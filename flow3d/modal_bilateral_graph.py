from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from modal_surface.motion_fill import query_knn_candidates


MODAL_BILATERAL_GRAPH_FORMAT = "modal_bilateral_gaussian_graph"
MODAL_BILATERAL_GRAPH_VERSION = 1

GRAPH_FILENAME = "graph.npz"
DIAGNOSTICS_FILENAME = "diagnostics.json"
EDGE_VISUALIZATION_FILENAME = "edge_strength_visualization.png"
EDGE_HISTOGRAM_FILENAME = "edge_strength_histogram.png"


@dataclass(frozen=True)
class ModalBilateralGraphResult:
    path: Path
    graph_path: Path
    diagnostics_path: Path
    edge_visualization_path: Path
    edge_histogram_path: Path


@dataclass(frozen=True)
class _CanonicalGaussians:
    points: np.ndarray
    scales: np.ndarray
    colors: np.ndarray
    checkpoint_size_bytes: int
    checkpoint_mtime_ns: int
    fingerprint_sha256: str


@dataclass(frozen=True)
class _BilateralGraph:
    edge_index: np.ndarray
    edge_distance: np.ndarray
    edge_scale_normalized_distance: np.ndarray
    edge_color_distance: np.ndarray
    edge_spatial_affinity: np.ndarray
    edge_color_affinity: np.ndarray
    edge_affinity: np.ndarray
    edge_weight: np.ndarray
    degree: np.ndarray
    component_index: np.ndarray
    component_sizes: np.ndarray
    directed_valid_candidate_count: int
    directed_selected_count: int
    directed_mutual_count: int


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_positive_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_settings(
    *,
    spatial_preselect_k: int,
    bilateral_k: int,
    max_distance: float,
    spatial_sigma: float,
    color_sigma: float,
    color_floor: float,
    scale_epsilon: float,
    visualization_max_edges: int,
    visualization_max_points: int,
) -> tuple[int, int, float, float, float, float, float, int, int]:
    preselect = _validate_positive_integer(
        spatial_preselect_k, "spatial_preselect_k"
    )
    final_k = _validate_positive_integer(bilateral_k, "bilateral_k")
    if final_k > preselect:
        raise ValueError("bilateral_k cannot exceed spatial_preselect_k")
    floor = float(color_floor)
    if not np.isfinite(floor) or floor < 0.0 or floor > 1.0:
        raise ValueError("color_floor must be finite and in [0,1]")
    return (
        preselect,
        final_k,
        _validate_positive_float(max_distance, "max_distance"),
        _validate_positive_float(spatial_sigma, "spatial_sigma"),
        _validate_positive_float(color_sigma, "color_sigma"),
        floor,
        _validate_positive_float(scale_epsilon, "scale_epsilon"),
        _validate_positive_integer(
            visualization_max_edges, "visualization_max_edges"
        ),
        _validate_positive_integer(
            visualization_max_points, "visualization_max_points"
        ),
    )


def _checkpoint_tensor(state: dict[str, Any], key: str) -> Any:
    if key not in state:
        raise ValueError(f"Checkpoint model state is missing {key!r}")
    value = state[key]
    if not hasattr(value, "detach"):
        raise ValueError(f"Checkpoint model field {key!r} is not a tensor")
    return value.detach().cpu()


def _canonical_fingerprint(
    points: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, array in (
        ("points", points),
        ("scales", scales),
        ("colors", colors),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _load_canonical_gaussians(checkpoint_path: str | Path) -> _CanonicalGaussians:
    try:
        import torch
    except ImportError as error:
        raise ImportError("Loading the static Gaussian checkpoint requires torch") from error

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state dictionary")

    means_tensor = _checkpoint_tensor(state, "fg.params.means").float()
    scales_tensor = _checkpoint_tensor(state, "fg.params.scales").float().exp()
    colors_tensor = _checkpoint_tensor(state, "fg.params.colors").float().sigmoid()
    points = means_tensor.numpy().astype(np.float32, copy=False)
    scales = scales_tensor.numpy().astype(np.float32, copy=False)
    colors = colors_tensor.numpy().astype(np.float32, copy=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Foreground means must have shape (N,3), got {points.shape}")
    if scales.shape != points.shape:
        raise ValueError(
            f"Activated foreground scales must have shape {points.shape}, got {scales.shape}"
        )
    if colors.shape != points.shape:
        raise ValueError(
            f"Activated foreground colors must have shape {points.shape}, got {colors.shape}"
        )
    if points.shape[0] < 2:
        raise ValueError("Bilateral graph construction requires at least two Gaussians")
    if not np.all(np.isfinite(points)):
        raise ValueError("Foreground means contain non-finite values")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("Activated foreground scales must be finite and positive")
    if (
        not np.all(np.isfinite(colors))
        or np.any(colors < 0.0)
        or np.any(colors > 1.0)
    ):
        raise ValueError("Activated foreground colors must be finite and in [0,1]")
    stat = path.stat()
    return _CanonicalGaussians(
        points=points,
        scales=scales,
        colors=colors,
        checkpoint_size_bytes=int(stat.st_size),
        checkpoint_mtime_ns=int(stat.st_mtime_ns),
        fingerprint_sha256=_canonical_fingerprint(points, scales, colors),
    )


def _stable_component_labels(
    num_points: int,
    edge_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as error:
        raise ImportError("Bilateral graph connectivity requires scipy.sparse") from error

    if edge_index.shape[0] == 0:
        labels = np.arange(num_points, dtype=np.int32)
        return labels, np.ones((num_points,), dtype=np.int32)
    source = np.concatenate((edge_index[:, 0], edge_index[:, 1]))
    target = np.concatenate((edge_index[:, 1], edge_index[:, 0]))
    adjacency = coo_matrix(
        (np.ones(source.shape[0], dtype=np.uint8), (source, target)),
        shape=(num_points, num_points),
    ).tocsr()
    component_count, labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    labels = np.asarray(labels, dtype=np.int64)
    first_point = np.full((component_count,), num_points, dtype=np.int64)
    np.minimum.at(first_point, labels, np.arange(num_points, dtype=np.int64))
    ordered_components = np.argsort(first_point, kind="stable")
    remap = np.empty((component_count,), dtype=np.int32)
    remap[ordered_components] = np.arange(component_count, dtype=np.int32)
    stable_labels = remap[labels]
    sizes = np.bincount(stable_labels, minlength=component_count).astype(np.int32)
    return stable_labels, sizes


def _build_bilateral_graph(
    canonical: _CanonicalGaussians,
    *,
    spatial_preselect_k: int,
    bilateral_k: int,
    max_distance: float,
    spatial_sigma: float,
    color_sigma: float,
    color_floor: float,
    scale_epsilon: float,
) -> _BilateralGraph:
    points = canonical.points
    scales = canonical.scales
    colors = canonical.colors
    num_points = int(points.shape[0])
    if spatial_preselect_k >= num_points:
        raise ValueError(
            "spatial_preselect_k must be smaller than the foreground Gaussian count"
        )
    candidates = query_knn_candidates(points, spatial_preselect_k)
    neighbors = np.asarray(candidates.neighbor_indices, dtype=np.int64)
    distances = np.asarray(candidates.neighbor_distances, dtype=np.float64)
    valid = distances <= max_distance
    valid_candidate_count = int(np.count_nonzero(valid))
    if valid_candidate_count == 0:
        raise ValueError("No spatial candidate edge satisfies max_distance")

    radii = np.sqrt(np.mean(np.square(scales.astype(np.float64)), axis=1))
    pair_radius = 0.5 * (radii[:, None] + radii[neighbors])
    normalized_distance = distances / np.maximum(pair_radius, scale_epsilon)
    color_delta = colors[:, None, :].astype(np.float64) - colors[neighbors].astype(
        np.float64
    )
    color_distance = np.linalg.norm(color_delta, axis=2)
    spatial_affinity = np.exp(
        -0.5 * np.square(normalized_distance / spatial_sigma)
    )
    raw_color_similarity = np.exp(-0.5 * np.square(color_distance / color_sigma))
    color_affinity = color_floor + (1.0 - color_floor) * raw_color_similarity
    affinity = spatial_affinity * color_affinity
    if not np.all(np.isfinite(affinity[valid])):
        raise ValueError("Bilateral affinity contains non-finite values")

    ranking_values = np.where(valid, affinity, -np.inf)
    order = np.argsort(-ranking_values, axis=1, kind="stable")[:, :bilateral_k]
    selected_neighbor = np.take_along_axis(neighbors, order, axis=1)
    selected_valid = np.take_along_axis(valid, order, axis=1)
    selected_distance = np.take_along_axis(distances, order, axis=1)
    selected_normalized_distance = np.take_along_axis(
        normalized_distance, order, axis=1
    )
    selected_color_distance = np.take_along_axis(color_distance, order, axis=1)
    selected_spatial_affinity = np.take_along_axis(spatial_affinity, order, axis=1)
    selected_color_affinity = np.take_along_axis(color_affinity, order, axis=1)
    selected_affinity = np.take_along_axis(affinity, order, axis=1)

    source = np.broadcast_to(
        np.arange(num_points, dtype=np.int64)[:, None], selected_neighbor.shape
    )[selected_valid]
    target = selected_neighbor[selected_valid]
    selected_distance = selected_distance[selected_valid]
    selected_normalized_distance = selected_normalized_distance[selected_valid]
    selected_color_distance = selected_color_distance[selected_valid]
    selected_spatial_affinity = selected_spatial_affinity[selected_valid]
    selected_color_affinity = selected_color_affinity[selected_valid]
    selected_affinity = selected_affinity[selected_valid]
    directed_selected_count = int(source.shape[0])

    directed_keys = source * num_points + target
    sorted_keys = np.sort(directed_keys)
    reverse_keys = target * num_points + source
    reverse_positions = np.searchsorted(sorted_keys, reverse_keys)
    reverse_in_range = reverse_positions < sorted_keys.shape[0]
    mutual = np.zeros(reverse_positions.shape, dtype=bool)
    mutual[reverse_in_range] = (
        sorted_keys[reverse_positions[reverse_in_range]]
        == reverse_keys[reverse_in_range]
    )
    directed_mutual_count = int(np.count_nonzero(mutual))
    undirected = mutual & (source < target) & (selected_affinity > 0.0)
    edge_index = np.column_stack((source[undirected], target[undirected]))
    if edge_index.shape[0] == 0:
        raise ValueError(
            "Bilateral top-K selection produced no positive mutual undirected edge"
        )
    edge_distance = selected_distance[undirected]
    edge_normalized_distance = selected_normalized_distance[undirected]
    edge_color_distance = selected_color_distance[undirected]
    edge_spatial_affinity = selected_spatial_affinity[undirected]
    edge_color_affinity = selected_color_affinity[undirected]
    edge_affinity = selected_affinity[undirected]
    if np.any(edge_index[:, 0] >= edge_index[:, 1]):
        raise RuntimeError("Bilateral graph edges are not canonical undirected pairs")
    if np.unique(edge_index, axis=0).shape[0] != edge_index.shape[0]:
        raise RuntimeError("Bilateral graph contains duplicate undirected edges")

    affinity_mean = float(np.mean(edge_affinity))
    if not np.isfinite(affinity_mean) or affinity_mean <= 0.0:
        raise ValueError("Bilateral graph has no positive finite mean affinity")
    edge_weight = edge_affinity / affinity_mean
    degree = np.zeros((num_points,), dtype=np.int32)
    np.add.at(degree, edge_index[:, 0], 1)
    np.add.at(degree, edge_index[:, 1], 1)
    if np.any(degree > bilateral_k):
        raise RuntimeError("Mutual top-K graph degree exceeds bilateral_k")
    component_index, component_sizes = _stable_component_labels(
        num_points, edge_index
    )
    return _BilateralGraph(
        edge_index=edge_index.astype(np.int64, copy=False),
        edge_distance=edge_distance.astype(np.float32),
        edge_scale_normalized_distance=edge_normalized_distance.astype(np.float32),
        edge_color_distance=edge_color_distance.astype(np.float32),
        edge_spatial_affinity=edge_spatial_affinity.astype(np.float32),
        edge_color_affinity=edge_color_affinity.astype(np.float32),
        edge_affinity=edge_affinity.astype(np.float32),
        edge_weight=edge_weight.astype(np.float32),
        degree=degree,
        component_index=component_index,
        component_sizes=component_sizes,
        directed_valid_candidate_count=valid_candidate_count,
        directed_selected_count=directed_selected_count,
        directed_mutual_count=directed_mutual_count,
    )


def _percentiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Diagnostics require a non-empty finite array")
    return {
        f"p{percentile}": float(np.percentile(array, percentile))
        for percentile in (0, 1, 10, 25, 50, 75, 90, 99, 100)
    }


def _diagnostics(
    *,
    checkpoint_path: Path,
    canonical: _CanonicalGaussians,
    graph: _BilateralGraph,
    spatial_preselect_k: int,
    bilateral_k: int,
    max_distance: float,
    spatial_sigma: float,
    color_sigma: float,
    color_floor: float,
    scale_epsilon: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    largest_component = int(np.max(graph.component_sizes))
    num_points = int(canonical.points.shape[0])
    return {
        "format": MODAL_BILATERAL_GRAPH_FORMAT,
        "version": MODAL_BILATERAL_GRAPH_VERSION,
        "source": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_size_bytes": canonical.checkpoint_size_bytes,
            "checkpoint_mtime_ns": canonical.checkpoint_mtime_ns,
            "canonical_fingerprint_sha256": canonical.fingerprint_sha256,
        },
        "settings": {
            "spatial_preselect_k": spatial_preselect_k,
            "bilateral_k": bilateral_k,
            "max_distance": max_distance,
            "gaussian_radius": "activated_scale_rms",
            "pair_radius": "arithmetic_mean",
            "spatial_sigma": spatial_sigma,
            "color_space": "activated_rgb",
            "color_sigma": color_sigma,
            "color_floor": color_floor,
            "scale_epsilon": scale_epsilon,
            "topology": "mutual_bilateral_top_k",
            "edge_weight_normalization": "global_mean_one",
        },
        "counts": {
            "gaussians": num_points,
            "directed_spatial_candidates_within_cutoff": (
                graph.directed_valid_candidate_count
            ),
            "directed_bilateral_top_k": graph.directed_selected_count,
            "directed_mutual_top_k": graph.directed_mutual_count,
            "undirected_edges": int(graph.edge_index.shape[0]),
            "isolated_gaussians": int(np.count_nonzero(graph.degree == 0)),
            "components": int(graph.component_sizes.shape[0]),
            "largest_component": largest_component,
        },
        "fractions": {
            "mutual_of_selected": (
                graph.directed_mutual_count / graph.directed_selected_count
            ),
            "isolated_gaussians": float(np.mean(graph.degree == 0)),
            "largest_component": largest_component / num_points,
        },
        "distributions": {
            "activated_scale_rms": _percentiles(
                np.sqrt(np.mean(np.square(canonical.scales), axis=1))
            ),
            "degree": _percentiles(graph.degree),
            "edge_distance": _percentiles(graph.edge_distance),
            "edge_scale_normalized_distance": _percentiles(
                graph.edge_scale_normalized_distance
            ),
            "edge_color_distance": _percentiles(graph.edge_color_distance),
            "edge_spatial_affinity": _percentiles(graph.edge_spatial_affinity),
            "edge_color_affinity": _percentiles(graph.edge_color_affinity),
            "edge_affinity": _percentiles(graph.edge_affinity),
            "edge_weight": _percentiles(graph.edge_weight),
            "component_size": _percentiles(graph.component_sizes),
        },
        "timings_seconds": {"total": float(elapsed_seconds)},
    }


def _even_rank_sample(values: np.ndarray, maximum_count: int) -> np.ndarray:
    count = int(values.shape[0])
    if count <= maximum_count:
        return np.arange(count, dtype=np.int64)
    order = np.argsort(values, kind="stable")
    positions = np.linspace(0, count - 1, maximum_count, dtype=np.int64)
    return order[positions]


def _point_sample(num_points: int, maximum_count: int) -> np.ndarray:
    if num_points <= maximum_count:
        return np.arange(num_points, dtype=np.int64)
    return np.linspace(0, num_points - 1, maximum_count, dtype=np.int64)


def _write_edge_visualization(
    path: Path,
    canonical: _CanonicalGaussians,
    graph: _BilateralGraph,
    maximum_edges: int,
    maximum_points: int,
) -> None:
    edge_selection = _even_rank_sample(graph.edge_affinity, maximum_edges)
    point_selection = _point_sample(canonical.points.shape[0], maximum_points)
    edge_points = canonical.points[graph.edge_index[edge_selection]]
    strengths = graph.edge_affinity[edge_selection]
    lower, upper = np.percentile(graph.edge_affinity, [1.0, 99.0])
    if upper <= lower:
        upper = lower + np.finfo(np.float32).eps
    norm = Normalize(vmin=float(lower), vmax=float(upper), clip=True)
    colors = plt.get_cmap("viridis")(norm(strengths))
    colors[:, 3] = 0.08 + 0.82 * norm(strengths)

    projections = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, (horizontal, vertical, horizontal_name, vertical_name) in zip(
        axes, projections
    ):
        segments = edge_points[:, :, (horizontal, vertical)]
        collection = LineCollection(
            segments,
            colors=colors,
            linewidths=0.45,
            rasterized=True,
        )
        axis.add_collection(collection)
        sampled_points = canonical.points[point_selection]
        axis.scatter(
            sampled_points[:, horizontal],
            sampled_points[:, vertical],
            s=0.15,
            c=canonical.colors[point_selection],
            alpha=0.2,
            linewidths=0.0,
            rasterized=True,
        )
        axis.autoscale()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(horizontal_name)
        axis.set_ylabel(vertical_name)
        axis.set_title(f"{horizontal_name.upper()}{vertical_name.upper()} projection")
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    scalar_map.set_array([])
    figure.colorbar(scalar_map, ax=axes, label="bilateral edge affinity", shrink=0.8)
    figure.suptitle(
        "Fixed bilateral Gaussian graph\n"
        f"showing {edge_selection.shape[0]:,} / {graph.edge_index.shape[0]:,} edges "
        "with deterministic rank sampling"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_edge_histogram(path: Path, graph: _BilateralGraph) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    entries = (
        (graph.edge_affinity, "combined bilateral affinity"),
        (graph.edge_spatial_affinity, "spatial affinity"),
        (graph.edge_color_affinity, "activated-RGB affinity"),
        (graph.edge_weight, "mean-normalized rigidity weight"),
    )
    for axis, (values, title) in zip(axes.flat, entries):
        axis.hist(values, bins=100, color="#2878B5", alpha=0.9)
        axis.set_title(title)
        axis.set_xlabel("value")
        axis.set_ylabel("edge count")
        axis.grid(alpha=0.2)
    figure.suptitle("Bilateral Gaussian edge-strength distributions")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_graph_npz(
    path: Path,
    checkpoint_path: Path,
    canonical: _CanonicalGaussians,
    graph: _BilateralGraph,
    *,
    spatial_preselect_k: int,
    bilateral_k: int,
    max_distance: float,
    spatial_sigma: float,
    color_sigma: float,
    color_floor: float,
    scale_epsilon: float,
) -> None:
    np.savez_compressed(
        path,
        format=np.array(MODAL_BILATERAL_GRAPH_FORMAT),
        version=np.array(MODAL_BILATERAL_GRAPH_VERSION, dtype=np.int32),
        source_checkpoint=np.array(str(checkpoint_path)),
        canonical_fingerprint_sha256=np.array(canonical.fingerprint_sha256),
        gaussian_indices=np.arange(canonical.points.shape[0], dtype=np.int32),
        points_world=canonical.points,
        activated_scales=canonical.scales,
        activated_rgb=canonical.colors,
        edge_index=graph.edge_index,
        edge_distance=graph.edge_distance,
        edge_scale_normalized_distance=graph.edge_scale_normalized_distance,
        edge_color_distance=graph.edge_color_distance,
        edge_spatial_affinity=graph.edge_spatial_affinity,
        edge_color_affinity=graph.edge_color_affinity,
        edge_affinity=graph.edge_affinity,
        edge_weight=graph.edge_weight,
        degree=graph.degree,
        component_index=graph.component_index,
        component_sizes=graph.component_sizes,
        isolated_mask=graph.degree == 0,
        spatial_preselect_k=np.array(spatial_preselect_k, dtype=np.int32),
        bilateral_k=np.array(bilateral_k, dtype=np.int32),
        max_distance=np.array(max_distance, dtype=np.float64),
        spatial_sigma=np.array(spatial_sigma, dtype=np.float64),
        color_sigma=np.array(color_sigma, dtype=np.float64),
        color_floor=np.array(color_floor, dtype=np.float64),
        scale_epsilon=np.array(scale_epsilon, dtype=np.float64),
        topology=np.array("mutual_bilateral_top_k"),
        gaussian_radius_method=np.array("activated_scale_rms"),
        pair_radius_method=np.array("arithmetic_mean"),
        color_space=np.array("activated_rgb"),
        edge_weight_normalization=np.array("global_mean_one"),
    )


def _validate_written_graph(path: Path, expected_points: int, expected_edges: int) -> None:
    required = {
        "format",
        "version",
        "gaussian_indices",
        "points_world",
        "activated_scales",
        "activated_rgb",
        "edge_index",
        "edge_affinity",
        "edge_weight",
        "degree",
        "component_index",
    }
    with np.load(path, allow_pickle=False) as artifact:
        missing = required - set(artifact.files)
        if missing:
            raise RuntimeError(f"Written bilateral graph is missing fields: {sorted(missing)}")
        if str(artifact["format"].item()) != MODAL_BILATERAL_GRAPH_FORMAT:
            raise RuntimeError("Written bilateral graph has the wrong format")
        if int(artifact["version"].item()) != MODAL_BILATERAL_GRAPH_VERSION:
            raise RuntimeError("Written bilateral graph has the wrong version")
        if artifact["points_world"].shape != (expected_points, 3):
            raise RuntimeError("Written bilateral graph has the wrong point shape")
        if artifact["edge_index"].shape != (expected_edges, 2):
            raise RuntimeError("Written bilateral graph has the wrong edge shape")
        if artifact["edge_affinity"].shape != (expected_edges,):
            raise RuntimeError("Written bilateral graph has the wrong affinity shape")


def build_modal_bilateral_graph(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    spatial_preselect_k: int = 32,
    bilateral_k: int = 8,
    max_distance: float = 0.008,
    spatial_sigma: float = 1.0,
    color_sigma: float = 0.15,
    color_floor: float = 0.05,
    scale_epsilon: float = 1e-8,
    visualization_max_edges: int = 50_000,
    visualization_max_points: int = 50_000,
) -> ModalBilateralGraphResult:
    (
        spatial_preselect_k,
        bilateral_k,
        max_distance,
        spatial_sigma,
        color_sigma,
        color_floor,
        scale_epsilon,
        visualization_max_edges,
        visualization_max_points,
    ) = _validate_settings(
        spatial_preselect_k=spatial_preselect_k,
        bilateral_k=bilateral_k,
        max_distance=max_distance,
        spatial_sigma=spatial_sigma,
        color_sigma=color_sigma,
        color_floor=color_floor,
        scale_epsilon=scale_epsilon,
        visualization_max_edges=visualization_max_edges,
        visualization_max_points=visualization_max_points,
    )
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    start = time.perf_counter()
    try:
        canonical = _load_canonical_gaussians(checkpoint)
        graph = _build_bilateral_graph(
            canonical,
            spatial_preselect_k=spatial_preselect_k,
            bilateral_k=bilateral_k,
            max_distance=max_distance,
            spatial_sigma=spatial_sigma,
            color_sigma=color_sigma,
            color_floor=color_floor,
            scale_epsilon=scale_epsilon,
        )
        graph_path = temporary / GRAPH_FILENAME
        _write_graph_npz(
            graph_path,
            checkpoint,
            canonical,
            graph,
            spatial_preselect_k=spatial_preselect_k,
            bilateral_k=bilateral_k,
            max_distance=max_distance,
            spatial_sigma=spatial_sigma,
            color_sigma=color_sigma,
            color_floor=color_floor,
            scale_epsilon=scale_epsilon,
        )
        edge_visualization_path = temporary / EDGE_VISUALIZATION_FILENAME
        _write_edge_visualization(
            edge_visualization_path,
            canonical,
            graph,
            visualization_max_edges,
            visualization_max_points,
        )
        edge_histogram_path = temporary / EDGE_HISTOGRAM_FILENAME
        _write_edge_histogram(edge_histogram_path, graph)
        elapsed = time.perf_counter() - start
        diagnostics_path = temporary / DIAGNOSTICS_FILENAME
        diagnostics = _diagnostics(
            checkpoint_path=checkpoint,
            canonical=canonical,
            graph=graph,
            spatial_preselect_k=spatial_preselect_k,
            bilateral_k=bilateral_k,
            max_distance=max_distance,
            spatial_sigma=spatial_sigma,
            color_sigma=color_sigma,
            color_floor=color_floor,
            scale_epsilon=scale_epsilon,
            elapsed_seconds=elapsed,
        )
        diagnostics_path.write_text(
            json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8"
        )
        _validate_written_graph(
            graph_path,
            expected_points=canonical.points.shape[0],
            expected_edges=graph.edge_index.shape[0],
        )
        for required_path in (
            diagnostics_path,
            edge_visualization_path,
            edge_histogram_path,
        ):
            if not required_path.is_file() or required_path.stat().st_size == 0:
                raise RuntimeError(f"Failed to write {required_path.name}")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return ModalBilateralGraphResult(
        path=output,
        graph_path=output / GRAPH_FILENAME,
        diagnostics_path=output / DIAGNOSTICS_FILENAME,
        edge_visualization_path=output / EDGE_VISUALIZATION_FILENAME,
        edge_histogram_path=output / EDGE_HISTOGRAM_FILENAME,
    )


__all__ = [
    "DIAGNOSTICS_FILENAME",
    "EDGE_HISTOGRAM_FILENAME",
    "EDGE_VISUALIZATION_FILENAME",
    "GRAPH_FILENAME",
    "MODAL_BILATERAL_GRAPH_FORMAT",
    "MODAL_BILATERAL_GRAPH_VERSION",
    "ModalBilateralGraphResult",
    "build_modal_bilateral_graph",
]
