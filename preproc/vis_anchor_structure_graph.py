"""Standalone browser viewer for observed anchor structure graph artifacts.

This tool intentionally depends only on NumPy and a modern Viser release. It is
designed to run in an isolated environment instead of the legacy Nerfview
environment used by the main Shape-of-Motion viewer.
"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


_GRAPH_REQUIRED_FIELDS = {
    "version",
    "mode_index",
    "freq_hz",
    "source_checkpoint",
    "num_foreground_gaussians",
    "anchor_gaussian_indices",
    "anchor_points_world",
    "anchor_colors_rgb",
    "anchor_observed_view_mask",
    "edge_index",
    "edge_distance",
    "edge_distance_weight",
    "edge_color_distance",
    "edge_color_weight",
    "edge_depth_score",
    "edge_combined_weight",
    "edge_view_support_mask",
    "edge_view_support_count",
    "edge_endpoint_gap_by_view",
    "edge_depth_jump_by_view",
    "degree",
    "component_index",
    "component_size",
    "isolated_mask",
    "view_ids",
    "endpoint_gap_median_by_view",
    "endpoint_gap_mad_by_view",
    "endpoint_gap_threshold_by_view",
    "depth_jump_median_by_view",
    "depth_jump_mad_by_view",
    "depth_jump_threshold_by_view",
    "color_distance_median",
    "color_distance_mad",
    "color_distance_threshold",
    "max_neighbors",
    "max_distance",
    "color_mad_multiplier",
    "depth_mad_multiplier",
    "depth_samples",
    "min_shared_views",
    "render_acc_min",
    "epsilon",
    "mad_scale",
    "knn_policy",
    "depth_source",
    "color_space",
    "distance_weight_method",
    "color_weight_method",
    "depth_weight_method",
    "knn_directed_candidate_count",
    "distance_rejected_directed_count",
    "nonmutual_rejected_pair_count",
    "mutual_distance_candidate_count",
    "anchor_count",
    "shared_observed_candidate_view_count",
    "raw_depth_valid_candidate_view_count",
    "endpoint_rejected_candidate_view_count",
    "jump_rejected_candidate_view_count",
    "supporting_candidate_view_count",
    "color_rejected_count",
    "color_retained_count",
    "depth_rejected_count",
    "depth_retained_count",
    "retained_edge_count",
    "isolated_anchor_count",
    "component_count",
}

_MANIFEST_PARAMETER_FIELDS = {
    "anchor_graph_version": "version",
    "anchor_graph_max_neighbors": "max_neighbors",
    "anchor_graph_max_distance": "max_distance",
    "anchor_graph_color_mad_multiplier": "color_mad_multiplier",
    "anchor_graph_depth_mad_multiplier": "depth_mad_multiplier",
    "anchor_graph_depth_samples": "depth_samples",
    "anchor_graph_min_shared_views": "min_shared_views",
    "anchor_graph_render_acc_min": "render_acc_min",
    "anchor_graph_epsilon": "epsilon",
}

_EXPECTED_SEMANTICS = {
    "knn_policy": "mutual_knn",
    "depth_source": "static_3dgs_rendered_depth",
    "color_space": "opencv_float_rgb_to_lab",
    "distance_weight_method": "inverse_distance",
    "color_weight_method": "gaussian_adaptive_threshold",
    "depth_weight_method": "supporting_view_gaussian_score",
}

_COUNT_FIELDS = {
    "knn_directed_candidate_count",
    "distance_rejected_directed_count",
    "nonmutual_rejected_pair_count",
    "mutual_distance_candidate_count",
    "anchor_count",
    "shared_observed_candidate_view_count",
    "raw_depth_valid_candidate_view_count",
    "endpoint_rejected_candidate_view_count",
    "jump_rejected_candidate_view_count",
    "supporting_candidate_view_count",
    "color_rejected_count",
    "color_retained_count",
    "depth_rejected_count",
    "depth_retained_count",
    "retained_edge_count",
    "isolated_anchor_count",
    "component_count",
}

_GAUSSIAN_BASE_REQUIRED_FIELDS = {
    "version",
    "point_type",
    "source_checkpoint",
    "has_background",
}

_GAUSSIAN_GROUP_FIELDS = (
    "gaussian_indices",
    "centers",
    "scales",
    "quats_wxyz",
    "rgbs",
    "opacities",
)

_COVERAGE_REQUIRED_FIELDS = {
    "version",
    "point_type",
    "source_checkpoint",
    "reference_observation_path",
    "num_foreground_gaussians",
    "gaussian_indices",
    "points_world",
    "view_ids",
    "k_values",
    "baseline_k",
    "baseline_k_index",
    "category_names",
    "preselect_hit_count_by_view",
    "positive_hit_count_by_view",
    "selected_hit_count_by_k_view",
    "best_positive_rank_by_view",
    "best_positive_score_by_view",
    "best_positive_score_ratio_by_view",
    "preselect_view_count",
    "positive_view_count",
    "selected_view_count_by_k",
    "selected_sample_count_by_k",
    "category_by_k",
    "category_count_by_k",
    "preselect_view_count_histogram",
    "positive_view_count_histogram",
    "selected_view_count_histogram_by_k",
    "mask_erode_iters",
    "pixel_sample_stride",
    "pixel_preselect_k",
    "pixel_render_acc_min",
    "pixel_min_contribution",
    "candidate_method",
    "replay_validation",
}

_COVERAGE_CATEGORY_NAMES = (
    "insufficient_preselect_views",
    "multiview_preselect_contribution_lost",
    "multiview_positive_topk_lost",
    "selected_multiview",
)

_COVERAGE_CATEGORY_COLORS = np.asarray(
    (
        (0.45, 0.45, 0.45),
        (1.0, 0.55, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.45, 1.0),
    ),
    dtype=np.float32,
)


@dataclass(frozen=True)
class AnchorGraphViewData:
    mode_index: int
    freq_hz: float
    graph_path: Path
    source_checkpoint: str
    num_foreground_gaussians: int
    anchor_gaussian_indices: np.ndarray
    anchor_points_world: np.ndarray
    anchor_colors_rgb: np.ndarray
    edge_index: np.ndarray
    edge_depth_score: np.ndarray
    edge_combined_weight: np.ndarray
    edge_view_support_count: np.ndarray
    component_index: np.ndarray
    isolated_mask: np.ndarray

    @property
    def label(self) -> str:
        return f"Mode {self.mode_index}: {self.freq_hz:.3f} Hz"


@dataclass(frozen=True)
class GaussianSplatGroup:
    centers: np.ndarray
    covariances: np.ndarray
    rgbs: np.ndarray
    opacities: np.ndarray


@dataclass(frozen=True)
class GaussianVisualizationData:
    sidecar_path: Path
    source_checkpoint: str
    foreground: GaussianSplatGroup
    background: GaussianSplatGroup | None


@dataclass(frozen=True)
class ObservationCoverageViewData:
    artifact_path: Path
    source_checkpoint: str
    points_world: np.ndarray
    k_values: np.ndarray
    category_names: tuple[str, ...]
    category_by_k: np.ndarray


def stable_uniform_indices(count: int, maximum: int) -> np.ndarray:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, np.integer))
        or maximum < 0
    ):
        raise ValueError("maximum must be a non-negative integer")
    visible_count = min(int(count), int(maximum))
    if visible_count == 0:
        return np.empty((0,), dtype=np.int64)
    if visible_count == int(count):
        return np.arange(count, dtype=np.int64)
    return np.floor(
        np.linspace(
            0,
            count,
            visible_count,
            endpoint=False,
            dtype=np.float64,
        )
    ).astype(np.int64)


def stable_uniform_edge_indices(
    edge_count: int,
    max_visible_edges: int,
) -> np.ndarray:
    return stable_uniform_indices(edge_count, max_visible_edges)


def observation_coverage_colors(category: np.ndarray) -> np.ndarray:
    values = np.asarray(category)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Coverage category must be a 1-D integer array")
    if np.any(values < 0) or np.any(values >= len(_COVERAGE_CATEGORY_NAMES)):
        raise ValueError("Coverage category contains an unknown value")
    return _COVERAGE_CATEGORY_COLORS[values]


def anchor_graph_component_colors(component_index: np.ndarray) -> np.ndarray:
    components = np.asarray(component_index)
    if components.ndim != 1 or not np.issubdtype(components.dtype, np.integer):
        raise ValueError("component_index must be a 1-D integer array")
    if np.any(components < 0):
        raise ValueError("component_index must be non-negative")
    hue = np.mod(components.astype(np.float64) * 0.6180339887498949, 1.0)
    h6 = hue * 6.0
    sector = np.floor(h6).astype(np.int64)
    fraction = h6 - sector
    saturation = 0.72
    value = 0.95
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    rgb = np.empty((components.shape[0], 3), dtype=np.float64)
    choices = (
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    )
    for index, values in enumerate(choices):
        mask = sector == index
        if np.any(mask):
            rgb[mask] = np.column_stack(
                [
                    np.full(np.count_nonzero(mask), channel)
                    if np.isscalar(channel)
                    else channel[mask]
                    for channel in values
                ]
            )
    return rgb.astype(np.float32)


def anchor_graph_scalar_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Anchor graph color metric must be a finite 1-D array")
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum == minimum:
        normalized = np.ones(values.shape, dtype=np.float32)
    else:
        normalized = (values - minimum) / (maximum - minimum)
    return np.column_stack(
        [1.0 - normalized, 0.25 + 0.75 * normalized, normalized]
    ).astype(np.float32)


def gaussian_covariances(
    scales: np.ndarray,
    quats_wxyz: np.ndarray,
) -> np.ndarray:
    scales = np.asarray(scales, dtype=np.float64)
    quaternions = np.asarray(quats_wxyz, dtype=np.float64)
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("Gaussian scales must have shape (N,3)")
    if quaternions.shape != (scales.shape[0], 4):
        raise ValueError("Gaussian quats_wxyz must have shape (N,4)")
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("Gaussian scales must be finite and positive")
    if not np.isfinite(quaternions).all():
        raise ValueError("Gaussian quats_wxyz must be finite")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Gaussian quats_wxyz must be nonzero")
    quaternions = quaternions / norms
    w, x, y, z = quaternions.T
    rotations = np.empty((scales.shape[0], 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    scaled_rotations = rotations * np.square(scales)[:, None, :]
    covariances = scaled_rotations @ np.swapaxes(rotations, 1, 2)
    if not np.isfinite(covariances).all():
        raise ValueError("Gaussian covariances must be finite")
    return covariances.astype(np.float32)


def scale_gaussian_opacities(
    opacities: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    opacities = np.asarray(opacities, dtype=np.float32)
    multiplier = float(multiplier)
    if opacities.ndim != 2 or opacities.shape[1] != 1:
        raise ValueError("Gaussian opacities must have shape (N,1)")
    if not np.isfinite(opacities).all() or np.any(
        (opacities < 0.0) | (opacities > 1.0)
    ):
        raise ValueError("Gaussian opacities must be finite and lie in [0,1]")
    if not np.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
        raise ValueError("Gaussian opacity multiplier must lie in [0,1]")
    return opacities * multiplier


def _scalar(array: np.ndarray, name: str, path: Path) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{path} {name} must be scalar")
    return value


def _scalar_string(array: np.ndarray, name: str, path: Path) -> str:
    value = _scalar(array, name, path).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} {name} must be a non-empty scalar string")
    return value


def _component_data(
    num_anchors: int,
    edge_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree = np.zeros((num_anchors,), dtype=np.int32)
    if edge_index.shape[0]:
        np.add.at(degree, edge_index[:, 0], 1)
        np.add.at(degree, edge_index[:, 1], 1)
    parent = np.arange(num_anchors, dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for start, end in edge_index:
        start_root = find(int(start))
        end_root = find(int(end))
        if start_root != end_root:
            parent[max(start_root, end_root)] = min(start_root, end_root)
    roots = np.asarray([find(index) for index in range(num_anchors)], dtype=np.int32)
    _, components = np.unique(roots, return_inverse=True)
    components = components.astype(np.int32)
    sizes = (
        np.bincount(
            components,
            minlength=int(components.max()) + 1,
        ).astype(np.int32)
        if num_anchors
        else np.empty((0,), dtype=np.int32)
    )
    return degree, components, sizes


def _load_graph_archive(
    graph_path: Path,
    *,
    expected_mode_index: int | None = None,
    expected_freq_hz: float | None = None,
    expected_source_checkpoint: str | None = None,
    manifest_parameters: dict[str, Any] | None = None,
) -> AnchorGraphViewData:
    if not graph_path.is_file():
        raise ValueError(f"Anchor graph does not exist: {graph_path}")
    with np.load(str(graph_path), allow_pickle=False) as archive:
        missing = sorted(_GRAPH_REQUIRED_FIELDS - set(archive.files))
        if missing:
            raise ValueError(f"{graph_path} missing required fields: {missing}")
        graph = {name: np.asarray(archive[name]) for name in archive.files}

    version_value = _scalar(graph["version"], "version", graph_path)
    if not np.issubdtype(version_value.dtype, np.integer) or int(
        version_value.item()
    ) != 1:
        raise ValueError(f"{graph_path} must be a version 1 anchor graph")
    mode_value = _scalar(graph["mode_index"], "mode_index", graph_path)
    if not np.issubdtype(mode_value.dtype, np.integer):
        raise ValueError(f"{graph_path} mode_index must be an integer scalar")
    mode_index = int(mode_value.item())
    freq_value = _scalar(graph["freq_hz"], "freq_hz", graph_path)
    if (
        not np.issubdtype(freq_value.dtype, np.number)
        or np.iscomplexobj(freq_value)
        or not np.isfinite(freq_value.item())
    ):
        raise ValueError(f"{graph_path} freq_hz must be a finite real scalar")
    freq_hz = float(freq_value.item())
    source_checkpoint = _scalar_string(
        graph["source_checkpoint"],
        "source_checkpoint",
        graph_path,
    )
    if expected_mode_index is not None and mode_index != expected_mode_index:
        raise ValueError(f"{graph_path} mode_index does not match manifest mode")
    if expected_freq_hz is not None and not np.isclose(
        freq_hz,
        expected_freq_hz,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(f"{graph_path} freq_hz does not match manifest mode")
    if (
        expected_source_checkpoint is not None
        and source_checkpoint != expected_source_checkpoint
    ):
        raise ValueError(f"{graph_path} source_checkpoint does not match manifest")

    for field_name, expected_value in _EXPECTED_SEMANTICS.items():
        if _scalar_string(graph[field_name], field_name, graph_path) != expected_value:
            raise ValueError(f"{graph_path} {field_name} is incompatible")
    mad_scale = float(_scalar(graph["mad_scale"], "mad_scale", graph_path).item())
    if not np.isclose(mad_scale, 1.4826, rtol=1e-6, atol=1e-8):
        raise ValueError(f"{graph_path} mad_scale is incompatible")

    numeric_parameters = {
        artifact_name: _scalar(graph[artifact_name], artifact_name, graph_path).item()
        for artifact_name in _MANIFEST_PARAMETER_FIELDS.values()
    }
    for field_name in ("max_neighbors", "depth_samples", "min_shared_views"):
        parameter_array = _scalar(graph[field_name], field_name, graph_path)
        if not np.issubdtype(parameter_array.dtype, np.integer):
            raise ValueError(f"{graph_path} {field_name} must be an integer scalar")
    max_neighbors = int(numeric_parameters["max_neighbors"])
    max_distance = float(numeric_parameters["max_distance"])
    color_mad_multiplier = float(numeric_parameters["color_mad_multiplier"])
    depth_mad_multiplier = float(numeric_parameters["depth_mad_multiplier"])
    depth_samples = int(numeric_parameters["depth_samples"])
    min_shared_views = int(numeric_parameters["min_shared_views"])
    render_acc_min = float(numeric_parameters["render_acc_min"])
    epsilon = float(numeric_parameters["epsilon"])
    if max_neighbors <= 0:
        raise ValueError(f"{graph_path} max_neighbors must be positive")
    if not np.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError(f"{graph_path} max_distance must be finite and positive")
    if not np.isfinite(color_mad_multiplier) or color_mad_multiplier < 0.0:
        raise ValueError(
            f"{graph_path} color_mad_multiplier must be finite and non-negative"
        )
    if not np.isfinite(depth_mad_multiplier) or depth_mad_multiplier < 0.0:
        raise ValueError(
            f"{graph_path} depth_mad_multiplier must be finite and non-negative"
        )
    if depth_samples < 2:
        raise ValueError(f"{graph_path} depth_samples must be at least 2")
    if min_shared_views <= 0:
        raise ValueError(f"{graph_path} min_shared_views must be positive")
    if not np.isfinite(render_acc_min) or not 0.0 <= render_acc_min <= 1.0:
        raise ValueError(f"{graph_path} render_acc_min must lie in [0,1]")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"{graph_path} epsilon must be finite and positive")
    if manifest_parameters is not None:
        missing_parameters = sorted(
            set(_MANIFEST_PARAMETER_FIELDS) - set(manifest_parameters)
        )
        if missing_parameters:
            raise ValueError(
                "Manifest missing anchor graph parameters: "
                f"{missing_parameters}"
            )
        for parameter_name, artifact_name in _MANIFEST_PARAMETER_FIELDS.items():
            manifest_value = manifest_parameters[parameter_name]
            if isinstance(manifest_value, bool) or not isinstance(
                manifest_value,
                (int, float),
            ):
                raise ValueError(
                    f"Manifest parameter {parameter_name} must be numeric"
                )
            if not np.isclose(
                float(manifest_value),
                float(numeric_parameters[artifact_name]),
                rtol=1e-6,
                atol=1e-8,
            ):
                raise ValueError(
                    f"{graph_path} {artifact_name} does not match manifest"
                )

    num_gaussians_value = _scalar(
        graph["num_foreground_gaussians"],
        "num_foreground_gaussians",
        graph_path,
    )
    if not np.issubdtype(num_gaussians_value.dtype, np.integer):
        raise ValueError(
            f"{graph_path} num_foreground_gaussians must be an integer scalar"
        )
    num_gaussians = int(num_gaussians_value.item())
    if num_gaussians < 0:
        raise ValueError(f"{graph_path} num_foreground_gaussians must be non-negative")

    anchor_indices = graph["anchor_gaussian_indices"]
    if anchor_indices.ndim != 1 or not np.issubdtype(anchor_indices.dtype, np.integer):
        raise ValueError(
            f"{graph_path} anchor_gaussian_indices must be 1-D integers"
        )
    anchor_indices = anchor_indices.astype(np.int64)
    if np.any(anchor_indices < 0) or np.any(anchor_indices >= num_gaussians):
        raise ValueError(f"{graph_path} anchor_gaussian_indices are out of range")
    if anchor_indices.size > 1 and np.any(np.diff(anchor_indices) <= 0):
        raise ValueError(
            f"{graph_path} anchor_gaussian_indices must be strictly increasing"
        )
    num_anchors = int(anchor_indices.shape[0])
    points = graph["anchor_points_world"].astype(np.float32)
    colors = graph["anchor_colors_rgb"].astype(np.float32)
    if points.shape != (num_anchors, 3) or not np.isfinite(points).all():
        raise ValueError(f"{graph_path} anchor_points_world must be finite (A,3)")
    if colors.shape != (num_anchors, 3) or not np.isfinite(colors).all():
        raise ValueError(f"{graph_path} anchor_colors_rgb must be finite (A,3)")
    if np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError(f"{graph_path} anchor_colors_rgb must lie in [0,1]")

    view_ids = graph["view_ids"]
    if view_ids.ndim != 1 or view_ids.shape[0] == 0:
        raise ValueError(f"{graph_path} view_ids must be non-empty and 1-D")
    normalized_view_ids = []
    for raw_view_id in view_ids:
        view_id = np.asarray(raw_view_id).item()
        if isinstance(view_id, bytes):
            view_id = view_id.decode("utf-8")
        if not isinstance(view_id, str) or not view_id:
            raise ValueError(f"{graph_path} view_ids must contain non-empty strings")
        normalized_view_ids.append(view_id)
    if len(set(normalized_view_ids)) != len(normalized_view_ids):
        raise ValueError(f"{graph_path} view_ids must be unique")
    num_views = len(normalized_view_ids)
    if min_shared_views > num_views:
        raise ValueError(
            f"{graph_path} min_shared_views cannot exceed the number of views"
        )
    if graph["anchor_observed_view_mask"].shape != (num_anchors, num_views):
        raise ValueError(
            f"{graph_path} anchor_observed_view_mask has invalid shape"
        )
    for field_name in (
        "endpoint_gap_median_by_view",
        "endpoint_gap_mad_by_view",
        "endpoint_gap_threshold_by_view",
        "depth_jump_median_by_view",
        "depth_jump_mad_by_view",
        "depth_jump_threshold_by_view",
    ):
        if graph[field_name].shape != (num_views,):
            raise ValueError(
                f"{graph_path} {field_name} must have shape ({num_views},)"
            )

    edges = graph["edge_index"]
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(
        edges.dtype,
        np.integer,
    ):
        raise ValueError(f"{graph_path} edge_index must be integer (E,2)")
    edges = edges.astype(np.int64)
    num_edges = int(edges.shape[0])
    if (
        np.any(edges < 0)
        or np.any(edges >= num_anchors)
        or np.any(edges[:, 0] >= edges[:, 1])
    ):
        raise ValueError(f"{graph_path} edge_index must contain ordered local pairs")
    if num_edges > 1:
        expected_order = np.lexsort((edges[:, 1], edges[:, 0]))
        if not np.array_equal(expected_order, np.arange(num_edges)):
            raise ValueError(
                f"{graph_path} edge_index must be lexicographically sorted"
            )
        if np.any(np.all(edges[1:] == edges[:-1], axis=1)):
            raise ValueError(f"{graph_path} edge_index contains duplicate edges")

    edge_vectors = (
        "edge_distance",
        "edge_distance_weight",
        "edge_color_distance",
        "edge_color_weight",
        "edge_depth_score",
        "edge_combined_weight",
        "edge_view_support_count",
    )
    for field_name in edge_vectors:
        if graph[field_name].shape != (num_edges,):
            raise ValueError(
                f"{graph_path} {field_name} must have shape ({num_edges},)"
            )
    for field_name in edge_vectors[:-1]:
        if not np.isfinite(graph[field_name]).all():
            raise ValueError(f"{graph_path} {field_name} must be finite")
    for field_name in (
        "edge_distance",
        "edge_distance_weight",
        "edge_color_distance",
        "edge_color_weight",
        "edge_depth_score",
        "edge_combined_weight",
    ):
        if np.any(graph[field_name] < 0.0):
            raise ValueError(f"{graph_path} {field_name} must be non-negative")
    if np.any(graph["edge_distance"] > max_distance + epsilon):
        raise ValueError(f"{graph_path} edge_distance exceeds max_distance")
    support_count = graph["edge_view_support_count"]
    if not np.issubdtype(support_count.dtype, np.integer):
        raise ValueError(
            f"{graph_path} edge_view_support_count must be integer-valued"
        )
    support_count = support_count.astype(np.int32)
    support_mask = graph["edge_view_support_mask"].astype(bool)
    if support_mask.shape != (num_edges, num_views):
        raise ValueError(f"{graph_path} edge_view_support_mask has invalid shape")
    if not np.array_equal(support_mask.sum(axis=1), support_count):
        raise ValueError(f"{graph_path} edge view support count is inconsistent")
    if np.any(support_count < min_shared_views):
        raise ValueError(f"{graph_path} contains an edge below min_shared_views")
    for field_name in (
        "edge_endpoint_gap_by_view",
        "edge_depth_jump_by_view",
    ):
        if graph[field_name].shape != (num_edges, num_views):
            raise ValueError(f"{graph_path} {field_name} has invalid shape")
        if not np.isfinite(graph[field_name][support_mask]).all():
            raise ValueError(
                f"{graph_path} {field_name} is invalid on supporting views"
            )

    expected_degree, expected_components, expected_sizes = _component_data(
        num_anchors,
        edges,
    )
    degree = graph["degree"]
    components = graph["component_index"]
    sizes = graph["component_size"]
    if not all(
        np.issubdtype(array.dtype, np.integer)
        for array in (degree, components, sizes)
    ):
        raise ValueError(
            f"{graph_path} degree/component arrays must be integer-valued"
        )
    if not np.array_equal(degree, expected_degree):
        raise ValueError(f"{graph_path} degree is inconsistent with edge_index")
    if not np.array_equal(components, expected_components):
        raise ValueError(
            f"{graph_path} component_index is inconsistent with edge_index"
        )
    if not np.array_equal(sizes, expected_sizes):
        raise ValueError(
            f"{graph_path} component_size is inconsistent with edge_index"
        )
    isolated = graph["isolated_mask"].astype(bool)
    if isolated.shape != (num_anchors,) or not np.array_equal(
        isolated,
        expected_degree == 0,
    ):
        raise ValueError(f"{graph_path} isolated_mask is inconsistent with degree")

    count_values: dict[str, int] = {}
    for field_name in _COUNT_FIELDS:
        count_value = _scalar(graph[field_name], field_name, graph_path)
        if not np.issubdtype(count_value.dtype, np.integer):
            raise ValueError(f"{graph_path} {field_name} must be an integer scalar")
        count_values[field_name] = int(count_value.item())
        if count_values[field_name] < 0:
            raise ValueError(f"{graph_path} {field_name} must be non-negative")
    if count_values["anchor_count"] != num_anchors:
        raise ValueError(f"{graph_path} anchor_count is inconsistent")
    if count_values["retained_edge_count"] != num_edges:
        raise ValueError(f"{graph_path} retained_edge_count is inconsistent")
    if count_values["isolated_anchor_count"] != int(isolated.sum()):
        raise ValueError(f"{graph_path} isolated_anchor_count is inconsistent")
    if count_values["component_count"] != expected_sizes.shape[0]:
        raise ValueError(f"{graph_path} component_count is inconsistent")

    return AnchorGraphViewData(
        mode_index=mode_index,
        freq_hz=freq_hz,
        graph_path=graph_path,
        source_checkpoint=source_checkpoint,
        num_foreground_gaussians=num_gaussians,
        anchor_gaussian_indices=anchor_indices,
        anchor_points_world=points,
        anchor_colors_rgb=colors,
        edge_index=edges.astype(np.int32),
        edge_depth_score=graph["edge_depth_score"].astype(np.float32),
        edge_combined_weight=graph["edge_combined_weight"].astype(np.float32),
        edge_view_support_count=support_count,
        component_index=components.astype(np.int32),
        isolated_mask=isolated,
    )


def load_anchor_graphs_from_manifest(
    manifest_path: Path,
) -> tuple[AnchorGraphViewData, ...]:
    if not manifest_path.is_file():
        raise ValueError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path} must contain a JSON mapping")
    manifest_version = payload.get("version")
    if isinstance(manifest_version, bool) or manifest_version != 1:
        raise ValueError(f"{manifest_path} must be a version 1 modal manifest")
    source_checkpoint = payload.get("source_checkpoint")
    if not isinstance(source_checkpoint, str) or not source_checkpoint:
        raise ValueError(f"{manifest_path} is missing source_checkpoint")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{manifest_path} is missing parameters")
    if parameters.get("anchor_graph_enabled") is not True:
        raise ValueError(
            f"{manifest_path} does not have anchor_graph_enabled=true"
        )
    modes = payload.get("modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError(f"{manifest_path} must contain a non-empty modes list")

    graphs = []
    seen_mode_indices: set[int] = set()
    for mode in modes:
        if not isinstance(mode, dict):
            raise ValueError(f"{manifest_path} modes must contain mappings")
        mode_index = mode.get("mode_index")
        if isinstance(mode_index, bool) or not isinstance(mode_index, int):
            raise ValueError(f"Mode entry in {manifest_path} has invalid mode_index")
        if mode_index in seen_mode_indices:
            raise ValueError(f"{manifest_path} contains duplicate mode_index values")
        seen_mode_indices.add(mode_index)
        freq_hz = mode.get("freq_hz")
        if (
            isinstance(freq_hz, bool)
            or not isinstance(freq_hz, (int, float))
            or not np.isfinite(freq_hz)
        ):
            raise ValueError(f"Mode entry in {manifest_path} has invalid freq_hz")
        graph_value = mode.get("anchor_graph_path")
        if not isinstance(graph_value, str) or not graph_value:
            raise ValueError(
                f"Mode {mode_index} in {manifest_path} is missing anchor_graph_path"
            )
        graph_path = Path(graph_value).expanduser()
        if not graph_path.is_absolute():
            graph_path = manifest_path.parent / graph_path
        graphs.append(
            _load_graph_archive(
                graph_path,
                expected_mode_index=mode_index,
                expected_freq_hz=float(freq_hz),
                expected_source_checkpoint=source_checkpoint,
                manifest_parameters=parameters,
            )
        )
    labels = [graph.label for graph in graphs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{manifest_path} produces duplicate graph mode labels")
    return tuple(graphs)


def load_anchor_graph_source(
    *,
    manifest_path: Path | None,
    graph_path: Path | None,
) -> tuple[AnchorGraphViewData, ...]:
    if (manifest_path is None) == (graph_path is None):
        raise ValueError("Exactly one of manifest_path or graph_path must be provided")
    if manifest_path is not None:
        return load_anchor_graphs_from_manifest(manifest_path)
    assert graph_path is not None
    return (_load_graph_archive(graph_path),)


def anchor_world_center(
    graphs: tuple[AnchorGraphViewData, ...],
) -> np.ndarray:
    points = [
        graph.anchor_points_world
        for graph in graphs
        if graph.anchor_points_world.shape[0]
    ]
    if not points:
        raise ValueError("Cannot center the viewer without anchor points")
    all_points = np.concatenate(points, axis=0).astype(np.float64)
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Anchor world center must be finite (3,)")
    return center.astype(np.float32)


def center_world_points(
    points: np.ndarray,
    world_center: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    center = np.asarray(world_center, dtype=np.float32)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("World points must be finite with final dimension 3")
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("world_center must be finite (3,)")
    return points - center


def _load_gaussian_splat_group(
    arrays: dict[str, np.ndarray],
    sidecar_path: Path,
    group_name: str,
) -> GaussianSplatGroup:
    short_name = "fg" if group_name == "foreground" else "bg"
    count_name = f"num_{group_name}_gaussians"
    required = {count_name} | {
        f"{short_name}_{field_name}" for field_name in _GAUSSIAN_GROUP_FIELDS
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{sidecar_path} missing required fields: {missing}")
    count_value = _scalar(arrays[count_name], count_name, sidecar_path)
    if not np.issubdtype(count_value.dtype, np.integer):
        raise ValueError(f"{sidecar_path} {count_name} must be an integer scalar")
    count = int(count_value.item())
    if count <= 0:
        raise ValueError(f"{sidecar_path} {count_name} must be positive")

    indices_name = f"{short_name}_gaussian_indices"
    indices = arrays[indices_name]
    if indices.shape != (count,) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{sidecar_path} {indices_name} must be integer ({count},)")
    if not np.array_equal(indices, np.arange(count, dtype=indices.dtype)):
        raise ValueError(f"{sidecar_path} {indices_name} must be contiguous")

    centers = arrays[f"{short_name}_centers"].astype(np.float32)
    scales = arrays[f"{short_name}_scales"].astype(np.float32)
    quaternions = arrays[f"{short_name}_quats_wxyz"].astype(np.float32)
    rgbs = arrays[f"{short_name}_rgbs"].astype(np.float32)
    opacities = arrays[f"{short_name}_opacities"].astype(np.float32)
    for field_name, array, expected_shape in (
        ("centers", centers, (count, 3)),
        ("scales", scales, (count, 3)),
        ("quats_wxyz", quaternions, (count, 4)),
        ("rgbs", rgbs, (count, 3)),
        ("opacities", opacities, (count, 1)),
    ):
        if array.shape != expected_shape or not np.isfinite(array).all():
            raise ValueError(
                f"{sidecar_path} {short_name}_{field_name} must be finite "
                f"{expected_shape}"
            )
    if np.any(scales <= 0.0):
        raise ValueError(f"{sidecar_path} {short_name}_scales must be positive")
    quaternion_norms = np.linalg.norm(quaternions.astype(np.float64), axis=1)
    if not np.allclose(quaternion_norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"{sidecar_path} {short_name}_quats_wxyz must be normalized"
        )
    if np.any(rgbs < 0.0) or np.any(rgbs > 1.0):
        raise ValueError(f"{sidecar_path} {short_name}_rgbs must lie in [0,1]")
    if np.any(opacities < 0.0) or np.any(opacities > 1.0):
        raise ValueError(
            f"{sidecar_path} {short_name}_opacities must lie in [0,1]"
        )
    return GaussianSplatGroup(
        centers=centers,
        covariances=gaussian_covariances(scales, quaternions),
        rgbs=rgbs,
        opacities=opacities,
    )


def load_gaussian_visualization_sidecar(
    sidecar_path: Path,
    graphs: tuple[AnchorGraphViewData, ...],
) -> GaussianVisualizationData:
    if not graphs:
        raise ValueError("Gaussian sidecar validation requires at least one graph")
    if not sidecar_path.is_file():
        raise ValueError(
            f"Gaussian visualization sidecar does not exist: {sidecar_path}"
        )
    with np.load(str(sidecar_path), allow_pickle=False) as archive:
        missing = sorted(_GAUSSIAN_BASE_REQUIRED_FIELDS - set(archive.files))
        if missing:
            raise ValueError(f"{sidecar_path} missing required fields: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    version_value = _scalar(arrays["version"], "version", sidecar_path)
    if not np.issubdtype(version_value.dtype, np.integer) or int(
        version_value.item()
    ) != 1:
        raise ValueError(f"{sidecar_path} must be a version 1 Gaussian sidecar")
    if (
        _scalar_string(arrays["point_type"], "point_type", sidecar_path)
        != "static_3dgs_activated_gaussians"
    ):
        raise ValueError(f"{sidecar_path} point_type is incompatible")
    source_checkpoint = _scalar_string(
        arrays["source_checkpoint"],
        "source_checkpoint",
        sidecar_path,
    )
    background_value = _scalar(
        arrays["has_background"],
        "has_background",
        sidecar_path,
    )
    if background_value.dtype != np.dtype(bool):
        raise ValueError(f"{sidecar_path} has_background must be a boolean scalar")
    has_background = bool(background_value.item())

    foreground = _load_gaussian_splat_group(
        arrays,
        sidecar_path,
        "foreground",
    )
    background_fields = {"num_background_gaussians"} | {
        f"bg_{field_name}" for field_name in _GAUSSIAN_GROUP_FIELDS
    }
    if has_background:
        background = _load_gaussian_splat_group(
            arrays,
            sidecar_path,
            "background",
        )
    else:
        unexpected = sorted(background_fields & set(arrays))
        if unexpected:
            raise ValueError(
                f"{sidecar_path} has_background=false but contains {unexpected}"
            )
        background = None

    for graph in graphs:
        if graph.source_checkpoint != source_checkpoint:
            raise ValueError(
                f"{sidecar_path} source_checkpoint does not match "
                f"{graph.graph_path}"
            )
        if graph.num_foreground_gaussians != foreground.centers.shape[0]:
            raise ValueError(
                f"{sidecar_path} foreground count does not match {graph.graph_path}"
            )
        if graph.anchor_gaussian_indices.shape[0]:
            sidecar_anchor_points = foreground.centers[
                graph.anchor_gaussian_indices
            ]
            if not np.allclose(
                sidecar_anchor_points,
                graph.anchor_points_world,
                rtol=1e-6,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{sidecar_path} foreground centers do not match "
                    f"{graph.graph_path} anchors"
                )
    return GaussianVisualizationData(
        sidecar_path=sidecar_path,
        source_checkpoint=source_checkpoint,
        foreground=foreground,
        background=background,
    )


def load_observation_coverage(
    path: Path,
    graphs: tuple[AnchorGraphViewData, ...],
    gaussians: GaussianVisualizationData | None = None,
) -> ObservationCoverageViewData:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(_COVERAGE_REQUIRED_FIELDS - set(archive.files))
        if missing:
            raise ValueError(f"{path} missing required fields: {missing}")
        arrays = {name: archive[name] for name in archive.files}
    version_value = int(_scalar(arrays["version"], "version", path))
    if version_value != 1:
        raise ValueError(f"{path} has unsupported version={version_value}")
    point_type = _scalar_string(arrays["point_type"], "point_type", path)
    if point_type != "foreground_gaussian_observation_coverage":
        raise ValueError(f"{path} has unsupported point_type={point_type!r}")
    source_checkpoint = _scalar_string(
        arrays["source_checkpoint"],
        "source_checkpoint",
        path,
    )
    candidate_method = _scalar_string(
        arrays["candidate_method"],
        "candidate_method",
        path,
    )
    if candidate_method != "rendered_depth_gaussian_contribution":
        raise ValueError(f"{path} has unsupported candidate_method")
    replay_validation = _scalar_string(
        arrays["replay_validation"],
        "replay_validation",
        path,
    )
    if replay_validation != "exact_reference_counts":
        raise ValueError(f"{path} was not validated against reference counts")
    num_points = int(
        _scalar(
            arrays["num_foreground_gaussians"],
            "num_foreground_gaussians",
            path,
        )
    )
    points = np.asarray(arrays["points_world"], dtype=np.float32)
    if points.shape != (num_points, 3) or not np.isfinite(points).all():
        raise ValueError(f"{path} points_world must be finite ({num_points},3)")
    indices = np.asarray(arrays["gaussian_indices"])
    if indices.shape != (num_points,) or not np.array_equal(
        indices,
        np.arange(num_points, dtype=indices.dtype),
    ):
        raise ValueError(f"{path} gaussian_indices must be contiguous")
    k_values = np.asarray(arrays["k_values"], dtype=np.int32)
    if (
        k_values.ndim != 1
        or k_values.shape[0] == 0
        or np.any(k_values <= 0)
        or (k_values.shape[0] > 1 and np.any(np.diff(k_values) <= 0))
    ):
        raise ValueError(f"{path} k_values must be positive and increasing")
    num_k = int(k_values.shape[0])
    baseline_k = int(_scalar(arrays["baseline_k"], "baseline_k", path))
    baseline_index = int(
        _scalar(arrays["baseline_k_index"], "baseline_k_index", path)
    )
    if not 0 <= baseline_index < num_k or k_values[baseline_index] != baseline_k:
        raise ValueError(f"{path} has inconsistent baseline K metadata")
    category_names = tuple(
        str(value) for value in np.asarray(arrays["category_names"]).tolist()
    )
    if category_names != _COVERAGE_CATEGORY_NAMES:
        raise ValueError(f"{path} has unsupported coverage category names")
    categories = np.asarray(arrays["category_by_k"])
    if (
        categories.shape != (num_k, num_points)
        or not np.issubdtype(categories.dtype, np.integer)
        or np.any(categories < 0)
        or np.any(categories >= len(category_names))
    ):
        raise ValueError(f"{path} category_by_k has invalid shape or values")
    category_counts = np.asarray(arrays["category_count_by_k"])
    expected_category_counts = np.stack(
        [
            np.bincount(categories[index], minlength=len(category_names))
            for index in range(num_k)
        ],
        axis=0,
    )
    if not np.array_equal(category_counts, expected_category_counts):
        raise ValueError(f"{path} category_count_by_k is inconsistent")
    view_ids = np.asarray(arrays["view_ids"])
    if view_ids.ndim != 1 or view_ids.shape[0] == 0:
        raise ValueError(f"{path} view_ids must be a non-empty 1-D array")
    num_views = int(view_ids.shape[0])
    expected_shapes = {
        "preselect_hit_count_by_view": (num_points, num_views),
        "positive_hit_count_by_view": (num_points, num_views),
        "selected_hit_count_by_k_view": (num_k, num_points, num_views),
        "best_positive_rank_by_view": (num_points, num_views),
        "best_positive_score_by_view": (num_points, num_views),
        "best_positive_score_ratio_by_view": (num_points, num_views),
        "preselect_view_count": (num_points,),
        "positive_view_count": (num_points,),
        "selected_view_count_by_k": (num_k, num_points),
        "selected_sample_count_by_k": (num_k, num_points),
        "preselect_view_count_histogram": (num_views + 1,),
        "positive_view_count_histogram": (num_views + 1,),
        "selected_view_count_histogram_by_k": (num_k, num_views + 1),
    }
    for name, expected_shape in expected_shapes.items():
        if np.asarray(arrays[name]).shape != expected_shape:
            raise ValueError(
                f"{path} field {name} must have shape {expected_shape}"
            )
    for graph in graphs:
        if graph.source_checkpoint != source_checkpoint:
            raise ValueError(f"{path} source_checkpoint does not match graph")
        if graph.num_foreground_gaussians != num_points:
            raise ValueError(f"{path} foreground count does not match graph")
        if graph.anchor_gaussian_indices.shape[0] and not np.allclose(
            points[graph.anchor_gaussian_indices],
            graph.anchor_points_world,
            rtol=1.0e-6,
            atol=1.0e-5,
        ):
            raise ValueError(f"{path} Gaussian centers do not match graph anchors")
    if gaussians is not None:
        if gaussians.source_checkpoint != source_checkpoint:
            raise ValueError(f"{path} source_checkpoint does not match sidecar")
        if not np.allclose(
            points,
            gaussians.foreground.centers,
            rtol=1.0e-6,
            atol=1.0e-5,
        ):
            raise ValueError(f"{path} Gaussian centers do not match sidecar")
    return ObservationCoverageViewData(
        artifact_path=path,
        source_checkpoint=source_checkpoint,
        points_world=points,
        k_values=k_values,
        category_names=category_names,
        category_by_k=categories.astype(np.int8),
    )


class StaticGaussianViewer:
    def __init__(
        self,
        server: Any,
        gaussians: GaussianVisualizationData,
        *,
        splat_scale: float,
        world_center: np.ndarray,
    ) -> None:
        self.foreground_opacities = gaussians.foreground.opacities.copy()
        self.background_opacities = (
            None
            if gaussians.background is None
            else gaussians.background.opacities.copy()
        )
        self.foreground_handle = server.scene.add_gaussian_splats(
            "/static_gaussians/foreground",
            centers=center_world_points(
                gaussians.foreground.centers,
                world_center,
            ),
            covariances=gaussians.foreground.covariances,
            rgbs=gaussians.foreground.rgbs,
            opacities=gaussians.foreground.opacities,
            scale=splat_scale,
            visible=True,
        )
        self.background_handle = None
        if gaussians.background is not None:
            self.background_handle = server.scene.add_gaussian_splats(
                "/static_gaussians/background",
                centers=center_world_points(
                    gaussians.background.centers,
                    world_center,
                ),
                covariances=gaussians.background.covariances,
                rgbs=gaussians.background.rgbs,
                opacities=gaussians.background.opacities,
                scale=splat_scale,
                visible=True,
            )
        with server.gui.add_folder("Static Gaussian checkpoint"):
            self.show_foreground = server.gui.add_checkbox(
                "Show foreground splats",
                True,
            )
            self.show_background = None
            if self.background_handle is not None:
                self.show_background = server.gui.add_checkbox(
                    "Show background splats",
                    True,
                )
            self.splat_scale = server.gui.add_slider(
                "Gaussian scale",
                min=0.1,
                max=3.0,
                step=0.05,
                initial_value=splat_scale,
            )
            self.gaussian_opacity = server.gui.add_slider(
                "Gaussian opacity",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=1.0,
            )
        self.show_foreground.on_update(self._update)
        if self.show_background is not None:
            self.show_background.on_update(self._update)
        self.splat_scale.on_update(self._update)
        self.gaussian_opacity.on_update(self._update_opacity)
        self._update()

    def _update(self, _event: Any = None) -> None:
        self.foreground_handle.visible = bool(self.show_foreground.value)
        scale = float(self.splat_scale.value)
        self.foreground_handle.scale = scale
        if self.background_handle is not None:
            assert self.show_background is not None
            self.background_handle.visible = bool(self.show_background.value)
            self.background_handle.scale = scale

    def _update_opacity(self, _event: Any = None) -> None:
        multiplier = float(self.gaussian_opacity.value)
        self.foreground_handle.opacities = scale_gaussian_opacities(
            self.foreground_opacities,
            multiplier,
        )
        if self.background_handle is not None:
            assert self.background_opacities is not None
            self.background_handle.opacities = scale_gaussian_opacities(
                self.background_opacities,
                multiplier,
            )


class AnchorGraphViewer:
    def __init__(
        self,
        server: Any,
        graphs: tuple[AnchorGraphViewData, ...],
        *,
        max_visible_edges: int,
        line_width: float,
        anchor_point_size: float,
        isolated_point_size: float,
        world_center: np.ndarray,
    ) -> None:
        if not graphs:
            raise ValueError("Anchor graph viewer requires at least one graph")
        self.server = server
        self.graphs = graphs
        self.labels = tuple(graph.label for graph in graphs)
        self.world_center = np.asarray(world_center, dtype=np.float32)
        if self.world_center.shape != (3,) or not np.isfinite(
            self.world_center
        ).all():
            raise ValueError("world_center must be finite (3,)")
        self._line_handle = None
        self._anchor_handle = None
        self._isolated_handle = None
        self._update_lock = threading.Lock()

        max_edges = max(int(graph.edge_index.shape[0]) for graph in graphs)
        edge_step = max(max_edges // 200, 1)
        with server.gui.add_folder("Anchor structure graph"):
            self.show_graph = server.gui.add_checkbox("Show graph", True)
            self.mode = server.gui.add_dropdown(
                "Mode",
                options=self.labels,
                initial_value=self.labels[0],
            )
            self.edge_color = server.gui.add_dropdown(
                "Edge color",
                options=("component", "depth support", "combined weight"),
                initial_value="component",
            )
            self.max_visible_edges = server.gui.add_slider(
                "Max visible edges",
                min=0,
                max=max(max_edges, 1),
                step=edge_step,
                initial_value=min(max_visible_edges, max_edges),
            )
            self.line_width = server.gui.add_slider(
                "Line width",
                min=0.1,
                max=10.0,
                step=0.1,
                initial_value=line_width,
            )
            self.show_anchors = server.gui.add_checkbox("Show anchors", True)
            self.anchor_point_size = server.gui.add_slider(
                "Anchor point size",
                min=0.0001,
                max=0.008,
                step=0.0001,
                initial_value=anchor_point_size,
            )
            self.show_isolated = server.gui.add_checkbox(
                "Show isolated anchors",
                True,
            )
            self.isolated_point_size = server.gui.add_slider(
                "Isolated-anchor point size",
                min=0.0001,
                max=0.008,
                step=0.0001,
                initial_value=isolated_point_size,
            )
        handles = (
            self.show_graph,
            self.mode,
            self.edge_color,
            self.max_visible_edges,
            self.line_width,
            self.show_anchors,
            self.anchor_point_size,
            self.show_isolated,
            self.isolated_point_size,
        )
        for handle in handles:
            handle.on_update(self._update)
        self._update()

    def _selected_graph(self) -> AnchorGraphViewData:
        selected = str(self.mode.value)
        if selected not in self.labels:
            raise ValueError(f"Unknown anchor graph mode: {selected}")
        return self.graphs[self.labels.index(selected)]

    def _remove_scene_nodes(self) -> None:
        for attribute in ("_line_handle", "_anchor_handle", "_isolated_handle"):
            handle = getattr(self, attribute)
            if handle is not None:
                handle.remove()
                setattr(self, attribute, None)

    def _update(self, _event: Any = None) -> None:
        with self._update_lock:
            self._remove_scene_nodes()
            graph = self._selected_graph()
            centered_points = center_world_points(
                graph.anchor_points_world,
                self.world_center,
            )
            if bool(self.show_graph.value):
                selected_edges = stable_uniform_edge_indices(
                    graph.edge_index.shape[0],
                    int(self.max_visible_edges.value),
                )
                edges = graph.edge_index[selected_edges]
                if edges.shape[0]:
                    color_mode = str(self.edge_color.value)
                    if color_mode == "component":
                        edge_colors = anchor_graph_component_colors(
                            graph.component_index[edges[:, 0]]
                        )
                    elif color_mode == "depth support":
                        edge_colors = anchor_graph_scalar_colors(
                            graph.edge_depth_score[selected_edges]
                        )
                    elif color_mode == "combined weight":
                        edge_colors = anchor_graph_scalar_colors(
                            np.log1p(graph.edge_combined_weight[selected_edges])
                        )
                    else:
                        raise ValueError(
                            f"Unknown anchor graph edge color: {color_mode}"
                        )
                    self._line_handle = self.server.scene.add_line_segments(
                        "/anchor_structure_graph/edges",
                        points=centered_points[edges],
                        colors=np.repeat(edge_colors[:, None, :], 2, axis=1),
                        line_width=float(self.line_width.value),
                    )
            if bool(self.show_anchors.value) and graph.anchor_points_world.shape[0]:
                self._anchor_handle = self.server.scene.add_point_cloud(
                    "/anchor_structure_graph/anchors",
                    points=centered_points,
                    colors=graph.anchor_colors_rgb,
                    point_size=float(self.anchor_point_size.value),
                    point_shape="circle",
                )
            if bool(self.show_isolated.value):
                isolated_points = centered_points[graph.isolated_mask]
                if isolated_points.shape[0]:
                    self._isolated_handle = self.server.scene.add_point_cloud(
                        "/anchor_structure_graph/isolated",
                        points=isolated_points,
                        colors=np.full(
                            (isolated_points.shape[0], 3),
                            [1.0, 0.0, 0.0],
                            dtype=np.float32,
                        ),
                        point_size=float(self.isolated_point_size.value),
                        point_shape="circle",
                    )


class ObservationCoverageViewer:
    def __init__(
        self,
        server: Any,
        coverage: ObservationCoverageViewData,
        *,
        max_visible_points: int,
        point_size: float,
        world_center: np.ndarray,
    ) -> None:
        self.server = server
        self.coverage = coverage
        self.world_center = np.asarray(world_center, dtype=np.float32)
        self.k_labels = tuple(f"K={int(value)}" for value in coverage.k_values)
        self.category_options = ("all categories",) + coverage.category_names
        self._point_handle = None
        self._update_lock = threading.Lock()
        num_points = int(coverage.points_world.shape[0])
        point_step = max(num_points // 200, 1)
        with server.gui.add_folder("Observation coverage"):
            self.show_coverage = server.gui.add_checkbox("Show coverage", True)
            self.k_value = server.gui.add_dropdown(
                "Candidate K",
                options=self.k_labels,
                initial_value=self.k_labels[0],
            )
            self.category = server.gui.add_dropdown(
                "Category",
                options=self.category_options,
                initial_value="all categories",
            )
            self.max_visible_points = server.gui.add_slider(
                "Max visible points",
                min=0,
                max=max(num_points, 1),
                step=point_step,
                initial_value=min(max_visible_points, num_points),
            )
            self.point_size = server.gui.add_slider(
                "Coverage point size",
                min=0.0001,
                max=0.008,
                step=0.0001,
                initial_value=point_size,
            )
        for handle in (
            self.show_coverage,
            self.k_value,
            self.category,
            self.max_visible_points,
            self.point_size,
        ):
            handle.on_update(self._update)
        self._update()

    def _update(self, _event: Any = None) -> None:
        with self._update_lock:
            if self._point_handle is not None:
                self._point_handle.remove()
                self._point_handle = None
            if not bool(self.show_coverage.value):
                return
            selected_k = str(self.k_value.value)
            if selected_k not in self.k_labels:
                raise ValueError(f"Unknown coverage K: {selected_k}")
            k_index = self.k_labels.index(selected_k)
            categories = self.coverage.category_by_k[k_index]
            selected_category = str(self.category.value)
            if selected_category == "all categories":
                gaussian_indices = np.arange(
                    self.coverage.points_world.shape[0],
                    dtype=np.int64,
                )
            elif selected_category in self.coverage.category_names:
                category_index = self.coverage.category_names.index(
                    selected_category
                )
                gaussian_indices = np.flatnonzero(categories == category_index)
            else:
                raise ValueError(
                    f"Unknown coverage category: {selected_category}"
                )
            visible = stable_uniform_indices(
                gaussian_indices.shape[0],
                int(self.max_visible_points.value),
            )
            gaussian_indices = gaussian_indices[visible]
            if gaussian_indices.shape[0] == 0:
                return
            self._point_handle = self.server.scene.add_point_cloud(
                "/observation_coverage/points",
                points=center_world_points(
                    self.coverage.points_world[gaussian_indices],
                    self.world_center,
                ),
                colors=observation_coverage_colors(
                    categories[gaussian_indices]
                ),
                point_size=float(self.point_size.value),
                point_shape="circle",
            )


def _configure_initial_camera(
    server: Any,
    graphs: tuple[AnchorGraphViewData, ...],
    world_center: np.ndarray,
) -> None:
    nonempty = [
        graph.anchor_points_world
        for graph in graphs
        if graph.anchor_points_world.shape[0]
    ]
    if not nonempty:
        return
    points = center_world_points(np.concatenate(nonempty, axis=0), world_center)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    extent = max(float(np.max(maximum - minimum)), 1.0e-3)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)
    server.initial_camera.position = tuple(
        float(value)
        for value in extent * np.asarray([1.2, -1.2, 0.8])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="View observed anchor structure graph artifacts in modern Viser",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        type=Path,
        help="Version-1 modal_modes_manifest.json containing anchor_graph_path fields",
    )
    source.add_argument(
        "--graph-npz",
        type=Path,
        help="Single version-1 anchor graph NPZ artifact",
    )
    parser.add_argument(
        "--gaussian-npz",
        type=Path,
        help=(
            "Optional static Gaussian sidecar produced by "
            "export_static_gaussians_for_viser.py"
        ),
    )
    parser.add_argument(
        "--coverage-npz",
        type=Path,
        help="Optional version-1 Gaussian observation coverage diagnostic",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-visible-edges", type=int, default=20000)
    parser.add_argument("--line-width", type=float, default=1.0)
    parser.add_argument("--anchor-point-size", type=float, default=0.0009)
    parser.add_argument("--isolated-point-size", type=float, default=0.002)
    parser.add_argument("--gaussian-scale", type=float, default=1.0)
    parser.add_argument("--coverage-max-visible-points", type=int, default=50000)
    parser.add_argument("--coverage-point-size", type=float, default=0.0009)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.port <= 0 or args.port > 65535:
        raise ValueError("--port must lie in [1,65535]")
    if args.max_visible_edges < 0:
        raise ValueError("--max-visible-edges must be non-negative")
    if args.coverage_max_visible_points < 0:
        raise ValueError("--coverage-max-visible-points must be non-negative")
    if not np.isfinite(args.line_width) or not 0.1 <= args.line_width <= 10.0:
        raise ValueError("--line-width must lie in [0.1,10.0]")
    if not np.isfinite(args.gaussian_scale) or not 0.1 <= args.gaussian_scale <= 3.0:
        raise ValueError("--gaussian-scale must lie in [0.1,3.0]")
    for name in (
        "anchor_point_size",
        "isolated_point_size",
        "coverage_point_size",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or not 0.0001 <= value <= 0.008:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} must lie in [0.0001,0.008]")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    graphs = load_anchor_graph_source(
        manifest_path=args.manifest,
        graph_path=args.graph_npz,
    )
    world_center = anchor_world_center(graphs)
    gaussians = (
        load_gaussian_visualization_sidecar(args.gaussian_npz, graphs)
        if args.gaussian_npz is not None
        else None
    )
    coverage = (
        load_observation_coverage(args.coverage_npz, graphs, gaussians)
        if args.coverage_npz is not None
        else None
    )

    try:
        import viser
        from viser._scene_api import SceneApi
    except ImportError as exc:
        raise RuntimeError(
            "This standalone viewer requires viser==1.0.30 in its isolated "
            "environment"
        ) from exc
    try:
        viser_version = version("viser")
    except PackageNotFoundError:
        viser_version = "unknown"
    if not hasattr(SceneApi, "add_line_segments"):
        raise RuntimeError(
            "Installed Viser lacks SceneApi.add_line_segments "
            f"(detected version {viser_version}); run this script in the isolated "
            "anchor_graph_viewer environment with viser==1.0.30"
        )
    if gaussians is not None and not hasattr(SceneApi, "add_gaussian_splats"):
        raise RuntimeError(
            "Installed Viser lacks SceneApi.add_gaussian_splats "
            f"(detected version {viser_version})"
        )

    server = viser.ViserServer(
        host=args.host,
        port=args.port,
        label="Anchor structure graph",
    )
    server.scene.set_up_direction("+z")
    _configure_initial_camera(server, graphs, world_center)
    if gaussians is not None:
        StaticGaussianViewer(
            server,
            gaussians,
            splat_scale=args.gaussian_scale,
            world_center=world_center,
        )
    AnchorGraphViewer(
        server,
        graphs,
        max_visible_edges=args.max_visible_edges,
        line_width=args.line_width,
        anchor_point_size=args.anchor_point_size,
        isolated_point_size=args.isolated_point_size,
        world_center=world_center,
    )
    if coverage is not None:
        ObservationCoverageViewer(
            server,
            coverage,
            max_visible_points=args.coverage_max_visible_points,
            point_size=args.coverage_point_size,
            world_center=world_center,
        )
    print(
        "Loaded "
        f"{len(graphs)} mode(s), "
        f"{sum(graph.edge_index.shape[0] for graph in graphs)} edge(s), "
        f"{sum(graph.anchor_points_world.shape[0] for graph in graphs)} anchor(s)."
    )
    print(
        "Viewer world origin is the anchor AABB center: "
        f"{world_center.tolist()}"
    )
    if gaussians is not None:
        background_count = (
            0
            if gaussians.background is None
            else gaussians.background.centers.shape[0]
        )
        print(
            "Loaded static Gaussian sidecar with "
            f"{gaussians.foreground.centers.shape[0]} foreground and "
            f"{background_count} background splat(s)."
        )
    if coverage is not None:
        print(
            f"Loaded observation coverage for {coverage.points_world.shape[0]} "
            f"Gaussians at K={coverage.k_values.tolist()}."
        )
    print(
        f"Viser {viser_version} listening on {server.get_host()}:{server.get_port()}"
    )
    server.sleep_forever()


if __name__ == "__main__":
    main()
