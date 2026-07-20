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


@dataclass(frozen=True)
class AnchorGraphViewData:
    mode_index: int
    freq_hz: float
    graph_path: Path
    source_checkpoint: str
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


def stable_uniform_edge_indices(
    edge_count: int,
    max_visible_edges: int,
) -> np.ndarray:
    if (
        isinstance(edge_count, bool)
        or not isinstance(edge_count, (int, np.integer))
        or edge_count < 0
    ):
        raise ValueError("edge_count must be a non-negative integer")
    if (
        isinstance(max_visible_edges, bool)
        or not isinstance(max_visible_edges, (int, np.integer))
        or max_visible_edges < 0
    ):
        raise ValueError("max_visible_edges must be a non-negative integer")
    visible_count = min(int(edge_count), int(max_visible_edges))
    if visible_count == 0:
        return np.empty((0,), dtype=np.int64)
    if visible_count == int(edge_count):
        return np.arange(edge_count, dtype=np.int64)
    return np.floor(
        np.linspace(
            0,
            edge_count,
            visible_count,
            endpoint=False,
            dtype=np.float64,
        )
    ).astype(np.int64)


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
    ) -> None:
        if not graphs:
            raise ValueError("Anchor graph viewer requires at least one graph")
        self.server = server
        self.graphs = graphs
        self.labels = tuple(graph.label for graph in graphs)
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
                        points=graph.anchor_points_world[edges],
                        colors=np.repeat(edge_colors[:, None, :], 2, axis=1),
                        line_width=float(self.line_width.value),
                    )
            if bool(self.show_anchors.value) and graph.anchor_points_world.shape[0]:
                self._anchor_handle = self.server.scene.add_point_cloud(
                    "/anchor_structure_graph/anchors",
                    points=graph.anchor_points_world,
                    colors=graph.anchor_colors_rgb,
                    point_size=float(self.anchor_point_size.value),
                    point_shape="circle",
                )
            if bool(self.show_isolated.value):
                isolated_points = graph.anchor_points_world[graph.isolated_mask]
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


def _configure_initial_camera(
    server: Any,
    graphs: tuple[AnchorGraphViewData, ...],
) -> None:
    nonempty = [
        graph.anchor_points_world
        for graph in graphs
        if graph.anchor_points_world.shape[0]
    ]
    if not nonempty:
        return
    points = np.concatenate(nonempty, axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    extent = max(float(np.max(maximum - minimum)), 1.0e-3)
    server.initial_camera.look_at = tuple(float(value) for value in center)
    server.initial_camera.position = tuple(
        float(value)
        for value in center + extent * np.asarray([1.2, -1.2, 0.8])
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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-visible-edges", type=int, default=20000)
    parser.add_argument("--line-width", type=float, default=1.0)
    parser.add_argument("--anchor-point-size", type=float, default=0.0009)
    parser.add_argument("--isolated-point-size", type=float, default=0.002)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.port <= 0 or args.port > 65535:
        raise ValueError("--port must lie in [1,65535]")
    if args.max_visible_edges < 0:
        raise ValueError("--max-visible-edges must be non-negative")
    if not np.isfinite(args.line_width) or not 0.1 <= args.line_width <= 10.0:
        raise ValueError("--line-width must lie in [0.1,10.0]")
    for name in ("anchor_point_size", "isolated_point_size"):
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

    server = viser.ViserServer(
        host=args.host,
        port=args.port,
        label="Anchor structure graph",
    )
    server.scene.set_up_direction("+z")
    _configure_initial_camera(server, graphs)
    AnchorGraphViewer(
        server,
        graphs,
        max_visible_edges=args.max_visible_edges,
        line_width=args.line_width,
        anchor_point_size=args.anchor_point_size,
        isolated_point_size=args.isolated_point_size,
    )
    print(
        "Loaded "
        f"{len(graphs)} mode(s), "
        f"{sum(graph.edge_index.shape[0] for graph in graphs)} edge(s), "
        f"{sum(graph.anchor_points_world.shape[0] for graph in graphs)} anchor(s)."
    )
    print(
        f"Viser {viser_version} listening on {server.get_host()}:{server.get_port()}"
    )
    server.sleep_forever()


if __name__ == "__main__":
    main()
