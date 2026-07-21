"""Structure graph over every positively observed foreground Gaussian."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


OBSERVED_STRUCTURE_GRAPH_VERSION = 2
OBSERVED_STRUCTURE_GRAPH_EPSILON = 1.0e-8
_MAD_SCALE = 1.4826
_PROFILE_BATCH_SIZE = 65536
_OBSERVED_GRAPH_REQUIRED_FIELDS = {
    "version",
    "graph_type",
    "node_selection",
    "mode_index",
    "freq_hz",
    "source_checkpoint",
    "source_observation_path",
    "num_foreground_gaussians",
    "node_gaussian_indices",
    "node_points_world",
    "node_colors_rgb",
    "node_observed_view_mask",
    "node_observed_view_count",
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
    "component_pruned_node_mask",
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
    "min_component_nodes",
    "min_component_edges",
    "render_acc_min",
    "epsilon",
    "mad_scale",
    "knn_policy",
    "depth_source",
    "color_space",
    "distance_weight_method",
    "color_weight_method",
    "depth_weight_method",
    "component_pruning_policy",
    "knn_directed_candidate_count",
    "distance_rejected_directed_count",
    "nonmutual_rejected_pair_count",
    "mutual_distance_candidate_count",
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
    "component_count",
    "observation_row_count",
    "positive_observation_row_count",
    "zero_weight_observation_row_count",
    "node_count",
    "single_view_node_count",
    "multi_view_node_count",
    "isolated_node_count",
    "component_pruned_component_count",
    "component_pruned_node_count",
    "component_pruned_edge_count",
}
_GRAPH_PARAMETER_FIELDS = (
    "version",
    "max_neighbors",
    "max_distance",
    "color_mad_multiplier",
    "depth_mad_multiplier",
    "depth_samples",
    "min_shared_views",
    "min_component_nodes",
    "min_component_edges",
    "render_acc_min",
    "epsilon",
)
_EXPECTED_SEMANTICS = {
    "knn_policy": "mutual_knn",
    "depth_source": "static_3dgs_rendered_depth",
    "color_space": "opencv_float_rgb_to_lab",
    "distance_weight_method": "inverse_distance",
    "color_weight_method": "gaussian_adaptive_threshold",
    "depth_weight_method": "supporting_view_gaussian_score",
    "component_pruning_policy": "remove_edges_keep_observed_nodes_isolated",
}
_OBSERVED_COUNT_FIELDS = {
    "knn_directed_candidate_count",
    "distance_rejected_directed_count",
    "nonmutual_rejected_pair_count",
    "mutual_distance_candidate_count",
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
    "component_count",
    "observation_row_count",
    "positive_observation_row_count",
    "zero_weight_observation_row_count",
    "node_count",
    "single_view_node_count",
    "multi_view_node_count",
    "isolated_node_count",
    "component_pruned_component_count",
    "component_pruned_node_count",
    "component_pruned_edge_count",
}


@dataclass(frozen=True)
class ObservedStructureGraphConfig:
    max_neighbors: int = 8
    max_distance: float = 0.0
    color_mad_multiplier: float = 3.0
    depth_mad_multiplier: float = 3.0
    depth_samples: int = 5
    min_shared_views: int = 1
    min_component_nodes: int = 4
    min_component_edges: int = 3
    render_acc_min: float = 0.05
    epsilon: float = OBSERVED_STRUCTURE_GRAPH_EPSILON

    def validate(self, num_nodes: int, num_views: int) -> None:
        if (
            isinstance(self.max_neighbors, (bool, np.bool_))
            or not isinstance(self.max_neighbors, (int, np.integer))
            or self.max_neighbors <= 0
        ):
            raise ValueError("observed graph max_neighbors must be a positive integer")
        if not np.isfinite(self.max_distance) or self.max_distance <= 0.0:
            raise ValueError("observed graph max_distance must be finite and positive")
        if (
            not np.isfinite(self.color_mad_multiplier)
            or self.color_mad_multiplier < 0.0
        ):
            raise ValueError(
                "observed graph color_mad_multiplier must be finite and non-negative"
            )
        if (
            not np.isfinite(self.depth_mad_multiplier)
            or self.depth_mad_multiplier < 0.0
        ):
            raise ValueError(
                "observed graph depth_mad_multiplier must be finite and non-negative"
            )
        if (
            isinstance(self.depth_samples, (bool, np.bool_))
            or not isinstance(self.depth_samples, (int, np.integer))
            or self.depth_samples < 2
        ):
            raise ValueError(
                "observed graph depth_samples must be an integer at least 2"
            )
        if (
            isinstance(self.min_shared_views, (bool, np.bool_))
            or not isinstance(self.min_shared_views, (int, np.integer))
            or self.min_shared_views <= 0
        ):
            raise ValueError(
                "observed graph min_shared_views must be a positive integer"
            )
        if self.min_shared_views > num_views:
            raise ValueError(
                "observed graph min_shared_views cannot exceed the number of views "
                f"({num_views})"
            )
        for name, value in (
            ("min_component_nodes", self.min_component_nodes),
            ("min_component_edges", self.min_component_edges),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(
                    f"observed graph {name} must be a positive integer"
                )
        if not np.isfinite(self.render_acc_min) or not 0.0 <= self.render_acc_min <= 1.0:
            raise ValueError("observed graph render_acc_min must lie in [0,1]")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("observed graph epsilon must be finite and positive")


@dataclass(frozen=True)
class ObservedStructureTopology:
    node_gaussian_indices: np.ndarray
    node_points_world: np.ndarray
    node_colors_rgb: np.ndarray
    node_observed_view_mask: np.ndarray
    edge_index: np.ndarray
    edge_distance: np.ndarray
    edge_distance_weight: np.ndarray
    edge_color_distance: np.ndarray
    edge_color_weight: np.ndarray
    edge_depth_score: np.ndarray
    edge_combined_weight: np.ndarray
    edge_view_support_mask: np.ndarray
    edge_view_support_count: np.ndarray
    edge_endpoint_gap_by_view: np.ndarray
    edge_depth_jump_by_view: np.ndarray
    degree: np.ndarray
    component_index: np.ndarray
    component_size: np.ndarray
    component_pruned_node_mask: np.ndarray
    isolated_mask: np.ndarray
    view_ids: np.ndarray
    endpoint_gap_median_by_view: np.ndarray
    endpoint_gap_mad_by_view: np.ndarray
    endpoint_gap_threshold_by_view: np.ndarray
    depth_jump_median_by_view: np.ndarray
    depth_jump_mad_by_view: np.ndarray
    depth_jump_threshold_by_view: np.ndarray
    color_distance_median: float
    color_distance_mad: float
    color_distance_threshold: float
    counts: Mapping[str, int]
    config: ObservedStructureGraphConfig


@dataclass(frozen=True)
class ObservedStructureGraph:
    node_gaussian_indices: np.ndarray
    node_points_world: np.ndarray
    node_colors_rgb: np.ndarray
    node_observed_view_mask: np.ndarray
    node_observed_view_count: np.ndarray
    topology: ObservedStructureTopology
    counts: Mapping[str, int]


@dataclass(frozen=True)
class LoadedObservedStructureGraph:
    graph_path: Path
    mode_index: int
    freq_hz: float
    source_checkpoint: str
    source_observation_path: str
    num_foreground_gaussians: int
    view_ids: tuple[str, ...]
    graph: ObservedStructureGraph

    @property
    def label(self) -> str:
        return f"Mode {self.mode_index}: {self.freq_hz:.3f} Hz"

    @property
    def node_gaussian_indices(self) -> np.ndarray:
        return self.graph.node_gaussian_indices

    @property
    def node_points_world(self) -> np.ndarray:
        return self.graph.node_points_world

    @property
    def node_colors_rgb(self) -> np.ndarray:
        return self.graph.node_colors_rgb

    @property
    def node_observed_view_mask(self) -> np.ndarray:
        return self.graph.node_observed_view_mask

    @property
    def edge_index(self) -> np.ndarray:
        return self.graph.topology.edge_index

    @property
    def edge_depth_score(self) -> np.ndarray:
        return self.graph.topology.edge_depth_score

    @property
    def edge_combined_weight(self) -> np.ndarray:
        return self.graph.topology.edge_combined_weight

    @property
    def edge_view_support_count(self) -> np.ndarray:
        return self.graph.topology.edge_view_support_count

    @property
    def component_index(self) -> np.ndarray:
        return self.graph.topology.component_index

    @property
    def component_pruned_node_mask(self) -> np.ndarray:
        return self.graph.topology.component_pruned_node_mask

    @property
    def isolated_mask(self) -> np.ndarray:
        return self.graph.topology.isolated_mask


def _positive_observation_masks(
    *,
    num_points: int,
    num_views: int,
    obs_point_index: np.ndarray,
    obs_view_index: np.ndarray,
    obs_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    point_index = np.asarray(obs_point_index)
    view_index = np.asarray(obs_view_index)
    weights = np.asarray(obs_weights)
    if point_index.shape != view_index.shape or point_index.shape != weights.shape:
        raise ValueError(
            "observed graph observation row arrays must have matching shapes"
        )
    if point_index.ndim != 1 or not np.issubdtype(point_index.dtype, np.integer):
        raise ValueError("observed graph obs_point_index must be a 1-D integer array")
    if not np.issubdtype(view_index.dtype, np.integer):
        raise ValueError("observed graph obs_view_index must be an integer array")
    if np.any(point_index < 0) or np.any(point_index >= num_points):
        raise ValueError("observed graph obs_point_index is out of range")
    if np.any(view_index < 0) or np.any(view_index >= num_views):
        raise ValueError("observed graph obs_view_index is out of range")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError(
            "observed graph observation weights must be finite and non-negative"
        )

    positive = weights > 0.0
    point_view_mask = np.zeros((num_points, num_views), dtype=bool)
    point_view_mask[point_index[positive], view_index[positive]] = True
    return (
        point_view_mask.any(axis=1),
        point_view_mask,
        int(np.count_nonzero(positive)),
    )


def _robust_threshold(
    values: np.ndarray,
    multiplier: float,
) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median, mad, median + float(multiplier) * _MAD_SCALE * mad


def _threshold_pass(
    values: np.ndarray,
    threshold: float,
    epsilon: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(threshold):
        return np.zeros(values.shape, dtype=bool)
    if threshold == 0.0:
        return np.isfinite(values) & (values == 0.0)
    return np.isfinite(values) & (values <= threshold + epsilon)


def _soft_weight(
    values: np.ndarray,
    threshold: float,
    epsilon: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if threshold == 0.0:
        return np.where(values == 0.0, 1.0, 0.0).astype(np.float32)
    return np.exp(
        -0.5 * np.square(values / max(threshold, epsilon))
    ).astype(np.float32)


def _rgb_to_opencv_lab(colors_rgb: np.ndarray) -> np.ndarray:
    import cv2

    colors = np.asarray(colors_rgb, dtype=np.float32)
    if colors.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    return cv2.cvtColor(
        colors.reshape(-1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3)


def _mutual_knn_edges(
    points: np.ndarray,
    max_neighbors: int,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    from modal_surface.motion_fill import query_knn_candidates

    num_points = int(points.shape[0])
    if num_points < 2:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            {
                "knn_directed_candidate_count": 0,
                "distance_rejected_directed_count": 0,
                "nonmutual_rejected_pair_count": 0,
                "mutual_distance_candidate_count": 0,
            },
        )

    k = min(int(max_neighbors), num_points - 1)
    candidates = query_knn_candidates(points, k)
    src = np.repeat(np.arange(num_points, dtype=np.int64), k)
    dst = candidates.neighbor_indices.reshape(-1)
    dist = candidates.neighbor_distances.reshape(-1)
    directed_total = int(src.shape[0])
    within = np.isfinite(dist) & (dist <= float(max_distance))
    src = src[within]
    dst = dst[within]
    dist = dist[within]
    directed_codes = src * num_points + dst
    directed_set = set(int(code) for code in directed_codes.tolist())
    mutual = np.asarray(
        [
            int(dst_i) * num_points + int(src_i) in directed_set
            for src_i, dst_i in zip(src, dst)
        ],
        dtype=bool,
    )
    lower = mutual & (src < dst)
    edge_index = np.column_stack([src[lower], dst[lower]]).astype(np.int32)
    edge_distance = dist[lower].astype(np.float32)
    if edge_index.shape[0]:
        order = np.lexsort((edge_index[:, 1], edge_index[:, 0]))
        edge_index = edge_index[order]
        edge_distance = edge_distance[order]
    within_undirected = {
        (min(int(i), int(j)), max(int(i), int(j))) for i, j in zip(src, dst)
    }
    return edge_index, edge_distance, {
        "knn_directed_candidate_count": directed_total,
        "distance_rejected_directed_count": directed_total - int(src.shape[0]),
        "nonmutual_rejected_pair_count": len(within_undirected)
        - int(edge_index.shape[0]),
        "mutual_distance_candidate_count": int(edge_index.shape[0]),
    }


def _bilinear_valid_mask(
    pixels_xy: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    return (
        np.isfinite(pixels).all(axis=-1)
        & (pixels[..., 0] >= 0.0)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 0] < width - 1)
        & (pixels[..., 1] < height - 1)
    )


def _sample_valid_pixels(
    image: np.ndarray,
    pixels_xy: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    from modal_surface.geometry import bilinear_sample

    result = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    if np.any(valid_mask):
        result[valid_mask] = bilinear_sample(
            image,
            np.asarray(pixels_xy)[valid_mask].reshape(-1, 2),
        ).astype(np.float32)
    return result


def _component_data(
    num_nodes: int,
    edge_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    degree = np.zeros((num_nodes,), dtype=np.int32)
    if edge_index.shape[0]:
        np.add.at(degree, edge_index[:, 0], 1)
        np.add.at(degree, edge_index[:, 1], 1)
    parent = np.arange(num_nodes, dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for start, end in edge_index:
        root_start = find(int(start))
        root_end = find(int(end))
        if root_start != root_end:
            parent[max(root_start, root_end)] = min(root_start, root_end)
    roots = np.asarray([find(i) for i in range(num_nodes)], dtype=np.int32)
    _, component_index = np.unique(roots, return_inverse=True)
    component_index = component_index.astype(np.int32)
    component_size = (
        np.bincount(
            component_index,
            minlength=int(component_index.max()) + 1,
        ).astype(np.int32)
        if num_nodes
        else np.empty((0,), dtype=np.int32)
    )
    return degree, component_index, component_size, degree == 0


def _build_observed_topology(
    *,
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    node_mask: np.ndarray,
    point_view_mask: np.ndarray,
    view_ids: np.ndarray,
    Ks: np.ndarray,
    world_to_cameras: np.ndarray,
    rendered_depths: list[np.ndarray],
    rendered_accs: list[np.ndarray],
    config: ObservedStructureGraphConfig,
) -> ObservedStructureTopology:
    from modal_surface.geometry import bilinear_sample, project_points

    points = np.asarray(points_world, dtype=np.float32)
    colors = np.asarray(colors_rgb, dtype=np.float32)
    nodes = np.asarray(node_mask, dtype=bool)
    observed_views = np.asarray(point_view_mask, dtype=bool)
    normalized_view_ids = np.asarray(view_ids).astype(str)
    intrinsics = np.asarray(Ks, dtype=np.float32)
    extrinsics = np.asarray(world_to_cameras, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("observed graph points_world must be a finite (N,3) array")
    if colors.shape != points.shape or not np.isfinite(colors).all():
        raise ValueError("observed graph colors_rgb must be a finite (N,3) array")
    if np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("observed graph colors_rgb must lie in [0,1]")
    if nodes.shape != (points.shape[0],):
        raise ValueError("observed graph node_mask must have shape (N,)")
    num_views = int(normalized_view_ids.shape[0])
    if (
        normalized_view_ids.ndim != 1
        or len(set(normalized_view_ids.tolist())) != num_views
    ):
        raise ValueError("observed graph view_ids must be a unique 1-D string array")
    if observed_views.shape != (points.shape[0], num_views):
        raise ValueError(
            "observed graph point_view_mask must have shape "
            f"({points.shape[0]},{num_views})"
        )
    if intrinsics.shape != (num_views, 3, 3):
        raise ValueError(f"observed graph Ks must have shape ({num_views},3,3)")
    if extrinsics.shape != (num_views, 4, 4):
        raise ValueError(
            f"observed graph world_to_cameras must have shape ({num_views},4,4)"
        )
    if len(rendered_depths) != num_views or len(rendered_accs) != num_views:
        raise ValueError(
            "observed graph rendered depth/alpha counts must match view_ids"
        )

    node_indices = np.flatnonzero(nodes).astype(np.int32)
    node_points = points[node_indices]
    node_colors = colors[node_indices]
    node_view_mask = observed_views[node_indices]
    config.validate(node_indices.shape[0], num_views)
    candidate_edges, candidate_distances, knn_counts = _mutual_knn_edges(
        node_points,
        config.max_neighbors,
        config.max_distance,
    )
    candidate_count = int(candidate_edges.shape[0])

    lab = _rgb_to_opencv_lab(node_colors)
    color_distances = (
        np.linalg.norm(
            lab[candidate_edges[:, 0]].astype(np.float64)
            - lab[candidate_edges[:, 1]].astype(np.float64),
            axis=1,
        ).astype(np.float32)
        if candidate_count
        else np.empty((0,), dtype=np.float32)
    )
    color_median, color_mad, color_threshold = _robust_threshold(
        color_distances,
        config.color_mad_multiplier,
    )
    color_pass = _threshold_pass(
        color_distances,
        color_threshold,
        config.epsilon,
    )

    endpoint_gap_by_view = np.full(
        (candidate_count, num_views),
        np.nan,
        dtype=np.float32,
    )
    jump_by_view = np.full(
        (candidate_count, num_views),
        np.nan,
        dtype=np.float32,
    )
    raw_depth_valid = np.zeros((candidate_count, num_views), dtype=bool)
    endpoint_medians = np.full((num_views,), np.nan, dtype=np.float32)
    endpoint_mads = np.full((num_views,), np.nan, dtype=np.float32)
    endpoint_thresholds = np.full((num_views,), np.nan, dtype=np.float32)
    jump_medians = np.full((num_views,), np.nan, dtype=np.float32)
    jump_mads = np.full((num_views,), np.nan, dtype=np.float32)
    jump_thresholds = np.full((num_views,), np.nan, dtype=np.float32)
    endpoint_gap_per_node = np.full(
        (node_points.shape[0], num_views),
        np.nan,
        dtype=np.float32,
    )
    projected_pixels: list[np.ndarray] = []
    endpoint_surface_valid: list[np.ndarray] = []

    for view_index in range(num_views):
        depth = np.asarray(rendered_depths[view_index], dtype=np.float32)
        acc = np.asarray(rendered_accs[view_index], dtype=np.float32)
        if depth.ndim != 2 or acc.shape != depth.shape:
            raise ValueError(
                "observed graph rendered depth/alpha for "
                f"{normalized_view_ids[view_index]} must be matching 2-D arrays"
            )
        pixels, camera_z = project_points(
            node_points,
            intrinsics[view_index],
            extrinsics[view_index],
        )
        pixel_valid = _bilinear_valid_mask(
            pixels,
            depth.shape[0],
            depth.shape[1],
        )
        sampled_depth = _sample_valid_pixels(depth, pixels, pixel_valid)
        sampled_acc = _sample_valid_pixels(acc, pixels, pixel_valid)
        surface_valid = (
            node_view_mask[:, view_index]
            & pixel_valid
            & np.isfinite(camera_z)
            & (camera_z > 0.0)
            & np.isfinite(sampled_depth)
            & (sampled_depth > 0.0)
            & np.isfinite(sampled_acc)
            & (sampled_acc >= config.render_acc_min)
        )
        gaps = np.full((node_points.shape[0],), np.nan, dtype=np.float32)
        gaps[surface_valid] = (
            np.abs(camera_z[surface_valid] - sampled_depth[surface_valid])
            / np.maximum(
                np.abs(sampled_depth[surface_valid]),
                config.epsilon,
            )
        ).astype(np.float32)
        endpoint_gap_per_node[:, view_index] = gaps
        projected_pixels.append(pixels)
        endpoint_surface_valid.append(surface_valid)

    line_fraction = np.linspace(
        0.0,
        1.0,
        config.depth_samples,
        dtype=np.float32,
    )
    for view_index in range(num_views):
        if candidate_count == 0:
            continue
        depth = np.asarray(rendered_depths[view_index], dtype=np.float32)
        acc = np.asarray(rendered_accs[view_index], dtype=np.float32)
        start = candidate_edges[:, 0]
        end = candidate_edges[:, 1]
        common = (
            node_view_mask[start, view_index]
            & node_view_mask[end, view_index]
        )
        endpoint_valid = (
            endpoint_surface_valid[view_index][start]
            & endpoint_surface_valid[view_index][end]
        )
        edge_endpoint_gap = np.maximum(
            endpoint_gap_per_node[start, view_index],
            endpoint_gap_per_node[end, view_index],
        )
        endpoint_gap_by_view[:, view_index] = edge_endpoint_gap
        for batch_start in range(0, candidate_count, _PROFILE_BATCH_SIZE):
            batch_end = min(batch_start + _PROFILE_BATCH_SIZE, candidate_count)
            rows = np.arange(batch_start, batch_end, dtype=np.int64)
            batch_possible = common[rows] & endpoint_valid[rows]
            if not np.any(batch_possible):
                continue
            possible_rows = rows[batch_possible]
            p0 = projected_pixels[view_index][start[possible_rows]]
            p1 = projected_pixels[view_index][end[possible_rows]]
            line_pixels = (
                p0[:, None, :] * (1.0 - line_fraction[None, :, None])
                + p1[:, None, :] * line_fraction[None, :, None]
            )
            profile_valid = _bilinear_valid_mask(
                line_pixels,
                depth.shape[0],
                depth.shape[1],
            ).all(axis=1)
            if not np.any(profile_valid):
                continue
            valid_rows = possible_rows[profile_valid]
            valid_pixels = line_pixels[profile_valid]
            sampled_depth = bilinear_sample(
                depth,
                valid_pixels.reshape(-1, 2),
            ).reshape(-1, config.depth_samples)
            sampled_acc = bilinear_sample(
                acc,
                valid_pixels.reshape(-1, 2),
            ).reshape(-1, config.depth_samples)
            samples_valid = (
                np.isfinite(sampled_depth).all(axis=1)
                & (sampled_depth > 0.0).all(axis=1)
                & np.isfinite(sampled_acc).all(axis=1)
                & (sampled_acc >= config.render_acc_min).all(axis=1)
            )
            valid_rows = valid_rows[samples_valid]
            sampled_depth = sampled_depth[samples_valid]
            if valid_rows.size == 0:
                continue
            adjacent_denom = np.maximum(
                0.5
                * (
                    np.abs(sampled_depth[:, :-1])
                    + np.abs(sampled_depth[:, 1:])
                ),
                config.epsilon,
            )
            jumps = np.max(
                np.abs(np.diff(sampled_depth, axis=1)) / adjacent_denom,
                axis=1,
            )
            jump_by_view[valid_rows, view_index] = jumps.astype(np.float32)
            raw_depth_valid[valid_rows, view_index] = True
        endpoint_median, endpoint_mad, endpoint_threshold = _robust_threshold(
            endpoint_gap_by_view[raw_depth_valid[:, view_index], view_index],
            config.depth_mad_multiplier,
        )
        endpoint_medians[view_index] = endpoint_median
        endpoint_mads[view_index] = endpoint_mad
        endpoint_thresholds[view_index] = endpoint_threshold
        median, mad, threshold = _robust_threshold(
            jump_by_view[raw_depth_valid[:, view_index], view_index],
            config.depth_mad_multiplier,
        )
        jump_medians[view_index] = median
        jump_mads[view_index] = mad
        jump_thresholds[view_index] = threshold

    view_support = np.zeros((candidate_count, num_views), dtype=bool)
    view_depth_scores = np.zeros((candidate_count, num_views), dtype=np.float32)
    for view_index in range(num_views):
        endpoint_pass = _threshold_pass(
            endpoint_gap_by_view[:, view_index],
            float(endpoint_thresholds[view_index]),
            config.epsilon,
        )
        jump_pass = _threshold_pass(
            jump_by_view[:, view_index],
            float(jump_thresholds[view_index]),
            config.epsilon,
        )
        support = raw_depth_valid[:, view_index] & endpoint_pass & jump_pass
        view_support[:, view_index] = support
        if np.any(support):
            endpoint_weight = _soft_weight(
                endpoint_gap_by_view[support, view_index],
                float(endpoint_thresholds[view_index]),
                config.epsilon,
            )
            jump_weight = _soft_weight(
                jump_by_view[support, view_index],
                float(jump_thresholds[view_index]),
                config.epsilon,
            )
            view_depth_scores[support, view_index] = (
                endpoint_weight * jump_weight
            )
    support_count = view_support.sum(axis=1).astype(np.int32)
    depth_pass = support_count >= config.min_shared_views
    retained = color_pass & depth_pass
    final_edges = candidate_edges[retained]
    final_distances = candidate_distances[retained]
    final_color_distances = color_distances[retained]
    final_support = view_support[retained]
    final_support_count = support_count[retained]
    final_endpoint_gaps = endpoint_gap_by_view[retained]
    final_jumps = jump_by_view[retained]
    final_depth_score = np.divide(
        view_depth_scores[retained].sum(axis=1),
        final_support_count,
        out=np.zeros(final_support_count.shape, dtype=np.float32),
        where=final_support_count > 0,
    ).astype(np.float32)
    (
        _preprune_degree,
        preprune_component_index,
        preprune_component_size,
        _preprune_isolated,
    ) = _component_data(node_indices.shape[0], final_edges)
    preprune_component_edge_count = np.zeros(
        preprune_component_size.shape,
        dtype=np.int32,
    )
    if final_edges.shape[0]:
        preprune_edge_component = preprune_component_index[final_edges[:, 0]]
        preprune_component_edge_count = np.bincount(
            preprune_edge_component,
            minlength=preprune_component_size.shape[0],
        ).astype(np.int32)
    else:
        preprune_edge_component = np.empty((0,), dtype=np.int32)
    connected_component = preprune_component_edge_count > 0
    pruned_component = connected_component & (
        (preprune_component_size < config.min_component_nodes)
        | (preprune_component_edge_count < config.min_component_edges)
    )
    component_pruned_node_mask = pruned_component[preprune_component_index]
    retained_component_edge = ~pruned_component[preprune_edge_component]
    component_pruned_edge_count = int(
        np.count_nonzero(~retained_component_edge)
    )
    final_edges = final_edges[retained_component_edge]
    final_distances = final_distances[retained_component_edge]
    final_color_distances = final_color_distances[retained_component_edge]
    final_support = final_support[retained_component_edge]
    final_support_count = final_support_count[retained_component_edge]
    final_endpoint_gaps = final_endpoint_gaps[retained_component_edge]
    final_jumps = final_jumps[retained_component_edge]
    final_depth_score = final_depth_score[retained_component_edge]
    distance_weight = (
        1.0 / np.maximum(final_distances, config.epsilon)
    ).astype(np.float32)
    color_weight = _soft_weight(
        final_color_distances,
        color_threshold,
        config.epsilon,
    )
    combined_weight = (
        distance_weight * color_weight * final_depth_score
    ).astype(np.float32)
    degree, component_index, component_size, isolated = _component_data(
        node_indices.shape[0],
        final_edges,
    )
    if np.any(component_pruned_node_mask & ~isolated):
        raise RuntimeError(
            "observed graph component pruning did not isolate every rejected node"
        )
    candidate_common_view_mask = (
        node_view_mask[candidate_edges[:, 0]]
        & node_view_mask[candidate_edges[:, 1]]
        if candidate_count
        else np.empty((0, num_views), dtype=bool)
    )
    endpoint_pass_all = np.zeros((candidate_count, num_views), dtype=bool)
    jump_pass_all = np.zeros((candidate_count, num_views), dtype=bool)
    for view_index in range(num_views):
        endpoint_pass_all[:, view_index] = _threshold_pass(
            endpoint_gap_by_view[:, view_index],
            float(endpoint_thresholds[view_index]),
            config.epsilon,
        )
        jump_pass_all[:, view_index] = _threshold_pass(
            jump_by_view[:, view_index],
            float(jump_thresholds[view_index]),
            config.epsilon,
        )
    counts = {
        **knn_counts,
        "node_count": int(node_indices.shape[0]),
        "shared_observed_candidate_view_count": int(
            np.count_nonzero(candidate_common_view_mask)
        ),
        "raw_depth_valid_candidate_view_count": int(
            np.count_nonzero(raw_depth_valid)
        ),
        "endpoint_rejected_candidate_view_count": int(
            np.count_nonzero(raw_depth_valid & ~endpoint_pass_all)
        ),
        "jump_rejected_candidate_view_count": int(
            np.count_nonzero(
                raw_depth_valid & endpoint_pass_all & ~jump_pass_all
            )
        ),
        "supporting_candidate_view_count": int(np.count_nonzero(view_support)),
        "color_rejected_count": int(np.count_nonzero(~color_pass)),
        "color_retained_count": int(np.count_nonzero(color_pass)),
        "depth_rejected_count": int(np.count_nonzero(color_pass & ~depth_pass)),
        "depth_retained_count": int(np.count_nonzero(color_pass & depth_pass)),
        "retained_edge_count": int(final_edges.shape[0]),
        "isolated_node_count": int(np.count_nonzero(isolated)),
        "component_count": int(component_size.shape[0]),
        "component_pruned_component_count": int(
            np.count_nonzero(pruned_component)
        ),
        "component_pruned_node_count": int(
            np.count_nonzero(component_pruned_node_mask)
        ),
        "component_pruned_edge_count": component_pruned_edge_count,
    }
    return ObservedStructureTopology(
        node_gaussian_indices=node_indices,
        node_points_world=node_points,
        node_colors_rgb=node_colors,
        node_observed_view_mask=node_view_mask,
        edge_index=final_edges,
        edge_distance=final_distances,
        edge_distance_weight=distance_weight,
        edge_color_distance=final_color_distances,
        edge_color_weight=color_weight,
        edge_depth_score=final_depth_score,
        edge_combined_weight=combined_weight,
        edge_view_support_mask=final_support,
        edge_view_support_count=final_support_count,
        edge_endpoint_gap_by_view=final_endpoint_gaps,
        edge_depth_jump_by_view=final_jumps,
        degree=degree,
        component_index=component_index,
        component_size=component_size,
        component_pruned_node_mask=component_pruned_node_mask,
        isolated_mask=isolated,
        view_ids=normalized_view_ids,
        endpoint_gap_median_by_view=endpoint_medians,
        endpoint_gap_mad_by_view=endpoint_mads,
        endpoint_gap_threshold_by_view=endpoint_thresholds,
        depth_jump_median_by_view=jump_medians,
        depth_jump_mad_by_view=jump_mads,
        depth_jump_threshold_by_view=jump_thresholds,
        color_distance_median=color_median,
        color_distance_mad=color_mad,
        color_distance_threshold=color_threshold,
        counts=counts,
        config=config,
    )


def build_observed_structure_graph(
    *,
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    obs_point_index: np.ndarray,
    obs_view_index: np.ndarray,
    obs_weights: np.ndarray,
    view_ids: np.ndarray,
    Ks: np.ndarray,
    world_to_cameras: np.ndarray,
    rendered_depths: list[np.ndarray],
    rendered_accs: list[np.ndarray],
    config: ObservedStructureGraphConfig,
) -> ObservedStructureGraph:
    points = np.asarray(points_world)
    normalized_view_ids = np.asarray(view_ids).astype(str)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("observed graph points_world must have shape (N,3)")
    if normalized_view_ids.ndim != 1 or normalized_view_ids.shape[0] == 0:
        raise ValueError("observed graph view_ids must be a non-empty 1-D array")
    node_mask, point_view_mask, positive_row_count = _positive_observation_masks(
        num_points=int(points.shape[0]),
        num_views=int(normalized_view_ids.shape[0]),
        obs_point_index=obs_point_index,
        obs_view_index=obs_view_index,
        obs_weights=obs_weights,
    )
    topology = _build_observed_topology(
        points_world=points_world,
        colors_rgb=colors_rgb,
        node_mask=node_mask,
        point_view_mask=point_view_mask,
        view_ids=normalized_view_ids,
        Ks=Ks,
        world_to_cameras=world_to_cameras,
        rendered_depths=rendered_depths,
        rendered_accs=rendered_accs,
        config=config,
    )
    node_indices = topology.node_gaussian_indices
    node_view_mask = point_view_mask[node_indices]
    if not np.array_equal(node_view_mask, topology.node_observed_view_mask):
        raise RuntimeError(
            "observed graph topology changed the positive observation view mask"
        )
    node_view_count = node_view_mask.sum(axis=1).astype(np.int32)
    counts = {
        **{
            name: int(value)
            for name, value in topology.counts.items()
            if name not in {"node_count", "isolated_node_count"}
        },
        "observation_row_count": int(np.asarray(obs_weights).shape[0]),
        "positive_observation_row_count": positive_row_count,
        "zero_weight_observation_row_count": int(
            np.asarray(obs_weights).shape[0] - positive_row_count
        ),
        "node_count": int(node_indices.shape[0]),
        "single_view_node_count": int(np.count_nonzero(node_view_count == 1)),
        "multi_view_node_count": int(np.count_nonzero(node_view_count >= 2)),
        "isolated_node_count": int(np.count_nonzero(topology.isolated_mask)),
    }
    return ObservedStructureGraph(
        node_gaussian_indices=node_indices,
        node_points_world=topology.node_points_world,
        node_colors_rgb=topology.node_colors_rgb,
        node_observed_view_mask=node_view_mask,
        node_observed_view_count=node_view_count,
        topology=topology,
        counts=counts,
    )


def write_observed_structure_graph(
    path: str | Path,
    graph: ObservedStructureGraph,
    *,
    mode_index: int,
    freq_hz: float,
    source_checkpoint: str,
    source_observation_path: str,
    num_foreground_gaussians: int,
) -> Path:
    from modal_surface.io import save_npz_compressed_atomic

    topology = graph.topology
    arrays: dict[str, np.ndarray] = {
        "version": np.array(OBSERVED_STRUCTURE_GRAPH_VERSION, dtype=np.int32),
        "graph_type": np.array("foreground_gaussian_observed_structure_graph"),
        "node_selection": np.array("positive_weight_observation_row"),
        "mode_index": np.array(mode_index, dtype=np.int32),
        "freq_hz": np.array(freq_hz, dtype=np.float32),
        "source_checkpoint": np.array(source_checkpoint),
        "source_observation_path": np.array(source_observation_path),
        "num_foreground_gaussians": np.array(
            num_foreground_gaussians,
            dtype=np.int32,
        ),
        "node_gaussian_indices": graph.node_gaussian_indices.astype(np.int32),
        "node_points_world": graph.node_points_world.astype(np.float32),
        "node_colors_rgb": graph.node_colors_rgb.astype(np.float32),
        "node_observed_view_mask": graph.node_observed_view_mask.astype(bool),
        "node_observed_view_count": graph.node_observed_view_count.astype(np.int32),
        "edge_index": topology.edge_index.astype(np.int32),
        "edge_distance": topology.edge_distance.astype(np.float32),
        "edge_distance_weight": topology.edge_distance_weight.astype(np.float32),
        "edge_color_distance": topology.edge_color_distance.astype(np.float32),
        "edge_color_weight": topology.edge_color_weight.astype(np.float32),
        "edge_depth_score": topology.edge_depth_score.astype(np.float32),
        "edge_combined_weight": topology.edge_combined_weight.astype(np.float32),
        "edge_view_support_mask": topology.edge_view_support_mask.astype(bool),
        "edge_view_support_count": topology.edge_view_support_count.astype(np.int32),
        "edge_endpoint_gap_by_view": topology.edge_endpoint_gap_by_view.astype(
            np.float32
        ),
        "edge_depth_jump_by_view": topology.edge_depth_jump_by_view.astype(np.float32),
        "degree": topology.degree.astype(np.int32),
        "component_index": topology.component_index.astype(np.int32),
        "component_size": topology.component_size.astype(np.int32),
        "component_pruned_node_mask": (
            topology.component_pruned_node_mask.astype(bool)
        ),
        "isolated_mask": topology.isolated_mask.astype(bool),
        "view_ids": topology.view_ids.astype(str),
        "endpoint_gap_median_by_view": topology.endpoint_gap_median_by_view.astype(
            np.float32
        ),
        "endpoint_gap_mad_by_view": topology.endpoint_gap_mad_by_view.astype(
            np.float32
        ),
        "endpoint_gap_threshold_by_view": topology.endpoint_gap_threshold_by_view.astype(
            np.float32
        ),
        "depth_jump_median_by_view": topology.depth_jump_median_by_view.astype(
            np.float32
        ),
        "depth_jump_mad_by_view": topology.depth_jump_mad_by_view.astype(np.float32),
        "depth_jump_threshold_by_view": topology.depth_jump_threshold_by_view.astype(
            np.float32
        ),
        "color_distance_median": np.array(
            topology.color_distance_median,
            dtype=np.float32,
        ),
        "color_distance_mad": np.array(topology.color_distance_mad, dtype=np.float32),
        "color_distance_threshold": np.array(
            topology.color_distance_threshold,
            dtype=np.float32,
        ),
        "max_neighbors": np.array(topology.config.max_neighbors, dtype=np.int32),
        "max_distance": np.array(topology.config.max_distance, dtype=np.float32),
        "color_mad_multiplier": np.array(
            topology.config.color_mad_multiplier,
            dtype=np.float32,
        ),
        "depth_mad_multiplier": np.array(
            topology.config.depth_mad_multiplier,
            dtype=np.float32,
        ),
        "depth_samples": np.array(topology.config.depth_samples, dtype=np.int32),
        "min_shared_views": np.array(
            topology.config.min_shared_views,
            dtype=np.int32,
        ),
        "min_component_nodes": np.array(
            topology.config.min_component_nodes,
            dtype=np.int32,
        ),
        "min_component_edges": np.array(
            topology.config.min_component_edges,
            dtype=np.int32,
        ),
        "render_acc_min": np.array(topology.config.render_acc_min, dtype=np.float32),
        "epsilon": np.array(topology.config.epsilon, dtype=np.float32),
        "mad_scale": np.array(_MAD_SCALE, dtype=np.float32),
        "knn_policy": np.array("mutual_knn"),
        "depth_source": np.array("static_3dgs_rendered_depth"),
        "color_space": np.array("opencv_float_rgb_to_lab"),
        "distance_weight_method": np.array("inverse_distance"),
        "color_weight_method": np.array("gaussian_adaptive_threshold"),
        "depth_weight_method": np.array("supporting_view_gaussian_score"),
        "component_pruning_policy": np.array(
            "remove_edges_keep_observed_nodes_isolated"
        ),
    }
    arrays.update(
        {name: np.array(value, dtype=np.int64) for name, value in graph.counts.items()}
    )
    return save_npz_compressed_atomic(path, arrays)


def _artifact_scalar(array: np.ndarray, name: str, path: Path) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{path} {name} must be scalar")
    return value


def _artifact_scalar_string(array: np.ndarray, name: str, path: Path) -> str:
    value = _artifact_scalar(array, name, path).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} {name} must be a non-empty scalar string")
    return value


def _artifact_string_vector(
    array: np.ndarray,
    name: str,
    path: Path,
) -> tuple[str, ...]:
    values = np.asarray(array)
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError(f"{path} {name} must be a non-empty 1-D array")
    normalized = []
    for raw_value in values:
        value = np.asarray(raw_value).item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path} {name} must contain non-empty strings")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{path} {name} must be unique")
    return tuple(normalized)


def _validate_adaptive_threshold_triplet(
    median: np.ndarray,
    mad: np.ndarray,
    threshold: np.ndarray,
    *,
    name: str,
    path: Path,
) -> None:
    median_values = np.asarray(median)
    mad_values = np.asarray(mad)
    threshold_values = np.asarray(threshold)
    if not (
        median_values.shape == mad_values.shape == threshold_values.shape
    ):
        raise ValueError(f"{path} {name} adaptive threshold arrays must match")
    if any(
        not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
        for values in (median_values, mad_values, threshold_values)
    ):
        raise ValueError(f"{path} {name} adaptive thresholds must be real-valued")
    all_nan = (
        np.isnan(median_values)
        & np.isnan(mad_values)
        & np.isnan(threshold_values)
    )
    all_finite = (
        np.isfinite(median_values)
        & np.isfinite(mad_values)
        & np.isfinite(threshold_values)
    )
    if not np.all(all_nan | all_finite):
        raise ValueError(
            f"{path} {name} adaptive thresholds must be all-finite or all-NaN"
        )
    if np.any(median_values[all_finite] < 0.0) or np.any(
        mad_values[all_finite] < 0.0
    ) or np.any(threshold_values[all_finite] < 0.0):
        raise ValueError(f"{path} {name} adaptive thresholds must be non-negative")


def _validate_adaptive_threshold_formula(
    median: np.ndarray,
    mad: np.ndarray,
    threshold: np.ndarray,
    *,
    multiplier: float,
    name: str,
    path: Path,
) -> None:
    median_values = np.asarray(median, dtype=np.float64)
    mad_values = np.asarray(mad, dtype=np.float64)
    threshold_values = np.asarray(threshold, dtype=np.float64)
    finite = np.isfinite(threshold_values)
    expected = median_values[finite] + multiplier * _MAD_SCALE * mad_values[finite]
    if not np.allclose(
        threshold_values[finite],
        expected,
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError(f"{path} {name} adaptive threshold is inconsistent")


def load_observed_structure_graph(
    path: str | Path,
) -> LoadedObservedStructureGraph:
    """Load and strictly validate a version-2 observed structure graph."""
    graph_path = Path(path)
    if not graph_path.is_file():
        raise ValueError(f"Observed graph does not exist: {graph_path}")
    with np.load(str(graph_path), allow_pickle=False) as archive:
        missing = sorted(_OBSERVED_GRAPH_REQUIRED_FIELDS - set(archive.files))
        if missing:
            raise ValueError(f"{graph_path} missing required fields: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    version_value = _artifact_scalar(arrays["version"], "version", graph_path)
    if not np.issubdtype(version_value.dtype, np.integer) or int(
        version_value.item()
    ) != OBSERVED_STRUCTURE_GRAPH_VERSION:
        raise ValueError(f"{graph_path} must be a version 2 observed graph")
    if (
        _artifact_scalar_string(arrays["graph_type"], "graph_type", graph_path)
        != "foreground_gaussian_observed_structure_graph"
    ):
        raise ValueError(f"{graph_path} graph_type is incompatible")
    if (
        _artifact_scalar_string(
            arrays["node_selection"],
            "node_selection",
            graph_path,
        )
        != "positive_weight_observation_row"
    ):
        raise ValueError(f"{graph_path} node_selection is incompatible")
    source_observation_path = _artifact_scalar_string(
        arrays["source_observation_path"],
        "source_observation_path",
        graph_path,
    )
    source_checkpoint = _artifact_scalar_string(
        arrays["source_checkpoint"],
        "source_checkpoint",
        graph_path,
    )
    mode_value = _artifact_scalar(arrays["mode_index"], "mode_index", graph_path)
    if not np.issubdtype(mode_value.dtype, np.integer):
        raise ValueError(f"{graph_path} mode_index must be an integer scalar")
    mode_index = int(mode_value.item())
    if mode_index < 0:
        raise ValueError(f"{graph_path} mode_index must be non-negative")
    freq_value = _artifact_scalar(arrays["freq_hz"], "freq_hz", graph_path)
    if (
        not np.issubdtype(freq_value.dtype, np.number)
        or np.iscomplexobj(freq_value)
        or not np.isfinite(freq_value.item())
    ):
        raise ValueError(f"{graph_path} freq_hz must be a finite real scalar")
    freq_hz = float(freq_value.item())

    for field_name, expected_value in _EXPECTED_SEMANTICS.items():
        if (
            _artifact_scalar_string(arrays[field_name], field_name, graph_path)
            != expected_value
        ):
            raise ValueError(f"{graph_path} {field_name} is incompatible")
    mad_scale = float(
        _artifact_scalar(arrays["mad_scale"], "mad_scale", graph_path).item()
    )
    if not np.isclose(mad_scale, _MAD_SCALE, rtol=1e-6, atol=1e-8):
        raise ValueError(f"{graph_path} mad_scale is incompatible")

    numeric_parameters = {
        field_name: _artifact_scalar(
            arrays[field_name],
            field_name,
            graph_path,
        ).item()
        for field_name in _GRAPH_PARAMETER_FIELDS
    }
    for field_name in (
        "max_neighbors",
        "depth_samples",
        "min_shared_views",
        "min_component_nodes",
        "min_component_edges",
    ):
        if not np.issubdtype(arrays[field_name].dtype, np.integer):
            raise ValueError(f"{graph_path} {field_name} must be an integer scalar")
    config = ObservedStructureGraphConfig(
        max_neighbors=int(numeric_parameters["max_neighbors"]),
        max_distance=float(numeric_parameters["max_distance"]),
        color_mad_multiplier=float(numeric_parameters["color_mad_multiplier"]),
        depth_mad_multiplier=float(numeric_parameters["depth_mad_multiplier"]),
        depth_samples=int(numeric_parameters["depth_samples"]),
        min_shared_views=int(numeric_parameters["min_shared_views"]),
        min_component_nodes=int(numeric_parameters["min_component_nodes"]),
        min_component_edges=int(numeric_parameters["min_component_edges"]),
        render_acc_min=float(numeric_parameters["render_acc_min"]),
        epsilon=float(numeric_parameters["epsilon"]),
    )

    num_gaussians_value = _artifact_scalar(
        arrays["num_foreground_gaussians"],
        "num_foreground_gaussians",
        graph_path,
    )
    if not np.issubdtype(num_gaussians_value.dtype, np.integer):
        raise ValueError(
            f"{graph_path} num_foreground_gaussians must be an integer scalar"
        )
    num_gaussians = int(num_gaussians_value.item())
    if num_gaussians < 0:
        raise ValueError(
            f"{graph_path} num_foreground_gaussians must be non-negative"
        )

    node_indices = arrays["node_gaussian_indices"]
    if node_indices.ndim != 1 or not np.issubdtype(node_indices.dtype, np.integer):
        raise ValueError(f"{graph_path} node_gaussian_indices must be 1-D integers")
    node_indices = node_indices.astype(np.int64)
    if np.any(node_indices < 0) or np.any(node_indices >= num_gaussians):
        raise ValueError(f"{graph_path} node_gaussian_indices are out of range")
    if node_indices.size > 1 and np.any(np.diff(node_indices) <= 0):
        raise ValueError(
            f"{graph_path} node_gaussian_indices must be strictly increasing"
        )
    num_nodes = int(node_indices.shape[0])
    points = arrays["node_points_world"].astype(np.float32)
    colors = arrays["node_colors_rgb"].astype(np.float32)
    if points.shape != (num_nodes, 3) or not np.isfinite(points).all():
        raise ValueError(f"{graph_path} node_points_world must be finite (N,3)")
    if colors.shape != (num_nodes, 3) or not np.isfinite(colors).all():
        raise ValueError(f"{graph_path} node_colors_rgb must be finite (N,3)")
    if np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError(f"{graph_path} node_colors_rgb must lie in [0,1]")

    view_ids = _artifact_string_vector(arrays["view_ids"], "view_ids", graph_path)
    num_views = len(view_ids)
    config.validate(num_nodes, num_views)
    observed_view_mask = arrays["node_observed_view_mask"]
    if (
        observed_view_mask.shape != (num_nodes, num_views)
        or observed_view_mask.dtype != np.bool_
    ):
        raise ValueError(
            f"{graph_path} node_observed_view_mask must be boolean "
            f"({num_nodes},{num_views})"
        )
    observed_view_count = arrays["node_observed_view_count"]
    if (
        observed_view_count.shape != (num_nodes,)
        or not np.issubdtype(observed_view_count.dtype, np.integer)
        or not np.array_equal(observed_view_count, observed_view_mask.sum(axis=1))
        or np.any(observed_view_count <= 0)
    ):
        raise ValueError(f"{graph_path} node_observed_view_count is inconsistent")
    observed_view_count = observed_view_count.astype(np.int32)

    endpoint_median = arrays["endpoint_gap_median_by_view"]
    endpoint_mad = arrays["endpoint_gap_mad_by_view"]
    endpoint_threshold = arrays["endpoint_gap_threshold_by_view"]
    jump_median = arrays["depth_jump_median_by_view"]
    jump_mad = arrays["depth_jump_mad_by_view"]
    jump_threshold = arrays["depth_jump_threshold_by_view"]
    for field_name in (
        "endpoint_gap_median_by_view",
        "endpoint_gap_mad_by_view",
        "endpoint_gap_threshold_by_view",
        "depth_jump_median_by_view",
        "depth_jump_mad_by_view",
        "depth_jump_threshold_by_view",
    ):
        if arrays[field_name].shape != (num_views,):
            raise ValueError(
                f"{graph_path} {field_name} must have shape ({num_views},)"
            )
    _validate_adaptive_threshold_triplet(
        endpoint_median,
        endpoint_mad,
        endpoint_threshold,
        name="endpoint gap",
        path=graph_path,
    )
    _validate_adaptive_threshold_formula(
        endpoint_median,
        endpoint_mad,
        endpoint_threshold,
        multiplier=config.depth_mad_multiplier,
        name="endpoint gap",
        path=graph_path,
    )
    _validate_adaptive_threshold_triplet(
        jump_median,
        jump_mad,
        jump_threshold,
        name="depth jump",
        path=graph_path,
    )
    _validate_adaptive_threshold_formula(
        jump_median,
        jump_mad,
        jump_threshold,
        multiplier=config.depth_mad_multiplier,
        name="depth jump",
        path=graph_path,
    )
    color_median = _artifact_scalar(
        arrays["color_distance_median"],
        "color_distance_median",
        graph_path,
    )
    color_mad = _artifact_scalar(
        arrays["color_distance_mad"],
        "color_distance_mad",
        graph_path,
    )
    color_threshold = _artifact_scalar(
        arrays["color_distance_threshold"],
        "color_distance_threshold",
        graph_path,
    )
    _validate_adaptive_threshold_triplet(
        color_median,
        color_mad,
        color_threshold,
        name="color distance",
        path=graph_path,
    )
    _validate_adaptive_threshold_formula(
        color_median,
        color_mad,
        color_threshold,
        multiplier=config.color_mad_multiplier,
        name="color distance",
        path=graph_path,
    )
    edges = arrays["edge_index"]
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or not np.issubdtype(edges.dtype, np.integer)
    ):
        raise ValueError(f"{graph_path} edge_index must be integer (E,2)")
    edges = edges.astype(np.int64)
    num_edges = int(edges.shape[0])
    if (
        np.any(edges < 0)
        or np.any(edges >= num_nodes)
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
    edge_vector_names = (
        "edge_distance",
        "edge_distance_weight",
        "edge_color_distance",
        "edge_color_weight",
        "edge_depth_score",
        "edge_combined_weight",
        "edge_view_support_count",
    )
    for field_name in edge_vector_names:
        if arrays[field_name].shape != (num_edges,):
            raise ValueError(
                f"{graph_path} {field_name} must have shape ({num_edges},)"
            )
    for field_name in edge_vector_names[:-1]:
        if not np.issubdtype(
            arrays[field_name].dtype, np.number
        ) or np.iscomplexobj(arrays[field_name]):
            raise ValueError(f"{graph_path} {field_name} must be real-valued")
        if not np.isfinite(arrays[field_name]).all():
            raise ValueError(f"{graph_path} {field_name} must be finite")
        if np.any(arrays[field_name] < 0.0):
            raise ValueError(f"{graph_path} {field_name} must be non-negative")
    edge_distance = arrays["edge_distance"].astype(np.float32)
    if np.any(edge_distance > config.max_distance + config.epsilon):
        raise ValueError(f"{graph_path} edge_distance exceeds max_distance")
    expected_distance = (
        np.linalg.norm(
            points[edges[:, 0]].astype(np.float64)
            - points[edges[:, 1]].astype(np.float64),
            axis=1,
        )
        if num_edges
        else np.empty((0,), dtype=np.float64)
    )
    if not np.allclose(edge_distance, expected_distance, rtol=1e-5, atol=1e-7):
        raise ValueError(f"{graph_path} edge_distance is inconsistent with nodes")
    expected_distance_weight = 1.0 / np.maximum(
        expected_distance, config.epsilon
    )
    if not np.allclose(
        arrays["edge_distance_weight"],
        expected_distance_weight,
        rtol=2.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError(f"{graph_path} edge_distance_weight is inconsistent")
    edge_color_distance = arrays["edge_color_distance"].astype(np.float64)
    if num_edges and not np.all(
        _threshold_pass(
            edge_color_distance,
            float(color_threshold.item()),
            config.epsilon,
        )
    ):
        raise ValueError(f"{graph_path} contains an edge above the color threshold")
    expected_color_weight = _soft_weight(
        edge_color_distance,
        float(color_threshold.item()),
        config.epsilon,
    )
    if not np.allclose(
        arrays["edge_color_weight"],
        expected_color_weight,
        rtol=2.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError(f"{graph_path} edge_color_weight is inconsistent")

    support_count = arrays["edge_view_support_count"]
    if not np.issubdtype(support_count.dtype, np.integer):
        raise ValueError(
            f"{graph_path} edge_view_support_count must be integer-valued"
        )
    support_count = support_count.astype(np.int32)
    support_mask = arrays["edge_view_support_mask"]
    if support_mask.dtype != np.bool_ or support_mask.shape != (
        num_edges,
        num_views,
    ):
        raise ValueError(f"{graph_path} edge_view_support_mask has invalid shape")
    if not np.array_equal(support_mask.sum(axis=1), support_count):
        raise ValueError(f"{graph_path} edge view support count is inconsistent")
    if np.any(support_count < config.min_shared_views):
        raise ValueError(f"{graph_path} contains an edge below min_shared_views")
    if num_edges and np.any(
        support_mask
        & ~(
            observed_view_mask[edges[:, 0]]
            & observed_view_mask[edges[:, 1]]
        )
    ):
        raise ValueError(f"{graph_path} edge support uses a non-shared observed view")
    for field_name in ("edge_endpoint_gap_by_view", "edge_depth_jump_by_view"):
        if arrays[field_name].shape != (num_edges, num_views):
            raise ValueError(f"{graph_path} {field_name} has invalid shape")
        if not np.issubdtype(
            arrays[field_name].dtype, np.number
        ) or np.iscomplexobj(arrays[field_name]):
            raise ValueError(f"{graph_path} {field_name} must be real-valued")
        finite_values = arrays[field_name][np.isfinite(arrays[field_name])]
        if np.any(finite_values < 0.0):
            raise ValueError(f"{graph_path} {field_name} contains negative values")
        if np.any(np.isinf(arrays[field_name])):
            raise ValueError(f"{graph_path} {field_name} contains infinite values")
        if not np.isfinite(arrays[field_name][support_mask]).all():
            raise ValueError(
                f"{graph_path} {field_name} is invalid on supporting views"
            )

    endpoint_gap = arrays["edge_endpoint_gap_by_view"].astype(np.float64)
    depth_jump = arrays["edge_depth_jump_by_view"].astype(np.float64)
    expected_support = np.zeros((num_edges, num_views), dtype=bool)
    expected_view_depth_score = np.zeros((num_edges, num_views), dtype=np.float64)
    for view_index in range(num_views):
        expected_support[:, view_index] = _threshold_pass(
            endpoint_gap[:, view_index],
            float(endpoint_threshold[view_index]),
            config.epsilon,
        ) & _threshold_pass(
            depth_jump[:, view_index],
            float(jump_threshold[view_index]),
            config.epsilon,
        )
        selected = expected_support[:, view_index]
        if np.any(selected):
            expected_view_depth_score[selected, view_index] = (
                _soft_weight(
                    endpoint_gap[selected, view_index],
                    float(endpoint_threshold[view_index]),
                    config.epsilon,
                ).astype(np.float64)
                * _soft_weight(
                    depth_jump[selected, view_index],
                    float(jump_threshold[view_index]),
                    config.epsilon,
                ).astype(np.float64)
            )
    if not np.array_equal(support_mask, expected_support):
        raise ValueError(f"{graph_path} edge_view_support_mask is inconsistent")
    expected_depth_score = np.divide(
        expected_view_depth_score.sum(axis=1),
        support_count,
        out=np.zeros((num_edges,), dtype=np.float64),
        where=support_count > 0,
    )
    if not np.allclose(
        arrays["edge_depth_score"],
        expected_depth_score,
        rtol=2.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError(f"{graph_path} edge_depth_score is inconsistent")

    combined_weight = arrays["edge_combined_weight"].astype(np.float64)
    expected_combined_weight = (
        arrays["edge_distance_weight"].astype(np.float64)
        * arrays["edge_color_weight"].astype(np.float64)
        * arrays["edge_depth_score"].astype(np.float64)
    )
    if not np.allclose(
        combined_weight,
        expected_combined_weight,
        rtol=2e-5,
        atol=1e-7,
    ):
        raise ValueError(f"{graph_path} edge_combined_weight is inconsistent")

    expected_degree, expected_components, expected_sizes, expected_isolated = (
        _component_data(num_nodes, edges)
    )
    degree = arrays["degree"]
    components = arrays["component_index"]
    sizes = arrays["component_size"]
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
    component_pruned_node_mask = arrays["component_pruned_node_mask"]
    if (
        component_pruned_node_mask.dtype != np.bool_
        or component_pruned_node_mask.shape != (num_nodes,)
    ):
        raise ValueError(
            f"{graph_path} component_pruned_node_mask must be boolean ({num_nodes},)"
        )
    isolated = arrays["isolated_mask"]
    if (
        isolated.dtype != np.bool_
        or isolated.shape != (num_nodes,)
        or not np.array_equal(isolated, expected_isolated)
    ):
        raise ValueError(f"{graph_path} isolated_mask is inconsistent with degree")
    if np.any(component_pruned_node_mask & ~isolated):
        raise ValueError(
            f"{graph_path} component-pruned nodes must remain isolated"
        )
    final_component_edge_count = np.zeros(expected_sizes.shape, dtype=np.int64)
    if num_edges:
        final_edge_component = expected_components[edges[:, 0]]
        final_component_edge_count = np.bincount(
            final_edge_component,
            minlength=expected_sizes.shape[0],
        )
    connected_component = final_component_edge_count > 0
    if np.any(
        connected_component
        & (
            (expected_sizes < config.min_component_nodes)
            | (final_component_edge_count < config.min_component_edges)
        )
    ):
        raise ValueError(
            f"{graph_path} contains a connected component below its size thresholds"
        )

    count_values: dict[str, int] = {}
    for field_name in _OBSERVED_COUNT_FIELDS:
        count_value = _artifact_scalar(arrays[field_name], field_name, graph_path)
        if not np.issubdtype(count_value.dtype, np.integer):
            raise ValueError(f"{graph_path} {field_name} must be an integer scalar")
        count_values[field_name] = int(count_value.item())
        if count_values[field_name] < 0:
            raise ValueError(f"{graph_path} {field_name} must be non-negative")
    if count_values["node_count"] != num_nodes:
        raise ValueError(f"{graph_path} node_count is inconsistent")
    if count_values["retained_edge_count"] != num_edges:
        raise ValueError(f"{graph_path} retained_edge_count is inconsistent")
    if count_values["isolated_node_count"] != int(isolated.sum()):
        raise ValueError(f"{graph_path} isolated_node_count is inconsistent")
    if count_values["component_count"] != expected_sizes.shape[0]:
        raise ValueError(f"{graph_path} component_count is inconsistent")
    if count_values["single_view_node_count"] != int(
        np.count_nonzero(observed_view_count == 1)
    ):
        raise ValueError(f"{graph_path} single_view_node_count is inconsistent")
    if count_values["multi_view_node_count"] != int(
        np.count_nonzero(observed_view_count >= 2)
    ):
        raise ValueError(f"{graph_path} multi_view_node_count is inconsistent")
    if count_values["single_view_node_count"] + count_values[
        "multi_view_node_count"
    ] != num_nodes:
        raise ValueError(f"{graph_path} observed node counts are inconsistent")
    if count_values["observation_row_count"] != (
        count_values["positive_observation_row_count"]
        + count_values["zero_weight_observation_row_count"]
    ):
        raise ValueError(f"{graph_path} observation row counts are inconsistent")
    if count_values["color_rejected_count"] + count_values[
        "color_retained_count"
    ] != count_values["mutual_distance_candidate_count"]:
        raise ValueError(f"{graph_path} color filtering counts are inconsistent")
    if count_values["depth_rejected_count"] + count_values[
        "depth_retained_count"
    ] != count_values["color_retained_count"]:
        raise ValueError(f"{graph_path} depth filtering counts are inconsistent")
    if count_values["depth_retained_count"] != (
        num_edges + count_values["component_pruned_edge_count"]
    ):
        raise ValueError(f"{graph_path} component pruning edge count is inconsistent")
    if count_values["component_pruned_node_count"] != int(
        np.count_nonzero(component_pruned_node_mask)
    ):
        raise ValueError(f"{graph_path} component pruning node count is inconsistent")
    if count_values["component_pruned_component_count"] == 0 and (
        count_values["component_pruned_node_count"] != 0
        or count_values["component_pruned_edge_count"] != 0
    ):
        raise ValueError(f"{graph_path} component pruning counts are inconsistent")
    if count_values["component_pruned_component_count"] > 0 and (
        count_values["component_pruned_node_count"] == 0
        or count_values["component_pruned_edge_count"] == 0
    ):
        raise ValueError(f"{graph_path} component pruning counts are inconsistent")
    if (
        2 * count_values["component_pruned_component_count"]
        > count_values["component_pruned_node_count"]
        or count_values["component_pruned_component_count"]
        > count_values["component_pruned_edge_count"]
    ):
        raise ValueError(f"{graph_path} component pruning counts are inconsistent")
    if count_values["raw_depth_valid_candidate_view_count"] != (
        count_values["endpoint_rejected_candidate_view_count"]
        + count_values["jump_rejected_candidate_view_count"]
        + count_values["supporting_candidate_view_count"]
    ):
        raise ValueError(f"{graph_path} depth-view filtering counts are inconsistent")
    retained_support_count = int(np.count_nonzero(support_mask))
    if count_values["supporting_candidate_view_count"] < retained_support_count:
        raise ValueError(
            f"{graph_path} supporting_candidate_view_count is inconsistent"
        )

    topology = ObservedStructureTopology(
        node_gaussian_indices=node_indices,
        node_points_world=points,
        node_colors_rgb=colors,
        node_observed_view_mask=observed_view_mask,
        edge_index=edges.astype(np.int32),
        edge_distance=edge_distance,
        edge_distance_weight=arrays["edge_distance_weight"].astype(np.float32),
        edge_color_distance=arrays["edge_color_distance"].astype(np.float32),
        edge_color_weight=arrays["edge_color_weight"].astype(np.float32),
        edge_depth_score=arrays["edge_depth_score"].astype(np.float32),
        edge_combined_weight=arrays["edge_combined_weight"].astype(np.float32),
        edge_view_support_mask=support_mask,
        edge_view_support_count=support_count,
        edge_endpoint_gap_by_view=arrays["edge_endpoint_gap_by_view"].astype(
            np.float32
        ),
        edge_depth_jump_by_view=arrays["edge_depth_jump_by_view"].astype(np.float32),
        degree=degree.astype(np.int32),
        component_index=components.astype(np.int32),
        component_size=sizes.astype(np.int32),
        component_pruned_node_mask=component_pruned_node_mask,
        isolated_mask=isolated,
        view_ids=np.asarray(view_ids),
        endpoint_gap_median_by_view=endpoint_median.astype(np.float32),
        endpoint_gap_mad_by_view=endpoint_mad.astype(np.float32),
        endpoint_gap_threshold_by_view=endpoint_threshold.astype(np.float32),
        depth_jump_median_by_view=jump_median.astype(np.float32),
        depth_jump_mad_by_view=jump_mad.astype(np.float32),
        depth_jump_threshold_by_view=jump_threshold.astype(np.float32),
        color_distance_median=float(color_median.item()),
        color_distance_mad=float(color_mad.item()),
        color_distance_threshold=float(color_threshold.item()),
        counts=count_values,
        config=config,
    )
    graph = ObservedStructureGraph(
        node_gaussian_indices=node_indices,
        node_points_world=points,
        node_colors_rgb=colors,
        node_observed_view_mask=observed_view_mask,
        node_observed_view_count=observed_view_count,
        topology=topology,
        counts=count_values,
    )
    return LoadedObservedStructureGraph(
        graph_path=graph_path,
        mode_index=mode_index,
        freq_hz=freq_hz,
        source_checkpoint=source_checkpoint,
        source_observation_path=source_observation_path,
        num_foreground_gaussians=num_gaussians,
        view_ids=view_ids,
        graph=graph,
    )


def validate_observed_structure_graph_sources(
    loaded: LoadedObservedStructureGraph,
    *,
    points_world: np.ndarray,
    gaussian_indices: np.ndarray,
    source_checkpoint: str,
    source_observation_path: str | Path,
    mode_index: int,
    freq_hz: float,
    view_ids: np.ndarray,
    obs_point_index: np.ndarray,
    obs_view_index: np.ndarray,
    obs_weights: np.ndarray,
    freq_tolerance_hz: float = 1.0e-6,
) -> None:
    """Cross-check a loaded graph against its checkpoint and observation source."""
    if not np.isfinite(freq_tolerance_hz) or freq_tolerance_hz < 0.0:
        raise ValueError("freq_tolerance_hz must be finite and non-negative")
    if loaded.source_checkpoint != str(source_checkpoint):
        raise ValueError(
            f"{loaded.graph_path} source_checkpoint does not match the checkpoint"
        )
    if loaded.source_observation_path != str(source_observation_path):
        raise ValueError(
            f"{loaded.graph_path} source_observation_path does not match the "
            "observation artifact"
        )
    if loaded.mode_index != int(mode_index):
        raise ValueError(f"{loaded.graph_path} mode_index does not match observations")
    if not np.isfinite(freq_hz) or not np.isclose(
        loaded.freq_hz,
        float(freq_hz),
        rtol=0.0,
        atol=freq_tolerance_hz,
    ):
        raise ValueError(f"{loaded.graph_path} frequency does not match observations")

    points = np.asarray(points_world)
    if (
        points.shape != (loaded.num_foreground_gaussians, 3)
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            "observed graph source points_world must be a finite foreground (N,3) "
            "array"
        )
    indices = np.asarray(gaussian_indices)
    expected_indices = np.arange(loaded.num_foreground_gaussians, dtype=np.int64)
    if (
        indices.shape != expected_indices.shape
        or not np.issubdtype(indices.dtype, np.integer)
        or not np.array_equal(indices, expected_indices)
    ):
        raise ValueError(
            "observed graph source gaussian_indices must be the canonical foreground "
            "index order"
        )
    node_source_points = points[loaded.node_gaussian_indices]
    if not np.allclose(
        node_source_points,
        loaded.node_points_world,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(
            f"{loaded.graph_path} node positions do not match source points_world"
        )

    normalized_view_ids = tuple(np.asarray(view_ids).astype(str).tolist())
    if normalized_view_ids != loaded.view_ids:
        raise ValueError(f"{loaded.graph_path} view_ids do not match observations")
    node_mask, point_view_mask, positive_row_count = _positive_observation_masks(
        num_points=loaded.num_foreground_gaussians,
        num_views=len(loaded.view_ids),
        obs_point_index=obs_point_index,
        obs_view_index=obs_view_index,
        obs_weights=obs_weights,
    )
    expected_node_indices = np.flatnonzero(node_mask)
    if not np.array_equal(expected_node_indices, loaded.node_gaussian_indices):
        raise ValueError(
            f"{loaded.graph_path} nodes do not match positive observation rows"
        )
    if not np.array_equal(
        point_view_mask[expected_node_indices],
        loaded.node_observed_view_mask,
    ):
        raise ValueError(
            f"{loaded.graph_path} node view mask does not match observations"
        )
    observation_row_count = int(np.asarray(obs_weights).shape[0])
    if loaded.graph.counts["observation_row_count"] != observation_row_count:
        raise ValueError(
            f"{loaded.graph_path} observation row count does not match observations"
        )
    if loaded.graph.counts["positive_observation_row_count"] != positive_row_count:
        raise ValueError(
            f"{loaded.graph_path} positive observation row count does not match"
        )
    if loaded.graph.counts["zero_weight_observation_row_count"] != (
        observation_row_count - positive_row_count
    ):
        raise ValueError(
            f"{loaded.graph_path} zero-weight observation row count does not match"
        )
