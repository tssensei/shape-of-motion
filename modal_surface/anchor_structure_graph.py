"""Observed full-rank anchor structure graph for modal diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from modal_surface.geometry import bilinear_sample, project_points
from modal_surface.io import save_npz_compressed_atomic
from modal_surface.motion_fill import query_knn_candidates


ANCHOR_STRUCTURE_GRAPH_VERSION = 1
ANCHOR_STRUCTURE_GRAPH_EPSILON = 1.0e-8
_MAD_SCALE = 1.4826
_PROFILE_BATCH_SIZE = 65536


@dataclass(frozen=True)
class AnchorStructureGraphConfig:
    max_neighbors: int = 8
    max_distance: float = 0.0
    color_mad_multiplier: float = 3.0
    depth_mad_multiplier: float = 3.0
    depth_samples: int = 5
    min_shared_views: int = 1
    render_acc_min: float = 0.05
    epsilon: float = ANCHOR_STRUCTURE_GRAPH_EPSILON

    def validate(self, num_anchors: int, num_views: int) -> None:
        if (
            isinstance(self.max_neighbors, (bool, np.bool_))
            or not isinstance(self.max_neighbors, (int, np.integer))
            or self.max_neighbors <= 0
        ):
            raise ValueError("anchor graph max_neighbors must be a positive integer")
        if not np.isfinite(self.max_distance) or self.max_distance <= 0.0:
            raise ValueError("anchor graph max_distance must be finite and positive")
        if not np.isfinite(self.color_mad_multiplier) or self.color_mad_multiplier < 0.0:
            raise ValueError("anchor graph color_mad_multiplier must be finite and non-negative")
        if not np.isfinite(self.depth_mad_multiplier) or self.depth_mad_multiplier < 0.0:
            raise ValueError("anchor graph depth_mad_multiplier must be finite and non-negative")
        if (
            isinstance(self.depth_samples, (bool, np.bool_))
            or not isinstance(self.depth_samples, (int, np.integer))
            or self.depth_samples < 2
        ):
            raise ValueError("anchor graph depth_samples must be an integer at least 2")
        if (
            isinstance(self.min_shared_views, (bool, np.bool_))
            or not isinstance(self.min_shared_views, (int, np.integer))
            or self.min_shared_views <= 0
        ):
            raise ValueError("anchor graph min_shared_views must be a positive integer")
        if self.min_shared_views > num_views:
            raise ValueError(
                "anchor graph min_shared_views cannot exceed the number of views "
                f"({num_views})"
            )
        if not np.isfinite(self.render_acc_min) or not 0.0 <= self.render_acc_min <= 1.0:
            raise ValueError("anchor graph render_acc_min must lie in [0,1]")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("anchor graph epsilon must be finite and positive")


@dataclass(frozen=True)
class AnchorStructureGraph:
    anchor_gaussian_indices: np.ndarray
    anchor_points_world: np.ndarray
    anchor_colors_rgb: np.ndarray
    anchor_observed_view_mask: np.ndarray
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
    config: AnchorStructureGraphConfig


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


def _threshold_pass(values: np.ndarray, threshold: float, epsilon: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(threshold):
        return np.zeros(values.shape, dtype=bool)
    if threshold == 0.0:
        return np.isfinite(values) & (values == 0.0)
    return np.isfinite(values) & (values <= threshold + epsilon)


def _soft_weight(values: np.ndarray, threshold: float, epsilon: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if threshold == 0.0:
        return np.where(values == 0.0, 1.0, 0.0).astype(np.float32)
    return np.exp(-0.5 * np.square(values / max(threshold, epsilon))).astype(np.float32)


def _mutual_knn_edges(
    points: np.ndarray,
    max_neighbors: int,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
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
        [int(dst_i) * num_points + int(src_i) in directed_set for src_i, dst_i in zip(src, dst)],
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
        "nonmutual_rejected_pair_count": len(within_undirected) - int(edge_index.shape[0]),
        "mutual_distance_candidate_count": int(edge_index.shape[0]),
    }


def _observed_view_mask(
    num_points: int,
    num_views: int,
    obs_point_index: np.ndarray,
    obs_view_index: np.ndarray,
    obs_weights: np.ndarray,
    alpha_identifiable_mask: np.ndarray,
) -> np.ndarray:
    point_index = np.asarray(obs_point_index)
    view_index = np.asarray(obs_view_index)
    weights = np.asarray(obs_weights)
    identifiable = np.asarray(alpha_identifiable_mask, dtype=bool)
    if point_index.shape != view_index.shape or point_index.shape != weights.shape:
        raise ValueError("anchor graph observation row arrays must have matching shapes")
    if point_index.ndim != 1 or not np.issubdtype(point_index.dtype, np.integer):
        raise ValueError("anchor graph obs_point_index must be a 1-D integer array")
    if not np.issubdtype(view_index.dtype, np.integer):
        raise ValueError("anchor graph obs_view_index must be an integer array")
    if identifiable.shape != (num_views,):
        raise ValueError(
            f"anchor graph alpha_identifiable_mask must have shape ({num_views},)"
        )
    if np.any(point_index < 0) or np.any(point_index >= num_points):
        raise ValueError("anchor graph obs_point_index is out of range")
    if np.any(view_index < 0) or np.any(view_index >= num_views):
        raise ValueError("anchor graph obs_view_index is out of range")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("anchor graph observation weights must be finite and non-negative")
    result = np.zeros((num_points, num_views), dtype=bool)
    keep = (weights > 0.0) & identifiable[view_index]
    result[point_index[keep], view_index[keep]] = True
    return result


def _bilinear_valid_mask(pixels_xy: np.ndarray, height: int, width: int) -> np.ndarray:
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
    result = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    if np.any(valid_mask):
        result[valid_mask] = bilinear_sample(
            image,
            np.asarray(pixels_xy)[valid_mask].reshape(-1, 2),
        ).astype(np.float32)
    return result


def _component_data(
    num_anchors: int,
    edge_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        root_start = find(int(start))
        root_end = find(int(end))
        if root_start != root_end:
            parent[max(root_start, root_end)] = min(root_start, root_end)
    roots = np.asarray([find(i) for i in range(num_anchors)], dtype=np.int32)
    _, component_index = np.unique(roots, return_inverse=True)
    component_index = component_index.astype(np.int32)
    component_size = np.bincount(component_index, minlength=int(component_index.max()) + 1).astype(np.int32) if num_anchors else np.empty((0,), dtype=np.int32)
    return degree, component_index, component_size, degree == 0


def build_anchor_structure_graph(
    *,
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    anchor_mask: np.ndarray,
    obs_point_index: np.ndarray,
    obs_view_index: np.ndarray,
    obs_weights: np.ndarray,
    alpha_identifiable_mask: np.ndarray,
    view_ids: np.ndarray,
    Ks: np.ndarray,
    world_to_cameras: np.ndarray,
    rendered_depths: list[np.ndarray],
    rendered_accs: list[np.ndarray],
    config: AnchorStructureGraphConfig,
) -> AnchorStructureGraph:
    points = np.asarray(points_world, dtype=np.float32)
    colors = np.asarray(colors_rgb, dtype=np.float32)
    anchors = np.asarray(anchor_mask, dtype=bool)
    view_ids = np.asarray(view_ids).astype(str)
    Ks = np.asarray(Ks, dtype=np.float32)
    world_to_cameras = np.asarray(world_to_cameras, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("anchor graph points_world must be a finite (N,3) array")
    if colors.shape != points.shape or not np.isfinite(colors).all():
        raise ValueError("anchor graph colors_rgb must be a finite (N,3) array")
    if np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("anchor graph colors_rgb must lie in [0,1]")
    if anchors.shape != (points.shape[0],):
        raise ValueError("anchor graph anchor_mask must have shape (N,)")
    num_views = int(view_ids.shape[0])
    if view_ids.ndim != 1 or len(set(view_ids.tolist())) != num_views:
        raise ValueError("anchor graph view_ids must be a unique 1-D string array")
    if Ks.shape != (num_views, 3, 3):
        raise ValueError(f"anchor graph Ks must have shape ({num_views},3,3)")
    if world_to_cameras.shape != (num_views, 4, 4):
        raise ValueError(f"anchor graph world_to_cameras must have shape ({num_views},4,4)")
    if len(rendered_depths) != num_views or len(rendered_accs) != num_views:
        raise ValueError("anchor graph rendered depth/alpha counts must match view_ids")

    anchor_indices = np.flatnonzero(anchors).astype(np.int32)
    anchor_points = points[anchor_indices]
    anchor_colors = colors[anchor_indices]
    config.validate(anchor_indices.shape[0], num_views)
    point_view_mask = _observed_view_mask(
        points.shape[0],
        num_views,
        obs_point_index,
        obs_view_index,
        obs_weights,
        alpha_identifiable_mask,
    )
    anchor_view_mask = point_view_mask[anchor_indices]
    candidate_edges, candidate_distances, knn_counts = _mutual_knn_edges(
        anchor_points,
        config.max_neighbors,
        config.max_distance,
    )
    candidate_count = int(candidate_edges.shape[0])

    lab = (
        cv2.cvtColor(anchor_colors.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        if anchor_colors.shape[0]
        else np.empty((0, 3), dtype=np.float32)
    )
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
    color_pass = _threshold_pass(color_distances, color_threshold, config.epsilon)

    endpoint_gap_by_view = np.full((candidate_count, num_views), np.nan, dtype=np.float32)
    jump_by_view = np.full((candidate_count, num_views), np.nan, dtype=np.float32)
    raw_depth_valid = np.zeros((candidate_count, num_views), dtype=bool)
    endpoint_medians = np.full((num_views,), np.nan, dtype=np.float32)
    endpoint_mads = np.full((num_views,), np.nan, dtype=np.float32)
    endpoint_thresholds = np.full((num_views,), np.nan, dtype=np.float32)
    jump_medians = np.full((num_views,), np.nan, dtype=np.float32)
    jump_mads = np.full((num_views,), np.nan, dtype=np.float32)
    jump_thresholds = np.full((num_views,), np.nan, dtype=np.float32)
    endpoint_gap_per_anchor = np.full((anchor_points.shape[0], num_views), np.nan, dtype=np.float32)
    projected_pixels: list[np.ndarray] = []
    endpoint_surface_valid: list[np.ndarray] = []

    for view_index in range(num_views):
        depth = np.asarray(rendered_depths[view_index], dtype=np.float32)
        acc = np.asarray(rendered_accs[view_index], dtype=np.float32)
        if depth.ndim != 2 or acc.shape != depth.shape:
            raise ValueError(
                f"anchor graph rendered depth/alpha for {view_ids[view_index]} must be matching 2-D arrays"
            )
        pixels, camera_z = project_points(
            anchor_points,
            Ks[view_index],
            world_to_cameras[view_index],
        )
        pixel_valid = _bilinear_valid_mask(pixels, depth.shape[0], depth.shape[1])
        sampled_depth = _sample_valid_pixels(depth, pixels, pixel_valid)
        sampled_acc = _sample_valid_pixels(acc, pixels, pixel_valid)
        surface_valid = (
            anchor_view_mask[:, view_index]
            & pixel_valid
            & np.isfinite(camera_z)
            & (camera_z > 0.0)
            & np.isfinite(sampled_depth)
            & (sampled_depth > 0.0)
            & np.isfinite(sampled_acc)
            & (sampled_acc >= config.render_acc_min)
        )
        gaps = np.full((anchor_points.shape[0],), np.nan, dtype=np.float32)
        gaps[surface_valid] = (
            np.abs(camera_z[surface_valid] - sampled_depth[surface_valid])
            / np.maximum(np.abs(sampled_depth[surface_valid]), config.epsilon)
        ).astype(np.float32)
        endpoint_gap_per_anchor[:, view_index] = gaps
        projected_pixels.append(pixels)
        endpoint_surface_valid.append(surface_valid)

    line_fraction = np.linspace(0.0, 1.0, config.depth_samples, dtype=np.float32)
    for view_index in range(num_views):
        if candidate_count == 0:
            continue
        depth = np.asarray(rendered_depths[view_index], dtype=np.float32)
        acc = np.asarray(rendered_accs[view_index], dtype=np.float32)
        start = candidate_edges[:, 0]
        end = candidate_edges[:, 1]
        common = anchor_view_mask[start, view_index] & anchor_view_mask[end, view_index]
        endpoint_valid = endpoint_surface_valid[view_index][start] & endpoint_surface_valid[view_index][end]
        edge_endpoint_gap = np.maximum(
            endpoint_gap_per_anchor[start, view_index],
            endpoint_gap_per_anchor[end, view_index],
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
            sampled_depth = bilinear_sample(depth, valid_pixels.reshape(-1, 2)).reshape(-1, config.depth_samples)
            sampled_acc = bilinear_sample(acc, valid_pixels.reshape(-1, 2)).reshape(-1, config.depth_samples)
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
                0.5 * (np.abs(sampled_depth[:, :-1]) + np.abs(sampled_depth[:, 1:])),
                config.epsilon,
            )
            jumps = np.max(np.abs(np.diff(sampled_depth, axis=1)) / adjacent_denom, axis=1)
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
            view_depth_scores[support, view_index] = endpoint_weight * jump_weight
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
    distance_weight = (1.0 / np.maximum(final_distances, config.epsilon)).astype(np.float32)
    color_weight = _soft_weight(final_color_distances, color_threshold, config.epsilon)
    combined_weight = (distance_weight * color_weight * final_depth_score).astype(np.float32)
    degree, component_index, component_size, isolated = _component_data(
        anchor_indices.shape[0],
        final_edges,
    )
    candidate_common_view_mask = (
        anchor_view_mask[candidate_edges[:, 0]]
        & anchor_view_mask[candidate_edges[:, 1]]
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
        "anchor_count": int(anchor_indices.shape[0]),
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
            np.count_nonzero(raw_depth_valid & endpoint_pass_all & ~jump_pass_all)
        ),
        "supporting_candidate_view_count": int(np.count_nonzero(view_support)),
        "color_rejected_count": int(np.count_nonzero(~color_pass)),
        "color_retained_count": int(np.count_nonzero(color_pass)),
        "depth_rejected_count": int(np.count_nonzero(color_pass & ~depth_pass)),
        "depth_retained_count": int(np.count_nonzero(color_pass & depth_pass)),
        "retained_edge_count": int(final_edges.shape[0]),
        "isolated_anchor_count": int(np.count_nonzero(isolated)),
        "component_count": int(component_size.shape[0]),
    }
    return AnchorStructureGraph(
        anchor_gaussian_indices=anchor_indices,
        anchor_points_world=anchor_points,
        anchor_colors_rgb=anchor_colors,
        anchor_observed_view_mask=anchor_view_mask,
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
        isolated_mask=isolated,
        view_ids=view_ids,
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


def write_anchor_structure_graph(
    path: str | Path,
    graph: AnchorStructureGraph,
    *,
    mode_index: int,
    freq_hz: float,
    source_checkpoint: str,
    num_foreground_gaussians: int,
) -> Path:
    arrays: dict[str, np.ndarray] = {
        "version": np.array(ANCHOR_STRUCTURE_GRAPH_VERSION, dtype=np.int32),
        "mode_index": np.array(mode_index, dtype=np.int32),
        "freq_hz": np.array(freq_hz, dtype=np.float32),
        "source_checkpoint": np.array(source_checkpoint),
        "num_foreground_gaussians": np.array(num_foreground_gaussians, dtype=np.int32),
        "anchor_gaussian_indices": graph.anchor_gaussian_indices.astype(np.int32),
        "anchor_points_world": graph.anchor_points_world.astype(np.float32),
        "anchor_colors_rgb": graph.anchor_colors_rgb.astype(np.float32),
        "anchor_observed_view_mask": graph.anchor_observed_view_mask.astype(bool),
        "edge_index": graph.edge_index.astype(np.int32),
        "edge_distance": graph.edge_distance.astype(np.float32),
        "edge_distance_weight": graph.edge_distance_weight.astype(np.float32),
        "edge_color_distance": graph.edge_color_distance.astype(np.float32),
        "edge_color_weight": graph.edge_color_weight.astype(np.float32),
        "edge_depth_score": graph.edge_depth_score.astype(np.float32),
        "edge_combined_weight": graph.edge_combined_weight.astype(np.float32),
        "edge_view_support_mask": graph.edge_view_support_mask.astype(bool),
        "edge_view_support_count": graph.edge_view_support_count.astype(np.int32),
        "edge_endpoint_gap_by_view": graph.edge_endpoint_gap_by_view.astype(np.float32),
        "edge_depth_jump_by_view": graph.edge_depth_jump_by_view.astype(np.float32),
        "degree": graph.degree.astype(np.int32),
        "component_index": graph.component_index.astype(np.int32),
        "component_size": graph.component_size.astype(np.int32),
        "isolated_mask": graph.isolated_mask.astype(bool),
        "view_ids": graph.view_ids.astype(str),
        "endpoint_gap_median_by_view": graph.endpoint_gap_median_by_view.astype(np.float32),
        "endpoint_gap_mad_by_view": graph.endpoint_gap_mad_by_view.astype(np.float32),
        "endpoint_gap_threshold_by_view": graph.endpoint_gap_threshold_by_view.astype(np.float32),
        "depth_jump_median_by_view": graph.depth_jump_median_by_view.astype(np.float32),
        "depth_jump_mad_by_view": graph.depth_jump_mad_by_view.astype(np.float32),
        "depth_jump_threshold_by_view": graph.depth_jump_threshold_by_view.astype(np.float32),
        "color_distance_median": np.array(graph.color_distance_median, dtype=np.float32),
        "color_distance_mad": np.array(graph.color_distance_mad, dtype=np.float32),
        "color_distance_threshold": np.array(graph.color_distance_threshold, dtype=np.float32),
        "max_neighbors": np.array(graph.config.max_neighbors, dtype=np.int32),
        "max_distance": np.array(graph.config.max_distance, dtype=np.float32),
        "color_mad_multiplier": np.array(graph.config.color_mad_multiplier, dtype=np.float32),
        "depth_mad_multiplier": np.array(graph.config.depth_mad_multiplier, dtype=np.float32),
        "depth_samples": np.array(graph.config.depth_samples, dtype=np.int32),
        "min_shared_views": np.array(graph.config.min_shared_views, dtype=np.int32),
        "render_acc_min": np.array(graph.config.render_acc_min, dtype=np.float32),
        "epsilon": np.array(graph.config.epsilon, dtype=np.float32),
        "mad_scale": np.array(_MAD_SCALE, dtype=np.float32),
        "knn_policy": np.array("mutual_knn"),
        "depth_source": np.array("static_3dgs_rendered_depth"),
        "color_space": np.array("opencv_float_rgb_to_lab"),
        "distance_weight_method": np.array("inverse_distance"),
        "color_weight_method": np.array("gaussian_adaptive_threshold"),
        "depth_weight_method": np.array("supporting_view_gaussian_score"),
    }
    arrays.update(
        {name: np.array(value, dtype=np.int64) for name, value in graph.counts.items()}
    )
    return save_npz_compressed_atomic(path, arrays)
