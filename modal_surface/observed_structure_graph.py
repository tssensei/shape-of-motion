"""Structure graph over every positively observed foreground Gaussian."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from modal_surface.anchor_structure_graph import (
    ANCHOR_STRUCTURE_GRAPH_EPSILON,
    AnchorStructureGraph,
    AnchorStructureGraphConfig,
    build_anchor_structure_graph,
)
from modal_surface.io import save_npz_compressed_atomic


OBSERVED_STRUCTURE_GRAPH_VERSION = 1
OBSERVED_STRUCTURE_GRAPH_EPSILON = ANCHOR_STRUCTURE_GRAPH_EPSILON
_MAD_SCALE = 1.4826


ObservedStructureGraphConfig = AnchorStructureGraphConfig


@dataclass(frozen=True)
class ObservedStructureGraph:
    node_gaussian_indices: np.ndarray
    node_points_world: np.ndarray
    node_colors_rgb: np.ndarray
    node_observed_view_mask: np.ndarray
    node_observed_view_count: np.ndarray
    topology: AnchorStructureGraph
    counts: Mapping[str, int]


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
    topology = build_anchor_structure_graph(
        points_world=points_world,
        colors_rgb=colors_rgb,
        anchor_mask=node_mask,
        obs_point_index=obs_point_index,
        obs_view_index=obs_view_index,
        obs_weights=obs_weights,
        alpha_identifiable_mask=np.ones(
            (normalized_view_ids.shape[0],),
            dtype=bool,
        ),
        view_ids=normalized_view_ids,
        Ks=Ks,
        world_to_cameras=world_to_cameras,
        rendered_depths=rendered_depths,
        rendered_accs=rendered_accs,
        config=config,
    )
    node_indices = topology.anchor_gaussian_indices
    node_view_mask = point_view_mask[node_indices]
    if not np.array_equal(node_view_mask, topology.anchor_observed_view_mask):
        raise RuntimeError(
            "observed graph topology changed the positive observation view mask"
        )
    node_view_count = node_view_mask.sum(axis=1).astype(np.int32)
    counts = {
        **{
            name: int(value)
            for name, value in topology.counts.items()
            if name not in {"anchor_count", "isolated_anchor_count"}
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
        node_points_world=topology.anchor_points_world,
        node_colors_rgb=topology.anchor_colors_rgb,
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
        "render_acc_min": np.array(topology.config.render_acc_min, dtype=np.float32),
        "epsilon": np.array(topology.config.epsilon, dtype=np.float32),
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
