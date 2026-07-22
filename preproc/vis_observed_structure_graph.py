"""Standalone browser viewer for observed Gaussian structure graph artifacts.

This tool intentionally depends only on NumPy and a modern Viser release. It is
designed to run in an isolated environment instead of the legacy Nerfview
environment used by the main Shape-of-Motion viewer.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from modal_surface.io import load_view_config
from modal_surface.observed_structure_graph import (
    LoadedObservedStructureGraph as ObservedGraphViewData,
    load_observed_structure_graph,
)

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

_RESIDUAL_REQUIRED_FIELDS = {
    "version",
    "point_type",
    "source_checkpoint",
    "source_observation_path",
    "source_solver_diagnostics_path",
    "num_foreground_gaussians",
    "gaussian_indices",
    "points_world",
    "view_ids",
    "mode_index",
    "freq_hz",
    "alphas",
    "alpha_identifiable_mask",
    "point_observable_rank",
    "point_condition",
    "point_distinct_view_count",
    "point_distinct_valid_view_count",
    "point_precompletion_residual",
    "staged_point_solution_status",
    "anchor_residual_threshold",
    "anchor_condition_max",
    "view_sample_count",
    "view_effective_weight",
    "view_mode_mean",
    "view_signal_energy",
    "view_within_sse",
    "view_cross_sse",
    "view_within_residual",
    "view_cross_residual",
    "point_effective_weight",
    "point_signal_energy",
    "point_signal_rms",
    "point_within_sse",
    "point_cross_sse",
    "point_total_sse",
    "point_within_residual",
    "point_cross_residual",
    "point_replayed_total_residual",
    "point_within_fraction",
    "point_worst_within_view",
    "point_worst_cross_view",
    "selected_multiview_mask",
    "residual_candidate_mask",
    "residual_rejected_mask",
    "anchor_mask",
    "low_modal_energy_mask",
    "residual_source_class",
    "residual_source_names",
    "low_modal_energy_threshold",
    "dominance_ratio",
    "mad_scale",
    "effective_weight_method",
    "decomposition_method",
    "replay_validation",
}

_RESIDUAL_SOURCE_NAMES = (
    "other",
    "accepted_anchor",
    "low_modal_energy",
    "within_view_dominated",
    "cross_view_dominated",
    "mixed",
)

_RESIDUAL_SOURCE_COLORS = np.asarray(
    (
        (0.45, 0.45, 0.45),
        (0.0, 0.45, 1.0),
        (0.8, 0.0, 0.8),
        (1.0, 0.15, 0.0),
        (0.0, 0.8, 0.2),
        (1.0, 0.65, 0.0),
    ),
    dtype=np.float32,
)

_STAGED_ANCHOR_STATUS = 0
_STAGED_PARTIAL_STATUS = 3
_STAGED_POINT_STATUS_COUNT = 8
_PARTIAL_POINT_COLOR = np.asarray((1.0, 0.0, 1.0), dtype=np.float32)
_RIGID_LATENT_REQUIRED_FIELDS = {
    "points_world",
    "phi",
    "gaussian_indices",
    "freq_hz",
    "mode_index",
    "obs_count_per_point",
    "point_type",
    "source_checkpoint",
}
_RIGID_DIAGNOSTIC_REQUIRED_FIELDS = {
    "solver_method",
    "solver_diagnostics_type",
    "rigidity_model",
    "rigid_component_connectivity_policy",
    "rigid_component_graph_path",
    "rigid_component_graph_source_path",
    "rigid_component_seed_policy",
    "rigid_seed_min_valid_views",
    "rigid_seed_min_singular_ratio",
    "rigid_seed_max_finite_drift",
    "rigid_seed_mask",
    "observed_mask",
    "fill_target_mask",
    "trusted_rigid_seed_mask",
    "effective_fill_target_mask",
    "completion_mask",
    "point_component_index",
    "rigid_phi_pre_fill",
    "trusted_rigid_phi_pre_fill",
    "final_phi",
    "component_graph_index",
    "component_node_count",
    "component_edge_count",
    "component_distinct_valid_view_count",
    "component_singular_values",
    "component_singular_ratio",
    "component_rank",
    "component_rank_deficient_mask",
    "component_seed_retained_mask",
    "component_valid_view_rejected_mask",
    "component_singular_rejected_mask",
    "component_finite_drift_rejected_mask",
    "component_normalized_weighted_residual",
    "component_finite_drift_max",
    "edge_component_index",
    "edge_finite_drift_max",
}
_MOTION_FILL_ROLE_NAMES = (
    "fixed_anchor",
    "constrained_variable",
    "free_variable",
    "excluded",
)
_RIGID_SEED_COLOR = np.asarray((0.0, 1.0, 1.0), dtype=np.float32)
_COMPLETED_FILL_COLOR = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
_UNRESOLVED_FILL_COLOR = np.asarray((1.0, 0.0, 1.0), dtype=np.float32)
_RIGID_DIAGNOSTIC_NORMAL_COLOR = np.asarray(
    (0.35, 0.35, 0.35), dtype=np.float32
)
_RIGID_DIAGNOSTIC_ANOMALY_COLOR = np.asarray(
    (1.0, 0.0, 0.0), dtype=np.float32
)
_QUARANTINED_RIGID_COLOR = np.asarray((1.0, 0.55, 0.0), dtype=np.float32)
_SINGLE_VIEW_RIGID_FILL_COLOR = np.asarray((0.15, 0.45, 1.0), dtype=np.float32)
_PARTIAL_MOTION_REJECTION_COLOR = np.asarray(
    (1.0, 0.55, 0.0), dtype=np.float32
)
_PARTIAL_BOTH_REJECTION_COLOR = np.asarray((1.0, 0.0, 1.0), dtype=np.float32)


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


@dataclass(frozen=True)
class AnchorResidualViewData:
    artifact_path: Path
    source_checkpoint: str
    mode_index: int
    freq_hz: float
    points_world: np.ndarray
    view_ids: tuple[str, ...]
    point_precompletion_residual: np.ndarray
    point_within_residual: np.ndarray
    point_cross_residual: np.ndarray
    point_within_fraction: np.ndarray
    point_signal_rms: np.ndarray
    point_worst_within_view: np.ndarray
    point_worst_cross_view: np.ndarray
    selected_multiview_mask: np.ndarray
    residual_rejected_mask: np.ndarray
    anchor_mask: np.ndarray
    partial_mask: np.ndarray
    residual_source_class: np.ndarray


@dataclass(frozen=True)
class RigidModeViewData:
    manifest_path: Path
    mode_index: int
    freq_hz: float
    latent_path: Path
    diagnostics_path: Path
    graph_path: Path
    graph_source_path: str
    points_world: np.ndarray
    phi: np.ndarray
    raw_rigid_phi: np.ndarray
    rigid_seed_mask: np.ndarray
    quarantined_rigid_mask: np.ndarray
    single_view_rigid_fill_mask: np.ndarray
    completed_fill_mask: np.ndarray
    unresolved_fill_mask: np.ndarray
    point_component_index: np.ndarray
    component_normalized_weighted_residual: np.ndarray
    component_rank: np.ndarray
    component_distinct_valid_view_count: np.ndarray
    component_singular_ratio: np.ndarray
    component_motion_rms: np.ndarray
    component_finite_drift_max: np.ndarray
    single_view_component_completion_mask: np.ndarray
    single_view_component_observable_rank: np.ndarray | None
    single_view_component_fill_nullity: np.ndarray | None
    single_view_component_trusted_knn_edge_count: np.ndarray | None
    single_view_component_ray_motion_ratio: np.ndarray | None
    single_view_component_normalized_motion_rms: np.ndarray | None
    single_view_component_finite_drift_rejected_mask: np.ndarray | None
    single_view_component_motion_rms_rejected_mask: np.ndarray | None
    single_view_component_postfill_retained_mask: np.ndarray | None
    edge_component_index: np.ndarray
    edge_finite_drift_max: np.ndarray


@dataclass(frozen=True)
class SourceCameraViewData:
    label: str
    image_width: int
    image_height: int
    K: np.ndarray
    world_to_camera: np.ndarray


@dataclass(frozen=True)
class RigidManifestViewData:
    manifest_path: Path
    source_checkpoint: str
    motion_fill_enabled: bool
    rigid_motion_fill_stage: str
    rigid_seed_min_valid_views: int
    rigid_seed_min_singular_ratio: float
    rigid_seed_max_finite_drift: float
    rigid_single_view_max_normalized_motion_rms: float | None
    source_cameras: tuple[SourceCameraViewData, ...]
    modes: tuple[RigidModeViewData, ...]


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


def anchor_residual_source_colors(source_class: np.ndarray) -> np.ndarray:
    values = np.asarray(source_class)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Residual source class must be a 1-D integer array")
    if np.any(values < 0) or np.any(values >= len(_RESIDUAL_SOURCE_NAMES)):
        raise ValueError("Residual source class contains an unknown value")
    return _RESIDUAL_SOURCE_COLORS[values]


def anchor_residual_fraction_colors(values: np.ndarray) -> np.ndarray:
    fraction = np.asarray(values, dtype=np.float32)
    if fraction.ndim != 1 or not np.isfinite(fraction).all():
        raise ValueError("Residual fraction must be a finite 1-D array")
    fraction = np.clip(fraction, 0.0, 1.0)
    return np.column_stack(
        [fraction, np.full(fraction.shape, 0.2), 1.0 - fraction]
    ).astype(np.float32)


def anchor_residual_scalar_colors(
    values: np.ndarray,
    *,
    logarithmic: bool,
) -> np.ndarray:
    metric = np.asarray(values, dtype=np.float32)
    if metric.ndim != 1:
        raise ValueError("Residual metric must be a 1-D array")
    colors = np.full((metric.shape[0], 3), 0.45, dtype=np.float32)
    finite = np.isfinite(metric)
    if np.any(finite):
        finite_values = metric[finite]
        if logarithmic:
            if np.any(finite_values < 0.0):
                raise ValueError("Logarithmic residual metric must be non-negative")
            finite_values = np.log1p(finite_values)
        colors[finite] = graph_scalar_colors(finite_values)
    return colors


def anchor_residual_view_colors(view_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(view_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Worst-view index must be a 1-D integer array")
    colors = np.full((indices.shape[0], 3), 0.45, dtype=np.float32)
    valid = indices >= 0
    if np.any(valid):
        colors[valid] = graph_component_colors(indices[valid])
    return colors


def staged_partial_mask(point_status: np.ndarray) -> np.ndarray:
    status = np.asarray(point_status)
    if status.ndim != 1 or not np.issubdtype(status.dtype, np.integer):
        raise ValueError("Staged point status must be a 1-D integer array")
    if np.any(status < 0) or np.any(status >= _STAGED_POINT_STATUS_COUNT):
        raise ValueError("Staged point status contains an unknown value")
    return status == _STAGED_PARTIAL_STATUS


def graph_component_colors(component_index: np.ndarray) -> np.ndarray:
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


def graph_scalar_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Graph color metric must be a finite 1-D array")
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


def rigid_component_singular_ratio(
    singular_values: np.ndarray,
    component_rank: np.ndarray,
) -> np.ndarray:
    singular = np.asarray(singular_values, dtype=np.float64)
    rank = np.asarray(component_rank)
    if singular.ndim != 2 or singular.shape[1] != 6:
        raise ValueError("Rigid component singular values must have shape (C,6)")
    if not np.isfinite(singular).all() or np.any(singular < 0.0):
        raise ValueError(
            "Rigid component singular values must be finite and non-negative"
        )
    if np.any(singular[:, 1:] > singular[:, :-1]):
        raise ValueError("Rigid component singular values must be non-increasing")
    if (
        rank.shape != (singular.shape[0],)
        or not np.issubdtype(rank.dtype, np.integer)
        or np.any(rank < 0)
        or np.any(rank > 6)
    ):
        raise ValueError("Rigid component rank is invalid")
    ratio = np.zeros((singular.shape[0],), dtype=np.float64)
    full_rank = (rank == 6) & (singular[:, 0] > 0.0)
    ratio[full_rank] = singular[full_rank, 5] / singular[full_rank, 0]
    if not np.isfinite(ratio).all() or np.any((ratio < 0.0) | (ratio > 1.0)):
        raise ValueError("Rigid component singular ratio is invalid")
    return ratio.astype(np.float32)


def rigid_component_motion_rms(
    phi: np.ndarray,
    point_component_index: np.ndarray,
    num_components: int,
) -> np.ndarray:
    field = np.asarray(phi)
    component_index = np.asarray(point_component_index)
    if (
        field.ndim != 2
        or field.shape[1] != 3
        or not np.issubdtype(field.dtype, np.complexfloating)
        or not np.isfinite(field).all()
    ):
        raise ValueError("Rigid component phi must be finite complex (N,3)")
    if (
        isinstance(num_components, bool)
        or not isinstance(num_components, (int, np.integer))
        or num_components <= 0
    ):
        raise ValueError("num_components must be a positive integer")
    if (
        component_index.shape != (field.shape[0],)
        or not np.issubdtype(component_index.dtype, np.integer)
        or np.any(component_index < -1)
        or np.any(component_index >= int(num_components))
    ):
        raise ValueError("Rigid point component indices are invalid")
    selected = component_index >= 0
    counts = np.bincount(
        component_index[selected], minlength=int(num_components)
    ).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError("Every rigid component must contain at least one point")
    selected_field = field[selected].astype(np.complex128)
    squared_norm = np.sum(
        np.square(selected_field.real) + np.square(selected_field.imag), axis=1
    )
    summed_squared_norm = np.bincount(
        component_index[selected],
        weights=squared_norm,
        minlength=int(num_components),
    )
    rms = np.sqrt(summed_squared_norm / counts)
    if not np.isfinite(rms).all():
        raise ValueError("Rigid component motion RMS is non-finite")
    return rms.astype(np.float32)


def rigid_component_anomaly_colors(anomalous: np.ndarray) -> np.ndarray:
    mask = np.asarray(anomalous)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("Rigid component anomaly mask must be a 1-D boolean array")
    colors = np.repeat(
        _RIGID_DIAGNOSTIC_NORMAL_COLOR[None], mask.shape[0], axis=0
    )
    colors[mask] = _RIGID_DIAGNOSTIC_ANOMALY_COLOR
    return colors


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


def _string_vector(array: np.ndarray, name: str, path: Path) -> tuple[str, ...]:
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


def _load_observed_graph_archive(
    graph_path: Path,
) -> ObservedGraphViewData:
    return load_observed_structure_graph(graph_path)


def observed_world_center(
    graphs: tuple[ObservedGraphViewData, ...],
) -> np.ndarray:
    points = [
        graph.node_points_world
        for graph in graphs
        if graph.node_points_world.shape[0]
    ]
    if not points:
        raise ValueError("Cannot center the viewer without observed graph nodes")
    all_points = np.concatenate(points, axis=0).astype(np.float64)
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Observed graph world center must be finite (3,)")
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


def deform_modal_points(
    points_world: np.ndarray,
    phi: np.ndarray,
    phase: float,
    motion_scale: float,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float32)
    field = np.asarray(phi)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Modal points_world must be a finite (N,3) array")
    if field.shape != points.shape or not np.issubdtype(
        field.dtype, np.complexfloating
    ):
        raise ValueError("Modal phi must be a complex (N,3) array")
    if not np.isfinite(field).all():
        raise ValueError("Modal phi must be finite")
    if not np.isfinite(phase):
        raise ValueError("Modal phase must be finite")
    if not np.isfinite(motion_scale) or motion_scale < 0.0:
        raise ValueError("Modal motion scale must be finite and non-negative")
    coefficient = np.exp(1j * float(phase))
    return (
        points.astype(np.float64)
        + float(motion_scale) * np.real(coefficient * field.astype(np.complex128))
    ).astype(np.float32)


def advance_playback_phase(
    phase: float,
    elapsed_seconds: float,
    frequency_hz: float,
    speed_multiplier: float,
) -> float:
    """Advance a modal phase using wall-clock time and the selected frequency."""

    values = np.asarray(
        [phase, elapsed_seconds, frequency_hz, speed_multiplier],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Playback phase inputs must be finite")
    if elapsed_seconds < 0.0:
        raise ValueError("Playback elapsed_seconds must be non-negative")
    if frequency_hz <= 0.0:
        raise ValueError("Playback frequency_hz must be positive")
    if speed_multiplier <= 0.0:
        raise ValueError("Playback speed_multiplier must be positive")
    return float(
        np.mod(
            phase
            + 2.0
            * np.pi
            * frequency_hz
            * speed_multiplier
            * elapsed_seconds,
            2.0 * np.pi,
        )
    )


def _manifest_string(value: Any, name: str, path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} {name} must be a non-empty string")
    return value


def _manifest_integer(value: Any, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} {name} must be an integer")
    return int(value)


def _manifest_number(value: Any, name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} {name} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{path} {name} must be finite")
    return result


def _manifest_artifact_path(
    manifest_path: Path,
    value: Any,
    name: str,
) -> Path:
    raw_path = Path(_manifest_string(value, name, manifest_path))
    return raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path


def _normalized_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _load_npz_arrays(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"Artifact does not exist: {path}")
    with np.load(str(path), allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path} missing required fields: {missing}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def _assert_observed_graph_identity(
    expected: ObservedGraphViewData,
    actual: ObservedGraphViewData,
) -> None:
    if expected.mode_index != actual.mode_index:
        raise ValueError(f"{actual.graph_path} mode_index does not match loaded graph")
    if not np.isclose(expected.freq_hz, actual.freq_hz, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{actual.graph_path} frequency does not match loaded graph")
    for field_name in (
        "source_checkpoint",
        "source_observation_path",
        "num_foreground_gaussians",
        "view_ids",
    ):
        if getattr(expected, field_name) != getattr(actual, field_name):
            raise ValueError(
                f"{actual.graph_path} {field_name} does not match loaded graph"
            )
    for field_name in (
        "node_gaussian_indices",
        "node_observed_view_mask",
        "edge_index",
        "component_index",
        "component_pruned_node_mask",
        "isolated_mask",
    ):
        if not np.array_equal(
            getattr(expected, field_name), getattr(actual, field_name)
        ):
            raise ValueError(
                f"{actual.graph_path} {field_name} does not match loaded graph"
            )
    if not np.allclose(
        expected.node_points_world,
        actual.node_points_world,
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise ValueError(
            f"{actual.graph_path} node positions do not match loaded graph"
        )
    for field_name in ("edge_depth_score", "edge_combined_weight"):
        if not np.allclose(
            getattr(expected, field_name),
            getattr(actual, field_name),
            rtol=1.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError(
                f"{actual.graph_path} {field_name} does not match loaded graph"
            )


def _rigid_reconstruction_quantization_metrics(
    persisted_phi: np.ndarray,
    persisted_translation: np.ndarray,
    persisted_rotation: np.ndarray,
    centered_points: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    actual = np.asarray(persisted_phi, dtype=np.complex128)
    translation = np.asarray(persisted_translation, dtype=np.complex128)
    rotation = np.asarray(persisted_rotation, dtype=np.complex128)
    centered = np.asarray(centered_points, dtype=np.float64)
    expected = translation + np.cross(rotation[None], centered)

    absolute_centered = np.abs(centered)

    def component_metrics(
        actual_part: np.ndarray,
        expected_part: np.ndarray,
        translation_part: np.ndarray,
        rotation_part: np.ndarray,
    ) -> tuple[float, float, float]:
        absolute_rotation = np.abs(rotation_part)
        cross_operand_scale = np.column_stack(
            (
                absolute_rotation[1] * absolute_centered[:, 2]
                + absolute_rotation[2] * absolute_centered[:, 1],
                absolute_rotation[2] * absolute_centered[:, 0]
                + absolute_rotation[0] * absolute_centered[:, 2],
                absolute_rotation[0] * absolute_centered[:, 1]
                + absolute_rotation[1] * absolute_centered[:, 0],
            )
        )
        operand_scale = (
            np.abs(actual_part)
            + np.abs(translation_part)[None]
            + cross_operand_scale
        )
        float32_epsilon = np.finfo(np.float32).eps
        float64_epsilon = np.finfo(np.float64).eps
        quantization_bound = (
            8.0
            * float32_epsilon
            * np.maximum(operand_scale, np.finfo(np.float32).tiny)
            + 64.0
            * float64_epsilon
            * np.maximum(operand_scale, np.finfo(np.float64).tiny)
        )
        error = np.abs(actual_part - expected_part)
        return (
            float(np.max(error)),
            float(np.max(quantization_bound)),
            float(np.max(error - quantization_bound)),
        )

    real_metrics = component_metrics(
        actual.real,
        expected.real,
        translation.real,
        rotation.real,
    )
    imaginary_metrics = component_metrics(
        actual.imag,
        expected.imag,
        translation.imag,
        rotation.imag,
    )
    return (*real_metrics, *imaginary_metrics)


def load_rigid_manifest(
    path: str | Path,
    graphs: tuple[ObservedGraphViewData, ...],
    gaussians: GaussianVisualizationData | None = None,
) -> RigidManifestViewData:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ValueError(f"Rigid manifest does not exist: {manifest_path}")
    if not graphs:
        raise ValueError("Rigid manifest validation requires an observed graph")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} root must be an object")
    if _manifest_integer(
        manifest.get("version"), "version", manifest_path
    ) != 1:
        raise ValueError(f"{manifest_path} must be version 1")
    if manifest.get("point_type") != "foreground_gaussian_center":
        raise ValueError(f"{manifest_path} point_type is incompatible")
    source_checkpoint = _manifest_string(
        manifest.get("source_checkpoint"),
        "source_checkpoint",
        manifest_path,
    )
    raw_source_view_configs = manifest.get("source_view_configs")
    if not isinstance(raw_source_view_configs, list) or not raw_source_view_configs:
        raise ValueError(
            f"{manifest_path} source_view_configs must be a non-empty list"
        )
    source_view_config_paths = tuple(
        Path(
            _manifest_string(
                value,
                f"source_view_configs[{index}]",
                manifest_path,
            )
        )
        for index, value in enumerate(raw_source_view_configs)
    )
    source_view_configs = tuple(
        load_view_config(path) for path in source_view_config_paths
    )
    expected_view_ids = graphs[0].view_ids
    if any(graph.view_ids != expected_view_ids for graph in graphs[1:]):
        raise ValueError("Observed graph inputs do not share one view order")
    source_view_ids = tuple(config.view_id for config in source_view_configs)
    if source_view_ids != expected_view_ids:
        raise ValueError(
            f"{manifest_path} source view-config order does not match observed graphs"
        )
    source_cameras: list[SourceCameraViewData] = []
    for config, config_path in zip(source_view_configs, source_view_config_paths):
        if (
            config.image_width <= 0
            or config.image_height <= 0
            or not np.isfinite(config.K).all()
            or config.K[1, 1] <= 0.0
            or not np.isfinite(config.world_to_camera).all()
        ):
            raise ValueError(f"{config_path} contains invalid camera geometry")
        try:
            np.linalg.inv(config.world_to_camera)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"{config_path} world_to_camera is singular"
            ) from exc
        source_cameras.append(
            SourceCameraViewData(
                label=config.view_id,
                image_width=config.image_width,
                image_height=config.image_height,
                K=config.K.astype(np.float64),
                world_to_camera=config.world_to_camera.astype(np.float64),
            )
        )
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{manifest_path} parameters must be an object")
    motion_fill_enabled = parameters.get("motion_fill_enabled")
    if not isinstance(motion_fill_enabled, bool):
        raise ValueError(
            f"{manifest_path} motion_fill_enabled must be a boolean"
        )
    rigid_motion_fill_stage = parameters.get("rigid_motion_fill_stage", "joint")
    if rigid_motion_fill_stage not in {"joint", "single-view-components"}:
        raise ValueError(
            f"{manifest_path} rigid_motion_fill_stage is incompatible"
        )
    component_fill_only = motion_fill_enabled and (
        rigid_motion_fill_stage == "single-view-components"
    )
    manifest_partial_observable_ratio = None
    manifest_partial_ray_fraction = None
    manifest_partial_max_normalized_motion_rms = None
    if component_fill_only:
        manifest_partial_observable_ratio = _manifest_number(
            parameters.get("rigid_single_view_observable_ratio"),
            "rigid_single_view_observable_ratio",
            manifest_path,
        )
        manifest_partial_ray_fraction = _manifest_number(
            parameters.get("rigid_single_view_ray_direction_min_fraction"),
            "rigid_single_view_ray_direction_min_fraction",
            manifest_path,
        )
        manifest_partial_max_normalized_motion_rms = _manifest_number(
            parameters.get("rigid_single_view_max_normalized_motion_rms"),
            "rigid_single_view_max_normalized_motion_rms",
            manifest_path,
        )
        if manifest_partial_max_normalized_motion_rms < 0.0:
            raise ValueError(
                f"{manifest_path} rigid_single_view_max_normalized_motion_rms "
                "must be non-negative"
            )
    rigid_seed_min_valid_views = _manifest_integer(
        parameters.get("rigid_seed_min_valid_views"),
        "rigid_seed_min_valid_views",
        manifest_path,
    )
    rigid_seed_min_singular_ratio = _manifest_number(
        parameters.get("rigid_seed_min_singular_ratio"),
        "rigid_seed_min_singular_ratio",
        manifest_path,
    )
    rigid_seed_max_finite_drift = _manifest_number(
        parameters.get("rigid_seed_max_finite_drift"),
        "rigid_seed_max_finite_drift",
        manifest_path,
    )
    if rigid_seed_min_valid_views <= 0:
        raise ValueError(
            f"{manifest_path} rigid_seed_min_valid_views must be positive"
        )
    if not 0.0 <= rigid_seed_min_singular_ratio <= 1.0:
        raise ValueError(
            f"{manifest_path} rigid_seed_min_singular_ratio must lie in [0,1]"
        )
    if rigid_seed_max_finite_drift < 0.0:
        raise ValueError(
            f"{manifest_path} rigid_seed_max_finite_drift must be non-negative"
        )
    expected_parameters = {
        "solver": "rigid_components",
        "rigidity_model": "complex_infinitesimal_se3",
        "rigid_component_selection": "observed_graph_degree_positive",
        "rigid_component_connectivity_policy": (
            "accepted_edge_transitive_components_bridges_merge"
        ),
        "rigid_component_rank_policy": "truncated_svd_minimum_norm_no_rejection",
        "rigid_component_residual_policy": "diagnostic_only",
        "rigid_component_edge_weight_use": "topology_only",
        "rigid_component_finite_rigidity": "first_order_only",
        "rigid_component_seed_policy": (
            "postsolve_valid_view_singular_ratio_and_finite_drift_gate"
        ),
        "nonseed_policy": (
            "single_view_partial_rigid_other_gaussians_zero"
            if component_fill_only
            else "single_view_component_rigid_else_free_motion_fill"
            if motion_fill_enabled
            else "zero_without_motion_fill"
        ),
        "single_view_component_fill_policy": (
            "observable_twist_plus_knn_filled_weak_and_ray_directions"
            if component_fill_only
            else "shared_unknown_normalized_infinitesimal_se3_twist"
            if motion_fill_enabled
            else "disabled"
        ),
    }
    for name, expected_value in expected_parameters.items():
        if parameters.get(name) != expected_value:
            raise ValueError(f"{manifest_path} parameter {name} is incompatible")
    raw_modes = manifest.get("modes")
    raw_mode_indices = manifest.get("mode_indices")
    if not isinstance(raw_modes, list) or not raw_modes:
        raise ValueError(f"{manifest_path} modes must be a non-empty list")
    if not isinstance(raw_mode_indices, list):
        raise ValueError(f"{manifest_path} mode_indices must be a list")
    mode_entries: dict[int, dict[str, Any]] = {}
    listed_indices: list[int] = []
    for entry_index, entry in enumerate(raw_modes):
        if not isinstance(entry, dict):
            raise ValueError(f"{manifest_path} modes[{entry_index}] must be an object")
        mode_index = _manifest_integer(
            entry.get("mode_index"),
            f"modes[{entry_index}].mode_index",
            manifest_path,
        )
        if mode_index in mode_entries:
            raise ValueError(f"{manifest_path} contains duplicate mode {mode_index}")
        mode_entries[mode_index] = entry
        listed_indices.append(mode_index)
    normalized_mode_indices = [
        _manifest_integer(value, f"mode_indices[{index}]", manifest_path)
        for index, value in enumerate(raw_mode_indices)
    ]
    if normalized_mode_indices != listed_indices:
        raise ValueError(f"{manifest_path} mode_indices do not match modes")
    graph_mode_indices = [graph.mode_index for graph in graphs]
    if len(set(graph_mode_indices)) != len(graph_mode_indices):
        raise ValueError("Observed graph inputs contain duplicate modes")
    if set(mode_entries) != set(graph_mode_indices):
        raise ValueError(
            f"{manifest_path} modes do not exactly match observed graph inputs"
        )

    loaded_modes: list[RigidModeViewData] = []
    for graph in graphs:
        if graph.source_checkpoint != source_checkpoint:
            raise ValueError(
                f"{manifest_path} source_checkpoint does not match {graph.graph_path}"
            )
        if graph.mode_index not in mode_entries:
            raise ValueError(
                f"{manifest_path} does not contain observed graph mode "
                f"{graph.mode_index}"
            )
        entry = mode_entries[graph.mode_index]
        mode_freq = _manifest_number(
            entry.get("freq_hz"),
            f"mode {graph.mode_index} freq_hz",
            manifest_path,
        )
        if not np.isclose(mode_freq, graph.freq_hz, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"{manifest_path} mode {graph.mode_index} frequency does not "
                "match graph"
            )
        if _manifest_string(
            entry.get("source_observation_path"),
            f"mode {graph.mode_index} source_observation_path",
            manifest_path,
        ) != graph.source_observation_path:
            raise ValueError(
                f"{manifest_path} mode {graph.mode_index} source observation "
                "does not match graph"
            )

        graph_path_value = _manifest_string(
            entry.get("rigid_component_graph_path"),
            f"mode {graph.mode_index} rigid_component_graph_path",
            manifest_path,
        )
        graph_path = _manifest_artifact_path(
            manifest_path,
            graph_path_value,
            f"mode {graph.mode_index} rigid_component_graph_path",
        )
        graph_source_path = _manifest_string(
            entry.get("rigid_component_graph_source_path"),
            f"mode {graph.mode_index} rigid_component_graph_source_path",
            manifest_path,
        )
        local_graph = _load_observed_graph_archive(graph_path)
        _assert_observed_graph_identity(graph, local_graph)
        allowed_graph_paths = {
            _normalized_path(graph_path),
            _normalized_path(graph_source_path),
        }
        if _normalized_path(graph.graph_path) not in allowed_graph_paths:
            raise ValueError(
                f"{graph.graph_path} is neither the manifest graph nor its source graph"
            )

        latent_path = _manifest_artifact_path(
            manifest_path,
            entry.get("latent_path"),
            f"mode {graph.mode_index} latent_path",
        )
        diagnostics_path = _manifest_artifact_path(
            manifest_path,
            entry.get("diagnostics_path"),
            f"mode {graph.mode_index} diagnostics_path",
        )
        component_diagnostics_path = _manifest_artifact_path(
            manifest_path,
            entry.get("component_diagnostics_path"),
            f"mode {graph.mode_index} component_diagnostics_path",
        )
        if _normalized_path(diagnostics_path) != _normalized_path(
            component_diagnostics_path
        ):
            raise ValueError(
                f"{manifest_path} mode {graph.mode_index} component diagnostics "
                "path differs"
            )
        latent = _load_npz_arrays(latent_path, _RIGID_LATENT_REQUIRED_FIELDS)
        diagnostics = _load_npz_arrays(
            diagnostics_path,
            _RIGID_DIAGNOSTIC_REQUIRED_FIELDS,
        )

        num_points = graph.num_foreground_gaussians
        points = np.asarray(latent["points_world"], dtype=np.float32)
        phi = np.asarray(latent["phi"])
        indices = np.asarray(latent["gaussian_indices"])
        if points.shape != (num_points, 3) or not np.isfinite(points).all():
            raise ValueError(
                f"{latent_path} points_world must be finite ({num_points},3)"
            )
        if (
            phi.shape != (num_points, 3)
            or not np.issubdtype(phi.dtype, np.complexfloating)
            or not np.isfinite(phi).all()
        ):
            raise ValueError(
                f"{latent_path} phi must be finite complex ({num_points},3)"
            )
        if (
            indices.shape != (num_points,)
            or not np.issubdtype(indices.dtype, np.integer)
            or not np.array_equal(indices, np.arange(num_points))
        ):
            raise ValueError(f"{latent_path} gaussian_indices must be canonical")
        if _scalar_string(
            latent["source_checkpoint"], "source_checkpoint", latent_path
        ) != source_checkpoint:
            raise ValueError(f"{latent_path} source_checkpoint does not match manifest")
        if _scalar_string(latent["point_type"], "point_type", latent_path) != (
            "foreground_gaussian_center"
        ):
            raise ValueError(f"{latent_path} point_type is incompatible")
        latent_mode = _scalar(latent["mode_index"], "mode_index", latent_path)
        if not np.issubdtype(latent_mode.dtype, np.integer) or int(
            latent_mode.item()
        ) != graph.mode_index:
            raise ValueError(f"{latent_path} mode_index does not match graph")
        latent_freq = float(
            _scalar(latent["freq_hz"], "freq_hz", latent_path).item()
        )
        if not np.isfinite(latent_freq) or not np.isclose(
            latent_freq, graph.freq_hz, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(f"{latent_path} frequency does not match graph")
        obs_count = np.asarray(latent["obs_count_per_point"])
        if (
            obs_count.shape != (num_points,)
            or not np.issubdtype(obs_count.dtype, np.integer)
            or np.any(obs_count < 0)
        ):
            raise ValueError(f"{latent_path} obs_count_per_point is invalid")
        if not np.allclose(
            points[graph.node_gaussian_indices],
            graph.node_points_world,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError(f"{latent_path} points do not match observed graph nodes")
        if gaussians is not None:
            if gaussians.source_checkpoint != source_checkpoint:
                raise ValueError(
                    f"{latent_path} checkpoint does not match Gaussian sidecar"
                )
            if not np.allclose(
                points,
                gaussians.foreground.centers,
                rtol=1.0e-5,
                atol=1.0e-6,
            ):
                raise ValueError(
                    f"{latent_path} points do not match Gaussian sidecar centers"
                )

        optional_role_fields = {
            "motion_fill_role",
            "motion_fill_role_names",
            "completion_mask",
        }
        present_role_fields = optional_role_fields & set(latent)
        if motion_fill_enabled and present_role_fields != optional_role_fields:
            raise ValueError(f"{latent_path} is missing rigid motion-fill role fields")
        if not motion_fill_enabled and present_role_fields:
            raise ValueError(f"{latent_path} unexpectedly contains motion-fill roles")

        for name, expected_value in (
            ("solver_method", "rigid_components"),
            ("solver_diagnostics_type", "rigid_component_twist_v1"),
            ("rigidity_model", "complex_infinitesimal_se3"),
            (
                "rigid_component_connectivity_policy",
                "accepted_edge_transitive_components_bridges_merge",
            ),
            (
                "rigid_component_seed_policy",
                "postsolve_valid_view_singular_ratio_and_finite_drift_gate",
            ),
        ):
            if (
                _scalar_string(diagnostics[name], name, diagnostics_path)
                != expected_value
            ):
                raise ValueError(f"{diagnostics_path} {name} is incompatible")
        diagnostic_min_valid_views = _scalar(
            diagnostics["rigid_seed_min_valid_views"],
            "rigid_seed_min_valid_views",
            diagnostics_path,
        )
        diagnostic_min_singular_ratio = _scalar(
            diagnostics["rigid_seed_min_singular_ratio"],
            "rigid_seed_min_singular_ratio",
            diagnostics_path,
        )
        diagnostic_max_finite_drift = _scalar(
            diagnostics["rigid_seed_max_finite_drift"],
            "rigid_seed_max_finite_drift",
            diagnostics_path,
        )
        if (
            not np.issubdtype(diagnostic_min_valid_views.dtype, np.integer)
            or int(diagnostic_min_valid_views.item())
            != rigid_seed_min_valid_views
        ):
            raise ValueError(
                f"{diagnostics_path} rigid seed minimum views differs from manifest"
            )
        diagnostic_ratio_value = float(
            diagnostic_min_singular_ratio.item()
        )
        if not np.isfinite(diagnostic_ratio_value) or not np.isclose(
            diagnostic_ratio_value,
            rigid_seed_min_singular_ratio,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                f"{diagnostics_path} rigid seed singular ratio differs from manifest"
            )
        diagnostic_drift_value = float(diagnostic_max_finite_drift.item())
        if not np.isfinite(diagnostic_drift_value) or not np.isclose(
            diagnostic_drift_value,
            rigid_seed_max_finite_drift,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                f"{diagnostics_path} rigid seed finite drift differs from manifest"
            )
        if _scalar_string(
            diagnostics["rigid_component_graph_path"],
            "rigid_component_graph_path",
            diagnostics_path,
        ) != graph_path_value:
            raise ValueError(f"{diagnostics_path} graph path does not match manifest")
        if _scalar_string(
            diagnostics["rigid_component_graph_source_path"],
            "rigid_component_graph_source_path",
            diagnostics_path,
        ) != graph_source_path:
            raise ValueError(
                f"{diagnostics_path} graph source path does not match manifest"
            )

        mask_names = (
            "rigid_seed_mask",
            "observed_mask",
            "fill_target_mask",
            "trusted_rigid_seed_mask",
            "effective_fill_target_mask",
            "completion_mask",
        )
        for name in mask_names:
            if (
                diagnostics[name].shape != (num_points,)
                or diagnostics[name].dtype != np.bool_
            ):
                raise ValueError(
                    f"{diagnostics_path} {name} must be boolean ({num_points},)"
                )
        rigid_seed_mask = diagnostics["rigid_seed_mask"]
        observed_mask = diagnostics["observed_mask"]
        fill_target_mask = diagnostics["fill_target_mask"]
        trusted_seed_mask = diagnostics["trusted_rigid_seed_mask"]
        effective_fill_target_mask = diagnostics["effective_fill_target_mask"]
        completion_mask = diagnostics["completion_mask"]
        expected_observed = np.zeros((num_points,), dtype=bool)
        expected_observed[graph.node_gaussian_indices] = True
        expected_seed = np.zeros((num_points,), dtype=bool)
        rigid_node_mask = graph.graph.topology.degree > 0
        expected_seed[graph.node_gaussian_indices[rigid_node_mask]] = True
        if not np.array_equal(observed_mask, expected_observed):
            raise ValueError(f"{diagnostics_path} observed_mask does not match graph")
        if not np.array_equal(rigid_seed_mask, expected_seed):
            raise ValueError(f"{diagnostics_path} rigid_seed_mask does not match graph")
        if not np.array_equal(fill_target_mask, ~expected_seed):
            raise ValueError(f"{diagnostics_path} fill_target_mask is inconsistent")
        if np.any(trusted_seed_mask & ~rigid_seed_mask):
            raise ValueError(f"{diagnostics_path} trusts a non-candidate rigid seed")
        if not np.array_equal(effective_fill_target_mask, ~trusted_seed_mask):
            raise ValueError(
                f"{diagnostics_path} effective_fill_target_mask is inconsistent"
            )
        if np.any(completion_mask & ~effective_fill_target_mask):
            raise ValueError(f"{diagnostics_path} completes a trusted rigid seed")
        if not np.array_equal(diagnostics["final_phi"], phi):
            raise ValueError(f"{diagnostics_path} final_phi does not match latent")

        selected_graph_components = np.unique(
            graph.component_index[rigid_node_mask]
        ).astype(np.int32)
        num_components = int(selected_graph_components.shape[0])
        graph_to_rigid = np.full(
            (graph.graph.topology.component_size.shape[0],),
            -1,
            dtype=np.int32,
        )
        graph_to_rigid[selected_graph_components] = np.arange(
            num_components, dtype=np.int32
        )
        expected_point_component = np.full((num_points,), -1, dtype=np.int32)
        expected_point_component[graph.node_gaussian_indices[rigid_node_mask]] = (
            graph_to_rigid[graph.component_index[rigid_node_mask]]
        )
        point_component = np.asarray(diagnostics["point_component_index"])
        if not np.issubdtype(point_component.dtype, np.integer) or not np.array_equal(
            point_component, expected_point_component
        ):
            raise ValueError(
                f"{diagnostics_path} point_component_index does not match graph"
            )
        component_graph_index = np.asarray(diagnostics["component_graph_index"])
        if not np.issubdtype(
            component_graph_index.dtype, np.integer
        ) or not np.array_equal(component_graph_index, selected_graph_components):
            raise ValueError(
                f"{diagnostics_path} component_graph_index does not match graph"
            )
        expected_node_count = np.bincount(
            expected_point_component[expected_point_component >= 0],
            minlength=num_components,
        )
        expected_edge_component = graph_to_rigid[
            graph.component_index[graph.edge_index[:, 0]]
        ]
        expected_edge_count = np.bincount(
            expected_edge_component,
            minlength=num_components,
        )
        for name, expected_values in (
            ("component_node_count", expected_node_count),
            ("component_edge_count", expected_edge_count),
            ("edge_component_index", expected_edge_component),
        ):
            values = np.asarray(diagnostics[name])
            if not np.issubdtype(values.dtype, np.integer) or not np.array_equal(
                values, expected_values
            ):
                raise ValueError(f"{diagnostics_path} {name} does not match graph")
        component_rank = np.asarray(diagnostics["component_rank"])
        if (
            component_rank.shape != (num_components,)
            or not np.issubdtype(component_rank.dtype, np.integer)
            or np.any(component_rank < 0)
            or np.any(component_rank > 6)
        ):
            raise ValueError(f"{diagnostics_path} component_rank is invalid")
        component_valid_view_count = np.asarray(
            diagnostics["component_distinct_valid_view_count"]
        )
        if (
            component_valid_view_count.shape != (num_components,)
            or not np.issubdtype(component_valid_view_count.dtype, np.integer)
            or np.any(component_valid_view_count < 1)
            or np.any(component_valid_view_count > len(graph.view_ids))
        ):
            raise ValueError(
                f"{diagnostics_path} component valid-view count is invalid"
            )
        try:
            component_singular_ratio = rigid_component_singular_ratio(
                diagnostics["component_singular_values"],
                component_rank,
            )
            component_motion_rms = rigid_component_motion_rms(
                phi,
                point_component,
                num_components,
            )
        except ValueError as exc:
            raise ValueError(f"{diagnostics_path} {exc}") from exc
        stored_singular_ratio = np.asarray(
            diagnostics["component_singular_ratio"], dtype=np.float32
        )
        if not np.array_equal(stored_singular_ratio, component_singular_ratio):
            raise ValueError(
                f"{diagnostics_path} component_singular_ratio is inconsistent"
            )
        component_finite_drift = np.asarray(
            diagnostics["component_finite_drift_max"], dtype=np.float32
        )
        if (
            component_finite_drift.shape != (num_components,)
            or not np.isfinite(component_finite_drift).all()
            or np.any(component_finite_drift < 0.0)
        ):
            raise ValueError(
                f"{diagnostics_path} component finite drift is invalid"
            )
        component_selection_masks = {
            name: np.asarray(diagnostics[name])
            for name in (
                "component_seed_retained_mask",
                "component_valid_view_rejected_mask",
                "component_singular_rejected_mask",
                "component_finite_drift_rejected_mask",
            )
        }
        for name, values in component_selection_masks.items():
            if values.shape != (num_components,) or values.dtype != np.bool_:
                raise ValueError(
                    f"{diagnostics_path} {name} must be boolean "
                    f"({num_components},)"
                )
        expected_valid_view_rejected = (
            component_valid_view_count < rigid_seed_min_valid_views
        )
        expected_singular_rejected = (
            component_singular_ratio < rigid_seed_min_singular_ratio
        )
        expected_finite_drift_rejected = (
            component_finite_drift > rigid_seed_max_finite_drift
        )
        expected_component_retained = ~(
            expected_valid_view_rejected
            | expected_singular_rejected
            | expected_finite_drift_rejected
        )
        for name, expected_values in (
            (
                "component_valid_view_rejected_mask",
                expected_valid_view_rejected,
            ),
            ("component_singular_rejected_mask", expected_singular_rejected),
            (
                "component_finite_drift_rejected_mask",
                expected_finite_drift_rejected,
            ),
            ("component_seed_retained_mask", expected_component_retained),
        ):
            if not np.array_equal(
                component_selection_masks[name], expected_values
            ):
                raise ValueError(f"{diagnostics_path} {name} is inconsistent")
        expected_trusted_seed = np.zeros((num_points,), dtype=bool)
        candidate_indices = np.flatnonzero(expected_seed)
        expected_trusted_seed[candidate_indices] = expected_component_retained[
            point_component[candidate_indices]
        ]
        if not np.array_equal(trusted_seed_mask, expected_trusted_seed):
            raise ValueError(
                f"{diagnostics_path} trusted_rigid_seed_mask is inconsistent"
            )
        raw_rigid_phi = np.asarray(diagnostics["rigid_phi_pre_fill"])
        trusted_rigid_phi = np.asarray(
            diagnostics["trusted_rigid_phi_pre_fill"]
        )
        for name, values in (
            ("rigid_phi_pre_fill", raw_rigid_phi),
            ("trusted_rigid_phi_pre_fill", trusted_rigid_phi),
        ):
            if (
                values.shape != (num_points, 3)
                or not np.issubdtype(values.dtype, np.complexfloating)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"{diagnostics_path} {name} must be finite complex "
                    f"({num_points},3)"
                )
        expected_trusted_phi = np.zeros((num_points, 3), dtype=np.complex64)
        expected_trusted_phi[trusted_seed_mask] = raw_rigid_phi[
            trusted_seed_mask
        ].astype(np.complex64)
        if not np.array_equal(trusted_rigid_phi, expected_trusted_phi):
            raise ValueError(
                f"{diagnostics_path} trusted_rigid_phi_pre_fill is inconsistent"
            )
        if not np.array_equal(
            phi[trusted_seed_mask], trusted_rigid_phi[trusted_seed_mask]
        ):
            raise ValueError(
                f"{latent_path} final phi changed a trusted rigid seed"
            )
        rank_deficient = np.asarray(diagnostics["component_rank_deficient_mask"])
        if rank_deficient.dtype != np.bool_ or not np.array_equal(
            rank_deficient, component_rank < 6
        ):
            raise ValueError(
                f"{diagnostics_path} component_rank_deficient_mask is inconsistent"
            )
        component_residual = np.asarray(
            diagnostics["component_normalized_weighted_residual"],
            dtype=np.float32,
        )
        edge_finite_drift = np.asarray(
            diagnostics["edge_finite_drift_max"],
            dtype=np.float32,
        )
        if (
            component_residual.shape != (num_components,)
            or not np.isfinite(component_residual).all()
            or np.any(component_residual < 0.0)
        ):
            raise ValueError(f"{diagnostics_path} component residual is invalid")
        if (
            edge_finite_drift.shape != (graph.edge_index.shape[0],)
            or not np.isfinite(edge_finite_drift).all()
            or np.any(edge_finite_drift < 0.0)
        ):
            raise ValueError(f"{diagnostics_path} finite edge drift is invalid")

        single_view_component_completion = np.zeros(
            (num_components,), dtype=bool
        )
        partial_observable_rank = None
        partial_fill_nullity = None
        partial_trusted_knn_count = None
        partial_ray_motion_ratio = None
        partial_normalized_motion_rms = None
        partial_finite_drift_rejected = None
        partial_motion_rms_rejected = None
        partial_postfill_retained = None
        if motion_fill_enabled:
            role_names = _string_vector(
                latent["motion_fill_role_names"],
                "motion_fill_role_names",
                latent_path,
            )
            if role_names != _MOTION_FILL_ROLE_NAMES:
                raise ValueError(
                    f"{latent_path} motion_fill_role_names is incompatible"
                )
            role = np.asarray(latent["motion_fill_role"])
            latent_completion = np.asarray(latent["completion_mask"])
            grouped_required = {
                "component_centroid",
                "single_view_component_fill_policy",
                "single_view_component_fill_mask",
                "single_view_rigid_fill_point_mask",
                "single_view_component_completion_mask",
                "single_view_component_translation",
                "single_view_component_rotation",
                "single_view_component_first_order_relative_max",
            }
            grouped_missing = sorted(grouped_required - set(diagnostics))
            if grouped_missing:
                raise ValueError(
                    f"{diagnostics_path} is missing grouped rigid-fill fields: "
                    f"{grouped_missing}"
                )
            expected_component_fill_policy = (
                "observable_twist_plus_knn_filled_weak_and_ray_directions"
                if component_fill_only
                else "shared_unknown_normalized_infinitesimal_se3_twist"
            )
            if _scalar_string(
                diagnostics["single_view_component_fill_policy"],
                "single_view_component_fill_policy",
                diagnostics_path,
            ) != expected_component_fill_policy:
                raise ValueError(
                    f"{diagnostics_path} single-view fill policy is incompatible"
                )
            single_view_component_fill = np.asarray(
                diagnostics["single_view_component_fill_mask"]
            )
            expected_single_view_component_fill = (
                (component_valid_view_count == 1)
                & ~component_selection_masks["component_seed_retained_mask"]
            )
            if (
                single_view_component_fill.dtype != np.bool_
                or single_view_component_fill.shape != (num_components,)
                or not np.array_equal(
                    single_view_component_fill,
                    expected_single_view_component_fill,
                )
            ):
                raise ValueError(
                    f"{diagnostics_path} single-view component fill mask is inconsistent"
                )
            single_view_fill_points = np.asarray(
                diagnostics["single_view_rigid_fill_point_mask"]
            )
            expected_single_view_fill_points = np.zeros(
                (num_points,), dtype=bool
            )
            component_points = point_component >= 0
            expected_single_view_fill_points[component_points] = (
                single_view_component_fill[point_component[component_points]]
            )
            if (
                single_view_fill_points.dtype != np.bool_
                or single_view_fill_points.shape != (num_points,)
                or not np.array_equal(
                    single_view_fill_points,
                    expected_single_view_fill_points,
                )
            ):
                raise ValueError(
                    f"{diagnostics_path} single-view rigid-fill point mask is inconsistent"
                )
            single_view_component_completion = np.asarray(
                diagnostics["single_view_component_completion_mask"]
            )
            if (
                single_view_component_completion.dtype != np.bool_
                or single_view_component_completion.shape != (num_components,)
                or np.any(
                    single_view_component_completion
                    & ~single_view_component_fill
                )
            ):
                raise ValueError(
                    f"{diagnostics_path} single-view component completion is invalid"
                )
            for component_idx in np.flatnonzero(
                single_view_component_fill
            ).tolist():
                member_completion = completion_mask[
                    point_component == component_idx
                ]
                if (
                    member_completion.size == 0
                    or np.any(member_completion) != np.all(member_completion)
                    or bool(np.all(member_completion))
                    != bool(single_view_component_completion[component_idx])
                ):
                    raise ValueError(
                        f"{diagnostics_path} component {component_idx} has partial "
                        "or inconsistent grouped completion"
                    )
            component_centroid = np.asarray(
                diagnostics["component_centroid"], dtype=np.float64
            )
            component_translation = np.asarray(
                diagnostics["single_view_component_translation"]
            )
            component_rotation = np.asarray(
                diagnostics["single_view_component_rotation"]
            )
            component_first_order = np.asarray(
                diagnostics[
                    "single_view_component_first_order_relative_max"
                ]
            )
            if (
                component_centroid.shape != (num_components, 3)
                or not np.isfinite(component_centroid).all()
            ):
                raise ValueError(
                    f"{diagnostics_path} component centroids are invalid"
                )
            for name, values in (
                ("single_view_component_translation", component_translation),
                ("single_view_component_rotation", component_rotation),
            ):
                if (
                    values.shape != (num_components, 3)
                    or not np.issubdtype(values.dtype, np.complexfloating)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(
                        f"{diagnostics_path} {name} must be finite complex "
                        f"({num_components},3)"
                    )
                if np.any(values[~single_view_component_fill] != 0):
                    raise ValueError(
                        f"{diagnostics_path} {name} is nonzero outside selected "
                        "single-view components"
                    )
            if (
                component_first_order.shape != (num_components,)
                or not np.isfinite(component_first_order).all()
                or np.any(component_first_order < 0.0)
            ):
                raise ValueError(
                    f"{diagnostics_path} single-view rigidity error is invalid"
                )
            if np.any(component_first_order[~single_view_component_fill] != 0):
                raise ValueError(
                    f"{diagnostics_path} single-view rigidity error is nonzero "
                    "outside selected components"
                )
            for component_idx in np.flatnonzero(
                single_view_component_fill
            ).tolist():
                members = np.flatnonzero(point_component == component_idx)
                centered = points[members].astype(np.float64) - component_centroid[
                    component_idx
                ]
                (
                    real_error,
                    real_bound,
                    real_excess,
                    imaginary_error,
                    imaginary_bound,
                    imaginary_excess,
                ) = _rigid_reconstruction_quantization_metrics(
                    phi[members],
                    component_translation[component_idx],
                    component_rotation[component_idx],
                    centered,
                )
                if real_excess > 0.0 or imaginary_excess > 0.0:
                    raise ValueError(
                        f"{latent_path} component {component_idx} is not represented "
                        "by its filled rigid twist within complex64 quantization: "
                        f"real_error={real_error:.9g}, real_bound={real_bound:.9g}, "
                        f"real_excess={real_excess:.9g}, "
                        f"imaginary_error={imaginary_error:.9g}, "
                        f"imaginary_bound={imaginary_bound:.9g}, "
                        f"imaginary_excess={imaginary_excess:.9g}"
                    )
            expected_role = np.full((num_points,), 2, dtype=np.int8)
            expected_role[trusted_seed_mask] = 0
            expected_role[single_view_fill_points] = 1
            if (
                not np.issubdtype(role.dtype, np.integer)
                or not np.array_equal(role, expected_role)
            ):
                raise ValueError(f"{latent_path} motion_fill_role is inconsistent")
            if latent_completion.dtype != np.bool_ or not np.array_equal(
                latent_completion, completion_mask
            ):
                raise ValueError(f"{latent_path} completion_mask is inconsistent")
            for name in (
                "motion_fill_method",
                "motion_fill_role",
                "completion_connected_to_anchor",
            ):
                if name not in diagnostics:
                    raise ValueError(f"{diagnostics_path} is missing {name}")
            expected_motion_fill_method = (
                "rigid_seed_single_view_partial_component_knn_lsmr"
                if component_fill_only
                else "rigid_seed_single_view_component_grouped_knn_lsmr"
            )
            if _scalar_string(
                diagnostics["motion_fill_method"],
                "motion_fill_method",
                diagnostics_path,
            ) != expected_motion_fill_method:
                raise ValueError(
                    f"{diagnostics_path} motion_fill_method is incompatible"
                )
            if not np.array_equal(diagnostics["motion_fill_role"], role):
                raise ValueError(
                    f"{diagnostics_path} motion_fill_role differs from latent"
                )
            connected = np.asarray(diagnostics["completion_connected_to_anchor"])
            if (
                connected.shape != (num_points,)
                or connected.dtype != np.bool_
                or not np.array_equal(
                    completion_mask, connected & ~trusted_seed_mask
                )
            ):
                raise ValueError(
                    f"{diagnostics_path} completion connectivity is inconsistent"
                )
            if component_fill_only:
                partial_required = {
                    "single_view_observable_singular_ratio_min",
                    "single_view_ray_direction_min_fraction",
                    "single_view_max_finite_drift",
                    "single_view_max_normalized_motion_rms",
                    "single_view_component_observable_rank",
                    "single_view_component_fill_nullity",
                    "single_view_component_ray_dominated_basis_count",
                    "single_view_component_trusted_knn_edge_count",
                    "single_view_component_cross_knn_edge_count",
                    "single_view_component_connected_to_trusted_mask",
                    "single_view_component_postfill_normalized_residual",
                    "single_view_component_ray_motion_rms",
                    "single_view_component_tangent_motion_rms",
                    "single_view_component_ray_motion_ratio",
                    "single_view_component_partial_finite_drift_max",
                    "single_view_component_normalized_motion_rms",
                    "single_view_component_finite_drift_rejected_mask",
                    "single_view_component_motion_rms_rejected_mask",
                    "single_view_component_postfill_retained_mask",
                }
                partial_missing = sorted(partial_required - set(diagnostics))
                if partial_missing:
                    raise ValueError(
                        f"{diagnostics_path} is missing partial component fields: "
                        f"{partial_missing}"
                    )
                partial_observable_ratio_min = float(
                    _scalar(
                        diagnostics[
                            "single_view_observable_singular_ratio_min"
                        ],
                        "single_view_observable_singular_ratio_min",
                        diagnostics_path,
                    ).item()
                )
                partial_ray_direction_min_fraction = float(
                    _scalar(
                        diagnostics[
                            "single_view_ray_direction_min_fraction"
                        ],
                        "single_view_ray_direction_min_fraction",
                        diagnostics_path,
                    ).item()
                )
                partial_max_finite_drift = float(
                    _scalar(
                        diagnostics["single_view_max_finite_drift"],
                        "single_view_max_finite_drift",
                        diagnostics_path,
                    ).item()
                )
                partial_max_normalized_motion_rms = float(
                    _scalar(
                        diagnostics["single_view_max_normalized_motion_rms"],
                        "single_view_max_normalized_motion_rms",
                        diagnostics_path,
                    ).item()
                )
                if (
                    not np.isfinite(partial_observable_ratio_min)
                    or not 0.0 < partial_observable_ratio_min <= 1.0
                    or not np.isfinite(partial_ray_direction_min_fraction)
                    or not 0.0 <= partial_ray_direction_min_fraction <= 1.0
                    or not np.isfinite(partial_max_finite_drift)
                    or partial_max_finite_drift < 0.0
                    or not np.isfinite(partial_max_normalized_motion_rms)
                    or partial_max_normalized_motion_rms < 0.0
                ):
                    raise ValueError(
                        f"{diagnostics_path} partial component thresholds are invalid"
                    )
                if not np.isclose(
                    partial_observable_ratio_min,
                    float(manifest_partial_observable_ratio),
                    rtol=0.0,
                    atol=1.0e-12,
                ) or not np.isclose(
                    partial_ray_direction_min_fraction,
                    float(manifest_partial_ray_fraction),
                    rtol=0.0,
                    atol=1.0e-12,
                ) or not np.isclose(
                    partial_max_finite_drift,
                    rigid_seed_max_finite_drift,
                    rtol=0.0,
                    atol=1.0e-12,
                ) or not np.isclose(
                    partial_max_normalized_motion_rms,
                    float(manifest_partial_max_normalized_motion_rms),
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise ValueError(
                        f"{diagnostics_path} partial thresholds differ from manifest"
                    )
                partial_observable_rank = np.asarray(
                    diagnostics["single_view_component_observable_rank"]
                )
                partial_fill_nullity = np.asarray(
                    diagnostics["single_view_component_fill_nullity"]
                )
                partial_trusted_knn_count = np.asarray(
                    diagnostics["single_view_component_trusted_knn_edge_count"]
                )
                partial_ray_motion_ratio = np.asarray(
                    diagnostics["single_view_component_ray_motion_ratio"],
                    dtype=np.float32,
                )
                partial_normalized_motion_rms = np.asarray(
                    diagnostics["single_view_component_normalized_motion_rms"],
                    dtype=np.float32,
                )
                partial_finite_drift_rejected = np.asarray(
                    diagnostics[
                        "single_view_component_finite_drift_rejected_mask"
                    ]
                )
                partial_motion_rms_rejected = np.asarray(
                    diagnostics[
                        "single_view_component_motion_rms_rejected_mask"
                    ]
                )
                partial_postfill_retained = np.asarray(
                    diagnostics["single_view_component_postfill_retained_mask"]
                )
                for name, values in (
                    (
                        "single_view_component_observable_rank",
                        partial_observable_rank,
                    ),
                    (
                        "single_view_component_fill_nullity",
                        partial_fill_nullity,
                    ),
                    (
                        "single_view_component_trusted_knn_edge_count",
                        partial_trusted_knn_count,
                    ),
                ):
                    if (
                        values.shape != (num_components,)
                        or not np.issubdtype(values.dtype, np.integer)
                        or np.any(values < 0)
                    ):
                        raise ValueError(
                            f"{diagnostics_path} {name} is invalid"
                        )
                if np.any(partial_observable_rank > 6) or np.any(
                    partial_fill_nullity > 6
                ):
                    raise ValueError(
                        f"{diagnostics_path} partial component dimensions exceed six"
                    )
                if (
                    partial_ray_motion_ratio.shape != (num_components,)
                    or not np.isfinite(partial_ray_motion_ratio).all()
                    or np.any(partial_ray_motion_ratio < 0.0)
                ):
                    raise ValueError(
                        f"{diagnostics_path} partial ray-motion ratio is invalid"
                    )
                if (
                    partial_normalized_motion_rms.shape != (num_components,)
                    or not np.isfinite(partial_normalized_motion_rms).all()
                    or np.any(partial_normalized_motion_rms < 0.0)
                    or np.any(
                        partial_normalized_motion_rms[
                            ~single_view_component_fill
                        ]
                        != 0.0
                    )
                ):
                    raise ValueError(
                        f"{diagnostics_path} partial normalized motion RMS is invalid"
                    )
                for name, values in (
                    (
                        "single_view_component_finite_drift_rejected_mask",
                        partial_finite_drift_rejected,
                    ),
                    (
                        "single_view_component_motion_rms_rejected_mask",
                        partial_motion_rms_rejected,
                    ),
                    (
                        "single_view_component_postfill_retained_mask",
                        partial_postfill_retained,
                    ),
                ):
                    if (
                        values.dtype != np.bool_
                        or values.shape != (num_components,)
                        or np.any(values & ~single_view_component_fill)
                    ):
                        raise ValueError(f"{diagnostics_path} {name} is invalid")
                expected_postfill_retained = (
                    single_view_component_fill
                    & ~partial_finite_drift_rejected
                    & ~partial_motion_rms_rejected
                )
                if not np.array_equal(
                    partial_postfill_retained, expected_postfill_retained
                ):
                    raise ValueError(
                        f"{diagnostics_path} partial post-fill retention is inconsistent"
                    )
                partial_connected = np.asarray(
                    diagnostics[
                        "single_view_component_connected_to_trusted_mask"
                    ]
                )
                if (
                    partial_connected.dtype != np.bool_
                    or partial_connected.shape != (num_components,)
                    or not np.array_equal(
                        partial_connected & partial_postfill_retained,
                        single_view_component_completion,
                    )
                ):
                    raise ValueError(
                        f"{diagnostics_path} partial trusted connectivity is inconsistent"
                    )
                rejected_components = (
                    single_view_component_fill & ~partial_postfill_retained
                )
                rejected_points = np.zeros((num_points,), dtype=bool)
                rejected_points[component_points] = rejected_components[
                    point_component[component_points]
                ]
                if np.any(phi[rejected_points] != 0):
                    raise ValueError(
                        f"{latent_path} rejected partial components retain motion"
                    )
                if (
                    np.any(component_translation[rejected_components] != 0)
                    or np.any(component_rotation[rejected_components] != 0)
                ):
                    raise ValueError(
                        f"{diagnostics_path} rejected partial rigid twists are nonzero"
                    )
        else:
            if np.any(completion_mask):
                raise ValueError(
                    f"{diagnostics_path} completion_mask is nonzero without "
                    "motion fill"
                )
            if not np.array_equal(phi, trusted_rigid_phi):
                raise ValueError(
                    f"{latent_path} phi differs from trusted rigid field "
                    "without motion fill"
                )

        loaded_modes.append(
            RigidModeViewData(
                manifest_path=manifest_path,
                mode_index=graph.mode_index,
                freq_hz=graph.freq_hz,
                latent_path=latent_path,
                diagnostics_path=diagnostics_path,
                graph_path=graph_path,
                graph_source_path=graph_source_path,
                points_world=points,
                phi=phi.astype(np.complex64),
                raw_rigid_phi=raw_rigid_phi.astype(np.complex64),
                rigid_seed_mask=trusted_seed_mask,
                quarantined_rigid_mask=rigid_seed_mask & ~trusted_seed_mask,
                single_view_rigid_fill_mask=(
                    single_view_fill_points & completion_mask
                    if motion_fill_enabled
                    else np.zeros((num_points,), dtype=bool)
                ),
                completed_fill_mask=completion_mask,
                unresolved_fill_mask=(
                    effective_fill_target_mask & ~completion_mask
                ),
                point_component_index=point_component.astype(np.int32),
                component_normalized_weighted_residual=component_residual,
                component_rank=component_rank.astype(np.int8),
                component_distinct_valid_view_count=(
                    component_valid_view_count.astype(np.int32)
                ),
                component_singular_ratio=component_singular_ratio,
                component_motion_rms=component_motion_rms,
                component_finite_drift_max=component_finite_drift,
                single_view_component_completion_mask=(
                    single_view_component_completion.astype(bool)
                ),
                single_view_component_observable_rank=(
                    None
                    if partial_observable_rank is None
                    else partial_observable_rank.astype(np.int8)
                ),
                single_view_component_fill_nullity=(
                    None
                    if partial_fill_nullity is None
                    else partial_fill_nullity.astype(np.int8)
                ),
                single_view_component_trusted_knn_edge_count=(
                    None
                    if partial_trusted_knn_count is None
                    else partial_trusted_knn_count.astype(np.int32)
                ),
                single_view_component_ray_motion_ratio=(
                    None
                    if partial_ray_motion_ratio is None
                    else partial_ray_motion_ratio.astype(np.float32)
                ),
                single_view_component_normalized_motion_rms=(
                    None
                    if partial_normalized_motion_rms is None
                    else partial_normalized_motion_rms.astype(np.float32)
                ),
                single_view_component_finite_drift_rejected_mask=(
                    None
                    if partial_finite_drift_rejected is None
                    else partial_finite_drift_rejected.astype(bool)
                ),
                single_view_component_motion_rms_rejected_mask=(
                    None
                    if partial_motion_rms_rejected is None
                    else partial_motion_rms_rejected.astype(bool)
                ),
                single_view_component_postfill_retained_mask=(
                    None
                    if partial_postfill_retained is None
                    else partial_postfill_retained.astype(bool)
                ),
                edge_component_index=expected_edge_component.astype(np.int32),
                edge_finite_drift_max=edge_finite_drift,
            )
        )

    return RigidManifestViewData(
        manifest_path=manifest_path,
        source_checkpoint=source_checkpoint,
        motion_fill_enabled=motion_fill_enabled,
        rigid_motion_fill_stage=rigid_motion_fill_stage,
        rigid_seed_min_valid_views=rigid_seed_min_valid_views,
        rigid_seed_min_singular_ratio=rigid_seed_min_singular_ratio,
        rigid_seed_max_finite_drift=rigid_seed_max_finite_drift,
        rigid_single_view_max_normalized_motion_rms=(
            manifest_partial_max_normalized_motion_rms
        ),
        source_cameras=tuple(source_cameras),
        modes=tuple(loaded_modes),
    )


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
    graphs: tuple[ObservedGraphViewData, ...],
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
        if graph.node_gaussian_indices.shape[0]:
            sidecar_graph_points = foreground.centers[
                graph.node_gaussian_indices
            ]
            if not np.allclose(
                sidecar_graph_points,
                graph.node_points_world,
                rtol=1e-6,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{sidecar_path} foreground centers do not match "
                    f"{graph.graph_path} nodes"
                )
    return GaussianVisualizationData(
        sidecar_path=sidecar_path,
        source_checkpoint=source_checkpoint,
        foreground=foreground,
        background=background,
    )


def load_observation_coverage(
    path: Path,
    graphs: tuple[ObservedGraphViewData, ...],
    gaussians: GaussianVisualizationData | None = None,
) -> ObservationCoverageViewData:
    if not graphs:
        raise ValueError("Observation coverage validation requires at least one graph")
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
    reference_observation_path = _scalar_string(
        arrays["reference_observation_path"],
        "reference_observation_path",
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
    view_ids = _string_vector(arrays["view_ids"], "view_ids", path)
    num_views = len(view_ids)
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
    selected_hits = np.asarray(arrays["selected_hit_count_by_k_view"])
    if not np.issubdtype(selected_hits.dtype, np.integer) or np.any(
        selected_hits < 0
    ):
        raise ValueError(
            f"{path} selected_hit_count_by_k_view must be non-negative integers"
        )
    baseline_view_mask = selected_hits[baseline_index] > 0
    baseline_node_indices = np.flatnonzero(
        baseline_view_mask.any(axis=1)
    ).astype(np.int64)
    for graph in graphs:
        if graph.source_checkpoint != source_checkpoint:
            raise ValueError(f"{path} source_checkpoint does not match graph")
        if graph.source_observation_path != reference_observation_path:
            raise ValueError(
                f"{path} reference_observation_path does not match graph"
            )
        if graph.num_foreground_gaussians != num_points:
            raise ValueError(f"{path} foreground count does not match graph")
        if graph.view_ids != view_ids:
            raise ValueError(f"{path} view_ids do not match graph")
        if not np.array_equal(
            graph.node_gaussian_indices,
            baseline_node_indices,
        ):
            raise ValueError(
                f"{path} baseline observed node indices do not match graph"
            )
        if not np.array_equal(
            graph.node_observed_view_mask,
            baseline_view_mask[baseline_node_indices],
        ):
            raise ValueError(
                f"{path} baseline observation mask does not match graph"
            )
        if graph.node_gaussian_indices.shape[0] and not np.allclose(
            points[graph.node_gaussian_indices],
            graph.node_points_world,
            rtol=1.0e-6,
            atol=1.0e-5,
        ):
            raise ValueError(f"{path} Gaussian centers do not match graph nodes")
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


def load_anchor_residual_diagnostics(
    path: Path,
    graphs: tuple[ObservedGraphViewData, ...],
    gaussians: GaussianVisualizationData | None = None,
) -> AnchorResidualViewData:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(_RESIDUAL_REQUIRED_FIELDS - set(archive.files))
        if missing:
            raise ValueError(f"{path} missing required fields: {missing}")
        arrays = {name: archive[name] for name in archive.files}
    version_value = int(_scalar(arrays["version"], "version", path))
    if version_value != 1:
        raise ValueError(f"{path} has unsupported version={version_value}")
    point_type = _scalar_string(arrays["point_type"], "point_type", path)
    if point_type != "foreground_gaussian_anchor_residual_decomposition":
        raise ValueError(f"{path} has unsupported point_type={point_type!r}")
    for name, expected in (
        (
            "effective_weight_method",
            "contribution_divided_by_point_view_multiplicity",
        ),
        (
            "decomposition_method",
            "exact_weighted_point_view_mean_sse_identity",
        ),
        ("replay_validation", "point_precompletion_residual_exact"),
    ):
        if _scalar_string(arrays[name], name, path) != expected:
            raise ValueError(f"{path} has unsupported {name}")
    source_checkpoint = _scalar_string(
        arrays["source_checkpoint"],
        "source_checkpoint",
        path,
    )
    source_observation_path = _scalar_string(
        arrays["source_observation_path"],
        "source_observation_path",
        path,
    )
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
    view_ids = _string_vector(arrays["view_ids"], "view_ids", path)
    num_views = len(view_ids)
    mode_index = int(_scalar(arrays["mode_index"], "mode_index", path))
    freq_hz = float(_scalar(arrays["freq_hz"], "freq_hz", path))
    matching_graphs = [graph for graph in graphs if graph.mode_index == mode_index]
    if len(matching_graphs) != 1:
        raise ValueError(
            f"{path} mode_index={mode_index} must match exactly one loaded graph"
        )
    matching_graph = matching_graphs[0]
    if not np.isclose(freq_hz, matching_graph.freq_hz, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{path} frequency does not match its observed graph")
    if matching_graph.source_observation_path != source_observation_path:
        raise ValueError(f"{path} source_observation_path does not match graph")
    if matching_graph.view_ids != view_ids:
        raise ValueError(f"{path} view_ids do not match graph")
    source_names = tuple(
        str(value) for value in np.asarray(arrays["residual_source_names"]).tolist()
    )
    if source_names != _RESIDUAL_SOURCE_NAMES:
        raise ValueError(f"{path} has unsupported residual source names")
    expected_shapes = {
        "alphas": (num_views,),
        "alpha_identifiable_mask": (num_views,),
        "point_observable_rank": (num_points,),
        "point_condition": (num_points,),
        "point_distinct_view_count": (num_points,),
        "point_distinct_valid_view_count": (num_points,),
        "point_precompletion_residual": (num_points,),
        "staged_point_solution_status": (num_points,),
        "view_sample_count": (num_points, num_views),
        "view_effective_weight": (num_points, num_views),
        "view_mode_mean": (num_points, num_views, 2),
        "view_signal_energy": (num_points, num_views),
        "view_within_sse": (num_points, num_views),
        "view_cross_sse": (num_points, num_views),
        "view_within_residual": (num_points, num_views),
        "view_cross_residual": (num_points, num_views),
        "point_effective_weight": (num_points,),
        "point_signal_energy": (num_points,),
        "point_signal_rms": (num_points,),
        "point_within_sse": (num_points,),
        "point_cross_sse": (num_points,),
        "point_total_sse": (num_points,),
        "point_within_residual": (num_points,),
        "point_cross_residual": (num_points,),
        "point_replayed_total_residual": (num_points,),
        "point_within_fraction": (num_points,),
        "point_worst_within_view": (num_points,),
        "point_worst_cross_view": (num_points,),
        "selected_multiview_mask": (num_points,),
        "residual_candidate_mask": (num_points,),
        "residual_rejected_mask": (num_points,),
        "anchor_mask": (num_points,),
        "low_modal_energy_mask": (num_points,),
        "residual_source_class": (num_points,),
    }
    for name, expected_shape in expected_shapes.items():
        if np.asarray(arrays[name]).shape != expected_shape:
            raise ValueError(f"{path} field {name} must have shape {expected_shape}")
    for name in (
        "alpha_identifiable_mask",
        "selected_multiview_mask",
        "residual_candidate_mask",
        "residual_rejected_mask",
        "anchor_mask",
        "low_modal_energy_mask",
    ):
        if np.asarray(arrays[name]).dtype != np.bool_:
            raise ValueError(f"{path} field {name} must be boolean")
    source_class = np.asarray(arrays["residual_source_class"])
    if (
        not np.issubdtype(source_class.dtype, np.integer)
        or np.any(source_class < 0)
        or np.any(source_class >= len(_RESIDUAL_SOURCE_NAMES))
    ):
        raise ValueError(f"{path} residual_source_class contains invalid values")
    staged_status = np.asarray(arrays["staged_point_solution_status"])
    partial_mask = staged_partial_mask(staged_status)
    selected_multiview_mask = np.asarray(
        arrays["selected_multiview_mask"],
        dtype=bool,
    )
    expected_selected_multiview = np.zeros((num_points,), dtype=bool)
    graph_multiview = (
        matching_graph.node_observed_view_mask.sum(axis=1) >= 2
    )
    expected_selected_multiview[
        matching_graph.node_gaussian_indices[graph_multiview]
    ] = True
    if not np.array_equal(
        selected_multiview_mask,
        expected_selected_multiview,
    ):
        raise ValueError(
            f"{path} selected_multiview_mask does not match observed graph"
        )
    for graph in graphs:
        if graph.source_checkpoint != source_checkpoint:
            raise ValueError(f"{path} source_checkpoint does not match graph")
        if graph.num_foreground_gaussians != num_points:
            raise ValueError(f"{path} foreground count does not match graph")
    if matching_graph.node_gaussian_indices.shape[0] and not np.allclose(
        points[matching_graph.node_gaussian_indices],
        matching_graph.node_points_world,
        rtol=1.0e-6,
        atol=1.0e-5,
    ):
        raise ValueError(f"{path} Gaussian centers do not match graph nodes")
    anchor_mask = np.asarray(arrays["anchor_mask"], dtype=bool)
    if not np.array_equal(anchor_mask, staged_status == _STAGED_ANCHOR_STATUS):
        raise ValueError(f"{path} staged point status does not match anchor_mask")
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
    return AnchorResidualViewData(
        artifact_path=path,
        source_checkpoint=source_checkpoint,
        mode_index=mode_index,
        freq_hz=freq_hz,
        points_world=points,
        view_ids=view_ids,
        point_precompletion_residual=np.asarray(
            arrays["point_precompletion_residual"], dtype=np.float32
        ),
        point_within_residual=np.asarray(
            arrays["point_within_residual"], dtype=np.float32
        ),
        point_cross_residual=np.asarray(
            arrays["point_cross_residual"], dtype=np.float32
        ),
        point_within_fraction=np.asarray(
            arrays["point_within_fraction"], dtype=np.float32
        ),
        point_signal_rms=np.asarray(arrays["point_signal_rms"], dtype=np.float32),
        point_worst_within_view=np.asarray(
            arrays["point_worst_within_view"], dtype=np.int8
        ),
        point_worst_cross_view=np.asarray(
            arrays["point_worst_cross_view"], dtype=np.int8
        ),
        selected_multiview_mask=selected_multiview_mask,
        residual_rejected_mask=np.asarray(
            arrays["residual_rejected_mask"], dtype=bool
        ),
        anchor_mask=anchor_mask,
        partial_mask=partial_mask,
        residual_source_class=source_class.astype(np.int8),
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


class ObservedGraphViewer:
    def __init__(
        self,
        server: Any,
        graphs: tuple[ObservedGraphViewData, ...],
        *,
        max_visible_edges: int,
        line_width: float,
        observed_point_size: float,
        isolated_point_size: float,
        world_center: np.ndarray,
        rigid_manifest: RigidManifestViewData | None = None,
    ) -> None:
        if not graphs:
            raise ValueError("Observed graph viewer requires at least one graph")
        self.server = server
        self.graphs = graphs
        self.labels = tuple(graph.label for graph in graphs)
        self.scene_prefix = "/observed_structure_graph"
        self.rigid_modes = (
            {mode.mode_index: mode for mode in rigid_manifest.modes}
            if rigid_manifest is not None
            else {}
        )
        if rigid_manifest is not None and set(self.rigid_modes) != {
            graph.mode_index for graph in graphs
        }:
            raise ValueError("Rigid manifest modes do not match observed graphs")
        self.world_center = np.asarray(world_center, dtype=np.float32)
        if self.world_center.shape != (3,) or not np.isfinite(
            self.world_center
        ).all():
            raise ValueError("world_center must be finite (3,)")
        self._line_handle = None
        self._node_handle = None
        self._isolated_handle = None
        self._rigid_seed_handle = None
        self._quarantined_rigid_handle = None
        self._single_view_rigid_fill_handle = None
        self._completed_fill_handle = None
        self._unresolved_fill_handle = None
        self._update_lock = threading.Lock()
        self._playback_thread = None
        self._display_mode_index = None
        self._selected_edge_indices = np.empty((0,), dtype=np.int64)

        max_edges = max(int(graph.edge_index.shape[0]) for graph in graphs)
        edge_step = max(max_edges // 200, 1)
        partial_component_colors = (
            (
                "partial fill status",
                "partial observable rank",
                "partial fill nullity",
                "trusted-KNN support",
                "partial ray-motion ratio",
                "partial post-fill rejection",
            )
            if rigid_manifest is not None
            and rigid_manifest.rigid_motion_fill_stage == "single-view-components"
            else ()
        )
        rigid_color_options = (
            (
                "component residual",
                "component rank",
                "finite-amplitude drift",
                "finite-drift anomaly",
                "single-view anomaly",
                "singular-ratio anomaly",
                "motion-RMS anomaly",
            )
            + partial_component_colors
            if rigid_manifest is not None
            else ()
        )
        with server.gui.add_folder("Observed Gaussian structure graph"):
            self.show_graph = server.gui.add_checkbox("Show graph", True)
            self.mode = server.gui.add_dropdown(
                "Mode",
                options=self.labels,
                initial_value=self.labels[0],
            )
            self.edge_color = server.gui.add_dropdown(
                "Edge color",
                options=(
                    "component",
                    "depth support",
                    "combined weight",
                )
                + rigid_color_options,
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
            self.show_nodes = server.gui.add_checkbox(
                "Show observed nodes",
                True,
            )
            self.node_point_size = server.gui.add_slider(
                "Observed-node point size",
                min=0.0001,
                max=0.008,
                step=0.0001,
                initial_value=observed_point_size,
            )
            self.show_isolated = server.gui.add_checkbox(
                "Show isolated observed nodes",
                True,
            )
            self.isolated_point_size = server.gui.add_slider(
                "Isolated-node point size",
                min=0.0001,
                max=0.008,
                step=0.0001,
                initial_value=isolated_point_size,
            )
            if rigid_manifest is None:
                self.graph_geometry = None
                self.phase = None
                self.motion_scale = None
                self.play = None
                self.playback_speed = None
                self.playback_fps = None
                self.minimum_valid_views = None
                self.minimum_log10_singular_ratio = None
                self.maximum_finite_drift = None
                self.motion_rms_percentile = None
                self.show_rigid_seeds = None
                self.show_quarantined_rigid = None
                self.show_single_view_rigid_fill = None
                self.show_completed_fill = None
                self.show_unresolved_fill = None
            else:
                self.graph_geometry = server.gui.add_dropdown(
                    "Graph geometry",
                    options=("canonical", "raw rigid", "deformed"),
                    initial_value="canonical",
                )
                self.phase = server.gui.add_slider(
                    "Phase (rad)",
                    min=0.0,
                    max=float(2.0 * np.pi),
                    step=float(2.0 * np.pi / 128.0),
                    initial_value=0.0,
                )
                self.motion_scale = server.gui.add_slider(
                    "Motion scale",
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    initial_value=1.0,
                )
                self.play = server.gui.add_checkbox("Play", False)
                self.playback_speed = server.gui.add_slider(
                    "Playback speed (x)",
                    min=0.1,
                    max=3.0,
                    step=0.05,
                    initial_value=1.0,
                )
                self.playback_fps = server.gui.add_slider(
                    "Playback FPS",
                    min=1,
                    max=30,
                    step=1,
                    initial_value=10,
                )
                self.minimum_valid_views = server.gui.add_slider(
                    "Minimum valid views",
                    min=1,
                    max=max(
                        2,
                        rigid_manifest.rigid_seed_min_valid_views,
                        max(len(graph.view_ids) for graph in graphs),
                    ),
                    step=1,
                    initial_value=rigid_manifest.rigid_seed_min_valid_views,
                )
                self.minimum_log10_singular_ratio = server.gui.add_slider(
                    "Minimum log10 singular ratio",
                    min=-12.0,
                    max=0.0,
                    step=0.25,
                    initial_value=max(
                        -12.0,
                        float(
                            np.log10(
                                max(
                                    rigid_manifest.rigid_seed_min_singular_ratio,
                                    1.0e-12,
                                )
                            )
                        ),
                    ),
                )
                self.maximum_finite_drift = server.gui.add_slider(
                    "Maximum finite drift",
                    min=0.0,
                    max=max(
                        10.0,
                        10.0 * rigid_manifest.rigid_seed_max_finite_drift,
                    ),
                    step=0.1,
                    initial_value=rigid_manifest.rigid_seed_max_finite_drift,
                )
                self.motion_rms_percentile = server.gui.add_slider(
                    "Motion RMS percentile",
                    min=50.0,
                    max=100.0,
                    step=0.5,
                    initial_value=99.0,
                )
                self.show_rigid_seeds = server.gui.add_checkbox(
                    "Show rigid seeds",
                    False,
                )
                self.show_quarantined_rigid = server.gui.add_checkbox(
                    "Show quarantined rigid candidates",
                    False,
                )
                if rigid_manifest.motion_fill_enabled:
                    self.show_single_view_rigid_fill = server.gui.add_checkbox(
                        "Show single-view rigid fill",
                        False,
                    )
                    self.show_completed_fill = server.gui.add_checkbox(
                        "Show completed fill",
                        False,
                    )
                    self.show_unresolved_fill = server.gui.add_checkbox(
                        "Show unresolved fill",
                        False,
                    )
                else:
                    self.show_single_view_rigid_fill = None
                    self.show_completed_fill = None
                    self.show_unresolved_fill = None
        rebuild_handles = (
            self.show_graph,
            self.mode,
            self.edge_color,
            self.max_visible_edges,
            self.show_nodes,
            self.show_isolated,
        )
        for handle in rebuild_handles:
            handle.on_update(self._update)
        for handle in (
            self.line_width,
            self.node_point_size,
            self.isolated_point_size,
        ):
            handle.on_update(self._update_style)
        if rigid_manifest is not None:
            assert self.graph_geometry is not None
            assert self.phase is not None
            assert self.motion_scale is not None
            assert self.play is not None
            assert self.playback_speed is not None
            assert self.playback_fps is not None
            assert self.minimum_valid_views is not None
            assert self.minimum_log10_singular_ratio is not None
            assert self.maximum_finite_drift is not None
            assert self.motion_rms_percentile is not None
            assert self.show_rigid_seeds is not None
            assert self.show_quarantined_rigid is not None
            geometry_handles = [
                self.graph_geometry,
                self.phase,
                self.motion_scale,
            ]
            for handle in geometry_handles:
                handle.on_update(self._update_geometry)
            for handle in (
                self.minimum_valid_views,
                self.minimum_log10_singular_ratio,
                self.maximum_finite_drift,
                self.motion_rms_percentile,
            ):
                handle.on_update(self._update)
            rigid_visibility_handles = [
                self.show_rigid_seeds,
                self.show_quarantined_rigid,
            ]
            if self.show_single_view_rigid_fill is not None:
                rigid_visibility_handles.append(
                    self.show_single_view_rigid_fill
                )
            if self.show_completed_fill is not None:
                rigid_visibility_handles.append(self.show_completed_fill)
            if self.show_unresolved_fill is not None:
                rigid_visibility_handles.append(self.show_unresolved_fill)
            for handle in rigid_visibility_handles:
                handle.on_update(self._update)
            self.play.on_update(self._on_playback_toggle)
        self._update()
        if rigid_manifest is not None:
            self._playback_thread = threading.Thread(
                target=self._playback_loop,
                daemon=True,
            )
            self._playback_thread.start()

    def _selected_graph(self) -> ObservedGraphViewData:
        selected = str(self.mode.value)
        if selected not in self.labels:
            raise ValueError(f"Unknown observed graph mode: {selected}")
        return self.graphs[self.labels.index(selected)]

    def _selected_rigid_mode(
        self,
        graph: ObservedGraphViewData,
    ) -> RigidModeViewData | None:
        return self.rigid_modes.get(graph.mode_index)

    def _on_playback_toggle(self, _event: Any = None) -> None:
        assert self.play is not None
        assert self.graph_geometry is not None
        if bool(self.play.value) and str(self.graph_geometry.value) != "deformed":
            self.graph_geometry.value = "deformed"

    def _playback_loop(self) -> None:
        assert self.play is not None
        assert self.graph_geometry is not None
        assert self.phase is not None
        assert self.playback_speed is not None
        assert self.playback_fps is not None
        last_time = time.perf_counter()
        while True:
            try:
                now = time.perf_counter()
                elapsed_seconds = max(now - last_time, 0.0)
                last_time = now
                if bool(self.play.value):
                    if str(self.graph_geometry.value) != "deformed":
                        self.graph_geometry.value = "deformed"
                    graph = self._selected_graph()
                    rigid_mode = self._selected_rigid_mode(graph)
                    if rigid_mode is None:
                        raise RuntimeError(
                            "Playback requires a rigid result for the selected mode"
                        )
                    self.phase.value = advance_playback_phase(
                        float(self.phase.value),
                        elapsed_seconds,
                        float(rigid_mode.freq_hz),
                        float(self.playback_speed.value),
                    )
                fps = max(float(self.playback_fps.value), 1.0)
                time.sleep(1.0 / fps)
            except Exception as exc:
                print(
                    "Observed graph playback thread failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                raise

    def _current_display_geometry(
        self,
    ) -> tuple[
        ObservedGraphViewData,
        RigidModeViewData | None,
        np.ndarray,
        np.ndarray | None,
    ]:
        graph = self._selected_graph()
        rigid_mode = self._selected_rigid_mode(graph)
        all_display_points = (
            rigid_mode.points_world if rigid_mode is not None else None
        )
        if rigid_mode is not None:
            assert self.graph_geometry is not None
            assert self.phase is not None
            assert self.motion_scale is not None
            if str(self.graph_geometry.value) == "deformed":
                all_display_points = deform_modal_points(
                    rigid_mode.points_world,
                    rigid_mode.phi,
                    float(self.phase.value),
                    float(self.motion_scale.value),
                )
            elif str(self.graph_geometry.value) == "raw rigid":
                all_display_points = deform_modal_points(
                    rigid_mode.points_world,
                    rigid_mode.raw_rigid_phi,
                    float(self.phase.value),
                    float(self.motion_scale.value),
                )
            elif str(self.graph_geometry.value) != "canonical":
                raise ValueError(
                    f"Unknown graph geometry: {self.graph_geometry.value}"
                )
        graph_points = (
            graph.node_points_world
            if all_display_points is None
            else all_display_points[graph.node_gaussian_indices]
        )
        centered_points = center_world_points(graph_points, self.world_center)
        centered_all_points = (
            None
            if all_display_points is None
            else center_world_points(all_display_points, self.world_center)
        )
        return graph, rigid_mode, centered_points, centered_all_points

    def _remove_scene_nodes(self) -> None:
        for attribute in (
            "_line_handle",
            "_node_handle",
            "_isolated_handle",
            "_rigid_seed_handle",
            "_quarantined_rigid_handle",
            "_single_view_rigid_fill_handle",
            "_completed_fill_handle",
            "_unresolved_fill_handle",
        ):
            handle = getattr(self, attribute)
            if handle is not None:
                handle.remove()
                setattr(self, attribute, None)

    def _update_geometry(self, _event: Any = None) -> None:
        with self._update_lock:
            graph, rigid_mode, centered_points, centered_all_points = (
                self._current_display_geometry()
            )
            if self._display_mode_index != graph.mode_index:
                # A mode-change rebuild callback may still be queued while the
                # playback thread advances phase. The rebuild owns that mode
                # transition; the following playback tick will update it.
                return
            if self._line_handle is not None:
                edges = graph.edge_index[self._selected_edge_indices]
                self._line_handle.points = centered_points[edges]
            if self._node_handle is not None:
                self._node_handle.points = centered_points
            if self._isolated_handle is not None:
                self._isolated_handle.points = centered_points[graph.isolated_mask]
            if rigid_mode is None:
                return
            assert centered_all_points is not None
            for attribute, mask in (
                ("_rigid_seed_handle", rigid_mode.rigid_seed_mask),
                (
                    "_quarantined_rigid_handle",
                    rigid_mode.quarantined_rigid_mask,
                ),
                (
                    "_single_view_rigid_fill_handle",
                    rigid_mode.single_view_rigid_fill_mask,
                ),
                ("_completed_fill_handle", rigid_mode.completed_fill_mask),
                ("_unresolved_fill_handle", rigid_mode.unresolved_fill_mask),
            ):
                handle = getattr(self, attribute)
                if handle is not None:
                    handle.points = centered_all_points[mask]

    def _update_style(self, _event: Any = None) -> None:
        with self._update_lock:
            if self._line_handle is not None:
                self._line_handle.line_width = float(self.line_width.value)
            for attribute in (
                "_node_handle",
                "_rigid_seed_handle",
                "_quarantined_rigid_handle",
                "_single_view_rigid_fill_handle",
                "_completed_fill_handle",
                "_unresolved_fill_handle",
            ):
                handle = getattr(self, attribute)
                if handle is not None:
                    handle.point_size = float(self.node_point_size.value)
            if self._isolated_handle is not None:
                self._isolated_handle.point_size = float(
                    self.isolated_point_size.value
                )

    def _update(self, _event: Any = None) -> None:
        with self._update_lock:
            self._remove_scene_nodes()
            graph, rigid_mode, centered_points, centered_all_points = (
                self._current_display_geometry()
            )
            self._display_mode_index = graph.mode_index
            self._selected_edge_indices = np.empty((0,), dtype=np.int64)
            if bool(self.show_graph.value):
                selected_edges = stable_uniform_edge_indices(
                    graph.edge_index.shape[0],
                    int(self.max_visible_edges.value),
                )
                self._selected_edge_indices = selected_edges
                edges = graph.edge_index[selected_edges]
                if edges.shape[0]:
                    color_mode = str(self.edge_color.value)
                    if color_mode == "component":
                        edge_colors = graph_component_colors(
                            graph.component_index[edges[:, 0]]
                        )
                    elif color_mode == "depth support":
                        edge_colors = graph_scalar_colors(
                            graph.edge_depth_score[selected_edges]
                        )
                    elif color_mode == "combined weight":
                        edge_colors = graph_scalar_colors(
                            np.log1p(graph.edge_combined_weight[selected_edges])
                        )
                    elif color_mode == "component residual":
                        if rigid_mode is None:
                            raise ValueError(
                                "Component residual color requires a rigid manifest"
                            )
                        edge_colors = graph_scalar_colors(
                            rigid_mode.component_normalized_weighted_residual[
                                rigid_mode.edge_component_index[selected_edges]
                            ]
                        )
                    elif color_mode == "component rank":
                        if rigid_mode is None:
                            raise ValueError(
                                "Component rank color requires a rigid manifest"
                            )
                        edge_colors = graph_scalar_colors(
                            rigid_mode.component_rank[
                                rigid_mode.edge_component_index[selected_edges]
                            ].astype(np.float32)
                        )
                    elif color_mode == "finite-amplitude drift":
                        if rigid_mode is None:
                            raise ValueError(
                                "Finite drift color requires a rigid manifest"
                            )
                        edge_colors = graph_scalar_colors(
                            rigid_mode.edge_finite_drift_max[selected_edges]
                        )
                    elif color_mode == "finite-drift anomaly":
                        if (
                            rigid_mode is None
                            or self.maximum_finite_drift is None
                        ):
                            raise ValueError(
                                "Finite drift anomaly color requires a rigid manifest"
                            )
                        component_anomaly = (
                            rigid_mode.component_finite_drift_max
                            > float(self.maximum_finite_drift.value)
                        )
                        edge_colors = rigid_component_anomaly_colors(
                            component_anomaly[
                                rigid_mode.edge_component_index[selected_edges]
                            ]
                        )
                    elif color_mode == "single-view anomaly":
                        if rigid_mode is None or self.minimum_valid_views is None:
                            raise ValueError(
                                "Valid-view anomaly color requires a rigid manifest"
                            )
                        component_anomaly = (
                            rigid_mode.component_distinct_valid_view_count
                            < int(self.minimum_valid_views.value)
                        )
                        edge_colors = rigid_component_anomaly_colors(
                            component_anomaly[
                                rigid_mode.edge_component_index[selected_edges]
                            ]
                        )
                    elif color_mode == "singular-ratio anomaly":
                        if (
                            rigid_mode is None
                            or self.minimum_log10_singular_ratio is None
                        ):
                            raise ValueError(
                                "Singular-ratio anomaly color requires a rigid manifest"
                            )
                        minimum_ratio = 10.0 ** float(
                            self.minimum_log10_singular_ratio.value
                        )
                        component_anomaly = (
                            rigid_mode.component_singular_ratio < minimum_ratio
                        )
                        edge_colors = rigid_component_anomaly_colors(
                            component_anomaly[
                                rigid_mode.edge_component_index[selected_edges]
                            ]
                        )
                    elif color_mode == "motion-RMS anomaly":
                        if rigid_mode is None or self.motion_rms_percentile is None:
                            raise ValueError(
                                "Motion-RMS anomaly color requires a rigid manifest"
                            )
                        motion_threshold = float(
                            np.percentile(
                                rigid_mode.component_motion_rms,
                                float(self.motion_rms_percentile.value),
                            )
                        )
                        component_anomaly = (
                            rigid_mode.component_motion_rms >= motion_threshold
                        )
                        edge_colors = rigid_component_anomaly_colors(
                            component_anomaly[
                                rigid_mode.edge_component_index[selected_edges]
                            ]
                        )
                    elif color_mode == "partial fill status":
                        if rigid_mode is None:
                            raise ValueError(
                                "Partial fill status requires a rigid manifest"
                            )
                        edge_components = rigid_mode.edge_component_index[
                            selected_edges
                        ]
                        edge_colors = np.full(
                            (selected_edges.size, 3),
                            _RIGID_DIAGNOSTIC_NORMAL_COLOR,
                            dtype=np.float32,
                        )
                        single_view = (
                            rigid_mode.component_distinct_valid_view_count[
                                edge_components
                            ]
                            == 1
                        )
                        edge_colors[single_view] = _RIGID_DIAGNOSTIC_ANOMALY_COLOR
                        completed = rigid_mode.single_view_component_completion_mask[
                            edge_components
                        ]
                        edge_colors[completed] = _SINGLE_VIEW_RIGID_FILL_COLOR
                    elif color_mode == "partial observable rank":
                        if (
                            rigid_mode is None
                            or rigid_mode.single_view_component_observable_rank is None
                        ):
                            raise ValueError(
                                "Partial observable rank requires component-only fill"
                            )
                        edge_colors = graph_scalar_colors(
                            rigid_mode.single_view_component_observable_rank[
                                rigid_mode.edge_component_index[selected_edges]
                            ].astype(np.float32)
                        )
                    elif color_mode == "partial fill nullity":
                        if (
                            rigid_mode is None
                            or rigid_mode.single_view_component_fill_nullity is None
                        ):
                            raise ValueError(
                                "Partial fill nullity requires component-only fill"
                            )
                        edge_colors = graph_scalar_colors(
                            rigid_mode.single_view_component_fill_nullity[
                                rigid_mode.edge_component_index[selected_edges]
                            ].astype(np.float32)
                        )
                    elif color_mode == "trusted-KNN support":
                        if (
                            rigid_mode is None
                            or rigid_mode.single_view_component_trusted_knn_edge_count
                            is None
                        ):
                            raise ValueError(
                                "Trusted-KNN support requires component-only fill"
                            )
                        edge_colors = graph_scalar_colors(
                            np.log1p(
                                rigid_mode.single_view_component_trusted_knn_edge_count[
                                    rigid_mode.edge_component_index[selected_edges]
                                ]
                            )
                        )
                    elif color_mode == "partial ray-motion ratio":
                        if (
                            rigid_mode is None
                            or rigid_mode.single_view_component_ray_motion_ratio
                            is None
                        ):
                            raise ValueError(
                                "Partial ray-motion ratio requires component-only fill"
                            )
                        edge_colors = graph_scalar_colors(
                            np.log1p(
                                rigid_mode.single_view_component_ray_motion_ratio[
                                    rigid_mode.edge_component_index[selected_edges]
                                ]
                            )
                        )
                    elif color_mode == "partial post-fill rejection":
                        if (
                            rigid_mode is None
                            or rigid_mode.single_view_component_finite_drift_rejected_mask
                            is None
                            or rigid_mode.single_view_component_motion_rms_rejected_mask
                            is None
                            or rigid_mode.single_view_component_postfill_retained_mask
                            is None
                        ):
                            raise ValueError(
                                "Partial post-fill rejection requires component-only fill"
                            )
                        edge_components = rigid_mode.edge_component_index[
                            selected_edges
                        ]
                        edge_colors = np.full(
                            (selected_edges.size, 3),
                            _RIGID_DIAGNOSTIC_NORMAL_COLOR,
                            dtype=np.float32,
                        )
                        retained = (
                            rigid_mode.single_view_component_postfill_retained_mask[
                                edge_components
                            ]
                        )
                        finite_rejected = (
                            rigid_mode.single_view_component_finite_drift_rejected_mask[
                                edge_components
                            ]
                        )
                        motion_rejected = (
                            rigid_mode.single_view_component_motion_rms_rejected_mask[
                                edge_components
                            ]
                        )
                        edge_colors[retained] = _SINGLE_VIEW_RIGID_FILL_COLOR
                        edge_colors[finite_rejected] = (
                            _RIGID_DIAGNOSTIC_ANOMALY_COLOR
                        )
                        edge_colors[motion_rejected] = (
                            _PARTIAL_MOTION_REJECTION_COLOR
                        )
                        edge_colors[finite_rejected & motion_rejected] = (
                            _PARTIAL_BOTH_REJECTION_COLOR
                        )
                    else:
                        raise ValueError(
                            f"Unknown observed graph edge color: {color_mode}"
                        )
                    self._line_handle = self.server.scene.add_line_segments(
                        f"{self.scene_prefix}/edges",
                        points=centered_points[edges],
                        colors=np.repeat(edge_colors[:, None, :], 2, axis=1),
                        line_width=float(self.line_width.value),
                    )
            if bool(self.show_nodes.value) and graph.node_points_world.shape[0]:
                self._node_handle = self.server.scene.add_point_cloud(
                    f"{self.scene_prefix}/nodes",
                    points=centered_points,
                    colors=graph.node_colors_rgb,
                    point_size=float(self.node_point_size.value),
                    point_shape="circle",
                )
            if bool(self.show_isolated.value):
                isolated_points = centered_points[graph.isolated_mask]
                if isolated_points.shape[0]:
                    self._isolated_handle = self.server.scene.add_point_cloud(
                        f"{self.scene_prefix}/isolated",
                        points=isolated_points,
                        colors=np.full(
                            (isolated_points.shape[0], 3),
                            [1.0, 0.0, 0.0],
                            dtype=np.float32,
                        ),
                        point_size=float(self.isolated_point_size.value),
                        point_shape="circle",
                    )
            if rigid_mode is not None:
                assert centered_all_points is not None
                assert self.show_rigid_seeds is not None
                assert self.show_quarantined_rigid is not None
                overlays = [
                    (
                        self.show_rigid_seeds,
                        rigid_mode.rigid_seed_mask,
                        _RIGID_SEED_COLOR,
                        "rigid_seeds",
                        "_rigid_seed_handle",
                    ),
                    (
                        self.show_quarantined_rigid,
                        rigid_mode.quarantined_rigid_mask,
                        _QUARANTINED_RIGID_COLOR,
                        "quarantined_rigid_candidates",
                        "_quarantined_rigid_handle",
                    ),
                ]
                if self.show_single_view_rigid_fill is not None:
                    overlays.append((
                        self.show_single_view_rigid_fill,
                        rigid_mode.single_view_rigid_fill_mask,
                        _SINGLE_VIEW_RIGID_FILL_COLOR,
                        "single_view_rigid_fill",
                        "_single_view_rigid_fill_handle",
                    ))
                if self.show_completed_fill is not None:
                    overlays.append((
                        self.show_completed_fill,
                        rigid_mode.completed_fill_mask,
                        _COMPLETED_FILL_COLOR,
                        "completed_fill",
                        "_completed_fill_handle",
                    ))
                if self.show_unresolved_fill is not None:
                    overlays.append((
                        self.show_unresolved_fill,
                        rigid_mode.unresolved_fill_mask,
                        _UNRESOLVED_FILL_COLOR,
                        "unresolved_fill",
                        "_unresolved_fill_handle",
                    ))
                for show_handle, mask, color, scene_name, attribute in overlays:
                    selected_points = centered_all_points[mask]
                    if bool(show_handle.value) and selected_points.shape[0]:
                        setattr(
                            self,
                            attribute,
                            self.server.scene.add_point_cloud(
                                f"{self.scene_prefix}/{scene_name}",
                                points=selected_points,
                                colors=np.broadcast_to(
                                    color,
                                    (selected_points.shape[0], 3),
                                ),
                                point_size=float(self.node_point_size.value),
                                point_shape="circle",
                            ),
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


class AnchorResidualDiagnosticViewer:
    def __init__(
        self,
        server: Any,
        diagnostics: AnchorResidualViewData,
        *,
        max_visible_points: int,
        point_size: float,
        world_center: np.ndarray,
    ) -> None:
        self.server = server
        self.diagnostics = diagnostics
        self.world_center = np.asarray(world_center, dtype=np.float32)
        self._point_handle = None
        self._partial_handle = None
        self._update_lock = threading.Lock()
        num_points = int(diagnostics.points_world.shape[0])
        partial_count = int(np.count_nonzero(diagnostics.partial_mask))
        point_step = max(num_points // 200, 1)
        with server.gui.add_folder("Anchor residual diagnostics"):
            self.show_diagnostics = server.gui.add_checkbox(
                "Show residual diagnostics",
                True,
            )
            self.filter = server.gui.add_dropdown(
                "Filter",
                options=(
                    "residual rejected",
                    "accepted anchors",
                    "all selected multiview",
                ),
                initial_value="residual rejected",
            )
            self.metric = server.gui.add_dropdown(
                "Color metric",
                options=(
                    "source class",
                    "within fraction",
                    "total residual",
                    "within residual",
                    "cross residual",
                    "modal signal RMS",
                    "worst within view",
                    "worst cross view",
                ),
                initial_value="within fraction",
            )
            self.show_partial = server.gui.add_checkbox(
                f"Show partial Gaussians ({partial_count})",
                False,
            )
            self.max_visible_points = server.gui.add_slider(
                "Max residual points",
                min=0,
                max=max(num_points, 1),
                step=point_step,
                initial_value=min(max_visible_points, num_points),
            )
            self.point_size = server.gui.add_slider(
                "Residual point size",
                min=0.0001,
                max=0.008,
                step=0.0001,
                initial_value=point_size,
            )
        for handle in (
            self.show_diagnostics,
            self.filter,
            self.metric,
            self.show_partial,
            self.max_visible_points,
            self.point_size,
        ):
            handle.on_update(self._update)
        self._update()

    def _selected_indices(self) -> np.ndarray:
        selected_filter = str(self.filter.value)
        if selected_filter == "residual rejected":
            mask = self.diagnostics.residual_rejected_mask
        elif selected_filter == "accepted anchors":
            mask = self.diagnostics.anchor_mask
        elif selected_filter == "all selected multiview":
            mask = self.diagnostics.selected_multiview_mask
        else:
            raise ValueError(f"Unknown residual diagnostic filter: {selected_filter}")
        indices = np.flatnonzero(mask)
        visible = stable_uniform_indices(
            indices.shape[0],
            int(self.max_visible_points.value),
        )
        return indices[visible]

    def _colors(self, indices: np.ndarray) -> np.ndarray:
        metric = str(self.metric.value)
        if metric == "source class":
            return anchor_residual_source_colors(
                self.diagnostics.residual_source_class[indices]
            )
        if metric == "within fraction":
            return anchor_residual_fraction_colors(
                self.diagnostics.point_within_fraction[indices]
            )
        if metric == "total residual":
            values = self.diagnostics.point_precompletion_residual[indices]
        elif metric == "within residual":
            values = self.diagnostics.point_within_residual[indices]
        elif metric == "cross residual":
            values = self.diagnostics.point_cross_residual[indices]
        elif metric == "modal signal RMS":
            values = self.diagnostics.point_signal_rms[indices]
        elif metric == "worst within view":
            return anchor_residual_view_colors(
                self.diagnostics.point_worst_within_view[indices]
            )
        elif metric == "worst cross view":
            return anchor_residual_view_colors(
                self.diagnostics.point_worst_cross_view[indices]
            )
        else:
            raise ValueError(f"Unknown residual diagnostic metric: {metric}")
        return anchor_residual_scalar_colors(values, logarithmic=True)

    def _update(self, _event: Any = None) -> None:
        with self._update_lock:
            if self._point_handle is not None:
                self._point_handle.remove()
                self._point_handle = None
            if self._partial_handle is not None:
                self._partial_handle.remove()
                self._partial_handle = None
            if bool(self.show_diagnostics.value):
                indices = self._selected_indices()
                if indices.shape[0]:
                    self._point_handle = self.server.scene.add_point_cloud(
                        "/anchor_residual_diagnostics/points",
                        points=center_world_points(
                            self.diagnostics.points_world[indices],
                            self.world_center,
                        ),
                        colors=self._colors(indices),
                        point_size=float(self.point_size.value),
                        point_shape="circle",
                    )
            if bool(self.show_partial.value):
                partial_indices = np.flatnonzero(self.diagnostics.partial_mask)
                visible = stable_uniform_indices(
                    partial_indices.shape[0],
                    int(self.max_visible_points.value),
                )
                partial_indices = partial_indices[visible]
                if partial_indices.shape[0]:
                    self._partial_handle = self.server.scene.add_point_cloud(
                        "/anchor_residual_diagnostics/partial_points",
                        points=center_world_points(
                            self.diagnostics.points_world[partial_indices],
                            self.world_center,
                        ),
                        colors=np.broadcast_to(
                            _PARTIAL_POINT_COLOR,
                            (partial_indices.shape[0], 3),
                        ).copy(),
                        point_size=float(self.point_size.value),
                        point_shape="circle",
                    )


def _observed_display_extent(
    graphs: tuple[ObservedGraphViewData, ...],
    world_center: np.ndarray,
) -> float:
    nonempty = [
        graph.node_points_world
        for graph in graphs
        if graph.node_points_world.shape[0]
    ]
    if not nonempty:
        return 1.0e-3
    points = center_world_points(np.concatenate(nonempty, axis=0), world_center)
    return max(float(np.max(points.max(axis=0) - points.min(axis=0))), 1.0e-3)


class SourceCameraViewer:
    def __init__(
        self,
        server: Any,
        cameras: tuple[SourceCameraViewData, ...],
        *,
        world_center: np.ndarray,
        scene_extent: float,
    ) -> None:
        import viser.transforms as vtf

        self.server = server
        self.camera_handles: dict[str, Any] = {}
        self.button_handles: list[Any] = []
        camera_colors = (
            (80, 150, 255),
            (255, 130, 70),
            (95, 200, 120),
            (210, 120, 255),
            (255, 210, 80),
            (80, 220, 220),
        )
        frustum_scale = 0.08 * float(scene_extent)
        with server.gui.add_folder("Source cameras"):
            show_cameras = server.gui.add_checkbox("Show cameras", True)
            reset_orbit = server.gui.add_button("Reset orbit center")
            for index, camera in enumerate(cameras):
                camera_to_world = np.linalg.inv(camera.world_to_camera)
                wxyz = vtf.SO3.from_matrix(camera_to_world[:3, :3]).wxyz
                position = (
                    camera_to_world[:3, 3]
                    - np.asarray(world_center, dtype=np.float64)
                )
                fov = float(
                    2.0
                    * np.arctan(
                        0.5 * camera.image_height / float(camera.K[1, 1])
                    )
                )
                aspect = float(camera.image_width) / float(camera.image_height)
                self.camera_handles[camera.label] = (
                    server.scene.add_camera_frustum(
                        f"/source_cameras/{camera.label}",
                        fov=fov,
                        aspect=aspect,
                        scale=frustum_scale,
                        color=camera_colors[index % len(camera_colors)],
                        wxyz=wxyz,
                        position=position,
                    )
                )
                button = server.gui.add_button(f"Go to {camera.label}")

                def _go_to_camera(
                    event: Any,
                    camera_wxyz: np.ndarray = wxyz.copy(),
                    camera_position: np.ndarray = position.copy(),
                    camera_fov: float = fov,
                ) -> None:
                    if event.client is None:
                        return
                    with event.client.atomic():
                        event.client.camera.position = camera_position
                        event.client.camera.look_at = (0.0, 0.0, 0.0)
                        event.client.camera.wxyz = camera_wxyz
                        event.client.camera.fov = camera_fov

                button.on_click(_go_to_camera)
                self.button_handles.append(button)

            def _toggle_cameras(event: Any) -> None:
                for handle in self.camera_handles.values():
                    handle.visible = bool(event.target.value)

            def _reset_orbit_center(event: Any) -> None:
                if event.client is not None:
                    event.client.camera.look_at = (0.0, 0.0, 0.0)

            show_cameras.on_update(_toggle_cameras)
            reset_orbit.on_click(_reset_orbit_center)


def _configure_initial_camera(
    server: Any,
    graphs: tuple[ObservedGraphViewData, ...],
    world_center: np.ndarray,
) -> None:
    nonempty = [
        graph.node_points_world
        for graph in graphs
        if graph.node_points_world.shape[0]
    ]
    if not nonempty:
        return
    extent = _observed_display_extent(graphs, world_center)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)
    server.initial_camera.position = tuple(
        float(value)
        for value in extent * np.asarray([1.2, -1.2, 0.8])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="View an observed Gaussian structure graph in modern Viser",
    )
    parser.add_argument(
        "--observed-graph-npz",
        type=Path,
        action="append",
        required=True,
        help=(
            "Version-1 observed Gaussian structure graph NPZ artifact. Repeat "
            "once per mode when viewing a multi-mode rigid manifest."
        ),
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
    parser.add_argument(
        "--residual-diagnostics-npz",
        type=Path,
        help="Optional version-1 anchor residual decomposition diagnostic",
    )
    parser.add_argument(
        "--rigid-manifest",
        type=Path,
        help=(
            "Optional version-1 rigid-components modal manifest for deformed "
            "graph and solve-result diagnostics"
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-visible-edges", type=int, default=20000)
    parser.add_argument("--line-width", type=float, default=1.0)
    parser.add_argument("--observed-point-size", type=float, default=0.0009)
    parser.add_argument("--isolated-point-size", type=float, default=0.002)
    parser.add_argument("--gaussian-scale", type=float, default=1.0)
    parser.add_argument("--coverage-max-visible-points", type=int, default=50000)
    parser.add_argument("--coverage-point-size", type=float, default=0.0009)
    parser.add_argument("--residual-max-visible-points", type=int, default=50000)
    parser.add_argument("--residual-point-size", type=float, default=0.0009)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.port <= 0 or args.port > 65535:
        raise ValueError("--port must lie in [1,65535]")
    if args.max_visible_edges < 0:
        raise ValueError("--max-visible-edges must be non-negative")
    if args.coverage_max_visible_points < 0:
        raise ValueError("--coverage-max-visible-points must be non-negative")
    if args.residual_max_visible_points < 0:
        raise ValueError("--residual-max-visible-points must be non-negative")
    if not np.isfinite(args.line_width) or not 0.1 <= args.line_width <= 10.0:
        raise ValueError("--line-width must lie in [0.1,10.0]")
    if not np.isfinite(args.gaussian_scale) or not 0.1 <= args.gaussian_scale <= 3.0:
        raise ValueError("--gaussian-scale must lie in [0.1,3.0]")
    for name in (
        "observed_point_size",
        "isolated_point_size",
        "coverage_point_size",
        "residual_point_size",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or not 0.0001 <= value <= 0.008:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} must lie in [0.0001,0.008]")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    graphs = tuple(
        _load_observed_graph_archive(path)
        for path in args.observed_graph_npz
    )
    mode_indices = [graph.mode_index for graph in graphs]
    if len(set(mode_indices)) != len(mode_indices):
        raise ValueError("--observed-graph-npz inputs contain duplicate modes")
    world_center = observed_world_center(graphs)
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
    residual_diagnostics = None
    if args.residual_diagnostics_npz is not None:
        residual_diagnostics = load_anchor_residual_diagnostics(
            args.residual_diagnostics_npz,
            graphs,
            gaussians,
        )
    rigid_manifest = (
        load_rigid_manifest(args.rigid_manifest, graphs, gaussians)
        if args.rigid_manifest is not None
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
    if rigid_manifest is not None and not hasattr(
        SceneApi, "add_camera_frustum"
    ):
        raise RuntimeError(
            "Installed Viser lacks SceneApi.add_camera_frustum "
            f"(detected version {viser_version})"
        )

    server = viser.ViserServer(
        host=args.host,
        port=args.port,
        label="Observed Gaussian structure graph",
    )
    server.scene.set_up_direction("+z")
    _configure_initial_camera(server, graphs, world_center)
    source_camera_viewer = (
        SourceCameraViewer(
            server,
            rigid_manifest.source_cameras,
            world_center=world_center,
            scene_extent=_observed_display_extent(graphs, world_center),
        )
        if rigid_manifest is not None
        else None
    )
    if gaussians is not None:
        StaticGaussianViewer(
            server,
            gaussians,
            splat_scale=args.gaussian_scale,
            world_center=world_center,
        )
    ObservedGraphViewer(
        server,
        graphs,
        max_visible_edges=args.max_visible_edges,
        line_width=args.line_width,
        observed_point_size=args.observed_point_size,
        isolated_point_size=args.isolated_point_size,
        world_center=world_center,
        rigid_manifest=rigid_manifest,
    )
    if coverage is not None:
        ObservationCoverageViewer(
            server,
            coverage,
            max_visible_points=args.coverage_max_visible_points,
            point_size=args.coverage_point_size,
            world_center=world_center,
        )
    if residual_diagnostics is not None:
        AnchorResidualDiagnosticViewer(
            server,
            residual_diagnostics,
            max_visible_points=args.residual_max_visible_points,
            point_size=args.residual_point_size,
            world_center=world_center,
        )
    print(
        "Loaded observed graph with "
        f"{graphs[0].edge_index.shape[0]} edge(s) and "
        f"{graphs[0].node_points_world.shape[0]} observed node(s); "
        f"pruned {graphs[0].graph.counts['component_pruned_component_count']} "
        "small component(s), "
        f"{graphs[0].graph.counts['component_pruned_edge_count']} edge(s), and "
        f"isolated {graphs[0].graph.counts['component_pruned_node_count']} node(s)."
    )
    print(
        "Viewer world origin is the observed-node AABB center: "
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
    if source_camera_viewer is not None:
        print(
            f"Loaded {len(source_camera_viewer.camera_handles)} source camera(s) "
            "from the rigid manifest view configs."
        )
    if coverage is not None:
        print(
            f"Loaded observation coverage for {coverage.points_world.shape[0]} "
            f"Gaussians at K={coverage.k_values.tolist()}."
        )
    if residual_diagnostics is not None:
        print(
            "Loaded anchor residual decomposition for mode "
            f"{residual_diagnostics.mode_index} at "
            f"{residual_diagnostics.freq_hz:.6g} Hz with "
            f"{int(np.count_nonzero(residual_diagnostics.partial_mask))} "
            "staged partial Gaussian(s)."
        )
    if rigid_manifest is not None:
        mode = rigid_manifest.modes[0]
        print(
            "Loaded rigid component result for mode "
            f"{mode.mode_index} at {mode.freq_hz:.6g} Hz with "
            f"{int(np.count_nonzero(mode.rigid_seed_mask))} trusted rigid "
            "seed(s), "
            f"{int(np.count_nonzero(mode.quarantined_rigid_mask))} quarantined "
            "rigid candidate(s), "
            f"{int(np.count_nonzero(mode.completed_fill_mask))} completed fill "
            "point(s), and "
            f"{int(np.count_nonzero(mode.unresolved_fill_mask))} unresolved fill "
            "point(s)."
        )
    print(
        f"Viser {viser_version} listening on {server.get_host()}:{server.get_port()}"
    )
    server.sleep_forever()


if __name__ == "__main__":
    main()
