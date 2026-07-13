"""Run spatial KNN nullspace completion on an existing three-view sphere toy.

This is an experimental post-processing step. It never rewrites the staged
solver outputs that serve as the no-fill baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from modal_surface.motion_fill import (
    KnnCandidateSet,
    KnnGraph,
    MotionFillResult,
    build_knn_graph,
    fill_nullspace_motion,
    query_knn_candidates,
    validate_motion_fill_inputs,
)
from modal_surface.optimization_staged import (
    POINT_STATUS_COMPLETED_OBSERVED,
    POINT_STATUS_COMPLETED_UNOBSERVED,
    POINT_STATUS_NAMES,
)


_TRACKS = (
    {
        "key": "linear",
        "label": "Linear",
        "observations": "observations/toy_sphere_observations.npz",
        "solved": "latents/solved_staged.npz",
        "ground_truth": "latents/gt_motion.npz",
    },
    {
        "key": "tilted_ellipse",
        "label": "Tilted ellipse",
        "observations": "observations/toy_sphere_tilted_ellipse_observations.npz",
        "solved": "latents/solved_staged_tilted_ellipse.npz",
        "ground_truth": "latents/gt_tilted_ellipse_motion.npz",
    },
)
_LSMR_ATOL = 1e-10
_LSMR_BTOL = 1e-10
_LSMR_CONLIM = 1e8
_NULLSPACE_OPERATOR_RTOL = 1e-4


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required three-view toy artifact does not exist: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _require(arrays: Mapping[str, np.ndarray], key: str, source: str) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"{source} is missing required array {key!r}.")
    return np.asarray(arrays[key])


def _require_shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")


def _integer_distribution(values: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return {}
    unique, counts = np.unique(selected.astype(np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts)}


def _preflight_track(
    label: str,
    observations: Mapping[str, np.ndarray],
    solved: Mapping[str, np.ndarray],
    ground_truth: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Validate that one source track is the clean three-view staged toy."""

    obs_points = _require(observations, "points_world", f"{label} observations")
    solved_points = _require(solved, "points_world", f"{label} staged latent")
    gt_points = _require(ground_truth, "points_world", f"{label} ground-truth latent")
    if obs_points.ndim != 2 or obs_points.shape[1] != 3:
        raise ValueError(f"{label} points_world must have shape (N, 3), got {obs_points.shape}.")
    num_points = int(obs_points.shape[0])
    _require_shape(solved_points, obs_points.shape, f"{label} staged points_world")
    _require_shape(gt_points, obs_points.shape, f"{label} ground-truth points_world")
    if not np.all(np.isfinite(obs_points)):
        raise ValueError(f"{label} points_world contains non-finite values.")
    if not np.array_equal(solved_points, obs_points) or not np.array_equal(gt_points, obs_points):
        raise ValueError(f"{label} staged, observation, and ground-truth point arrays must match exactly.")

    view_ids = _require(observations, "view_ids", f"{label} observations").reshape(-1)
    if view_ids.shape[0] != 3:
        raise ValueError(
            f"{label} motion fill requires an existing three-view toy; found {view_ids.shape[0]} views."
        )
    num_views = 3
    solved_view_ids = _require(solved, "view_ids", f"{label} staged latent").reshape(-1)
    if not np.array_equal(solved_view_ids, view_ids):
        raise ValueError(f"{label} staged and observation view_ids must match exactly.")

    alphas = _require(solved, "alphas", f"{label} staged latent")
    true_alphas = _require(observations, "true_alphas", f"{label} observations")
    alpha_identifiable = _require(
        solved, "alpha_identifiable_mask", f"{label} staged latent"
    )
    _require_shape(alphas, (num_views,), f"{label} alphas")
    _require_shape(true_alphas, (num_views,), f"{label} true_alphas")
    _require_shape(alpha_identifiable, (num_views,), f"{label} alpha_identifiable_mask")
    if alpha_identifiable.dtype != np.bool_ or not np.all(alpha_identifiable):
        raise ValueError(f"{label} requires all three view alphas to be identifiable.")
    if not np.all(np.isfinite(alphas)) or not np.all(np.isfinite(true_alphas)):
        raise ValueError(f"{label} alpha arrays must be finite.")

    phi = _require(solved, "phi", f"{label} staged latent")
    phi_observable = _require(solved, "phi_observable", f"{label} staged latent")
    phi_gt = _require(observations, "phi_gt", f"{label} observations")
    latent_phi_gt = _require(ground_truth, "phi", f"{label} ground-truth latent")
    for value, name in (
        (phi, "phi"),
        (phi_observable, "phi_observable"),
        (phi_gt, "observation phi_gt"),
        (latent_phi_gt, "ground-truth phi"),
    ):
        _require_shape(value, (num_points, 3), f"{label} {name}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{label} {name} contains non-finite values.")
    if not np.array_equal(phi, phi_observable):
        raise ValueError(f"{label} staged phi must still equal phi_observable before motion fill.")
    if not np.array_equal(phi_gt, latent_phi_gt):
        raise ValueError(f"{label} observation and latent ground-truth fields must match exactly.")
    if not np.allclose(phi_gt, phi_gt[:1], rtol=0.0, atol=1e-7):
        raise ValueError(f"{label} ground-truth motion must be spatially constant for this experiment.")

    source_completion = _require(solved, "completion_mask", f"{label} staged latent")
    source_correction = _require(
        solved, "phi_nullspace_correction", f"{label} staged latent"
    )
    _require_shape(source_completion, (num_points,), f"{label} completion_mask")
    _require_shape(source_correction, (num_points, 3), f"{label} phi_nullspace_correction")
    if np.any(source_completion) or np.any(source_correction != 0):
        raise ValueError(f"{label} source latent is already completed; expected the staged baseline.")
    solver_method = str(
        _require(solved, "solver_method", f"{label} staged latent").reshape(()).item()
    )
    if solver_method != "staged_overlap_observable":
        raise ValueError(
            f"{label} requires solver_method='staged_overlap_observable', got {solver_method!r}."
        )
    _require(solved, "solver_version", f"{label} staged latent").reshape(()).item()

    masks: dict[str, np.ndarray] = {}
    for name in (
        "anchor_mask",
        "partial_mask",
        "unobserved_mask",
        "rejected_mask",
        "alpha_unresolved_mask",
        "no_usable_observation_mask",
    ):
        mask = _require(solved, name, f"{label} staged latent")
        _require_shape(mask, (num_points,), f"{label} {name}")
        if mask.dtype != np.bool_:
            raise ValueError(f"{label} {name} must have boolean dtype, got {mask.dtype}.")
        masks[name] = mask
    for name in ("rejected_mask", "alpha_unresolved_mask", "no_usable_observation_mask"):
        if np.any(masks[name]):
            raise ValueError(
                f"{label} has {int(np.count_nonzero(masks[name]))} unexpected {name} points."
            )
    partition_count = (
        masks["anchor_mask"].astype(np.int8)
        + masks["partial_mask"].astype(np.int8)
        + masks["unobserved_mask"].astype(np.int8)
    )
    if np.any(partition_count != 1):
        raise ValueError(
            f"{label} anchor, partial, and unobserved masks must be mutually exclusive and exhaustive."
        )

    count_names = (
        "obs_count_per_point",
        "obs_sample_count_per_point",
        "point_distinct_view_count",
        "point_distinct_valid_view_count",
        "point_usable_observation_row_count",
    )
    counts: dict[str, np.ndarray] = {}
    for name in count_names:
        value = _require(solved, name, f"{label} staged latent")
        _require_shape(value, (num_points,), f"{label} {name}")
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{label} {name} must have integer dtype, got {value.dtype}.")
        counts[name] = value.astype(np.int64, copy=False)
    reference_count = counts["obs_count_per_point"]
    for name in count_names[1:]:
        if not np.array_equal(counts[name], reference_count):
            raise ValueError(
                f"{label} toy preflight requires obs/sample/distinct/valid/usable counts to agree; "
                f"{name} differs from obs_count_per_point."
            )
    obs_count = _require(observations, "obs_count_per_point", f"{label} observations")
    obs_sample_count = _require(
        observations, "obs_sample_count_per_point", f"{label} observations"
    )
    if not np.array_equal(obs_count, reference_count) or not np.array_equal(
        obs_sample_count, reference_count
    ):
        raise ValueError(f"{label} observation and staged point counts must match exactly.")

    obs_point_index = _require(solved, "obs_point_index", f"{label} staged latent")
    obs_view_index = _require(solved, "obs_view_index", f"{label} staged latent")
    obs_y = _require(solved, "obs_y", f"{label} staged latent")
    obs_jacobian = _require(solved, "obs_J", f"{label} staged latent")
    obs_weight = _require(solved, "obs_effective_weight", f"{label} staged latent")
    num_observations = int(obs_point_index.shape[0])
    if not np.issubdtype(obs_point_index.dtype, np.integer):
        raise ValueError(f"{label} obs_point_index must have integer dtype.")
    if not np.issubdtype(obs_view_index.dtype, np.integer):
        raise ValueError(f"{label} obs_view_index must have integer dtype.")
    for value, name in (
        (obs_view_index, "obs_view_index"),
        (obs_weight, "obs_effective_weight"),
    ):
        _require_shape(value, (num_observations,), f"{label} {name}")
    _require_shape(obs_y, (num_observations, 2), f"{label} obs_y")
    _require_shape(obs_jacobian, (num_observations, 2, 3), f"{label} obs_J")
    if np.any(obs_point_index < 0) or np.any(obs_point_index >= num_points):
        raise ValueError(f"{label} obs_point_index contains an out-of-range index.")
    if np.any(obs_view_index < 0) or np.any(obs_view_index >= num_views):
        raise ValueError(f"{label} obs_view_index contains an out-of-range index.")
    if np.any(~np.isfinite(obs_weight)) or np.any(obs_weight <= 0.0):
        raise ValueError(f"{label} clean toy observations must all have positive finite weight.")
    if not np.all(np.isfinite(obs_y)) or not np.all(np.isfinite(obs_jacobian)):
        raise ValueError(f"{label} observation values and Jacobians must be finite.")
    for name, solved_value in (
        ("obs_point_index", obs_point_index),
        ("obs_view_index", obs_view_index),
        ("obs_y", obs_y),
        ("obs_J", obs_jacobian),
    ):
        observation_value = _require(observations, name, f"{label} observations")
        if not np.array_equal(observation_value, solved_value):
            raise ValueError(f"{label} staged and source observation arrays differ for {name}.")
    derived_count = np.bincount(obs_point_index, minlength=num_points)
    if not np.array_equal(derived_count, reference_count):
        raise ValueError(f"{label} obs_point_index does not reproduce obs_count_per_point.")
    point_view_key = obs_point_index.astype(np.int64) * num_views + obs_view_index.astype(np.int64)
    if np.unique(point_view_key).size != num_observations:
        raise ValueError(f"{label} must contain exactly one observation row per point-view pair.")

    predicted_gt = true_alphas[obs_view_index, None] * np.einsum(
        "nij,nj->ni", obs_jacobian, phi_gt[obs_point_index]
    )
    if not np.allclose(predicted_gt, obs_y, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{label} observation rows are not the expected noise-free ground truth.")

    rank = _require(solved, "point_observable_rank", f"{label} staged latent")
    nullity = _require(solved, "point_nullity", f"{label} staged latent")
    nullspace_basis = _require(
        solved, "point_nullspace_basis", f"{label} staged latent"
    )
    _require_shape(rank, (num_points,), f"{label} point_observable_rank")
    _require_shape(nullity, (num_points,), f"{label} point_nullity")
    _require_shape(
        nullspace_basis,
        (num_points, 3, 3),
        f"{label} point_nullspace_basis",
    )
    if not np.issubdtype(rank.dtype, np.integer) or np.any(rank < 0) or np.any(
        rank > 3
    ):
        raise ValueError(
            f"{label} point_observable_rank must contain integer values in [0, 3]."
        )
    if not np.issubdtype(nullity.dtype, np.integer) or np.any(nullity < 0) or np.any(
        nullity > 3
    ):
        raise ValueError(f"{label} point_nullity must contain integer values in [0, 3].")
    if np.any(rank.astype(np.int8) + nullity.astype(np.int8) != 3):
        raise ValueError(f"{label} point_observable_rank + point_nullity must equal 3.")
    if not np.all(np.isfinite(nullspace_basis)):
        raise ValueError(f"{label} point_nullspace_basis contains non-finite values.")
    validate_motion_fill_inputs(
        phi_observable,
        nullspace_basis,
        nullity,
        masks["anchor_mask"],
        masks["partial_mask"],
        masks["unobserved_mask"],
    )
    weighted_operator = (
        np.sqrt(obs_weight)[:, None, None]
        * alphas[obs_view_index, None, None]
        * obs_jacobian
    )
    operator_energy = np.bincount(
        obs_point_index,
        weights=np.sum(np.abs(weighted_operator) ** 2, axis=(1, 2)),
        minlength=num_points,
    )
    nullspace_energy = np.zeros((num_points,), dtype=np.float64)
    observed_nullity = nullity[obs_point_index]
    for dimension in (1, 2, 3):
        selected = observed_nullity == dimension
        if not np.any(selected):
            continue
        projected_nullspace = np.einsum(
            "nij,njk->nik",
            weighted_operator[selected],
            nullspace_basis[obs_point_index[selected], :, :dimension],
        )
        np.add.at(
            nullspace_energy,
            obs_point_index[selected],
            np.sum(np.abs(projected_nullspace) ** 2, axis=(1, 2)),
        )
    checked_nullspace = (nullity > 0) & (operator_energy > 0.0)
    nullspace_operator_relative_error = np.zeros((num_points,), dtype=np.float64)
    nullspace_operator_relative_error[checked_nullspace] = np.sqrt(
        nullspace_energy[checked_nullspace] / operator_energy[checked_nullspace]
    )
    max_nullspace_operator_relative_error = float(
        np.max(nullspace_operator_relative_error, initial=0.0)
    )
    if max_nullspace_operator_relative_error > _NULLSPACE_OPERATOR_RTOL:
        point_index = int(np.argmax(nullspace_operator_relative_error))
        raise ValueError(
            f"{label} stored nullspace fails A_i N_i = 0 at point {point_index}: "
            f"relative error {nullspace_operator_relative_error[point_index]:.6g}."
        )
    point_status = _require(solved, "point_solution_status", f"{label} staged latent")
    _require_shape(point_status, (num_points,), f"{label} point_solution_status")
    colors = _require(observations, "colors", f"{label} observations")
    _require_shape(colors, (num_points, 3), f"{label} colors")
    class_summary: dict[str, Any] = {}
    for class_name in ("anchor", "partial", "unobserved"):
        mask = masks[f"{class_name}_mask"]
        class_summary[class_name] = {
            "point_count": int(np.count_nonzero(mask)),
            "observable_rank_distribution": _integer_distribution(rank, mask),
            "nullity_distribution": _integer_distribution(nullity, mask),
            "valid_view_count_distribution": _integer_distribution(
                counts["point_distinct_valid_view_count"], mask
            ),
        }
    return {
        "num_points": num_points,
        "num_views": num_views,
        "num_observations": num_observations,
        "nullspace_operator_validation": {
            "relative_tolerance": _NULLSPACE_OPERATOR_RTOL,
            "max_relative_error": max_nullspace_operator_relative_error,
        },
        "classes": class_summary,
    }


def _validate_k_values(k_values: list[int] | tuple[int, ...], num_points: int) -> tuple[int, ...]:
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in k_values
    ):
        raise ValueError(f"K values must be integers, got {tuple(k_values)}.")
    values = tuple(int(value) for value in k_values)
    if not values:
        raise ValueError("At least one K value is required.")
    if any(value <= 0 for value in values):
        raise ValueError(f"K values must be positive, got {values}.")
    if len(set(values)) != len(values):
        raise ValueError(f"K values must be unique, got {values}.")
    if max(values) >= num_points:
        raise ValueError(f"Every K must be smaller than the point count ({num_points}), got {values}.")
    return values


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": int(array.size),
            "finite_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def _graph_coherence(phi: np.ndarray, graph: KnnGraph) -> float | None:
    if graph.edge_index.shape[0] == 0:
        return None
    delta = phi[graph.edge_index[:, 0]] - phi[graph.edge_index[:, 1]]
    numerator = np.sum(graph.edge_weight * np.sum(np.abs(delta) ** 2, axis=1))
    denominator = float(np.sum(graph.edge_weight))
    return float(numerator / denominator)


def _observation_drift(
    solved: Mapping[str, np.ndarray],
    correction: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_points = correction.shape[0]
    point_index = np.asarray(solved["obs_point_index"], dtype=np.int64)
    view_index = np.asarray(solved["obs_view_index"], dtype=np.int64)
    jacobian = np.asarray(solved["obs_J"], dtype=np.float64)
    measurement = np.asarray(solved["obs_y"], dtype=np.complex128)
    weight = np.asarray(solved["obs_effective_weight"], dtype=np.float64)
    alphas = np.asarray(solved["alphas"], dtype=np.complex128)
    sqrt_weight = np.sqrt(weight)
    projected = np.einsum("nij,nj->ni", jacobian, correction[point_index])
    weighted_drift = sqrt_weight[:, None] * alphas[view_index, None] * projected
    weighted_measurement = sqrt_weight[:, None] * measurement
    drift_squared = np.bincount(
        point_index,
        weights=np.sum(np.abs(weighted_drift) ** 2, axis=1),
        minlength=num_points,
    )
    measurement_squared = np.bincount(
        point_index,
        weights=np.sum(np.abs(weighted_measurement) ** 2, axis=1),
        minlength=num_points,
    )
    absolute = np.sqrt(drift_squared)
    relative = absolute / np.maximum(np.sqrt(measurement_squared), float(epsilon))
    observed = np.bincount(point_index, minlength=num_points) > 0
    absolute[~observed] = np.nan
    relative[~observed] = np.nan
    return absolute, relative


def _solver_metadata(metadata: Any) -> dict[str, Any]:
    def finite_or_none(value: float) -> float | None:
        value = float(value)
        return value if np.isfinite(value) else None

    return {
        "performed": bool(metadata.performed),
        "converged": bool(metadata.converged),
        "stop_code": int(metadata.stop_code),
        "iterations": int(metadata.iterations),
        "residual_norm": finite_or_none(metadata.residual_norm),
        "normal_residual_norm": finite_or_none(metadata.normal_residual_norm),
        "matrix_norm": finite_or_none(metadata.matrix_norm),
        "condition_estimate": finite_or_none(metadata.condition_estimate),
        "solution_norm": finite_or_none(metadata.solution_norm),
    }


def _class_diagnostics(
    mask: np.ndarray,
    solved: Mapping[str, np.ndarray],
    phi_gt: np.ndarray,
    result: MotionFillResult,
    observation_drift_absolute: np.ndarray,
    observation_drift_relative: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    observable_error = np.linalg.norm(result.phi_observable - phi_gt, axis=1)
    filled_error = np.linalg.norm(result.phi - phi_gt, axis=1)
    gt_norm = np.linalg.norm(phi_gt, axis=1)
    missing_gt = phi_gt - result.phi_observable
    correction_error = np.linalg.norm(result.phi_nullspace_correction - missing_gt, axis=1)
    denominator = np.maximum(gt_norm, float(epsilon))
    completion_count = int(np.count_nonzero(mask & result.completion_mask))
    point_count = int(np.count_nonzero(mask))
    connected_count = int(np.count_nonzero(mask & result.completion_connected_to_anchor))
    hop_mask = mask & (result.connectivity.hop_distance >= 0)
    return {
        "point_count": point_count,
        "observable_rank_distribution": _integer_distribution(
            solved["point_observable_rank"], mask
        ),
        "nullity_distribution": _integer_distribution(solved["point_nullity"], mask),
        "valid_view_count_distribution": _integer_distribution(
            solved["point_distinct_valid_view_count"], mask
        ),
        "complex_mode_relative_error_observable": _stats(
            observable_error[mask] / denominator[mask]
        ),
        "complex_mode_relative_error_filled": _stats(filled_error[mask] / denominator[mask]),
        "trajectory_rmse_observable": _stats(observable_error[mask] / np.sqrt(2.0)),
        "trajectory_rmse_filled": _stats(filled_error[mask] / np.sqrt(2.0)),
        "nullspace_correction_absolute_error": _stats(correction_error[mask]),
        "nullspace_correction_relative_error": _stats(correction_error[mask] / denominator[mask]),
        "observation_drift_absolute": _stats(observation_drift_absolute[mask]),
        "observation_drift_relative": _stats(observation_drift_relative[mask]),
        "completion": {
            "numerator": completion_count,
            "denominator": point_count,
            "fraction": float(completion_count / point_count) if point_count else None,
        },
        "connected_to_anchor": {
            "numerator": connected_count,
            "denominator": point_count,
            "fraction": float(connected_count / point_count) if point_count else None,
        },
        "anchor_hop_distance": _stats(result.connectivity.hop_distance[hop_mask]),
        "anchor_free_point_count": int(np.count_nonzero(mask & ~result.completion_connected_to_anchor)),
    }


def _track_diagnostics(
    solved: Mapping[str, np.ndarray],
    observations: Mapping[str, np.ndarray],
    phi_gt: np.ndarray,
    graph: KnnGraph,
    result: MotionFillResult,
    epsilon: float,
) -> dict[str, Any]:
    anchor = np.asarray(solved["anchor_mask"], dtype=bool)
    partial = np.asarray(solved["partial_mask"], dtype=bool)
    unobserved = np.asarray(solved["unobserved_mask"], dtype=bool)
    valid_view_count = np.asarray(solved["point_distinct_valid_view_count"], dtype=np.int32)
    observation_absolute, observation_relative = _observation_drift(
        solved, result.phi_nullspace_correction, epsilon
    )
    classes = {
        "anchor": _class_diagnostics(
            anchor,
            solved,
            phi_gt,
            result,
            observation_absolute,
            observation_relative,
            epsilon,
        ),
        "partial": _class_diagnostics(
            partial,
            solved,
            phi_gt,
            result,
            observation_absolute,
            observation_relative,
            epsilon,
        ),
        "unobserved": _class_diagnostics(
            unobserved,
            solved,
            phi_gt,
            result,
            observation_absolute,
            observation_relative,
            epsilon,
        ),
        "anchor_valid_view_count_2": _class_diagnostics(
            anchor & (valid_view_count == 2),
            solved,
            phi_gt,
            result,
            observation_absolute,
            observation_relative,
            epsilon,
        ),
        "anchor_valid_view_count_3": _class_diagnostics(
            anchor & (valid_view_count == 3),
            solved,
            phi_gt,
            result,
            observation_absolute,
            observation_relative,
            epsilon,
        ),
    }

    anchor_drift = np.linalg.norm(result.phi[anchor] - result.phi_observable[anchor], axis=1)
    anchor_gt_error = np.linalg.norm(result.phi[anchor] - phi_gt[anchor], axis=1)
    alphas = np.asarray(solved["alphas"], dtype=np.complex128)
    true_alphas = np.asarray(observations["true_alphas"], dtype=np.complex128)
    component_has_anchor = result.connectivity.component_has_anchor
    return {
        "classes": classes,
        "anchor_drift": _stats(anchor_drift),
        "anchor_to_ground_truth_error": _stats(anchor_gt_error),
        "partial_observation_drift_absolute": _stats(observation_absolute[partial]),
        "partial_observation_drift_relative": _stats(observation_relative[partial]),
        "alpha_error": {
            "complex_absolute": _stats(np.abs(alphas - true_alphas)),
            "phase_degrees": _stats(
                np.degrees(np.abs(np.angle(alphas * np.conj(true_alphas))))
            ),
            "gain_absolute": _stats(np.abs(np.abs(alphas) - np.abs(true_alphas))),
        },
        "graph_coherence": {
            "ground_truth": _graph_coherence(phi_gt, graph),
            "observable": _graph_coherence(result.phi_observable, graph),
            "filled": _graph_coherence(result.phi, graph),
        },
        "connectivity": {
            "component_count": int(component_has_anchor.shape[0]),
            "anchor_connected_component_count": int(np.count_nonzero(component_has_anchor)),
            "anchor_free_component_count": int(np.count_nonzero(~component_has_anchor)),
            "anchor_connected_point_count": int(
                np.count_nonzero(result.completion_connected_to_anchor)
            ),
            "anchor_free_point_count": int(
                np.count_nonzero(~result.completion_connected_to_anchor)
            ),
            "component_anchor_count_distribution": _integer_distribution(
                result.connectivity.component_anchor_count,
                np.ones_like(result.connectivity.component_anchor_count, dtype=bool),
            ),
            "anchor_hop_distance": _stats(
                result.connectivity.hop_distance[result.connectivity.hop_distance >= 0]
            ),
            "anchor_free_hop_distance_sentinel": -1,
        },
        "sparse_system": {
            "row_count": int(result.system_row_count),
            "column_count": int(result.system_column_count),
            "active_edge_count": int(result.active_edge_count),
            "real": _solver_metadata(result.real_solver),
            "imaginary": _solver_metadata(result.imag_solver),
        },
    }


def _write_npz(path: Path, **arrays: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def _tint_colors(colors: np.ndarray, tint: np.ndarray, strength: float = 0.4) -> np.ndarray:
    base = np.asarray(colors, dtype=np.float32)
    output = (1.0 - strength) * base + strength * np.asarray(tint, dtype=np.float32)[None, :]
    return np.rint(np.clip(output, 0.0, 255.0)).astype(np.uint8)


def _point_status(solved: Mapping[str, np.ndarray], result: MotionFillResult) -> np.ndarray:
    status = np.asarray(solved["point_solution_status"], dtype=np.int8).copy()
    partial = np.asarray(solved["partial_mask"], dtype=bool)
    unobserved = np.asarray(solved["unobserved_mask"], dtype=bool)
    status[partial & result.completion_mask] = POINT_STATUS_COMPLETED_OBSERVED
    status[unobserved & result.completion_mask] = POINT_STATUS_COMPLETED_UNOBSERVED
    return status


def _write_track_outputs(
    output_dir: Path,
    track_key: str,
    observations: Mapping[str, np.ndarray],
    solved: Mapping[str, np.ndarray],
    ground_truth: Mapping[str, np.ndarray],
    graph: KnnGraph,
    result: MotionFillResult,
    source_observations: str,
    source_staged_latent: str,
) -> tuple[Path, Path]:
    """Write one lean completed latent and its GT/observable/filled overlay."""

    points = np.asarray(solved["points_world"], dtype=np.float32)
    phi_gt = np.asarray(ground_truth["phi"], dtype=np.complex64)
    colors = np.asarray(observations["colors"], dtype=np.uint8)
    obs_count = np.asarray(solved["obs_count_per_point"], dtype=np.int32)
    freq_hz = np.asarray(solved["freq_hz"], dtype=np.float32)
    mode_index = np.asarray(solved["mode_index"], dtype=np.int32)
    filled_path = _write_npz(
        output_dir / f"{track_key}_filled.npz",
        points_world=points,
        phi=np.asarray(result.phi, dtype=np.complex64),
        phi_observable=np.asarray(result.phi_observable, dtype=np.complex64),
        phi_nullspace_correction=np.asarray(result.phi_nullspace_correction, dtype=np.complex64),
        phi_gt=phi_gt,
        point_nullspace_basis=np.asarray(solved["point_nullspace_basis"], dtype=np.complex64),
        point_nullity=np.asarray(solved["point_nullity"], dtype=np.int8),
        anchor_mask=np.asarray(solved["anchor_mask"], dtype=bool),
        partial_mask=np.asarray(solved["partial_mask"], dtype=bool),
        unobserved_mask=np.asarray(solved["unobserved_mask"], dtype=bool),
        rejected_mask=np.asarray(solved["rejected_mask"], dtype=bool),
        alpha_unresolved_mask=np.asarray(solved["alpha_unresolved_mask"], dtype=bool),
        no_usable_observation_mask=np.asarray(solved["no_usable_observation_mask"], dtype=bool),
        completion_mask=np.asarray(result.completion_mask, dtype=bool),
        completion_connected_to_anchor=np.asarray(
            result.completion_connected_to_anchor, dtype=bool
        ),
        point_solution_status=_point_status(solved, result),
        point_solution_status_names=np.asarray(POINT_STATUS_NAMES),
        point_observable_rank=np.asarray(solved["point_observable_rank"], dtype=np.int8),
        point_distinct_valid_view_count=np.asarray(
            solved["point_distinct_valid_view_count"], dtype=np.int32
        ),
        point_connected_component_index=np.asarray(graph.component_index, dtype=np.int32),
        point_anchor_hop_distance=np.asarray(result.connectivity.hop_distance, dtype=np.int32),
        component_has_anchor=np.asarray(result.connectivity.component_has_anchor, dtype=bool),
        component_anchor_count=np.asarray(result.connectivity.component_anchor_count, dtype=np.int32),
        coefficient_values=np.asarray(result.coefficient_values, dtype=np.complex64),
        coefficient_offsets=np.asarray(result.coefficient_offsets, dtype=np.int64),
        alphas=np.asarray(solved["alphas"], dtype=np.complex64),
        alpha_identifiable_mask=np.asarray(solved["alpha_identifiable_mask"], dtype=bool),
        true_alphas=np.asarray(observations["true_alphas"], dtype=np.complex64),
        colors=colors,
        freq_hz=freq_hz,
        mode_index=mode_index,
        obs_count_per_point=obs_count,
        obs_sample_count_per_point=np.asarray(
            solved["obs_sample_count_per_point"], dtype=np.int32
        ),
        graph_k=np.array(graph.k, dtype=np.int32),
        graph_max_distance=np.array(graph.max_distance, dtype=np.float64),
        graph_epsilon=np.array(graph.epsilon, dtype=np.float64),
        motion_fill_method=np.array("joint_knn_nullspace_lsmr"),
        motion_fill_lsmr_atol=np.array(_LSMR_ATOL, dtype=np.float64),
        motion_fill_lsmr_btol=np.array(_LSMR_BTOL, dtype=np.float64),
        motion_fill_lsmr_conlim=np.array(_LSMR_CONLIM, dtype=np.float64),
        source_solver_method=np.asarray(solved["solver_method"]),
        source_solver_version=np.asarray(solved["solver_version"]),
        source_staged_latent=np.array(source_staged_latent),
        source_observations=np.array(source_observations),
    )

    num_points = points.shape[0]
    overlay_path = _write_npz(
        output_dir / f"{track_key}_overlay.npz",
        points_world=np.concatenate([points, points, points], axis=0),
        phi=np.concatenate(
            [phi_gt, result.phi_observable.astype(np.complex64), result.phi.astype(np.complex64)],
            axis=0,
        ),
        colors=np.concatenate(
            [
                _tint_colors(colors, np.array([40, 220, 120], dtype=np.uint8)),
                _tint_colors(colors, np.array([230, 159, 0], dtype=np.uint8)),
                _tint_colors(colors, np.array([240, 80, 220], dtype=np.uint8)),
            ],
            axis=0,
        ),
        freq_hz=freq_hz,
        mode_index=mode_index,
        obs_count_per_point=np.tile(obs_count, 3),
        obs_sample_count_per_point=np.tile(obs_count, 3),
        point_distinct_valid_view_count=np.tile(
            np.asarray(solved["point_distinct_valid_view_count"], dtype=np.int32), 3
        ),
        point_group=np.repeat(np.arange(3, dtype=np.int32), num_points),
        point_group_names=np.array(["GT", "Observable", f"Filled K={graph.k}"]),
    )
    return filled_path, overlay_path


def _write_graph(path: Path, candidates: KnnCandidateSet, graph: KnnGraph) -> Path:
    return _write_npz(
        path,
        candidate_neighbor_indices=np.asarray(
            candidates.neighbor_indices[:, : graph.k], dtype=np.int64
        ),
        candidate_neighbor_distances=np.asarray(
            candidates.neighbor_distances[:, : graph.k], dtype=np.float64
        ),
        edge_index=np.asarray(graph.edge_index, dtype=np.int64),
        edge_distance=np.asarray(graph.edge_distance, dtype=np.float64),
        edge_weight=np.asarray(graph.edge_weight, dtype=np.float64),
        degree=np.asarray(graph.degree, dtype=np.int32),
        component_index=np.asarray(graph.component_index, dtype=np.int32),
        component_sizes=np.asarray(graph.component_sizes, dtype=np.int32),
        isolated_mask=np.asarray(graph.isolated_mask, dtype=bool),
        k=np.array(graph.k, dtype=np.int32),
        max_distance=np.array(graph.max_distance, dtype=np.float64),
        epsilon=np.array(graph.epsilon, dtype=np.float64),
        candidate_directed_count=np.array(graph.candidate_directed_count, dtype=np.int64),
        retained_directed_count=np.array(graph.retained_directed_count, dtype=np.int64),
        pruned_directed_count=np.array(graph.pruned_directed_count, dtype=np.int64),
        unique_undirected_edge_count=np.array(
            graph.unique_undirected_edge_count, dtype=np.int64
        ),
    )


def _relative_path(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def _write_manifest(
    path: Path,
    k: int,
    max_distance: float,
    epsilon: float,
    modes: list[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "parameters": {
            "synthetic_test": "visibility_sphere",
            "experiment": "toy_knn_motion_fill",
            "motion_fill_method": "joint_knn_nullspace_lsmr",
            "knn_k": int(k),
            "knn_max_distance": float(max_distance),
            "knn_epsilon": float(epsilon),
            "lsmr_atol": _LSMR_ATOL,
            "lsmr_btol": _LSMR_BTOL,
            "lsmr_conlim": _LSMR_CONLIM,
        },
        "modes": modes,
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _write_sweep_summary(
    path: Path,
    k_values: tuple[int, ...],
    max_distance: float,
    epsilon: float,
    tracks: list[dict[str, Any]],
    diagnostics_by_k: Mapping[int, dict[str, Any]],
    manifest_paths_by_k: Mapping[int, Path],
    filled_fields_by_track: Mapping[str, Mapping[int, np.ndarray]],
) -> Path:
    track_summaries: dict[str, Any] = {}
    for track in tracks:
        track_key = str(track["key"])
        solved = track["solved_arrays"]
        phi_gt = np.asarray(track["ground_truth_arrays"]["phi"], dtype=np.complex128)
        denominator = np.maximum(np.linalg.norm(phi_gt, axis=1), float(epsilon))
        class_masks = {
            class_name: np.asarray(solved[f"{class_name}_mask"], dtype=bool)
            for class_name in ("anchor", "partial", "unobserved")
        }
        by_k: dict[str, Any] = {}
        for k in k_values:
            track_diagnostics = diagnostics_by_k[k]["tracks"][track_key]
            class_diagnostics = track_diagnostics["classes"]
            by_k[str(k)] = {
                "diagnostics_path": f"k{k}/diagnostics.json",
                "manifest_path": _relative_path(manifest_paths_by_k[k], path.parent),
                "anchor_drift": track_diagnostics["anchor_drift"],
                "graph_coherence_filled": track_diagnostics["graph_coherence"]["filled"],
                "classes": {
                    class_name: {
                        "completion": class_diagnostics[class_name]["completion"],
                        "connected_to_anchor": class_diagnostics[class_name][
                            "connected_to_anchor"
                        ],
                        "complex_mode_relative_error_filled": class_diagnostics[class_name][
                            "complex_mode_relative_error_filled"
                        ],
                        "trajectory_rmse_filled": class_diagnostics[class_name][
                            "trajectory_rmse_filled"
                        ],
                    }
                    for class_name in class_masks
                },
            }

        pairwise: dict[str, Any] = {}
        for first_index, first_k in enumerate(k_values):
            first_phi = np.asarray(
                filled_fields_by_track[track_key][first_k], dtype=np.complex128
            )
            for second_k in k_values[first_index + 1 :]:
                second_phi = np.asarray(
                    filled_fields_by_track[track_key][second_k], dtype=np.complex128
                )
                difference = np.linalg.norm(first_phi - second_phi, axis=1)
                pairwise[f"k{first_k}_vs_k{second_k}"] = {
                    class_name: {
                        "complex_mode_relative_difference": _stats(
                            difference[mask] / denominator[mask]
                        ),
                        "trajectory_rmse_difference": _stats(
                            difference[mask] / np.sqrt(2.0)
                        ),
                    }
                    for class_name, mask in class_masks.items()
                }
        track_summaries[track_key] = {
            "track_label": track["label"],
            "by_k": by_k,
            "pairwise_filled_field_difference": pairwise,
        }

    payload = {
        "version": 1,
        "experiment": "toy_knn_motion_fill_k_sensitivity",
        "k_values": [int(k) for k in k_values],
        "primary_k": 8,
        "primary_k_in_sweep": 8 in k_values,
        "max_distance": float(max_distance),
        "epsilon": float(epsilon),
        "tracks": track_summaries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path


def run_experiment(
    toy_dir: Path,
    k_values: list[int] | tuple[int, ...] = (4, 8, 16),
    max_distance: float = 0.1,
    epsilon: float = 1e-8,
) -> list[Path]:
    toy_dir = Path(toy_dir).expanduser()
    if not toy_dir.is_dir():
        raise FileNotFoundError(f"Three-view toy directory does not exist: {toy_dir}")
    max_distance = float(max_distance)
    epsilon = float(epsilon)
    if not np.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("max_distance must be finite and positive.")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive.")

    tracks: list[dict[str, Any]] = []
    for spec in _TRACKS:
        observations_path = toy_dir / spec["observations"]
        solved_path = toy_dir / spec["solved"]
        ground_truth_path = toy_dir / spec["ground_truth"]
        observations = _load_npz(observations_path)
        solved = _load_npz(solved_path)
        ground_truth = _load_npz(ground_truth_path)
        preflight = _preflight_track(spec["label"], observations, solved, ground_truth)
        tracks.append(
            {
                **spec,
                "observations_path": observations_path,
                "solved_path": solved_path,
                "ground_truth_path": ground_truth_path,
                "observations_arrays": observations,
                "solved_arrays": solved,
                "ground_truth_arrays": ground_truth,
                "preflight": preflight,
            }
        )

    reference_points = np.asarray(tracks[0]["solved_arrays"]["points_world"])
    for track in tracks[1:]:
        if not np.array_equal(track["solved_arrays"]["points_world"], reference_points):
            raise ValueError("Linear and tilted-ellipse tracks must share the same rest points.")
        if not np.array_equal(
            track["solved_arrays"]["obs_count_per_point"],
            tracks[0]["solved_arrays"]["obs_count_per_point"],
        ):
            raise ValueError(
                "Linear and tilted-ellipse tracks must share observation counts for one Viser manifest."
            )
    values = _validate_k_values(k_values, reference_points.shape[0])
    candidates = query_knn_candidates(reference_points, max(values))

    manifest_paths: list[Path] = []
    diagnostics_by_k: dict[int, dict[str, Any]] = {}
    manifest_paths_by_k: dict[int, Path] = {}
    filled_fields_by_track: dict[str, dict[int, np.ndarray]] = {
        str(track["key"]): {} for track in tracks
    }
    for k in values:
        graph = build_knn_graph(candidates, k, max_distance, epsilon)
        output_dir = toy_dir / "motion_fill" / f"k{k}"
        graph_path = _write_graph(output_dir / "graph.npz", candidates, graph)
        diagnostics: dict[str, Any] = {
            "version": 1,
            "experiment": "toy_knn_motion_fill",
            "solver_parameters": {
                "method": "joint_knn_nullspace_lsmr",
                "lsmr_atol": _LSMR_ATOL,
                "lsmr_btol": _LSMR_BTOL,
                "lsmr_conlim": _LSMR_CONLIM,
                "lsmr_maxiter": None,
            },
            "graph": {
                "k": int(k),
                "max_distance": float(max_distance),
                "epsilon": float(epsilon),
                "candidate_directed_count": int(graph.candidate_directed_count),
                "retained_directed_count": int(graph.retained_directed_count),
                "pruned_directed_count": int(graph.pruned_directed_count),
                "unique_undirected_edge_count": int(graph.unique_undirected_edge_count),
                "component_count": int(graph.component_sizes.shape[0]),
                "isolated_point_count": int(np.count_nonzero(graph.isolated_mask)),
                "degree": _stats(graph.degree),
                "edge_distance": _stats(graph.edge_distance),
                "edge_weight": _stats(graph.edge_weight),
                "nearest_neighbor_distance": _stats(candidates.neighbor_distances[:, 0]),
                "kth_neighbor_distance": _stats(candidates.neighbor_distances[:, k - 1]),
                "graph_path": _relative_path(graph_path, toy_dir),
            },
            "tracks": {},
        }
        manifest_path = (
            toy_dir / "manifests" / "motion_fill" / f"k{k}" / "modal_modes_manifest.json"
        )
        manifest_modes: list[dict[str, Any]] = []
        for track in tracks:
            solved = track["solved_arrays"]
            observations = track["observations_arrays"]
            ground_truth = track["ground_truth_arrays"]
            result = fill_nullspace_motion(
                graph,
                solved["phi_observable"],
                solved["point_nullspace_basis"],
                solved["point_nullity"],
                solved["anchor_mask"],
                solved["partial_mask"],
                solved["unobserved_mask"],
                lsmr_atol=_LSMR_ATOL,
                lsmr_btol=_LSMR_BTOL,
                lsmr_conlim=_LSMR_CONLIM,
            )
            filled_path, overlay_path = _write_track_outputs(
                output_dir,
                track["key"],
                observations,
                solved,
                ground_truth,
                graph,
                result,
                source_observations=track["observations"],
                source_staged_latent=track["solved"],
            )
            track_diagnostics = _track_diagnostics(
                solved,
                observations,
                np.asarray(ground_truth["phi"], dtype=np.complex128),
                graph,
                result,
                epsilon,
            )
            track_diagnostics.update(
                {
                    "preflight": track["preflight"],
                    "source_observations": track["observations"],
                    "source_staged_latent": track["solved"],
                    "source_ground_truth": track["ground_truth"],
                    "filled_latent": _relative_path(filled_path, toy_dir),
                    "overlay_latent": _relative_path(overlay_path, toy_dir),
                }
            )
            diagnostics["tracks"][track["key"]] = track_diagnostics
            filled_fields_by_track[track["key"]][k] = np.asarray(
                result.phi, dtype=np.complex128
            ).copy()
            manifest_modes.append(
                {
                    "mode_index": int(np.asarray(solved["mode_index"]).item()),
                    "freq_hz": float(np.asarray(solved["freq_hz"]).item()),
                    "label": f"GT vs observable vs KNN fill (K={k})",
                    "latent_path": _relative_path(overlay_path, manifest_path.parent),
                    "track_label": track["label"],
                }
            )
        diagnostics_path = output_dir / "diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8"
        )
        _write_manifest(manifest_path, k, max_distance, epsilon, manifest_modes)
        manifest_paths.append(manifest_path)
        diagnostics_by_k[k] = diagnostics
        manifest_paths_by_k[k] = manifest_path
        print(f"K={k}: graph {graph_path}")
        print(f"K={k}: diagnostics {diagnostics_path}")
        print(f"K={k}: Viser manifest {manifest_path}")
    summary_path = _write_sweep_summary(
        toy_dir / "motion_fill" / "summary.json",
        values,
        max_distance,
        epsilon,
        tracks,
        diagnostics_by_k,
        manifest_paths_by_k,
        filled_fields_by_track,
    )
    print(f"K sensitivity summary: {summary_path}")
    return manifest_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill staged three-view sphere-toy nullspaces with a spatial KNN prior."
    )
    parser.add_argument(
        "--toy-dir",
        type=Path,
        required=True,
        help="Existing output directory produced by the three-view sphere toy.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="KNN neighborhood sizes to evaluate (default: 4 8 16).",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.1,
        help="Fixed maximum spatial edge distance shared by every K (default: 0.1).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Positive denominator epsilon for inverse-distance graph weights (default: 1e-8).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_experiment(args.toy_dir, args.k_values, args.max_distance, args.epsilon)


if __name__ == "__main__":
    main()
