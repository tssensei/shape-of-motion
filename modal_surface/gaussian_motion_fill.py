"""Production motion-fill post-processing for staged Gaussian modal fields."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from modal_surface.io import save_npz_compressed_atomic
from modal_surface.motion_fill import (
    KnnCandidateSet,
    KnnGraph,
    MotionFillResult,
    fill_nullspace_motion,
)
from modal_surface.optimization_staged import (
    POINT_STATUS_COMPLETED_OBSERVED,
    POINT_STATUS_COMPLETED_UNOBSERVED,
    AlphaSyncResult,
    ObservableSolveResult,
    PreparedObservations,
    StagedSolveResult,
    compute_prediction_and_residuals,
)


MOTION_FILL_ROLE_FIXED_ANCHOR = 0
MOTION_FILL_ROLE_CONSTRAINED_VARIABLE = 1
MOTION_FILL_ROLE_FREE_VARIABLE = 2
MOTION_FILL_ROLE_EXCLUDED = 3
MOTION_FILL_ROLE_NAMES = (
    "fixed_anchor",
    "constrained_variable",
    "free_variable",
    "excluded",
)

MOTION_FILL_EXCLUDED_NONE = 0
MOTION_FILL_EXCLUDED_REJECTED = 1
MOTION_FILL_EXCLUDED_ALPHA_UNRESOLVED = 2
MOTION_FILL_EXCLUDED_NO_USABLE_OBSERVATION = 3
MOTION_FILL_EXCLUDED_FULL_RANK_NONANCHOR = 4
MOTION_FILL_EXCLUDED_REASON_NAMES = (
    "none",
    "rejected",
    "alpha_unresolved",
    "no_usable_observation",
    "full_rank_nonanchor",
)

MOTION_FILL_METHOD = "joint_knn_nullspace_lsmr"
MOTION_FILL_VERSION = 1
MOTION_FILL_NUMERICAL_RANK_POLICY = "svd_max_shape_float64_epsilon"
MOTION_FILL_EPSILON = 1e-8
MOTION_FILL_LSMR_ATOL = 1e-10
MOTION_FILL_LSMR_BTOL = 1e-10
MOTION_FILL_LSMR_CONLIM = 1e8
MOTION_FILL_NULLSPACE_RTOL = 1e-4
MOTION_FILL_OBSERVATION_DRIFT_RTOL = 1e-4


@dataclass(frozen=True)
class GaussianMotionFillRoles:
    role: np.ndarray
    excluded_reason: np.ndarray
    fixed_anchor_mask: np.ndarray
    constrained_variable_mask: np.ndarray
    free_variable_mask: np.ndarray
    excluded_mask: np.ndarray

    @property
    def active_mask(self) -> np.ndarray:
        return ~self.excluded_mask


@dataclass(frozen=True)
class GaussianObservationOperator:
    point_index: np.ndarray
    view_index: np.ndarray
    jacobian: np.ndarray
    weight: np.ndarray
    alphas: np.ndarray
    valid_rows: np.ndarray
    weighted_operator: np.ndarray


@dataclass(frozen=True)
class GaussianMotionFillSubspaces:
    basis: np.ndarray
    nullity: np.ndarray
    valid_rows: np.ndarray


@dataclass(frozen=True)
class GaussianMotionFillResult:
    motion: MotionFillResult
    roles: GaussianMotionFillRoles
    numerical_nullity: np.ndarray
    staged_nullity_refined_mask: np.ndarray
    point_solution_status: np.ndarray
    obs_pred_y: np.ndarray
    obs_residual: np.ndarray
    obs_residual_valid_mask: np.ndarray
    point_residual: np.ndarray
    point_residual_valid_mask: np.ndarray
    diagnostics: dict[str, Any]


def _require_boolean_mask(
    value: np.ndarray,
    key: str,
    num_points: int,
) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != (num_points,):
        raise ValueError(f"{key} must have shape ({num_points},), got {mask.shape}.")
    if mask.dtype != np.bool_:
        raise ValueError(f"{key} must have boolean dtype, got {mask.dtype}.")
    return mask


def derive_gaussian_motion_fill_roles(
    observable: ObservableSolveResult,
    point_nullity: np.ndarray,
) -> GaussianMotionFillRoles:
    """Map staged solver states to the four production motion-fill roles."""

    phi = np.asarray(observable.phi_observable)
    if phi.ndim != 2 or phi.shape[1] != 3:
        raise ValueError(f"phi_observable must have shape (N,3), got {phi.shape}.")
    num_points = int(phi.shape[0])
    anchor = _require_boolean_mask(observable.anchor_mask, "anchor_mask", num_points)
    partial = _require_boolean_mask(observable.partial_mask, "partial_mask", num_points)
    unobserved = _require_boolean_mask(
        observable.unobserved_mask, "unobserved_mask", num_points
    )
    rejected = _require_boolean_mask(observable.rejected_mask, "rejected_mask", num_points)
    alpha_unresolved = _require_boolean_mask(
        observable.alpha_unresolved_mask, "alpha_unresolved_mask", num_points
    )
    no_usable = _require_boolean_mask(
        observable.no_usable_observation_mask,
        "no_usable_observation_mask",
        num_points,
    )

    staged_state_count = sum(
        mask.astype(np.int8)
        for mask in (
            anchor,
            partial,
            unobserved,
            rejected,
            alpha_unresolved,
            no_usable,
        )
    )
    if np.any(staged_state_count != 1):
        raise ValueError(
            "Staged anchor, partial, unobserved, rejected, alpha-unresolved, and "
            "no-usable masks must be mutually exclusive and exhaustive."
        )

    nullity_source = np.asarray(point_nullity)
    if nullity_source.shape != (num_points,):
        raise ValueError(
            f"point_nullity must have shape ({num_points},), got {nullity_source.shape}."
        )
    if not np.issubdtype(nullity_source.dtype, np.integer):
        raise ValueError(f"point_nullity must have integer dtype, got {nullity_source.dtype}.")
    if np.any(nullity_source < 0) or np.any(nullity_source > 3):
        raise ValueError("point_nullity values must lie in [0,3].")
    nullity = nullity_source.astype(np.int8, copy=False)

    constrained = partial & (nullity > 0)
    full_rank_nonanchor = partial & (nullity == 0)
    excluded = rejected | alpha_unresolved | no_usable | full_rank_nonanchor
    free = unobserved
    role_count = (
        anchor.astype(np.int8)
        + constrained.astype(np.int8)
        + free.astype(np.int8)
        + excluded.astype(np.int8)
    )
    if np.any(role_count != 1):
        raise RuntimeError("Derived motion-fill roles are not mutually exclusive and exhaustive.")

    role = np.full((num_points,), -1, dtype=np.int8)
    role[anchor] = MOTION_FILL_ROLE_FIXED_ANCHOR
    role[constrained] = MOTION_FILL_ROLE_CONSTRAINED_VARIABLE
    role[free] = MOTION_FILL_ROLE_FREE_VARIABLE
    role[excluded] = MOTION_FILL_ROLE_EXCLUDED
    excluded_reason = np.full(
        (num_points,), MOTION_FILL_EXCLUDED_NONE, dtype=np.int8
    )
    excluded_reason[rejected] = MOTION_FILL_EXCLUDED_REJECTED
    excluded_reason[alpha_unresolved] = MOTION_FILL_EXCLUDED_ALPHA_UNRESOLVED
    excluded_reason[no_usable] = MOTION_FILL_EXCLUDED_NO_USABLE_OBSERVATION
    excluded_reason[full_rank_nonanchor] = MOTION_FILL_EXCLUDED_FULL_RANK_NONANCHOR
    return GaussianMotionFillRoles(
        role=role,
        excluded_reason=excluded_reason,
        fixed_anchor_mask=anchor,
        constrained_variable_mask=constrained,
        free_variable_mask=free,
        excluded_mask=excluded,
    )


def _build_observation_operator(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
) -> GaussianObservationOperator:
    point_index_source = np.asarray(prepared.obs_point_index)
    view_index_source = np.asarray(prepared.obs_view_index)
    jacobian_source = np.asarray(prepared.obs_J)
    weight_source = np.asarray(prepared.obs_weights)
    alphas_source = np.asarray(alpha.alphas)
    identifiable = np.asarray(alpha.identifiable_mask)
    num_points = int(prepared.points.shape[0])
    if point_index_source.ndim != 1:
        raise ValueError(
            f"obs_point_index must be a 1-D array, got {point_index_source.shape}."
        )
    num_observations = int(point_index_source.shape[0])
    if not np.issubdtype(point_index_source.dtype, np.integer):
        raise ValueError("obs_point_index must have integer dtype.")
    if not np.issubdtype(view_index_source.dtype, np.integer):
        raise ValueError("obs_view_index must have integer dtype.")
    if view_index_source.shape != (num_observations,):
        raise ValueError("obs_view_index must match obs_point_index length.")
    if jacobian_source.shape != (num_observations, 2, 3):
        raise ValueError(f"obs_J must have shape ({num_observations},2,3).")
    if weight_source.shape != (num_observations,):
        raise ValueError("obs_effective_weight must match obs_point_index length.")
    if alphas_source.ndim != 1 or alphas_source.size == 0:
        raise ValueError(
            f"alphas must be a non-empty 1-D array, got {alphas_source.shape}."
        )
    if identifiable.shape != alphas_source.shape or identifiable.dtype != np.bool_:
        raise ValueError("alpha_identifiable_mask must be boolean and match alphas.")
    point_index = point_index_source.astype(np.int64, copy=False)
    view_index = view_index_source.astype(np.int64, copy=False)
    if np.any(point_index < 0) or np.any(point_index >= num_points):
        raise ValueError("obs_point_index contains an out-of-range point index.")
    if np.any(view_index < 0) or np.any(view_index >= alphas_source.shape[0]):
        raise ValueError("obs_view_index contains an out-of-range view index.")
    jacobian = jacobian_source.astype(np.float64)
    weight = weight_source.astype(np.float64)
    alphas = alphas_source.astype(np.complex128)
    if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("obs_effective_weight must contain finite non-negative values.")

    valid_rows = identifiable[view_index] & (weight > 0.0)
    if np.any(~np.isfinite(jacobian[valid_rows])):
        raise ValueError("obs_J must be finite on positive-weight identifiable rows.")
    valid_alphas = alphas[view_index[valid_rows]]
    if np.any(~np.isfinite(valid_alphas)) or np.any(np.abs(valid_alphas) == 0.0):
        raise ValueError(
            "alphas must be finite and non-zero on positive-weight identifiable rows."
        )
    weighted_operator = np.zeros(
        (num_observations, 2, 3), dtype=np.complex128
    )
    weighted_operator[valid_rows] = (
        np.sqrt(weight[valid_rows])[:, None, None]
        * valid_alphas[:, None, None]
        * jacobian[valid_rows]
    )
    return GaussianObservationOperator(
        point_index=point_index,
        view_index=view_index,
        jacobian=jacobian,
        weight=weight,
        alphas=alphas,
        valid_rows=valid_rows,
        weighted_operator=weighted_operator,
    )


def _derive_numerical_completion_subspaces(
    operator: GaussianObservationOperator,
    num_points: int,
) -> GaussianMotionFillSubspaces:
    """Derive completion-only nullspaces without changing staged rank fields."""

    basis = np.broadcast_to(
        np.eye(3, dtype=np.float64), (num_points, 3, 3)
    ).copy()
    nullity = np.full((num_points,), 3, dtype=np.int8)
    valid_indices = np.flatnonzero(operator.valid_rows)
    if valid_indices.size == 0:
        return GaussianMotionFillSubspaces(
            basis=basis,
            nullity=nullity,
            valid_rows=operator.valid_rows,
        )

    ordered_rows = valid_indices[
        np.argsort(operator.point_index[valid_indices], kind="stable")
    ]
    counts = np.bincount(
        operator.point_index[ordered_rows], minlength=num_points
    ).astype(np.int64)
    offsets = np.zeros((num_points + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    epsilon = np.finfo(np.float64).eps
    for point in np.flatnonzero(counts):
        rows = ordered_rows[offsets[point] : offsets[point + 1]]
        geometry = (
            np.sqrt(operator.weight[rows])[:, None, None]
            * np.abs(operator.alphas[operator.view_index[rows]])[:, None, None]
            * operator.jacobian[rows]
        ).reshape(-1, 3)
        _, singular, vh = np.linalg.svd(
            geometry,
            full_matrices=geometry.shape[0] < geometry.shape[1],
        )
        if singular.size == 0 or singular[0] <= 0.0:
            raise ValueError(
                f"Point {int(point)} has positive-weight observations but zero "
                "operator energy."
            )
        tolerance = float(singular[0]) * max(geometry.shape) * epsilon
        rank = int(np.count_nonzero(singular > tolerance))
        dimension = 3 - rank
        basis[point] = 0.0
        if dimension:
            basis[point, :, :dimension] = vh[rank:].T
        nullity[point] = dimension
    return GaussianMotionFillSubspaces(
        basis=basis,
        nullity=nullity,
        valid_rows=operator.valid_rows,
    )


def _validate_observation_nullspaces(
    subspaces: GaussianMotionFillSubspaces,
    operator: GaussianObservationOperator,
    constrained_mask: np.ndarray,
) -> float:
    num_points = int(constrained_mask.shape[0])
    basis = np.asarray(subspaces.basis)
    nullity = np.asarray(subspaces.nullity)
    if basis.shape != (num_points, 3, 3) or nullity.shape != (num_points,):
        raise ValueError("Point nullspace arrays do not match points_world length.")
    point_index = operator.point_index
    weighted_operator = operator.weighted_operator
    valid_rows = operator.valid_rows
    active_rows = constrained_mask[point_index] & valid_rows
    operator_energy = np.zeros((num_points,), dtype=np.float64)
    nullspace_energy = np.zeros((num_points,), dtype=np.float64)
    np.add.at(
        operator_energy,
        point_index[active_rows],
        np.sum(np.abs(weighted_operator[active_rows]) ** 2, axis=(1, 2)),
    )
    observed_nullity = nullity[point_index]
    for dimension in (1, 2, 3):
        selected = active_rows & (observed_nullity == dimension)
        if not np.any(selected):
            continue
        projected = np.einsum(
            "nij,njk->nik",
            weighted_operator[selected],
            basis[point_index[selected], :, :dimension].astype(np.complex128),
        )
        np.add.at(
            nullspace_energy,
            point_index[selected],
            np.sum(np.abs(projected) ** 2, axis=(1, 2)),
        )
    if np.any(constrained_mask & (operator_energy <= 0.0)):
        point = int(np.where(constrained_mask & (operator_energy <= 0.0))[0][0])
        raise ValueError(f"Constrained point {point} has no positive-weight observation operator.")
    relative = np.zeros((num_points,), dtype=np.float64)
    relative[constrained_mask] = np.sqrt(
        nullspace_energy[constrained_mask] / operator_energy[constrained_mask]
    )
    maximum = float(np.max(relative, initial=0.0))
    if maximum > MOTION_FILL_NULLSPACE_RTOL:
        point = int(np.argmax(relative))
        raise ValueError(
            f"Motion-fill nullspace fails A_i N_i = 0 at point {point}: "
            f"relative error {relative[point]:.6g} exceeds "
            f"{MOTION_FILL_NULLSPACE_RTOL:.6g}."
        )
    return maximum


def _observation_drift(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    correction: np.ndarray,
    constrained_mask: np.ndarray,
    valid_rows: np.ndarray,
) -> float:
    correction_complex = np.asarray(correction, dtype=np.complex128)
    num_points = correction_complex.shape[0]
    point_index = np.asarray(prepared.obs_point_index, dtype=np.int64)
    view_index = np.asarray(prepared.obs_view_index, dtype=np.int64)
    jacobian = np.asarray(prepared.obs_J, dtype=np.float64)
    measurement = np.asarray(prepared.obs_y, dtype=np.complex128)
    weight = np.asarray(prepared.obs_weights, dtype=np.float64)
    alphas = np.asarray(alpha.alphas, dtype=np.complex128)
    if valid_rows.shape != point_index.shape or valid_rows.dtype != np.bool_:
        raise ValueError("valid_rows must be boolean and match the observation count.")
    projected = np.einsum(
        "nij,nj->ni", jacobian, correction_complex[point_index]
    )
    sqrt_weight = np.sqrt(weight)
    weighted_drift = sqrt_weight[:, None] * alphas[view_index, None] * projected
    weighted_measurement = sqrt_weight[:, None] * measurement
    drift_energy = np.zeros((num_points,), dtype=np.float64)
    measurement_energy = np.zeros((num_points,), dtype=np.float64)
    np.add.at(
        drift_energy,
        point_index[valid_rows],
        np.sum(np.abs(weighted_drift[valid_rows]) ** 2, axis=1),
    )
    np.add.at(
        measurement_energy,
        point_index[valid_rows],
        np.sum(np.abs(weighted_measurement[valid_rows]) ** 2, axis=1),
    )
    relative = np.zeros((num_points,), dtype=np.float64)
    relative[constrained_mask] = np.sqrt(
        drift_energy[constrained_mask]
    ) / np.maximum(
        np.sqrt(measurement_energy[constrained_mask]), MOTION_FILL_EPSILON
    )
    maximum = float(np.max(relative, initial=0.0))
    if maximum > MOTION_FILL_OBSERVATION_DRIFT_RTOL:
        point = int(np.argmax(relative))
        raise RuntimeError(
            f"Motion fill changed observations at point {point}: "
            f"relative drift {relative[point]:.6g} exceeds "
            f"{MOTION_FILL_OBSERVATION_DRIFT_RTOL:.6g}."
        )
    return maximum


def _json_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _solver_diagnostics(metadata: Any) -> dict[str, Any]:
    return {
        "performed": bool(metadata.performed),
        "converged": bool(metadata.converged),
        "stop_code": int(metadata.stop_code),
        "iterations": int(metadata.iterations),
        "residual_norm": _json_float(metadata.residual_norm),
        "normal_residual_norm": _json_float(metadata.normal_residual_norm),
        "matrix_norm": _json_float(metadata.matrix_norm),
        "condition_estimate": _json_float(metadata.condition_estimate),
        "solution_norm": _json_float(metadata.solution_norm),
    }


def apply_gaussian_motion_fill(
    staged: StagedSolveResult,
    graph: KnnGraph,
    graph_path: str,
) -> GaussianMotionFillResult:
    """Complete one staged Gaussian field without reading or writing artifacts."""

    if not graph_path:
        raise ValueError("graph_path must be non-empty.")
    prepared = staged.prepared
    alpha = staged.alpha
    observable = staged.observable
    points = np.asarray(prepared.points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points.shape}.")
    phi_observable = np.asarray(observable.phi_observable)
    num_points = int(points.shape[0])
    if phi_observable.shape != (num_points, 3):
        raise ValueError(
            f"phi_observable must have shape ({num_points},3), got "
            f"{phi_observable.shape}."
        )
    if np.asarray(observable.phi).shape != (num_points, 3):
        raise ValueError(
            f"phi must have shape ({num_points},3), got {np.asarray(observable.phi).shape}."
        )
    if not np.array_equal(observable.phi, phi_observable):
        raise ValueError("Staged phi must equal phi_observable before motion fill.")
    if graph.num_points != num_points:
        raise ValueError(
            f"Motion-fill graph has {graph.num_points} points, expected {num_points}."
        )

    staged_roles = derive_gaussian_motion_fill_roles(observable, observable.nullity)
    operator = _build_observation_operator(prepared, alpha)
    subspaces = _derive_numerical_completion_subspaces(operator, num_points)
    roles = derive_gaussian_motion_fill_roles(observable, subspaces.nullity)
    staged_nullity = np.asarray(observable.nullity, dtype=np.int8)
    refined_partial_mask = (
        observable.partial_mask & (staged_nullity != subspaces.nullity)
    )
    nullspace_error = _validate_observation_nullspaces(
        subspaces,
        operator,
        roles.constrained_variable_mask,
    )
    motion = fill_nullspace_motion(
        graph,
        phi_observable,
        subspaces.basis,
        subspaces.nullity,
        roles.fixed_anchor_mask,
        roles.constrained_variable_mask,
        roles.free_variable_mask,
        excluded_mask=roles.excluded_mask,
        lsmr_atol=MOTION_FILL_LSMR_ATOL,
        lsmr_btol=MOTION_FILL_LSMR_BTOL,
        lsmr_conlim=MOTION_FILL_LSMR_CONLIM,
    )
    fixed_mask = roles.fixed_anchor_mask | roles.excluded_mask
    if not np.array_equal(motion.phi[fixed_mask], phi_observable[fixed_mask]):
        raise RuntimeError("Motion fill changed a fixed anchor or excluded point.")
    if np.any(motion.phi_nullspace_correction[fixed_mask] != 0):
        raise RuntimeError("Motion fill assigned a correction to a fixed or excluded point.")
    observation_drift = _observation_drift(
        prepared,
        alpha,
        motion.phi_nullspace_correction,
        roles.constrained_variable_mask,
        subspaces.valid_rows,
    )

    pred, obs_residual, obs_residual_valid, point_residual, point_residual_valid = (
        compute_prediction_and_residuals(prepared, alpha, motion.phi)
    )
    point_status = np.asarray(observable.point_status, dtype=np.int8).copy()
    point_status[
        roles.constrained_variable_mask & motion.completion_mask
    ] = POINT_STATUS_COMPLETED_OBSERVED
    point_status[
        roles.free_variable_mask & motion.completion_mask
    ] = POINT_STATUS_COMPLETED_UNOBSERVED
    active_edges = roles.active_mask[graph.edge_index[:, 0]] & roles.active_mask[
        graph.edge_index[:, 1]
    ]

    role_counts = {
        name: int(np.count_nonzero(roles.role == index))
        for index, name in enumerate(MOTION_FILL_ROLE_NAMES)
    }
    excluded_reason_counts = {
        name: int(np.count_nonzero(roles.excluded_reason == index))
        for index, name in enumerate(MOTION_FILL_EXCLUDED_REASON_NAMES)
        if index != MOTION_FILL_EXCLUDED_NONE
    }
    numerical_nullity_counts = {
        str(dimension): int(np.count_nonzero(subspaces.nullity == dimension))
        for dimension in range(4)
    }
    diagnostics = {
        "method": MOTION_FILL_METHOD,
        "version": MOTION_FILL_VERSION,
        "role_counts": role_counts,
        "excluded_reason_counts": excluded_reason_counts,
        "numerical_subspace": {
            "rank_policy": MOTION_FILL_NUMERICAL_RANK_POLICY,
            "nullity_counts": numerical_nullity_counts,
            "refined_partial_count": int(np.count_nonzero(refined_partial_mask)),
            "staged_constrained_count": int(
                np.count_nonzero(staged_roles.constrained_variable_mask)
            ),
        },
        "completion_count": int(np.count_nonzero(motion.completion_mask)),
        "anchor_connected_point_count": int(
            np.count_nonzero(motion.completion_connected_to_anchor)
        ),
        "active_component_count": int(motion.connectivity.component_sizes.shape[0]),
        "anchor_connected_component_count": int(
            np.count_nonzero(motion.connectivity.component_has_anchor)
        ),
        "nullspace_operator_max_relative_error": nullspace_error,
        "observation_drift_max_relative": observation_drift,
        "tolerances": {
            "nullspace_operator_relative": MOTION_FILL_NULLSPACE_RTOL,
            "observation_drift_relative": MOTION_FILL_OBSERVATION_DRIFT_RTOL,
            "relative_denominator_epsilon": MOTION_FILL_EPSILON,
        },
        "system": {
            "row_count": int(motion.system_row_count),
            "column_count": int(motion.system_column_count),
            "active_edge_count": int(motion.active_edge_count),
            "eligible_edge_count": int(np.count_nonzero(active_edges)),
        },
        "lsmr_real": _solver_diagnostics(motion.real_solver),
        "lsmr_imaginary": _solver_diagnostics(motion.imag_solver),
    }
    return GaussianMotionFillResult(
        motion=motion,
        roles=roles,
        numerical_nullity=subspaces.nullity,
        staged_nullity_refined_mask=refined_partial_mask,
        point_solution_status=point_status,
        obs_pred_y=pred,
        obs_residual=obs_residual,
        obs_residual_valid_mask=obs_residual_valid,
        point_residual=point_residual,
        point_residual_valid_mask=point_residual_valid,
        diagnostics=diagnostics,
    )


def write_motion_fill_graph(
    path: str | Path,
    points_world: np.ndarray,
    candidates: KnnCandidateSet,
    graph: KnnGraph,
) -> Path:
    """Write the shared Gaussian KNN topology once for every solved mode."""

    out = Path(path)
    points = np.asarray(points_world, dtype=np.float32)
    if points.shape != (graph.num_points, 3):
        raise ValueError(
            f"points_world must have shape ({graph.num_points},3), got {points.shape}."
        )
    save_npz_compressed_atomic(
        out,
        {
            "points_world": points,
            "gaussian_indices": np.arange(graph.num_points, dtype=np.int32),
            "candidate_neighbor_indices": np.asarray(
                candidates.neighbor_indices[:, : graph.k], dtype=np.int64
            ),
            "candidate_neighbor_distances": np.asarray(
                candidates.neighbor_distances[:, : graph.k], dtype=np.float64
            ),
            "edge_index": np.asarray(graph.edge_index, dtype=np.int64),
            "edge_distance": np.asarray(graph.edge_distance, dtype=np.float64),
            "edge_weight": np.asarray(graph.edge_weight, dtype=np.float64),
            "degree": np.asarray(graph.degree, dtype=np.int32),
            "component_index": np.asarray(graph.component_index, dtype=np.int32),
            "component_sizes": np.asarray(graph.component_sizes, dtype=np.int32),
            "isolated_mask": np.asarray(graph.isolated_mask, dtype=bool),
            "k": np.array(graph.k, dtype=np.int32),
            "max_distance": np.array(graph.max_distance, dtype=np.float64),
            "epsilon": np.array(graph.epsilon, dtype=np.float64),
            "candidate_directed_count": np.array(
                graph.candidate_directed_count, dtype=np.int64
            ),
            "retained_directed_count": np.array(
                graph.retained_directed_count, dtype=np.int64
            ),
            "pruned_directed_count": np.array(
                graph.pruned_directed_count, dtype=np.int64
            ),
            "unique_undirected_edge_count": np.array(
                graph.unique_undirected_edge_count, dtype=np.int64
            ),
            "zero_distance_edge_count": np.array(
                np.count_nonzero(graph.edge_distance == 0.0), dtype=np.int64
            ),
        },
    )
    return out


def write_motion_fill_diagnostics(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return out
