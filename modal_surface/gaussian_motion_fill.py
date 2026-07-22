"""Production motion-fill post-processing for Gaussian modal fields."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from modal_surface.io import save_npz_compressed_atomic
from modal_surface.motion_fill import (
    AnchorConnectivity,
    KnnCandidateSet,
    KnnGraph,
    MotionFillResult,
    SparseSolveMetadata,
    build_shared_group_membership,
    fill_nullspace_motion,
)
from modal_surface.observed_structure_graph import ObservedStructureGraph
from modal_surface.optimization_staged import (
    POINT_STATUS_COMPLETED_OBSERVED,
    POINT_STATUS_COMPLETED_UNOBSERVED,
    AlphaSyncResult,
    ObservableSolveResult,
    PreparedObservations,
    StagedSolveResult,
    compute_prediction_and_residuals,
)
from modal_surface.rigid_component_solver import (
    RigidComponentSeedSelectionResult,
    RigidComponentSolveResult,
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
RIGID_SEQUENTIAL_MOTION_FILL_METHOD = (
    "rigid_seed_partial_component_then_independent_gaussian_knn_lsmr"
)
RIGID_SINGLE_VIEW_PARTIAL_FILL_METHOD = (
    "rigid_seed_single_view_partial_component_knn_lsmr"
)
RIGID_SINGLE_VIEW_PARTIAL_FILL_POLICY = (
    "observable_twist_plus_knn_filled_weak_and_ray_directions"
)


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


@dataclass(frozen=True)
class RigidSeedMotionFillResult:
    """Rigid-component and independent-Gaussian completion from fixed seeds."""

    motion: MotionFillResult
    roles: GaussianMotionFillRoles
    numerical_nullity: np.ndarray
    observed_mask: np.ndarray
    usable_observed_mask: np.ndarray
    fill_target_mask: np.ndarray
    single_view_component_fill_mask: np.ndarray
    single_view_rigid_fill_point_mask: np.ndarray
    single_view_component_completion_mask: np.ndarray
    single_view_component_translation: np.ndarray
    single_view_component_rotation: np.ndarray
    single_view_component_first_order_relative_max: np.ndarray
    obs_pred_y: np.ndarray
    obs_residual: np.ndarray
    obs_residual_valid_mask: np.ndarray
    point_residual: np.ndarray
    point_residual_valid_mask: np.ndarray
    diagnostics: dict[str, Any]
    single_view_partial_diagnostics: RigidSingleViewPartialDiagnostics | None = None


@dataclass(frozen=True)
class RigidSingleViewPartialDiagnostics:
    observable_singular_ratio_min: float
    ray_direction_min_fraction: float
    max_finite_drift: float
    component_observable_rank: np.ndarray
    component_fill_nullity: np.ndarray
    component_ray_dominated_basis_count: np.ndarray
    component_trusted_knn_edge_count: np.ndarray
    component_cross_knn_edge_count: np.ndarray
    component_connected_to_trusted_mask: np.ndarray
    component_postfill_normalized_residual: np.ndarray
    component_ray_motion_rms: np.ndarray
    component_tangent_motion_rms: np.ndarray
    component_ray_motion_ratio: np.ndarray
    component_finite_drift_max: np.ndarray
    component_finite_drift_rejected_mask: np.ndarray
    component_postfill_retained_mask: np.ndarray


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


def derive_rigid_seed_motion_fill_roles(
    rigid_seed_mask: np.ndarray,
    constrained_component_mask: np.ndarray,
) -> GaussianMotionFillRoles:
    """Assign fixed seeds, optional component variables, and free points."""

    mask_source = np.asarray(rigid_seed_mask)
    if mask_source.ndim != 1:
        raise ValueError(
            f"rigid_seed_mask must be a 1-D array, got {mask_source.shape}."
        )
    seed = _require_boolean_mask(
        mask_source, "rigid_seed_mask", int(mask_source.shape[0])
    )
    constrained = _require_boolean_mask(
        constrained_component_mask,
        "constrained_component_mask",
        int(mask_source.shape[0]),
    )
    if not np.any(seed):
        raise ValueError("rigid_seed_mask must contain at least one fixed seed.")
    if np.any(seed & constrained):
        raise ValueError(
            "rigid_seed_mask and constrained_component_mask must be disjoint"
        )
    free = ~(seed | constrained)
    empty = np.zeros(seed.shape, dtype=bool)
    role = np.full(seed.shape, MOTION_FILL_ROLE_FREE_VARIABLE, dtype=np.int8)
    role[seed] = MOTION_FILL_ROLE_FIXED_ANCHOR
    role[constrained] = MOTION_FILL_ROLE_CONSTRAINED_VARIABLE
    return GaussianMotionFillRoles(
        role=role,
        excluded_reason=np.full(
            seed.shape, MOTION_FILL_EXCLUDED_NONE, dtype=np.int8
        ),
        fixed_anchor_mask=seed,
        constrained_variable_mask=constrained,
        free_variable_mask=free,
        excluded_mask=empty,
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


def _rigid_point_blocks(
    points: np.ndarray,
    centroid: np.ndarray,
    radius: float,
) -> np.ndarray:
    centered = np.asarray(points, dtype=np.float64) - np.asarray(
        centroid, dtype=np.float64
    )
    blocks = np.zeros((centered.shape[0], 3, 6), dtype=np.float64)
    blocks[:, :, :3] = np.eye(3, dtype=np.float64)[None]
    if radius > MOTION_FILL_EPSILON:
        skew = np.zeros((centered.shape[0], 3, 3), dtype=np.float64)
        skew[:, 0, 1] = -centered[:, 2]
        skew[:, 0, 2] = centered[:, 1]
        skew[:, 1, 0] = centered[:, 2]
        skew[:, 1, 2] = -centered[:, 0]
        skew[:, 2, 0] = -centered[:, 1]
        skew[:, 2, 1] = centered[:, 0]
        blocks[:, :, 3:] = -skew / radius
    return blocks


def _component_lsmr(
    matrix: Any,
    right_hand_side: np.ndarray,
) -> tuple[np.ndarray, SparseSolveMetadata, int]:
    try:
        from scipy.sparse.linalg import lsmr
    except ImportError as exc:
        raise ImportError(
            "Single-view component fill requires scipy.sparse.linalg.lsmr"
        ) from exc
    maxiter = max(1000, 4 * int(matrix.shape[1]))
    solved = lsmr(
        matrix,
        np.asarray(right_hand_side, dtype=np.float64),
        atol=MOTION_FILL_LSMR_ATOL,
        btol=MOTION_FILL_LSMR_BTOL,
        conlim=MOTION_FILL_LSMR_CONLIM,
        maxiter=maxiter,
    )
    stop_code = int(solved[1])
    metadata = SparseSolveMetadata(
        performed=True,
        converged=stop_code in {0, 1, 2, 4, 5},
        stop_code=stop_code,
        iterations=int(solved[2]),
        residual_norm=float(solved[3]),
        normal_residual_norm=float(solved[4]),
        matrix_norm=float(solved[5]),
        condition_estimate=float(solved[6]),
        solution_norm=float(solved[7]),
    )
    return np.asarray(solved[0], dtype=np.float64), metadata, maxiter


def _empty_component_lsmr(right_hand_side: np.ndarray) -> SparseSolveMetadata:
    return SparseSolveMetadata(
        performed=False,
        converged=True,
        stop_code=0,
        iterations=0,
        residual_norm=float(np.linalg.norm(right_hand_side)),
        normal_residual_norm=0.0,
        matrix_norm=0.0,
        condition_estimate=1.0,
        solution_norm=0.0,
    )


def _component_finite_drift_max(
    points: np.ndarray,
    phi: np.ndarray,
    global_edges: np.ndarray,
    edge_components: np.ndarray,
    completed_components: np.ndarray,
    phase_angles: np.ndarray,
    num_components: int,
) -> np.ndarray:
    out = np.zeros((num_components,), dtype=np.float64)
    selected = completed_components[edge_components]
    selected_indices = np.flatnonzero(selected)
    cos_phase = np.cos(phase_angles)[None]
    sin_phase = np.sin(phase_angles)[None]
    for start in range(0, selected_indices.size, 32768):
        indices = selected_indices[start : start + 32768]
        edges = global_edges[indices]
        base = points[edges[:, 1]] - points[edges[:, 0]]
        delta = phi[edges[:, 1]] - phi[edges[:, 0]]
        displacement = (
            delta.real[:, :, None] * cos_phase[:, None, :]
            - delta.imag[:, :, None] * sin_phase[:, None, :]
        )
        base_length = np.linalg.norm(base, axis=1)
        deformed_length = np.linalg.norm(base[:, :, None] + displacement, axis=1)
        edge_maximum = np.max(
            np.abs(deformed_length - base_length[:, None])
            / np.maximum(base_length[:, None], MOTION_FILL_EPSILON),
            axis=1,
        )
        np.maximum.at(out, edge_components[indices], edge_maximum)
    return out


def apply_single_view_component_partial_fill(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    rigid: RigidComponentSolveResult,
    seed_selection: RigidComponentSeedSelectionResult,
    observed_graph: ObservedStructureGraph,
    graph: KnnGraph,
    graph_path: str,
    *,
    observable_singular_ratio_min: float,
    ray_direction_min_fraction: float,
    max_finite_drift: float,
    timings: dict[str, float] | None = None,
    progress: Callable[[str], None] | None = None,
) -> RigidSeedMotionFillResult:
    """Fill only weak single-view component twists from trusted-component KNNs."""

    total_started = perf_counter()
    preparation_started = perf_counter()
    if progress is not None:
        progress("single-view partial component preparation started")
    if not graph_path:
        raise ValueError("graph_path must be non-empty")
    if (
        not np.isfinite(observable_singular_ratio_min)
        or not 0.0 < observable_singular_ratio_min <= 1.0
    ):
        raise ValueError(
            "observable_singular_ratio_min must be finite and lie in (0,1]"
        )
    if (
        not np.isfinite(ray_direction_min_fraction)
        or not 0.0 <= ray_direction_min_fraction <= 1.0
    ):
        raise ValueError(
            "ray_direction_min_fraction must be finite and lie in [0,1]"
        )
    if not np.isfinite(max_finite_drift) or max_finite_drift < 0.0:
        raise ValueError("max_finite_drift must be finite and non-negative")
    points = np.asarray(prepared.points, dtype=np.float64)
    num_points = int(points.shape[0])
    if points.shape != (num_points, 3) or graph.num_points != num_points:
        raise ValueError("prepared points and motion-fill graph are inconsistent")
    num_components = rigid.num_components
    point_component = np.asarray(rigid.point_component_index)
    component_view_count = np.asarray(
        seed_selection.component_supported_valid_view_count
    )
    component_retained = np.asarray(
        seed_selection.component_seed_retained_mask, dtype=bool
    )
    if (
        point_component.shape != (num_points,)
        or component_view_count.shape != (num_components,)
        or component_retained.shape != (num_components,)
    ):
        raise ValueError("rigid component metadata is inconsistent")
    single_view_component_fill_mask = (
        (component_view_count == 1) & ~component_retained
    )
    selected_components = np.flatnonzero(single_view_component_fill_mask)
    if selected_components.size == 0:
        raise ValueError(
            "Component-only motion fill found no rejected single-view component"
        )
    component_to_group = np.full((num_components,), -1, dtype=np.int32)
    component_to_group[selected_components] = np.arange(
        selected_components.size, dtype=np.int32
    )
    shared_group_index = np.full((num_points,), -1, dtype=np.int32)
    component_points = point_component >= 0
    selected_point_mask = np.zeros((num_points,), dtype=bool)
    selected_point_mask[component_points] = single_view_component_fill_mask[
        point_component[component_points]
    ]
    shared_group_index[selected_point_mask] = component_to_group[
        point_component[selected_point_mask]
    ]
    membership = build_shared_group_membership(shared_group_index, num_points)
    trusted_seed_mask = np.asarray(
        seed_selection.trusted_rigid_seed_mask, dtype=bool
    )
    roles = derive_rigid_seed_motion_fill_roles(
        trusted_seed_mask,
        selected_point_mask,
    )

    component_centroid = np.asarray(rigid.component_centroid, dtype=np.float64)
    component_radius = np.asarray(rigid.component_radius, dtype=np.float64)
    if (
        component_centroid.shape != (num_components, 3)
        or component_radius.shape != (num_components,)
        or not np.isfinite(component_centroid).all()
        or not np.isfinite(component_radius).all()
    ):
        raise ValueError("rigid component geometry is invalid")

    usable_row_mask = (
        (prepared.obs_weights > 0.0)
        & alpha.identifiable_mask[prepared.obs_view_index]
        & (point_component[prepared.obs_point_index] >= 0)
    )
    usable_rows = np.flatnonzero(usable_row_mask)
    usable_row_components = point_component[prepared.obs_point_index[usable_rows]]
    row_order = np.argsort(usable_row_components, kind="stable")
    usable_rows = usable_rows[row_order]
    usable_row_components = usable_row_components[row_order]
    component_row_count = np.bincount(
        usable_row_components, minlength=num_components
    ).astype(np.int64)
    row_offsets = np.zeros((num_components + 1,), dtype=np.int64)
    row_offsets[1:] = np.cumsum(component_row_count, dtype=np.int64)

    ray_by_point = np.zeros((num_points, 3), dtype=np.float64)
    row_points = prepared.obs_point_index[usable_rows]
    unique_points, first_row_positions = np.unique(
        row_points, return_index=True
    )
    first_rows = usable_rows[first_row_positions]
    row_jacobians = prepared.obs_J[first_rows].astype(np.float64)
    rays = np.cross(row_jacobians[:, 0], row_jacobians[:, 1])
    ray_norm = np.linalg.norm(rays, axis=1)
    if np.any(ray_norm <= MOTION_FILL_EPSILON):
        raise ValueError("an observation projection Jacobian has no viewing ray")
    ray_by_point[unique_points] = rays / ray_norm[:, None]
    if np.any(np.linalg.norm(ray_by_point[membership.ordered_points], axis=1) < 0.5):
        raise ValueError("a single-view component point has no usable viewing ray")

    point_blocks = np.zeros((num_points, 3, 6), dtype=np.float64)
    observable_twist = np.zeros((num_components, 6), dtype=np.complex128)
    fill_basis = np.zeros((num_components, 6, 6), dtype=np.float64)
    observable_rank = np.zeros((num_components,), dtype=np.int8)
    fill_nullity = np.zeros((num_components,), dtype=np.int8)
    ray_dominated_count = np.zeros((num_components,), dtype=np.int8)
    observable_phi = np.zeros((num_points, 3), dtype=np.complex128)
    observable_phi[trusted_seed_mask] = np.asarray(
        seed_selection.phi[trusted_seed_mask], dtype=np.complex128
    )

    for group_idx, component_idx in enumerate(selected_components.tolist()):
        member_start = int(membership.offsets[group_idx])
        member_end = int(membership.offsets[group_idx + 1])
        members = membership.ordered_points[member_start:member_end]
        blocks = _rigid_point_blocks(
            points[members],
            component_centroid[component_idx],
            float(component_radius[component_idx]),
        )
        point_blocks[members] = blocks
        rows = usable_rows[
            row_offsets[component_idx] : row_offsets[component_idx + 1]
        ]
        if rows.size == 0:
            raise ValueError(
                f"single-view component {component_idx} has no usable observation row"
            )
        row_component_blocks = _rigid_point_blocks(
            points[prepared.obs_point_index[rows]],
            component_centroid[component_idx],
            float(component_radius[component_idx]),
        )
        projected_blocks = np.einsum(
            "rij,rjk->rik",
            prepared.obs_J[rows].astype(np.float64),
            row_component_blocks,
        )
        sqrt_weight = np.sqrt(prepared.obs_weights[rows])
        row_alpha = alpha.alphas[prepared.obs_view_index[rows]].astype(
            np.complex128
        )
        design = (
            sqrt_weight[:, None, None]
            * row_alpha[:, None, None]
            * projected_blocks.astype(np.complex128)
        ).reshape(-1, 6)
        target = (
            sqrt_weight[:, None] * prepared.obs_y[rows].astype(np.complex128)
        ).reshape(-1)
        gram = design.conj().T @ design
        imaginary_gram_max = float(np.max(np.abs(gram.imag), initial=0.0))
        if imaginary_gram_max > 1.0e-8 * max(
            1.0, float(np.max(np.abs(gram.real), initial=0.0))
        ):
            raise RuntimeError("single-view component Gram matrix is not real")
        eigenvalues, eigenvectors = np.linalg.eigh(gram.real)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        right_vectors = eigenvectors[:, order]
        singular = np.sqrt(eigenvalues)
        singular_ratio = singular / max(float(singular[0]), MOTION_FILL_EPSILON)
        normal_rhs = design.conj().T @ target
        mode_coefficients = np.zeros((6,), dtype=np.complex128)
        positive = eigenvalues > MOTION_FILL_EPSILON**2
        mode_coefficients[positive] = (
            right_vectors[:, positive].T @ normal_rhs
        ) / eigenvalues[positive]

        induced = np.einsum("nij,jk->nik", blocks, right_vectors)
        induced_energy = np.sum(np.square(induced), axis=(0, 1))
        radial = np.einsum(
            "ni,nik->nk", ray_by_point[members], induced
        )
        radial_energy = np.sum(np.square(radial), axis=0)
        radial_fraction = np.sqrt(
            radial_energy / np.maximum(induced_energy, MOTION_FILL_EPSILON**2)
        )
        physical = induced_energy > MOTION_FILL_EPSILON**2
        ray_dominated = physical & (
            radial_fraction >= ray_direction_min_fraction
        )
        stable = (
            physical
            & (singular_ratio >= observable_singular_ratio_min)
            & ~ray_dominated
        )
        fill_directions = physical & ~stable
        observable_rank[component_idx] = int(np.count_nonzero(stable))
        fill_nullity[component_idx] = int(np.count_nonzero(fill_directions))
        ray_dominated_count[component_idx] = int(
            np.count_nonzero(ray_dominated)
        )
        observable_twist[component_idx] = (
            right_vectors[:, stable] @ mode_coefficients[stable]
        )
        dimension = int(fill_nullity[component_idx])
        if dimension:
            fill_basis[component_idx, :, :dimension] = right_vectors[
                :, fill_directions
            ]
        observable_phi[members] = np.einsum(
            "nij,j->ni", blocks, observable_twist[component_idx]
        )

    if timings is not None:
        timings["preparation_seconds"] = float(
            perf_counter() - preparation_started
        )
    if progress is not None:
        progress(
            "single-view partial component preparation finished in "
            f"{perf_counter() - preparation_started:.3f} s: "
            f"groups={selected_components.size}, points={membership.ordered_points.size}"
        )

    connectivity_started = perf_counter()
    if progress is not None:
        progress("single-view component KNN connectivity started")
    edges = np.asarray(graph.edge_index, dtype=np.int64)
    endpoint_trusted = trusted_seed_mask[edges]
    endpoint_groups = shared_group_index[edges]
    endpoint_single = endpoint_groups >= 0
    eligible_edge_mask = (
        (endpoint_trusted[:, 0] | endpoint_single[:, 0])
        & (endpoint_trusted[:, 1] | endpoint_single[:, 1])
        & (endpoint_single[:, 0] | endpoint_single[:, 1])
        & ~(
            endpoint_single[:, 0]
            & endpoint_single[:, 1]
            & (endpoint_groups[:, 0] == endpoint_groups[:, 1])
        )
    )
    eligible_indices = np.flatnonzero(eligible_edge_mask)
    group_adjacency: list[list[int]] = [
        [] for _ in range(selected_components.size)
    ]
    group_hop = np.full((selected_components.size,), -1, dtype=np.int32)
    queue: list[int] = []
    trusted_knn_count = np.zeros((num_components,), dtype=np.int32)
    cross_knn_count = np.zeros((num_components,), dtype=np.int32)
    for edge_idx in eligible_indices.tolist():
        group_i = int(endpoint_groups[edge_idx, 0])
        group_j = int(endpoint_groups[edge_idx, 1])
        if group_i >= 0:
            component_i = int(selected_components[group_i])
            cross_knn_count[component_i] += 1
            if endpoint_trusted[edge_idx, 1]:
                trusted_knn_count[component_i] += 1
                if group_hop[group_i] < 0:
                    group_hop[group_i] = 1
                    queue.append(group_i)
        if group_j >= 0:
            component_j = int(selected_components[group_j])
            cross_knn_count[component_j] += 1
            if endpoint_trusted[edge_idx, 0]:
                trusted_knn_count[component_j] += 1
                if group_hop[group_j] < 0:
                    group_hop[group_j] = 1
                    queue.append(group_j)
        if group_i >= 0 and group_j >= 0:
            group_adjacency[group_i].append(group_j)
            group_adjacency[group_j].append(group_i)
    queue_position = 0
    while queue_position < len(queue):
        group_idx = queue[queue_position]
        queue_position += 1
        next_hop = int(group_hop[group_idx]) + 1
        for neighbor in group_adjacency[group_idx]:
            if group_hop[neighbor] >= 0:
                continue
            group_hop[neighbor] = next_hop
            queue.append(neighbor)
    connected_groups = group_hop >= 0
    component_connected = np.zeros((num_components,), dtype=bool)
    component_connected[selected_components] = connected_groups
    active_edge_mask = eligible_edge_mask.copy()
    for side in (0, 1):
        side_single = endpoint_single[:, side]
        active_edge_mask[side_single] &= connected_groups[
            endpoint_groups[side_single, side]
        ]
    active_edge_indices = np.flatnonzero(active_edge_mask)
    active_edges = edges[active_edge_indices]
    if timings is not None:
        timings["connectivity_seconds"] = float(
            perf_counter() - connectivity_started
        )
    if progress is not None:
        progress(
            "single-view component KNN connectivity finished in "
            f"{perf_counter() - connectivity_started:.3f} s: "
            f"eligible_edges={eligible_indices.size}, active_edges={active_edges.shape[0]}, "
            f"connected_groups={int(np.count_nonzero(connected_groups))}"
        )

    assembly_started = perf_counter()
    if progress is not None:
        progress("single-view component sparse assembly started")
    group_dimensions = fill_nullity[selected_components].astype(np.int32)
    active_dimensions = np.where(
        connected_groups, group_dimensions, 0
    ).astype(np.int64)
    coefficient_offsets = np.zeros(
        (selected_components.size + 1,), dtype=np.int64
    )
    coefficient_offsets[1:] = np.cumsum(active_dimensions, dtype=np.int64)
    correction_blocks = np.zeros((num_points, 3, 6), dtype=np.float64)
    for group_idx, component_idx in enumerate(selected_components.tolist()):
        dimension = int(group_dimensions[group_idx])
        if dimension == 0:
            continue
        members = membership.ordered_points[
            membership.offsets[group_idx] : membership.offsets[group_idx + 1]
        ]
        correction_blocks[members, :, :dimension] = np.einsum(
            "nij,jk->nik",
            point_blocks[members],
            fill_basis[component_idx, :, :dimension],
        )
    phi_observable = np.zeros((num_points, 3), dtype=np.complex128)
    phi_observable[trusted_seed_mask] = observable_phi[trusted_seed_mask]
    phi_observable[selected_point_mask] = observable_phi[selected_point_mask]
    sqrt_weight = np.sqrt(graph.edge_weight[active_edge_indices])
    right_hand_side = (
        sqrt_weight[:, None]
        * (
            phi_observable[active_edges[:, 1]]
            - phi_observable[active_edges[:, 0]]
        )
    ).reshape(-1)
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    edge_rows = np.arange(active_edges.shape[0], dtype=np.int64)
    coordinate = np.arange(3, dtype=np.int64)
    for side, sign in ((0, 1.0), (1, -1.0)):
        side_points = active_edges[:, side]
        side_groups = shared_group_index[side_points]
        for dimension in range(1, 7):
            selected = (side_groups >= 0) & (
                group_dimensions[np.maximum(side_groups, 0)] == dimension
            )
            if not np.any(selected):
                continue
            selected_rows = edge_rows[selected]
            selected_points = side_points[selected]
            selected_groups = side_groups[selected]
            block = (
                sign
                * sqrt_weight[selected, None, None]
                * correction_blocks[selected_points, :, :dimension]
            )
            rows = np.broadcast_to(
                selected_rows[:, None, None] * 3 + coordinate[None, :, None],
                block.shape,
            )
            columns = np.broadcast_to(
                coefficient_offsets[selected_groups, None, None]
                + np.arange(dimension, dtype=np.int64)[None, None],
                block.shape,
            )
            row_parts.append(rows.reshape(-1))
            column_parts.append(columns.reshape(-1))
            value_parts.append(block.reshape(-1))
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise ImportError(
            "Single-view component fill requires scipy.sparse"
        ) from exc
    row_count = int(active_edges.shape[0] * 3)
    column_count = int(coefficient_offsets[-1])
    if value_parts:
        matrix = coo_matrix(
            (
                np.concatenate(value_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=(row_count, column_count),
            dtype=np.float64,
        ).tocsr()
        matrix.eliminate_zeros()
    else:
        matrix = coo_matrix(
            (row_count, column_count), dtype=np.float64
        ).tocsr()
    if timings is not None:
        timings["system_assembly_seconds"] = float(
            perf_counter() - assembly_started
        )
    if progress is not None:
        progress(
            "single-view component sparse assembly finished in "
            f"{perf_counter() - assembly_started:.3f} s: rows={row_count}, "
            f"cols={column_count}, nnz={int(matrix.nnz)}"
        )

    if column_count:
        solve_started = perf_counter()
        if progress is not None:
            progress("single-view component LSMR real started")
        real_coefficients, real_solver, lsmr_maxiter = _component_lsmr(
            matrix, right_hand_side.real
        )
        if timings is not None:
            timings["lsmr_real_seconds"] = float(
                perf_counter() - solve_started
            )
        if progress is not None:
            progress(
                "single-view component LSMR real finished in "
                f"{perf_counter() - solve_started:.3f} s: "
                f"iterations={real_solver.iterations}, stop_code={real_solver.stop_code}, "
                f"maxiter={lsmr_maxiter}"
            )
        solve_started = perf_counter()
        if progress is not None:
            progress("single-view component LSMR imaginary started")
        (
            imaginary_coefficients,
            imag_solver,
            imaginary_lsmr_maxiter,
        ) = _component_lsmr(matrix, right_hand_side.imag)
        if imaginary_lsmr_maxiter != lsmr_maxiter:
            raise RuntimeError("real and imaginary component LSMR budgets differ")
        if timings is not None:
            timings["lsmr_imaginary_seconds"] = float(
                perf_counter() - solve_started
            )
        if progress is not None:
            progress(
                "single-view component LSMR imaginary finished in "
                f"{perf_counter() - solve_started:.3f} s: "
                f"iterations={imag_solver.iterations}, stop_code={imag_solver.stop_code}, "
                f"maxiter={lsmr_maxiter}"
            )
        if not real_solver.converged or not imag_solver.converged:
            raise RuntimeError(
                "Single-view component LSMR did not converge: "
                f"real stop_code={real_solver.stop_code}, "
                f"imaginary stop_code={imag_solver.stop_code}"
            )
        coefficient_values = (
            real_coefficients + 1j * imaginary_coefficients
        )
    else:
        coefficient_values = np.empty((0,), dtype=np.complex128)
        real_solver = _empty_component_lsmr(right_hand_side.real)
        imag_solver = _empty_component_lsmr(right_hand_side.imag)
        if timings is not None:
            timings["lsmr_real_seconds"] = 0.0
            timings["lsmr_imaginary_seconds"] = 0.0

    validation_started = perf_counter()
    if progress is not None:
        progress("single-view component reconstruction and validation started")
    final_phi = phi_observable.copy()
    correction = np.zeros((num_points, 3), dtype=np.complex128)
    component_translation = np.zeros((num_components, 3), dtype=np.complex128)
    component_rotation = np.zeros((num_components, 3), dtype=np.complex128)
    component_final_twist = observable_twist.copy()
    for group_idx, component_idx in enumerate(selected_components.tolist()):
        dimension = (
            int(group_dimensions[group_idx])
            if connected_groups[group_idx]
            else 0
        )
        if dimension > 0:
            start = int(coefficient_offsets[group_idx])
            coefficients = coefficient_values[start : start + dimension]
            component_final_twist[component_idx] += (
                fill_basis[component_idx, :, :dimension] @ coefficients
            )
        members = membership.ordered_points[
            membership.offsets[group_idx] : membership.offsets[group_idx + 1]
        ]
        final_phi[members] = np.einsum(
            "nij,j->ni",
            point_blocks[members],
            component_final_twist[component_idx],
        )
        correction[members] = final_phi[members] - phi_observable[members]
        component_translation[component_idx] = component_final_twist[
            component_idx, :3
        ]
        radius = float(component_radius[component_idx])
        if radius > MOTION_FILL_EPSILON:
            component_rotation[component_idx] = component_final_twist[
                component_idx, 3:
            ] / radius
    final_phi[trusted_seed_mask] = np.asarray(
        seed_selection.phi[trusted_seed_mask], dtype=np.complex128
    )

    local_edges = np.asarray(observed_graph.topology.edge_index, dtype=np.int64)
    node_indices = np.asarray(observed_graph.node_gaussian_indices, dtype=np.int64)
    global_edges = node_indices[local_edges]
    edge_components = np.asarray(rigid.edge_component_index, dtype=np.int64)
    edge_vectors = points[global_edges[:, 1]] - points[global_edges[:, 0]]
    edge_delta_phi = final_phi[global_edges[:, 1]] - final_phi[global_edges[:, 0]]
    edge_axial = np.einsum("ij,ij->i", edge_vectors, edge_delta_phi)
    edge_squared_length = np.einsum("ij,ij->i", edge_vectors, edge_vectors)
    edge_relative = np.maximum(
        np.abs(edge_axial.real), np.abs(edge_axial.imag)
    ) / np.maximum(edge_squared_length, MOTION_FILL_EPSILON)
    component_first_order_relative_max = np.zeros(
        (num_components,), dtype=np.float64
    )
    selected_component_edges = single_view_component_fill_mask[edge_components]
    np.maximum.at(
        component_first_order_relative_max,
        edge_components[selected_component_edges],
        edge_relative[selected_component_edges],
    )
    selected_first_order_max = float(
        np.max(
            component_first_order_relative_max[single_view_component_fill_mask],
            initial=0.0,
        )
    )
    if selected_first_order_max > rigid.config.first_order_rtol:
        raise RuntimeError(
            "Single-view partial component fill violated first-order rigidity: "
            f"max_relative={selected_first_order_max:.9g}"
        )

    postfill_residual = np.zeros((num_components,), dtype=np.float64)
    ray_motion_rms = np.zeros((num_components,), dtype=np.float64)
    tangent_motion_rms = np.zeros((num_components,), dtype=np.float64)
    ray_motion_ratio = np.zeros((num_components,), dtype=np.float64)
    for group_idx, component_idx in enumerate(selected_components.tolist()):
        rows = usable_rows[
            row_offsets[component_idx] : row_offsets[component_idx + 1]
        ]
        row_component_blocks = _rigid_point_blocks(
            points[prepared.obs_point_index[rows]],
            component_centroid[component_idx],
            float(component_radius[component_idx]),
        )
        projected_blocks = np.einsum(
            "rij,rjk->rik",
            prepared.obs_J[rows].astype(np.float64),
            row_component_blocks,
        )
        sqrt_weight_rows = np.sqrt(prepared.obs_weights[rows])
        design = (
            sqrt_weight_rows[:, None, None]
            * alpha.alphas[prepared.obs_view_index[rows]][:, None, None]
            * projected_blocks
        ).reshape(-1, 6)
        target = (
            sqrt_weight_rows[:, None] * prepared.obs_y[rows]
        ).reshape(-1)
        residual_norm = float(
            np.linalg.norm(design @ component_final_twist[component_idx] - target)
        )
        postfill_residual[component_idx] = residual_norm / max(
            float(np.linalg.norm(target)), MOTION_FILL_EPSILON
        )
        members = membership.ordered_points[
            membership.offsets[group_idx] : membership.offsets[group_idx + 1]
        ]
        member_phi = final_phi[members]
        radial_amplitude = np.einsum(
            "ni,ni->n", ray_by_point[members], member_phi
        )
        tangent_phi = (
            member_phi - radial_amplitude[:, None] * ray_by_point[members]
        )
        ray_motion_rms[component_idx] = np.sqrt(
            float(np.mean(np.abs(radial_amplitude) ** 2))
        )
        tangent_motion_rms[component_idx] = np.sqrt(
            float(np.mean(np.sum(np.abs(tangent_phi) ** 2, axis=1)))
        )
        ray_motion_ratio[component_idx] = ray_motion_rms[component_idx] / max(
            tangent_motion_rms[component_idx], MOTION_FILL_EPSILON
        )
    component_finite_drift = _component_finite_drift_max(
        points,
        final_phi,
        global_edges,
        edge_components,
        single_view_component_fill_mask,
        np.asarray(rigid.phase_angles, dtype=np.float64),
        num_components,
    )

    selected_members = membership.ordered_points
    component_finite_drift_rejected = (
        single_view_component_fill_mask
        & (component_finite_drift > max_finite_drift)
    )
    component_postfill_retained = (
        single_view_component_fill_mask
        & ~component_finite_drift_rejected
    )
    component_postfill_rejected = (
        single_view_component_fill_mask & ~component_postfill_retained
    )
    rejected_point_mask = np.zeros((num_points,), dtype=bool)
    rejected_point_mask[selected_point_mask] = component_postfill_rejected[
        point_component[selected_point_mask]
    ]
    final_phi[rejected_point_mask] = 0.0
    phi_observable[rejected_point_mask] = 0.0
    correction[rejected_point_mask] = 0.0
    component_final_twist[component_postfill_rejected] = 0.0
    component_translation[component_postfill_rejected] = 0.0
    component_rotation[component_postfill_rejected] = 0.0

    effective_connected_groups = (
        connected_groups & component_postfill_retained[selected_components]
    )
    component_completion = np.zeros((num_components,), dtype=bool)
    component_completion[selected_components] = effective_connected_groups
    completion_mask = np.zeros((num_points,), dtype=bool)
    if selected_members.size:
        completion_mask[selected_members] = effective_connected_groups[
            shared_group_index[selected_members]
        ]
    if progress is not None:
        progress(
            "single-view component post-fill gate: "
            f"retained={int(np.count_nonzero(component_postfill_retained))}, "
            f"finite_drift_rejected={int(np.count_nonzero(component_finite_drift_rejected))}, "
            f"total_rejected={int(np.count_nonzero(component_postfill_rejected))}"
        )

    pred, obs_residual, obs_residual_valid, point_residual, point_residual_valid = (
        compute_prediction_and_residuals(prepared, alpha, final_phi)
    )
    operator = _build_observation_operator(prepared, alpha)
    observed_mask = np.zeros((num_points,), dtype=bool)
    positive_rows = operator.weight > 0.0
    observed_mask[np.unique(operator.point_index[positive_rows])] = True
    usable_observed_mask = np.zeros((num_points,), dtype=bool)
    usable_observed_mask[np.unique(operator.point_index[operator.valid_rows])] = True
    fill_target_mask = selected_point_mask

    active_mask = trusted_seed_mask | selected_point_mask
    connected_to_anchor = trusted_seed_mask | completion_mask
    unconnected_groups = np.flatnonzero(~effective_connected_groups)
    active_component_count = 1 + int(unconnected_groups.size)
    active_component_index = np.full((num_points,), -1, dtype=np.int32)
    active_component_index[connected_to_anchor] = 0
    for active_idx, group_idx in enumerate(unconnected_groups.tolist(), start=1):
        members = membership.ordered_points[
            membership.offsets[group_idx] : membership.offsets[group_idx + 1]
        ]
        active_component_index[members] = active_idx
    active_component_sizes = np.bincount(
        active_component_index[active_mask], minlength=active_component_count
    ).astype(np.int32)
    active_component_has_anchor = np.zeros(
        (active_component_count,), dtype=bool
    )
    active_component_has_anchor[0] = True
    active_component_anchor_count = np.zeros(
        (active_component_count,), dtype=np.int32
    )
    active_component_anchor_count[0] = int(np.count_nonzero(trusted_seed_mask))
    hop_distance = np.full((num_points,), -1, dtype=np.int32)
    hop_distance[trusted_seed_mask] = 0
    if membership.ordered_points.size:
        member_groups = shared_group_index[membership.ordered_points]
        connected_members = effective_connected_groups[member_groups]
        hop_distance[membership.ordered_points[connected_members]] = group_hop[
            member_groups[connected_members]
        ]
    connectivity = AnchorConnectivity(
        connected_to_anchor=connected_to_anchor,
        active_mask=active_mask,
        component_index=active_component_index,
        component_sizes=active_component_sizes,
        component_has_anchor=active_component_has_anchor,
        component_anchor_count=active_component_anchor_count,
        hop_distance=hop_distance,
    )
    point_group_index = np.full((num_points,), -1, dtype=np.int32)
    point_group_index[selected_point_mask] = shared_group_index[selected_point_mask]
    motion = MotionFillResult(
        phi=final_phi.astype(np.complex64),
        phi_observable=phi_observable.astype(np.complex64),
        phi_nullspace_correction=correction.astype(np.complex64),
        completion_mask=completion_mask,
        completion_connected_to_anchor=connected_to_anchor,
        coefficient_values=coefficient_values,
        coefficient_offsets=coefficient_offsets,
        connectivity=connectivity,
        real_solver=real_solver,
        imag_solver=imag_solver,
        system_row_count=row_count,
        system_column_count=column_count,
        active_edge_count=int(active_edges.shape[0]),
        point_coefficient_group_index=point_group_index,
        coefficient_group_dimensions=group_dimensions,
    )
    numerical_nullity = np.full((num_points,), 3, dtype=np.int8)
    numerical_nullity[trusted_seed_mask] = 0
    numerical_nullity[selected_point_mask] = fill_nullity[
        point_component[selected_point_mask]
    ]
    partial_diagnostics = RigidSingleViewPartialDiagnostics(
        observable_singular_ratio_min=float(observable_singular_ratio_min),
        ray_direction_min_fraction=float(ray_direction_min_fraction),
        max_finite_drift=float(max_finite_drift),
        component_observable_rank=observable_rank,
        component_fill_nullity=fill_nullity,
        component_ray_dominated_basis_count=ray_dominated_count,
        component_trusted_knn_edge_count=trusted_knn_count,
        component_cross_knn_edge_count=cross_knn_count,
        component_connected_to_trusted_mask=component_connected,
        component_postfill_normalized_residual=postfill_residual.astype(np.float32),
        component_ray_motion_rms=ray_motion_rms.astype(np.float32),
        component_tangent_motion_rms=tangent_motion_rms.astype(np.float32),
        component_ray_motion_ratio=ray_motion_ratio.astype(np.float32),
        component_finite_drift_max=component_finite_drift.astype(np.float32),
        component_finite_drift_rejected_mask=(
            component_finite_drift_rejected
        ),
        component_postfill_retained_mask=component_postfill_retained,
    )
    diagnostics = {
        "method": RIGID_SINGLE_VIEW_PARTIAL_FILL_METHOD,
        "version": MOTION_FILL_VERSION,
        "graph_path": graph_path,
        "single_view_component_fill": {
            "policy": RIGID_SINGLE_VIEW_PARTIAL_FILL_POLICY,
            "component_count": int(selected_components.size),
            "completed_component_count": int(np.count_nonzero(component_completion)),
            "unresolved_component_count": int(
                np.count_nonzero(single_view_component_fill_mask & ~component_completion)
            ),
            "observable_singular_ratio_min": float(observable_singular_ratio_min),
            "ray_direction_min_fraction": float(ray_direction_min_fraction),
            "max_finite_drift": float(max_finite_drift),
            "finite_drift_rejected_component_count": int(
                np.count_nonzero(component_finite_drift_rejected)
            ),
            "postfill_rejected_component_count": int(
                np.count_nonzero(component_postfill_rejected)
            ),
        },
        "system": {
            "row_count": row_count,
            "column_count": column_count,
            "active_edge_count": int(active_edges.shape[0]),
            "eligible_edge_count": int(eligible_indices.size),
        },
        "lsmr_real": _solver_diagnostics(real_solver),
        "lsmr_imaginary": _solver_diagnostics(imag_solver),
    }
    result = RigidSeedMotionFillResult(
        motion=motion,
        roles=roles,
        numerical_nullity=numerical_nullity,
        observed_mask=observed_mask,
        usable_observed_mask=usable_observed_mask,
        fill_target_mask=fill_target_mask,
        single_view_component_fill_mask=single_view_component_fill_mask,
        single_view_rigid_fill_point_mask=selected_point_mask,
        single_view_component_completion_mask=component_completion,
        single_view_component_translation=component_translation.astype(np.complex64),
        single_view_component_rotation=component_rotation.astype(np.complex64),
        single_view_component_first_order_relative_max=(
            component_first_order_relative_max.astype(np.float32)
        ),
        obs_pred_y=pred,
        obs_residual=obs_residual,
        obs_residual_valid_mask=obs_residual_valid,
        point_residual=point_residual,
        point_residual_valid_mask=point_residual_valid,
        diagnostics=diagnostics,
        single_view_partial_diagnostics=partial_diagnostics,
    )
    if timings is not None:
        timings["validation_and_residual_seconds"] = float(
            perf_counter() - validation_started
        )
        timings["total_seconds"] = float(perf_counter() - total_started)
    if progress is not None:
        progress(
            "single-view component reconstruction and validation finished in "
            f"{perf_counter() - validation_started:.3f} s"
        )
    return result


def apply_sequential_rigid_motion_fill(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    rigid: RigidComponentSolveResult,
    seed_selection: RigidComponentSeedSelectionResult,
    observed_graph: ObservedStructureGraph,
    graph: KnnGraph,
    graph_path: str,
    *,
    observable_singular_ratio_min: float,
    ray_direction_min_fraction: float,
    max_finite_drift: float,
    timings: dict[str, float] | None = None,
    progress: Callable[[str], None] | None = None,
) -> RigidSeedMotionFillResult:
    """Fill trusted single-view components, then independent Gaussian vectors."""

    total_started = perf_counter()
    component_timings: dict[str, float] = {}
    component_result = apply_single_view_component_partial_fill(
        prepared,
        alpha,
        rigid,
        seed_selection,
        observed_graph,
        graph,
        graph_path,
        observable_singular_ratio_min=observable_singular_ratio_min,
        ray_direction_min_fraction=ray_direction_min_fraction,
        max_finite_drift=max_finite_drift,
        timings=component_timings,
        progress=progress,
    )

    num_points = int(prepared.points.shape[0])
    trusted_seed_mask = np.asarray(
        seed_selection.trusted_rigid_seed_mask,
        dtype=bool,
    )
    component_anchor_mask = np.asarray(
        component_result.motion.completion_mask,
        dtype=bool,
    )
    if (
        trusted_seed_mask.shape != (num_points,)
        or component_anchor_mask.shape != (num_points,)
        or np.any(trusted_seed_mask & component_anchor_mask)
    ):
        raise RuntimeError("Sequential rigid motion-fill anchor masks are invalid")
    fixed_anchor_mask = trusted_seed_mask | component_anchor_mask
    if not np.any(fixed_anchor_mask):
        raise ValueError("Sequential rigid motion fill requires at least one anchor")

    phi_fixed = np.zeros((num_points, 3), dtype=np.complex64)
    phi_fixed[fixed_anchor_mask] = component_result.motion.phi[fixed_anchor_mask]
    point_nullspace_basis = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (num_points, 3, 3),
    )
    point_nullity = np.full((num_points,), 3, dtype=np.int8)
    point_nullity[fixed_anchor_mask] = 0
    point_target_mask = ~fixed_anchor_mask
    empty_mask = np.zeros((num_points,), dtype=bool)
    roles = derive_rigid_seed_motion_fill_roles(
        fixed_anchor_mask,
        empty_mask,
    )

    pointwise_timings: dict[str, float] = {}
    point_motion = fill_nullspace_motion(
        graph,
        phi_fixed,
        point_nullspace_basis,
        point_nullity,
        fixed_anchor_mask,
        empty_mask,
        point_target_mask,
        excluded_mask=empty_mask,
        lsmr_atol=MOTION_FILL_LSMR_ATOL,
        lsmr_btol=MOTION_FILL_LSMR_BTOL,
        lsmr_conlim=MOTION_FILL_LSMR_CONLIM,
        timings=pointwise_timings,
        progress=progress,
    )
    if not np.array_equal(
        point_motion.phi[fixed_anchor_mask],
        phi_fixed[fixed_anchor_mask],
    ):
        raise RuntimeError("Pointwise motion fill changed a rigid-component anchor")

    overall_completion = (
        point_motion.completion_mask | component_anchor_mask
    )
    motion = MotionFillResult(
        phi=point_motion.phi,
        phi_observable=point_motion.phi_observable,
        phi_nullspace_correction=point_motion.phi_nullspace_correction,
        completion_mask=overall_completion,
        completion_connected_to_anchor=(
            point_motion.completion_connected_to_anchor
        ),
        coefficient_values=point_motion.coefficient_values,
        coefficient_offsets=point_motion.coefficient_offsets,
        connectivity=point_motion.connectivity,
        real_solver=point_motion.real_solver,
        imag_solver=point_motion.imag_solver,
        system_row_count=point_motion.system_row_count,
        system_column_count=point_motion.system_column_count,
        active_edge_count=point_motion.active_edge_count,
        point_coefficient_group_index=None,
        coefficient_group_dimensions=None,
    )

    validation_started = perf_counter()
    if progress is not None:
        progress("sequential validation and residuals started")
    pred, obs_residual, obs_residual_valid, point_residual, point_residual_valid = (
        compute_prediction_and_residuals(prepared, alpha, motion.phi)
    )
    observed_mask = component_result.observed_mask
    completed_observed = motion.completion_mask & ~trusted_seed_mask & observed_mask
    completed_unobserved = motion.completion_mask & ~trusted_seed_mask & ~observed_mask
    unresolved_target = ~trusted_seed_mask & ~motion.completion_mask
    diagnostics = {
        "method": RIGID_SEQUENTIAL_MOTION_FILL_METHOD,
        "version": MOTION_FILL_VERSION,
        "graph_path": graph_path,
        "pipeline": (
            "single_view_partial_components_then_independent_3d_gaussians"
        ),
        "component_stage": component_result.diagnostics,
        "role_counts": {
            name: int(np.count_nonzero(roles.role == index))
            for index, name in enumerate(MOTION_FILL_ROLE_NAMES)
        },
        "completion": {
            "component_anchor_point_count": int(
                np.count_nonzero(component_anchor_mask)
            ),
            "pointwise_target_count": int(np.count_nonzero(point_target_mask)),
            "completed_observed_count": int(np.count_nonzero(completed_observed)),
            "completed_unobserved_count": int(
                np.count_nonzero(completed_unobserved)
            ),
            "unresolved_target_count": int(np.count_nonzero(unresolved_target)),
        },
        "system": {
            "row_count": int(motion.system_row_count),
            "column_count": int(motion.system_column_count),
            "active_edge_count": int(motion.active_edge_count),
            "eligible_edge_count": int(graph.edge_index.shape[0]),
        },
        "lsmr_real": _solver_diagnostics(motion.real_solver),
        "lsmr_imaginary": _solver_diagnostics(motion.imag_solver),
    }
    result = RigidSeedMotionFillResult(
        motion=motion,
        roles=roles,
        numerical_nullity=point_nullity,
        observed_mask=observed_mask,
        usable_observed_mask=component_result.usable_observed_mask,
        fill_target_mask=~trusted_seed_mask,
        single_view_component_fill_mask=(
            component_result.single_view_component_fill_mask
        ),
        single_view_rigid_fill_point_mask=(
            component_result.single_view_rigid_fill_point_mask
        ),
        single_view_component_completion_mask=(
            component_result.single_view_component_completion_mask
        ),
        single_view_component_translation=(
            component_result.single_view_component_translation
        ),
        single_view_component_rotation=(
            component_result.single_view_component_rotation
        ),
        single_view_component_first_order_relative_max=(
            component_result.single_view_component_first_order_relative_max
        ),
        obs_pred_y=pred,
        obs_residual=obs_residual,
        obs_residual_valid_mask=obs_residual_valid,
        point_residual=point_residual,
        point_residual_valid_mask=point_residual_valid,
        diagnostics=diagnostics,
        single_view_partial_diagnostics=(
            component_result.single_view_partial_diagnostics
        ),
    )
    validation_seconds = float(perf_counter() - validation_started)
    if timings is not None:
        for name, seconds in component_timings.items():
            timings[f"component_{name}"] = float(seconds)
        for name, seconds in pointwise_timings.items():
            timings[f"pointwise_{name}"] = float(seconds)
        timings["validation_and_residual_seconds"] = validation_seconds
        timings["total_seconds"] = float(perf_counter() - total_started)
    if progress is not None:
        progress(
            "sequential validation and residuals finished in "
            f"{validation_seconds:.3f} s"
        )
    return result


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
