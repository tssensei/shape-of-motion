"""Staged multi-view modal solve without alpha/phi alternation.

The staged solver implemented here intentionally stops after two operations:

1. synchronize per-view modal phase/gain from multi-view overlap constraints;
2. recover each point's observable 3D component and classify reliable anchors.

Rank-deficient points are left at their truncated-SVD minimum-norm solution.
No spatial/nullspace completion is performed in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_EPS = 1e-12
_ANCHOR_CONDITION_MAX = 100.0
POINT_STATUS_ANCHOR = 0
POINT_STATUS_COMPLETED_OBSERVED = 1
POINT_STATUS_COMPLETED_UNOBSERVED = 2
POINT_STATUS_PARTIAL_UNRESOLVED = 3
POINT_STATUS_UNOBSERVED_UNRESOLVED = 4
POINT_STATUS_REJECTED_UNRESOLVED = 5
POINT_STATUS_ALPHA_UNRESOLVED = 6
POINT_STATUS_NO_USABLE_OBSERVATION = 7
POINT_STATUS_NAMES = (
    "anchor",
    "completed_observed",
    "completed_unobserved",
    "partial_unresolved",
    "unobserved_unresolved",
    "rejected_unresolved",
    "alpha_unresolved",
    "no_usable_observation",
)


@dataclass(frozen=True)
class StagedSolverConfig:
    alpha_model: str = "phase"
    alpha_gain_min: float = 0.25
    alpha_gain_max: float = 4.0
    alpha_min_shared_points: int = 16
    alpha_rank_ratio_min: float = 1e-4
    alpha_info_ratio_min: float = 1e-4
    alpha_failure: str = "exclude"
    anchor_svd_ratio_min: float = 1e-2
    anchor_residual_max: float = 0.1

    def validate(self) -> None:
        if self.alpha_model not in {"phase", "bounded-complex"}:
            raise ValueError("alpha_model must be 'phase' or 'bounded-complex'.")
        if not (0.0 < self.alpha_gain_min <= 1.0 <= self.alpha_gain_max):
            raise ValueError("alpha gain bounds must satisfy 0 < min <= 1 <= max.")
        if self.alpha_min_shared_points <= 0:
            raise ValueError("alpha_min_shared_points must be positive.")
        if not (0.0 < self.alpha_rank_ratio_min <= 1.0):
            raise ValueError("alpha_rank_ratio_min must be in (0,1].")
        if not (0.0 < self.alpha_info_ratio_min <= 1.0):
            raise ValueError("alpha_info_ratio_min must be in (0,1].")
        if self.alpha_failure not in {"exclude", "error"}:
            raise ValueError("alpha_failure must be 'exclude' or 'error'.")
        if not (0.0 < self.anchor_svd_ratio_min <= 1.0):
            raise ValueError("anchor_svd_ratio_min must be in (0,1].")
        if self.anchor_residual_max <= 0.0:
            raise ValueError("anchor_residual_max must be positive.")


@dataclass
class PreparedObservations:
    arrays: dict[str, np.ndarray]
    points: np.ndarray
    obs_point_index: np.ndarray
    obs_view_index: np.ndarray
    obs_pixels_xy: np.ndarray
    obs_y: np.ndarray
    obs_J: np.ndarray
    obs_confidence: np.ndarray
    obs_weights: np.ndarray
    obs_count_per_point: np.ndarray
    obs_sample_count_per_point: np.ndarray
    derived_view_count_per_point: np.ndarray
    rows_by_point: list[np.ndarray]
    view_ids: np.ndarray
    num_views: int


@dataclass
class AlphaConstraint:
    point_index: int
    rows: np.ndarray
    views: np.ndarray
    C: np.ndarray
    observation_energy: float


@dataclass
class AlphaComponentSolve:
    alphas: np.ndarray
    consistency_residual: float
    singular_values: np.ndarray
    rank_ratio: float
    information_ratio: float
    condition: float
    phase_std: np.ndarray
    log_gain_std: np.ndarray
    constraint_information: np.ndarray
    parameter_information: np.ndarray
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    gain_bound_active_mask: np.ndarray
    information_kind: str


@dataclass
class AlphaSyncResult:
    alphas: np.ndarray
    identifiable_mask: np.ndarray
    component_index: np.ndarray
    structural_component_index: np.ndarray
    shared_count_component_index: np.ndarray
    information_component_index: np.ndarray
    exclusion_reason: np.ndarray
    shared_point_count: np.ndarray
    edge_point_count: np.ndarray
    edge_information: np.ndarray
    constraint_count_per_view: np.ndarray
    information_matrix: np.ndarray
    singular_values: np.ndarray
    rank_ratio: float
    information_ratio: float
    condition: float
    consistency_residual: float
    phase_std: np.ndarray
    log_gain_std: np.ndarray
    parameter_information: np.ndarray
    parameter_view_indices: np.ndarray
    parameter_order: str
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    gain_bound_active_mask: np.ndarray
    information_kind: str


@dataclass
class ObservableSolveResult:
    phi: np.ndarray
    phi_observable: np.ndarray
    nullspace_basis: np.ndarray
    nullity: np.ndarray
    singular_values: np.ndarray
    observable_rank: np.ndarray
    condition: np.ndarray
    distinct_valid_view_count: np.ndarray
    usable_observation_row_count: np.ndarray
    precompletion_residual: np.ndarray
    anchor_residual_threshold: float
    anchor_mask: np.ndarray
    partial_mask: np.ndarray
    rejected_mask: np.ndarray
    unobserved_mask: np.ndarray
    alpha_unresolved_mask: np.ndarray
    no_usable_observation_mask: np.ndarray
    point_status: np.ndarray


def _rows_by_point(num_points: int, point_index: np.ndarray) -> list[np.ndarray]:
    rows: list[list[int]] = [[] for _ in range(num_points)]
    for row, point in enumerate(point_index.tolist()):
        rows[int(point)].append(row)
    return [np.asarray(value, dtype=np.int64) for value in rows]


def prepare_observations(data: Mapping[str, np.ndarray]) -> PreparedObservations:
    required = (
        "points_world",
        "obs_point_index",
        "obs_view_index",
        "obs_pixels_xy",
        "obs_y",
        "obs_J",
        "obs_confidence",
        "obs_count_per_point",
        "view_ids",
        "freq_hz",
        "mode_index",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Observation graph is missing required arrays: {missing}.")

    arrays = {key: np.asarray(value) for key, value in data.items()}
    points = arrays["points_world"].astype(np.float32)
    point_index = arrays["obs_point_index"].astype(np.int64)
    view_index = arrays["obs_view_index"].astype(np.int64)
    pixels = arrays["obs_pixels_xy"].astype(np.float32)
    obs_y = arrays["obs_y"].astype(np.complex64)
    obs_J = arrays["obs_J"].astype(np.float32)
    confidence = arrays["obs_confidence"].astype(np.float32)
    obs_count = arrays["obs_count_per_point"].astype(np.int32)
    sample_count = (
        arrays["obs_sample_count_per_point"].astype(np.int32)
        if "obs_sample_count_per_point" in arrays
        else np.bincount(point_index, minlength=points.shape[0]).astype(np.int32)
    )
    view_ids = arrays["view_ids"]
    if view_ids.ndim != 1 or view_ids.size == 0:
        raise ValueError(f"view_ids must be a non-empty 1-D array, got {view_ids.shape}.")
    num_views = int(view_ids.shape[0])
    num_obs = int(obs_y.shape[0])

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points.shape}.")
    if point_index.shape != (num_obs,) or view_index.shape != (num_obs,):
        raise ValueError("Observation point/view index arrays must match obs_y length.")
    if pixels.shape != (num_obs, 2) or obs_y.shape != (num_obs, 2):
        raise ValueError("obs_pixels_xy and obs_y must have shape (O,2).")
    if obs_J.shape != (num_obs, 2, 3):
        raise ValueError(f"obs_J must have shape ({num_obs},2,3), got {obs_J.shape}.")
    if confidence.shape != (num_obs,):
        raise ValueError(f"obs_confidence must have shape ({num_obs},), got {confidence.shape}.")
    if obs_count.shape != (points.shape[0],) or sample_count.shape != (points.shape[0],):
        raise ValueError("Per-point observation count arrays must match points_world length.")
    if np.any(point_index < 0) or np.any(point_index >= points.shape[0]):
        raise ValueError("obs_point_index contains invalid point indices.")
    if np.any(view_index < 0) or np.any(view_index >= num_views):
        raise ValueError("obs_view_index contains invalid view indices.")
    if np.any(~np.isfinite(confidence)) or np.any(confidence < 0):
        raise ValueError("obs_confidence must be finite and non-negative.")
    if np.any(~np.isfinite(points)):
        raise ValueError("points_world must be finite.")
    if np.any(~np.isfinite(obs_J)):
        raise ValueError("obs_J must be finite.")
    if np.any(~np.isfinite(np.real(obs_y))) or np.any(~np.isfinite(np.imag(obs_y))):
        raise ValueError("obs_y must be finite.")
    if np.any(obs_count < 0) or np.any(sample_count < 0):
        raise ValueError("Per-point observation counts must be non-negative.")
    if np.unique(view_ids.astype(str)).size != num_views:
        raise ValueError("view_ids must be unique.")
    if np.asarray(arrays["freq_hz"]).size != 1 or np.asarray(arrays["mode_index"]).size != 1:
        raise ValueError("freq_hz and mode_index must be scalar arrays.")
    if "view_freqs_hz" in arrays and arrays["view_freqs_hz"].shape != (num_views,):
        raise ValueError(f"view_freqs_hz must have shape ({num_views},).")
    for key in ("view_image_width", "view_image_height"):
        if key in arrays and arrays[key].shape != (num_views,):
            raise ValueError(f"{key} must have shape ({num_views},).")
    for key in (
        "colors",
        "gaussian_indices",
    ):
        if key in arrays and arrays[key].shape[:1] != (points.shape[0],):
            raise ValueError(f"{key} must have first dimension {points.shape[0]}.")
    if "gaussian_indices" in arrays and arrays["gaussian_indices"].shape != (points.shape[0],):
        raise ValueError(f"gaussian_indices must have shape ({points.shape[0]},).")
    if "colors" in arrays and arrays["colors"].shape != (points.shape[0], 3):
        raise ValueError(f"colors must have shape ({points.shape[0]},3).")

    point_view_mask = np.zeros((points.shape[0], num_views), dtype=bool)
    point_view_mask[point_index, view_index] = True
    derived_count = point_view_mask.sum(axis=1).astype(np.int32)
    if not np.array_equal(derived_count, obs_count):
        bad = int(np.count_nonzero(derived_count != obs_count))
        raise ValueError(
            f"obs_count_per_point disagrees with unique observation views for {bad} points."
        )
    derived_sample_count = np.bincount(point_index, minlength=points.shape[0]).astype(np.int32)
    if not np.array_equal(derived_sample_count, sample_count):
        bad = int(np.count_nonzero(derived_sample_count != sample_count))
        raise ValueError(
            f"obs_sample_count_per_point disagrees with observation rows for {bad} points."
        )

    pair_key = point_index * max(num_views, 1) + view_index
    _, inverse, counts = np.unique(pair_key, return_inverse=True, return_counts=True)
    row_multiplicity = counts[inverse].astype(np.float64)
    weights = confidence.astype(np.float64) / np.maximum(row_multiplicity, 1.0)

    return PreparedObservations(
        arrays=arrays,
        points=points,
        obs_point_index=point_index,
        obs_view_index=view_index,
        obs_pixels_xy=pixels,
        obs_y=obs_y,
        obs_J=obs_J,
        obs_confidence=confidence,
        obs_weights=weights,
        obs_count_per_point=obs_count,
        obs_sample_count_per_point=sample_count,
        derived_view_count_per_point=derived_count,
        rows_by_point=_rows_by_point(points.shape[0], point_index),
        view_ids=view_ids,
        num_views=num_views,
    )


def _build_alpha_constraints(
    prepared: PreparedObservations,
    allowed_views: np.ndarray | None = None,
) -> list[AlphaConstraint]:
    constraints: list[AlphaConstraint] = []
    if allowed_views is None:
        allowed_views = np.ones((prepared.num_views,), dtype=bool)
    for point_idx, point_rows in enumerate(prepared.rows_by_point):
        if point_rows.size == 0:
            continue
        keep = allowed_views[prepared.obs_view_index[point_rows]] & (prepared.obs_weights[point_rows] > 0)
        rows = point_rows[keep]
        views = np.unique(prepared.obs_view_index[rows])
        if views.size < 2:
            continue

        sqrt_w = np.sqrt(prepared.obs_weights[rows])
        A = (sqrt_w[:, None, None] * prepared.obs_J[rows].astype(np.float64)).reshape(-1, 3)
        B = np.zeros((rows.size * 2, prepared.num_views), dtype=np.complex128)
        for local_row, obs_row in enumerate(rows.tolist()):
            view_idx = int(prepared.obs_view_index[obs_row])
            B[2 * local_row : 2 * local_row + 2, view_idx] = (
                sqrt_w[local_row] * prepared.obs_y[obs_row].astype(np.complex128)
            )

        U, singular, _ = np.linalg.svd(A, full_matrices=True)
        if singular.size == 0 or singular[0] <= _EPS:
            continue
        rank = int(np.count_nonzero(singular > 1e-8 * singular[0]))
        left_null = U[:, rank:]
        if left_null.shape[1] == 0:
            continue
        C = left_null.conj().T @ B
        if float(np.linalg.norm(C)) <= _EPS:
            # Keep exact/near-exact zero constraints out of the numerical solve,
            # but edge accounting below will still classify this geometry as weak.
            pass
        constraints.append(
            AlphaConstraint(
                point_index=point_idx,
                rows=rows,
                views=views,
                C=C,
                observation_energy=float(np.linalg.norm(B)),
            )
        )
    return constraints


def _alpha_edge_statistics(
    constraints: list[AlphaConstraint],
    num_views: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((num_views, num_views), dtype=np.int32)
    information = np.zeros((num_views, num_views), dtype=np.float64)
    for constraint in constraints:
        energy = max(constraint.observation_energy, _EPS)
        H = (constraint.C.conj().T @ constraint.C) / (energy * energy)
        scale = max(float(np.real(np.trace(H))), _EPS)
        for i, view_a in enumerate(constraint.views.tolist()):
            for view_b in constraint.views[i + 1 :].tolist():
                value = float(np.abs(H[int(view_a), int(view_b)]))
                if value > 1e-12 * scale:
                    counts[int(view_a), int(view_b)] += 1
                    counts[int(view_b), int(view_a)] += 1
                    information[int(view_a), int(view_b)] += value
                    information[int(view_b), int(view_a)] += value
    return counts, information


def _raw_shared_point_counts(prepared: PreparedObservations) -> np.ndarray:
    counts = np.zeros((prepared.num_views, prepared.num_views), dtype=np.int32)
    for point_rows in prepared.rows_by_point:
        positive_rows = point_rows[prepared.obs_weights[point_rows] > 0.0]
        views = np.unique(prepared.obs_view_index[positive_rows])
        for local, view_a in enumerate(views.tolist()):
            for view_b in views[local + 1 :].tolist():
                counts[int(view_a), int(view_b)] += 1
                counts[int(view_b), int(view_a)] += 1
    return counts


def _graph_components(adjacency: np.ndarray) -> np.ndarray:
    num_views = adjacency.shape[0]
    components = np.full((num_views,), -1, dtype=np.int32)
    component = 0
    for start in range(num_views):
        if components[start] >= 0:
            continue
        queue = [start]
        components[start] = component
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            for neighbor in np.where(adjacency[node])[0].tolist():
                if components[neighbor] >= 0:
                    continue
                components[neighbor] = component
                queue.append(int(neighbor))
        component += 1
    return components


def _reference_connected_subset(
    adjacency: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    allowed = np.zeros((adjacency.shape[0],), dtype=bool)
    allowed[candidate] = True
    connected = np.zeros_like(allowed)
    if not allowed[0]:
        return np.asarray([], dtype=np.int64)
    queue = [0]
    connected[0] = True
    cursor = 0
    while cursor < len(queue):
        view_idx = queue[cursor]
        cursor += 1
        for neighbor in np.where(adjacency[view_idx] & allowed)[0].tolist():
            if connected[neighbor]:
                continue
            connected[neighbor] = True
            queue.append(int(neighbor))
    return np.where(connected)[0].astype(np.int64)


def _constraint_information_blocks(
    constraints: list[AlphaConstraint],
    views: np.ndarray,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for constraint in constraints:
        energy = max(constraint.observation_energy, _EPS)
        C = constraint.C[:, views] / energy
        blocks.append(C.conj().T @ C)
    if not blocks:
        return np.zeros((0, views.size, views.size), dtype=np.complex128)
    return np.stack(blocks, axis=0).astype(np.complex128)


def _phase_initialization(H: np.ndarray, reference_local: int) -> np.ndarray:
    _, vectors = np.linalg.eigh(H)
    beta = vectors[:, 0].astype(np.complex128)
    ref = beta[reference_local]
    if abs(ref) <= _EPS:
        beta = np.ones((H.shape[0],), dtype=np.complex128)
    else:
        beta = beta / ref
    beta = np.exp(1j * np.angle(beta))
    beta[reference_local] = 1.0 + 0.0j
    return beta


def _complex_initialization(H: np.ndarray, reference_local: int) -> np.ndarray:
    beta = np.ones((H.shape[0],), dtype=np.complex128)
    unknown = np.asarray([i for i in range(H.shape[0]) if i != reference_local], dtype=np.int64)
    if unknown.size:
        H_unknown = H[np.ix_(unknown, unknown)]
        rhs = -H[unknown, reference_local]
        solution = np.linalg.lstsq(H_unknown, rhs, rcond=None)[0]
        beta[unknown] = solution
    beta[reference_local] = 1.0 + 0.0j
    return beta


def _parameterized_beta(
    parameters: np.ndarray,
    num_component_views: int,
    reference_local: int,
    model: str,
) -> np.ndarray:
    beta = np.ones((num_component_views,), dtype=np.complex128)
    unknown = [i for i in range(num_component_views) if i != reference_local]
    count = len(unknown)
    theta = parameters[:count]
    if model == "phase":
        values = np.exp(-1j * theta)
    else:
        gains = parameters[count:]
        values = np.exp(-gains - 1j * theta)
    beta[np.asarray(unknown, dtype=np.int64)] = values
    return beta


def _alpha_parameter_derivatives(
    beta: np.ndarray,
    reference_local: int,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    unknown = np.asarray(
        [i for i in range(beta.size) if i != reference_local], dtype=np.int64
    )
    count = int(unknown.size)
    if count == 0:
        return unknown, np.zeros((beta.size, 0), dtype=np.complex128)
    parameter_count = count if model == "phase" else 2 * count
    derivatives = np.zeros((beta.size, parameter_count), dtype=np.complex128)
    for local, view_idx in enumerate(unknown.tolist()):
        derivatives[view_idx, local] = -1j * beta[view_idx]
        if model == "bounded-complex":
            derivatives[view_idx, count + local] = -beta[view_idx]
    return unknown, derivatives


def _alpha_parameter_gradient(
    H: np.ndarray,
    beta: np.ndarray,
    reference_local: int,
    model: str,
) -> np.ndarray:
    _, derivatives = _alpha_parameter_derivatives(beta, reference_local, model)
    if derivatives.shape[1] == 0:
        return np.zeros((0,), dtype=np.float64)
    return (2.0 * np.real(derivatives.conj().T @ H @ beta)).astype(np.float64)


def _alpha_parameter_information(
    H: np.ndarray,
    beta: np.ndarray,
    reference_local: int,
    model: str,
) -> np.ndarray:
    unknown, derivatives = _alpha_parameter_derivatives(beta, reference_local, model)
    count = int(unknown.size)
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)

    if model == "phase":
        # This is the exact phase Hessian of beta^H H beta.  Writing it from
        # off-diagonal cross-view terms prevents diagonal residual energy from
        # being mistaken for phase information when the data are inconsistent.
        cross = 2.0 * np.real(
            np.conj(beta)[:, None] * H * beta[None, :]
        ).astype(np.float64)
        np.fill_diagonal(cross, 0.0)
        np.fill_diagonal(cross, -np.sum(cross, axis=1))
        information = cross[np.ix_(unknown, unknown)]
        return 0.5 * (information + information.T)

    information = 2.0 * np.real(derivatives.conj().T @ H @ derivatives)

    # Add the exact second-derivative terms.  They matter away from a zero
    # residual and stop within-view residual energy from creating false phase
    # curvature in the bounded-gain model as well.
    H_beta = H @ beta
    for local, view_idx in enumerate(unknown.tolist()):
        second_theta = -beta[view_idx]
        second_gain = beta[view_idx]
        second_cross = 1j * beta[view_idx]
        information[local, local] += 2.0 * np.real(
            np.conj(second_theta) * H_beta[view_idx]
        )
        information[count + local, count + local] += 2.0 * np.real(
            np.conj(second_gain) * H_beta[view_idx]
        )
        cross_value = 2.0 * np.real(np.conj(second_cross) * H_beta[view_idx])
        information[local, count + local] += cross_value
        information[count + local, local] += cross_value
    return 0.5 * (information + information.T)


def _huber_parameter_information(
    information_blocks: np.ndarray,
    beta: np.ndarray,
    reference_local: int,
    model: str,
    block_residuals: np.ndarray,
    f_scale: float,
) -> np.ndarray:
    parameter_count = (beta.size - 1) * (2 if model == "bounded-complex" else 1)
    information = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    for block, block_residual in zip(information_blocks, block_residuals, strict=True):
        block_hessian = _alpha_parameter_information(
            block, beta, reference_local, model
        )
        if block_residual <= f_scale:
            information += 0.5 * block_hessian
            continue
        block_gradient = _alpha_parameter_gradient(
            block, beta, reference_local, model
        )
        residual_safe = max(float(block_residual), _EPS)
        information += 0.5 * (f_scale / residual_safe) * block_hessian
        information -= (f_scale / (4.0 * residual_safe**3)) * np.outer(
            block_gradient, block_gradient
        )
    return 0.5 * (information + information.T)


def _block_residuals(information_blocks: np.ndarray, beta: np.ndarray) -> np.ndarray:
    squared = np.real(
        np.einsum("i,bij,j->b", np.conj(beta), information_blocks, beta)
    )
    return np.sqrt(np.maximum(squared, 0.0)).astype(np.float64)


def _profiled_observation_residual_blocks(
    prepared: PreparedObservations,
    constraints: list[AlphaConstraint],
    component_views: np.ndarray,
    parameters: np.ndarray,
    reference_local: int,
) -> list[np.ndarray]:
    beta = _parameterized_beta(
        parameters, component_views.size, reference_local, "bounded-complex"
    )
    alpha = 1.0 / beta
    global_to_local = np.full((prepared.num_views,), -1, dtype=np.int64)
    global_to_local[component_views] = np.arange(component_views.size, dtype=np.int64)
    blocks: list[np.ndarray] = []
    for constraint in constraints:
        rows = constraint.rows
        sqrt_w = np.sqrt(prepared.obs_weights[rows])
        local_views = global_to_local[prepared.obs_view_index[rows]]
        alpha_rows = alpha[local_views]
        A = (
            sqrt_w[:, None, None]
            * alpha_rows[:, None, None]
            * prepared.obs_J[rows].astype(np.complex128)
        ).reshape(-1, 3)
        b = (sqrt_w[:, None] * prepared.obs_y[rows].astype(np.complex128)).reshape(-1)
        point_phi = np.linalg.lstsq(A, b, rcond=None)[0]
        normalization = max(float(np.linalg.norm(b)), _EPS)
        blocks.append((A @ point_phi - b) / normalization)
    return blocks


def _flatten_weighted_complex_blocks(
    blocks: list[np.ndarray],
    block_weights: np.ndarray,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for block, weight in zip(blocks, block_weights, strict=True):
        scale = np.sqrt(max(float(weight), 0.0))
        pieces.append(scale * np.concatenate([np.real(block), np.imag(block)]))
    return np.concatenate(pieces).astype(np.float64) if pieces else np.zeros((0,), dtype=np.float64)


def _information_summary(
    parameter_information: np.ndarray,
    constraint_count: int,
) -> tuple[np.ndarray, float, float, float]:
    eigenvalues = (
        np.linalg.eigvalsh(0.5 * (parameter_information + parameter_information.T))
        if parameter_information.size
        else np.zeros((0,), dtype=np.float64)
    )
    singular = np.sqrt(np.maximum(eigenvalues, 0.0))[::-1]
    if not singular.size:
        return singular, 1.0, 0.0, 1.0
    rank_ratio = float(singular[-1] / max(singular[0], _EPS))
    information_ratio = float(singular[-1] / np.sqrt(max(constraint_count, 1)))
    condition = float(singular[0] / max(singular[-1], _EPS))
    return singular, rank_ratio, information_ratio, condition


def _refine_alpha_component(
    prepared: PreparedObservations,
    constraints: list[AlphaConstraint],
    component_views: np.ndarray,
    config: StagedSolverConfig,
) -> AlphaComponentSolve:
    reference_local = int(np.where(component_views == 0)[0][0])
    information_blocks = _constraint_information_blocks(constraints, component_views)
    aggregate = np.sum(information_blocks, axis=0)
    if config.alpha_model == "phase":
        beta_init = _phase_initialization(aggregate, reference_local)
    else:
        beta_init = _complex_initialization(aggregate, reference_local)

    unknown = [i for i in range(component_views.size) if i != reference_local]
    theta0 = -np.angle(beta_init[unknown])
    if config.alpha_model == "phase":
        x0 = theta0.astype(np.float64)
        lower = np.full_like(x0, -np.inf)
        upper = np.full_like(x0, np.inf)
    else:
        log_gain0 = np.clip(
            -np.log(np.maximum(np.abs(beta_init[unknown]), _EPS)),
            np.log(config.alpha_gain_min),
            np.log(config.alpha_gain_max),
        )
        x0 = np.concatenate([theta0, log_gain0]).astype(np.float64)
        lower = np.concatenate(
            [
                np.full_like(theta0, -np.inf),
                np.full_like(log_gain0, np.log(config.alpha_gain_min)),
            ]
        )
        upper = np.concatenate(
            [
                np.full_like(theta0, np.inf),
                np.full_like(log_gain0, np.log(config.alpha_gain_max)),
            ]
        )

    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise ImportError("The staged alpha solver requires scipy.optimize.least_squares.") from exc

    if config.alpha_model == "phase":
        def phase_residual(parameters: np.ndarray) -> np.ndarray:
            beta_value = _parameterized_beta(
                parameters, component_views.size, reference_local, "phase"
            )
            return _block_residuals(information_blocks, beta_value)

        initial_residual = phase_residual(x0)
        f_scale = max(
            float(np.median(initial_residual)) if initial_residual.size else 1.0,
            1e-6,
        )
        result = least_squares(
            phase_residual,
            x0,
            bounds=(lower, upper),
            loss="huber",
            f_scale=f_scale,
            max_nfev=100,
        )
        beta = _parameterized_beta(
            result.x, component_views.size, reference_local, "phase"
        )
        final_block_residual = phase_residual(result.x)
        robust_weights = np.ones_like(final_block_residual)
        large = final_block_residual > f_scale
        robust_weights[large] = f_scale / np.maximum(
            final_block_residual[large], _EPS
        )
        parameter_information = _huber_parameter_information(
            information_blocks,
            beta,
            reference_local,
            "phase",
            final_block_residual,
            f_scale,
        )
        effective_row_count = float(np.sum(robust_weights))
        weighted_squared_residual = float(
            np.sum(robust_weights * final_block_residual**2)
        )
        gain_bound_active = np.zeros((component_views.size,), dtype=bool)
        information_kind = "exact_block_huber_hessian"
        optimizer_success = bool(result.success)
        optimizer_status = int(result.status)
        optimizer_message = str(result.message)
    else:
        initial_blocks = _profiled_observation_residual_blocks(
            prepared, constraints, component_views, x0, reference_local
        )
        initial_norms = np.asarray(
            [np.linalg.norm(block) for block in initial_blocks], dtype=np.float64
        )
        f_scale = max(
            float(np.median(initial_norms)) if initial_norms.size else 1.0,
            1e-6,
        )
        robust_weights = np.ones((len(constraints),), dtype=np.float64)
        current = x0

        def profiled_residual(
            parameters: np.ndarray,
            fixed_weights: np.ndarray,
        ) -> np.ndarray:
            blocks = _profiled_observation_residual_blocks(
                prepared, constraints, component_views, parameters, reference_local
            )
            return _flatten_weighted_complex_blocks(blocks, fixed_weights)

        result = None
        irls_converged = False
        for _ in range(12):
            weights_used = robust_weights.copy()
            result = least_squares(
                lambda parameters: profiled_residual(parameters, weights_used),
                current,
                bounds=(lower, upper),
                loss="linear",
                max_nfev=100,
            )
            current = result.x
            current_blocks = _profiled_observation_residual_blocks(
                prepared, constraints, component_views, current, reference_local
            )
            current_norms = np.asarray(
                [np.linalg.norm(block) for block in current_blocks], dtype=np.float64
            )
            updated_weights = np.ones_like(current_norms)
            large = current_norms > f_scale
            updated_weights[large] = f_scale / np.maximum(current_norms[large], _EPS)
            if np.allclose(updated_weights, weights_used, atol=1e-4, rtol=1e-3):
                robust_weights = weights_used
                irls_converged = True
                break
            robust_weights = updated_weights
        if not irls_converged:
            result = least_squares(
                lambda parameters: profiled_residual(parameters, robust_weights),
                current,
                bounds=(lower, upper),
                loss="linear",
                max_nfev=100,
            )
        beta = _parameterized_beta(
            result.x, component_views.size, reference_local, "bounded-complex"
        )
        final_blocks = _profiled_observation_residual_blocks(
            prepared, constraints, component_views, result.x, reference_local
        )
        final_block_residual = np.asarray(
            [np.linalg.norm(block) for block in final_blocks], dtype=np.float64
        )
        jacobian = np.asarray(result.jac, dtype=np.float64)
        parameter_information = jacobian.T @ jacobian
        effective_row_count = float(jacobian.shape[0])
        weighted_squared_residual = float(np.sum(np.asarray(result.fun) ** 2))
        gain_bound_active = np.zeros((component_views.size,), dtype=bool)
        active_mask = np.asarray(result.active_mask, dtype=np.int8)
        gain_active = active_mask[len(unknown) :] != 0
        gain_bound_active[np.asarray(unknown, dtype=np.int64)] = gain_active
        information_kind = "profiled_frozen_huber_gauss_newton"
        optimizer_success = bool(result.success) and irls_converged
        optimizer_status = int(result.status) if irls_converged else -2
        optimizer_message = (
            str(result.message)
            if irls_converged
            else "bounded-complex Huber IRLS did not reach a fixed point"
        )

    alpha = 1.0 / beta
    alpha[reference_local] = 1.0 + 0.0j
    robust_information = np.einsum(
        "b,bij->ij", robust_weights, information_blocks
    ).astype(np.complex128)
    singular, rank_ratio, information_ratio, condition = _information_summary(
        parameter_information, len(constraints)
    )
    consistency = (
        float(np.sqrt(np.mean(final_block_residual**2)))
        if final_block_residual.size
        else 0.0
    )

    phase_uncertainty = np.full((component_views.size,), np.inf, dtype=np.float64)
    phase_uncertainty[reference_local] = 0.0
    log_gain_uncertainty = (
        np.zeros((component_views.size,), dtype=np.float64)
        if config.alpha_model == "phase"
        else np.full((component_views.size,), np.inf, dtype=np.float64)
    )
    log_gain_uncertainty[reference_local] = 0.0
    if parameter_information.size and singular.size and singular[-1] > _EPS:
        covariance = np.linalg.pinv(parameter_information)
        degrees = max(effective_row_count - parameter_information.shape[0], 1.0)
        variance = weighted_squared_residual / degrees
        param_std = np.sqrt(np.maximum(np.diag(covariance) * variance, 0.0))
        unknown_array = np.asarray(unknown, dtype=np.int64)
        phase_uncertainty[unknown_array] = param_std[: len(unknown)]
        if config.alpha_model == "bounded-complex":
            log_gain_uncertainty[unknown_array] = param_std[len(unknown) :]
            log_gain_uncertainty[gain_bound_active] = np.inf
    return AlphaComponentSolve(
        alphas=alpha,
        consistency_residual=consistency,
        singular_values=singular,
        rank_ratio=rank_ratio,
        information_ratio=information_ratio,
        condition=condition,
        phase_std=phase_uncertainty,
        log_gain_std=log_gain_uncertainty,
        constraint_information=robust_information,
        parameter_information=parameter_information,
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
        gain_bound_active_mask=gain_bound_active,
        information_kind=information_kind,
    )


def solve_alpha_sync(
    prepared: PreparedObservations,
    config: StagedSolverConfig,
    *,
    enforce_failure: bool = True,
) -> AlphaSyncResult:
    config.validate()
    num_views = prepared.num_views
    alphas = np.ones((num_views,), dtype=np.complex128)
    identifiable = np.zeros((num_views,), dtype=bool)
    reasons = np.full((num_views,), "disconnected", dtype="<U32")
    phase_std = np.full((num_views,), np.inf, dtype=np.float64)
    log_gain_std = np.full((num_views,), np.inf, dtype=np.float64)
    gain_bound_active = np.zeros((num_views,), dtype=bool)

    reference_usable = bool(
        np.any((prepared.obs_view_index == 0) & (prepared.obs_weights > 0.0))
    )
    if reference_usable:
        identifiable[0] = True
        reasons[0] = "reference"
        phase_std[0] = 0.0
        log_gain_std[0] = 0.0
    else:
        reasons[0] = "empty_reference"

    all_constraints = _build_alpha_constraints(prepared)
    edge_counts, edge_information = _alpha_edge_statistics(all_constraints, num_views)
    raw_shared_counts = _raw_shared_point_counts(prepared)
    raw_adjacency = raw_shared_counts > 0
    np.fill_diagonal(raw_adjacency, False)
    structural_components = _graph_components(raw_adjacency)
    shared_count_adjacency = raw_shared_counts >= int(config.alpha_min_shared_points)
    np.fill_diagonal(shared_count_adjacency, False)
    shared_count_components = _graph_components(shared_count_adjacency)
    information_adjacency = edge_counts >= int(config.alpha_min_shared_points)
    np.fill_diagonal(information_adjacency, False)
    information_components = _graph_components(information_adjacency)
    solved_components = np.full((num_views,), -1, dtype=np.int32)
    if reference_usable:
        solved_components[0] = 0
    reference_component = int(information_components[0])
    candidate = (
        np.where(information_components == reference_component)[0]
        if reference_usable
        else np.asarray([0], dtype=np.int64)
    )

    for view_idx in range(1, num_views):
        if not reference_usable:
            reasons[view_idx] = "disconnected"
        elif information_components[view_idx] != reference_component:
            if structural_components[view_idx] != structural_components[0]:
                reasons[view_idx] = "disconnected"
            elif shared_count_components[view_idx] != shared_count_components[0]:
                reasons[view_idx] = "insufficient_overlap"
            else:
                reasons[view_idx] = "degenerate_geometry"

    final_constraints: list[AlphaConstraint] = []
    final_information = np.zeros((num_views, num_views), dtype=np.complex128)
    final_singular = np.zeros((0,), dtype=np.float64)
    final_rank_ratio = 1.0
    final_info_ratio = 0.0
    final_condition = 1.0
    final_consistency = 0.0
    final_parameter_information = np.zeros((0, 0), dtype=np.float64)
    final_parameter_view_indices = np.zeros((0,), dtype=np.int32)
    final_parameter_order = "none"
    final_optimizer_success = bool(reference_usable)
    final_optimizer_status = 0 if reference_usable else -1
    final_optimizer_message = (
        "reference-only; no relative alpha parameters"
        if reference_usable
        else "reference view has no positive-weight observations"
    )
    final_information_kind = "not_computed"

    while candidate.size > 1:
        connected_candidate = _reference_connected_subset(
            information_adjacency, candidate
        )
        disconnected_after_exclusion = np.setdiff1d(
            candidate, connected_candidate, assume_unique=True
        )
        if disconnected_after_exclusion.size:
            reasons[disconnected_after_exclusion] = "insufficient_overlap"
            candidate = connected_candidate
            if candidate.size <= 1:
                break
        allowed = np.zeros((num_views,), dtype=bool)
        allowed[candidate] = True
        constraints = _build_alpha_constraints(prepared, allowed)
        if not constraints:
            weakest = int(candidate[-1])
            reasons[weakest] = "insufficient_overlap"
            candidate = candidate[candidate != weakest]
            continue

        component = _refine_alpha_component(
            prepared, constraints, candidate, config
        )
        reference_local = int(np.where(candidate == 0)[0][0])
        parameter_count = (candidate.size - 1) * (2 if config.alpha_model == "bounded-complex" else 1)
        full_rank = component.singular_values.size == parameter_count and (
            parameter_count == 0 or component.singular_values[-1] > _EPS
        )
        no_active_gain_bound = not bool(np.any(component.gain_bound_active_mask))
        numerically_identifiable = (
            component.optimizer_success
            and no_active_gain_bound
            and full_rank
            and component.rank_ratio >= config.alpha_rank_ratio_min
            and component.information_ratio >= config.alpha_info_ratio_min
        )
        final_constraints = constraints
        final_information.fill(0.0)
        final_information[np.ix_(candidate, candidate)] = component.constraint_information
        final_singular = component.singular_values
        final_rank_ratio = component.rank_ratio
        final_info_ratio = component.information_ratio
        final_condition = component.condition
        final_consistency = component.consistency_residual
        final_parameter_information = component.parameter_information
        nonreference_candidate = candidate[candidate != 0].astype(np.int32)
        if config.alpha_model == "phase":
            final_parameter_view_indices = nonreference_candidate
            final_parameter_order = "phase"
        else:
            final_parameter_view_indices = np.concatenate(
                [nonreference_candidate, nonreference_candidate]
            )
            final_parameter_order = "phase_then_log_gain"
        final_optimizer_success = component.optimizer_success
        final_optimizer_status = component.optimizer_status
        final_optimizer_message = component.optimizer_message
        final_information_kind = component.information_kind
        gain_bound_active[candidate] = component.gain_bound_active_mask
        if numerically_identifiable:
            alphas[candidate] = component.alphas
            identifiable[candidate] = True
            reasons[candidate] = "estimated"
            reasons[0] = "reference"
            phase_std[candidate] = component.phase_std
            log_gain_std[candidate] = component.log_gain_std
            solved_components[candidate] = 0
            break

        active_nonreference = candidate[
            (candidate != 0) & component.gain_bound_active_mask
        ]
        if active_nonreference.size:
            weakest = int(active_nonreference[0])
            failure_reason = "gain_bound_limited"
        elif candidate.size == 2:
            weakest = int(candidate[candidate != 0][0])
            failure_reason = (
                "optimizer_failure"
                if not component.optimizer_success
                else "degenerate_geometry"
            )
        else:
            diagonal = np.abs(np.diag(component.parameter_information))
            phase_strength = diagonal[: candidate.size - 1]
            if config.alpha_model == "bounded-complex":
                gain_strength = diagonal[candidate.size - 1 :]
                phase_strength = np.minimum(phase_strength, gain_strength)
            nonreference = candidate[candidate != 0]
            weakest = int(nonreference[int(np.argmin(phase_strength))])
            failure_reason = (
                "optimizer_failure"
                if not component.optimizer_success
                else "degenerate_geometry"
            )
        reasons[weakest] = failure_reason
        candidate = candidate[candidate != weakest]

    if config.alpha_failure == "error" and enforce_failure:
        observed_views = np.zeros((num_views,), dtype=bool)
        observed_views[
            np.unique(prepared.obs_view_index[prepared.obs_weights > 0.0])
        ] = True
        invalid = np.where(observed_views & ~identifiable)[0]
        if invalid.size:
            labels = [str(prepared.view_ids[idx]) for idx in invalid.tolist()]
            raise ValueError(f"Unidentifiable alpha for observed views: {labels}.")

    constraint_counts = np.zeros((num_views,), dtype=np.int32)
    for constraint in final_constraints:
        nonzero = np.linalg.norm(constraint.C, axis=0) > _EPS
        constraint_counts[nonzero] += 1

    return AlphaSyncResult(
        alphas=alphas.astype(np.complex64),
        identifiable_mask=identifiable,
        component_index=solved_components,
        structural_component_index=structural_components,
        shared_count_component_index=shared_count_components,
        information_component_index=information_components,
        exclusion_reason=reasons,
        shared_point_count=raw_shared_counts,
        edge_point_count=edge_counts,
        edge_information=edge_information.astype(np.float32),
        constraint_count_per_view=constraint_counts,
        information_matrix=final_information.astype(np.complex64),
        singular_values=final_singular.astype(np.float32),
        rank_ratio=float(final_rank_ratio),
        information_ratio=float(final_info_ratio),
        condition=float(final_condition),
        consistency_residual=float(final_consistency),
        phase_std=phase_std.astype(np.float32),
        log_gain_std=log_gain_std.astype(np.float32),
        parameter_information=final_parameter_information.astype(np.float32),
        parameter_view_indices=final_parameter_view_indices,
        parameter_order=final_parameter_order,
        optimizer_success=bool(final_optimizer_success),
        optimizer_status=int(final_optimizer_status),
        optimizer_message=str(final_optimizer_message),
        gain_bound_active_mask=gain_bound_active,
        information_kind=str(final_information_kind),
    )


def solve_observable_points(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    config: StagedSolverConfig,
) -> ObservableSolveResult:
    num_points = prepared.points.shape[0]
    phi = np.zeros((num_points, 3), dtype=np.complex64)
    nullspace = np.zeros((num_points, 3, 3), dtype=np.complex64)
    nullity = np.full((num_points,), 3, dtype=np.int8)
    singular_values = np.zeros((num_points, 3), dtype=np.float32)
    rank_out = np.zeros((num_points,), dtype=np.int8)
    condition = np.full((num_points,), np.inf, dtype=np.float32)
    valid_view_count = np.zeros((num_points,), dtype=np.int32)
    residual = np.full((num_points,), np.nan, dtype=np.float32)

    for point_idx, point_rows in enumerate(prepared.rows_by_point):
        if point_rows.size == 0:
            nullspace[point_idx] = np.eye(3, dtype=np.complex64)
            continue
        keep = (
            alpha.identifiable_mask[prepared.obs_view_index[point_rows]]
            & (prepared.obs_weights[point_rows] > 0)
        )
        rows = point_rows[keep]
        if rows.size == 0:
            nullspace[point_idx] = np.eye(3, dtype=np.complex64)
            continue
        valid_view_count[point_idx] = int(np.unique(prepared.obs_view_index[rows]).size)
        sqrt_w = np.sqrt(prepared.obs_weights[rows])
        geometry = (
            sqrt_w[:, None, None] * prepared.obs_J[rows].astype(np.float64)
        ).reshape(-1, 3)
        _, singular, Vh = np.linalg.svd(geometry, full_matrices=True)
        singular_values[point_idx, : singular.size] = singular.astype(np.float32)
        rank = (
            int(np.count_nonzero(singular >= config.anchor_svd_ratio_min * singular[0]))
            if singular.size and singular[0] > _EPS
            else 0
        )
        rank_out[point_idx] = rank
        nullity[point_idx] = 3 - rank
        V = Vh.conj().T
        if rank < 3:
            nullspace[point_idx, :, : 3 - rank] = V[:, rank:].astype(np.complex64)
        if rank:
            condition[point_idx] = float(singular[0] / max(singular[rank - 1], _EPS))

        alpha_rows = alpha.alphas[prepared.obs_view_index[rows]].astype(np.complex128)
        A = (
            sqrt_w[:, None, None]
            * alpha_rows[:, None, None]
            * prepared.obs_J[rows].astype(np.complex128)
        ).reshape(-1, 3)
        b = (sqrt_w[:, None] * prepared.obs_y[rows].astype(np.complex128)).reshape(-1)
        if rank:
            observable_basis = V[:, :rank]
            coefficients = np.linalg.lstsq(A @ observable_basis, b, rcond=None)[0]
            solved = observable_basis @ coefficients
            phi[point_idx] = solved.astype(np.complex64)
        model_residual = A @ phi[point_idx].astype(np.complex128) - b
        residual[point_idx] = float(np.linalg.norm(model_residual) / max(float(np.linalg.norm(b)), _EPS))

    usable_observed = (valid_view_count > 0) & np.isfinite(residual)
    anchor_candidates = (
        (valid_view_count >= 2)
        & (rank_out == 3)
        & (condition <= _ANCHOR_CONDITION_MAX)
        & np.isfinite(residual)
    )
    anchor_candidate_residual = residual[anchor_candidates]
    if anchor_candidate_residual.size:
        median = float(np.median(anchor_candidate_residual))
        mad = float(np.median(np.abs(anchor_candidate_residual - median)))
        residual_threshold = min(
            float(config.anchor_residual_max),
            max(1e-3, median + 3.0 * mad),
        )
    else:
        residual_threshold = float(config.anchor_residual_max)
    residual_accepted = usable_observed & (residual <= residual_threshold)
    anchor = anchor_candidates & residual_accepted
    rejected = usable_observed & ~residual_accepted
    unobserved = prepared.obs_sample_count_per_point == 0
    positive_weight_row_count = np.bincount(
        prepared.obs_point_index[prepared.obs_weights > 0.0],
        minlength=num_points,
    )
    no_usable_observation = (~unobserved) & (positive_weight_row_count == 0)
    alpha_unresolved = (positive_weight_row_count > 0) & (valid_view_count == 0)
    partial = residual_accepted & ~anchor

    status = np.full(
        (num_points,), POINT_STATUS_PARTIAL_UNRESOLVED, dtype=np.int8
    )
    status[anchor] = POINT_STATUS_ANCHOR
    status[unobserved] = POINT_STATUS_UNOBSERVED_UNRESOLVED
    status[rejected] = POINT_STATUS_REJECTED_UNRESOLVED
    status[alpha_unresolved] = POINT_STATUS_ALPHA_UNRESOLVED
    status[no_usable_observation] = POINT_STATUS_NO_USABLE_OBSERVATION

    return ObservableSolveResult(
        phi=phi.copy(),
        phi_observable=phi.copy(),
        nullspace_basis=nullspace,
        nullity=nullity,
        singular_values=singular_values,
        observable_rank=rank_out,
        condition=condition,
        distinct_valid_view_count=valid_view_count,
        usable_observation_row_count=positive_weight_row_count.astype(np.int32),
        precompletion_residual=residual,
        anchor_residual_threshold=residual_threshold,
        anchor_mask=anchor,
        partial_mask=partial,
        rejected_mask=rejected,
        unobserved_mask=unobserved,
        alpha_unresolved_mask=alpha_unresolved,
        no_usable_observation_mask=no_usable_observation,
        point_status=status,
    )


def _prediction_and_residuals(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_obs = prepared.obs_y.shape[0]
    pred = np.full((num_obs, 2), np.nan + 1j * np.nan, dtype=np.complex64)
    obs_residual = np.full((num_obs,), np.nan, dtype=np.float32)
    valid = (
        alpha.identifiable_mask[prepared.obs_view_index]
        & (prepared.obs_weights > 0.0)
    )
    valid_rows = np.where(valid)[0]
    if valid_rows.size:
        projected = np.einsum(
            "oij,oj->oi",
            prepared.obs_J[valid_rows].astype(np.float32),
            phi[prepared.obs_point_index[valid_rows]].astype(np.complex64),
        )
        pred[valid_rows] = (
            alpha.alphas[prepared.obs_view_index[valid_rows], None] * projected
        ).astype(np.complex64)
        obs_residual[valid_rows] = np.linalg.norm(
            prepared.obs_y[valid_rows] - pred[valid_rows], axis=1
        ).astype(np.float32)

    sums = np.zeros((prepared.points.shape[0],), dtype=np.float64)
    counts = np.zeros((prepared.points.shape[0],), dtype=np.int32)
    if valid_rows.size:
        np.add.at(sums, prepared.obs_point_index[valid_rows], obs_residual[valid_rows].astype(np.float64) ** 2)
        np.add.at(counts, prepared.obs_point_index[valid_rows], 1)
    point_residual = np.full((prepared.points.shape[0],), np.nan, dtype=np.float32)
    has_residual = counts > 0
    point_residual[has_residual] = np.sqrt(sums[has_residual] / counts[has_residual]).astype(np.float32)
    return pred, obs_residual, valid, point_residual, has_residual


def _optional_output_fields(prepared: PreparedObservations) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for key in (
        "colors",
        "gaussian_indices",
        "point_type",
        "source_checkpoint",
    ):
        if key in prepared.arrays:
            fields[key] = prepared.arrays[key]
    return fields


def optimize_multi_view_staged(
    observations_path: str | Path,
    out_path: str | Path,
    vis_dir: str | Path | None = None,
    config: StagedSolverConfig | None = None,
) -> Path:
    config = config or StagedSolverConfig()
    config.validate()
    # copies every array from the open NPZ into a regular dictionary
    with np.load(str(observations_path), allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    prepared = prepare_observations(data)
    alpha = solve_alpha_sync(prepared, config, enforce_failure=False)
    observable = solve_observable_points(prepared, alpha, config)
    pred, obs_residual, obs_residual_valid, point_residual, point_residual_valid = _prediction_and_residuals(
        prepared, alpha, observable.phi
    )

    alpha_freqs = (
        prepared.arrays["view_freqs_hz"].astype(np.float32)
        if "view_freqs_hz" in prepared.arrays
        else np.full(
            (prepared.num_views,),
            float(np.asarray(prepared.arrays["freq_hz"]).item()),
            dtype=np.float32,
        )
    )
    num_points = prepared.points.shape[0]
    false_mask = np.zeros((num_points,), dtype=bool)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points_world=prepared.points.astype(np.float32),
        phi=observable.phi.astype(np.complex64),
        phi_observable=observable.phi_observable.astype(np.complex64),
        phi_nullspace_correction=np.zeros_like(observable.phi, dtype=np.complex64),
        point_nullspace_basis=observable.nullspace_basis.astype(np.complex64),
        point_nullity=observable.nullity.astype(np.int8),
        alphas=alpha.alphas.astype(np.complex64),
        alpha_by_view=alpha.alphas.astype(np.complex64),
        alpha_semantics=np.array("per_view_per_mode"),
        alpha_reference_view_index=np.array(0, dtype=np.int32),
        alpha_view_freqs_hz=alpha_freqs,
        alpha_identifiable_mask=alpha.identifiable_mask.astype(bool),
        alpha_component_index=alpha.component_index.astype(np.int32),
        alpha_structural_component_index=alpha.structural_component_index.astype(np.int32),
        alpha_shared_count_component_index=alpha.shared_count_component_index.astype(np.int32),
        alpha_information_component_index=alpha.information_component_index.astype(np.int32),
        alpha_exclusion_reason=alpha.exclusion_reason,
        alpha_phase=np.angle(alpha.alphas).astype(np.float32),
        alpha_gain=np.abs(alpha.alphas).astype(np.float32),
        alpha_shared_point_count=alpha.shared_point_count.astype(np.int32),
        alpha_edge_point_count=alpha.edge_point_count.astype(np.int32),
        alpha_edge_information=alpha.edge_information.astype(np.float32),
        alpha_constraint_count_per_view=alpha.constraint_count_per_view.astype(np.int32),
        alpha_information_matrix=alpha.information_matrix.astype(np.complex64),
        alpha_parameter_information=alpha.parameter_information.astype(np.float32),
        alpha_parameter_view_indices=alpha.parameter_view_indices.astype(np.int32),
        alpha_parameter_order=np.array(alpha.parameter_order),
        alpha_singular_values=alpha.singular_values.astype(np.float32),
        alpha_rank_ratio=np.array(alpha.rank_ratio, dtype=np.float32),
        alpha_information_ratio=np.array(alpha.information_ratio, dtype=np.float32),
        alpha_condition=np.array(alpha.condition, dtype=np.float32),
        alpha_consistency_residual=np.array(alpha.consistency_residual, dtype=np.float32),
        alpha_phase_std=alpha.phase_std.astype(np.float32),
        alpha_log_gain_std=alpha.log_gain_std.astype(np.float32),
        alpha_gain_std=(np.abs(alpha.alphas) * alpha.log_gain_std).astype(np.float32),
        alpha_gain_bound_active_mask=alpha.gain_bound_active_mask.astype(bool),
        alpha_optimizer_success=np.array(alpha.optimizer_success),
        alpha_optimizer_status=np.array(alpha.optimizer_status, dtype=np.int32),
        alpha_optimizer_message=np.array(alpha.optimizer_message),
        alpha_information_kind=np.array(alpha.information_kind),
        view_ids=prepared.view_ids,
        freq_hz=prepared.arrays["freq_hz"].astype(np.float32),
        mode_index=prepared.arrays["mode_index"].astype(np.int32),
        solver_method=np.array("staged_overlap_observable"),
        solver_version=np.array(2, dtype=np.int32),
        solver_alpha_model=np.array(config.alpha_model),
        point_singular_values=observable.singular_values.astype(np.float32),
        point_observable_rank=observable.observable_rank.astype(np.int8),
        point_condition=observable.condition.astype(np.float32),
        point_distinct_view_count=prepared.derived_view_count_per_point.astype(np.int32),
        point_distinct_valid_view_count=observable.distinct_valid_view_count.astype(np.int32),
        point_usable_observation_row_count=observable.usable_observation_row_count.astype(np.int32),
        point_precompletion_residual=observable.precompletion_residual.astype(np.float32),
        anchor_residual_threshold=np.array(observable.anchor_residual_threshold, dtype=np.float32),
        anchor_condition_max=np.array(_ANCHOR_CONDITION_MAX, dtype=np.float32),
        anchor_mask=observable.anchor_mask.astype(bool),
        partial_mask=observable.partial_mask.astype(bool),
        rejected_mask=observable.rejected_mask.astype(bool),
        unobserved_mask=observable.unobserved_mask.astype(bool),
        alpha_unresolved_mask=observable.alpha_unresolved_mask.astype(bool),
        no_usable_observation_mask=observable.no_usable_observation_mask.astype(bool),
        completion_mask=false_mask,
        completion_connected_to_anchor=false_mask,
        point_solution_status=observable.point_status.astype(np.int8),
        point_solution_status_names=np.asarray(POINT_STATUS_NAMES),
        point_residual=point_residual.astype(np.float32),
        point_residual_valid_mask=point_residual_valid.astype(bool),
        obs_count_per_point=prepared.obs_count_per_point.astype(np.int32),
        obs_sample_count_per_point=prepared.obs_sample_count_per_point.astype(np.int32),
        obs_point_index=prepared.obs_point_index.astype(np.int32),
        obs_view_index=prepared.obs_view_index.astype(np.int32),
        obs_pixels_xy=prepared.obs_pixels_xy.astype(np.float32),
        obs_y=prepared.obs_y.astype(np.complex64),
        obs_J=prepared.obs_J.astype(np.float32),
        obs_confidence=prepared.obs_confidence.astype(np.float32),
        obs_effective_weight=prepared.obs_weights.astype(np.float32),
        obs_pred_y=pred.astype(np.complex64),
        obs_residual=obs_residual.astype(np.float32),
        obs_residual_valid_mask=obs_residual_valid.astype(bool),
        source_observations=np.array(str(observations_path)),
        staged_summary=np.asarray(
            [
                int(alpha.identifiable_mask.sum()),
                int(observable.anchor_mask.sum()),
                int(observable.partial_mask.sum()),
                int(observable.unobserved_mask.sum()),
                int(observable.rejected_mask.sum()),
                int(observable.alpha_unresolved_mask.sum()),
                int(observable.no_usable_observation_mask.sum()),
            ],
            dtype=np.int64,
        ),
        staged_summary_names=np.asarray(
            [
                "identifiable_views",
                "anchors",
                "partial",
                "unobserved",
                "rejected",
                "alpha_unresolved",
                "no_usable_observation",
            ]
        ),
        **_optional_output_fields(prepared),
    )

    if config.alpha_failure == "error":
        observed_views = np.zeros((prepared.num_views,), dtype=bool)
        observed_views[
            np.unique(prepared.obs_view_index[prepared.obs_weights > 0.0])
        ] = True
        invalid = np.where(observed_views & ~alpha.identifiable_mask)[0]
        if invalid.size:
            labels = [str(prepared.view_ids[idx]) for idx in invalid.tolist()]
            raise ValueError(
                f"Unidentifiable alpha for observed views after writing diagnostics to {out}: {labels}."
            )

    if vis_dir is not None:
        from modal_surface.optimization_visualization import (
            _amplitude_colors,
            _phase_colors,
            _scatter_mode_image,
            _write_ply,
        )

        if "view_image_width" not in prepared.arrays or "view_image_height" not in prepared.arrays:
            raise ValueError("Visualization requires view_image_width and view_image_height.")
        vis = Path(vis_dir)
        vis.mkdir(parents=True, exist_ok=True)
        widths = prepared.arrays["view_image_width"].astype(np.int32)
        heights = prepared.arrays["view_image_height"].astype(np.int32)
        for view_idx in range(prepared.num_views):
            rows = np.where((prepared.obs_view_index == view_idx) & obs_residual_valid)[0]
            if rows.size == 0:
                continue
            view_name = str(prepared.view_ids[view_idx])
            _scatter_mode_image(
                vis / f"{view_name}_observed.png",
                prepared.obs_pixels_xy[rows],
                prepared.obs_y[rows],
                int(widths[view_idx]),
                int(heights[view_idx]),
                f"{view_name} observed",
            )
            _scatter_mode_image(
                vis / f"{view_name}_predicted.png",
                prepared.obs_pixels_xy[rows],
                pred[rows],
                int(widths[view_idx]),
                int(heights[view_idx]),
                f"{view_name} predicted",
            )
            _scatter_mode_image(
                vis / f"{view_name}_residual.png",
                prepared.obs_pixels_xy[rows],
                prepared.obs_y[rows] - pred[rows],
                int(widths[view_idx]),
                int(heights[view_idx]),
                f"{view_name} residual",
                cmap="viridis",
                normalize=False,
            )
        _write_ply(vis / "pointcloud_amplitude.ply", prepared.points, _amplitude_colors(observable.phi))
        _write_ply(vis / "pointcloud_phase_u.ply", prepared.points, _phase_colors(observable.phi))

    return out
