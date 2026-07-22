"""Exact within-view and cross-view decomposition of staged anchor residuals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from modal_surface.io import save_npz_compressed_atomic
from modal_surface.optimization_staged import prepare_observations


ANCHOR_RESIDUAL_DIAGNOSTIC_VERSION = 1
ANCHOR_STATUS = 0
RESIDUAL_SOURCE_NAMES = (
    "other",
    "accepted_anchor",
    "low_modal_energy",
    "within_view_dominated",
    "cross_view_dominated",
    "mixed",
)
_EPS = 1.0e-12
_MAD_SCALE = 1.4826
_DOMINANCE_RATIO = 2.0


@dataclass(frozen=True)
class AnchorResidualDiagnostics:
    points_world: np.ndarray
    view_ids: np.ndarray
    mode_index: int
    freq_hz: float
    alphas: np.ndarray
    alpha_identifiable_mask: np.ndarray
    point_observable_rank: np.ndarray
    point_condition: np.ndarray
    point_distinct_view_count: np.ndarray
    point_distinct_valid_view_count: np.ndarray
    point_precompletion_residual: np.ndarray
    staged_point_solution_status: np.ndarray
    anchor_residual_threshold: float
    anchor_condition_max: float
    view_sample_count: np.ndarray
    view_effective_weight: np.ndarray
    view_mode_mean: np.ndarray
    view_signal_energy: np.ndarray
    view_within_sse: np.ndarray
    view_cross_sse: np.ndarray
    view_within_residual: np.ndarray
    view_cross_residual: np.ndarray
    point_effective_weight: np.ndarray
    point_signal_energy: np.ndarray
    point_signal_rms: np.ndarray
    point_within_sse: np.ndarray
    point_cross_sse: np.ndarray
    point_total_sse: np.ndarray
    point_within_residual: np.ndarray
    point_cross_residual: np.ndarray
    point_replayed_total_residual: np.ndarray
    point_within_fraction: np.ndarray
    point_worst_within_view: np.ndarray
    point_worst_cross_view: np.ndarray
    selected_multiview_mask: np.ndarray
    residual_candidate_mask: np.ndarray
    residual_rejected_mask: np.ndarray
    anchor_mask: np.ndarray
    low_modal_energy_mask: np.ndarray
    residual_source_class: np.ndarray
    low_modal_energy_threshold: float


def _require_diagnostic_arrays(
    diagnostics: Mapping[str, np.ndarray],
    num_points: int,
    num_views: int,
) -> dict[str, np.ndarray]:
    required = {
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
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        raise ValueError(f"Solver diagnostics missing required fields: {missing}")
    arrays = {name: np.asarray(diagnostics[name]) for name in required}
    for name in (
        "point_observable_rank",
        "point_condition",
        "point_distinct_view_count",
        "point_distinct_valid_view_count",
        "point_precompletion_residual",
        "staged_point_solution_status",
    ):
        if arrays[name].shape != (num_points,):
            raise ValueError(
                f"Solver diagnostics field {name} must have shape ({num_points},)"
            )
    if arrays["alphas"].shape != (num_views,):
        raise ValueError(f"Solver diagnostics alphas must have shape ({num_views},)")
    if arrays["alpha_identifiable_mask"].shape != (num_views,):
        raise ValueError(
            "Solver diagnostics alpha_identifiable_mask must match view count"
        )
    for name in ("anchor_residual_threshold", "anchor_condition_max"):
        if arrays[name].shape != ():
            raise ValueError(f"Solver diagnostics field {name} must be scalar")
    alphas = arrays["alphas"]
    if not np.isfinite(np.real(alphas)).all() or not np.isfinite(
        np.imag(alphas)
    ).all():
        raise ValueError("Solver diagnostics alphas must be finite")
    if arrays["alpha_identifiable_mask"].dtype != np.bool_:
        raise ValueError("alpha_identifiable_mask must be boolean")
    return arrays


def _robust_low_energy_threshold(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size < 8:
        return float("nan")
    logarithm = np.log(positive)
    median = float(np.median(logarithm))
    mad = float(np.median(np.abs(logarithm - median)))
    if mad <= _EPS:
        return float("nan")
    return float(np.exp(median - 3.0 * _MAD_SCALE * mad))


def build_anchor_residual_diagnostics(
    observations: Mapping[str, np.ndarray],
    solver_diagnostics: Mapping[str, np.ndarray],
) -> AnchorResidualDiagnostics:
    prepared = prepare_observations(observations)
    num_points = int(prepared.points.shape[0])
    num_views = int(prepared.num_views)
    diagnostic = _require_diagnostic_arrays(
        solver_diagnostics,
        num_points,
        num_views,
    )
    alphas = diagnostic["alphas"].astype(np.complex128)
    identifiable = diagnostic["alpha_identifiable_mask"].astype(bool)
    rank = diagnostic["point_observable_rank"].astype(np.int8)
    condition = diagnostic["point_condition"].astype(np.float64)
    raw_view_count = diagnostic["point_distinct_view_count"].astype(np.int32)
    valid_view_count = diagnostic["point_distinct_valid_view_count"].astype(
        np.int32
    )
    saved_residual = diagnostic["point_precompletion_residual"].astype(
        np.float64
    )
    status = diagnostic["staged_point_solution_status"].astype(np.int8)
    residual_threshold = float(diagnostic["anchor_residual_threshold"].item())
    condition_max = float(diagnostic["anchor_condition_max"].item())
    if not np.isfinite(residual_threshold) or residual_threshold <= 0.0:
        raise ValueError("anchor_residual_threshold must be finite and positive")
    if not np.isfinite(condition_max) or condition_max <= 0.0:
        raise ValueError("anchor_condition_max must be finite and positive")

    view_sample_count = np.zeros((num_points, num_views), dtype=np.int32)
    view_weight = np.zeros((num_points, num_views), dtype=np.float64)
    view_mean = np.zeros((num_points, num_views, 2), dtype=np.complex64)
    view_signal = np.zeros((num_points, num_views), dtype=np.float64)
    view_within = np.zeros((num_points, num_views), dtype=np.float64)
    view_cross = np.zeros((num_points, num_views), dtype=np.float64)
    replayed_residual = np.full((num_points,), np.nan, dtype=np.float64)

    for point_index, point_rows in enumerate(prepared.rows_by_point):
        if point_rows.size == 0:
            continue
        keep = (
            identifiable[prepared.obs_view_index[point_rows]]
            & (prepared.obs_weights[point_rows] > 0.0)
        )
        rows = point_rows[keep]
        if rows.size == 0:
            continue
        weights = prepared.obs_weights[rows].astype(np.float64)
        sqrt_weight = np.sqrt(weights)
        geometry = (
            sqrt_weight[:, None, None]
            * prepared.obs_J[rows].astype(np.float64)
        ).reshape(-1, 3)
        _, _, vh = np.linalg.svd(geometry, full_matrices=True)
        point_rank = int(rank[point_index])
        if point_rank < 0 or point_rank > 3:
            raise ValueError("point_observable_rank must lie in [0,3]")
        basis = vh.conj().T[:, :point_rank]
        alpha_rows = alphas[prepared.obs_view_index[rows]]
        matrix = (
            sqrt_weight[:, None, None]
            * alpha_rows[:, None, None]
            * prepared.obs_J[rows].astype(np.complex128)
        ).reshape(-1, 3)
        target = (
            sqrt_weight[:, None]
            * prepared.obs_y[rows].astype(np.complex128)
        ).reshape(-1)
        phi = np.zeros((3,), dtype=np.complex64)
        if point_rank:
            coefficients = np.linalg.lstsq(
                matrix @ basis,
                target,
                rcond=None,
            )[0]
            phi = (basis @ coefficients).astype(np.complex64)
        direct_error = matrix @ phi.astype(np.complex128) - target
        direct_sse = float(np.vdot(direct_error, direct_error).real)
        signal_energy = float(np.vdot(target, target).real)
        replayed_residual[point_index] = np.sqrt(direct_sse) / max(
            np.sqrt(signal_energy),
            _EPS,
        )

        row_views = prepared.obs_view_index[rows]
        for view_index in np.unique(row_views).tolist():
            view_rows = rows[row_views == view_index]
            view_weights = prepared.obs_weights[view_rows].astype(np.float64)
            total_weight = float(np.sum(view_weights))
            if total_weight <= 0.0:
                continue
            jacobians = prepared.obs_J[view_rows].astype(np.float64)
            if not np.array_equal(jacobians, np.broadcast_to(jacobians[0], jacobians.shape)):
                raise ValueError(
                    "Observation Jacobians differ within one Gaussian-view group"
                )
            values = prepared.obs_y[view_rows].astype(np.complex128)
            mean = np.sum(view_weights[:, None] * values, axis=0) / total_weight
            centered = values - mean[None, :]
            within_sse = float(
                np.sum(view_weights * np.sum(np.abs(centered) ** 2, axis=1))
            )
            predicted = (
                alphas[view_index]
                * (jacobians[0].astype(np.complex128) @ phi.astype(np.complex128))
            )
            cross_sse = float(total_weight * np.sum(np.abs(predicted - mean) ** 2))
            group_signal = float(
                np.sum(view_weights * np.sum(np.abs(values) ** 2, axis=1))
            )
            view_sample_count[point_index, view_index] = int(view_rows.size)
            view_weight[point_index, view_index] = total_weight
            view_mean[point_index, view_index] = mean.astype(np.complex64)
            view_signal[point_index, view_index] = group_signal
            view_within[point_index, view_index] = within_sse
            view_cross[point_index, view_index] = cross_sse

        decomposed_sse = float(
            np.sum(view_within[point_index]) + np.sum(view_cross[point_index])
        )
        if not np.isclose(
            decomposed_sse,
            direct_sse,
            rtol=2.0e-6,
            atol=1.0e-10,
        ):
            raise ValueError(
                "Within/cross residual decomposition does not reproduce direct SSE "
                f"for Gaussian {point_index}"
            )

    saved_finite = np.isfinite(saved_residual)
    replayed_finite = np.isfinite(replayed_residual)
    if not np.array_equal(saved_finite, replayed_finite):
        mismatch = int(np.count_nonzero(saved_finite != replayed_finite))
        raise ValueError(
            f"Replayed residual validity disagrees with solver diagnostics for {mismatch} Gaussians"
        )
    if not np.allclose(
        replayed_residual[saved_finite],
        saved_residual[saved_finite],
        rtol=3.0e-5,
        atol=2.0e-6,
    ):
        mismatch = int(
            np.count_nonzero(
                ~np.isclose(
                    replayed_residual[saved_finite],
                    saved_residual[saved_finite],
                    rtol=3.0e-5,
                    atol=2.0e-6,
                )
            )
        )
        raise ValueError(
            f"Replayed residual disagrees with solver diagnostics for {mismatch} Gaussians"
        )

    point_weight = np.sum(view_weight, axis=1)
    point_signal = np.sum(view_signal, axis=1)
    point_within = np.sum(view_within, axis=1)
    point_cross = np.sum(view_cross, axis=1)
    point_total = point_within + point_cross
    point_signal_rms = np.sqrt(
        np.divide(
            point_signal,
            point_weight,
            out=np.zeros((num_points,), dtype=np.float64),
            where=point_weight > 0.0,
        )
    )
    point_within_residual = np.sqrt(
        np.divide(
            point_within,
            point_signal,
            out=np.zeros((num_points,), dtype=np.float64),
            where=point_signal > 0.0,
        )
    )
    point_cross_residual = np.sqrt(
        np.divide(
            point_cross,
            point_signal,
            out=np.zeros((num_points,), dtype=np.float64),
            where=point_signal > 0.0,
        )
    )
    within_fraction = np.divide(
        point_within,
        point_total,
        out=np.full((num_points,), 0.5, dtype=np.float64),
        where=point_total > 0.0,
    )
    view_within_residual = np.sqrt(
        np.divide(
            view_within,
            view_signal,
            out=np.zeros_like(view_within),
            where=view_signal > 0.0,
        )
    )
    view_cross_residual = np.sqrt(
        np.divide(
            view_cross,
            view_signal,
            out=np.zeros_like(view_cross),
            where=view_signal > 0.0,
        )
    )
    worst_within = np.argmax(view_within, axis=1).astype(np.int8)
    worst_cross = np.argmax(view_cross, axis=1).astype(np.int8)
    no_weight = point_weight <= 0.0
    worst_within[no_weight] = -1
    worst_cross[no_weight] = -1

    selected_multiview = raw_view_count >= 2
    residual_candidate = (
        (valid_view_count >= 2)
        & (rank == 3)
        & np.isfinite(condition)
        & (condition <= condition_max)
        & saved_finite
    )
    residual_rejected = residual_candidate & (saved_residual > residual_threshold)
    anchor = status == ANCHOR_STATUS
    expected_anchor = residual_candidate & (saved_residual <= residual_threshold)
    if not np.array_equal(anchor, expected_anchor):
        mismatch = int(np.count_nonzero(anchor != expected_anchor))
        raise ValueError(
            f"Reconstructed anchor rule disagrees with saved status for {mismatch} Gaussians"
        )
    low_energy_threshold = _robust_low_energy_threshold(
        point_signal_rms[residual_rejected]
    )
    low_energy = (
        residual_rejected
        & np.isfinite(low_energy_threshold)
        & (point_signal_rms <= low_energy_threshold)
    )
    source_class = np.zeros((num_points,), dtype=np.int8)
    source_class[anchor] = 1
    source_class[low_energy] = 2
    remaining = residual_rejected & ~low_energy
    source_class[remaining & (point_within >= _DOMINANCE_RATIO * point_cross)] = 3
    source_class[remaining & (point_cross >= _DOMINANCE_RATIO * point_within)] = 4
    source_class[remaining & (source_class == 0)] = 5

    mode_index = int(np.asarray(observations["mode_index"]).item())
    freq_hz = float(np.asarray(observations["freq_hz"]).item())
    return AnchorResidualDiagnostics(
        points_world=prepared.points.astype(np.float32),
        view_ids=prepared.view_ids.astype(str),
        mode_index=mode_index,
        freq_hz=freq_hz,
        alphas=alphas.astype(np.complex64),
        alpha_identifiable_mask=identifiable,
        point_observable_rank=rank,
        point_condition=condition.astype(np.float32),
        point_distinct_view_count=raw_view_count,
        point_distinct_valid_view_count=valid_view_count,
        point_precompletion_residual=saved_residual.astype(np.float32),
        staged_point_solution_status=status,
        anchor_residual_threshold=residual_threshold,
        anchor_condition_max=condition_max,
        view_sample_count=view_sample_count,
        view_effective_weight=view_weight.astype(np.float32),
        view_mode_mean=view_mean,
        view_signal_energy=view_signal.astype(np.float32),
        view_within_sse=view_within.astype(np.float32),
        view_cross_sse=view_cross.astype(np.float32),
        view_within_residual=view_within_residual.astype(np.float32),
        view_cross_residual=view_cross_residual.astype(np.float32),
        point_effective_weight=point_weight.astype(np.float32),
        point_signal_energy=point_signal.astype(np.float32),
        point_signal_rms=point_signal_rms.astype(np.float32),
        point_within_sse=point_within.astype(np.float32),
        point_cross_sse=point_cross.astype(np.float32),
        point_total_sse=point_total.astype(np.float32),
        point_within_residual=point_within_residual.astype(np.float32),
        point_cross_residual=point_cross_residual.astype(np.float32),
        point_replayed_total_residual=replayed_residual.astype(np.float32),
        point_within_fraction=within_fraction.astype(np.float32),
        point_worst_within_view=worst_within,
        point_worst_cross_view=worst_cross,
        selected_multiview_mask=selected_multiview,
        residual_candidate_mask=residual_candidate,
        residual_rejected_mask=residual_rejected,
        anchor_mask=anchor,
        low_modal_energy_mask=low_energy,
        residual_source_class=source_class,
        low_modal_energy_threshold=low_energy_threshold,
    )


def write_anchor_residual_diagnostics(
    path: Path,
    diagnostics: AnchorResidualDiagnostics,
    *,
    source_checkpoint: str,
    source_observation_path: Path,
    source_solver_diagnostics_path: Path,
) -> Path:
    num_points = int(diagnostics.points_world.shape[0])
    arrays = {
        "version": np.array(ANCHOR_RESIDUAL_DIAGNOSTIC_VERSION, dtype=np.int32),
        "point_type": np.array("foreground_gaussian_anchor_residual_decomposition"),
        "source_checkpoint": np.array(source_checkpoint),
        "source_observation_path": np.array(str(source_observation_path)),
        "source_solver_diagnostics_path": np.array(
            str(source_solver_diagnostics_path)
        ),
        "num_foreground_gaussians": np.array(num_points, dtype=np.int64),
        "gaussian_indices": np.arange(num_points, dtype=np.int64),
        "points_world": diagnostics.points_world,
        "view_ids": diagnostics.view_ids,
        "mode_index": np.array(diagnostics.mode_index, dtype=np.int32),
        "freq_hz": np.array(diagnostics.freq_hz, dtype=np.float32),
        "alphas": diagnostics.alphas,
        "alpha_identifiable_mask": diagnostics.alpha_identifiable_mask,
        "point_observable_rank": diagnostics.point_observable_rank,
        "point_condition": diagnostics.point_condition,
        "point_distinct_view_count": diagnostics.point_distinct_view_count,
        "point_distinct_valid_view_count": (
            diagnostics.point_distinct_valid_view_count
        ),
        "point_precompletion_residual": diagnostics.point_precompletion_residual,
        "staged_point_solution_status": diagnostics.staged_point_solution_status,
        "anchor_residual_threshold": np.array(
            diagnostics.anchor_residual_threshold,
            dtype=np.float32,
        ),
        "anchor_condition_max": np.array(
            diagnostics.anchor_condition_max,
            dtype=np.float32,
        ),
        "view_sample_count": diagnostics.view_sample_count,
        "view_effective_weight": diagnostics.view_effective_weight,
        "view_mode_mean": diagnostics.view_mode_mean,
        "view_signal_energy": diagnostics.view_signal_energy,
        "view_within_sse": diagnostics.view_within_sse,
        "view_cross_sse": diagnostics.view_cross_sse,
        "view_within_residual": diagnostics.view_within_residual,
        "view_cross_residual": diagnostics.view_cross_residual,
        "point_effective_weight": diagnostics.point_effective_weight,
        "point_signal_energy": diagnostics.point_signal_energy,
        "point_signal_rms": diagnostics.point_signal_rms,
        "point_within_sse": diagnostics.point_within_sse,
        "point_cross_sse": diagnostics.point_cross_sse,
        "point_total_sse": diagnostics.point_total_sse,
        "point_within_residual": diagnostics.point_within_residual,
        "point_cross_residual": diagnostics.point_cross_residual,
        "point_replayed_total_residual": (
            diagnostics.point_replayed_total_residual
        ),
        "point_within_fraction": diagnostics.point_within_fraction,
        "point_worst_within_view": diagnostics.point_worst_within_view,
        "point_worst_cross_view": diagnostics.point_worst_cross_view,
        "selected_multiview_mask": diagnostics.selected_multiview_mask,
        "residual_candidate_mask": diagnostics.residual_candidate_mask,
        "residual_rejected_mask": diagnostics.residual_rejected_mask,
        "anchor_mask": diagnostics.anchor_mask,
        "low_modal_energy_mask": diagnostics.low_modal_energy_mask,
        "residual_source_class": diagnostics.residual_source_class,
        "residual_source_names": np.asarray(RESIDUAL_SOURCE_NAMES),
        "low_modal_energy_threshold": np.array(
            diagnostics.low_modal_energy_threshold,
            dtype=np.float32,
        ),
        "dominance_ratio": np.array(_DOMINANCE_RATIO, dtype=np.float32),
        "mad_scale": np.array(_MAD_SCALE, dtype=np.float32),
        "effective_weight_method": np.array(
            "contribution_divided_by_point_view_multiplicity"
        ),
        "decomposition_method": np.array(
            "exact_weighted_point_view_mean_sse_identity"
        ),
        "replay_validation": np.array("point_precompletion_residual_exact"),
    }
    return save_npz_compressed_atomic(path, arrays)
