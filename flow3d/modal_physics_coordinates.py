from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import matplotlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, onenormest, splu

from flow3d.modal_flow_coordinates import (
    COORDINATE_FILENAME,
    LEGACY_MODAL_FLOW_COORDINATE_SOLVER,
    MODAL_FLOW_COORDINATE_FORMAT,
    MODAL_FLOW_COORDINATE_GAUGE,
    MODAL_FLOW_COORDINATE_SOLVER,
    MODAL_FLOW_COORDINATE_VERSION,
    MODAL_PHYSICS_COORDINATE_SOLVER,
    ModalFlowCoordinates,
    evaluate_modal_flow_coordinate_sets,
    load_modal_coordinate_provenance,
    load_modal_flow_coordinates,
)


PHYSICS_COORDINATE_FORMAT = "modal_physics_coordinates"
PHYSICS_COORDINATE_VERSION = 1
PHYSICS_DIAGNOSTICS_NPZ_FILENAME = "physics_diagnostics.npz"
PHYSICS_DIAGNOSTICS_JSON_FILENAME = "diagnostics.json"
PHYSICS_PLOT_FILENAME = "physics_coordinate_summary.png"

__all__ = [
    "PHYSICS_COORDINATE_FORMAT",
    "PHYSICS_COORDINATE_VERSION",
    "PHYSICS_DIAGNOSTICS_JSON_FILENAME",
    "PHYSICS_DIAGNOSTICS_NPZ_FILENAME",
    "PHYSICS_PLOT_FILENAME",
    "postfit_modal_physics_coordinates",
]


def _validate_nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_difference_weights(
    sample_times: np.ndarray,
    evaluation_time: float,
    derivative_order: int,
) -> np.ndarray:
    offsets = np.asarray(sample_times, dtype=np.float64) - float(evaluation_time)
    count = offsets.size
    powers = np.arange(count, dtype=np.int64)[:, None]
    system = offsets[None, :] ** powers
    rhs = np.zeros((count,), dtype=np.float64)
    rhs[derivative_order] = math.factorial(derivative_order)
    return np.linalg.solve(system, rhs)


def _finite_difference_matrix(times_sec: np.ndarray, derivative_order: int) -> sparse.csr_matrix:
    times = np.asarray(times_sec, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("Finite-difference times must be a non-empty 1-D array")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("Finite-difference times must be finite and strictly increasing")
    if derivative_order not in (1, 2):
        raise ValueError("Only first and second derivatives are supported")
    count = times.size
    if count == 1 or (count == 2 and derivative_order == 2):
        return sparse.csr_matrix((count, count), dtype=np.float64)
    if count == 2:
        step = float(times[1] - times[0])
        return sparse.csr_matrix(
            np.asarray([[-1.0 / step, 1.0 / step], [-1.0 / step, 1.0 / step]])
        )

    row_values: list[int] = []
    column_values: list[int] = []
    data_values: list[float] = []
    for row in range(count):
        if row == 0:
            columns = np.asarray([0, 1, 2], dtype=np.int64)
        elif row == count - 1:
            columns = np.asarray([count - 3, count - 2, count - 1], dtype=np.int64)
        else:
            columns = np.asarray([row - 1, row, row + 1], dtype=np.int64)
        weights = _finite_difference_weights(
            times[columns],
            float(times[row]),
            derivative_order,
        )
        row_values.extend([row] * columns.size)
        column_values.extend(columns.tolist())
        data_values.extend(weights.tolist())
    return sparse.coo_matrix(
        (data_values, (row_values, column_values)),
        shape=(count, count),
        dtype=np.float64,
    ).tocsr()


def _forcing_difference_matrix(count: int) -> sparse.csr_matrix:
    if count <= 1:
        return sparse.csr_matrix((0, count), dtype=np.float64)
    return sparse.diags(
        diagonals=(-np.ones(count - 1), np.ones(count - 1)),
        offsets=(0, 1),
        shape=(count - 1, count),
        format="csr",
        dtype=np.float64,
    )


def _condition_estimate(system: sparse.csc_matrix, factor: Any) -> float:
    system_norm = float(onenormest(system))

    def solve(value: np.ndarray) -> np.ndarray:
        return np.asarray(factor.solve(np.asarray(value, dtype=np.float64)))

    inverse = LinearOperator(
        system.shape,
        matvec=solve,
        rmatvec=solve,
        matmat=solve,
        rmatmat=solve,
        dtype=np.float64,
    )
    result = system_norm * float(onenormest(inverse))
    if not np.isfinite(result) or result < 1.0:
        raise ValueError("Physics coordinate system condition estimate is invalid")
    return result


def _spectral_metrics(
    coordinates: np.ndarray,
    times_sec: np.ndarray,
    assigned_frequency_hz: float,
    band_half_width_hz: float,
) -> tuple[float, float]:
    values = np.asarray(coordinates, dtype=np.complex128)
    times = np.asarray(times_sec, dtype=np.float64)
    if values.ndim != 1 or times.shape != values.shape:
        raise ValueError("Spectral coordinate inputs must be matching 1-D arrays")
    if values.size < 2:
        return 0.0, 0.0
    time_steps = np.diff(times)
    time_step = float(np.median(time_steps))
    if not np.allclose(time_steps, time_step, rtol=1e-6, atol=1e-9):
        raise ValueError("Spectral diagnostics require uniform frame times within a view")
    centered = values - np.mean(values)
    energy = np.square(np.abs(np.fft.fft(centered)))
    frequencies = np.fft.fftfreq(values.size, d=time_step)
    energy[np.isclose(frequencies, 0.0, rtol=0.0, atol=1e-15)] = 0.0
    total = float(np.sum(energy))
    if total <= np.finfo(np.float64).eps:
        return 0.0, 0.0
    dominant = float(frequencies[int(np.argmax(energy))])
    band = np.abs(np.abs(frequencies) - assigned_frequency_hz) <= band_half_width_hz
    ratio = float(np.clip(np.sum(energy[band]) / total, 0.0, 1.0))
    return dominant, ratio


def _complex_correlation(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.complex128)
    centered = values - np.mean(values, axis=0, keepdims=True)
    norms = np.sqrt(np.sum(np.square(np.abs(centered)), axis=0))
    denominator = norms[:, None] * norms[None, :]
    gram = centered.conj().T @ centered
    correlation = np.zeros(denominator.shape, dtype=np.float64)
    valid = denominator > np.finfo(np.float64).eps
    correlation[valid] = np.abs(gram[valid]) / denominator[valid]
    return np.clip(correlation, 0.0, 1.0)


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values)
    return float(np.sqrt(np.mean(np.square(np.abs(array)))))


def _solve_view_mode(
    source: np.ndarray,
    times_sec: np.ndarray,
    frequency_hz: float,
    damping_ratio: float,
    forcing_weight: float,
    forcing_difference_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    values = np.asarray(source, dtype=np.complex128)
    times = np.asarray(times_sec, dtype=np.float64)
    count = values.size
    first = _finite_difference_matrix(times, 1)
    second = _finite_difference_matrix(times, 2)
    omega = 2.0 * np.pi * float(frequency_hz)
    oscillator = (
        second
        + (2.0 * damping_ratio * omega) * first
        + (omega * omega) * sparse.eye(count, format="csr", dtype=np.float64)
    ).tocsr()
    force_difference = _forcing_difference_matrix(count)

    if forcing_weight == 0.0 and forcing_difference_weight == 0.0:
        fitted = values.copy()
        condition = 1.0
    else:
        system = sparse.eye(count, format="csr", dtype=np.float64) / count
        if forcing_weight > 0.0:
            system = system + (
                forcing_weight
                * (oscillator.T @ oscillator)
                / (count * omega**4)
            )
        if forcing_difference_weight > 0.0 and count > 1:
            difference_oscillator = force_difference @ oscillator
            system = system + (
                forcing_difference_weight
                * (difference_oscillator.T @ difference_oscillator)
                / ((count - 1) * omega**4)
            )
        system = system.tocsc()
        factor = splu(system)
        rhs = np.column_stack(
            [values.real / count, values.imag / count, np.ones(count, dtype=np.float64)]
        )
        solved = factor.solve(rhs)
        constraint_direction = solved[:, 2]
        denominator = float(np.sum(constraint_direction))
        if not np.isfinite(denominator) or abs(denominator) <= np.finfo(np.float64).eps:
            raise ValueError("Physics coordinate mean-zero constraint is singular")
        fitted_real = solved[:, 0] - constraint_direction * (
            float(np.sum(solved[:, 0])) / denominator
        )
        fitted_imag = solved[:, 1] - constraint_direction * (
            float(np.sum(solved[:, 1])) / denominator
        )
        fitted = fitted_real + 1j * fitted_imag
        condition = _condition_estimate(system, factor)

    input_force = oscillator @ values
    output_force = oscillator @ fitted
    input_force_difference = force_difference @ input_force
    output_force_difference = force_difference @ output_force
    scale = max(_rms(values), 1e-12)
    input_magnitude = np.abs(values)
    output_magnitude = np.abs(fitted)
    input_p90 = float(np.percentile(input_magnitude, 90))
    input_p99 = float(np.percentile(input_magnitude, 99))
    output_p90 = float(np.percentile(output_magnitude, 90))
    output_p99 = float(np.percentile(output_magnitude, 99))
    diagnostics = {
        "input_coordinate_rms": _rms(values),
        "output_coordinate_rms": _rms(fitted),
        "input_coordinate_p90": input_p90,
        "output_coordinate_p90": output_p90,
        "input_coordinate_p99": input_p99,
        "output_coordinate_p99": output_p99,
        "input_coordinate_max": float(np.max(input_magnitude)),
        "output_coordinate_max": float(np.max(output_magnitude)),
        "fidelity_nrmse": _rms(fitted - values) / scale,
        "rms_retention": _rms(fitted) / scale,
        "p90_retention": output_p90 / max(input_p90, 1e-12),
        "p99_retention": output_p99 / max(input_p99, 1e-12),
        "input_first_derivative_rms": _rms(first @ values),
        "output_first_derivative_rms": _rms(first @ fitted),
        "input_second_derivative_rms": _rms(second @ values),
        "output_second_derivative_rms": _rms(second @ fitted),
        "input_forcing_normalized_rms": _rms(input_force) / (omega * omega * scale),
        "output_forcing_normalized_rms": _rms(output_force) / (omega * omega * scale),
        "input_forcing_difference_normalized_rms": (
            _rms(input_force_difference) / (omega * omega * scale)
            if input_force_difference.size
            else 0.0
        ),
        "output_forcing_difference_normalized_rms": (
            _rms(output_force_difference) / (omega * omega * scale)
            if output_force_difference.size
            else 0.0
        ),
        "system_condition_estimate": condition,
        "solve_seconds": float(time.perf_counter() - start),
    }
    if not np.isfinite(fitted.real).all() or not np.isfinite(fitted.imag).all():
        raise ValueError("Physics coordinate solution is non-finite")
    return fitted, diagnostics


def _write_coordinate_artifact(
    path: Path,
    source: ModalFlowCoordinates,
    coordinates: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        format=np.array(MODAL_FLOW_COORDINATE_FORMAT),
        version=np.array(MODAL_FLOW_COORDINATE_VERSION, dtype=np.int32),
        view_ids=np.asarray(source.view_ids),
        frame_names=np.asarray(source.frame_names),
        frame_view_indices=source.frame_view_indices.astype(np.int64),
        frame_local_indices=source.frame_local_indices.astype(np.int64),
        frame_times_sec=source.frame_times_sec.astype(np.float64),
        mode_indices=source.mode_indices.astype(np.int64),
        frequencies_hz=source.frequencies_hz.astype(np.float64),
        coordinate_real=coordinates.real.astype(np.float32),
        coordinate_imag=coordinates.imag.astype(np.float32),
        reference_local_indices=source.reference_local_indices.astype(np.int64),
        ridge_relative=np.array(source.ridge_relative, dtype=np.float64),
        source_modal_manifest=np.array(str(source.source_modal_manifest)),
        source_modal_frame_map=np.array(str(source.source_modal_frame_map)),
        source_flow_cache_dirs=np.asarray(
            [str(path_value) for path_value in source.source_flow_cache_dirs]
        ),
    )


def _write_plot(
    path: Path,
    frequencies_hz: np.ndarray,
    diagnostics: dict[str, np.ndarray],
) -> None:
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    input_band = np.mean(diagnostics["input_assigned_frequency_energy_ratio"], axis=0)
    output_band = np.mean(diagnostics["output_assigned_frequency_energy_ratio"], axis=0)
    fidelity = np.mean(diagnostics["fidelity_nrmse"], axis=0)
    p99_retention = np.mean(diagnostics["p99_retention"], axis=0)
    input_force = np.mean(diagnostics["input_forcing_normalized_rms"], axis=0)
    output_force = np.mean(diagnostics["output_forcing_normalized_rms"], axis=0)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(frequencies_hz, input_band, "o-", label="input")
    axes[0, 0].plot(frequencies_hz, output_band, "o-", label="physics")
    axes[0, 0].set_title("Assigned-frequency band energy")
    axes[0, 0].set_ylabel("energy ratio")
    axes[0, 0].legend()
    axes[0, 1].bar(np.arange(frequencies_hz.size), fidelity)
    axes[0, 1].set_title("Coordinate fidelity NRMSE")
    axes[0, 1].set_xticks(np.arange(frequencies_hz.size), [f"{x:.3g}" for x in frequencies_hz], rotation=60)
    axes[1, 0].bar(np.arange(frequencies_hz.size), p99_retention)
    axes[1, 0].axhline(0.8, color="red", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Strong-amplitude p99 retention")
    axes[1, 0].set_xticks(np.arange(frequencies_hz.size), [f"{x:.3g}" for x in frequencies_hz], rotation=60)
    axes[1, 1].plot(frequencies_hz, input_force, "o-", label="input")
    axes[1, 1].plot(frequencies_hz, output_force, "o-", label="physics")
    axes[1, 1].set_title("Normalized latent forcing RMS")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("assigned frequency (Hz)")
        axis.grid(alpha=0.25)
    figure.suptitle("Latent-force oscillator coordinate post-fit")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def postfit_modal_physics_coordinates(
    *,
    input_coordinates: str | Path,
    out_dir: str | Path,
    damping_ratio: float = 0.05,
    forcing_weight: float = 0.1,
    forcing_difference_weight: float = 0.0,
    assigned_band_half_width_hz: float = 0.1,
    frame_chunk_size: int = 64,
) -> ModalFlowCoordinates:
    """Fit fixed coordinates to a damped oscillator with a regularized latent force."""

    damping = _validate_nonnegative_finite(damping_ratio, "damping_ratio")
    forcing = _validate_nonnegative_finite(forcing_weight, "forcing_weight")
    forcing_difference = _validate_nonnegative_finite(
        forcing_difference_weight, "forcing_difference_weight"
    )
    band_half_width = float(assigned_band_half_width_hz)
    if not np.isfinite(band_half_width) or band_half_width <= 0.0:
        raise ValueError("assigned_band_half_width_hz must be finite and positive")
    if (
        isinstance(frame_chunk_size, bool)
        or int(frame_chunk_size) != frame_chunk_size
        or frame_chunk_size < 1
    ):
        raise ValueError("frame_chunk_size must be a positive integer")
    chunk_size = int(frame_chunk_size)
    target = Path(out_dir).expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Physics coordinate output already exists: {target}")

    total_start = time.perf_counter()
    source = load_modal_flow_coordinates(input_coordinates)
    input_provenance = load_modal_coordinate_provenance(source)
    if input_provenance["solver"] not in {
        LEGACY_MODAL_FLOW_COORDINATE_SOLVER,
        MODAL_FLOW_COORDINATE_SOLVER,
    }:
        raise ValueError(
            "Physics post-fit requires a reference-flow or rendered-projection ridge "
            "coordinate artifact"
        )
    input_values = source.coordinates.astype(np.complex128)
    output_values = np.empty_like(input_values)
    num_views = len(source.view_ids)
    num_modes = source.mode_indices.size
    diagnostic_names = (
        "input_coordinate_rms",
        "output_coordinate_rms",
        "input_coordinate_p90",
        "output_coordinate_p90",
        "input_coordinate_p99",
        "output_coordinate_p99",
        "input_coordinate_max",
        "output_coordinate_max",
        "fidelity_nrmse",
        "rms_retention",
        "p90_retention",
        "p99_retention",
        "input_first_derivative_rms",
        "output_first_derivative_rms",
        "input_second_derivative_rms",
        "output_second_derivative_rms",
        "input_forcing_normalized_rms",
        "output_forcing_normalized_rms",
        "input_forcing_difference_normalized_rms",
        "output_forcing_difference_normalized_rms",
        "system_condition_estimate",
        "solve_seconds",
    )
    diagnostics = {
        name: np.empty((num_views, num_modes), dtype=np.float64)
        for name in diagnostic_names
    }
    input_band = np.empty((num_views, num_modes), dtype=np.float64)
    output_band = np.empty_like(input_band)
    input_dominant = np.empty_like(input_band)
    output_dominant = np.empty_like(input_band)
    input_correlation = np.empty((num_views, num_modes, num_modes), dtype=np.float64)
    output_correlation = np.empty_like(input_correlation)

    solve_start = time.perf_counter()
    for view_index, view_id in enumerate(source.view_ids):
        rows = np.flatnonzero(source.frame_view_indices == view_index)
        order = np.argsort(source.frame_local_indices[rows])
        ordered_rows = rows[order]
        times = source.frame_times_sec[ordered_rows]
        view_input = input_values[ordered_rows]
        view_output = np.empty_like(view_input)
        for mode_slot, frequency_hz in enumerate(source.frequencies_hz):
            fitted, mode_diagnostics = _solve_view_mode(
                view_input[:, mode_slot],
                times,
                float(frequency_hz),
                damping,
                forcing,
                forcing_difference,
            )
            view_output[:, mode_slot] = fitted
            for name in diagnostic_names:
                diagnostics[name][view_index, mode_slot] = float(mode_diagnostics[name])
            input_dominant[view_index, mode_slot], input_band[view_index, mode_slot] = (
                _spectral_metrics(
                    view_input[:, mode_slot], times, float(frequency_hz), band_half_width
                )
            )
            output_dominant[view_index, mode_slot], output_band[view_index, mode_slot] = (
                _spectral_metrics(
                    fitted, times, float(frequency_hz), band_half_width
                )
            )
        if not np.isfinite(view_output.real).all() or not np.isfinite(view_output.imag).all():
            raise ValueError(f"Physics coordinates are non-finite for view {view_id!r}")
        if forcing > 0.0 or forcing_difference > 0.0:
            view_output -= np.mean(view_output, axis=0, keepdims=True)
        output_values[ordered_rows] = view_output
        input_correlation[view_index] = _complex_correlation(view_input)
        output_correlation[view_index] = _complex_correlation(view_output)
    solve_seconds = float(time.perf_counter() - solve_start)

    output_values = output_values.astype(np.complex64).astype(np.complex128)
    flow_evaluation = evaluate_modal_flow_coordinate_sets(
        source,
        {"input": input_values, "output": output_values},
        frame_chunk_size=chunk_size,
    )
    diagnostics.update(
        {
            "input_assigned_frequency_energy_ratio": input_band,
            "output_assigned_frequency_energy_ratio": output_band,
            "input_dominant_signed_frequency_hz": input_dominant,
            "output_dominant_signed_frequency_hz": output_dominant,
            "input_cross_mode_correlation": input_correlation,
            "output_cross_mode_correlation": output_correlation,
        }
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    )
    try:
        write_start = time.perf_counter()
        _write_coordinate_artifact(temporary / COORDINATE_FILENAME, source, output_values)
        npz_payload: dict[str, np.ndarray] = {
            "view_ids": np.asarray(source.view_ids),
            "mode_indices": source.mode_indices.astype(np.int64),
            "frequencies_hz": source.frequencies_hz.astype(np.float64),
            **diagnostics,
        }
        for label, values in flow_evaluation.items():
            for name, value in values.items():
                npz_payload[f"{label}_{name}"] = np.asarray(value)
        np.savez_compressed(
            temporary / PHYSICS_DIAGNOSTICS_NPZ_FILENAME,
            **npz_payload,
        )
        _write_plot(
            temporary / PHYSICS_PLOT_FILENAME,
            source.frequencies_hz,
            diagnostics,
        )
        write_seconds = float(time.perf_counter() - write_start)

        views: list[dict[str, Any]] = []
        for view_index, view_id in enumerate(source.view_ids):
            modes: list[dict[str, Any]] = []
            for mode_slot, frequency_hz in enumerate(source.frequencies_hz):
                modes.append(
                    {
                        "mode_slot": mode_slot,
                        "mode_index": int(source.mode_indices[mode_slot]),
                        "frequency_hz": float(frequency_hz),
                        "fidelity_nrmse": float(diagnostics["fidelity_nrmse"][view_index, mode_slot]),
                        "rms_retention": float(diagnostics["rms_retention"][view_index, mode_slot]),
                        "p90_retention": float(diagnostics["p90_retention"][view_index, mode_slot]),
                        "p99_retention": float(diagnostics["p99_retention"][view_index, mode_slot]),
                        "input_assigned_frequency_energy_ratio": float(input_band[view_index, mode_slot]),
                        "output_assigned_frequency_energy_ratio": float(output_band[view_index, mode_slot]),
                        "input_dominant_signed_frequency_hz": float(input_dominant[view_index, mode_slot]),
                        "output_dominant_signed_frequency_hz": float(output_dominant[view_index, mode_slot]),
                        "input_forcing_normalized_rms": float(diagnostics["input_forcing_normalized_rms"][view_index, mode_slot]),
                        "output_forcing_normalized_rms": float(diagnostics["output_forcing_normalized_rms"][view_index, mode_slot]),
                        "system_condition_estimate": float(diagnostics["system_condition_estimate"][view_index, mode_slot]),
                    }
                )
            views.append(
                {
                    "view_id": view_id,
                    "frame_count": int(np.count_nonzero(source.frame_view_indices == view_index)),
                    "input_flow_r2": float(flow_evaluation["input"]["view_flow_r2"][view_index]),
                    "output_flow_r2": float(flow_evaluation["output"]["view_flow_r2"][view_index]),
                    "input_strong_motion_flow_r2": float(flow_evaluation["input"]["view_strong_motion_flow_r2"][view_index]),
                    "output_strong_motion_flow_r2": float(flow_evaluation["output"]["view_strong_motion_flow_r2"][view_index]),
                    "modes": modes,
                }
            )
        payload = {
            "format": PHYSICS_COORDINATE_FORMAT,
            "version": PHYSICS_COORDINATE_VERSION,
            "solver": MODAL_PHYSICS_COORDINATE_SOLVER,
            "gauge": MODAL_FLOW_COORDINATE_GAUGE,
            "coordinate_artifact": COORDINATE_FILENAME,
            "source_coordinate": str(source.path),
            "source_coordinate_solver": input_provenance["solver"],
            "settings": {
                "damping_ratio": damping,
                "forcing_weight": forcing,
                "forcing_difference_weight": forcing_difference,
                "assigned_band_half_width_hz": band_half_width,
                "frame_chunk_size": chunk_size,
                "derivative_scheme": "actual_time_three_point_second_order",
                "forcing_difference": "adjacent_sample_difference",
                "complex_channels": "shared_operator_independent_real_imaginary_rhs",
            },
            "overall": {
                "input_flow_r2": float(flow_evaluation["input"]["overall_flow_r2"]),
                "output_flow_r2": float(flow_evaluation["output"]["overall_flow_r2"]),
                "flow_r2_delta": float(flow_evaluation["output"]["overall_flow_r2"] - flow_evaluation["input"]["overall_flow_r2"]),
                "mean_fidelity_nrmse": float(np.mean(diagnostics["fidelity_nrmse"])),
                "mean_rms_retention": float(np.mean(diagnostics["rms_retention"])),
                "mean_p90_retention": float(np.mean(diagnostics["p90_retention"])),
                "mean_p99_retention": float(np.mean(diagnostics["p99_retention"])),
                "mean_input_assigned_frequency_energy_ratio": float(np.mean(input_band)),
                "mean_output_assigned_frequency_energy_ratio": float(np.mean(output_band)),
                "mean_input_forcing_normalized_rms": float(np.mean(diagnostics["input_forcing_normalized_rms"])),
                "mean_output_forcing_normalized_rms": float(np.mean(diagnostics["output_forcing_normalized_rms"])),
            },
            "timings_seconds": {
                "physics_solve": solve_seconds,
                "write": write_seconds,
                "total": float(time.perf_counter() - total_start),
            },
            "views": views,
        }
        if input_provenance["solver"] == MODAL_FLOW_COORDINATE_SOLVER:
            payload.update(
                {
                    name: input_provenance[name]
                    for name in (
                        "rendered_design_source",
                        "rendered_design_identity",
                        "rendered_design_normalization",
                    )
                }
            )
        with (temporary / PHYSICS_DIAGNOSTICS_JSON_FILENAME).open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)

        validated = load_modal_flow_coordinates(temporary / COORDINATE_FILENAME)
        provenance = load_modal_coordinate_provenance(validated)
        if provenance["solver"] != MODAL_PHYSICS_COORDINATE_SOLVER:
            raise ValueError("Published physics coordinate provenance did not validate")
        max_mean = 0.0
        for view_index in range(num_views):
            rows = np.flatnonzero(validated.frame_view_indices == view_index)
            mean = np.mean(validated.coordinates[rows], axis=0)
            max_mean = max(max_mean, float(np.max(np.abs(mean))))
        if max_mean > 1e-6:
            raise ValueError(
                f"Published physics coordinate temporal mean {max_mean:.6g} exceeds 1e-6"
            )
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Physics coordinate output already exists: {target}")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_modal_flow_coordinates(target / COORDINATE_FILENAME)
