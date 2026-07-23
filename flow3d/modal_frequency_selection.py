from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Sequence, SupportsFloat

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow3d.modal_flow_coordinates import (
    _load_and_validate_caches,
    _load_frame_map,
    _view_pixel_groups,
    load_flow_observation_topology,
)
from modal_peak_pick.core.cache import ModalAnalysisCache


MODAL_FREQUENCY_SELECTION_FORMAT = "modal_frequency_selection"
MODAL_FREQUENCY_SELECTION_VERSION = 2

SUMMARY_FILENAME = "frequency_selection_summary.json"
DIAGNOSTICS_FILENAME = "frequency_selection_diagnostics.npz"
PLOT_FILENAME = "frequency_selection_curves.png"

_PIXEL_CHUNK_SIZE = 1024
_GRID_ROWS = 4
_GRID_COLS = 4
_GREEDY_TIE_TOLERANCE = 1e-12

__all__ = [
    "DIAGNOSTICS_FILENAME",
    "MODAL_FREQUENCY_SELECTION_FORMAT",
    "MODAL_FREQUENCY_SELECTION_VERSION",
    "ModalFrequencySelectionResult",
    "PLOT_FILENAME",
    "SUMMARY_FILENAME",
    "run_modal_frequency_selection",
]


@dataclass(frozen=True)
class ModalFrequencySelectionResult:
    path: Path
    summary_path: Path
    diagnostics_path: Path
    plot_path: Path


@dataclass(frozen=True)
class _ViewStatistics:
    cache: ModalAnalysisCache
    pixels: np.ndarray
    reference_index: int
    gram: np.ndarray
    cross_flow: np.ndarray
    energy_per_frame: np.ndarray
    flow_energy: float
    pair_scales: np.ndarray
    fft_detrend: bool
    fft_window: str


@dataclass(frozen=True)
class _Fit:
    coefficients: np.ndarray
    frame_sse: np.ndarray
    sse: float
    r2: float
    numerical_rank: int
    observable_condition: float


def _json_float(value: SupportsFloat) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def _topology_provenance(path: Path) -> dict[str, Any]:
    scalar_fields = (
        "topology_id",
        "source_checkpoint",
        "mask_erode_iters",
        "pixel_sample_stride",
        "pixel_candidate_k",
        "pixel_preselect_k",
        "pixel_render_acc_min",
        "pixel_min_contribution",
        "pixel_candidate_method",
    )
    required_fields = set(scalar_fields) | {"source_view_configs"}
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required_fields - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing topology provenance fields: {missing}")
        values: dict[str, Any] = {}
        for name in scalar_fields:
            array = np.asarray(archive[name])
            if array.shape != ():
                raise ValueError(f"{path} {name} must be scalar")
            value = array.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            values[name] = value
        source_view_configs_array = np.asarray(archive["source_view_configs"])
        if source_view_configs_array.ndim != 1:
            raise ValueError(f"{path} source_view_configs must be one-dimensional")
        source_view_configs = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in source_view_configs_array.tolist()
        ]

    for name in ("topology_id", "source_checkpoint", "pixel_candidate_method"):
        if not isinstance(values[name], str) or not values[name]:
            raise ValueError(f"{path} {name} must be a non-empty string")
    integer_names = (
        "mask_erode_iters",
        "pixel_sample_stride",
        "pixel_candidate_k",
        "pixel_preselect_k",
    )
    for name in integer_names:
        if isinstance(values[name], bool) or not isinstance(
            values[name], (int, np.integer)
        ):
            raise ValueError(f"{path} {name} must be an integer")
        values[name] = int(values[name])
    for name in ("pixel_render_acc_min", "pixel_min_contribution"):
        value = float(values[name])
        if not np.isfinite(value):
            raise ValueError(f"{path} {name} must be finite")
        values[name] = value
    values["source_view_configs"] = source_view_configs
    return values


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _frequency_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    values = (float(minimum), float(maximum), float(step))
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Frequency limits and step must be finite")
    if minimum <= 0.0 or maximum <= minimum or step <= 0.0:
        raise ValueError(
            "Frequency limits must satisfy 0 < min_freq_hz < max_freq_hz and step > 0"
        )
    interval_count_float = (maximum - minimum) / step
    interval_count = int(round(interval_count_float))
    tolerance = 1e-10 * max(1.0, abs(interval_count_float))
    if abs(interval_count_float - interval_count) > tolerance:
        raise ValueError(
            "The frequency range endpoints must align to an integral number of steps"
        )
    frequencies = minimum + step * np.arange(interval_count + 1, dtype=np.float64)
    frequencies[-1] = maximum
    if not np.all(np.diff(frequencies) > 0.0):
        raise ValueError("Frequency grid must be strictly increasing")
    return frequencies


def _validate_mode_counts(mode_counts: Sequence[int], candidate_count: int) -> np.ndarray:
    if len(mode_counts) == 0:
        raise ValueError("mode_counts must contain at least one value")
    values = np.asarray(
        [_validate_positive_int(value, "mode_counts entry") for value in mode_counts],
        dtype=np.int64,
    )
    if not np.array_equal(values, np.unique(values)):
        raise ValueError("mode_counts must be strictly increasing and unique")
    if np.any(np.diff(values) <= 0):
        raise ValueError("mode_counts must be strictly increasing")
    if int(values[-1]) > candidate_count:
        raise ValueError(
            f"Largest mode count {int(values[-1])} exceeds {candidate_count} candidates"
        )
    return values


def _candidate_pixels(
    topology: Any,
    view_index: int,
    cache: ModalAnalysisCache,
    pixel_stride: int,
) -> np.ndarray:
    _, _, pixels, _, _ = _view_pixel_groups(topology, view_index, cache)
    height, width = cache.flow_u.shape[1:]
    selected = (
        (pixels[:, 0] >= 1)
        & (pixels[:, 0] < width - 1)
        & (pixels[:, 1] >= 1)
        & (pixels[:, 1] < height - 1)
        & ((pixels[:, 0] - 1) % pixel_stride == 0)
        & ((pixels[:, 1] - 1) % pixel_stride == 0)
    )
    if not bool(np.all(selected)):
        invalid_count = int(np.count_nonzero(~selected))
        raise ValueError(
            f"View {topology.view_ids[view_index]!r} has {invalid_count} candidate pixels "
            f"outside the interior --pixel-stride={pixel_stride} grid"
        )
    return pixels.astype(np.int64, copy=False)


def _temporal_basis(
    num_frames: int,
    fps: float,
    frequencies_hz: np.ndarray,
    window: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    times = np.arange(num_frames, dtype=np.float64) / float(fps)
    basis = np.exp(
        (-2j * np.pi) * frequencies_hz[:, None] * times[None, :]
    ).astype(np.complex64, copy=False)
    normalized_window = window.lower()
    if normalized_window == "hann":
        indices = np.arange(num_frames, dtype=np.float64)
        temporal_window = (
            0.5
            - 0.5 * np.cos(2.0 * np.pi * indices / max(1, num_frames - 1))
        ).astype(np.float32)
    elif normalized_window in {"none", "boxcar", "rect"}:
        temporal_window = None
    else:
        raise ValueError(f"Unsupported cached FFT window {window!r}")
    return basis, temporal_window


def _mode_flow_chunks(
    cache: ModalAnalysisCache,
    pixels: np.ndarray,
    reference_index: int,
    frequencies_hz: np.ndarray,
    detrend: bool,
    window: str,
):
    basis, temporal_window = _temporal_basis(
        cache.flow_u.shape[0], cache.fps, frequencies_hz, window
    )
    for pixel_start in range(0, pixels.shape[0], _PIXEL_CHUNK_SIZE):
        pixel_end = min(pixel_start + _PIXEL_CHUNK_SIZE, pixels.shape[0])
        current = pixels[pixel_start:pixel_end]
        x = current[:, 0]
        y = current[:, 1]
        raw_u = np.asarray(cache.flow_u[:, y, x], dtype=np.float32)
        raw_v = np.asarray(cache.flow_v[:, y, x], dtype=np.float32)
        if not np.isfinite(raw_u).all() or not np.isfinite(raw_v).all():
            raise ValueError(
                f"Flow cache {cache.path} contains non-finite candidate flow values"
            )

        flow_u = raw_u.astype(np.float64)
        flow_v = raw_v.astype(np.float64)
        flow_u -= flow_u[reference_index : reference_index + 1]
        flow_v -= flow_v[reference_index : reference_index + 1]
        flow = np.empty((2 * current.shape[0], cache.flow_u.shape[0]), dtype=np.float64)
        flow[0::2] = flow_u.T
        flow[1::2] = flow_v.T

        processed_u = raw_u.copy()
        processed_v = raw_v.copy()
        if detrend:
            processed_u -= processed_u.mean(axis=0, keepdims=True, dtype=np.float32)
            processed_v -= processed_v.mean(axis=0, keepdims=True, dtype=np.float32)
        if temporal_window is not None:
            processed_u *= temporal_window[:, None]
            processed_v *= temporal_window[:, None]
        mode_u = basis @ processed_u
        mode_v = basis @ processed_v
        mode_count = frequencies_hz.size
        design = np.empty((2 * current.shape[0], 2 * mode_count), dtype=np.float64)
        design[0::2, 0::2] = mode_u.T.real
        design[1::2, 0::2] = mode_v.T.real
        design[0::2, 1::2] = -mode_u.T.imag
        design[1::2, 1::2] = -mode_v.T.imag
        if not np.isfinite(design).all():
            raise ValueError(f"Exact-DFT design from {cache.path} is non-finite")
        yield slice(pixel_start, pixel_end), design, flow


def _accumulate_view_statistics(
    cache: ModalAnalysisCache,
    pixels: np.ndarray,
    frequencies_hz: np.ndarray,
) -> _ViewStatistics:
    reference_index = int(cache.metadata["analysis"]["reference_frame_index"])
    fft = cache.metadata["analysis"]["fft"]
    detrend = bool(fft["detrend"])
    window = str(fft["window"])
    column_count = 2 * frequencies_hz.size
    gram = np.zeros((column_count, column_count), dtype=np.float64)
    cross_flow = np.zeros((column_count, cache.flow_u.shape[0]), dtype=np.float64)
    energy_per_frame = np.zeros((cache.flow_u.shape[0],), dtype=np.float64)
    for _, design, flow in _mode_flow_chunks(
        cache,
        pixels,
        reference_index,
        frequencies_hz,
        detrend,
        window,
    ):
        gram += design.T @ design
        cross_flow += design.T @ flow
        energy_per_frame += np.sum(flow * flow, axis=0)
    flow_energy = float(np.sum(energy_per_frame))
    if not np.isfinite(flow_energy) or flow_energy <= np.finfo(np.float64).eps:
        raise ValueError(f"Candidate flow for {cache.path} has zero or invalid energy")
    pair_scales = np.empty((frequencies_hz.size,), dtype=np.float64)
    denominator = float(2 * pixels.shape[0])
    for mode_index in range(frequencies_hz.size):
        pair = slice(2 * mode_index, 2 * mode_index + 2)
        pair_scales[mode_index] = np.sqrt(float(np.trace(gram[pair, pair])) / denominator)
    if not np.isfinite(pair_scales).all():
        raise ValueError(f"Exact-DFT pair scales from {cache.path} are non-finite")
    pair_scales[pair_scales <= np.finfo(np.float64).eps] = 1.0
    return _ViewStatistics(
        cache=cache,
        pixels=pixels,
        reference_index=reference_index,
        gram=gram,
        cross_flow=cross_flow,
        energy_per_frame=energy_per_frame,
        flow_energy=flow_energy,
        pair_scales=pair_scales,
        fft_detrend=detrend,
        fft_window=window,
    )


def _selected_columns(selected: Sequence[int]) -> np.ndarray:
    columns = np.empty((2 * len(selected),), dtype=np.int64)
    columns[0::2] = 2 * np.asarray(selected, dtype=np.int64)
    columns[1::2] = columns[0::2] + 1
    return columns


def _fit(statistics: _ViewStatistics, selected: Sequence[int]) -> _Fit:
    columns = _selected_columns(selected)
    if columns.size == 0:
        return _Fit(
            coefficients=np.empty((0, statistics.energy_per_frame.size), dtype=np.float64),
            frame_sse=statistics.energy_per_frame.copy(),
            sse=statistics.flow_energy,
            r2=0.0,
            numerical_rank=0,
            observable_condition=1.0,
        )
    pair_scales = statistics.pair_scales[np.asarray(selected, dtype=np.int64)]
    scales = np.repeat(pair_scales, 2)
    gram = statistics.gram[np.ix_(columns, columns)] / (
        scales[:, None] * scales[None, :]
    )
    gram = (gram + gram.T) * 0.5
    cross = statistics.cross_flow[columns] / scales[:, None]
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = max(float(eigenvalues[-1]), 0.0)
    tolerance = (
        np.finfo(np.float64).eps
        * max(gram.shape[0], 2 * statistics.pixels.shape[0])
        * largest
    )
    positive = eigenvalues > tolerance
    numerical_rank = int(np.count_nonzero(positive))
    if numerical_rank:
        observable_vectors = eigenvectors[:, positive]
        scaled_coefficients = observable_vectors @ (
            (observable_vectors.T @ cross) / eigenvalues[positive, None]
        )
        observable_condition = float(
            np.sqrt(eigenvalues[positive][-1] / eigenvalues[positive][0])
        )
    else:
        scaled_coefficients = np.zeros_like(cross)
        observable_condition = 1.0
    coefficients = scaled_coefficients / scales[:, None]
    captured_per_frame = np.sum(cross * scaled_coefficients, axis=0)
    frame_sse = statistics.energy_per_frame - captured_per_frame
    negative_tolerance = 1e-9 * max(statistics.flow_energy, 1.0)
    if float(np.min(frame_sse)) < -negative_tolerance:
        raise ValueError("Modal projection captured more energy than the input flow contains")
    frame_sse = np.maximum(frame_sse, 0.0)
    sse = float(np.sum(frame_sse))
    return _Fit(
        coefficients=coefficients,
        frame_sse=frame_sse,
        sse=sse,
        r2=float(1.0 - sse / statistics.flow_energy),
        numerical_rank=numerical_rank,
        observable_condition=observable_condition,
    )


def _pair_redundancy(
    statistics: _ViewStatistics,
    previous: Sequence[int],
    new_mode: int,
) -> float:
    if not previous:
        return 0.0
    previous_columns = _selected_columns(previous)
    new_columns = _selected_columns([new_mode])
    previous_scales = np.repeat(
        statistics.pair_scales[np.asarray(previous, dtype=np.int64)], 2
    )
    new_scales = np.repeat(statistics.pair_scales[[new_mode]], 2)
    old_gram = statistics.gram[np.ix_(previous_columns, previous_columns)] / (
        previous_scales[:, None] * previous_scales[None, :]
    )
    cross = statistics.gram[np.ix_(previous_columns, new_columns)] / (
        previous_scales[:, None] * new_scales[None, :]
    )
    new_gram = statistics.gram[np.ix_(new_columns, new_columns)] / (
        new_scales[:, None] * new_scales[None, :]
    )
    denominator = float(np.trace(new_gram))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0
    old_gram = (old_gram + old_gram.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(old_gram)
    largest = max(float(eigenvalues[-1]), 0.0)
    tolerance = (
        np.finfo(np.float64).eps
        * max(old_gram.shape[0], 2 * statistics.pixels.shape[0])
        * largest
    )
    valid = eigenvalues > tolerance
    if np.any(valid):
        projected = eigenvectors[:, valid].T @ cross
        explained = float(
            np.sum(projected * projected / eigenvalues[valid, None])
        )
    else:
        explained = 0.0
    value = explained / denominator
    if value < -1e-8 or value > 1.0 + 1e-6:
        raise ValueError("Computed mode-pair redundancy is outside [0,1]")
    return float(np.clip(value, 0.0, 1.0))


def _inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(matrix)
    largest = max(float(values[-1]), 0.0)
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * largest
    inverse = np.zeros_like(values)
    valid = values > tolerance
    inverse[valid] = 1.0 / np.sqrt(values[valid])
    return (vectors * inverse[None, :]) @ vectors.T


def _maximum_pair_correlation(
    statistics: _ViewStatistics,
    previous: Sequence[int],
    new_mode: int,
) -> float:
    if not previous:
        return 0.0
    new_columns = _selected_columns([new_mode])
    new_gram = statistics.gram[np.ix_(new_columns, new_columns)]
    new_inverse = _inverse_sqrt(new_gram)
    result = 0.0
    for old_mode in previous:
        old_columns = _selected_columns([old_mode])
        old_gram = statistics.gram[np.ix_(old_columns, old_columns)]
        cross = statistics.gram[np.ix_(old_columns, new_columns)]
        whitened = _inverse_sqrt(old_gram) @ cross @ new_inverse
        value = float(np.linalg.svd(whitened, compute_uv=False)[0])
        result = max(result, value)
    if result > 1.0 + 1e-6:
        raise ValueError("Computed canonical mode-pair correlation exceeds one")
    return float(np.clip(result, 0.0, 1.0))


def _greedy_select(
    view_statistics: Sequence[_ViewStatistics],
    frequencies_hz: np.ndarray,
    maximum_count: int,
) -> dict[str, np.ndarray]:
    view_count = len(view_statistics)
    candidate_count = frequencies_hz.size
    selected: list[int] = []
    available = set(range(candidate_count))
    selected_indices = np.empty((maximum_count,), dtype=np.int64)
    view_r2 = np.empty((view_count, maximum_count), dtype=np.float64)
    view_sse = np.empty((view_count, maximum_count), dtype=np.float64)
    numerical_rank = np.empty((view_count, maximum_count), dtype=np.int64)
    condition = np.empty((view_count, maximum_count), dtype=np.float64)
    redundancy = np.empty((view_count, maximum_count), dtype=np.float64)
    pair_correlation = np.empty((view_count, maximum_count), dtype=np.float64)
    marginal_gain = np.empty((maximum_count,), dtype=np.float64)
    previous_macro = 0.0

    standalone_view_r2 = np.empty((view_count, candidate_count), dtype=np.float64)
    for view_index, statistics in enumerate(view_statistics):
        for candidate in range(candidate_count):
            standalone_view_r2[view_index, candidate] = _fit(
                statistics, [candidate]
            ).r2

    for step in range(maximum_count):
        best_candidate: int | None = None
        best_macro = -np.inf
        best_fits: list[_Fit] | None = None
        for candidate in sorted(available):
            trial = [*selected, candidate]
            fits = [_fit(statistics, trial) for statistics in view_statistics]
            macro = float(np.mean([fit.r2 for fit in fits]))
            if macro > best_macro + _GREEDY_TIE_TOLERANCE:
                best_candidate = candidate
                best_macro = macro
                best_fits = fits
        if best_candidate is None or best_fits is None:
            raise RuntimeError(f"Greedy frequency selection failed at step {step + 1}")
        gain = best_macro - previous_macro
        if gain < -1e-9:
            raise ValueError("Greedy macro R2 decreased after adding a mode pair")
        selected_indices[step] = best_candidate
        for view_index, (statistics, fit) in enumerate(
            zip(view_statistics, best_fits)
        ):
            view_r2[view_index, step] = fit.r2
            view_sse[view_index, step] = fit.sse
            numerical_rank[view_index, step] = fit.numerical_rank
            condition[view_index, step] = fit.observable_condition
            redundancy[view_index, step] = _pair_redundancy(
                statistics, selected, best_candidate
            )
            pair_correlation[view_index, step] = _maximum_pair_correlation(
                statistics, selected, best_candidate
            )
        marginal_gain[step] = max(gain, 0.0)
        selected.append(best_candidate)
        available.remove(best_candidate)
        previous_macro = best_macro

    energy = np.asarray(
        [statistics.flow_energy for statistics in view_statistics], dtype=np.float64
    )
    pooled_r2 = 1.0 - np.sum(view_sse, axis=0) / float(np.sum(energy))
    macro_r2 = np.mean(view_r2, axis=0)
    worst_r2 = np.min(view_r2, axis=0)
    standalone_sse = energy[:, None] * (1.0 - standalone_view_r2)
    standalone_pooled = 1.0 - np.sum(standalone_sse, axis=0) / float(np.sum(energy))
    return {
        "selected_frequency_indices": selected_indices,
        "selected_frequencies_hz": frequencies_hz[selected_indices],
        "marginal_macro_r2_gain": marginal_gain,
        "view_r2": view_r2,
        "view_sse": view_sse,
        "pooled_r2": pooled_r2,
        "macro_r2": macro_r2,
        "worst_view_r2": worst_r2,
        "numerical_rank": numerical_rank,
        "observable_condition": condition,
        "selected_pair_redundancy": redundancy,
        "selected_pair_max_canonical_correlation": pair_correlation,
        "standalone_view_r2": standalone_view_r2,
        "standalone_pooled_r2": standalone_pooled,
        "standalone_macro_r2": np.mean(standalone_view_r2, axis=0),
        "standalone_worst_view_r2": np.min(standalone_view_r2, axis=0),
    }


def _regional_diagnostics(
    view_statistics: Sequence[_ViewStatistics],
    selected_indices: np.ndarray,
    frequencies_hz: np.ndarray,
    mode_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    view_count = len(view_statistics)
    count_count = mode_counts.size
    region_count = _GRID_ROWS * _GRID_COLS
    regional_energy = np.zeros((view_count, region_count), dtype=np.float64)
    regional_sse = np.zeros((view_count, count_count, region_count), dtype=np.float64)
    regional_pixel_count = np.zeros((view_count, region_count), dtype=np.int64)
    direct_sse = np.zeros((view_count, count_count), dtype=np.float64)

    selected_frequencies = frequencies_hz[selected_indices]
    for view_index, statistics in enumerate(view_statistics):
        height, width = statistics.cache.flow_u.shape[1:]
        pixel_regions = (
            np.minimum(statistics.pixels[:, 1] * _GRID_ROWS // height, _GRID_ROWS - 1)
            * _GRID_COLS
            + np.minimum(
                statistics.pixels[:, 0] * _GRID_COLS // width, _GRID_COLS - 1
            )
        ).astype(np.int64)
        regional_pixel_count[view_index] = np.bincount(
            pixel_regions, minlength=region_count
        )
        fit_coefficients = [
            _fit(statistics, selected_indices[: int(mode_count)]).coefficients
            for mode_count in mode_counts
        ]
        for pixel_slice, design, flow in _mode_flow_chunks(
            statistics.cache,
            statistics.pixels,
            statistics.reference_index,
            selected_frequencies,
            statistics.fft_detrend,
            statistics.fft_window,
        ):
            feature_regions = np.repeat(pixel_regions[pixel_slice], 2)
            regional_energy[view_index] += np.bincount(
                feature_regions,
                weights=np.sum(flow * flow, axis=1),
                minlength=region_count,
            )
            for count_index, (mode_count, coefficients) in enumerate(
                zip(mode_counts, fit_coefficients)
            ):
                prediction = design[:, : 2 * int(mode_count)] @ coefficients
                residual = prediction - flow
                feature_sse = np.sum(residual * residual, axis=1)
                regional_sse[view_index, count_index] += np.bincount(
                    feature_regions,
                    weights=feature_sse,
                    minlength=region_count,
                )
                direct_sse[view_index, count_index] += float(np.sum(feature_sse))

        for count_index, mode_count in enumerate(mode_counts):
            expected = _fit(
                statistics, selected_indices[: int(mode_count)]
            ).sse
            tolerance = 1e-7 * max(expected, statistics.flow_energy, 1.0)
            if abs(direct_sse[view_index, count_index] - expected) > tolerance:
                raise ValueError(
                    f"Direct residual check failed for view {view_index}, K={int(mode_count)}"
                )

    valid = (regional_pixel_count > 0) & (
        regional_energy > np.finfo(np.float64).eps
    )
    regional_r2 = np.full_like(regional_sse, np.nan)
    for view_index in range(view_count):
        view_valid = valid[view_index]
        view_sse = regional_sse[view_index]
        view_energy = regional_energy[view_index]
        regional_r2[view_index][:, view_valid] = (
            1.0
            - view_sse[:, view_valid]
            / view_energy[view_valid][None, :]
        )
    pooled_energy = np.sum(regional_energy, axis=0)
    pooled_valid = pooled_energy > np.finfo(np.float64).eps
    pooled_r2 = np.full((count_count, region_count), np.nan, dtype=np.float64)
    pooled_r2[:, pooled_valid] = 1.0 - np.sum(regional_sse, axis=0)[
        :, pooled_valid
    ] / pooled_energy[None, pooled_valid]
    worst_r2 = np.full((count_count, region_count), np.nan, dtype=np.float64)
    for region in range(region_count):
        valid_views = valid[:, region]
        if np.any(valid_views):
            worst_r2[:, region] = np.min(
                regional_r2[:, :, region][valid_views], axis=0
            )
    return {
        "regional_view_r2": regional_r2,
        "regional_view_sse": regional_sse,
        "regional_flow_energy": regional_energy,
        "regional_pixel_count": regional_pixel_count,
        "regional_valid": valid,
        "regional_pooled_r2": pooled_r2,
        "regional_worst_view_r2": worst_r2,
        "direct_view_sse": direct_sse,
    }


def _plot(path: Path, arrays: dict[str, np.ndarray]) -> None:
    steps = np.arange(1, arrays["selected_frequencies_hz"].size + 1)
    frequencies = arrays["candidate_frequencies_hz"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(steps, arrays["pooled_r2"], label="pooled")
    axes[0, 0].plot(steps, arrays["macro_r2"], label="equal-view macro")
    axes[0, 0].plot(steps, arrays["worst_view_r2"], label="worst view")
    axes[0, 0].set_xlabel("Selected complex mode count")
    axes[0, 0].set_ylabel("Candidate-flow R2")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].bar(steps, arrays["marginal_macro_r2_gain"])
    axes[0, 1].set_xlabel("Greedy selection step")
    axes[0, 1].set_ylabel("Marginal macro R2")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].plot(frequencies, arrays["standalone_pooled_r2"])
    axes[1, 0].scatter(
        arrays["selected_frequencies_hz"],
        arrays["standalone_pooled_r2"][arrays["selected_frequency_indices"]],
        c=steps,
        cmap="viridis",
        s=24,
    )
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Standalone pooled R2")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(
        steps,
        np.max(arrays["selected_pair_redundancy"], axis=0),
        label="max redundancy",
    )
    axes[1, 1].plot(
        steps,
        np.max(arrays["selected_pair_max_canonical_correlation"], axis=0),
        label="max pair correlation",
    )
    axes[1, 1].set_xlabel("Greedy selection step")
    axes[1, 1].set_ylabel("Redundancy diagnostic")
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    figure.suptitle("Dense exact-DFT grouped frequency selection (in-sample)")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_shortlists(
    directory: Path,
    mode_counts: np.ndarray,
    selected_frequencies_hz: np.ndarray,
) -> list[str]:
    filenames: list[str] = []
    for mode_count in mode_counts:
        count = int(mode_count)
        greedy_order = selected_frequencies_hz[:count]
        filename = f"selected_frequencies_k{count}.json"
        payload = {
            "format": "modal_frequency_shortlist",
            "version": 1,
            "mode_count": count,
            "selected_peaks_hz": sorted(float(value) for value in greedy_order),
            "greedy_selection_order_hz": [float(value) for value in greedy_order],
        }
        with (directory / filename).open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
        filenames.append(filename)
    return filenames


def _validate_output(path: Path, mode_counts: np.ndarray, view_count: int) -> None:
    summary_path = path / SUMMARY_FILENAME
    diagnostics_path = path / DIAGNOSTICS_FILENAME
    plot_path = path / PLOT_FILENAME
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if (
        summary.get("format") != MODAL_FREQUENCY_SELECTION_FORMAT
        or summary.get("version") != MODAL_FREQUENCY_SELECTION_VERSION
    ):
        raise ValueError(f"{summary_path} has invalid format/version")
    if summary.get("evaluation_scope") != "in_sample_full_time_series":
        raise ValueError(f"{summary_path} must state its in-sample evaluation scope")
    with np.load(diagnostics_path, allow_pickle=False) as archive:
        required = {
            "format",
            "version",
            "view_ids",
            "candidate_frequencies_hz",
            "mode_counts",
            "selected_frequency_indices",
            "selected_frequencies_hz",
            "view_r2",
            "view_sse",
            "flow_energy",
            "pooled_r2",
            "macro_r2",
            "worst_view_r2",
            "regional_view_r2",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{diagnostics_path} is missing fields: {missing}")
        if str(np.asarray(archive["format"]).item()) != MODAL_FREQUENCY_SELECTION_FORMAT:
            raise ValueError(f"{diagnostics_path} has invalid format")
        if int(np.asarray(archive["version"]).item()) != MODAL_FREQUENCY_SELECTION_VERSION:
            raise ValueError(f"{diagnostics_path} has invalid version")
        candidates = np.asarray(archive["candidate_frequencies_hz"], dtype=np.float64)
        indices = np.asarray(archive["selected_frequency_indices"], dtype=np.int64)
        selected = np.asarray(archive["selected_frequencies_hz"], dtype=np.float64)
        if (
            indices.shape != selected.shape
            or np.unique(indices).size != indices.size
            or np.any(indices < 0)
            or np.any(indices >= candidates.size)
            or not np.array_equal(selected, candidates[indices])
        ):
            raise ValueError(f"{diagnostics_path} has invalid selected frequencies")
        if np.asarray(archive["view_ids"]).shape != (view_count,):
            raise ValueError(f"{diagnostics_path} has invalid view_ids shape")
        if not np.array_equal(np.asarray(archive["mode_counts"]), mode_counts):
            raise ValueError(f"{diagnostics_path} mode_counts changed during write")
        view_r2 = np.asarray(archive["view_r2"], dtype=np.float64)
        view_sse = np.asarray(archive["view_sse"], dtype=np.float64)
        flow_energy = np.asarray(archive["flow_energy"], dtype=np.float64)
        if (
            view_r2.shape != (view_count, selected.size)
            or view_sse.shape != view_r2.shape
            or flow_energy.shape != (view_count,)
            or not np.isfinite(view_r2).all()
            or not np.isfinite(view_sse).all()
            or not np.isfinite(flow_energy).all()
            or np.any(flow_energy <= 0.0)
        ):
            raise ValueError(f"{diagnostics_path} has invalid view fit arrays")
        expected_curves = {
            "pooled_r2": 1.0 - np.sum(view_sse, axis=0) / float(np.sum(flow_energy)),
            "macro_r2": np.mean(view_r2, axis=0),
            "worst_view_r2": np.min(view_r2, axis=0),
        }
        for name in ("pooled_r2", "macro_r2", "worst_view_r2"):
            values = np.asarray(archive[name])
            if (
                values.shape != selected.shape
                or not np.isfinite(values).all()
                or not np.allclose(values, expected_curves[name], rtol=1e-12, atol=1e-12)
            ):
                raise ValueError(f"{diagnostics_path} {name} is invalid")
    for mode_count in mode_counts:
        shortlist_path = path / f"selected_frequencies_k{int(mode_count)}.json"
        with shortlist_path.open("r", encoding="utf-8") as file:
            shortlist = json.load(file)
        count = int(mode_count)
        greedy_values = shortlist.get("greedy_selection_order_hz")
        sorted_values = shortlist.get("selected_peaks_hz")
        expected_prefix = selected[:count]
        if (
            not isinstance(greedy_values, list)
            or not isinstance(sorted_values, list)
            or not np.array_equal(np.asarray(greedy_values, dtype=np.float64), expected_prefix)
            or not np.array_equal(
                np.asarray(sorted_values, dtype=np.float64), np.sort(expected_prefix)
            )
        ):
            raise ValueError(f"{shortlist_path} has an invalid frequency count")
    if not plot_path.is_file() or plot_path.stat().st_size == 0:
        raise ValueError(f"{plot_path} was not written")


def run_modal_frequency_selection(
    *,
    flow_cache_specs: Sequence[str],
    observation_topology_path: str | Path,
    modal_frame_map_path: str | Path,
    output_dir: str | Path,
    min_freq_hz: float,
    max_freq_hz: float,
    frequency_step_hz: float,
    mode_counts: Sequence[int],
    pixel_stride: int,
) -> ModalFrequencySelectionResult:
    """Select a shared dense exact-DFT frequency set by equal-view grouped greedy fit."""

    total_start = time.perf_counter()
    output_path = Path(output_dir).expanduser()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stride = _validate_positive_int(pixel_stride, "pixel_stride")
    frequencies_hz = _frequency_grid(
        float(min_freq_hz), float(max_freq_hz), float(frequency_step_hz)
    )
    requested_counts = _validate_mode_counts(mode_counts, frequencies_hz.size)

    frame_map = _load_frame_map(modal_frame_map_path)
    topology_source = Path(observation_topology_path).expanduser().resolve(strict=True)
    topology = load_flow_observation_topology(topology_source)
    topology_provenance = _topology_provenance(topology_source)
    if len(topology_provenance["source_view_configs"]) != len(topology.view_ids):
        raise ValueError(
            f"{topology_source} source_view_configs count does not match its views"
        )
    if stride != topology_provenance["pixel_sample_stride"]:
        raise ValueError(
            f"--pixel-stride={stride} does not match observation topology "
            f"pixel_sample_stride={topology_provenance['pixel_sample_stride']}"
        )
    caches = _load_and_validate_caches(frame_map, topology, flow_cache_specs)
    for view_id, cache in zip(frame_map.view_ids, caches):
        if float(frequencies_hz[-1]) > 0.5 * cache.fps + 1e-12:
            raise ValueError(
                f"Frequency grid exceeds Nyquist for view {view_id!r}: "
                f"{frequencies_hz[-1]:.9g} > {0.5 * cache.fps:.9g} Hz"
            )

    validation_seconds = float(time.perf_counter() - total_start)
    view_statistics: list[_ViewStatistics] = []
    view_seconds: list[float] = []
    for view_index, cache in enumerate(caches):
        view_start = time.perf_counter()
        pixels = _candidate_pixels(topology, view_index, cache, stride)
        print(
            f"Exact-DFT statistics {frame_map.view_ids[view_index]}: "
            f"pixels={pixels.shape[0]}, frames={cache.flow_u.shape[0]}, "
            f"candidates={frequencies_hz.size}",
            flush=True,
        )
        view_statistics.append(
            _accumulate_view_statistics(cache, pixels, frequencies_hz)
        )
        elapsed = float(time.perf_counter() - view_start)
        view_seconds.append(elapsed)
        print(
            f"Finished {frame_map.view_ids[view_index]} exact-DFT statistics "
            f"in {elapsed:.3f}s",
            flush=True,
        )

    selection_start = time.perf_counter()
    print(
        f"Selecting {int(requested_counts[-1])} shared complex frequency groups",
        flush=True,
    )
    selection = _greedy_select(
        view_statistics, frequencies_hz, int(requested_counts[-1])
    )
    selection_seconds = float(time.perf_counter() - selection_start)
    print(f"Finished grouped greedy selection in {selection_seconds:.3f}s", flush=True)
    for mode_count in requested_counts:
        selected_prefix = selection["selected_frequencies_hz"][: int(mode_count)]
        print(
            f"  K={int(mode_count)} frequencies (Hz): "
            + ", ".join(f"{float(value):.6g}" for value in selected_prefix),
            flush=True,
        )
    regional_start = time.perf_counter()
    print(
        "Computing direct residual and regional diagnostics for K="
        + ",".join(str(int(value)) for value in requested_counts),
        flush=True,
    )
    regional = _regional_diagnostics(
        view_statistics,
        selection["selected_frequency_indices"],
        frequencies_hz,
        requested_counts,
    )
    regional_seconds = float(time.perf_counter() - regional_start)
    print(
        f"Finished direct regional diagnostics in {regional_seconds:.3f}s",
        flush=True,
    )

    arrays: dict[str, np.ndarray] = {
        "format": np.array(MODAL_FREQUENCY_SELECTION_FORMAT),
        "version": np.array(MODAL_FREQUENCY_SELECTION_VERSION, dtype=np.int32),
        "view_ids": np.asarray(frame_map.view_ids),
        "candidate_frequencies_hz": frequencies_hz,
        "mode_counts": requested_counts,
        "greedy_steps": np.arange(
            1, int(requested_counts[-1]) + 1, dtype=np.int64
        ),
        "candidate_pixel_count": np.asarray(
            [statistics.pixels.shape[0] for statistics in view_statistics],
            dtype=np.int64,
        ),
        "flow_energy": np.asarray(
            [statistics.flow_energy for statistics in view_statistics], dtype=np.float64
        ),
        "pair_scales": np.stack(
            [statistics.pair_scales for statistics in view_statistics]
        ),
        **selection,
        **regional,
    }

    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        np.savez_compressed(temp_path / DIAGNOSTICS_FILENAME, **arrays)
        _plot(temp_path / PLOT_FILENAME, arrays)
        shortlist_filenames = _write_shortlists(
            temp_path, requested_counts, selection["selected_frequencies_hz"]
        )
        requested_positions = requested_counts - 1
        views_summary: list[dict[str, Any]] = []
        for view_index, (view_id, statistics) in enumerate(
            zip(frame_map.view_ids, view_statistics)
        ):
            views_summary.append(
                {
                    "view_id": view_id,
                    "candidate_pixel_count": int(statistics.pixels.shape[0]),
                    "frame_count": int(statistics.cache.flow_u.shape[0]),
                    "flow_energy": statistics.flow_energy,
                    "r2": selection["view_r2"][view_index].tolist(),
                    "r2_at_mode_counts": {
                        str(int(count)): float(selection["view_r2"][view_index, position])
                        for count, position in zip(requested_counts, requested_positions)
                    },
                    "observable_condition_at_mode_counts": {
                        str(int(count)): float(
                            selection["observable_condition"][view_index, position]
                        )
                        for count, position in zip(requested_counts, requested_positions)
                    },
                    "numerical_rank_at_mode_counts": {
                        str(int(count)): int(
                            selection["numerical_rank"][view_index, position]
                        )
                        for count, position in zip(requested_counts, requested_positions)
                    },
                    "worst_regional_r2_at_mode_counts": {
                        str(int(count)): _json_float(
                            np.nanmin(
                                regional["regional_view_r2"][view_index, count_index]
                            )
                        )
                        for count_index, count in enumerate(requested_counts)
                    },
                }
            )
        summary = {
            "format": MODAL_FREQUENCY_SELECTION_FORMAT,
            "version": MODAL_FREQUENCY_SELECTION_VERSION,
            "evaluation_scope": "in_sample_full_time_series",
            "interpretation_note": (
                "All exact-DFT spatial modes and reported R2 values use the same complete "
                "time series. Results measure basis capacity and are not held-out "
                "prediction scores."
            ),
            "candidate_frequencies_hz": frequencies_hz.tolist(),
            "mode_counts": requested_counts.tolist(),
            "selection": {
                "objective": "equal_view_macro_r2_grouped_complex_pair",
                "selected_frequency_indices": selection[
                    "selected_frequency_indices"
                ].tolist(),
                "selected_frequencies_hz": selection[
                    "selected_frequencies_hz"
                ].tolist(),
                "marginal_macro_r2_gain": selection[
                    "marginal_macro_r2_gain"
                ].tolist(),
                "max_view_redundancy_by_step": np.max(
                    selection["selected_pair_redundancy"], axis=0
                ).tolist(),
                "max_view_pair_correlation_by_step": np.max(
                    selection["selected_pair_max_canonical_correlation"], axis=0
                ).tolist(),
            },
            "overall": {
                "pooled_r2": selection["pooled_r2"].tolist(),
                "macro_r2": selection["macro_r2"].tolist(),
                "worst_view_r2": selection["worst_view_r2"].tolist(),
                "pooled_r2_at_mode_counts": {
                    str(int(count)): float(selection["pooled_r2"][position])
                    for count, position in zip(requested_counts, requested_positions)
                },
                "macro_r2_at_mode_counts": {
                    str(int(count)): float(selection["macro_r2"][position])
                    for count, position in zip(requested_counts, requested_positions)
                },
                "worst_view_r2_at_mode_counts": {
                    str(int(count)): float(selection["worst_view_r2"][position])
                    for count, position in zip(requested_counts, requested_positions)
                },
                "worst_pooled_regional_r2_at_mode_counts": {
                    str(int(count)): _json_float(
                        np.nanmin(regional["regional_pooled_r2"][count_index])
                    )
                    for count_index, count in enumerate(requested_counts)
                },
            },
            "views": views_summary,
            "settings": {
                "min_freq_hz": float(min_freq_hz),
                "max_freq_hz": float(max_freq_hz),
                "frequency_step_hz": float(frequency_step_hz),
                "pixel_stride": stride,
                "pixel_chunk_size": _PIXEL_CHUNK_SIZE,
                "grid_rows": _GRID_ROWS,
                "grid_cols": _GRID_COLS,
                "complex_pair_columns": ["real", "negative_imaginary"],
            },
            "source": {
                "observation_topology": str(topology_source),
                "topology_id": topology_provenance["topology_id"],
                "source_checkpoint": topology_provenance["source_checkpoint"],
                "source_view_configs": topology_provenance["source_view_configs"],
                "topology_view_ids": list(topology.view_ids),
                "topology_sampling": {
                    "mask_erode_iters": topology_provenance["mask_erode_iters"],
                    "pixel_sample_stride": topology_provenance[
                        "pixel_sample_stride"
                    ],
                    "pixel_candidate_k": topology_provenance[
                        "pixel_candidate_k"
                    ],
                    "pixel_preselect_k": topology_provenance[
                        "pixel_preselect_k"
                    ],
                    "pixel_render_acc_min": topology_provenance[
                        "pixel_render_acc_min"
                    ],
                    "pixel_min_contribution": topology_provenance[
                        "pixel_min_contribution"
                    ],
                    "pixel_candidate_method": topology_provenance[
                        "pixel_candidate_method"
                    ],
                },
                "modal_frame_map": str(frame_map.path),
                "flow_caches": [str(cache.path.resolve()) for cache in caches],
            },
            "timings_seconds": {
                "input_validation": validation_seconds,
                "view_statistics": view_seconds,
                "selection": selection_seconds,
                "regional_residual": regional_seconds,
                "analysis_total": float(time.perf_counter() - total_start),
            },
            "diagnostics_file": DIAGNOSTICS_FILENAME,
            "plot_file": PLOT_FILENAME,
            "shortlist_files": shortlist_filenames,
        }
        with (temp_path / SUMMARY_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
            file.write("\n")
        _validate_output(temp_path, requested_counts, len(frame_map.view_ids))
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError(f"Output directory appeared during analysis: {output_path}")
        os.replace(temp_path, output_path)
    except BaseException:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise

    return ModalFrequencySelectionResult(
        path=output_path,
        summary_path=output_path / SUMMARY_FILENAME,
        diagnostics_path=output_path / DIAGNOSTICS_FILENAME,
        plot_path=output_path / PLOT_FILENAME,
    )
