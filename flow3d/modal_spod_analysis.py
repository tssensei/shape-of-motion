from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow3d.modal_flow_coordinates import (
    _load_and_validate_caches,
    _load_frame_map,
    _load_manifest,
)
from flow3d.modal_frequency_selection import _candidate_pixels
from modal_peak_pick.core.cache import ModalAnalysisCache


MODAL_SPOD_ANALYSIS_FORMAT = "modal_spod_analysis"
MODAL_SPOD_ANALYSIS_VERSION = 1

SUMMARY_FILENAME = "spod_summary.json"
DIAGNOSTICS_FILENAME = "spod_diagnostics.npz"
BANDS_FILENAME = "frequency_bands.json"
SPECTRA_PLOT_FILENAME = "spod_spectra.png"
STABILITY_PLOT_FILENAME = "spod_stability.png"
MODE_PREVIEW_DIRNAME = "mode_previews"

SLOW_SWAY_LABEL = "slow_sway_low_frequency"
SLOW_SWAY_CUTOFF_HZ = 0.2

_PIXEL_CHUNK_SIZE = 1024
_TOP_EIGENVALUE_COUNT = 3
_LOCAL_PROMINENCE_RADIUS = 3
_HANN_EFFECTIVE_BANDWIDTH_BINS = 1.5
_SHAPE_MAC_MERGE_THRESHOLD = 0.9
_SHAPE_MAC_MAX_GAP_MULTIPLIER = 2.0

__all__ = [
    "BANDS_FILENAME",
    "DIAGNOSTICS_FILENAME",
    "MODAL_SPOD_ANALYSIS_FORMAT",
    "MODAL_SPOD_ANALYSIS_VERSION",
    "ModalSpodAnalysisResult",
    "SPECTRA_PLOT_FILENAME",
    "STABILITY_PLOT_FILENAME",
    "SUMMARY_FILENAME",
    "run_modal_spod_analysis",
]


@dataclass(frozen=True)
class ModalSpodAnalysisResult:
    path: Path
    summary_path: Path
    diagnostics_path: Path
    bands_path: Path
    spectra_plot_path: Path
    stability_plot_path: Path


@dataclass(frozen=True)
class _WindowSpec:
    length: int
    hop: int
    starts: np.ndarray
    frequency_indices: np.ndarray
    frequencies_hz: np.ndarray
    hann: np.ndarray
    snapshot_scale: np.ndarray


@dataclass(frozen=True)
class _SpodMetrics:
    spec: _WindowSpec
    gram: np.ndarray
    top_eigenvalues: np.ndarray
    leading_vectors: np.ndarray
    total_power: np.ndarray
    dominance: np.ndarray
    separation: np.ndarray
    split_half_mac: np.ndarray
    loo_mac_median: np.ndarray
    loo_mac_min: np.ndarray
    nonoverlap_mac_median: np.ndarray
    nonoverlap_pair_count: int
    window_power: np.ndarray
    window_power_cv: np.ndarray
    normalized_leading_power: np.ndarray
    local_prominence: np.ndarray


@dataclass(frozen=True)
class _ViewAnalysis:
    view_id: str
    cache: ModalAnalysisCache
    pixels: np.ndarray
    configurations: tuple[_SpodMetrics, ...]
    cross_primary_frequency_indices: np.ndarray
    cross_secondary_frequency_indices: np.ndarray
    cross_resolution_frequency_hz: np.ndarray
    cross_resolution_mac: np.ndarray


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_settings(
    window_lengths: Sequence[int],
    overlap_fraction: float,
    min_freq_hz: float,
    max_freq_hz: float,
    pixel_stride: int,
) -> tuple[tuple[int, int], float, float, float, int]:
    if len(window_lengths) != 2:
        raise ValueError("window_lengths must contain exactly two values")
    lengths = tuple(
        _validate_positive_int(value, "window_lengths entry")
        for value in window_lengths
    )
    if lengths[0] <= lengths[1]:
        raise ValueError(
            "window_lengths must list the longer spectral window before the shorter "
            "stability window"
        )
    overlap = float(overlap_fraction)
    minimum = float(min_freq_hz)
    maximum = float(max_freq_hz)
    if not np.isfinite(overlap) or overlap < 0.0 or overlap >= 1.0:
        raise ValueError("overlap_fraction must be finite and in [0,1)")
    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or minimum <= 0.0
        or maximum <= minimum
    ):
        raise ValueError(
            "Frequency limits must be finite and satisfy 0 < min_freq_hz < max_freq_hz"
        )
    for length in lengths:
        hop_float = length * (1.0 - overlap)
        hop = int(round(hop_float))
        if hop <= 0 or not math.isclose(
            hop_float, hop, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "Each window length times (1 - overlap_fraction) must be a "
                "positive integer"
            )
    return (
        (lengths[0], lengths[1]),
        overlap,
        minimum,
        maximum,
        _validate_positive_int(pixel_stride, "pixel_stride"),
    )


def _build_window_spec(
    *,
    num_frames: int,
    fps: float,
    measurement_count: int,
    length: int,
    overlap_fraction: float,
    min_freq_hz: float,
    max_freq_hz: float,
) -> _WindowSpec:
    if length > num_frames:
        raise ValueError(
            f"Window length {length} exceeds the available {num_frames} frames"
        )
    hop = int(round(length * (1.0 - overlap_fraction)))
    starts = np.arange(0, num_frames - length + 1, hop, dtype=np.int64)
    if starts.size < 4:
        raise ValueError(
            f"Window length {length} yields only {starts.size} segments; at least four "
            "are required for split-half stability"
        )
    full_frequencies = np.fft.rfftfreq(length, d=1.0 / fps)
    selected = np.flatnonzero(
        (full_frequencies >= min_freq_hz - 1e-12)
        & (full_frequencies <= max_freq_hz + 1e-12)
    ).astype(np.int64)
    if selected.size == 0:
        raise ValueError(
            f"Window length {length} has no native FFT bins in the requested range"
        )
    if int(selected[0]) == 0:
        raise ValueError("The SPOD analysis range must not include the DC bin")
    hann = np.hanning(length).astype(np.float32)
    hann_energy = float(np.sum(hann.astype(np.float64) ** 2))
    if not np.isfinite(hann_energy) or hann_energy <= 0.0:
        raise ValueError(f"Window length {length} produced invalid Hann energy")
    one_sided_factor = np.full(selected.shape, 2.0, dtype=np.float64)
    if length % 2 == 0:
        one_sided_factor[selected == length // 2] = 1.0
    snapshot_scale = np.sqrt(
        one_sided_factor
        / (float(fps) * hann_energy * float(measurement_count))
    ).astype(np.float32)
    return _WindowSpec(
        length=length,
        hop=hop,
        starts=starts,
        frequency_indices=selected,
        frequencies_hz=full_frequencies[selected].astype(np.float64, copy=False),
        hann=hann,
        snapshot_scale=snapshot_scale,
    )


def _window_snapshots(
    raw_u: np.ndarray,
    raw_v: np.ndarray,
    spec: _WindowSpec,
) -> np.ndarray:
    if raw_u.shape != raw_v.shape or raw_u.ndim != 2:
        raise ValueError("Chunked flow must have matching [T,P] u/v arrays")
    pixel_count = raw_u.shape[1]
    snapshots = np.empty(
        (
            spec.frequencies_hz.size,
            spec.starts.size,
            2 * pixel_count,
        ),
        dtype=np.complex64,
    )
    window = spec.hann[:, None]
    for window_index, start in enumerate(spec.starts):
        stop = int(start) + spec.length
        flow_u = np.asarray(raw_u[int(start) : stop], dtype=np.float32).copy()
        flow_v = np.asarray(raw_v[int(start) : stop], dtype=np.float32).copy()
        flow_u -= flow_u.mean(axis=0, keepdims=True, dtype=np.float32)
        flow_v -= flow_v.mean(axis=0, keepdims=True, dtype=np.float32)
        flow_u *= window
        flow_v *= window
        spectrum_u = np.fft.rfft(flow_u, axis=0)[spec.frequency_indices]
        spectrum_v = np.fft.rfft(flow_v, axis=0)[spec.frequency_indices]
        snapshots[:, window_index, 0::2] = spectrum_u.astype(
            np.complex64, copy=False
        )
        snapshots[:, window_index, 1::2] = spectrum_v.astype(
            np.complex64, copy=False
        )
    snapshots *= spec.snapshot_scale[:, None, None]
    if not np.isfinite(snapshots).all():
        raise ValueError("Windowed SPOD snapshots contain non-finite values")
    return snapshots


def _matched_frequency_indices(
    primary: np.ndarray,
    secondary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    primary_indices: list[int] = []
    secondary_indices: list[int] = []
    for primary_index, frequency in enumerate(primary):
        distances = np.abs(secondary - frequency)
        secondary_index = int(np.argmin(distances))
        if float(distances[secondary_index]) <= 1e-10:
            primary_indices.append(primary_index)
            secondary_indices.append(secondary_index)
    if not primary_indices:
        raise ValueError("Window configurations have no exactly shared native FFT bins")
    return (
        np.asarray(primary_indices, dtype=np.int64),
        np.asarray(secondary_indices, dtype=np.int64),
    )


def _frequency_axes_match(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and bool(
        np.allclose(left, right, rtol=0.0, atol=1e-10)
    )


def _accumulate_snapshot_grams(
    cache: ModalAnalysisCache,
    pixels: np.ndarray,
    specs: tuple[_WindowSpec, _WindowSpec],
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    primary_indices, secondary_indices = _matched_frequency_indices(
        specs[0].frequencies_hz, specs[1].frequencies_hz
    )
    grams = (
        np.zeros(
            (
                specs[0].frequencies_hz.size,
                specs[0].starts.size,
                specs[0].starts.size,
            ),
            dtype=np.complex128,
        ),
        np.zeros(
            (
                specs[1].frequencies_hz.size,
                specs[1].starts.size,
                specs[1].starts.size,
            ),
            dtype=np.complex128,
        ),
    )
    cross_gram = np.zeros(
        (primary_indices.size, specs[0].starts.size, specs[1].starts.size),
        dtype=np.complex128,
    )
    for pixel_start in range(0, pixels.shape[0], _PIXEL_CHUNK_SIZE):
        pixel_end = min(pixel_start + _PIXEL_CHUNK_SIZE, pixels.shape[0])
        current = pixels[pixel_start:pixel_end]
        x = current[:, 0]
        y = current[:, 1]
        raw_u = np.asarray(cache.flow_u[:, y, x], dtype=np.float32)
        raw_v = np.asarray(cache.flow_v[:, y, x], dtype=np.float32)
        if not np.isfinite(raw_u).all() or not np.isfinite(raw_v).all():
            raise ValueError(f"Flow cache {cache.path} has non-finite candidate flow")
        snapshots = tuple(
            _window_snapshots(raw_u, raw_v, spec) for spec in specs
        )
        for gram, values in zip(grams, snapshots):
            gram += np.einsum(
                "fwm,fvm->fwv",
                values.conj(),
                values,
                optimize=True,
                dtype=np.complex128,
            )
        cross_gram += np.einsum(
            "qwm,qvm->qwv",
            snapshots[0][primary_indices].conj(),
            snapshots[1][secondary_indices],
            optimize=True,
            dtype=np.complex128,
        )
    normalized = (
        grams[0] / float(specs[0].starts.size),
        grams[1] / float(specs[1].starts.size),
    )
    return normalized, cross_gram, primary_indices, secondary_indices


def _embedded_leading_vector(gram: np.ndarray, indices: np.ndarray) -> np.ndarray:
    result = np.zeros((gram.shape[0],), dtype=np.complex128)
    subgram = gram[np.ix_(indices, indices)]
    subgram = (subgram + subgram.conj().T) * 0.5
    values, vectors = np.linalg.eigh(subgram)
    if float(values[-1]) <= np.finfo(np.float64).eps:
        return result
    result[indices] = vectors[:, -1]
    return result


def _coefficient_mac(first: np.ndarray, second: np.ndarray, gram: np.ndarray) -> float:
    first_norm = float(np.real(np.vdot(first, gram @ first)))
    second_norm = float(np.real(np.vdot(second, gram @ second)))
    denominator = first_norm * second_norm
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    inner = np.vdot(first, gram @ second)
    value = float(abs(inner) ** 2 / denominator)
    if value > 1.0 + 1e-6:
        raise ValueError(f"Computed SPOD MAC exceeds one: {value}")
    return float(np.clip(value, 0.0, 1.0))


def _local_prominence(values: np.ndarray) -> np.ndarray:
    result = np.ones_like(values, dtype=np.float64)
    for index, value in enumerate(values):
        start = max(0, index - _LOCAL_PROMINENCE_RADIUS)
        stop = min(values.size, index + _LOCAL_PROMINENCE_RADIUS + 1)
        neighbors = np.concatenate((values[start:index], values[index + 1 : stop]))
        if neighbors.size:
            baseline = float(np.median(neighbors))
            result[index] = float(value) / max(
                baseline, np.finfo(np.float64).eps
            )
    return result


def _compute_metrics(spec: _WindowSpec, gram: np.ndarray) -> _SpodMetrics:
    frequency_count = spec.frequencies_hz.size
    window_count = spec.starts.size
    top_eigenvalues = np.zeros(
        (frequency_count, _TOP_EIGENVALUE_COUNT), dtype=np.float64
    )
    leading_vectors = np.zeros(
        (frequency_count, window_count), dtype=np.complex128
    )
    total_power = np.zeros((frequency_count,), dtype=np.float64)
    dominance = np.zeros_like(total_power)
    separation = np.zeros_like(total_power)
    split_half_mac = np.zeros_like(total_power)
    loo_mac_median = np.zeros_like(total_power)
    loo_mac_min = np.zeros_like(total_power)
    nonoverlap_mac_median = np.zeros_like(total_power)
    window_power = np.zeros((frequency_count, window_count), dtype=np.float64)
    first_half = np.arange(0, window_count // 2, dtype=np.int64)
    second_half = np.arange(window_count // 2, window_count, dtype=np.int64)
    nonoverlap_pairs = [
        (left, right)
        for left in range(window_count)
        for right in range(left + 1, window_count)
        if int(spec.starts[right]) - int(spec.starts[left]) >= spec.length
    ]
    all_indices = np.arange(window_count, dtype=np.int64)

    for frequency_index in range(frequency_count):
        current = gram[frequency_index]
        hermitian_error = float(np.max(np.abs(current - current.conj().T)))
        if hermitian_error > 1e-10 * max(float(np.max(np.abs(current))), 1.0):
            raise ValueError("Accumulated SPOD Gram matrix is not Hermitian")
        current = (current + current.conj().T) * 0.5
        values, vectors = np.linalg.eigh(current)
        negative_tolerance = 1e-10 * max(float(values[-1]), 1.0)
        if float(values[0]) < -negative_tolerance:
            raise ValueError("Accumulated SPOD Gram matrix is not positive semidefinite")
        values = np.maximum(values, 0.0)
        descending = values[::-1]
        retained = min(_TOP_EIGENVALUE_COUNT, descending.size)
        top_eigenvalues[frequency_index, :retained] = descending[:retained]
        leading_vectors[frequency_index] = vectors[:, -1]
        total = float(np.sum(values))
        leading = float(descending[0])
        total_power[frequency_index] = total
        dominance[frequency_index] = leading / max(
            total, np.finfo(np.float64).eps
        )
        second = float(descending[1]) if descending.size > 1 else 0.0
        separation[frequency_index] = leading / max(
            second, np.finfo(np.float64).eps * max(leading, 1.0)
        )
        window_power[frequency_index] = (
            np.maximum(np.real(np.diag(current)), 0.0) * window_count
        )

        first_vector = _embedded_leading_vector(current, first_half)
        second_vector = _embedded_leading_vector(current, second_half)
        split_half_mac[frequency_index] = _coefficient_mac(
            first_vector, second_vector, current
        )

        full_vector = leading_vectors[frequency_index]
        loo_values: list[float] = []
        for omitted in range(window_count):
            retained_indices = all_indices[all_indices != omitted]
            retained_vector = _embedded_leading_vector(current, retained_indices)
            loo_values.append(
                _coefficient_mac(full_vector, retained_vector, current)
            )
        loo_mac_median[frequency_index] = float(np.median(loo_values))
        loo_mac_min[frequency_index] = float(np.min(loo_values))

        snapshot_macs: list[float] = []
        for left, right in nonoverlap_pairs:
            denominator = float(
                np.real(current[left, left]) * np.real(current[right, right])
            )
            if denominator <= np.finfo(np.float64).eps:
                snapshot_macs.append(0.0)
            else:
                snapshot_macs.append(
                    float(
                        np.clip(
                            abs(current[left, right]) ** 2 / denominator,
                            0.0,
                            1.0,
                        )
                    )
                )
        if snapshot_macs:
            nonoverlap_mac_median[frequency_index] = float(
                np.median(snapshot_macs)
            )

    window_power_mean = np.mean(window_power, axis=1)
    window_power_cv = np.std(window_power, axis=1) / np.maximum(
        window_power_mean, np.finfo(np.float64).eps
    )
    leading_power = top_eigenvalues[:, 0]
    normalized_leading_power = leading_power / max(
        float(np.max(leading_power)), np.finfo(np.float64).eps
    )
    arrays = (
        top_eigenvalues,
        leading_vectors,
        total_power,
        dominance,
        separation,
        split_half_mac,
        loo_mac_median,
        loo_mac_min,
        nonoverlap_mac_median,
        window_power,
        window_power_cv,
        normalized_leading_power,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("SPOD diagnostics contain non-finite values")
    return _SpodMetrics(
        spec=spec,
        gram=gram,
        top_eigenvalues=top_eigenvalues,
        leading_vectors=leading_vectors,
        total_power=total_power,
        dominance=dominance,
        separation=separation,
        split_half_mac=split_half_mac,
        loo_mac_median=loo_mac_median,
        loo_mac_min=loo_mac_min,
        nonoverlap_mac_median=nonoverlap_mac_median,
        nonoverlap_pair_count=len(nonoverlap_pairs),
        window_power=window_power,
        window_power_cv=window_power_cv,
        normalized_leading_power=normalized_leading_power,
        local_prominence=_local_prominence(leading_power),
    )


def _cross_resolution_mac(
    primary: _SpodMetrics,
    secondary: _SpodMetrics,
    cross_gram: np.ndarray,
    primary_indices: np.ndarray,
    secondary_indices: np.ndarray,
) -> np.ndarray:
    values = np.zeros((primary_indices.size,), dtype=np.float64)
    primary_window_count = primary.spec.starts.size
    secondary_window_count = secondary.spec.starts.size
    for match_index, (primary_index, secondary_index) in enumerate(
        zip(primary_indices, secondary_indices)
    ):
        primary_value = float(primary.top_eigenvalues[int(primary_index), 0])
        secondary_value = float(secondary.top_eigenvalues[int(secondary_index), 0])
        denominator = math.sqrt(
            primary_window_count
            * primary_value
            * secondary_window_count
            * secondary_value
        )
        if denominator <= np.finfo(np.float64).eps:
            continue
        primary_vector = primary.leading_vectors[int(primary_index)]
        secondary_vector = secondary.leading_vectors[int(secondary_index)]
        inner = np.vdot(
            primary_vector,
            cross_gram[match_index] @ secondary_vector,
        )
        current = float(abs(inner) ** 2 / (denominator * denominator))
        if current > 1.0 + 1e-5:
            raise ValueError(
                f"Cross-resolution SPOD MAC exceeds one: {current}"
            )
        values[match_index] = float(np.clip(current, 0.0, 1.0))
    if not np.isfinite(values).all():
        raise ValueError("Cross-resolution SPOD MAC is non-finite")
    return values


def _analyze_view(
    *,
    view_id: str,
    cache: ModalAnalysisCache,
    pixels: np.ndarray,
    window_lengths: tuple[int, int],
    overlap_fraction: float,
    min_freq_hz: float,
    max_freq_hz: float,
) -> _ViewAnalysis:
    measurement_count = 2 * pixels.shape[0]
    specs = tuple(
        _build_window_spec(
            num_frames=cache.flow_u.shape[0],
            fps=cache.fps,
            measurement_count=measurement_count,
            length=length,
            overlap_fraction=overlap_fraction,
            min_freq_hz=min_freq_hz,
            max_freq_hz=max_freq_hz,
        )
        for length in window_lengths
    )
    if len(specs) != 2:
        raise RuntimeError("Expected exactly two SPOD window specifications")
    print(
        f"Windowed SPOD {view_id}: pixels={pixels.shape[0]}, "
        f"frames={cache.flow_u.shape[0]}, "
        f"segments=[{specs[0].starts.size},{specs[1].starts.size}]",
        flush=True,
    )
    grams, cross_gram, primary_indices, secondary_indices = (
        _accumulate_snapshot_grams(cache, pixels, (specs[0], specs[1]))
    )
    configurations = (
        _compute_metrics(specs[0], grams[0]),
        _compute_metrics(specs[1], grams[1]),
    )
    cross_values = _cross_resolution_mac(
        configurations[0],
        configurations[1],
        cross_gram,
        primary_indices,
        secondary_indices,
    )
    return _ViewAnalysis(
        view_id=view_id,
        cache=cache,
        pixels=pixels,
        configurations=configurations,
        cross_primary_frequency_indices=primary_indices,
        cross_secondary_frequency_indices=secondary_indices,
        cross_resolution_frequency_hz=specs[0].frequencies_hz[primary_indices],
        cross_resolution_mac=cross_values,
    )


def _local_maxima(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Peak input must be a non-empty one-dimensional array")
    selected: list[int] = []
    for index in range(1, values.size - 1):
        value = values[index]
        left = values[index - 1]
        right = values[index + 1]
        if value >= left and value >= right and (value > left or value > right):
            selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def _candidate_peak_indices(
    frequencies_hz: np.ndarray,
    shared_support: np.ndarray,
) -> np.ndarray:
    peaks = _local_maxima(shared_support).tolist()
    low_indices = np.flatnonzero(frequencies_hz < SLOW_SWAY_CUTOFF_HZ)
    if low_indices.size:
        low_representative = int(
            low_indices[np.argmax(shared_support[low_indices])]
        )
        peaks.append(low_representative)
    return np.asarray(sorted(set(peaks)), dtype=np.int64)


def _spatial_mode_mac(modes: np.ndarray) -> np.ndarray:
    if modes.ndim != 3 or modes.shape[2] != 2 or modes.shape[0] == 0:
        raise ValueError("Spatial mode MAC requires non-empty [M,P,2] modes")
    flattened = modes.reshape(modes.shape[0], -1).astype(
        np.complex128, copy=False
    )
    norms = np.sum(np.abs(flattened) ** 2, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("Spatial mode MAC received an invalid mode norm")
    inner = flattened.conj() @ flattened.T
    mac = np.abs(inner) ** 2 / (norms[:, None] * norms[None, :])
    mac = np.clip(np.real(mac), 0.0, 1.0)
    if not np.isfinite(mac).all():
        raise ValueError("Spatial mode MAC is non-finite")
    return mac.astype(np.float64, copy=False)


def _shared_spatial_mode_mac(
    modes_by_view: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if len(modes_by_view) < 2:
        raise ValueError("Shared spatial mode MAC requires at least two views")
    per_view = np.stack([_spatial_mode_mac(modes) for modes in modes_by_view])
    return per_view, np.sort(per_view, axis=0)[-2]


def _shared_support(view_analyses: Sequence[_ViewAnalysis]) -> np.ndarray:
    stacked = np.stack(
        [
            analysis.configurations[0].normalized_leading_power
            for analysis in view_analyses
        ]
    )
    if stacked.shape[0] < 2:
        raise ValueError("Shared SPOD support requires at least two views")
    return np.sort(stacked, axis=0)[-2]


def _band_cross_resolution_index(
    analysis: _ViewAnalysis,
    representative_frequency_hz: float,
    band_start_hz: float,
    band_end_hz: float,
) -> int | None:
    frequencies = analysis.cross_resolution_frequency_hz
    matches = np.flatnonzero(
        (frequencies >= band_start_hz - 1e-12)
        & (frequencies <= band_end_hz + 1e-12)
    )
    if matches.size == 0:
        nearest = int(np.argmin(np.abs(frequencies - representative_frequency_hz)))
        secondary_resolution = (
            analysis.cache.fps / analysis.configurations[1].spec.length
        )
        if (
            abs(float(frequencies[nearest]) - representative_frequency_hz)
            > 0.5 * float(secondary_resolution) + 1e-12
        ):
            return None
        return nearest
    distances = np.abs(frequencies[matches] - representative_frequency_hz)
    return int(matches[int(np.argmin(distances))])


def _build_frequency_bands(
    view_analyses: Sequence[_ViewAnalysis],
    shared_support: np.ndarray,
    candidate_peak_indices: np.ndarray,
    shared_spatial_mac: np.ndarray,
    min_freq_hz: float,
    max_freq_hz: float,
) -> list[dict[str, Any]]:
    primary_frequencies = view_analyses[0].configurations[0].spec.frequencies_hz
    primary_length = view_analyses[0].configurations[0].spec.length
    fps = view_analyses[0].cache.fps
    effective_bandwidth = (
        _HANN_EFFECTIVE_BANDWIDTH_BINS * fps / float(primary_length)
    )
    shape_merge_max_gap = _SHAPE_MAC_MAX_GAP_MULTIPLIER * effective_bandwidth
    if shared_spatial_mac.shape != (
        candidate_peak_indices.size,
        candidate_peak_indices.size,
    ):
        raise ValueError("Shared spatial MAC does not match candidate peak indices")
    candidate_position = {
        int(frequency_index): position
        for position, frequency_index in enumerate(candidate_peak_indices)
    }
    bands: list[dict[str, Any]] = []

    low_indices = np.flatnonzero(primary_frequencies < SLOW_SWAY_CUTOFF_HZ)
    if low_indices.size:
        representative = int(low_indices[np.argmax(shared_support[low_indices])])
        low_peaks = [
            int(index)
            for index in candidate_peak_indices
            if float(primary_frequencies[int(index)]) < SLOW_SWAY_CUTOFF_HZ
        ]
        bands.append(
            {
                "label": SLOW_SWAY_LABEL,
                "peak_indices": low_peaks,
                "representative_index": representative,
                "start_hz": float(primary_frequencies[int(low_indices[0])]),
                "end_hz": min(SLOW_SWAY_CUTOFF_HZ, max_freq_hz),
                "merge_links": [],
            }
        )

    ordinary_peaks = [
        int(index)
        for index in candidate_peak_indices
        if float(primary_frequencies[int(index)]) >= SLOW_SWAY_CUTOFF_HZ
    ]
    groups: list[list[int]] = []
    group_links: list[list[dict[str, Any]]] = []
    for peak in ordinary_peaks:
        if not groups:
            groups.append([peak])
            group_links.append([])
            continue
        previous = groups[-1][-1]
        adjacent_gap = float(
            primary_frequencies[peak] - primary_frequencies[previous]
        )
        peak_position = candidate_position[peak]
        group_positions = [candidate_position[index] for index in groups[-1]]
        group_macs = shared_spatial_mac[peak_position, group_positions]
        group_frequency_gaps = np.asarray(
            [
                float(primary_frequencies[peak] - primary_frequencies[index])
                for index in groups[-1]
            ],
            dtype=np.float64,
        )
        eligible_offsets = np.flatnonzero(
            group_frequency_gaps <= shape_merge_max_gap + 1e-12
        )
        if eligible_offsets.size:
            best_group_offset = int(
                eligible_offsets[
                    np.argmax(group_macs[eligible_offsets])
                ]
            )
        else:
            best_group_offset = len(groups[-1]) - 1
        best_group_peak = groups[-1][best_group_offset]
        best_shape_mac = float(group_macs[best_group_offset])
        matched_gap = float(group_frequency_gaps[best_group_offset])
        merge_reason: str | None = None
        if adjacent_gap <= effective_bandwidth + 1e-12:
            merge_reason = "hann_effective_bandwidth"
        elif (
            matched_gap <= shape_merge_max_gap + 1e-12
            and best_shape_mac >= _SHAPE_MAC_MERGE_THRESHOLD
        ):
            merge_reason = "shared_spatial_mac"
        if merge_reason is None:
            groups.append([peak])
            group_links.append([])
        else:
            groups[-1].append(peak)
            group_links[-1].append(
                {
                    "frequency_hz": float(primary_frequencies[peak]),
                    "matched_frequency_hz": float(
                        primary_frequencies[best_group_peak]
                    ),
                    "adjacent_frequency_gap_hz": adjacent_gap,
                    "matched_frequency_gap_hz": matched_gap,
                    "shared_two_view_spatial_mac": best_shape_mac,
                    "reason": merge_reason,
                }
            )
    for group, merge_links in zip(groups, group_links):
        group_indices = np.asarray(group, dtype=np.int64)
        representative = int(
            group_indices[np.argmax(shared_support[group_indices])]
        )
        bands.append(
            {
                "label": "candidate_frequency_band",
                "peak_indices": group,
                "representative_index": representative,
                "start_hz": max(
                    min_freq_hz,
                    SLOW_SWAY_CUTOFF_HZ,
                    float(primary_frequencies[group[0]])
                    - 0.5 * effective_bandwidth,
                ),
                "end_hz": min(
                    max_freq_hz,
                    float(primary_frequencies[group[-1]])
                    + 0.5 * effective_bandwidth,
                ),
                "merge_links": merge_links,
            }
        )

    ranked_indices = sorted(
        range(len(bands)),
        key=lambda index: (
            -float(shared_support[bands[index]["representative_index"]]),
            float(primary_frequencies[bands[index]["representative_index"]]),
        ),
    )
    support_rank = {band_index: rank + 1 for rank, band_index in enumerate(ranked_indices)}
    for band_index, band in enumerate(bands):
        representative = int(band.pop("representative_index"))
        frequency = float(primary_frequencies[representative])
        per_view: list[dict[str, Any]] = []
        for analysis in view_analyses:
            metrics = analysis.configurations[0]
            cross_index = _band_cross_resolution_index(
                analysis,
                frequency,
                float(band["start_hz"]),
                float(band["end_hz"]),
            )
            per_view.append(
                {
                    "view_id": analysis.view_id,
                    "normalized_leading_power": float(
                        metrics.normalized_leading_power[representative]
                    ),
                    "local_prominence": float(
                        metrics.local_prominence[representative]
                    ),
                    "dominance": float(metrics.dominance[representative]),
                    "eigenvalue_separation": float(
                        metrics.separation[representative]
                    ),
                    "split_half_mac": float(
                        metrics.split_half_mac[representative]
                    ),
                    "loo_mac_median": float(
                        metrics.loo_mac_median[representative]
                    ),
                    "loo_mac_min": float(metrics.loo_mac_min[representative]),
                    "nonoverlap_mac_median": (
                        float(metrics.nonoverlap_mac_median[representative])
                        if metrics.nonoverlap_pair_count > 0
                        else None
                    ),
                    "nonoverlap_pair_count": metrics.nonoverlap_pair_count,
                    "window_power_cv": float(
                        metrics.window_power_cv[representative]
                    ),
                    "cross_resolution_frequency_hz": (
                        float(analysis.cross_resolution_frequency_hz[cross_index])
                        if cross_index is not None
                        else None
                    ),
                    "cross_resolution_mac": (
                        float(analysis.cross_resolution_mac[cross_index])
                        if cross_index is not None
                        else None
                    ),
                }
            )
        band.update(
            {
                "band_index": band_index,
                "support_rank": support_rank[band_index],
                "representative_frequency_hz": frequency,
                "shared_two_view_support": float(shared_support[representative]),
                "peak_frequencies_hz": [
                    float(primary_frequencies[int(index)])
                    for index in band["peak_indices"]
                ],
                "contains_shape_redundancy_merge": any(
                    link["reason"] == "shared_spatial_mac"
                    for link in band["merge_links"]
                ),
                "per_view": per_view,
            }
        )
    return bands


def _representative_indices(bands: Sequence[dict[str, Any]], frequencies: np.ndarray) -> np.ndarray:
    indices: list[int] = []
    for band in bands:
        frequency = float(band["representative_frequency_hz"])
        index = int(np.argmin(np.abs(frequencies - frequency)))
        if abs(float(frequencies[index]) - frequency) > 1e-10:
            raise ValueError("Frequency band representative is not a primary FFT bin")
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def _reconstruct_leading_modes(
    analysis: _ViewAnalysis,
    representative_indices: np.ndarray,
) -> np.ndarray:
    metrics = analysis.configurations[0]
    spec = metrics.spec
    modes = np.zeros(
        (representative_indices.size, analysis.pixels.shape[0], 2),
        dtype=np.complex64,
    )
    for pixel_start in range(0, analysis.pixels.shape[0], _PIXEL_CHUNK_SIZE):
        pixel_end = min(pixel_start + _PIXEL_CHUNK_SIZE, analysis.pixels.shape[0])
        current = analysis.pixels[pixel_start:pixel_end]
        x = current[:, 0]
        y = current[:, 1]
        raw_u = np.asarray(analysis.cache.flow_u[:, y, x], dtype=np.float32)
        raw_v = np.asarray(analysis.cache.flow_v[:, y, x], dtype=np.float32)
        snapshots = _window_snapshots(raw_u, raw_v, spec)
        for mode_index, frequency_index in enumerate(representative_indices):
            leading_value = float(
                metrics.top_eigenvalues[int(frequency_index), 0]
            )
            denominator = math.sqrt(spec.starts.size * leading_value)
            if denominator <= np.finfo(np.float64).eps:
                continue
            vector = np.einsum(
                "wm,w->m",
                snapshots[int(frequency_index)],
                metrics.leading_vectors[int(frequency_index)],
                optimize=True,
            ) / denominator
            modes[mode_index, pixel_start:pixel_end, 0] = vector[0::2]
            modes[mode_index, pixel_start:pixel_end, 1] = vector[1::2]
    for mode_index in range(modes.shape[0]):
        flattened = modes[mode_index].reshape(-1)
        norm = float(np.linalg.norm(flattened.astype(np.complex128)))
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
            raise ValueError("Representative SPOD mode has zero or invalid norm")
        modes[mode_index] /= np.float32(norm)
        flattened = modes[mode_index].reshape(-1)
        gauge_index = int(np.argmax(np.abs(flattened)))
        gauge_value = flattened[gauge_index]
        phase = np.complex64(np.exp(-1j * np.angle(gauge_value)))
        modes[mode_index] *= phase
        if float(np.real(modes[mode_index].reshape(-1)[gauge_index])) < 0.0:
            modes[mode_index] *= np.complex64(-1.0)
    if not np.isfinite(modes).all():
        raise ValueError("Representative SPOD modes are non-finite")
    return modes


def _configuration_key(window_length: int) -> str:
    return f"w{window_length}"


def _build_diagnostic_arrays(
    view_analyses: Sequence[_ViewAnalysis],
    window_lengths: tuple[int, int],
    overlap_fraction: float,
    shared_support: np.ndarray,
    candidate_peak_indices: np.ndarray,
    per_view_spatial_mac: np.ndarray,
    shared_spatial_mac: np.ndarray,
    bands: Sequence[dict[str, Any]],
) -> dict[str, np.ndarray]:
    primary_frequencies = view_analyses[0].configurations[0].spec.frequencies_hz
    endpoint_maxima = np.array(
        [
            shared_support.size > 1 and shared_support[0] > shared_support[1],
            shared_support.size > 1 and shared_support[-1] > shared_support[-2],
        ],
        dtype=bool,
    )
    arrays: dict[str, np.ndarray] = {
        "format": np.array(MODAL_SPOD_ANALYSIS_FORMAT),
        "version": np.array(MODAL_SPOD_ANALYSIS_VERSION, dtype=np.int32),
        "view_ids": np.asarray([analysis.view_id for analysis in view_analyses]),
        "window_lengths": np.asarray(window_lengths, dtype=np.int64),
        "overlap_fraction": np.array(overlap_fraction, dtype=np.float64),
        "slow_sway_cutoff_hz": np.array(SLOW_SWAY_CUTOFF_HZ, dtype=np.float64),
        "candidate_pixel_count": np.asarray(
            [analysis.pixels.shape[0] for analysis in view_analyses],
            dtype=np.int64,
        ),
        "frame_count": np.asarray(
            [analysis.cache.flow_u.shape[0] for analysis in view_analyses],
            dtype=np.int64,
        ),
        "shared_two_view_support": shared_support.astype(np.float64, copy=False),
        "candidate_peak_frequency_indices": candidate_peak_indices.astype(
            np.int64, copy=False
        ),
        "candidate_peak_frequencies_hz": primary_frequencies[
            candidate_peak_indices
        ].astype(np.float64, copy=False),
        "candidate_peak_spatial_mac_per_view": per_view_spatial_mac.astype(
            np.float64, copy=False
        ),
        "candidate_peak_spatial_mac_shared_two_view": shared_spatial_mac.astype(
            np.float64, copy=False
        ),
        "search_endpoint_frequencies_hz": primary_frequencies[[0, -1]].astype(
            np.float64, copy=False
        ),
        "search_endpoint_one_sided_maximum": endpoint_maxima,
        "ordinary_endpoint_peaks_excluded": np.array(True),
        "hann_effective_bandwidth_bins": np.array(
            _HANN_EFFECTIVE_BANDWIDTH_BINS, dtype=np.float64
        ),
        "shape_mac_merge_threshold": np.array(
            _SHAPE_MAC_MERGE_THRESHOLD, dtype=np.float64
        ),
        "shape_mac_max_gap_multiplier": np.array(
            _SHAPE_MAC_MAX_GAP_MULTIPLIER, dtype=np.float64
        ),
        "band_representative_frequencies_hz": np.asarray(
            [band["representative_frequency_hz"] for band in bands],
            dtype=np.float64,
        ),
        "band_support_rank": np.asarray(
            [band["support_rank"] for band in bands], dtype=np.int64
        ),
    }
    view_count = len(view_analyses)
    for configuration_index, window_length in enumerate(window_lengths):
        key = _configuration_key(window_length)
        reference_frequencies = view_analyses[0].configurations[
            configuration_index
        ].spec.frequencies_hz
        for analysis in view_analyses[1:]:
            frequencies = analysis.configurations[
                configuration_index
            ].spec.frequencies_hz
            if not _frequency_axes_match(frequencies, reference_frequencies):
                raise ValueError("SPOD frequency axes differ between views")
        max_segments = max(
            analysis.configurations[configuration_index].spec.starts.size
            for analysis in view_analyses
        )
        frequency_count = reference_frequencies.size
        starts = np.full((view_count, max_segments), -1, dtype=np.int64)
        segment_valid = np.zeros((view_count, max_segments), dtype=bool)
        window_power = np.zeros(
            (view_count, frequency_count, max_segments), dtype=np.float64
        )
        grams = np.zeros(
            (view_count, frequency_count, max_segments, max_segments),
            dtype=np.complex128,
        )
        leading_vectors = np.zeros(
            (view_count, frequency_count, max_segments), dtype=np.complex128
        )
        for view_index, analysis in enumerate(view_analyses):
            metrics = analysis.configurations[configuration_index]
            segment_count = metrics.spec.starts.size
            starts[view_index, :segment_count] = metrics.spec.starts
            segment_valid[view_index, :segment_count] = True
            window_power[view_index, :, :segment_count] = metrics.window_power
            grams[view_index, :, :segment_count, :segment_count] = metrics.gram
            leading_vectors[view_index, :, :segment_count] = metrics.leading_vectors
        arrays.update(
            {
                f"frequency_hz_{key}": reference_frequencies.astype(
                    np.float64, copy=False
                ),
                f"frequency_resolution_hz_{key}": np.array(
                    view_analyses[0].cache.fps / window_length,
                    dtype=np.float64,
                ),
                f"hop_frames_{key}": np.array(
                    view_analyses[0].configurations[
                        configuration_index
                    ].spec.hop,
                    dtype=np.int64,
                ),
                f"segment_count_{key}": np.asarray(
                    [
                        analysis.configurations[
                            configuration_index
                        ].spec.starts.size
                        for analysis in view_analyses
                    ],
                    dtype=np.int64,
                ),
                f"window_starts_{key}": starts,
                f"window_valid_{key}": segment_valid,
                f"window_power_{key}": window_power,
                f"snapshot_gram_{key}": grams,
                f"leading_snapshot_vector_{key}": leading_vectors,
                f"top_eigenvalues_{key}": np.stack(
                    [
                        analysis.configurations[
                            configuration_index
                        ].top_eigenvalues
                        for analysis in view_analyses
                    ]
                ),
                f"total_power_{key}": np.stack(
                    [
                        analysis.configurations[configuration_index].total_power
                        for analysis in view_analyses
                    ]
                ),
                f"dominance_{key}": np.stack(
                    [
                        analysis.configurations[configuration_index].dominance
                        for analysis in view_analyses
                    ]
                ),
                f"eigenvalue_separation_{key}": np.stack(
                    [
                        analysis.configurations[
                            configuration_index
                        ].separation
                        for analysis in view_analyses
                    ]
                ),
                f"split_half_mac_{key}": np.stack(
                    [
                        analysis.configurations[
                            configuration_index
                        ].split_half_mac
                        for analysis in view_analyses
                    ]
                ),
                f"loo_mac_median_{key}": np.stack(
                    [
                        analysis.configurations[
                            configuration_index
                        ].loo_mac_median
                        for analysis in view_analyses
                    ]
                ),
                f"loo_mac_min_{key}": np.stack(
                    [
                        analysis.configurations[configuration_index].loo_mac_min
                        for analysis in view_analyses
                    ]
                ),
                f"nonoverlap_mac_median_{key}": np.stack(
                    [
                        analysis.configurations[
                            configuration_index
                        ].nonoverlap_mac_median
                        for analysis in view_analyses
                    ]
                ),
                f"nonoverlap_pair_count_{key}": np.asarray(
                    [
                        analysis.configurations[
                            configuration_index
                        ].nonoverlap_pair_count
                        for analysis in view_analyses
                    ],
                    dtype=np.int64,
                ),
                f"window_power_cv_{key}": np.stack(
                    [
                        analysis.configurations[
                            configuration_index
                        ].window_power_cv
                        for analysis in view_analyses
                    ]
                ),
                f"normalized_leading_power_{key}": np.stack(
                    [
                        analysis.configurations[
                            configuration_index
                        ].normalized_leading_power
                        for analysis in view_analyses
                    ]
                ),
                f"local_prominence_{key}": np.stack(
                    [
                        analysis.configurations[
                            configuration_index
                        ].local_prominence
                        for analysis in view_analyses
                    ]
                ),
            }
        )
    cross_frequencies = view_analyses[0].cross_resolution_frequency_hz
    for analysis in view_analyses[1:]:
        if not _frequency_axes_match(
            analysis.cross_resolution_frequency_hz, cross_frequencies
        ):
            raise ValueError("Cross-resolution frequency axes differ between views")
    arrays["cross_resolution_frequency_hz"] = cross_frequencies
    arrays["cross_resolution_mac"] = np.stack(
        [analysis.cross_resolution_mac for analysis in view_analyses]
    )
    for name, array in arrays.items():
        if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
            raise ValueError(f"Diagnostic array {name!r} is non-finite")
    return arrays


def _plot_spectra(
    path: Path,
    view_analyses: Sequence[_ViewAnalysis],
    shared_support: np.ndarray,
    bands: Sequence[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(
        len(view_analyses) + 1,
        1,
        figsize=(12.0, 3.0 * (len(view_analyses) + 1)),
        squeeze=False,
    )
    for view_index, analysis in enumerate(view_analyses):
        axis = axes[view_index, 0]
        for metrics in analysis.configurations:
            axis.plot(
                metrics.spec.frequencies_hz,
                metrics.normalized_leading_power,
                label=f"{metrics.spec.length}-frame",
            )
        axis.axvspan(
            analysis.configurations[0].spec.frequencies_hz[0],
            SLOW_SWAY_CUTOFF_HZ,
            color="tab:gray",
            alpha=0.15,
        )
        axis.set_ylabel("normalized lambda1")
        axis.set_title(analysis.view_id)
        axis.grid(alpha=0.25)
        axis.legend()
    shared_axis = axes[-1, 0]
    primary_frequencies = view_analyses[0].configurations[0].spec.frequencies_hz
    shared_axis.plot(primary_frequencies, shared_support, color="black")
    shared_axis.axvspan(
        primary_frequencies[0],
        SLOW_SWAY_CUTOFF_HZ,
        color="tab:gray",
        alpha=0.15,
        label=SLOW_SWAY_LABEL,
    )
    for band in bands:
        shared_axis.axvline(
            band["representative_frequency_hz"], color="tab:red", alpha=0.3
        )
    shared_axis.set_xlabel("Frequency (Hz)")
    shared_axis.set_ylabel("second-highest view support")
    shared_axis.set_title("Shared scalar frequency support (no cross-view phase)")
    shared_axis.grid(alpha=0.25)
    shared_axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_stability(
    path: Path,
    view_analyses: Sequence[_ViewAnalysis],
) -> None:
    figure, axes = plt.subplots(
        len(view_analyses),
        3,
        figsize=(16.0, 3.5 * len(view_analyses)),
        squeeze=False,
    )
    for view_index, analysis in enumerate(view_analyses):
        primary = analysis.configurations[0]
        axes[view_index, 0].plot(
            primary.spec.frequencies_hz, primary.dominance, label="dominance"
        )
        axes[view_index, 0].plot(
            primary.spec.frequencies_hz,
            primary.split_half_mac,
            label="split-half MAC",
        )
        axes[view_index, 0].set_title(f"{analysis.view_id}: primary stability")
        axes[view_index, 0].set_ylim(-0.02, 1.02)
        axes[view_index, 0].legend()
        axes[view_index, 1].plot(
            primary.spec.frequencies_hz,
            primary.loo_mac_median,
            label="LOO median",
        )
        axes[view_index, 1].plot(
            primary.spec.frequencies_hz,
            primary.loo_mac_min,
            label="LOO minimum",
        )
        axes[view_index, 1].set_title(f"{analysis.view_id}: leave-one-window-out")
        axes[view_index, 1].set_ylim(-0.02, 1.02)
        axes[view_index, 1].legend()
        axes[view_index, 2].plot(
            analysis.cross_resolution_frequency_hz,
            analysis.cross_resolution_mac,
            label="512/256 MAC",
        )
        axes[view_index, 2].set_title(f"{analysis.view_id}: cross-resolution")
        axes[view_index, 2].set_ylim(-0.02, 1.02)
        axes[view_index, 2].legend()
        for axis in axes[view_index]:
            axis.axvspan(
                primary.spec.frequencies_hz[0],
                SLOW_SWAY_CUTOFF_HZ,
                color="tab:gray",
                alpha=0.15,
            )
            axis.set_xlabel("Frequency (Hz)")
            axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_mode_previews(
    path: Path,
    analysis: _ViewAnalysis,
    modes: np.ndarray,
    frequencies_hz: np.ndarray,
) -> None:
    column_count = 4
    row_count = int(math.ceil(modes.shape[0] / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.0 * column_count, 3.0 * row_count),
        squeeze=False,
    )
    height, width = analysis.cache.flow_u.shape[1:]
    x = analysis.pixels[:, 0]
    y = analysis.pixels[:, 1]
    for mode_index, axis in enumerate(axes.reshape(-1)):
        if mode_index >= modes.shape[0]:
            axis.axis("off")
            continue
        magnitude = np.sqrt(
            np.abs(modes[mode_index, :, 0]) ** 2
            + np.abs(modes[mode_index, :, 1]) ** 2
        ).astype(np.float32)
        image = np.full((height, width), np.nan, dtype=np.float32)
        image[y, x] = magnitude
        axis.imshow(image, cmap="magma", interpolation="nearest")
        axis.set_title(f"{float(frequencies_hz[mode_index]):.6g} Hz")
        axis.axis("off")
    figure.suptitle(f"{analysis.view_id} leading SPOD mode magnitude")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_mode_previews(
    output_path: Path,
    view_analyses: Sequence[_ViewAnalysis],
    bands: Sequence[dict[str, Any]],
    candidate_peak_indices: np.ndarray,
    candidate_modes_by_view: Sequence[np.ndarray],
) -> list[str]:
    preview_path = output_path / MODE_PREVIEW_DIRNAME
    preview_path.mkdir(parents=True, exist_ok=False)
    frequencies = np.asarray(
        [band["representative_frequency_hz"] for band in bands],
        dtype=np.float64,
    )
    band_indices = np.asarray([band["band_index"] for band in bands], dtype=np.int64)
    candidate_position = {
        int(frequency_index): position
        for position, frequency_index in enumerate(candidate_peak_indices)
    }
    filenames: list[str] = []
    if len(candidate_modes_by_view) != len(view_analyses):
        raise ValueError("Candidate SPOD modes do not match the analyzed views")
    for analysis, candidate_modes in zip(view_analyses, candidate_modes_by_view):
        if Path(analysis.view_id).name != analysis.view_id:
            raise ValueError(f"View ID is not safe for an output filename: {analysis.view_id!r}")
        representative_indices = _representative_indices(
            bands, analysis.configurations[0].spec.frequencies_hz
        )
        try:
            representative_positions = np.asarray(
                [candidate_position[int(index)] for index in representative_indices],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "Frequency band representative is not a reconstructed candidate peak"
            ) from error
        modes = candidate_modes[representative_positions]
        npz_name = f"{analysis.view_id}_spod_modes.npz"
        np.savez_compressed(
            preview_path / npz_name,
            format=np.array("modal_spod_mode_previews"),
            version=np.array(1, dtype=np.int32),
            view_id=np.array(analysis.view_id),
            window_length=np.array(
                analysis.configurations[0].spec.length, dtype=np.int64
            ),
            candidate_pixels_xy=analysis.pixels.astype(np.int64, copy=False),
            band_indices=band_indices,
            frequencies_hz=frequencies,
            mode_u=modes[:, :, 0],
            mode_v=modes[:, :, 1],
        )
        plot_name = f"{analysis.view_id}_spod_mode_previews.png"
        _plot_mode_previews(preview_path / plot_name, analysis, modes, frequencies)
        filenames.extend(
            [
                f"{MODE_PREVIEW_DIRNAME}/{npz_name}",
                f"{MODE_PREVIEW_DIRNAME}/{plot_name}",
            ]
        )
    return filenames


def _validate_written_output(path: Path, view_ids: Sequence[str]) -> None:
    required = (
        SUMMARY_FILENAME,
        DIAGNOSTICS_FILENAME,
        BANDS_FILENAME,
        SPECTRA_PLOT_FILENAME,
        STABILITY_PLOT_FILENAME,
    )
    for filename in required:
        current = path / filename
        if not current.is_file() or current.stat().st_size == 0:
            raise ValueError(f"SPOD output is missing or empty: {current}")
    with (path / SUMMARY_FILENAME).open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if (
        summary.get("format") != MODAL_SPOD_ANALYSIS_FORMAT
        or summary.get("version") != MODAL_SPOD_ANALYSIS_VERSION
    ):
        raise ValueError("Written SPOD summary has an invalid format or version")
    with (path / BANDS_FILENAME).open("r", encoding="utf-8") as file:
        bands = json.load(file)
    if (
        bands.get("format") != "modal_spod_frequency_bands"
        or bands.get("version") != 1
        or not isinstance(bands.get("bands"), list)
        or len(bands["bands"]) != summary.get("frequency_band_count")
    ):
        raise ValueError("Written SPOD frequency bands are invalid")
    with np.load(path / DIAGNOSTICS_FILENAME, allow_pickle=False) as archive:
        if str(np.asarray(archive["format"]).item()) != MODAL_SPOD_ANALYSIS_FORMAT:
            raise ValueError("Written SPOD diagnostics have an invalid format")
        if int(np.asarray(archive["version"]).item()) != MODAL_SPOD_ANALYSIS_VERSION:
            raise ValueError("Written SPOD diagnostics have an invalid version")
        for key in (
            "candidate_peak_frequencies_hz",
            "candidate_peak_spatial_mac_per_view",
            "candidate_peak_spatial_mac_shared_two_view",
            "band_representative_frequencies_hz",
        ):
            values = np.asarray(archive[key])
            if values.dtype.kind not in {"f", "c"} or not np.isfinite(values).all():
                raise ValueError(f"Written SPOD diagnostic {key!r} is invalid")
    for view_id in view_ids:
        preview = path / MODE_PREVIEW_DIRNAME / f"{view_id}_spod_modes.npz"
        if not preview.is_file() or preview.stat().st_size == 0:
            raise ValueError(f"SPOD mode preview is missing: {preview}")
        with np.load(preview, allow_pickle=False) as archive:
            if (
                str(np.asarray(archive["format"]).item())
                != "modal_spod_mode_previews"
                or int(np.asarray(archive["version"]).item()) != 1
                or str(np.asarray(archive["view_id"]).item()) != view_id
            ):
                raise ValueError(f"SPOD mode preview metadata is invalid: {preview}")
            mode_u = np.asarray(archive["mode_u"])
            mode_v = np.asarray(archive["mode_v"])
            pixels = np.asarray(archive["candidate_pixels_xy"])
            frequencies = np.asarray(archive["frequencies_hz"])
            band_indices = np.asarray(archive["band_indices"])
            if (
                mode_u.shape != mode_v.shape
                or mode_u.ndim != 2
                or pixels.shape != (mode_u.shape[1], 2)
                or frequencies.shape != (mode_u.shape[0],)
                or band_indices.shape != (mode_u.shape[0],)
                or not np.isfinite(mode_u).all()
                or not np.isfinite(mode_v).all()
                or not np.isfinite(frequencies).all()
            ):
                raise ValueError(f"SPOD mode preview arrays are invalid: {preview}")
        preview_plot = (
            path / MODE_PREVIEW_DIRNAME / f"{view_id}_spod_mode_previews.png"
        )
        if not preview_plot.is_file() or preview_plot.stat().st_size == 0:
            raise ValueError(f"SPOD mode preview plot is missing: {preview_plot}")


def run_modal_spod_analysis(
    *,
    flow_cache_specs: Sequence[str],
    modal_manifest_path: str | Path,
    modal_frame_map_path: str | Path,
    output_dir: str | Path,
    window_lengths: Sequence[int],
    overlap_fraction: float,
    min_freq_hz: float,
    max_freq_hz: float,
    pixel_stride: int,
) -> ModalSpodAnalysisResult:
    """Analyze per-view cached flow with windowed snapshot FDD/SPOD."""

    total_start = time.perf_counter()
    (
        lengths,
        overlap,
        minimum_frequency,
        maximum_frequency,
        stride,
    ) = _validate_settings(
        window_lengths,
        overlap_fraction,
        min_freq_hz,
        max_freq_hz,
        pixel_stride,
    )
    output_path = Path(output_dir).expanduser()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_map = _load_frame_map(modal_frame_map_path)
    manifest = _load_manifest(modal_manifest_path)
    topology = manifest.topology
    manifest_source = manifest.path
    caches = _load_and_validate_caches(frame_map, topology, flow_cache_specs)
    del manifest
    if len(caches) < 2:
        raise ValueError("Windowed SPOD requires at least two views")
    fps = float(caches[0].fps)
    for view_id, cache in zip(frame_map.view_ids, caches):
        if not math.isclose(float(cache.fps), fps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Windowed SPOD currently requires identical FPS across views; "
                f"{view_id!r} has {cache.fps}, expected {fps}"
            )
        if maximum_frequency > 0.5 * float(cache.fps) + 1e-12:
            raise ValueError(
                f"Maximum frequency exceeds Nyquist for view {view_id!r}"
            )

    validation_seconds = float(time.perf_counter() - total_start)
    view_analyses: list[_ViewAnalysis] = []
    view_seconds: list[float] = []
    for view_index, (view_id, cache) in enumerate(zip(frame_map.view_ids, caches)):
        view_start = time.perf_counter()
        pixels = _candidate_pixels(topology, view_index, cache, stride)
        analysis = _analyze_view(
            view_id=view_id,
            cache=cache,
            pixels=pixels,
            window_lengths=lengths,
            overlap_fraction=overlap,
            min_freq_hz=minimum_frequency,
            max_freq_hz=maximum_frequency,
        )
        elapsed = float(time.perf_counter() - view_start)
        view_analyses.append(analysis)
        view_seconds.append(elapsed)
        print(f"Finished {view_id} windowed SPOD in {elapsed:.3f}s", flush=True)

    primary_frequencies = view_analyses[0].configurations[0].spec.frequencies_hz
    secondary_frequencies = view_analyses[0].configurations[1].spec.frequencies_hz
    for analysis in view_analyses[1:]:
        if not _frequency_axes_match(
            analysis.configurations[0].spec.frequencies_hz,
            primary_frequencies,
        ) or not _frequency_axes_match(
            analysis.configurations[1].spec.frequencies_hz,
            secondary_frequencies,
        ):
            raise ValueError("Native SPOD frequency axes differ between views")

    aggregation_start = time.perf_counter()
    shared_support = _shared_support(view_analyses)
    candidate_peak_indices = _candidate_peak_indices(
        primary_frequencies, shared_support
    )
    if candidate_peak_indices.size == 0:
        raise ValueError(
            "Windowed SPOD found no interior frequency peaks; search endpoints are "
            "excluded because their support may be clipped by the requested range"
        )
    candidate_modes_by_view = tuple(
        _reconstruct_leading_modes(analysis, candidate_peak_indices)
        for analysis in view_analyses
    )
    per_view_spatial_mac, shared_spatial_mac = _shared_spatial_mode_mac(
        candidate_modes_by_view
    )
    bands = _build_frequency_bands(
        view_analyses,
        shared_support,
        candidate_peak_indices,
        shared_spatial_mac,
        minimum_frequency,
        maximum_frequency,
    )
    if not bands:
        raise ValueError("Windowed SPOD did not produce any candidate frequency bands")
    diagnostics = _build_diagnostic_arrays(
        view_analyses,
        lengths,
        overlap,
        shared_support,
        candidate_peak_indices,
        per_view_spatial_mac,
        shared_spatial_mac,
        bands,
    )
    aggregation_seconds = float(time.perf_counter() - aggregation_start)

    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        write_start = time.perf_counter()
        np.savez_compressed(temp_path / DIAGNOSTICS_FILENAME, **diagnostics)
        with (temp_path / BANDS_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "format": "modal_spod_frequency_bands",
                    "version": 1,
                    "interpretation": (
                        "Bands rank gauge-invariant scalar support from asynchronous "
                        "views. They are diagnostic candidates, not solver-ready natural "
                        "mode declarations."
                    ),
                    "slow_sway_cutoff_hz": SLOW_SWAY_CUTOFF_HZ,
                    "primary_window_length": lengths[0],
                    "primary_frequency_resolution_hz": fps / lengths[0],
                    "ordinary_endpoint_peaks_excluded": True,
                    "search_endpoint_one_sided_maximum": [
                        bool(
                            shared_support.size > 1
                            and shared_support[0] > shared_support[1]
                        ),
                        bool(
                            shared_support.size > 1
                            and shared_support[-1] > shared_support[-2]
                        ),
                    ],
                    "band_merge_settings": {
                        "hann_effective_bandwidth_bins": (
                            _HANN_EFFECTIVE_BANDWIDTH_BINS
                        ),
                        "shape_mac_threshold": _SHAPE_MAC_MERGE_THRESHOLD,
                        "shape_mac_max_gap_multiplier": (
                            _SHAPE_MAC_MAX_GAP_MULTIPLIER
                        ),
                    },
                    "bands": bands,
                },
                file,
                indent=2,
                allow_nan=False,
            )
            file.write("\n")
        _plot_spectra(
            temp_path / SPECTRA_PLOT_FILENAME,
            view_analyses,
            shared_support,
            bands,
        )
        _plot_stability(temp_path / STABILITY_PLOT_FILENAME, view_analyses)
        preview_files = _write_mode_previews(
            temp_path,
            view_analyses,
            bands,
            candidate_peak_indices,
            candidate_modes_by_view,
        )
        write_seconds = float(time.perf_counter() - write_start)

        views_summary: list[dict[str, Any]] = []
        for analysis, elapsed in zip(view_analyses, view_seconds):
            configurations = []
            for metrics in analysis.configurations:
                configurations.append(
                    {
                        "window_length": metrics.spec.length,
                        "hop_frames": metrics.spec.hop,
                        "segment_count": int(metrics.spec.starts.size),
                        "frequency_resolution_hz": fps / metrics.spec.length,
                        "native_frequency_count": int(
                            metrics.spec.frequencies_hz.size
                        ),
                        "frequency_range_hz": [
                            float(metrics.spec.frequencies_hz[0]),
                            float(metrics.spec.frequencies_hz[-1]),
                        ],
                        "nonoverlap_pair_count": metrics.nonoverlap_pair_count,
                    }
                )
            views_summary.append(
                {
                    "view_id": analysis.view_id,
                    "frame_count": int(analysis.cache.flow_u.shape[0]),
                    "candidate_pixel_count": int(analysis.pixels.shape[0]),
                    "analysis_seconds": elapsed,
                    "configurations": configurations,
                }
            )

        summary = {
            "format": MODAL_SPOD_ANALYSIS_FORMAT,
            "version": MODAL_SPOD_ANALYSIS_VERSION,
            "method": "per_view_windowed_snapshot_spod_v1",
            "interpretation_note": (
                "Each asynchronous view is analyzed independently. Only scalar, "
                "gauge-invariant frequency confidence is aggregated across views; no "
                "cross-view complex phase, CSD, or raw 2D mode-shape comparison is "
                "performed. Within-view spatial MAC values are aggregated only as "
                "second-highest scalar redundancy support."
            ),
            "slow_sway_interpretation": (
                "Frequencies below 0.2 Hz are retained as one physical slow-sway band. "
                "The label does not classify them as camera drift or noise."
            ),
            "settings": {
                "window_lengths": list(lengths),
                "overlap_fraction": overlap,
                "hop_frames": [
                    int(round(length * (1.0 - overlap))) for length in lengths
                ],
                "window": "hann",
                "detrend": "per_window_mean",
                "frequency_sampling": "native_rfft_bins",
                "min_freq_hz": minimum_frequency,
                "max_freq_hz": maximum_frequency,
                "pixel_stride": stride,
                "pixel_chunk_size": _PIXEL_CHUNK_SIZE,
                "slow_sway_cutoff_hz": SLOW_SWAY_CUTOFF_HZ,
                "cross_view_support": "second_highest_normalized_leading_power",
                "ordinary_endpoint_peaks_excluded": True,
                "hann_effective_bandwidth_bins": (
                    _HANN_EFFECTIVE_BANDWIDTH_BINS
                ),
                "shape_mac_merge_threshold": _SHAPE_MAC_MERGE_THRESHOLD,
                "shape_mac_max_gap_multiplier": (
                    _SHAPE_MAC_MAX_GAP_MULTIPLIER
                ),
            },
            "views": views_summary,
            "frequency_band_count": len(bands),
            "frequency_bands": bands,
            "source": {
                "modal_manifest": str(manifest_source),
                "modal_frame_map": str(frame_map.path),
                "flow_caches": [str(cache.path.resolve()) for cache in caches],
            },
            "timings_seconds": {
                "input_validation": validation_seconds,
                "view_analysis": view_seconds,
                "aggregation": aggregation_seconds,
                "write_and_previews": write_seconds,
                "total": float(time.perf_counter() - total_start),
            },
            "files": {
                "diagnostics": DIAGNOSTICS_FILENAME,
                "frequency_bands": BANDS_FILENAME,
                "spectra_plot": SPECTRA_PLOT_FILENAME,
                "stability_plot": STABILITY_PLOT_FILENAME,
                "mode_previews": preview_files,
            },
        }
        with (temp_path / SUMMARY_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, allow_nan=False)
            file.write("\n")
        _validate_written_output(temp_path, frame_map.view_ids)
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError(
                f"Output directory appeared during analysis: {output_path}"
            )
        os.replace(temp_path, output_path)
    except BaseException:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise

    return ModalSpodAnalysisResult(
        path=output_path,
        summary_path=output_path / SUMMARY_FILENAME,
        diagnostics_path=output_path / DIAGNOSTICS_FILENAME,
        bands_path=output_path / BANDS_FILENAME,
        spectra_plot_path=output_path / SPECTRA_PLOT_FILENAME,
        stability_plot_path=output_path / STABILITY_PLOT_FILENAME,
    )
