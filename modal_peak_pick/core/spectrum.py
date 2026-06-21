from __future__ import annotations

from typing import Optional

import numpy as np


def _hann(num_samples: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(num_samples) / max(1, num_samples - 1))


def fft_over_time(
    u: np.ndarray,
    v: np.ndarray,
    fps: float,
    detrend: bool = True,
    window: str = "hann",
    fft_block_w: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temporal rFFT for per-pixel image-plane displacement signals."""
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError("u and v must have the same shape [T,H,W].")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if fft_block_w <= 0:
        raise ValueError("fft_block_w must be positive.")

    num_samples, h, w = u.shape
    if num_samples < 3:
        raise ValueError("Need at least 3 temporal samples for FFT.")

    num_freqs = num_samples // 2 + 1
    U = np.empty((num_freqs, h, w), dtype=np.complex64)
    V = np.empty((num_freqs, h, w), dtype=np.complex64)

    temporal_window = None
    if window.lower() == "hann":
        temporal_window = _hann(num_samples).astype(np.float32)[:, None, None]
    elif window.lower() not in {"none", "boxcar", "rect"}:
        raise ValueError(f"Unsupported window: {window}")

    block_w = min(int(fft_block_w), w)
    for x0 in range(0, w, block_w):
        x1 = min(w, x0 + block_w)
        ub = np.asarray(u[:, :, x0:x1], dtype=np.float32)
        vb = np.asarray(v[:, :, x0:x1], dtype=np.float32)

        if detrend:
            ub = ub - ub.mean(axis=0, keepdims=True, dtype=np.float32)
            vb = vb - vb.mean(axis=0, keepdims=True, dtype=np.float32)
        else:
            ub = ub.copy()
            vb = vb.copy()

        if temporal_window is not None:
            ub *= temporal_window
            vb *= temporal_window

        U[:, :, x0:x1] = np.fft.rfft(ub, axis=0).astype(np.complex64, copy=False)
        V[:, :, x0:x1] = np.fft.rfft(vb, axis=0).astype(np.complex64, copy=False)

    freqs_hz = np.fft.rfftfreq(num_samples, d=1.0 / fps).astype(np.float32)
    return freqs_hz, U, V


def amplitude_map(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    return np.sqrt((np.abs(U) ** 2 + np.abs(V) ** 2).astype(np.float32))


def global_power_spectrum(
    U: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mean modal amplitude over the mask for each frequency bin."""
    if U.shape != V.shape or U.ndim != 3:
        raise ValueError("U and V must have the same shape [F,H,W].")

    amp = amplitude_map(U, V)
    if mask is None:
        return amp.mean(axis=(1, 2)).astype(np.float32)

    m = mask.astype(bool)
    if m.shape != U.shape[1:]:
        raise ValueError(f"mask shape {m.shape} does not match field shape {U.shape[1:]}.")
    if int(m.sum()) < 10:
        return amp.mean(axis=(1, 2)).astype(np.float32)
    return amp[:, m].mean(axis=1).astype(np.float32)


def snap_to_local_peak(
    freqs_hz: np.ndarray,
    power_spectrum: np.ndarray,
    f_click_hz: float,
    window_hz: float = 1.0,
) -> float:
    """Snap a requested frequency to the strongest spectrum bin nearby."""
    if freqs_hz.ndim != 1 or power_spectrum.ndim != 1:
        raise ValueError("freqs_hz and power_spectrum must be 1D arrays.")
    if freqs_hz.shape[0] != power_spectrum.shape[0]:
        raise ValueError("freqs_hz and power_spectrum must have the same length.")
    if freqs_hz.shape[0] == 0:
        raise ValueError("Cannot snap with an empty frequency axis.")
    if window_hz < 0:
        raise ValueError("window_hz must be non-negative.")

    lo = float(f_click_hz) - float(window_hz)
    hi = float(f_click_hz) + float(window_hz)
    idx = np.where((freqs_hz >= lo) & (freqs_hz <= hi))[0]
    if idx.size == 0:
        return float(freqs_hz[np.argmin(np.abs(freqs_hz - float(f_click_hz)))])
    best = idx[np.argmax(power_spectrum[idx])]
    return float(freqs_hz[best])

