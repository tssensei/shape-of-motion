from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from modal_peak_pick.core.flow import compute_dense_flow_to_reference, contrast_weighted_smooth
from modal_peak_pick.core.spectrum import fft_over_time, global_power_spectrum, snap_to_local_peak
from modal_peak_pick.core.video_io import load_video_clip


@dataclass
class ModalAnalysisResult:
    frames_gray: np.ndarray
    fps: float
    mask: Optional[np.ndarray]
    frame_ref: np.ndarray
    t_ref_idx: int
    t_ref_s: float
    u: np.ndarray
    v: np.ndarray
    freqs_hz: np.ndarray
    U: np.ndarray
    V: np.ndarray
    power_spectrum: np.ndarray


@dataclass(frozen=True)
class ModeSlice:
    mode_idx: int
    freq_input_hz: float
    freq_selected_hz: float
    U_slice: np.ndarray
    V_slice: np.ndarray


def parse_freqs(text: str) -> list[float]:
    vals: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("At least one frequency is required.")
    return vals


def load_mask(mask_path: Optional[str], h: int, w: int) -> Optional[np.ndarray]:
    if mask_path is None:
        return None

    path = Path(mask_path)
    if path.suffix.lower() == ".npy":
        mask = np.load(str(path))
        if mask.ndim != 2:
            raise ValueError(f"Mask .npy must be 2D, got shape {mask.shape}.")
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask image: {mask_path}")

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    if np.issubdtype(mask.dtype, np.floating):
        return mask > 0.5
    return mask > 127


def run_modal_analysis(
    frames_gray: np.ndarray,
    fps: float,
    mask: Optional[np.ndarray],
    flow_method: str = "farneback",
    no_smooth: bool = False,
    sigma_b: float = 3.0,
    sigma_c: float = 0.0,
    t0: float = 0.0,
) -> ModalAnalysisResult:
    if frames_gray.ndim != 3:
        raise ValueError("frames_gray must be [T,H,W].")
    if frames_gray.shape[0] < 3:
        raise ValueError("Need at least 3 frames for modal analysis.")
    if mask is not None and mask.shape != frames_gray.shape[1:]:
        raise ValueError(f"mask shape {mask.shape} does not match frame shape {frames_gray.shape[1:]}.")

    t_ref_idx = int(frames_gray.shape[0] // 2)
    frame_ref = frames_gray[t_ref_idx]
    u, v = compute_dense_flow_to_reference(frames_gray, method=flow_method)

    if not no_smooth:
        u, v = contrast_weighted_smooth(
            u,
            v,
            frame_ref=frame_ref,
            sigma_b=sigma_b,
            sigma_c=sigma_c,
            mask=mask,
        )

    freqs_hz, U, V = fft_over_time(u, v, fps=fps, detrend=True, window="hann")
    power_spectrum = global_power_spectrum(U, V, mask=mask)
    return ModalAnalysisResult(
        frames_gray=frames_gray,
        fps=float(fps),
        mask=mask,
        frame_ref=frame_ref,
        t_ref_idx=t_ref_idx,
        t_ref_s=float(t0 + (t_ref_idx / fps)),
        u=u,
        v=v,
        freqs_hz=freqs_hz,
        U=U,
        V=V,
        power_spectrum=power_spectrum,
    )


def run_modal_analysis_from_video(
    video_path: str,
    t0: float = 0.0,
    t1: Optional[float] = None,
    resize: Optional[int] = None,
    max_frames: Optional[int] = None,
    flow_method: str = "farneback",
    no_smooth: bool = False,
    sigma_b: float = 3.0,
    sigma_c: float = 0.0,
    mask_path: Optional[str] = None,
) -> ModalAnalysisResult:
    frames_gray, fps = load_video_clip(
        video_path,
        t0=t0,
        t1=t1,
        resize=resize,
        grayscale=True,
        max_frames=max_frames,
    )
    h, w = int(frames_gray.shape[1]), int(frames_gray.shape[2])
    mask = load_mask(mask_path, h, w)
    return run_modal_analysis(
        frames_gray=frames_gray,
        fps=fps,
        mask=mask,
        flow_method=flow_method,
        no_smooth=no_smooth,
        sigma_b=sigma_b,
        sigma_c=sigma_c,
        t0=t0,
    )


def select_mode_slice(
    result: ModalAnalysisResult,
    f_click_hz: float,
    peak_window_hz: float,
    mode_idx: int = 1,
) -> ModeSlice:
    f_selected = snap_to_local_peak(
        result.freqs_hz,
        result.power_spectrum,
        f_click_hz,
        window_hz=peak_window_hz,
    )
    k = int(np.argmin(np.abs(result.freqs_hz - f_selected)))
    return ModeSlice(
        mode_idx=int(mode_idx),
        freq_input_hz=float(f_click_hz),
        freq_selected_hz=float(result.freqs_hz[k]),
        U_slice=result.U[k].astype(np.complex64, copy=False),
        V_slice=result.V[k].astype(np.complex64, copy=False),
    )

