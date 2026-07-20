from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np


def compute_dense_flow_pair(
    reference_gray: np.ndarray,
    current_gray: np.ndarray,
    method: str = "farneback",
) -> np.ndarray:
    """Compute dense image-plane flow from one grayscale frame to another."""
    if reference_gray.ndim != 2 or current_gray.ndim != 2:
        raise ValueError("reference_gray and current_gray must both be [H,W].")
    if reference_gray.shape != current_gray.shape:
        raise ValueError(
            "reference_gray and current_gray must have the same shape; "
            f"got {reference_gray.shape} and {current_gray.shape}."
        )
    if not np.isfinite(reference_gray).all() or not np.isfinite(current_gray).all():
        raise ValueError("Dense-flow input frames must be finite.")

    reference = np.clip(reference_gray * 255.0, 0.0, 255.0).astype(np.uint8)
    current = np.clip(current_gray * 255.0, 0.0, 255.0).astype(np.uint8)
    method_l = method.lower()
    if method_l == "farneback":
        return cv2.calcOpticalFlowFarneback(
            reference,
            current,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=15,
            iterations=4,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        ).astype(np.float32, copy=False)

    if method_l in {"tvl1", "tv-l1", "tv_l1"}:
        if not hasattr(cv2, "optflow") or not hasattr(
            cv2.optflow, "DualTVL1OpticalFlow_create"
        ):
            raise RuntimeError(
                "TV-L1 requires an OpenCV build with "
                "cv2.optflow.DualTVL1OpticalFlow_create."
            )
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        tvl1.setTau(0.25)
        tvl1.setLambda(0.05)
        tvl1.setTheta(0.3)
        tvl1.setScalesNumber(4)
        tvl1.setWarpingsNumber(5)
        tvl1.setEpsilon(0.01)
        tvl1.setInnerIterations(30)
        tvl1.setOuterIterations(10)
        tvl1.setScaleStep(0.8)
        tvl1.setGamma(0.0)
        tvl1.setMedianFiltering(0)
        tvl1.setUseInitialFlow(False)
        return tvl1.calc(reference, current, None).astype(np.float32, copy=False)

    raise ValueError(f"Unknown flow method: {method}")


def compute_dense_flow_to_reference(
    frames_gray: np.ndarray,
    method: str = "farneback",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute dense optical flow from the middle reference frame to every frame.

    Returns horizontal and vertical image-plane displacements with shape [T,H,W].
    """
    if frames_gray.ndim != 3:
        raise ValueError("frames_gray must be [T,H,W].")
    if frames_gray.shape[0] < 3:
        raise ValueError("Need at least 3 frames for modal analysis.")

    num_frames, h, w = frames_gray.shape
    u = np.zeros((num_frames, h, w), dtype=np.float32)
    v = np.zeros((num_frames, h, w), dtype=np.float32)

    t_ref = num_frames // 2
    reference = frames_gray[t_ref]
    for t in range(num_frames):
        flow = compute_dense_flow_pair(reference, frames_gray[t], method=method)
        u[t] = flow[..., 0]
        v[t] = flow[..., 1]
    return u, v


def _pyramid_sobel(frame: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    if levels < 0:
        raise ValueError("levels must be >= 0.")

    img = frame.astype(np.float32, copy=False)
    for _ in range(int(levels)):
        img = cv2.pyrDown(img)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    if levels == 0:
        return gx, gy

    h, w = frame.shape
    gx = cv2.resize(gx, (w, h), interpolation=cv2.INTER_LINEAR)
    gy = cv2.resize(gy, (w, h), interpolation=cv2.INTER_LINEAR)
    scale = float(2 ** levels)
    return gx * scale, gy * scale


def _weighted_pyramid_sobel(
    frame: np.ndarray,
    grad_pyr_weights: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    if len(grad_pyr_weights) == 0:
        raise ValueError("grad_pyr_weights must not be empty.")

    weights = np.asarray(grad_pyr_weights, dtype=np.float32)
    if np.any(weights < 0):
        raise ValueError("grad_pyr_weights must be non-negative.")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("grad_pyr_weights must contain at least one positive value.")
    weights = weights / total

    gx_acc = np.zeros_like(frame, dtype=np.float32)
    gy_acc = np.zeros_like(frame, dtype=np.float32)
    for level, alpha in enumerate(weights.tolist()):
        if alpha <= 0:
            continue
        gx_l, gy_l = _pyramid_sobel(frame, int(level))
        gx_acc += float(alpha) * gx_l
        gy_acc += float(alpha) * gy_l
    return gx_acc, gy_acc


def contrast_weighted_smooth(
    u: np.ndarray,
    v: np.ndarray,
    frame_ref: np.ndarray,
    sigma_b: float = 3.0,
    sigma_c: float = 0.0,
    eps: float = 1e-6,
    mask: Optional[np.ndarray] = None,
    grad_pyr_levels: int = 2,
    grad_pyr_weights: Optional[Sequence[float]] = (0.5, 0.3, 0.2),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth image-plane flow with edge-aware weights from the reference frame.
    """
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError("u and v must have the same shape [T,H,W].")
    if frame_ref.ndim != 2:
        raise ValueError("frame_ref must be [H,W].")
    if frame_ref.shape != u.shape[1:]:
        raise ValueError(f"frame_ref shape {frame_ref.shape} does not match flow shape {u.shape[1:]}.")
    if mask is not None and mask.shape != frame_ref.shape:
        raise ValueError(f"mask shape {mask.shape} does not match frame_ref shape {frame_ref.shape}.")
    if sigma_b <= 0:
        raise ValueError("sigma_b must be positive.")
    if grad_pyr_levels < 0:
        raise ValueError("grad_pyr_levels must be >= 0.")

    image = frame_ref.astype(np.float32, copy=False)
    if sigma_c > 0:
        kernel_c = int(round(6 * sigma_c + 1))
        if kernel_c % 2 == 0:
            kernel_c += 1
        image = cv2.GaussianBlur(
            image,
            (kernel_c, kernel_c),
            sigmaX=sigma_c,
            sigmaY=sigma_c,
            borderType=cv2.BORDER_REFLECT,
        )

    if grad_pyr_weights is None:
        gx, gy = _pyramid_sobel(image, int(grad_pyr_levels))
    else:
        gx, gy = _weighted_pyramid_sobel(image, grad_pyr_weights)

    wx = np.abs(gx).astype(np.float32)
    wy = np.abs(gy).astype(np.float32)
    mask_f = None
    if mask is not None:
        mask_f = (mask > 0).astype(np.float32)
        wx *= mask_f
        wy *= mask_f

    kernel_b = int(round(6 * sigma_b + 1))
    if kernel_b % 2 == 0:
        kernel_b += 1

    den_u = cv2.GaussianBlur(
        wx,
        (kernel_b, kernel_b),
        sigmaX=sigma_b,
        sigmaY=sigma_b,
        borderType=cv2.BORDER_REFLECT,
    ) + eps
    den_v = cv2.GaussianBlur(
        wy,
        (kernel_b, kernel_b),
        sigmaX=sigma_b,
        sigmaY=sigma_b,
        borderType=cv2.BORDER_REFLECT,
    ) + eps

    u_filt = np.empty_like(u, dtype=np.float32)
    v_filt = np.empty_like(v, dtype=np.float32)
    for t in range(u.shape[0]):
        num_u = cv2.GaussianBlur(
            u[t].astype(np.float32, copy=False) * wx,
            (kernel_b, kernel_b),
            sigmaX=sigma_b,
            sigmaY=sigma_b,
            borderType=cv2.BORDER_REFLECT,
        )
        num_v = cv2.GaussianBlur(
            v[t].astype(np.float32, copy=False) * wy,
            (kernel_b, kernel_b),
            sigmaX=sigma_b,
            sigmaY=sigma_b,
            borderType=cv2.BORDER_REFLECT,
        )
        u_filt[t] = num_u / den_u
        v_filt[t] = num_v / den_v
        if mask_f is not None:
            u_filt[t] *= mask_f
            v_filt[t] *= mask_f

    return u_filt, v_filt

