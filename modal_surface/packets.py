from __future__ import annotations

from pathlib import Path

import numpy as np

from modal_surface.geometry import depth_edge_keep_mask, erode_mask, unproject_pixels
from modal_surface.io import ViewConfig, ensure_modal_shape, load_depth, load_mask, load_modal_npz, load_view_config


def _mode_amplitude(mode_u: np.ndarray, mode_v: np.ndarray) -> np.ndarray:
    return np.sqrt((np.abs(mode_u) ** 2 + np.abs(mode_v) ** 2).astype(np.float32))


def _confidence_from_amplitude(amplitude: np.ndarray) -> np.ndarray:
    if amplitude.size == 0:
        return amplitude.astype(np.float32)
    hi = np.percentile(amplitude, 95, axis=1)
    hi = np.maximum(hi, 1e-6).astype(np.float32)
    return np.clip(amplitude / hi[:, None], 0.0, 1.0).astype(np.float32)


def make_surface_packet(
    view_config_path: str | Path,
    modal_npz_path: str | Path,
    out_path: str | Path,
    stride: int = 4,
    mask_erode_iters: int = 2,
    depth_edge_tau: float = 0.12,
    min_amplitude_percentile: float = 5.0,
) -> Path:
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if not (0.0 <= min_amplitude_percentile < 100.0):
        raise ValueError("min_amplitude_percentile must be in [0, 100).")

    cfg = load_view_config(view_config_path)
    expected_shape = (cfg.image_height, cfg.image_width)
    modal = load_modal_npz(modal_npz_path)
    ensure_modal_shape(modal, expected_shape)
    depth = load_depth(cfg.depth_path, expected_shape, depth_scale=cfg.depth_scale)
    mask = load_mask(cfg.mask_path, expected_shape)

    mode_u = modal["mode_u"].astype(np.complex64)
    mode_v = modal["mode_v"].astype(np.complex64)
    amp_images = _mode_amplitude(mode_u, mode_v)

    valid = mask & np.isfinite(depth) & (depth > 0)
    valid = erode_mask(valid, mask_erode_iters)
    valid = depth_edge_keep_mask(depth, valid, edge_tau=depth_edge_tau, kernel_size=5)

    yy, xx = np.mgrid[: cfg.image_height : stride, : cfg.image_width : stride]
    candidate_y = yy.ravel()
    candidate_x = xx.ravel()
    keep = valid[candidate_y, candidate_x]
    if not np.any(keep):
        raise ValueError("No valid surface pixels after mask/depth filtering.")
    px = candidate_x[keep].astype(np.int64)
    py = candidate_y[keep].astype(np.int64)

    amplitude = amp_images[:, py, px].astype(np.float32)
    max_amp = amplitude.max(axis=0)
    amp_thresh = float(np.percentile(max_amp, min_amplitude_percentile)) if max_amp.size else 0.0
    amp_keep = max_amp >= amp_thresh
    px = px[amp_keep]
    py = py[amp_keep]
    amplitude = amplitude[:, amp_keep]
    if px.size == 0:
        raise ValueError("No valid points remain after modal amplitude filtering.")

    pixels_xy = np.stack([px.astype(np.float32), py.astype(np.float32)], axis=1)
    point_depth = depth[py, px].astype(np.float32)
    points_world = unproject_pixels(pixels_xy, point_depth, cfg.K, cfg.world_to_camera)
    mode_u_points = mode_u[:, py, px].astype(np.complex64)
    mode_v_points = mode_v[:, py, px].astype(np.complex64)
    confidence = _confidence_from_amplitude(amplitude)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points_world=points_world.astype(np.float32),
        pixels_xy=pixels_xy.astype(np.float32),
        depth=point_depth.astype(np.float32),
        mode_u=mode_u_points.astype(np.complex64),
        mode_v=mode_v_points.astype(np.complex64),
        amplitude=amplitude.astype(np.float32),
        confidence=confidence.astype(np.float32),
        selected_freqs_hz=modal["selected_freqs_hz"].astype(np.float32),
        view_id=np.array(cfg.view_id),
        K=cfg.K.astype(np.float32),
        world_to_camera=cfg.world_to_camera.astype(np.float32),
        image_width=np.array(cfg.image_width, dtype=np.int32),
        image_height=np.array(cfg.image_height, dtype=np.int32),
        source_modal_npz=np.array(str(modal_npz_path)),
        source_view_config=np.array(str(view_config_path)),
    )
    return out

