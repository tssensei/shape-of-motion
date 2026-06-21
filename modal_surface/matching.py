from __future__ import annotations

from pathlib import Path

import numpy as np

from modal_surface.geometry import (
    bilinear_sample,
    depth_edge_keep_mask,
    erode_mask,
    in_image_with_margin,
    project_points,
    projection_jacobian,
)
from modal_surface.io import ensure_modal_shape, load_depth, load_mask, load_modal_npz, load_view_config


def _mode_amplitude(mode_u: np.ndarray, mode_v: np.ndarray) -> np.ndarray:
    return np.sqrt((np.abs(mode_u) ** 2 + np.abs(mode_v) ** 2).astype(np.float32))


def _window_depth_stats(depth: np.ndarray, x: int, y: int, radius: int) -> tuple[float, float, float, float] | None:
    h, w = depth.shape
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    vals = depth[y0:y1, x0:x1]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < max(3, (2 * radius + 1)):
        return None
    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    return float(p10), float(p50), float(p90), float(vals.size)


def _view2_confidence_map(mode_u: np.ndarray, mode_v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    amp = _mode_amplitude(mode_u, mode_v)
    vals = amp[mask] if np.any(mask) else amp.ravel()
    scale = float(np.percentile(vals, 95)) if vals.size else 1.0
    scale = max(scale, 1e-6)
    return np.clip(amp / scale, 0.0, 1.0).astype(np.float32)


def match_two_views(
    view1_packet_path: str | Path,
    view2_config_path: str | Path,
    view2_modal_npz_path: str | Path,
    out_path: str | Path,
    mode_index: int = 0,
    mask_erode_iters: int = 2,
    depth_tau: float = 0.08,
    edge_tau: float = 0.12,
    depth_window_radius: int = 2,
    freq_tolerance_hz: float = 0.1,
) -> Path:
    if mode_index < 0:
        raise ValueError("mode_index must be non-negative.")
    if depth_tau <= 0:
        raise ValueError("depth_tau must be positive.")
    if edge_tau <= 0:
        raise ValueError("edge_tau must be positive.")

    p1 = np.load(str(view1_packet_path), allow_pickle=False)
    cfg2 = load_view_config(view2_config_path)
    expected_shape2 = (cfg2.image_height, cfg2.image_width)
    modal2 = load_modal_npz(view2_modal_npz_path)
    ensure_modal_shape(modal2, expected_shape2)
    if mode_index >= p1["mode_u"].shape[0] or mode_index >= modal2["mode_u"].shape[0]:
        raise ValueError("mode_index is out of range for one of the modal files.")

    freq1 = float(p1["selected_freqs_hz"][mode_index])
    freq2 = float(modal2["selected_freqs_hz"][mode_index])
    if abs(freq1 - freq2) > freq_tolerance_hz:
        raise ValueError(
            f"Mode frequency mismatch at mode_index={mode_index}: view1={freq1:.6f}, view2={freq2:.6f}."
        )

    points_world = p1["points_world"].astype(np.float32)
    p2_xy, z2 = project_points(points_world, cfg2.K, cfg2.world_to_camera)
    inside = in_image_with_margin(p2_xy, cfg2.image_width, cfg2.image_height, margin=depth_window_radius + 1)
    if not np.any(inside):
        raise ValueError("No view1 packet points project inside view2.")

    depth2 = load_depth(cfg2.depth_path, expected_shape2, depth_scale=cfg2.depth_scale)
    mask2 = load_mask(cfg2.mask_path, expected_shape2)
    valid2 = erode_mask(mask2 & np.isfinite(depth2) & (depth2 > 0), mask_erode_iters)
    valid2 = depth_edge_keep_mask(depth2, valid2, edge_tau=edge_tau, kernel_size=5)

    mode_u2 = modal2["mode_u"][mode_index].astype(np.complex64)
    mode_v2 = modal2["mode_v"][mode_index].astype(np.complex64)
    conf2_map = _view2_confidence_map(mode_u2, mode_v2, valid2)

    keep_indices: list[int] = []
    sampled_p2: list[list[float]] = []
    sampled_y2: list[list[complex]] = []
    sampled_c2: list[float] = []
    for i in np.where(inside)[0].tolist():
        x = int(round(float(p2_xy[i, 0])))
        y = int(round(float(p2_xy[i, 1])))
        if not valid2[y, x]:
            continue
        stats = _window_depth_stats(depth2, x, y, depth_window_radius)
        if stats is None:
            continue
        d_front, d_med, d_back, _ = stats
        if d_med <= 0:
            continue
        if abs(float(z2[i]) - d_front) / d_med >= depth_tau:
            continue
        if (d_back - d_front) / d_med > edge_tau:
            continue
        y2_u = bilinear_sample(mode_u2, p2_xy[i : i + 1])[0]
        y2_v = bilinear_sample(mode_v2, p2_xy[i : i + 1])[0]
        c2 = float(bilinear_sample(conf2_map, p2_xy[i : i + 1])[0])
        keep_indices.append(i)
        sampled_p2.append([float(p2_xy[i, 0]), float(p2_xy[i, 1])])
        sampled_y2.append([complex(y2_u), complex(y2_v)])
        sampled_c2.append(c2)

    if len(keep_indices) == 0:
        raise ValueError("No cross-view matches survived depth visibility filtering.")

    idx = np.asarray(keep_indices, dtype=np.int64)
    matched_points = points_world[idx]
    K1 = p1["K"].astype(np.float64)
    w2c1 = p1["world_to_camera"].astype(np.float64)
    J1 = projection_jacobian(matched_points, K1, w2c1)
    J2 = projection_jacobian(matched_points, cfg2.K, cfg2.world_to_camera)
    y1 = np.stack([p1["mode_u"][mode_index, idx], p1["mode_v"][mode_index, idx]], axis=1).astype(np.complex64)
    y2 = np.asarray(sampled_y2, dtype=np.complex64)
    c1 = p1["confidence"][mode_index, idx].astype(np.float32)
    c2 = np.asarray(sampled_c2, dtype=np.float32)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points_world=matched_points.astype(np.float32),
        p1_xy=p1["pixels_xy"][idx].astype(np.float32),
        p2_xy=np.asarray(sampled_p2, dtype=np.float32),
        y1=y1,
        y2=y2,
        J1=J1.astype(np.float32),
        J2=J2.astype(np.float32),
        c1=c1,
        c2=c2,
        freq_hz=np.array(freq1, dtype=np.float32),
        mode_index=np.array(mode_index, dtype=np.int32),
        view1_id=p1["view_id"],
        view2_id=np.array(cfg2.view_id),
        image1_width=p1["image_width"],
        image1_height=p1["image_height"],
        image2_width=np.array(cfg2.image_width, dtype=np.int32),
        image2_height=np.array(cfg2.image_height, dtype=np.int32),
        source_view1_packet=np.array(str(view1_packet_path)),
        source_view2_config=np.array(str(view2_config_path)),
        source_view2_modal_npz=np.array(str(view2_modal_npz_path)),
    )
    return out

