"""Build N-view modal observation graphs from surface packets.

The multi-view stage generalizes the current two-view matches file. Instead of
hard-coding y1/y2/J1/J2 arrays, it stores one row per visible observation:

    obs_point_index[o] -> which canonical 3D point was observed
    obs_view_index[o]  -> which camera/view produced the observation
    obs_y[o]           -> complex 2D modal response in that view
    obs_J[o]           -> projection Jacobian at the 3D point for that view

The first implementation still uses a canonical surface packet, usually view1,
as the 3D anchor set. Target views add observations only for canonical points
that pass mask and depth visibility checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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


def _scalar_str(value: np.ndarray | str | bytes | object) -> str:
    arr = np.asarray(value)
    item = arr.item() if arr.shape == () else arr.reshape(-1)[0].item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _mode_amplitude(mode_u: np.ndarray, mode_v: np.ndarray) -> np.ndarray:
    return np.sqrt((np.abs(mode_u) ** 2 + np.abs(mode_v) ** 2).astype(np.float32))


def _confidence_map(mode_u: np.ndarray, mode_v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    amp = _mode_amplitude(mode_u, mode_v)
    vals = amp[mask] if np.any(mask) else amp.ravel()
    scale = float(np.percentile(vals, 95)) if vals.size else 1.0
    scale = max(scale, 1e-6)
    return np.clip(amp / scale, 0.0, 1.0).astype(np.float32)


def _window_depth_stats(depth: np.ndarray, x: int, y: int, radius: int) -> tuple[float, float, float] | None:
    h, w = depth.shape
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    vals = depth[y0:y1, x0:x1]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < max(3, 2 * radius + 1):
        return None
    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    return float(p10), float(p50), float(p90)


def _append_canonical_observations(
    packet: np.lib.npyio.NpzFile,
    mode_index: int,
    obs_point_indices: list[int],
    obs_view_indices: list[int],
    obs_pixels: list[list[float]],
    obs_y: list[list[complex]],
    obs_j: list[np.ndarray],
    obs_confidence: list[float],
) -> None:
    points = packet["points_world"].astype(np.float32)
    K = packet["K"].astype(np.float64)
    w2c = packet["world_to_camera"].astype(np.float64)
    J = projection_jacobian(points, K, w2c)
    pixels = packet["pixels_xy"].astype(np.float32)
    y = np.stack([packet["mode_u"][mode_index], packet["mode_v"][mode_index]], axis=1).astype(np.complex64)
    confidence = packet["confidence"][mode_index].astype(np.float32)

    for i in range(points.shape[0]):
        obs_point_indices.append(i)
        obs_view_indices.append(0)
        obs_pixels.append([float(pixels[i, 0]), float(pixels[i, 1])])
        obs_y.append([complex(y[i, 0]), complex(y[i, 1])])
        obs_j.append(J[i].astype(np.float32))
        obs_confidence.append(float(confidence[i]))


def _append_target_observations(
    points_world: np.ndarray,
    canonical_freq_hz: float,
    view_index: int,
    view_config_path: str | Path,
    modal_npz_path: str | Path,
    mode_index: int,
    mask_erode_iters: int,
    depth_tau: float,
    edge_tau: float,
    depth_window_radius: int,
    freq_tolerance_hz: float,
    obs_point_indices: list[int],
    obs_view_indices: list[int],
    obs_pixels: list[list[float]],
    obs_y: list[list[complex]],
    obs_j: list[np.ndarray],
    obs_confidence: list[float],
) -> tuple[str, int, int, float]:
    cfg = load_view_config(view_config_path)
    expected_shape = (cfg.image_height, cfg.image_width)
    modal = load_modal_npz(modal_npz_path)
    ensure_modal_shape(modal, expected_shape)
    if mode_index >= modal["mode_u"].shape[0]:
        raise ValueError(f"mode_index={mode_index} is out of range for {modal_npz_path}.")

    target_freq_hz = float(modal["selected_freqs_hz"][mode_index])
    if abs(canonical_freq_hz - target_freq_hz) > freq_tolerance_hz:
        raise ValueError(
            f"Mode frequency mismatch for {cfg.view_id}: canonical={canonical_freq_hz:.6f}, "
            f"target={target_freq_hz:.6f}."
        )

    pixels_xy, z = project_points(points_world, cfg.K, cfg.world_to_camera)
    inside = in_image_with_margin(pixels_xy, cfg.image_width, cfg.image_height, margin=depth_window_radius + 1) & (z > 0)
    if not np.any(inside):
        return cfg.view_id, cfg.image_width, cfg.image_height, target_freq_hz

    depth = load_depth(cfg.depth_path, expected_shape, depth_scale=cfg.depth_scale)
    mask = load_mask(cfg.mask_path, expected_shape)
    valid = erode_mask(mask & np.isfinite(depth) & (depth > 0), mask_erode_iters)
    valid = depth_edge_keep_mask(depth, valid, edge_tau=edge_tau, kernel_size=5)

    mode_u = modal["mode_u"][mode_index].astype(np.complex64)
    mode_v = modal["mode_v"][mode_index].astype(np.complex64)
    confidence_map = _confidence_map(mode_u, mode_v, valid)

    for i in np.where(inside)[0].tolist():
        x = int(round(float(pixels_xy[i, 0])))
        y = int(round(float(pixels_xy[i, 1])))
        if not valid[y, x]:
            continue
        stats = _window_depth_stats(depth, x, y, depth_window_radius)
        if stats is None:
            continue
        d_front, d_med, d_back = stats
        if d_med <= 0:
            continue
        if abs(float(z[i]) - d_front) / d_med >= depth_tau:
            continue
        if (d_back - d_front) / d_med > edge_tau:
            continue

        y_u = bilinear_sample(mode_u, pixels_xy[i : i + 1])[0]
        y_v = bilinear_sample(mode_v, pixels_xy[i : i + 1])[0]
        confidence = float(bilinear_sample(confidence_map, pixels_xy[i : i + 1])[0])
        obs_point_indices.append(i)
        obs_view_indices.append(view_index)
        obs_pixels.append([float(pixels_xy[i, 0]), float(pixels_xy[i, 1])])
        obs_y.append([complex(y_u), complex(y_v)])
        obs_j.append(projection_jacobian(points_world[i : i + 1], cfg.K, cfg.world_to_camera)[0].astype(np.float32))
        obs_confidence.append(confidence)

    return cfg.view_id, cfg.image_width, cfg.image_height, target_freq_hz


def build_multiview_observation_graph(
    canonical_packet_path: str | Path,
    target_view_config_paths: Sequence[str | Path],
    target_modal_npz_paths: Sequence[str | Path],
    out_path: str | Path,
    mode_index: int = 0,
    min_observations: int = 2,
    mask_erode_iters: int = 2,
    depth_tau: float = 0.08,
    edge_tau: float = 0.12,
    depth_window_radius: int = 2,
    freq_tolerance_hz: float = 0.1,
) -> Path:
    """Build an N-view observation graph using one canonical surface packet."""
    if mode_index < 0:
        raise ValueError("mode_index must be non-negative.")
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1.")
    if len(target_view_config_paths) != len(target_modal_npz_paths):
        raise ValueError("--target-view-configs and --target-modal-npzs must have the same length.")
    if len(target_view_config_paths) == 0:
        raise ValueError("At least one target view is required.")
    if depth_tau <= 0:
        raise ValueError("depth_tau must be positive.")
    if edge_tau <= 0:
        raise ValueError("edge_tau must be positive.")

    packet = np.load(str(canonical_packet_path), allow_pickle=False)
    if mode_index >= packet["mode_u"].shape[0]:
        raise ValueError(f"mode_index={mode_index} is out of range for {canonical_packet_path}.")

    points_world_all = packet["points_world"].astype(np.float32)
    canonical_freq_hz = float(packet["selected_freqs_hz"][mode_index])
    view_ids = [_scalar_str(packet["view_id"])]
    view_widths = [int(packet["image_width"])]
    view_heights = [int(packet["image_height"])]
    view_freqs_hz = [canonical_freq_hz]
    source_view_configs = [_scalar_str(packet["source_view_config"]) if "source_view_config" in packet.files else ""]
    source_modal_npzs = [_scalar_str(packet["source_modal_npz"]) if "source_modal_npz" in packet.files else ""]

    obs_point_indices: list[int] = []
    obs_view_indices: list[int] = []
    obs_pixels: list[list[float]] = []
    obs_y: list[list[complex]] = []
    obs_j: list[np.ndarray] = []
    obs_confidence: list[float] = []

    _append_canonical_observations(
        packet,
        mode_index,
        obs_point_indices,
        obs_view_indices,
        obs_pixels,
        obs_y,
        obs_j,
        obs_confidence,
    )

    for target_i, (cfg_path, modal_path) in enumerate(zip(target_view_config_paths, target_modal_npz_paths), start=1):
        view_id, width, height, freq_hz = _append_target_observations(
            points_world_all,
            canonical_freq_hz,
            target_i,
            cfg_path,
            modal_path,
            mode_index,
            mask_erode_iters,
            depth_tau,
            edge_tau,
            depth_window_radius,
            freq_tolerance_hz,
            obs_point_indices,
            obs_view_indices,
            obs_pixels,
            obs_y,
            obs_j,
            obs_confidence,
        )
        view_ids.append(view_id)
        view_widths.append(width)
        view_heights.append(height)
        view_freqs_hz.append(freq_hz)
        source_view_configs.append(str(cfg_path))
        source_modal_npzs.append(str(modal_path))

    obs_point_arr = np.asarray(obs_point_indices, dtype=np.int64)
    obs_view_arr = np.asarray(obs_view_indices, dtype=np.int32)
    counts = np.bincount(obs_point_arr, minlength=points_world_all.shape[0])
    keep_points = counts >= int(min_observations)
    if not np.any(keep_points):
        raise ValueError("No canonical points satisfy min_observations.")

    old_to_new = np.full(points_world_all.shape[0], -1, dtype=np.int64)
    active_old_indices = np.where(keep_points)[0]
    old_to_new[active_old_indices] = np.arange(active_old_indices.size, dtype=np.int64)
    keep_obs = keep_points[obs_point_arr]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points_world=points_world_all[active_old_indices].astype(np.float32),
        canonical_point_indices=active_old_indices.astype(np.int32),
        obs_point_index=old_to_new[obs_point_arr[keep_obs]].astype(np.int32),
        obs_view_index=obs_view_arr[keep_obs].astype(np.int32),
        obs_pixels_xy=np.asarray(obs_pixels, dtype=np.float32)[keep_obs],
        obs_y=np.asarray(obs_y, dtype=np.complex64)[keep_obs],
        obs_J=np.asarray(obs_j, dtype=np.float32)[keep_obs],
        obs_confidence=np.asarray(obs_confidence, dtype=np.float32)[keep_obs],
        obs_count_per_point=counts[active_old_indices].astype(np.int32),
        view_ids=np.asarray(view_ids),
        view_image_width=np.asarray(view_widths, dtype=np.int32),
        view_image_height=np.asarray(view_heights, dtype=np.int32),
        view_freqs_hz=np.asarray(view_freqs_hz, dtype=np.float32),
        freq_hz=np.array(canonical_freq_hz, dtype=np.float32),
        mode_index=np.array(mode_index, dtype=np.int32),
        min_observations=np.array(min_observations, dtype=np.int32),
        source_canonical_packet=np.array(str(canonical_packet_path)),
        source_view_configs=np.asarray(source_view_configs),
        source_modal_npzs=np.asarray(source_modal_npzs),
    )
    return out
