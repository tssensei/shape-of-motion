"""Build modal observation graphs from a VGGT carrier point cloud.

This module implements the VGGT carrier route. The geometry support is no
longer a depth map sampled from one reference view. Instead, a dense VGGT point
cloud is treated as the shared set of candidate 3D points:

    carrier points X_i
    -> project X_i into every modal view
    -> robust z-buffer + mask decides visibility
    -> sample complex mode_u/mode_v and projection Jacobian J
    -> save the same observation graph consumed by optimize-multi-view

The visibility test intentionally does not read view_config.depth_path. The
front surface is estimated from projected carrier depths in a local image
window, which avoids reintroducing the old cross-depth-map alignment problem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from modal_surface.geometry import bilinear_sample, erode_mask, in_image_with_margin, project_points, projection_jacobian, unproject_pixels
from modal_surface.io import ViewConfig, ensure_modal_shape, load_mask, load_modal_npz, load_view_config


def _load_carrier_points(path: str | Path) -> dict[str, np.ndarray]:
    z = np.load(str(path), allow_pickle=False)
    required = ["points_world"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise ValueError(f"Carrier point file {path} missing required keys: {missing}.")

    points = z["points_world"].astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points.shape}.")

    keep = np.all(np.isfinite(points), axis=1)
    if not np.any(keep):
        raise ValueError("No finite carrier points were found.")

    out: dict[str, np.ndarray] = {
        "points_world": points[keep],
        "original_indices": np.where(keep)[0].astype(np.int32),
    }
    if "colors" in z.files:
        colors = z["colors"]
        if colors.shape != (points.shape[0], 3):
            raise ValueError(f"colors must have shape (N,3), got {colors.shape}.")
        out["colors"] = colors[keep].astype(np.uint8)
    if "source_view_index" in z.files:
        source_view_index = z["source_view_index"].astype(np.int32)
        if source_view_index.shape != (points.shape[0],):
            raise ValueError(f"source_view_index must have shape (N,), got {source_view_index.shape}.")
        out["source_view_index"] = source_view_index[keep]
    if "source_pixels_xy" in z.files:
        source_pixels_xy = z["source_pixels_xy"].astype(np.float32)
        if source_pixels_xy.shape != (points.shape[0], 2):
            raise ValueError(f"source_pixels_xy must have shape (N,2), got {source_pixels_xy.shape}.")
        out["source_pixels_xy"] = source_pixels_xy[keep]
    for key in ("source_image_height", "source_image_width"):
        if key in z.files:
            value = int(np.asarray(z[key]).item())
            if value <= 0:
                raise ValueError(f"{key} must be positive, got {value}.")
            out[key] = np.array(value, dtype=np.int32)
    return out


def _load_view_inputs(
    view_config_paths: Sequence[str | Path],
    modal_npz_paths: Sequence[str | Path],
    mode_index: int,
    freq_tolerance_hz: float,
) -> tuple[list[ViewConfig], list[dict[str, np.ndarray]], np.ndarray, float]:
    if len(view_config_paths) != len(modal_npz_paths):
        raise ValueError("--view-config and --modal-npz must be supplied the same number of times.")
    if len(view_config_paths) < 1:
        raise ValueError("At least one view is required.")
    if mode_index < 0:
        raise ValueError("mode_index must be non-negative.")
    if freq_tolerance_hz < 0:
        raise ValueError("freq_tolerance_hz must be non-negative.")

    configs: list[ViewConfig] = []
    modals: list[dict[str, np.ndarray]] = []
    freqs: list[float] = []
    for cfg_path, modal_path in zip(view_config_paths, modal_npz_paths):
        cfg = load_view_config(cfg_path)
        modal = load_modal_npz(modal_path)
        ensure_modal_shape(modal, (cfg.image_height, cfg.image_width))
        if mode_index >= modal["mode_u"].shape[0]:
            raise ValueError(f"mode_index={mode_index} is out of range for {modal_path}.")
        freq_hz = float(modal["selected_freqs_hz"][mode_index])
        configs.append(cfg)
        modals.append(modal)
        freqs.append(freq_hz)

    reference_freq_hz = freqs[0]
    for cfg, freq_hz in zip(configs[1:], freqs[1:]):
        if abs(reference_freq_hz - freq_hz) > freq_tolerance_hz:
            raise ValueError(
                f"Mode frequency mismatch for {cfg.view_id}: reference={reference_freq_hz:.6f}, "
                f"target={freq_hz:.6f}."
            )
    return configs, modals, np.asarray(freqs, dtype=np.float32), reference_freq_hz


def _local_snr(
    modal: dict[str, np.ndarray],
    selected_freq_hz: float,
    snr_band_hz: float,
    snr_exclude_hz: float,
) -> tuple[float, float, float, float]:
    missing = [key for key in ("freqs_hz", "power_spectrum") if key not in modal]
    if missing:
        raise ValueError(f"local-snr weighting requires modal npz keys {missing}. Re-run run_modal_peak_pick.py export.")
    freqs = np.asarray(modal["freqs_hz"], dtype=np.float64).reshape(-1)
    power = np.asarray(modal["power_spectrum"], dtype=np.float64).reshape(-1)
    if freqs.shape != power.shape or freqs.size == 0:
        raise ValueError("freqs_hz and power_spectrum must be non-empty arrays with matching shape.")
    finite = np.isfinite(freqs) & np.isfinite(power)
    if not np.any(finite):
        raise ValueError("freqs_hz/power_spectrum contain no finite entries.")
    freq = float(selected_freq_hz)
    finite_indices = np.where(finite)[0]
    nearest_idx = int(finite_indices[np.argmin(np.abs(freqs[finite_indices] - freq))])
    signal = float(power[nearest_idx])
    bin_hz = float(freqs[nearest_idx])
    noise_mask = (
        finite
        & (freqs >= freq - float(snr_band_hz))
        & (freqs <= freq + float(snr_band_hz))
        & (np.abs(freqs - freq) >= float(snr_exclude_hz))
    )
    if not np.any(noise_mask):
        raise ValueError(
            f"No frequency bins available for local SNR noise estimate around {freq:.6f} Hz. "
            "Increase --snr-band-hz or decrease --snr-exclude-hz."
        )
    noise = float(np.median(power[noise_mask]))
    snr = signal / max(noise, np.finfo(np.float64).eps)
    return signal, noise, snr, bin_hz


def _view_frequency_reliability(
    modals: Sequence[dict[str, np.ndarray]],
    view_freqs_hz: np.ndarray,
    weighting: str,
    snr_band_hz: float,
    snr_exclude_hz: float,
    snr_good: float,
    view_weight_min: float,
) -> dict[str, np.ndarray]:
    if weighting not in {"none", "local-snr"}:
        raise ValueError("view_frequency_weighting must be 'none' or 'local-snr'.")
    if snr_band_hz <= 0:
        raise ValueError("snr_band_hz must be positive.")
    if snr_exclude_hz < 0:
        raise ValueError("snr_exclude_hz must be non-negative.")
    if snr_exclude_hz >= snr_band_hz:
        raise ValueError("snr_exclude_hz must be smaller than snr_band_hz.")
    if snr_good <= 1:
        raise ValueError("snr_good must be greater than 1.")
    if not (0.0 <= view_weight_min <= 1.0):
        raise ValueError("view_weight_min must be in [0, 1].")

    num_views = len(modals)
    weights = np.ones((num_views,), dtype=np.float32)
    snr = np.full((num_views,), np.nan, dtype=np.float32)
    signal = np.full((num_views,), np.nan, dtype=np.float32)
    noise = np.full((num_views,), np.nan, dtype=np.float32)
    bin_hz = np.full((num_views,), np.nan, dtype=np.float32)
    if weighting == "local-snr":
        for view_idx, (modal, selected_freq_hz) in enumerate(zip(modals, view_freqs_hz)):
            sig, noi, ratio, freq_bin = _local_snr(modal, float(selected_freq_hz), snr_band_hz, snr_exclude_hz)
            signal[view_idx] = np.float32(sig)
            noise[view_idx] = np.float32(noi)
            snr[view_idx] = np.float32(ratio)
            bin_hz[view_idx] = np.float32(freq_bin)
        q = np.maximum(snr.astype(np.float64) - 1.0, 0.0)
        denom = max(float(np.max(q)), float(snr_good) - 1.0)
        weights = np.clip(q / max(denom, np.finfo(np.float64).eps), float(view_weight_min), 1.0).astype(np.float32)
    return {
        "weights": weights,
        "snr": snr,
        "signal": signal,
        "noise": noise,
        "bin_hz": bin_hz,
    }


def _candidate_mask(
    pixels_xy: np.ndarray,
    z: np.ndarray,
    valid_mask: np.ndarray,
    width: int,
    height: int,
    margin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inside = in_image_with_margin(pixels_xy, width, height, margin=margin) & np.isfinite(z) & (z > 0)
    candidate = np.zeros_like(inside, dtype=bool)
    rounded_x = np.full((pixels_xy.shape[0],), -1, dtype=np.int32)
    rounded_y = np.full((pixels_xy.shape[0],), -1, dtype=np.int32)
    if not np.any(inside):
        return candidate, rounded_x, rounded_y
    inside_idx = np.where(inside)[0]
    x = np.rint(pixels_xy[inside_idx, 0]).astype(np.int32)
    y = np.rint(pixels_xy[inside_idx, 1]).astype(np.int32)
    in_mask = valid_mask[y, x]
    selected = inside_idx[in_mask]
    candidate[selected] = True
    rounded_x[inside_idx] = x
    rounded_y[inside_idx] = y
    return candidate, rounded_x, rounded_y


def _build_pixel_buckets(
    point_indices: np.ndarray,
    rounded_x: np.ndarray,
    rounded_y: np.ndarray,
    width: int,
) -> dict[int, np.ndarray]:
    if point_indices.size == 0:
        return {}
    linear = rounded_y[point_indices].astype(np.int64) * int(width) + rounded_x[point_indices].astype(np.int64)
    order = np.argsort(linear)
    linear_sorted = linear[order]
    point_sorted = point_indices[order]
    keys, starts, counts = np.unique(linear_sorted, return_index=True, return_counts=True)
    return {int(k): point_sorted[int(s) : int(s + c)] for k, s, c in zip(keys, starts, counts)}


def _local_depth_stats(
    buckets: dict[int, np.ndarray],
    z: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int,
    front_percentile: float,
    min_samples: int,
) -> tuple[float, float] | None:
    parts: list[np.ndarray] = []
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    for yy in range(y0, y1):
        base = yy * width
        for xx in range(x0, x1):
            idx = buckets.get(base + xx)
            if idx is not None:
                parts.append(idx)
    if not parts:
        return None
    indices = parts[0] if len(parts) == 1 else np.concatenate(parts)
    vals = z[indices]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < min_samples:
        return None
    z_front, z_med = np.percentile(vals, [front_percentile, 50])
    return float(z_front), float(z_med)


def _zbuffer_weight(
    z_value: float,
    z_front: float,
    z_med: float,
    zbuffer_tau: float,
    zbuffer_mode: str,
    zbuffer_soft_sigma: float,
    zbuffer_soft_min_weight: float,
) -> float | None:
    rel_depth_delta = abs(float(z_value) - float(z_front)) / max(abs(float(z_med)), 1e-6)
    if zbuffer_mode == "hard":
        return 1.0 if rel_depth_delta < float(zbuffer_tau) else None
    if zbuffer_mode != "soft":
        raise ValueError("zbuffer_mode must be 'hard' or 'soft'.")
    weight = float(np.exp(-0.5 * (rel_depth_delta / float(zbuffer_soft_sigma)) ** 2))
    if weight < float(zbuffer_soft_min_weight):
        return None
    return weight


def _bucket_indices_in_radius(
    buckets: dict[int, np.ndarray],
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    for yy in range(y0, y1):
        base = yy * width
        for xx in range(x0, x1):
            idx = buckets.get(base + xx)
            if idx is not None:
                parts.append(idx)
    if not parts:
        return np.zeros((0,), dtype=np.int64)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _require_scipy_kdtree():
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("pixel-candidates contribution sampling requires scipy.spatial.cKDTree.") from exc
    return cKDTree


def _quat_wxyz_to_rotmat(quats: np.ndarray) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float64)
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"gaussian_quats must have shape (N,4), got {quats.shape}.")
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("gaussian_quats contains invalid zero or non-finite entries.")
    q = quats / norms
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    mats = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    mats[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    mats[:, 0, 1] = 2.0 * (x * y - w * z)
    mats[:, 0, 2] = 2.0 * (x * z + w * y)
    mats[:, 1, 0] = 2.0 * (x * y + w * z)
    mats[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    mats[:, 1, 2] = 2.0 * (y * z - w * x)
    mats[:, 2, 0] = 2.0 * (x * z - w * y)
    mats[:, 2, 1] = 2.0 * (y * z + w * x)
    mats[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return mats


def _validate_gaussian_contribution_inputs(
    points_world: np.ndarray,
    gaussian_scales: np.ndarray | None,
    gaussian_quats: np.ndarray | None,
    gaussian_opacities: np.ndarray | None,
    rendered_depths: Sequence[np.ndarray] | None,
    rendered_accs: Sequence[np.ndarray] | None,
    configs: Sequence[ViewConfig],
    require_rendered: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    if gaussian_scales is None:
        raise ValueError("Contribution-based Gaussian sampling requires gaussian_scales.")
    if gaussian_quats is None:
        raise ValueError("Contribution-based Gaussian sampling requires gaussian_quats.")
    if gaussian_opacities is None:
        raise ValueError("Contribution-based Gaussian sampling requires gaussian_opacities.")
    if require_rendered and rendered_depths is None:
        raise ValueError("pixel-candidates requires rendered_depths.")
    if require_rendered and rendered_accs is None:
        raise ValueError("pixel-candidates requires rendered_accs.")

    scales = np.asarray(gaussian_scales, dtype=np.float32)
    quats = np.asarray(gaussian_quats, dtype=np.float32)
    opacities = np.asarray(gaussian_opacities, dtype=np.float32).reshape(-1)
    if scales.shape != points_world.shape:
        raise ValueError(f"gaussian_scales must have shape {points_world.shape}, got {scales.shape}.")
    if quats.shape != (points_world.shape[0], 4):
        raise ValueError(f"gaussian_quats must have shape ({points_world.shape[0]},4), got {quats.shape}.")
    if opacities.shape != (points_world.shape[0],):
        raise ValueError(f"gaussian_opacities must have shape ({points_world.shape[0]},), got {opacities.shape}.")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("gaussian_scales must be finite and positive.")
    if not np.all(np.isfinite(opacities)) or np.any(opacities < 0):
        raise ValueError("gaussian_opacities must be finite and non-negative.")

    depth_arrays: list[np.ndarray] = []
    acc_arrays: list[np.ndarray] = []
    if rendered_depths is not None or rendered_accs is not None:
        if rendered_depths is None or rendered_accs is None:
            raise ValueError("rendered_depths and rendered_accs must either both be supplied or both be omitted.")
        if len(rendered_depths) != len(configs) or len(rendered_accs) != len(configs):
            raise ValueError("rendered_depths/rendered_accs must have one array per view.")
        for cfg, depth, acc in zip(configs, rendered_depths, rendered_accs):
            expected_shape = (cfg.image_height, cfg.image_width)
            depth_arr = np.asarray(depth, dtype=np.float32)
            acc_arr = np.asarray(acc, dtype=np.float32)
            if depth_arr.shape != expected_shape:
                raise ValueError(f"Rendered depth for {cfg.view_id} must have shape {expected_shape}, got {depth_arr.shape}.")
            if acc_arr.shape != expected_shape:
                raise ValueError(f"Rendered alpha for {cfg.view_id} must have shape {expected_shape}, got {acc_arr.shape}.")
            depth_arrays.append(depth_arr)
            acc_arrays.append(acc_arr)
    return scales, quats, opacities, depth_arrays, acc_arrays


def _append_view_contribution_observations(
    points_world: np.ndarray,
    gaussian_scales: np.ndarray,
    gaussian_rotmats: np.ndarray,
    gaussian_opacities: np.ndarray,
    rendered_depth: np.ndarray,
    rendered_acc: np.ndarray,
    view_index: int,
    cfg: ViewConfig,
    modal: dict[str, np.ndarray],
    mode_index: int,
    mask_erode_iters: int,
    obs_point_indices: list[int],
    obs_view_indices: list[int],
    obs_pixels: list[list[float]],
    obs_y: list[list[complex]],
    obs_j: list[np.ndarray],
    obs_confidence: list[float],
    obs_depth_weight: list[float],
    obs_camera_z: list[float],
    obs_contribution_weight: list[float],
    obs_contribution_score: list[float],
    obs_contribution_sum: list[float],
    obs_surface_pixels: list[list[float]],
    obs_surface_camera_z: list[float],
    view_confidence: float,
    depth_weighting: str,
    depth_weight_power: float,
    depth_weight_min: float,
    depth_weight_reference_percentile: float,
    pixel_sample_stride: int,
    pixel_candidate_k: int,
    pixel_preselect_k: int,
    pixel_min_mode_amp_percentile: float,
    pixel_max_samples_per_view: int,
    pixel_render_acc_min: float,
    pixel_min_contribution: float,
    obs_zbuffer_weight: list[float],
) -> tuple[int, float]:
    expected_shape = (cfg.image_height, cfg.image_width)
    mask = load_mask(cfg.mask_path, expected_shape)
    valid_mask = erode_mask(mask, mask_erode_iters)

    valid_depth = np.isfinite(rendered_depth) & (rendered_depth > 0)
    valid_acc = np.isfinite(rendered_acc) & (rendered_acc >= float(pixel_render_acc_min))
    visible_mask = valid_mask & valid_depth & valid_acc
    if not np.any(visible_mask):
        return 0, float("nan")

    mode_u = modal["mode_u"][mode_index].astype(np.complex64)
    mode_v = modal["mode_v"][mode_index].astype(np.complex64)
    mode_amp = np.sqrt(np.abs(mode_u) ** 2 + np.abs(mode_v) ** 2).astype(np.float32)
    valid_amp = mode_amp[visible_mask]
    if valid_amp.size == 0:
        return 0, float("nan")
    amp_threshold = float(np.percentile(valid_amp, pixel_min_mode_amp_percentile))

    ys = np.arange(1, cfg.image_height - 1, int(pixel_sample_stride), dtype=np.int32)
    xs = np.arange(1, cfg.image_width - 1, int(pixel_sample_stride), dtype=np.int32)
    if ys.size == 0 or xs.size == 0:
        return 0, float("nan")
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    keep = visible_mask[flat_y, flat_x] & (mode_amp[flat_y, flat_x] >= amp_threshold)
    flat_x = flat_x[keep]
    flat_y = flat_y[keep]
    if flat_x.size == 0:
        return 0, float("nan")
    if flat_x.size > int(pixel_max_samples_per_view):
        selected = np.linspace(0, flat_x.size - 1, int(pixel_max_samples_per_view)).astype(np.int64)
        flat_x = flat_x[selected]
        flat_y = flat_y[selected]

    depths = rendered_depth[flat_y, flat_x].astype(np.float32)
    surface_pixels = np.stack([flat_x.astype(np.float32), flat_y.astype(np.float32)], axis=1)
    surface_points = unproject_pixels(surface_pixels, depths, cfg.K, cfg.world_to_camera)
    finite_surface = np.all(np.isfinite(surface_points), axis=1) & np.isfinite(depths) & (depths > 0)
    if not np.any(finite_surface):
        return 0, float("nan")
    flat_x = flat_x[finite_surface]
    flat_y = flat_y[finite_surface]
    depths = depths[finite_surface]
    surface_pixels = surface_pixels[finite_surface]
    surface_points = surface_points[finite_surface]

    z_reference = float(np.percentile(depths, depth_weight_reference_percentile))
    cKDTree = _require_scipy_kdtree()
    preselect_k = min(int(pixel_preselect_k), points_world.shape[0])
    tree = cKDTree(points_world.astype(np.float64))
    _, candidate_rows = tree.query(surface_points.astype(np.float64), k=preselect_k)
    if preselect_k == 1:
        candidate_rows = candidate_rows[:, None]

    _, point_camera_z = project_points(points_world, cfg.K, cfg.world_to_camera)
    added = 0
    min_contribution = float(pixel_min_contribution)
    scale_eps = np.float32(1e-8)
    for row, (x, y, depth, surface_point, candidates) in enumerate(
        zip(flat_x.tolist(), flat_y.tolist(), depths.tolist(), surface_points, candidate_rows)
    ):
        candidate_indices = np.asarray(candidates, dtype=np.int64)
        candidate_indices = candidate_indices[(candidate_indices >= 0) & (candidate_indices < points_world.shape[0])]
        if candidate_indices.size == 0:
            continue

        delta = surface_point[None, :].astype(np.float64) - points_world[candidate_indices].astype(np.float64)
        local = np.einsum("nij,nj->ni", np.swapaxes(gaussian_rotmats[candidate_indices], 1, 2), delta)
        scaled = local / np.maximum(gaussian_scales[candidate_indices].astype(np.float64), float(scale_eps))
        mahalanobis2 = np.sum(scaled * scaled, axis=1)
        scores = gaussian_opacities[candidate_indices].astype(np.float64) * np.exp(-0.5 * mahalanobis2)
        valid = np.isfinite(scores) & (scores >= min_contribution)
        if not np.any(valid):
            continue
        candidate_indices = candidate_indices[valid]
        scores = scores[valid]
        order = np.argsort(scores)[::-1][: int(pixel_candidate_k)]
        candidate_indices = candidate_indices[order]
        scores = scores[order]
        positive_z = np.isfinite(point_camera_z[candidate_indices]) & (point_camera_z[candidate_indices] > 0)
        if not np.any(positive_z):
            continue
        candidate_indices = candidate_indices[positive_z]
        scores = scores[positive_z]
        denom = float(np.sum(scores))
        if denom <= 0 or not np.isfinite(denom):
            continue
        weights = (scores / denom).astype(np.float32)
        jacobians = projection_jacobian(points_world[candidate_indices], cfg.K, cfg.world_to_camera)
        y_u = complex(mode_u[int(y), int(x)])
        y_v = complex(mode_v[int(y), int(x)])
        for row_idx, (point_idx, score, contribution_weight) in enumerate(
            zip(candidate_indices.tolist(), scores.tolist(), weights.tolist())
        ):
            candidate_z = float(point_camera_z[int(point_idx)])
            depth_weight = _depth_weight(
                candidate_z,
                z_reference,
                depth_weighting,
                depth_weight_power,
                depth_weight_min,
            )
            obs_point_indices.append(int(point_idx))
            obs_view_indices.append(view_index)
            obs_pixels.append([float(x), float(y)])
            obs_y.append([y_u, y_v])
            obs_j.append(jacobians[row_idx].astype(np.float32))
            obs_confidence.append(float(view_confidence) * depth_weight * float(contribution_weight))
            obs_depth_weight.append(depth_weight)
            obs_camera_z.append(candidate_z)
            obs_contribution_weight.append(float(contribution_weight))
            obs_contribution_score.append(float(score))
            obs_contribution_sum.append(denom)
            obs_surface_pixels.append([float(surface_pixels[row, 0]), float(surface_pixels[row, 1])])
            obs_surface_camera_z.append(float(depth))
            obs_zbuffer_weight.append(1.0)
            added += 1
    return added, z_reference


def _projected_covariances_2d(
    jacobians: np.ndarray,
    gaussian_scales: np.ndarray,
    gaussian_rotmats: np.ndarray,
    cov_eps_px: float,
) -> np.ndarray:
    scales2 = gaussian_scales.astype(np.float64) ** 2
    rotmats = gaussian_rotmats.astype(np.float64)
    cov3d = np.einsum("nai,ni,nbi->nab", rotmats, scales2, rotmats)
    tmp = np.einsum("nij,njk->nik", jacobians.astype(np.float64), cov3d)
    cov2d = np.einsum("nij,nkj->nik", tmp, jacobians.astype(np.float64))
    eps = float(cov_eps_px)
    cov2d[:, 0, 0] += eps
    cov2d[:, 1, 1] += eps
    return cov2d


def _append_view_gaussian_contribution_observations(
    points_world: np.ndarray,
    gaussian_scales: np.ndarray,
    gaussian_rotmats: np.ndarray,
    gaussian_opacities: np.ndarray,
    view_index: int,
    cfg: ViewConfig,
    modal: dict[str, np.ndarray],
    mode_index: int,
    mask_erode_iters: int,
    gaussian_contribution_radius: int,
    gaussian_contribution_min_share: float,
    gaussian_contribution_min_score: float,
    gaussian_contribution_cov_eps_px: float,
    obs_point_indices: list[int],
    obs_view_indices: list[int],
    obs_pixels: list[list[float]],
    obs_y: list[list[complex]],
    obs_j: list[np.ndarray],
    obs_confidence: list[float],
    obs_depth_weight: list[float],
    obs_camera_z: list[float],
    obs_contribution_weight: list[float],
    obs_contribution_score: list[float],
    obs_contribution_sum: list[float],
    view_confidence: float,
    depth_weighting: str,
    depth_weight_power: float,
    depth_weight_min: float,
    depth_weight_reference_percentile: float,
    obs_zbuffer_weight: list[float],
) -> tuple[int, float]:
    expected_shape = (cfg.image_height, cfg.image_width)
    mask = load_mask(cfg.mask_path, expected_shape)
    valid_mask = erode_mask(mask, mask_erode_iters)

    pixels_xy, z = project_points(points_world, cfg.K, cfg.world_to_camera)
    candidate, rounded_x, rounded_y = _candidate_mask(
        pixels_xy,
        z,
        valid_mask,
        cfg.image_width,
        cfg.image_height,
        margin=max(1, int(gaussian_contribution_radius) + 1),
    )
    candidate_indices = np.where(candidate)[0]
    if candidate_indices.size == 0:
        return 0, float("nan")
    z_reference = float(np.percentile(z[candidate_indices], depth_weight_reference_percentile))

    buckets = _build_pixel_buckets(candidate_indices, rounded_x, rounded_y, cfg.image_width)
    jacobians = projection_jacobian(points_world[candidate_indices], cfg.K, cfg.world_to_camera)
    candidate_row = np.full(points_world.shape[0], -1, dtype=np.int64)
    candidate_row[candidate_indices] = np.arange(candidate_indices.size, dtype=np.int64)
    cov2d = _projected_covariances_2d(
        jacobians,
        gaussian_scales[candidate_indices],
        gaussian_rotmats[candidate_indices],
        gaussian_contribution_cov_eps_px,
    )

    mode_u = modal["mode_u"][mode_index].astype(np.complex64)
    mode_v = modal["mode_v"][mode_index].astype(np.complex64)
    min_score = float(gaussian_contribution_min_score)
    added = 0
    for point_idx in candidate_indices.tolist():
        x = int(rounded_x[point_idx])
        y = int(rounded_y[point_idx])
        nearby = _bucket_indices_in_radius(
            buckets,
            x,
            y,
            cfg.image_width,
            cfg.image_height,
            int(gaussian_contribution_radius),
        )
        if nearby.size == 0:
            continue

        rows = candidate_row[nearby]
        valid_rows = rows >= 0
        if not np.any(valid_rows):
            continue
        nearby = nearby[valid_rows]
        rows = rows[valid_rows]

        delta = pixels_xy[point_idx][None, :].astype(np.float64) - pixels_xy[nearby].astype(np.float64)
        a = cov2d[rows, 0, 0]
        b = cov2d[rows, 0, 1]
        c = cov2d[rows, 1, 1]
        det = a * c - b * b
        valid_det = np.isfinite(det) & (det > 0)
        if not np.any(valid_det):
            continue
        dx = delta[:, 0]
        dy = delta[:, 1]
        quad = np.full((nearby.shape[0],), np.inf, dtype=np.float64)
        quad[valid_det] = (
            c[valid_det] * dx[valid_det] * dx[valid_det]
            - 2.0 * b[valid_det] * dx[valid_det] * dy[valid_det]
            + a[valid_det] * dy[valid_det] * dy[valid_det]
        ) / det[valid_det]
        scores = gaussian_opacities[nearby].astype(np.float64) * np.exp(-0.5 * quad)
        valid_scores = np.isfinite(scores) & (scores >= min_score)
        if not np.any(valid_scores):
            continue
        current = np.where(nearby == point_idx)[0]
        if current.size == 0:
            continue
        current_score = float(scores[int(current[0])])
        if not np.isfinite(current_score) or current_score < min_score:
            continue
        contribution_sum = float(np.sum(scores[valid_scores]))
        if contribution_sum <= 0.0 or not np.isfinite(contribution_sum):
            continue
        contribution_share = current_score / contribution_sum

        sample_xy = pixels_xy[point_idx : point_idx + 1]
        y_u = bilinear_sample(mode_u, sample_xy)[0]
        y_v = bilinear_sample(mode_v, sample_xy)[0]
        depth_weight = _depth_weight(
            float(z[point_idx]),
            z_reference,
            depth_weighting,
            depth_weight_power,
            depth_weight_min,
        )
        obs_point_indices.append(point_idx)
        obs_view_indices.append(view_index)
        obs_pixels.append([float(pixels_xy[point_idx, 0]), float(pixels_xy[point_idx, 1])])
        obs_y.append([complex(y_u), complex(y_v)])
        obs_j.append(jacobians[int(candidate_row[point_idx])].astype(np.float32))
        obs_confidence.append(float(view_confidence) * depth_weight * float(contribution_share))
        obs_depth_weight.append(depth_weight)
        obs_camera_z.append(float(z[point_idx]))
        obs_contribution_weight.append(float(contribution_share))
        obs_contribution_score.append(current_score)
        obs_contribution_sum.append(contribution_sum)
        obs_zbuffer_weight.append(1.0)
        added += 1
    return added, z_reference


def _depth_weight(
    z_value: float,
    z_reference: float,
    weighting: str,
    power: float,
    min_weight: float,
) -> float:
    if weighting == "none":
        return 1.0
    if weighting != "inverse-z":
        raise ValueError("depth_weighting must be 'none' or 'inverse-z'.")
    if z_value <= 0 or z_reference <= 0:
        return float(min_weight)
    weight = (float(z_reference) / float(z_value)) ** float(power)
    return float(np.clip(weight, float(min_weight), 1.0))


def _parse_pair_weight_specs(
    pair_weight_specs: Sequence[str] | None,
    configs: Sequence[ViewConfig],
) -> tuple[dict[tuple[int, int], float], np.ndarray]:
    if not pair_weight_specs:
        return {}, np.asarray([], dtype=str)
    view_ids = [cfg.view_id for cfg in configs]
    if len(set(view_ids)) != len(view_ids):
        raise ValueError("View ids must be unique when using --pair-weight.")
    view_to_index = {view_id: idx for idx, view_id in enumerate(view_ids)}
    pair_weights: dict[tuple[int, int], float] = {}
    normalized_specs: list[str] = []
    for raw in pair_weight_specs:
        parts = [part.strip() for part in str(raw).split(",")]
        if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
            raise ValueError(f"--pair-weight must have format viewA,viewB,weight, got: {raw}")
        view_a, view_b, weight_text = parts
        if view_a not in view_to_index:
            raise ValueError(f"Unknown view id in --pair-weight: {view_a}. Known views: {view_ids}")
        if view_b not in view_to_index:
            raise ValueError(f"Unknown view id in --pair-weight: {view_b}. Known views: {view_ids}")
        idx_a = view_to_index[view_a]
        idx_b = view_to_index[view_b]
        if idx_a == idx_b:
            raise ValueError(f"--pair-weight must reference two different views, got: {raw}")
        weight = float(weight_text)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"--pair-weight weight must be finite and non-negative, got: {raw}")
        key = tuple(sorted((idx_a, idx_b)))
        if key in pair_weights:
            raise ValueError(f"Duplicate --pair-weight for pair {view_ids[key[0]]},{view_ids[key[1]]}.")
        pair_weights[key] = weight
        normalized_specs.append(f"{view_ids[key[0]]},{view_ids[key[1]]},{weight:g}")
    return pair_weights, np.asarray(normalized_specs, dtype=str)


def _pair_weight_per_point(
    num_points: int,
    obs_point_arr: np.ndarray,
    obs_view_arr: np.ndarray,
    counts: np.ndarray,
    pair_weight_specs: Sequence[str] | None,
    configs: Sequence[ViewConfig],
) -> tuple[np.ndarray, np.ndarray]:
    pair_weights, normalized_specs = _parse_pair_weight_specs(pair_weight_specs, configs)
    out = np.ones((num_points,), dtype=np.float32)
    if not pair_weights:
        return out, normalized_specs
    point_view_mask = np.zeros((num_points, len(configs)), dtype=bool)
    point_view_mask[obs_point_arr, obs_view_arr] = True
    for point_idx in np.where(counts == 2)[0].tolist():
        views = np.where(point_view_mask[point_idx])[0]
        if views.size != 2:
            raise ValueError(f"Point {point_idx} has two observations but not two distinct observed views.")
        key = tuple(sorted((int(views[0]), int(views[1]))))
        out[point_idx] = float(pair_weights.get(key, 1.0))
    return out, normalized_specs


def _subset_carrier_points(carrier: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    n = carrier["points_world"].shape[0]
    out: dict[str, np.ndarray] = {}
    for key, value in carrier.items():
        if value.shape[:1] == (n,):
            out[key] = value[keep]
        else:
            out[key] = value
    return out


def _source_mask_keep(carrier: dict[str, np.ndarray], configs: Sequence[ViewConfig], erode_iters: int) -> np.ndarray:
    if erode_iters < 0:
        raise ValueError("source_mask_erode_iters must be non-negative.")
    required = ["source_view_index", "source_pixels_xy", "source_image_height", "source_image_width"]
    missing = [key for key in required if key not in carrier]
    if missing:
        raise ValueError(
            "Carrier point file is missing source-mask metadata "
            f"{missing}. Re-run preproc/run_vggt.py --export-points with the current code."
        )

    source_view_index = carrier["source_view_index"]
    source_pixels_xy = carrier["source_pixels_xy"]
    source_h = int(np.asarray(carrier["source_image_height"]).item())
    source_w = int(np.asarray(carrier["source_image_width"]).item())
    keep = np.zeros((carrier["points_world"].shape[0],), dtype=bool)

    for view_idx, cfg in enumerate(configs):
        rows = np.where(source_view_index == view_idx)[0]
        if rows.size == 0:
            continue
        mask = load_mask(cfg.mask_path, (cfg.image_height, cfg.image_width))
        valid_mask = erode_mask(mask, erode_iters)
        scale_x = float(cfg.image_width) / float(source_w)
        scale_y = float(cfg.image_height) / float(source_h)
        finite = np.all(np.isfinite(source_pixels_xy[rows]), axis=1)
        x = np.full((rows.size,), -1, dtype=np.int32)
        y = np.full((rows.size,), -1, dtype=np.int32)
        x[finite] = np.rint(source_pixels_xy[rows[finite], 0] * scale_x).astype(np.int32)
        y[finite] = np.rint(source_pixels_xy[rows[finite], 1] * scale_y).astype(np.int32)
        inside = finite & (x >= 0) & (x < cfg.image_width) & (y >= 0) & (y < cfg.image_height)
        if not np.any(inside):
            continue
        inside_rows = rows[inside]
        keep[inside_rows] = valid_mask[y[inside], x[inside]]
    return keep


def _append_view_observations(
    points_world: np.ndarray,
    view_index: int,
    cfg: ViewConfig,
    modal: dict[str, np.ndarray],
    mode_index: int,
    mask_erode_iters: int,
    zbuffer_radius: int,
    front_percentile: float,
    zbuffer_tau: float,
    min_zbuffer_samples: int,
    obs_point_indices: list[int],
    obs_view_indices: list[int],
    obs_pixels: list[list[float]],
    obs_y: list[list[complex]],
    obs_j: list[np.ndarray],
    obs_confidence: list[float],
    obs_depth_weight: list[float],
    obs_camera_z: list[float],
    view_confidence: float,
    depth_weighting: str,
    depth_weight_power: float,
    depth_weight_min: float,
    depth_weight_reference_percentile: float,
    obs_zbuffer_weight: list[float] | None = None,
    zbuffer_mode: str = "hard",
    zbuffer_soft_sigma: float = 0.10,
    zbuffer_soft_min_weight: float = 0.05,
) -> tuple[int, float]:
    expected_shape = (cfg.image_height, cfg.image_width)
    mask = load_mask(cfg.mask_path, expected_shape)
    valid_mask = erode_mask(mask, mask_erode_iters)

    pixels_xy, z = project_points(points_world, cfg.K, cfg.world_to_camera)
    candidate, rounded_x, rounded_y = _candidate_mask(
        pixels_xy,
        z,
        valid_mask,
        cfg.image_width,
        cfg.image_height,
        margin=max(1, zbuffer_radius + 1),
    )
    candidate_indices = np.where(candidate)[0]
    if candidate_indices.size == 0:
        return 0, float("nan")
    z_reference = float(np.percentile(z[candidate_indices], depth_weight_reference_percentile))

    buckets = _build_pixel_buckets(candidate_indices, rounded_x, rounded_y, cfg.image_width)
    jacobians = projection_jacobian(points_world[candidate_indices], cfg.K, cfg.world_to_camera)
    candidate_to_row = {int(point_idx): row for row, point_idx in enumerate(candidate_indices.tolist())}

    mode_u = modal["mode_u"][mode_index].astype(np.complex64)
    mode_v = modal["mode_v"][mode_index].astype(np.complex64)
    added = 0
    for point_idx in candidate_indices.tolist():
        x = int(rounded_x[point_idx])
        y = int(rounded_y[point_idx])
        stats = _local_depth_stats(
            buckets,
            z,
            x,
            y,
            cfg.image_width,
            cfg.image_height,
            zbuffer_radius,
            front_percentile,
            min_zbuffer_samples,
        )
        if stats is None:
            continue
        z_front, z_med = stats
        if z_med <= 0:
            continue
        z_weight = _zbuffer_weight(
            float(z[point_idx]),
            z_front,
            z_med,
            zbuffer_tau,
            zbuffer_mode,
            zbuffer_soft_sigma,
            zbuffer_soft_min_weight,
        )
        if z_weight is None:
            continue

        sample_xy = pixels_xy[point_idx : point_idx + 1]
        y_u = bilinear_sample(mode_u, sample_xy)[0]
        y_v = bilinear_sample(mode_v, sample_xy)[0]
        obs_point_indices.append(point_idx)
        obs_view_indices.append(view_index)
        obs_pixels.append([float(pixels_xy[point_idx, 0]), float(pixels_xy[point_idx, 1])])
        obs_y.append([complex(y_u), complex(y_v)])
        obs_j.append(jacobians[candidate_to_row[point_idx]].astype(np.float32))
        depth_weight = _depth_weight(
            float(z[point_idx]),
            z_reference,
            depth_weighting,
            depth_weight_power,
            depth_weight_min,
        )
        obs_confidence.append(float(view_confidence) * depth_weight * float(z_weight))
        obs_depth_weight.append(depth_weight)
        obs_camera_z.append(float(z[point_idx]))
        if obs_zbuffer_weight is not None:
            obs_zbuffer_weight.append(float(z_weight))
        added += 1
    return added, z_reference


def _write_point_observation_graph(
    points_world_all: np.ndarray,
    configs: Sequence[ViewConfig],
    modals: Sequence[dict[str, np.ndarray]],
    view_freqs_hz: np.ndarray,
    reference_freq_hz: float,
    reliability: dict[str, np.ndarray],
    out_path: str | Path,
    mode_index: int,
    mask_erode_iters: int,
    zbuffer_radius: int,
    front_percentile: float,
    zbuffer_tau: float,
    min_zbuffer_samples: int,
    min_observations: int,
    view_frequency_weighting: str,
    snr_band_hz: float,
    snr_exclude_hz: float,
    snr_good: float,
    view_weight_min: float,
    depth_weighting: str,
    depth_weight_power: float,
    depth_weight_min: float,
    depth_weight_reference_percentile: float,
    pair_weight_specs: Sequence[str] | None,
    source_view_config_paths: Sequence[str | Path],
    source_modal_npz_paths: Sequence[str | Path],
    preserve_all_points: bool = False,
    optional_point_fields: dict[str, np.ndarray] | None = None,
    extra_metadata: dict[str, np.ndarray] | None = None,
    observation_sampling: str = "gaussian-center",
    pixel_sample_stride: int = 4,
    pixel_candidate_k: int = 4,
    pixel_preselect_k: int = 32,
    pixel_render_acc_min: float = 0.05,
    pixel_min_contribution: float = 1e-12,
    pixel_min_mode_amp_percentile: float = 0.0,
    pixel_max_samples_per_view: int = 20000,
    gaussian_contribution_radius: int = 8,
    gaussian_contribution_min_share: float = 1e-4,
    gaussian_contribution_min_score: float = 1e-12,
    gaussian_contribution_cov_eps_px: float = 0.25,
    zbuffer_mode: str = "hard",
    zbuffer_soft_sigma: float = 0.10,
    zbuffer_soft_min_weight: float = 0.05,
    gaussian_scales: np.ndarray | None = None,
    gaussian_quats: np.ndarray | None = None,
    gaussian_opacities: np.ndarray | None = None,
    rendered_depths: Sequence[np.ndarray] | None = None,
    rendered_accs: Sequence[np.ndarray] | None = None,
) -> Path:
    points_world_all = np.asarray(points_world_all, dtype=np.float32)
    if points_world_all.ndim != 2 or points_world_all.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points_world_all.shape}.")
    contribution_modes = {"pixel-candidates", "gaussian-center-contribution"}
    if observation_sampling not in ({"gaussian-center"} | contribution_modes):
        raise ValueError("observation_sampling must be 'gaussian-center', 'pixel-candidates', or 'gaussian-center-contribution'.")
    if zbuffer_mode not in {"hard", "soft"}:
        raise ValueError("zbuffer_mode must be 'hard' or 'soft'.")
    contribution_inputs = None
    contribution_rotmats = None
    if observation_sampling in contribution_modes:
        contribution_inputs = _validate_gaussian_contribution_inputs(
            points_world_all,
            gaussian_scales,
            gaussian_quats,
            gaussian_opacities,
            rendered_depths,
            rendered_accs,
            configs,
            require_rendered=observation_sampling == "pixel-candidates",
        )
        contribution_rotmats = _quat_wxyz_to_rotmat(contribution_inputs[1])
    if observation_sampling == "pixel-candidates":
        if pixel_preselect_k < 1:
            raise ValueError("pixel_preselect_k must be at least 1.")
        if not (0.0 <= pixel_render_acc_min <= 1.0):
            raise ValueError("pixel_render_acc_min must be in [0, 1].")
        if pixel_min_contribution < 0:
            raise ValueError("pixel_min_contribution must be non-negative.")
    if observation_sampling == "gaussian-center-contribution":
        if gaussian_contribution_radius < 0:
            raise ValueError("gaussian_contribution_radius must be non-negative.")
        if not (0.0 <= gaussian_contribution_min_share <= 1.0):
            raise ValueError("gaussian_contribution_min_share must be in [0, 1].")
        if gaussian_contribution_min_score < 0:
            raise ValueError("gaussian_contribution_min_score must be non-negative.")
        if gaussian_contribution_cov_eps_px <= 0:
            raise ValueError("gaussian_contribution_cov_eps_px must be positive.")

    view_frequency_weights = reliability["weights"]
    obs_point_indices: list[int] = []
    obs_view_indices: list[int] = []
    obs_pixels: list[list[float]] = []
    obs_y: list[list[complex]] = []
    obs_j: list[np.ndarray] = []
    obs_confidence: list[float] = []
    obs_depth_weight: list[float] = []
    obs_camera_z: list[float] = []
    obs_contribution_weight: list[float] = []
    obs_contribution_score: list[float] = []
    obs_contribution_sum: list[float] = []
    obs_surface_pixels: list[list[float]] = []
    obs_surface_camera_z: list[float] = []
    obs_zbuffer_weight: list[float] = []
    observations_per_view: list[int] = []
    view_depth_reference_z: list[float] = []
    for view_index, (cfg, modal) in enumerate(zip(configs, modals)):
        if observation_sampling == "gaussian-center":
            count, z_reference = _append_view_observations(
                points_world_all,
                view_index,
                cfg,
                modal,
                mode_index,
                mask_erode_iters,
                zbuffer_radius,
                front_percentile,
                zbuffer_tau,
                min_zbuffer_samples,
                obs_point_indices,
                obs_view_indices,
                obs_pixels,
                obs_y,
                obs_j,
                obs_confidence,
                obs_depth_weight,
                obs_camera_z,
                float(view_frequency_weights[view_index]),
                depth_weighting,
                depth_weight_power,
                depth_weight_min,
                depth_weight_reference_percentile,
                obs_zbuffer_weight,
                zbuffer_mode,
                zbuffer_soft_sigma,
                zbuffer_soft_min_weight,
            )
        elif observation_sampling == "pixel-candidates":
            assert contribution_inputs is not None
            assert contribution_rotmats is not None
            contribution_scales, _, contribution_opacities, contribution_depths, contribution_accs = contribution_inputs
            count, z_reference = _append_view_contribution_observations(
                points_world_all,
                contribution_scales,
                contribution_rotmats,
                contribution_opacities,
                contribution_depths[view_index],
                contribution_accs[view_index],
                view_index,
                cfg,
                modal,
                mode_index,
                mask_erode_iters,
                obs_point_indices,
                obs_view_indices,
                obs_pixels,
                obs_y,
                obs_j,
                obs_confidence,
                obs_depth_weight,
                obs_camera_z,
                obs_contribution_weight,
                obs_contribution_score,
                obs_contribution_sum,
                obs_surface_pixels,
                obs_surface_camera_z,
                float(view_frequency_weights[view_index]),
                depth_weighting,
                depth_weight_power,
                depth_weight_min,
                depth_weight_reference_percentile,
                pixel_sample_stride,
                pixel_candidate_k,
                pixel_preselect_k,
                pixel_min_mode_amp_percentile,
                pixel_max_samples_per_view,
                pixel_render_acc_min,
                pixel_min_contribution,
                obs_zbuffer_weight,
            )
        else:
            assert contribution_inputs is not None
            assert contribution_rotmats is not None
            contribution_scales, _, contribution_opacities, _, _ = contribution_inputs
            count, z_reference = _append_view_gaussian_contribution_observations(
                points_world_all,
                contribution_scales,
                contribution_rotmats,
                contribution_opacities,
                view_index,
                cfg,
                modal,
                mode_index,
                mask_erode_iters,
                gaussian_contribution_radius,
                gaussian_contribution_min_share,
                gaussian_contribution_min_score,
                gaussian_contribution_cov_eps_px,
                obs_point_indices,
                obs_view_indices,
                obs_pixels,
                obs_y,
                obs_j,
                obs_confidence,
                obs_depth_weight,
                obs_camera_z,
                obs_contribution_weight,
                obs_contribution_score,
                obs_contribution_sum,
                float(view_frequency_weights[view_index]),
                depth_weighting,
                depth_weight_power,
                depth_weight_min,
                depth_weight_reference_percentile,
                obs_zbuffer_weight,
            )
        observations_per_view.append(count)
        view_depth_reference_z.append(z_reference)

    obs_point_arr = np.asarray(obs_point_indices, dtype=np.int64)
    obs_view_arr = np.asarray(obs_view_indices, dtype=np.int32)
    if obs_point_arr.size == 0:
        raise ValueError("No observations survived point z-buffer and mask checks.")

    gaussian_contribution_max_share_per_point: np.ndarray | None = None
    gaussian_contribution_contrast_keep: np.ndarray | None = None
    if observation_sampling == "gaussian-center-contribution":
        if len(obs_contribution_weight) != obs_point_arr.shape[0]:
            raise ValueError("Internal error: contribution weight count does not match observations.")
        contribution_weight_arr = np.asarray(obs_contribution_weight, dtype=np.float32)
        gaussian_contribution_max_share_per_point = np.zeros(points_world_all.shape[0], dtype=np.float32)
        np.maximum.at(gaussian_contribution_max_share_per_point, obs_point_arr, contribution_weight_arr)
        contrast_ratio = float(gaussian_contribution_min_share)
        gaussian_contribution_contrast_keep = contribution_weight_arr >= (
            contrast_ratio * gaussian_contribution_max_share_per_point[obs_point_arr]
        )
        if not np.any(gaussian_contribution_contrast_keep):
            raise ValueError(
                "No gaussian-center-contribution observations survived relative contrast gating. "
                "Lower --gaussian-contribution-min-share or use 0 to disable it."
            )
        if not np.all(gaussian_contribution_contrast_keep):
            keep_indices = np.flatnonzero(gaussian_contribution_contrast_keep)
            obs_point_arr = obs_point_arr[keep_indices]
            obs_view_arr = obs_view_arr[keep_indices]
            obs_point_indices = obs_point_arr.tolist()
            obs_view_indices = obs_view_arr.tolist()
            obs_pixels = [obs_pixels[int(i)] for i in keep_indices]
            obs_y = [obs_y[int(i)] for i in keep_indices]
            obs_j = [obs_j[int(i)] for i in keep_indices]
            obs_confidence = [obs_confidence[int(i)] for i in keep_indices]
            obs_depth_weight = [obs_depth_weight[int(i)] for i in keep_indices]
            obs_camera_z = [obs_camera_z[int(i)] for i in keep_indices]
            obs_contribution_weight = [obs_contribution_weight[int(i)] for i in keep_indices]
            obs_contribution_score = [obs_contribution_score[int(i)] for i in keep_indices]
            obs_contribution_sum = [obs_contribution_sum[int(i)] for i in keep_indices]
            obs_zbuffer_weight = [obs_zbuffer_weight[int(i)] for i in keep_indices]
            observations_per_view = np.bincount(obs_view_arr, minlength=len(configs)).astype(np.int32).tolist()
        gaussian_contribution_contrast_keep = np.ones(obs_point_arr.shape[0], dtype=bool)

    sample_counts = np.bincount(obs_point_arr, minlength=points_world_all.shape[0])
    point_view_mask = np.zeros((points_world_all.shape[0], len(configs)), dtype=bool)
    point_view_mask[obs_point_arr, obs_view_arr] = True
    counts = point_view_mask.sum(axis=1).astype(np.int32)
    pair_weight_per_point, normalized_pair_weight_specs = _pair_weight_per_point(
        points_world_all.shape[0],
        obs_point_arr,
        obs_view_arr,
        counts,
        pair_weight_specs,
        configs,
    )
    obs_pair_weight = pair_weight_per_point[obs_point_arr].astype(np.float32)
    obs_confidence_arr = np.asarray(obs_confidence, dtype=np.float32) * obs_pair_weight
    observation_keep_points = counts >= int(min_observations)
    if not np.any(observation_keep_points):
        raise ValueError("No points satisfy min_observations.")
    used_counts = counts.copy()
    used_counts[~observation_keep_points] = 0

    if preserve_all_points:
        active_old_indices = np.arange(points_world_all.shape[0], dtype=np.int64)
        old_to_new = active_old_indices.copy()
    else:
        active_old_indices = np.where(observation_keep_points)[0]
        old_to_new = np.full(points_world_all.shape[0], -1, dtype=np.int64)
        old_to_new[active_old_indices] = np.arange(active_old_indices.size, dtype=np.int64)
    keep_obs = observation_keep_points[obs_point_arr]
    obs_fields_out: dict[str, np.ndarray] = {}
    if observation_sampling in contribution_modes:
        for key, values in (
            ("obs_contribution_weight", obs_contribution_weight),
            ("obs_contribution_score", obs_contribution_score),
            ("obs_contribution_sum", obs_contribution_sum),
        ):
            if len(values) != obs_point_arr.shape[0]:
                raise ValueError(f"Internal error: {key} count does not match observations.")
        obs_fields_out["obs_contribution_weight"] = np.asarray(obs_contribution_weight, dtype=np.float32)[keep_obs]
        obs_fields_out["obs_contribution_score"] = np.asarray(obs_contribution_score, dtype=np.float32)[keep_obs]
        obs_fields_out["obs_contribution_sum"] = np.asarray(obs_contribution_sum, dtype=np.float32)[keep_obs]
        if observation_sampling == "gaussian-center-contribution":
            if gaussian_contribution_contrast_keep is None:
                raise ValueError("Internal error: gaussian contribution contrast keep mask was not computed.")
            obs_fields_out["gaussian_contribution_contrast_keep"] = gaussian_contribution_contrast_keep[keep_obs]
    if observation_sampling == "pixel-candidates":
        for key, values in (
            ("obs_surface_pixels_xy", obs_surface_pixels),
            ("obs_surface_camera_z", obs_surface_camera_z),
        ):
            if len(values) != obs_point_arr.shape[0]:
                raise ValueError(f"Internal error: {key} count does not match observations.")
        obs_fields_out["obs_surface_pixels_xy"] = np.asarray(obs_surface_pixels, dtype=np.float32)[keep_obs]
        obs_fields_out["obs_surface_camera_z"] = np.asarray(obs_surface_camera_z, dtype=np.float32)[keep_obs]
    if len(obs_zbuffer_weight) != obs_point_arr.shape[0]:
        raise ValueError("Internal error: z-buffer weight count does not match observations.")
    obs_fields_out["obs_zbuffer_weight"] = np.asarray(obs_zbuffer_weight, dtype=np.float32)[keep_obs]

    optional_point_fields = dict(optional_point_fields or {})
    point_fields_out: dict[str, np.ndarray] = {}
    for key, value in optional_point_fields.items():
        value = np.asarray(value)
        if value.shape[:1] != (points_world_all.shape[0],):
            raise ValueError(
                f"Optional point field {key} must have first dimension "
                f"{points_world_all.shape[0]}, got {value.shape}."
            )
        point_fields_out[key] = value[active_old_indices]
    if gaussian_contribution_max_share_per_point is not None:
        point_fields_out["gaussian_contribution_max_share_per_point"] = gaussian_contribution_max_share_per_point[
            active_old_indices
        ]

    metadata = dict(extra_metadata or {})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points_world=points_world_all[active_old_indices].astype(np.float32),
        obs_point_index=old_to_new[obs_point_arr[keep_obs]].astype(np.int32),
        obs_view_index=obs_view_arr[keep_obs].astype(np.int32),
        obs_pixels_xy=np.asarray(obs_pixels, dtype=np.float32)[keep_obs],
        obs_y=np.asarray(obs_y, dtype=np.complex64)[keep_obs],
        obs_J=np.asarray(obs_j, dtype=np.float32)[keep_obs],
        obs_confidence=obs_confidence_arr[keep_obs],
        obs_depth_weight=np.asarray(obs_depth_weight, dtype=np.float32)[keep_obs],
        obs_pair_weight=obs_pair_weight[keep_obs],
        obs_camera_z=np.asarray(obs_camera_z, dtype=np.float32)[keep_obs],
        obs_count_per_point=used_counts[active_old_indices].astype(np.int32),
        obs_sample_count_per_point=sample_counts[active_old_indices].astype(np.int32),
        pair_weight_per_point=pair_weight_per_point[active_old_indices],
        pair_weight_specs=normalized_pair_weight_specs,
        view_ids=np.asarray([cfg.view_id for cfg in configs]),
        view_image_width=np.asarray([cfg.image_width for cfg in configs], dtype=np.int32),
        view_image_height=np.asarray([cfg.image_height for cfg in configs], dtype=np.int32),
        view_freqs_hz=view_freqs_hz.astype(np.float32),
        view_frequency_weighting=np.array(str(view_frequency_weighting)),
        view_frequency_weights=view_frequency_weights.astype(np.float32),
        view_frequency_snr=reliability["snr"].astype(np.float32),
        view_frequency_signal=reliability["signal"].astype(np.float32),
        view_frequency_noise=reliability["noise"].astype(np.float32),
        view_frequency_bin_hz=reliability["bin_hz"].astype(np.float32),
        view_depth_reference_z=np.asarray(view_depth_reference_z, dtype=np.float32),
        depth_weighting=np.array(str(depth_weighting)),
        depth_weight_power=np.array(depth_weight_power, dtype=np.float32),
        depth_weight_min=np.array(depth_weight_min, dtype=np.float32),
        depth_weight_reference_percentile=np.array(depth_weight_reference_percentile, dtype=np.float32),
        snr_band_hz=np.array(snr_band_hz, dtype=np.float32),
        snr_exclude_hz=np.array(snr_exclude_hz, dtype=np.float32),
        snr_good=np.array(snr_good, dtype=np.float32),
        view_weight_min=np.array(view_weight_min, dtype=np.float32),
        freq_hz=np.array(reference_freq_hz, dtype=np.float32),
        mode_index=np.array(mode_index, dtype=np.int32),
        min_observations=np.array(min_observations, dtype=np.int32),
        zbuffer_radius=np.array(zbuffer_radius, dtype=np.int32),
        zbuffer_mode=np.array(str(zbuffer_mode)),
        zbuffer_soft_sigma=np.array(zbuffer_soft_sigma, dtype=np.float32),
        zbuffer_soft_min_weight=np.array(zbuffer_soft_min_weight, dtype=np.float32),
        front_percentile=np.array(front_percentile, dtype=np.float32),
        zbuffer_tau=np.array(zbuffer_tau, dtype=np.float32),
        min_zbuffer_samples=np.array(min_zbuffer_samples, dtype=np.int32),
        mask_erode_iters=np.array(mask_erode_iters, dtype=np.int32),
        candidate_point_count=np.array(points_world_all.shape[0], dtype=np.int32),
        preserved_all_points=np.array(bool(preserve_all_points)),
        observation_sampling=np.array(str(observation_sampling)),
        pixel_sample_stride=np.array(pixel_sample_stride, dtype=np.int32),
        pixel_candidate_k=np.array(pixel_candidate_k, dtype=np.int32),
        pixel_preselect_k=np.array(pixel_preselect_k, dtype=np.int32),
        pixel_render_acc_min=np.array(pixel_render_acc_min, dtype=np.float32),
        pixel_min_contribution=np.array(pixel_min_contribution, dtype=np.float32),
        pixel_min_mode_amp_percentile=np.array(pixel_min_mode_amp_percentile, dtype=np.float32),
        pixel_max_samples_per_view=np.array(pixel_max_samples_per_view, dtype=np.int32),
        gaussian_contribution_radius=np.array(gaussian_contribution_radius, dtype=np.int32),
        gaussian_contribution_min_share=np.array(gaussian_contribution_min_share, dtype=np.float32),
        gaussian_contribution_min_score=np.array(gaussian_contribution_min_score, dtype=np.float32),
        gaussian_contribution_cov_eps_px=np.array(gaussian_contribution_cov_eps_px, dtype=np.float32),
        pixel_candidate_method=np.array(
            "rendered_depth_gaussian_contribution" if observation_sampling == "pixel-candidates" else ""
        ),
        gaussian_candidate_method=np.array(
            "projected_gaussian_contribution_share" if observation_sampling == "gaussian-center-contribution" else ""
        ),
        observations_per_view=np.asarray(observations_per_view, dtype=np.int32),
        source_view_configs=np.asarray([str(path) for path in source_view_config_paths]),
        source_modal_npzs=np.asarray([str(path) for path in source_modal_npz_paths]),
        **obs_fields_out,
        **point_fields_out,
        **metadata,
    )
    return out


def _validate_observation_graph_args(
    min_observations: int,
    zbuffer_radius: int,
    front_percentile: float,
    zbuffer_tau: float,
    min_zbuffer_samples: int,
    depth_weighting: str,
    depth_weight_power: float,
    depth_weight_min: float,
    depth_weight_reference_percentile: float,
    zbuffer_mode: str = "hard",
    zbuffer_soft_sigma: float = 0.10,
    zbuffer_soft_min_weight: float = 0.05,
) -> None:
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1.")
    if zbuffer_radius < 0:
        raise ValueError("zbuffer_radius must be non-negative.")
    if not (0.0 <= front_percentile <= 100.0):
        raise ValueError("front_percentile must be in [0, 100].")
    if zbuffer_tau <= 0:
        raise ValueError("zbuffer_tau must be positive.")
    if zbuffer_mode not in {"hard", "soft"}:
        raise ValueError("zbuffer_mode must be 'hard' or 'soft'.")
    if zbuffer_soft_sigma <= 0:
        raise ValueError("zbuffer_soft_sigma must be positive.")
    if not (0.0 <= zbuffer_soft_min_weight <= 1.0):
        raise ValueError("zbuffer_soft_min_weight must be in [0, 1].")
    if min_zbuffer_samples < 1:
        raise ValueError("min_zbuffer_samples must be at least 1.")
    if depth_weighting not in {"none", "inverse-z"}:
        raise ValueError("depth_weighting must be 'none' or 'inverse-z'.")
    if depth_weight_power <= 0:
        raise ValueError("depth_weight_power must be positive.")
    if not (0.0 <= depth_weight_min <= 1.0):
        raise ValueError("depth_weight_min must be in [0, 1].")
    if not (0.0 <= depth_weight_reference_percentile <= 100.0):
        raise ValueError("depth_weight_reference_percentile must be in [0, 100].")


def _validate_pixel_candidate_args(
    pixel_sample_stride: int,
    pixel_candidate_k: int,
    pixel_preselect_k: int,
    pixel_render_acc_min: float,
    pixel_min_contribution: float,
    pixel_min_mode_amp_percentile: float,
    pixel_max_samples_per_view: int,
) -> None:
    if pixel_sample_stride < 1:
        raise ValueError("pixel_sample_stride must be at least 1.")
    if pixel_candidate_k < 1:
        raise ValueError("pixel_candidate_k must be at least 1.")
    if pixel_preselect_k < pixel_candidate_k:
        raise ValueError("pixel_preselect_k must be at least pixel_candidate_k.")
    if not (0.0 <= pixel_render_acc_min <= 1.0):
        raise ValueError("pixel_render_acc_min must be in [0, 1].")
    if pixel_min_contribution < 0:
        raise ValueError("pixel_min_contribution must be non-negative.")
    if not (0.0 <= pixel_min_mode_amp_percentile <= 100.0):
        raise ValueError("pixel_min_mode_amp_percentile must be in [0, 100].")
    if pixel_max_samples_per_view < 1:
        raise ValueError("pixel_max_samples_per_view must be at least 1.")


def _validate_gaussian_contribution_args(
    gaussian_contribution_radius: int,
    gaussian_contribution_min_share: float,
    gaussian_contribution_min_score: float,
    gaussian_contribution_cov_eps_px: float,
) -> None:
    if gaussian_contribution_radius < 0:
        raise ValueError("gaussian_contribution_radius must be non-negative.")
    if not (0.0 <= gaussian_contribution_min_share <= 1.0):
        raise ValueError("gaussian_contribution_min_share must be in [0, 1].")
    if gaussian_contribution_min_score < 0:
        raise ValueError("gaussian_contribution_min_score must be non-negative.")
    if gaussian_contribution_cov_eps_px <= 0:
        raise ValueError("gaussian_contribution_cov_eps_px must be positive.")


def build_points_observation_graph(
    points_world: np.ndarray,
    view_config_paths: Sequence[str | Path],
    modal_npz_paths: Sequence[str | Path],
    out_path: str | Path,
    mode_index: int = 0,
    mask_erode_iters: int = 1,
    zbuffer_radius: int = 5,
    front_percentile: float = 10.0,
    zbuffer_tau: float = 0.05,
    min_zbuffer_samples: int = 5,
    min_observations: int = 1,
    freq_tolerance_hz: float = 0.1,
    view_frequency_weighting: str = "none",
    snr_band_hz: float = 0.3,
    snr_exclude_hz: float = 0.08,
    snr_good: float = 3.0,
    view_weight_min: float = 0.05,
    depth_weighting: str = "none",
    depth_weight_power: float = 2.0,
    depth_weight_min: float = 0.02,
    depth_weight_reference_percentile: float = 50.0,
    pair_weight_specs: Sequence[str] | None = None,
    preserve_all_points: bool = False,
    optional_point_fields: dict[str, np.ndarray] | None = None,
    extra_metadata: dict[str, np.ndarray] | None = None,
    observation_sampling: str = "gaussian-center",
    pixel_sample_stride: int = 4,
    pixel_candidate_k: int = 4,
    pixel_preselect_k: int = 32,
    pixel_render_acc_min: float = 0.05,
    pixel_min_contribution: float = 1e-12,
    pixel_min_mode_amp_percentile: float = 0.0,
    pixel_max_samples_per_view: int = 20000,
    gaussian_contribution_radius: int = 8,
    gaussian_contribution_min_share: float = 1e-4,
    gaussian_contribution_min_score: float = 1e-12,
    gaussian_contribution_cov_eps_px: float = 0.25,
    zbuffer_mode: str = "hard",
    zbuffer_soft_sigma: float = 0.10,
    zbuffer_soft_min_weight: float = 0.05,
    gaussian_scales: np.ndarray | None = None,
    gaussian_quats: np.ndarray | None = None,
    gaussian_opacities: np.ndarray | None = None,
    rendered_depths: Sequence[np.ndarray] | None = None,
    rendered_accs: Sequence[np.ndarray] | None = None,
) -> Path:
    """Build an N-view observation graph for an arbitrary fixed 3D point set."""
    _validate_observation_graph_args(
        min_observations,
        zbuffer_radius,
        front_percentile,
        zbuffer_tau,
        min_zbuffer_samples,
        depth_weighting,
        depth_weight_power,
        depth_weight_min,
        depth_weight_reference_percentile,
        zbuffer_mode,
        zbuffer_soft_sigma,
        zbuffer_soft_min_weight,
    )
    if observation_sampling not in {"gaussian-center", "pixel-candidates", "gaussian-center-contribution"}:
        raise ValueError("observation_sampling must be 'gaussian-center', 'pixel-candidates', or 'gaussian-center-contribution'.")
    if observation_sampling == "pixel-candidates":
        _validate_pixel_candidate_args(
            pixel_sample_stride,
            pixel_candidate_k,
            pixel_preselect_k,
            pixel_render_acc_min,
            pixel_min_contribution,
            pixel_min_mode_amp_percentile,
            pixel_max_samples_per_view,
        )
    if observation_sampling == "gaussian-center-contribution":
        _validate_gaussian_contribution_args(
            gaussian_contribution_radius,
            gaussian_contribution_min_share,
            gaussian_contribution_min_score,
            gaussian_contribution_cov_eps_px,
        )
    configs, modals, view_freqs_hz, reference_freq_hz = _load_view_inputs(
        view_config_paths,
        modal_npz_paths,
        mode_index,
        freq_tolerance_hz,
    )
    reliability = _view_frequency_reliability(
        modals,
        view_freqs_hz,
        view_frequency_weighting,
        snr_band_hz,
        snr_exclude_hz,
        snr_good,
        view_weight_min,
    )
    return _write_point_observation_graph(
        points_world,
        configs,
        modals,
        view_freqs_hz,
        reference_freq_hz,
        reliability,
        out_path,
        mode_index,
        mask_erode_iters,
        zbuffer_radius,
        front_percentile,
        zbuffer_tau,
        min_zbuffer_samples,
        min_observations,
        view_frequency_weighting,
        snr_band_hz,
        snr_exclude_hz,
        snr_good,
        view_weight_min,
        depth_weighting,
        depth_weight_power,
        depth_weight_min,
        depth_weight_reference_percentile,
        pair_weight_specs,
        view_config_paths,
        modal_npz_paths,
        preserve_all_points=preserve_all_points,
        optional_point_fields=optional_point_fields,
        extra_metadata=extra_metadata,
        observation_sampling=observation_sampling,
        pixel_sample_stride=pixel_sample_stride,
        pixel_candidate_k=pixel_candidate_k,
        pixel_preselect_k=pixel_preselect_k,
        pixel_render_acc_min=pixel_render_acc_min,
        pixel_min_contribution=pixel_min_contribution,
        pixel_min_mode_amp_percentile=pixel_min_mode_amp_percentile,
        pixel_max_samples_per_view=pixel_max_samples_per_view,
        gaussian_contribution_radius=gaussian_contribution_radius,
        gaussian_contribution_min_share=gaussian_contribution_min_share,
        gaussian_contribution_min_score=gaussian_contribution_min_score,
        gaussian_contribution_cov_eps_px=gaussian_contribution_cov_eps_px,
        zbuffer_mode=zbuffer_mode,
        zbuffer_soft_sigma=zbuffer_soft_sigma,
        zbuffer_soft_min_weight=zbuffer_soft_min_weight,
        gaussian_scales=gaussian_scales,
        gaussian_quats=gaussian_quats,
        gaussian_opacities=gaussian_opacities,
        rendered_depths=rendered_depths,
        rendered_accs=rendered_accs,
    )


def build_carrier_observation_graph(
    carrier_points_path: str | Path,
    view_config_paths: Sequence[str | Path],
    modal_npz_paths: Sequence[str | Path],
    out_path: str | Path,
    mode_index: int = 0,
    mask_erode_iters: int = 1,
    source_mask_erode_iters: int = 1,
    zbuffer_radius: int = 5,
    front_percentile: float = 10.0,
    zbuffer_tau: float = 0.05,
    min_zbuffer_samples: int = 5,
    min_observations: int = 1,
    freq_tolerance_hz: float = 0.1,
    view_frequency_weighting: str = "none",
    snr_band_hz: float = 0.3,
    snr_exclude_hz: float = 0.08,
    snr_good: float = 3.0,
    view_weight_min: float = 0.05,
    depth_weighting: str = "none",
    depth_weight_power: float = 2.0,
    depth_weight_min: float = 0.02,
    depth_weight_reference_percentile: float = 50.0,
    pair_weight_specs: Sequence[str] | None = None,
) -> Path:
    """Build an N-view observation graph using VGGT carrier points."""
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1.")
    if zbuffer_radius < 0:
        raise ValueError("zbuffer_radius must be non-negative.")
    if not (0.0 <= front_percentile <= 100.0):
        raise ValueError("front_percentile must be in [0, 100].")
    if zbuffer_tau <= 0:
        raise ValueError("zbuffer_tau must be positive.")
    if min_zbuffer_samples < 1:
        raise ValueError("min_zbuffer_samples must be at least 1.")
    if depth_weighting not in {"none", "inverse-z"}:
        raise ValueError("depth_weighting must be 'none' or 'inverse-z'.")
    if depth_weight_power <= 0:
        raise ValueError("depth_weight_power must be positive.")
    if not (0.0 <= depth_weight_min <= 1.0):
        raise ValueError("depth_weight_min must be in [0, 1].")
    if not (0.0 <= depth_weight_reference_percentile <= 100.0):
        raise ValueError("depth_weight_reference_percentile must be in [0, 100].")

    carrier = _load_carrier_points(carrier_points_path)
    configs, modals, view_freqs_hz, reference_freq_hz = _load_view_inputs(
        view_config_paths,
        modal_npz_paths,
        mode_index,
        freq_tolerance_hz,
    )
    reliability = _view_frequency_reliability(
        modals,
        view_freqs_hz,
        view_frequency_weighting,
        snr_band_hz,
        snr_exclude_hz,
        snr_good,
        view_weight_min,
    )
    view_frequency_weights = reliability["weights"]
    source_mask_candidate_count = int(carrier["points_world"].shape[0])
    source_keep = _source_mask_keep(carrier, configs, source_mask_erode_iters)
    source_mask_kept_count = int(source_keep.sum())
    if source_mask_kept_count == 0:
        raise ValueError("No VGGT carrier points survived source-mask filtering.")
    carrier = _subset_carrier_points(carrier, source_keep)
    points_world_all = carrier["points_world"]

    obs_point_indices: list[int] = []
    obs_view_indices: list[int] = []
    obs_pixels: list[list[float]] = []
    obs_y: list[list[complex]] = []
    obs_j: list[np.ndarray] = []
    obs_confidence: list[float] = []
    obs_depth_weight: list[float] = []
    obs_camera_z: list[float] = []
    observations_per_view: list[int] = []
    view_depth_reference_z: list[float] = []
    for view_index, (cfg, modal) in enumerate(zip(configs, modals)):
        count, z_reference = _append_view_observations(
            points_world_all,
            view_index,
            cfg,
            modal,
            mode_index,
            mask_erode_iters,
            zbuffer_radius,
            front_percentile,
            zbuffer_tau,
            min_zbuffer_samples,
            obs_point_indices,
            obs_view_indices,
            obs_pixels,
            obs_y,
            obs_j,
            obs_confidence,
            obs_depth_weight,
            obs_camera_z,
            float(view_frequency_weights[view_index]),
            depth_weighting,
            depth_weight_power,
            depth_weight_min,
            depth_weight_reference_percentile,
        )
        observations_per_view.append(count)
        view_depth_reference_z.append(z_reference)

    obs_point_arr = np.asarray(obs_point_indices, dtype=np.int64)
    obs_view_arr = np.asarray(obs_view_indices, dtype=np.int32)
    if obs_point_arr.size == 0:
        raise ValueError("No observations survived carrier z-buffer and mask checks.")
    counts = np.bincount(obs_point_arr, minlength=points_world_all.shape[0])
    pair_weight_per_point, normalized_pair_weight_specs = _pair_weight_per_point(
        points_world_all.shape[0],
        obs_point_arr,
        obs_view_arr,
        counts,
        pair_weight_specs,
        configs,
    )
    obs_pair_weight = pair_weight_per_point[obs_point_arr].astype(np.float32)
    obs_confidence_arr = np.asarray(obs_confidence, dtype=np.float32) * obs_pair_weight
    keep_points = counts >= int(min_observations)
    if not np.any(keep_points):
        raise ValueError("No carrier points satisfy min_observations.")

    old_to_new = np.full(points_world_all.shape[0], -1, dtype=np.int64)
    active_old_indices = np.where(keep_points)[0]
    old_to_new[active_old_indices] = np.arange(active_old_indices.size, dtype=np.int64)
    keep_obs = keep_points[obs_point_arr]

    optional_point_fields = {}
    if "source_view_index" in carrier:
        source_view_index = carrier["source_view_index"]
        source_mask = np.zeros((points_world_all.shape[0], len(configs)), dtype=bool)
        valid_source = (source_view_index >= 0) & (source_view_index < len(configs))
        source_mask[np.where(valid_source)[0], source_view_index[valid_source]] = True
        optional_point_fields["point_source_view_mask"] = source_mask[active_old_indices]
        optional_point_fields["point_source_count"] = source_mask[active_old_indices].sum(axis=1).astype(np.int32)
    if "source_pixels_xy" in carrier:
        optional_point_fields["source_pixels_xy"] = carrier["source_pixels_xy"][active_old_indices].astype(np.float32)
    if "colors" in carrier:
        optional_point_fields["colors"] = carrier["colors"][active_old_indices].astype(np.uint8)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points_world=points_world_all[active_old_indices].astype(np.float32),
        carrier_point_indices=carrier["original_indices"][active_old_indices].astype(np.int32),
        obs_point_index=old_to_new[obs_point_arr[keep_obs]].astype(np.int32),
        obs_view_index=obs_view_arr[keep_obs].astype(np.int32),
        obs_pixels_xy=np.asarray(obs_pixels, dtype=np.float32)[keep_obs],
        obs_y=np.asarray(obs_y, dtype=np.complex64)[keep_obs],
        obs_J=np.asarray(obs_j, dtype=np.float32)[keep_obs],
        obs_confidence=obs_confidence_arr[keep_obs],
        obs_depth_weight=np.asarray(obs_depth_weight, dtype=np.float32)[keep_obs],
        obs_pair_weight=obs_pair_weight[keep_obs],
        obs_camera_z=np.asarray(obs_camera_z, dtype=np.float32)[keep_obs],
        obs_count_per_point=counts[active_old_indices].astype(np.int32),
        pair_weight_per_point=pair_weight_per_point[active_old_indices],
        pair_weight_specs=normalized_pair_weight_specs,
        view_ids=np.asarray([cfg.view_id for cfg in configs]),
        view_image_width=np.asarray([cfg.image_width for cfg in configs], dtype=np.int32),
        view_image_height=np.asarray([cfg.image_height for cfg in configs], dtype=np.int32),
        view_freqs_hz=view_freqs_hz.astype(np.float32),
        view_frequency_weighting=np.array(str(view_frequency_weighting)),
        view_frequency_weights=view_frequency_weights.astype(np.float32),
        view_frequency_snr=reliability["snr"].astype(np.float32),
        view_frequency_signal=reliability["signal"].astype(np.float32),
        view_frequency_noise=reliability["noise"].astype(np.float32),
        view_frequency_bin_hz=reliability["bin_hz"].astype(np.float32),
        view_depth_reference_z=np.asarray(view_depth_reference_z, dtype=np.float32),
        depth_weighting=np.array(str(depth_weighting)),
        depth_weight_power=np.array(depth_weight_power, dtype=np.float32),
        depth_weight_min=np.array(depth_weight_min, dtype=np.float32),
        depth_weight_reference_percentile=np.array(depth_weight_reference_percentile, dtype=np.float32),
        snr_band_hz=np.array(snr_band_hz, dtype=np.float32),
        snr_exclude_hz=np.array(snr_exclude_hz, dtype=np.float32),
        snr_good=np.array(snr_good, dtype=np.float32),
        view_weight_min=np.array(view_weight_min, dtype=np.float32),
        freq_hz=np.array(reference_freq_hz, dtype=np.float32),
        mode_index=np.array(mode_index, dtype=np.int32),
        min_observations=np.array(min_observations, dtype=np.int32),
        zbuffer_radius=np.array(zbuffer_radius, dtype=np.int32),
        front_percentile=np.array(front_percentile, dtype=np.float32),
        zbuffer_tau=np.array(zbuffer_tau, dtype=np.float32),
        min_zbuffer_samples=np.array(min_zbuffer_samples, dtype=np.int32),
        mask_erode_iters=np.array(mask_erode_iters, dtype=np.int32),
        source_mask_erode_iters=np.array(source_mask_erode_iters, dtype=np.int32),
        source_mask_candidate_count=np.array(source_mask_candidate_count, dtype=np.int32),
        source_mask_kept_count=np.array(source_mask_kept_count, dtype=np.int32),
        candidate_point_count=np.array(points_world_all.shape[0], dtype=np.int32),
        observations_per_view=np.asarray(observations_per_view, dtype=np.int32),
        source_carrier_points=np.array(str(carrier_points_path)),
        source_view_configs=np.asarray([str(path) for path in view_config_paths]),
        source_modal_npzs=np.asarray([str(path) for path in modal_npz_paths]),
        **optional_point_fields,
    )
    return out
