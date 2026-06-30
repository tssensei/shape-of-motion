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

from modal_surface.geometry import bilinear_sample, erode_mask, in_image_with_margin, project_points, projection_jacobian
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
    if len(view_config_paths) < 2:
        raise ValueError("At least two views are required.")
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
    view_confidence: float,
) -> int:
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
        return 0

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
        if abs(float(z[point_idx]) - z_front) / max(abs(z_med), 1e-6) >= zbuffer_tau:
            continue

        sample_xy = pixels_xy[point_idx : point_idx + 1]
        y_u = bilinear_sample(mode_u, sample_xy)[0]
        y_v = bilinear_sample(mode_v, sample_xy)[0]
        obs_point_indices.append(point_idx)
        obs_view_indices.append(view_index)
        obs_pixels.append([float(pixels_xy[point_idx, 0]), float(pixels_xy[point_idx, 1])])
        obs_y.append([complex(y_u), complex(y_v)])
        obs_j.append(jacobians[candidate_to_row[point_idx]].astype(np.float32))
        obs_confidence.append(float(view_confidence))
        added += 1
    return added


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
    observations_per_view: list[int] = []
    for view_index, (cfg, modal) in enumerate(zip(configs, modals)):
        count = _append_view_observations(
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
            float(view_frequency_weights[view_index]),
        )
        observations_per_view.append(count)

    obs_point_arr = np.asarray(obs_point_indices, dtype=np.int64)
    obs_view_arr = np.asarray(obs_view_indices, dtype=np.int32)
    if obs_point_arr.size == 0:
        raise ValueError("No observations survived carrier z-buffer and mask checks.")
    counts = np.bincount(obs_point_arr, minlength=points_world_all.shape[0])
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
        obs_confidence=np.asarray(obs_confidence, dtype=np.float32)[keep_obs],
        obs_count_per_point=counts[active_old_indices].astype(np.int32),
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
