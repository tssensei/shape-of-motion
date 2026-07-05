import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger as guru

from modal_surface.geometry import (
    bilinear_sample,
    erode_mask,
    project_points,
    projection_jacobian,
)
from modal_surface.io import load_view_config


@dataclass(frozen=True)
class ModalModeData:
    mode_index: int
    freq_hz: float
    latent_path: Path
    points_world: np.ndarray
    phi: np.ndarray


@dataclass(frozen=True)
class ModalFrameMap:
    view_ids: list[str]
    frame_view_indices: torch.Tensor
    frame_local_indices: torch.Tensor
    smooth_triplets: torch.Tensor


@dataclass(frozen=True)
class ModalConsistencyData:
    y_real: torch.Tensor
    y_imag: torch.Tensor
    J: torch.Tensor
    gaussian_indices: torch.Tensor
    mode_indices: torch.Tensor
    group_indices: torch.Tensor
    group_count: int


def load_modal_modes(manifest_path: str) -> list[ModalModeData]:
    manifest = Path(manifest_path)
    with manifest.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    modes_payload = payload.get("modes")
    if not isinstance(modes_payload, list) or not modes_payload:
        raise ValueError(f"{manifest_path} does not contain non-empty modes list")

    modes = []
    for mode_payload in modes_payload:
        latent_rel = mode_payload.get("latent_path")
        if latent_rel is None:
            raise ValueError(f"Mode entry in {manifest_path} is missing latent_path")
        latent_path = Path(latent_rel)
        if not latent_path.is_absolute():
            latent_path = manifest.parent / latent_path

        latent = np.load(str(latent_path), allow_pickle=False)
        required = {"points_world", "phi"}
        missing = sorted(required - set(latent.files))
        if missing:
            raise ValueError(f"{latent_path} missing required fields: {missing}")

        points_world = latent["points_world"].astype(np.float32)
        phi = latent["phi"]
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError(f"{latent_path} points_world must have shape (N, 3)")
        if phi.shape != points_world.shape:
            raise ValueError(
                f"{latent_path} phi must have shape {points_world.shape}, got {phi.shape}"
            )
        if not np.iscomplexobj(phi):
            raise ValueError(f"{latent_path} phi must be complex-valued")

        freq_hz = mode_payload.get("freq_hz")
        if freq_hz is None:
            freq_hz = (
                float(np.asarray(latent["freq_hz"]).reshape(()))
                if "freq_hz" in latent.files
                else 0.0
            )

        modes.append(
            ModalModeData(
                mode_index=int(mode_payload.get("mode_index", len(modes))),
                freq_hz=float(freq_hz),
                latent_path=latent_path,
                points_world=points_world,
                phi=phi.astype(np.complex64),
            )
        )

    return modes


def interpolate_modal_modes_to_gaussians(
    gaussian_means: torch.Tensor,
    modes: list[ModalModeData],
    knn: int,
    power: float,
    eps: float,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    if not modes:
        raise ValueError("Cannot interpolate empty modal mode list")
    if knn <= 0:
        raise ValueError(f"modal knn must be positive, got {knn}")
    if power <= 0:
        raise ValueError(f"modal interpolation power must be positive, got {power}")
    if eps <= 0:
        raise ValueError(f"modal interpolation eps must be positive, got {eps}")

    gaussian_means = gaussian_means.detach()
    device = gaussian_means.device
    dtype = gaussian_means.dtype
    num_modes = len(modes)
    num_gaussians = gaussian_means.shape[0]
    phi_real = gaussian_means.new_empty((num_modes, num_gaussians, 3))
    phi_imag = gaussian_means.new_empty((num_modes, num_gaussians, 3))
    nearest_dists = []

    for mode_idx, mode in enumerate(modes):
        points = torch.as_tensor(mode.points_world, device=device, dtype=dtype)
        phi_r = torch.as_tensor(mode.phi.real, device=device, dtype=dtype)
        phi_i = torch.as_tensor(mode.phi.imag, device=device, dtype=dtype)
        k = min(knn, points.shape[0])

        for start in range(0, num_gaussians, chunk_size):
            end = min(start + chunk_size, num_gaussians)
            means_chunk = gaussian_means[start:end]
            dists = torch.cdist(means_chunk, points)
            knn_dists, knn_indices = torch.topk(dists, k=k, largest=False)
            weights = (knn_dists + eps).pow(-power)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)

            phi_real[mode_idx, start:end] = (
                phi_r[knn_indices] * weights[..., None]
            ).sum(dim=1)
            phi_imag[mode_idx, start:end] = (
                phi_i[knn_indices] * weights[..., None]
            ).sum(dim=1)
            nearest_dists.append(knn_dists[:, 0].detach().cpu())

    all_nearest = torch.cat(nearest_dists)
    stats = {
        "nearest_p50": float(torch.quantile(all_nearest, 0.50).item()),
        "nearest_p90": float(torch.quantile(all_nearest, 0.90).item()),
        "nearest_p95": float(torch.quantile(all_nearest, 0.95).item()),
        "nearest_max": float(all_nearest.max().item()),
    }
    guru.info(
        "Modal carrier-to-Gaussian nearest distance stats: "
        f"p50={stats['nearest_p50']:.6g}, "
        f"p90={stats['nearest_p90']:.6g}, "
        f"p95={stats['nearest_p95']:.6g}, "
        f"max={stats['nearest_max']:.6g}"
    )
    freqs_hz = torch.tensor([mode.freq_hz for mode in modes], device=device, dtype=dtype)
    return phi_real, phi_imag, freqs_hz, stats


def _modal_mask_from_npz(modal: Any, shape: tuple[int, int]) -> np.ndarray:
    if "mask" not in modal.files:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(modal["mask"])
    if mask.ndim != 2:
        raise ValueError(f"modal mask must be 2D, got {mask.shape}")
    if mask.shape != shape:
        raise ValueError(f"modal mask shape {mask.shape} does not match {shape}")
    if np.issubdtype(mask.dtype, np.floating):
        return mask > 0.5
    return mask > 0


def _build_pixel_buckets(
    point_indices: np.ndarray,
    rounded_x: np.ndarray,
    rounded_y: np.ndarray,
    width: int,
) -> dict[int, np.ndarray]:
    if point_indices.size == 0:
        return {}
    linear = (
        rounded_y[point_indices].astype(np.int64) * int(width)
        + rounded_x[point_indices].astype(np.int64)
    )
    order = np.argsort(linear)
    linear_sorted = linear[order]
    point_sorted = point_indices[order]
    keys, starts, counts = np.unique(
        linear_sorted, return_index=True, return_counts=True
    )
    return {
        int(k): point_sorted[int(s) : int(s + c)]
        for k, s, c in zip(keys, starts, counts)
    }


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


def _zbuffer_visible_indices(
    candidate_indices: np.ndarray,
    rounded_x: np.ndarray,
    rounded_y: np.ndarray,
    z: np.ndarray,
    width: int,
    height: int,
    zbuffer_radius: int,
    front_percentile: float,
    zbuffer_tau: float,
    min_zbuffer_samples: int,
) -> np.ndarray:
    buckets = _build_pixel_buckets(candidate_indices, rounded_x, rounded_y, width)
    keep = []
    for point_idx in candidate_indices.tolist():
        stats = _local_depth_stats(
            buckets,
            z,
            int(rounded_x[point_idx]),
            int(rounded_y[point_idx]),
            width,
            height,
            zbuffer_radius,
            front_percentile,
            min_zbuffer_samples,
        )
        if stats is None:
            continue
        z_front, z_med = stats
        if z_med <= 0:
            continue
        rel = abs(float(z[point_idx]) - z_front) / max(abs(z_med), 1e-6)
        if rel < zbuffer_tau:
            keep.append(point_idx)
    return np.asarray(keep, dtype=np.int64)


def load_modal_consistency_data(
    gaussian_means: torch.Tensor,
    modes: list[ModalModeData],
    view_config_paths: tuple[str, ...],
    modal_npz_paths: tuple[str, ...],
    freq_tolerance_hz: float,
    mask_erode_iters: int,
    zbuffer_radius: int = 5,
    front_percentile: float = 10.0,
    zbuffer_tau: float = 0.05,
    min_zbuffer_samples: int = 5,
) -> ModalConsistencyData | None:
    if not view_config_paths and not modal_npz_paths:
        return None
    if len(view_config_paths) != len(modal_npz_paths):
        raise ValueError(
            "modal consistency view configs and modal npzs must have the same length"
        )
    if not modes:
        raise ValueError("modal consistency requires at least one modal mode")
    if freq_tolerance_hz < 0:
        raise ValueError("modal consistency freq tolerance must be non-negative")
    if mask_erode_iters < 0:
        raise ValueError("modal consistency mask_erode_iters must be non-negative")
    if zbuffer_radius < 0:
        raise ValueError("modal consistency zbuffer_radius must be non-negative")
    if not (0.0 <= front_percentile <= 100.0):
        raise ValueError("modal consistency front_percentile must be in [0, 100]")
    if zbuffer_tau <= 0:
        raise ValueError("modal consistency zbuffer_tau must be positive")
    if min_zbuffer_samples < 1:
        raise ValueError(
            "modal consistency min_zbuffer_samples must be at least 1"
        )

    device = gaussian_means.device
    dtype = gaussian_means.dtype
    points_world = gaussian_means.detach().cpu().numpy().astype(np.float32)
    obs_y: list[np.ndarray] = []
    obs_J: list[np.ndarray] = []
    gaussian_indices: list[np.ndarray] = []
    mode_indices: list[np.ndarray] = []
    group_indices: list[np.ndarray] = []
    group_count = 0

    for view_cfg_path, modal_path in zip(view_config_paths, modal_npz_paths):
        cfg = load_view_config(view_cfg_path)
        modal = np.load(str(modal_path), allow_pickle=False)
        required = ["mode_u", "mode_v", "selected_freqs_hz"]
        missing = [key for key in required if key not in modal.files]
        if missing:
            raise ValueError(f"{modal_path} missing required modal arrays: {missing}")
        mode_u_all = modal["mode_u"]
        mode_v_all = modal["mode_v"]
        selected_freqs = modal["selected_freqs_hz"].astype(np.float32).reshape(-1)
        if mode_u_all.ndim != 3 or mode_v_all.shape != mode_u_all.shape:
            raise ValueError(f"{modal_path} mode_u/mode_v must have matching shape (K,H,W)")
        height, width = int(mode_u_all.shape[1]), int(mode_u_all.shape[2])
        if (height, width) != (cfg.image_height, cfg.image_width):
            raise ValueError(
                f"{modal_path} modal image shape {(height, width)} does not match "
                f"view config {(cfg.image_height, cfg.image_width)}"
            )
        mask = erode_mask(_modal_mask_from_npz(modal, (height, width)), mask_erode_iters)
        pixels_xy, z = project_points(points_world, cfg.K, cfg.world_to_camera)
        finite_pixels = np.isfinite(pixels_xy).all(axis=1)
        rounded_x = np.full((pixels_xy.shape[0],), -1, dtype=np.int64)
        rounded_y = np.full((pixels_xy.shape[0],), -1, dtype=np.int64)
        rounded_x[finite_pixels] = np.rint(pixels_xy[finite_pixels, 0]).astype(np.int64)
        rounded_y[finite_pixels] = np.rint(pixels_xy[finite_pixels, 1]).astype(np.int64)
        valid = (
            finite_pixels
            & np.isfinite(z)
            & (z > 0)
            & (pixels_xy[:, 0] >= 0)
            & (pixels_xy[:, 0] < width - 1)
            & (pixels_xy[:, 1] >= 0)
            & (pixels_xy[:, 1] < height - 1)
            & (rounded_x >= 0)
            & (rounded_x < width)
            & (rounded_y >= 0)
            & (rounded_y < height)
        )
        valid_indices = np.where(valid)[0]
        if valid_indices.size == 0:
            raise ValueError(f"No modal consistency Gaussian projections survived for {cfg.view_id}")
        valid_indices = valid_indices[mask[rounded_y[valid_indices], rounded_x[valid_indices]]]
        if valid_indices.size == 0:
            raise ValueError(f"No modal consistency Gaussian projections survived mask for {cfg.view_id}")
        mask_valid_count = int(valid_indices.size)
        valid_indices = _zbuffer_visible_indices(
            valid_indices,
            rounded_x,
            rounded_y,
            z,
            width,
            height,
            zbuffer_radius,
            front_percentile,
            zbuffer_tau,
            min_zbuffer_samples,
        )
        if valid_indices.size == 0:
            raise ValueError(
                f"No modal consistency Gaussian projections survived z-buffer for {cfg.view_id}"
            )
        guru.info(
            f"Modal consistency view {cfg.view_id}: z-buffer kept "
            f"{valid_indices.size}/{mask_valid_count} masked projections"
        )
        jacobians = projection_jacobian(points_world[valid_indices], cfg.K, cfg.world_to_camera)
        sample_xy = pixels_xy[valid_indices]

        for internal_mode_idx, mode in enumerate(modes):
            source_mode_idx = int(mode.mode_index)
            if source_mode_idx < 0 or source_mode_idx >= mode_u_all.shape[0]:
                raise ValueError(
                    f"{modal_path} does not contain mode_index={source_mode_idx}"
                )
            view_freq = float(selected_freqs[source_mode_idx])
            if abs(view_freq - float(mode.freq_hz)) > freq_tolerance_hz:
                raise ValueError(
                    f"{modal_path} mode_index={source_mode_idx} frequency {view_freq:.6f} Hz "
                    f"does not match manifest frequency {mode.freq_hz:.6f} Hz"
                )
            y_u = bilinear_sample(mode_u_all[source_mode_idx].astype(np.complex64), sample_xy)
            y_v = bilinear_sample(mode_v_all[source_mode_idx].astype(np.complex64), sample_xy)
            obs_y.append(np.stack([y_u, y_v], axis=1).astype(np.complex64))
            obs_J.append(jacobians.astype(np.float32))
            gaussian_indices.append(valid_indices.astype(np.int64))
            mode_indices.append(
                np.full((valid_indices.size,), internal_mode_idx, dtype=np.int64)
            )
            group_indices.append(
                np.full((valid_indices.size,), group_count, dtype=np.int64)
            )
            group_count += 1

    if not obs_y:
        raise ValueError("No modal consistency observations were created")
    obs_y_arr = np.concatenate(obs_y, axis=0)
    obs_J_arr = np.concatenate(obs_J, axis=0)
    gaussian_idx_arr = np.concatenate(gaussian_indices, axis=0)
    mode_idx_arr = np.concatenate(mode_indices, axis=0)
    group_idx_arr = np.concatenate(group_indices, axis=0)
    guru.info(
        f"Loaded {obs_y_arr.shape[0]} activation modal consistency observations "
        f"across {group_count} view-frequency groups"
    )
    return ModalConsistencyData(
        y_real=torch.as_tensor(obs_y_arr.real, device=device, dtype=dtype),
        y_imag=torch.as_tensor(obs_y_arr.imag, device=device, dtype=dtype),
        J=torch.as_tensor(obs_J_arr, device=device, dtype=dtype),
        gaussian_indices=torch.as_tensor(gaussian_idx_arr, device=device, dtype=torch.long),
        mode_indices=torch.as_tensor(mode_idx_arr, device=device, dtype=torch.long),
        group_indices=torch.as_tensor(group_idx_arr, device=device, dtype=torch.long),
        group_count=group_count,
    )


def load_modal_frame_map(
    frame_map_path: str,
    train_dataset: Any,
    num_frames: int,
    device: torch.device,
) -> ModalFrameMap:
    with open(frame_map_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    view_ids = payload.get("views")
    frames = payload.get("frames")
    if not isinstance(view_ids, list) or not view_ids:
        raise ValueError(f"{frame_map_path} must contain non-empty views list")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{frame_map_path} must contain non-empty frames list")

    frame_names = getattr(train_dataset, "frame_names", None)
    if frame_names is None:
        raise ValueError("modal_activation requires train_dataset.frame_names")
    if len(frame_names) != train_dataset.num_frames:
        raise ValueError("train_dataset.frame_names length does not match num_frames")

    if hasattr(train_dataset, "time_ids"):
        time_ids = torch.as_tensor(train_dataset.time_ids).to(torch.long).cpu()
    else:
        time_ids = torch.arange(train_dataset.num_frames, dtype=torch.long)
    if len(time_ids) != len(frame_names):
        raise ValueError("train_dataset.time_ids length does not match frame_names")
    if int(torch.unique(time_ids).numel()) != int(time_ids.numel()):
        raise ValueError("modal_activation requires unique global ts per frame")

    by_name = {}
    for record in frames:
        frame_name = record.get("frame_name")
        if frame_name is None:
            raise ValueError(f"Frame record in {frame_map_path} is missing frame_name")
        if frame_name in by_name:
            raise ValueError(f"Duplicate frame_name in modal frame map: {frame_name}")
        by_name[frame_name] = record

    view_to_index = {view_id: idx for idx, view_id in enumerate(view_ids)}
    frame_view_indices = torch.full((num_frames,), -1, dtype=torch.long)
    frame_local_indices = torch.full((num_frames,), -1, dtype=torch.long)
    per_view_local_to_ts: list[dict[int, int]] = [dict() for _ in view_ids]
    missing = []

    for dataset_index, frame_name in enumerate(frame_names):
        if frame_name not in by_name:
            missing.append(frame_name)
            continue
        record = by_name[frame_name]
        view_id = record.get("view_id")
        if view_id not in view_to_index:
            raise ValueError(f"Unknown view_id {view_id!r} for frame {frame_name}")
        if "local_index" not in record:
            raise ValueError(f"Frame record {frame_name} is missing local_index")
        local_index = int(record["local_index"])
        if local_index < 0:
            raise ValueError(f"local_index must be non-negative for {frame_name}")

        ts = int(time_ids[dataset_index].item())
        if not 0 <= ts < num_frames:
            raise ValueError(f"Frame {frame_name} has ts={ts}, outside [0, {num_frames})")
        view_index = view_to_index[view_id]
        if local_index in per_view_local_to_ts[view_index]:
            raise ValueError(
                f"Duplicate local_index={local_index} for view_id={view_id!r}"
            )
        per_view_local_to_ts[view_index][local_index] = ts
        frame_view_indices[ts] = view_index
        frame_local_indices[ts] = local_index

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{frame_map_path} is missing {len(missing)} training frames, "
            f"first missing: {preview}"
        )
    smooth_triplets = []
    for local_to_ts in per_view_local_to_ts:
        for local_index, center_ts in sorted(local_to_ts.items()):
            prev_ts = local_to_ts.get(local_index - 1)
            next_ts = local_to_ts.get(local_index + 1)
            if prev_ts is not None and next_ts is not None:
                smooth_triplets.append([prev_ts, center_ts, next_ts])

    if smooth_triplets:
        triplets_tensor = torch.tensor(smooth_triplets, dtype=torch.long, device=device)
    else:
        triplets_tensor = torch.empty((0, 3), dtype=torch.long, device=device)
        guru.warning("Modal frame map produced no activation smoothness triplets")

    return ModalFrameMap(
        view_ids=list(view_ids),
        frame_view_indices=frame_view_indices.to(device),
        frame_local_indices=frame_local_indices.to(device),
        smooth_triplets=triplets_tensor,
    )


def resolve_required_modal_paths(modal_manifest: str | None, modal_frame_map: str | None):
    if modal_manifest is None:
        raise ValueError("trajectory_type='modal_activation' requires modal_manifest")
    if modal_frame_map is None:
        raise ValueError("trajectory_type='modal_activation' requires modal_frame_map")
    if not os.path.exists(modal_manifest):
        raise ValueError(f"modal_manifest does not exist: {modal_manifest}")
    if not os.path.exists(modal_frame_map):
        raise ValueError(f"modal_frame_map does not exist: {modal_frame_map}")
