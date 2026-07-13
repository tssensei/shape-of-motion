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
class GaussianModalFieldData:
    modes: list[ModalModeData]
    phi_real: torch.Tensor
    phi_imag: torch.Tensor
    freqs_hz: torch.Tensor
    obs_count_per_point: torch.Tensor


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


def load_gaussian_modal_fields(
    manifest_path: str,
    gaussian_means: torch.Tensor,
) -> GaussianModalFieldData:
    manifest = Path(manifest_path)
    with manifest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if int(payload.get("version", -1)) != 1:
        raise ValueError(f"{manifest_path} must be a version 1 modal manifest")
    if payload.get("point_type") != "foreground_gaussian_center":
        raise ValueError(
            f"{manifest_path} point_type={payload.get('point_type')!r}; "
            "expected foreground_gaussian_center"
        )
    source_checkpoint = payload.get("source_checkpoint")
    if not isinstance(source_checkpoint, str) or not source_checkpoint:
        raise ValueError(f"{manifest_path} is missing source_checkpoint")
    modes_payload = payload.get("modes")
    if not isinstance(modes_payload, list) or not modes_payload:
        raise ValueError(f"{manifest_path} does not contain non-empty modes list")

    gaussian_means = gaussian_means.detach()
    gaussian_points = gaussian_means.cpu().numpy().astype(np.float32)
    if gaussian_points.ndim != 2 or gaussian_points.shape[1] != 3:
        raise ValueError(
            "foreground Gaussian means must have shape (N, 3), "
            f"got {gaussian_points.shape}"
        )
    if not np.isfinite(gaussian_points).all():
        raise ValueError("foreground Gaussian means must be finite")
    num_gaussians = gaussian_points.shape[0]
    expected_indices = np.arange(num_gaussians, dtype=np.int64)
    modes: list[ModalModeData] = []
    phi_values: list[np.ndarray] = []
    freqs_hz: list[float] = []
    obs_counts: list[np.ndarray] = []

    for mode_payload in modes_payload:
        latent_rel = mode_payload.get("latent_path")
        if not isinstance(latent_rel, str) or not latent_rel:
            raise ValueError(f"Mode entry in {manifest_path} is missing latent_path")
        latent_path = Path(latent_rel)
        if not latent_path.is_absolute():
            latent_path = manifest.parent / latent_path

        with np.load(str(latent_path), allow_pickle=False) as latent:
            required = {
                "points_world",
                "phi",
                "gaussian_indices",
                "freq_hz",
                "mode_index",
                "obs_count_per_point",
                "point_type",
                "source_checkpoint",
            }
            missing = sorted(required - set(latent.files))
            if missing:
                raise ValueError(f"{latent_path} missing required fields: {missing}")

            point_type = np.asarray(latent["point_type"])
            if point_type.shape != ():
                raise ValueError(f"{latent_path} point_type must be a scalar string")
            if str(point_type.item()) != "foreground_gaussian_center":
                raise ValueError(
                    f"{latent_path} point_type={str(point_type.item())!r}; "
                    "expected foreground_gaussian_center"
                )

            latent_source_checkpoint = np.asarray(latent["source_checkpoint"])
            if latent_source_checkpoint.shape != ():
                raise ValueError(f"{latent_path} source_checkpoint must be a scalar string")
            if str(latent_source_checkpoint.item()) != source_checkpoint:
                raise ValueError(
                    f"{latent_path} source_checkpoint does not match {manifest_path}"
                )

            points_world = latent["points_world"].astype(np.float32)
            phi = latent["phi"]
            gaussian_indices = latent["gaussian_indices"]
            obs_count = latent["obs_count_per_point"]
            freq = np.asarray(latent["freq_hz"])
            mode_index = np.asarray(latent["mode_index"])

        expected_point_shape = (num_gaussians, 3)
        if points_world.shape != expected_point_shape:
            raise ValueError(
                f"{latent_path} points_world shape {points_world.shape} does not match "
                f"foreground {expected_point_shape}"
            )
        if phi.shape != expected_point_shape:
            raise ValueError(
                f"{latent_path} phi shape {phi.shape} does not match foreground "
                f"{expected_point_shape}"
            )
        if not np.iscomplexobj(phi):
            raise ValueError(f"{latent_path} phi must be complex-valued")
        if not np.isfinite(phi.real).all() or not np.isfinite(phi.imag).all():
            raise ValueError(f"{latent_path} phi must be finite")
        if gaussian_indices.shape != (num_gaussians,):
            raise ValueError(
                f"{latent_path} gaussian_indices shape {gaussian_indices.shape} does not "
                f"match foreground {(num_gaussians,)}"
            )
        if not np.issubdtype(gaussian_indices.dtype, np.integer):
            raise ValueError(f"{latent_path} gaussian_indices must be integer-valued")
        if not np.array_equal(gaussian_indices, expected_indices):
            raise ValueError(
                f"{latent_path} gaussian_indices are not contiguous foreground indices"
            )
        if obs_count.shape != (num_gaussians,):
            raise ValueError(
                f"{latent_path} obs_count_per_point shape {obs_count.shape} does not "
                f"match foreground {(num_gaussians,)}"
            )
        if not np.issubdtype(obs_count.dtype, np.integer):
            raise ValueError(f"{latent_path} obs_count_per_point must be integer-valued")
        if np.any(obs_count < 0):
            raise ValueError(f"{latent_path} obs_count_per_point must be non-negative")
        if (
            freq.shape != ()
            or not np.issubdtype(freq.dtype, np.number)
            or np.iscomplexobj(freq)
            or not np.isfinite(freq.item())
        ):
            raise ValueError(f"{latent_path} freq_hz must be a finite real scalar")
        manifest_mode_index = mode_payload.get("mode_index")
        if not isinstance(manifest_mode_index, int) or isinstance(
            manifest_mode_index, bool
        ):
            raise ValueError(f"Mode entry in {manifest_path} has invalid mode_index")
        if mode_index.shape != () or not np.issubdtype(mode_index.dtype, np.integer):
            raise ValueError(f"{latent_path} mode_index must be an integer scalar")
        if int(mode_index.item()) != manifest_mode_index:
            raise ValueError(f"{latent_path} mode_index does not match {manifest_path}")
        manifest_freq_hz = mode_payload.get("freq_hz")
        if (
            not isinstance(manifest_freq_hz, (int, float))
            or isinstance(manifest_freq_hz, bool)
            or not np.isfinite(manifest_freq_hz)
        ):
            raise ValueError(f"Mode entry in {manifest_path} has invalid freq_hz")
        if not np.isclose(
            float(freq.item()), float(manifest_freq_hz), rtol=1e-6, atol=1e-6
        ):
            raise ValueError(f"{latent_path} freq_hz does not match {manifest_path}")
        if not np.isfinite(points_world).all():
            raise ValueError(f"{latent_path} points_world must be finite")
        max_position_delta = (
            float(np.max(np.abs(points_world - gaussian_points)))
            if num_gaussians > 0
            else 0.0
        )
        if max_position_delta > 1e-5:
            raise ValueError(
                f"{latent_path} points_world differs from foreground Gaussian means by "
                f"max {max_position_delta:.6g}, above 1e-05"
            )

        freq_hz = float(freq.item())
        mode = ModalModeData(
            mode_index=manifest_mode_index,
            freq_hz=freq_hz,
            latent_path=latent_path,
            points_world=points_world,
            phi=phi.astype(np.complex64),
        )
        modes.append(mode)
        phi_values.append(mode.phi)
        freqs_hz.append(freq_hz)
        obs_counts.append(obs_count.astype(np.int64))

    phi_all = np.stack(phi_values, axis=0)
    device = gaussian_means.device
    dtype = gaussian_means.dtype
    return GaussianModalFieldData(
        modes=modes,
        phi_real=torch.as_tensor(phi_all.real, device=device, dtype=dtype),
        phi_imag=torch.as_tensor(phi_all.imag, device=device, dtype=dtype),
        freqs_hz=torch.as_tensor(freqs_hz, device=device, dtype=dtype),
        obs_count_per_point=torch.as_tensor(
            np.stack(obs_counts, axis=0), device=device, dtype=torch.long
        ),
    )


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
