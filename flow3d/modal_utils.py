import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger as guru


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
