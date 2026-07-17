import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modal_surface.gaussian_motion_fill import (
    MOTION_FILL_ROLE_CONSTRAINED_VARIABLE,
    MOTION_FILL_ROLE_EXCLUDED,
    MOTION_FILL_ROLE_FIXED_ANCHOR,
    MOTION_FILL_ROLE_FREE_VARIABLE,
    MOTION_FILL_ROLE_NAMES,
)


MOTION_FILL_DISPLAY_ANCHOR = 0
MOTION_FILL_DISPLAY_PARTIAL = 1
MOTION_FILL_DISPLAY_FILLED = 2
MOTION_FILL_DISPLAY_UNOBSERVED = 3
MOTION_FILL_DISPLAY_EXCLUDED = 4
MOTION_FILL_DISPLAY_NAMES = (
    "anchor",
    "partial",
    "filled",
    "unobserved",
    "excluded",
)
MOTION_FILL_DISPLAY_COLORS = np.asarray(
    [
        [0.05, 0.55, 1.0],
        [1.0, 0.55, 0.1],
        [0.1, 0.85, 0.3],
        [0.65, 0.35, 1.0],
        [0.45, 0.45, 0.45],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class ModalModeData:
    mode_index: int
    freq_hz: float
    latent_path: Path
    points_world: np.ndarray
    phi: np.ndarray
    motion_fill_display_class: np.ndarray | None = None


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
    frame_times_sec: torch.Tensor


def classify_motion_fill_display_points(
    motion_fill_role: np.ndarray,
    completion_mask: np.ndarray,
) -> np.ndarray:
    """Derive exclusive anchor/partial/filled/unobserved/excluded display classes."""

    role = np.asarray(motion_fill_role)
    completion = np.asarray(completion_mask)
    if role.ndim != 1:
        raise ValueError(f"motion_fill_role must be a 1-D array, got {role.shape}")
    if not np.issubdtype(role.dtype, np.integer):
        raise ValueError(f"motion_fill_role must have integer dtype, got {role.dtype}")
    if completion.shape != role.shape:
        raise ValueError(
            f"completion_mask must have shape {role.shape}, got {completion.shape}"
        )
    if completion.dtype != np.bool_:
        raise ValueError(
            f"completion_mask must have boolean dtype, got {completion.dtype}"
        )
    if np.any(role < 0) or np.any(role >= len(MOTION_FILL_ROLE_NAMES)):
        raise ValueError(
            f"motion_fill_role values must lie in [0,{len(MOTION_FILL_ROLE_NAMES) - 1}]"
        )
    completable = (role == MOTION_FILL_ROLE_CONSTRAINED_VARIABLE) | (
        role == MOTION_FILL_ROLE_FREE_VARIABLE
    )
    if np.any(completion & ~completable):
        point = int(np.where(completion & ~completable)[0][0])
        raise ValueError(
            "completion_mask is true for non-completable motion_fill_role "
            f"at point {point}"
        )

    display_class = np.full(role.shape, -1, dtype=np.int8)
    display_class[role == MOTION_FILL_ROLE_FIXED_ANCHOR] = MOTION_FILL_DISPLAY_ANCHOR
    display_class[
        (role == MOTION_FILL_ROLE_CONSTRAINED_VARIABLE) & ~completion
    ] = MOTION_FILL_DISPLAY_PARTIAL
    display_class[completion] = MOTION_FILL_DISPLAY_FILLED
    display_class[
        (role == MOTION_FILL_ROLE_FREE_VARIABLE) & ~completion
    ] = MOTION_FILL_DISPLAY_UNOBSERVED
    display_class[role == MOTION_FILL_ROLE_EXCLUDED] = MOTION_FILL_DISPLAY_EXCLUDED
    if np.any(display_class < 0):
        point = int(np.where(display_class < 0)[0][0])
        raise RuntimeError(
            f"Motion-fill display class was not assigned at point {point}"
        )
    return display_class


def _validated_motion_fill_display_classes(display_class: np.ndarray) -> np.ndarray:
    classes = np.asarray(display_class)
    if not np.issubdtype(classes.dtype, np.integer):
        raise ValueError(
            f"motion-fill display classes must have integer dtype, got {classes.dtype}"
        )
    if np.any(classes < 0) or np.any(classes >= len(MOTION_FILL_DISPLAY_NAMES)):
        raise ValueError(
            "motion-fill display classes must lie in "
            f"[0,{len(MOTION_FILL_DISPLAY_NAMES) - 1}]"
        )
    return classes


def motion_fill_display_colors(display_class: np.ndarray) -> np.ndarray:
    """Map motion-fill display classes to deterministic float RGB colors."""

    classes = _validated_motion_fill_display_classes(display_class)
    return MOTION_FILL_DISPLAY_COLORS[classes]


def select_motion_fill_display_indices(
    display_class: np.ndarray,
    enabled_classes: np.ndarray,
    max_count: int,
) -> np.ndarray:
    """Select up to one shared count from the enabled display-class union."""

    classes = _validated_motion_fill_display_classes(display_class)
    enabled = np.asarray(enabled_classes)
    if classes.ndim != 1:
        raise ValueError(
            f"motion-fill display classes must be a 1-D array, got {classes.shape}"
        )
    if enabled.shape != (len(MOTION_FILL_DISPLAY_NAMES),):
        raise ValueError(
            "enabled_classes must have shape "
            f"({len(MOTION_FILL_DISPLAY_NAMES)},), got {enabled.shape}"
        )
    if enabled.dtype != np.bool_:
        raise ValueError(
            f"enabled_classes must have boolean dtype, got {enabled.dtype}"
        )
    if isinstance(max_count, (bool, np.bool_)) or not isinstance(
        max_count, (int, np.integer)
    ):
        raise ValueError(
            f"max_count must be an integer, got {type(max_count).__name__}"
        )
    if max_count < 0:
        raise ValueError(f"max_count must be non-negative, got {max_count}")
    return np.flatnonzero(enabled[classes])[: int(max_count)]


def stack_modal_motion_fill_display_classes(
    modes: list[ModalModeData],
) -> np.ndarray | None:
    """Stack per-mode display classes, preserving manifests without role metadata."""

    has_metadata = [mode.motion_fill_display_class is not None for mode in modes]
    if not any(has_metadata):
        return None
    if not all(has_metadata):
        raise ValueError(
            "Manifest modes must either all include motion_fill_role metadata "
            "or none do"
        )
    classes = np.stack(
        [np.asarray(mode.motion_fill_display_class) for mode in modes], axis=0
    )
    return _validated_motion_fill_display_classes(classes).astype(
        np.int8, copy=False
    )


def _load_motion_fill_display_class(
    latent: Any,
    latent_path: Path,
    num_points: int,
) -> np.ndarray | None:
    if "motion_fill_role" not in latent.files:
        return None
    required = {"completion_mask", "motion_fill_role_names"}
    missing = sorted(required - set(latent.files))
    if missing:
        raise ValueError(
            f"{latent_path} has motion_fill_role but is missing required fields: {missing}"
        )
    raw_names = np.asarray(latent["motion_fill_role_names"])
    if raw_names.ndim != 1:
        raise ValueError(
            f"{latent_path} motion_fill_role_names must be a 1-D string array"
        )
    names = []
    for item in raw_names:
        value = np.asarray(item).item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        names.append(str(value))
    if tuple(names) != tuple(MOTION_FILL_ROLE_NAMES):
        raise ValueError(
            f"{latent_path} motion_fill_role_names={tuple(names)!r}; "
            f"expected {tuple(MOTION_FILL_ROLE_NAMES)!r}"
        )
    role = np.asarray(latent["motion_fill_role"])
    completion = np.asarray(latent["completion_mask"])
    if role.shape != (num_points,):
        raise ValueError(
            f"{latent_path} motion_fill_role must have shape ({num_points},), "
            f"got {role.shape}"
        )
    return classify_motion_fill_display_points(role, completion)


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
        motion_fill_display_class = _load_motion_fill_display_class(
            latent,
            latent_path,
            points_world.shape[0],
        )

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
                motion_fill_display_class=motion_fill_display_class,
            )
        )

    has_motion_fill_metadata = [
        mode.motion_fill_display_class is not None for mode in modes
    ]
    if any(has_motion_fill_metadata) and not all(has_motion_fill_metadata):
        raise ValueError(
            "Manifest modes must either all include motion_fill_role metadata "
            "or none do"
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


def load_modal_frame_map(
    frame_map_path: str,
    train_dataset: Any,
    num_frames: int,
    device: torch.device,
) -> ModalFrameMap:
    with open(frame_map_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    version = payload.get("version")
    if isinstance(version, bool) or version != 1:
        raise ValueError(f"{frame_map_path} must use modal frame map version 1")
    view_ids = payload.get("views")
    view_fps_hz = payload.get("view_fps_hz")
    frames = payload.get("frames")
    if not isinstance(view_ids, list) or not view_ids:
        raise ValueError(f"{frame_map_path} must contain non-empty views list")
    if any(not isinstance(view_id, str) or not view_id for view_id in view_ids):
        raise ValueError(f"{frame_map_path} views must be non-empty strings")
    if len(set(view_ids)) != len(view_ids):
        raise ValueError(f"{frame_map_path} views must be unique")
    if not isinstance(view_fps_hz, dict):
        raise ValueError(f"{frame_map_path} must contain view_fps_hz object")
    if set(view_fps_hz) != set(view_ids):
        raise ValueError(
            f"{frame_map_path} view_fps_hz keys must exactly match views"
        )
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{frame_map_path} must contain non-empty frames list")

    fps_by_view: dict[str, float] = {}
    for view_id in view_ids:
        value = view_fps_hz[view_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"FPS for view {view_id!r} must be numeric")
        fps = float(value)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"FPS for view {view_id!r} must be finite and positive")
        fps_by_view[view_id] = fps

    frame_names = getattr(train_dataset, "frame_names", None)
    if frame_names is None:
        raise ValueError("modal_activation requires train_dataset.frame_names")
    if len(frame_names) != train_dataset.num_frames:
        raise ValueError("train_dataset.frame_names length does not match num_frames")
    if train_dataset.num_frames != num_frames:
        raise ValueError(
            "modal_activation model frame count does not match train dataset"
        )
    if num_frames <= 0:
        raise ValueError("modal_activation requires at least one training frame")
    if len(set(frame_names)) != len(frame_names):
        raise ValueError("modal_activation requires unique train_dataset.frame_names")

    if hasattr(train_dataset, "time_ids"):
        time_ids = torch.as_tensor(train_dataset.time_ids).cpu()
    else:
        time_ids = torch.arange(train_dataset.num_frames, dtype=torch.long)
    if time_ids.ndim != 1 or len(time_ids) != len(frame_names):
        raise ValueError("train_dataset.time_ids length does not match frame_names")
    if time_ids.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("train_dataset.time_ids must have integer dtype")
    time_ids = time_ids.to(torch.long)
    if int(torch.unique(time_ids).numel()) != int(time_ids.numel()):
        raise ValueError("modal_activation requires unique global ts per frame")
    expected_time_ids = torch.arange(num_frames, dtype=torch.long)
    if not torch.equal(torch.sort(time_ids).values, expected_time_ids):
        raise ValueError(
            f"modal_activation global ts must be a permutation of [0, {num_frames})"
        )

    view_to_index = {view_id: idx for idx, view_id in enumerate(view_ids)}
    by_name: dict[str, tuple[int, int, float]] = {}
    per_view_full_local_indices: list[set[int]] = [set() for _ in view_ids]
    for record in frames:
        if not isinstance(record, dict):
            raise ValueError(f"Frame records in {frame_map_path} must be objects")
        frame_name = record.get("frame_name")
        if not isinstance(frame_name, str) or not frame_name:
            raise ValueError(
                f"Frame record in {frame_map_path} has invalid frame_name"
            )
        if frame_name in by_name:
            raise ValueError(f"Duplicate frame_name in modal frame map: {frame_name}")
        view_id = record.get("view_id")
        if not isinstance(view_id, str):
            raise ValueError(f"view_id must be a string for frame {frame_name}")
        if view_id not in view_to_index:
            raise ValueError(f"Unknown view_id {view_id!r} for frame {frame_name}")
        local_index_value = record.get("local_index")
        if isinstance(local_index_value, bool) or not isinstance(
            local_index_value, int
        ):
            raise ValueError(f"local_index must be an integer for {frame_name}")
        local_index = int(local_index_value)
        if local_index < 0:
            raise ValueError(f"local_index must be non-negative for {frame_name}")
        time_sec_value = record.get("time_sec")
        if isinstance(time_sec_value, bool) or not isinstance(
            time_sec_value, (int, float)
        ):
            raise ValueError(f"time_sec must be numeric for {frame_name}")
        time_sec = float(time_sec_value)
        if not np.isfinite(time_sec) or time_sec < 0:
            raise ValueError(f"time_sec must be finite and non-negative for {frame_name}")
        expected_time_sec = local_index / fps_by_view[view_id]
        if abs(time_sec - expected_time_sec) > 1e-9:
            raise ValueError(
                f"time_sec={time_sec:.12g} for {frame_name} does not match "
                f"local_index/fps={expected_time_sec:.12g}"
            )

        view_index = view_to_index[view_id]
        if local_index in per_view_full_local_indices[view_index]:
            raise ValueError(
                f"Duplicate local_index={local_index} for view_id={view_id!r}"
            )
        per_view_full_local_indices[view_index].add(local_index)
        by_name[frame_name] = (view_index, local_index, time_sec)

    for view_index, view_id in enumerate(view_ids):
        local_indices = per_view_full_local_indices[view_index]
        if not local_indices:
            raise ValueError(f"Modal frame map view {view_id!r} has no frames")
        expected = set(range(len(local_indices)))
        if local_indices != expected:
            raise ValueError(
                f"Modal frame map local_index values for view {view_id!r} "
                f"must be contiguous from 0 to {len(local_indices) - 1}"
            )

    frame_view_indices = torch.full((num_frames,), -1, dtype=torch.long)
    frame_local_indices = torch.full((num_frames,), -1, dtype=torch.long)
    frame_times_sec = torch.full((num_frames,), float("nan"), dtype=torch.float32)
    missing: list[str] = []

    for dataset_index, frame_name in enumerate(frame_names):
        if frame_name not in by_name:
            missing.append(frame_name)
            continue
        view_index, local_index, time_sec = by_name[frame_name]
        ts = int(time_ids[dataset_index].item())
        frame_view_indices[ts] = view_index
        frame_local_indices[ts] = local_index
        frame_times_sec[ts] = time_sec

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{frame_map_path} is missing {len(missing)} training frames, "
            f"first missing: {preview}"
        )
    if bool((frame_view_indices < 0).any().item()):
        raise ValueError("Modal frame map did not assign every model frame to a view")
    if bool((frame_local_indices < 0).any().item()):
        raise ValueError("Modal frame map did not assign every model frame a local index")
    if not bool(torch.isfinite(frame_times_sec).all().item()):
        raise ValueError("Modal frame map did not assign every model frame a time_sec")

    return ModalFrameMap(
        view_ids=list(view_ids),
        frame_view_indices=frame_view_indices.to(device),
        frame_local_indices=frame_local_indices.to(device),
        frame_times_sec=frame_times_sec.to(device),
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
