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
class GaussianModalRefinementData:
    source_mode_indices: tuple[int, ...]
    refinement_mask: torch.Tensor
    display_class: torch.Tensor
    role_counts: torch.Tensor
    staged_refinement_energy: torch.Tensor
    refinement_edge_index: torch.Tensor
    refinement_edge_mode_indices: torch.Tensor
    refinement_edge_weights: torch.Tensor


@dataclass(frozen=True)
class ModalFrameMap:
    view_ids: list[str]
    frame_view_indices: torch.Tensor
    frame_local_indices: torch.Tensor
    frame_times_sec: torch.Tensor


@dataclass(frozen=True)
class ModalEnvelopeLayout:
    knot_offsets: torch.Tensor
    knot_times_sec: torch.Tensor
    frame_left_indices: torch.Tensor
    frame_right_indices: torch.Tensor
    frame_lerp_weights: torch.Tensor


def build_modal_envelope_layout(
    frame_map: ModalFrameMap,
    knot_interval_sec: float,
) -> ModalEnvelopeLayout:
    interval = float(knot_interval_sec)
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("modal envelope knot interval must be finite and positive")
    if not frame_map.view_ids:
        raise ValueError("modal envelope layout requires at least one view")

    frame_view_indices = frame_map.frame_view_indices.detach().cpu().numpy()
    frame_times_sec = frame_map.frame_times_sec.detach().cpu().numpy().astype(
        np.float64,
        copy=False,
    )
    if frame_view_indices.ndim != 1 or frame_times_sec.shape != frame_view_indices.shape:
        raise ValueError("modal envelope frame view/time arrays must be matching 1-D arrays")
    if not np.issubdtype(frame_view_indices.dtype, np.integer):
        raise ValueError("modal envelope frame view indices must have integer dtype")
    if not np.isfinite(frame_times_sec).all() or np.any(frame_times_sec < 0.0):
        raise ValueError("modal envelope frame times must be finite and non-negative")

    num_views = len(frame_map.view_ids)
    if np.any(frame_view_indices < 0) or np.any(frame_view_indices >= num_views):
        raise ValueError("modal envelope frame view indices are out of range")

    knot_offsets = [0]
    knot_times_by_view: list[np.ndarray] = []
    frame_left_indices = np.full(frame_times_sec.shape, -1, dtype=np.int64)
    frame_right_indices = np.full(frame_times_sec.shape, -1, dtype=np.int64)
    frame_lerp_weights = np.full(frame_times_sec.shape, np.nan, dtype=np.float32)

    for view_index, view_id in enumerate(frame_map.view_ids):
        frame_indices = np.flatnonzero(frame_view_indices == view_index)
        view_times = frame_times_sec[frame_indices]
        if view_times.size == 0:
            knot_times = np.asarray([0.0], dtype=np.float64)
        else:
            min_time = float(view_times.min())
            max_time = float(view_times.max())
            if abs(min_time) > 1.0e-9:
                raise ValueError(
                    f"modal envelope view {view_id!r} must include its t=0 frame"
                )
            if max_time == 0.0:
                knot_times = np.asarray([0.0], dtype=np.float64)
            else:
                knot_count = int(np.ceil(max_time / interval)) + 1
                knot_times = np.linspace(
                    0.0,
                    max_time,
                    knot_count,
                    dtype=np.float64,
                )

        view_offset = knot_offsets[-1]
        knot_offsets.append(view_offset + int(knot_times.shape[0]))
        knot_times_by_view.append(knot_times)
        if view_times.size == 0:
            continue

        if knot_times.shape[0] == 1:
            frame_left_indices[frame_indices] = view_offset
            frame_right_indices[frame_indices] = view_offset
            frame_lerp_weights[frame_indices] = 0.0
            continue

        for frame_index, time_sec in zip(frame_indices, view_times, strict=True):
            upper = int(np.searchsorted(knot_times, time_sec, side="right"))
            if upper >= knot_times.shape[0]:
                if time_sec > knot_times[-1] + 1.0e-9:
                    raise ValueError(
                        f"modal envelope frame time {time_sec:.12g} for view "
                        f"{view_id!r} exceeds its knot range"
                    )
                left_local = knot_times.shape[0] - 1
                right_local = left_local
                lerp = 0.0
            else:
                left_local = upper - 1
                right_local = upper
                if left_local < 0:
                    raise ValueError(
                        f"modal envelope frame time {time_sec:.12g} for view "
                        f"{view_id!r} precedes its knot range"
                    )
                left_time = float(knot_times[left_local])
                right_time = float(knot_times[right_local])
                if right_time <= left_time:
                    raise ValueError("modal envelope knot times must be increasing")
                lerp = (float(time_sec) - left_time) / (right_time - left_time)
                if lerp < -1.0e-6 or lerp > 1.0 + 1.0e-6:
                    raise ValueError("modal envelope interpolation weight is out of range")
                lerp = float(np.clip(lerp, 0.0, 1.0))
            frame_left_indices[frame_index] = view_offset + left_local
            frame_right_indices[frame_index] = view_offset + right_local
            frame_lerp_weights[frame_index] = lerp

    if np.any(frame_left_indices < 0) or np.any(frame_right_indices < 0):
        raise ValueError("modal envelope layout did not assign every frame")
    if not np.isfinite(frame_lerp_weights).all():
        raise ValueError("modal envelope layout produced non-finite interpolation weights")

    device = frame_map.frame_times_sec.device
    return ModalEnvelopeLayout(
        knot_offsets=torch.as_tensor(knot_offsets, dtype=torch.long, device=device),
        knot_times_sec=torch.as_tensor(
            np.concatenate(knot_times_by_view),
            dtype=frame_map.frame_times_sec.dtype,
            device=device,
        ),
        frame_left_indices=torch.as_tensor(
            frame_left_indices,
            dtype=torch.long,
            device=device,
        ),
        frame_right_indices=torch.as_tensor(
            frame_right_indices,
            dtype=torch.long,
            device=device,
        ),
        frame_lerp_weights=torch.as_tensor(
            frame_lerp_weights,
            dtype=frame_map.frame_times_sec.dtype,
            device=device,
        ),
    )


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
    version = payload.get("version")
    if isinstance(version, bool) or version != 1:
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


def _resolve_modal_artifact_path(
    manifest_path: Path,
    value: Any,
    field_name: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Mode entry in {manifest_path} is missing {field_name}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.is_file():
        raise ValueError(f"{field_name} does not exist: {path}")
    return path


def _modal_scalar_string(value: np.ndarray, field_name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{path} {field_name} must be a scalar string")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str) or not item:
        raise ValueError(f"{path} {field_name} must be a non-empty scalar string")
    return item


def _modal_string_vector(value: np.ndarray, field_name: str, path: Path) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{path} {field_name} must be a 1-D string array")
    result: list[str] = []
    for raw_item in array:
        item = np.asarray(raw_item).item()
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if not isinstance(item, str) or not item:
            raise ValueError(f"{path} {field_name} must contain non-empty strings")
        result.append(item)
    return tuple(result)


def _load_modal_refinement_graph(
    manifest_path: Path,
    parameters: dict[str, Any],
    gaussian_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if parameters.get("motion_fill_enabled") is not True:
        raise ValueError(
            f"{manifest_path} must describe a motion-fill-enabled Gaussian field"
        )
    graph_path = _resolve_modal_artifact_path(
        manifest_path,
        parameters.get("motion_fill_graph_path"),
        "parameters.motion_fill_graph_path",
    )
    with np.load(str(graph_path), allow_pickle=False) as loaded:
        graph = {key: loaded[key] for key in loaded.files}
    required = {
        "points_world",
        "gaussian_indices",
        "edge_index",
        "edge_distance",
        "edge_weight",
        "k",
        "max_distance",
        "epsilon",
    }
    missing = sorted(required - set(graph))
    if missing:
        raise ValueError(f"{graph_path} missing required graph fields: {missing}")
    num_gaussians = int(gaussian_points.shape[0])
    graph_points = np.asarray(graph["points_world"], dtype=np.float32)
    if graph_points.shape != (num_gaussians, 3):
        raise ValueError(
            f"{graph_path} points_world must have shape ({num_gaussians},3)"
        )
    if not np.isfinite(graph_points).all():
        raise ValueError(f"{graph_path} points_world must be finite")
    max_delta = (
        float(np.max(np.abs(graph_points - gaussian_points)))
        if num_gaussians > 0
        else 0.0
    )
    if max_delta > 1e-5:
        raise ValueError(
            f"{graph_path} points_world differs from foreground Gaussian means by "
            f"max {max_delta:.6g}, above 1e-05"
        )
    gaussian_indices = np.asarray(graph["gaussian_indices"])
    if gaussian_indices.shape != (num_gaussians,) or not np.issubdtype(
        gaussian_indices.dtype, np.integer
    ):
        raise ValueError(
            f"{graph_path} gaussian_indices must be an integer array with shape "
            f"({num_gaussians},)"
        )
    if not np.array_equal(
        gaussian_indices.astype(np.int64), np.arange(num_gaussians, dtype=np.int64)
    ):
        raise ValueError(f"{graph_path} gaussian_indices must preserve foreground order")

    edge_index = np.asarray(graph["edge_index"])
    edge_distance = np.asarray(graph["edge_distance"], dtype=np.float64)
    edge_weight = np.asarray(graph["edge_weight"], dtype=np.float64)
    if edge_index.ndim != 2 or edge_index.shape[1] != 2 or not np.issubdtype(
        edge_index.dtype, np.integer
    ):
        raise ValueError(f"{graph_path} edge_index must have integer shape (E,2)")
    edge_index = edge_index.astype(np.int64)
    num_edges = int(edge_index.shape[0])
    if edge_distance.shape != (num_edges,) or edge_weight.shape != (num_edges,):
        raise ValueError(
            f"{graph_path} edge_distance and edge_weight must have shape ({num_edges},)"
        )
    if np.any(edge_index < 0) or np.any(edge_index >= num_gaussians):
        raise ValueError(f"{graph_path} edge_index contains an out-of-range index")
    if np.any(edge_index[:, 0] >= edge_index[:, 1]):
        raise ValueError(
            f"{graph_path} edge_index must contain ordered undirected edges with i < j"
        )
    if np.unique(edge_index, axis=0).shape[0] != num_edges:
        raise ValueError(f"{graph_path} edge_index contains duplicate edges")
    if np.any(~np.isfinite(edge_distance)) or np.any(edge_distance < 0.0):
        raise ValueError(f"{graph_path} edge_distance must be finite and non-negative")
    if np.any(~np.isfinite(edge_weight)) or np.any(edge_weight <= 0.0):
        raise ValueError(f"{graph_path} edge_weight must be finite and positive")
    if "unique_undirected_edge_count" in graph:
        count = np.asarray(graph["unique_undirected_edge_count"])
        if count.shape != () or int(count.item()) != num_edges:
            raise ValueError(
                f"{graph_path} unique_undirected_edge_count does not match edge_index"
            )
    if "degree" in graph:
        degree = np.asarray(graph["degree"])
        expected_degree = np.zeros((num_gaussians,), dtype=np.int64)
        np.add.at(expected_degree, edge_index[:, 0], 1)
        np.add.at(expected_degree, edge_index[:, 1], 1)
        if degree.shape != (num_gaussians,) or not np.array_equal(
            degree.astype(np.int64), expected_degree
        ):
            raise ValueError(f"{graph_path} degree does not match edge_index")
    for graph_key, parameter_key in (
        ("k", "motion_fill_k"),
        ("max_distance", "motion_fill_max_distance"),
        ("epsilon", "motion_fill_epsilon"),
    ):
        if parameter_key not in parameters:
            raise ValueError(f"{manifest_path} parameters is missing {parameter_key}")
        graph_value = np.asarray(graph.get(graph_key))
        if graph_value.shape != ():
            raise ValueError(f"{graph_path} {graph_key} must be scalar")
        expected = parameters[parameter_key]
        if graph_key == "k":
            matches = int(graph_value.item()) == int(expected)
        else:
            matches = np.isclose(
                float(graph_value.item()), float(expected), rtol=1e-9, atol=1e-12
            )
        if not matches:
            raise ValueError(
                f"{graph_path} {graph_key} does not match manifest {parameter_key}"
            )
    for field_name in ("edge_weighting", "color_metric", "color_sigma_source"):
        if field_name not in parameters:
            continue
        if field_name not in graph:
            raise ValueError(f"{graph_path} is missing manifest field {field_name}")
        if (
            _modal_scalar_string(graph[field_name], field_name, graph_path)
            != parameters[field_name]
        ):
            raise ValueError(f"{graph_path} {field_name} does not match the manifest")
    if "color_sigma" in parameters:
        if "color_sigma" not in graph:
            raise ValueError(f"{graph_path} is missing manifest field color_sigma")
        color_sigma = np.asarray(graph["color_sigma"])
        expected_color_sigma = parameters["color_sigma"]
        if color_sigma.shape != ():
            raise ValueError(f"{graph_path} color_sigma does not match the manifest")
        graph_color_sigma = float(color_sigma.item())
        if expected_color_sigma is None:
            matches_color_sigma = np.isnan(graph_color_sigma)
        else:
            matches_color_sigma = np.isclose(
                graph_color_sigma,
                float(expected_color_sigma),
                rtol=1e-9,
                atol=1e-12,
            )
        if not matches_color_sigma:
            raise ValueError(f"{graph_path} color_sigma does not match the manifest")
    return edge_index, edge_weight


def load_gaussian_modal_refinement_data(
    manifest_path: str,
    gaussian_means: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> GaussianModalRefinementData:
    """Load role masks and spatial constraints for formal phi refinement."""

    manifest = Path(manifest_path)
    with manifest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if int(payload.get("version", -1)) != 1:
        raise ValueError(f"{manifest_path} must be a version 1 modal manifest")
    if payload.get("point_type") != "foreground_gaussian_center":
        raise ValueError(
            f"{manifest_path} point_type must be foreground_gaussian_center"
        )
    if gaussian_means.ndim != 2 or gaussian_means.shape[1] != 3:
        raise ValueError(
            "foreground Gaussian means must have shape (N,3), "
            f"got {tuple(gaussian_means.shape)}"
        )
    if not torch.is_floating_point(gaussian_means):
        raise ValueError("foreground Gaussian means must have floating-point dtype")
    if not bool(torch.isfinite(gaussian_means).all().item()):
        raise ValueError("foreground Gaussian means must be finite")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("modal refinement tensors require floating-point dtype")
    gaussian_points = gaussian_means.detach().cpu().numpy().astype(np.float32)
    num_gaussians = int(gaussian_points.shape[0])

    modes_payload = payload.get("modes")
    if not isinstance(modes_payload, list) or not modes_payload:
        raise ValueError(f"{manifest_path} does not contain a non-empty modes list")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{manifest_path} parameters must be an object")

    modal_fields = load_gaussian_modal_fields(manifest_path, gaussian_means)
    if len(modal_fields.modes) != len(modes_payload):
        raise RuntimeError("Manifest mode loading changed the mode count")
    source_mode_indices = tuple(mode.mode_index for mode in modal_fields.modes)
    graph_edge_index, graph_edge_weight = _load_modal_refinement_graph(
        manifest, parameters, gaussian_points
    )

    display_classes: list[np.ndarray] = []
    refinement_masks: list[np.ndarray] = []
    role_counts: list[np.ndarray] = []
    staged_refinement_energies: list[float] = []
    for mode in modal_fields.modes:
        latent_path = mode.latent_path
        with np.load(str(latent_path), allow_pickle=False) as loaded:
            required = {
                "motion_fill_role",
                "motion_fill_role_names",
                "completion_mask",
            }
            missing = sorted(required - set(loaded.files))
            if missing:
                raise ValueError(
                    f"{latent_path} missing motion-fill role fields required for "
                    f"role-based refinement: {missing}"
                )
            role = np.asarray(loaded["motion_fill_role"])
            role_names = _modal_string_vector(
                loaded["motion_fill_role_names"],
                "motion_fill_role_names",
                latent_path,
            )
            completion = np.asarray(loaded["completion_mask"])
        if role_names != tuple(MOTION_FILL_ROLE_NAMES):
            raise ValueError(
                f"{latent_path} motion_fill_role_names does not match formal roles"
            )
        if role.shape != (num_gaussians,) or not np.issubdtype(
            role.dtype, np.integer
        ):
            raise ValueError(
                f"{latent_path} motion_fill_role must be an integer array with "
                f"shape ({num_gaussians},)"
            )
        display_class = classify_motion_fill_display_points(role, completion)
        refinement_mask = (
            (display_class == MOTION_FILL_DISPLAY_ANCHOR)
            | (display_class == MOTION_FILL_DISPLAY_PARTIAL)
            | (display_class == MOTION_FILL_DISPLAY_FILLED)
        )
        if not np.any(refinement_mask):
            raise ValueError(
                f"{latent_path} has no anchor, partial, or filled Gaussians"
            )
        counts = np.asarray(
            [
                np.count_nonzero(display_class == display_role)
                for display_role in (
                    MOTION_FILL_DISPLAY_ANCHOR,
                    MOTION_FILL_DISPLAY_PARTIAL,
                    MOTION_FILL_DISPLAY_FILLED,
                )
            ],
            dtype=np.int64,
        )
        staged_energy = float(
            np.mean(
                np.sum(np.abs(mode.phi[refinement_mask]) ** 2, axis=1),
                dtype=np.float64,
            )
        )
        if not np.isfinite(staged_energy) or staged_energy <= 1e-12:
            raise ValueError(
                f"{latent_path} staged refinement energy must be finite and "
                "above 1e-12"
            )
        display_classes.append(display_class)
        refinement_masks.append(refinement_mask)
        role_counts.append(counts)
        staged_refinement_energies.append(staged_energy)

    refinement_edges: list[np.ndarray] = []
    refinement_edge_modes: list[np.ndarray] = []
    refinement_edge_weights: list[np.ndarray] = []
    for mode_slot, refinement_mask in enumerate(refinement_masks):
        keep_edges = refinement_mask[graph_edge_index[:, 0]] & refinement_mask[
            graph_edge_index[:, 1]
        ]
        selected_edges = graph_edge_index[keep_edges]
        selected_weights = graph_edge_weight[keep_edges]
        if selected_edges.shape[0] == 0:
            raise ValueError(
                f"Mode {source_mode_indices[mode_slot]} has no trainable graph edges"
            )
        mean_weight = float(np.mean(selected_weights, dtype=np.float64))
        if not np.isfinite(mean_weight) or mean_weight <= 0.0:
            raise ValueError(
                f"Mode {source_mode_indices[mode_slot]} refinement-edge mean "
                "weight is invalid"
            )
        normalized_weights = selected_weights / mean_weight
        if not np.isfinite(normalized_weights).all() or np.any(
            normalized_weights <= 0.0
        ):
            raise ValueError(
                f"Mode {source_mode_indices[mode_slot]} normalized refinement-edge "
                "weights must be finite and positive"
            )
        refinement_edges.append(selected_edges)
        refinement_edge_modes.append(
            np.full((selected_edges.shape[0],), mode_slot, dtype=np.int64)
        )
        refinement_edge_weights.append(normalized_weights)

    edge_index_array = np.concatenate(refinement_edges, axis=0).T.copy()
    edge_mode_array = np.concatenate(refinement_edge_modes, axis=0)
    edge_weight_array = np.concatenate(refinement_edge_weights, axis=0)
    return GaussianModalRefinementData(
        source_mode_indices=source_mode_indices,
        refinement_mask=torch.as_tensor(
            np.stack(refinement_masks, axis=0),
            device=device,
            dtype=torch.bool,
        ),
        display_class=torch.as_tensor(
            np.stack(display_classes, axis=0),
            device=device,
            dtype=torch.long,
        ),
        role_counts=torch.as_tensor(
            np.stack(role_counts, axis=0),
            device=device,
            dtype=torch.long,
        ),
        staged_refinement_energy=torch.as_tensor(
            staged_refinement_energies,
            device=device,
            dtype=dtype,
        ),
        refinement_edge_index=torch.as_tensor(
            edge_index_array,
            device=device,
            dtype=torch.long,
        ),
        refinement_edge_mode_indices=torch.as_tensor(
            edge_mode_array,
            device=device,
            dtype=torch.long,
        ),
        refinement_edge_weights=torch.as_tensor(
            edge_weight_array,
            device=device,
            dtype=dtype,
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
