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
    view_ids: tuple[str, ...]
    source_mode_indices: tuple[int, ...]
    anchor_mask: torch.Tensor
    obs_y_real: torch.Tensor
    obs_y_imag: torch.Tensor
    obs_J: torch.Tensor
    obs_gaussian_indices: torch.Tensor
    obs_mode_indices: torch.Tensor
    obs_view_indices: torch.Tensor
    obs_group_indices: torch.Tensor
    obs_alpha_real: torch.Tensor
    obs_alpha_imag: torch.Tensor
    obs_effective_weights: torch.Tensor
    group_target_energy: torch.Tensor
    group_mode_indices: torch.Tensor
    group_view_indices: torch.Tensor
    anchor_edge_index: torch.Tensor
    anchor_edge_mode_indices: torch.Tensor
    anchor_edge_weights: torch.Tensor
    staged_anchor_energy: torch.Tensor


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


def _validate_modal_point_identity(
    arrays: dict[str, np.ndarray],
    path: Path,
    gaussian_points: np.ndarray,
    source_checkpoint: str,
) -> None:
    required = {"points_world", "gaussian_indices", "point_type", "source_checkpoint"}
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{path} missing Gaussian identity fields: {missing}")
    num_gaussians = int(gaussian_points.shape[0])
    points_world = np.asarray(arrays["points_world"], dtype=np.float32)
    if points_world.shape != (num_gaussians, 3):
        raise ValueError(
            f"{path} points_world must have shape ({num_gaussians},3), "
            f"got {points_world.shape}"
        )
    if not np.isfinite(points_world).all():
        raise ValueError(f"{path} points_world must be finite")
    max_delta = (
        float(np.max(np.abs(points_world - gaussian_points)))
        if num_gaussians > 0
        else 0.0
    )
    if max_delta > 1e-5:
        raise ValueError(
            f"{path} points_world differs from foreground Gaussian means by "
            f"max {max_delta:.6g}, above 1e-05"
        )
    gaussian_indices = np.asarray(arrays["gaussian_indices"])
    if gaussian_indices.shape != (num_gaussians,) or not np.issubdtype(
        gaussian_indices.dtype, np.integer
    ):
        raise ValueError(
            f"{path} gaussian_indices must be an integer array with shape "
            f"({num_gaussians},)"
        )
    if not np.array_equal(
        gaussian_indices.astype(np.int64), np.arange(num_gaussians, dtype=np.int64)
    ):
        raise ValueError(f"{path} gaussian_indices must preserve foreground order")
    if _modal_scalar_string(arrays["point_type"], "point_type", path) != (
        "foreground_gaussian_center"
    ):
        raise ValueError(f"{path} point_type must be foreground_gaussian_center")
    if (
        _modal_scalar_string(
            arrays["source_checkpoint"], "source_checkpoint", path
        )
        != source_checkpoint
    ):
        raise ValueError(f"{path} source_checkpoint does not match the manifest")


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
    view_ids: list[str] | tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> GaussianModalRefinementData:
    """Load fixed-anchor pixel-candidate constraints for formal phi refinement."""

    manifest = Path(manifest_path)
    with manifest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if int(payload.get("version", -1)) != 1:
        raise ValueError(f"{manifest_path} must be a version 1 modal manifest")
    if payload.get("point_type") != "foreground_gaussian_center":
        raise ValueError(
            f"{manifest_path} point_type must be foreground_gaussian_center"
        )
    source_checkpoint = payload.get("source_checkpoint")
    if not isinstance(source_checkpoint, str) or not source_checkpoint:
        raise ValueError(f"{manifest_path} is missing source_checkpoint")
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

    ordered_view_ids = tuple(view_ids)
    if not ordered_view_ids or any(
        not isinstance(view_id, str) or not view_id for view_id in ordered_view_ids
    ):
        raise ValueError("view_ids must contain non-empty strings")
    if len(set(ordered_view_ids)) != len(ordered_view_ids):
        raise ValueError("view_ids must be unique")
    num_views = len(ordered_view_ids)

    modes_payload = payload.get("modes")
    if not isinstance(modes_payload, list) or not modes_payload:
        raise ValueError(f"{manifest_path} does not contain a non-empty modes list")
    source_mode_indices = tuple(
        mode_payload.get("mode_index")
        for mode_payload in modes_payload
        if isinstance(mode_payload, dict)
    )
    if len(source_mode_indices) != len(modes_payload) or any(
        isinstance(mode_index, bool) or not isinstance(mode_index, int)
        for mode_index in source_mode_indices
    ):
        raise ValueError(f"{manifest_path} modes must contain integer mode_index values")
    if len(set(source_mode_indices)) != len(source_mode_indices):
        raise ValueError(f"{manifest_path} mode_index values must be unique")
    manifest_mode_indices = payload.get("mode_indices")
    if manifest_mode_indices != list(source_mode_indices):
        raise ValueError(f"{manifest_path} mode_indices does not match modes order")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{manifest_path} parameters must be an object")
    if parameters.get("alpha_model") != "per_view_per_mode":
        raise ValueError(
            f"{manifest_path} must use alpha_model='per_view_per_mode'"
        )
    reference_view = parameters.get("alpha_reference_view_index")
    if isinstance(reference_view, bool) or reference_view != 0:
        raise ValueError(f"{manifest_path} alpha reference view index must be 0")

    modal_fields = load_gaussian_modal_fields(manifest_path, gaussian_means)
    graph_edge_index, graph_edge_weight = _load_modal_refinement_graph(
        manifest, parameters, gaussian_points
    )

    anchor_masks: list[np.ndarray] = []
    staged_anchor_energies: list[float] = []
    y_values: list[np.ndarray] = []
    J_values: list[np.ndarray] = []
    gaussian_index_values: list[np.ndarray] = []
    mode_index_values: list[np.ndarray] = []
    view_index_values: list[np.ndarray] = []
    group_index_values: list[np.ndarray] = []
    alpha_values: list[np.ndarray] = []
    effective_weight_values: list[np.ndarray] = []

    for mode_slot, (mode_payload, mode) in enumerate(
        zip(modes_payload, modal_fields.modes)
    ):
        assert isinstance(mode_payload, dict)
        manifest_freq = mode_payload.get("freq_hz")
        if (
            isinstance(manifest_freq, bool)
            or not isinstance(manifest_freq, (int, float))
            or not np.isfinite(manifest_freq)
            or float(manifest_freq) <= 0.0
        ):
            raise ValueError(
                f"Mode {mode.mode_index} in {manifest_path} has invalid freq_hz"
            )
        latent_path = mode.latent_path
        with np.load(str(latent_path), allow_pickle=False) as loaded:
            required = {"motion_fill_role", "motion_fill_role_names", "completion_mask"}
            missing = sorted(required - set(loaded.files))
            if missing:
                raise ValueError(
                    f"{latent_path} missing motion-fill role fields required for "
                    f"anchor refinement: {missing}"
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
        classify_motion_fill_display_points(role, completion)
        anchor_mask = role == MOTION_FILL_ROLE_FIXED_ANCHOR
        anchor_count = int(np.count_nonzero(anchor_mask))
        if anchor_count == 0:
            raise ValueError(f"{latent_path} does not contain fixed-anchor Gaussians")
        anchor_masks.append(anchor_mask)
        staged_energy = float(
            np.mean(np.sum(np.abs(mode.phi[anchor_mask]) ** 2, axis=1), dtype=np.float64)
        )
        if not np.isfinite(staged_energy) or staged_energy <= 1e-12:
            raise ValueError(
                f"{latent_path} staged fixed-anchor energy must be finite and above 1e-12"
            )
        staged_anchor_energies.append(staged_energy)

        observation_path = _resolve_modal_artifact_path(
            manifest, mode_payload.get("observation_path"), "observation_path"
        )
        with np.load(str(observation_path), allow_pickle=False) as loaded:
            observations = {key: loaded[key] for key in loaded.files}
        required_observation = {
            "points_world",
            "gaussian_indices",
            "point_type",
            "source_checkpoint",
            "obs_point_index",
            "obs_view_index",
            "obs_pixels_xy",
            "obs_y",
            "obs_J",
            "obs_contribution_weight",
            "obs_count_per_point",
            "obs_sample_count_per_point",
            "view_ids",
            "view_freqs_hz",
            "freq_hz",
            "mode_index",
            "observations_per_view",
            "pixel_candidate_method",
            "preserved_all_points",
            "candidate_point_count",
        }
        missing = sorted(required_observation - set(observations))
        if missing:
            raise ValueError(
                f"{observation_path} missing required pixel-candidate fields: {missing}"
            )
        _validate_modal_point_identity(
            observations, observation_path, gaussian_points, source_checkpoint
        )
        if (
            _modal_scalar_string(
                observations["pixel_candidate_method"],
                "pixel_candidate_method",
                observation_path,
            )
            != "rendered_depth_gaussian_contribution"
        ):
            raise ValueError(
                f"{observation_path} is not a formal pixel-candidate observation graph"
            )
        preserved = np.asarray(observations["preserved_all_points"])
        if preserved.shape != () or preserved.dtype != np.bool_ or not bool(
            preserved.item()
        ):
            raise ValueError(
                f"{observation_path} must preserve every foreground Gaussian"
            )
        candidate_point_count = np.asarray(observations["candidate_point_count"])
        if (
            candidate_point_count.shape != ()
            or not np.issubdtype(candidate_point_count.dtype, np.integer)
            or int(candidate_point_count.item()) != num_gaussians
        ):
            raise ValueError(
                f"{observation_path} candidate_point_count does not match foreground"
            )
        observation_view_ids = _modal_string_vector(
            observations["view_ids"], "view_ids", observation_path
        )
        if observation_view_ids != ordered_view_ids:
            raise ValueError(
                f"{observation_path} view_ids {observation_view_ids!r} do not match "
                f"frame-map order {ordered_view_ids!r}"
            )
        mode_index = np.asarray(observations["mode_index"])
        freq_hz = np.asarray(observations["freq_hz"])
        if mode_index.shape != () or not np.issubdtype(mode_index.dtype, np.integer):
            raise ValueError(f"{observation_path} mode_index must be an integer scalar")
        if int(mode_index.item()) != mode.mode_index:
            raise ValueError(f"{observation_path} mode_index does not match the manifest")
        if (
            freq_hz.shape != ()
            or not np.issubdtype(freq_hz.dtype, np.number)
            or np.iscomplexobj(freq_hz)
            or not np.isfinite(freq_hz.item())
            or not np.isclose(
                float(freq_hz.item()), float(manifest_freq), rtol=1e-6, atol=1e-6
            )
        ):
            raise ValueError(f"{observation_path} freq_hz does not match the manifest")
        view_freqs = np.asarray(observations["view_freqs_hz"], dtype=np.float64)
        manifest_view_freqs = np.asarray(mode_payload.get("freqs_hz_by_view"))
        if (
            view_freqs.shape != (num_views,)
            or manifest_view_freqs.shape != (num_views,)
            or not np.isfinite(view_freqs).all()
            or not np.isfinite(manifest_view_freqs.astype(np.float64)).all()
            or not np.allclose(
                view_freqs,
                manifest_view_freqs.astype(np.float64),
                rtol=1e-6,
                atol=1e-6,
            )
        ):
            raise ValueError(
                f"{observation_path} view frequencies do not match the manifest"
            )

        obs_y = np.asarray(observations["obs_y"])
        obs_J = np.asarray(observations["obs_J"])
        point_indices = np.asarray(observations["obs_point_index"])
        obs_view_indices = np.asarray(observations["obs_view_index"])
        pixels = np.asarray(observations["obs_pixels_xy"])
        contribution_weight = np.asarray(observations["obs_contribution_weight"])
        num_observations = int(obs_y.shape[0])
        if obs_y.shape != (num_observations, 2) or not np.iscomplexobj(obs_y):
            raise ValueError(f"{observation_path} obs_y must have complex shape (O,2)")
        if (
            obs_J.shape != (num_observations, 2, 3)
            or not np.issubdtype(obs_J.dtype, np.number)
            or np.iscomplexobj(obs_J)
        ):
            raise ValueError(f"{observation_path} obs_J must have real shape (O,2,3)")
        if pixels.shape != (num_observations, 2) or not np.issubdtype(
            pixels.dtype, np.number
        ):
            raise ValueError(
                f"{observation_path} obs_pixels_xy must have shape (O,2)"
            )
        for values, name in (
            (point_indices, "obs_point_index"),
            (obs_view_indices, "obs_view_index"),
        ):
            if values.shape != (num_observations,) or not np.issubdtype(
                values.dtype, np.integer
            ):
                raise ValueError(
                    f"{observation_path} {name} must be an integer array with shape (O,)"
                )
        point_indices = point_indices.astype(np.int64)
        obs_view_indices = obs_view_indices.astype(np.int64)
        if np.any(point_indices < 0) or np.any(point_indices >= num_gaussians):
            raise ValueError(f"{observation_path} obs_point_index is out of range")
        if np.any(obs_view_indices < 0) or np.any(obs_view_indices >= num_views):
            raise ValueError(f"{observation_path} obs_view_index is out of range")
        if (
            contribution_weight.shape != (num_observations,)
            or not np.issubdtype(contribution_weight.dtype, np.number)
            or np.iscomplexobj(contribution_weight)
        ):
            raise ValueError(
                f"{observation_path} obs_contribution_weight must have shape (O,)"
            )
        if (
            not np.isfinite(obs_y.real).all()
            or not np.isfinite(obs_y.imag).all()
            or not np.isfinite(obs_J).all()
            or not np.isfinite(pixels).all()
            or not np.isfinite(contribution_weight).all()
            or np.any(contribution_weight < 0.0)
        ):
            raise ValueError(f"{observation_path} observation arrays must be finite")
        derived_sample_count = np.bincount(
            point_indices, minlength=num_gaussians
        ).astype(np.int64)
        stored_sample_count = np.asarray(observations["obs_sample_count_per_point"])
        if (
            stored_sample_count.shape != (num_gaussians,)
            or not np.issubdtype(stored_sample_count.dtype, np.integer)
            or not np.array_equal(
                stored_sample_count.astype(np.int64), derived_sample_count
            )
        ):
            raise ValueError(
                f"{observation_path} obs_sample_count_per_point does not match rows"
            )
        point_view_mask = np.zeros((num_gaussians, num_views), dtype=bool)
        point_view_mask[point_indices, obs_view_indices] = True
        derived_view_count = point_view_mask.sum(axis=1).astype(np.int64)
        stored_view_count = np.asarray(observations["obs_count_per_point"])
        if (
            stored_view_count.shape != (num_gaussians,)
            or not np.issubdtype(stored_view_count.dtype, np.integer)
            or not np.array_equal(
                stored_view_count.astype(np.int64), derived_view_count
            )
        ):
            raise ValueError(
                f"{observation_path} obs_count_per_point does not match unique views"
            )
        observations_per_view = np.asarray(observations["observations_per_view"])
        derived_per_view = np.bincount(
            obs_view_indices, minlength=num_views
        ).astype(np.int64)
        if (
            observations_per_view.shape != (num_views,)
            or not np.issubdtype(observations_per_view.dtype, np.integer)
            or not np.array_equal(
                observations_per_view.astype(np.int64), derived_per_view
            )
        ):
            raise ValueError(
                f"{observation_path} observations_per_view does not match rows"
            )
        pair_key = point_indices * num_views + obs_view_indices
        _, inverse, multiplicity = np.unique(
            pair_key, return_inverse=True, return_counts=True
        )
        effective_weight = contribution_weight.astype(np.float64) / multiplicity[
            inverse
        ].astype(np.float64)

        diagnostics_path = _resolve_modal_artifact_path(
            manifest, mode_payload.get("diagnostics_path"), "diagnostics_path"
        )
        with np.load(str(diagnostics_path), allow_pickle=False) as loaded:
            diagnostics = {key: loaded[key] for key in loaded.files}
        required_diagnostics = {
            "alphas",
            "alpha_identifiable_mask",
            "alpha_reference_view_index",
            "alpha_view_freqs_hz",
        }
        missing = sorted(required_diagnostics - set(diagnostics))
        if missing:
            raise ValueError(
                f"{diagnostics_path} missing required alpha fields: {missing}"
            )
        alphas = np.asarray(diagnostics["alphas"])
        identifiable = np.asarray(diagnostics["alpha_identifiable_mask"])
        diagnostics_reference = np.asarray(diagnostics["alpha_reference_view_index"])
        diagnostics_freqs = np.asarray(
            diagnostics["alpha_view_freqs_hz"], dtype=np.float64
        )
        if alphas.shape != (num_views,) or not np.iscomplexobj(alphas):
            raise ValueError(f"{diagnostics_path} alphas must have complex shape (V,)")
        if identifiable.shape != (num_views,) or identifiable.dtype != np.bool_:
            raise ValueError(
                f"{diagnostics_path} alpha_identifiable_mask must have boolean shape (V,)"
            )
        if (
            diagnostics_reference.shape != ()
            or not np.issubdtype(diagnostics_reference.dtype, np.integer)
            or int(diagnostics_reference.item()) != 0
        ):
            raise ValueError(
                f"{diagnostics_path} alpha_reference_view_index must be 0"
            )
        if (
            diagnostics_freqs.shape != (num_views,)
            or not np.allclose(
                diagnostics_freqs, view_freqs, rtol=1e-6, atol=1e-6
            )
        ):
            raise ValueError(
                f"{diagnostics_path} alpha frequencies do not match observations"
            )
        if not np.isfinite(alphas.real).all() or not np.isfinite(alphas.imag).all():
            raise ValueError(f"{diagnostics_path} alphas must be finite")

        alpha_by_view = mode_payload.get("alpha_by_view")
        if not isinstance(alpha_by_view, list) or len(alpha_by_view) != num_views:
            raise ValueError(
                f"Mode {mode.mode_index} in {manifest_path} has invalid alpha_by_view"
            )
        manifest_alphas = np.empty((num_views,), dtype=np.complex128)
        manifest_identifiable = np.empty((num_views,), dtype=bool)
        for view_index, (view_id, alpha_entry) in enumerate(
            zip(ordered_view_ids, alpha_by_view)
        ):
            if not isinstance(alpha_entry, dict) or alpha_entry.get("view_id") != view_id:
                raise ValueError(
                    f"Mode {mode.mode_index} alpha_by_view does not match view order"
                )
            real = alpha_entry.get("real")
            imaginary = alpha_entry.get("imag")
            alpha_identifiable = alpha_entry.get("identifiable")
            alpha_freq = alpha_entry.get("freq_hz")
            if (
                isinstance(real, bool)
                or not isinstance(real, (int, float))
                or isinstance(imaginary, bool)
                or not isinstance(imaginary, (int, float))
                or not np.isfinite(real)
                or not np.isfinite(imaginary)
                or not isinstance(alpha_identifiable, bool)
                or isinstance(alpha_freq, bool)
                or not isinstance(alpha_freq, (int, float))
                or not np.isfinite(alpha_freq)
                or not np.isclose(
                    float(alpha_freq), view_freqs[view_index], rtol=1e-6, atol=1e-6
                )
            ):
                raise ValueError(
                    f"Mode {mode.mode_index} alpha_by_view has invalid numeric fields"
                )
            manifest_alphas[view_index] = complex(real, imaginary)
            manifest_identifiable[view_index] = alpha_identifiable
        if not np.allclose(
            manifest_alphas, alphas.astype(np.complex128), rtol=1e-6, atol=1e-7
        ) or not np.array_equal(manifest_identifiable, identifiable):
            raise ValueError(
                f"{diagnostics_path} alpha values do not match the manifest"
            )

        keep = (
            anchor_mask[point_indices]
            & identifiable[obs_view_indices]
            & (effective_weight > 0.0)
        )
        kept_rows = np.where(keep)[0]
        if kept_rows.size == 0:
            raise ValueError(
                f"Mode {mode.mode_index} has no positive-weight identifiable anchor rows"
            )
        kept_views = obs_view_indices[kept_rows]
        kept_y = obs_y[kept_rows].astype(np.complex64)
        kept_J = obs_J[kept_rows].astype(np.float32)
        kept_points = point_indices[kept_rows]
        kept_weights = effective_weight[kept_rows]
        kept_groups = mode_slot * num_views + kept_views

        y_values.append(kept_y)
        J_values.append(kept_J)
        gaussian_index_values.append(kept_points)
        mode_index_values.append(
            np.full((kept_rows.size,), mode_slot, dtype=np.int64)
        )
        view_index_values.append(kept_views)
        group_index_values.append(kept_groups)
        alpha_values.append(alphas[kept_views].astype(np.complex64))
        effective_weight_values.append(kept_weights)

    anchor_mask_array = np.stack(anchor_masks, axis=0)
    anchor_edges: list[np.ndarray] = []
    anchor_edge_modes: list[np.ndarray] = []
    anchor_edge_weights: list[np.ndarray] = []
    for mode_slot, anchor_mask in enumerate(anchor_masks):
        keep_edges = anchor_mask[graph_edge_index[:, 0]] & anchor_mask[
            graph_edge_index[:, 1]
        ]
        selected_edges = graph_edge_index[keep_edges]
        selected_weights = graph_edge_weight[keep_edges]
        if selected_edges.shape[0] == 0:
            raise ValueError(
                f"Mode {source_mode_indices[mode_slot]} has no anchor-anchor graph edges"
            )
        mean_weight = float(np.mean(selected_weights, dtype=np.float64))
        if not np.isfinite(mean_weight) or mean_weight <= 0.0:
            raise ValueError(
                f"Mode {source_mode_indices[mode_slot]} anchor-edge mean weight is invalid"
            )
        normalized_weights = selected_weights / mean_weight
        if not np.isfinite(normalized_weights).all() or np.any(
            normalized_weights <= 0.0
        ):
            raise ValueError(
                f"Mode {source_mode_indices[mode_slot]} normalized anchor-edge weights "
                "must be finite and positive"
            )
        anchor_edges.append(selected_edges)
        anchor_edge_modes.append(
            np.full((selected_edges.shape[0],), mode_slot, dtype=np.int64)
        )
        anchor_edge_weights.append(normalized_weights)

    y_array = np.concatenate(y_values, axis=0)
    J_array = np.concatenate(J_values, axis=0)
    gaussian_index_array = np.concatenate(gaussian_index_values, axis=0)
    mode_index_array = np.concatenate(mode_index_values, axis=0)
    view_index_array = np.concatenate(view_index_values, axis=0)
    group_index_array = np.concatenate(group_index_values, axis=0)
    alpha_array = np.concatenate(alpha_values, axis=0)
    effective_weight_array = np.concatenate(effective_weight_values, axis=0)
    active_group_ids = np.unique(group_index_array)
    group_mode_array = active_group_ids // num_views
    group_view_array = active_group_ids % num_views
    active_groups_per_mode = np.bincount(
        group_mode_array, minlength=len(modes_payload)
    )
    if np.any(active_groups_per_mode == 0):
        missing_modes = [
            source_mode_indices[index]
            for index in np.where(active_groups_per_mode == 0)[0].tolist()
        ]
        raise RuntimeError(
            f"Internal error: no active modal observation group for modes {missing_modes}"
        )
    group_index_array = np.searchsorted(active_group_ids, group_index_array)
    num_groups = int(active_group_ids.shape[0])
    group_target_energy = np.bincount(
        group_index_array,
        weights=(
            effective_weight_array
            * np.sum(np.abs(y_array.astype(np.complex128)) ** 2, axis=1)
        ),
        minlength=num_groups,
    )
    group_row_count = np.bincount(group_index_array, minlength=num_groups)
    if group_target_energy.shape != (num_groups,) or np.any(group_row_count == 0):
        raise RuntimeError("Internal error: active modal observation group is empty")
    if np.any(~np.isfinite(group_target_energy)) or np.any(
        group_target_energy <= 1e-12
    ):
        bad_groups = np.where(
            (~np.isfinite(group_target_energy)) | (group_target_energy <= 1e-12)
        )[0]
        labels = [
            f"mode {source_mode_indices[int(group_mode_array[index])]} "
            f"view {ordered_view_ids[int(group_view_array[index])]}"
            for index in bad_groups.tolist()
        ]
        raise ValueError(
            "Anchor target energy must be finite and above 1e-12 for active "
            f"groups: {labels}"
        )

    edge_index_array = np.concatenate(anchor_edges, axis=0).T.copy()
    edge_mode_array = np.concatenate(anchor_edge_modes, axis=0)
    edge_weight_array = np.concatenate(anchor_edge_weights, axis=0)
    return GaussianModalRefinementData(
        view_ids=ordered_view_ids,
        source_mode_indices=tuple(int(index) for index in source_mode_indices),
        anchor_mask=torch.as_tensor(anchor_mask_array, device=device, dtype=torch.bool),
        obs_y_real=torch.as_tensor(y_array.real, device=device, dtype=dtype),
        obs_y_imag=torch.as_tensor(y_array.imag, device=device, dtype=dtype),
        obs_J=torch.as_tensor(J_array, device=device, dtype=dtype),
        obs_gaussian_indices=torch.as_tensor(
            gaussian_index_array, device=device, dtype=torch.long
        ),
        obs_mode_indices=torch.as_tensor(
            mode_index_array, device=device, dtype=torch.long
        ),
        obs_view_indices=torch.as_tensor(
            view_index_array, device=device, dtype=torch.long
        ),
        obs_group_indices=torch.as_tensor(
            group_index_array, device=device, dtype=torch.long
        ),
        obs_alpha_real=torch.as_tensor(alpha_array.real, device=device, dtype=dtype),
        obs_alpha_imag=torch.as_tensor(alpha_array.imag, device=device, dtype=dtype),
        obs_effective_weights=torch.as_tensor(
            effective_weight_array, device=device, dtype=dtype
        ),
        group_target_energy=torch.as_tensor(
            group_target_energy, device=device, dtype=dtype
        ),
        group_mode_indices=torch.as_tensor(
            group_mode_array, device=device, dtype=torch.long
        ),
        group_view_indices=torch.as_tensor(
            group_view_array, device=device, dtype=torch.long
        ),
        anchor_edge_index=torch.as_tensor(
            edge_index_array, device=device, dtype=torch.long
        ),
        anchor_edge_mode_indices=torch.as_tensor(
            edge_mode_array, device=device, dtype=torch.long
        ),
        anchor_edge_weights=torch.as_tensor(
            edge_weight_array, device=device, dtype=dtype
        ),
        staged_anchor_energy=torch.as_tensor(
            staged_anchor_energies, device=device, dtype=dtype
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
