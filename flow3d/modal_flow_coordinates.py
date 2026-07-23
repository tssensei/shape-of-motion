from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import solve_triangular

from modal_peak_pick.core.cache import ModalAnalysisCache, load_analysis_cache
from modal_surface.gaussian_observations import load_gaussian_observation_topology


MODAL_FLOW_COORDINATE_FORMAT = "modal_flow_coordinates"
MODAL_FLOW_COORDINATE_VERSION = 1
MODAL_FLOW_COORDINATE_PARAMETERIZATION = "per_frame_flow_coordinates_v1"
MODAL_FLOW_COORDINATE_SOLVER = "reference_flow_ridge_v1"
MODAL_PHYSICS_COORDINATE_SOLVER = "latent_force_oscillator_postfit_v1"
MODAL_FLOW_COORDINATE_GAUGE = "per_view_temporal_mean_zero"
SUPPORTED_MODAL_FLOW_COORDINATE_SOLVERS = {
    MODAL_FLOW_COORDINATE_SOLVER,
    MODAL_PHYSICS_COORDINATE_SOLVER,
}

COORDINATE_FILENAME = "modal_flow_coordinates.npz"
DIAGNOSTICS_NPZ_FILENAME = "coordinate_diagnostics.npz"
DIAGNOSTICS_JSON_FILENAME = "diagnostics.json"

__all__ = [
    "COORDINATE_FILENAME",
    "DIAGNOSTICS_JSON_FILENAME",
    "DIAGNOSTICS_NPZ_FILENAME",
    "MODAL_FLOW_COORDINATE_FORMAT",
    "MODAL_FLOW_COORDINATE_GAUGE",
    "MODAL_FLOW_COORDINATE_PARAMETERIZATION",
    "MODAL_FLOW_COORDINATE_SOLVER",
    "MODAL_PHYSICS_COORDINATE_SOLVER",
    "MODAL_FLOW_COORDINATE_VERSION",
    "SUPPORTED_MODAL_FLOW_COORDINATE_SOLVERS",
    "ModalFlowCoordinates",
    "evaluate_modal_flow_coordinate_sets",
    "load_flow_observation_topology",
    "load_modal_coordinate_provenance",
    "load_modal_flow_coordinates",
    "parse_flow_cache_specs",
    "solve_modal_flow_coordinates",
]

_COORDINATE_FIELDS = {
    "format",
    "version",
    "view_ids",
    "frame_names",
    "frame_view_indices",
    "frame_local_indices",
    "frame_times_sec",
    "mode_indices",
    "frequencies_hz",
    "coordinate_real",
    "coordinate_imag",
    "reference_local_indices",
    "ridge_relative",
    "source_modal_manifest",
    "source_modal_frame_map",
    "source_flow_cache_dirs",
}

_DIAGNOSTIC_FIELDS = {
    "view_ids",
    "frame_names",
    "frame_view_indices",
    "frame_local_indices",
    "mode_indices",
    "frequencies_hz",
    "candidate_row_count",
    "unique_pixel_count",
    "candidate_weight_sum_min",
    "candidate_weight_sum_max",
    "candidate_weight_sum_max_abs_error",
    "mode_pair_scales",
    "singular_values",
    "numerical_rank",
    "condition_number",
    "ridge_condition_number",
    "reference_local_indices",
    "per_frame_flow_rmse",
    "per_frame_relative_residual",
    "per_frame_flow_r2",
    "view_flow_rmse",
    "view_relative_residual",
    "view_flow_r2",
    "view_reference_residual_norm",
    "coordinate_rms",
    "coordinate_max",
    "coordinate_temporal_mean_abs",
    "coordinate_dominant_signed_frequency_hz",
    "coordinate_assigned_frequency_energy_ratio",
    "coordinate_cross_mode_correlation",
    "view_solve_seconds",
}


@dataclass(frozen=True)
class ModalFlowCoordinates:
    """Validated fixed per-frame complex coordinates and their provenance."""

    path: Path
    view_ids: tuple[str, ...]
    frame_names: tuple[str, ...]
    frame_view_indices: np.ndarray
    frame_local_indices: np.ndarray
    frame_times_sec: np.ndarray
    mode_indices: np.ndarray
    frequencies_hz: np.ndarray
    coordinate_real: np.ndarray
    coordinate_imag: np.ndarray
    reference_local_indices: np.ndarray
    ridge_relative: float
    source_modal_manifest: Path
    source_modal_frame_map: Path
    source_flow_cache_dirs: tuple[Path, ...]

    @property
    def coordinates(self) -> np.ndarray:
        return self.coordinate_real + 1j * self.coordinate_imag


def load_modal_coordinate_provenance(
    coordinates: ModalFlowCoordinates,
) -> dict[str, Any]:
    """Read solver provenance without changing the strict version-1 NPZ schema."""

    provenance: dict[str, Any] = {
        "solver": MODAL_FLOW_COORDINATE_SOLVER,
        "gauge": MODAL_FLOW_COORDINATE_GAUGE,
    }
    diagnostics_path = coordinates.path.parent / DIAGNOSTICS_JSON_FILENAME
    if not diagnostics_path.is_file():
        return provenance
    payload = _read_json_object(diagnostics_path)
    artifact_format = payload.get("format")
    if artifact_format == MODAL_FLOW_COORDINATE_FORMAT:
        solver = payload.get("solver")
        gauge = payload.get("gauge")
        if solver != MODAL_FLOW_COORDINATE_SOLVER or gauge != MODAL_FLOW_COORDINATE_GAUGE:
            raise ValueError(
                f"{diagnostics_path} has incompatible coordinate solver provenance"
            )
        return provenance
    if artifact_format != "modal_physics_coordinates" or payload.get("version") != 1:
        raise ValueError(
            f"{diagnostics_path} has unsupported coordinate diagnostics format/version"
        )
    solver = payload.get("solver")
    gauge = payload.get("gauge")
    if solver != MODAL_PHYSICS_COORDINATE_SOLVER:
        raise ValueError(f"{diagnostics_path} has unsupported physics coordinate solver")
    if gauge != MODAL_FLOW_COORDINATE_GAUGE:
        raise ValueError(f"{diagnostics_path} has unsupported physics coordinate gauge")
    output_name = payload.get("coordinate_artifact")
    if output_name != coordinates.path.name:
        raise ValueError(
            f"{diagnostics_path} coordinate_artifact does not identify {coordinates.path.name}"
        )
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"{diagnostics_path} settings must be an object")
    provenance.update(
        {
            "solver": solver,
            "gauge": gauge,
            "physics": settings,
            "source_coordinate": payload.get("source_coordinate"),
        }
    )
    return provenance


@dataclass(frozen=True)
class _FrameMapData:
    path: Path
    view_ids: tuple[str, ...]
    fps_hz: np.ndarray
    frame_names: tuple[str, ...]
    frame_view_indices: np.ndarray
    frame_local_indices: np.ndarray
    frame_times_sec: np.ndarray


@dataclass(frozen=True)
class _ObservationTopology:
    points_world: np.ndarray
    gaussian_indices: np.ndarray
    obs_point_index: np.ndarray
    obs_view_index: np.ndarray
    obs_pixels_xy: np.ndarray
    obs_J: np.ndarray
    obs_contribution_weight: np.ndarray
    view_ids: tuple[str, ...]
    view_image_width: np.ndarray
    view_image_height: np.ndarray


@dataclass(frozen=True)
class _ManifestData:
    path: Path
    mode_indices: np.ndarray
    frequencies_hz: np.ndarray
    phi: np.ndarray
    topology: _ObservationTopology


def _scalar_string(value: np.ndarray, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{path} {name} must be a scalar string")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str) or not item:
        raise ValueError(f"{path} {name} must be a non-empty scalar string")
    return item


def _string_vector(value: np.ndarray, name: str, path: Path) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{path} {name} must be a 1-D string array")
    result: list[str] = []
    for raw in array:
        item = np.asarray(raw).item()
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if not isinstance(item, str) or not item:
            raise ValueError(f"{path} {name} must contain non-empty strings")
        result.append(item)
    return tuple(result)


def _scalar_int(value: np.ndarray, name: str, path: Path) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{path} {name} must be an integer scalar")
    return int(array.item())


def _scalar_float(value: np.ndarray, name: str, path: Path) -> float:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{path} {name} must be a real scalar")
    result = float(array.item())
    if not np.isfinite(result):
        raise ValueError(f"{path} {name} must be finite")
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _resolve_artifact_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{root} is missing non-empty {name}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _load_frame_map(path_value: str | Path) -> _FrameMapData:
    path = Path(path_value).expanduser()
    payload = _read_json_object(path)
    version = payload.get("version")
    if isinstance(version, bool) or version != 1:
        raise ValueError(f"{path} must use modal frame map version 1")

    raw_view_ids = payload.get("views")
    raw_fps = payload.get("view_fps_hz")
    raw_frames = payload.get("frames")
    if not isinstance(raw_view_ids, list) or not raw_view_ids:
        raise ValueError(f"{path} must contain a non-empty views list")
    if any(not isinstance(value, str) or not value for value in raw_view_ids):
        raise ValueError(f"{path} views must contain non-empty strings")
    view_ids = tuple(raw_view_ids)
    if len(set(view_ids)) != len(view_ids):
        raise ValueError(f"{path} views must be unique")
    if not isinstance(raw_fps, dict) or set(raw_fps) != set(view_ids):
        raise ValueError(f"{path} view_fps_hz keys must exactly match views")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError(f"{path} must contain a non-empty frames list")

    fps_values: list[float] = []
    for view_id in view_ids:
        value = raw_fps[view_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"FPS for view {view_id!r} must be numeric")
        fps = float(value)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"FPS for view {view_id!r} must be finite and positive")
        fps_values.append(fps)

    view_to_index = {view_id: index for index, view_id in enumerate(view_ids)}
    frame_names: list[str] = []
    frame_view_indices: list[int] = []
    frame_local_indices: list[int] = []
    frame_times_sec: list[float] = []
    local_indices_by_view: list[set[int]] = [set() for _ in view_ids]
    seen_names: set[str] = set()
    for record in raw_frames:
        if not isinstance(record, dict):
            raise ValueError(f"{path} frame records must be objects")
        frame_name = record.get("frame_name")
        view_id = record.get("view_id")
        local_index_value = record.get("local_index")
        time_sec_value = record.get("time_sec")
        if not isinstance(frame_name, str) or not frame_name:
            raise ValueError(f"{path} frame_name values must be non-empty strings")
        if frame_name in seen_names:
            raise ValueError(f"{path} contains duplicate frame_name {frame_name!r}")
        if not isinstance(view_id, str) or not view_id:
            raise ValueError(f"{path} view_id for {frame_name!r} must be a non-empty string")
        if view_id not in view_to_index:
            raise ValueError(f"{path} frame {frame_name!r} has unknown view_id {view_id!r}")
        if isinstance(local_index_value, bool) or not isinstance(local_index_value, int):
            raise ValueError(f"{path} local_index for {frame_name!r} must be an integer")
        local_index = int(local_index_value)
        if local_index < 0:
            raise ValueError(f"{path} local_index for {frame_name!r} must be non-negative")
        if isinstance(time_sec_value, bool) or not isinstance(time_sec_value, (int, float)):
            raise ValueError(f"{path} time_sec for {frame_name!r} must be numeric")
        time_sec = float(time_sec_value)
        view_index = view_to_index[view_id]
        expected_time_sec = local_index / fps_values[view_index]
        if not np.isfinite(time_sec) or time_sec < 0.0:
            raise ValueError(f"{path} time_sec for {frame_name!r} must be finite and non-negative")
        if abs(time_sec - expected_time_sec) > 1e-9:
            raise ValueError(
                f"{path} time_sec={time_sec:.12g} for {frame_name!r} does not match "
                f"local_index/fps={expected_time_sec:.12g}"
            )
        if local_index in local_indices_by_view[view_index]:
            raise ValueError(
                f"{path} contains duplicate local_index={local_index} for view {view_id!r}"
            )
        seen_names.add(frame_name)
        local_indices_by_view[view_index].add(local_index)
        frame_names.append(frame_name)
        frame_view_indices.append(view_index)
        frame_local_indices.append(local_index)
        frame_times_sec.append(time_sec)

    for view_index, view_id in enumerate(view_ids):
        local_indices = local_indices_by_view[view_index]
        if not local_indices:
            raise ValueError(f"{path} view {view_id!r} has no frames")
        expected = set(range(len(local_indices)))
        if local_indices != expected:
            raise ValueError(
                f"{path} local_index values for view {view_id!r} must be contiguous "
                f"from 0 to {len(local_indices) - 1}"
            )

    return _FrameMapData(
        path=path.resolve(),
        view_ids=view_ids,
        fps_hz=np.asarray(fps_values, dtype=np.float64),
        frame_names=tuple(frame_names),
        frame_view_indices=np.asarray(frame_view_indices, dtype=np.int64),
        frame_local_indices=np.asarray(frame_local_indices, dtype=np.int64),
        frame_times_sec=np.asarray(frame_times_sec, dtype=np.float64),
    )


def parse_flow_cache_specs(specs: Sequence[str]) -> tuple[tuple[str, Path], ...]:
    """Parse ordered ``VIEW_ID=PATH`` cache specifications without reordering."""

    if not specs:
        raise ValueError("At least one --flow-caches VIEW_ID=PATH value is required")
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, str) or spec.count("=") != 1:
            raise ValueError(f"Flow cache specification must be VIEW_ID=PATH, got {spec!r}")
        view_id, raw_path = spec.split("=", 1)
        if not view_id or not raw_path:
            raise ValueError(f"Flow cache specification must be VIEW_ID=PATH, got {spec!r}")
        if view_id in seen:
            raise ValueError(f"Duplicate flow cache view_id {view_id!r}")
        seen.add(view_id)
        parsed.append((view_id, Path(raw_path).expanduser()))
    return tuple(parsed)


def _require_npz_fields(archive: np.lib.npyio.NpzFile, required: set[str], path: Path) -> None:
    missing = sorted(required - set(archive.files))
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")


def _load_latent_mode(
    path: Path,
    manifest_path: Path,
    source_checkpoint: str,
    mode_index: int,
    frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "points_world",
            "phi",
            "gaussian_indices",
            "freq_hz",
            "mode_index",
            "point_type",
            "source_checkpoint",
        }
        _require_npz_fields(archive, required, path)
        point_type = _scalar_string(archive["point_type"], "point_type", path)
        latent_source = _scalar_string(archive["source_checkpoint"], "source_checkpoint", path)
        latent_mode_index = _scalar_int(archive["mode_index"], "mode_index", path)
        latent_frequency = _scalar_float(archive["freq_hz"], "freq_hz", path)
        points_world = np.asarray(archive["points_world"], dtype=np.float32)
        phi = np.asarray(archive["phi"])
        gaussian_indices = np.asarray(archive["gaussian_indices"])

    if point_type != "foreground_gaussian_center":
        raise ValueError(f"{path} point_type must be 'foreground_gaussian_center'")
    if latent_source != source_checkpoint:
        raise ValueError(f"{path} source_checkpoint does not match {manifest_path}")
    if latent_mode_index != mode_index:
        raise ValueError(f"{path} mode_index does not match {manifest_path}")
    if not np.isclose(latent_frequency, frequency_hz, rtol=1e-6, atol=1e-6):
        raise ValueError(f"{path} freq_hz does not match {manifest_path}")
    if (
        points_world.ndim != 2
        or points_world.shape[0] == 0
        or points_world.shape[1] != 3
        or not np.isfinite(points_world).all()
    ):
        raise ValueError(f"{path} points_world must be finite with shape [G,3]")
    if phi.shape != points_world.shape or not np.iscomplexobj(phi):
        raise ValueError(f"{path} phi must be complex with shape {points_world.shape}")
    if not np.isfinite(phi.real).all() or not np.isfinite(phi.imag).all():
        raise ValueError(f"{path} phi must be finite")
    if gaussian_indices.shape != (points_world.shape[0],) or not np.issubdtype(
        gaussian_indices.dtype, np.integer
    ):
        raise ValueError(f"{path} gaussian_indices must be an integer [G] array")
    if not np.array_equal(gaussian_indices, np.arange(points_world.shape[0])):
        raise ValueError(f"{path} gaussian_indices must be contiguous foreground indices")
    return points_world, phi.astype(np.complex64), gaussian_indices.astype(np.int64)


def _load_observation_topology(
    path: Path,
    manifest_path: Path,
    source_checkpoint: str | None,
    mode_index: int | None,
    frequency_hz: float | None,
) -> _ObservationTopology:
    split_format = mode_index is None
    if split_format:
        arrays = load_gaussian_observation_topology(path)
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}

    required = {
        "points_world",
        "gaussian_indices",
        "point_type",
        "source_checkpoint",
        "obs_point_index",
        "obs_J",
        "obs_contribution_weight",
        "view_ids",
        "view_image_width",
        "view_image_height",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    point_type = _scalar_string(arrays["point_type"], "point_type", path)
    observation_source = _scalar_string(
        arrays["source_checkpoint"], "source_checkpoint", path
    )
    points_world = np.asarray(arrays["points_world"], dtype=np.float32)
    gaussian_indices = np.asarray(arrays["gaussian_indices"])
    obs_point_index = np.asarray(arrays["obs_point_index"])
    if split_format:
        obs_sample_index = np.asarray(arrays["obs_sample_index"], dtype=np.int64)
        obs_view_index = np.asarray(arrays["sample_view_index"])[obs_sample_index]
        obs_pixels_xy = np.asarray(arrays["sample_pixels_xy"])[obs_sample_index]
    else:
        obs_view_index = np.asarray(arrays["obs_view_index"])
        obs_pixels_xy = np.asarray(arrays["obs_pixels_xy"])
    obs_j = np.asarray(arrays["obs_J"])
    obs_weight = np.asarray(arrays["obs_contribution_weight"])
    view_ids = _string_vector(arrays["view_ids"], "view_ids", path)
    view_width = np.asarray(arrays["view_image_width"])
    view_height = np.asarray(arrays["view_image_height"])

    if not split_format:
        assert mode_index is not None
        assert frequency_hz is not None
        required = {
            "freq_hz",
            "mode_index",
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        observation_mode_index = _scalar_int(
            arrays["mode_index"], "mode_index", path
        )
        observation_frequency = _scalar_float(
            arrays["freq_hz"], "freq_hz", path
        )

    if point_type != "foreground_gaussian_center":
        raise ValueError(f"{path} point_type must be 'foreground_gaussian_center'")
    if source_checkpoint is not None and observation_source != source_checkpoint:
        raise ValueError(f"{path} source_checkpoint does not match {manifest_path}")
    if not split_format and observation_mode_index != mode_index:
        raise ValueError(f"{path} mode_index does not match {manifest_path}")
    if not split_format and not np.isclose(
        observation_frequency, frequency_hz, rtol=1e-6, atol=1e-6
    ):
        raise ValueError(f"{path} freq_hz does not match {manifest_path}")
    if (
        points_world.ndim != 2
        or points_world.shape[0] == 0
        or points_world.shape[1] != 3
        or not np.isfinite(points_world).all()
    ):
        raise ValueError(f"{path} points_world must be finite with shape [G,3]")
    num_points = points_world.shape[0]
    if gaussian_indices.shape != (num_points,) or not np.issubdtype(
        gaussian_indices.dtype, np.integer
    ):
        raise ValueError(f"{path} gaussian_indices must be an integer [G] array")
    if not np.array_equal(gaussian_indices, np.arange(num_points)):
        raise ValueError(f"{path} gaussian_indices must be contiguous foreground indices")
    num_rows = obs_point_index.size
    if obs_point_index.shape != (num_rows,) or not np.issubdtype(obs_point_index.dtype, np.integer):
        raise ValueError(f"{path} obs_point_index must be an integer [O] array")
    if obs_view_index.shape != (num_rows,) or not np.issubdtype(obs_view_index.dtype, np.integer):
        raise ValueError(f"{path} obs_view_index must be an integer [O] array")
    if obs_pixels_xy.shape != (num_rows, 2):
        raise ValueError(f"{path} obs_pixels_xy must have shape [O,2]")
    if obs_j.shape != (num_rows, 2, 3):
        raise ValueError(f"{path} obs_J must have shape [O,2,3]")
    if obs_weight.shape != (num_rows,):
        raise ValueError(f"{path} obs_contribution_weight must have shape [O]")
    if num_rows == 0:
        raise ValueError(f"{path} contains no observation rows")
    if np.any(obs_point_index < 0) or np.any(obs_point_index >= num_points):
        raise ValueError(f"{path} obs_point_index contains out-of-range values")
    if len(set(view_ids)) != len(view_ids) or not view_ids:
        raise ValueError(f"{path} view_ids must be non-empty and unique")
    num_views = len(view_ids)
    if np.any(obs_view_index < 0) or np.any(obs_view_index >= num_views):
        raise ValueError(f"{path} obs_view_index contains out-of-range values")
    if view_width.shape != (num_views,) or view_height.shape != (num_views,):
        raise ValueError(f"{path} view image dimensions must have shape [V]")
    if not np.issubdtype(view_width.dtype, np.integer) or not np.issubdtype(
        view_height.dtype, np.integer
    ):
        raise ValueError(f"{path} view image dimensions must be integer-valued")
    if np.any(view_width <= 0) or np.any(view_height <= 0):
        raise ValueError(f"{path} view image dimensions must be positive")
    for name, values in (
        ("obs_pixels_xy", obs_pixels_xy),
        ("obs_J", obs_j),
        ("obs_contribution_weight", obs_weight),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{path} {name} must be finite")
    if np.any(obs_weight < 0.0):
        raise ValueError(f"{path} obs_contribution_weight must be non-negative")

    rounded_pixels = np.rint(obs_pixels_xy)
    if np.max(np.abs(obs_pixels_xy - rounded_pixels)) > 1e-6:
        raise ValueError(f"{path} obs_pixels_xy must contain integer pixel coordinates")
    pixels = rounded_pixels.astype(np.int64)
    row_width = view_width[obs_view_index.astype(np.int64)]
    row_height = view_height[obs_view_index.astype(np.int64)]
    if (
        np.any(pixels[:, 0] < 0)
        or np.any(pixels[:, 0] >= row_width)
        or np.any(pixels[:, 1] < 0)
        or np.any(pixels[:, 1] >= row_height)
    ):
        raise ValueError(f"{path} obs_pixels_xy contains out-of-bounds pixels")

    return _ObservationTopology(
        points_world=points_world,
        gaussian_indices=gaussian_indices.astype(np.int64),
        obs_point_index=obs_point_index.astype(np.int64),
        obs_view_index=obs_view_index.astype(np.int64),
        obs_pixels_xy=pixels,
        obs_J=obs_j.astype(np.float64),
        obs_contribution_weight=obs_weight.astype(np.float64),
        view_ids=view_ids,
        view_image_width=view_width.astype(np.int64),
        view_image_height=view_height.astype(np.int64),
    )


def load_flow_observation_topology(
    path_value: str | Path,
) -> _ObservationTopology:
    """Load standalone split observation topology for flow-space analysis."""

    path = Path(path_value).expanduser().resolve(strict=True)
    return _load_observation_topology(path, path, None, None, None)


def _same_topology(current: _ObservationTopology, expected: _ObservationTopology, path: Path) -> None:
    fields = (
        "points_world",
        "gaussian_indices",
        "obs_point_index",
        "obs_view_index",
        "obs_pixels_xy",
        "obs_J",
        "obs_contribution_weight",
        "view_image_width",
        "view_image_height",
    )
    for field in fields:
        if not np.array_equal(getattr(current, field), getattr(expected, field)):
            raise ValueError(f"{path} {field} differs from the first mode observation topology")
    if current.view_ids != expected.view_ids:
        raise ValueError(f"{path} view_ids differ from the first mode observation topology")


def _load_manifest(path_value: str | Path) -> _ManifestData:
    path = Path(path_value).expanduser()
    payload = _read_json_object(path)
    version = payload.get("version")
    if isinstance(version, bool) or version != 1:
        raise ValueError(f"{path} must be a version 1 modal manifest")
    if payload.get("point_type") != "foreground_gaussian_center":
        raise ValueError(f"{path} point_type must be 'foreground_gaussian_center'")
    source_checkpoint = payload.get("source_checkpoint")
    if not isinstance(source_checkpoint, str) or not source_checkpoint:
        raise ValueError(f"{path} is missing source_checkpoint")
    raw_modes = payload.get("modes")
    if not isinstance(raw_modes, list) or not raw_modes:
        raise ValueError(f"{path} must contain a non-empty modes list")

    mode_indices: list[int] = []
    frequencies: list[float] = []
    phi_values: list[np.ndarray] = []
    expected_points: np.ndarray | None = None
    expected_indices: np.ndarray | None = None
    expected_topology: _ObservationTopology | None = None
    shared_topology_path: Path | None = None
    if "observation_topology_path" in payload:
        shared_topology_path = _resolve_artifact_path(
            path,
            payload.get("observation_topology_path"),
            "observation_topology_path",
        )
        expected_topology = _load_observation_topology(
            shared_topology_path,
            path,
            source_checkpoint,
            None,
            None,
        )
    for slot, raw_mode in enumerate(raw_modes):
        if not isinstance(raw_mode, dict):
            raise ValueError(f"{path} mode entries must be objects")
        mode_index_value = raw_mode.get("mode_index")
        frequency_value = raw_mode.get("freq_hz")
        if isinstance(mode_index_value, bool) or not isinstance(mode_index_value, int):
            raise ValueError(f"{path} mode slot {slot} has invalid mode_index")
        if isinstance(frequency_value, bool) or not isinstance(frequency_value, (int, float)):
            raise ValueError(f"{path} mode slot {slot} has invalid freq_hz")
        mode_index = int(mode_index_value)
        frequency_hz = float(frequency_value)
        if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError(f"{path} mode slot {slot} freq_hz must be finite and positive")
        if mode_index in mode_indices:
            raise ValueError(f"{path} contains duplicate mode_index {mode_index}")

        alpha_by_view = raw_mode.get("alpha_by_view")
        if not isinstance(alpha_by_view, list) or not alpha_by_view:
            raise ValueError(f"{path} mode {mode_index} is missing alpha_by_view diagnostics")
        alpha_view_ids: list[str] = []
        for alpha in alpha_by_view:
            if not isinstance(alpha, dict):
                raise ValueError(f"{path} mode {mode_index} alpha_by_view entries must be objects")
            alpha_view_id = alpha.get("view_id")
            if not isinstance(alpha_view_id, str) or not alpha_view_id:
                raise ValueError(f"{path} mode {mode_index} alpha view_id must be non-empty")
            if alpha.get("identifiable") is not True:
                raise ValueError(
                    f"{path} mode {mode_index} alpha for view {alpha_view_id!r} is not identifiable"
                )
            for field in ("real", "imag"):
                value = alpha.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(
                        f"{path} mode {mode_index} alpha {field} for view {alpha_view_id!r} must be finite"
                    )
            alpha_view_ids.append(alpha_view_id)
        if len(set(alpha_view_ids)) != len(alpha_view_ids):
            raise ValueError(f"{path} mode {mode_index} alpha_by_view contains duplicate views")

        latent_path = _resolve_artifact_path(path, raw_mode.get("latent_path"), "latent_path")
        points, phi, gaussian_indices = _load_latent_mode(
            latent_path,
            path,
            source_checkpoint,
            mode_index,
            frequency_hz,
        )
        if shared_topology_path is None:
            observation_path = _resolve_artifact_path(
                path, raw_mode.get("observation_path"), "observation_path"
            )
            topology = _load_observation_topology(
                observation_path,
                path,
                source_checkpoint,
                mode_index,
                frequency_hz,
            )
        else:
            observation_path = shared_topology_path
            assert expected_topology is not None
            topology = expected_topology
        if alpha_view_ids != list(topology.view_ids):
            raise ValueError(
                f"{path} mode {mode_index} alpha_by_view order must match observation view_ids"
            )
        if points.shape != topology.points_world.shape or np.max(
            np.abs(points - topology.points_world)
        ) > 1e-5:
            raise ValueError(f"{latent_path} points_world does not match {observation_path}")
        if not np.array_equal(gaussian_indices, topology.gaussian_indices):
            raise ValueError(f"{latent_path} gaussian_indices does not match {observation_path}")

        if expected_points is None:
            expected_points = points
            expected_indices = gaussian_indices
            if expected_topology is None:
                expected_topology = topology
        else:
            assert expected_indices is not None
            assert expected_topology is not None
            if not np.array_equal(points, expected_points):
                raise ValueError(f"{latent_path} points_world differs from the first mode latent")
            if not np.array_equal(gaussian_indices, expected_indices):
                raise ValueError(f"{latent_path} gaussian_indices differs from the first mode latent")
            if shared_topology_path is None:
                _same_topology(topology, expected_topology, observation_path)

        mode_indices.append(mode_index)
        frequencies.append(frequency_hz)
        phi_values.append(phi)

    assert expected_topology is not None
    manifest_mode_indices = payload.get("mode_indices")
    if manifest_mode_indices is not None and manifest_mode_indices != mode_indices:
        raise ValueError(f"{path} mode_indices does not match modes order")
    return _ManifestData(
        path=path.resolve(),
        mode_indices=np.asarray(mode_indices, dtype=np.int64),
        frequencies_hz=np.asarray(frequencies, dtype=np.float64),
        phi=np.stack(phi_values, axis=0),
        topology=expected_topology,
    )


def _load_and_validate_caches(
    frame_map: _FrameMapData,
    topology: _ObservationTopology,
    cache_specs: Sequence[str],
) -> tuple[ModalAnalysisCache, ...]:
    parsed = parse_flow_cache_specs(cache_specs)
    parsed_view_ids = tuple(view_id for view_id, _ in parsed)
    if parsed_view_ids != frame_map.view_ids:
        raise ValueError(
            "--flow-caches view IDs and order must exactly match modal frame map views: "
            f"expected {frame_map.view_ids}, got {parsed_view_ids}"
        )
    if topology.view_ids != frame_map.view_ids:
        raise ValueError(
            f"Observation view_ids {topology.view_ids} do not match frame map views {frame_map.view_ids}"
        )

    caches: list[ModalAnalysisCache] = []
    seen_cache_paths: set[Path] = set()
    for view_index, (view_id, cache_path) in enumerate(parsed):
        cache = load_analysis_cache(cache_path)
        resolved_cache_path = cache.path.resolve()
        if resolved_cache_path in seen_cache_paths:
            raise ValueError(f"Flow cache directory is reused by multiple views: {resolved_cache_path}")
        seen_cache_paths.add(resolved_cache_path)
        flow_method = cache.metadata["analysis"]["flow_method"]
        if flow_method != "farneback":
            raise ValueError(
                f"Flow cache for {view_id!r} uses {flow_method!r}; expected 'farneback'"
            )
        frame_range = cache.metadata["video"]["frame_range"]
        t0_s = float(frame_range["t0_s"])
        if abs(t0_s) > 1e-12:
            raise ValueError(f"Flow cache for {view_id!r} must start at t0_s=0, got {t0_s}")
        expected_fps = float(frame_map.fps_hz[view_index])
        if not np.isclose(cache.fps, expected_fps, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Flow cache FPS for {view_id!r} is {cache.fps}, expected {expected_fps}"
            )
        frame_rows = np.flatnonzero(frame_map.frame_view_indices == view_index)
        if cache.flow_u.shape[0] != frame_rows.size:
            raise ValueError(
                f"Flow cache for {view_id!r} has {cache.flow_u.shape[0]} frames, "
                f"expected {frame_rows.size}"
            )
        expected_shape = (
            int(topology.view_image_height[view_index]),
            int(topology.view_image_width[view_index]),
        )
        if cache.flow_u.shape[1:] != expected_shape:
            raise ValueError(
                f"Flow cache for {view_id!r} has spatial shape {cache.flow_u.shape[1:]}, "
                f"expected {expected_shape}"
            )
        if cache.mask is None:
            raise ValueError(f"Flow cache for {view_id!r} must contain a foreground mask")
        reference_index = int(cache.metadata["analysis"]["reference_frame_index"])
        expected_reference_index = cache.flow_u.shape[0] // 2
        if reference_index != expected_reference_index:
            raise ValueError(
                f"Flow cache reference index for {view_id!r} is {reference_index}, "
                f"expected middle frame {expected_reference_index}"
            )
        expected_reference_time = reference_index / expected_fps
        if abs(cache.t_ref_s - expected_reference_time) > 1e-9:
            raise ValueError(
                f"Flow cache reference time for {view_id!r} does not match reference index/FPS"
            )
        caches.append(cache)
    return tuple(caches)


def _view_pixel_groups(
    topology: _ObservationTopology,
    view_index: int,
    cache: ModalAnalysisCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.flatnonzero(topology.obs_view_index == view_index)
    if rows.size == 0:
        raise ValueError(f"Observation topology has no rows for view {topology.view_ids[view_index]!r}")
    pixels = topology.obs_pixels_xy[rows]
    width = int(topology.view_image_width[view_index])
    pixel_ids = pixels[:, 1] * width + pixels[:, 0]
    order = np.argsort(pixel_ids, kind="stable")
    sorted_rows = rows[order]
    sorted_pixel_ids = pixel_ids[order]
    starts = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.flatnonzero(np.diff(sorted_pixel_ids)) + 1]
    )
    unique_pixels = topology.obs_pixels_xy[sorted_rows[starts]]
    weights = topology.obs_contribution_weight[sorted_rows]
    weight_sums = np.add.reduceat(weights, starts)
    weight_error = np.abs(weight_sums - 1.0)
    max_error = float(np.max(weight_error))
    if max_error > 1e-5:
        pixel_index = int(np.argmax(weight_error))
        x, y = unique_pixels[pixel_index].tolist()
        raise ValueError(
            f"Candidate weights for view {topology.view_ids[view_index]!r} pixel "
            f"({x},{y}) sum to {weight_sums[pixel_index]:.9g}; max allowed error is 1e-5"
        )
    mask = cache.mask
    assert mask is not None
    if not bool(np.all(mask[unique_pixels[:, 1], unique_pixels[:, 0]])):
        raise ValueError(
            f"Observation pixels for view {topology.view_ids[view_index]!r} fall outside cache mask"
        )
    return sorted_rows, starts, unique_pixels, weights, weight_sums


def _build_design_matrix(
    manifest: _ManifestData,
    sorted_rows: np.ndarray,
    starts: np.ndarray,
    sorted_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    topology = manifest.topology
    num_pixels = starts.size
    num_modes = manifest.mode_indices.size
    design = np.empty((2 * num_pixels, 2 * num_modes), dtype=np.float64)
    point_indices = topology.obs_point_index[sorted_rows]
    jacobians = topology.obs_J[sorted_rows]
    for mode_slot in range(num_modes):
        point_phi = manifest.phi[mode_slot, point_indices].astype(np.complex128)
        projected = np.einsum("oij,oj->oi", jacobians, point_phi, optimize=True)
        projected *= sorted_weights[:, None]
        per_pixel = np.add.reduceat(projected, starts, axis=0)
        design[0::2, 2 * mode_slot] = per_pixel[:, 0].real
        design[1::2, 2 * mode_slot] = per_pixel[:, 1].real
        design[0::2, 2 * mode_slot + 1] = -per_pixel[:, 0].imag
        design[1::2, 2 * mode_slot + 1] = -per_pixel[:, 1].imag

    if not np.isfinite(design).all():
        raise ValueError("Projected modal design matrix contains non-finite values")
    pair_scales = np.empty((num_modes,), dtype=np.float64)
    denominator = float(2 * num_pixels)
    for mode_slot in range(num_modes):
        pair = design[:, 2 * mode_slot : 2 * mode_slot + 2]
        pair_scales[mode_slot] = np.sqrt(float(np.sum(pair * pair)) / denominator)
    if not np.isfinite(pair_scales).all():
        invalid = np.flatnonzero(~np.isfinite(pair_scales))
        raise ValueError(
            "Projected modal design has non-finite mode-pair scale at slots "
            f"{invalid.tolist()}"
        )
    pair_scales[pair_scales <= np.finfo(np.float64).eps] = 1.0
    return design, pair_scales


def _flow_matrix(
    cache: ModalAnalysisCache,
    pixels: np.ndarray,
    start: int,
    end: int,
    reference_u: np.ndarray,
    reference_v: np.ndarray,
) -> np.ndarray:
    x = pixels[:, 0]
    y = pixels[:, 1]
    flow_u = np.asarray(cache.flow_u[start:end, y, x], dtype=np.float64)
    flow_v = np.asarray(cache.flow_v[start:end, y, x], dtype=np.float64)
    if not np.isfinite(flow_u).all() or not np.isfinite(flow_v).all():
        raise ValueError(
            f"Flow cache {cache.path} contains non-finite values on observation pixels "
            f"for frames [{start}, {end})"
        )
    flow_u -= reference_u[None, :]
    flow_v -= reference_v[None, :]
    matrix = np.empty((2 * pixels.shape[0], end - start), dtype=np.float64)
    matrix[0::2] = flow_u.T
    matrix[1::2] = flow_v.T
    return matrix


def _coordinate_spectral_diagnostics(
    coordinates: np.ndarray,
    fps_hz: float,
    assigned_frequencies_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames, num_modes = coordinates.shape
    spectrum = np.fft.fft(coordinates, axis=0)
    frequencies = np.fft.fftfreq(num_frames, d=1.0 / fps_hz)
    energy = np.abs(spectrum) ** 2
    if num_frames:
        energy[0] = 0.0
    dominant = np.zeros((num_modes,), dtype=np.float64)
    assigned_ratio = np.zeros((num_modes,), dtype=np.float64)
    for mode_slot, assigned_frequency in enumerate(assigned_frequencies_hz):
        total = float(np.sum(energy[:, mode_slot]))
        if total <= np.finfo(np.float64).eps:
            continue
        dominant[mode_slot] = float(frequencies[int(np.argmax(energy[:, mode_slot]))])
        positive = int(np.argmin(np.abs(frequencies - assigned_frequency)))
        negative = int(np.argmin(np.abs(frequencies + assigned_frequency)))
        selected = np.unique(np.asarray([positive, negative], dtype=np.int64))
        assigned_ratio[mode_slot] = float(np.sum(energy[selected, mode_slot]) / total)

    norms = np.sqrt(np.sum(np.abs(coordinates) ** 2, axis=0))
    correlation = np.zeros((num_modes, num_modes), dtype=np.float64)
    gram = coordinates.conj().T @ coordinates
    denominator = norms[:, None] * norms[None, :]
    valid = denominator > np.finfo(np.float64).eps
    correlation[valid] = np.abs(gram[valid]) / denominator[valid]
    return dominant, assigned_ratio, correlation


def _solve_view(
    manifest: _ManifestData,
    frame_map: _FrameMapData,
    cache: ModalAnalysisCache,
    view_index: int,
    ridge_relative: float,
    frame_chunk_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    solve_start = time.perf_counter()
    topology = manifest.topology
    sorted_rows, starts, pixels, sorted_weights, weight_sums = _view_pixel_groups(
        topology, view_index, cache
    )
    design, pair_scales = _build_design_matrix(
        manifest, sorted_rows, starts, sorted_weights
    )
    num_pixels = pixels.shape[0]
    num_modes = manifest.mode_indices.size
    normalized = design.copy()
    for mode_slot, scale in enumerate(pair_scales):
        normalized[:, 2 * mode_slot : 2 * mode_slot + 2] /= scale
    normalizer = float(2 * num_pixels)
    gram = normalized.T @ normalized / normalizer
    system = gram + ridge_relative * np.eye(2 * num_modes, dtype=np.float64)
    cholesky = np.linalg.cholesky(system)

    eigenvalues = np.linalg.eigvalsh(gram)
    scaled_singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))[::-1]
    tolerance = (
        np.finfo(np.float64).eps
        * max(normalized.shape)
        * float(scaled_singular_values[0])
    )
    numerical_rank = int(np.count_nonzero(scaled_singular_values > tolerance))
    condition_number = (
        float(scaled_singular_values[0] / scaled_singular_values[-1])
        if numerical_rank == 2 * num_modes and scaled_singular_values[-1] > 0.0
        else float("inf")
    )
    ridge_condition = float(
        (float(eigenvalues[-1]) + ridge_relative)
        / (max(float(eigenvalues[0]), 0.0) + ridge_relative)
    )

    reference_index = int(cache.metadata["analysis"]["reference_frame_index"])
    x = pixels[:, 0]
    y = pixels[:, 1]
    reference_u = np.asarray(cache.flow_u[reference_index, y, x], dtype=np.float64)
    reference_v = np.asarray(cache.flow_v[reference_index, y, x], dtype=np.float64)
    if not np.isfinite(reference_u).all() or not np.isfinite(reference_v).all():
        raise ValueError(f"Flow cache {cache.path} reference flow is non-finite on observation pixels")

    num_frames = cache.flow_u.shape[0]
    relative_coordinates = np.empty((num_frames, num_modes), dtype=np.complex128)
    for start in range(0, num_frames, frame_chunk_size):
        end = min(start + frame_chunk_size, num_frames)
        flow = _flow_matrix(cache, pixels, start, end, reference_u, reference_v)
        rhs = normalized.T @ flow / normalizer
        intermediate = solve_triangular(
            cholesky,
            rhs,
            lower=True,
            check_finite=False,
        )
        scaled_solution = solve_triangular(
            cholesky.T,
            intermediate,
            lower=False,
            check_finite=False,
        )
        solution = scaled_solution.copy()
        for mode_slot, scale in enumerate(pair_scales):
            solution[2 * mode_slot : 2 * mode_slot + 2] /= scale
        relative_coordinates[start:end] = (
            solution[0::2].T + 1j * solution[1::2].T
        )
    if not np.isfinite(relative_coordinates.real).all() or not np.isfinite(
        relative_coordinates.imag
    ).all():
        raise ValueError(f"Coordinate solve for view {frame_map.view_ids[view_index]!r} is non-finite")

    relative_coordinates -= relative_coordinates[reference_index : reference_index + 1]
    coordinates = relative_coordinates - np.mean(relative_coordinates, axis=0, keepdims=True)
    coordinates = coordinates.astype(np.complex64).astype(np.complex128)

    per_frame_rmse = np.empty((num_frames,), dtype=np.float64)
    per_frame_relative = np.empty((num_frames,), dtype=np.float64)
    per_frame_r2 = np.empty((num_frames,), dtype=np.float64)
    total_residual_sq = 0.0
    total_flow_sq = 0.0
    reference_residual_norm = 0.0
    reference_coordinates = coordinates[reference_index]
    for start in range(0, num_frames, frame_chunk_size):
        end = min(start + frame_chunk_size, num_frames)
        flow = _flow_matrix(cache, pixels, start, end, reference_u, reference_v)
        relative = coordinates[start:end] - reference_coordinates[None, :]
        packed = np.empty((2 * num_modes, end - start), dtype=np.float64)
        packed[0::2] = relative.real.T
        packed[1::2] = relative.imag.T
        residual = design @ packed - flow
        residual_sq = np.sum(residual * residual, axis=0)
        flow_sq = np.sum(flow * flow, axis=0)
        chunk_rmse = np.sqrt(residual_sq / normalizer)
        chunk_relative = np.zeros((end - start,), dtype=np.float64)
        chunk_r2 = np.ones((end - start,), dtype=np.float64)
        positive_energy = flow_sq > np.finfo(np.float64).eps
        chunk_relative[positive_energy] = np.sqrt(
            residual_sq[positive_energy] / flow_sq[positive_energy]
        )
        chunk_r2[positive_energy] = (
            1.0 - residual_sq[positive_energy] / flow_sq[positive_energy]
        )
        zero_energy_bad = (~positive_energy) & (residual_sq > np.finfo(np.float64).eps)
        if np.any(zero_energy_bad):
            raise ValueError(
                f"View {frame_map.view_ids[view_index]!r} has non-zero prediction on zero reference-relative flow"
            )
        per_frame_rmse[start:end] = chunk_rmse
        per_frame_relative[start:end] = chunk_relative
        per_frame_r2[start:end] = chunk_r2
        if start <= reference_index < end:
            reference_residual_norm = float(
                np.sqrt(residual_sq[reference_index - start])
            )
        total_residual_sq += float(np.sum(residual_sq))
        total_flow_sq += float(np.sum(flow_sq))

    view_rmse = float(np.sqrt(total_residual_sq / (normalizer * num_frames)))
    if total_flow_sq <= np.finfo(np.float64).eps:
        view_relative = 0.0
        view_r2 = 1.0
    else:
        view_relative = float(np.sqrt(total_residual_sq / total_flow_sq))
        view_r2 = float(1.0 - total_residual_sq / total_flow_sq)
    dominant, assigned_ratio, correlation = _coordinate_spectral_diagnostics(
        coordinates,
        float(frame_map.fps_hz[view_index]),
        manifest.frequencies_hz,
    )
    coordinate_abs = np.abs(coordinates)
    diagnostics = {
        "candidate_row_count": int(sorted_rows.size),
        "unique_pixel_count": int(num_pixels),
        "candidate_weight_sum_min": float(np.min(weight_sums)),
        "candidate_weight_sum_max": float(np.max(weight_sums)),
        "candidate_weight_sum_max_abs_error": float(np.max(np.abs(weight_sums - 1.0))),
        "mode_pair_scales": pair_scales,
        "singular_values": scaled_singular_values,
        "numerical_rank": numerical_rank,
        "condition_number": condition_number,
        "ridge_condition_number": ridge_condition,
        "reference_local_index": reference_index,
        "per_frame_flow_rmse": per_frame_rmse,
        "per_frame_relative_residual": per_frame_relative,
        "per_frame_flow_r2": per_frame_r2,
        "flow_rmse": view_rmse,
        "relative_residual": view_relative,
        "flow_r2": view_r2,
        "reference_residual_norm": reference_residual_norm,
        "residual_sum_squares": total_residual_sq,
        "flow_sum_squares": total_flow_sq,
        "coordinate_rms": np.sqrt(np.mean(coordinate_abs * coordinate_abs, axis=0)),
        "coordinate_max": np.max(coordinate_abs, axis=0),
        "coordinate_temporal_mean_abs": np.abs(np.mean(coordinates, axis=0)),
        "dominant_signed_frequency_hz": dominant,
        "assigned_frequency_energy_ratio": assigned_ratio,
        "cross_mode_correlation": correlation,
        "solve_seconds": float(time.perf_counter() - solve_start),
    }
    return coordinates.astype(np.complex64), diagnostics


def _json_float(value: float) -> float | None:
    result = float(value)
    if not np.isfinite(result):
        return None
    return result


def _diagnostics_json(
    manifest: _ManifestData,
    frame_map: _FrameMapData,
    view_diagnostics: Sequence[Mapping[str, Any]],
    ridge_relative: float,
    frame_chunk_size: int,
    solve_seconds: float,
    write_seconds: float,
) -> dict[str, Any]:
    total_residual_sq = float(sum(float(values["residual_sum_squares"]) for values in view_diagnostics))
    total_flow_sq = float(sum(float(values["flow_sum_squares"]) for values in view_diagnostics))
    total_elements = float(
        sum(
            2 * int(values["unique_pixel_count"]) * int(np.count_nonzero(frame_map.frame_view_indices == view_index))
            for view_index, values in enumerate(view_diagnostics)
        )
    )
    overall_rmse = float(np.sqrt(total_residual_sq / total_elements))
    overall_relative = (
        0.0 if total_flow_sq <= np.finfo(np.float64).eps else float(np.sqrt(total_residual_sq / total_flow_sq))
    )
    overall_r2 = (
        1.0 if total_flow_sq <= np.finfo(np.float64).eps else float(1.0 - total_residual_sq / total_flow_sq)
    )

    views: list[dict[str, Any]] = []
    for view_index, values in enumerate(view_diagnostics):
        mode_summaries = []
        for mode_slot, mode_index in enumerate(manifest.mode_indices):
            mode_summaries.append(
                {
                    "mode_slot": mode_slot,
                    "mode_index": int(mode_index),
                    "frequency_hz": float(manifest.frequencies_hz[mode_slot]),
                    "pair_scale": float(values["mode_pair_scales"][mode_slot]),
                    "coordinate_rms": float(values["coordinate_rms"][mode_slot]),
                    "coordinate_max": float(values["coordinate_max"][mode_slot]),
                    "coordinate_temporal_mean_abs": float(
                        values["coordinate_temporal_mean_abs"][mode_slot]
                    ),
                    "dominant_signed_frequency_hz": float(
                        values["dominant_signed_frequency_hz"][mode_slot]
                    ),
                    "assigned_frequency_energy_ratio": float(
                        values["assigned_frequency_energy_ratio"][mode_slot]
                    ),
                }
            )
        views.append(
            {
                "view_id": frame_map.view_ids[view_index],
                "frame_count": int(np.count_nonzero(frame_map.frame_view_indices == view_index)),
                "candidate_row_count": int(values["candidate_row_count"]),
                "unique_pixel_count": int(values["unique_pixel_count"]),
                "candidate_weight_sum_min": float(values["candidate_weight_sum_min"]),
                "candidate_weight_sum_max": float(values["candidate_weight_sum_max"]),
                "candidate_weight_sum_max_abs_error": float(
                    values["candidate_weight_sum_max_abs_error"]
                ),
                "numerical_rank": int(values["numerical_rank"]),
                "column_count": int(2 * manifest.mode_indices.size),
                "condition_number": _json_float(values["condition_number"]),
                "ridge_condition_number": float(values["ridge_condition_number"]),
                "reference_local_index": int(values["reference_local_index"]),
                "reference_residual_norm": float(values["reference_residual_norm"]),
                "flow_rmse": float(values["flow_rmse"]),
                "relative_residual": float(values["relative_residual"]),
                "flow_r2": float(values["flow_r2"]),
                "solve_seconds": float(values["solve_seconds"]),
                "modes": mode_summaries,
            }
        )
    return {
        "version": 1,
        "format": MODAL_FLOW_COORDINATE_FORMAT,
        "parameterization": MODAL_FLOW_COORDINATE_PARAMETERIZATION,
        "solver": MODAL_FLOW_COORDINATE_SOLVER,
        "gauge": MODAL_FLOW_COORDINATE_GAUGE,
        "projection_basis": "candidate_weighted_J_phi_without_alpha",
        "frequency_forward_role": "label_only",
        "temporal_regularization": "none",
        "spectral_diagnostics": {
            "transform": "complex_fft_of_mean_zero_coordinates",
            "assigned_frequency_energy": "nearest_positive_and_negative_fft_bins",
        },
        "source_modal_manifest": str(manifest.path),
        "source_modal_frame_map": str(frame_map.path),
        "ridge_relative": float(ridge_relative),
        "frame_chunk_size": int(frame_chunk_size),
        "mode_count": int(manifest.mode_indices.size),
        "frame_count": len(frame_map.frame_names),
        "overall": {
            "flow_rmse": overall_rmse,
            "relative_residual": overall_relative,
            "flow_r2": overall_r2,
        },
        "timings_seconds": {
            "solve": float(solve_seconds),
            "write": float(write_seconds),
            "total": float(solve_seconds + write_seconds),
        },
        "views": views,
    }


def _write_artifacts(
    output_dir: Path,
    manifest: _ManifestData,
    frame_map: _FrameMapData,
    caches: Sequence[ModalAnalysisCache],
    coordinates: np.ndarray,
    view_diagnostics: Sequence[Mapping[str, Any]],
    ridge_relative: float,
    frame_chunk_size: int,
    solve_seconds: float,
) -> None:
    write_start = time.perf_counter()
    coordinate_real = coordinates.real.astype(np.float32)
    coordinate_imag = coordinates.imag.astype(np.float32)
    reference_indices = np.asarray(
        [values["reference_local_index"] for values in view_diagnostics], dtype=np.int64
    )
    np.savez_compressed(
        output_dir / COORDINATE_FILENAME,
        format=np.array(MODAL_FLOW_COORDINATE_FORMAT),
        version=np.array(MODAL_FLOW_COORDINATE_VERSION, dtype=np.int32),
        view_ids=np.asarray(frame_map.view_ids),
        frame_names=np.asarray(frame_map.frame_names),
        frame_view_indices=frame_map.frame_view_indices.astype(np.int64),
        frame_local_indices=frame_map.frame_local_indices.astype(np.int64),
        frame_times_sec=frame_map.frame_times_sec.astype(np.float64),
        mode_indices=manifest.mode_indices.astype(np.int64),
        frequencies_hz=manifest.frequencies_hz.astype(np.float64),
        coordinate_real=coordinate_real,
        coordinate_imag=coordinate_imag,
        reference_local_indices=reference_indices,
        ridge_relative=np.array(ridge_relative, dtype=np.float64),
        source_modal_manifest=np.array(str(manifest.path)),
        source_modal_frame_map=np.array(str(frame_map.path)),
        source_flow_cache_dirs=np.asarray([str(cache.path.resolve()) for cache in caches]),
    )

    num_views = len(frame_map.view_ids)
    num_modes = manifest.mode_indices.size
    num_columns = 2 * num_modes
    per_frame_rmse = np.empty((len(frame_map.frame_names),), dtype=np.float64)
    per_frame_relative = np.empty_like(per_frame_rmse)
    per_frame_r2 = np.empty_like(per_frame_rmse)
    for view_index, values in enumerate(view_diagnostics):
        rows = np.flatnonzero(frame_map.frame_view_indices == view_index)
        order = np.argsort(frame_map.frame_local_indices[rows])
        ordered_rows = rows[order]
        per_frame_rmse[ordered_rows] = values["per_frame_flow_rmse"]
        per_frame_relative[ordered_rows] = values["per_frame_relative_residual"]
        per_frame_r2[ordered_rows] = values["per_frame_flow_r2"]
    np.savez_compressed(
        output_dir / DIAGNOSTICS_NPZ_FILENAME,
        view_ids=np.asarray(frame_map.view_ids),
        frame_names=np.asarray(frame_map.frame_names),
        frame_view_indices=frame_map.frame_view_indices.astype(np.int64),
        frame_local_indices=frame_map.frame_local_indices.astype(np.int64),
        mode_indices=manifest.mode_indices.astype(np.int64),
        frequencies_hz=manifest.frequencies_hz.astype(np.float64),
        candidate_row_count=np.asarray(
            [values["candidate_row_count"] for values in view_diagnostics], dtype=np.int64
        ),
        unique_pixel_count=np.asarray(
            [values["unique_pixel_count"] for values in view_diagnostics], dtype=np.int64
        ),
        candidate_weight_sum_min=np.asarray(
            [values["candidate_weight_sum_min"] for values in view_diagnostics], dtype=np.float64
        ),
        candidate_weight_sum_max=np.asarray(
            [values["candidate_weight_sum_max"] for values in view_diagnostics], dtype=np.float64
        ),
        candidate_weight_sum_max_abs_error=np.asarray(
            [values["candidate_weight_sum_max_abs_error"] for values in view_diagnostics],
            dtype=np.float64,
        ),
        mode_pair_scales=np.stack(
            [values["mode_pair_scales"] for values in view_diagnostics]
        ).reshape(num_views, num_modes),
        singular_values=np.stack(
            [values["singular_values"] for values in view_diagnostics]
        ).reshape(num_views, num_columns),
        numerical_rank=np.asarray(
            [values["numerical_rank"] for values in view_diagnostics], dtype=np.int32
        ),
        condition_number=np.asarray(
            [values["condition_number"] for values in view_diagnostics], dtype=np.float64
        ),
        ridge_condition_number=np.asarray(
            [values["ridge_condition_number"] for values in view_diagnostics], dtype=np.float64
        ),
        reference_local_indices=reference_indices,
        per_frame_flow_rmse=per_frame_rmse,
        per_frame_relative_residual=per_frame_relative,
        per_frame_flow_r2=per_frame_r2,
        view_flow_rmse=np.asarray(
            [values["flow_rmse"] for values in view_diagnostics], dtype=np.float64
        ),
        view_relative_residual=np.asarray(
            [values["relative_residual"] for values in view_diagnostics], dtype=np.float64
        ),
        view_flow_r2=np.asarray(
            [values["flow_r2"] for values in view_diagnostics], dtype=np.float64
        ),
        view_reference_residual_norm=np.asarray(
            [values["reference_residual_norm"] for values in view_diagnostics], dtype=np.float64
        ),
        coordinate_rms=np.stack(
            [values["coordinate_rms"] for values in view_diagnostics]
        ).reshape(num_views, num_modes),
        coordinate_max=np.stack(
            [values["coordinate_max"] for values in view_diagnostics]
        ).reshape(num_views, num_modes),
        coordinate_temporal_mean_abs=np.stack(
            [values["coordinate_temporal_mean_abs"] for values in view_diagnostics]
        ).reshape(num_views, num_modes),
        coordinate_dominant_signed_frequency_hz=np.stack(
            [values["dominant_signed_frequency_hz"] for values in view_diagnostics]
        ).reshape(num_views, num_modes),
        coordinate_assigned_frequency_energy_ratio=np.stack(
            [values["assigned_frequency_energy_ratio"] for values in view_diagnostics]
        ).reshape(num_views, num_modes),
        coordinate_cross_mode_correlation=np.stack(
            [values["cross_mode_correlation"] for values in view_diagnostics]
        ).reshape(num_views, num_modes, num_modes),
        view_solve_seconds=np.asarray(
            [values["solve_seconds"] for values in view_diagnostics], dtype=np.float64
        ),
    )

    write_seconds = float(time.perf_counter() - write_start)
    diagnostics = _diagnostics_json(
        manifest,
        frame_map,
        view_diagnostics,
        ridge_relative,
        frame_chunk_size,
        solve_seconds,
        write_seconds,
    )
    with (output_dir / DIAGNOSTICS_JSON_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=2, sort_keys=True, allow_nan=False)


def _validate_diagnostic_artifacts(output_dir: Path, coordinate_data: ModalFlowCoordinates) -> None:
    diagnostics_path = output_dir / DIAGNOSTICS_NPZ_FILENAME
    json_path = output_dir / DIAGNOSTICS_JSON_FILENAME
    if not diagnostics_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Coordinate output is missing diagnostics files: {output_dir}")
    with np.load(diagnostics_path, allow_pickle=False) as archive:
        if set(archive.files) != _DIAGNOSTIC_FIELDS:
            raise ValueError(
                f"{diagnostics_path} fields must be exactly {sorted(_DIAGNOSTIC_FIELDS)}"
            )
        if _string_vector(archive["view_ids"], "view_ids", diagnostics_path) != coordinate_data.view_ids:
            raise ValueError(f"{diagnostics_path} view_ids do not match coordinate artifact")
        if _string_vector(archive["frame_names"], "frame_names", diagnostics_path) != coordinate_data.frame_names:
            raise ValueError(f"{diagnostics_path} frame_names do not match coordinate artifact")
        if not np.array_equal(
            archive["frame_view_indices"], coordinate_data.frame_view_indices
        ):
            raise ValueError(
                f"{diagnostics_path} frame_view_indices do not match coordinate artifact"
            )
        if not np.array_equal(
            archive["frame_local_indices"], coordinate_data.frame_local_indices
        ):
            raise ValueError(
                f"{diagnostics_path} frame_local_indices do not match coordinate artifact"
            )
        if not np.array_equal(archive["mode_indices"], coordinate_data.mode_indices):
            raise ValueError(f"{diagnostics_path} mode_indices do not match coordinate artifact")
        if not np.allclose(
            archive["frequencies_hz"], coordinate_data.frequencies_hz, rtol=0.0, atol=0.0
        ):
            raise ValueError(f"{diagnostics_path} frequencies_hz do not match coordinate artifact")
        if not np.array_equal(
            archive["reference_local_indices"], coordinate_data.reference_local_indices
        ):
            raise ValueError(
                f"{diagnostics_path} reference_local_indices do not match coordinate artifact"
            )

        num_views = len(coordinate_data.view_ids)
        num_frames = len(coordinate_data.frame_names)
        num_modes = coordinate_data.mode_indices.size
        expected_shapes = {
            "candidate_row_count": (num_views,),
            "unique_pixel_count": (num_views,),
            "candidate_weight_sum_min": (num_views,),
            "candidate_weight_sum_max": (num_views,),
            "candidate_weight_sum_max_abs_error": (num_views,),
            "mode_pair_scales": (num_views, num_modes),
            "singular_values": (num_views, 2 * num_modes),
            "numerical_rank": (num_views,),
            "condition_number": (num_views,),
            "ridge_condition_number": (num_views,),
            "per_frame_flow_rmse": (num_frames,),
            "per_frame_relative_residual": (num_frames,),
            "per_frame_flow_r2": (num_frames,),
            "view_flow_rmse": (num_views,),
            "view_relative_residual": (num_views,),
            "view_flow_r2": (num_views,),
            "view_reference_residual_norm": (num_views,),
            "coordinate_rms": (num_views, num_modes),
            "coordinate_max": (num_views, num_modes),
            "coordinate_temporal_mean_abs": (num_views, num_modes),
            "coordinate_dominant_signed_frequency_hz": (num_views, num_modes),
            "coordinate_assigned_frequency_energy_ratio": (num_views, num_modes),
            "coordinate_cross_mode_correlation": (num_views, num_modes, num_modes),
            "view_solve_seconds": (num_views,),
        }
        for name, expected_shape in expected_shapes.items():
            values = np.asarray(archive[name])
            if values.shape != expected_shape:
                raise ValueError(
                    f"{diagnostics_path} {name} shape {values.shape} does not match "
                    f"{expected_shape}"
                )
            if name == "condition_number":
                if np.isnan(values).any() or np.any(values <= 0.0):
                    raise ValueError(
                        f"{diagnostics_path} condition_number must be positive or infinity"
                    )
            elif not np.isfinite(values).all():
                raise ValueError(f"{diagnostics_path} {name} must be finite")
        if np.any(archive["candidate_row_count"] <= 0) or np.any(
            archive["unique_pixel_count"] <= 0
        ):
            raise ValueError(f"{diagnostics_path} row and pixel counts must be positive")
        if np.any(archive["numerical_rank"] < 0) or np.any(
            archive["numerical_rank"] > 2 * num_modes
        ):
            raise ValueError(f"{diagnostics_path} numerical_rank is out of range")
        if np.any(archive["candidate_weight_sum_max_abs_error"] > 1e-5):
            raise ValueError(
                f"{diagnostics_path} candidate weight sum error exceeds 1e-5"
            )
        if np.any(archive["coordinate_assigned_frequency_energy_ratio"] < 0.0) or np.any(
            archive["coordinate_assigned_frequency_energy_ratio"] > 1.0 + 1e-12
        ):
            raise ValueError(
                f"{diagnostics_path} assigned-frequency energy ratio is outside [0,1]"
            )
    payload = _read_json_object(json_path)
    if payload.get("version") != 1 or payload.get("format") != MODAL_FLOW_COORDINATE_FORMAT:
        raise ValueError(f"{json_path} has unsupported diagnostics format/version")
    if payload.get("parameterization") != MODAL_FLOW_COORDINATE_PARAMETERIZATION:
        raise ValueError(f"{json_path} has unsupported parameterization")
    if payload.get("solver") != MODAL_FLOW_COORDINATE_SOLVER:
        raise ValueError(f"{json_path} has unsupported coordinate solver")
    if payload.get("gauge") != MODAL_FLOW_COORDINATE_GAUGE:
        raise ValueError(f"{json_path} has unsupported coordinate gauge")
    if payload.get("projection_basis") != "candidate_weighted_J_phi_without_alpha":
        raise ValueError(f"{json_path} has unsupported projection basis")
    if payload.get("frequency_forward_role") != "label_only":
        raise ValueError(f"{json_path} has unsupported frequency forward role")
    if payload.get("temporal_regularization") != "none":
        raise ValueError(f"{json_path} has unsupported temporal regularization")
    if payload.get("frame_count") != len(coordinate_data.frame_names):
        raise ValueError(f"{json_path} frame_count does not match coordinate artifact")
    if payload.get("mode_count") != coordinate_data.mode_indices.size:
        raise ValueError(f"{json_path} mode_count does not match coordinate artifact")
    views = payload.get("views")
    if (
        not isinstance(views, list)
        or any(not isinstance(view, dict) for view in views)
        or [view.get("view_id") for view in views] != list(coordinate_data.view_ids)
    ):
        raise ValueError(f"{json_path} views do not match coordinate artifact")
    overall = payload.get("overall")
    if not isinstance(overall, dict):
        raise ValueError(f"{json_path} overall must be an object")
    for name in ("flow_rmse", "relative_residual", "flow_r2"):
        value = overall.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{json_path} overall.{name} must be finite")


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def load_modal_flow_coordinates(path_value: str | Path) -> ModalFlowCoordinates:
    """Load and strictly validate a version-1 modal flow-coordinate artifact."""

    path = Path(path_value).expanduser()
    if path.is_dir():
        path = path / COORDINATE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Modal flow coordinate artifact does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        fields = set(archive.files)
        if fields != _COORDINATE_FIELDS:
            missing = sorted(_COORDINATE_FIELDS - fields)
            extra = sorted(fields - _COORDINATE_FIELDS)
            raise ValueError(f"{path} coordinate fields mismatch; missing={missing}, extra={extra}")
        artifact_format = _scalar_string(archive["format"], "format", path)
        version = _scalar_int(archive["version"], "version", path)
        view_ids = _string_vector(archive["view_ids"], "view_ids", path)
        frame_names = _string_vector(archive["frame_names"], "frame_names", path)
        frame_view_indices = np.asarray(archive["frame_view_indices"])
        frame_local_indices = np.asarray(archive["frame_local_indices"])
        frame_times_sec = np.asarray(archive["frame_times_sec"])
        mode_indices = np.asarray(archive["mode_indices"])
        frequencies_hz = np.asarray(archive["frequencies_hz"])
        coordinate_real = np.asarray(archive["coordinate_real"])
        coordinate_imag = np.asarray(archive["coordinate_imag"])
        reference_local_indices = np.asarray(archive["reference_local_indices"])
        ridge_relative = _scalar_float(archive["ridge_relative"], "ridge_relative", path)
        source_manifest = _scalar_string(
            archive["source_modal_manifest"], "source_modal_manifest", path
        )
        source_frame_map = _scalar_string(
            archive["source_modal_frame_map"], "source_modal_frame_map", path
        )
        source_cache_dirs = _string_vector(
            archive["source_flow_cache_dirs"], "source_flow_cache_dirs", path
        )

    if artifact_format != MODAL_FLOW_COORDINATE_FORMAT:
        raise ValueError(
            f"{path} format={artifact_format!r}; expected {MODAL_FLOW_COORDINATE_FORMAT!r}"
        )
    if version != MODAL_FLOW_COORDINATE_VERSION:
        raise ValueError(
            f"{path} version={version}; expected {MODAL_FLOW_COORDINATE_VERSION}"
        )
    if not view_ids or len(set(view_ids)) != len(view_ids):
        raise ValueError(f"{path} view_ids must be non-empty and unique")
    if not frame_names or len(set(frame_names)) != len(frame_names):
        raise ValueError(f"{path} frame_names must be non-empty and unique")
    num_views = len(view_ids)
    num_frames = len(frame_names)
    if len(source_cache_dirs) != num_views:
        raise ValueError(f"{path} source_flow_cache_dirs must have one entry per view")
    if len(set(source_cache_dirs)) != num_views:
        raise ValueError(f"{path} source_flow_cache_dirs must be unique")
    for name, values in (
        ("frame_view_indices", frame_view_indices),
        ("frame_local_indices", frame_local_indices),
    ):
        if values.shape != (num_frames,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{path} {name} must be an integer [T] array")
    if frame_times_sec.shape != (num_frames,) or not np.issubdtype(frame_times_sec.dtype, np.floating):
        raise ValueError(f"{path} frame_times_sec must be a floating [T] array")
    if not np.isfinite(frame_times_sec).all() or np.any(frame_times_sec < 0.0):
        raise ValueError(f"{path} frame_times_sec must be finite and non-negative")
    if np.any(frame_view_indices < 0) or np.any(frame_view_indices >= num_views):
        raise ValueError(f"{path} frame_view_indices contains out-of-range values")
    if np.any(frame_local_indices < 0):
        raise ValueError(f"{path} frame_local_indices must be non-negative")
    if mode_indices.ndim != 1 or not np.issubdtype(mode_indices.dtype, np.integer):
        raise ValueError(f"{path} mode_indices must be an integer [K] array")
    if mode_indices.size == 0 or np.unique(mode_indices).size != mode_indices.size:
        raise ValueError(f"{path} mode_indices must be non-empty and unique")
    num_modes = mode_indices.size
    if frequencies_hz.shape != (num_modes,) or not np.issubdtype(
        frequencies_hz.dtype, np.floating
    ):
        raise ValueError(f"{path} frequencies_hz must be a floating [K] array")
    if not np.isfinite(frequencies_hz).all() or np.any(frequencies_hz <= 0.0):
        raise ValueError(f"{path} frequencies_hz must be finite and positive")
    if coordinate_real.dtype != np.float32 or coordinate_imag.dtype != np.float32:
        raise ValueError(f"{path} coordinate_real/imag must have dtype float32")
    if coordinate_real.shape != (num_frames, num_modes) or coordinate_imag.shape != (
        num_frames,
        num_modes,
    ):
        raise ValueError(f"{path} coordinate_real/imag must have shape [T,K]")
    if not np.isfinite(coordinate_real).all() or not np.isfinite(coordinate_imag).all():
        raise ValueError(f"{path} coordinate_real/imag must be finite")
    if reference_local_indices.shape != (num_views,) or not np.issubdtype(
        reference_local_indices.dtype, np.integer
    ):
        raise ValueError(f"{path} reference_local_indices must be an integer [V] array")
    if ridge_relative <= 0.0:
        raise ValueError(f"{path} ridge_relative must be positive")
    for view_index, view_id in enumerate(view_ids):
        rows = np.flatnonzero(frame_view_indices == view_index)
        if rows.size == 0:
            raise ValueError(f"{path} view {view_id!r} has no frames")
        local_indices = frame_local_indices[rows]
        if not np.array_equal(np.sort(local_indices), np.arange(rows.size)):
            raise ValueError(f"{path} local indices for view {view_id!r} must be contiguous")
        local_order = np.argsort(local_indices)
        ordered_times = frame_times_sec[rows[local_order]]
        if abs(float(ordered_times[0])) > 1e-12:
            raise ValueError(f"{path} frame times for view {view_id!r} must start at zero")
        if ordered_times.size > 1:
            time_steps = np.diff(ordered_times)
            if np.any(time_steps <= 0.0) or not np.allclose(
                time_steps,
                time_steps[0],
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"{path} frame times for view {view_id!r} must have one constant positive step"
                )
        reference_index = int(reference_local_indices[view_index])
        if reference_index < 0 or reference_index >= rows.size:
            raise ValueError(f"{path} reference index for view {view_id!r} is out of range")
        view_coordinates = coordinate_real[rows].astype(np.float64) + 1j * coordinate_imag[
            rows
        ].astype(np.float64)
        max_mean = float(np.max(np.abs(np.mean(view_coordinates, axis=0))))
        if max_mean > 1e-6:
            raise ValueError(
                f"{path} view {view_id!r} coordinate temporal mean {max_mean:.6g} exceeds 1e-6"
            )

    return ModalFlowCoordinates(
        path=path.resolve(),
        view_ids=view_ids,
        frame_names=frame_names,
        frame_view_indices=_readonly(frame_view_indices.astype(np.int64, copy=False)),
        frame_local_indices=_readonly(frame_local_indices.astype(np.int64, copy=False)),
        frame_times_sec=_readonly(frame_times_sec.astype(np.float64, copy=False)),
        mode_indices=_readonly(mode_indices.astype(np.int64, copy=False)),
        frequencies_hz=_readonly(frequencies_hz.astype(np.float64, copy=False)),
        coordinate_real=_readonly(coordinate_real),
        coordinate_imag=_readonly(coordinate_imag),
        reference_local_indices=_readonly(reference_local_indices.astype(np.int64, copy=False)),
        ridge_relative=ridge_relative,
        source_modal_manifest=Path(source_manifest),
        source_modal_frame_map=Path(source_frame_map),
        source_flow_cache_dirs=tuple(Path(value) for value in source_cache_dirs),
    )


def evaluate_modal_flow_coordinate_sets(
    source: ModalFlowCoordinates,
    coordinate_sets: Mapping[str, np.ndarray],
    *,
    frame_chunk_size: int = 64,
) -> dict[str, dict[str, np.ndarray | float]]:
    """Evaluate one or more coordinate fields against the source reference flow."""

    if not coordinate_sets:
        raise ValueError("coordinate_sets must be non-empty")
    if (
        isinstance(frame_chunk_size, bool)
        or int(frame_chunk_size) != frame_chunk_size
        or frame_chunk_size < 1
    ):
        raise ValueError("frame_chunk_size must be a positive integer")
    chunk_size = int(frame_chunk_size)
    expected_shape = (len(source.frame_names), source.mode_indices.size)
    validated_sets: dict[str, np.ndarray] = {}
    for label, values in coordinate_sets.items():
        if not isinstance(label, str) or not label:
            raise ValueError("coordinate set labels must be non-empty strings")
        array = np.asarray(values, dtype=np.complex128)
        if array.shape != expected_shape or not np.isfinite(array.real).all() or not np.isfinite(
            array.imag
        ).all():
            raise ValueError(
                f"Coordinate set {label!r} must be finite complex {expected_shape}"
            )
        validated_sets[label] = array

    frame_map = _load_frame_map(source.source_modal_frame_map)
    manifest = _load_manifest(source.source_modal_manifest)
    if frame_map.view_ids != source.view_ids:
        raise ValueError("Coordinate source view order differs from its frame map")
    if frame_map.frame_names != source.frame_names:
        raise ValueError("Coordinate source frame names differ from its frame map")
    if not np.array_equal(frame_map.frame_view_indices, source.frame_view_indices):
        raise ValueError("Coordinate source frame view indices differ from its frame map")
    if not np.array_equal(frame_map.frame_local_indices, source.frame_local_indices):
        raise ValueError("Coordinate source local indices differ from its frame map")
    if not np.allclose(frame_map.frame_times_sec, source.frame_times_sec, rtol=0.0, atol=0.0):
        raise ValueError("Coordinate source frame times differ from its frame map")
    if not np.array_equal(manifest.mode_indices, source.mode_indices) or not np.allclose(
        manifest.frequencies_hz, source.frequencies_hz, rtol=0.0, atol=0.0
    ):
        raise ValueError("Coordinate source modes differ from its modal manifest")
    cache_specs = [
        f"{view_id}={cache_path}"
        for view_id, cache_path in zip(source.view_ids, source.source_flow_cache_dirs)
    ]
    caches = _load_and_validate_caches(frame_map, manifest.topology, cache_specs)

    num_frames = len(source.frame_names)
    num_views = len(source.view_ids)
    num_modes = source.mode_indices.size
    results: dict[str, dict[str, np.ndarray | float]] = {}
    accumulators: dict[str, dict[str, float]] = {}
    total_elements = 0.0
    for label in validated_sets:
        results[label] = {
            "per_frame_flow_rmse": np.empty((num_frames,), dtype=np.float64),
            "per_frame_relative_residual": np.empty((num_frames,), dtype=np.float64),
            "per_frame_flow_r2": np.empty((num_frames,), dtype=np.float64),
            "view_flow_rmse": np.empty((num_views,), dtype=np.float64),
            "view_relative_residual": np.empty((num_views,), dtype=np.float64),
            "view_flow_r2": np.empty((num_views,), dtype=np.float64),
            "view_strong_motion_flow_r2": np.empty((num_views,), dtype=np.float64),
        }
        accumulators[label] = {"residual_sum_squares": 0.0, "flow_sum_squares": 0.0}

    for view_index, cache in enumerate(caches):
        sorted_rows, starts, pixels, sorted_weights, _ = _view_pixel_groups(
            manifest.topology, view_index, cache
        )
        design, _ = _build_design_matrix(manifest, sorted_rows, starts, sorted_weights)
        normalizer = float(2 * pixels.shape[0])
        reference_index = int(source.reference_local_indices[view_index])
        cache_reference_index = int(cache.metadata["analysis"]["reference_frame_index"])
        if reference_index != cache_reference_index:
            raise ValueError(
                f"Coordinate reference index for {source.view_ids[view_index]!r} does not "
                "match its source flow cache"
            )
        x = pixels[:, 0]
        y = pixels[:, 1]
        reference_u = np.asarray(cache.flow_u[reference_index, y, x], dtype=np.float64)
        reference_v = np.asarray(cache.flow_v[reference_index, y, x], dtype=np.float64)
        rows = np.flatnonzero(source.frame_view_indices == view_index)
        order = np.argsort(source.frame_local_indices[rows])
        ordered_rows = rows[order]
        num_view_frames = ordered_rows.size
        total_elements += normalizer * num_view_frames
        flow_energy = np.empty((num_view_frames,), dtype=np.float64)
        residual_energy = {
            label: np.empty((num_view_frames,), dtype=np.float64)
            for label in validated_sets
        }
        reference_coordinates = {
            label: values[ordered_rows[reference_index]]
            for label, values in validated_sets.items()
        }
        for start in range(0, num_view_frames, chunk_size):
            end = min(start + chunk_size, num_view_frames)
            flow = _flow_matrix(cache, pixels, start, end, reference_u, reference_v)
            flow_energy[start:end] = np.sum(flow * flow, axis=0)
            for label, values in validated_sets.items():
                relative = (
                    values[ordered_rows[start:end]] - reference_coordinates[label][None, :]
                )
                packed = np.empty((2 * num_modes, end - start), dtype=np.float64)
                packed[0::2] = relative.real.T
                packed[1::2] = relative.imag.T
                residual = design @ packed - flow
                residual_energy[label][start:end] = np.sum(residual * residual, axis=0)

        positive_flow = flow_energy > np.finfo(np.float64).eps
        strong_count = max(1, int(np.ceil(0.1 * num_view_frames)))
        strong_rows = np.argpartition(flow_energy, -strong_count)[-strong_count:]
        for label in validated_sets:
            residual_sq = residual_energy[label]
            per_frame_rmse = np.sqrt(residual_sq / normalizer)
            per_frame_relative = np.zeros_like(per_frame_rmse)
            per_frame_r2 = np.ones_like(per_frame_rmse)
            per_frame_relative[positive_flow] = np.sqrt(
                residual_sq[positive_flow] / flow_energy[positive_flow]
            )
            per_frame_r2[positive_flow] = (
                1.0 - residual_sq[positive_flow] / flow_energy[positive_flow]
            )
            zero_flow_bad = (~positive_flow) & (
                residual_sq > np.finfo(np.float64).eps
            )
            if np.any(zero_flow_bad):
                raise ValueError(
                    f"Coordinate set {label!r} predicts non-zero motion for a zero-flow "
                    f"frame in view {source.view_ids[view_index]!r}"
                )
            total_residual = float(np.sum(residual_sq))
            total_flow = float(np.sum(flow_energy))
            relative = 0.0 if total_flow <= np.finfo(np.float64).eps else float(
                np.sqrt(total_residual / total_flow)
            )
            r2 = 1.0 if total_flow <= np.finfo(np.float64).eps else float(
                1.0 - total_residual / total_flow
            )
            strong_flow = float(np.sum(flow_energy[strong_rows]))
            strong_residual = float(np.sum(residual_sq[strong_rows]))
            strong_r2 = 1.0 if strong_flow <= np.finfo(np.float64).eps else float(
                1.0 - strong_residual / strong_flow
            )
            result = results[label]
            result["per_frame_flow_rmse"][ordered_rows] = per_frame_rmse
            result["per_frame_relative_residual"][ordered_rows] = per_frame_relative
            result["per_frame_flow_r2"][ordered_rows] = per_frame_r2
            result["view_flow_rmse"][view_index] = float(
                np.sqrt(total_residual / (normalizer * num_view_frames))
            )
            result["view_relative_residual"][view_index] = relative
            result["view_flow_r2"][view_index] = r2
            result["view_strong_motion_flow_r2"][view_index] = strong_r2
            accumulators[label]["residual_sum_squares"] += total_residual
            accumulators[label]["flow_sum_squares"] += total_flow

    for label, result in results.items():
        residual_sum = accumulators[label]["residual_sum_squares"]
        flow_sum = accumulators[label]["flow_sum_squares"]
        result["overall_flow_rmse"] = float(np.sqrt(residual_sum / total_elements))
        result["overall_relative_residual"] = (
            0.0 if flow_sum <= np.finfo(np.float64).eps else float(np.sqrt(residual_sum / flow_sum))
        )
        result["overall_flow_r2"] = (
            1.0 if flow_sum <= np.finfo(np.float64).eps else float(1.0 - residual_sum / flow_sum)
        )
    return results


def solve_modal_flow_coordinates(
    *,
    modal_manifest: str | Path,
    modal_frame_map: str | Path,
    flow_caches: Sequence[str],
    out_dir: str | Path,
    ridge_relative: float = 1e-4,
    frame_chunk_size: int = 64,
) -> ModalFlowCoordinates:
    """Solve fixed per-frame modal coordinates and atomically publish artifacts."""

    ridge = float(ridge_relative)
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge_relative must be finite and positive")
    if isinstance(frame_chunk_size, bool) or int(frame_chunk_size) != frame_chunk_size or frame_chunk_size < 1:
        raise ValueError("frame_chunk_size must be a positive integer")
    chunk_size = int(frame_chunk_size)
    target = Path(out_dir).expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Modal flow coordinate output already exists: {target}")

    solve_start = time.perf_counter()
    frame_map = _load_frame_map(modal_frame_map)
    manifest = _load_manifest(modal_manifest)
    if manifest.topology.view_ids != frame_map.view_ids:
        raise ValueError(
            f"Manifest observation views {manifest.topology.view_ids} do not match "
            f"frame map views {frame_map.view_ids}"
        )
    caches = _load_and_validate_caches(frame_map, manifest.topology, flow_caches)

    num_frames = len(frame_map.frame_names)
    num_modes = manifest.mode_indices.size
    coordinates = np.empty((num_frames, num_modes), dtype=np.complex64)
    view_diagnostics: list[dict[str, Any]] = []
    for view_index, cache in enumerate(caches):
        view_coordinates, diagnostics = _solve_view(
            manifest,
            frame_map,
            cache,
            view_index,
            ridge,
            chunk_size,
        )
        rows = np.flatnonzero(frame_map.frame_view_indices == view_index)
        order = np.argsort(frame_map.frame_local_indices[rows])
        coordinates[rows[order]] = view_coordinates
        view_diagnostics.append(diagnostics)
    solve_seconds = float(time.perf_counter() - solve_start)
    if not np.isfinite(coordinates.real).all() or not np.isfinite(coordinates.imag).all():
        raise ValueError("Solved modal flow coordinates are non-finite")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    )
    try:
        _write_artifacts(
            temporary,
            manifest,
            frame_map,
            caches,
            coordinates,
            view_diagnostics,
            ridge,
            chunk_size,
            solve_seconds,
        )
        validated = load_modal_flow_coordinates(temporary)
        _validate_diagnostic_artifacts(temporary, validated)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Modal flow coordinate output already exists: {target}")
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_modal_flow_coordinates(target)
