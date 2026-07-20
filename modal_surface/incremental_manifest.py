"""Strict helpers for extending a Gaussian modal manifest without recomputing it."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_MODE_ARTIFACT_FIELDS = (
    "observation_path",
    "latent_path",
    "diagnostics_path",
    "vis_dir",
)

_OBSERVATION_TOPOLOGY_FIELDS = (
    "points_world",
    "gaussian_indices",
    "obs_point_index",
    "obs_view_index",
    "obs_pixels_xy",
    "obs_J",
    "obs_contribution_weight",
    "view_ids",
    "view_image_width",
    "view_image_height",
)


@dataclass(frozen=True)
class IncrementalManifestBase:
    """Validated base modes and shared artifacts used by an incremental solve."""

    path: Path
    modes: tuple[dict[str, Any], ...]
    mode_indices: tuple[int, ...]
    observation_topology: Mapping[str, np.ndarray]
    motion_fill_graph_path: Path | None


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _resolve_artifact_path(manifest_path: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{manifest_path} is missing non-empty {field}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _resolve_source_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _without_artifact_paths(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in parameters.items()
        if key != "motion_fill_graph_path"
    }


def _load_observation_topology(
    manifest_path: Path,
    mode: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    observation_path = _resolve_artifact_path(
        manifest_path,
        mode.get("observation_path"),
        "observation_path",
    )
    with np.load(observation_path, allow_pickle=False) as archive:
        missing = sorted(set(_OBSERVATION_TOPOLOGY_FIELDS) - set(archive.files))
        if missing:
            raise ValueError(
                f"{observation_path} is missing observation topology fields: {missing}"
            )
        return {
            field: np.asarray(archive[field]).copy()
            for field in _OBSERVATION_TOPOLOGY_FIELDS
        }


def _rebase_mode_artifacts(
    manifest_path: Path,
    raw_mode: Mapping[str, Any],
) -> dict[str, Any]:
    mode = deepcopy(dict(raw_mode))
    for field in _MODE_ARTIFACT_FIELDS:
        value = mode.get(field)
        if value is None and field == "vis_dir":
            continue
        mode[field] = str(_resolve_artifact_path(manifest_path, value, field))
    return mode


def load_incremental_manifest_base(
    path_value: str | Path,
    *,
    source_checkpoint: str,
    source_view_configs: Sequence[str],
    frequencies_by_view: Sequence[np.ndarray],
    extension_mode_indices: Sequence[int],
    expected_parameters: Mapping[str, Any],
    frequency_tolerance_hz: float,
) -> IncrementalManifestBase:
    """Load and validate a prefix manifest for a solve of only later mode slots."""

    path = Path(path_value).expanduser().resolve()
    payload = _read_json_object(path)
    if payload.get("version") != 1:
        raise ValueError(f"{path} must be a version 1 modal manifest")
    if payload.get("point_type") != "foreground_gaussian_center":
        raise ValueError(
            f"{path} point_type must be 'foreground_gaussian_center'"
        )

    base_source = payload.get("source_checkpoint")
    if base_source != source_checkpoint:
        raise ValueError(
            "--input-ckpt must exactly match the base manifest source_checkpoint "
            "so reused and newly written latent artifacts share one checkpoint identity"
        )

    base_view_configs = payload.get("source_view_configs")
    if not isinstance(base_view_configs, list) or not all(
        isinstance(value, str) and value for value in base_view_configs
    ):
        raise ValueError(f"{path} has invalid source_view_configs")
    resolved_base_views = tuple(
        _resolve_source_path(path, value) for value in base_view_configs
    )
    resolved_current_views = tuple(
        Path(value).expanduser().resolve() for value in source_view_configs
    )
    if resolved_base_views != resolved_current_views:
        raise ValueError(
            "--view-config paths and order must match the base manifest exactly"
        )

    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{path} has invalid parameters")
    base_parameters = _without_artifact_paths(parameters)
    current_parameters = _without_artifact_paths(expected_parameters)
    if base_parameters != current_parameters:
        differing = sorted(
            key
            for key in set(base_parameters) | set(current_parameters)
            if base_parameters.get(key) != current_parameters.get(key)
        )
        raise ValueError(
            "Incremental solve parameters differ from the base manifest; all staged, "
            "pixel-candidate, alpha, and motion-fill settings must remain identical. "
            f"Differing fields: {differing}"
        )

    raw_modes = payload.get("modes")
    raw_indices = payload.get("mode_indices")
    if not isinstance(raw_modes, list) or not raw_modes:
        raise ValueError(f"{path} must contain a non-empty modes list")
    if not isinstance(raw_indices, list) or len(raw_indices) != len(raw_modes):
        raise ValueError(f"{path} mode_indices must match its modes list")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_indices
    ):
        raise ValueError(f"{path} mode_indices must contain integers")
    base_indices = tuple(int(value) for value in raw_indices)
    if base_indices != tuple(range(len(base_indices))):
        raise ValueError(
            "Incremental extension requires the base manifest to contain the contiguous "
            "prefix mode indices 0..K-1 in order"
        )
    expected_extension = tuple(
        range(len(base_indices), len(base_indices) + len(extension_mode_indices))
    )
    if tuple(int(value) for value in extension_mode_indices) != expected_extension:
        raise ValueError(
            "--mode-indices must be exactly the next contiguous slots after the base "
            f"manifest: expected {list(expected_extension)}"
        )

    if len(frequencies_by_view) != len(source_view_configs):
        raise ValueError("Frequency/view count mismatch during incremental validation")
    tolerance = float(frequency_tolerance_hz)
    for slot, raw_mode in enumerate(raw_modes):
        if not isinstance(raw_mode, dict):
            raise ValueError(f"{path} mode slot {slot} must be an object")
        if raw_mode.get("mode_index") != base_indices[slot]:
            raise ValueError(f"{path} modes and mode_indices differ at slot {slot}")
        per_view = raw_mode.get("freqs_hz_by_view")
        if not isinstance(per_view, list) or len(per_view) != len(frequencies_by_view):
            raise ValueError(
                f"{path} mode {base_indices[slot]} has invalid freqs_hz_by_view"
            )
        for view_index, frequencies in enumerate(frequencies_by_view):
            if base_indices[slot] >= int(frequencies.shape[0]):
                raise ValueError(
                    f"Current modal NPZ view {view_index} does not contain base mode "
                    f"index {base_indices[slot]}"
                )
            old_frequency = float(per_view[view_index])
            current_frequency = float(frequencies[base_indices[slot]])
            if (
                not np.isfinite(old_frequency)
                or abs(old_frequency - current_frequency) > tolerance
            ):
                raise ValueError(
                    f"Base mode {base_indices[slot]} frequency for view {view_index} "
                    f"({old_frequency:.9g} Hz) does not match the current modal NPZ "
                    f"({current_frequency:.9g} Hz) within {tolerance:.9g} Hz"
                )

    modes = tuple(_rebase_mode_artifacts(path, mode) for mode in raw_modes)
    observation_topology = _load_observation_topology(path, raw_modes[0])

    graph_path: Path | None = None
    graph_value = parameters.get("motion_fill_graph_path")
    if bool(parameters.get("motion_fill_enabled")):
        graph_path = _resolve_artifact_path(
            path,
            graph_value,
            "parameters.motion_fill_graph_path",
        )
    elif graph_value is not None:
        raise ValueError(
            f"{path} declares motion_fill_graph_path while motion fill is disabled"
        )

    return IncrementalManifestBase(
        path=path,
        modes=modes,
        mode_indices=base_indices,
        observation_topology=observation_topology,
        motion_fill_graph_path=graph_path,
    )


def validate_incremental_observation_topology(
    base: IncrementalManifestBase,
    observations: Mapping[str, np.ndarray],
    observation_path: str | Path,
) -> None:
    """Require a new mode to preserve the base pixel-candidate row topology."""

    missing = sorted(set(_OBSERVATION_TOPOLOGY_FIELDS) - set(observations))
    if missing:
        raise ValueError(
            f"{observation_path} is missing observation topology fields: {missing}"
        )
    for field, expected in base.observation_topology.items():
        current = np.asarray(observations[field])
        if not np.array_equal(current, expected):
            raise ValueError(
                f"{observation_path} {field} differs from the base manifest topology"
            )
