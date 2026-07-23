"""Strictly combine solved Gaussian modal manifests without recomputing modes."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


_MODE_ARTIFACT_FIELDS = (
    "observation_path",
    "latent_path",
    "diagnostics_path",
    "component_diagnostics_path",
    "rigid_component_graph_path",
    "vis_dir",
)

_OPTIONAL_MODE_ARTIFACT_FIELDS = {
    "component_diagnostics_path",
    "rigid_component_graph_path",
    "vis_dir",
}

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

_PER_MANIFEST_PARAMETER_FIELDS = {
    "motion_fill_graph_path",
    "rigid_component_graph_count",
}


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _required_string_list(
    manifest_path: Path,
    payload: Mapping[str, Any],
    field: str,
) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{manifest_path} has invalid {field}")
    return value


def _resolve_artifact_path(
    manifest_path: Path,
    value: object,
    field: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{manifest_path} is missing non-empty {field}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _relative_to_output(path: Path, output_path: Path) -> str:
    return Path(os.path.relpath(path, output_path.parent)).as_posix()


def _validated_mode_indices(
    manifest_path: Path,
    payload: Mapping[str, Any],
) -> tuple[int, ...]:
    raw_modes = payload.get("modes")
    raw_indices = payload.get("mode_indices")
    if not isinstance(raw_modes, list) or not raw_modes:
        raise ValueError(f"{manifest_path} must contain a non-empty modes list")
    if not isinstance(raw_indices, list) or len(raw_indices) != len(raw_modes):
        raise ValueError(f"{manifest_path} mode_indices must match its modes list")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_indices
    ):
        raise ValueError(f"{manifest_path} mode_indices must contain integers")
    indices = tuple(int(value) for value in raw_indices)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{manifest_path} contains duplicate mode indices")
    for slot, (mode_index, raw_mode) in enumerate(zip(indices, raw_modes)):
        if not isinstance(raw_mode, dict):
            raise ValueError(f"{manifest_path} mode slot {slot} must be an object")
        if raw_mode.get("mode_index") != mode_index:
            raise ValueError(
                f"{manifest_path} modes and mode_indices differ at slot {slot}"
            )
        frequency = raw_mode.get("freq_hz")
        if (
            isinstance(frequency, bool)
            or not isinstance(frequency, (int, float))
            or not np.isfinite(float(frequency))
            or float(frequency) <= 0.0
        ):
            raise ValueError(
                f"{manifest_path} mode {mode_index} has invalid freq_hz"
            )
    return indices


def _load_observation_topology(
    manifest_path: Path,
    raw_mode: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    observation_path = _resolve_artifact_path(
        manifest_path,
        raw_mode.get("observation_path"),
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


def _validate_same_topology(
    current: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    observation_path: Path,
) -> None:
    for field in _OBSERVATION_TOPOLOGY_FIELDS:
        if not np.array_equal(current[field], expected[field]):
            raise ValueError(
                f"{observation_path} {field} differs from the first mode topology"
            )


def _rebase_mode(
    manifest_path: Path,
    raw_mode: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    mode = deepcopy(dict(raw_mode))
    for field in _MODE_ARTIFACT_FIELDS:
        value = mode.get(field)
        if value is None and field in _OPTIONAL_MODE_ARTIFACT_FIELDS:
            continue
        try:
            path = _resolve_artifact_path(manifest_path, value, field)
        except FileNotFoundError:
            if field == "vis_dir":
                mode.pop(field)
                continue
            raise
        mode[field] = _relative_to_output(path, output_path)
    return mode


def _shared_parameters(
    manifests: Sequence[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    shared: dict[str, Any] | None = None
    rigid_count_declared: list[int] = []
    for manifest_path, payload in manifests:
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"{manifest_path} has invalid parameters")
        normalized = {
            key: value
            for key, value in parameters.items()
            if key not in _PER_MANIFEST_PARAMETER_FIELDS
        }
        if shared is None:
            shared = deepcopy(normalized)
        elif normalized != shared:
            differing = sorted(
                key
                for key in set(shared) | set(normalized)
                if shared.get(key) != normalized.get(key)
            )
            raise ValueError(
                "Input manifest parameters differ outside per-manifest artifact "
                f"fields: {differing}"
            )
        rigid_count = parameters.get("rigid_component_graph_count")
        if rigid_count is not None:
            if isinstance(rigid_count, bool) or not isinstance(rigid_count, int):
                raise ValueError(
                    f"{manifest_path} rigid_component_graph_count must be an integer"
                )
            rigid_count_declared.append(int(rigid_count))

    assert shared is not None
    if rigid_count_declared and len(rigid_count_declared) != len(manifests):
        raise ValueError(
            "rigid_component_graph_count must be declared by every input manifest"
        )
    if rigid_count_declared:
        graph_policy = shared.get("rigid_component_graph_policy", "mode_bound")
        if graph_policy not in {"mode_bound", "shared_observation_topology"}:
            raise ValueError(
                "rigid_component_graph_policy must be mode_bound or "
                "shared_observation_topology"
            )
        shared["rigid_component_graph_count"] = (
            1
            if graph_policy == "shared_observation_topology"
            else sum(rigid_count_declared)
        )
    return shared


def _motion_fill_graph_paths(
    manifests: Sequence[tuple[Path, Mapping[str, Any]]],
    output_path: Path,
) -> list[dict[str, str]]:
    paths: list[dict[str, str]] = []
    for manifest_path, payload in manifests:
        parameters = payload["parameters"]
        value = parameters.get("motion_fill_graph_path")
        if value is None:
            if bool(parameters.get("motion_fill_enabled")):
                raise ValueError(
                    f"{manifest_path} enables motion fill without motion_fill_graph_path"
                )
            continue
        if not bool(parameters.get("motion_fill_enabled")):
            raise ValueError(
                f"{manifest_path} declares motion_fill_graph_path while motion fill "
                "is disabled"
            )
        graph_path = _resolve_artifact_path(
            manifest_path,
            value,
            "parameters.motion_fill_graph_path",
        )
        paths.append(
            {
                "manifest": _relative_to_output(manifest_path, output_path),
                "path": _relative_to_output(graph_path, output_path),
            }
        )
    return paths


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        assert temporary is not None
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def combine_modal_manifests(
    manifest_paths: Sequence[str | Path],
    output_path: str | Path,
) -> Path:
    """Combine solved mode entries and rebase their artifacts to one manifest."""

    if len(manifest_paths) < 2:
        raise ValueError("At least two input manifests are required")
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    manifests: list[tuple[Path, dict[str, Any]]] = []
    for value in manifest_paths:
        path = Path(value).expanduser().resolve()
        payload = _read_json_object(path)
        if payload.get("version") != 1:
            raise ValueError(f"{path} must be a version 1 modal manifest")
        if payload.get("point_type") != "foreground_gaussian_center":
            raise ValueError(
                f"{path} point_type must be 'foreground_gaussian_center'"
            )
        source_checkpoint = payload.get("source_checkpoint")
        if not isinstance(source_checkpoint, str) or not source_checkpoint:
            raise ValueError(f"{path} has invalid source_checkpoint")
        _required_string_list(path, payload, "source_view_configs")
        _required_string_list(path, payload, "source_modal_npzs")
        manifests.append((path, payload))

    first_path, first = manifests[0]
    for path, payload in manifests[1:]:
        for field in ("source_checkpoint", "source_view_configs"):
            if payload.get(field) != first.get(field):
                raise ValueError(
                    f"{path} {field} differs from {first_path}"
                )

    shared_parameters = _shared_parameters(manifests)
    motion_fill_graphs = _motion_fill_graph_paths(manifests, output)
    if motion_fill_graphs:
        shared_parameters["motion_fill_graph_paths"] = [
            item["path"] for item in motion_fill_graphs
        ]

    combined_modes: list[dict[str, Any]] = []
    source_mode_indices: list[dict[str, Any]] = []
    expected_topology: dict[str, np.ndarray] | None = None
    shared_rigid_graph_path: Path | None = None
    seen_mode_indices: set[int] = set()
    for path, payload in manifests:
        indices = _validated_mode_indices(path, payload)
        parameters = payload["parameters"]
        graph_count = parameters.get("rigid_component_graph_count")
        graph_policy = parameters.get(
            "rigid_component_graph_policy",
            "mode_bound",
        )
        if graph_count is not None:
            expected_graph_count = (
                1
                if graph_policy == "shared_observation_topology"
                else len(indices)
            )
            if int(graph_count) != expected_graph_count:
                raise ValueError(
                    f"{path} rigid_component_graph_count does not match "
                    f"{graph_policy} policy"
                )
        source_mode_indices.append(
            {
                "manifest": _relative_to_output(path, output),
                "mode_indices": list(indices),
            }
        )
        for raw_mode in payload["modes"]:
            mode_index = int(raw_mode["mode_index"])
            if mode_index in seen_mode_indices:
                raise ValueError(f"Duplicate mode index across manifests: {mode_index}")
            seen_mode_indices.add(mode_index)
            if graph_policy == "shared_observation_topology":
                mode_graph_path = _resolve_artifact_path(
                    path,
                    raw_mode.get("rigid_component_graph_path"),
                    "rigid_component_graph_path",
                )
                if shared_rigid_graph_path is None:
                    shared_rigid_graph_path = mode_graph_path
                elif mode_graph_path != shared_rigid_graph_path:
                    raise ValueError(
                        "Shared rigid graph paths differ across modes or manifests: "
                        f"{shared_rigid_graph_path} and {mode_graph_path}"
                    )
            topology = _load_observation_topology(path, raw_mode)
            observation_path = _resolve_artifact_path(
                path,
                raw_mode.get("observation_path"),
                "observation_path",
            )
            if expected_topology is None:
                expected_topology = topology
            else:
                _validate_same_topology(
                    topology,
                    expected_topology,
                    observation_path,
                )
            combined_modes.append(_rebase_mode(path, raw_mode, output))

    combined_modes.sort(key=lambda mode: int(mode["mode_index"]))
    combined_indices = [int(mode["mode_index"]) for mode in combined_modes]
    if combined_indices != list(range(len(combined_indices))):
        raise ValueError(
            "Combined mode indices must be the contiguous zero-based range; "
            f"got {combined_indices}"
        )

    source_modal_npzs = [
        payload.get("source_modal_npzs") for _, payload in manifests
    ]
    combined = deepcopy(first)
    combined.pop("incremental_extension", None)
    combined["mode_indices"] = combined_indices
    combined["parameters"] = shared_parameters
    combined["modes"] = combined_modes
    if all(value == source_modal_npzs[0] for value in source_modal_npzs[1:]):
        combined["source_modal_npzs"] = source_modal_npzs[0]
    else:
        combined.pop("source_modal_npzs", None)

    combination: dict[str, Any] = {
        "method": "ordered_manifest_concatenation_v1",
        "sources": source_mode_indices,
    }
    if not all(value == source_modal_npzs[0] for value in source_modal_npzs[1:]):
        combination["source_modal_npzs"] = [
            {
                "manifest": _relative_to_output(path, output),
                "paths": payload.get("source_modal_npzs"),
            }
            for path, payload in manifests
        ]
    if motion_fill_graphs:
        combination["motion_fill_graphs"] = motion_fill_graphs
    combined["manifest_combination"] = combination

    _write_json_atomic(output, combined)
    return output
