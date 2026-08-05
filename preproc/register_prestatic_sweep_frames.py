"""Register a denser canonical sweep into an existing pre-static COLMAP model.

The completed image-first pre-static run is an immutable seed.  This controller
copies its database with SQLite's backup API, adds only missing target-grid
frames, matches them to registered seed frames and temporal target neighbours,
and publishes a new version-1 static dataset containing only registered frames
from the requested target time grid.  Canonical RGB and mask bytes are never
decoded, resized, rotated, cropped, or rewritten.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from preproc.prepare_joint_colmap_video_dataset import (
    ModelCandidate,
    compute_colmap_scene_norm,
    export_camera_records,
    get_intrinsics_extrinsics,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
    resolve_executable,
    run_command,
)
from preproc.run_prestatic_pipeline import (
    FrameRecord,
    ReferenceSelection,
    SequenceConfig,
    ValidatedSequence,
    _canonical_json_identity,
    _ensure_unique_colmap_basenames,
    _files_have_same_content,
    _read_image,
    _read_json,
    _require_torch,
    _validate_binary_mask,
    _validate_rgb_image,
    _write_json_atomic,
    select_sweep_frames,
    validate_camera_group_assignments,
    validate_sequence,
)


CONFIG_FORMAT = "som_prestatic_registration"
CONFIG_VERSION = 1
STATE_FORMAT = "prestatic_registration_state"
STATE_VERSION = 1
READY_FORMAT = "static_colmap_dataset_ready"
READY_VERSION = 1
SOURCE_MANIFEST_FORMAT = "canonical_source_sequences"
SOURCE_MANIFEST_VERSION = 1
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
STAGES = (
    "validate_base",
    "stage_workspace",
    "extract_features",
    "import_matches",
    "register_images",
    "triangulate_points",
    "bundle_adjust",
    "validate_registration",
    "package_static_dataset",
    "export_reference_cameras",
    "validate_static_ready",
)


@dataclass(frozen=True)
class RegistrationConfig:
    config_path: Path
    scene_id: str
    scene_root: Path
    run_id: str
    base_prestatic_run_id: str
    target_sweep_fps_hz: float
    min_target_registration_ratio: float
    seed_neighbors_per_side: int
    target_neighbor_radius: int
    colmap_command: str


@dataclass(frozen=True)
class BaseSeedFrame:
    source_index: int
    staged_relative_name: str


@dataclass(frozen=True)
class TargetFrame:
    staged_relative_name: str
    source_frame: FrameRecord
    target_time_sec: float
    reuses_seed: bool

    @property
    def staged_name(self) -> str:
        return Path(self.staged_relative_name).name


@dataclass(frozen=True)
class RegistrationPaths:
    run_dir: Path
    reports_dir: Path
    workspace_dir: Path
    workspace_images: Path
    seed_database: Path
    features_database: Path
    database_path: Path
    base_model_dir: Path
    registered_model_dir: Path
    triangulated_model_dir: Path
    final_model_dir: Path
    new_image_list: Path
    pair_list: Path
    dataset_dir: Path
    references_dir: Path
    resolved_config_path: Path
    state_path: Path


@dataclass(frozen=True)
class BaseContext:
    run_dir: Path
    ready_path: Path
    ready: Mapping[str, Any]
    resolved: Mapping[str, Any]
    source_manifest_path: Path
    source_manifest: Mapping[str, Any]
    sweep: ValidatedSequence
    static_views: tuple[ValidatedSequence, ...]
    references: tuple[ReferenceSelection, ...]
    base_seed_frames: tuple[BaseSeedFrame, ...]
    registered_seed_by_source_index: Mapping[int, str]
    workspace_images: Path
    database_path: Path
    selected_model_dir: Path
    sweep_camera_id: int
    camera_group_by_source: Mapping[str, str]
    artifact_hashes: Mapping[str, str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _validate_keys(
    payload: Mapping[str, Any], label: str, required: set[str]
) -> None:
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")
    if unknown:
        raise ValueError(f"{label} contains unsupported keys: {unknown}")


def _parse_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{label} must start with a letter or digit and contain only letters, "
            "digits, underscores, or hyphens"
        )
    return value


def _parse_positive_float(value: Any, label: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must not exceed {maximum}")
    return result


def _parse_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_registration_config(path: str | Path) -> RegistrationConfig:
    config_path = Path(path).expanduser().resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    values = _require_mapping(payload, str(config_path))
    _validate_keys(
        values,
        str(config_path),
        {
            "format",
            "version",
            "scene_id",
            "scene_root",
            "run_id",
            "base_prestatic_run_id",
            "target_sweep_fps_hz",
            "min_target_registration_ratio",
            "seed_neighbors_per_side",
            "target_neighbor_radius",
            "colmap_command",
        },
    )
    if (
        values["format"] != CONFIG_FORMAT
        or type(values["version"]) is not int
        or values["version"] != CONFIG_VERSION
    ):
        raise ValueError(
            f"{config_path} must use format={CONFIG_FORMAT!r}, version={CONFIG_VERSION}"
        )
    scene_id = _parse_identifier(values["scene_id"], "scene_id")
    run_id = _parse_identifier(values["run_id"], "run_id")
    base_run_id = _parse_identifier(
        values["base_prestatic_run_id"], "base_prestatic_run_id"
    )
    if run_id == base_run_id:
        raise ValueError("run_id must differ from base_prestatic_run_id")
    scene_root_value = values["scene_root"]
    if not isinstance(scene_root_value, str) or not scene_root_value:
        raise ValueError("scene_root must be a non-empty path")
    scene_root = Path(scene_root_value).expanduser().resolve()
    if scene_root.name != scene_id:
        raise ValueError(
            f"scene_root basename {scene_root.name!r} does not match scene_id {scene_id!r}"
        )
    command = values["colmap_command"]
    if not isinstance(command, str) or not command:
        raise ValueError("colmap_command must be a non-empty executable name or path")
    return RegistrationConfig(
        config_path=config_path,
        scene_id=scene_id,
        scene_root=scene_root,
        run_id=run_id,
        base_prestatic_run_id=base_run_id,
        target_sweep_fps_hz=_parse_positive_float(
            values["target_sweep_fps_hz"], "target_sweep_fps_hz"
        ),
        min_target_registration_ratio=_parse_positive_float(
            values["min_target_registration_ratio"],
            "min_target_registration_ratio",
            maximum=1.0,
        ),
        seed_neighbors_per_side=_parse_positive_int(
            values["seed_neighbors_per_side"], "seed_neighbors_per_side"
        ),
        target_neighbor_radius=_parse_positive_int(
            values["target_neighbor_radius"], "target_neighbor_radius"
        ),
        colmap_command=command,
    )


def registration_paths(config: RegistrationConfig) -> RegistrationPaths:
    run_dir = config.scene_root / "shared" / "preprocessing" / config.run_id
    reports = run_dir / "reports"
    workspace = run_dir / "colmap_workspace"
    return RegistrationPaths(
        run_dir=run_dir,
        reports_dir=reports,
        workspace_dir=workspace,
        workspace_images=workspace / "images",
        seed_database=workspace / "database.seed.db",
        features_database=workspace / "database.features.db",
        database_path=workspace / "database.db",
        base_model_dir=workspace / "base_model",
        registered_model_dir=workspace / "registered_model",
        triangulated_model_dir=workspace / "triangulated_model",
        final_model_dir=workspace / "final_model",
        new_image_list=workspace / "new_images.txt",
        pair_list=workspace / "temporal_pairs.txt",
        dataset_dir=run_dir / "sweep_colmap_dataset",
        references_dir=run_dir / "references",
        resolved_config_path=run_dir / "resolved_config.json",
        state_path=reports / "pipeline_state.json",
    )


def select_registration_targets(
    sweep: ValidatedSequence,
    target_fps_hz: float,
    base_seed_names_by_source_index: Mapping[int, str],
) -> tuple[TargetFrame, ...]:
    sampled = select_sweep_frames(sweep, target_fps_hz)
    width = max(6, len(str(max(0, len(sweep.frames) - 1))))
    targets: list[TargetFrame] = []
    names: list[str] = []
    for selection in sampled:
        source_index = selection.source_frame.source_index
        seed_name = base_seed_names_by_source_index.get(source_index)
        if seed_name is None:
            relative_name = (
                f"sweep/dense_sweep_src_{source_index:0{width}d}.png"
            )
            reuses_seed = False
        else:
            normalized = str(seed_name).replace("\\", "/")
            if not normalized.startswith("sweep/") or Path(normalized).suffix.lower() != ".png":
                raise ValueError(
                    f"Invalid base seed staged name for source index {source_index}: "
                    f"{seed_name!r}"
                )
            relative_name = normalized
            reuses_seed = True
        names.append(relative_name)
        targets.append(
            TargetFrame(
                staged_relative_name=relative_name,
                source_frame=selection.source_frame,
                target_time_sec=selection.target_time_sec,
                reuses_seed=reuses_seed,
            )
        )
    if len(set(names)) != len(names):
        raise ValueError("Target sweep selection produced duplicate staged names")
    _ensure_unique_colmap_basenames(names)
    return tuple(targets)


def build_temporal_match_pairs(
    targets: Sequence[TargetFrame],
    registered_seed_names_by_source_index: Mapping[int, str],
    *,
    seed_neighbors_per_side: int,
    target_neighbor_radius: int,
) -> tuple[tuple[str, str], ...]:
    if seed_neighbors_per_side <= 0 or target_neighbor_radius <= 0:
        raise ValueError("Temporal pair radii must be positive")
    seed_items = sorted(
        (int(index), str(name).replace("\\", "/"))
        for index, name in registered_seed_names_by_source_index.items()
    )
    if not seed_items:
        raise ValueError("No successfully registered base sweep seed is available")
    seed_indices = [item[0] for item in seed_items]
    registered_seed_names = {name for _, name in seed_items}
    registration_target_names = {
        target.staged_relative_name
        for target in targets
        if target.staged_relative_name not in registered_seed_names
    }
    pairs: set[tuple[str, str]] = set()

    def add(left: str, right: str) -> None:
        if left == right:
            return
        pairs.add(tuple(sorted((left, right))))

    for target in targets:
        if target.staged_relative_name not in registration_target_names:
            continue
        source_index = target.source_frame.source_index
        insert_at = bisect.bisect_left(seed_indices, source_index)
        neighbours = seed_items[
            max(0, insert_at - seed_neighbors_per_side) : insert_at
        ] + seed_items[insert_at : insert_at + seed_neighbors_per_side]
        if not neighbours:
            raise RuntimeError(
                f"No registered seed neighbour for target source index {source_index}"
            )
        for _, seed_name in neighbours:
            add(target.staged_relative_name, seed_name)

    for target_index, target in enumerate(targets):
        begin = max(0, target_index - target_neighbor_radius)
        end = min(len(targets), target_index + target_neighbor_radius + 1)
        for neighbour in targets[begin:end]:
            if (
                target.staged_relative_name in registration_target_names
                or neighbour.staged_relative_name in registration_target_names
            ):
                add(target.staged_relative_name, neighbour.staged_relative_name)
    if registration_target_names and not pairs:
        raise RuntimeError(
            "No temporal match pair was generated for unregistered target frames"
        )
    covered = {name for pair in pairs for name in pair} & registration_target_names
    missing = sorted(registration_target_names - covered)
    if missing:
        raise RuntimeError(f"New target frames lack temporal match pairs: {missing[:5]}")
    return tuple(sorted(pairs))


def build_feature_extractor_command(
    colmap: str,
    database_path: Path,
    image_path: Path,
    image_list_path: Path,
    camera_id: int,
) -> list[str]:
    return [
        colmap,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--image_list_path",
        str(image_list_path),
        "--ImageReader.existing_camera_id",
        str(camera_id),
    ]


def build_matches_importer_command(
    colmap: str, database_path: Path, pair_list_path: Path
) -> list[str]:
    return [
        colmap,
        "matches_importer",
        "--database_path",
        str(database_path),
        "--match_list_path",
        str(pair_list_path),
        "--match_type",
        "pairs",
    ]


def build_image_registrator_command(
    colmap: str,
    database_path: Path,
    input_model: Path,
    output_model: Path,
) -> list[str]:
    return [
        colmap,
        "image_registrator",
        "--database_path",
        str(database_path),
        "--input_path",
        str(input_model),
        "--output_path",
        str(output_model),
    ]


def build_point_triangulator_command(
    colmap: str,
    database_path: Path,
    image_path: Path,
    input_model: Path,
    output_model: Path,
) -> list[str]:
    return [
        colmap,
        "point_triangulator",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--input_path",
        str(input_model),
        "--output_path",
        str(output_model),
    ]


def build_bundle_adjuster_command(
    colmap: str, input_model: Path, output_model: Path
) -> list[str]:
    return [
        colmap,
        "bundle_adjuster",
        "--input_path",
        str(input_model),
        "--output_path",
        str(output_model),
        "--BundleAdjustment.refine_focal_length",
        "0",
        "--BundleAdjustment.refine_principal_point",
        "0",
        "--BundleAdjustment.refine_extra_params",
        "0",
    ]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def backup_sqlite_database(source_path: Path, destination_path: Path) -> None:
    if not source_path.is_file() or source_path.is_symlink():
        raise FileNotFoundError(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(destination_path)
    temporary = destination_path.with_name(
        f".{destination_path.name}.{os.getpid()}.tmp"
    )
    temporary.unlink(missing_ok=True)
    try:
        source_uri = f"file:{source_path.as_posix()}?mode=ro"
        with sqlite3.connect(source_uri, uri=True) as source, sqlite3.connect(
            temporary
        ) as destination:
            source.backup(destination)
            result = destination.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                raise RuntimeError(
                    f"SQLite backup integrity check failed for {temporary}: {result}"
                )
        os.replace(temporary, destination_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_manifest_sequence(
    payload: Any,
    *,
    is_static: bool,
) -> ValidatedSequence:
    record = _require_mapping(payload, "source manifest sequence")
    _validate_keys(
        record,
        "source manifest sequence",
        {
            "view_id",
            "image_dir",
            "mask_dir",
            "fps_hz",
            "camera_group",
            "frame_count",
            "image_width",
            "image_height",
            "image_extension",
            "frame_names",
            "source_identity",
        },
    )
    sequence_config = SequenceConfig(
        view_id=_parse_identifier(record["view_id"], "source view_id"),
        image_dir=Path(str(record["image_dir"])).expanduser().resolve(strict=True),
        mask_dir=Path(str(record["mask_dir"])).expanduser().resolve(strict=True),
        fps_hz=_parse_positive_float(record["fps_hz"], "source fps_hz"),
        camera_group=_parse_identifier(
            record["camera_group"], "source camera_group"
        ),
        reference_policy="middle" if is_static else None,
    )
    sequence = validate_sequence(sequence_config)
    expected_names = record["frame_names"]
    if not isinstance(expected_names, list) or any(
        not isinstance(name, str) for name in expected_names
    ):
        raise ValueError("Source manifest frame_names must be a list of strings")
    actual_names = [frame.frame_name for frame in sequence.frames]
    if expected_names != actual_names:
        raise ValueError(
            f"Canonical source names changed for {sequence.config.view_id}"
        )
    expected = {
        "frame_count": len(sequence.frames),
        "image_width": sequence.image_width,
        "image_height": sequence.image_height,
        "image_extension": sequence.image_extension,
        "source_identity": sequence.source_identity,
    }
    mismatches = {
        key: (record[key], value)
        for key, value in expected.items()
        if record[key] != value
    }
    if mismatches:
        raise ValueError(
            f"Canonical source sequence changed for {sequence.config.view_id}: "
            f"{mismatches}"
        )
    return sequence


def _require_complete_model(model_dir: Path, label: str) -> None:
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise ValueError(f"{label} is not a regular directory: {model_dir}")
    missing = [
        str(model_dir / name)
        for name in ("cameras.bin", "images.bin", "points3D.bin")
        if not (model_dir / name).is_file()
        or (model_dir / name).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(f"{label} is incomplete: {missing}")


def _validate_sqlite_database(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    uri = f"file:{path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        result = connection.execute("PRAGMA quick_check").fetchone()
        if result is None or result[0] != "ok":
            raise RuntimeError(f"SQLite integrity check failed for {path}: {result}")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"cameras", "images", "keypoints", "descriptors"}
        if not required.issubset(tables):
            raise ValueError(
                f"COLMAP database lacks tables {sorted(required - tables)}: {path}"
            )


def _load_base_context(config: RegistrationConfig) -> BaseContext:
    base_run = (
        config.scene_root
        / "shared"
        / "preprocessing"
        / config.base_prestatic_run_id
    )
    ready_path = base_run / "reports" / "STATIC_DATASET_READY.json"
    ready = _require_mapping(_read_json(ready_path), "base STATIC_DATASET_READY")
    if (
        ready.get("format") != READY_FORMAT
        or ready.get("version") != READY_VERSION
        or ready.get("scene_id") != config.scene_id
        or ready.get("run_id") != config.base_prestatic_run_id
    ):
        raise ValueError(f"Incompatible base READY marker: {ready_path}")
    if Path(str(ready.get("static_dataset"))).expanduser().resolve(strict=True) != (
        base_run / "sweep_colmap_dataset"
    ).resolve(strict=True):
        raise ValueError("Base READY static_dataset path is outside the base run")
    if Path(str(ready.get("reference_cameras"))).expanduser().resolve(strict=True) != (
        base_run / "references" / "reference_cameras.json"
    ).resolve(strict=True):
        raise ValueError("Base READY reference_cameras path is unexpected")
    base_state = _require_mapping(
        _read_json(base_run / "reports" / "pipeline_state.json"),
        "base pipeline state",
    )
    if (
        base_state.get("format") != "prestatic_pipeline_state"
        or base_state.get("version") != 1
        or base_state.get("completed_stages")
        != [
            "validate_inputs",
            "stage_colmap_images",
            "run_colmap",
            "select_model",
            "package_static_dataset",
            "export_reference_cameras",
            "validate_static_ready",
        ]
    ):
        raise ValueError("Base pre-static pipeline is not completely finalized")
    resolved_path = base_run / "resolved_config.json"
    resolved = _require_mapping(_read_json(resolved_path), "base resolved config")
    if (
        resolved.get("format") != "som_prestatic_pipeline"
        or resolved.get("version") != 1
        or resolved.get("scene_id") != config.scene_id
        or resolved.get("run_id") != config.base_prestatic_run_id
    ):
        raise ValueError(f"Incompatible base pre-static config: {resolved_path}")
    transforms = _require_mapping(resolved.get("transforms"), "base transforms")
    if any(
        transforms.get(key) is not False
        for key in ("video_decode", "rotate", "crop", "resize")
    ):
        raise ValueError("Base pre-static run did not preserve canonical image bytes")

    source_manifest_path = Path(str(ready.get("source_manifest"))).expanduser().resolve(strict=True)
    resolved_manifest_path = (
        Path(str(resolved.get("source_manifest"))).expanduser().resolve(strict=True)
    )
    if source_manifest_path != resolved_manifest_path:
        raise ValueError("Base READY and resolved config reference different source manifests")
    source_manifest = _require_mapping(
        _read_json(source_manifest_path), "canonical source manifest"
    )
    if (
        source_manifest.get("format") != SOURCE_MANIFEST_FORMAT
        or source_manifest.get("version") != SOURCE_MANIFEST_VERSION
        or source_manifest.get("scene_id") != config.scene_id
    ):
        raise ValueError(f"Incompatible source manifest: {source_manifest_path}")
    sweep = _validate_manifest_sequence(source_manifest.get("sweep"), is_static=False)
    static_payload = source_manifest.get("static_views")
    if not isinstance(static_payload, list) or not static_payload:
        raise ValueError("Canonical source manifest contains no static views")
    static_views = tuple(
        _validate_manifest_sequence(item, is_static=True) for item in static_payload
    )

    reference_path = base_run / "reports" / "reference_selection.json"
    reference_payload = _require_mapping(
        _read_json(reference_path), "base reference selection"
    )
    if (
        reference_payload.get("format") != "fixed_reference_selection"
        or reference_payload.get("version") != 1
    ):
        raise ValueError(f"Incompatible base reference selection: {reference_path}")
    reference_values = reference_payload.get("references")
    if not isinstance(reference_values, list):
        raise ValueError("Base reference selection must contain a references list")
    reference_by_view = {
        str(item.get("view_id")): _require_mapping(item, "reference record")
        for item in reference_values
        if isinstance(item, Mapping)
    }
    if len(reference_by_view) != len(reference_values):
        raise ValueError("Base reference selection repeats or invalidates a view_id")
    references: list[ReferenceSelection] = []
    for view in static_views:
        record = reference_by_view.get(view.config.view_id)
        if record is None:
            raise ValueError(
                f"Base reference selection is missing {view.config.view_id!r}"
            )
        source_index = record.get("source_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or source_index >= len(view.frames)
        ):
            raise ValueError(f"Invalid reference source index for {view.config.view_id}")
        source_frame = view.frames[source_index]
        if record.get("source_frame_name") != source_frame.frame_name:
            raise ValueError(
                f"Reference source index/name mismatch for {view.config.view_id}"
            )
        staged_name = record.get("staged_relative_name")
        if not isinstance(staged_name, str) or not staged_name:
            raise ValueError(f"Invalid staged reference name for {view.config.view_id}")
        if record.get("camera_group") != view.config.camera_group:
            raise ValueError(f"Reference camera group changed for {view.config.view_id}")
        references.append(
            ReferenceSelection(
                view_id=view.config.view_id,
                camera_group=view.config.camera_group,
                staged_relative_name=staged_name.replace("\\", "/"),
                source_frame=source_frame,
            )
        )
    if set(reference_by_view) != {view.config.view_id for view in static_views}:
        raise ValueError("Base references and source static views differ")

    seed_selection_path = base_run / "reports" / "sweep_frame_selection.json"
    seed_payload = _require_mapping(
        _read_json(seed_selection_path), "base sweep selection"
    )
    if (
        seed_payload.get("format") != "time_sampled_sweep_frames"
        or seed_payload.get("version") != 1
        or seed_payload.get("view_id") != sweep.config.view_id
    ):
        raise ValueError(f"Incompatible base sweep selection: {seed_selection_path}")
    selections = seed_payload.get("selection")
    if not isinstance(selections, list) or not selections:
        raise ValueError("Base sweep selection contains no seed frames")
    base_seed_frames: list[BaseSeedFrame] = []
    seen_source_indices: set[int] = set()
    for item in selections:
        record = _require_mapping(item, "base sweep selection record")
        source_index = record.get("source_index")
        staged_name = record.get("staged_name")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or source_index >= len(sweep.frames)
            or source_index in seen_source_indices
        ):
            raise ValueError(f"Invalid base sweep source index: {source_index!r}")
        if not isinstance(staged_name, str) or Path(staged_name).name != staged_name:
            raise ValueError(f"Invalid base sweep staged name: {staged_name!r}")
        source_frame = sweep.frames[source_index]
        if record.get("source_frame_name") != source_frame.frame_name:
            raise ValueError(f"Base sweep source name mismatch at {source_index}")
        seen_source_indices.add(source_index)
        base_seed_frames.append(
            BaseSeedFrame(source_index, f"sweep/{staged_name}")
        )

    workspace_images = base_run / "colmap_workspace" / "images"
    database_path = base_run / "colmap_workspace" / "database.db"
    _validate_sqlite_database(database_path)
    selected_report_path = base_run / "reports" / "selected_model.json"
    selected_report = _require_mapping(
        _read_json(selected_report_path), "base selected model"
    )
    selected_model_dir = (
        Path(str(selected_report.get("model_dir"))).expanduser().resolve(strict=True)
    )
    sparse_root = (base_run / "colmap_workspace" / "sparse_models").resolve(strict=True)
    if selected_model_dir.parent != sparse_root:
        raise ValueError(
            f"Base selected model is outside its sparse-model directory: {selected_model_dir}"
        )
    _require_complete_model(selected_model_dir, "base selected model")
    model_images = read_images_binary(selected_model_dir / "images.bin")
    registered_names = {
        image.name.replace("\\", "/") for image in model_images.values()
    }
    reference_names = {reference.staged_relative_name for reference in references}
    missing_references = sorted(reference_names - registered_names)
    if missing_references:
        raise RuntimeError(
            f"Base selected model is missing reference frames: {missing_references}"
        )
    registered_seed_by_source_index = {
        seed.source_index: seed.staged_relative_name
        for seed in base_seed_frames
        if seed.staged_relative_name in registered_names
    }
    if not registered_seed_by_source_index:
        raise RuntimeError("Base selected model registered no sweep seed frame")
    sweep_camera_ids = {
        int(image.camera_id)
        for image in model_images.values()
        if image.name.replace("\\", "/")
        in set(registered_seed_by_source_index.values())
    }
    if len(sweep_camera_ids) != 1:
        raise RuntimeError(
            f"Base registered sweep seeds use multiple camera IDs: {sweep_camera_ids}"
        )
    sweep_camera_id = next(iter(sweep_camera_ids))
    with sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True) as connection:
        camera_row = connection.execute(
            "SELECT camera_id FROM cameras WHERE camera_id=?", (sweep_camera_id,)
        ).fetchone()
        if camera_row is None:
            raise RuntimeError(
                f"Base COLMAP database lacks sweep camera ID {sweep_camera_id}"
            )

    if not workspace_images.is_dir() or workspace_images.is_symlink():
        raise FileNotFoundError(workspace_images)
    for seed in base_seed_frames:
        source = sweep.frames[seed.source_index].image_path
        staged = workspace_images / seed.staged_relative_name
        if not staged.is_file() or not _files_have_same_content(staged, source):
            raise ValueError(
                f"Base staged sweep image differs from canonical source: {staged}"
            )
    for reference in references:
        staged = workspace_images / reference.staged_relative_name
        if not staged.is_file() or not _files_have_same_content(
            staged, reference.source_frame.image_path
        ):
            raise ValueError(
                f"Base staged reference differs from canonical source: {staged}"
            )

    artifact_hashes = {
        "database.db": _sha256_file(database_path),
        "cameras.bin": _sha256_file(selected_model_dir / "cameras.bin"),
        "images.bin": _sha256_file(selected_model_dir / "images.bin"),
        "points3D.bin": _sha256_file(selected_model_dir / "points3D.bin"),
        "sweep_frame_selection.json": _sha256_file(seed_selection_path),
        "reference_selection.json": _sha256_file(reference_path),
    }
    camera_group_by_source = {
        "sweep": sweep.config.camera_group,
        **{
            view.config.view_id: view.config.camera_group
            for view in static_views
        },
    }
    return BaseContext(
        run_dir=base_run,
        ready_path=ready_path,
        ready=ready,
        resolved=resolved,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        sweep=sweep,
        static_views=static_views,
        references=tuple(references),
        base_seed_frames=tuple(base_seed_frames),
        registered_seed_by_source_index=registered_seed_by_source_index,
        workspace_images=workspace_images,
        database_path=database_path,
        selected_model_dir=selected_model_dir,
        sweep_camera_id=sweep_camera_id,
        camera_group_by_source=camera_group_by_source,
        artifact_hashes=artifact_hashes,
    )


def _resolved_config_payload(
    config: RegistrationConfig,
    paths: RegistrationPaths,
    base: BaseContext,
    targets: Sequence[TargetFrame],
    pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "format": CONFIG_FORMAT,
        "version": CONFIG_VERSION,
        "source_config": str(config.config_path),
        "scene_id": config.scene_id,
        "scene_root": str(config.scene_root),
        "run_id": config.run_id,
        "run_dir": str(paths.run_dir),
        "base_prestatic_run_id": config.base_prestatic_run_id,
        "base_run_dir": str(base.run_dir),
        "source_manifest": str(base.source_manifest_path),
        "target_sweep_fps_hz": config.target_sweep_fps_hz,
        "min_target_registration_ratio": config.min_target_registration_ratio,
        "seed_neighbors_per_side": config.seed_neighbors_per_side,
        "target_neighbor_radius": config.target_neighbor_radius,
        "colmap_command": config.colmap_command,
        "source_sweep_fps_hz": base.sweep.config.fps_hz,
        "source_sweep_frame_count": len(base.sweep.frames),
        "target_frame_count": len(targets),
        "reused_seed_target_count": sum(target.reuses_seed for target in targets),
        "new_target_count": sum(not target.reuses_seed for target in targets),
        "temporal_pair_count": len(pairs),
        "sweep_camera_id": base.sweep_camera_id,
        "base_artifact_hashes": dict(base.artifact_hashes),
        "transforms": {
            "video_decode": False,
            "rotate": False,
            "crop": False,
            "resize": False,
        },
    }


def _target_selection_payload(
    base: BaseContext,
    config: RegistrationConfig,
    targets: Sequence[TargetFrame],
) -> dict[str, Any]:
    return {
        "format": "time_sampled_sweep_frames",
        "version": 1,
        "view_id": base.sweep.config.view_id,
        "source_fps_hz": base.sweep.config.fps_hz,
        "target_fps_hz": config.target_sweep_fps_hz,
        "source_frame_count": len(base.sweep.frames),
        "selected_frame_count": len(targets),
        "selection": [
            {
                "staged_name": target.staged_name,
                "staged_relative_name": target.staged_relative_name,
                "source_frame_name": target.source_frame.frame_name,
                "source_index": target.source_frame.source_index,
                "target_time_sec": target.target_time_sec,
                "source_time_sec": target.source_frame.time_sec,
                "time_error_sec": (
                    target.source_frame.time_sec - target.target_time_sec
                ),
                "source_image": str(target.source_frame.image_path),
                "source_mask": str(target.source_frame.mask_path),
                "reuses_base_seed": target.reuses_seed,
            }
            for target in targets
        ],
    }


def _initial_state(config_identity: str) -> dict[str, Any]:
    return {
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        "config_identity": config_identity,
        "completed_stages": [],
        "stage_completed_at": {},
    }


def _load_state(paths: RegistrationPaths, config_identity: str) -> dict[str, Any]:
    state = _read_json(paths.state_path)
    if (
        not isinstance(state, dict)
        or state.get("format") != STATE_FORMAT
        or state.get("version") != STATE_VERSION
        or state.get("config_identity") != config_identity
    ):
        raise ValueError(
            f"Existing registration state is incompatible: {paths.state_path}. "
            "Use a new run_id."
        )
    completed = state.get("completed_stages")
    if not isinstance(completed, list) or completed != list(STAGES[: len(completed)]):
        raise ValueError(
            f"Completed registration stages are not a contiguous prefix: {completed}"
        )
    return state


def _initialize_or_load_state(
    paths: RegistrationPaths, config_identity: str
) -> dict[str, Any]:
    if paths.run_dir.exists():
        if not paths.run_dir.is_dir() or paths.run_dir.is_symlink():
            raise ValueError(f"Registration run path is invalid: {paths.run_dir}")
        if not paths.state_path.is_file():
            raise FileExistsError(
                f"Run directory exists without pipeline state: {paths.run_dir}"
            )
        return _load_state(paths, config_identity)
    paths.run_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{paths.run_dir.name}.",
            suffix=".init.tmp",
            dir=str(paths.run_dir.parent),
        )
    )
    state = _initial_state(config_identity)
    try:
        reports = temporary / "reports"
        reports.mkdir()
        _write_json_atomic(reports / paths.state_path.name, state)
        os.replace(temporary, paths.run_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return state


def _complete_stage(
    paths: RegistrationPaths, state: dict[str, Any], stage: str
) -> None:
    completed = list(state["completed_stages"])
    expected = STAGES[len(completed)]
    if stage != expected:
        raise RuntimeError(f"Cannot complete stage {stage!r}; expected {expected!r}")
    completed.append(stage)
    state["completed_stages"] = completed
    completed_at = dict(state.get("stage_completed_at", {}))
    completed_at[stage] = _utc_now()
    state["stage_completed_at"] = completed_at
    _write_json_atomic(paths.state_path, state)


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                if "\n" in line or "\r" in line:
                    raise ValueError(f"Manifest line contains a newline: {line!r}")
                handle.write(line)
                handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_workspace(
    paths: RegistrationPaths,
    base: BaseContext,
    targets: Sequence[TargetFrame],
    pairs: Sequence[tuple[str, str]],
) -> None:
    if not paths.workspace_images.is_dir() or paths.workspace_images.is_symlink():
        raise ValueError(f"Invalid expanded image stage: {paths.workspace_images}")
    for target in targets:
        staged = paths.workspace_images / target.staged_relative_name
        if not staged.is_file() or not _files_have_same_content(
            staged, target.source_frame.image_path
        ):
            raise ValueError(
                f"Expanded target differs from canonical source: {staged}"
            )
    for reference in base.references:
        staged = paths.workspace_images / reference.staged_relative_name
        if not staged.is_file() or not _files_have_same_content(
            staged, reference.source_frame.image_path
        ):
            raise ValueError(f"Expanded reference differs from source: {staged}")
    _validate_sqlite_database(paths.seed_database)
    _require_complete_model(paths.base_model_dir, "staged base model")
    new_names = [
        target.staged_relative_name for target in targets if not target.reuses_seed
    ]
    if paths.new_image_list.read_text(encoding="utf-8").splitlines() != new_names:
        raise ValueError(f"New-image list changed: {paths.new_image_list}")
    pair_lines = [f"{left} {right}" for left, right in pairs]
    if paths.pair_list.read_text(encoding="utf-8").splitlines() != pair_lines:
        raise ValueError(f"Temporal pair list changed: {paths.pair_list}")


def _stage_workspace(
    paths: RegistrationPaths,
    base: BaseContext,
    targets: Sequence[TargetFrame],
    pairs: Sequence[tuple[str, str]],
) -> None:
    if paths.workspace_dir.exists() or paths.workspace_dir.is_symlink():
        _validate_workspace(paths, base, targets, pairs)
        return
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".colmap_workspace.", suffix=".tmp", dir=str(paths.run_dir)
        )
    )
    try:
        images = temporary / "images"
        shutil.copytree(base.workspace_images, images)
        for target in targets:
            destination = images / target.staged_relative_name
            if target.reuses_seed:
                if not destination.is_file() or not _files_have_same_content(
                    destination, target.source_frame.image_path
                ):
                    raise ValueError(
                        f"Reused seed does not match target source: {destination}"
                    )
            else:
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target.source_frame.image_path, destination)
        shutil.copytree(base.selected_model_dir, temporary / "base_model")
        backup_sqlite_database(base.database_path, temporary / "database.seed.db")
        _write_lines(
            temporary / "new_images.txt",
            [
                target.staged_relative_name
                for target in targets
                if not target.reuses_seed
            ],
        )
        _write_lines(
            temporary / "temporal_pairs.txt",
            [f"{left} {right}" for left, right in pairs],
        )
        os.replace(temporary, paths.workspace_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _validate_workspace(paths, base, targets, pairs)


def _database_image_rows(path: Path) -> dict[str, tuple[int, int]]:
    _validate_sqlite_database(path)
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        return {
            str(name).replace("\\", "/"): (int(image_id), int(camera_id))
            for image_id, name, camera_id in connection.execute(
                "SELECT image_id, name, camera_id FROM images"
            )
        }


def _validate_target_database_images(
    path: Path,
    targets: Sequence[TargetFrame],
    sweep_camera_id: int,
    *,
    require_new_features: bool,
) -> None:
    rows = _database_image_rows(path)
    missing = sorted(
        target.staged_relative_name
        for target in targets
        if target.staged_relative_name not in rows
    )
    if missing:
        raise RuntimeError(f"COLMAP database lacks target images: {missing[:5]}")
    wrong_camera = {
        target.staged_relative_name: rows[target.staged_relative_name][1]
        for target in targets
        if rows[target.staged_relative_name][1] != sweep_camera_id
    }
    if wrong_camera:
        raise RuntimeError(
            f"Target images do not reuse sweep camera ID {sweep_camera_id}: "
            f"{dict(list(wrong_camera.items())[:5])}"
        )
    if not require_new_features:
        return
    new_ids = [
        rows[target.staged_relative_name][0]
        for target in targets
        if not target.reuses_seed
    ]
    if not new_ids:
        return
    placeholders = ",".join("?" for _ in new_ids)
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        keypoint_ids = {
            int(row[0])
            for row in connection.execute(
                f"SELECT image_id FROM keypoints WHERE image_id IN ({placeholders}) "
                "AND rows > 0",
                new_ids,
            )
        }
        descriptor_ids = {
            int(row[0])
            for row in connection.execute(
                f"SELECT image_id FROM descriptors WHERE image_id IN ({placeholders}) "
                "AND rows > 0",
                new_ids,
            )
        }
    missing_features = sorted(set(new_ids) - keypoint_ids - descriptor_ids)
    incomplete_features = sorted(
        (set(new_ids) - keypoint_ids) | (set(new_ids) - descriptor_ids)
    )
    if missing_features or incomplete_features:
        raise RuntimeError(
            "Feature extraction did not produce keypoints and descriptors for "
            f"new image IDs: {incomplete_features[:8]}"
        )


def _run_database_stage(
    input_database: Path,
    output_database: Path,
    command_builder: Any | None,
    log_path: Path,
) -> None:
    if output_database.exists() or output_database.is_symlink():
        _validate_sqlite_database(output_database)
        return
    temporary = output_database.with_name(
        f".{output_database.name}.{os.getpid()}.tmp"
    )
    temporary.unlink(missing_ok=True)
    backup_sqlite_database(input_database, temporary)
    try:
        if command_builder is not None:
            run_command(command_builder(temporary), log_path)
        _validate_sqlite_database(temporary)
        os.replace(temporary, output_database)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _extract_features_stage(
    colmap: str,
    paths: RegistrationPaths,
    base: BaseContext,
    targets: Sequence[TargetFrame],
) -> None:
    new_targets = [target for target in targets if not target.reuses_seed]
    builder = None
    if new_targets:
        builder = lambda database: build_feature_extractor_command(
            colmap,
            database,
            paths.workspace_images,
            paths.new_image_list,
            base.sweep_camera_id,
        )
    _run_database_stage(
        paths.seed_database,
        paths.features_database,
        builder,
        paths.reports_dir / "commands.log",
    )
    _validate_target_database_images(
        paths.features_database,
        targets,
        base.sweep_camera_id,
        require_new_features=True,
    )


def _import_matches_stage(
    colmap: str,
    paths: RegistrationPaths,
    base: BaseContext,
    targets: Sequence[TargetFrame],
    pairs: Sequence[tuple[str, str]],
) -> None:
    builder = None
    if pairs:
        builder = lambda database: build_matches_importer_command(
            colmap, database, paths.pair_list
        )
    _run_database_stage(
        paths.features_database,
        paths.database_path,
        builder,
        paths.reports_dir / "commands.log",
    )
    _validate_target_database_images(
        paths.database_path,
        targets,
        base.sweep_camera_id,
        require_new_features=True,
    )


def _run_model_stage(
    output_model: Path,
    command_builder: Any,
    log_path: Path,
) -> None:
    if output_model.exists() or output_model.is_symlink():
        _require_complete_model(output_model, "existing COLMAP stage model")
        return
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_model.name}.",
            suffix=".tmp",
            dir=str(output_model.parent),
        )
    )
    try:
        run_command(command_builder(temporary), log_path)
        _require_complete_model(temporary, "temporary COLMAP stage model")
        os.replace(temporary, output_model)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _register_images_stage(
    colmap: str, paths: RegistrationPaths
) -> None:
    _run_model_stage(
        paths.registered_model_dir,
        lambda output: build_image_registrator_command(
            colmap,
            paths.database_path,
            paths.base_model_dir,
            output,
        ),
        paths.reports_dir / "commands.log",
    )


def _triangulate_points_stage(
    colmap: str, paths: RegistrationPaths
) -> None:
    _run_model_stage(
        paths.triangulated_model_dir,
        lambda output: build_point_triangulator_command(
            colmap,
            paths.database_path,
            paths.workspace_images,
            paths.registered_model_dir,
            output,
        ),
        paths.reports_dir / "commands.log",
    )


def _bundle_adjust_stage(colmap: str, paths: RegistrationPaths) -> None:
    _run_model_stage(
        paths.final_model_dir,
        lambda output: build_bundle_adjuster_command(
            colmap, paths.triangulated_model_dir, output
        ),
        paths.reports_dir / "commands.log",
    )


def _registered_targets(
    model_dir: Path, targets: Sequence[TargetFrame]
) -> tuple[TargetFrame, ...]:
    images = read_images_binary(model_dir / "images.bin")
    registered_names = {
        image.name.replace("\\", "/") for image in images.values()
    }
    return tuple(
        target
        for target in targets
        if target.staged_relative_name in registered_names
    )


def _validate_registered_targets(
    config: RegistrationConfig,
    paths: RegistrationPaths,
    base: BaseContext,
    targets: Sequence[TargetFrame],
) -> ModelCandidate:
    _require_complete_model(paths.final_model_dir, "final registered COLMAP model")
    cameras = read_cameras_binary(paths.final_model_dir / "cameras.bin")
    images = read_images_binary(paths.final_model_dir / "images.bin")
    points = read_points3d_binary(paths.final_model_dir / "points3D.bin")
    if not points:
        raise RuntimeError("Final registered model contains no sparse 3D point")
    registered_by_name = {
        image.name.replace("\\", "/"): image for image in images.values()
    }
    reference_names = {
        reference.staged_relative_name for reference in base.references
    }
    missing_references = sorted(reference_names - set(registered_by_name))
    if missing_references:
        raise RuntimeError(
            f"Dense registration lost fixed reference frames: {missing_references}"
        )
    registered = [
        target
        for target in targets
        if target.staged_relative_name in registered_by_name
    ]
    missing = [
        target.staged_relative_name
        for target in targets
        if target.staged_relative_name not in registered_by_name
    ]
    ratio = len(registered) / len(targets)
    target_camera_ids = {
        int(registered_by_name[target.staged_relative_name].camera_id)
        for target in registered
    }
    if target_camera_ids != {base.sweep_camera_id}:
        raise RuntimeError(
            "Registered target frames do not all use the seed sweep camera ID: "
            f"expected={base.sweep_camera_id}, actual={sorted(target_camera_ids)}"
        )
    nonfinite_poses = [
        image_name
        for image_name, image in registered_by_name.items()
        if not np.isfinite(
            np.concatenate(
                [
                    np.asarray(image.qvec, dtype=np.float64),
                    np.asarray(image.tvec, dtype=np.float64),
                ]
            )
        ).all()
    ]
    if nonfinite_poses:
        raise RuntimeError(
            f"Registered image poses contain non-finite values: {nonfinite_poses[:5]}"
        )
    base_cameras = read_cameras_binary(base.selected_model_dir / "cameras.bin")
    if base.sweep_camera_id not in cameras or base.sweep_camera_id not in base_cameras:
        raise RuntimeError("Sweep camera ID is missing from base or final model")
    base_camera = base_cameras[base.sweep_camera_id]
    final_camera = cameras[base.sweep_camera_id]
    if not np.isfinite(np.asarray(final_camera.params, dtype=np.float64)).all():
        raise RuntimeError("Final sweep camera intrinsics contain non-finite values")
    if (
        base_camera.model != final_camera.model
        or int(base_camera.width) != int(final_camera.width)
        or int(base_camera.height) != int(final_camera.height)
        or not np.array_equal(
            np.asarray(base_camera.params), np.asarray(final_camera.params)
        )
    ):
        raise RuntimeError(
            "Final bundle adjustment changed the seed sweep camera intrinsics"
        )
    report = {
        "format": "target_sweep_registration",
        "version": 1,
        "target_fps_hz": config.target_sweep_fps_hz,
        "requested_target_count": len(targets),
        "registered_target_count": len(registered),
        "registration_ratio": ratio,
        "minimum_registration_ratio": config.min_target_registration_ratio,
        "missing_target_names": missing,
        "registered_target_names": [
            target.staged_relative_name for target in registered
        ],
        "sweep_camera_id": base.sweep_camera_id,
        "point_count": len(points),
        "final_model": str(paths.final_model_dir),
    }
    _write_json_atomic(paths.reports_dir / "target_registration.json", report)
    _write_json_atomic(
        paths.reports_dir / "selected_model.json",
        {
            "model_dir": str(paths.final_model_dir),
            "registered_names": sorted(registered_by_name),
            "registered_sweep_count": sum(
                name.startswith("sweep/") for name in registered_by_name
            ),
            "registered_target_count": len(registered),
            "point_count": len(points),
        },
    )
    if ratio < config.min_target_registration_ratio:
        raise RuntimeError(
            f"Target registration ratio {ratio:.6g} is below required "
            f"{config.min_target_registration_ratio:.6g}; see "
            f"{paths.reports_dir / 'target_registration.json'}"
        )
    return ModelCandidate(
        path=paths.final_model_dir,
        registered_names=frozenset(registered_by_name),
        registered_sweep_count=sum(
            name.startswith("sweep/") for name in registered_by_name
        ),
        point_count=len(points),
    )


def _verify_base_immutable(base: BaseContext) -> None:
    current = {
        "database.db": _sha256_file(base.database_path),
        "cameras.bin": _sha256_file(base.selected_model_dir / "cameras.bin"),
        "images.bin": _sha256_file(base.selected_model_dir / "images.bin"),
        "points3D.bin": _sha256_file(base.selected_model_dir / "points3D.bin"),
        "sweep_frame_selection.json": _sha256_file(
            base.run_dir / "reports" / "sweep_frame_selection.json"
        ),
        "reference_selection.json": _sha256_file(
            base.run_dir / "reports" / "reference_selection.json"
        ),
    }
    if current != dict(base.artifact_hashes):
        raise RuntimeError(
            "The immutable base pre-static run changed during dense registration"
        )


def _validate_existing_dataset(
    paths: RegistrationPaths,
    registered_targets: Sequence[TargetFrame],
) -> None:
    if not paths.dataset_dir.is_dir() or paths.dataset_dir.is_symlink():
        raise ValueError(f"Invalid static dataset: {paths.dataset_dir}")
    expected = {target.staged_name: target for target in registered_targets}
    image_dir = paths.dataset_dir / "images"
    mask_dir = paths.dataset_dir / "masks"
    invalid_image_entries = [
        path.name
        for path in image_dir.iterdir()
        if not path.is_file() or path.suffix.lower() != ".png"
    ]
    invalid_mask_entries = [
        path.name
        for path in mask_dir.iterdir()
        if not path.is_file() or path.suffix.lower() != ".png"
    ]
    if invalid_image_entries or invalid_mask_entries:
        raise ValueError(
            "Dense static dataset contains non-PNG frame entries: "
            f"images={invalid_image_entries[:5]}, masks={invalid_mask_entries[:5]}"
        )
    actual_images = {
        path.name: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }
    actual_masks = {
        path.name: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }
    if set(actual_images) != set(expected) or set(actual_masks) != set(expected):
        raise ValueError("Existing dense static dataset is not target-only")
    for name, target in expected.items():
        if not _files_have_same_content(
            actual_images[name], target.source_frame.image_path
        ) or not _files_have_same_content(
            actual_masks[name], target.source_frame.mask_path
        ):
            raise ValueError(f"Packaged RGB/mask differs from source for {name}")
    packaged_model = paths.dataset_dir / "colmap" / "sparse" / "0"
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not _files_have_same_content(
            packaged_model / name, paths.final_model_dir / name
        ):
            raise ValueError(f"Packaged COLMAP model differs: {name}")


def _validate_existing_references(
    paths: RegistrationPaths,
    base: BaseContext,
) -> None:
    if not paths.references_dir.is_dir() or paths.references_dir.is_symlink():
        raise ValueError(f"Invalid fixed-reference directory: {paths.references_dir}")
    expected_references = {
        f"{reference.view_id}_ref.png": reference.source_frame.image_path
        for reference in base.references
    }
    actual_references = {
        path.name: path
        for path in paths.references_dir.glob("*.png")
        if path.is_file()
    }
    if set(actual_references) != set(expected_references):
        raise ValueError("Packaged fixed-reference images changed")
    for name, source in expected_references.items():
        if not _files_have_same_content(actual_references[name], source):
            raise ValueError(f"Packaged reference differs from source: {name}")


def _validate_existing_package(
    paths: RegistrationPaths,
    base: BaseContext,
    registered_targets: Sequence[TargetFrame],
) -> None:
    _validate_existing_dataset(paths, registered_targets)
    _validate_existing_references(paths, base)


def _package_static_dataset(
    config: RegistrationConfig,
    paths: RegistrationPaths,
    base: BaseContext,
    targets: Sequence[TargetFrame],
) -> None:
    registered_targets = _registered_targets(paths.final_model_dir, targets)
    if not registered_targets:
        raise RuntimeError("No registered target frame is available for packaging")
    report = {
        "format": "registered_static_training_frames",
        "version": 1,
        "target_fps_hz": config.target_sweep_fps_hz,
        "requested_target_count": len(targets),
        "registered_target_count": len(registered_targets),
        "frames": [
            {
                "registered_name": target.staged_relative_name,
                "training_name": target.staged_name,
                "source_frame_name": target.source_frame.frame_name,
                "source_index": target.source_frame.source_index,
                "source_time_sec": target.source_frame.time_sec,
                "target_time_sec": target.target_time_sec,
                "source_image": str(target.source_frame.image_path),
                "source_mask": str(target.source_frame.mask_path),
                "reuses_base_seed": target.reuses_seed,
            }
            for target in registered_targets
        ],
    }
    report_path = paths.reports_dir / "packaged_sweep_frames.json"
    dataset_exists = paths.dataset_dir.exists() or paths.dataset_dir.is_symlink()
    references_exist = (
        paths.references_dir.exists() or paths.references_dir.is_symlink()
    )
    if dataset_exists:
        _validate_existing_dataset(paths, registered_targets)
    if references_exist:
        _validate_existing_references(paths, base)
    if dataset_exists and references_exist:
        if report_path.exists() and _read_json(report_path) != report:
            raise ValueError(f"Packaged-frame report changed: {report_path}")
        _write_json_atomic(report_path, report)
        return

    temporary = Path(
        tempfile.mkdtemp(prefix=".package.", suffix=".tmp", dir=str(paths.run_dir))
    )
    try:
        if not dataset_exists:
            dataset = temporary / "sweep_colmap_dataset"
            images = dataset / "images"
            masks = dataset / "masks"
            model = dataset / "colmap" / "sparse" / "0"
            cache = dataset / "flow3d_preprocessed"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            model.parent.mkdir(parents=True)
            cache.mkdir(parents=True)
            for target in registered_targets:
                shutil.copy2(
                    target.source_frame.image_path, images / target.staged_name
                )
                shutil.copy2(
                    target.source_frame.mask_path, masks / target.staged_name
                )
            shutil.copytree(paths.final_model_dir, model)
        if not references_exist:
            references_dir = temporary / "references"
            references_dir.mkdir()
            for reference in base.references:
                shutil.copy2(
                    reference.source_frame.image_path,
                    references_dir / f"{reference.view_id}_ref.png",
                )
        if not references_exist:
            os.replace(temporary / "references", paths.references_dir)
        if not dataset_exists:
            os.replace(temporary / "sweep_colmap_dataset", paths.dataset_dir)
        _write_json_atomic(report_path, report)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _reference_name_to_label(
    references: Sequence[ReferenceSelection],
) -> dict[str, str]:
    return {
        reference.staged_relative_name.replace("\\", "/"): reference.view_id
        for reference in references
    }


def _export_reference_cameras(
    config: RegistrationConfig,
    paths: RegistrationPaths,
    base: BaseContext,
    targets: Sequence[TargetFrame],
) -> None:
    cameras = read_cameras_binary(paths.final_model_dir / "cameras.bin")
    images = read_images_binary(paths.final_model_dir / "images.bin")
    points = read_points3d_binary(paths.final_model_dir / "points3D.bin")
    target_names = {target.staged_relative_name for target in targets}
    registered_target_names = sorted(
        target_names
        & {image.name.replace("\\", "/") for image in images.values()}
    )
    if not registered_target_names:
        raise RuntimeError("Final model has no registered requested target frame")
    target_w2cs: list[np.ndarray] = []
    for image in images.values():
        if image.name.replace("\\", "/") in set(registered_target_names):
            _, world_to_camera = get_intrinsics_extrinsics(image, cameras)
            target_w2cs.append(world_to_camera)
    scene_scale, scene_transform = compute_colmap_scene_norm(
        np.stack([point.xyz for point in points.values()]).astype(np.float64),
        np.stack(target_w2cs).astype(np.float64),
    )
    if (
        not math.isfinite(scene_scale)
        or scene_scale <= 0.0
        or not np.isfinite(scene_transform).all()
    ):
        raise RuntimeError("Dense COLMAP scene normalization is non-finite")
    reference_mapping = _reference_name_to_label(base.references)
    records, reference_records = export_camera_records(
        paths.final_model_dir,
        reference_mapping,
        scene_scale,
        scene_transform,
    )
    camera_group_ids = validate_camera_group_assignments(
        records, base.camera_group_by_source
    )
    torch = _require_torch()
    scene_norm_path = (
        paths.dataset_dir / "flow3d_preprocessed" / "scene_norm_dict.pth"
    )
    temporary_scene_norm = scene_norm_path.with_name(
        f".{scene_norm_path.name}.{os.getpid()}.tmp"
    )
    try:
        torch.save(
            {
                "scale": float(scene_scale),
                "transfm": torch.from_numpy(scene_transform.astype(np.float32)),
            },
            temporary_scene_norm,
        )
        os.replace(temporary_scene_norm, scene_norm_path)
    except BaseException:
        temporary_scene_norm.unlink(missing_ok=True)
        raise
    _write_json_atomic(
        paths.reports_dir / "registered_cameras.json", {"cameras": records}
    )
    _write_json_atomic(
        paths.references_dir / "reference_cameras.json",
        {
            "format": "fixed_reference_cameras",
            "version": 1,
            "coordinate_systems": {
                "raw": "COLMAP sparse model world",
                "normalized": "scene_norm_dict.pth world",
            },
            "references": reference_records,
        },
    )
    errors = np.asarray(
        [float(point.error) for point in points.values()], dtype=np.float64
    )
    if not np.isfinite(errors).all():
        raise RuntimeError("Final COLMAP point reprojection errors are non-finite")
    missing_target_names = sorted(target_names - set(registered_target_names))
    summary = {
        "format": "joint_colmap_frame_dataset",
        "version": 1,
        "source_config": str(config.config_path),
        "source_manifest": str(base.source_manifest_path),
        "base_prestatic_run": str(base.run_dir),
        "transforms": {
            "video_decode": False,
            "rotate": False,
            "crop": False,
            "resize": False,
        },
        "colmap": {
            "camera_model": cameras[base.sweep_camera_id].model,
            "camera_grouping": "existing_camera_id",
            "camera_group_ids": camera_group_ids,
            "matcher": "matches_importer_pairs",
            "selected_model": str(paths.final_model_dir),
            "requested_sweep_count": len(targets),
            "registered_sweep_count": len(registered_target_names),
            "total_registered_sweep_count": sum(
                record["source"] == "sweep" for record in records
            ),
            "sweep_registration_ratio": len(registered_target_names) / len(targets),
            "minimum_sweep_registration_ratio": (
                config.min_target_registration_ratio
            ),
            "missing_sweep_names": missing_target_names,
            "registered_reference_labels": sorted(reference_records),
            "registered_image_count": len(records),
            "point_count": len(points),
            "mean_point_reprojection_error": float(errors.mean()),
            "median_point_reprojection_error": float(np.median(errors)),
        },
        "scene_normalization": {
            "scale": scene_scale,
            "transform": scene_transform.astype(float).tolist(),
            "camera_subset": "successfully_registered_target_grid_only",
        },
        "outputs": {
            "dataset_dir": str(paths.dataset_dir),
            "training_images": str(paths.dataset_dir / "images"),
            "training_masks": str(paths.dataset_dir / "masks"),
            "sparse_model": str(paths.dataset_dir / "colmap" / "sparse" / "0"),
            "scene_norm": str(scene_norm_path),
            "registered_cameras": str(
                paths.reports_dir / "registered_cameras.json"
            ),
            "reference_cameras": str(
                paths.references_dir / "reference_cameras.json"
            ),
        },
        "remaining_training_inputs": [],
        "registered_sweep_names": registered_target_names,
    }
    _write_json_atomic(
        paths.reports_dir / "colmap_registration_summary.json", summary
    )


def _validate_static_dataset(
    config: RegistrationConfig,
    paths: RegistrationPaths,
    base: BaseContext,
    targets: Sequence[TargetFrame],
) -> dict[str, Any]:
    registered_targets = _registered_targets(paths.final_model_dir, targets)
    _validate_existing_package(paths, base, registered_targets)
    image_dir = paths.dataset_dir / "images"
    mask_dir = paths.dataset_dir / "masks"
    model_dir = paths.dataset_dir / "colmap" / "sparse" / "0"
    image_paths = sorted(image_dir.glob("*.png"))
    mask_by_name = {path.name: path for path in mask_dir.glob("*.png")}
    if not image_paths or {path.name for path in image_paths} != set(mask_by_name):
        raise ValueError("Static training image and mask names differ")
    model_images = read_images_binary(model_dir / "images.bin")
    model_cameras = read_cameras_binary(model_dir / "cameras.bin")
    by_basename: dict[str, Any] = {}
    for image in model_images.values():
        basename = Path(image.name).name
        if basename in by_basename:
            raise ValueError(f"COLMAP model repeats image basename {basename!r}")
        by_basename[basename] = image
    expected_size: tuple[int, int] | None = None
    for image_path in image_paths:
        image = _read_image(image_path, "dense packaged RGB")
        mask = _read_image(mask_by_name[image_path.name], "dense packaged mask")
        _validate_rgb_image(image, image_path)
        _validate_binary_mask(mask, mask_by_name[image_path.name])
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"RGB/mask size mismatch for {image_path.name}")
        size = (image.shape[1], image.shape[0])
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise ValueError("Dense packaged sweep changes resolution")
        if image_path.name not in by_basename:
            raise KeyError(f"Packaged target is absent from model: {image_path.name}")
        camera = model_cameras[by_basename[image_path.name].camera_id]
        if (int(camera.width), int(camera.height)) != size:
            raise ValueError(f"COLMAP camera size differs for {image_path.name}")
    scene_norm_path = (
        paths.dataset_dir / "flow3d_preprocessed" / "scene_norm_dict.pth"
    )
    torch = _require_torch()
    scene_norm = torch.load(scene_norm_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(scene_norm, dict)
        or set(scene_norm) != {"scale", "transfm"}
        or not math.isfinite(float(scene_norm["scale"]))
        or float(scene_norm["scale"]) <= 0.0
        or tuple(scene_norm["transfm"].shape) != (4, 4)
        or not bool(torch.isfinite(scene_norm["transfm"]).all())
    ):
        raise ValueError(f"Invalid scene normalization: {scene_norm_path}")
    reference_path = paths.references_dir / "reference_cameras.json"
    reference_payload = _require_mapping(
        _read_json(reference_path), "fixed reference cameras"
    )
    expected_view_ids = {reference.view_id for reference in base.references}
    if (
        reference_payload.get("format") != "fixed_reference_cameras"
        or reference_payload.get("version") != 1
        or set(_require_mapping(reference_payload.get("references"), "references"))
        != expected_view_ids
    ):
        raise ValueError(f"Invalid fixed reference camera manifest: {reference_path}")
    reference_records = _require_mapping(
        reference_payload.get("references"), "references"
    )
    for reference in base.references:
        reference_rgb_path = paths.references_dir / f"{reference.view_id}_ref.png"
        reference_rgb = _read_image(reference_rgb_path, "fixed reference RGB")
        _validate_rgb_image(reference_rgb, reference_rgb_path)
        record = _require_mapping(
            reference_records[reference.view_id],
            f"reference camera {reference.view_id}",
        )
        if "image_width" not in record or "image_height" not in record:
            raise ValueError(
                f"Reference camera lacks dimensions for {reference.view_id}"
            )
        if (reference_rgb.shape[1], reference_rgb.shape[0]) != (
            int(record["image_width"]),
            int(record["image_height"]),
        ):
            raise ValueError(
                f"Reference RGB/COLMAP size mismatch for {reference.view_id}"
            )
    return {
        "format": READY_FORMAT,
        "version": READY_VERSION,
        "scene_id": config.scene_id,
        "run_id": config.run_id,
        "static_dataset": str(paths.dataset_dir),
        "training_frame_count": len(image_paths),
        "training_resolution": {
            "width": expected_size[0] if expected_size else None,
            "height": expected_size[1] if expected_size else None,
        },
        "reference_camera_count": len(expected_view_ids),
        "reference_cameras": str(reference_path),
        "source_manifest": str(base.source_manifest_path),
        "static_training_requirements": {
            "trajectory_type": "static",
            "camera_type": "colmap",
            "load_from_cache": True,
            "depth_loss_weights": {
                "w_depth_reg": 0.0,
                "w_depth_grad": 0.0,
                "w_depth_const": 0.0,
            },
        },
    }


def _validate_completed_stage(paths: RegistrationPaths, stage: str) -> None:
    required: dict[str, tuple[Path, ...]] = {
        "validate_base": (
            paths.resolved_config_path,
            paths.reports_dir / "base_provenance.json",
            paths.reports_dir / "sweep_frame_selection.json",
            paths.reports_dir / "reference_selection.json",
        ),
        "stage_workspace": (
            paths.workspace_images,
            paths.seed_database,
            paths.base_model_dir,
            paths.new_image_list,
            paths.pair_list,
        ),
        "extract_features": (paths.features_database,),
        "import_matches": (paths.database_path,),
        "register_images": (paths.registered_model_dir,),
        "triangulate_points": (paths.triangulated_model_dir,),
        "bundle_adjust": (paths.final_model_dir,),
        "validate_registration": (
            paths.reports_dir / "target_registration.json",
            paths.reports_dir / "selected_model.json",
        ),
        "package_static_dataset": (
            paths.dataset_dir / "images",
            paths.dataset_dir / "masks",
            paths.dataset_dir / "colmap" / "sparse" / "0",
            paths.references_dir,
            paths.reports_dir / "packaged_sweep_frames.json",
        ),
        "export_reference_cameras": (
            paths.dataset_dir / "flow3d_preprocessed" / "scene_norm_dict.pth",
            paths.references_dir / "reference_cameras.json",
            paths.reports_dir / "registered_cameras.json",
            paths.reports_dir / "colmap_registration_summary.json",
        ),
        "validate_static_ready": (
            paths.reports_dir / "STATIC_DATASET_READY.json",
        ),
    }
    missing = [str(path) for path in required[stage] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Stage {stage!r} is marked complete but artifacts are missing: {missing}"
        )


def _stage_bounds(
    from_stage: str | None, until_stage: str | None
) -> tuple[int, int]:
    start = 0 if from_stage is None else STAGES.index(from_stage)
    end = len(STAGES) - 1 if until_stage is None else STAGES.index(until_stage)
    if start > end:
        raise ValueError("--from-stage must not come after --until-stage")
    return start, end


def run_registration_pipeline(
    config: RegistrationConfig,
    *,
    dry_run: bool = False,
    from_stage: str | None = None,
    until_stage: str | None = None,
) -> Path | None:
    base = _load_base_context(config)
    base_seed_names = {
        seed.source_index: seed.staged_relative_name
        for seed in base.base_seed_frames
    }
    targets = select_registration_targets(
        base.sweep, config.target_sweep_fps_hz, base_seed_names
    )
    if config.target_sweep_fps_hz > base.sweep.config.fps_hz:
        raise ValueError(
            "target_sweep_fps_hz must not exceed the canonical source FPS"
        )
    if config.target_sweep_fps_hz <= float(
        _require_mapping(base.resolved.get("sweep"), "base sweep config").get(
            "sample_fps_hz"
        )
    ):
        raise ValueError(
            "target_sweep_fps_hz must exceed the base pre-static sampling FPS"
        )
    pairs = build_temporal_match_pairs(
        targets,
        base.registered_seed_by_source_index,
        seed_neighbors_per_side=config.seed_neighbors_per_side,
        target_neighbor_radius=config.target_neighbor_radius,
    )
    paths = registration_paths(config)
    resolved = _resolved_config_payload(config, paths, base, targets, pairs)
    config_identity = _canonical_json_identity(
        {
            "resolved_config": resolved,
            "source_manifest_identity": _canonical_json_identity(
                base.source_manifest
            ),
        }
    )
    start, end = _stage_bounds(from_stage, until_stage)
    if dry_run:
        print(
            json.dumps(
                {
                    "config_identity": config_identity,
                    "run_dir": str(paths.run_dir),
                    "base_run_dir": str(base.run_dir),
                    "stages": list(STAGES[start : end + 1]),
                    "source_sweep_frames": len(base.sweep.frames),
                    "base_seed_frames": len(base.base_seed_frames),
                    "registered_base_seed_frames": len(
                        base.registered_seed_by_source_index
                    ),
                    "target_frames": len(targets),
                    "reused_seed_targets": sum(
                        target.reuses_seed for target in targets
                    ),
                    "new_images": sum(not target.reuses_seed for target in targets),
                    "registration_targets": sum(
                        target.staged_relative_name
                        not in set(base.registered_seed_by_source_index.values())
                        for target in targets
                    ),
                    "temporal_pairs": len(pairs),
                    "transforms": resolved["transforms"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return None

    state = _initialize_or_load_state(paths, config_identity)
    completed = list(state["completed_stages"])
    if start > len(completed):
        raise ValueError(
            f"Cannot start from {STAGES[start]!r}; prerequisite stage "
            f"{STAGES[len(completed)]!r} is not complete"
        )
    for stage in completed:
        _validate_completed_stage(paths, stage)
    if "validate_base" in completed:
        if _read_json(paths.resolved_config_path) != resolved:
            raise ValueError(
                f"Resolved registration config changed: {paths.resolved_config_path}"
            )
        expected_selection = _target_selection_payload(base, config, targets)
        if (
            _read_json(paths.reports_dir / "sweep_frame_selection.json")
            != expected_selection
        ):
            raise ValueError("Resolved target sweep selection changed")

    colmap: str | None = None
    command_stages = {
        "extract_features",
        "import_matches",
        "register_images",
        "triangulate_points",
        "bundle_adjust",
    }
    if any(
        STAGES[index] in command_stages
        and STAGES[index] not in state["completed_stages"]
        for index in range(start, end + 1)
    ):
        colmap = resolve_executable(config.colmap_command, "COLMAP")

    for stage_index in range(start, end + 1):
        stage = STAGES[stage_index]
        if stage in state["completed_stages"]:
            print(f"Skipping completed stage: {stage}", flush=True)
            continue
        if stage_index != len(state["completed_stages"]):
            expected = STAGES[len(state["completed_stages"])]
            raise RuntimeError(f"Cannot run {stage!r}; expected {expected!r}")
        print(f"Running stage: {stage}", flush=True)
        if stage == "validate_base":
            _write_json_atomic(paths.resolved_config_path, resolved)
            _write_json_atomic(
                paths.reports_dir / "base_provenance.json",
                {
                    "format": "immutable_prestatic_base",
                    "version": 1,
                    "base_run_id": config.base_prestatic_run_id,
                    "base_run_dir": str(base.run_dir),
                    "base_ready": str(base.ready_path),
                    "source_manifest": str(base.source_manifest_path),
                    "artifact_hashes": dict(base.artifact_hashes),
                },
            )
            _write_json_atomic(
                paths.reports_dir / "sweep_frame_selection.json",
                _target_selection_payload(base, config, targets),
            )
            _write_json_atomic(
                paths.reports_dir / "reference_selection.json",
                _read_json(base.run_dir / "reports" / "reference_selection.json"),
            )
        elif stage == "stage_workspace":
            _stage_workspace(paths, base, targets, pairs)
        elif stage == "extract_features":
            assert colmap is not None
            _extract_features_stage(colmap, paths, base, targets)
        elif stage == "import_matches":
            assert colmap is not None
            _import_matches_stage(colmap, paths, base, targets, pairs)
        elif stage == "register_images":
            assert colmap is not None
            _register_images_stage(colmap, paths)
        elif stage == "triangulate_points":
            assert colmap is not None
            _triangulate_points_stage(colmap, paths)
        elif stage == "bundle_adjust":
            assert colmap is not None
            _bundle_adjust_stage(colmap, paths)
        elif stage == "validate_registration":
            _validate_registered_targets(config, paths, base, targets)
            _verify_base_immutable(base)
        elif stage == "package_static_dataset":
            _package_static_dataset(config, paths, base, targets)
        elif stage == "export_reference_cameras":
            _export_reference_cameras(config, paths, base, targets)
        elif stage == "validate_static_ready":
            ready = _validate_static_dataset(config, paths, base, targets)
            _verify_base_immutable(base)
            _write_json_atomic(
                paths.reports_dir / "STATIC_DATASET_READY.json", ready
            )
        else:
            raise AssertionError(stage)
        _complete_stage(paths, state, stage)
        state = _load_state(paths, config_identity)

    ready_path = paths.reports_dir / "STATIC_DATASET_READY.json"
    if ready_path.is_file():
        print(
            "Dense pre-static registration completed successfully:\n"
            f"  static dataset: {paths.dataset_dir}\n"
            f"  reference cameras: {paths.references_dir / 'reference_cameras.json'}\n"
            f"  ready report: {ready_path}",
            flush=True,
        )
        return ready_path
    print(
        f"Dense registration stopped after {STAGES[end]!r}; output: {paths.run_dir}",
        flush=True,
    )
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Register a denser canonical sweep time grid into an immutable "
            "completed pre-static COLMAP model and package a target-only static "
            "3DGS dataset without image transforms."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-stage", choices=STAGES, default=None)
    parser.add_argument("--until-stage", choices=STAGES, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_registration_config(args.config)
    run_registration_pipeline(
        config,
        dry_run=args.dry_run,
        from_stage=args.from_stage,
        until_stage=args.until_stage,
    )


if __name__ == "__main__":
    main()
