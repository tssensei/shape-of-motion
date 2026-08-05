"""Prepare one image-first joint-COLMAP dataset before static 3DGS training.

The manually exported RGB and mask sequences are the canonical experiment
inputs.  This controller never decodes a video and never rotates, crops, or
resizes an image.  It selects fixed-camera references, samples the sweep on a
time grid, reconstructs every source in one COLMAP model, and packages the
registered sweep frames and masks for the static Shape-of-Motion loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from preproc.prepare_joint_colmap_video_dataset import (
    ModelCandidate,
    choose_model,
    compute_colmap_scene_norm,
    export_camera_records,
    get_intrinsics_extrinsics,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
    resolve_executable,
    run_colmap,
    validate_sweep_registration,
)


CONFIG_FORMAT = "som_prestatic_pipeline"
CONFIG_VERSION = 1
SOURCE_MANIFEST_FORMAT = "canonical_source_sequences"
SOURCE_MANIFEST_VERSION = 1
PIPELINE_STATE_FORMAT = "prestatic_pipeline_state"
PIPELINE_STATE_VERSION = 1
READY_FORMAT = "static_colmap_dataset_ready"
READY_VERSION = 1
STAGES = (
    "validate_inputs",
    "stage_colmap_images",
    "run_colmap",
    "select_model",
    "package_static_dataset",
    "export_reference_cameras",
    "validate_static_ready",
)
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
NUMBERED_STEM_PATTERN = re.compile(r"^(?P<prefix>.*?)(?P<number>[0-9]+)$")
CAMERA_MODELS = frozenset(
    {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"}
)


@dataclass(frozen=True)
class SequenceConfig:
    view_id: str
    image_dir: Path
    mask_dir: Path
    fps_hz: float
    camera_group: str
    reference_policy: str | None = None
    reference_frame: str | None = None


@dataclass(frozen=True)
class ColmapConfig:
    sweep_sample_fps_hz: float
    camera_model: str
    matcher: str
    min_sweep_registration_ratio: float
    command: str


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    scene_id: str
    scene_root: Path
    run_id: str
    sweep: SequenceConfig
    static_views: tuple[SequenceConfig, ...]
    colmap: ColmapConfig


@dataclass(frozen=True)
class FrameRecord:
    frame_name: str
    frame_stem: str
    source_index: int
    time_sec: float
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class ValidatedSequence:
    config: SequenceConfig
    image_width: int
    image_height: int
    image_extension: str
    frames: tuple[FrameRecord, ...]
    source_identity: str


@dataclass(frozen=True)
class SweepSelection:
    staged_name: str
    source_frame: FrameRecord
    target_time_sec: float


@dataclass(frozen=True)
class ReferenceSelection:
    view_id: str
    camera_group: str
    staged_relative_name: str
    source_frame: FrameRecord


@dataclass(frozen=True)
class PipelinePaths:
    run_dir: Path
    reports_dir: Path
    workspace_dir: Path
    workspace_images: Path
    database_path: Path
    sparse_models_dir: Path
    dataset_dir: Path
    references_dir: Path
    resolved_config_path: Path
    source_manifest_path: Path
    state_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _validate_keys(
    payload: Mapping[str, Any],
    label: str,
    *,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required - optional)
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")
    if unknown:
        raise ValueError(f"{label} contains unsupported keys: {unknown}")


def _parse_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must start with a letter or digit and contain only "
            "letters, digits, underscores, or hyphens"
        )
    return value


def _parse_positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    return result


def _parse_sequence(
    payload: Any,
    label: str,
    *,
    is_static: bool,
) -> SequenceConfig:
    values = _require_mapping(payload, label)
    optional = {"reference_policy", "reference_frame"} if is_static else set()
    _validate_keys(
        values,
        label,
        required={"view_id", "image_dir", "mask_dir", "fps_hz", "camera_group"},
        optional=optional,
    )
    view_id = _parse_identifier(values["view_id"], f"{label}.view_id")
    camera_group = _parse_identifier(
        values["camera_group"], f"{label}.camera_group"
    )
    image_dir_value = values["image_dir"]
    mask_dir_value = values["mask_dir"]
    if not isinstance(image_dir_value, str) or not image_dir_value:
        raise ValueError(f"{label}.image_dir must be a non-empty path")
    if not isinstance(mask_dir_value, str) or not mask_dir_value:
        raise ValueError(f"{label}.mask_dir must be a non-empty path")
    image_dir = Path(image_dir_value).expanduser().resolve()
    mask_dir = Path(mask_dir_value).expanduser().resolve()
    if image_dir == mask_dir:
        raise ValueError(f"{label}.image_dir and mask_dir must be different")
    reference_policy = values.get("reference_policy")
    reference_frame = values.get("reference_frame")
    if is_static:
        if reference_policy is None and reference_frame is None:
            reference_policy = "middle"
        if reference_policy is not None:
            if reference_policy != "middle":
                raise ValueError(
                    f"{label}.reference_policy currently supports only 'middle'"
                )
            if reference_frame is not None:
                raise ValueError(
                    f"{label} cannot set both reference_policy and reference_frame"
                )
        if reference_frame is not None and (
            not isinstance(reference_frame, str) or not reference_frame
        ):
            raise ValueError(f"{label}.reference_frame must be a non-empty filename")
        if reference_frame is not None and (
            Path(reference_frame).name != reference_frame
            or Path(reference_frame).suffix.lower() != ".png"
        ):
            raise ValueError(
                f"{label}.reference_frame must be one PNG filename, not a path"
            )
    return SequenceConfig(
        view_id=view_id,
        image_dir=image_dir,
        mask_dir=mask_dir,
        fps_hz=_parse_positive_float(values["fps_hz"], f"{label}.fps_hz"),
        camera_group=camera_group,
        reference_policy=reference_policy,
        reference_frame=reference_frame,
    )


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    values = _require_mapping(payload, str(config_path))
    _validate_keys(
        values,
        str(config_path),
        required={
            "format",
            "version",
            "scene_id",
            "scene_root",
            "run_id",
            "sweep",
            "static_views",
            "colmap",
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
    scene_root_value = values["scene_root"]
    if not isinstance(scene_root_value, str) or not scene_root_value:
        raise ValueError("scene_root must be a non-empty path")
    scene_root = Path(scene_root_value).expanduser().resolve()
    if scene_root.name != scene_id:
        raise ValueError(
            f"scene_root basename {scene_root.name!r} does not match "
            f"scene_id {scene_id!r}"
        )
    sweep = _parse_sequence(values["sweep"], "sweep", is_static=False)
    static_payload = values["static_views"]
    if not isinstance(static_payload, list) or not static_payload:
        raise ValueError("static_views must be a non-empty list")
    static_views = tuple(
        _parse_sequence(item, f"static_views[{index}]", is_static=True)
        for index, item in enumerate(static_payload)
    )
    if any(view.view_id == "sweep" for view in static_views):
        raise ValueError("Static view_id 'sweep' is reserved for COLMAP sweep frames")
    view_ids = [sweep.view_id, *(view.view_id for view in static_views)]
    if len(set(view_ids)) != len(view_ids):
        raise ValueError(f"Every sequence view_id must be unique, got {view_ids}")
    if any(view.camera_group == sweep.camera_group for view in static_views):
        raise ValueError(
            "The sweep and static references must use distinct camera_group values"
        )

    colmap_payload = _require_mapping(values["colmap"], "colmap")
    _validate_keys(
        colmap_payload,
        "colmap",
        required={
            "sweep_sample_fps_hz",
            "camera_model",
            "matcher",
            "min_sweep_registration_ratio",
            "command",
        },
    )
    camera_model = colmap_payload["camera_model"]
    if not isinstance(camera_model, str) or camera_model not in CAMERA_MODELS:
        raise ValueError(
            f"colmap.camera_model must be one of {sorted(CAMERA_MODELS)}, "
            f"got {camera_model!r}"
        )
    if colmap_payload["matcher"] != "exhaustive":
        raise ValueError("colmap.matcher currently supports only 'exhaustive'")
    registration_ratio = _parse_positive_float(
        colmap_payload["min_sweep_registration_ratio"],
        "colmap.min_sweep_registration_ratio",
    )
    if registration_ratio > 1.0:
        raise ValueError("colmap.min_sweep_registration_ratio must not exceed 1")
    command = colmap_payload["command"]
    if not isinstance(command, str) or not command:
        raise ValueError("colmap.command must be a non-empty executable name or path")
    colmap = ColmapConfig(
        sweep_sample_fps_hz=_parse_positive_float(
            colmap_payload["sweep_sample_fps_hz"],
            "colmap.sweep_sample_fps_hz",
        ),
        camera_model=str(camera_model),
        matcher="exhaustive",
        min_sweep_registration_ratio=registration_ratio,
        command=command,
    )
    return PipelineConfig(
        config_path=config_path,
        scene_id=scene_id,
        scene_root=scene_root,
        run_id=run_id,
        sweep=sweep,
        static_views=static_views,
        colmap=colmap,
    )


def _read_image(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError(f"Could not read {label}: {path}")
    return image


def _ordered_pngs(
    directory: Path,
    label: str,
    *,
    require_contiguous: bool = False,
) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    entries = sorted(directory.iterdir())
    non_files = [entry.name for entry in entries if not entry.is_file()]
    if non_files:
        raise ValueError(
            f"{label} must contain only frame files; found entries {non_files[:5]}"
        )
    wrong_suffix = [entry.name for entry in entries if entry.suffix.lower() != ".png"]
    if wrong_suffix:
        raise ValueError(
            f"{label} accepts only PNG frames; found {wrong_suffix[:5]}"
        )
    if not entries:
        raise ValueError(f"{label} contains no PNG frames: {directory}")
    parsed: list[tuple[str, int, int, Path]] = []
    for entry in entries:
        match = NUMBERED_STEM_PATTERN.fullmatch(entry.stem)
        if match is None:
            raise ValueError(
                f"{label} frame names must end in a numeric index: {entry.name}"
            )
        parsed.append(
            (
                match.group("prefix"),
                int(match.group("number")),
                len(match.group("number")),
                entry,
            )
        )
    prefixes = {value[0] for value in parsed}
    widths = {value[2] for value in parsed}
    if len(prefixes) != 1 or len(widths) != 1:
        raise ValueError(
            f"{label} frame names must share one prefix and fixed-width numeric suffix"
        )
    numeric_indices = [value[1] for value in parsed]
    if len(set(numeric_indices)) != len(numeric_indices):
        raise ValueError(f"{label} contains duplicate numeric frame indices")
    ordered = sorted(parsed, key=lambda value: value[1])
    if require_contiguous and any(
        current[1] != previous[1] + 1
        for previous, current in zip(ordered, ordered[1:])
    ):
        raise ValueError(
            f"{label} numeric frame indices must be contiguous for FPS timing"
        )
    return [value[3] for value in ordered]


def _validate_rgb_image(image: np.ndarray, path: Path) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"RGB frame must be an 8-bit three-channel PNG, got "
            f"dtype={image.dtype}, shape={image.shape}: {path}"
        )


def _validate_binary_mask(mask: np.ndarray, path: Path) -> None:
    if mask.ndim == 3:
        first = mask[..., 0]
        channels_match = all(
            np.array_equal(first, mask[..., channel])
            for channel in range(1, mask.shape[2])
        )
        if not channels_match:
            raise ValueError(f"Mask color channels are not identical: {path}")
        mask = first
    if mask.ndim != 2:
        raise ValueError(f"Mask must be single-channel or replicated grayscale: {path}")
    values = np.unique(mask)
    if values.size > 2:
        raise ValueError(
            f"Mask must contain at most background and foreground values: {path}, "
            f"found {values[:8].tolist()}"
        )
    if values.size == 2 and values[0] != 0:
        raise ValueError(f"Two-valued mask must use zero for background: {path}")


def _sequence_identity(
    config: SequenceConfig,
    frames: Sequence[FrameRecord],
    image_width: int,
    image_height: int,
) -> str:
    hasher = hashlib.sha256()
    header = {
        "view_id": config.view_id,
        "image_dir": str(config.image_dir),
        "mask_dir": str(config.mask_dir),
        "fps_hz": config.fps_hz,
        "camera_group": config.camera_group,
        "image_width": image_width,
        "image_height": image_height,
    }
    hasher.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    for frame in frames:
        image_stat = frame.image_path.stat()
        mask_stat = frame.mask_path.stat()
        hasher.update(frame.frame_name.encode("utf-8"))
        hasher.update(
            (
                f"{image_stat.st_size}:{image_stat.st_mtime_ns}:"
                f"{mask_stat.st_size}:{mask_stat.st_mtime_ns}"
            ).encode("utf-8")
        )
    return hasher.hexdigest()


def validate_sequence(config: SequenceConfig) -> ValidatedSequence:
    image_paths = _ordered_pngs(
        config.image_dir,
        f"{config.view_id} image_dir",
        require_contiguous=True,
    )
    mask_paths = _ordered_pngs(
        config.mask_dir,
        f"{config.view_id} mask_dir",
        require_contiguous=True,
    )
    masks_by_stem = {path.stem: path for path in mask_paths}
    image_stems = {path.stem for path in image_paths}
    missing_masks = sorted(image_stems - set(masks_by_stem))
    extra_masks = sorted(set(masks_by_stem) - image_stems)
    if missing_masks or extra_masks:
        raise ValueError(
            f"{config.view_id} image/mask stems differ: "
            f"missing_masks={missing_masks[:5]}, extra_masks={extra_masks[:5]}"
        )

    expected_shape: tuple[int, int] | None = None
    frames: list[FrameRecord] = []
    for source_index, image_path in enumerate(image_paths):
        mask_path = masks_by_stem[image_path.stem]
        image = _read_image(image_path, "RGB frame")
        mask = _read_image(mask_path, "mask frame")
        _validate_rgb_image(image, image_path)
        image_shape = image.shape[:2]
        if expected_shape is None:
            expected_shape = image_shape
        elif image_shape != expected_shape:
            raise ValueError(
                f"{config.view_id} image resolution changed from "
                f"{expected_shape[::-1]} "
                f"to {image_shape[::-1]} at {image_path}"
            )
        if mask.shape[:2] != image_shape:
            raise ValueError(
                f"Image/mask dimensions differ for {config.view_id}/{image_path.name}: "
                f"image={image_shape[::-1]}, mask={mask.shape[:2][::-1]}"
            )
        _validate_binary_mask(mask, mask_path)
        frames.append(
            FrameRecord(
                frame_name=image_path.name,
                frame_stem=image_path.stem,
                source_index=source_index,
                time_sec=source_index / config.fps_hz,
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    assert expected_shape is not None
    identity = _sequence_identity(
        config,
        frames,
        expected_shape[1],
        expected_shape[0],
    )
    return ValidatedSequence(
        config=config,
        image_width=expected_shape[1],
        image_height=expected_shape[0],
        image_extension=".png",
        frames=tuple(frames),
        source_identity=identity,
    )


def validate_source_sequences(
    config: PipelineConfig,
) -> tuple[ValidatedSequence, tuple[ValidatedSequence, ...]]:
    sweep = validate_sequence(config.sweep)
    static_views = tuple(validate_sequence(view) for view in config.static_views)
    by_group: dict[str, tuple[int, int]] = {}
    for sequence in (sweep, *static_views):
        size = (sequence.image_width, sequence.image_height)
        previous = by_group.setdefault(sequence.config.camera_group, size)
        if previous != size:
            raise ValueError(
                f"camera_group {sequence.config.camera_group!r} mixes resolutions "
                f"{previous} and {size}"
            )
    static_sizes = {(view.image_width, view.image_height) for view in static_views}
    if len(static_sizes) != 1:
        raise ValueError(
            "Current modal dataset contract requires every static view to use the "
            f"same resolution, got {sorted(static_sizes)}"
        )
    return sweep, static_views


def select_sweep_frames(
    sequence: ValidatedSequence,
    target_fps_hz: float,
) -> tuple[SweepSelection, ...]:
    target_fps = _parse_positive_float(target_fps_hz, "target_fps_hz")
    count = len(sequence.frames)
    if target_fps >= sequence.config.fps_hz:
        source_indices = list(range(count))
        target_times = [frame.time_sec for frame in sequence.frames]
    else:
        duration = sequence.frames[-1].time_sec
        target_count = int(math.floor(duration * target_fps + 1e-12)) + 1
        candidate_times = [index / target_fps for index in range(target_count)]
        source_indices = []
        target_times = []
        for target_time in candidate_times:
            source_value = target_time * sequence.config.fps_hz
            source_index = min(count - 1, int(math.floor(source_value + 0.5)))
            if source_indices and source_index == source_indices[-1]:
                continue
            source_indices.append(source_index)
            target_times.append(target_time)
    if not source_indices:
        raise ValueError("Sweep selection produced no frames")
    return tuple(
        SweepSelection(
            staged_name=f"sweep_{selection_index:06d}.png",
            source_frame=sequence.frames[source_index],
            target_time_sec=target_time,
        )
        for selection_index, (source_index, target_time) in enumerate(
            zip(source_indices, target_times, strict=True)
        )
    )


def select_reference_frame(sequence: ValidatedSequence) -> ReferenceSelection:
    reference_name = sequence.config.reference_frame
    if reference_name is not None:
        matches = [
            frame for frame in sequence.frames if frame.frame_name == reference_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Reference frame {reference_name!r} was not found exactly once in "
                f"{sequence.config.image_dir}"
            )
        frame = matches[0]
    else:
        if sequence.config.reference_policy != "middle":
            raise ValueError(
                f"Unsupported reference policy for {sequence.config.view_id}: "
                f"{sequence.config.reference_policy!r}"
            )
        frame = sequence.frames[len(sequence.frames) // 2]
    folder = f"static_{sequence.config.camera_group}"
    return ReferenceSelection(
        view_id=sequence.config.view_id,
        camera_group=sequence.config.camera_group,
        staged_relative_name=f"{folder}/{sequence.config.view_id}_ref.png",
        source_frame=frame,
    )


def _pipeline_paths(config: PipelineConfig) -> PipelinePaths:
    run_dir = (
        config.scene_root / "shared" / "preprocessing" / config.run_id
    )
    reports = run_dir / "reports"
    workspace = run_dir / "colmap_workspace"
    return PipelinePaths(
        run_dir=run_dir,
        reports_dir=reports,
        workspace_dir=workspace,
        workspace_images=workspace / "images",
        database_path=workspace / "database.db",
        sparse_models_dir=workspace / "sparse_models",
        dataset_dir=run_dir / "sweep_colmap_dataset",
        references_dir=run_dir / "references",
        resolved_config_path=run_dir / "resolved_config.json",
        source_manifest_path=(
            config.scene_root / "shared" / "inputs" / "source_sequences_v1.json"
        ),
        state_path=reports / "pipeline_state.json",
    )


def _source_manifest_payload(
    config: PipelineConfig,
    sweep: ValidatedSequence,
    static_views: Sequence[ValidatedSequence],
) -> dict[str, Any]:
    def sequence_payload(sequence: ValidatedSequence) -> dict[str, Any]:
        return {
            "view_id": sequence.config.view_id,
            "image_dir": str(sequence.config.image_dir),
            "mask_dir": str(sequence.config.mask_dir),
            "fps_hz": sequence.config.fps_hz,
            "camera_group": sequence.config.camera_group,
            "frame_count": len(sequence.frames),
            "image_width": sequence.image_width,
            "image_height": sequence.image_height,
            "image_extension": sequence.image_extension,
            "frame_names": [frame.frame_name for frame in sequence.frames],
            "source_identity": sequence.source_identity,
        }

    return {
        "format": SOURCE_MANIFEST_FORMAT,
        "version": SOURCE_MANIFEST_VERSION,
        "scene_id": config.scene_id,
        "sweep": sequence_payload(sweep),
        "static_views": [sequence_payload(view) for view in static_views],
    }


def _resolved_config_payload(
    config: PipelineConfig,
    paths: PipelinePaths,
    sweep: ValidatedSequence,
    static_views: Sequence[ValidatedSequence],
    sweep_selection: Sequence[SweepSelection],
    references: Sequence[ReferenceSelection],
) -> dict[str, Any]:
    return {
        "format": CONFIG_FORMAT,
        "version": CONFIG_VERSION,
        "source_config": str(config.config_path),
        "scene_id": config.scene_id,
        "scene_root": str(config.scene_root),
        "run_id": config.run_id,
        "run_dir": str(paths.run_dir),
        "source_manifest": str(paths.source_manifest_path),
        "sweep": {
            "view_id": sweep.config.view_id,
            "source_identity": sweep.source_identity,
            "source_fps_hz": sweep.config.fps_hz,
            "sample_fps_hz": config.colmap.sweep_sample_fps_hz,
            "source_frame_count": len(sweep.frames),
            "selected_frame_count": len(sweep_selection),
            "image_width": sweep.image_width,
            "image_height": sweep.image_height,
            "camera_group": sweep.config.camera_group,
        },
        "static_views": [
            {
                "view_id": view.config.view_id,
                "source_identity": view.source_identity,
                "fps_hz": view.config.fps_hz,
                "frame_count": len(view.frames),
                "image_width": view.image_width,
                "image_height": view.image_height,
                "camera_group": view.config.camera_group,
                "reference_frame": reference.source_frame.frame_name,
                "reference_local_index": reference.source_frame.source_index,
                "reference_time_sec": reference.source_frame.time_sec,
                "staged_relative_name": reference.staged_relative_name,
            }
            for view, reference in zip(static_views, references, strict=True)
        ],
        "colmap": {
            "camera_model": config.colmap.camera_model,
            "matcher": config.colmap.matcher,
            "min_sweep_registration_ratio": (
                config.colmap.min_sweep_registration_ratio
            ),
            "command": config.colmap.command,
            "camera_grouping": "single_camera_per_folder",
        },
        "transforms": {
            "video_decode": False,
            "rotate": False,
            "crop": False,
            "resize": False,
        },
    }


def _canonical_json_identity(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _files_have_same_content(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _require_same_file_content(
    actual: Path,
    expected: Path,
    label: str,
) -> None:
    if not actual.is_file() or actual.is_symlink():
        raise ValueError(f"{label} is not a regular file: {actual}")
    if not _files_have_same_content(actual, expected):
        raise ValueError(
            f"{label} differs from its canonical source: {actual} != {expected}"
        )


def _initial_state(config_identity: str) -> dict[str, Any]:
    return {
        "format": PIPELINE_STATE_FORMAT,
        "version": PIPELINE_STATE_VERSION,
        "config_identity": config_identity,
        "completed_stages": [],
        "stage_completed_at": {},
    }


def _initialize_or_load_state(
    paths: PipelinePaths,
    config_identity: str,
) -> dict[str, Any]:
    if paths.run_dir.exists():
        if not paths.run_dir.is_dir() or paths.run_dir.is_symlink():
            raise ValueError(f"Pipeline run path is not a directory: {paths.run_dir}")
        if not paths.state_path.is_file():
            raise FileExistsError(
                f"Run directory exists without pipeline state: {paths.run_dir}. "
                "Use a new run_id or inspect the incomplete directory."
            )
        return _load_state(paths, config_identity)

    paths.run_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_run = Path(
        tempfile.mkdtemp(
            prefix=f".{paths.run_dir.name}.",
            suffix=".init.tmp",
            dir=str(paths.run_dir.parent),
        )
    )
    state = _initial_state(config_identity)
    try:
        temporary_reports = temporary_run / "reports"
        temporary_reports.mkdir()
        _write_json_atomic(
            temporary_reports / paths.state_path.name,
            state,
        )
        os.replace(temporary_run, paths.run_dir)
    except BaseException:
        shutil.rmtree(temporary_run, ignore_errors=True)
        raise
    return state


def _load_state(paths: PipelinePaths, config_identity: str) -> dict[str, Any]:
    if not paths.state_path.exists():
        return _initial_state(config_identity)
    state = _read_json(paths.state_path)
    if (
        not isinstance(state, dict)
        or state.get("format") != PIPELINE_STATE_FORMAT
        or state.get("version") != PIPELINE_STATE_VERSION
        or state.get("config_identity") != config_identity
    ):
        raise ValueError(
            f"Existing pipeline state is incompatible with the current inputs: "
            f"{paths.state_path}. Use a new run_id."
        )
    completed = state.get("completed_stages")
    if not isinstance(completed, list) or any(
        stage not in STAGES for stage in completed
    ):
        raise ValueError(f"Invalid completed_stages in {paths.state_path}")
    expected_prefix = list(STAGES[: len(completed)])
    if completed != expected_prefix:
        raise ValueError(
            f"Completed pipeline stages are not a contiguous prefix: {completed}"
        )
    return state


def _complete_stage(paths: PipelinePaths, state: dict[str, Any], stage: str) -> None:
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


def _sweep_selection_payload(
    sequence: ValidatedSequence,
    target_fps_hz: float,
    selections: Sequence[SweepSelection],
) -> dict[str, Any]:
    return {
        "format": "time_sampled_sweep_frames",
        "version": 1,
        "view_id": sequence.config.view_id,
        "source_fps_hz": sequence.config.fps_hz,
        "target_fps_hz": target_fps_hz,
        "source_frame_count": len(sequence.frames),
        "selected_frame_count": len(selections),
        "selection": [
            {
                "staged_name": item.staged_name,
                "source_frame_name": item.source_frame.frame_name,
                "source_index": item.source_frame.source_index,
                "target_time_sec": item.target_time_sec,
                "source_time_sec": item.source_frame.time_sec,
                "time_error_sec": item.source_frame.time_sec - item.target_time_sec,
                "source_image": str(item.source_frame.image_path),
                "source_mask": str(item.source_frame.mask_path),
            }
            for item in selections
        ],
    }


def _reference_selection_payload(
    references: Sequence[ReferenceSelection],
) -> dict[str, Any]:
    return {
        "format": "fixed_reference_selection",
        "version": 1,
        "references": [
            {
                "view_id": reference.view_id,
                "camera_group": reference.camera_group,
                "staged_relative_name": reference.staged_relative_name,
                "source_frame_name": reference.source_frame.frame_name,
                "source_index": reference.source_frame.source_index,
                "source_time_sec": reference.source_frame.time_sec,
                "source_image": str(reference.source_frame.image_path),
            }
            for reference in references
        ],
    }


def _ensure_unique_colmap_basenames(relative_names: Sequence[str]) -> None:
    basenames: dict[str, str] = {}
    for relative_name in relative_names:
        basename = Path(relative_name).name
        previous = basenames.setdefault(basename, relative_name)
        if previous != relative_name:
            raise ValueError(
                f"COLMAP image basenames must be globally unique, but {basename!r} "
                f"is used by {previous!r} and {relative_name!r}"
            )


def _validate_staged_colmap_images(
    root: Path,
    sweep_selection: Sequence[SweepSelection],
    references: Sequence[ReferenceSelection],
) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"COLMAP image stage is not a regular directory: {root}")
    expected = {
        **{
            f"sweep/{selection.staged_name}": selection.source_frame.image_path
            for selection in sweep_selection
        },
        **{
            reference.staged_relative_name: reference.source_frame.image_path
            for reference in references
        },
    }
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }
    if set(actual) != set(expected):
        raise ValueError(
            "Existing COLMAP image stage does not match the resolved selection: "
            f"missing={sorted(set(expected) - set(actual))[:5]}, "
            f"extra={sorted(set(actual) - set(expected))[:5]}"
        )
    for relative_name, source in expected.items():
        _require_same_file_content(
            actual[relative_name],
            source,
            f"Staged COLMAP image {relative_name}",
        )


def _stage_colmap_images(
    paths: PipelinePaths,
    sweep_selection: Sequence[SweepSelection],
    references: Sequence[ReferenceSelection],
    sweep: ValidatedSequence,
    target_fps_hz: float,
) -> None:
    if paths.workspace_images.exists() or paths.workspace_images.is_symlink():
        _validate_staged_colmap_images(
            paths.workspace_images,
            sweep_selection,
            references,
        )
    else:
        relative_names = [
            *(f"sweep/{selection.staged_name}" for selection in sweep_selection),
            *(reference.staged_relative_name for reference in references),
        ]
        _ensure_unique_colmap_basenames(relative_names)
        paths.workspace_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=".images.", suffix=".tmp", dir=str(paths.workspace_dir)
            )
        )
        try:
            sweep_dir = temporary / "sweep"
            sweep_dir.mkdir()
            for selection in sweep_selection:
                shutil.copy2(
                    selection.source_frame.image_path,
                    sweep_dir / selection.staged_name,
                )
            for reference in references:
                destination = temporary / reference.staged_relative_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(reference.source_frame.image_path, destination)
            os.replace(temporary, paths.workspace_images)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    _write_json_atomic(
        paths.reports_dir / "sweep_frame_selection.json",
        _sweep_selection_payload(sweep, target_fps_hz, sweep_selection),
    )
    _write_json_atomic(
        paths.reports_dir / "reference_selection.json",
        _reference_selection_payload(references),
    )


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to write scene_norm_dict.pth. Run this controller "
            "from an environment containing Shape-of-Motion dependencies."
        ) from exc
    return torch


def _run_colmap_stage(config: PipelineConfig, paths: PipelinePaths) -> None:
    if paths.database_path.exists() or paths.sparse_models_dir.exists():
        raise FileExistsError(
            "Incomplete COLMAP output already exists without a completion marker. "
            f"Use a new run_id or inspect {paths.workspace_dir}."
        )
    _require_torch()
    colmap = resolve_executable(config.colmap.command, "COLMAP")
    run_colmap(
        colmap,
        paths.workspace_images,
        paths.database_path,
        paths.sparse_models_dir,
        config.colmap.camera_model,
        paths.reports_dir / "commands.log",
    )


def _reference_name_to_label(
    references: Sequence[ReferenceSelection],
) -> dict[str, str]:
    return {
        reference.staged_relative_name.replace("\\", "/"): reference.view_id
        for reference in references
    }


def _select_model_stage(
    config: PipelineConfig,
    paths: PipelinePaths,
    sweep_selection: Sequence[SweepSelection],
    references: Sequence[ReferenceSelection],
) -> ModelCandidate:
    expected_references = set(_reference_name_to_label(references))
    selected = choose_model(
        paths.sparse_models_dir,
        expected_references,
        paths.reports_dir / "model_candidates.json",
    )
    staged_sweep = [
        paths.workspace_images / "sweep" / selection.staged_name
        for selection in sweep_selection
    ]
    validate_sweep_registration(
        selected,
        staged_sweep,
        config.colmap.min_sweep_registration_ratio,
        paths.reports_dir / "sweep_registration.json",
    )
    _write_json_atomic(
        paths.reports_dir / "selected_model.json",
        {
            "model_dir": str(selected.path),
            "registered_names": sorted(selected.registered_names),
            "registered_sweep_count": selected.registered_sweep_count,
            "point_count": selected.point_count,
        },
    )
    return selected


def _load_selected_model(paths: PipelinePaths) -> ModelCandidate:
    selection_path = paths.reports_dir / "selected_model.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    payload = _read_json(selection_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid selected model report: {selection_path}")
    model_dir_value = payload.get("model_dir")
    registered_names = payload.get("registered_names")
    if not isinstance(model_dir_value, str) or not isinstance(registered_names, list):
        raise ValueError(f"Invalid selected model report: {selection_path}")
    model_dir = Path(model_dir_value).expanduser().resolve(strict=True)
    required = tuple(
        model_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin")
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Selected COLMAP model is incomplete: {missing}")
    return ModelCandidate(
        path=model_dir,
        registered_names=frozenset(str(name) for name in registered_names),
        registered_sweep_count=int(payload["registered_sweep_count"]),
        point_count=int(payload["point_count"]),
    )


def _packaged_sweep_records(
    selected: ModelCandidate,
    sweep_selection: Sequence[SweepSelection],
) -> tuple[list[SweepSelection], list[dict[str, Any]]]:
    packaged: list[SweepSelection] = []
    records: list[dict[str, Any]] = []
    for selection in sweep_selection:
        registered_name = f"sweep/{selection.staged_name}"
        if registered_name not in selected.registered_names:
            continue
        packaged.append(selection)
        records.append(
            {
                "registered_name": registered_name,
                "training_name": selection.staged_name,
                "source_frame_name": selection.source_frame.frame_name,
                "source_index": selection.source_frame.source_index,
                "source_time_sec": selection.source_frame.time_sec,
                "source_image": str(selection.source_frame.image_path),
                "source_mask": str(selection.source_frame.mask_path),
            }
        )
    if not packaged:
        raise RuntimeError("Selected COLMAP model registered no packaged sweep frame")
    return packaged, records


def _validate_existing_packaged_dataset(
    dataset_dir: Path,
    selected: ModelCandidate,
    packaged: Sequence[SweepSelection],
) -> None:
    if not dataset_dir.is_dir() or dataset_dir.is_symlink():
        raise ValueError(f"Static dataset is not a regular directory: {dataset_dir}")
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    image_paths = _ordered_pngs(image_dir, "existing packaged images")
    mask_paths = _ordered_pngs(mask_dir, "existing packaged masks")
    expected_names = {selection.staged_name for selection in packaged}
    if {path.name for path in image_paths} != expected_names:
        raise ValueError(
            "Existing packaged image names do not match selected COLMAP frames"
        )
    if {path.name for path in mask_paths} != expected_names:
        raise ValueError(
            "Existing packaged mask names do not match selected COLMAP frames"
        )
    by_name = {selection.staged_name: selection for selection in packaged}
    for name, selection in by_name.items():
        _require_same_file_content(
            image_dir / name,
            selection.source_frame.image_path,
            f"Packaged training image {name}",
        )
        _require_same_file_content(
            mask_dir / name,
            selection.source_frame.mask_path,
            f"Packaged training mask {name}",
        )
    packaged_model = dataset_dir / "colmap" / "sparse" / "0"
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        _require_same_file_content(
            packaged_model / filename,
            selected.path / filename,
            f"Packaged COLMAP model file {filename}",
        )
    cache_dir = dataset_dir / "flow3d_preprocessed"
    if not cache_dir.is_dir() or cache_dir.is_symlink():
        raise ValueError(f"Static dataset cache is not a directory: {cache_dir}")


def _validate_existing_packaged_references(
    references_dir: Path,
    references: Sequence[ReferenceSelection],
) -> None:
    if not references_dir.is_dir() or references_dir.is_symlink():
        raise ValueError(
            f"Reference output is not a regular directory: {references_dir}"
        )
    expected = {
        f"{reference.view_id}_ref.png": reference.source_frame.image_path
        for reference in references
    }
    actual_names = {
        path.name for path in references_dir.glob("*.png") if path.is_file()
    }
    if actual_names != set(expected):
        raise ValueError(
            "Existing packaged references do not match resolved references: "
            f"expected={sorted(expected)}, actual={sorted(actual_names)}"
        )
    for name, source in expected.items():
        _require_same_file_content(
            references_dir / name,
            source,
            f"Packaged reference image {name}",
        )


def _package_static_dataset(
    paths: PipelinePaths,
    selected: ModelCandidate,
    sweep_selection: Sequence[SweepSelection],
    references: Sequence[ReferenceSelection],
) -> None:
    packaged, packaged_records = _packaged_sweep_records(
        selected,
        sweep_selection,
    )
    report_payload = {
        "format": "registered_static_training_frames",
        "version": 1,
        "frames": packaged_records,
    }
    report_path = paths.reports_dir / "packaged_sweep_frames.json"
    if paths.dataset_dir.exists() or paths.dataset_dir.is_symlink():
        _validate_existing_packaged_dataset(paths.dataset_dir, selected, packaged)
    if paths.references_dir.exists() or paths.references_dir.is_symlink():
        _validate_existing_packaged_references(paths.references_dir, references)
    if report_path.exists() and _read_json(report_path) != report_payload:
        raise ValueError(
            f"Existing packaged-frame report does not match resolved inputs: "
            f"{report_path}"
        )
    if paths.dataset_dir.exists() and paths.references_dir.exists():
        _write_json_atomic(report_path, report_payload)
        return

    temporary_root = Path(
        tempfile.mkdtemp(prefix=".package.", suffix=".tmp", dir=str(paths.run_dir))
    )
    try:
        if not paths.dataset_dir.exists():
            dataset = temporary_root / "sweep_colmap_dataset"
            images = dataset / "images"
            masks = dataset / "masks"
            model = dataset / "colmap" / "sparse" / "0"
            cache = dataset / "flow3d_preprocessed"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            model.parent.mkdir(parents=True)
            cache.mkdir(parents=True)
            for selection in packaged:
                shutil.copy2(
                    selection.source_frame.image_path,
                    images / selection.staged_name,
                )
                shutil.copy2(
                    selection.source_frame.mask_path,
                    masks / selection.staged_name,
                )
            shutil.copytree(selected.path, model)
        if not paths.references_dir.exists():
            references_dir = temporary_root / "references"
            references_dir.mkdir()
            for reference in references:
                shutil.copy2(
                    reference.source_frame.image_path,
                    references_dir / f"{reference.view_id}_ref.png",
                )
        if not paths.references_dir.exists():
            os.replace(temporary_root / "references", paths.references_dir)
        if not paths.dataset_dir.exists():
            os.replace(
                temporary_root / "sweep_colmap_dataset",
                paths.dataset_dir,
            )
        _write_json_atomic(report_path, report_payload)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def validate_camera_group_assignments(
    records: Sequence[dict[str, Any]],
    source_to_group: Mapping[str, str],
) -> dict[str, int]:
    ids_by_source: dict[str, set[int]] = {source: set() for source in source_to_group}
    for record in records:
        source = str(record["source"])
        if source not in ids_by_source:
            raise ValueError(f"Unexpected camera source in registered model: {source}")
        ids_by_source[source].add(int(record["camera_id"]))
    invalid_sources = {
        source: sorted(camera_ids)
        for source, camera_ids in ids_by_source.items()
        if len(camera_ids) != 1
    }
    if invalid_sources:
        raise RuntimeError(
            "Every source must use exactly one COLMAP camera ID; got "
            f"{invalid_sources}"
        )
    ids_by_group: dict[str, set[int]] = {}
    for source, group in source_to_group.items():
        ids_by_group.setdefault(group, set()).update(ids_by_source[source])
    invalid_groups = {
        group: sorted(camera_ids)
        for group, camera_ids in ids_by_group.items()
        if len(camera_ids) != 1
    }
    if invalid_groups:
        raise RuntimeError(
            "Every configured camera_group must use exactly one COLMAP camera ID; "
            f"got {invalid_groups}"
        )
    group_ids = {
        group: next(iter(camera_ids)) for group, camera_ids in ids_by_group.items()
    }
    reverse: dict[int, list[str]] = {}
    for group, camera_id in group_ids.items():
        reverse.setdefault(camera_id, []).append(group)
    reused = {
        camera_id: sorted(groups)
        for camera_id, groups in reverse.items()
        if len(groups) != 1
    }
    if reused:
        raise RuntimeError(
            "Distinct camera_group values unexpectedly share COLMAP camera IDs: "
            f"{reused}"
        )
    return group_ids


def _validate_registered_basenames(model_dir: Path) -> None:
    images = read_images_binary(model_dir / "images.bin")
    relative_names = [image.name.replace("\\", "/") for image in images.values()]
    _ensure_unique_colmap_basenames(relative_names)


def _export_reference_cameras(
    config: PipelineConfig,
    paths: PipelinePaths,
    selected: ModelCandidate,
    sweep_selection: Sequence[SweepSelection],
    references: Sequence[ReferenceSelection],
) -> None:
    _validate_registered_basenames(selected.path)
    cameras = read_cameras_binary(selected.path / "cameras.bin")
    images = read_images_binary(selected.path / "images.bin")
    points = read_points3d_binary(selected.path / "points3D.bin")
    sweep_w2cs: list[np.ndarray] = []
    for image in images.values():
        if image.name.replace("\\", "/").startswith("sweep/"):
            _, world_to_camera = get_intrinsics_extrinsics(image, cameras)
            sweep_w2cs.append(world_to_camera)
    if not sweep_w2cs:
        raise RuntimeError("Selected COLMAP model has no registered sweep cameras")
    scene_scale, scene_transform = compute_colmap_scene_norm(
        np.stack([point.xyz for point in points.values()]).astype(np.float64),
        np.stack(sweep_w2cs).astype(np.float64),
    )
    reference_mapping = _reference_name_to_label(references)
    records, reference_records = export_camera_records(
        selected.path,
        reference_mapping,
        scene_scale,
        scene_transform,
    )
    source_to_group = {
        "sweep": config.sweep.camera_group,
        **{view.view_id: view.camera_group for view in config.static_views},
    }
    camera_group_ids = validate_camera_group_assignments(records, source_to_group)
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
        paths.reports_dir / "registered_cameras.json",
        {"cameras": records},
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
    registered_sweep_names = sorted(
        record["image_name"] for record in records if record["source"] == "sweep"
    )
    requested_sweep_names = sorted(
        f"sweep/{selection.staged_name}" for selection in sweep_selection
    )
    missing_sweep_names = sorted(
        set(requested_sweep_names) - set(registered_sweep_names)
    )
    summary = {
        "format": "joint_colmap_frame_dataset",
        "version": 1,
        "source_config": str(config.config_path),
        "source_manifest": str(paths.source_manifest_path),
        "transforms": {
            "video_decode": False,
            "rotate": False,
            "crop": False,
            "resize": False,
        },
        "colmap": {
            "camera_model": config.colmap.camera_model,
            "camera_grouping": "single_camera_per_folder",
            "camera_group_ids": camera_group_ids,
            "matcher": "exhaustive_matcher",
            "selected_model": str(selected.path),
            "requested_sweep_count": len(requested_sweep_names),
            "registered_sweep_count": len(registered_sweep_names),
            "sweep_registration_ratio": (
                len(registered_sweep_names) / len(requested_sweep_names)
            ),
            "minimum_sweep_registration_ratio": (
                config.colmap.min_sweep_registration_ratio
            ),
            "missing_sweep_names": missing_sweep_names,
            "registered_reference_labels": sorted(reference_records),
            "registered_image_count": len(records),
            "point_count": len(points),
            "mean_point_reprojection_error": float(errors.mean()),
            "median_point_reprojection_error": float(np.median(errors)),
        },
        "scene_normalization": {
            "scale": scene_scale,
            "transform": scene_transform.astype(float).tolist(),
        },
        "outputs": {
            "dataset_dir": str(paths.dataset_dir),
            "training_images": str(paths.dataset_dir / "images"),
            "training_masks": str(paths.dataset_dir / "masks"),
            "sparse_model": str(paths.dataset_dir / "colmap" / "sparse" / "0"),
            "scene_norm": str(scene_norm_path),
            "registered_cameras": str(paths.reports_dir / "registered_cameras.json"),
            "reference_cameras": str(paths.references_dir / "reference_cameras.json"),
        },
        "remaining_training_inputs": [],
        "registered_sweep_names": registered_sweep_names,
    }
    _write_json_atomic(
        paths.reports_dir / "colmap_registration_summary.json",
        summary,
    )


def _validate_static_dataset(
    config: PipelineConfig,
    paths: PipelinePaths,
    references: Sequence[ReferenceSelection],
) -> dict[str, Any]:
    image_dir = paths.dataset_dir / "images"
    mask_dir = paths.dataset_dir / "masks"
    model_dir = paths.dataset_dir / "colmap" / "sparse" / "0"
    scene_norm_path = paths.dataset_dir / "flow3d_preprocessed" / "scene_norm_dict.pth"
    image_paths = _ordered_pngs(image_dir, "static training images")
    mask_paths = _ordered_pngs(mask_dir, "static training masks")
    if [path.stem for path in image_paths] != [path.stem for path in mask_paths]:
        raise ValueError("Packaged static training image and mask stems do not match")
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        path = model_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    _validate_registered_basenames(model_dir)
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    registered = read_images_binary(model_dir / "images.bin")
    by_basename = {Path(image.name).name: image for image in registered.values()}
    expected_size: tuple[int, int] | None = None
    for image_path, mask_path in zip(image_paths, mask_paths, strict=True):
        image = _read_image(image_path, "packaged training image")
        mask = _read_image(mask_path, "packaged training mask")
        _validate_rgb_image(image, image_path)
        _validate_binary_mask(mask, mask_path)
        size = (image.shape[1], image.shape[0])
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise ValueError(
                f"Packaged sweep changes resolution from {expected_size} to {size}"
            )
        if mask.shape[:2] != image.shape[:2]:
            raise ValueError(f"Packaged image/mask size mismatch for {image_path.name}")
        if image_path.name not in by_basename:
            raise KeyError(
                f"Packaged image is not registered by COLMAP: {image_path.name}"
            )
        camera = cameras[by_basename[image_path.name].camera_id]
        if (int(camera.width), int(camera.height)) != size:
            raise ValueError(
                f"COLMAP camera size for {image_path.name} is "
                f"{camera.width}x{camera.height}, expected {size[0]}x{size[1]}"
            )
    if not scene_norm_path.is_file():
        raise FileNotFoundError(scene_norm_path)
    torch = _require_torch()
    scene_norm = torch.load(scene_norm_path, map_location="cpu", weights_only=False)
    if not isinstance(scene_norm, dict) or set(scene_norm) != {"scale", "transfm"}:
        raise ValueError(f"Invalid scene normalization payload: {scene_norm_path}")
    scale = float(scene_norm["scale"])
    transform = scene_norm["transfm"]
    if not math.isfinite(scale) or scale <= 0.0 or tuple(transform.shape) != (4, 4):
        raise ValueError(f"Invalid scene normalization values: {scene_norm_path}")
    reference_path = paths.references_dir / "reference_cameras.json"
    reference_payload = _read_json(reference_path)
    expected_reference_ids = {reference.view_id for reference in references}
    reference_records = (
        reference_payload.get("references")
        if isinstance(reference_payload, dict)
        else None
    )
    if (
        not isinstance(reference_payload, dict)
        or reference_payload.get("format") != "fixed_reference_cameras"
        or reference_payload.get("version") != 1
        or not isinstance(reference_records, Mapping)
        or set(reference_records) != expected_reference_ids
    ):
        raise ValueError(f"Invalid fixed reference camera manifest: {reference_path}")
    assert isinstance(reference_records, Mapping)
    expected_reference_names = {
        f"{reference.view_id}_ref.png" for reference in references
    }
    actual_reference_names = {
        path.name for path in paths.references_dir.glob("*.png") if path.is_file()
    }
    if actual_reference_names != expected_reference_names:
        raise ValueError(
            "Packaged reference RGB files do not match configured views: "
            f"expected={sorted(expected_reference_names)}, "
            f"actual={sorted(actual_reference_names)}"
        )
    for reference in references:
        reference_rgb_path = paths.references_dir / f"{reference.view_id}_ref.png"
        reference_rgb = _read_image(reference_rgb_path, "packaged reference RGB")
        _validate_rgb_image(reference_rgb, reference_rgb_path)
        record = reference_records[reference.view_id]
        if not isinstance(record, Mapping):
            raise ValueError(
                f"Invalid reference camera record for {reference.view_id}: "
                f"{reference_path}"
            )
        if "image_width" not in record or "image_height" not in record:
            raise ValueError(
                f"Reference camera record lacks dimensions for {reference.view_id}: "
                f"{reference_path}"
            )
        expected_reference_size = (
            int(record["image_width"]),
            int(record["image_height"]),
        )
        reference_size = (reference_rgb.shape[1], reference_rgb.shape[0])
        if reference_size != expected_reference_size:
            raise ValueError(
                f"Reference RGB/COLMAP size mismatch for {reference.view_id}: "
                f"RGB={reference_size}, COLMAP={expected_reference_size}"
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
        "reference_camera_count": len(expected_reference_ids),
        "reference_cameras": str(reference_path),
        "source_manifest": str(paths.source_manifest_path),
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


def _validate_completed_stage(paths: PipelinePaths, stage: str) -> None:
    required_by_stage: dict[str, tuple[Path, ...]] = {
        "validate_inputs": (
            paths.resolved_config_path,
            paths.source_manifest_path,
        ),
        "stage_colmap_images": (
            paths.workspace_images,
            paths.reports_dir / "sweep_frame_selection.json",
            paths.reports_dir / "reference_selection.json",
        ),
        "run_colmap": (
            paths.database_path,
            paths.sparse_models_dir,
            paths.reports_dir / "commands.log",
        ),
        "select_model": (
            paths.reports_dir / "selected_model.json",
            paths.reports_dir / "model_candidates.json",
            paths.reports_dir / "sweep_registration.json",
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
    missing = [str(path) for path in required_by_stage[stage] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Stage {stage!r} is marked complete but artifacts are missing: {missing}"
        )


def _stage_bounds(
    from_stage: str | None,
    until_stage: str | None,
) -> tuple[int, int]:
    start = 0 if from_stage is None else STAGES.index(from_stage)
    end = len(STAGES) - 1 if until_stage is None else STAGES.index(until_stage)
    if start > end:
        raise ValueError("--from-stage must not come after --until-stage")
    return start, end


def run_pipeline(
    config: PipelineConfig,
    *,
    dry_run: bool = False,
    from_stage: str | None = None,
    until_stage: str | None = None,
) -> Path | None:
    sweep, static_views = validate_source_sequences(config)
    sweep_selection = select_sweep_frames(
        sweep, config.colmap.sweep_sample_fps_hz
    )
    references = tuple(select_reference_frame(view) for view in static_views)
    paths = _pipeline_paths(config)
    source_manifest = _source_manifest_payload(config, sweep, static_views)
    resolved_config = _resolved_config_payload(
        config,
        paths,
        sweep,
        static_views,
        sweep_selection,
        references,
    )
    config_identity = _canonical_json_identity(
        {"resolved_config": resolved_config, "source_manifest": source_manifest}
    )
    start, end = _stage_bounds(from_stage, until_stage)
    if dry_run:
        print(
            json.dumps(
                {
                    "config_identity": config_identity,
                    "run_dir": str(paths.run_dir),
                    "stages": list(STAGES[start : end + 1]),
                    "sweep_source_frames": len(sweep.frames),
                    "sweep_selected_frames": len(sweep_selection),
                    "static_references": {
                        reference.view_id: reference.source_frame.frame_name
                        for reference in references
                    },
                    "transforms": resolved_config["transforms"],
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
    if "validate_inputs" in completed:
        if _read_json(paths.source_manifest_path) != source_manifest:
            raise ValueError(
                f"Canonical source manifest changed after validation: "
                f"{paths.source_manifest_path}"
            )
        if _read_json(paths.resolved_config_path) != resolved_config:
            raise ValueError(
                f"Resolved pipeline config changed after validation: "
                f"{paths.resolved_config_path}"
            )

    for stage_index in range(start, end + 1):
        stage = STAGES[stage_index]
        if stage in state["completed_stages"]:
            print(f"Skipping completed stage: {stage}", flush=True)
            continue
        if stage_index != len(state["completed_stages"]):
            expected = STAGES[len(state["completed_stages"])]
            raise RuntimeError(
                f"Cannot run {stage!r}; expected next stage {expected!r}"
            )
        print(f"Running stage: {stage}", flush=True)
        if stage == "validate_inputs":
            if paths.source_manifest_path.exists():
                existing = _read_json(paths.source_manifest_path)
                if existing != source_manifest:
                    raise ValueError(
                        f"Canonical source manifest already exists with different "
                        f"inputs: {paths.source_manifest_path}"
                    )
            else:
                _write_json_atomic(paths.source_manifest_path, source_manifest)
            _write_json_atomic(paths.resolved_config_path, resolved_config)
        elif stage == "stage_colmap_images":
            _stage_colmap_images(
                paths,
                sweep_selection,
                references,
                sweep,
                config.colmap.sweep_sample_fps_hz,
            )
        elif stage == "run_colmap":
            _run_colmap_stage(config, paths)
        elif stage == "select_model":
            _select_model_stage(
                config,
                paths,
                sweep_selection,
                references,
            )
        elif stage == "package_static_dataset":
            _package_static_dataset(
                paths,
                _load_selected_model(paths),
                sweep_selection,
                references,
            )
        elif stage == "export_reference_cameras":
            _export_reference_cameras(
                config,
                paths,
                _load_selected_model(paths),
                sweep_selection,
                references,
            )
        elif stage == "validate_static_ready":
            ready = _validate_static_dataset(config, paths, references)
            _write_json_atomic(
                paths.reports_dir / "STATIC_DATASET_READY.json",
                ready,
            )
        else:
            raise AssertionError(stage)
        _complete_stage(paths, state, stage)
        state = _load_state(paths, config_identity)
    ready_path = paths.reports_dir / "STATIC_DATASET_READY.json"
    if ready_path.is_file():
        print(
            "Pre-static pipeline completed successfully:\n"
            f"  static dataset: {paths.dataset_dir}\n"
            f"  reference cameras: {paths.references_dir / 'reference_cameras.json'}\n"
            f"  ready report: {ready_path}",
            flush=True,
        )
        return ready_path
    print(
        f"Pre-static pipeline stopped after {STAGES[end]!r}; output: {paths.run_dir}",
        flush=True,
    )
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate canonical RGB/mask sequences, select one reference per "
            "fixed camera, sample a sweep, run joint COLMAP, and package the "
            "static 3DGS training dataset without resizing."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-stage", choices=STAGES, default=None)
    parser.add_argument("--until-stage", choices=STAGES, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_pipeline_config(args.config)
    run_pipeline(
        config,
        dry_run=args.dry_run,
        from_stage=args.from_stage,
        until_stage=args.until_stage,
    )


if __name__ == "__main__":
    main()
