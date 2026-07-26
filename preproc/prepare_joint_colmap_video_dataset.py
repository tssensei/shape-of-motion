"""Build one joint COLMAP model from a sweep video and static references.

The script intentionally uses only decoded RGB video frames. It assigns the
calibration sweep to one COLMAP camera and all static reference images to a
second shared camera, requires every reference to register in one sparse model,
and packages that model for the repository's static COLMAP dataset loader.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COLMAP_READER_PATH = REPO_ROOT / "flow3d" / "data" / "colmap.py"
COLMAP_READER_SPEC = importlib.util.spec_from_file_location(
    "flow3d_data_colmap_joint_video",
    COLMAP_READER_PATH,
)
if COLMAP_READER_SPEC is None or COLMAP_READER_SPEC.loader is None:
    raise ImportError(f"Could not load {COLMAP_READER_PATH}")
colmap_reader = importlib.util.module_from_spec(COLMAP_READER_SPEC)
COLMAP_READER_SPEC.loader.exec_module(colmap_reader)
get_intrinsics_extrinsics = colmap_reader.get_intrinsics_extrinsics
read_cameras_binary = colmap_reader.read_cameras_binary
read_images_binary = colmap_reader.read_images_binary
read_points3d_binary = colmap_reader.read_points3d_binary


REFERENCE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class ReferenceSpec:
    label: str
    video_path: Path
    timestamp_sec: float


@dataclass(frozen=True)
class ModelCandidate:
    path: Path
    registered_names: frozenset[str]
    registered_sweep_count: int
    point_count: int


def parse_reference_spec(value: str) -> ReferenceSpec:
    try:
        label, remainder = value.split("=", 1)
        video_text, timestamp_text = remainder.rsplit("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--reference must be LABEL=VIDEO_PATH=TIMESTAMP_SECONDS"
        ) from exc
    if not REFERENCE_LABEL_PATTERN.fullmatch(label):
        raise argparse.ArgumentTypeError(
            "reference LABEL must contain only letters, digits, underscores, or "
            "hyphens and must start with a letter or digit"
        )
    if not video_text:
        raise argparse.ArgumentTypeError("reference VIDEO_PATH must not be empty")
    try:
        timestamp_sec = float(timestamp_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"reference timestamp must be a number, got {timestamp_text!r}"
        ) from exc
    if not math.isfinite(timestamp_sec) or timestamp_sec < 0.0:
        raise argparse.ArgumentTypeError(
            "reference timestamp must be finite and non-negative"
        )
    return ReferenceSpec(
        label=label,
        video_path=Path(video_text).expanduser(),
        timestamp_sec=timestamp_sec,
    )


def resolve_executable(command: str, label: str) -> str:
    candidate = Path(command).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(
            f"{label} executable {command!r} was not found. Activate the intended "
            "cluster environment or pass its path explicitly."
        )
    return resolved


def format_float(value: float) -> str:
    return format(value, ".12g")


def run_command(command: Sequence[str], log_path: Path) -> None:
    rendered = shlex.join([str(part) for part in command])
    print(f"$ {rendered}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n")
        log.flush()
        process = subprocess.Popen(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError(f"Failed to capture output for command: {rendered}")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
        log.write(f"[exit {return_code}]\n")
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {rendered}")


def validate_inputs(
    sweep_video: Path,
    references: Sequence[ReferenceSpec],
    out_dir: Path,
    sweep_fps: float,
    sweep_start_sec: float,
    sweep_end_sec: float | None,
) -> None:
    if not sweep_video.is_file():
        raise FileNotFoundError(sweep_video)
    if not references:
        raise ValueError("At least one --reference is required")
    labels = [reference.label for reference in references]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Reference labels must be unique, got {labels}")
    for reference in references:
        if reference.label == "sweep":
            raise ValueError("Reference label 'sweep' is reserved")
        if not reference.video_path.is_file():
            raise FileNotFoundError(reference.video_path)
    if not math.isfinite(sweep_fps) or sweep_fps <= 0.0:
        raise ValueError("--sweep-fps must be finite and positive")
    if not math.isfinite(sweep_start_sec) or sweep_start_sec < 0.0:
        raise ValueError("--sweep-start-sec must be finite and non-negative")
    if sweep_end_sec is not None:
        if not math.isfinite(sweep_end_sec) or sweep_end_sec <= sweep_start_sec:
            raise ValueError(
                "--sweep-end-sec must be finite and greater than --sweep-start-sec"
            )
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(
            f"Output directory already exists: {out_dir}. Choose a new output path."
        )


def extract_sweep_frames(
    ffmpeg: str,
    video_path: Path,
    output_dir: Path,
    fps: float,
    start_sec: float,
    end_sec: float | None,
    log_path: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True)
    command = [ffmpeg, "-hide_banner", "-nostdin", "-n"]
    if start_sec > 0.0:
        command.extend(["-ss", format_float(start_sec)])
    if end_sec is not None:
        duration_sec = end_sec - start_sec
        command.extend(["-t", format_float(duration_sec)])
    command.extend(
        [
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-vf",
            f"fps={format_float(fps)}",
            "-start_number",
            "0",
            str(output_dir / "sweep_%06d.png"),
        ]
    )
    run_command(command, log_path)
    frames = sorted(output_dir.glob("sweep_*.png"))
    if not frames:
        raise RuntimeError(f"FFmpeg exported no sweep frames from {video_path}")
    return frames


def extract_reference_frame(
    ffmpeg: str,
    reference: ReferenceSpec,
    output_dir: Path,
    log_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{reference.label}_ref.png"
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-n",
        "-ss",
        format_float(reference.timestamp_sec),
        "-i",
        str(reference.video_path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-an",
        "-sn",
        str(output_path),
    ]
    run_command(command, log_path)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"FFmpeg did not export {reference.label} at "
            f"{reference.timestamp_sec:.6g} seconds"
        )
    return output_path


def run_colmap(
    colmap: str,
    image_dir: Path,
    database_path: Path,
    sparse_models_dir: Path,
    camera_model: str,
    log_path: Path,
) -> None:
    run_command(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--ImageReader.camera_model",
            camera_model,
            "--ImageReader.single_camera_per_folder",
            "1",
        ],
        log_path,
    )
    run_command(
        [
            colmap,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
        ],
        log_path,
    )
    sparse_models_dir.mkdir()
    run_command(
        [
            colmap,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_models_dir),
        ],
        log_path,
    )


def model_sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.name))
    except ValueError:
        return (1, path.name)


def inspect_model_candidates(
    sparse_models_dir: Path,
    expected_reference_names: set[str],
) -> tuple[list[dict[str, Any]], list[ModelCandidate]]:
    summaries: list[dict[str, Any]] = []
    accepted: list[ModelCandidate] = []
    model_dirs = sorted(
        [path for path in sparse_models_dir.iterdir() if path.is_dir()],
        key=model_sort_key,
    )
    for model_dir in model_dirs:
        cameras_path = model_dir / "cameras.bin"
        images_path = model_dir / "images.bin"
        points_path = model_dir / "points3D.bin"
        if (
            not cameras_path.is_file()
            or not images_path.is_file()
            or not points_path.is_file()
        ):
            summaries.append(
                {
                    "model_dir": str(model_dir),
                    "complete_binary_model": False,
                }
            )
            continue
        images = read_images_binary(images_path)
        points = read_points3d_binary(points_path)
        registered_names = frozenset(
            image.name.replace("\\", "/") for image in images.values()
        )
        registered_sweep = sorted(
            name for name in registered_names if name.startswith("sweep/")
        )
        missing_references = sorted(expected_reference_names - registered_names)
        summary = {
            "model_dir": str(model_dir),
            "complete_binary_model": True,
            "registered_image_count": len(registered_names),
            "registered_sweep_count": len(registered_sweep),
            "point_count": len(points),
            "missing_references": missing_references,
        }
        summaries.append(summary)
        if not missing_references and registered_sweep and points:
            accepted.append(
                ModelCandidate(
                    path=model_dir,
                    registered_names=registered_names,
                    registered_sweep_count=len(registered_sweep),
                    point_count=len(points),
                )
            )
    return summaries, accepted


def choose_model(
    sparse_models_dir: Path,
    expected_reference_names: set[str],
    report_path: Path,
) -> ModelCandidate:
    summaries, accepted = inspect_model_candidates(
        sparse_models_dir,
        expected_reference_names,
    )
    write_json(report_path, {"models": summaries})
    if not accepted:
        raise RuntimeError(
            "No COLMAP model registered every reference image together with at "
            f"least one sweep frame. See {report_path}."
        )
    return max(
        accepted,
        key=lambda candidate: (
            candidate.registered_sweep_count,
            len(candidate.registered_names),
            candidate.point_count,
        ),
    )


def validate_sweep_registration(
    selected: ModelCandidate,
    sweep_frames: Sequence[Path],
    min_registration_ratio: float,
    report_path: Path,
) -> None:
    requested_names = {f"sweep/{frame.name}" for frame in sweep_frames}
    registered_names = requested_names & selected.registered_names
    missing_names = sorted(requested_names - registered_names)
    ratio = len(registered_names) / len(requested_names)
    write_json(
        report_path,
        {
            "requested_sweep_count": len(requested_names),
            "registered_sweep_count": len(registered_names),
            "registration_ratio": ratio,
            "minimum_registration_ratio": min_registration_ratio,
            "missing_sweep_names": missing_names,
        },
    )
    if ratio < min_registration_ratio:
        raise RuntimeError(
            f"Sweep registration ratio {ratio:.6g} is below the required "
            f"{min_registration_ratio:.6g}. See {report_path}."
        )


def rotation_aligning_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm <= 1e-12 or target_norm <= 1e-12:
        raise ValueError("Cannot align a zero-length direction vector")
    source = source / source_norm
    target = target / target_norm
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-12:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(source)))] = 1.0
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    x, y, z = cross
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + skew
        + skew @ skew * ((1.0 - cosine) / (sine * sine))
    )


def compute_colmap_scene_norm(
    points_xyz: np.ndarray,
    sweep_w2cs: np.ndarray,
) -> tuple[float, np.ndarray]:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3 or len(points_xyz) == 0:
        raise ValueError(
            f"Expected non-empty points shaped (N,3), got {points_xyz.shape}"
        )
    if sweep_w2cs.ndim != 3 or sweep_w2cs.shape[1:] != (4, 4) or len(sweep_w2cs) == 0:
        raise ValueError(
            f"Expected non-empty sweep poses shaped (N,4,4), got {sweep_w2cs.shape}"
        )
    finite_points = points_xyz[np.isfinite(points_xyz).all(axis=1)]
    if len(finite_points) == 0:
        raise ValueError("COLMAP sparse model contains no finite 3D points")
    scene_center = finite_points.mean(axis=0)
    low = np.quantile(finite_points - scene_center, 0.05, axis=0)
    high = np.quantile(finite_points - scene_center, 0.95, axis=0)
    scale = float(np.max(high - low) / 2.0)
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"Invalid COLMAP scene scale: {scale}")

    original_up = -sweep_w2cs[:, 1, :3].mean(axis=0)
    rotation = rotation_aligning_vectors(
        original_up,
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ scene_center
    return scale, transform


def normalized_world_to_camera(
    world_to_camera: np.ndarray,
    scale: float,
    transform: np.ndarray,
) -> np.ndarray:
    normalized = world_to_camera @ np.linalg.inv(transform)
    normalized[:3, 3] /= scale
    return normalized


def source_for_image(
    image_name: str,
    reference_name_to_label: dict[str, str],
) -> str:
    normalized = image_name.replace("\\", "/")
    if normalized.startswith("sweep/"):
        return "sweep"
    if normalized in reference_name_to_label:
        return reference_name_to_label[normalized]
    raise ValueError(f"Unexpected COLMAP image name: {image_name}")


def export_camera_records(
    model_dir: Path,
    reference_name_to_label: dict[str, str],
    scale: float,
    transform: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    images = read_images_binary(model_dir / "images.bin")
    reference_labels = set(reference_name_to_label.values())
    records: list[dict[str, Any]] = []
    references: dict[str, dict[str, Any]] = {}
    for image in sorted(images.values(), key=lambda value: value.name):
        camera = cameras[image.camera_id]
        K4, world_to_camera = get_intrinsics_extrinsics(image, cameras)
        camera_to_world = np.linalg.inv(world_to_camera)
        normalized_w2c = normalized_world_to_camera(
            world_to_camera,
            scale,
            transform,
        )
        normalized_c2w = np.linalg.inv(normalized_w2c)
        source = source_for_image(image.name, reference_name_to_label)
        record = {
            "image_name": image.name.replace("\\", "/"),
            "source": source,
            "image_id": int(image.id),
            "camera_id": int(image.camera_id),
            "camera_model": camera.model,
            "image_width": int(camera.width),
            "image_height": int(camera.height),
            "camera_params": np.asarray(camera.params, dtype=float).tolist(),
            "K": K4[:3, :3].astype(float).tolist(),
            "qvec": np.asarray(image.qvec, dtype=float).tolist(),
            "tvec": np.asarray(image.tvec, dtype=float).tolist(),
            "world_to_camera": world_to_camera.astype(float).tolist(),
            "camera_to_world": camera_to_world.astype(float).tolist(),
            "camera_center_world": camera_to_world[:3, 3].astype(float).tolist(),
            "normalized_world_to_camera": normalized_w2c.astype(float).tolist(),
            "normalized_camera_to_world": normalized_c2w.astype(float).tolist(),
            "normalized_camera_center_world": normalized_c2w[:3, 3]
            .astype(float)
            .tolist(),
        }
        records.append(record)
        if source in reference_labels:
            if source in references:
                raise ValueError(f"Multiple registered reference images for {source}")
            references[source] = record
    missing = sorted(reference_labels - set(references))
    if missing:
        raise RuntimeError(f"Registered model is missing reference records: {missing}")
    return records, references


def validate_camera_groups(
    records: Sequence[dict[str, Any]],
    reference_labels: set[str],
) -> dict[str, int]:
    ids_by_source: dict[str, set[int]] = {
        "sweep": set(),
        **{label: set() for label in reference_labels},
    }
    for record in records:
        source = str(record["source"])
        if source not in ids_by_source:
            raise ValueError(f"Unexpected camera source in manifest: {source}")
        ids_by_source[source].add(int(record["camera_id"]))
    invalid = {
        source: sorted(camera_ids)
        for source, camera_ids in ids_by_source.items()
        if len(camera_ids) != 1
    }
    if invalid:
        raise RuntimeError(
            f"Each source must use exactly one COLMAP camera ID; got {invalid}."
        )
    sweep_camera_id = next(iter(ids_by_source["sweep"]))
    static_camera_ids = {next(iter(ids_by_source[label])) for label in reference_labels}
    if len(static_camera_ids) != 1:
        raise RuntimeError(
            "All static references must share one COLMAP camera ID; got "
            f"{ {label: sorted(ids_by_source[label]) for label in reference_labels} }."
        )
    static_camera_id = next(iter(static_camera_ids))
    if sweep_camera_id == static_camera_id:
        raise RuntimeError(
            "Sweep and static references must use different COLMAP camera IDs, "
            f"but both use {sweep_camera_id}."
        )
    return {
        "sweep": sweep_camera_id,
        "static_refs": static_camera_id,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def copy_selected_outputs(
    selected_model: Path,
    registered_names: frozenset[str],
    sweep_frames: Sequence[Path],
    reference_frames: dict[str, Path],
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    dataset_dir = out_dir / "sweep_colmap_dataset"
    dataset_images = dataset_dir / "images"
    dataset_model = dataset_dir / "colmap" / "sparse" / "0"
    dataset_cache = dataset_dir / "flow3d_preprocessed"
    references_dir = out_dir / "references"
    dataset_images.mkdir(parents=True)
    dataset_model.parent.mkdir(parents=True)
    dataset_cache.mkdir(parents=True)
    references_dir.mkdir(parents=True)
    for frame in sweep_frames:
        if f"sweep/{frame.name}" in registered_names:
            shutil.copy2(frame, dataset_images / frame.name)
    shutil.copytree(selected_model, dataset_model)
    for label, frame in reference_frames.items():
        shutil.copy2(frame, references_dir / f"{label}_ref.png")
    return dataset_dir, dataset_cache, references_dir


def build_summary(
    args: argparse.Namespace,
    selected: ModelCandidate,
    sweep_frames: Sequence[Path],
    records: Sequence[dict[str, Any]],
    points: dict[int, Any],
    scene_scale: float,
    scene_transform: np.ndarray,
    camera_group_ids: dict[str, int],
    dataset_dir: Path,
) -> dict[str, Any]:
    registered_sweep_names = sorted(
        record["image_name"] for record in records if record["source"] == "sweep"
    )
    requested_sweep_names = [f"sweep/{frame.name}" for frame in sweep_frames]
    missing_sweep_names = sorted(
        set(requested_sweep_names) - set(registered_sweep_names)
    )
    errors = np.asarray(
        [float(point.error) for point in points.values()],
        dtype=np.float64,
    )
    return {
        "format": "joint_colmap_video_dataset",
        "version": 1,
        "inputs": {
            "sweep_video": str(args.sweep_video),
            "references": [
                {
                    "label": reference.label,
                    "video_path": str(reference.video_path),
                    "timestamp_sec": reference.timestamp_sec,
                }
                for reference in args.reference
            ],
        },
        "extraction": {
            "sweep_fps": args.sweep_fps,
            "sweep_start_sec": args.sweep_start_sec,
            "sweep_end_sec": args.sweep_end_sec,
            "ffmpeg_autorotate": True,
            "cropping": False,
            "resizing": False,
        },
        "colmap": {
            "camera_model": args.camera_model,
            "camera_grouping": "single_camera_per_folder",
            "camera_group_ids": camera_group_ids,
            "matcher": "exhaustive_matcher",
            "selected_model": str(selected.path),
            "requested_sweep_count": len(sweep_frames),
            "registered_sweep_count": len(registered_sweep_names),
            "minimum_sweep_registration_ratio": (args.min_sweep_registration_ratio),
            "missing_sweep_names": missing_sweep_names,
            "registered_reference_labels": sorted(
                record["source"] for record in records if record["source"] != "sweep"
            ),
            "registered_image_count": len(records),
            "point_count": len(points),
            "mean_point_reprojection_error": (
                float(errors.mean()) if errors.size else None
            ),
            "median_point_reprojection_error": (
                float(np.median(errors)) if errors.size else None
            ),
        },
        "scene_normalization": {
            "scale": scene_scale,
            "transform": scene_transform.astype(float).tolist(),
        },
        "outputs": {
            "dataset_dir": str(dataset_dir),
            "training_images": str(dataset_dir / "images"),
            "sparse_model": str(dataset_dir / "colmap" / "sparse" / "0"),
            "scene_norm": str(
                dataset_dir / "flow3d_preprocessed" / "scene_norm_dict.pth"
            ),
            "registered_cameras": str(
                args.out_dir / "reports" / "registered_cameras.json"
            ),
            "reference_cameras": str(
                args.out_dir / "references" / "reference_cameras.json"
            ),
        },
        "remaining_training_inputs": [
            "Foreground/background masks under sweep_colmap_dataset/masks",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a calibration sweep and fixed-camera reference frames, run "
            "one joint COLMAP reconstruction, and package the registered camera "
            "poses for the static Shape-of-Motion pipeline."
        )
    )
    parser.add_argument("--sweep-video", required=True, type=Path)
    parser.add_argument(
        "--reference",
        action="append",
        required=True,
        type=parse_reference_spec,
        help="LABEL=VIDEO_PATH=TIMESTAMP_SECONDS; repeat once per static view",
    )
    parser.add_argument("--sweep-fps", type=float, default=3.0)
    parser.add_argument("--sweep-start-sec", type=float, default=0.0)
    parser.add_argument("--sweep-end-sec", type=float, default=None)
    parser.add_argument(
        "--min-sweep-registration-ratio",
        type=float,
        default=1.0,
        help="Fail if fewer than this fraction of extracted sweep frames register",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg-command", default="ffmpeg")
    parser.add_argument("--colmap-command", default="colmap")
    parser.add_argument(
        "--camera-model",
        choices=("SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"),
        default="SIMPLE_RADIAL",
    )
    args = parser.parse_args()

    args.sweep_video = args.sweep_video.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.reference = [
        ReferenceSpec(
            label=reference.label,
            video_path=reference.video_path.expanduser().resolve(),
            timestamp_sec=reference.timestamp_sec,
        )
        for reference in args.reference
    ]
    validate_inputs(
        args.sweep_video,
        args.reference,
        args.out_dir,
        args.sweep_fps,
        args.sweep_start_sec,
        args.sweep_end_sec,
    )
    if (
        not math.isfinite(args.min_sweep_registration_ratio)
        or args.min_sweep_registration_ratio <= 0.0
        or args.min_sweep_registration_ratio > 1.0
    ):
        raise ValueError("--min-sweep-registration-ratio must be in (0, 1]")
    ffmpeg = resolve_executable(args.ffmpeg_command, "FFmpeg")
    colmap = resolve_executable(args.colmap_command, "COLMAP")
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to write scene_norm_dict.pth. Run this script "
            "from the Shape-of-Motion environment."
        ) from exc

    args.out_dir.mkdir(parents=True)
    reports_dir = args.out_dir / "reports"
    reports_dir.mkdir()
    command_log = reports_dir / "commands.log"
    workspace_dir = args.out_dir / "colmap_workspace"
    colmap_images = workspace_dir / "images"
    sweep_image_dir = colmap_images / "sweep"
    database_path = workspace_dir / "database.db"
    sparse_models_dir = workspace_dir / "sparse_models"

    sweep_frames = extract_sweep_frames(
        ffmpeg,
        args.sweep_video,
        sweep_image_dir,
        args.sweep_fps,
        args.sweep_start_sec,
        args.sweep_end_sec,
        command_log,
    )
    reference_frames: dict[str, Path] = {}
    static_reference_dir = colmap_images / "static_refs"
    for reference in args.reference:
        reference_frames[reference.label] = extract_reference_frame(
            ffmpeg,
            reference,
            static_reference_dir,
            command_log,
        )

    run_colmap(
        colmap,
        colmap_images,
        database_path,
        sparse_models_dir,
        args.camera_model,
        command_log,
    )
    reference_name_to_label = {
        f"static_refs/{frame.name}": label for label, frame in reference_frames.items()
    }
    expected_reference_names = set(reference_name_to_label)
    selected = choose_model(
        sparse_models_dir,
        expected_reference_names,
        reports_dir / "model_candidates.json",
    )
    validate_sweep_registration(
        selected,
        sweep_frames,
        args.min_sweep_registration_ratio,
        reports_dir / "sweep_registration.json",
    )

    cameras = read_cameras_binary(selected.path / "cameras.bin")
    images = read_images_binary(selected.path / "images.bin")
    points = read_points3d_binary(selected.path / "points3D.bin")
    sweep_w2cs = []
    for image in images.values():
        if image.name.replace("\\", "/").startswith("sweep/"):
            _, world_to_camera = get_intrinsics_extrinsics(image, cameras)
            sweep_w2cs.append(world_to_camera)
    scene_scale, scene_transform = compute_colmap_scene_norm(
        np.stack([point.xyz for point in points.values()]).astype(np.float64),
        np.stack(sweep_w2cs).astype(np.float64),
    )
    records, reference_records = export_camera_records(
        selected.path,
        reference_name_to_label,
        scene_scale,
        scene_transform,
    )
    camera_group_ids = validate_camera_groups(
        records,
        set(reference_frames),
    )

    dataset_dir, dataset_cache, references_dir = copy_selected_outputs(
        selected.path,
        selected.registered_names,
        sweep_frames,
        reference_frames,
        args.out_dir,
    )
    torch.save(
        {
            "scale": float(scene_scale),
            "transfm": torch.from_numpy(scene_transform.astype(np.float32)),
        },
        dataset_cache / "scene_norm_dict.pth",
    )
    write_json(reports_dir / "registered_cameras.json", {"cameras": records})
    write_json(
        references_dir / "reference_cameras.json",
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
    summary = build_summary(
        args,
        selected,
        sweep_frames,
        records,
        points,
        scene_scale,
        scene_transform,
        camera_group_ids,
        dataset_dir,
    )
    write_json(reports_dir / "colmap_registration_summary.json", summary)

    print(
        "Joint COLMAP dataset prepared successfully:\n"
        f"  output: {args.out_dir}\n"
        f"  sweep registered: {selected.registered_sweep_count}/{len(sweep_frames)}\n"
        f"  references registered: {', '.join(sorted(reference_frames))}\n"
        f"  sparse points: {len(points)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
