"""Extract complete fixed-camera video frame sequences for modal analysis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Sequence


FRAME_DATASET_FORMAT = "fixed_camera_video_frames"
FRAME_DATASET_VERSION = 1
VIEW_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
ROTATION_CHOICES = ("auto", "clockwise", "counterclockwise")
ROTATION_FILTERS = {
    "clockwise": "transpose=clock",
    "counterclockwise": "transpose=cclock",
}


@dataclass(frozen=True)
class FixedCameraVideoSpec:
    label: str
    video_path: Path
    reference_timestamp_sec: float


def parse_video_spec(value: str) -> FixedCameraVideoSpec:
    try:
        label, remainder = value.split("=", 1)
        video_text, timestamp_text = remainder.rsplit("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--video must be LABEL=VIDEO_PATH=REFERENCE_TIMESTAMP_SECONDS"
        ) from exc
    if not VIEW_LABEL_PATTERN.fullmatch(label):
        raise argparse.ArgumentTypeError(
            "video LABEL must contain only letters, digits, underscores, or "
            "hyphens and must start with a letter or digit"
        )
    if not video_text:
        raise argparse.ArgumentTypeError("video VIDEO_PATH must not be empty")
    try:
        reference_timestamp_sec = float(timestamp_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"reference timestamp must be a number, got {timestamp_text!r}"
        ) from exc
    if (
        not math.isfinite(reference_timestamp_sec)
        or reference_timestamp_sec < 0.0
    ):
        raise argparse.ArgumentTypeError(
            "reference timestamp must be finite and non-negative"
        )
    return FixedCameraVideoSpec(
        label=label,
        video_path=Path(video_text).expanduser(),
        reference_timestamp_sec=reference_timestamp_sec,
    )


def resolve_executable(command: str) -> str:
    candidate = Path(command).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(
            f"FFmpeg executable {command!r} was not found. Activate the intended "
            "cluster environment or pass its path explicitly."
        )
    return resolved


def format_float(value: float) -> str:
    return format(value, ".12g")


def validate_frame_transform(rotation: str, scale: float) -> None:
    if rotation not in ROTATION_CHOICES:
        raise ValueError(
            f"rotation must be one of {ROTATION_CHOICES}, got {rotation!r}"
        )
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")


def build_video_filter(fps: float | None, rotation: str, scale: float) -> str:
    validate_frame_transform(rotation, scale)
    filters = []
    if fps is not None:
        filters.append(f"fps={format_float(fps)}")
    if rotation != "auto":
        filters.append(ROTATION_FILTERS[rotation])
    if scale != 1.0:
        scale_text = format_float(scale)
        filters.append(
            f"scale=trunc(iw*{scale_text}):trunc(ih*{scale_text}):flags=lanczos"
        )
    return ",".join(filters)


def validate_extraction_inputs(
    videos: Sequence[FixedCameraVideoSpec],
    start_sec: float,
    fps: float,
    out_dir: Path,
    rotation: str = "auto",
    scale: float = 1.0,
) -> None:
    if not videos:
        raise ValueError("At least one fixed-camera video is required")
    labels = [video.label for video in videos]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Fixed-camera video labels must be unique, got {labels}")
    if not math.isfinite(start_sec) or start_sec < 0.0:
        raise ValueError("start_sec must be finite and non-negative")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    validate_frame_transform(rotation, scale)
    for video in videos:
        if not VIEW_LABEL_PATTERN.fullmatch(video.label):
            raise ValueError(f"Invalid fixed-camera video label: {video.label!r}")
        if not video.video_path.is_file():
            raise FileNotFoundError(video.video_path)
        if video.reference_timestamp_sec < start_sec:
            raise ValueError(
                f"{video.label} reference timestamp "
                f"{video.reference_timestamp_sec:.6g} precedes extraction start "
                f"{start_sec:.6g}"
            )
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists: {out_dir}")


def _run_command(command: Sequence[str], log_path: Path) -> None:
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def extract_fixed_camera_video_frames(
    videos: Sequence[FixedCameraVideoSpec],
    start_sec: float,
    fps: float,
    out_dir: Path,
    ffmpeg_command: str = "ffmpeg",
    rotation: str = "auto",
    scale: float = 1.0,
) -> Path:
    resolved_videos = tuple(
        FixedCameraVideoSpec(
            label=video.label,
            video_path=video.video_path.expanduser().resolve(),
            reference_timestamp_sec=video.reference_timestamp_sec,
        )
        for video in videos
    )
    out_dir = out_dir.expanduser().resolve()
    validate_extraction_inputs(
        resolved_videos,
        start_sec,
        fps,
        out_dir,
        rotation,
        scale,
    )
    ffmpeg = resolve_executable(ffmpeg_command)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=str(out_dir.parent))
    )
    try:
        frame_names_dir = temp_dir / "frame_names"
        frame_names_dir.mkdir()
        command_log = temp_dir / "commands.log"
        view_records: list[dict[str, Any]] = []
        for video in resolved_videos:
            image_dir = temp_dir / "images" / video.label
            image_dir.mkdir(parents=True)
            output_pattern = image_dir / f"{video.label}_%06d.png"
            command = [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-n",
                "-ss",
                format_float(start_sec),
            ]
            if rotation != "auto":
                command.append("-noautorotate")
            command.extend(
                [
                    "-i",
                    str(video.video_path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-sn",
                    "-vf",
                    build_video_filter(fps, rotation, scale),
                    "-start_number",
                    "0",
                    str(output_pattern),
                ]
            )
            _run_command(
                command,
                command_log,
            )
            frames = sorted(image_dir.glob(f"{video.label}_*.png"))
            if not frames:
                raise RuntimeError(
                    f"FFmpeg exported no frames for {video.label} from "
                    f"{video.video_path}"
                )
            frame_names = [frame.stem for frame in frames]
            frame_names_path = frame_names_dir / f"{video.label}.json"
            _write_json(frame_names_path, frame_names)
            view_records.append(
                {
                    "label": video.label,
                    "video_path": str(video.video_path),
                    "start_sec": start_sec,
                    "fps_hz": fps,
                    "frame_count": len(frames),
                    "first_frame_name": frame_names[0],
                    "last_frame_name": frame_names[-1],
                    "reference_label": f"{video.label}_ref",
                    "reference_timestamp_sec": video.reference_timestamp_sec,
                    "reference_local_time_sec": (
                        video.reference_timestamp_sec - start_sec
                    ),
                    "image_dir": str(out_dir / "images" / video.label),
                    "frame_names_json": str(
                        out_dir / "frame_names" / f"{video.label}.json"
                    ),
                }
            )
        _write_json(
            temp_dir / "metadata.json",
            {
                "format": FRAME_DATASET_FORMAT,
                "version": FRAME_DATASET_VERSION,
                "start_sec": start_sec,
                "fps_hz": fps,
                "ffmpeg_autorotate": rotation == "auto",
                "manual_rotation": rotation,
                "scale_factor": scale,
                "cropping": False,
                "resizing": scale != 1.0,
                "views": view_records,
            },
        )
        temp_dir.replace(out_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    print(
        "Fixed-camera video frames prepared successfully:\n"
        f"  output: {out_dir}\n"
        f"  start: {start_sec:.6g} sec\n"
        f"  fps: {fps:.6g}\n"
        + "\n".join(
            f"  {record['label']}: {record['frame_count']} frames"
            for record in view_records
        ),
        flush=True,
    )
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract complete fixed-camera videos after an initial trim while "
            "optionally applying an explicit rotation and uniform scale."
        )
    )
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        type=parse_video_spec,
        help=(
            "LABEL=VIDEO_PATH=REFERENCE_TIMESTAMP_SECONDS; repeat once per "
            "fixed camera"
        ),
    )
    parser.add_argument("--start-sec", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--rotation", choices=ROTATION_CHOICES, default="auto")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg-command", default="ffmpeg")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    extract_fixed_camera_video_frames(
        args.video,
        args.start_sec,
        args.fps,
        args.out_dir,
        args.ffmpeg_command,
        args.rotation,
        args.scale,
    )


if __name__ == "__main__":
    main()
