from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata for a video file."""

    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


def probe_video(path: str) -> VideoInfo:
    """Read video metadata."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration_s = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    return VideoInfo(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_s=duration_s,
    )


def _resize_frame(frame_bgr: np.ndarray, resize: Optional[int]) -> np.ndarray:
    if resize is None:
        return frame_bgr
    if resize <= 0:
        raise ValueError("resize must be positive when provided.")

    h, w = frame_bgr.shape[:2]
    side = max(h, w)
    if side == resize:
        return frame_bgr

    scale = float(resize) / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_video_clip(
    path: str,
    t0: float = 0.0,
    t1: Optional[float] = None,
    resize: Optional[int] = None,
    grayscale: bool = True,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, float]:
    """
    Load a video segment into memory as float32 frames in [0, 1].

    Returns [T,H,W] for grayscale=True and [T,H,W,3] in BGR order otherwise.
    """
    info = probe_video(path)
    fps = info.fps if info.fps > 0 else 30.0
    if t0 < 0:
        raise ValueError("t0 must be non-negative.")
    if t1 is not None and t1 <= t0:
        raise ValueError("t1 must be greater than t0.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided.")

    start_frame = max(0, int(round(t0 * fps)))
    if t1 is None:
        end_frame = info.frame_count
    else:
        end_frame = min(info.frame_count, int(round(t1 * fps)))
    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid clip range t0={t0}, t1={t1}; video duration is {info.duration_s:.3f}s."
        )

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[np.ndarray] = []
    idx = start_frame
    while idx < end_frame:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_bgr = _resize_frame(frame_bgr, resize)
        if grayscale:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            frame = frame_bgr.astype(np.float32) / 255.0
        frames.append(frame)
        idx += 1
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if len(frames) < 1:
        raise ValueError("Clip produced no frames.")
    return np.stack(frames, axis=0), fps


def load_ordered_image_sequence(
    directory: str | Path,
    frame_names: tuple[str, ...],
    *,
    resize: Optional[int] = None,
    grayscale: bool = True,
) -> np.ndarray:
    """Load an image sequence in an explicit sidecar order."""
    source = Path(directory).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Image sequence directory does not exist: {source}")
    if not frame_names:
        raise ValueError("Image sequence frame_names must not be empty")

    indexed: dict[str, Path] = {}
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in indexed:
            raise ValueError(
                f"Image sequence repeats stem {path.stem!r}: "
                f"{indexed[path.stem]} and {path}"
            )
        indexed[path.stem] = path
    expected = set(frame_names)
    actual = set(indexed)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if extra:
            details.append(f"extra={extra[:5]}")
        raise ValueError(
            "Image sequence names do not match the frame-name sidecar: "
            + ", ".join(details)
        )

    frames: list[np.ndarray] = []
    output_shape: tuple[int, ...] | None = None
    for frame_name in frame_names:
        frame_bgr = cv2.imread(str(indexed[frame_name]), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError(f"Failed to decode image sequence frame: {indexed[frame_name]}")
        frame_bgr = _resize_frame(frame_bgr, resize)
        if grayscale:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            frame = frame_bgr.astype(np.float32) / 255.0
        if output_shape is None:
            output_shape = frame.shape
        elif frame.shape != output_shape:
            raise ValueError(
                f"Image sequence frame shape mismatch for {indexed[frame_name]}: "
                f"{frame.shape} versus {output_shape}"
            )
        frames.append(frame)
    return np.stack(frames, axis=0)

