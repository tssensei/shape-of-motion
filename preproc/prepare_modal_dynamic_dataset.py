from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np


DATASET_FORMAT = "modal_dynamic_dataset"
DATASET_VERSION = 1
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
VIEW_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    image_dir: Path
    mask_dir: Path
    frame_names_json: Path
    fps_hz: float


@dataclass(frozen=True)
class _ValidatedView:
    spec: ViewSpec
    frame_names: tuple[str, ...]
    image_paths: tuple[Path, ...]
    mask_paths: tuple[Path, ...]
    source_height: int
    source_width: int


def parse_view_spec(value: str) -> ViewSpec:
    parts = value.split("=")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "--view must be VIEW_ID=IMAGE_DIR=MASK_DIR=FRAME_NAMES_JSON=FPS"
        )
    view_id, image_dir, mask_dir, frame_names_json, fps_text = parts
    if not VIEW_ID_PATTERN.fullmatch(view_id):
        raise argparse.ArgumentTypeError(
            "VIEW_ID must contain only letters, digits, underscores, or hyphens "
            "and must start with a letter or digit"
        )
    for label, text in (
        ("IMAGE_DIR", image_dir),
        ("MASK_DIR", mask_dir),
        ("FRAME_NAMES_JSON", frame_names_json),
    ):
        if not text:
            raise argparse.ArgumentTypeError(f"{label} in --view must not be empty")
    try:
        fps_hz = float(fps_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("FPS in --view must be a number") from exc
    if not math.isfinite(fps_hz) or fps_hz <= 0.0:
        raise argparse.ArgumentTypeError("FPS in --view must be finite and positive")
    return ViewSpec(
        view_id=view_id,
        image_dir=Path(image_dir).expanduser(),
        mask_dir=Path(mask_dir).expanduser(),
        frame_names_json=Path(frame_names_json).expanduser(),
        fps_hz=fps_hz,
    )


def _load_frame_names(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Frame-name sidecar does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Frame-name sidecar must contain a non-empty JSON list: {path}")
    frame_names: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(payload):
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Frame-name sidecar entry {index} must be a non-empty string: {path}"
            )
        if Path(value).name != value:
            raise ValueError(
                f"Frame-name sidecar entry must be a filename stem, got {value!r}: {path}"
            )
        if value in seen:
            raise ValueError(f"Duplicate frame name {value!r} in {path}")
        seen.add(value)
        frame_names.append(value)
    return tuple(frame_names)


def _index_images(directory: Path, label: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    by_stem: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in by_stem:
            raise ValueError(
                f"{label} directory contains duplicate stem {path.stem!r}: "
                f"{by_stem[path.stem]} and {path}"
            )
        by_stem[path.stem] = path
    if not by_stem:
        raise ValueError(f"{label} directory contains no supported images: {directory}")
    return by_stem


def _read_image(path: Path, flags: int, label: str) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"Failed to decode {label}: {path}")
    return image


def _validate_view(spec: ViewSpec) -> _ValidatedView:
    frame_names = _load_frame_names(spec.frame_names_json)
    images_by_stem = _index_images(spec.image_dir, f"{spec.view_id} image")
    masks_by_stem = _index_images(spec.mask_dir, f"{spec.view_id} mask")
    expected = set(frame_names)
    for label, indexed in (("image", images_by_stem), ("mask", masks_by_stem)):
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
                f"{spec.view_id} {label} names do not match its frame-name sidecar: "
                + ", ".join(details)
            )

    image_paths = tuple(images_by_stem[name] for name in frame_names)
    mask_paths = tuple(masks_by_stem[name] for name in frame_names)
    source_height = -1
    source_width = -1
    for frame_name, image_path, mask_path in zip(frame_names, image_paths, mask_paths):
        image = _read_image(image_path, cv2.IMREAD_COLOR, "RGB image")
        mask = _read_image(mask_path, cv2.IMREAD_UNCHANGED, "mask")
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image dimensions for {image_path}: {image.shape}")
        if mask.shape[:2] != (height, width):
            raise ValueError(
                f"Image/mask dimensions differ for {spec.view_id}/{frame_name}: "
                f"image={(height, width)}, mask={mask.shape[:2]}"
            )
        if source_height < 0:
            source_height, source_width = height, width
        elif (height, width) != (source_height, source_width):
            raise ValueError(
                f"Source image dimensions vary within {spec.view_id}: "
                f"expected={(source_height, source_width)}, "
                f"got={(height, width)} for {image_path}"
            )

    return _ValidatedView(
        spec=spec,
        frame_names=frame_names,
        image_paths=image_paths,
        mask_paths=mask_paths,
        source_height=source_height,
        source_width=source_width,
    )


def _write_png(path: Path, array: np.ndarray) -> None:
    if not cv2.imwrite(str(path), array):
        raise OSError(f"Failed to write PNG: {path}")


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def validate_prepared_dataset(path: Path) -> None:
    frame_map_path = path / "modal_frame_map.json"
    metadata_path = path / "metadata.json"
    image_dir = path / "images"
    mask_dir = path / "masks"
    if not frame_map_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Prepared dataset is missing JSON metadata: {path}")
    with frame_map_path.open("r", encoding="utf-8") as f:
        frame_map = json.load(f)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if frame_map.get("version") != DATASET_VERSION:
        raise ValueError("Prepared modal frame map has an unsupported version")
    views = frame_map.get("views")
    view_fps_hz = frame_map.get("view_fps_hz")
    frames = frame_map.get("frames")
    if (
        not isinstance(views, list)
        or not views
        or any(not isinstance(view_id, str) or not view_id for view_id in views)
        or len(set(views)) != len(views)
    ):
        raise ValueError("Prepared modal frame map has invalid views")
    if not isinstance(view_fps_hz, dict) or set(view_fps_hz) != set(views):
        raise ValueError("Prepared modal frame map has invalid view_fps_hz")
    for view_id in views:
        fps_hz = view_fps_hz[view_id]
        if (
            isinstance(fps_hz, bool)
            or not isinstance(fps_hz, (int, float))
            or not math.isfinite(float(fps_hz))
            or float(fps_hz) <= 0.0
        ):
            raise ValueError(f"Prepared modal frame map has invalid FPS for {view_id}")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Prepared modal frame map has invalid frames")
    if metadata.get("version") != DATASET_VERSION or metadata.get("format") != DATASET_FORMAT:
        raise ValueError("Prepared dataset metadata has an unsupported format or version")
    if metadata.get("view_count") != len(views):
        raise ValueError("Prepared dataset metadata has an invalid view_count")
    if metadata.get("frame_count") != len(frames):
        raise ValueError("Prepared dataset metadata has an invalid frame_count")
    metadata_views = metadata.get("views")
    if not isinstance(metadata_views, list) or len(metadata_views) != len(views):
        raise ValueError("Prepared dataset metadata has invalid views")
    metadata_by_view: dict[str, dict[str, Any]] = {}
    for record in metadata_views:
        if not isinstance(record, dict):
            raise ValueError("Prepared dataset metadata view must be an object")
        view_id = record.get("view_id")
        if (
            not isinstance(view_id, str)
            or view_id not in views
            or view_id in metadata_by_view
        ):
            raise ValueError("Prepared dataset metadata has invalid view ids")
        metadata_by_view[view_id] = record
    target = metadata.get("target_resolution")
    if not isinstance(target, dict):
        raise ValueError("Prepared dataset metadata is missing target_resolution")
    target_height = int(target.get("height", 0))
    target_width = int(target.get("width", 0))
    if target_height <= 0 or target_width <= 0:
        raise ValueError("Prepared dataset target resolution is invalid")
    expected_names: list[str] = []
    view_local_indices: dict[str, list[int]] = {str(view): [] for view in views}
    for record in frames:
        if not isinstance(record, dict):
            raise ValueError("Prepared modal frame record must be an object")
        frame_name = record.get("frame_name")
        view_id = record.get("view_id")
        local_index = record.get("local_index")
        source_frame_name = record.get("source_frame_name")
        time_sec = record.get("time_sec")
        if not isinstance(frame_name, str) or not frame_name:
            raise ValueError("Prepared modal frame record has invalid frame_name")
        if not isinstance(view_id, str) or view_id not in view_local_indices:
            raise ValueError(f"Prepared modal frame record has unknown view_id={view_id!r}")
        if not isinstance(local_index, int) or local_index < 0:
            raise ValueError(f"Prepared modal frame {frame_name} has invalid local_index")
        if not isinstance(source_frame_name, str) or not source_frame_name:
            raise ValueError(f"Prepared modal frame {frame_name} has invalid source_frame_name")
        if not isinstance(time_sec, (int, float)) or not math.isfinite(float(time_sec)):
            raise ValueError(f"Prepared modal frame {frame_name} has invalid time_sec")
        expected_time_sec = local_index / float(view_fps_hz[view_id])
        if not math.isclose(
            float(time_sec), expected_time_sec, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"Prepared modal frame {frame_name} has inconsistent time_sec")
        expected_names.append(frame_name)
        view_local_indices[view_id].append(local_index)
    if len(set(expected_names)) != len(expected_names):
        raise ValueError("Prepared modal frame map contains duplicate frame_name values")
    for view_id, local_indices in view_local_indices.items():
        if local_indices != list(range(len(local_indices))):
            raise ValueError(f"Prepared modal frame indices are not contiguous for {view_id}")
        metadata_view = metadata_by_view[view_id]
        if metadata_view.get("frame_count") != len(local_indices):
            raise ValueError(f"Prepared dataset metadata has invalid frame_count for {view_id}")
        metadata_fps = metadata_view.get("fps_hz")
        if (
            isinstance(metadata_fps, bool)
            or not isinstance(metadata_fps, (int, float))
            or float(metadata_fps) != float(view_fps_hz[view_id])
        ):
            raise ValueError(f"Prepared dataset metadata has inconsistent FPS for {view_id}")
    expected_files = {f"{name}.png" for name in expected_names}
    for label, directory in (("image", image_dir), ("mask", mask_dir)):
        actual_files = {item.name for item in directory.iterdir() if item.is_file()}
        if actual_files != expected_files:
            raise ValueError(f"Prepared {label} files do not match the frame map")
    for name in expected_names:
        image = _read_image(image_dir / f"{name}.png", cv2.IMREAD_COLOR, "prepared RGB image")
        mask = _read_image(mask_dir / f"{name}.png", cv2.IMREAD_UNCHANGED, "prepared mask")
        if image.shape[:2] != (target_height, target_width):
            raise ValueError(f"Prepared image has an invalid shape: {name}")
        if mask.shape != (target_height, target_width):
            raise ValueError(f"Prepared mask has an invalid shape: {name}")
        values = set(np.unique(mask).tolist())
        if not values.issubset({0, 255}):
            raise ValueError(f"Prepared mask is not binary: {name}")


def prepare_modal_dynamic_dataset(
    views: Sequence[ViewSpec],
    target_width: int,
    out_dir: Path,
) -> Path:
    return _prepare_modal_dynamic_dataset(
        views=views,
        target_width=target_width,
        out_dir=out_dir,
        native_resolution=False,
    )


def prepare_modal_dynamic_dataset_native(
    views: Sequence[ViewSpec],
    out_dir: Path,
) -> Path:
    return _prepare_modal_dynamic_dataset(
        views=views,
        target_width=None,
        out_dir=out_dir,
        native_resolution=True,
    )


def _prepare_modal_dynamic_dataset(
    views: Sequence[ViewSpec],
    target_width: int | None,
    out_dir: Path,
    native_resolution: bool,
) -> Path:
    if not views:
        raise ValueError("At least one --view is required")
    if not native_resolution and (target_width is None or target_width <= 0):
        raise ValueError("target_width must be positive")
    for view in views:
        if not VIEW_ID_PATTERN.fullmatch(view.view_id):
            raise ValueError(f"Invalid view_id: {view.view_id!r}")
        if not math.isfinite(view.fps_hz) or view.fps_hz <= 0.0:
            raise ValueError(f"FPS must be finite and positive for {view.view_id}")
    view_ids = [view.view_id for view in views]
    if len(set(view_ids)) != len(view_ids):
        raise ValueError("View IDs must be unique")
    out_dir = out_dir.expanduser()
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    validated_views = [_validate_view(view) for view in views]
    reference = validated_views[0]
    for view in validated_views[1:]:
        if native_resolution and (
            view.source_height != reference.source_height
            or view.source_width != reference.source_width
        ):
            raise ValueError(
                "All views must have the same native resolution: "
                f"{reference.spec.view_id}="
                f"{reference.source_width}x{reference.source_height}, "
                f"{view.spec.view_id}={view.source_width}x{view.source_height}"
            )
        if not native_resolution and (
            view.source_height * reference.source_width
            != reference.source_height * view.source_width
        ):
            raise ValueError(
                "All views must have the same aspect ratio: "
                f"{reference.spec.view_id}="
                f"{reference.source_width}x{reference.source_height}, "
                f"{view.spec.view_id}={view.source_width}x{view.source_height}"
            )
    if native_resolution:
        output_width = reference.source_width
        output_height = reference.source_height
    else:
        assert target_width is not None
        output_width = target_width
        output_height = int(
            round(output_width * reference.source_height / reference.source_width)
        )
        if output_height <= 0:
            raise ValueError("target_width produces a zero-height output")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=str(out_dir.parent))
    )
    try:
        image_out_dir = temp_dir / "images"
        mask_out_dir = temp_dir / "masks"
        image_out_dir.mkdir()
        mask_out_dir.mkdir()
        frame_records: list[dict[str, Any]] = []
        metadata_views: list[dict[str, Any]] = []
        for view in validated_views:
            for local_index, (source_name, image_path, mask_path) in enumerate(
                zip(view.frame_names, view.image_paths, view.mask_paths)
            ):
                frame_name = f"{view.spec.view_id}_{local_index:06d}"
                mask = _read_image(mask_path, cv2.IMREAD_UNCHANGED, "mask")
                if mask.ndim == 2:
                    binary_mask = mask > 0
                else:
                    binary_mask = np.any(mask > 0, axis=2)
                if native_resolution:
                    if image_path.suffix.lower() != ".png":
                        raise ValueError(
                            "Native-resolution RGB inputs must be PNG files so their "
                            f"pixels can be copied without re-encoding: {image_path}"
                        )
                    image = _read_image(
                        image_path,
                        cv2.IMREAD_UNCHANGED,
                        "native-resolution RGB image",
                    )
                    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                        raise ValueError(
                            "Native-resolution RGB inputs must be three-channel uint8 "
                            f"PNG files, got dtype={image.dtype}, shape={image.shape}: "
                            f"{image_path}"
                        )
                    shutil.copy2(image_path, image_out_dir / f"{frame_name}.png")
                    _write_png(
                        mask_out_dir / f"{frame_name}.png",
                        binary_mask.astype(np.uint8) * 255,
                    )
                else:
                    image = _read_image(image_path, cv2.IMREAD_COLOR, "RGB image")
                    resized_image = cv2.resize(
                        image,
                        (output_width, output_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    resized_mask = cv2.resize(
                        binary_mask.astype(np.uint8),
                        (output_width, output_height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    _write_png(image_out_dir / f"{frame_name}.png", resized_image)
                    _write_png(mask_out_dir / f"{frame_name}.png", resized_mask * 255)
                frame_records.append(
                    {
                        "frame_name": frame_name,
                        "view_id": view.spec.view_id,
                        "local_index": local_index,
                        "time_sec": local_index / view.spec.fps_hz,
                        "source_frame_name": source_name,
                    }
                )
            metadata_views.append(
                {
                    "view_id": view.spec.view_id,
                    "image_dir": str(view.spec.image_dir),
                    "mask_dir": str(view.spec.mask_dir),
                    "frame_names_json": str(view.spec.frame_names_json),
                    "fps_hz": view.spec.fps_hz,
                    "frame_count": len(view.frame_names),
                    "source_resolution": {
                        "width": view.source_width,
                        "height": view.source_height,
                    },
                }
            )

        _write_json(
            temp_dir / "modal_frame_map.json",
            {
                "version": DATASET_VERSION,
                "views": view_ids,
                "view_fps_hz": {
                    view.spec.view_id: view.spec.fps_hz for view in validated_views
                },
                "frames": frame_records,
            },
        )
        _write_json(
            temp_dir / "metadata.json",
            {
                "version": DATASET_VERSION,
                "format": DATASET_FORMAT,
                "view_count": len(validated_views),
                "frame_count": len(frame_records),
                "target_resolution": {
                    "width": output_width,
                    "height": output_height,
                },
                "spatial_processing": {
                    "mode": "native_resolution" if native_resolution else "target_width",
                    "image_resize": "none" if native_resolution else "area",
                    "mask_resize": "none" if native_resolution else "nearest",
                },
                "views": metadata_views,
            },
        )
        validate_prepared_dataset(temp_dir)
        temp_dir.replace(out_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic multi-view RGB/mask dataset for modal training."
    )
    parser.add_argument(
        "--view",
        action="append",
        required=True,
        type=parse_view_spec,
        help="VIEW_ID=IMAGE_DIR=MASK_DIR=FRAME_NAMES_JSON=FPS; repeat per view.",
    )
    resolution_group = parser.add_mutually_exclusive_group(required=True)
    resolution_group.add_argument("--target-width", type=int)
    resolution_group.add_argument("--native-resolution", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.native_resolution:
        output = prepare_modal_dynamic_dataset_native(args.view, args.out_dir)
    else:
        output = prepare_modal_dynamic_dataset(args.view, args.target_width, args.out_dir)
    print(f"Saved modal dynamic dataset -> {output}")


if __name__ == "__main__":
    main()
