"""Prepare modal view geometry directly from joint COLMAP and static 3DGS.

This is the COLMAP-only replacement for the former VGGT view-geometry stage.
It consumes the normalized fixed-reference cameras exported by
``prepare_joint_colmap_video_dataset.py``, scales their intrinsics to the modal
video resolution, copies the supplied foreground ROI masks into the standard
modal view-config layout, and renders matching foreground depth from the final
static Gaussian checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np

from modal_surface.checkpoint_render_inputs import (
    load_fg_pixel_candidate_inputs_from_checkpoint,
)


FORMAT_NAME = "colmap_modal_view_configs"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    mask_path: Path


@dataclass(frozen=True)
class PreparedView:
    view_id: str
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    K: np.ndarray
    world_to_camera: np.ndarray
    mask: np.ndarray
    mask_source: Path
    mask_source_width: int
    mask_source_height: int
    mask_resized: bool


def parse_view_spec(value: str) -> ViewSpec:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--view must be VIEW_ID=MASK_PATH")
    view_id, mask_path = parts
    if not view_id or view_id.strip() != view_id:
        raise argparse.ArgumentTypeError(
            "VIEW_ID must be non-empty without surrounding spaces"
        )
    if "/" in view_id or "\\" in view_id:
        raise argparse.ArgumentTypeError("VIEW_ID must be a simple file prefix")
    if not mask_path:
        raise argparse.ArgumentTypeError("MASK_PATH must not be empty")
    return ViewSpec(view_id=view_id, mask_path=Path(mask_path).expanduser())


def _load_reference_cameras(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Reference-camera manifest must be a JSON object: {path}")
    if (
        payload.get("format") != "fixed_reference_cameras"
        or payload.get("version") != 1
    ):
        raise ValueError(f"Unsupported reference-camera manifest contract: {path}")
    coordinate_systems = payload.get("coordinate_systems")
    if (
        not isinstance(coordinate_systems, dict)
        or coordinate_systems.get("normalized") != "scene_norm_dict.pth world"
    ):
        raise ValueError(
            "Reference-camera manifest has no supported normalized coordinate "
            f"system: {path}"
        )
    references = payload.get("references")
    if not isinstance(references, dict) or not references:
        raise ValueError(f"Reference-camera manifest contains no references: {path}")
    if any(
        not isinstance(key, str) or not isinstance(value, dict)
        for key, value in references.items()
    ):
        raise ValueError(f"Reference-camera records are malformed: {path}")
    return references


def _finite_matrix(value: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"{label} must be a finite matrix with shape {shape}, got {matrix.shape}"
        )
    return matrix


def _load_binary_mask(
    path: Path,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    allow_resize: bool = True,
) -> tuple[np.ndarray, tuple[int, int], bool]:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read foreground ROI mask: {path}")
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    elif mask.ndim == 2:
        mask = mask > 0
    else:
        raise ValueError(f"Foreground ROI mask must be 2-D or 3-D, got {mask.shape}: {path}")
    loaded_shape = (int(mask.shape[0]), int(mask.shape[1]))
    resized = False
    if loaded_shape == target_shape:
        pass
    elif allow_resize and loaded_shape == source_shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        resized = True
    elif allow_resize:
        raise ValueError(
            f"Foreground ROI mask shape {loaded_shape} matches neither reference "
            f"source {source_shape} nor target {target_shape}: {path}"
        )
    else:
        raise ValueError(
            f"Native-resolution foreground ROI mask shape {loaded_shape} must exactly "
            f"match the registered reference shape {target_shape}: {path}"
        )
    if not bool(np.any(mask)):
        raise ValueError(f"Foreground ROI mask is empty: {path}")
    return mask.astype(np.uint8), loaded_shape, resized


def _prepare_view(
    spec: ViewSpec,
    record: dict[str, Any],
    target_width: int | None,
    native_resolution: bool = False,
) -> PreparedView:
    source_width = record.get("image_width")
    source_height = record.get("image_height")
    if (
        isinstance(source_width, bool)
        or not isinstance(source_width, int)
        or source_width <= 0
        or isinstance(source_height, bool)
        or not isinstance(source_height, int)
        or source_height <= 0
    ):
        raise ValueError(f"Reference {spec.view_id!r} has invalid source dimensions")
    if native_resolution:
        prepared_width = source_width
        target_height = source_height
    else:
        if target_width is None or target_width <= 0:
            raise ValueError("--target-width must be positive")
        prepared_width = target_width
        target_height = int(round(prepared_width * source_height / source_width))
        if target_height <= 0:
            raise ValueError(f"Reference {spec.view_id!r} produced an invalid target height")

    K = _finite_matrix(record.get("K"), (3, 3), f"{spec.view_id} K")
    world_to_camera = _finite_matrix(
        record.get("normalized_world_to_camera"),
        (4, 4),
        f"{spec.view_id} normalized_world_to_camera",
    )
    scaled_K = K.copy()
    if not native_resolution:
        scaled_K[0, :] *= prepared_width / source_width
        scaled_K[1, :] *= target_height / source_height

    mask_path = spec.mask_path.resolve(strict=True)
    mask, mask_source_shape, mask_resized = _load_binary_mask(
        mask_path,
        (source_height, source_width),
        (target_height, prepared_width),
        allow_resize=not native_resolution,
    )
    return PreparedView(
        view_id=spec.view_id,
        source_width=source_width,
        source_height=source_height,
        target_width=prepared_width,
        target_height=target_height,
        K=scaled_K,
        world_to_camera=world_to_camera,
        mask=mask,
        mask_source=mask_path,
        mask_source_width=mask_source_shape[1],
        mask_source_height=mask_source_shape[0],
        mask_resized=mask_resized,
    )


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _write_view_config(path: Path, view: PreparedView) -> None:
    _write_json(
        path,
        {
            "view_id": view.view_id,
            "image_width": view.target_width,
            "image_height": view.target_height,
            "K": view.K.astype(float).tolist(),
            "world_to_camera": view.world_to_camera.astype(float).tolist(),
            "depth_path": f"{view.view_id}_depth.npy",
            "mask_path": f"{view.view_id}_mask.npy",
            "depth_scale": 1.0,
        },
    )


def prepare_colmap_modal_view_configs(
    input_checkpoint: Path,
    reference_cameras: Path,
    views: Sequence[ViewSpec],
    target_width: int,
    out_dir: Path,
) -> Path:
    return _prepare_colmap_modal_view_configs(
        input_checkpoint=input_checkpoint,
        reference_cameras=reference_cameras,
        views=views,
        target_width=target_width,
        out_dir=out_dir,
        native_resolution=False,
    )


def prepare_colmap_modal_view_configs_native(
    input_checkpoint: Path,
    reference_cameras: Path,
    views: Sequence[ViewSpec],
    out_dir: Path,
) -> Path:
    return _prepare_colmap_modal_view_configs(
        input_checkpoint=input_checkpoint,
        reference_cameras=reference_cameras,
        views=views,
        target_width=None,
        out_dir=out_dir,
        native_resolution=True,
    )


def _prepare_colmap_modal_view_configs(
    input_checkpoint: Path,
    reference_cameras: Path,
    views: Sequence[ViewSpec],
    target_width: int | None,
    out_dir: Path,
    native_resolution: bool,
) -> Path:
    if not native_resolution and (target_width is None or target_width <= 0):
        raise ValueError("--target-width must be positive")
    if not views:
        raise ValueError("At least one --view is required")
    view_ids = [view.view_id for view in views]
    if len(set(view_ids)) != len(view_ids):
        raise ValueError(f"Duplicate view IDs are not allowed: {view_ids}")

    checkpoint_path = input_checkpoint.expanduser().resolve(strict=True)
    reference_path = reference_cameras.expanduser().resolve(strict=True)
    output_path = out_dir.expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output_path}")

    references = _load_reference_cameras(reference_path)
    missing = sorted(set(view_ids) - set(references))
    if missing:
        raise KeyError(f"Reference-camera manifest is missing requested views: {missing}")
    prepared = [
        _prepare_view(
            spec,
            references[spec.view_id],
            target_width,
            native_resolution=native_resolution,
        )
        for spec in views
    ]
    target_shapes = {(view.target_height, view.target_width) for view in prepared}
    if len(target_shapes) != 1:
        raise ValueError(
            "All modal views must share one target resolution, got "
            f"{sorted(target_shapes)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        config_paths: list[str] = []
        for view in prepared:
            np.save(temporary / f"{view.view_id}_mask.npy", view.mask)
            config_path = temporary / f"{view.view_id}_config.json"
            _write_view_config(config_path, view)
            config_paths.append(str(config_path))

        (
            _foreground_means,
            _foreground_scales,
            _foreground_quaternions,
            _foreground_opacities,
            _foreground_colors,
            rendered_depths,
            rendered_accumulations,
        ) = load_fg_pixel_candidate_inputs_from_checkpoint(
            str(checkpoint_path),
            config_paths,
        )
        if (
            len(rendered_depths) != len(prepared)
            or len(rendered_accumulations) != len(prepared)
        ):
            raise ValueError("Static checkpoint renderer returned the wrong view count")

        view_diagnostics: list[dict[str, Any]] = []
        for view, depth, accumulation in zip(
            prepared,
            rendered_depths,
            rendered_accumulations,
        ):
            expected_shape = (view.target_height, view.target_width)
            if depth.shape != expected_shape or accumulation.shape != expected_shape:
                raise ValueError(
                    f"Rendered geometry for {view.view_id} does not match {expected_shape}: "
                    f"depth={depth.shape}, accumulation={accumulation.shape}"
                )
            if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
                raise ValueError(
                    f"Rendered depth for {view.view_id} is not finite and non-negative"
                )
            if not np.all(np.isfinite(accumulation)):
                raise ValueError(
                    f"Rendered accumulation for {view.view_id} is not finite"
                )
            visible = (view.mask > 0) & (depth > 0.0) & (accumulation >= 0.05)
            visible_count = int(np.count_nonzero(visible))
            if visible_count == 0:
                raise ValueError(
                    f"Reference pose and ROI mask have no foreground-render overlap for {view.view_id}"
                )
            np.save(
                temporary / f"{view.view_id}_depth.npy",
                depth.astype(np.float32),
            )
            mask_count = int(np.count_nonzero(view.mask))
            view_diagnostics.append(
                {
                    "view_id": view.view_id,
                    "source_resolution": {
                        "width": view.source_width,
                        "height": view.source_height,
                    },
                    "target_resolution": {
                        "width": view.target_width,
                        "height": view.target_height,
                    },
                    "mask_source": str(view.mask_source),
                    "mask_source_resolution": {
                        "width": view.mask_source_width,
                        "height": view.mask_source_height,
                    },
                    "mask_resized_with_nearest": view.mask_resized,
                    "mask_pixel_count": mask_count,
                    "render_overlap_pixel_count": visible_count,
                    "render_overlap_fraction_of_mask": visible_count / mask_count,
                }
            )

        _write_json(
            temporary / "metadata.json",
            {
                "format": FORMAT_NAME,
                "version": FORMAT_VERSION,
                "source_checkpoint": str(checkpoint_path),
                "source_reference_cameras": str(reference_path),
                "coordinate_system": "scene_norm_dict.pth world",
                "depth_source": "foreground render from static 3DGS checkpoint",
                "spatial_processing": {
                    "mode": "native_resolution" if native_resolution else "target_width",
                    "intrinsics_scaled": not native_resolution,
                    "roi_resize_allowed": not native_resolution,
                },
                "view_order": view_ids,
                "views": view_diagnostics,
            },
        )
        temporary.replace(output_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare standard modal view configs from normalized joint-COLMAP "
            "reference cameras and a final static 3DGS checkpoint."
        )
    )
    parser.add_argument("--input-ckpt", required=True, type=Path)
    parser.add_argument("--reference-cameras", required=True, type=Path)
    parser.add_argument(
        "--view",
        action="append",
        required=True,
        type=parse_view_spec,
        help="VIEW_ID=MASK_PATH; repeat once per fixed-camera view",
    )
    resolution_group = parser.add_mutually_exclusive_group(required=True)
    resolution_group.add_argument("--target-width", type=int)
    resolution_group.add_argument("--native-resolution", action="store_true")
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.native_resolution:
        output_path = prepare_colmap_modal_view_configs_native(
            input_checkpoint=args.input_ckpt,
            reference_cameras=args.reference_cameras,
            views=args.view,
            out_dir=args.out_dir,
        )
    else:
        output_path = prepare_colmap_modal_view_configs(
            input_checkpoint=args.input_ckpt,
            reference_cameras=args.reference_cameras,
            views=args.view,
            target_width=args.target_width,
            out_dir=args.out_dir,
        )
    print(f"Saved COLMAP modal view configs -> {output_path}")


if __name__ == "__main__":
    main()
