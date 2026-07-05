import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from flow3d.data.utils import UINT16_MAX
from modal_surface.io import load_depth as load_modal_depth
from modal_surface.io import load_view_config


def dataset_subdir(data_dir: str, name: str, res: str) -> Path:
    path = Path(data_dir) / name
    if res:
        path = path / res
    return path


def load_raw_disparity(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        disp = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".png":
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Cannot read raw disparity: {path}")
        if raw.ndim == 3:
            raw = raw[..., 0]
        disp = raw.astype(np.float32)
        if np.issubdtype(raw.dtype, np.integer):
            disp /= float(UINT16_MAX)
    else:
        raise ValueError(f"Unsupported raw depth extension: {path.suffix}")
    if disp.ndim != 2:
        raise ValueError(f"Raw disparity must be 2D, got {disp.shape} from {path}")
    return disp


def load_fg_mask(mask_dir: Path, frame_name: str, shape: tuple[int, int]) -> np.ndarray:
    candidates = [
        mask_dir / f"{frame_name}.png",
        mask_dir / f"{frame_name}.jpg",
        mask_dir / f"{frame_name}.jpeg",
    ]
    mask_path = next((path for path in candidates if path.exists()), None)
    if mask_path is None:
        raise FileNotFoundError(
            f"Cannot find mask for {frame_name}; tried {', '.join(str(p) for p in candidates)}"
        )
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask file: {mask_path}")
    if mask.ndim == 3:
        mask = mask.max(axis=-1)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def resize_depth_to_shape(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth
    return cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def fit_median_affine(
    raw_disp: np.ndarray,
    ref_disp: np.ndarray,
    bg_mask: np.ndarray,
    min_bg_pixels: int,
    center_eps: float,
    frame_name: str,
    view_id: str,
) -> tuple[float, float, int, float]:
    valid = (
        bg_mask
        & np.isfinite(raw_disp)
        & np.isfinite(ref_disp)
        & (raw_disp > 0)
        & (ref_disp > 0)
    )
    num_bg = int(valid.sum())
    if num_bg < min_bg_pixels:
        raise ValueError(
            f"{frame_name} view_id={view_id} has only {num_bg} valid background pixels; "
            f"requires at least {min_bg_pixels}"
        )

    raw_vals = raw_disp[valid].astype(np.float64)
    ref_vals = ref_disp[valid].astype(np.float64)
    raw_centered = raw_vals - np.median(raw_vals)
    ref_centered = ref_vals - np.median(ref_vals)
    scale_valid = np.isfinite(raw_centered) & (np.abs(raw_centered) > center_eps)
    if int(scale_valid.sum()) == 0:
        raise ValueError(
            f"{frame_name} view_id={view_id} has no usable background disparity variation"
        )
    ratios = ref_centered[scale_valid] / raw_centered[scale_valid]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        raise ValueError(f"{frame_name} view_id={view_id} produced no finite scale ratios")
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= center_eps:
        raise ValueError(f"{frame_name} view_id={view_id} produced invalid scale={scale}")
    shift = float(np.median(ref_vals - scale * raw_vals))
    if not np.isfinite(shift):
        raise ValueError(f"{frame_name} view_id={view_id} produced invalid shift={shift}")
    residual = np.abs((scale * raw_vals + shift) - ref_vals)
    return scale, shift, num_bg, float(np.median(residual))


def main():
    parser = argparse.ArgumentParser(
        description="Align per-frame Depth Anything disparity to VGGT reference depth using background pixels."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--res", default="")
    parser.add_argument("--raw-depth-type", default="depth_anything_v2")
    parser.add_argument("--raw-depth-ext", choices=["png", "npy"], default="png")
    parser.add_argument("--out-depth-type", default="aligned_depth_anything_vggt_bg")
    parser.add_argument("--mask-type", default="masks")
    parser.add_argument("--modal-frame-map", required=True)
    parser.add_argument("--view-config", action="append", required=True)
    parser.add_argument("--bg-erode-iters", type=int, default=3)
    parser.add_argument("--min-bg-pixels", type=int, default=1000)
    parser.add_argument("--center-eps", type=float, default=1e-6)
    args = parser.parse_args()

    raw_dir = dataset_subdir(args.data_dir, args.raw_depth_type, args.res)
    out_dir = dataset_subdir(args.data_dir, args.out_depth_type, args.res)
    mask_dir = dataset_subdir(args.data_dir, args.mask_type, args.res)
    out_dir.mkdir(parents=True, exist_ok=True)

    view_configs = [load_view_config(path) for path in args.view_config]
    view_by_id = {cfg.view_id: cfg for cfg in view_configs}
    if len(view_by_id) != len(view_configs):
        raise ValueError("VGGT view configs must have unique view_id values")

    ref_depth_by_view = {}
    for cfg in view_configs:
        ref_depth_by_view[cfg.view_id] = load_modal_depth(
            cfg.depth_path,
            (cfg.image_height, cfg.image_width),
            cfg.depth_scale,
        ).astype(np.float32)

    with open(args.modal_frame_map, "r", encoding="utf-8") as f:
        frame_map = json.load(f)
    frames = frame_map.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{args.modal_frame_map} must contain a non-empty frames list")

    diagnostics = []
    bg_kernel = np.ones((3, 3), np.uint8)
    for record in tqdm(frames, desc="Aligning Depth Anything to VGGT"):
        frame_name = record.get("frame_name")
        view_id = record.get("view_id")
        if not frame_name:
            raise ValueError("Frame record is missing frame_name")
        if view_id not in view_by_id:
            raise ValueError(f"Frame {frame_name} uses unknown view_id={view_id!r}")

        raw_path = raw_dir / f"{frame_name}.{args.raw_depth_ext}"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw disparity file not found: {raw_path}")
        raw_disp = load_raw_disparity(raw_path)
        fg_mask = load_fg_mask(mask_dir, frame_name, raw_disp.shape)
        bg_mask = ~fg_mask
        if args.bg_erode_iters > 0:
            bg_mask = cv2.erode(
                bg_mask.astype(np.uint8),
                bg_kernel,
                iterations=args.bg_erode_iters,
            ).astype(bool)

        ref_depth = resize_depth_to_shape(ref_depth_by_view[view_id], raw_disp.shape)
        ref_disp = np.zeros_like(ref_depth, dtype=np.float32)
        valid_ref_depth = np.isfinite(ref_depth) & (ref_depth > 1e-6)
        ref_disp[valid_ref_depth] = 1.0 / ref_depth[valid_ref_depth]

        scale, shift, num_bg, residual = fit_median_affine(
            raw_disp,
            ref_disp,
            bg_mask,
            args.min_bg_pixels,
            args.center_eps,
            frame_name,
            view_id,
        )
        aligned = (scale * raw_disp + shift).astype(np.float32)
        aligned[~np.isfinite(aligned) | (aligned <= 0)] = 0.0
        np.save(out_dir / f"{frame_name}.npy", aligned)

        diagnostics.append(
            {
                "frame_name": frame_name,
                "view_id": view_id,
                "local_index": int(record.get("local_index", -1)),
                "scale": scale,
                "shift": shift,
                "num_bg_pixels": num_bg,
                "median_abs_residual": residual,
            }
        )

    csv_path = out_dir / "_alignment_diagnostics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_name",
                "view_id",
                "local_index",
                "scale",
                "shift",
                "num_bg_pixels",
                "median_abs_residual",
            ],
        )
        writer.writeheader()
        writer.writerows(diagnostics)

    json_path = out_dir / "_alignment_diagnostics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "frames": diagnostics,
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(diagnostics)} aligned disparity maps to {out_dir}")
    print(f"Wrote diagnostics to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
