import argparse
from pathlib import Path

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    return image_paths


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def load_image(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    image = imageio.imread(path)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image at {path}, got shape {image.shape}")
    return image.astype(np.uint8)


def resize_disp_to_metric(mono_disp: np.ndarray, metric_shape: tuple[int, int]):
    import cv2

    interp = cv2.INTER_NEAREST_EXACT
    return cv2.resize(
        mono_disp,
        (metric_shape[1], metric_shape[0]),
        interpolation=interp,
    ).astype(np.float32)


def compute_megasam_alignment(
    image_paths: list[Path],
    mono_depth_dir: Path,
    metric_depth_dir: Path,
):
    mono_disp_list = []
    scales = []
    shifts = []
    fovs = []
    metric_shape = None

    for image_path in image_paths:
        mono_path = mono_depth_dir / f"{image_path.stem}.npy"
        metric_path = metric_depth_dir / f"{image_path.stem}.npz"
        require_file(mono_path)
        require_file(metric_path)

        da_disp = np.load(mono_path).astype(np.float32)
        uni_data = np.load(metric_path)
        if "depth" not in uni_data or "fov" not in uni_data:
            raise KeyError(f"{metric_path} must contain depth and fov")
        metric_depth = np.squeeze(uni_data["depth"]).astype(np.float32)
        if metric_depth.ndim != 2:
            raise ValueError(
                f"Expected 2D metric depth in {metric_path}, got {metric_depth.shape}"
            )
        if metric_shape is None:
            metric_shape = metric_depth.shape
        elif metric_depth.shape != metric_shape:
            raise ValueError(
                f"Metric depth shape changed from {metric_shape} to "
                f"{metric_depth.shape} at {metric_path}"
            )

        da_disp = resize_disp_to_metric(da_disp, metric_depth.shape)
        mono_disp_list.append(da_disp)
        fovs.append(np.asarray(uni_data["fov"]).item())

        gt_disp = 1.0 / (metric_depth + 1e-8)
        valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
        gt_disp[valid_mask] = 1e-2

        gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
        da_disp_ms = da_disp - np.median(da_disp) + 1e-8
        scale = np.median(gt_disp_ms / da_disp_ms)
        shift = np.median(gt_disp - scale * da_disp)
        if not np.isfinite(scale) or not np.isfinite(shift):
            raise ValueError(f"Invalid MegaSAM scale/shift at {image_path.name}")
        scales.append(scale)
        shifts.append(shift)

    mono_disp_stack = np.stack(mono_disp_list, axis=0)
    scales = np.asarray(scales, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.float64)
    ss_product = scales * shifts
    med_idx = int(np.argmin(np.abs(ss_product - np.median(ss_product))))
    align_scale = scales[med_idx]
    align_shift = shifts[med_idx]
    normalize_scale = (
        np.percentile(align_scale * mono_disp_stack + align_shift, 98) / 2.0
    )
    if (
        not np.isfinite(align_scale)
        or not np.isfinite(align_shift)
        or not np.isfinite(normalize_scale)
        or normalize_scale <= 0
    ):
        raise ValueError(
            "Invalid global MegaSAM alignment: "
            f"scale={align_scale}, shift={align_shift}, "
            f"normalize_scale={normalize_scale}"
        )

    denom = (1.0 / normalize_scale) * (
        align_scale * mono_disp_stack + align_shift
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        depths = np.clip(1.0 / denom, 1e-4, 1e4).astype(np.float32)
    depths[depths < 1e-2] = 0.0
    if not np.isfinite(depths).all():
        raise ValueError("High-res MegaSAM depths contain non-finite values")

    return depths, {
        "align_scale": align_scale,
        "align_shift": align_shift,
        "normalize_scale": normalize_scale,
        "median_fov": float(np.median(fovs)),
        "metric_shape": metric_shape,
    }


def load_camera_npz(camera_npz: Path, num_frames: int):
    require_file(camera_npz)
    cams = np.load(camera_npz)
    for key in ["intrinsic", "cam_c2w", "depths"]:
        if key not in cams:
            raise KeyError(f"{camera_npz} must contain {key}")

    cam_c2w = cams["cam_c2w"].astype(np.float32)
    if cam_c2w.shape[0] != num_frames:
        raise ValueError(
            f"cam_c2w has {cam_c2w.shape[0]} frames, expected {num_frames}"
        )
    camera_depths = cams["depths"]
    if camera_depths.shape[0] != num_frames:
        raise ValueError(
            f"camera depths have {camera_depths.shape[0]} frames, expected {num_frames}"
        )
    camera_depth_shape = np.squeeze(camera_depths[0]).shape[:2]
    if len(camera_depth_shape) != 2:
        raise ValueError(f"Invalid camera depth shape: {camera_depth_shape}")

    intrinsic = cams["intrinsic"].astype(np.float32).copy()
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic shape (3, 3), got {intrinsic.shape}")
    return intrinsic, cam_c2w, camera_depth_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mono-depth-dir", required=True, type=Path)
    parser.add_argument("--metric-depth-dir", required=True, type=Path)
    parser.add_argument("--camera-npz", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.camera_npz.resolve() == args.out.resolve():
        raise ValueError("--out must not overwrite --camera-npz")
    if not args.mono_depth_dir.is_dir():
        raise NotADirectoryError(args.mono_depth_dir)
    if not args.metric_depth_dir.is_dir():
        raise NotADirectoryError(args.metric_depth_dir)

    image_paths = list_images(args.image_dir)
    mono_files = sorted(args.mono_depth_dir.glob("*.npy"))
    metric_files = sorted(args.metric_depth_dir.glob("*.npz"))
    if len(mono_files) != len(image_paths):
        raise ValueError(
            f"Found {len(mono_files)} mono depth files, expected {len(image_paths)}"
        )
    if len(metric_files) != len(image_paths):
        raise ValueError(
            f"Found {len(metric_files)} metric depth files, expected {len(image_paths)}"
        )

    depths, align_info = compute_megasam_alignment(
        image_paths, args.mono_depth_dir, args.metric_depth_dir
    )
    output_depth_h, output_depth_w = depths.shape[1:3]
    intrinsic, cam_c2w, camera_depth_shape = load_camera_npz(
        args.camera_npz, len(image_paths)
    )
    camera_depth_h, camera_depth_w = camera_depth_shape
    intrinsic[0, :] *= float(output_depth_w) / float(camera_depth_w)
    intrinsic[1, :] *= float(output_depth_h) / float(camera_depth_h)

    images = np.stack([load_image(path) for path in image_paths], axis=0)
    if images.shape[0] != depths.shape[0]:
        raise ValueError(f"Image/depth count mismatch: {images.shape}, {depths.shape}")
    if images.shape[1:3] != depths.shape[1:3]:
        raise ValueError(
            "Expected high-res depths to match image resolution, got "
            f"images={images.shape[1:3]} depths={depths.shape[1:3]}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        images=images,
        depths=depths.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        cam_c2w=cam_c2w.astype(np.float32),
    )

    print(f"Wrote {args.out}")
    print(f"frames: {len(image_paths)}")
    print(f"images: {images.shape}")
    print(f"depths: {depths.shape}")
    print(f"camera depth shape: {camera_depth_shape}")
    print(f"output intrinsic:\n{intrinsic}")
    print(
        "MegaSAM alignment: "
        f"scale={align_info['align_scale']}, "
        f"shift={align_info['align_shift']}, "
        f"normalize_scale={align_info['normalize_scale']}, "
        f"median_fov={align_info['median_fov']}"
    )


if __name__ == "__main__":
    main()
