import argparse
import json
import os
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def list_images(img_dir: Path) -> list[Path]:
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_image(path: Path) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def pick_image_dir(data_dir: Path, img_dir: str | None) -> Path:
    candidates = []
    if img_dir is not None:
        candidates.append(Path(img_dir))
    candidates.extend(
        [
            data_dir / "images",
            data_dir / "images" / "480p",
            data_dir / "JPEGImages",
            data_dir / "JPEGImages" / "480p",
        ]
    )
    for path in candidates:
        if path.exists() and list_images(path):
            return path
    raise FileNotFoundError(
        "Could not find image directory. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def pick_depth_dir(data_dir: Path, depth_dir: str | None) -> Path:
    candidates = []
    if depth_dir is not None:
        candidates.append(Path(depth_dir))
    candidates.extend(
        [
            data_dir / "aligned_depth_anything",
            data_dir / "aligned_depth_anything" / "480p",
        ]
    )
    for path in candidates:
        if path.exists() and any(path.glob("*.npy")):
            return path
    raise FileNotFoundError(
        "Could not find aligned depth directory. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def pick_megasam_path(data_dir: Path, megasam_path: str | None) -> Path:
    candidates = []
    if megasam_path is not None:
        candidates.append(Path(megasam_path))
    candidates.extend([data_dir / "megasam.npz", data_dir / f"{data_dir.name}.npz"])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find MegaSAM npz. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def to_4x4_poses(poses: np.ndarray, key: str) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3:
        raise ValueError(
            f"{key} must have shape (N, 4, 4) or (N, 3, 4), got {poses.shape}"
        )
    if poses.shape[-2:] == (4, 4):
        return poses
    if poses.shape[-2:] == (3, 4):
        bottom = np.broadcast_to(
            np.array([0, 0, 0, 1], dtype=np.float32), (poses.shape[0], 1, 4)
        )
        return np.concatenate([poses, bottom], axis=1)
    raise ValueError(
        f"{key} must have shape (N, 4, 4) or (N, 3, 4), got {poses.shape}"
    )


def load_megasam_c2w(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("cam_c2w", "c2ws", "camera_c2w", "traj_c2w"):
        if key in npz:
            return to_4x4_poses(npz[key], key)
    for key in ("cam_w2c", "w2cs", "camera_w2c", "traj_w2c"):
        if key in npz:
            return np.linalg.inv(to_4x4_poses(npz[key], key)).astype(np.float32)
    raise KeyError(f"Could not find MegaSAM poses in keys: {list(npz.keys())}")


def infer_npz_hw(npz: np.lib.npyio.NpzFile) -> tuple[int, int] | None:
    if "images" in npz:
        images = npz["images"]
        if images.ndim == 4:
            if images.shape[-1] in (1, 3, 4):
                return int(images.shape[1]), int(images.shape[2])
            return int(images.shape[2]), int(images.shape[3])
    for key in ("depths", "depth", "pred_depths", "pred_depth", "disps", "disp"):
        if key in npz:
            arr = npz[key]
            if arr.ndim >= 3:
                return int(arr.shape[-2]), int(arr.shape[-1])
    return None


def intrinsic_to_vec(arr: np.ndarray, image_hw: tuple[int, int], key: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-2:] == (3, 3):
        K = np.median(arr, axis=0)
        vec = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
    elif arr.ndim == 2 and arr.shape == (3, 3):
        vec = np.array([arr[0, 0], arr[1, 1], arr[0, 2], arr[1, 2]], dtype=np.float32)
    elif arr.ndim == 2 and arr.shape[-1] >= 4:
        vec = np.median(arr[..., :4], axis=0).astype(np.float32)
    elif arr.ndim == 1 and arr.shape[0] >= 4:
        vec = arr[:4].astype(np.float32)
    else:
        raise ValueError(f"Unsupported intrinsic shape for {key}: {arr.shape}")

    h, w = image_hw
    if np.nanmax(np.abs(vec)) <= 4.0:
        vec = vec.copy()
        vec[[0, 2]] *= w
        vec[[1, 3]] *= h
    return vec.astype(np.float32)


def load_calib_intrinsics(calib_path: str, frame_names: list[str]) -> np.ndarray:
    with open(calib_path, "r") as f:
        calib = json.load(f)

    intrinsics = []
    for name in frame_names:
        stem = Path(name).stem
        if stem not in calib:
            raise KeyError(f"Missing intrinsics for frame {stem} in {calib_path}")
        intrinsics.append(calib[stem][:4])
    return np.asarray(intrinsics, dtype=np.float32).mean(axis=0)


def load_megasam_intrinsics(
    npz: np.lib.npyio.NpzFile, source_hw: tuple[int, int], image_hw: tuple[int, int]
) -> np.ndarray:
    for key in ("intrinsic", "intrinsics", "K", "Ks", "cam_K", "cam_intrinsic"):
        if key in npz:
            vec = intrinsic_to_vec(npz[key], source_hw, key)
            sy = image_hw[0] / source_hw[0]
            sx = image_hw[1] / source_hw[1]
            vec = vec.copy()
            vec[[0, 2]] *= sx
            vec[[1, 3]] *= sy
            return vec.astype(np.float32)
    raise KeyError(f"Could not find MegaSAM intrinsics in keys: {list(npz.keys())}")


def load_megasam_depth_array(
    npz: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, bool] | None:
    for key in ("depths", "depth", "pred_depths", "pred_depth"):
        if key in npz:
            return np.asarray(npz[key], dtype=np.float32), False
    for key in ("disps", "disp", "disparities", "disparity"):
        if key in npz:
            return np.asarray(npz[key], dtype=np.float32), True
    return None


def load_aligned_depth(depth_dir: Path, image_path: Path, image_hw: tuple[int, int]) -> np.ndarray:
    disp_path = depth_dir / f"{image_path.stem}.npy"
    if not disp_path.exists():
        raise FileNotFoundError(f"Missing aligned depth: {disp_path}")
    disp = np.squeeze(np.load(disp_path).astype(np.float32))
    if disp.shape != image_hw:
        disp = cv2.resize(disp, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_LINEAR)
    return 1.0 / np.clip(disp, 1e-6, 1e6)


def megasam_depth_at(
    depth_arr: np.ndarray, index: int, is_disparity: bool, image_hw: tuple[int, int]
) -> np.ndarray:
    depth = np.squeeze(depth_arr[index]).astype(np.float32)
    if is_disparity:
        depth = 1.0 / np.clip(depth, 1e-6, 1e6)
    if depth.shape != image_hw:
        depth = cv2.resize(depth, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_LINEAR)
    return depth


def estimate_pose_scale(
    aligned_depths: list[np.ndarray],
    megasam_depths: np.ndarray,
    is_disparity: bool,
    image_hw: tuple[int, int],
) -> tuple[float, np.ndarray]:
    ratios = []
    for idx in range(min(len(aligned_depths), len(megasam_depths))):
        target = aligned_depths[idx]
        source = megasam_depth_at(megasam_depths, idx, is_disparity, image_hw)
        target_valid = target[np.isfinite(target) & (target > 1e-6)]
        source_valid = source[np.isfinite(source) & (source > 1e-6)]
        if len(target_valid) > 1024 and len(source_valid) > 1024:
            ratios.append(float(np.median(target_valid) / np.median(source_valid)))
    ratios_arr = np.asarray(ratios, dtype=np.float32)
    if len(ratios_arr) == 0:
        return 1.0, ratios_arr
    scale = float(np.median(ratios_arr))
    if not np.isfinite(scale) or scale <= 0:
        return 1.0, ratios_arr
    return scale, ratios_arr


def downsample_intrinsics(intrinsics: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return intrinsics.astype(np.float32)
    return (intrinsics / np.array([stride, stride, stride, stride], dtype=np.float32)).astype(
        np.float32
    )


def unproject_depth(depth: np.ndarray, intrinsics: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics
    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    x = (xs - cx) / fx * depth
    y = (ys - cy) / fy * depth
    cam_points = np.stack([x, y, depth], axis=-1)
    points = cam_points @ c2w[:3, :3].T + c2w[:3, 3][None, None, :]
    return points.astype(np.float32)


def sampled_view(arr: np.ndarray, stride: int, offset: int) -> np.ndarray:
    if stride <= 1:
        return arr
    offset = min(max(offset, 0), stride - 1)
    return arr[offset::stride, offset::stride]


def print_camera_stats(label: str, c2w: np.ndarray, depth_ref: float):
    centers = c2w[:, :3, 3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    radius = np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()
    print(f"{label} camera radius: {radius:.6g}")
    print(f"{label} camera path length: {steps.sum():.6g}")
    print(f"{label} step median/max: {np.median(steps):.6g} / {steps.max():.6g}")
    if depth_ref > 0:
        print(f"{label} path_length / aligned_depth_median: {steps.sum() / depth_ref:.6g}")
        print(f"{label} radius / aligned_depth_median: {radius / depth_ref:.6g}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a MegaSAM npz into the DROID-style droid_recon.npy used by SOM."
    )
    parser.add_argument("--data-dir", required=True, help="SOM training sequence directory.")
    parser.add_argument("--megasam", default=None, help="Path to MegaSAM output npz.")
    parser.add_argument("--img-dir", default=None, help="Image directory override.")
    parser.add_argument("--depth-dir", default=None, help="aligned_depth_anything directory override.")
    parser.add_argument("--calib", default=None, help="Optional SOM calibration JSON.")
    parser.add_argument("--out-path", default=None, help="Output .npy path.")
    parser.add_argument(
        "--pose-scale",
        type=float,
        default=None,
        help="Explicit multiplier for MegaSAM camera translations.",
    )
    parser.add_argument(
        "--no-auto-scale",
        action="store_true",
        help="Do not estimate pose scale from aligned depth and MegaSAM depth.",
    )
    parser.add_argument("--points-stride", type=int, default=8)
    parser.add_argument("--sample-offset", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    img_dir = pick_image_dir(data_dir, args.img_dir)
    depth_dir = pick_depth_dir(data_dir, args.depth_dir)
    megasam_path = pick_megasam_path(data_dir, args.megasam)
    out_path = Path(args.out_path) if args.out_path else data_dir / "droid_recon.npy"

    image_paths = list_images(img_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    first_image = read_image(image_paths[0])
    image_hw = first_image.shape[:2]

    with np.load(megasam_path) as megasam:
        traj_c2w = load_megasam_c2w(megasam)
        if len(traj_c2w) < len(image_paths):
            raise ValueError(
                f"Frame count mismatch: {len(image_paths)} images but {len(traj_c2w)} poses"
            )
        if len(traj_c2w) > len(image_paths):
            print(f"Trimming {len(traj_c2w)} MegaSAM poses to {len(image_paths)} frames")
            traj_c2w = traj_c2w[: len(image_paths)]

        source_hw = infer_npz_hw(megasam) or image_hw
        if args.calib:
            intrinsics_full = load_calib_intrinsics(
                args.calib, [path.name for path in image_paths]
            )
        else:
            intrinsics_full = load_megasam_intrinsics(megasam, source_hw, image_hw)

        megasam_depth_pack = load_megasam_depth_array(megasam)
        if megasam_depth_pack is None:
            megasam_depths = None
            megasam_is_disparity = False
        else:
            megasam_depths, megasam_is_disparity = megasam_depth_pack

    aligned_depths = [
        load_aligned_depth(depth_dir, image_path, image_hw) for image_path in image_paths
    ]
    aligned_medians = np.asarray(
        [
            np.median(depth[np.isfinite(depth) & (depth > 1e-6)])
            for depth in aligned_depths
        ],
        dtype=np.float32,
    )
    aligned_depth_ref = float(np.median(aligned_medians))

    if args.pose_scale is not None:
        pose_scale = args.pose_scale
        ratios = np.asarray([], dtype=np.float32)
        print(f"Using explicit pose scale: {pose_scale:.6g}")
    elif args.no_auto_scale or megasam_depths is None:
        pose_scale = 1.0
        ratios = np.asarray([], dtype=np.float32)
        print("Using pose scale: 1")
    else:
        pose_scale, ratios = estimate_pose_scale(
            aligned_depths, megasam_depths, megasam_is_disparity, image_hw
        )
        if len(ratios):
            print(
                "aligned_depth / megasam_depth ratio median/min/max: "
                f"{np.median(ratios):.6g} / {ratios.min():.6g} / {ratios.max():.6g}"
            )
        print(f"Using estimated pose scale: {pose_scale:.6g}")

    print_camera_stats("raw", traj_c2w, aligned_depth_ref)
    traj_c2w = traj_c2w.astype(np.float32).copy()
    traj_c2w[:, :3, 3] *= np.float32(pose_scale)
    print_camera_stats("scaled", traj_c2w, aligned_depth_ref)

    intrinsics_out = downsample_intrinsics(intrinsics_full, args.points_stride)
    images = []
    points = []
    masks = []
    for idx, image_path in enumerate(image_paths):
        image = read_image(image_path).astype(np.float32) / 255.0
        if image.shape[:2] != image_hw:
            image = cv2.resize(image, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_AREA)
        depth = aligned_depths[idx]
        image_small = sampled_view(image, args.points_stride, args.sample_offset)
        depth_small = sampled_view(depth, args.points_stride, args.sample_offset)
        valid = np.isfinite(depth_small) & (depth_small > 1e-6)
        images.append(image_small.astype(np.float32))
        points.append(unproject_depth(depth_small, intrinsics_out, traj_c2w[idx]))
        masks.append(valid)

    save_dict = {
        "tstamps": np.arange(len(image_paths), dtype=np.int64),
        "images": np.stack(images, axis=0).astype(np.float32),
        "points": np.stack(points, axis=0).astype(np.float32),
        "masks": np.stack(masks, axis=0),
        "map_c2w": traj_c2w.astype(np.float32),
        "traj_c2w": traj_c2w.astype(np.float32),
        "intrinsics": intrinsics_out.astype(np.float32),
        "img_shape": np.asarray(images[0].shape[:2], dtype=np.int64),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.array(save_dict, dtype=object))
    print(f"Wrote {out_path}")
    for key, value in save_dict.items():
        print(f"{key}: {value.shape if isinstance(value, np.ndarray) else type(value)}")


if __name__ == "__main__":
    main()
