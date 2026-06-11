import argparse
import json
import os

import cv2
import imageio.v2 as iio
import numpy as np
from tqdm import tqdm


def list_images(img_dir: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg")
    return sorted(f for f in os.listdir(img_dir) if f.lower().endswith(exts))


def read_image(path: str) -> np.ndarray:
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def load_calib_intrinsics(calib_path: str, frame_names: list[str]) -> np.ndarray:
    with open(calib_path, "r") as f:
        calib = json.load(f)

    values = []
    for name in frame_names:
        stem = os.path.splitext(name)[0]
        if stem not in calib:
            raise KeyError(f"Missing intrinsics for frame {stem} in {calib_path}")
        values.append(calib[stem][:4])
    return np.asarray(values, dtype=np.float32).mean(axis=0)


def to_4x4_poses(poses: np.ndarray, key: str) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3:
        raise ValueError(f"{key} must have shape (N, 4, 4) or (N, 3, 4), got {poses.shape}")
    if poses.shape[-2:] == (4, 4):
        return poses
    if poses.shape[-2:] == (3, 4):
        bottom = np.broadcast_to(
            np.array([0, 0, 0, 1], dtype=np.float32), (poses.shape[0], 1, 4)
        )
        return np.concatenate([poses, bottom], axis=1)
    raise ValueError(f"{key} must have shape (N, 4, 4) or (N, 3, 4), got {poses.shape}")


def load_megasam_c2w(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    c2w_keys = ("cam_c2w", "c2ws", "camera_c2w", "traj_c2w")
    w2c_keys = ("cam_w2c", "w2cs", "camera_w2c", "traj_w2c")

    for key in c2w_keys:
        if key in npz:
            return to_4x4_poses(npz[key], key)
    for key in w2c_keys:
        if key in npz:
            return np.linalg.inv(to_4x4_poses(npz[key], key)).astype(np.float32)

    raise KeyError(
        "Could not find MegaSaM camera poses. "
        f"Expected one of {c2w_keys + w2c_keys}; found {list(npz.keys())}"
    )


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
        vec[0] *= w
        vec[2] *= w
        vec[1] *= h
        vec[3] *= h
    return vec.astype(np.float32)


def load_megasam_intrinsics(
    npz: np.lib.npyio.NpzFile, image_hw: tuple[int, int]
) -> np.ndarray | None:
    for key in ("intrinsic", "intrinsics", "K", "Ks", "cam_K", "cam_intrinsic"):
        if key in npz:
            return intrinsic_to_vec(npz[key], image_hw, key)
    return None


def load_megasam_depths(npz: np.lib.npyio.NpzFile) -> tuple[np.ndarray, bool] | None:
    for key in ("depths", "depth", "pred_depths", "pred_depth"):
        if key in npz:
            return np.asarray(npz[key], dtype=np.float32), False
    for key in ("disps", "disp", "disparities", "disparity"):
        if key in npz:
            return np.asarray(npz[key], dtype=np.float32), True
    return None


def resize_hw(arr: np.ndarray, out_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    out_h, out_w = out_hw
    return cv2.resize(arr, (out_w, out_h), interpolation=interpolation)


def load_depth_from_disp(depth_dir: str, frame_name: str, image_hw: tuple[int, int]) -> np.ndarray:
    stem = os.path.splitext(frame_name)[0]
    path = os.path.join(depth_dir, stem + ".npy")
    disp = np.load(path).astype(np.float32)
    disp = np.squeeze(disp)
    if disp.shape != image_hw:
        disp = resize_hw(disp, image_hw, cv2.INTER_LINEAR)
    return 1.0 / np.clip(disp, 1e-6, 1e6)


def resize_megasam_depth(
    depth_arr: np.ndarray,
    index: int,
    is_disparity: bool,
    image_hw: tuple[int, int],
) -> np.ndarray:
    depth = np.squeeze(depth_arr[index]).astype(np.float32)
    if is_disparity:
        depth = 1.0 / np.clip(depth, 1e-6, 1e6)
    if depth.shape != image_hw:
        depth = resize_hw(depth, image_hw, cv2.INTER_LINEAR)
    return depth


def estimate_pose_scale(
    aligned_depths: list[np.ndarray],
    megasam_depths: np.ndarray,
    is_disparity: bool,
    image_hw: tuple[int, int],
) -> float:
    ratios = []
    for idx, target_depth in enumerate(aligned_depths):
        source_depth = resize_megasam_depth(megasam_depths, idx, is_disparity, image_hw)
        valid = (
            np.isfinite(source_depth)
            & np.isfinite(target_depth)
            & (source_depth > 1e-6)
            & (target_depth > 1e-6)
        )
        if valid.sum() > 1024:
            ratios.append(np.median(target_depth[valid] / source_depth[valid]))

    if not ratios:
        return 1.0
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    return scale


def scaled_intrinsics(intrinsics: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    out = intrinsics.astype(np.float32).copy()
    out[0] *= scale_x
    out[2] *= scale_x
    out[1] *= scale_y
    out[3] *= scale_y
    return out


def unproject_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
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
    world_points = (
        cam_points @ c2w[:3, :3].T + c2w[:3, 3][None, None, :]
    ).astype(np.float32)
    return world_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--megasam", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument(
        "--intrinsics-source",
        choices=("auto", "megasam", "calib"),
        default="auto",
    )
    parser.add_argument("--point-long-edge", type=int, default=320)
    parser.add_argument("--pose-scale", type=float, default=1.0)
    parser.add_argument("--no-scale-align", action="store_true")
    args = parser.parse_args()

    img_files = list_images(args.img_dir)
    if not img_files:
        raise ValueError(f"No images found in {args.img_dir}")

    first_img = read_image(os.path.join(args.img_dir, img_files[0]))
    full_h, full_w = first_img.shape[:2]
    full_hw = (full_h, full_w)

    megasam = np.load(args.megasam)
    print(f"MegaSaM keys: {list(megasam.keys())}")

    traj_c2w = load_megasam_c2w(megasam)
    if len(traj_c2w) != len(img_files):
        raise ValueError(
            f"Frame count mismatch: {len(img_files)} images but {len(traj_c2w)} poses"
        )

    calib_intrinsics = load_calib_intrinsics(args.calib, img_files)
    megasam_intrinsics = load_megasam_intrinsics(megasam, full_hw)
    if args.intrinsics_source == "megasam":
        if megasam_intrinsics is None:
            raise KeyError("Requested MegaSaM intrinsics, but none were found in npz")
        full_intrinsics = megasam_intrinsics
    elif args.intrinsics_source == "calib":
        full_intrinsics = calib_intrinsics
    else:
        full_intrinsics = (
            megasam_intrinsics if megasam_intrinsics is not None else calib_intrinsics
        )

    if args.point_long_edge > 0:
        point_scale = min(1.0, args.point_long_edge / max(full_h, full_w))
    else:
        point_scale = 1.0
    out_h = max(1, int(round(full_h * point_scale)))
    out_w = max(1, int(round(full_w * point_scale)))
    out_hw = (out_h, out_w)
    out_intrinsics = scaled_intrinsics(
        full_intrinsics, out_w / full_w, out_h / full_h
    )

    full_depths = [
        load_depth_from_disp(args.depth_dir, frame_name, full_hw)
        for frame_name in tqdm(img_files, desc="loading aligned depths")
    ]

    pose_scale = args.pose_scale
    megasam_depth_info = load_megasam_depths(megasam)
    if not args.no_scale_align and megasam_depth_info is not None:
        depth_arr, is_disparity = megasam_depth_info
        if len(depth_arr) == len(img_files):
            pose_scale *= estimate_pose_scale(
                full_depths, depth_arr, is_disparity, full_hw
            )
        else:
            print(
                "Skipping MegaSaM depth scale alignment because depth count "
                f"{len(depth_arr)} != image count {len(img_files)}"
            )
    traj_c2w = traj_c2w.copy()
    traj_c2w[:, :3, 3] *= pose_scale

    images = []
    points = []
    masks = []
    for idx, frame_name in enumerate(tqdm(img_files, desc="building recon")):
        img = read_image(os.path.join(args.img_dir, frame_name))
        img_small = resize_hw(img, out_hw, cv2.INTER_AREA)
        depth_small = resize_hw(full_depths[idx], out_hw, cv2.INTER_LINEAR)
        valid = np.isfinite(depth_small) & (depth_small > 1e-6)

        images.append(img_small.astype(np.float32) / 255.0)
        masks.append(valid)
        points.append(unproject_depth(depth_small, out_intrinsics, traj_c2w[idx]))

    save_dict = {
        "tstamps": np.arange(len(img_files), dtype=np.int64),
        "images": np.stack(images, axis=0).astype(np.float32),
        "points": np.stack(points, axis=0).astype(np.float32),
        "masks": np.stack(masks, axis=0),
        "map_c2w": traj_c2w.astype(np.float32),
        "traj_c2w": traj_c2w.astype(np.float32),
        "intrinsics": out_intrinsics.astype(np.float32),
        "img_shape": np.array(out_hw, dtype=np.int64),
    }

    out_dir = os.path.dirname(args.out_path.rstrip("/"))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out_path, np.array(save_dict))

    print(f"pose_scale={pose_scale}")
    for key, value in save_dict.items():
        print(f"{key} {value.shape if isinstance(value, np.ndarray) else value}")


if __name__ == "__main__":
    main()
