from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

import cv2
import numpy as np


MAX_IMAGE_ID = 2_147_483_647
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def read_mask(path: Path, image_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        mask = mask.reshape((*mask.shape[:2], -1)).max(axis=-1)
    mask = mask > 0
    if mask.shape != image_hw:
        mask = cv2.resize(mask.astype(np.uint8), (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def load_modal_shape(modal_npz: Path) -> tuple[int, int]:
    z = np.load(str(modal_npz), allow_pickle=False)
    if "mode_u" not in z:
        raise KeyError(f"{modal_npz} missing mode_u")
    mode_u = z["mode_u"]
    if mode_u.ndim != 3:
        raise ValueError(f"Expected mode_u shape (K,H,W), got {mode_u.shape}")
    return int(mode_u.shape[1]), int(mode_u.shape[2])


def load_modal_mask_or_resize(modal_npz: Path, fallback_mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    z = np.load(str(modal_npz), allow_pickle=False)
    if "has_mask" in z and bool(np.asarray(z["has_mask"]).item()) and "mask" in z:
        mask = z["mask"]
        if mask.shape == target_hw and mask.size > 0:
            return mask.astype(bool)
    return cv2.resize(
        fallback_mask.astype(np.uint8),
        (target_hw[1], target_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    ) > 0


def load_unidepth_intrinsics(path: Path, view_name: str) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    stem = Path(view_name).stem
    if stem not in raw:
        raise KeyError(f"{stem} not found in {path}; found {sorted(raw.keys())}")
    vals = np.asarray(raw[stem][:4], dtype=np.float64)
    if vals.shape != (4,):
        raise ValueError(f"Expected four intrinsics for {stem}, got {raw[stem]}")
    fx, fy, cx, cy = vals.tolist()
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def scale_K(K: np.ndarray, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    sx = float(target_w) / float(source_w)
    sy = float(target_h) / float(source_h)
    out = K.astype(np.float64).copy()
    out[0, :] *= sx
    out[1, :] *= sy
    return out


def load_depth_from_unidepth_disp(path: Path, source_hw: tuple[int, int]) -> np.ndarray:
    disp = np.load(str(path)).astype(np.float32)
    disp = np.squeeze(disp)
    if disp.ndim != 2:
        raise ValueError(f"Expected 2D disparity at {path}, got {disp.shape}")
    if disp.shape != source_hw:
        disp = cv2.resize(disp, (source_hw[1], source_hw[0]), interpolation=cv2.INTER_LINEAR)
    depth = 1.0 / np.clip(disp, 1e-6, 1e6)
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def resize_depth(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if depth.shape == target_hw:
        return depth.astype(np.float32, copy=False)
    return cv2.resize(depth.astype(np.float32), (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)


def load_database_images(con: sqlite3.Connection) -> dict[str, int]:
    rows = con.execute("SELECT image_id, name FROM images").fetchall()
    return {str(name): int(image_id) for image_id, name in rows}


def load_keypoints(con: sqlite3.Connection, image_id: int) -> np.ndarray:
    row = con.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
    if row is None:
        raise KeyError(f"No keypoints found for image_id={image_id}")
    rows, cols, blob = int(row[0]), int(row[1]), row[2]
    if rows <= 0 or cols < 2:
        raise ValueError(f"Invalid keypoints shape for image_id={image_id}: rows={rows}, cols={cols}")
    arr = np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)
    return arr[:, :2].astype(np.float32)


def load_verified_match_indices(
    con: sqlite3.Connection,
    image_id1: int,
    image_id2: int,
) -> np.ndarray:
    pair_id = image_ids_to_pair_id(image_id1, image_id2)
    row = con.execute("SELECT rows, cols, data FROM two_view_geometries WHERE pair_id=?", (pair_id,)).fetchone()
    if row is None:
        raise KeyError(f"No two_view_geometries row found for pair_id={pair_id}")
    rows, cols, blob = int(row[0]), int(row[1]), row[2]
    if rows <= 0 or cols != 2 or blob is None:
        raise ValueError(f"Invalid verified matches for pair_id={pair_id}: rows={rows}, cols={cols}")
    matches = np.frombuffer(blob, dtype=np.uint32).reshape(rows, cols).astype(np.int64)
    if image_id1 > image_id2:
        matches = matches[:, ::-1]
    return matches


def normalize_pixels(pixels_xy: np.ndarray, K: np.ndarray) -> np.ndarray:
    x = (pixels_xy[:, 0].astype(np.float64) - K[0, 2]) / K[0, 0]
    y = (pixels_xy[:, 1].astype(np.float64) - K[1, 2]) / K[1, 1]
    return np.stack([x, y], axis=1)


def recover_pose_from_matches(
    pts1_xy: np.ndarray,
    pts2_xy: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    ransac_thresh_px: float,
    ransac_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(pts1_xy) < 8:
        raise ValueError(f"Need at least 8 verified matches, got {len(pts1_xy)}")
    n1 = normalize_pixels(pts1_xy, K1)
    n2 = normalize_pixels(pts2_xy, K2)
    mean_focal = float(np.mean([K1[0, 0], K1[1, 1], K2[0, 0], K2[1, 1]]))
    threshold = float(ransac_thresh_px) / max(mean_focal, 1e-6)

    E, mask = cv2.findEssentialMat(
        n1,
        n2,
        np.eye(3, dtype=np.float64),
        method=cv2.RANSAC,
        prob=float(ransac_prob),
        threshold=threshold,
    )
    if E is None:
        raise RuntimeError("cv2.findEssentialMat failed")
    mask = np.ones((len(n1), 1), dtype=np.uint8) if mask is None else mask.astype(np.uint8)

    E = np.asarray(E, dtype=np.float64)
    candidates = [E] if E.shape == (3, 3) else [E[i : i + 3] for i in range(0, E.shape[0], 3)]
    best = None
    for E_i in candidates:
        if E_i.shape != (3, 3):
            continue
        retval, R, t, pose_mask = cv2.recoverPose(E_i, n1, n2, np.eye(3, dtype=np.float64), mask=mask.copy())
        score = int(retval)
        if best is None or score > best[0]:
            best = (score, R, t.reshape(3), pose_mask)
    if best is None or best[0] < 8:
        raise RuntimeError("cv2.recoverPose failed to recover a usable two-view pose")

    _, R, t, pose_mask = best
    inliers = pose_mask.reshape(-1) > 0
    return R.astype(np.float64), t.astype(np.float64), inliers, E.astype(np.float64)


def bilinear_sample(image: np.ndarray, pixels_xy: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = pixels_xy[:, 0].astype(np.float64)
    y = pixels_xy[:, 1].astype(np.float64)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    out = np.full(len(pixels_xy), np.nan, dtype=np.float64)
    if not np.any(valid):
        return out
    wx = x[valid] - x0[valid]
    wy = y[valid] - y0[valid]
    v00 = image[y0[valid], x0[valid]]
    v01 = image[y0[valid], x1[valid]]
    v10 = image[y1[valid], x0[valid]]
    v11 = image[y1[valid], x1[valid]]
    out[valid] = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v01
        + (1.0 - wx) * wy * v10
        + wx * wy * v11
    )
    return out


def unproject_pixels(pixels_xy: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    x = (pixels_xy[:, 0].astype(np.float64) - K[0, 2]) * depth / K[0, 0]
    y = (pixels_xy[:, 1].astype(np.float64) - K[1, 2]) * depth / K[1, 1]
    return np.stack([x, y, depth], axis=1)


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return np.stack([u, v], axis=1), z


def estimate_translation_scale(
    pts1_xy: np.ndarray,
    pts2_xy: np.ndarray,
    inliers: np.ndarray,
    depth1: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    grid_size: int,
) -> tuple[float, dict, np.ndarray]:
    pts1_i = pts1_xy[inliers]
    pts2_i = pts2_xy[inliers]
    d1 = bilinear_sample(depth1, pts1_i)
    valid = np.isfinite(d1) & (d1 > 0)
    if valid.sum() < 8:
        raise RuntimeError(f"Not enough pose inliers with valid view1 depth: {int(valid.sum())}")

    pts1_i = pts1_i[valid]
    pts2_i = pts2_i[valid]
    d1 = d1[valid]
    X1 = unproject_pixels(pts1_i, d1, K1)
    RX1 = (R @ X1.T).T
    ray2 = np.concatenate([normalize_pixels(pts2_i, K2), np.ones((len(pts2_i), 1), dtype=np.float64)], axis=1)

    A = np.cross(ray2, t.reshape(1, 3))
    b = -np.cross(ray2, RX1)
    denom = np.sum(A * A, axis=1)
    per_match_scale = np.sum(A * b, axis=1) / np.maximum(denom, 1e-12)
    positive = per_match_scale[np.isfinite(per_match_scale) & (per_match_scale > 1e-6)]
    if positive.size == 0:
        initial = 1.0
    else:
        initial = float(np.median(positive))

    lo = max(initial / 10.0, 1e-5)
    hi = max(initial * 10.0, lo * 1.01)
    candidates = np.geomspace(lo, hi, max(int(grid_size), 3))
    best_scale = initial
    best_median = np.inf
    best_errors = None
    for scale in candidates:
        X2 = RX1 + float(scale) * t.reshape(1, 3)
        positive_z = X2[:, 2] > 1e-6
        if positive_z.sum() < 8:
            continue
        proj, _ = project_points(X2[positive_z], K2)
        err = np.linalg.norm(proj - pts2_i[positive_z], axis=1)
        med = float(np.median(err))
        if med < best_median:
            best_median = med
            best_scale = float(scale)
            best_errors = err

    if best_errors is None:
        best_errors = np.full(0, np.nan, dtype=np.float64)
    stats = {
        "depth_valid_inliers": int(valid.sum()),
        "initial_scale_median": float(initial),
        "scale": float(best_scale),
        "median_reprojection_error_px": float(best_median),
        "mean_reprojection_error_px": float(np.mean(best_errors)) if best_errors.size else None,
        "per_match_positive_scale_count": int(positive.size),
    }
    return float(best_scale), stats, np.where(inliers)[0][valid]


def estimate_view2_depth_alignment(
    pts1_xy: np.ndarray,
    pts2_xy: np.ndarray,
    depth1: np.ndarray,
    depth2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    R: np.ndarray,
    t_scaled: np.ndarray,
    selected_indices: np.ndarray,
) -> tuple[float, dict]:
    if selected_indices.size == 0:
        return 1.0, {"view2_depth_scale": 1.0, "valid_pairs": 0}
    p1 = pts1_xy[selected_indices]
    p2 = pts2_xy[selected_indices]
    d1 = bilinear_sample(depth1, p1)
    d2 = bilinear_sample(depth2, p2)
    valid = np.isfinite(d1) & np.isfinite(d2) & (d1 > 0) & (d2 > 0)
    if valid.sum() < 8:
        return 1.0, {"view2_depth_scale": 1.0, "valid_pairs": int(valid.sum())}
    X1 = unproject_pixels(p1[valid], d1[valid], K1)
    X2 = (R @ X1.T).T + t_scaled.reshape(1, 3)
    z2 = X2[:, 2]
    valid_z = np.isfinite(z2) & (z2 > 0)
    ratios = z2[valid_z] / d2[valid][valid_z]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size < 8:
        return 1.0, {"view2_depth_scale": 1.0, "valid_pairs": int(ratios.size)}
    scale = float(np.median(ratios))
    return scale, {
        "view2_depth_scale": scale,
        "valid_pairs": int(ratios.size),
        "ratio_p10": float(np.percentile(ratios, 10)),
        "ratio_p90": float(np.percentile(ratios, 90)),
    }


def save_view_config(
    path: Path,
    view_id: str,
    image_width: int,
    image_height: int,
    K: np.ndarray,
    world_to_camera: np.ndarray,
    depth_name: str,
    mask_name: str,
) -> None:
    payload = {
        "view_id": view_id,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "K": K.astype(float).tolist(),
        "world_to_camera": world_to_camera.astype(float).tolist(),
        "depth_path": depth_name,
        "mask_path": mask_name,
        "depth_scale": 1.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export two-view pose/depth configs for modal_surface from COLMAP verified matches."
    )
    parser.add_argument("--database", required=True, type=Path, help="COLMAP database.db path.")
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory with view1/view2 reference images.")
    parser.add_argument("--mask-dir", required=True, type=Path, help="Directory with view1/view2 foreground masks.")
    parser.add_argument("--unidepth-disp-dir", required=True, type=Path, help="UniDepth disparity .npy directory.")
    parser.add_argument("--unidepth-intrins", required=True, type=Path, help="UniDepth intrinsics JSON.")
    parser.add_argument("--view1-name", required=True, help="View 1 image filename in COLMAP database.")
    parser.add_argument("--view2-name", required=True, help="View 2 image filename in COLMAP database.")
    parser.add_argument("--view1-modal-npz", required=True, type=Path, help="View 1 modal_analysis npz.")
    parser.add_argument("--view2-modal-npz", required=True, type=Path, help="View 2 modal_analysis npz.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for modal_surface geometry files.")
    parser.add_argument("--ransac-thresh-px", type=float, default=2.0, help="Essential matrix RANSAC threshold in pixels.")
    parser.add_argument("--ransac-prob", type=float, default=0.999, help="Essential matrix RANSAC confidence.")
    parser.add_argument("--scale-grid-size", type=int, default=81, help="Grid size for translation scale refinement.")
    parser.add_argument("--disable-view2-depth-alignment", action="store_true", help="Do not scale view2 depth to the recovered pose scale.")
    args = parser.parse_args()

    if args.ransac_thresh_px <= 0:
        raise ValueError("--ransac-thresh-px must be positive")
    if not args.database.exists():
        raise FileNotFoundError(args.database)

    image1 = read_image(args.image_dir / args.view1_name)
    image2 = read_image(args.image_dir / args.view2_name)
    if image1.shape[:2] != image2.shape[:2]:
        raise ValueError(f"Image shapes differ: {image1.shape[:2]} vs {image2.shape[:2]}")
    source_hw = image1.shape[:2]

    target_hw1 = load_modal_shape(args.view1_modal_npz)
    target_hw2 = load_modal_shape(args.view2_modal_npz)
    if target_hw1 != target_hw2:
        raise ValueError(f"Modal shapes differ: {target_hw1} vs {target_hw2}")
    target_hw = target_hw1
    target_h, target_w = target_hw

    with sqlite3.connect(str(args.database)) as con:
        images = load_database_images(con)
        if args.view1_name not in images or args.view2_name not in images:
            raise KeyError(f"Database images are {sorted(images.keys())}, missing requested views")
        image_id1 = images[args.view1_name]
        image_id2 = images[args.view2_name]
        keypoints1 = load_keypoints(con, image_id1)
        keypoints2 = load_keypoints(con, image_id2)
        match_indices = load_verified_match_indices(con, image_id1, image_id2)

    pts1 = keypoints1[match_indices[:, 0]]
    pts2 = keypoints2[match_indices[:, 1]]
    K1_source = load_unidepth_intrinsics(args.unidepth_intrins, args.view1_name)
    K2_source = load_unidepth_intrinsics(args.unidepth_intrins, args.view2_name)
    R, t_unit, pose_inliers, E = recover_pose_from_matches(
        pts1,
        pts2,
        K1_source,
        K2_source,
        args.ransac_thresh_px,
        args.ransac_prob,
    )

    depth1_source = load_depth_from_unidepth_disp(
        args.unidepth_disp_dir / f"{Path(args.view1_name).stem}.npy",
        source_hw,
    )
    depth2_source = load_depth_from_unidepth_disp(
        args.unidepth_disp_dir / f"{Path(args.view2_name).stem}.npy",
        source_hw,
    )
    scale, scale_stats, scale_match_indices = estimate_translation_scale(
        pts1,
        pts2,
        pose_inliers,
        depth1_source,
        K1_source,
        K2_source,
        R,
        t_unit,
        args.scale_grid_size,
    )
    t_scaled = scale * t_unit

    view2_depth_scale = 1.0
    view2_depth_stats = {"view2_depth_scale": 1.0, "valid_pairs": 0}
    if not args.disable_view2_depth_alignment:
        view2_depth_scale, view2_depth_stats = estimate_view2_depth_alignment(
            pts1,
            pts2,
            depth1_source,
            depth2_source,
            K1_source,
            K2_source,
            R,
            t_scaled,
            scale_match_indices,
        )
        depth2_source = depth2_source * view2_depth_scale

    K1_target = scale_K(K1_source, source_hw, target_hw)
    K2_target = scale_K(K2_source, source_hw, target_hw)
    depth1_target = resize_depth(depth1_source, target_hw)
    depth2_target = resize_depth(depth2_source, target_hw)

    mask1_source = read_mask(args.mask_dir / f"{Path(args.view1_name).stem}.png", source_hw)
    mask2_source = read_mask(args.mask_dir / f"{Path(args.view2_name).stem}.png", source_hw)
    mask1_target = load_modal_mask_or_resize(args.view1_modal_npz, mask1_source, target_hw)
    mask2_target = load_modal_mask_or_resize(args.view2_modal_npz, mask2_source, target_hw)

    w2c1 = np.eye(4, dtype=np.float64)
    w2c2 = np.eye(4, dtype=np.float64)
    w2c2[:3, :3] = R
    w2c2[:3, 3] = t_scaled

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "view1_depth.npy", depth1_target.astype(np.float32))
    np.save(args.out_dir / "view2_depth.npy", depth2_target.astype(np.float32))
    np.save(args.out_dir / "view1_mask.npy", mask1_target.astype(np.uint8))
    np.save(args.out_dir / "view2_mask.npy", mask2_target.astype(np.uint8))
    save_view_config(
        args.out_dir / "view1_config.json",
        "view1",
        target_w,
        target_h,
        K1_target,
        w2c1,
        "view1_depth.npy",
        "view1_mask.npy",
    )
    save_view_config(
        args.out_dir / "view2_config.json",
        "view2",
        target_w,
        target_h,
        K2_target,
        w2c2,
        "view2_depth.npy",
        "view2_mask.npy",
    )

    np.savez_compressed(
        args.out_dir / "verified_matches_debug.npz",
        pts1_xy=pts1.astype(np.float32),
        pts2_xy=pts2.astype(np.float32),
        pose_inliers=pose_inliers.astype(np.uint8),
        essential_matrix=E.astype(np.float64),
        R=R.astype(np.float64),
        t_unit=t_unit.astype(np.float64),
        t_scaled=t_scaled.astype(np.float64),
        K1_source=K1_source.astype(np.float64),
        K2_source=K2_source.astype(np.float64),
        K1_target=K1_target.astype(np.float64),
        K2_target=K2_target.astype(np.float64),
        source_hw=np.asarray(source_hw, dtype=np.int32),
        target_hw=np.asarray(target_hw, dtype=np.int32),
    )
    stats = {
        "view1_name": args.view1_name,
        "view2_name": args.view2_name,
        "image_id1": int(image_id1),
        "image_id2": int(image_id2),
        "source_height": int(source_hw[0]),
        "source_width": int(source_hw[1]),
        "target_height": int(target_h),
        "target_width": int(target_w),
        "verified_matches": int(len(match_indices)),
        "pose_inliers": int(pose_inliers.sum()),
        "pose_inlier_fraction": float(pose_inliers.mean()),
        "translation_unit": t_unit.astype(float).tolist(),
        "translation_scaled": t_scaled.astype(float).tolist(),
        "rotation_view1_to_view2": R.astype(float).tolist(),
        "scale_estimation": scale_stats,
        "view2_depth_alignment": view2_depth_stats,
        "outputs": {
            "view1_config": "view1_config.json",
            "view2_config": "view2_config.json",
            "view1_depth": "view1_depth.npy",
            "view2_depth": "view2_depth.npy",
            "view1_mask": "view1_mask.npy",
            "view2_mask": "view2_mask.npy",
            "verified_matches_debug": "verified_matches_debug.npz",
        },
    }
    with (args.out_dir / "two_view_pose_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"verified matches: {len(match_indices)}")
    print(f"pose inliers: {int(pose_inliers.sum())}/{len(pose_inliers)}")
    print(f"translation scale: {scale:.6g}")
    print(f"view2 depth alignment scale: {view2_depth_scale:.6g}")
    print(f"saved modal_surface geometry -> {args.out_dir}")


if __name__ == "__main__":
    main()
