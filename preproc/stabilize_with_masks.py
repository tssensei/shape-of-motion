from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def list_frame_paths(img_dir: str) -> list[Path]:
    paths = [
        Path(img_dir) / name
        for name in sorted(os.listdir(img_dir))
        if name.lower().endswith(IMAGE_EXTS)
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return paths


def find_mask_path(mask_dir: str, stem: str) -> Path:
    matches = [
        Path(mask_dir) / name
        for name in os.listdir(mask_dir)
        if Path(name).stem == stem and name.lower().endswith(IMAGE_EXTS)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely find mask for {stem} in {mask_dir}. "
            f"Found matches: {matches}"
        )
    return matches[0]


def read_rgb(path: Path) -> np.ndarray:
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    return img[..., :3]


def read_fg_mask(path: Path, image_hw: tuple[int, int]) -> np.ndarray:
    mask = iio.imread(path)
    fg = mask.reshape((*mask.shape[:2], -1)).max(axis=-1) > 0
    h, w = image_hw
    if fg.shape != (h, w):
        fg = cv2.resize(fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return fg


def erode_background(fg_mask: np.ndarray, iterations: int) -> np.ndarray:
    bg = (~fg_mask).astype(np.uint8) * 255
    if iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        bg = cv2.erode(bg, kernel, iterations=iterations)
    return bg


def make_orb(nfeatures: int):
    return cv2.ORB_create(nfeatures=nfeatures, fastThreshold=7)


def detect_features(orb, image: np.ndarray, bg_mask: np.ndarray, frame_name: str):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, desc = orb.detectAndCompute(gray, bg_mask)
    if desc is None or len(keypoints) == 0:
        raise RuntimeError(f"No background ORB features found for frame {frame_name}")
    return keypoints, desc


def match_features(cur_desc, ref_desc, ratio: float, max_matches: int):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(cur_desc, ref_desc, k=2)
    matches = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            matches.append(m)
    matches = sorted(matches, key=lambda m: m.distance)
    if max_matches > 0:
        matches = matches[:max_matches]
    return matches


def estimate_transform(
    cur_kp,
    ref_kp,
    matches,
    model: str,
    ransac_thresh: float,
    min_matches: int,
    min_inliers: int,
    frame_name: str,
):
    if len(matches) < min_matches:
        raise RuntimeError(
            f"Not enough background matches for {frame_name}: "
            f"{len(matches)} < {min_matches}"
        )

    cur_pts = np.float32([cur_kp[m.queryIdx].pt for m in matches])
    ref_pts = np.float32([ref_kp[m.trainIdx].pt for m in matches])

    if model == "homography":
        mat, inlier_mask = cv2.findHomography(
            cur_pts,
            ref_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=3000,
            confidence=0.995,
        )
        if mat is None:
            raise RuntimeError(f"Homography estimation failed for {frame_name}")
        if abs(float(mat[2, 2])) > 1e-8:
            mat = mat / mat[2, 2]
    elif model == "affine":
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            cur_pts,
            ref_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=3000,
            confidence=0.995,
        )
        if affine is None:
            raise RuntimeError(f"Affine estimation failed for {frame_name}")
        mat = np.eye(3, dtype=np.float64)
        mat[:2] = affine
    else:
        raise ValueError(f"Unknown model: {model}")

    inliers = inlier_mask.reshape(-1).astype(bool) if inlier_mask is not None else None
    num_inliers = int(inliers.sum()) if inliers is not None else len(matches)
    if num_inliers < min_inliers:
        raise RuntimeError(
            f"Not enough RANSAC inliers for {frame_name}: "
            f"{num_inliers} < {min_inliers}"
        )

    error = reprojection_error(mat, cur_pts[inliers], ref_pts[inliers])
    stats = {
        "frame": frame_name,
        "matches": len(matches),
        "inliers": num_inliers,
        "inlier_ratio": float(num_inliers / max(1, len(matches))),
        "mean_reprojection_error": float(error),
    }
    return mat.astype(np.float32), stats


def reprojection_error(mat: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    if len(src_pts) == 0:
        return float("nan")
    projected = cv2.perspectiveTransform(src_pts[:, None, :], mat)[:, 0, :]
    return float(np.linalg.norm(projected - dst_pts, axis=1).mean())


def warp_image(image: np.ndarray, mat: np.ndarray, model: str, size: tuple[int, int]):
    if model == "affine":
        return cv2.warpAffine(
            image,
            mat[:2],
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return cv2.warpPerspective(
        image,
        mat,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def warp_mask(mask: np.ndarray, mat: np.ndarray, model: str, size: tuple[int, int]):
    src = mask.astype(np.uint8) * 255
    if model == "affine":
        warped = cv2.warpAffine(
            src,
            mat[:2],
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        warped = cv2.warpPerspective(
            src,
            mat,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return warped > 0


def default_sidecar_path(out_video: str, out_img_dir: str, suffix: str) -> str:
    if out_video:
        video_path = Path(out_video)
        return str(video_path.with_name(video_path.stem + suffix))
    out_dir = Path(out_img_dir)
    return str(out_dir.parent / f"{out_dir.name}{suffix}")


def ensure_parent(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def write_json(path: str, data) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Stabilize an image sequence using only SOM background masks."
    )
    parser.add_argument("--img-dir", required=True, help="Input image sequence directory.")
    parser.add_argument("--mask-dir", required=True, help="Input SOM foreground mask directory.")
    parser.add_argument("--out-img-dir", required=True, help="Output stabilized image directory.")
    parser.add_argument("--out-mask-dir", required=True, help="Output stabilized foreground mask directory.")
    parser.add_argument("--out-video", default="", help="Optional output stabilized mp4 path.")
    parser.add_argument("--out-mask-npy", default="", help="Optional foreground ROI mask .npy path.")
    parser.add_argument("--out-transforms", default="", help="Optional transforms .npy path.")
    parser.add_argument("--out-frame-names", default="", help="Optional frame names .json path.")
    parser.add_argument("--out-stats", default="", help="Optional stats .json path.")
    parser.add_argument("--ref-name", default="", help="Reference frame stem. Defaults to middle frame.")
    parser.add_argument("--model", choices=["affine", "homography"], default="homography")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--image-ext", default="png", help="Output image extension.")
    parser.add_argument("--mask-erode-iters", type=int, default=7, help="Background mask erosion iterations.")
    parser.add_argument("--orb-features", type=int, default=8000, help="ORB feature budget.")
    parser.add_argument("--match-ratio", type=float, default=0.75, help="Lowe ratio threshold.")
    parser.add_argument("--max-matches", type=int, default=4000, help="Maximum matches passed to RANSAC.")
    parser.add_argument("--min-matches", type=int, default=80, help="Minimum feature matches per frame.")
    parser.add_argument("--min-inliers", type=int, default=40, help="Minimum RANSAC inliers per frame.")
    parser.add_argument("--ransac-thresh", type=float, default=3.0, help="RANSAC reprojection threshold.")
    parser.add_argument(
        "--mask-npy-mode",
        choices=["reference", "union"],
        default="reference",
        help="How to build --out-mask-npy from stabilized foreground masks.",
    )
    args = parser.parse_args()

    if args.mask_erode_iters < 0:
        raise ValueError("--mask-erode-iters must be non-negative")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if Path(args.img_dir).resolve() == Path(args.out_img_dir).resolve():
        raise ValueError("--out-img-dir must differ from --img-dir")
    if Path(args.mask_dir).resolve() == Path(args.out_mask_dir).resolve():
        raise ValueError("--out-mask-dir must differ from --mask-dir")

    frame_paths = list_frame_paths(args.img_dir)
    frame_names = [p.stem for p in frame_paths]
    ref_name = args.ref_name or frame_names[len(frame_names) // 2]
    if ref_name not in frame_names:
        raise ValueError(f"Reference frame {ref_name} not found in {args.img_dir}")
    image_ext = args.image_ext.lstrip(".")
    if not image_ext:
        raise ValueError("--image-ext must not be empty")

    os.makedirs(args.out_img_dir, exist_ok=True)
    os.makedirs(args.out_mask_dir, exist_ok=True)

    ref_path = frame_paths[frame_names.index(ref_name)]
    ref_img = read_rgb(ref_path)
    h, w = ref_img.shape[:2]
    size = (w, h)
    ref_fg = read_fg_mask(find_mask_path(args.mask_dir, ref_name), (h, w))
    ref_bg = erode_background(ref_fg, args.mask_erode_iters)

    orb = make_orb(args.orb_features)
    ref_kp, ref_desc = detect_features(orb, ref_img, ref_bg, ref_name)

    transforms = []
    warped_masks = []
    stats = []
    writer = None
    if args.out_video:
        out_video_dir = os.path.dirname(args.out_video)
        if out_video_dir:
            os.makedirs(out_video_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_video, fourcc, args.fps, size)
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {args.out_video}")

    try:
        for frame_path, frame_name in zip(frame_paths, frame_names):
            img = read_rgb(frame_path)
            if img.shape[:2] != (h, w):
                raise ValueError(
                    f"Image size mismatch for {frame_name}: {img.shape[:2]} != {(h, w)}"
                )
            fg = read_fg_mask(find_mask_path(args.mask_dir, frame_name), (h, w))

            if frame_name == ref_name:
                mat = np.eye(3, dtype=np.float32)
                frame_stats = {
                    "frame": frame_name,
                    "matches": 0,
                    "inliers": 0,
                    "inlier_ratio": 1.0,
                    "mean_reprojection_error": 0.0,
                }
            else:
                bg = erode_background(fg, args.mask_erode_iters)
                cur_kp, cur_desc = detect_features(orb, img, bg, frame_name)
                matches = match_features(cur_desc, ref_desc, args.match_ratio, args.max_matches)
                mat, frame_stats = estimate_transform(
                    cur_kp,
                    ref_kp,
                    matches,
                    args.model,
                    args.ransac_thresh,
                    args.min_matches,
                    args.min_inliers,
                    frame_name,
                )
                frame_stats["keypoints_current"] = len(cur_kp)
                frame_stats["keypoints_reference"] = len(ref_kp)

            warped_img = warp_image(img, mat, args.model, size)
            warped_fg = warp_mask(fg, mat, args.model, size)
            transforms.append(mat)
            warped_masks.append(warped_fg)
            stats.append(frame_stats)

            iio.imwrite(
                Path(args.out_img_dir) / f"{frame_name}.{image_ext}",
                warped_img,
            )
            iio.imwrite(
                Path(args.out_mask_dir) / f"{frame_name}.png",
                warped_fg.astype(np.uint8) * 255,
            )
            if writer is not None:
                writer.write(cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()

    transforms_path = args.out_transforms or default_sidecar_path(
        args.out_video, args.out_img_dir, "_homographies.npy"
    )
    frame_names_path = args.out_frame_names or default_sidecar_path(
        args.out_video, args.out_img_dir, "_frame_names.json"
    )
    stats_path = args.out_stats or default_sidecar_path(
        args.out_video, args.out_img_dir, "_stats.json"
    )
    ensure_parent(transforms_path)
    np.save(transforms_path, np.stack(transforms, axis=0).astype(np.float32))
    write_json(frame_names_path, frame_names)
    write_json(
        stats_path,
        {
            "reference_frame": ref_name,
            "model": args.model,
            "frame_count": len(frame_names),
            "stats": stats,
        },
    )

    if args.out_mask_npy:
        if args.mask_npy_mode == "reference":
            out_mask = warped_masks[frame_names.index(ref_name)]
        else:
            out_mask = np.stack(warped_masks, axis=0).any(axis=0)
        out_mask_dir = os.path.dirname(args.out_mask_npy)
        if out_mask_dir:
            os.makedirs(out_mask_dir, exist_ok=True)
        np.save(args.out_mask_npy, out_mask.astype(np.uint8))

    print(f"saved stabilized images -> {args.out_img_dir}")
    print(f"saved stabilized masks -> {args.out_mask_dir}")
    if args.out_video:
        print(f"saved stabilized video -> {args.out_video}")
    if args.out_mask_npy:
        print(f"saved modal ROI mask -> {args.out_mask_npy}")
    print(f"saved transforms -> {transforms_path}")
    print(f"saved stats -> {stats_path}")


if __name__ == "__main__":
    main()
