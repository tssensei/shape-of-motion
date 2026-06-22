from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
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


def build_mask_map(mask_dir: str) -> dict[str, Path]:
    mask_map: dict[str, Path] = {}
    for name in sorted(os.listdir(mask_dir)):
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        path = Path(mask_dir) / name
        stem = path.stem
        if stem in mask_map:
            raise FileExistsError(
                f"Duplicate mask stem {stem} in {mask_dir}: {mask_map[stem]} and {path}"
            )
        mask_map[stem] = path
    if not mask_map:
        raise FileNotFoundError(f"No masks found in {mask_dir}")
    return mask_map


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def read_fg_mask(path: Path, image_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        fg = mask.reshape((*mask.shape[:2], -1)).max(axis=-1) > 0
    else:
        fg = mask > 0
    h, w = image_hw
    if fg.shape != (h, w):
        fg = cv2.resize(fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return fg


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, data) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool, copy=False)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations) > 0


def detect_reference_features(
    ref_gray: np.ndarray,
    exclude_mask: np.ndarray,
    max_corners: int,
    quality: float,
    min_distance: int,
    block_size: int,
    border_margin: int,
) -> np.ndarray:
    detect_mask = (~exclude_mask).astype(np.uint8) * 255
    if border_margin > 0:
        m = int(border_margin)
        detect_mask[:m, :] = 0
        detect_mask[-m:, :] = 0
        detect_mask[:, :m] = 0
        detect_mask[:, -m:] = 0

    pts = cv2.goodFeaturesToTrack(
        ref_gray,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=block_size,
        useHarrisDetector=False,
        mask=detect_mask,
    )
    if pts is None or len(pts) < 4:
        raise RuntimeError(
            "Could not detect enough background features in the reference frame. "
            "Try reducing --mask-dilate-iters, --feature-quality, or --feature-min-distance."
        )
    return pts.astype(np.float32)


def points_inside_image(points_xy: np.ndarray, height: int, width: int) -> np.ndarray:
    return (
        np.isfinite(points_xy).all(axis=1)
        & (points_xy[:, 0] >= 0.0)
        & (points_xy[:, 0] <= width - 1.0)
        & (points_xy[:, 1] >= 0.0)
        & (points_xy[:, 1] <= height - 1.0)
    )


def points_on_background(points_xy: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    h, w = fg_mask.shape
    inside = points_inside_image(points_xy, h, w)
    keep = np.zeros(len(points_xy), dtype=bool)
    if not np.any(inside):
        return keep
    xi = np.rint(points_xy[inside, 0]).astype(np.int32)
    yi = np.rint(points_xy[inside, 1]).astype(np.int32)
    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)
    keep_inside = ~fg_mask[yi, xi]
    keep[np.flatnonzero(inside)] = keep_inside
    return keep


def track_reference_points(
    ref_gray: np.ndarray,
    gray: np.ndarray,
    ref_pts: np.ndarray,
    current_exclude_mask: np.ndarray,
    lk_params: dict,
    fb_thresh: float,
    reject_current_foreground: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(ref_pts)
    cur_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(ref_gray, gray, ref_pts, None, **lk_params)
    if cur_pts is None or st_fwd is None:
        return np.zeros((n, 2), dtype=np.float32), np.zeros(n, dtype=bool), np.full(n, np.inf)

    back_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(gray, ref_gray, cur_pts, None, **lk_params)
    if back_pts is None or st_bwd is None:
        return cur_pts.reshape(-1, 2), np.zeros(n, dtype=bool), np.full(n, np.inf)

    ref_xy = ref_pts.reshape(-1, 2)
    cur_xy = cur_pts.reshape(-1, 2)
    back_xy = back_pts.reshape(-1, 2)
    fb_err = np.linalg.norm(ref_xy - back_xy, axis=1)
    good = (
        (st_fwd.reshape(-1) == 1)
        & (st_bwd.reshape(-1) == 1)
        & np.isfinite(fb_err)
        & (fb_err <= fb_thresh)
        & points_inside_image(cur_xy, gray.shape[0], gray.shape[1])
    )
    if reject_current_foreground:
        good &= points_on_background(cur_xy, current_exclude_mask)
    return cur_xy.astype(np.float32), good, fb_err.astype(np.float32)


def estimate_homography_to_ref(
    ref_xy: np.ndarray,
    cur_xy: np.ndarray,
    good: np.ndarray,
    min_matches: int,
    min_inliers: int,
    ransac_thresh: float,
) -> tuple[np.ndarray | None, dict]:
    src_cur = cur_xy[good]
    dst_ref = ref_xy[good]
    stats = {
        "tracked": int(good.sum()),
        "inliers": 0,
        "valid": False,
        "reason": "",
    }
    if len(src_cur) < min_matches:
        stats["reason"] = f"tracked<{min_matches}"
        return None, stats

    H, inlier_mask = cv2.findHomography(
        src_cur,
        dst_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=3000,
        confidence=0.995,
    )
    if H is None or not np.isfinite(H).all() or abs(float(H[2, 2])) < 1e-8:
        stats["reason"] = "findHomography_failed"
        return None, stats

    inliers = inlier_mask.reshape(-1).astype(bool) if inlier_mask is not None else np.ones(len(src_cur), dtype=bool)
    stats["inliers"] = int(inliers.sum())
    if stats["inliers"] < min_inliers:
        stats["reason"] = f"inliers<{min_inliers}"
        return None, stats

    H = H / H[2, 2]
    reproj = cv2.perspectiveTransform(src_cur[inliers, None, :], H)[:, 0, :]
    err = np.linalg.norm(reproj - dst_ref[inliers], axis=1)
    stats["valid"] = True
    stats["reason"] = "ok"
    stats["mean_reprojection_error"] = float(err.mean()) if len(err) else 0.0
    stats["median_reprojection_error"] = float(np.median(err)) if len(err) else 0.0
    return H.astype(np.float32), stats


def fill_missing_homographies(
    H_raw: list[np.ndarray | None],
    max_fallback_fraction: float,
) -> tuple[list[np.ndarray], list[dict]]:
    valid_indices = [i for i, H in enumerate(H_raw) if H is not None]
    if not valid_indices:
        raise RuntimeError("No valid homographies were estimated.")

    fallback_count = len(H_raw) - len(valid_indices)
    fallback_fraction = fallback_count / max(1, len(H_raw))
    if fallback_fraction > max_fallback_fraction:
        raise RuntimeError(
            f"Too many failed homographies: {fallback_count}/{len(H_raw)} "
            f"({fallback_fraction:.1%}) > --max-fallback-fraction {max_fallback_fraction:.1%}."
        )

    filled = []
    fill_stats = []
    for i, H in enumerate(H_raw):
        if H is not None:
            filled.append(H)
            fill_stats.append({"frame_index": i, "filled": False, "source_index": i})
            continue
        nearest = min(valid_indices, key=lambda j: abs(j - i))
        filled.append(H_raw[nearest].copy())
        fill_stats.append({"frame_index": i, "filled": True, "source_index": nearest})
    return filled, fill_stats


def smooth_homographies(H_list: list[np.ndarray], radius: int) -> list[np.ndarray]:
    if len(H_list) == 0 or radius <= 0:
        return H_list

    params = []
    for H in H_list:
        H = H.astype(np.float64)
        if abs(float(H[2, 2])) < 1e-8 or not np.isfinite(H).all():
            H = np.eye(3, dtype=np.float64)
        else:
            H = H / H[2, 2]
        params.append(
            [
                H[0, 0],
                H[0, 1],
                H[0, 2],
                H[1, 0],
                H[1, 1],
                H[1, 2],
                H[2, 0],
                H[2, 1],
            ]
        )

    params_arr = np.array(params, dtype=np.float64)
    out = params_arr.copy()
    n = len(params_arr)
    radius = int(radius)
    sigma = max(radius / 3.0, 1e-6)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)

    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        k_lo = lo - (i - radius)
        k_hi = k_lo + (hi - lo)
        weights = kernel[k_lo:k_hi]
        weights = weights / weights.sum()
        out[i] = (params_arr[lo:hi] * weights[:, None]).sum(axis=0)

    return [
        np.array(
            [
                [p[0], p[1], p[2]],
                [p[3], p[4], p[5]],
                [p[6], p[7], 1.0],
            ],
            dtype=np.float32,
        )
        for p in out
    ]


def warp_image(image: np.ndarray, H: np.ndarray, size: tuple[int, int], border_mode: str) -> np.ndarray:
    mode = cv2.BORDER_REPLICATE if border_mode == "replicate" else cv2.BORDER_CONSTANT
    return cv2.warpPerspective(
        image,
        H,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=mode,
        borderValue=0,
    )


def warp_mask(mask: np.ndarray, H: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    warped = cv2.warpPerspective(
        mask.astype(np.uint8) * 255,
        H,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 127


def crop_and_resize(image: np.ndarray, crop_ratio: float) -> np.ndarray:
    if crop_ratio <= 0:
        return image
    h, w = image.shape[:2]
    dx = int(round(w * crop_ratio))
    dy = int(round(h * crop_ratio))
    if w - 2 * dx < 20 or h - 2 * dy < 20:
        raise ValueError("--crop-ratio is too large for the input image size")
    return cv2.resize(image[dy : h - dy, dx : w - dx], (w, h), interpolation=cv2.INTER_LINEAR)


def crop_mask_and_resize(mask: np.ndarray, crop_ratio: float) -> np.ndarray:
    if crop_ratio <= 0:
        return mask
    h, w = mask.shape
    dx = int(round(w * crop_ratio))
    dy = int(round(h * crop_ratio))
    if w - 2 * dx < 20 or h - 2 * dy < 20:
        raise ValueError("--crop-ratio is too large for the input mask size")
    resized = cv2.resize(
        mask[dy : h - dy, dx : w - dx].astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def default_sidecar_path(out_video: str, out_img_dir: str, suffix: str) -> Path:
    if out_video:
        video_path = Path(out_video)
        return video_path.with_name(video_path.stem + suffix)
    out_dir = Path(out_img_dir)
    return out_dir.parent / f"{out_dir.name}{suffix}"


def default_work_dir(out_video: str, out_img_dir: str) -> Path:
    if out_video:
        video_path = Path(out_video)
        return video_path.with_name(video_path.stem + "_work")
    out_dir = Path(out_img_dir)
    return out_dir.parent / f"{out_dir.name}_ref_anchor_work"


def run_command(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def write_h264_video(
    ffmpeg_bin: str,
    video_frame_dir: Path,
    out_video: str,
    fps: float,
    crf: int,
    preset: str,
) -> None:
    ensure_parent(out_video)
    run_command(
        [
            ffmpeg_bin,
            "-nostdin",
            "-y",
            "-hide_banner",
            "-framerate",
            f"{fps:g}",
            "-start_number",
            "1",
            "-i",
            str(video_frame_dir / "%08d.png"),
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-movflags",
            "+faststart",
            out_video,
        ]
    )


def choose_reference(frame_names: list[str], ref_name: str, ref_index: int | None) -> tuple[str, int]:
    if ref_name:
        if ref_name not in frame_names:
            raise ValueError(f"Reference frame {ref_name} not found in input frames")
        return ref_name, frame_names.index(ref_name)
    if ref_index is not None:
        if ref_index < 0 or ref_index >= len(frame_names):
            raise ValueError(f"--ref-index {ref_index} is outside [0, {len(frame_names) - 1}]")
        return frame_names[ref_index], ref_index
    idx = len(frame_names) // 2
    return frame_names[idx], idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reference-anchored frame stabilization using SOM masks, Shi-Tomasi "
            "background points, and LK forward-backward tracking."
        )
    )
    parser.add_argument("--img-dir", required=True, help="Input image sequence directory.")
    parser.add_argument("--mask-dir", required=True, help="Input SOM foreground mask directory.")
    parser.add_argument("--out-img-dir", required=True, help="Output stabilized image directory.")
    parser.add_argument("--out-mask-dir", required=True, help="Output stabilized foreground mask directory.")
    parser.add_argument("--out-video", default="", help="Optional output H.264 mp4 path.")
    parser.add_argument("--out-mask-npy", default="", help="Optional foreground ROI mask .npy path.")
    parser.add_argument("--out-homographies", default="", help="Optional smoothed homographies .npy path.")
    parser.add_argument("--out-frame-names", default="", help="Optional frame names .json path.")
    parser.add_argument("--out-stats", default="", help="Optional stabilization stats .json path.")
    parser.add_argument("--work-dir", default="", help="Temporary directory for mp4 frame assembly.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable for --out-video.")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--ref-name", default="", help="Reference frame stem. Defaults to middle frame.")
    parser.add_argument("--ref-index", type=int, default=None, help="Reference frame index. Ignored if --ref-name is set.")
    parser.add_argument("--image-ext", default="png", help="Output image extension.")
    parser.add_argument("--mask-dilate-iters", type=int, default=7, help="Dilate foreground masks before background tracking.")
    parser.add_argument("--border-margin", type=int, default=12, help="Do not detect reference features near image borders.")
    parser.add_argument("--max-corners", type=int, default=1000, help="Maximum Shi-Tomasi reference points.")
    parser.add_argument("--feature-quality", type=float, default=0.01, help="Shi-Tomasi qualityLevel.")
    parser.add_argument("--feature-min-distance", type=int, default=10, help="Shi-Tomasi minDistance.")
    parser.add_argument("--feature-block-size", type=int, default=7, help="Shi-Tomasi blockSize.")
    parser.add_argument("--lk-window", type=int, default=21, help="LK optical-flow window size.")
    parser.add_argument("--lk-levels", type=int, default=3, help="LK pyramid levels.")
    parser.add_argument("--lk-max-iter", type=int, default=60, help="LK max iterations.")
    parser.add_argument("--lk-eps", type=float, default=0.005, help="LK epsilon.")
    parser.add_argument("--pass1-fb-thresh", type=float, default=1.0, help="Forward-backward threshold for stable-point screening.")
    parser.add_argument("--pass2-fb-thresh", type=float, default=2.0, help="Forward-backward threshold for homography estimation.")
    parser.add_argument("--min-track-success", type=float, default=0.80, help="Minimum per-point success rate in pass 1.")
    parser.add_argument("--max-avg-fb-err", type=float, default=1.5, help="Maximum per-point average FB error in pass 1.")
    parser.add_argument("--min-stable-points", type=int, default=25, help="Fail if fewer stable points survive pass 1.")
    parser.add_argument("--min-matches", type=int, default=12, help="Minimum tracked points before RANSAC.")
    parser.add_argument("--min-inliers", type=int, default=10, help="Minimum RANSAC inliers per frame.")
    parser.add_argument("--ransac-thresh", type=float, default=3.0, help="RANSAC reprojection threshold in pixels.")
    parser.add_argument("--smooth-radius", type=int, default=10, help="Temporal smoothing radius for anchored homographies.")
    parser.add_argument("--max-fallback-fraction", type=float, default=0.20, help="Abort if more frames need nearest-valid transform fill.")
    parser.add_argument("--disable-current-mask-check", action="store_true", help="Do not reject tracked points landing in current foreground masks.")
    parser.add_argument("--border-mode", choices=["replicate", "black"], default="replicate", help="Warp border mode.")
    parser.add_argument("--crop-ratio", type=float, default=0.0, help="Optional crop-and-resize ratio after stabilization.")
    parser.add_argument(
        "--mask-npy-mode",
        choices=["reference", "union"],
        default="reference",
        help="How to build --out-mask-npy from stabilized foreground masks.",
    )
    parser.add_argument("--mask-npy-ref-name", default="", help="Reference frame stem for --mask-npy-mode reference.")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF for --out-video.")
    parser.add_argument("--preset", default="medium", help="H.264 preset for --out-video.")
    parser.add_argument("--log-every", type=int, default=100, help="Progress print interval in frames.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep temporary mp4 frames.")
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.mask_dilate_iters < 0:
        raise ValueError("--mask-dilate-iters must be non-negative")
    if args.smooth_radius < 0:
        raise ValueError("--smooth-radius must be non-negative")
    if not (0.0 < args.min_track_success <= 1.0):
        raise ValueError("--min-track-success must be in (0, 1]")
    if not (0.0 <= args.max_fallback_fraction <= 1.0):
        raise ValueError("--max-fallback-fraction must be in [0, 1]")
    if args.crop_ratio < 0 or args.crop_ratio >= 0.45:
        raise ValueError("--crop-ratio must be in [0, 0.45)")
    if Path(args.img_dir).resolve() == Path(args.out_img_dir).resolve():
        raise ValueError("--out-img-dir must differ from --img-dir")
    if Path(args.mask_dir).resolve() == Path(args.out_mask_dir).resolve():
        raise ValueError("--out-mask-dir must differ from --mask-dir")

    frame_paths = list_frame_paths(args.img_dir)
    frame_names = [path.stem for path in frame_paths]
    mask_map = build_mask_map(args.mask_dir)
    missing_masks = [name for name in frame_names if name not in mask_map]
    if missing_masks:
        raise FileNotFoundError(f"No mask found for frame {missing_masks[0]}")

    image_ext = args.image_ext.lstrip(".")
    if not image_ext:
        raise ValueError("--image-ext must not be empty")

    ref_name, ref_index = choose_reference(frame_names, args.ref_name, args.ref_index)
    mask_npy_ref_name = args.mask_npy_ref_name or ref_name
    if mask_npy_ref_name not in frame_names:
        raise ValueError(f"Mask npy reference frame {mask_npy_ref_name} not found")

    first = read_bgr(frame_paths[0])
    image_hw = first.shape[:2]
    for path in frame_paths[1:]:
        image = read_bgr(path)
        if image.shape[:2] != image_hw:
            raise ValueError(f"Image size mismatch for {path}: {image.shape[:2]} != {image_hw}")
    h, w = image_hw
    size = (w, h)

    ref_img = read_bgr(frame_paths[ref_index])
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_fg = read_fg_mask(mask_map[ref_name], image_hw)
    ref_exclude = dilate_mask(ref_fg, args.mask_dilate_iters)
    ref_pts_all = detect_reference_features(
        ref_gray,
        ref_exclude,
        args.max_corners,
        args.feature_quality,
        args.feature_min_distance,
        args.feature_block_size,
        args.border_margin,
    )
    print(f"[reference] frame={ref_name} index={ref_index}/{len(frame_names)-1}")
    print(f"[reference] detected background points: {len(ref_pts_all)}")

    lk_params = {
        "winSize": (args.lk_window, args.lk_window),
        "maxLevel": args.lk_levels,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            args.lk_max_iter,
            args.lk_eps,
        ),
    }

    success_count = np.zeros(len(ref_pts_all), dtype=np.int32)
    fb_err_sum = np.zeros(len(ref_pts_all), dtype=np.float64)
    reject_current_fg = not args.disable_current_mask_check

    print("[pass1] tracking all reference points to screen stable background points")
    for i, (frame_path, frame_name) in enumerate(zip(frame_paths, frame_names)):
        if i == ref_index:
            success_count += 1
            continue
        image = read_bgr(frame_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_fg = read_fg_mask(mask_map[frame_name], image_hw)
        current_exclude = dilate_mask(current_fg, args.mask_dilate_iters)
        _, good, fb_err = track_reference_points(
            ref_gray,
            gray,
            ref_pts_all,
            current_exclude,
            lk_params,
            args.pass1_fb_thresh,
            reject_current_fg,
        )
        success_count += good.astype(np.int32)
        fb_err_sum += np.where(good, fb_err, 0.0)
        if args.log_every > 0 and ((i + 1) % args.log_every == 0 or i + 1 == len(frame_paths)):
            print(f"  [pass1] {i+1}/{len(frame_paths)}")

    success_rate = success_count / len(frame_paths)
    avg_fb_err = np.divide(
        fb_err_sum,
        success_count,
        out=np.full(len(ref_pts_all), np.inf),
        where=success_count > 0,
    )
    stable = (success_rate >= args.min_track_success) & (avg_fb_err <= args.max_avg_fb_err)
    if int(stable.sum()) < args.min_stable_points:
        raise RuntimeError(
            f"Only {int(stable.sum())} stable points survived pass 1. "
            f"Need at least {args.min_stable_points}. Try lowering --min-track-success, "
            "--max-avg-fb-err, or --mask-dilate-iters."
        )
    ref_pts = ref_pts_all[stable]
    ref_xy = ref_pts.reshape(-1, 2)
    print(
        f"[pass1] stable points: {len(ref_pts_all)} -> {len(ref_pts)} "
        f"(min success {success_rate[stable].min():.1%}, max avg fb {avg_fb_err[stable].max():.3f}px)"
    )

    print("[pass2] estimating frame-to-reference homographies")
    H_raw: list[np.ndarray | None] = []
    pass2_stats = []
    for i, (frame_path, frame_name) in enumerate(zip(frame_paths, frame_names)):
        if i == ref_index:
            H_raw.append(np.eye(3, dtype=np.float32))
            pass2_stats.append(
                {
                    "frame": frame_name,
                    "frame_index": i,
                    "tracked": len(ref_pts),
                    "inliers": len(ref_pts),
                    "valid": True,
                    "reason": "reference",
                    "mean_reprojection_error": 0.0,
                    "median_reprojection_error": 0.0,
                }
            )
            continue

        image = read_bgr(frame_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_fg = read_fg_mask(mask_map[frame_name], image_hw)
        current_exclude = dilate_mask(current_fg, args.mask_dilate_iters)
        cur_xy, good, fb_err = track_reference_points(
            ref_gray,
            gray,
            ref_pts,
            current_exclude,
            lk_params,
            args.pass2_fb_thresh,
            reject_current_fg,
        )
        H, stats = estimate_homography_to_ref(
            ref_xy,
            cur_xy,
            good,
            args.min_matches,
            args.min_inliers,
            args.ransac_thresh,
        )
        stats.update(
            {
                "frame": frame_name,
                "frame_index": i,
                "mean_fb_error": float(np.mean(fb_err[good])) if np.any(good) else None,
                "median_fb_error": float(np.median(fb_err[good])) if np.any(good) else None,
            }
        )
        H_raw.append(H)
        pass2_stats.append(stats)
        if args.log_every > 0 and ((i + 1) % args.log_every == 0 or i + 1 == len(frame_paths)):
            valid_so_far = sum(1 for H_i in H_raw if H_i is not None)
            print(f"  [pass2] {i+1}/{len(frame_paths)} valid={valid_so_far}/{len(H_raw)}")

    H_filled, fill_stats = fill_missing_homographies(H_raw, args.max_fallback_fraction)
    H_smooth = smooth_homographies(H_filled, args.smooth_radius)

    out_img_dir = Path(args.out_img_dir)
    out_mask_dir = Path(args.out_mask_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(args.work_dir) if args.work_dir else default_work_dir(args.out_video, args.out_img_dir)
    video_frame_dir = None
    if args.out_video:
        video_frame_dir = work_dir / "_ref_anchor_video_frames"
        if video_frame_dir.exists():
            shutil.rmtree(video_frame_dir)
        video_frame_dir.mkdir(parents=True, exist_ok=True)

    ref_mask_out = None
    union_mask_out = None
    print("[write] warping images and masks")
    for i, (frame_path, frame_name, H) in enumerate(zip(frame_paths, frame_names, H_smooth)):
        image = read_bgr(frame_path)
        fg = read_fg_mask(mask_map[frame_name], image_hw)
        stable_img = warp_image(image, H, size, args.border_mode)
        stable_fg = warp_mask(fg, H, size)
        stable_img = crop_and_resize(stable_img, args.crop_ratio)
        stable_fg = crop_mask_and_resize(stable_fg, args.crop_ratio)

        write_image(out_img_dir / f"{frame_name}.{image_ext}", stable_img)
        write_image(out_mask_dir / f"{frame_name}.png", stable_fg.astype(np.uint8) * 255)
        if video_frame_dir is not None:
            write_image(video_frame_dir / f"{i+1:08d}.png", stable_img)

        if frame_name == mask_npy_ref_name:
            ref_mask_out = stable_fg
        union_mask_out = stable_fg if union_mask_out is None else (union_mask_out | stable_fg)
        if args.log_every > 0 and ((i + 1) % args.log_every == 0 or i + 1 == len(frame_paths)):
            print(f"  [write] {i+1}/{len(frame_paths)}")

    if args.out_video and video_frame_dir is not None:
        write_h264_video(args.ffmpeg_bin, video_frame_dir, args.out_video, args.fps, args.crf, args.preset)
        if not args.keep_work_dir:
            shutil.rmtree(video_frame_dir)

    if args.out_mask_npy:
        ensure_parent(args.out_mask_npy)
        if args.mask_npy_mode == "reference":
            if ref_mask_out is None:
                raise RuntimeError(f"Mask npy reference frame {mask_npy_ref_name} was not written")
            mask_npy = ref_mask_out
        else:
            mask_npy = union_mask_out
        np.save(args.out_mask_npy, mask_npy.astype(np.uint8))

    homographies_path = Path(args.out_homographies) if args.out_homographies else default_sidecar_path(
        args.out_video, args.out_img_dir, "_ref_anchor_homographies.npy"
    )
    frame_names_path = Path(args.out_frame_names) if args.out_frame_names else default_sidecar_path(
        args.out_video, args.out_img_dir, "_frame_names.json"
    )
    stats_path = Path(args.out_stats) if args.out_stats else default_sidecar_path(
        args.out_video, args.out_img_dir, "_ref_anchor_stats.json"
    )
    ensure_parent(homographies_path)
    np.save(homographies_path, np.stack(H_smooth, axis=0).astype(np.float32))
    write_json(frame_names_path, frame_names)
    write_json(
        stats_path,
        {
            "frame_count": len(frame_names),
            "image_height": h,
            "image_width": w,
            "fps": args.fps,
            "reference_frame": ref_name,
            "reference_index": ref_index,
            "detected_reference_points": int(len(ref_pts_all)),
            "stable_reference_points": int(len(ref_pts)),
            "min_stable_success_rate": float(success_rate[stable].min()),
            "max_stable_avg_fb_error": float(avg_fb_err[stable].max()),
            "valid_homographies": int(sum(1 for H in H_raw if H is not None)),
            "filled_homographies": int(sum(1 for H in H_raw if H is None)),
            "smooth_radius": args.smooth_radius,
            "mask_dilate_iters": args.mask_dilate_iters,
            "reject_current_foreground": reject_current_fg,
            "pass2": pass2_stats,
            "fill": fill_stats,
        },
    )

    print(f"saved stabilized images -> {args.out_img_dir}")
    print(f"saved stabilized masks -> {args.out_mask_dir}")
    if args.out_video:
        print(f"saved stabilized video -> {args.out_video}")
    if args.out_mask_npy:
        print(f"saved modal ROI mask -> {args.out_mask_npy}")
    print(f"saved homographies -> {homographies_path}")
    print(f"saved frame names -> {frame_names_path}")
    print(f"saved stats -> {stats_path}")


if __name__ == "__main__":
    main()
