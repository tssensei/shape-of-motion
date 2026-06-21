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
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


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


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool, copy=False)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations) > 0


def odd_kernel_size(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value if value % 2 == 1 else value + 1


def make_background_frame(
    image: np.ndarray,
    exclude_mask: np.ndarray,
    fill_mode: str,
    blur_ksize: int,
    feather_ksize: int,
) -> np.ndarray:
    alpha = exclude_mask.astype(np.float32)
    if feather_ksize > 1:
        k = odd_kernel_size(feather_ksize, "--feather-ksize")
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    if fill_mode == "black":
        fill = np.zeros_like(image)
    elif fill_mode == "gray":
        fill = np.full_like(image, 127)
    elif fill_mode == "blur":
        k = odd_kernel_size(blur_ksize, "--blur-ksize")
        fill = cv2.GaussianBlur(image, (k, k), 0)
    else:
        raise ValueError(f"Unknown fill mode: {fill_mode}")

    blended = image.astype(np.float32) * (1.0 - alpha) + fill.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def ensure_parent(path: str | Path) -> None:
    out_dir = Path(path).parent
    if str(out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, data) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
    return out_dir.parent / f"{out_dir.name}_vidstab_work"


def ffmpeg_filter_path(path: Path) -> str:
    text = str(path).replace("\\", "/")
    return text.replace(":", "\\:").replace(",", "\\,").replace("'", "\\'")


def run_ffmpeg(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def check_vidstab_filters(ffmpeg_bin: str) -> None:
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-filters"],
        check=True,
        text=True,
        capture_output=True,
    )
    text = proc.stdout + proc.stderr
    missing = [
        name
        for name in ("vidstabdetect", "vidstabtransform")
        if name not in text
    ]
    if missing:
        raise RuntimeError(
            f"ffmpeg is missing required filters: {missing}. "
            "Use an ffmpeg build compiled with --enable-libvidstab."
        )


def make_detect_filter(args, trf_path: Path) -> str:
    parts = [
        f"shakiness={args.shakiness}",
        f"accuracy={args.accuracy}",
        f"stepsize={args.stepsize}",
        f"mincontrast={args.mincontrast:g}",
        f"result={ffmpeg_filter_path(trf_path)}",
    ]
    if args.tripod:
        parts.append("tripod=1")
    return "vidstabdetect=" + ":".join(parts)


def make_transform_filter(args, trf_path: Path) -> str:
    parts = [
        f"input={ffmpeg_filter_path(trf_path)}",
        f"smoothing={args.smoothing}",
        f"optzoom={args.optzoom}",
        f"crop={args.crop}",
        f"interpol={args.interpol}",
    ]
    if args.zoom:
        parts.append(f"zoom={args.zoom:g}")
    if args.tripod:
        parts.append("tripod=1")
    return "vidstabtransform=" + ":".join(parts)


def prepare_session_dir(work_dir: Path) -> Path:
    session_dir = work_dir / "_vidstab_tmp"
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True)
    return session_dir


def write_sequence_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def collect_union_mask(
    frame_paths: list[Path],
    mask_map: dict[str, Path],
    image_hw: tuple[int, int],
) -> np.ndarray:
    union = np.zeros(image_hw, dtype=bool)
    for frame_path in frame_paths:
        mask_path = mask_map.get(frame_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for frame {frame_path.stem}")
        union |= read_fg_mask(mask_path, image_hw)
    return union


def create_temp_sequences(
    frame_paths: list[Path],
    mask_map: dict[str, Path],
    session_dir: Path,
    union_exclude: np.ndarray | None,
    args,
) -> dict[str, Path | list[float]]:
    original_dir = session_dir / "original"
    mask_dir = session_dir / "masks"
    bg_dir = session_dir / "background_only"
    original_dir.mkdir()
    mask_dir.mkdir()
    bg_dir.mkdir()

    bg_ratios = []
    for idx, frame_path in enumerate(frame_paths, start=1):
        image = read_bgr(frame_path)
        mask_path = mask_map.get(frame_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for frame {frame_path.stem}")
        fg = read_fg_mask(mask_path, image.shape[:2])
        exclude = union_exclude
        if exclude is None:
            exclude = dilate_mask(fg, args.mask_dilate_iters)
        bg_ratios.append(float((~exclude).mean()))

        out_name = f"{idx:08d}.png"
        write_sequence_image(original_dir / out_name, image)
        write_sequence_image(mask_dir / out_name, fg.astype(np.uint8) * 255)
        bg_frame = make_background_frame(
            image,
            exclude,
            args.fill_mode,
            args.blur_ksize,
            args.feather_ksize,
        )
        write_sequence_image(bg_dir / out_name, bg_frame)

    min_bg_ratio = min(bg_ratios)
    if min_bg_ratio < args.min_background_ratio:
        raise RuntimeError(
            f"Only {min_bg_ratio:.3f} background pixels remain after masking. "
            f"Minimum required by --min-background-ratio is {args.min_background_ratio:.3f}."
        )

    return {
        "original_dir": original_dir,
        "mask_dir": mask_dir,
        "bg_dir": bg_dir,
        "background_ratios": bg_ratios,
    }


def require_temp_outputs(seq_dir: Path, count: int) -> list[Path]:
    paths = [seq_dir / f"{idx:08d}.png" for idx in range(1, count + 1)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing ffmpeg output frames. First missing: {missing[0]}")
    return paths


def write_named_image(src: Path, dst: Path, image_ext: str) -> None:
    if image_ext.lower() == "png":
        shutil.copyfile(src, dst)
        return
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read ffmpeg output image: {src}")
    if not cv2.imwrite(str(dst), image):
        raise RuntimeError(f"Could not write image: {dst}")


def materialize_outputs(
    stable_img_dir: Path,
    stable_mask_dir: Path,
    frame_names: list[str],
    out_img_dir: Path,
    out_mask_dir: Path,
    image_ext: str,
    mask_npy_mode: str,
    mask_npy_ref_name: str,
) -> tuple[np.ndarray | None, dict]:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    stable_imgs = require_temp_outputs(stable_img_dir, len(frame_names))
    stable_masks = require_temp_outputs(stable_mask_dir, len(frame_names))

    ref_mask = None
    union_mask = None
    for src_img, src_mask, frame_name in zip(stable_imgs, stable_masks, frame_names):
        write_named_image(src_img, out_img_dir / f"{frame_name}.{image_ext}", image_ext)

        mask_img = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise FileNotFoundError(f"Cannot read ffmpeg output mask: {src_mask}")
        fg = mask_img > 127
        write_sequence_image(out_mask_dir / f"{frame_name}.png", fg.astype(np.uint8) * 255)
        if frame_name == mask_npy_ref_name:
            ref_mask = fg
        union_mask = fg if union_mask is None else (union_mask | fg)

    if mask_npy_mode == "reference":
        if ref_mask is None:
            raise ValueError(f"Reference mask frame {mask_npy_ref_name} was not generated")
        out_mask = ref_mask
    elif mask_npy_mode == "union":
        out_mask = union_mask
    else:
        raise ValueError(f"Unknown mask npy mode: {mask_npy_mode}")

    stats = {
        "output_image_dir": str(out_img_dir),
        "output_mask_dir": str(out_mask_dir),
        "mask_npy_mode": mask_npy_mode,
        "mask_npy_reference_frame": mask_npy_ref_name,
    }
    return out_mask, stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stabilize a frame sequence with ffmpeg vidstab while estimating transforms "
            "from SOM-background-only frames."
        )
    )
    parser.add_argument("--img-dir", required=True, help="Input image sequence directory.")
    parser.add_argument("--mask-dir", required=True, help="Input SOM foreground mask directory.")
    parser.add_argument("--out-img-dir", required=True, help="Output stabilized image directory.")
    parser.add_argument("--out-mask-dir", required=True, help="Output stabilized foreground mask directory.")
    parser.add_argument("--out-video", default="", help="Optional output stabilized H.264 mp4 path.")
    parser.add_argument("--out-mask-npy", default="", help="Optional foreground ROI mask .npy path.")
    parser.add_argument("--out-trf", default="", help="Optional vidstab transform .trf path.")
    parser.add_argument("--out-frame-names", default="", help="Optional frame names .json path.")
    parser.add_argument("--out-stats", default="", help="Optional stats .json path.")
    parser.add_argument("--work-dir", default="", help="Directory for temporary image sequences.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable.")
    parser.add_argument("--fps", type=float, default=30.0, help="Input/output FPS.")
    parser.add_argument("--image-ext", default="png", help="Output image extension.")
    parser.add_argument("--mask-dilate-iters", type=int, default=9, help="Dilate foreground masks before background fill.")
    parser.add_argument(
        "--background-mask-mode",
        choices=["union", "frame"],
        default="union",
        help="Use one union exclusion mask for all frames, or each frame mask independently.",
    )
    parser.add_argument(
        "--fill-mode",
        choices=["blur", "black", "gray"],
        default="blur",
        help="How to fill foreground pixels for vidstab detection frames.",
    )
    parser.add_argument("--blur-ksize", type=int, default=101, help="Gaussian blur kernel for --fill-mode blur.")
    parser.add_argument("--feather-ksize", type=int, default=31, help="Feather kernel for foreground fill alpha.")
    parser.add_argument("--min-background-ratio", type=float, default=0.15, help="Fail if too little background remains.")
    parser.add_argument("--shakiness", type=int, default=5, help="vidstabdetect shakiness.")
    parser.add_argument("--accuracy", type=int, default=15, help="vidstabdetect accuracy.")
    parser.add_argument("--stepsize", type=int, default=6, help="vidstabdetect stepsize.")
    parser.add_argument("--mincontrast", type=float, default=0.25, help="vidstabdetect mincontrast.")
    parser.add_argument("--smoothing", type=int, default=60, help="vidstabtransform smoothing window.")
    parser.add_argument("--optzoom", type=int, default=0, help="vidstabtransform optzoom.")
    parser.add_argument("--zoom", type=float, default=0.0, help="vidstabtransform fixed extra zoom.")
    parser.add_argument("--crop", choices=["keep", "black"], default="black", help="vidstabtransform crop mode.")
    parser.add_argument(
        "--interpol",
        choices=["no", "linear", "bilinear", "bicubic"],
        default="bilinear",
        help="vidstabtransform interpolation.",
    )
    parser.add_argument("--tripod", action="store_true", help="Use vidstab virtual tripod mode.")
    parser.add_argument(
        "--mask-npy-mode",
        choices=["reference", "union"],
        default="reference",
        help="How to build --out-mask-npy from stabilized foreground masks.",
    )
    parser.add_argument("--mask-npy-ref-name", default="", help="Reference frame stem for --mask-npy-mode reference.")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF for --out-video.")
    parser.add_argument("--preset", default="medium", help="H.264 preset for --out-video.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep temporary sequences after success.")
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.mask_dilate_iters < 0:
        raise ValueError("--mask-dilate-iters must be non-negative")
    if args.min_background_ratio <= 0 or args.min_background_ratio >= 1:
        raise ValueError("--min-background-ratio must be between 0 and 1")
    if Path(args.img_dir).resolve() == Path(args.out_img_dir).resolve():
        raise ValueError("--out-img-dir must differ from --img-dir")
    if Path(args.mask_dir).resolve() == Path(args.out_mask_dir).resolve():
        raise ValueError("--out-mask-dir must differ from --mask-dir")

    check_vidstab_filters(args.ffmpeg_bin)

    frame_paths = list_frame_paths(args.img_dir)
    frame_names = [path.stem for path in frame_paths]
    image_ext = args.image_ext.lstrip(".")
    if not image_ext:
        raise ValueError("--image-ext must not be empty")

    first_img = read_bgr(frame_paths[0])
    image_hw = first_img.shape[:2]
    for frame_path in frame_paths[1:]:
        image = read_bgr(frame_path)
        if image.shape[:2] != image_hw:
            raise ValueError(
                f"Image size mismatch for {frame_path}: {image.shape[:2]} != {image_hw}"
            )

    mask_map = build_mask_map(args.mask_dir)
    missing_masks = [name for name in frame_names if name not in mask_map]
    if missing_masks:
        raise FileNotFoundError(f"No mask found for frame {missing_masks[0]}")

    mask_npy_ref_name = args.mask_npy_ref_name or frame_names[len(frame_names) // 2]
    if mask_npy_ref_name not in frame_names:
        raise ValueError(f"Reference mask frame {mask_npy_ref_name} not found in input frames")

    work_dir = Path(args.work_dir) if args.work_dir else default_work_dir(args.out_video, args.out_img_dir)
    session_dir = prepare_session_dir(work_dir)
    trf_path = Path(args.out_trf) if args.out_trf else default_sidecar_path(
        args.out_video, args.out_img_dir, "_vidstab.trf"
    )
    frame_names_path = Path(args.out_frame_names) if args.out_frame_names else default_sidecar_path(
        args.out_video, args.out_img_dir, "_frame_names.json"
    )
    stats_path = Path(args.out_stats) if args.out_stats else default_sidecar_path(
        args.out_video, args.out_img_dir, "_vidstab_stats.json"
    )
    ensure_parent(trf_path)
    ensure_parent(frame_names_path)
    ensure_parent(stats_path)

    union_exclude = None
    if args.background_mask_mode == "union":
        fg_union = collect_union_mask(frame_paths, mask_map, image_hw)
        union_exclude = dilate_mask(fg_union, args.mask_dilate_iters)

    sequence_info = create_temp_sequences(
        frame_paths,
        mask_map,
        session_dir,
        union_exclude,
        args,
    )
    original_dir = sequence_info["original_dir"]
    input_mask_dir = sequence_info["mask_dir"]
    bg_dir = sequence_info["bg_dir"]

    detect_filter = make_detect_filter(args, trf_path)
    run_ffmpeg(
        [
            args.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-framerate",
            f"{args.fps:g}",
            "-start_number",
            "1",
            "-i",
            str(bg_dir / "%08d.png"),
            "-vf",
            f"format=gray,{detect_filter}",
            "-f",
            "null",
            "-",
        ]
    )

    transform_filter = make_transform_filter(args, trf_path)
    stable_img_tmp = session_dir / "stable_images"
    stable_mask_tmp = session_dir / "stable_masks"
    stable_img_tmp.mkdir()
    stable_mask_tmp.mkdir()
    run_ffmpeg(
        [
            args.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-framerate",
            f"{args.fps:g}",
            "-start_number",
            "1",
            "-i",
            str(original_dir / "%08d.png"),
            "-vf",
            f"{transform_filter},format=rgb24",
            str(stable_img_tmp / "%08d.png"),
        ]
    )
    run_ffmpeg(
        [
            args.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-framerate",
            f"{args.fps:g}",
            "-start_number",
            "1",
            "-i",
            str(input_mask_dir / "%08d.png"),
            "-vf",
            f"{transform_filter},format=gray",
            str(stable_mask_tmp / "%08d.png"),
        ]
    )

    out_mask, output_stats = materialize_outputs(
        stable_img_tmp,
        stable_mask_tmp,
        frame_names,
        Path(args.out_img_dir),
        Path(args.out_mask_dir),
        image_ext,
        args.mask_npy_mode,
        mask_npy_ref_name,
    )

    if args.out_mask_npy:
        ensure_parent(args.out_mask_npy)
        np.save(args.out_mask_npy, out_mask.astype(np.uint8))

    if args.out_video:
        ensure_parent(args.out_video)
        run_ffmpeg(
            [
                args.ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-framerate",
                f"{args.fps:g}",
                "-start_number",
                "1",
                "-i",
                str(stable_img_tmp / "%08d.png"),
                "-vf",
                "format=yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                str(args.crf),
                "-preset",
                args.preset,
                "-movflags",
                "+faststart",
                args.out_video,
            ]
        )

    write_json(frame_names_path, frame_names)
    write_json(
        stats_path,
        {
            "frame_count": len(frame_names),
            "image_height": image_hw[0],
            "image_width": image_hw[1],
            "fps": args.fps,
            "background_mask_mode": args.background_mask_mode,
            "fill_mode": args.fill_mode,
            "mask_dilate_iters": args.mask_dilate_iters,
            "blur_ksize": args.blur_ksize,
            "feather_ksize": args.feather_ksize,
            "min_background_ratio": min(sequence_info["background_ratios"]),
            "mean_background_ratio": float(np.mean(sequence_info["background_ratios"])),
            "vidstab_trf": str(trf_path),
            "detect_filter": detect_filter,
            "transform_filter": transform_filter,
            **output_stats,
        },
    )

    if not args.keep_work_dir:
        shutil.rmtree(session_dir)

    print(f"saved stabilized images -> {args.out_img_dir}")
    print(f"saved stabilized masks -> {args.out_mask_dir}")
    if args.out_video:
        print(f"saved stabilized video -> {args.out_video}")
    if args.out_mask_npy:
        print(f"saved modal ROI mask -> {args.out_mask_npy}")
    print(f"saved vidstab transforms -> {trf_path}")
    print(f"saved frame names -> {frame_names_path}")
    print(f"saved stats -> {stats_path}")


if __name__ == "__main__":
    main()
