from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np

from modal_peak_pick.core.cache import write_analysis_cache
from modal_peak_pick.core.background_compensation import load_frame_names
from modal_peak_pick.core.flow import compute_dense_flow_to_reference, contrast_weighted_smooth
from modal_peak_pick.core.pipeline import load_mask
from modal_peak_pick.core.spectrum import fft_over_time, global_power_spectrum
from modal_peak_pick.core.video_io import load_ordered_image_sequence, load_video_clip


def add_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Input video path.")
    source.add_argument("--image-dir", help="Ordered image-sequence directory.")
    parser.add_argument(
        "--frame-names-json",
        default=None,
        help="Ordered JSON list of image filename stems; required with --image-dir.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Image-sequence FPS; required with --image-dir.",
    )
    parser.add_argument(
        "--reference-frame",
        default=None,
        help="Explicit reference-frame stem; required with --image-dir.",
    )
    parser.add_argument(
        "--source-identity",
        default=None,
        help="Optional immutable source identity recorded for ordered image input.",
    )
    parser.add_argument("--cache-dir", required=True, help="New immutable modal-analysis cache directory.")
    parser.add_argument("--mask", default=None, help="Optional binary ROI mask path (.npy or image).")
    parser.add_argument("--t0", type=float, default=None, help="Clip start time in seconds; video input only.")
    parser.add_argument("--t1", type=float, default=None, help="Clip end time in seconds.")
    parser.add_argument("--resize", type=int, default=None, help="Resize max(H,W) before analysis.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum decoded frames.")
    parser.add_argument("--flow-method", choices=["farneback", "tvl1"], default="farneback")
    parser.add_argument("--no-smooth", action="store_true", help="Disable contrast-weighted flow smoothing.")
    parser.add_argument("--sigma-b", type=float, default=3.0, help="Spatial smoothing sigma.")
    parser.add_argument("--sigma-c", type=float, default=0.0, help="Reference pre-blur sigma.")
    parser.add_argument("--analysis-mask-dilate-iters", type=int, default=0, help="Dilate the analysis mask with a 3x3 kernel before flow smoothing and spectrum computation.")


def _source_metadata(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser()
    stat = path.stat()
    return {
        "path": str(path_text),
        "resolved_path": str(path.resolve()),
        "fingerprint": {
            "method": "stat",
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
    }


def _flow_parameters(method: str) -> dict[str, Any]:
    if method == "farneback":
        return {
            "pyr_scale": 0.5,
            "levels": 4,
            "winsize": 15,
            "iterations": 4,
            "poly_n": 5,
            "poly_sigma": 1.1,
            "flags": 0,
        }
    return {
        "tau": 0.25,
        "lambda": 0.05,
        "theta": 0.3,
        "scales": 4,
        "warpings": 5,
        "epsilon": 0.01,
        "inner_iterations": 30,
        "outer_iterations": 10,
        "scale_step": 0.8,
        "gamma": 0.0,
        "median_filtering": 0,
        "use_initial_flow": False,
    }


def run(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir).expanduser()
    if cache_dir.exists() or cache_dir.is_symlink():
        raise FileExistsError(f"Modal analysis cache target already exists: {cache_dir}")

    image_mode = args.image_dir is not None
    if image_mode:
        missing = [
            option
            for option, value in (
                ("--frame-names-json", args.frame_names_json),
                ("--fps", args.fps),
                ("--reference-frame", args.reference_frame),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "--image-dir requires " + ", ".join(missing)
            )
        forbidden = [
            option
            for option, value in (
                ("--resize", args.resize),
                ("--t0", args.t0),
                ("--t1", args.t1),
                ("--max-frames", args.max_frames),
            )
            if value is not None
        ]
        if forbidden:
            raise ValueError(
                "Image-sequence analysis does not allow " + ", ".join(forbidden)
            )
        fps = float(args.fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("--fps must be finite and positive")
    else:
        image_only = [
            option
            for option, value in (
                ("--frame-names-json", args.frame_names_json),
                ("--fps", args.fps),
                ("--reference-frame", args.reference_frame),
                ("--source-identity", args.source_identity),
            )
            if value is not None
        ]
        if image_only:
            raise ValueError(
                "Video analysis does not allow image-sequence options: "
                + ", ".join(image_only)
            )

    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    frame_names: tuple[str, ...] | None = None
    reference_frame_name: str | None = None
    if image_mode:
        frame_names = load_frame_names(args.frame_names_json)
        reference_frame_name = str(args.reference_frame)
        if reference_frame_name not in frame_names:
            raise ValueError(
                f"Reference frame {reference_frame_name!r} is not present in "
                f"{args.frame_names_json}"
            )
        frames_gray = load_ordered_image_sequence(
            args.image_dir,
            frame_names,
            resize=None,
            grayscale=True,
        )
        reference_frame_index = frame_names.index(reference_frame_name)
        frame_range = {
            "t0_s": 0.0,
            "t1_s": None,
            "max_frames": None,
            "decoded_frame_count": int(frames_gray.shape[0]),
        }
    else:
        t0 = 0.0 if args.t0 is None else float(args.t0)
        frames_gray, fps = load_video_clip(
            args.video,
            t0=t0,
            t1=args.t1,
            resize=args.resize,
            grayscale=True,
            max_frames=args.max_frames,
        )
        reference_frame_index = int(frames_gray.shape[0] // 2)
        frame_range = {
            "t0_s": t0,
            "t1_s": None if args.t1 is None else float(args.t1),
            "max_frames": None if args.max_frames is None else int(args.max_frames),
            "decoded_frame_count": int(frames_gray.shape[0]),
        }
    height, width = int(frames_gray.shape[1]), int(frames_gray.shape[2])
    mask = load_mask(
        args.mask,
        height,
        width,
        dilate_iters=int(args.analysis_mask_dilate_iters),
    )
    decode_seconds = time.perf_counter() - stage_start

    reference_frame = frames_gray[reference_frame_index]
    stage_start = time.perf_counter()
    flow_u, flow_v = compute_dense_flow_to_reference(
        frames_gray,
        method=args.flow_method,
        reference_frame_index=reference_frame_index,
    )
    flow_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    if not args.no_smooth:
        flow_u, flow_v = contrast_weighted_smooth(
            flow_u,
            flow_v,
            frame_ref=reference_frame,
            sigma_b=args.sigma_b,
            sigma_c=args.sigma_c,
            mask=mask,
        )
    smooth_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    freqs_hz, spectrum_u, spectrum_v = fft_over_time(
        flow_u,
        flow_v,
        fps=fps,
        detrend=True,
        window="hann",
    )
    fft_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    power_spectrum = global_power_spectrum(spectrum_u, spectrum_v, mask=mask)
    spectrum_seconds = time.perf_counter() - stage_start
    analysis_total = time.perf_counter() - total_start

    source_mask = None if args.mask is None else _source_metadata(args.mask)
    primary_source = _source_metadata(
        args.image_dir if image_mode else args.video
    )
    sources: dict[str, Any] = {
        "video": primary_source,
        "mask": source_mask,
    }
    if image_mode:
        assert frame_names is not None
        assert reference_frame_name is not None
        sources.update(
            {
                "input_type": "image_sequence",
                "image_sequence": {
                    "image_dir": primary_source,
                    "frame_names_json": _source_metadata(args.frame_names_json),
                    "ordered_frame_names": list(frame_names),
                    "reference_frame": reference_frame_name,
                    **(
                        {"source_identity": str(args.source_identity)}
                        if args.source_identity is not None
                        else {}
                    ),
                },
            }
        )
    metadata = {
        "sources": sources,
        "video": {
            "fps": float(fps),
            "frame_range": frame_range,
            "resize_max_side": None if image_mode or args.resize is None else int(args.resize),
        },
        "analysis": {
            "flow_method": str(args.flow_method),
            "flow_parameters": _flow_parameters(str(args.flow_method)),
            "smoothing": {
                "disabled": bool(args.no_smooth),
                "sigma_b": float(args.sigma_b),
                "sigma_c": float(args.sigma_c),
                "analysis_mask_dilate_iters": int(args.analysis_mask_dilate_iters),
                "gradient_pyramid_weights": [0.5, 0.3, 0.2],
            },
            "fft": {
                "detrend": True,
                "window": "hann",
                "block_width": 128,
            },
            "spectrum": {
                "method": "mean_image_plane_amplitude",
            },
            "reference_frame_index": reference_frame_index,
            "reference_time_s": float(frame_range["t0_s"] + reference_frame_index / fps),
            **(
                {"reference_frame_name": reference_frame_name}
                if reference_frame_name is not None
                else {}
            ),
        },
        "timings_seconds": {
            "decode": float(decode_seconds),
            "flow": float(flow_seconds),
            "smooth": float(smooth_seconds),
            "fft": float(fft_seconds),
            "spectrum": float(spectrum_seconds),
            "analysis_total": float(analysis_total),
        },
    }
    cache = write_analysis_cache(
        cache_dir,
        flow_u=flow_u,
        flow_v=flow_v,
        spectrum_u=spectrum_u,
        spectrum_v=spectrum_v,
        freqs_hz=freqs_hz,
        power_spectrum=power_spectrum,
        reference_frame=reference_frame,
        mask=mask,
        metadata=metadata,
    )

    timings = cache.metadata["timings_seconds"]
    print(f"Saved modal analysis cache -> {cache.path}")
    print(
        "Stage timings (s): "
        + ", ".join(
            f"{name}={float(timings[name]):.3f}"
            for name in ("decode", "flow", "smooth", "fft", "spectrum", "cache_write", "total")
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze a video once and write an immutable modal cache.")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
