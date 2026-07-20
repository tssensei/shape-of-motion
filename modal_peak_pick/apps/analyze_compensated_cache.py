from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
import uuid

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modal_peak_pick.core.background_compensation import (
    BackgroundCompensationSettings,
    compute_background_compensated_flow,
    json_float,
    load_frame_names,
    load_ordered_foreground_masks,
    transform_summary,
)
from modal_peak_pick.core.cache import load_analysis_cache, write_analysis_cache
from modal_peak_pick.core.flow import contrast_weighted_smooth
from modal_peak_pick.core.spectrum import fft_over_time, global_power_spectrum
from modal_peak_pick.core.video_io import load_ordered_image_sequence, load_video_clip


DIAGNOSTICS_FILENAME = "camera_compensation_diagnostics.npz"
SUMMARY_FILENAME = "camera_compensation_summary.json"
CURVES_FILENAME = "camera_compensation_curves.png"
PREVIEW_FILENAME = "camera_compensation_previews.png"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Raw, unstabilized input video.")
    source.add_argument(
        "--image-dir",
        help="Raw, unstabilized image sequence indexed by the frame-name sidecar.",
    )
    parser.add_argument(
        "--foreground-mask-dir",
        required=True,
        help="Per-frame raw foreground masks indexed by the frame-name sidecar.",
    )
    parser.add_argument(
        "--frame-names-json",
        required=True,
        help="Ordered JSON list of raw mask filename stems.",
    )
    parser.add_argument(
        "--reference-cache",
        required=True,
        help="Current stabilized cache defining the target grid and reference frame.",
    )
    parser.add_argument("--cache-dir", required=True, help="New immutable output cache.")
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Raw video max-side resize; defaults to the reference cache setting.",
    )
    parser.add_argument("--mask-dilate-px", type=int, default=16)
    parser.add_argument(
        "--transform-model",
        choices=["similarity", "affine", "homography"],
        default="similarity",
    )
    parser.add_argument("--max-corners", type=int, default=4000)
    parser.add_argument("--feature-quality", type=float, default=0.01)
    parser.add_argument("--feature-min-distance-px", type=float, default=8.0)
    parser.add_argument("--lk-window-px", type=int, default=31)
    parser.add_argument("--lk-max-level", type=int, default=4)
    parser.add_argument("--max-forward-backward-error-px", type=float, default=1.5)
    parser.add_argument("--ransac-threshold-px", type=float, default=2.0)
    parser.add_argument("--min-inliers", type=int, default=30)
    parser.add_argument("--min-inlier-fraction", type=float, default=0.1)
    parser.add_argument("--min-background-coverage", type=float, default=0.05)
    parser.add_argument("--min-candidate-valid-fraction", type=float, default=0.99)
    parser.add_argument(
        "--flow-method", choices=["farneback", "tvl1"], default="farneback"
    )
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--sigma-b", type=float, default=3.0)
    parser.add_argument("--sigma-c", type=float, default=0.0)


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


def _statistics(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": json_float(np.mean(array)),
        "p50": json_float(np.percentile(array, 50)),
        "p90": json_float(np.percentile(array, 90)),
        "p99": json_float(np.percentile(array, 99)),
        "max": json_float(np.max(array)),
    }


def _write_diagnostics(
    directory: Path,
    result: Any,
    *,
    settings: BackgroundCompensationSettings,
    reference_index: int,
    reference_cache_path: Path,
    raw_source: Path,
) -> None:
    np.savez_compressed(
        directory / DIAGNOSTICS_FILENAME,
        format=np.asarray("background_compensated_flow"),
        version=np.asarray(1, dtype=np.int64),
        reference_frame_index=np.asarray(reference_index, dtype=np.int64),
        raw_reference_to_stabilized_reference=result.raw_reference_to_stabilized_reference,
        raw_reference_to_raw_frame=result.raw_reference_to_raw_frame,
        raw_frame_to_stabilized_reference=result.raw_frame_to_stabilized_reference,
        residual_background_transform=result.residual_background_transform,
        tracked_count=result.tracked_count,
        inlier_count=result.inlier_count,
        inlier_fraction=result.inlier_fraction,
        forward_backward_rmse=result.forward_backward_rmse,
        reprojection_rmse=result.reprojection_rmse,
        spatial_coverage=result.spatial_coverage,
        valid_fraction=result.valid_fraction,
        candidate_valid_fraction=result.candidate_valid_fraction,
        residual_background_rms=result.residual_background_rms,
        residual_background_p90=result.residual_background_p90,
        camera_flow_foreground_rms=result.camera_flow_foreground_rms,
        residual_camera_flow_foreground_rms=result.residual_camera_flow_foreground_rms,
        aligned_preview_indices=result.aligned_preview_indices,
        aligned_previews=result.aligned_previews,
    )
    frame_transforms = [
        transform_summary(matrix) for matrix in result.raw_reference_to_raw_frame
    ]
    summary = {
        "format": "background_compensated_flow",
        "version": 1,
        "source": {
            "raw_input": str(raw_source.resolve()),
            "reference_cache": str(reference_cache_path.resolve()),
        },
        "settings": {
            name: getattr(settings, name)
            for name in settings.__dataclass_fields__
        },
        "reference_frame_index": int(reference_index),
        "frame_count": int(result.flow_u.shape[0]),
        "raw_reference_to_stabilized_reference": transform_summary(
            result.raw_reference_to_stabilized_reference
        ),
        "statistics": {
            "tracked_count": _statistics(result.tracked_count),
            "inlier_count": _statistics(result.inlier_count),
            "inlier_fraction": _statistics(result.inlier_fraction),
            "forward_backward_rmse_px": _statistics(
                result.forward_backward_rmse
            ),
            "reprojection_rmse_px": _statistics(result.reprojection_rmse),
            "spatial_coverage": _statistics(result.spatial_coverage),
            "valid_fraction": _statistics(result.valid_fraction),
            "candidate_valid_fraction": _statistics(
                result.candidate_valid_fraction
            ),
            "residual_background_rms_px": _statistics(
                result.residual_background_rms
            ),
            "residual_background_p90_px": _statistics(
                result.residual_background_p90
            ),
            "camera_flow_foreground_rms_px": _statistics(
                result.camera_flow_foreground_rms
            ),
            "residual_camera_flow_foreground_rms_px": _statistics(
                result.residual_camera_flow_foreground_rms
            ),
        },
        "per_frame_transform": frame_transforms,
    }
    with (directory / SUMMARY_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")

    frames = np.arange(result.flow_u.shape[0], dtype=np.int64)
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(frames, result.inlier_fraction, label="RANSAC inlier fraction")
    axes[0].plot(frames, result.spatial_coverage, label="background coverage")
    axes[0].legend()
    axes[0].set_ylabel("fraction")
    axes[1].plot(frames, result.reprojection_rmse, label="raw transform reprojection")
    axes[1].plot(frames, result.residual_background_rms, label="post-warp background")
    axes[1].set_ylabel("pixels RMS")
    axes[1].legend()
    axes[2].plot(
        frames, result.camera_flow_foreground_rms, label="camera flow on foreground"
    )
    axes[2].plot(
        frames,
        result.residual_camera_flow_foreground_rms,
        label="post-warp residual camera flow",
    )
    axes[2].set_xlabel("frame index")
    axes[2].set_ylabel("pixels RMS")
    axes[2].legend()
    figure.suptitle("Raw-video background motion compensation")
    figure.tight_layout()
    figure.savefig(directory / CURVES_FILENAME, dpi=160)
    plt.close(figure)

    preview_count = int(result.aligned_previews.shape[0])
    figure, axes = plt.subplots(1, preview_count, figsize=(6 * preview_count, 5))
    axes_array = np.atleast_1d(axes)
    for axis, frame_index, preview in zip(
        axes_array, result.aligned_preview_indices, result.aligned_previews
    ):
        axis.imshow(preview, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"aligned raw frame {int(frame_index)}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(directory / PREVIEW_FILENAME, dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    target = Path(args.cache_dir).expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Modal analysis cache target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    reference_cache_path = Path(args.reference_cache).expanduser()
    reference_cache = load_analysis_cache(reference_cache_path)
    if reference_cache.mask is None:
        raise ValueError("The reference cache must contain a foreground analysis mask")

    settings = BackgroundCompensationSettings(
        mask_dilate_px=int(args.mask_dilate_px),
        transform_model=str(args.transform_model),
        max_corners=int(args.max_corners),
        feature_quality=float(args.feature_quality),
        feature_min_distance_px=float(args.feature_min_distance_px),
        lk_window_px=int(args.lk_window_px),
        lk_max_level=int(args.lk_max_level),
        max_forward_backward_error_px=float(args.max_forward_backward_error_px),
        ransac_threshold_px=float(args.ransac_threshold_px),
        min_inliers=int(args.min_inliers),
        min_inlier_fraction=float(args.min_inlier_fraction),
        min_background_coverage=float(args.min_background_coverage),
        min_candidate_valid_fraction=float(args.min_candidate_valid_fraction),
    )
    resize = args.resize
    if resize is None:
        resize = reference_cache.metadata["video"]["resize_max_side"]
    frame_range = reference_cache.metadata["video"]["frame_range"]

    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    frame_names = load_frame_names(args.frame_names_json)
    if args.image_dir is not None:
        raw_source = Path(args.image_dir).expanduser()
        raw_frames = load_ordered_image_sequence(
            raw_source,
            frame_names,
            resize=resize,
            grayscale=True,
        )
        raw_fps = float(reference_cache.fps)
    else:
        raw_source = Path(args.video).expanduser()
        raw_frames, raw_fps = load_video_clip(
            args.video,
            t0=float(frame_range["t0_s"]),
            t1=(
                None
                if frame_range["t1_s"] is None
                else float(frame_range["t1_s"])
            ),
            resize=resize,
            grayscale=True,
            max_frames=frame_range["max_frames"],
        )
        if len(frame_names) != raw_frames.shape[0]:
            raise ValueError(
                f"Frame-name sidecar has {len(frame_names)} entries but raw video "
                f"decoded {raw_frames.shape[0]} frames"
            )
    raw_masks = load_ordered_foreground_masks(
        args.foreground_mask_dir,
        frame_names,
        height=int(raw_frames.shape[1]),
        width=int(raw_frames.shape[2]),
    )
    decode_seconds = time.perf_counter() - stage_start
    if raw_frames.shape[0] != reference_cache.flow_u.shape[0]:
        raise ValueError(
            "Raw video frame count does not match the reference cache: "
            f"{raw_frames.shape[0]} versus {reference_cache.flow_u.shape[0]}"
        )
    if not np.isclose(raw_fps, reference_cache.fps, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Raw video FPS {raw_fps} does not match reference cache FPS "
            f"{reference_cache.fps}"
        )

    stage_start = time.perf_counter()
    compensated = compute_background_compensated_flow(
        raw_frames,
        raw_masks,
        np.asarray(reference_cache.reference_frame, dtype=np.float32),
        np.asarray(reference_cache.mask, dtype=np.uint8),
        int(reference_cache.metadata["analysis"]["reference_frame_index"]),
        settings,
        flow_method=str(args.flow_method),
    )
    compensation_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    flow_u = compensated.flow_u
    flow_v = compensated.flow_v
    if not args.no_smooth:
        flow_u, flow_v = contrast_weighted_smooth(
            flow_u,
            flow_v,
            frame_ref=np.asarray(reference_cache.reference_frame, dtype=np.float32),
            sigma_b=float(args.sigma_b),
            sigma_c=float(args.sigma_c),
            mask=np.asarray(reference_cache.mask, dtype=bool),
        )
    smooth_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    freqs_hz, spectrum_u, spectrum_v = fft_over_time(
        flow_u, flow_v, fps=raw_fps, detrend=True, window="hann"
    )
    fft_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    power_spectrum = global_power_spectrum(
        spectrum_u, spectrum_v, mask=reference_cache.mask
    )
    spectrum_seconds = time.perf_counter() - stage_start
    analysis_total = time.perf_counter() - total_start

    metadata = {
        "sources": {
            "video": _source_metadata(str(raw_source)),
            "raw_input_type": "image_sequence" if args.image_dir is not None else "video",
            "mask": _source_metadata(args.frame_names_json),
            "foreground_mask_dir": str(Path(args.foreground_mask_dir).resolve()),
            "reference_cache": str(reference_cache_path.resolve()),
        },
        "video": {
            "fps": float(raw_fps),
            "frame_range": dict(frame_range),
            "resize_max_side": None if resize is None else int(resize),
        },
        "analysis": {
            "flow_method": str(args.flow_method),
            "flow_parameters": _flow_parameters(str(args.flow_method)),
            "smoothing": {
                "disabled": bool(args.no_smooth),
                "sigma_b": float(args.sigma_b),
                "sigma_c": float(args.sigma_c),
                "analysis_mask_dilate_iters": 0,
                "gradient_pyramid_weights": [0.5, 0.3, 0.2],
            },
            "fft": {"detrend": True, "window": "hann", "block_width": 128},
            "spectrum": {"method": "mean_image_plane_amplitude"},
            "reference_frame_index": int(
                reference_cache.metadata["analysis"]["reference_frame_index"]
            ),
            "reference_time_s": float(
                reference_cache.metadata["analysis"]["reference_time_s"]
            ),
            "camera_compensation": {
                "format": "background_compensated_flow",
                "version": 1,
                "target_grid": "stabilized_reference_cache",
                "direct_to_reference": True,
                "sequential_accumulation": False,
                "single_resampling_per_frame": True,
                "settings": {
                    name: getattr(settings, name)
                    for name in settings.__dataclass_fields__
                },
            },
        },
        "timings_seconds": {
            "decode": float(decode_seconds),
            "flow": float(compensation_seconds),
            "smooth": float(smooth_seconds),
            "fft": float(fft_seconds),
            "spectrum": float(spectrum_seconds),
            "analysis_total": float(analysis_total),
        },
    }

    staged_target = target.parent / f".{target.name}.{uuid.uuid4().hex}.stage"
    try:
        write_analysis_cache(
            staged_target,
            flow_u=flow_u,
            flow_v=flow_v,
            spectrum_u=spectrum_u,
            spectrum_v=spectrum_v,
            freqs_hz=freqs_hz,
            power_spectrum=power_spectrum,
            reference_frame=np.asarray(reference_cache.reference_frame, dtype=np.float32),
            mask=np.asarray(reference_cache.mask, dtype=bool),
            metadata=metadata,
        )
        _write_diagnostics(
            staged_target,
            compensated,
            settings=settings,
            reference_index=int(
                reference_cache.metadata["analysis"]["reference_frame_index"]
            ),
            reference_cache_path=reference_cache_path,
            raw_source=raw_source,
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Modal analysis cache target already exists: {target}")
        os.replace(staged_target, target)
    except Exception:
        shutil.rmtree(staged_target, ignore_errors=True)
        raise

    print(f"Saved background-compensated modal cache -> {target}")
    print(
        "Compensation diagnostics: "
        f"candidate_valid_min={float(np.min(compensated.candidate_valid_fraction)):.6f}, "
        f"background_residual_rms_p90="
        f"{float(np.percentile(compensated.residual_background_rms, 90)):.6f}px, "
        f"foreground_camera_rms_mean="
        f"{float(np.mean(compensated.camera_flow_foreground_rms)):.6f}px"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standard modal cache from an unstabilized sequence using "
            "direct-to-reference background motion compensation."
        )
    )
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
