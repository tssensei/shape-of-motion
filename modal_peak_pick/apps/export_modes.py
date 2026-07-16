from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from modal_peak_pick.core.cache import load_analysis_cache
from modal_peak_pick.core.pipeline import parse_freqs
from modal_peak_pick.core.spectrum import dft_at_frequencies


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", required=True, help="Modal-analysis cache directory.")
    parser.add_argument("--out", default="outputs/modal_analysis.npz", help="Output .npz path.")
    parser.add_argument("--freqs", default=None, help="Comma-separated selected frequencies in Hz.")
    parser.add_argument("--peaks-json", default=None, help="JSON file with selected_peaks_hz.")
    parser.add_argument("--mode-amp-clamp", choices=["none", "local-ratio"], default="none", help="Optional robust amplitude clamp for exported complex modes.")
    parser.add_argument("--mode-amp-local-window", type=int, default=31, help="Odd local median window for --mode-amp-clamp local-ratio.")
    parser.add_argument("--mode-amp-ratio", type=float, default=5.0, help="Local median amplitude multiplier for --mode-amp-clamp local-ratio.")
    parser.add_argument("--mode-amp-global-percentile", type=float, default=99.7, help="Global mask percentile cap for --mode-amp-clamp local-ratio.")


def _read_requested_freqs(freqs_text: str | None, peaks_json: str | None) -> list[float]:
    if freqs_text is not None and peaks_json is not None:
        raise ValueError("Use either --freqs or --peaks-json, not both.")
    if freqs_text is not None:
        return parse_freqs(freqs_text)
    if peaks_json is not None:
        with Path(peaks_json).open("r", encoding="utf-8") as f:
            data = json.load(f)
        vals = data.get("selected_peaks_hz")
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(f"{peaks_json} must contain a non-empty selected_peaks_hz list.")
        return [float(v) for v in vals]
    raise ValueError("Provide --freqs or --peaks-json.")


def _opt_text(value: str | None) -> np.ndarray:
    return np.array("" if value is None else str(value))


def _opt_int(value: int | None) -> np.ndarray:
    return np.array(-1 if value is None else int(value), dtype=np.int32)


def _validate_amp_clamp_args(method: str, local_window: int, ratio: float, global_percentile: float) -> None:
    if method not in {"none", "local-ratio"}:
        raise ValueError("mode_amp_clamp must be 'none' or 'local-ratio'.")
    if local_window <= 0 or local_window % 2 == 0:
        raise ValueError("mode_amp_local_window must be a positive odd integer.")
    if ratio <= 0:
        raise ValueError("mode_amp_ratio must be positive.")
    if not (0.0 < global_percentile <= 100.0):
        raise ValueError("mode_amp_global_percentile must be in (0, 100].")


def _amp_stats(amp: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    vals = amp[valid & np.isfinite(amp)]
    if vals.size == 0:
        raise ValueError("No finite in-mask mode amplitudes are available for clamp statistics.")
    return float(np.percentile(vals, 95)), float(np.percentile(vals, 99)), float(vals.max())


def _median_filter_2d(image: np.ndarray, window: int, chunk_rows: int = 32) -> np.ndarray:
    radius = int(window) // 2
    padded = np.pad(image.astype(np.float32, copy=False), ((radius, radius), (radius, radius)), mode="edge")
    out = np.empty_like(image, dtype=np.float32)
    for row0 in range(0, image.shape[0], int(chunk_rows)):
        row1 = min(image.shape[0], row0 + int(chunk_rows))
        block = padded[row0 : row1 + 2 * radius, :]
        windows = sliding_window_view(block, (int(window), int(window)))
        out[row0:row1] = np.median(windows, axis=(-1, -2)).astype(np.float32)
    return out


def _clamp_mode_amplitude(
    mode_u: np.ndarray,
    mode_v: np.ndarray,
    mask: np.ndarray | None,
    method: str,
    local_window: int,
    ratio: float,
    global_percentile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    amp = np.sqrt((np.abs(mode_u) ** 2 + np.abs(mode_v) ** 2).astype(np.float32))
    valid = np.ones(amp.shape, dtype=bool) if mask is None else mask.astype(bool, copy=False)
    original_p95, original_p99, original_max = _amp_stats(amp, valid)

    if method == "none":
        return mode_u, mode_v, {
            "clamped_fraction": 0.0,
            "scale_min": 1.0,
            "original_p95": original_p95,
            "original_p99": original_p99,
            "original_max": original_max,
            "clamped_p95": original_p95,
            "clamped_p99": original_p99,
            "clamped_max": original_max,
        }

    vals = amp[valid & np.isfinite(amp)]
    fill = float(np.median(vals))
    amp_for_local = np.where(valid & np.isfinite(amp), amp, fill).astype(np.float32)
    local_median = _median_filter_2d(amp_for_local, int(local_window))
    local_cap = local_median * np.float32(ratio)
    global_cap = np.float32(np.percentile(vals, float(global_percentile)))
    target = np.minimum(amp, np.minimum(local_cap, global_cap))
    eps = np.float32(1e-8)
    scale = np.ones_like(amp, dtype=np.float32)
    clamp_pixels = valid & np.isfinite(target) & np.isfinite(amp) & (amp > eps) & (target < amp)
    scale[clamp_pixels] = target[clamp_pixels] / np.maximum(amp[clamp_pixels], eps)
    mode_u_out = (mode_u * scale).astype(np.complex64, copy=False)
    mode_v_out = (mode_v * scale).astype(np.complex64, copy=False)
    clamped_amp = amp * scale
    clamped_p95, clamped_p99, clamped_max = _amp_stats(clamped_amp, valid)
    valid_count = int(valid.sum())
    return mode_u_out, mode_v_out, {
        "clamped_fraction": float(clamp_pixels.sum() / max(valid_count, 1)),
        "scale_min": float(scale[valid].min()) if valid_count > 0 else 1.0,
        "original_p95": original_p95,
        "original_p99": original_p99,
        "original_max": original_max,
        "clamped_p95": clamped_p95,
        "clamped_p99": clamped_p99,
        "clamped_max": clamped_max,
    }


def run(args: argparse.Namespace) -> None:
    mode_amp_clamp = str(getattr(args, "mode_amp_clamp", "none"))
    mode_amp_local_window = int(getattr(args, "mode_amp_local_window", 31))
    mode_amp_ratio = float(getattr(args, "mode_amp_ratio", 5.0))
    mode_amp_global_percentile = float(getattr(args, "mode_amp_global_percentile", 99.7))
    _validate_amp_clamp_args(mode_amp_clamp, mode_amp_local_window, mode_amp_ratio, mode_amp_global_percentile)

    requested_freqs = _read_requested_freqs(args.freqs, args.peaks_json)
    cache = load_analysis_cache(args.cache_dir)
    video_metadata = cache.metadata["video"]
    analysis_metadata = cache.metadata["analysis"]
    sources = cache.metadata["sources"]
    smoothing_metadata = analysis_metadata["smoothing"]
    fft_metadata = analysis_metadata["fft"]
    selected_freqs_hz, exact_u, exact_v = dft_at_frequencies(
        cache.flow_u,
        cache.flow_v,
        fps=cache.fps,
        freqs_hz=requested_freqs,
        detrend=bool(fft_metadata["detrend"]),
        window=str(fft_metadata["window"]),
    )

    modes_u = []
    modes_v = []
    clamp_stats: list[dict[str, float]] = []
    for idx in range(len(requested_freqs)):
        mode_u, mode_v, stats = _clamp_mode_amplitude(
            exact_u[idx],
            exact_v[idx],
            cache.mask,
            mode_amp_clamp,
            mode_amp_local_window,
            mode_amp_ratio,
            mode_amp_global_percentile,
        )
        modes_u.append(mode_u)
        modes_v.append(mode_v)
        clamp_stats.append(stats)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_arr = np.zeros((0, 0), dtype=np.uint8) if cache.mask is None else cache.mask.astype(np.uint8)
    np.savez_compressed(
        out_path,
        freqs_hz=cache.freqs_hz.astype(np.float32, copy=False),
        power_spectrum=cache.power_spectrum.astype(np.float32, copy=False),
        mode_u=np.stack(modes_u, axis=0).astype(np.complex64, copy=False),
        mode_v=np.stack(modes_v, axis=0).astype(np.complex64, copy=False),
        mask=mask_arr,
        has_mask=np.array(cache.mask is not None, dtype=np.uint8),
        reference_frame=cache.reference_frame.astype(np.float32, copy=False),
        fps=np.array(cache.fps, dtype=np.float32),
        t0=np.array(video_metadata["frame_range"]["t0_s"], dtype=np.float32),
        t1=np.array(-1.0 if video_metadata["frame_range"]["t1_s"] is None else video_metadata["frame_range"]["t1_s"], dtype=np.float32),
        resize=_opt_int(video_metadata["resize_max_side"]),
        source_video=np.array(str(sources["video"]["path"])),
        source_mask=_opt_text(None if sources["mask"] is None else str(sources["mask"]["path"])),
        requested_freqs_hz=np.asarray(requested_freqs, dtype=np.float32),
        selected_freqs_hz=selected_freqs_hz.astype(np.float32, copy=False),
        frequency_method=np.array("exact_dft"),
        flow_method=np.array(str(analysis_metadata["flow_method"])),
        no_smooth=np.array(bool(smoothing_metadata["disabled"]), dtype=np.uint8),
        sigma_b=np.array(smoothing_metadata["sigma_b"], dtype=np.float32),
        sigma_c=np.array(smoothing_metadata["sigma_c"], dtype=np.float32),
        t_ref_s=np.array(cache.t_ref_s, dtype=np.float32),
        mode_amp_clamp_method=np.array(mode_amp_clamp),
        mode_amp_local_window=np.array(mode_amp_local_window, dtype=np.int32),
        mode_amp_ratio=np.array(mode_amp_ratio, dtype=np.float32),
        mode_amp_global_percentile=np.array(mode_amp_global_percentile, dtype=np.float32),
        mode_amp_clamped_fraction=np.asarray([s["clamped_fraction"] for s in clamp_stats], dtype=np.float32),
        mode_amp_scale_min=np.asarray([s["scale_min"] for s in clamp_stats], dtype=np.float32),
        mode_amp_original_p95=np.asarray([s["original_p95"] for s in clamp_stats], dtype=np.float32),
        mode_amp_original_p99=np.asarray([s["original_p99"] for s in clamp_stats], dtype=np.float32),
        mode_amp_original_max=np.asarray([s["original_max"] for s in clamp_stats], dtype=np.float32),
        mode_amp_clamped_p95=np.asarray([s["clamped_p95"] for s in clamp_stats], dtype=np.float32),
        mode_amp_clamped_p99=np.asarray([s["clamped_p99"] for s in clamp_stats], dtype=np.float32),
        mode_amp_clamped_max=np.asarray([s["clamped_max"] for s in clamp_stats], dtype=np.float32),
    )
    print(f"Saved modal analysis -> {out_path}")
    print(f"Selected frequencies: {[round(float(f), 6) for f in selected_freqs_hz]}")
    if mode_amp_clamp != "none":
        print(f"Mode amplitude clamp fractions: {[round(float(s['clamped_fraction']), 6) for s in clamp_stats]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export selected 2D complex modal slices.")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
