from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

from modal_peak_pick.core.cache import load_analysis_cache
from modal_peak_pick.core.spectrum import amplitude_map


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", required=True, help="Modal-analysis cache directory.")
    parser.add_argument("--out-dir", required=True, help="Output directory for PNG/JSON inspection files.")
    parser.add_argument("--num-peaks", type=int, default=8, help="Number of candidate peaks to export.")
    parser.add_argument("--min-freq-hz", type=float, default=0.2, help="Ignore candidates below this frequency.")
    parser.add_argument("--max-freq-hz", type=float, default=None, help="Ignore candidates above this frequency.")
    parser.add_argument("--peak-window-hz", type=float, default=0.6, help="Minimum spacing between candidate peaks.")
    parser.add_argument("--preview-percentile", type=float, default=99.0, help="Display percentile for mode previews.")


def _phase_hsv(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    phase = np.angle(z)
    mag = np.abs(z).astype(np.float32)
    hue = (phase + np.pi) / (2.0 * np.pi)
    sat = np.ones_like(hue, dtype=np.float32)
    val = np.clip((mag - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0).astype(np.float32)
    return hsv_to_rgb(np.stack([hue.astype(np.float32), sat, val], axis=-1))


def _normalize01(x: np.ndarray, hi: float, lo: float = 0.0) -> np.ndarray:
    return np.clip((x.astype(np.float32) - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0)


def _valid_frequency_mask(freqs_hz: np.ndarray, min_freq_hz: float, max_freq_hz: float | None) -> np.ndarray:
    if min_freq_hz < 0:
        raise ValueError("min-freq-hz must be non-negative.")
    mask = freqs_hz >= float(min_freq_hz)
    if max_freq_hz is not None:
        if max_freq_hz <= min_freq_hz:
            raise ValueError("max-freq-hz must be greater than min-freq-hz.")
        mask &= freqs_hz <= float(max_freq_hz)
    return mask


def _local_maxima(power: np.ndarray) -> np.ndarray:
    if power.ndim != 1:
        raise ValueError("power spectrum must be 1D.")
    if power.shape[0] < 3:
        return np.zeros(power.shape[0], dtype=bool)
    mid = power[1:-1]
    is_peak = ((mid >= power[:-2]) & (mid > power[2:])) | ((mid > power[:-2]) & (mid >= power[2:]))
    out = np.zeros(power.shape[0], dtype=bool)
    out[1:-1] = is_peak
    return out


def _choose_candidate_peaks(
    freqs_hz: np.ndarray,
    power: np.ndarray,
    num_peaks: int,
    min_freq_hz: float,
    max_freq_hz: float | None,
    peak_window_hz: float,
) -> list[int]:
    if num_peaks <= 0:
        raise ValueError("num-peaks must be positive.")
    if peak_window_hz < 0:
        raise ValueError("peak-window-hz must be non-negative.")

    valid = _valid_frequency_mask(freqs_hz, min_freq_hz, max_freq_hz)
    candidates = np.where(valid & _local_maxima(power))[0]
    if candidates.size == 0:
        raise ValueError(
            "No local peaks found in the requested frequency range. "
            "Try lowering --min-freq-hz or inspecting a longer clip."
        )

    ordered = candidates[np.argsort(power[candidates])[::-1]]
    chosen: list[int] = []
    for idx in ordered.tolist():
        freq = float(freqs_hz[idx])
        if all(abs(freq - float(freqs_hz[j])) >= peak_window_hz for j in chosen):
            chosen.append(int(idx))
        if len(chosen) >= num_peaks:
            break
    return chosen


def _masked_values(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is not None and np.any(mask):
        return values[mask]
    return values.ravel()


def _mode_display_scale(U_slice: np.ndarray, V_slice: np.ndarray, mask: np.ndarray | None, percentile: float) -> float:
    if not (0 < percentile <= 100):
        raise ValueError("preview-percentile must be in (0, 100].")
    mag = np.sqrt((np.abs(U_slice) ** 2 + np.abs(V_slice) ** 2).astype(np.float32))
    vals = _masked_values(mag, mask)
    if vals.size == 0:
        return 1.0
    hi = float(np.percentile(vals, percentile))
    if hi <= 0:
        hi = float(vals.max(initial=1.0))
    return max(hi, 1e-6)


def _save_spectrum_plot(
    out_path: Path,
    freqs_hz: np.ndarray,
    power: np.ndarray,
    peak_indices: list[int],
    min_freq_hz: float,
    max_freq_hz: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs_hz, power, lw=1.2)
    for rank, idx in enumerate(peak_indices, start=1):
        freq = float(freqs_hz[idx])
        ax.axvline(freq, color="tab:red", lw=0.8, alpha=0.55)
        ax.scatter([freq], [float(power[idx])], s=22, color="tab:red")
        ax.text(freq, float(power[idx]), f" {rank}: {freq:.3f} Hz", fontsize=8, va="bottom")
    ax.set_title("Global power spectrum with candidate peaks")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean image-plane amplitude")
    ax.set_xlim(left=max(0.0, min_freq_hz))
    if max_freq_hz is not None:
        ax.set_xlim(right=max_freq_hz)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_mode_preview(
    out_path: Path,
    frame_ref: np.ndarray,
    mask: np.ndarray | None,
    U_slice: np.ndarray,
    V_slice: np.ndarray,
    freq_hz: float,
    power: float,
    percentile: float,
) -> None:
    hi = _mode_display_scale(U_slice, V_slice, mask, percentile)
    amp = amplitude_map(U_slice, V_slice)
    amp_img = _normalize01(amp, hi=hi)
    u_rgb = _phase_hsv(U_slice, 0.0, hi)
    v_rgb = _phase_hsv(V_slice, 0.0, hi)
    if mask is not None:
        mask3 = mask[..., None].astype(np.float32)
        amp_img = amp_img * mask.astype(np.float32)
        u_rgb = u_rgb * mask3
        v_rgb = v_rgb * mask3

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].imshow(frame_ref, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Reference frame")
    axes[0, 1].imshow(amp_img, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title(f"Amplitude, p{percentile:g} scale")
    axes[1, 0].imshow(u_rgb)
    axes[1, 0].set_title("U phase hue, magnitude value")
    axes[1, 1].imshow(v_rgb)
    axes[1, 1].set_title("V phase hue, magnitude value")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Candidate mode at {freq_hz:.4f} Hz, P={power:.6g}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = load_analysis_cache(args.cache_dir)
    video_metadata = cache.metadata["video"]
    analysis_metadata = cache.metadata["analysis"]
    sources = cache.metadata["sources"]
    smoothing_metadata = analysis_metadata["smoothing"]

    peak_indices = _choose_candidate_peaks(
        freqs_hz=cache.freqs_hz,
        power=cache.power_spectrum,
        num_peaks=args.num_peaks,
        min_freq_hz=args.min_freq_hz,
        max_freq_hz=args.max_freq_hz,
        peak_window_hz=args.peak_window_hz,
    )

    _save_spectrum_plot(
        out_dir / "spectrum.png",
        freqs_hz=cache.freqs_hz,
        power=cache.power_spectrum,
        peak_indices=peak_indices,
        min_freq_hz=args.min_freq_hz,
        max_freq_hz=args.max_freq_hz,
    )

    peaks = []
    for rank, idx in enumerate(peak_indices, start=1):
        freq = float(cache.freqs_hz[idx])
        power = float(cache.power_spectrum[idx])
        preview_name = f"mode_{rank:03d}_{freq:.4f}hz.png"
        _save_mode_preview(
            out_dir / preview_name,
            frame_ref=cache.reference_frame,
            mask=cache.mask,
            U_slice=cache.spectrum_u[idx],
            V_slice=cache.spectrum_v[idx],
            freq_hz=freq,
            power=power,
            percentile=args.preview_percentile,
        )
        peaks.append(
            {
                "rank": int(rank),
                "bin_index": int(idx),
                "freq_hz": freq,
                "power": power,
                "preview_png": preview_name,
            }
        )

    peak_freqs = [p["freq_hz"] for p in peaks]
    payload = {
        "selected_peaks_hz": peak_freqs,
        "peaks": peaks,
        "source_video": str(sources["video"]["path"]),
        "source_mask": None if sources["mask"] is None else str(sources["mask"]["path"]),
        "t0": float(video_metadata["frame_range"]["t0_s"]),
        "t1": video_metadata["frame_range"]["t1_s"],
        "resize": video_metadata["resize_max_side"],
        "num_peaks": int(args.num_peaks),
        "min_freq_hz": float(args.min_freq_hz),
        "max_freq_hz": None if args.max_freq_hz is None else float(args.max_freq_hz),
        "peak_window_hz": float(args.peak_window_hz),
        "flow_method": str(analysis_metadata["flow_method"]),
        "no_smooth": bool(smoothing_metadata["disabled"]),
        "sigma_b": float(smoothing_metadata["sigma_b"]),
        "sigma_c": float(smoothing_metadata["sigma_c"]),
        "analysis_mask_dilate_iters": int(smoothing_metadata["analysis_mask_dilate_iters"]),
    }
    with (out_dir / "top_peaks.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    np.savez_compressed(
        out_dir / "spectrum_data.npz",
        freqs_hz=cache.freqs_hz.astype(np.float32, copy=False),
        power_spectrum=cache.power_spectrum.astype(np.float32, copy=False),
        top_peak_indices=np.asarray(peak_indices, dtype=np.int32),
        top_peak_freqs_hz=np.asarray(peak_freqs, dtype=np.float32),
        top_peak_power=np.asarray([p["power"] for p in peaks], dtype=np.float32),
    )

    print(f"Saved inspection outputs -> {out_dir}")
    print(f"Candidate peaks: {[round(float(f), 6) for f in peak_freqs]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Non-GUI modal peak inspection.")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
