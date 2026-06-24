from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from modal_peak_pick.core.pipeline import parse_freqs, run_modal_analysis_from_video, select_mode_slice


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--out", default="outputs/modal_analysis.npz", help="Output .npz path.")
    parser.add_argument("--freqs", default=None, help="Comma-separated selected frequencies in Hz.")
    parser.add_argument("--peaks-json", default=None, help="JSON file with selected_peaks_hz.")
    parser.add_argument("--mask", default=None, help="Optional binary ROI mask path (.npy or image).")
    parser.add_argument("--t0", type=float, default=0.0, help="Clip start time in seconds.")
    parser.add_argument("--t1", type=float, default=None, help="Clip end time in seconds.")
    parser.add_argument("--resize", type=int, default=None, help="Resize max(H,W) before analysis.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum decoded frames.")
    parser.add_argument("--flow-method", choices=["farneback", "tvl1"], default="farneback")
    parser.add_argument("--no-smooth", action="store_true", help="Disable contrast-weighted flow smoothing.")
    parser.add_argument("--sigma-b", type=float, default=3.0, help="Spatial smoothing sigma.")
    parser.add_argument("--sigma-c", type=float, default=0.0, help="Reference pre-blur sigma.")
    parser.add_argument("--analysis-mask-dilate-iters", type=int, default=0, help="Dilate the analysis mask with a 3x3 kernel before flow smoothing and spectrum computation.")


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


def run(args: argparse.Namespace) -> None:
    requested_freqs = _read_requested_freqs(args.freqs, args.peaks_json)
    result = run_modal_analysis_from_video(
        video_path=args.video,
        t0=args.t0,
        t1=args.t1,
        resize=args.resize,
        max_frames=args.max_frames,
        flow_method=args.flow_method,
        no_smooth=args.no_smooth,
        sigma_b=args.sigma_b,
        sigma_c=args.sigma_c,
        mask_path=args.mask,
        analysis_mask_dilate_iters=getattr(args, "analysis_mask_dilate_iters", 0),
    )

    modes_u = []
    modes_v = []
    selected_freqs = []
    for idx, freq in enumerate(requested_freqs, start=1):
        mode = select_mode_slice(result, freq, mode_idx=idx)
        modes_u.append(mode.U_slice)
        modes_v.append(mode.V_slice)
        selected_freqs.append(mode.freq_selected_hz)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_arr = np.zeros((0, 0), dtype=np.uint8) if result.mask is None else result.mask.astype(np.uint8)
    np.savez_compressed(
        out_path,
        freqs_hz=result.freqs_hz.astype(np.float32, copy=False),
        power_spectrum=result.power_spectrum.astype(np.float32, copy=False),
        mode_u=np.stack(modes_u, axis=0).astype(np.complex64, copy=False),
        mode_v=np.stack(modes_v, axis=0).astype(np.complex64, copy=False),
        mask=mask_arr,
        has_mask=np.array(result.mask is not None, dtype=np.uint8),
        reference_frame=result.frame_ref.astype(np.float32, copy=False),
        fps=np.array(result.fps, dtype=np.float32),
        t0=np.array(args.t0, dtype=np.float32),
        t1=np.array(-1.0 if args.t1 is None else args.t1, dtype=np.float32),
        resize=_opt_int(args.resize),
        source_video=np.array(str(args.video)),
        source_mask=_opt_text(args.mask),
        requested_freqs_hz=np.asarray(requested_freqs, dtype=np.float32),
        selected_freqs_hz=np.asarray(selected_freqs, dtype=np.float32),
        frequency_method=np.array("exact_dft"),
        flow_method=np.array(str(args.flow_method)),
        no_smooth=np.array(bool(args.no_smooth), dtype=np.uint8),
        sigma_b=np.array(args.sigma_b, dtype=np.float32),
        sigma_c=np.array(args.sigma_c, dtype=np.float32),
        t_ref_s=np.array(result.t_ref_s, dtype=np.float32),
    )
    print(f"Saved modal analysis -> {out_path}")
    print(f"Selected frequencies: {[round(float(f), 6) for f in selected_freqs]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export selected 2D complex modal slices.")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
