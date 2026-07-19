from __future__ import annotations

import argparse
import math
from pathlib import Path


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def _positive_finite_float(value: str) -> float:
    result = _finite_float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze cached multi-view optical flow with per-view windowed "
            "FDD/SPOD diagnostics."
        )
    )
    parser.add_argument(
        "--flow-caches",
        nargs="+",
        required=True,
        metavar="VIEW_ID=PATH",
        help="Ordered view-to-cache mappings; order must exactly match the frame map.",
    )
    parser.add_argument(
        "--modal-manifest",
        required=True,
        help="Current Gaussian modal manifest used to define the candidate ROI.",
    )
    parser.add_argument(
        "--modal-frame-map",
        required=True,
        help="Version-1 modal_frame_map.json for the full videos.",
    )
    parser.add_argument("--out-dir", required=True, help="New immutable output directory.")
    parser.add_argument(
        "--window-lengths",
        type=_positive_int,
        nargs="+",
        default=(512, 256),
        metavar="FRAMES",
        help=(
            "Exactly two temporal window lengths, longer primary first and shorter "
            "stability window second (default: 512 256)."
        ),
    )
    parser.add_argument(
        "--overlap-fraction",
        type=_finite_float,
        default=0.75,
        help="Shared fractional window overlap in [0,1) (default: 0.75).",
    )
    parser.add_argument(
        "--min-freq-hz",
        type=_positive_finite_float,
        default=0.05,
        help="Inclusive lower frequency bound in Hz (default: 0.05).",
    )
    parser.add_argument(
        "--max-freq-hz",
        type=_positive_finite_float,
        default=2.2,
        help="Inclusive upper frequency bound in Hz (default: 2.2).",
    )
    parser.add_argument(
        "--pixel-stride",
        type=_positive_int,
        default=2,
        help=(
            "Expected interior sampling-grid stride of the manifest candidate pixels "
            "(default: 2)."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_freq_hz <= args.min_freq_hz:
        raise ValueError("--max-freq-hz must be greater than --min-freq-hz")
    if args.overlap_fraction < 0.0 or args.overlap_fraction >= 1.0:
        raise ValueError("--overlap-fraction must be in [0,1)")
    if len(args.window_lengths) != 2:
        raise ValueError("--window-lengths requires exactly two values")
    if args.window_lengths[0] <= args.window_lengths[1]:
        raise ValueError(
            "--window-lengths must list the longer primary window first"
        )
    for window_length in args.window_lengths:
        hop_float = window_length * (1.0 - args.overlap_fraction)
        hop = int(round(hop_float))
        if hop <= 0 or not math.isclose(hop_float, hop, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Each window length times (1 - overlap) must be a positive integer"
            )

    output_path = Path(args.out_dir).expanduser()
    if output_path.exists() or output_path.is_symlink():
        raise ValueError(f"--out-dir already exists: {output_path}")

    seen_view_ids: set[str] = set()
    for spec in args.flow_caches:
        if "=" not in spec:
            raise ValueError(f"--flow-caches value must be VIEW_ID=PATH, got {spec!r}")
        view_id, cache_path_text = spec.split("=", 1)
        if not view_id or view_id.strip() != view_id:
            raise ValueError(f"--flow-caches contains an invalid view ID in {spec!r}")
        if view_id in seen_view_ids:
            raise ValueError(f"--flow-caches repeats view ID {view_id!r}")
        seen_view_ids.add(view_id)
        if not cache_path_text:
            raise ValueError(f"--flow-caches has an empty path for {view_id!r}")
        cache_path = Path(cache_path_text).expanduser()
        if not cache_path.is_dir():
            raise ValueError(f"Flow cache directory does not exist: {cache_path}")

    for argument_name in ("modal_manifest", "modal_frame_map"):
        path = Path(getattr(args, argument_name)).expanduser()
        if not path.is_file():
            option_name = argument_name.replace("_", "-")
            raise ValueError(f"--{option_name} does not exist: {path}")


def run(args: argparse.Namespace) -> None:
    _validate_args(args)

    from flow3d.modal_spod_analysis import run_modal_spod_analysis

    result = run_modal_spod_analysis(
        flow_cache_specs=args.flow_caches,
        modal_manifest_path=args.modal_manifest,
        modal_frame_map_path=args.modal_frame_map,
        output_dir=args.out_dir,
        window_lengths=args.window_lengths,
        overlap_fraction=args.overlap_fraction,
        min_freq_hz=args.min_freq_hz,
        max_freq_hz=args.max_freq_hz,
        pixel_stride=args.pixel_stride,
    )
    print(f"Saved modal SPOD analysis -> {result.path}")
    print(f"  summary: {result.summary_path}")
    print(f"  diagnostics: {result.diagnostics_path}")
    print(f"  frequency bands: {result.bands_path}")
    print(f"  spectra plot: {result.spectra_plot_path}")
    print(f"  stability plot: {result.stability_plot_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    run(args)


if __name__ == "__main__":
    main()
