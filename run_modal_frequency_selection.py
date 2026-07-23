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
            "Select a compact shared frequency set from cached multi-view optical "
            "flow using grouped complex exact-DFT modes and greedy residual reduction."
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
        "--observation-topology",
        required=True,
        help=(
            "Standalone Gaussian observation topology used to define the "
            "candidate ROI."
        ),
    )
    parser.add_argument(
        "--modal-frame-map",
        required=True,
        help="Version-1 modal_frame_map.json for the full videos.",
    )
    parser.add_argument("--out-dir", required=True, help="New immutable output directory.")
    parser.add_argument(
        "--min-freq-hz",
        type=_positive_finite_float,
        required=True,
        help="Inclusive lower bound of the exact-DFT candidate grid.",
    )
    parser.add_argument(
        "--max-freq-hz",
        type=_positive_finite_float,
        required=True,
        help="Inclusive upper bound of the exact-DFT candidate grid.",
    )
    parser.add_argument(
        "--frequency-step-hz",
        type=_positive_finite_float,
        required=True,
        help="Positive spacing of the exact-DFT candidate grid.",
    )
    parser.add_argument(
        "--mode-counts",
        type=_positive_int,
        nargs="+",
        required=True,
        metavar="K",
        help="Strictly increasing complex-mode counts to report.",
    )
    parser.add_argument(
        "--pixel-stride",
        type=_positive_int,
        default=2,
        help=(
            "Expected interior sampling-grid stride of the topology's candidate pixels; "
            "this validates the existing ROI and does not resample it."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_freq_hz <= args.min_freq_hz:
        raise ValueError("--max-freq-hz must be greater than --min-freq-hz")
    if any(
        right <= left
        for left, right in zip(args.mode_counts, args.mode_counts[1:])
    ):
        raise ValueError("--mode-counts must be strictly increasing and unique")

    seen_view_ids: set[str] = set()
    for spec in args.flow_caches:
        if "=" not in spec:
            raise ValueError(f"--flow-caches value must be VIEW_ID=PATH, got {spec!r}")
        view_id, cache_path_text = spec.split("=", 1)
        if not view_id or view_id.strip() != view_id:
            raise ValueError(
                f"--flow-caches contains an invalid view ID in {spec!r}"
            )
        if view_id in seen_view_ids:
            raise ValueError(f"--flow-caches repeats view ID {view_id!r}")
        seen_view_ids.add(view_id)
        if not cache_path_text:
            raise ValueError(f"--flow-caches has an empty path for {view_id!r}")
        cache_path = Path(cache_path_text).expanduser()
        if not cache_path.is_dir():
            raise ValueError(f"Flow cache directory does not exist: {cache_path}")

    for argument_name in ("observation_topology", "modal_frame_map"):
        path = Path(getattr(args, argument_name)).expanduser()
        if not path.is_file():
            option_name = argument_name.replace("_", "-")
            raise ValueError(f"--{option_name} does not exist: {path}")


def run(args: argparse.Namespace) -> None:
    _validate_args(args)

    from flow3d.modal_frequency_selection import run_modal_frequency_selection

    result = run_modal_frequency_selection(
        flow_cache_specs=args.flow_caches,
        observation_topology_path=args.observation_topology,
        modal_frame_map_path=args.modal_frame_map,
        output_dir=args.out_dir,
        min_freq_hz=args.min_freq_hz,
        max_freq_hz=args.max_freq_hz,
        frequency_step_hz=args.frequency_step_hz,
        mode_counts=args.mode_counts,
        pixel_stride=args.pixel_stride,
    )
    print(f"Saved modal frequency selection -> {result.path}")
    print(f"  summary: {result.summary_path}")
    print(f"  diagnostics: {result.diagnostics_path}")
    print(f"  plot: {result.plot_path}")


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
