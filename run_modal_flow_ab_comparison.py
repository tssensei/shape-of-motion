from __future__ import annotations

import argparse
from pathlib import Path

from flow3d.modal_flow_ab_comparison import run_modal_flow_ab_comparison


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current stabilized Farneback caches against raw-video caches "
            "built with direct-to-reference background motion compensation."
        )
    )
    parser.add_argument(
        "--stabilized-caches",
        nargs="+",
        required=True,
        metavar="VIEW_ID=PATH",
    )
    parser.add_argument(
        "--compensated-caches",
        nargs="+",
        required=True,
        metavar="VIEW_ID=PATH",
    )
    parser.add_argument("--modal-manifest", required=True)
    parser.add_argument("--modal-frame-map", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pixel-stride", type=_positive_int, default=2)
    parser.add_argument("--grid-rows", type=_positive_int, default=4)
    parser.add_argument("--grid-cols", type=_positive_int, default=4)
    parser.add_argument(
        "--background-mask-dilate-px", type=_positive_int, default=16
    )
    return parser


def _validate_paths(args: argparse.Namespace) -> None:
    for option in ("modal_manifest", "modal_frame_map"):
        path = Path(getattr(args, option)).expanduser()
        if not path.is_file():
            raise ValueError(f"--{option.replace('_', '-')} does not exist: {path}")
    output = Path(args.out_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError(f"--out-dir already exists: {output}")


def run(args: argparse.Namespace) -> None:
    _validate_paths(args)
    result = run_modal_flow_ab_comparison(
        stabilized_cache_specs=args.stabilized_caches,
        compensated_cache_specs=args.compensated_caches,
        modal_manifest_path=args.modal_manifest,
        modal_frame_map_path=args.modal_frame_map,
        output_dir=args.out_dir,
        pixel_stride=args.pixel_stride,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        background_mask_dilate_px=args.background_mask_dilate_px,
    )
    print(f"Saved stabilized-vs-compensated flow comparison -> {result.path}")
    print(f"  summary: {result.summary_path}")
    print(f"  diagnostics: {result.diagnostics_path}")
    print(f"  spectra: {result.spectrum_plot_path}")
    print(f"  flow energy: {result.flow_plot_path}")
    print(f"  regional: {result.regional_plot_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        _validate_paths(args)
    except ValueError as error:
        parser.error(str(error))
    run(args)


if __name__ == "__main__":
    main()

