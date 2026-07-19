from __future__ import annotations

import argparse
import math
from pathlib import Path


def _positive_integer(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _unit_interval_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise argparse.ArgumentTypeError("must be finite and in [0,1]")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project a fixed bilateral Gaussian graph into one calibrated camera "
            "view and color edges by affinity."
        )
    )
    parser.add_argument("--graph", required=True, help="Version-1 bilateral graph NPZ.")
    parser.add_argument(
        "--view-config",
        required=True,
        help="Modal view-config JSON supplying intrinsics and world-to-camera pose.",
    )
    parser.add_argument(
        "--background-image",
        default=None,
        help="Optional RGB image with exactly the view-config resolution.",
    )
    parser.add_argument(
        "--min-affinity",
        type=_unit_interval_float,
        required=True,
        help="Only display edges whose combined bilateral affinity reaches this value.",
    )
    parser.add_argument(
        "--max-edges",
        type=_positive_integer,
        default=100_000,
        help="Maximum deterministically sampled projected edges to draw.",
    )
    parser.add_argument("--out", required=True, help="New output PNG path.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("graph", "view_config"):
        path = Path(getattr(args, name)).expanduser()
        if not path.is_file():
            raise ValueError(f"--{name.replace('_', '-')} does not exist: {path}")
    if args.background_image is not None:
        background = Path(args.background_image).expanduser()
        if not background.is_file():
            raise ValueError(f"--background-image does not exist: {background}")
    output = Path(args.out).expanduser()
    if output.suffix.lower() != ".png":
        raise ValueError("--out must use a .png extension")
    if output.exists() or output.is_symlink():
        raise ValueError(f"--out already exists: {output}")


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    from flow3d.modal_bilateral_graph_view import render_modal_bilateral_graph_view

    result = render_modal_bilateral_graph_view(
        graph_path=args.graph,
        view_config_path=args.view_config,
        output_path=args.out,
        minimum_affinity=args.min_affinity,
        maximum_edges=args.max_edges,
        background_image_path=args.background_image,
    )
    print(f"Saved {result.view_id} bilateral graph projection -> {result.path}")
    print(f"  graph edges: {result.graph_edge_count}")
    print(f"  above threshold: {result.threshold_edge_count}")
    print(f"  projected inside view: {result.projected_edge_count}")
    print(f"  displayed: {result.displayed_edge_count}")


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
