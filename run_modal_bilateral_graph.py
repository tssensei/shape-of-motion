from __future__ import annotations

import argparse
import math
from pathlib import Path


def _positive_integer(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _positive_finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return result


def _unit_interval_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise argparse.ArgumentTypeError("must be finite and in [0,1]")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and visualize a fixed bilateral foreground-Gaussian graph for "
            "subsequent rigidity experiments."
        )
    )
    parser.add_argument(
        "--input-ckpt",
        required=True,
        help="Static SceneModel checkpoint supplying canonical foreground Gaussians.",
    )
    parser.add_argument("--out-dir", required=True, help="New immutable output directory.")
    parser.add_argument(
        "--spatial-preselect-k",
        type=_positive_integer,
        default=32,
        help="Spatial nearest-neighbor candidates per Gaussian (default: 32).",
    )
    parser.add_argument(
        "--bilateral-k",
        type=_positive_integer,
        default=8,
        help="Strongest bilateral candidates retained before mutual filtering (default: 8).",
    )
    parser.add_argument(
        "--max-distance",
        type=_positive_finite_float,
        default=0.008,
        help="Maximum canonical center distance in scene units (default: 0.008).",
    )
    parser.add_argument(
        "--spatial-sigma",
        type=_positive_finite_float,
        default=1.0,
        help="Gaussian sigma for scale-normalized center distance (default: 1.0).",
    )
    parser.add_argument(
        "--color-sigma",
        type=_positive_finite_float,
        default=0.15,
        help="Gaussian sigma for activated-RGB distance (default: 0.15).",
    )
    parser.add_argument(
        "--color-floor",
        type=_unit_interval_float,
        default=0.05,
        help="Minimum multiplicative RGB affinity (default: 0.05).",
    )
    parser.add_argument(
        "--scale-epsilon",
        type=_positive_finite_float,
        default=1e-8,
        help="Lower bound for pair Gaussian radius normalization (default: 1e-8).",
    )
    parser.add_argument(
        "--visualization-max-edges",
        type=_positive_integer,
        default=50_000,
        help="Maximum deterministically rank-sampled edges drawn in the preview.",
    )
    parser.add_argument(
        "--visualization-max-points",
        type=_positive_integer,
        default=50_000,
        help="Maximum deterministically sampled Gaussian centers drawn in the preview.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.bilateral_k > args.spatial_preselect_k:
        raise ValueError("--bilateral-k cannot exceed --spatial-preselect-k")
    checkpoint = Path(args.input_ckpt).expanduser()
    if not checkpoint.is_file():
        raise ValueError(f"--input-ckpt does not exist: {checkpoint}")
    output = Path(args.out_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError(f"--out-dir already exists: {output}")


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    from flow3d.modal_bilateral_graph import build_modal_bilateral_graph

    result = build_modal_bilateral_graph(
        checkpoint_path=args.input_ckpt,
        output_dir=args.out_dir,
        spatial_preselect_k=args.spatial_preselect_k,
        bilateral_k=args.bilateral_k,
        max_distance=args.max_distance,
        spatial_sigma=args.spatial_sigma,
        color_sigma=args.color_sigma,
        color_floor=args.color_floor,
        scale_epsilon=args.scale_epsilon,
        visualization_max_edges=args.visualization_max_edges,
        visualization_max_points=args.visualization_max_points,
    )
    print(f"Saved bilateral Gaussian graph -> {result.path}")
    print(f"  graph: {result.graph_path}")
    print(f"  diagnostics: {result.diagnostics_path}")
    print(f"  edge visualization: {result.edge_visualization_path}")
    print(f"  edge histogram: {result.edge_histogram_path}")


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
