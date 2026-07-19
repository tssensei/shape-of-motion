from __future__ import annotations

import argparse
import json

from flow3d.modal_basis_diagnostics import run_modal_basis_diagnostics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how much cached reference-frame optical flow is represented by "
            "the current direct 2D and lifted 3D modal bases, with a same-rank PCA upper bound."
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
        "--modal-npzs",
        nargs="+",
        required=True,
        metavar="VIEW_ID=PATH",
        help="Exact ordered 2D modal NPZ files recorded by the Gaussian manifest.",
    )
    parser.add_argument(
        "--modal-manifest",
        required=True,
        help="Gaussian modal_modes_manifest.json containing fixed phi and observations.",
    )
    parser.add_argument(
        "--modal-frame-map",
        required=True,
        help="Version-1 modal_frame_map.json for the full videos.",
    )
    parser.add_argument(
        "--dynamic-data-dir",
        required=True,
        help="Prepared dynamic RGB dataset used for the photometric flow-warp check.",
    )
    parser.add_argument("--out-dir", required=True, help="New immutable output directory.")
    parser.add_argument(
        "--max-real-rank",
        type=int,
        default=20,
        help="Largest shared real rank; must not exceed twice the manifest mode count.",
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=2,
        help="Integer sampling stride for the full analysis-mask comparison.",
    )
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument(
        "--warp-frame-count",
        type=int,
        default=32,
        help="Deterministically sampled non-reference frames per view for the warp check.",
    )
    parser.add_argument(
        "--production-ridge-relative",
        type=float,
        default=1e-4,
        help="Pair-normalized ridge used for the separate solver-style full-basis score.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    result = run_modal_basis_diagnostics(
        flow_caches=args.flow_caches,
        modal_npzs=args.modal_npzs,
        modal_manifest=args.modal_manifest,
        modal_frame_map=args.modal_frame_map,
        dynamic_data_dir=args.dynamic_data_dir,
        out_dir=args.out_dir,
        max_real_rank=args.max_real_rank,
        pixel_stride=args.pixel_stride,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        warp_frame_count=args.warp_frame_count,
        production_ridge_relative=args.production_ridge_relative,
    )
    with result.summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    endpoint = summary["overall"]["candidate_endpoint_gaps"]
    candidate = summary["overall"]["candidate_pooled_r2"]
    final_rank = summary["real_ranks"][-1]
    print(f"Saved modal basis capacity diagnostics -> {result.path}")
    print(f"Candidate ROI pooled R2 at real rank {final_rank}:")
    for method in ("pca_upper_bound", "direct_2d", "lifted_3d"):
        print(f"  {method}: {float(candidate[method][-1]):.6g}")
    print(
        "Endpoint gaps: "
        f"PCA-2D={float(endpoint['pca_minus_direct_2d']):.6g}, "
        f"2D-lifted={float(endpoint['direct_2d_minus_lifted_3d']):.6g}"
    )
    for view in summary["views"]:
        warp_ratio = view["warp"]["warp_to_zero_mae_ratio"]
        warp_text = "null" if warp_ratio is None else f"{float(warp_ratio):.6g}"
        print(
            f"  {view['view_id']}: candidate_pixels={int(view['candidate_pixel_count'])}, "
            f"warp/zero MAE={warp_text}"
        )


def main(argv: list[str] | None = None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
