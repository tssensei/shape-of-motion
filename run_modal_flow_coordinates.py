from __future__ import annotations

import argparse
import json

from flow3d.modal_flow_coordinates import (
    DIAGNOSTICS_JSON_FILENAME,
    solve_modal_flow_coordinates,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Invert cached reference-frame optical flow into fixed per-frame "
            "complex coordinates using a rendered modal projection design."
        )
    )
    parser.add_argument(
        "--rendered-design",
        required=True,
        help="Rendered modal design artifact or its immutable output directory.",
    )
    parser.add_argument(
        "--modal-frame-map",
        required=True,
        help="Version-1 modal_frame_map.json for all full-length views.",
    )
    parser.add_argument(
        "--flow-caches",
        nargs="+",
        required=True,
        metavar="VIEW_ID=PATH",
        help="Ordered view-to-cache mappings; order must exactly match the frame map.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="New immutable output directory.",
    )
    parser.add_argument(
        "--ridge-relative",
        type=float,
        default=1e-4,
        help="Positive ridge weight after shared real/imag mode-pair normalization.",
    )
    parser.add_argument(
        "--frame-chunk-size",
        type=int,
        default=64,
        help="Number of cached flow frames read and solved at once.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    result = solve_modal_flow_coordinates(
        rendered_design=args.rendered_design,
        modal_frame_map=args.modal_frame_map,
        flow_caches=args.flow_caches,
        out_dir=args.out_dir,
        ridge_relative=args.ridge_relative,
        frame_chunk_size=args.frame_chunk_size,
    )
    diagnostics_path = result.path.parent / DIAGNOSTICS_JSON_FILENAME
    with diagnostics_path.open("r", encoding="utf-8") as file:
        diagnostics = json.load(file)
    print(f"Saved modal flow coordinates -> {result.path.parent}")
    print(
        "Solved "
        f"views={len(result.view_ids)}, frames={len(result.frame_names)}, "
        f"modes={result.mode_indices.size}, ridge={result.ridge_relative:.6g}"
    )
    overall = diagnostics["overall"]
    print(
        "Flow fit: "
        f"RMSE={float(overall['flow_rmse']):.6g}, "
        f"relative={float(overall['relative_residual']):.6g}, "
        f"R2={float(overall['flow_r2']):.6g}"
    )
    for view in diagnostics["views"]:
        condition = view["condition_number"]
        condition_text = "inf" if condition is None else f"{float(condition):.6g}"
        print(
            f"  {view['view_id']}: frames={int(view['frame_count'])}, "
            f"pixels={int(view['unique_pixel_count'])}, "
            f"rank={int(view['numerical_rank'])}/{int(view['column_count'])}, "
            f"condition={condition_text}, R2={float(view['flow_r2']):.6g}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
