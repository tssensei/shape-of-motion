from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VGGT-carrier latent 3D modal field tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_match_carrier = sub.add_parser("match-carrier-views", help="Build an N-view observation graph from VGGT carrier points.")
    p_match_carrier.add_argument("--carrier-points", required=True, help="VGGT carrier points .npz path.")
    p_match_carrier.add_argument("--view-config", action="append", required=True, help="View JSON config path. Repeat per view.")
    p_match_carrier.add_argument("--modal-npz", action="append", required=True, help="modal_analysis.npz path. Repeat per view.")
    p_match_carrier.add_argument("--out", required=True, help="Output observation graph .npz path.")
    p_match_carrier.add_argument("--mode-index", type=int, default=0, help="Selected modal frequency index.")
    p_match_carrier.add_argument("--mask-erode-iters", type=int, default=1, help="3x3 modal mask erosion iterations.")
    p_match_carrier.add_argument("--source-mask-erode-iters", type=int, default=1, help="3x3 erosion iterations for filtering VGGT carrier points by their source-view mask.")
    p_match_carrier.add_argument("--zbuffer-radius", type=int, default=5, help="Local robust z-buffer window radius in pixels.")
    p_match_carrier.add_argument("--front-percentile", type=float, default=10.0, help="Local depth percentile treated as front surface.")
    p_match_carrier.add_argument("--zbuffer-tau", type=float, default=0.05, help="Relative depth threshold against local front depth.")
    p_match_carrier.add_argument("--min-zbuffer-samples", type=int, default=5, help="Minimum local carrier depths for visibility.")
    p_match_carrier.add_argument(
        "--min-observations",
        type=int,
        default=1,
        help="Minimum observed views per carrier point. Default keeps view-local carrier observations.",
    )
    p_match_carrier.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")

    p_opt_multi = sub.add_parser("optimize-multi-view", help="Optimize latent 3D modal displacement from N-view observations.")
    p_opt_multi.add_argument("--observations", required=True, help="Multi-view observation graph .npz path.")
    p_opt_multi.add_argument("--out", required=True, help="Output latent_field multi-view .npz path.")
    p_opt_multi.add_argument("--vis-dir", default=None, help="Optional visualization directory.")
    p_opt_multi.add_argument("--iterations", type=int, default=8, help="Alternating optimization iterations.")
    p_opt_multi.add_argument("--ridge-mu", type=float, default=1e-4, help="Per-point ridge regularization.")
    p_opt_multi.add_argument("--outlier-frac", type=float, default=0.05, help="Fraction of worst residual points dropped per iteration.")
    p_opt_multi.add_argument("--single-view-smooth-lambda", type=float, default=0.0, help="Anchor-prior weight for refining points observed by only one view.")
    p_opt_multi.add_argument("--single-view-smooth-k", type=int, default=8, help="Number of reliable anchor neighbors used for single-view refinement.")
    p_opt_multi.add_argument(
        "--single-view-anchor-min-observations",
        type=int,
        default=2,
        help="Minimum observation count for points used as single-view smoothing anchors.",
    )

    p_solve = sub.add_parser("solve-carrier-modes", help="Batch solve VGGT carrier modal fields for multiple mode indices.")
    p_solve.add_argument("--carrier-points", required=True, help="VGGT carrier points .npz path.")
    p_solve.add_argument("--view-config", action="append", required=True, help="View JSON config path. Repeat per view.")
    p_solve.add_argument("--modal-npz", action="append", required=True, help="modal_analysis.npz path. Repeat per view.")
    p_solve.add_argument("--out-dir", required=True, help="Output directory for observations, latents, vis, and manifest.")
    p_solve.add_argument("--mode-indices", default="all", help="Comma-separated zero-based mode indices, or 'all'.")
    p_solve.add_argument("--mask-erode-iters", type=int, default=1, help="3x3 modal mask erosion iterations.")
    p_solve.add_argument("--source-mask-erode-iters", type=int, default=1, help="3x3 erosion iterations for filtering VGGT carrier points by their source-view mask.")
    p_solve.add_argument("--zbuffer-radius", type=int, default=5, help="Local robust z-buffer window radius in pixels.")
    p_solve.add_argument("--front-percentile", type=float, default=10.0, help="Local depth percentile treated as front surface.")
    p_solve.add_argument("--zbuffer-tau", type=float, default=0.05, help="Relative depth threshold against local front depth.")
    p_solve.add_argument("--min-zbuffer-samples", type=int, default=5, help="Minimum local carrier depths for visibility.")
    p_solve.add_argument("--min-observations", type=int, default=1, help="Minimum observed views per carrier point.")
    p_solve.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")
    p_solve.add_argument("--iterations", type=int, default=8, help="Alternating optimization iterations.")
    p_solve.add_argument("--ridge-mu", type=float, default=1e-4, help="Per-point ridge regularization.")
    p_solve.add_argument("--outlier-frac", type=float, default=0.0, help="Fraction of worst residual points dropped per iteration.")
    p_solve.add_argument("--single-view-smooth-lambda", type=float, default=0.0, help="Anchor-prior weight for refining points observed by only one view.")
    p_solve.add_argument("--single-view-smooth-k", type=int, default=8, help="Number of reliable anchor neighbors used for single-view refinement.")
    p_solve.add_argument(
        "--single-view-anchor-min-observations",
        type=int,
        default=2,
        help="Minimum observation count for points used as single-view smoothing anchors.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "match-carrier-views":
        from modal_surface.apps import match_carrier_views

        match_carrier_views.run(args)
        return
    if args.command == "optimize-multi-view":
        from modal_surface.apps import optimize_multi_view

        optimize_multi_view.run(args)
        return
    if args.command == "solve-carrier-modes":
        from modal_surface.apps import solve_carrier_modes

        solve_carrier_modes.run(args)
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
