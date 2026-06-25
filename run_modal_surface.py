from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Surface-based latent 3D modal field tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_packet = sub.add_parser("make-packet", help="Build a single-view surface modal packet.")
    p_packet.add_argument("--view-config", required=True, help="View JSON config path.")
    p_packet.add_argument("--modal-npz", required=True, help="modal_analysis.npz from modal peak-pick export.")
    p_packet.add_argument("--out", required=True, help="Output surface packet .npz path.")
    p_packet.add_argument("--stride", type=int, default=4, help="Pixel sampling stride.")
    p_packet.add_argument("--mask-erode-iters", type=int, default=2, help="3x3 mask erosion iterations.")
    p_packet.add_argument("--depth-edge-tau", type=float, default=0.12, help="Depth local range threshold.")
    p_packet.add_argument("--min-amplitude-percentile", type=float, default=5.0, help="Drop points below this max-amplitude percentile.")

    p_match = sub.add_parser("match-two-views", help="Match view1 surface points into view2 with depth visibility.")
    p_match.add_argument("--view1-packet", required=True, help="View 1 surface packet .npz path.")
    p_match.add_argument("--view2-config", required=True, help="View 2 JSON config path.")
    p_match.add_argument("--view2-modal-npz", required=True, help="View 2 modal_analysis.npz path.")
    p_match.add_argument("--out", required=True, help="Output matches .npz path.")
    p_match.add_argument("--mode-index", type=int, default=0, help="Selected modal frequency index.")
    p_match.add_argument("--mask-erode-iters", type=int, default=2, help="View 2 mask erosion iterations.")
    p_match.add_argument("--depth-tau", type=float, default=0.08, help="Relative depth consistency threshold.")
    p_match.add_argument("--edge-tau", type=float, default=0.12, help="Relative local depth range threshold.")
    p_match.add_argument("--depth-window-radius", type=int, default=2, help="Depth consistency window radius.")
    p_match.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")

    p_opt = sub.add_parser("optimize-two-view", help="Optimize latent 3D modal displacement from two-view matches.")
    p_opt.add_argument("--matches", required=True, help="matches_12.npz path.")
    p_opt.add_argument("--out", required=True, help="Output latent_field_12.npz path.")
    p_opt.add_argument("--vis-dir", default=None, help="Optional visualization directory.")
    p_opt.add_argument("--iterations", type=int, default=8, help="Alternating optimization iterations.")
    p_opt.add_argument("--ridge-mu", type=float, default=1e-4, help="Per-point ridge regularization.")
    p_opt.add_argument("--outlier-frac", type=float, default=0.05, help="Fraction of worst residual points dropped per iteration.")

    p_match_multi = sub.add_parser("match-multi-views", help="Build an N-view modal observation graph from surface packets.")
    p_match_multi.add_argument("--surface-packets", nargs="+", required=True, help="Surface packet .npz paths from all views.")
    p_match_multi.add_argument("--out", required=True, help="Output observation graph .npz path.")
    p_match_multi.add_argument("--mode-index", type=int, default=0, help="Selected modal frequency index.")
    p_match_multi.add_argument("--merge-voxel-size", type=float, default=0.03, help="World-space voxel size used to merge packet points.")
    p_match_multi.add_argument("--min-observations", type=int, default=2, help="Minimum observations per pooled surface point.")
    p_match_multi.add_argument("--mask-erode-iters", type=int, default=2, help="Target view mask erosion iterations.")
    p_match_multi.add_argument("--depth-tau", type=float, default=0.08, help="Relative depth consistency threshold.")
    p_match_multi.add_argument("--edge-tau", type=float, default=0.12, help="Relative local depth range threshold.")
    p_match_multi.add_argument("--depth-window-radius", type=int, default=2, help="Depth consistency window radius.")
    p_match_multi.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")

    p_match_carrier = sub.add_parser("match-carrier-views", help="Build an N-view observation graph from VGGT carrier points.")
    p_match_carrier.add_argument("--carrier-points", required=True, help="VGGT carrier points .npz path.")
    p_match_carrier.add_argument("--view-config", action="append", required=True, help="View JSON config path. Repeat per view.")
    p_match_carrier.add_argument("--modal-npz", action="append", required=True, help="modal_analysis.npz path. Repeat per view.")
    p_match_carrier.add_argument("--out", required=True, help="Output observation graph .npz path.")
    p_match_carrier.add_argument("--mode-index", type=int, default=0, help="Selected modal frequency index.")
    p_match_carrier.add_argument("--mask-erode-iters", type=int, default=1, help="3x3 modal mask erosion iterations.")
    p_match_carrier.add_argument("--zbuffer-radius", type=int, default=5, help="Local robust z-buffer window radius in pixels.")
    p_match_carrier.add_argument("--front-percentile", type=float, default=10.0, help="Local depth percentile treated as front surface.")
    p_match_carrier.add_argument("--zbuffer-tau", type=float, default=0.05, help="Relative depth threshold against local front depth.")
    p_match_carrier.add_argument("--min-zbuffer-samples", type=int, default=5, help="Minimum local carrier depths for visibility.")
    p_match_carrier.add_argument("--min-observations", type=int, default=2, help="Minimum observed views per carrier point.")
    p_match_carrier.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")

    p_opt_multi = sub.add_parser("optimize-multi-view", help="Optimize latent 3D modal displacement from N-view observations.")
    p_opt_multi.add_argument("--observations", required=True, help="Multi-view observation graph .npz path.")
    p_opt_multi.add_argument("--out", required=True, help="Output latent_field multi-view .npz path.")
    p_opt_multi.add_argument("--vis-dir", default=None, help="Optional visualization directory.")
    p_opt_multi.add_argument("--iterations", type=int, default=8, help="Alternating optimization iterations.")
    p_opt_multi.add_argument("--ridge-mu", type=float, default=1e-4, help="Per-point ridge regularization.")
    p_opt_multi.add_argument("--outlier-frac", type=float, default=0.05, help="Fraction of worst residual points dropped per iteration.")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "make-packet":
        from modal_surface.apps import make_packet

        make_packet.run(args)
        return
    if args.command == "match-two-views":
        from modal_surface.apps import match_two_views

        match_two_views.run(args)
        return
    if args.command == "optimize-two-view":
        from modal_surface.apps import optimize_two_view

        optimize_two_view.run(args)
        return
    if args.command == "match-multi-views":
        from modal_surface.apps import match_multi_views

        match_multi_views.run(args)
        return
    if args.command == "match-carrier-views":
        from modal_surface.apps import match_carrier_views

        match_carrier_views.run(args)
        return
    if args.command == "optimize-multi-view":
        from modal_surface.apps import optimize_multi_view

        optimize_multi_view.run(args)
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
