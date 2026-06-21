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
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
