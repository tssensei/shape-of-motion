"""CLI adapter for the match-two-views stage."""

from __future__ import annotations

import argparse

from modal_surface.matching import match_two_views


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register match-two-views command-line arguments."""
    parser.add_argument("--view1-packet", required=True, help="View 1 surface packet .npz path.")
    parser.add_argument("--view2-config", required=True, help="View 2 JSON config path.")
    parser.add_argument("--view2-modal-npz", required=True, help="View 2 modal_analysis.npz path.")
    parser.add_argument("--out", required=True, help="Output matches .npz path.")
    parser.add_argument("--mode-index", type=int, default=0, help="Selected modal frequency index.")
    parser.add_argument("--mask-erode-iters", type=int, default=2, help="View 2 mask erosion iterations.")
    parser.add_argument("--depth-tau", type=float, default=0.08, help="Relative depth consistency threshold.")
    parser.add_argument("--edge-tau", type=float, default=0.12, help="Relative local depth range threshold.")
    parser.add_argument("--depth-window-radius", type=int, default=2, help="Depth consistency window radius.")
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")


def run(args: argparse.Namespace) -> None:
    """Run match-two-views from an argparse namespace."""
    out = match_two_views(
        view1_packet_path=args.view1_packet,
        view2_config_path=args.view2_config,
        view2_modal_npz_path=args.view2_modal_npz,
        out_path=args.out,
        mode_index=args.mode_index,
        mask_erode_iters=args.mask_erode_iters,
        depth_tau=args.depth_tau,
        edge_tau=args.edge_tau,
        depth_window_radius=args.depth_window_radius,
        freq_tolerance_hz=args.freq_tolerance_hz,
    )
    print(f"Saved matches -> {out}")
