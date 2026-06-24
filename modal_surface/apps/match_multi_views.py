"""CLI adapter for the match-multi-views stage."""

from __future__ import annotations

import argparse

from modal_surface.multiview import build_multiview_observation_graph


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register match-multi-views command-line arguments."""
    parser.add_argument("--canonical-packet", required=True, help="Canonical surface packet .npz path, usually view1.")
    parser.add_argument("--target-view-configs", nargs="+", required=True, help="Target view JSON config paths.")
    parser.add_argument("--target-modal-npzs", nargs="+", required=True, help="Target modal_analysis.npz paths.")
    parser.add_argument("--out", required=True, help="Output observation graph .npz path.")
    parser.add_argument("--mode-index", type=int, default=0, help="Selected modal frequency index.")
    parser.add_argument("--min-observations", type=int, default=2, help="Minimum observations per canonical point, including the canonical view.")
    parser.add_argument("--mask-erode-iters", type=int, default=2, help="Target view mask erosion iterations.")
    parser.add_argument("--depth-tau", type=float, default=0.08, help="Relative depth consistency threshold.")
    parser.add_argument("--edge-tau", type=float, default=0.12, help="Relative local depth range threshold.")
    parser.add_argument("--depth-window-radius", type=int, default=2, help="Depth consistency window radius.")
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")


def run(args: argparse.Namespace) -> None:
    """Run match-multi-views from an argparse namespace."""
    out = build_multiview_observation_graph(
        canonical_packet_path=args.canonical_packet,
        target_view_config_paths=args.target_view_configs,
        target_modal_npz_paths=args.target_modal_npzs,
        out_path=args.out,
        mode_index=args.mode_index,
        min_observations=args.min_observations,
        mask_erode_iters=args.mask_erode_iters,
        depth_tau=args.depth_tau,
        edge_tau=args.edge_tau,
        depth_window_radius=args.depth_window_radius,
        freq_tolerance_hz=args.freq_tolerance_hz,
    )
    print(f"Saved multi-view observations -> {out}")
