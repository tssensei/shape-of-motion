"""CLI adapter for the match-multi-views stage."""

from __future__ import annotations

import argparse

from modal_surface.multiview import build_multiview_observation_graph


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register match-multi-views command-line arguments."""
    parser.add_argument("--surface-packets", nargs="+", required=True, help="Surface packet .npz paths from all views.")
    parser.add_argument("--out", required=True, help="Output observation graph .npz path.")
    parser.add_argument("--mode-index", type=int, default=0, help="Selected modal frequency index.")
    parser.add_argument("--merge-voxel-size", type=float, default=0.03, help="World-space voxel size used to merge packet points.")
    parser.add_argument("--min-observations", type=int, default=2, help="Minimum observations per pooled surface point.")
    parser.add_argument("--mask-erode-iters", type=int, default=2, help="Target view mask erosion iterations.")
    parser.add_argument("--depth-tau", type=float, default=0.08, help="Relative depth consistency threshold.")
    parser.add_argument("--edge-tau", type=float, default=0.12, help="Relative local depth range threshold.")
    parser.add_argument("--depth-window-radius", type=int, default=2, help="Depth consistency window radius.")
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")


def run(args: argparse.Namespace) -> None:
    """Run match-multi-views from an argparse namespace."""
    out = build_multiview_observation_graph(
        surface_packet_paths=args.surface_packets,
        out_path=args.out,
        mode_index=args.mode_index,
        merge_voxel_size=args.merge_voxel_size,
        min_observations=args.min_observations,
        mask_erode_iters=args.mask_erode_iters,
        depth_tau=args.depth_tau,
        edge_tau=args.edge_tau,
        depth_window_radius=args.depth_window_radius,
        freq_tolerance_hz=args.freq_tolerance_hz,
    )
    print(f"Saved multi-view observations -> {out}")
