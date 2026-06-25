"""CLI adapter for building observations from a VGGT carrier point cloud."""

from __future__ import annotations

import argparse

from modal_surface.carrier import build_carrier_observation_graph


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register match-carrier-views command-line arguments."""
    parser.add_argument("--carrier-points", required=True, help="VGGT carrier points .npz path.")
    parser.add_argument("--view-config", action="append", required=True, help="View JSON config path. Repeat per view.")
    parser.add_argument("--modal-npz", action="append", required=True, help="modal_analysis.npz path. Repeat per view.")
    parser.add_argument("--out", required=True, help="Output observation graph .npz path.")
    parser.add_argument("--mode-index", type=int, default=0, help="Selected modal frequency index.")
    parser.add_argument("--mask-erode-iters", type=int, default=1, help="3x3 modal mask erosion iterations.")
    parser.add_argument("--zbuffer-radius", type=int, default=5, help="Local robust z-buffer window radius in pixels.")
    parser.add_argument("--front-percentile", type=float, default=10.0, help="Local depth percentile treated as front surface.")
    parser.add_argument("--zbuffer-tau", type=float, default=0.05, help="Relative depth threshold against local front depth.")
    parser.add_argument("--min-zbuffer-samples", type=int, default=5, help="Minimum local carrier depths for visibility.")
    parser.add_argument("--min-observations", type=int, default=2, help="Minimum observed views per carrier point.")
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")


def run(args: argparse.Namespace) -> None:
    """Run match-carrier-views from an argparse namespace."""
    out = build_carrier_observation_graph(
        carrier_points_path=args.carrier_points,
        view_config_paths=args.view_config,
        modal_npz_paths=args.modal_npz,
        out_path=args.out,
        mode_index=args.mode_index,
        mask_erode_iters=args.mask_erode_iters,
        zbuffer_radius=args.zbuffer_radius,
        front_percentile=args.front_percentile,
        zbuffer_tau=args.zbuffer_tau,
        min_zbuffer_samples=args.min_zbuffer_samples,
        min_observations=args.min_observations,
        freq_tolerance_hz=args.freq_tolerance_hz,
    )
    print(f"Saved carrier-view observations -> {out}")
