from __future__ import annotations

import argparse

from modal_surface.packets import make_surface_packet


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--view-config", required=True, help="View JSON config path.")
    parser.add_argument("--modal-npz", required=True, help="modal_analysis.npz from modal peak-pick export.")
    parser.add_argument("--out", required=True, help="Output surface packet .npz path.")
    parser.add_argument("--stride", type=int, default=4, help="Pixel sampling stride.")
    parser.add_argument("--mask-erode-iters", type=int, default=2, help="3x3 mask erosion iterations.")
    parser.add_argument("--depth-edge-tau", type=float, default=0.12, help="Depth local range threshold.")
    parser.add_argument("--min-amplitude-percentile", type=float, default=5.0, help="Drop points below this max-amplitude percentile.")


def run(args: argparse.Namespace) -> None:
    out = make_surface_packet(
        view_config_path=args.view_config,
        modal_npz_path=args.modal_npz,
        out_path=args.out,
        stride=args.stride,
        mask_erode_iters=args.mask_erode_iters,
        depth_edge_tau=args.depth_edge_tau,
        min_amplitude_percentile=args.min_amplitude_percentile,
    )
    print(f"Saved surface packet -> {out}")

