"""CLI adapter for the optimize-multi-view stage."""

from __future__ import annotations

import argparse

from modal_surface.optimization_staged import optimize_multi_view_staged
from modal_surface.solver_cli import add_staged_solver_arguments, staged_solver_config


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register optimize-multi-view command-line arguments."""
    parser.add_argument("--observations", required=True, help="Multi-view observation graph .npz path.")
    parser.add_argument("--out", required=True, help="Output latent_field multi-view .npz path.")
    parser.add_argument("--vis-dir", default=None, help="Optional visualization directory.")
    add_staged_solver_arguments(parser)


def run(args: argparse.Namespace) -> None:
    """Run optimize-multi-view from an argparse namespace."""
    out = optimize_multi_view_staged(
        observations_path=args.observations,
        out_path=args.out,
        vis_dir=args.vis_dir,
        config=staged_solver_config(args),
    )
    print(f"Saved multi-view latent field -> {out}")
