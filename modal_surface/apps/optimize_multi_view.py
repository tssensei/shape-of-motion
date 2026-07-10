"""CLI adapter for the optimize-multi-view stage."""

from __future__ import annotations

import argparse

from modal_surface.optimization_multi import optimize_multi_view
from modal_surface.solver_cli import add_solver_arguments, solver_kwargs, validate_solver_args


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register optimize-multi-view command-line arguments."""
    parser.add_argument("--observations", required=True, help="Multi-view observation graph .npz path.")
    parser.add_argument("--out", required=True, help="Output latent_field multi-view .npz path.")
    parser.add_argument("--vis-dir", default=None, help="Optional visualization directory.")
    add_solver_arguments(parser, outlier_default=None)


def run(args: argparse.Namespace) -> None:
    """Run optimize-multi-view from an argparse namespace."""
    validate_solver_args(args, legacy_outlier_default=0.05)
    out = optimize_multi_view(
        observations_path=args.observations,
        out_path=args.out,
        vis_dir=args.vis_dir,
        **solver_kwargs(args),
    )
    print(f"Saved multi-view latent field -> {out}")
