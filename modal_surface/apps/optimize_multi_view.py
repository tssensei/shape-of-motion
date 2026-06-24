"""CLI adapter for the optimize-multi-view stage."""

from __future__ import annotations

import argparse

from modal_surface.optimization_multi import optimize_multi_view


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register optimize-multi-view command-line arguments."""
    parser.add_argument("--observations", required=True, help="Multi-view observation graph .npz path.")
    parser.add_argument("--out", required=True, help="Output latent_field multi-view .npz path.")
    parser.add_argument("--vis-dir", default=None, help="Optional visualization directory.")
    parser.add_argument("--iterations", type=int, default=8, help="Alternating optimization iterations.")
    parser.add_argument("--ridge-mu", type=float, default=1e-4, help="Per-point ridge regularization.")
    parser.add_argument("--outlier-frac", type=float, default=0.05, help="Fraction of worst residual points dropped per iteration.")


def run(args: argparse.Namespace) -> None:
    """Run optimize-multi-view from an argparse namespace."""
    out = optimize_multi_view(
        observations_path=args.observations,
        out_path=args.out,
        vis_dir=args.vis_dir,
        iterations=args.iterations,
        ridge_mu=args.ridge_mu,
        outlier_frac=args.outlier_frac,
    )
    print(f"Saved multi-view latent field -> {out}")
