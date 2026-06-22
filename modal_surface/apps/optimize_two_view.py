"""CLI adapter for the optimize-two-view stage."""

from __future__ import annotations

import argparse

from modal_surface.optimization import optimize_two_view


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register optimize-two-view command-line arguments."""
    parser.add_argument("--matches", required=True, help="matches_12.npz path.")
    parser.add_argument("--out", required=True, help="Output latent_field_12.npz path.")
    parser.add_argument("--vis-dir", default=None, help="Optional visualization directory.")
    parser.add_argument("--iterations", type=int, default=8, help="Alternating optimization iterations.")
    parser.add_argument("--ridge-mu", type=float, default=1e-4, help="Per-point ridge regularization.")
    parser.add_argument("--outlier-frac", type=float, default=0.05, help="Fraction of worst residual points dropped per iteration.")


def run(args: argparse.Namespace) -> None:
    """Run optimize-two-view from an argparse namespace."""
    out = optimize_two_view(
        matches_path=args.matches,
        out_path=args.out,
        vis_dir=args.vis_dir,
        iterations=args.iterations,
        ridge_mu=args.ridge_mu,
        outlier_frac=args.outlier_frac,
    )
    print(f"Saved latent field -> {out}")
