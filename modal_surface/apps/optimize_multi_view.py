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
    parser.add_argument("--single-view-smooth-lambda", type=float, default=0.0, help="Anchor-prior weight for refining points observed by only one view.")
    parser.add_argument("--single-view-smooth-k", type=int, default=8, help="Number of reliable anchor neighbors used for single-view refinement.")
    parser.add_argument(
        "--single-view-anchor-min-observations",
        type=int,
        default=2,
        help="Minimum observation count for points used as single-view smoothing anchors.",
    )
    parser.add_argument("--graph-smooth-lambda", type=float, default=0.0, help="Shared graph smoothness weight for all active points.")
    parser.add_argument("--graph-smooth-k", type=int, default=8, help="Number of nearest neighbors used to build the graph.")
    parser.add_argument("--graph-auto-radius-scale", type=float, default=2.5, help="Multiplier on median kth-neighbor distance for graph edge pruning.")
    parser.add_argument("--graph-min-shared-views", type=int, default=1, help="Minimum shared observed views required for a graph edge.")
    parser.add_argument("--obs-count-weight-1", type=float, default=0.25, help="Data weight multiplier for points observed by one view.")
    parser.add_argument("--obs-count-weight-2", type=float, default=0.75, help="Data weight multiplier for points observed by two views.")
    parser.add_argument("--obs-count-weight-3plus", type=float, default=1.0, help="Data weight multiplier for points observed by three or more views.")


def run(args: argparse.Namespace) -> None:
    """Run optimize-multi-view from an argparse namespace."""
    out = optimize_multi_view(
        observations_path=args.observations,
        out_path=args.out,
        vis_dir=args.vis_dir,
        iterations=args.iterations,
        ridge_mu=args.ridge_mu,
        outlier_frac=args.outlier_frac,
        single_view_smooth_lambda=args.single_view_smooth_lambda,
        single_view_smooth_k=args.single_view_smooth_k,
        single_view_anchor_min_observations=args.single_view_anchor_min_observations,
        graph_smooth_lambda=args.graph_smooth_lambda,
        graph_smooth_k=args.graph_smooth_k,
        graph_auto_radius_scale=args.graph_auto_radius_scale,
        graph_min_shared_views=args.graph_min_shared_views,
        obs_count_weight_1=args.obs_count_weight_1,
        obs_count_weight_2=args.obs_count_weight_2,
        obs_count_weight_3plus=args.obs_count_weight_3plus,
    )
    print(f"Saved multi-view latent field -> {out}")
