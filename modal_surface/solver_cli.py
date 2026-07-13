"""Shared CLI arguments for configuring the staged modal solver."""

from __future__ import annotations

import argparse
from typing import Any

from modal_surface.optimization_staged import StagedSolverConfig


def add_staged_solver_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--alpha-model",
        choices=["phase", "bounded-complex"],
        default="phase",
        help="Estimate unit-magnitude phase offsets or bounded complex phase/gain offsets.",
    )
    parser.add_argument("--alpha-gain-min", type=float, default=0.25, help="Minimum alpha magnitude in bounded-complex mode.")
    parser.add_argument("--alpha-gain-max", type=float, default=4.0, help="Maximum alpha magnitude in bounded-complex mode.")
    parser.add_argument(
        "--alpha-min-shared-points",
        type=int,
        default=16,
        help="Minimum informative shared points required to connect two views for alpha synchronization.",
    )
    parser.add_argument(
        "--alpha-rank-ratio-min",
        type=float,
        default=1e-4,
        help="Minimum gauge-reduced alpha Jacobian sigma_min/sigma_max ratio.",
    )
    parser.add_argument(
        "--alpha-info-ratio-min",
        type=float,
        default=1e-4,
        help="Minimum absolute alpha information normalized by observation energy.",
    )
    parser.add_argument(
        "--alpha-failure",
        choices=["exclude", "error"],
        default="exclude",
        help="Exclude and mark unidentifiable views, or fail the whole mode.",
    )
    parser.add_argument(
        "--anchor-svd-ratio-min",
        type=float,
        default=1e-2,
        help="Minimum point-system singular ratio retained as observable; anchors also require condition <= 100.",
    )
    parser.add_argument(
        "--anchor-residual-max",
        type=float,
        default=0.1,
        help="Maximum normalized residual threshold used by reliable-anchor classification.",
    )


def staged_solver_config(args: argparse.Namespace) -> StagedSolverConfig:
    return StagedSolverConfig(
        alpha_model=str(args.alpha_model),
        alpha_gain_min=float(args.alpha_gain_min),
        alpha_gain_max=float(args.alpha_gain_max),
        alpha_min_shared_points=int(args.alpha_min_shared_points),
        alpha_rank_ratio_min=float(args.alpha_rank_ratio_min),
        alpha_info_ratio_min=float(args.alpha_info_ratio_min),
        alpha_failure=str(args.alpha_failure),
        anchor_svd_ratio_min=float(args.anchor_svd_ratio_min),
        anchor_residual_max=float(args.anchor_residual_max),
    )


def staged_solver_manifest_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "solver": "staged",
        "alpha_solver_model": str(args.alpha_model),
        "alpha_gain_min": float(args.alpha_gain_min),
        "alpha_gain_max": float(args.alpha_gain_max),
        "alpha_min_shared_points": int(args.alpha_min_shared_points),
        "alpha_rank_ratio_min": float(args.alpha_rank_ratio_min),
        "alpha_info_ratio_min": float(args.alpha_info_ratio_min),
        "alpha_failure": str(args.alpha_failure),
        "anchor_svd_ratio_min": float(args.anchor_svd_ratio_min),
        "anchor_residual_max": float(args.anchor_residual_max),
    }
