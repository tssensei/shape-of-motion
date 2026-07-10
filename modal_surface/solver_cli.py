"""Shared CLI arguments for selecting and configuring the modal solver."""

from __future__ import annotations

import argparse
from typing import Any


def add_staged_solver_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--solver",
        choices=["staged", "legacy-als"],
        default="staged",
        help="Staged overlap/observable solver (default) or transitional legacy ALS.",
    )
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


def add_solver_arguments(
    parser: argparse.ArgumentParser,
    *,
    outlier_default: float | None = 0.0,
    include_modal_rigid: bool = False,
    include_modal_fill: bool = False,
) -> None:
    """Register the shared legacy-transition and staged solver controls."""
    parser.add_argument("--iterations", type=int, default=8, help="Legacy ALS iteration count.")
    parser.add_argument("--ridge-mu", type=float, default=1e-4, help="Legacy ALS per-point ridge regularization.")
    parser.add_argument(
        "--outlier-frac",
        type=float,
        default=outlier_default,
        help="Legacy ALS fraction of worst residual points dropped per iteration.",
    )
    parser.add_argument("--single-view-smooth-lambda", type=float, default=0.0, help="Legacy single-view anchor-prior weight.")
    parser.add_argument("--single-view-smooth-k", type=int, default=8, help="Legacy single-view anchor-neighbor count.")
    parser.add_argument("--single-view-anchor-min-observations", type=int, default=2, help="Legacy minimum observation count for single-view anchors.")
    parser.add_argument("--graph-smooth-lambda", type=float, default=0.0, help="Legacy graph-smoothing weight.")
    parser.add_argument("--graph-smooth-k", type=int, default=8, help="Legacy graph nearest-neighbor count.")
    parser.add_argument("--graph-auto-radius-scale", type=float, default=2.5, help="Legacy graph radius scale.")
    parser.add_argument("--graph-min-shared-views", type=int, default=1, help="Legacy minimum shared views for graph edges.")
    if include_modal_rigid:
        parser.add_argument("--modal-rigid-lambda", type=float, default=0.0, help="Legacy local modal-consensus weight.")
        parser.add_argument("--modal-rigid-k", type=int, default=8, help="Legacy modal-consensus neighbor count.")
        parser.add_argument("--modal-rigid-auto-radius-scale", type=float, default=2.0, help="Legacy modal-consensus radius scale.")
        parser.add_argument("--modal-rigid-min-shared-views", type=int, default=0, help="Legacy minimum shared views for modal-consensus edges.")
    if include_modal_fill:
        parser.add_argument("--modal-fill-unobserved", action="store_true", help="Deprecated legacy spatial fill for unobserved points.")
        parser.add_argument("--modal-fill-k", type=int, default=4, help="Legacy modal-fill neighbor count.")
        parser.add_argument("--modal-fill-auto-radius-scale", type=float, default=1.0, help="Legacy modal-fill radius scale.")
        parser.add_argument("--modal-fill-anchor-min-observations", type=int, default=1, help="Legacy minimum observation count for fill anchors.")
        parser.add_argument("--modal-fill-ridge-mu", type=float, default=1e-6, help="Legacy modal-fill ridge regularization.")
    parser.add_argument("--obs-count-weight-1", type=float, default=0.25, help="Legacy data multiplier for one-view points.")
    parser.add_argument("--obs-count-weight-2", type=float, default=0.75, help="Legacy data multiplier for two-view points.")
    parser.add_argument("--obs-count-weight-3plus", type=float, default=1.0, help="Legacy data multiplier for points with at least three views.")
    add_staged_solver_arguments(parser)


def staged_solver_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "solver": str(args.solver),
        "alpha_model": str(args.alpha_model),
        "alpha_gain_min": float(args.alpha_gain_min),
        "alpha_gain_max": float(args.alpha_gain_max),
        "alpha_min_shared_points": int(args.alpha_min_shared_points),
        "alpha_rank_ratio_min": float(args.alpha_rank_ratio_min),
        "alpha_info_ratio_min": float(args.alpha_info_ratio_min),
        "alpha_failure": str(args.alpha_failure),
        "anchor_svd_ratio_min": float(args.anchor_svd_ratio_min),
        "anchor_residual_max": float(args.anchor_residual_max),
    }


def solver_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Collect every solver kwarg registered on an argparse namespace."""
    kwargs = staged_solver_kwargs(args)
    for name in (
        "iterations",
        "ridge_mu",
        "outlier_frac",
        "single_view_smooth_lambda",
        "single_view_smooth_k",
        "single_view_anchor_min_observations",
        "graph_smooth_lambda",
        "graph_smooth_k",
        "graph_auto_radius_scale",
        "graph_min_shared_views",
        "modal_rigid_lambda",
        "modal_rigid_k",
        "modal_rigid_auto_radius_scale",
        "modal_rigid_min_shared_views",
        "modal_fill_unobserved",
        "modal_fill_k",
        "modal_fill_auto_radius_scale",
        "modal_fill_anchor_min_observations",
        "modal_fill_ridge_mu",
        "obs_count_weight_1",
        "obs_count_weight_2",
        "obs_count_weight_3plus",
    ):
        if hasattr(args, name):
            kwargs[name] = getattr(args, name)
    return kwargs


def validate_solver_args(
    args: argparse.Namespace,
    *,
    legacy_outlier_default: float = 0.0,
) -> None:
    if hasattr(args, "outlier_frac") and args.outlier_frac is None:
        args.outlier_frac = (
            0.0 if str(args.solver) == "staged" else float(legacy_outlier_default)
        )
    if str(args.solver) != "staged":
        staged_defaults: dict[str, Any] = {
            "alpha_model": "phase",
            "alpha_gain_min": 0.25,
            "alpha_gain_max": 4.0,
            "alpha_min_shared_points": 16,
            "alpha_rank_ratio_min": 1e-4,
            "alpha_info_ratio_min": 1e-4,
            "alpha_failure": "exclude",
            "anchor_svd_ratio_min": 1e-2,
            "anchor_residual_max": 0.1,
        }
        incompatible = [
            f"--{name.replace('_', '-')}"
            for name, default in staged_defaults.items()
            if hasattr(args, name) and getattr(args, name) != default
        ]
        if incompatible:
            raise ValueError(
                "The legacy ALS solver cannot use staged-only controls: "
                f"{', '.join(incompatible)}. Select --solver staged to use them."
            )
        return
    incompatible: list[str] = []
    legacy_defaults: dict[str, Any] = {
        "iterations": 8,
        "ridge_mu": 1e-4,
        "single_view_smooth_lambda": 0.0,
        "single_view_smooth_k": 8,
        "single_view_anchor_min_observations": 2,
        "graph_smooth_lambda": 0.0,
        "graph_smooth_k": 8,
        "graph_auto_radius_scale": 2.5,
        "graph_min_shared_views": 1,
        "modal_rigid_lambda": 0.0,
        "modal_rigid_k": 8,
        "modal_rigid_auto_radius_scale": 2.0,
        "modal_rigid_min_shared_views": 0,
        "modal_fill_unobserved": False,
        "modal_fill_k": 4,
        "modal_fill_auto_radius_scale": 1.0,
        "modal_fill_anchor_min_observations": 1,
        "modal_fill_ridge_mu": 1e-6,
        "obs_count_weight_1": 0.25,
        "obs_count_weight_2": 0.75,
        "obs_count_weight_3plus": 1.0,
    }
    for name, default in legacy_defaults.items():
        if hasattr(args, name) and getattr(args, name) != default:
            incompatible.append(f"--{name.replace('_', '-')}")
    if hasattr(args, "outlier_frac") and float(args.outlier_frac) != 0.0:
        incompatible.append("--outlier-frac")
    if incompatible:
        joined = ", ".join(incompatible)
        raise ValueError(
            f"The staged solver stops after observable-anchor selection and cannot use {joined}. "
            "Select --solver legacy-als to use those options."
        )


def staged_solver_manifest_parameters(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.solver) == "legacy-als":
        return {"solver": "legacy-als"}
    return {
        "solver": str(args.solver),
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
