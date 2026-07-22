"""Shared CLI arguments for selecting and configuring modal solvers."""

from __future__ import annotations

import argparse
from typing import Any

from modal_surface.optimization_staged import StagedSolverConfig


STAGED_ANCHOR_SVD_RATIO_DEFAULT = 1e-2
STAGED_ANCHOR_RESIDUAL_MAX_DEFAULT = 0.1
RIGID_COMPONENT_RCOND_DEFAULT = 1e-8
RIGID_SEED_MIN_VALID_VIEWS_DEFAULT = 2
RIGID_SEED_MIN_SECONDARY_VIEW_NODE_RATIO_DEFAULT = 1.0 / 3.0
RIGID_SEED_MIN_SINGULAR_RATIO_DEFAULT = 1e-3
RIGID_SEED_MAX_FINITE_DRIFT_DEFAULT = 2.0
RIGID_MOTION_FILL_STAGE_DEFAULT = "joint"
RIGID_SINGLE_VIEW_OBSERVABLE_RATIO_DEFAULT = 1.0e-2
RIGID_SINGLE_VIEW_RAY_DIRECTION_MIN_FRACTION_DEFAULT = 0.8
RIGID_SINGLE_VIEW_MAX_NORMALIZED_MOTION_RMS_DEFAULT = 2.0


def add_solve_method_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--solve-method",
        choices=["staged", "rigid-components"],
        default="staged",
        help=(
            "Run the existing staged per-Gaussian solve, or consume prebuilt "
            "observed structure graphs and solve one rigid twist per component."
        ),
    )
    parser.add_argument(
        "--rigid-component-graph",
        action="append",
        default=[],
        help=(
            "Observed Gaussian structure graph NPZ. Repeat once per requested "
            "mode when --solve-method=rigid-components."
        ),
    )
    parser.add_argument(
        "--rigid-component-rcond",
        type=float,
        default=RIGID_COMPONENT_RCOND_DEFAULT,
        help=(
            "Relative singular-value cutoff for deterministic minimum-norm "
            "rigid-component solves (default: 1e-8)."
        ),
    )
    parser.add_argument(
        "--rigid-seed-min-valid-views",
        type=int,
        default=RIGID_SEED_MIN_VALID_VIEWS_DEFAULT,
        help=(
            "Minimum node-coverage-supported alpha-identifiable views required "
            "to retain a solved rigid component as a trusted seed (default: 2)."
        ),
    )
    parser.add_argument(
        "--rigid-seed-min-singular-ratio",
        type=float,
        default=RIGID_SEED_MIN_SINGULAR_RATIO_DEFAULT,
        help=(
            "Minimum full-system sigma_min/sigma_max ratio required to retain "
            "a solved rigid component as a trusted seed (default: 1e-3)."
        ),
    )
    parser.add_argument(
        "--rigid-seed-min-secondary-view-node-ratio",
        type=float,
        default=RIGID_SEED_MIN_SECONDARY_VIEW_NODE_RATIO_DEFAULT,
        help=(
            "Minimum distinct-node coverage relative to the dominant view for "
            "another view to support a rigid seed (default: 1/3)."
        ),
    )
    parser.add_argument(
        "--rigid-seed-max-finite-drift",
        type=float,
        default=RIGID_SEED_MAX_FINITE_DRIFT_DEFAULT,
        help=(
            "Maximum component finite-amplitude edge-length drift retained as "
            "a trusted rigid seed (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--rigid-motion-fill-stage",
        choices=["joint", "single-view-components"],
        default=RIGID_MOTION_FILL_STAGE_DEFAULT,
        help=(
            "Run the existing joint component/point fill, or only complete "
            "single-view component weak directions before any pointwise fill "
            f"(default: {RIGID_MOTION_FILL_STAGE_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--rigid-single-view-observable-ratio",
        type=float,
        default=RIGID_SINGLE_VIEW_OBSERVABLE_RATIO_DEFAULT,
        help=(
            "Minimum component observation singular-value ratio retained in "
            "the single-view observable twist (default: 1e-2)."
        ),
    )
    parser.add_argument(
        "--rigid-single-view-ray-direction-min-fraction",
        type=float,
        default=RIGID_SINGLE_VIEW_RAY_DIRECTION_MIN_FRACTION_DEFAULT,
        help=(
            "Minimum induced radial-motion fraction that moves a single-view "
            "twist basis direction into KNN completion (default: 0.8)."
        ),
    )
    parser.add_argument(
        "--rigid-single-view-max-normalized-motion-rms",
        type=float,
        default=RIGID_SINGLE_VIEW_MAX_NORMALIZED_MOTION_RMS_DEFAULT,
        help=(
            "Maximum post-fill point-motion RMS divided by component radius; "
            "larger single-view components are zeroed as a whole (default: 2.0)."
        ),
    )


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
        default=STAGED_ANCHOR_SVD_RATIO_DEFAULT,
        help="Minimum point-system singular ratio retained as observable; anchors also require condition <= 100.",
    )
    parser.add_argument(
        "--anchor-residual-max",
        type=float,
        default=STAGED_ANCHOR_RESIDUAL_MAX_DEFAULT,
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


def rigid_component_manifest_parameters(args: argparse.Namespace) -> dict[str, Any]:
    motion_fill_stage = str(args.rigid_motion_fill_stage)
    motion_fill_enabled = bool(getattr(args, "motion_fill", False))
    component_fill_only = motion_fill_enabled and (
        motion_fill_stage == "single-view-components"
    )
    if component_fill_only:
        nonseed_policy = "single_view_partial_rigid_other_gaussians_zero"
        component_fill_policy = (
            "observable_twist_plus_knn_filled_weak_and_ray_directions"
        )
    elif motion_fill_enabled:
        nonseed_policy = "single_view_component_rigid_else_free_motion_fill"
        component_fill_policy = "shared_unknown_normalized_infinitesimal_se3_twist"
    else:
        nonseed_policy = "zero_without_motion_fill"
        component_fill_policy = "disabled"
    return {
        "solver": "rigid_components",
        "rigidity_model": "complex_infinitesimal_se3",
        "rigid_component_rcond": float(args.rigid_component_rcond),
        "rigid_component_selection": "observed_graph_degree_positive",
        "rigid_component_connectivity_policy": (
            "accepted_edge_transitive_components_bridges_merge"
        ),
        "rigid_component_rank_policy": "truncated_svd_minimum_norm_no_rejection",
        "rigid_component_residual_policy": "diagnostic_only",
        "rigid_component_edge_weight_use": "topology_only",
        "rigid_component_first_order_rtol": 1e-6,
        "rigid_component_first_order_validation": (
            "complex128_model_strict_complex64_quantization_accounted"
        ),
        "rigid_component_persisted_strain_policy": (
            "diagnostic_with_per_edge_actual_cast_error_bound"
        ),
        "rigid_component_finite_phase_samples": 64,
        "rigid_component_finite_rigidity": "first_order_only",
        "rigid_component_seed_policy": (
            "postsolve_valid_view_singular_ratio_and_finite_drift_gate"
        ),
        "rigid_seed_min_valid_views": int(args.rigid_seed_min_valid_views),
        "rigid_seed_min_secondary_view_node_ratio": float(
            args.rigid_seed_min_secondary_view_node_ratio
        ),
        "rigid_seed_min_singular_ratio": float(
            args.rigid_seed_min_singular_ratio
        ),
        "rigid_seed_max_finite_drift": float(
            args.rigid_seed_max_finite_drift
        ),
        "nonseed_policy": nonseed_policy,
        "single_view_component_fill_policy": component_fill_policy,
        "rigid_motion_fill_stage": motion_fill_stage,
        "rigid_single_view_observable_ratio": float(
            args.rigid_single_view_observable_ratio
        ),
        "rigid_single_view_ray_direction_min_fraction": float(
            args.rigid_single_view_ray_direction_min_fraction
        ),
        "rigid_single_view_max_normalized_motion_rms": float(
            args.rigid_single_view_max_normalized_motion_rms
        ),
        "alpha_solver_model": str(args.alpha_model),
        "alpha_gain_min": float(args.alpha_gain_min),
        "alpha_gain_max": float(args.alpha_gain_max),
        "alpha_min_shared_points": int(args.alpha_min_shared_points),
        "alpha_rank_ratio_min": float(args.alpha_rank_ratio_min),
        "alpha_info_ratio_min": float(args.alpha_info_ratio_min),
        "alpha_failure": str(args.alpha_failure),
    }
