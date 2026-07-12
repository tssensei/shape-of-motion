"""Optimize one latent 3D modal displacement from N-view observations.

Each call solves one modal frequency. The saved ``alphas`` vector has shape
``(V,)`` and represents the per-view slice ``alpha[:, k]`` for that mode.
"""

from __future__ import annotations

from pathlib import Path

from modal_surface.optimization_staged import StagedSolverConfig, optimize_multi_view_staged


def optimize_multi_view(
    observations_path: str | Path,
    out_path: str | Path,
    vis_dir: str | Path | None = None,
    alpha_model: str = "phase",
    alpha_gain_min: float = 0.25,
    alpha_gain_max: float = 4.0,
    alpha_min_shared_points: int = 16,
    alpha_rank_ratio_min: float = 1e-4,
    alpha_info_ratio_min: float = 1e-4,
    alpha_failure: str = "exclude",
    anchor_svd_ratio_min: float = 1e-2,
    anchor_residual_max: float = 0.1,
) -> Path:
    """Solve one mode with overlap synchronization and observable reconstruction."""
    config = StagedSolverConfig(
        alpha_model=alpha_model,
        alpha_gain_min=alpha_gain_min,
        alpha_gain_max=alpha_gain_max,
        alpha_min_shared_points=alpha_min_shared_points,
        alpha_rank_ratio_min=alpha_rank_ratio_min,
        alpha_info_ratio_min=alpha_info_ratio_min,
        alpha_failure=alpha_failure,
        anchor_svd_ratio_min=anchor_svd_ratio_min,
        anchor_residual_max=anchor_residual_max,
    )
    return optimize_multi_view_staged(
        observations_path=observations_path,
        out_path=out_path,
        vis_dir=vis_dir,
        config=config,
    )
