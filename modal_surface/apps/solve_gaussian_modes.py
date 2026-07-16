"""Batch solve 3D modal fields directly on foreground 3DGS Gaussians."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, SupportsFloat

import numpy as np

from modal_surface.gaussian_observations import build_gaussian_observation_graph
from modal_surface.gaussian_motion_fill import (
    MOTION_FILL_EDGE_WEIGHTING_SPATIAL,
    MOTION_FILL_EDGE_WEIGHTING_SPATIAL_RGB,
    MOTION_FILL_EPSILON,
    MOTION_FILL_LSMR_ATOL,
    MOTION_FILL_LSMR_BTOL,
    MOTION_FILL_LSMR_CONLIM,
    MOTION_FILL_NULLSPACE_RTOL,
    MOTION_FILL_NUMERICAL_RANK_POLICY,
    MOTION_FILL_OBSERVATION_DRIFT_RTOL,
    MOTION_FILL_METHOD,
    MOTION_FILL_ROLE_NAMES,
    MOTION_FILL_VERSION,
    GaussianGraphWeightingResult,
    GaussianMotionFillResult,
    apply_gaussian_motion_fill,
    weight_gaussian_motion_fill_graph,
    write_motion_fill_diagnostics,
    write_motion_fill_graph,
)
from modal_surface.io import (
    load_modal_freqs,
    load_view_config,
    save_npz_compressed_atomic,
)
from modal_surface.motion_fill import KnnGraph, build_knn_graph, query_knn_candidates
from modal_surface.optimization_staged import (
    ANCHOR_CONDITION_MAX,
    POINT_STATUS_NAMES,
    StagedSolveResult,
    enforce_alpha_failure,
    optimize_multi_view_staged,
)
from modal_surface.optimization_visualization import write_solve_visualizations
from modal_surface.solver_cli import (
    add_staged_solver_arguments,
    staged_solver_config,
    staged_solver_manifest_parameters,
)


_DEFAULT_MOTION_FILL_K = 8


def parse_mode_indices(raw: str, num_modes: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(num_modes))
    out: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        idx = int(text)
        if idx < 0 or idx >= num_modes:
            raise ValueError(f"mode index {idx} is outside [0,{num_modes - 1}].")
        out.append(idx)
    if not out:
        raise ValueError("--mode-indices must contain at least one index, or 'all'.")
    return out


def freq_slug(freq_hz: float) -> str:
    text = f"{freq_hz:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def latent_stats(
    staged: StagedSolveResult,
    motion_fill: GaussianMotionFillResult | None,
    observations: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    observable = staged.observable
    alpha = staged.alpha
    if motion_fill is None:
        obs_residual = staged.obs_residual
        obs_residual_valid = staged.obs_residual_valid_mask
        point_residual = staged.point_residual
        point_residual_valid = staged.point_residual_valid_mask
    else:
        obs_residual = motion_fill.obs_residual
        obs_residual_valid = motion_fill.obs_residual_valid_mask
        point_residual = motion_fill.point_residual
        point_residual_valid = motion_fill.point_residual_valid_mask
    obs_residual = np.asarray(obs_residual)[np.asarray(obs_residual_valid, dtype=bool)]
    point_residual = np.asarray(point_residual)[np.asarray(point_residual_valid, dtype=bool)]
    return {
        "num_points": int(staged.prepared.points.shape[0]),
        "num_observations": int(staged.prepared.obs_point_index.shape[0]),
        "obs_residual_median": _finite_percentile(obs_residual, 50),
        "obs_residual_p90": _finite_percentile(obs_residual, 90),
        "point_residual_median": _finite_percentile(point_residual, 50),
        "point_residual_p90": _finite_percentile(point_residual, 90),
        "observations_per_view": observations["observations_per_view"].astype(int).tolist(),
        "solver_method": "staged_overlap_observable",
        "alpha_identifiable_count": int(alpha.identifiable_mask.sum()),
        "alpha_optimizer_success": bool(alpha.optimizer_success),
        "anchor_count": int(observable.anchor_mask.sum()),
        "partial_unresolved_count": int(observable.partial_mask.sum()),
        "rejected_count": int(observable.rejected_mask.sum()),
        "alpha_unresolved_point_count": int(observable.alpha_unresolved_mask.sum()),
        "no_usable_observation_point_count": int(
            observable.no_usable_observation_mask.sum()
        ),
    }


def json_float(value: SupportsFloat) -> float | None:
    result = float(value)
    if not np.isfinite(result):
        return None
    return result


def alpha_by_view_diagnostics(
    staged: StagedSolveResult,
) -> list[dict[str, Any]]:
    alpha = staged.alpha
    view_ids = [str(v) for v in staged.prepared.view_ids.tolist()]
    alphas = alpha.alphas.astype(np.complex64).reshape(-1)
    freqs_hz = staged.alpha_view_freqs_hz.astype(np.float32).reshape(-1)
    if alphas.shape[0] != len(view_ids):
        raise ValueError("Alpha count does not match view_ids.")
    if freqs_hz.shape[0] != len(view_ids):
        raise ValueError("alpha_view_freqs_hz length does not match view_ids.")
    identifiable = alpha.identifiable_mask.astype(bool).reshape(-1)
    reasons = alpha.exclusion_reason.astype(str).reshape(-1)
    phase_std = alpha.phase_std.astype(np.float32).reshape(-1)
    log_gain_std = alpha.log_gain_std.astype(np.float32).reshape(-1)
    gain_std = (np.abs(alphas) * log_gain_std).astype(np.float32)
    gain_bound_active = alpha.gain_bound_active_mask.astype(bool).reshape(-1)
    staged_diagnostics = {
        "alpha_identifiable_mask": identifiable,
        "alpha_exclusion_reason": reasons,
        "alpha_phase_std": phase_std,
        "alpha_gain_std": gain_std,
        "alpha_log_gain_std": log_gain_std,
        "alpha_gain_bound_active_mask": gain_bound_active,
    }
    invalid = [name for name, values in staged_diagnostics.items() if values.shape[0] != len(view_ids)]
    if invalid:
        raise ValueError(f"Staged alpha diagnostics have invalid lengths: {invalid}.")
    return [
        {
            "view_id": view_id,
            "freq_hz": json_float(freqs_hz[idx]),
            "real": json_float(np.real(alphas[idx])),
            "imag": json_float(np.imag(alphas[idx])),
            "abs": json_float(np.abs(alphas[idx])),
            "phase_rad": json_float(np.angle(alphas[idx])),
            "identifiable": bool(identifiable[idx]),
            "reason": str(reasons[idx]),
            "phase_std": json_float(phase_std[idx]),
            "gain_std": json_float(gain_std[idx]),
            "log_gain_std": json_float(log_gain_std[idx]),
            "gain_bound_active": bool(gain_bound_active[idx]),
        }
        for idx, view_id in enumerate(view_ids)
    ]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-ckpt", required=True, help="Static 3DGS checkpoint containing foreground Gaussian centers.")
    parser.add_argument("--view-config", action="append", required=True, help="View JSON config path. Repeat per view.")
    parser.add_argument("--modal-npz", action="append", required=True, help="modal_analysis.npz path. Repeat per view.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for observations, diagnostics, latents, vis, and manifest.",
    )
    parser.add_argument("--mode-indices", default="all", help="Comma-separated zero-based mode indices, or 'all'.")
    parser.add_argument("--pixel-sample-stride", type=int, default=4, help="Pixel grid stride for pixel-candidates sampling.")
    parser.add_argument("--pixel-candidate-k", type=int, default=4, help="Number of top contribution Gaussians supervised by each sampled pixel.")
    parser.add_argument("--pixel-preselect-k", type=int, default=32, help="Number of 3D nearest Gaussians scored before top-k contribution selection.")
    parser.add_argument("--pixel-render-acc-min", type=float, default=0.05, help="Minimum rendered foreground alpha for sampled modal pixels.")
    parser.add_argument("--pixel-min-contribution", type=float, default=1e-12, help="Minimum unnormalized Gaussian contribution retained for a sampled modal pixel.")
    parser.add_argument("--mask-erode-iters", type=int, default=1, help="3x3 modal mask erosion iterations.")
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")
    parser.add_argument(
        "--motion-fill",
        action="store_true",
        help="Fill accepted Gaussian nullspaces and truly unobserved Gaussians after the staged solve.",
    )
    parser.add_argument(
        "--motion-fill-k",
        type=int,
        default=_DEFAULT_MOTION_FILL_K,
        help="Spatial KNN neighborhood size used when --motion-fill is enabled (default: 8).",
    )
    parser.add_argument(
        "--motion-fill-max-distance",
        type=float,
        default=None,
        help="Required maximum KNN edge distance in scene units when --motion-fill is enabled.",
    )
    parser.add_argument(
        "--motion-fill-lsmr-atol",
        type=float,
        default=MOTION_FILL_LSMR_ATOL,
        help="LSMR normal-equation tolerance used by motion fill (default: 1e-8).",
    )
    parser.add_argument(
        "--motion-fill-edge-weighting",
        choices=(
            MOTION_FILL_EDGE_WEIGHTING_SPATIAL,
            MOTION_FILL_EDGE_WEIGHTING_SPATIAL_RGB,
        ),
        default=MOTION_FILL_EDGE_WEIGHTING_SPATIAL,
        help="Use spatial inverse-distance weights or add activated-RGB affinity.",
    )
    parser.add_argument(
        "--motion-fill-color-sigma",
        type=float,
        default=None,
        help="Activated-RGB affinity sigma; spatial-rgb defaults to retained-edge median.",
    )
    add_staged_solver_arguments(parser)


def _validate_motion_fill_arguments(
    args: argparse.Namespace,
    num_points: int | None = None,
) -> None:
    if not bool(args.motion_fill):
        if args.motion_fill_max_distance is not None:
            raise ValueError("--motion-fill-max-distance requires --motion-fill.")
        if args.motion_fill_k != _DEFAULT_MOTION_FILL_K:
            raise ValueError("A custom --motion-fill-k requires --motion-fill.")
        if args.motion_fill_lsmr_atol != MOTION_FILL_LSMR_ATOL:
            raise ValueError("A custom --motion-fill-lsmr-atol requires --motion-fill.")
        if args.motion_fill_edge_weighting != MOTION_FILL_EDGE_WEIGHTING_SPATIAL:
            raise ValueError(
                "Non-default --motion-fill-edge-weighting requires --motion-fill."
            )
        if args.motion_fill_color_sigma is not None:
            raise ValueError("--motion-fill-color-sigma requires --motion-fill.")
        return
    if args.motion_fill_max_distance is None:
        raise ValueError("--motion-fill requires --motion-fill-max-distance in scene units.")
    if (
        isinstance(args.motion_fill_k, (bool, np.bool_))
        or not isinstance(args.motion_fill_k, (int, np.integer))
        or int(args.motion_fill_k) <= 0
    ):
        raise ValueError("--motion-fill-k must be a positive integer.")
    max_distance = float(args.motion_fill_max_distance)
    if not np.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("--motion-fill-max-distance must be finite and positive.")
    if isinstance(args.motion_fill_lsmr_atol, (bool, np.bool_)):
        raise ValueError("--motion-fill-lsmr-atol must be finite and positive.")
    lsmr_atol = float(args.motion_fill_lsmr_atol)
    if not np.isfinite(lsmr_atol) or lsmr_atol <= 0.0:
        raise ValueError("--motion-fill-lsmr-atol must be finite and positive.")
    if args.motion_fill_edge_weighting == MOTION_FILL_EDGE_WEIGHTING_SPATIAL:
        if args.motion_fill_color_sigma is not None:
            raise ValueError(
                "--motion-fill-color-sigma requires "
                "--motion-fill-edge-weighting spatial-rgb."
            )
    elif args.motion_fill_edge_weighting == MOTION_FILL_EDGE_WEIGHTING_SPATIAL_RGB:
        if args.motion_fill_color_sigma is not None:
            if isinstance(args.motion_fill_color_sigma, (bool, np.bool_)):
                raise ValueError(
                    "--motion-fill-color-sigma must be finite and positive."
                )
            color_sigma = float(args.motion_fill_color_sigma)
            if not np.isfinite(color_sigma) or color_sigma <= 0.0:
                raise ValueError(
                    "--motion-fill-color-sigma must be finite and positive."
                )
    else:
        raise ValueError(
            "--motion-fill-edge-weighting must be 'spatial' or 'spatial-rgb'."
        )
    if num_points is not None and int(args.motion_fill_k) >= int(num_points):
        raise ValueError(
            f"--motion-fill-k must be smaller than the foreground Gaussian count ({num_points})."
        )


def _load_fg_pixel_candidate_inputs_from_checkpoint(
    path: str,
    view_config_paths: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
]:
    import torch
    from flow3d.scene_model import SceneModel
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sceneModel = SceneModel.init_from_state_dict(state).to(device)
    sceneModel.eval()
    fg_means = sceneModel.fg.params["means"].detach().cpu().float().numpy().astype(np.float32)
    fg_scales = sceneModel.fg.get_scales().detach().cpu().float().numpy().astype(np.float32)
    fg_quats = sceneModel.fg.get_quats().detach().cpu().float().numpy().astype(np.float32)
    fg_opacities = sceneModel.fg.get_opacities().detach().cpu().float().numpy().reshape(-1).astype(np.float32)
    fg_colors = sceneModel.fg.get_colors().detach().cpu().float().numpy().astype(np.float32)
    if fg_means.ndim != 2 or fg_means.shape[1] != 3:
        raise ValueError(f"fg.params.means must have shape (N,3), got {fg_means.shape}")
    if fg_scales.shape != fg_means.shape:
        raise ValueError(f"activated foreground scales must have shape {fg_means.shape}, got {fg_scales.shape}")
    if fg_quats.shape != (fg_means.shape[0], 4):
        raise ValueError(f"activated foreground quats must have shape ({fg_means.shape[0]},4), got {fg_quats.shape}")
    if fg_opacities.shape != (fg_means.shape[0],):
        raise ValueError(f"activated foreground opacities must have shape ({fg_means.shape[0]},), got {fg_opacities.shape}")
    if fg_colors.shape != fg_means.shape:
        raise ValueError(
            f"activated foreground colors must have shape {fg_means.shape}, got {fg_colors.shape}"
        )
    if not np.all(np.isfinite(fg_means)):
        raise ValueError(f"{path} contains non-finite foreground Gaussian centers")
    if not np.all(np.isfinite(fg_colors)):
        raise ValueError(f"{path} contains non-finite activated foreground Gaussian colors")
    if np.any((fg_colors < 0.0) | (fg_colors > 1.0)):
        raise ValueError(f"{path} contains activated foreground Gaussian colors outside [0,1]")

    rendered_depths: list[np.ndarray] = []
    rendered_accs: list[np.ndarray] = []
    with torch.no_grad():
        for cfg_path in view_config_paths:
            cfg = load_view_config(cfg_path)
            w2c = torch.from_numpy(cfg.world_to_camera).to(device=device, dtype=torch.float32)[None]
            K = torch.from_numpy(cfg.K).to(device=device, dtype=torch.float32)[None]
            rendered = sceneModel.render(
                None,
                w2c,
                K,
                (cfg.image_width, cfg.image_height),
                return_depth=True,
                return_mask=False,
                fg_only=True,
            )
            depth = rendered["depth"][0, ..., 0].detach().cpu().float().numpy().astype(np.float32)
            acc_tensor = rendered["acc"][0]
            if acc_tensor.ndim == 3:
                acc_tensor = acc_tensor[..., 0]
            acc = acc_tensor.detach().cpu().float().numpy().astype(np.float32)
            if depth.shape != (cfg.image_height, cfg.image_width):
                raise ValueError(f"Rendered depth for {cfg.view_id} has unexpected shape {depth.shape}.")
            if acc.shape != (cfg.image_height, cfg.image_width):
                raise ValueError(f"Rendered alpha for {cfg.view_id} has unexpected shape {acc.shape}.")
            rendered_depths.append(depth)
            rendered_accs.append(acc)
    return (
        fg_means,
        fg_scales,
        fg_quats,
        fg_opacities,
        fg_colors,
        rendered_depths,
        rendered_accs,
    )


def _gaussian_latent_stats(
    staged: StagedSolveResult,
    motion_fill: GaussianMotionFillResult | None,
    observations: Mapping[str, np.ndarray],
    num_fg: int,
) -> dict[str, Any]:
    stats = latent_stats(staged, motion_fill, observations)
    prepared = staged.prepared
    observable = staged.observable
    obs_count = prepared.obs_count_per_point.astype(np.int32)
    obs_sample_count = prepared.obs_sample_count_per_point.astype(np.int32)
    if "gaussian_indices" not in prepared.arrays:
        raise ValueError("Gaussian observations are missing gaussian_indices.")
    gaussian_indices = prepared.arrays["gaussian_indices"].astype(np.int32)
    stats.update(
        {
            "num_foreground_gaussians": int(num_fg),
            "num_output_points": int(prepared.points.shape[0]),
            "unobserved_output_count": int((obs_count == 0).sum()),
            "obs_count_p50": json_float(np.percentile(obs_count, 50)),
            "obs_count_p90": json_float(np.percentile(obs_count, 90)),
            "obs_sample_count_p50": json_float(np.percentile(obs_sample_count, 50)),
            "obs_sample_count_p90": json_float(np.percentile(obs_sample_count, 90)),
            "gaussian_indices_contiguous": bool(np.array_equal(gaussian_indices, np.arange(gaussian_indices.shape[0], dtype=np.int32))),
            "preserved_all_points": bool(np.asarray(observations["preserved_all_points"]).item()),
            "pixel_candidate_method": str(np.asarray(observations["pixel_candidate_method"]).item()),
        }
    )
    if motion_fill is not None:
        completion = motion_fill.motion.completion_mask.astype(bool)
        partial = observable.partial_mask.astype(bool)
        unobserved = observable.unobserved_mask.astype(bool)
        staged_solver_method = "staged_overlap_observable"
        motion_fill_method = MOTION_FILL_METHOD
        stats.update(
            {
                "staged_solver_method": staged_solver_method,
                "motion_fill_method": motion_fill_method,
                "effective_field_method": f"{staged_solver_method}+{motion_fill_method}",
                "partial_staged_count": int(partial.sum()),
                "partial_unresolved_count": int((partial & ~completion).sum()),
                "completed_observed_count": int((partial & completion).sum()),
                "unobserved_unresolved_count": int((unobserved & ~completion).sum()),
                "completed_unobserved_count": int((unobserved & completion).sum()),
            }
        )
    contribution_weight = observations["obs_contribution_weight"].astype(np.float32)
    contribution_score = observations["obs_contribution_score"].astype(np.float32)
    contribution_sum = observations["obs_contribution_sum"].astype(np.float32)
    stats.update(
        {
            "contribution_weight_p50": json_float(np.percentile(contribution_weight, 50)),
            "contribution_weight_p90": json_float(np.percentile(contribution_weight, 90)),
            "contribution_weight_max": json_float(contribution_weight.max()),
            "contribution_score_p50": json_float(np.percentile(contribution_score, 50)),
            "contribution_score_p90": json_float(np.percentile(contribution_score, 90)),
            "contribution_score_max": json_float(contribution_score.max()),
            "contribution_sum_p50": json_float(np.percentile(contribution_sum, 50)),
            "contribution_sum_p90": json_float(np.percentile(contribution_sum, 90)),
            "contribution_sum_max": json_float(contribution_sum.max()),
        }
    )
    return stats


def _required_scalar_array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    dtype: Any,
) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"Gaussian observations are missing {key}.")
    value = np.asarray(arrays[key])
    if value.shape != ():
        raise ValueError(f"Gaussian observation field {key} must be scalar, got {value.shape}.")
    return value.astype(dtype)


def _write_compact_gaussian_latent(
    out_path: str | Path,
    staged: StagedSolveResult,
    motion_fill: GaussianMotionFillResult | None,
) -> Path:
    prepared = staged.prepared
    num_points = int(prepared.points.shape[0])
    if "gaussian_indices" not in prepared.arrays:
        raise ValueError("Gaussian observations are missing gaussian_indices.")
    gaussian_indices = np.asarray(prepared.arrays["gaussian_indices"])
    if gaussian_indices.shape != (num_points,):
        raise ValueError(
            f"gaussian_indices must have shape ({num_points},), got {gaussian_indices.shape}."
        )
    phi = staged.observable.phi if motion_fill is None else motion_fill.motion.phi
    phi = np.asarray(phi)
    if phi.shape != (num_points, 3):
        raise ValueError(f"Final phi must have shape ({num_points},3), got {phi.shape}.")

    arrays: dict[str, np.ndarray] = {
        "points_world": prepared.points.astype(np.float32),
        "phi": phi.astype(np.complex64),
        "gaussian_indices": gaussian_indices.astype(np.int32),
        "freq_hz": _required_scalar_array(prepared.arrays, "freq_hz", np.float32),
        "mode_index": _required_scalar_array(prepared.arrays, "mode_index", np.int32),
        "obs_count_per_point": prepared.obs_count_per_point.astype(np.int32),
        "point_type": _required_scalar_array(prepared.arrays, "point_type", str),
        "source_checkpoint": _required_scalar_array(
            prepared.arrays, "source_checkpoint", str
        ),
    }
    if motion_fill is not None:
        arrays.update(
            {
                "motion_fill_role": motion_fill.roles.role.astype(np.int8),
                "motion_fill_role_names": np.asarray(MOTION_FILL_ROLE_NAMES),
                "completion_mask": motion_fill.motion.completion_mask.astype(bool),
            }
        )
    return save_npz_compressed_atomic(out_path, arrays)


def _motion_fill_solver_arrays(prefix: str, metadata: Any) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_performed": np.array(bool(metadata.performed)),
        f"{prefix}_converged": np.array(bool(metadata.converged)),
        f"{prefix}_stop_code": np.array(int(metadata.stop_code), dtype=np.int32),
        f"{prefix}_iterations": np.array(int(metadata.iterations), dtype=np.int32),
        f"{prefix}_iterations_max": np.array(
            int(metadata.iterations_max), dtype=np.int32
        ),
        f"{prefix}_residual_norm": np.array(float(metadata.residual_norm), dtype=np.float64),
        f"{prefix}_normal_residual_norm": np.array(
            float(metadata.normal_residual_norm), dtype=np.float64
        ),
        f"{prefix}_matrix_norm": np.array(float(metadata.matrix_norm), dtype=np.float64),
        f"{prefix}_condition_estimate": np.array(
            float(metadata.condition_estimate), dtype=np.float64
        ),
        f"{prefix}_solution_norm": np.array(float(metadata.solution_norm), dtype=np.float64),
    }


def _motion_fill_component_arrays(motion: Any) -> dict[str, np.ndarray]:
    components = motion.component_solvers
    arrays: dict[str, np.ndarray] = {
        "motion_fill_solver_scope": np.array(motion.solver_scope),
        "motion_fill_parallel_channels": np.array(bool(motion.parallel_channels)),
        "motion_fill_solver_component_index": np.asarray(
            [component.component_index for component in components], dtype=np.int32
        ),
        "motion_fill_solver_component_point_count": np.asarray(
            [component.point_count for component in components], dtype=np.int32
        ),
        "motion_fill_solver_component_edge_count": np.asarray(
            [component.edge_count for component in components], dtype=np.int64
        ),
        "motion_fill_solver_component_row_count": np.asarray(
            [component.row_count for component in components], dtype=np.int64
        ),
        "motion_fill_solver_component_column_count": np.asarray(
            [component.column_count for component in components], dtype=np.int64
        ),
    }
    for channel, attribute in (("real", "real_solver"), ("imaginary", "imag_solver")):
        metadata = [getattr(component, attribute) for component in components]
        prefix = f"motion_fill_lsmr_{channel}_component"
        arrays.update(
            {
                f"{prefix}_performed": np.asarray(
                    [item.performed for item in metadata], dtype=bool
                ),
                f"{prefix}_converged": np.asarray(
                    [item.converged for item in metadata], dtype=bool
                ),
                f"{prefix}_stop_code": np.asarray(
                    [item.stop_code for item in metadata], dtype=np.int32
                ),
                f"{prefix}_iterations": np.asarray(
                    [item.iterations for item in metadata], dtype=np.int32
                ),
                f"{prefix}_residual_norm": np.asarray(
                    [item.residual_norm for item in metadata], dtype=np.float64
                ),
                f"{prefix}_normal_residual_norm": np.asarray(
                    [item.normal_residual_norm for item in metadata], dtype=np.float64
                ),
                f"{prefix}_matrix_norm": np.asarray(
                    [item.matrix_norm for item in metadata], dtype=np.float64
                ),
                f"{prefix}_condition_estimate": np.asarray(
                    [item.condition_estimate for item in metadata], dtype=np.float64
                ),
                f"{prefix}_solution_norm": np.asarray(
                    [item.solution_norm for item in metadata], dtype=np.float64
                ),
            }
        )
    return arrays


def _write_solver_diagnostics(
    out_path: str | Path,
    staged: StagedSolveResult,
    motion_fill: GaussianMotionFillResult | None,
    graph: KnnGraph | None = None,
    graph_path: str | None = None,
) -> Path:
    alpha = staged.alpha
    observable = staged.observable
    final_status = (
        observable.point_status
        if motion_fill is None
        else motion_fill.point_solution_status
    )
    final_point_residual = (
        staged.point_residual
        if motion_fill is None
        else motion_fill.point_residual
    )
    final_point_residual_valid = (
        staged.point_residual_valid_mask
        if motion_fill is None
        else motion_fill.point_residual_valid_mask
    )
    arrays: dict[str, np.ndarray] = {
        "alphas": alpha.alphas.astype(np.complex64),
        "alpha_reference_view_index": np.array(0, dtype=np.int32),
        "alpha_view_freqs_hz": staged.alpha_view_freqs_hz.astype(np.float32),
        "alpha_identifiable_mask": alpha.identifiable_mask.astype(bool),
        "alpha_reference_connected_mask": alpha.reference_connected_mask.astype(bool),
        "alpha_exclusion_reason": alpha.exclusion_reason,
        "alpha_shared_point_count": alpha.shared_point_count.astype(np.int32),
        "alpha_edge_point_count": alpha.edge_point_count.astype(np.int32),
        "alpha_edge_information": alpha.edge_information.astype(np.float32),
        "alpha_constraint_count_per_view": alpha.constraint_count_per_view.astype(np.int32),
        "alpha_information_matrix": alpha.information_matrix.astype(np.complex64),
        "alpha_parameter_information": alpha.parameter_information.astype(np.float32),
        "alpha_parameter_view_indices": alpha.parameter_view_indices.astype(np.int32),
        "alpha_parameter_order": np.array(alpha.parameter_order),
        "alpha_singular_values": alpha.singular_values.astype(np.float32),
        "alpha_rank_ratio": np.array(alpha.rank_ratio, dtype=np.float32),
        "alpha_information_ratio": np.array(alpha.information_ratio, dtype=np.float32),
        "alpha_condition": np.array(alpha.condition, dtype=np.float32),
        "alpha_consistency_residual": np.array(
            alpha.consistency_residual, dtype=np.float32
        ),
        "alpha_phase_std": alpha.phase_std.astype(np.float32),
        "alpha_log_gain_std": alpha.log_gain_std.astype(np.float32),
        "alpha_gain_bound_active_mask": alpha.gain_bound_active_mask.astype(bool),
        "alpha_optimizer_success": np.array(alpha.optimizer_success),
        "alpha_optimizer_status": np.array(alpha.optimizer_status, dtype=np.int32),
        "alpha_optimizer_message": np.array(alpha.optimizer_message),
        "alpha_information_kind": np.array(alpha.information_kind),
        "point_singular_values": observable.singular_values.astype(np.float32),
        "point_observable_rank": observable.observable_rank.astype(np.int8),
        "point_nullity": observable.nullity.astype(np.int8),
        "point_condition": observable.condition.astype(np.float32),
        "point_distinct_view_count": staged.prepared.derived_view_count_per_point.astype(
            np.int32
        ),
        "point_distinct_valid_view_count": observable.distinct_valid_view_count.astype(
            np.int32
        ),
        "point_usable_observation_row_count": observable.usable_observation_row_count.astype(
            np.int32
        ),
        "point_precompletion_residual": observable.precompletion_residual.astype(
            np.float32
        ),
        "staged_point_solution_status": observable.point_status.astype(np.int8),
        "final_point_solution_status": np.asarray(final_status, dtype=np.int8),
        "point_solution_status_names": np.asarray(POINT_STATUS_NAMES),
        "point_residual": np.asarray(final_point_residual, dtype=np.float32),
        "point_residual_valid_mask": np.asarray(
            final_point_residual_valid, dtype=bool
        ),
        "obs_sample_count_per_point": staged.prepared.obs_sample_count_per_point.astype(
            np.int32
        ),
        "anchor_residual_threshold": np.array(
            observable.anchor_residual_threshold, dtype=np.float32
        ),
        "anchor_condition_max": np.array(ANCHOR_CONDITION_MAX, dtype=np.float32),
    }
    if motion_fill is not None:
        if graph is None or graph_path is None:
            raise ValueError("Motion-fill diagnostics require graph metadata.")
        motion = motion_fill.motion
        diagnostics = motion_fill.diagnostics
        system = diagnostics["system"]
        arrays.update(
            {
                "phi_nullspace_correction": motion.phi_nullspace_correction.astype(
                    np.complex64
                ),
                "motion_fill_role": motion_fill.roles.role.astype(np.int8),
                "motion_fill_excluded_reason": motion_fill.roles.excluded_reason.astype(
                    np.int8
                ),
                "motion_fill_point_numerical_nullity": motion_fill.numerical_nullity.astype(
                    np.int8
                ),
                "motion_fill_staged_nullity_refined_mask": motion_fill.staged_nullity_refined_mask.astype(
                    bool
                ),
                "completion_mask": motion.completion_mask.astype(bool),
                "completion_connected_to_anchor": motion.completion_connected_to_anchor.astype(
                    bool
                ),
                "point_active_component_index": motion.connectivity.component_index.astype(
                    np.int32
                ),
                "point_anchor_hop_distance": motion.connectivity.hop_distance.astype(
                    np.int32
                ),
                "active_component_sizes": motion.connectivity.component_sizes.astype(
                    np.int32
                ),
                "active_component_has_anchor": motion.connectivity.component_has_anchor.astype(
                    bool
                ),
                "active_component_anchor_count": motion.connectivity.component_anchor_count.astype(
                    np.int32
                ),
                "motion_fill_method": np.array(MOTION_FILL_METHOD),
                "motion_fill_version": np.array(MOTION_FILL_VERSION, dtype=np.int32),
                "motion_fill_numerical_rank_policy": np.array(
                    MOTION_FILL_NUMERICAL_RANK_POLICY
                ),
                "motion_fill_excluded_policy": np.array(
                    "retain_observable_exclude_from_graph"
                ),
                "motion_fill_graph_path": np.array(graph_path),
                "motion_fill_graph_k": np.array(graph.k, dtype=np.int32),
                "motion_fill_graph_max_distance": np.array(
                    graph.max_distance, dtype=np.float64
                ),
                "motion_fill_graph_epsilon": np.array(
                    graph.epsilon, dtype=np.float64
                ),
                "motion_fill_nullspace_operator_max_relative_error": np.array(
                    diagnostics["nullspace_operator_max_relative_error"],
                    dtype=np.float64,
                ),
                "motion_fill_nullspace_operator_rtol": np.array(
                    MOTION_FILL_NULLSPACE_RTOL, dtype=np.float64
                ),
                "motion_fill_observation_drift_max_relative": np.array(
                    diagnostics["observation_drift_max_relative"], dtype=np.float64
                ),
                "motion_fill_observation_drift_rtol": np.array(
                    MOTION_FILL_OBSERVATION_DRIFT_RTOL, dtype=np.float64
                ),
                "motion_fill_relative_denominator_epsilon": np.array(
                    MOTION_FILL_EPSILON, dtype=np.float64
                ),
                "motion_fill_system_row_count": np.array(
                    motion.system_row_count, dtype=np.int64
                ),
                "motion_fill_system_column_count": np.array(
                    motion.system_column_count, dtype=np.int64
                ),
                "motion_fill_active_edge_count": np.array(
                    motion.active_edge_count, dtype=np.int64
                ),
                "motion_fill_eligible_edge_count": np.array(
                    system["eligible_edge_count"], dtype=np.int64
                ),
                "motion_fill_lsmr_atol": np.array(
                    diagnostics["tolerances"]["lsmr_atol"], dtype=np.float64
                ),
                "motion_fill_lsmr_btol": np.array(
                    MOTION_FILL_LSMR_BTOL, dtype=np.float64
                ),
                "motion_fill_lsmr_conlim": np.array(
                    MOTION_FILL_LSMR_CONLIM, dtype=np.float64
                ),
                "motion_fill_lsmr_maxiter": np.array(
                    diagnostics["tolerances"]["lsmr_maxiter"], dtype=np.int64
                ),
                "motion_fill_source_solver_method": np.array(
                    "staged_overlap_observable"
                ),
                **_motion_fill_solver_arrays(
                    "motion_fill_lsmr_real", motion.real_solver
                ),
                **_motion_fill_solver_arrays(
                    "motion_fill_lsmr_imaginary", motion.imag_solver
                ),
                **_motion_fill_component_arrays(motion),
            }
        )
    return save_npz_compressed_atomic(out_path, arrays)


def _print_observation_sanity(
    observations: Mapping[str, np.ndarray],
    num_fg: int,
) -> None:
    obs_count = observations["obs_count_per_point"].astype(np.int32)
    obs_sample_count = observations["obs_sample_count_per_point"].astype(np.int32)
    view_ids = observations["view_ids"].astype(str)
    obs_view_index = observations["obs_view_index"].astype(np.int32)
    per_view = np.bincount(obs_view_index, minlength=view_ids.shape[0])
    per_view_text = ", ".join(f"{view_ids[i]}={int(per_view[i])}" for i in range(view_ids.shape[0]))
    obs_percentiles = np.percentile(obs_count, [0, 50, 90, 100])
    obs_sample_percentiles = np.percentile(obs_sample_count, [0, 50, 90, 100])
    print(
        "Observation sanity: "
        f"points={num_fg}, total_obs={obs_view_index.shape[0]}, per_view=[{per_view_text}], "
        f"view_count p0/p50/p90/max="
        f"{obs_percentiles[0]:.0f}/{obs_percentiles[1]:.0f}/{obs_percentiles[2]:.0f}/{obs_percentiles[3]:.0f}, "
        f"sample_count p0/p50/p90/max="
        f"{obs_sample_percentiles[0]:.0f}/{obs_sample_percentiles[1]:.0f}/"
        f"{obs_sample_percentiles[2]:.0f}/{obs_sample_percentiles[3]:.0f}, "
        f"unobserved={int((obs_count == 0).sum())}"
    )


def run(args: argparse.Namespace) -> None:
    _validate_motion_fill_arguments(args)
    view_configs_paths = list(args.view_config)
    modal_npzs_paths = list(args.modal_npz)
    if len(view_configs_paths) != len(modal_npzs_paths):
        raise ValueError("--view-config and --modal-npz must be supplied the same number of times.")
    
    (
        fg_means,
        fg_scales,
        fg_quats,
        fg_opacities,
        fg_colors,
        rendered_depths,
        rendered_accs,
    ) = _load_fg_pixel_candidate_inputs_from_checkpoint(
        args.input_ckpt,
        view_configs_paths,
    )

    from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

    gaussian_tree = cKDTree(fg_means.astype(np.float64))

    _validate_motion_fill_arguments(args, fg_means.shape[0])
    
    freqs_per_view = load_modal_freqs(modal_npzs_paths)
    # e.g. freqs_per_view = [
    # np.array([0.357, 0.714]),  # view 1
    # np.array([0.359, 0.711]),  # view 2
    # np.array([0.356, 0.716]),  # view 3 ]
    mode_indices = parse_mode_indices(args.mode_indices, int(freqs_per_view[0].shape[0]))
    out_dir = Path(args.out_dir)
    obs_dir = out_dir / "observations"
    latent_dir = out_dir / "latents"
    diagnostics_dir = out_dir / "diagnostics"
    vis_dir = out_dir / "vis"
    obs_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Build KNN graph from gaussian centers
    motion_fill_graph = None
    motion_fill_graph_weighting: GaussianGraphWeightingResult | None = None
    motion_fill_graph_path: Path | None = None
    motion_fill_mode_diagnostics: dict[str, Any] = {}
    if args.motion_fill:
        candidates = query_knn_candidates(
            fg_means,
            int(args.motion_fill_k),
            tree=gaussian_tree,
        )
        spatial_graph = build_knn_graph(
            candidates,
            int(args.motion_fill_k),
            float(args.motion_fill_max_distance),
            MOTION_FILL_EPSILON,
        )
        motion_fill_graph_weighting = weight_gaussian_motion_fill_graph(
            spatial_graph,
            fg_colors,
            str(args.motion_fill_edge_weighting),
            args.motion_fill_color_sigma,
        )
        motion_fill_graph = motion_fill_graph_weighting.graph
        motion_fill_graph_path = write_motion_fill_graph(
            out_dir / "motion_fill" / "graph.npz",
            fg_means,
            candidates,
            motion_fill_graph_weighting,
        )

    modes: list[dict[str, Any]] = []
    for mode_index in mode_indices:
        reference_freq = float(freqs_per_view[0][mode_index])
        mode_name = f"mode_{mode_index:03d}_{freq_slug(reference_freq)}hz"
        obs_path = obs_dir / f"{mode_name}.npz"
        latent_path = latent_dir / f"{mode_name}.npz"
        diagnostics_path = diagnostics_dir / f"{mode_name}.npz"
        mode_vis_dir = vis_dir / mode_name
        print(f"Solving Gaussian mode {mode_name}")
        latent_path.unlink(missing_ok=True)
        build_gaussian_observation_graph(
            points_world=fg_means,
            view_config_paths=view_configs_paths,
            modal_npz_paths=modal_npzs_paths,
            out_path=obs_path,
            source_checkpoint=args.input_ckpt,
            mode_index=mode_index,
            mask_erode_iters=args.mask_erode_iters,
            freq_tolerance_hz=args.freq_tolerance_hz,  #
            pixel_sample_stride=args.pixel_sample_stride,
            pixel_candidate_k=args.pixel_candidate_k,
            pixel_preselect_k=args.pixel_preselect_k,
            pixel_render_acc_min=args.pixel_render_acc_min,
            pixel_min_contribution=args.pixel_min_contribution,
            gaussian_scales=fg_scales,
            gaussian_quats=fg_quats,
            gaussian_opacities=fg_opacities,
            rendered_depths=rendered_depths,
            rendered_accs=rendered_accs,
            gaussian_tree=gaussian_tree,
        )
        with np.load(str(obs_path), allow_pickle=False) as loaded:
            observations = {key: loaded[key] for key in loaded.files}
        _print_observation_sanity(observations, fg_means.shape[0])
        staged = optimize_multi_view_staged(
            observations=observations,
            config=staged_solver_config(args),
        )
        if (
            staged.config.alpha_failure == "error"
            and staged.unidentifiable_observed_view_indices.size
        ):
            _write_solver_diagnostics(diagnostics_path, staged, None)
            enforce_alpha_failure(staged, diagnostics_path)

        motion_fill_result = None
        if motion_fill_graph is not None:
            assert motion_fill_graph_path is not None
            graph_relative_path = relative_path(motion_fill_graph_path, out_dir)
            try:
                motion_fill_result = apply_gaussian_motion_fill(
                    staged,
                    motion_fill_graph,
                    graph_relative_path,
                    lsmr_atol=float(args.motion_fill_lsmr_atol),
                )
            except Exception:
                _write_solver_diagnostics(diagnostics_path, staged, None)
                raise
            motion_fill_mode_diagnostics[mode_name] = motion_fill_result.diagnostics

        if motion_fill_result is None:
            final_phi = staged.observable.phi
            final_prediction = staged.obs_pred_y
            final_residual_valid = staged.obs_residual_valid_mask
        else:
            final_phi = motion_fill_result.motion.phi
            final_prediction = motion_fill_result.obs_pred_y
            final_residual_valid = motion_fill_result.obs_residual_valid_mask
        write_solve_visualizations(
            staged,
            final_phi,
            final_prediction,
            final_residual_valid,
            mode_vis_dir,
        )
        graph_relative_path = (
            relative_path(motion_fill_graph_path, out_dir)
            if motion_fill_graph_path is not None
            else None
        )
        _write_solver_diagnostics(
            diagnostics_path,
            staged,
            motion_fill_result,
            motion_fill_graph,
            graph_relative_path,
        )
        _write_compact_gaussian_latent(latent_path, staged, motion_fill_result)

        mode_stats = _gaussian_latent_stats(
            staged,
            motion_fill_result,
            observations,
            fg_means.shape[0],
        )
        if motion_fill_result is not None:
            mode_stats["motion_fill"] = motion_fill_result.diagnostics
        freqs_by_view = [float(freqs[mode_index]) for freqs in freqs_per_view]
        modes.append(
            {
                "mode_index": int(mode_index),
                "freq_hz": reference_freq,
                "freqs_hz_by_view": freqs_by_view,
                "label": f"{mode_index}: {reference_freq:.6f} Hz",
                "observation_path": relative_path(obs_path, out_dir),
                "latent_path": relative_path(latent_path, out_dir),
                "diagnostics_path": relative_path(diagnostics_path, out_dir),
                "vis_dir": relative_path(mode_vis_dir, out_dir),
                "alpha_by_view": alpha_by_view_diagnostics(staged),
                "stats": mode_stats,
            }
        )

    manifest_parameters = {
        "mask_erode_iters": int(args.mask_erode_iters),
        "pixel_sample_stride": int(args.pixel_sample_stride),
        "pixel_candidate_k": int(args.pixel_candidate_k),
        "pixel_preselect_k": int(args.pixel_preselect_k),
        "pixel_render_acc_min": float(args.pixel_render_acc_min),
        "pixel_min_contribution": float(args.pixel_min_contribution),
        "freq_tolerance_hz": float(args.freq_tolerance_hz),
        "alpha_model": "per_view_per_mode",
        "alpha_reference_view_index": 0,
        "motion_fill_enabled": bool(args.motion_fill),
        **staged_solver_manifest_parameters(args),
    }
    if motion_fill_graph is not None:
        assert motion_fill_graph_path is not None
        assert motion_fill_graph_weighting is not None
        manifest_parameters.update(
            {
                "motion_fill_method": MOTION_FILL_METHOD,
                "motion_fill_version": MOTION_FILL_VERSION,
                "motion_fill_k": int(motion_fill_graph.k),
                "motion_fill_max_distance": float(motion_fill_graph.max_distance),
                "motion_fill_epsilon": float(motion_fill_graph.epsilon),
                "motion_fill_nullspace_operator_rtol": MOTION_FILL_NULLSPACE_RTOL,
                "motion_fill_observation_drift_rtol": MOTION_FILL_OBSERVATION_DRIFT_RTOL,
                "motion_fill_lsmr_atol": float(args.motion_fill_lsmr_atol),
                "motion_fill_lsmr_maxiter_by_mode": {
                    mode_name: int(diagnostics["tolerances"]["lsmr_maxiter"])
                    for mode_name, diagnostics in motion_fill_mode_diagnostics.items()
                },
                "motion_fill_lsmr_maxiter_policy": "global_system_min_dimension",
                "motion_fill_solver_scope": "componentwise",
                "motion_fill_parallel_channels": True,
                "motion_fill_graph_path": relative_path(motion_fill_graph_path, out_dir),
                "motion_fill_excluded_policy": "retain_observable_exclude_from_graph",
                "edge_weighting": motion_fill_graph_weighting.edge_weighting,
                "color_metric": motion_fill_graph_weighting.color_metric,
                "color_sigma": motion_fill_graph_weighting.resolved_color_sigma,
                "color_sigma_source": motion_fill_graph_weighting.color_sigma_source,
            }
        )
        write_motion_fill_diagnostics(
            out_dir / "motion_fill" / "diagnostics.json",
            {
                "version": MOTION_FILL_VERSION,
                "method": MOTION_FILL_METHOD,
                "graph_path": relative_path(motion_fill_graph_path, out_dir),
                "graph": {
                    "k": int(motion_fill_graph.k),
                    "max_distance": float(motion_fill_graph.max_distance),
                    "epsilon": float(motion_fill_graph.epsilon),
                    "edge_weighting": motion_fill_graph_weighting.edge_weighting,
                    "color_metric": motion_fill_graph_weighting.color_metric,
                    "color_sigma": motion_fill_graph_weighting.resolved_color_sigma,
                    "color_sigma_source": motion_fill_graph_weighting.color_sigma_source,
                    "edge_count": int(motion_fill_graph.edge_index.shape[0]),
                    "zero_distance_edge_count": int(
                        np.count_nonzero(motion_fill_graph.edge_distance == 0.0)
                    ),
                    "component_count": int(motion_fill_graph.component_sizes.shape[0]),
                    "isolated_point_count": int(
                        np.count_nonzero(motion_fill_graph.isolated_mask)
                    ),
                },
                "modes": motion_fill_mode_diagnostics,
            },
        )

    manifest = {
        "version": 1,
        "source_checkpoint": str(args.input_ckpt),
        "point_type": "foreground_gaussian_center",
        "source_view_configs": view_configs_paths,
        "source_modal_npzs": modal_npzs_paths,
        "mode_indices": mode_indices,
        "parameters": manifest_parameters,
        "modes": modes,
    }
    manifest_path = out_dir / "modal_modes_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved Gaussian modal modes manifest -> {manifest_path}")
