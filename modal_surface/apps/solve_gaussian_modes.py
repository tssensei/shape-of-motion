"""Batch solve 3D modal fields directly on foreground 3DGS Gaussians."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modal_surface.carrier import build_points_observation_graph
from modal_surface.io import load_view_config
from modal_surface.optimization_multi import optimize_multi_view
from modal_surface.apps.solve_carrier_modes import (
    _alpha_by_view_diagnostics,
    _freq_slug,
    _json_float,
    _latent_stats,
    _load_modal_freqs,
    _parse_mode_indices,
    _rel,
    _view_frequency_reliability,
)
from flow3d.scene_model import SceneModel


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-ckpt", required=True, help="Static 3DGS checkpoint containing foreground Gaussian centers.")
    parser.add_argument("--view-config", action="append", required=True, help="View JSON config path. Repeat per view.")
    parser.add_argument("--modal-npz", action="append", required=True, help="modal_analysis.npz path. Repeat per view.")
    parser.add_argument("--out-dir", required=True, help="Output directory for observations, latents, vis, and manifest.")
    parser.add_argument("--mode-indices", default="all", help="Comma-separated zero-based mode indices, or 'all'.")
    parser.add_argument(
        "--observation-sampling",
        choices=["gaussian-center", "pixel-candidates", "gaussian-center-contribution"],
        default="gaussian-center",
        help="Observation graph sampling mode.",
    )
    parser.add_argument("--pixel-sample-stride", type=int, default=4, help="Pixel grid stride for pixel-candidates sampling.")
    parser.add_argument("--pixel-candidate-k", type=int, default=4, help="Number of top contribution Gaussians supervised by each sampled pixel.")
    parser.add_argument("--pixel-preselect-k", type=int, default=32, help="Number of 3D nearest Gaussians scored before top-k contribution selection.")
    parser.add_argument("--pixel-render-acc-min", type=float, default=0.05, help="Minimum rendered foreground alpha for sampled modal pixels.")
    parser.add_argument("--pixel-min-contribution", type=float, default=1e-12, help="Minimum unnormalized Gaussian contribution retained for a sampled modal pixel.")
    parser.add_argument("--pixel-min-mode-amp-percentile", type=float, default=0.0, help="Discard sampled modal pixels below this foreground amplitude percentile.")
    parser.add_argument("--pixel-max-samples-per-view", type=int, default=20000, help="Maximum sampled modal pixels per view before candidate expansion.")
    parser.add_argument("--gaussian-contribution-radius", type=int, default=8, help="Pixel radius used to estimate projected Gaussian contribution share.")
    parser.add_argument("--gaussian-contribution-min-share", type=float, default=1e-4, help="Minimum normalized contribution share for gaussian-center-contribution observations.")
    parser.add_argument("--gaussian-contribution-min-score", type=float, default=1e-12, help="Minimum unnormalized projected Gaussian contribution score.")
    parser.add_argument("--gaussian-contribution-cov-eps-px", type=float, default=0.25, help="2D covariance diagonal epsilon in pixels for projected Gaussian contribution.")
    parser.add_argument("--mask-erode-iters", type=int, default=1, help="3x3 modal mask erosion iterations.")
    parser.add_argument("--zbuffer-radius", type=int, default=5, help="Local robust z-buffer window radius in pixels.")
    parser.add_argument("--zbuffer-mode", choices=["hard", "soft"], default="hard", help="Hard z-buffer filtering or soft z-buffer confidence weighting.")
    parser.add_argument("--zbuffer-soft-sigma", type=float, default=0.10, help="Normalized depth sigma used by --zbuffer-mode soft.")
    parser.add_argument("--zbuffer-soft-min-weight", type=float, default=0.05, help="Minimum z-buffer weight retained by --zbuffer-mode soft.")
    parser.add_argument("--front-percentile", type=float, default=10.0, help="Local depth percentile treated as front surface.")
    parser.add_argument("--zbuffer-tau", type=float, default=0.05, help="Relative depth threshold against local front depth.")
    parser.add_argument("--min-zbuffer-samples", type=int, default=5, help="Minimum local point depths for visibility.")
    parser.add_argument("--view-frequency-weighting", choices=["none", "local-snr"], default="none", help="View-frequency reliability weighting method.")
    parser.add_argument("--snr-band-hz", type=float, default=0.3, help="Half-width of the local spectrum band used for local-SNR noise estimation.")
    parser.add_argument("--snr-exclude-hz", type=float, default=0.08, help="Half-width around the selected frequency excluded from local-SNR noise estimation.")
    parser.add_argument("--snr-good", type=float, default=3.0, help="SNR treated as a clear frequency peak for view-frequency weighting.")
    parser.add_argument("--view-weight-min", type=float, default=0.05, help="Minimum view-frequency reliability weight.")
    parser.add_argument("--depth-weighting", choices=["none", "inverse-z"], default="none", help="Depth-based observation weighting method.")
    parser.add_argument("--depth-weight-power", type=float, default=2.0, help="Power used by inverse-z depth weighting.")
    parser.add_argument("--depth-weight-min", type=float, default=0.02, help="Minimum inverse-z depth weight.")
    parser.add_argument("--depth-weight-reference-percentile", type=float, default=50.0, help="Per-view candidate-depth percentile used as inverse-z reference.")
    parser.add_argument("--pair-weight", action="append", default=None, help="Weight for points observed by exactly two views, formatted as viewA,viewB,weight. Repeat per pair.")
    parser.add_argument("--min-observations", type=int, default=1, help="Minimum observed views whose observations are used for solving.")
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")
    parser.add_argument("--iterations", type=int, default=8, help="Alternating optimization iterations.")
    parser.add_argument("--ridge-mu", type=float, default=1e-4, help="Per-point ridge regularization.")
    parser.add_argument("--outlier-frac", type=float, default=0.0, help="Fraction of worst residual points dropped per iteration.")
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
    parser.add_argument("--modal-rigid-lambda", type=float, default=0.0, help="Local full-vector modal consensus weight.")
    parser.add_argument("--modal-rigid-k", type=int, default=8, help="Nearest-neighbor count used for local modal consensus edges.")
    parser.add_argument("--modal-rigid-auto-radius-scale", type=float, default=2.0, help="Multiplier on median kth-neighbor distance for local modal consensus edge pruning.")
    parser.add_argument("--modal-rigid-min-shared-views", type=int, default=0, help="Minimum shared observed views required for a local modal consensus edge.")
    parser.add_argument("--modal-fill-unobserved", action="store_true", help="Propagate solved modal motion from observed Gaussians to unobserved Gaussians.")
    parser.add_argument("--modal-fill-k", type=int, default=4, help="Nearest-neighbor count used for modal motion propagation.")
    parser.add_argument("--modal-fill-auto-radius-scale", type=float, default=1.0, help="Multiplier on median neighbor distance for modal motion propagation edges.")
    parser.add_argument("--modal-fill-anchor-min-observations", type=int, default=1, help="Minimum observation count for Gaussians used as modal propagation anchors.")
    parser.add_argument("--modal-fill-ridge-mu", type=float, default=1e-6, help="Ridge regularization for propagated modal motion.")
    parser.add_argument("--obs-count-weight-1", type=float, default=0.25, help="Data weight multiplier for points observed by one view.")
    parser.add_argument("--obs-count-weight-2", type=float, default=0.75, help="Data weight multiplier for points observed by two views.")
    parser.add_argument("--obs-count-weight-3plus", type=float, default=1.0, help="Data weight multiplier for points observed by three or more views.")


def _load_fg_means_from_checkpoint(path: str) -> np.ndarray:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state")
    if "fg.params.means" not in state:
        raise ValueError(f"{path} is missing fg.params.means")
    means = state["fg.params.means"].detach().cpu().float().numpy().astype(np.float32)
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"fg.params.means must have shape (N,3), got {means.shape}")
    finite = np.all(np.isfinite(means), axis=1)
    if not np.all(finite):
        bad = int((~finite).sum())
        raise ValueError(f"{path} contains {bad} non-finite foreground Gaussian centers")
    return means


def _load_fg_contribution_inputs_from_checkpoint(
    path: str,
    view_config_paths: list[str],
    render_views: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SceneModel.init_from_state_dict(state).to(device)
    model.eval()
    fg_means = model.fg.params["means"].detach().cpu().float().numpy().astype(np.float32)
    fg_scales = model.fg.get_scales().detach().cpu().float().numpy().astype(np.float32)
    fg_quats = model.fg.get_quats().detach().cpu().float().numpy().astype(np.float32)
    fg_opacities = model.fg.get_opacities().detach().cpu().float().numpy().reshape(-1).astype(np.float32)
    if fg_means.ndim != 2 or fg_means.shape[1] != 3:
        raise ValueError(f"fg.params.means must have shape (N,3), got {fg_means.shape}")
    if fg_scales.shape != fg_means.shape:
        raise ValueError(f"activated foreground scales must have shape {fg_means.shape}, got {fg_scales.shape}")
    if fg_quats.shape != (fg_means.shape[0], 4):
        raise ValueError(f"activated foreground quats must have shape ({fg_means.shape[0]},4), got {fg_quats.shape}")
    if fg_opacities.shape != (fg_means.shape[0],):
        raise ValueError(f"activated foreground opacities must have shape ({fg_means.shape[0]},), got {fg_opacities.shape}")
    if not np.all(np.isfinite(fg_means)):
        raise ValueError(f"{path} contains non-finite foreground Gaussian centers")

    rendered_depths: list[np.ndarray] = []
    rendered_accs: list[np.ndarray] = []
    if render_views:
        with torch.no_grad():
            for cfg_path in view_config_paths:
                cfg = load_view_config(cfg_path)
                w2c = torch.from_numpy(cfg.world_to_camera).to(device=device, dtype=torch.float32)[None]
                K = torch.from_numpy(cfg.K).to(device=device, dtype=torch.float32)[None]
                rendered = model.render(
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
    return fg_means, fg_scales, fg_quats, fg_opacities, rendered_depths, rendered_accs


def _gaussian_latent_stats(latent_path: Path, observation_path: Path, num_fg: int) -> dict[str, Any]:
    stats = _latent_stats(latent_path, observation_path)
    latent = np.load(str(latent_path), allow_pickle=False)
    observations = np.load(str(observation_path), allow_pickle=False)
    obs_count = latent["obs_count_per_point"].astype(np.int32)
    obs_sample_count = (
        latent["obs_sample_count_per_point"].astype(np.int32)
        if "obs_sample_count_per_point" in latent.files
        else obs_count
    )
    gaussian_indices = latent["gaussian_indices"].astype(np.int32)
    stats.update(
        {
            "num_foreground_gaussians": int(num_fg),
            "num_output_points": int(latent["points_world"].shape[0]),
            "unobserved_output_count": int((obs_count == 0).sum()),
            "obs_count_p50": _json_float(np.percentile(obs_count, 50)),
            "obs_count_p90": _json_float(np.percentile(obs_count, 90)),
            "obs_sample_count_p50": _json_float(np.percentile(obs_sample_count, 50)),
            "obs_sample_count_p90": _json_float(np.percentile(obs_sample_count, 90)),
            "gaussian_indices_contiguous": bool(np.array_equal(gaussian_indices, np.arange(gaussian_indices.shape[0], dtype=np.int32))),
            "preserved_all_points": bool(np.asarray(observations["preserved_all_points"]).item())
            if "preserved_all_points" in observations.files
            else False,
            "observation_sampling": str(np.asarray(observations["observation_sampling"]).item())
            if "observation_sampling" in observations.files
            else "gaussian-center",
        }
    )
    if "pixel_candidate_method" in observations.files:
        stats["pixel_candidate_method"] = str(np.asarray(observations["pixel_candidate_method"]).item())
    if "gaussian_candidate_method" in observations.files:
        stats["gaussian_candidate_method"] = str(np.asarray(observations["gaussian_candidate_method"]).item())
    if "obs_contribution_weight" in observations.files:
        contribution_weight = observations["obs_contribution_weight"].astype(np.float32)
        contribution_score = observations["obs_contribution_score"].astype(np.float32)
        contribution_sum = observations["obs_contribution_sum"].astype(np.float32)
        stats.update(
            {
                "contribution_weight_p50": _json_float(np.percentile(contribution_weight, 50)),
                "contribution_weight_p90": _json_float(np.percentile(contribution_weight, 90)),
                "contribution_weight_max": _json_float(contribution_weight.max()),
                "contribution_score_p50": _json_float(np.percentile(contribution_score, 50)),
                "contribution_score_p90": _json_float(np.percentile(contribution_score, 90)),
                "contribution_score_max": _json_float(contribution_score.max()),
                "contribution_sum_p50": _json_float(np.percentile(contribution_sum, 50)),
                "contribution_sum_p90": _json_float(np.percentile(contribution_sum, 90)),
                "contribution_sum_max": _json_float(contribution_sum.max()),
            }
        )
    if "modal_rigid_edge_count" in latent.files:
        modal_rigid_degree = latent["modal_rigid_degree"].astype(np.int32)
        modal_rigid_residual = latent["modal_rigid_residual"].astype(np.float32)
        stats.update(
            {
                "modal_rigid_edge_count": int(np.asarray(latent["modal_rigid_edge_count"]).item()),
                "modal_rigid_degree_p50": _json_float(np.percentile(modal_rigid_degree, 50)),
                "modal_rigid_degree_p90": _json_float(np.percentile(modal_rigid_degree, 90)),
                "modal_rigid_degree_max": int(modal_rigid_degree.max()) if modal_rigid_degree.size else 0,
                "modal_rigid_residual_median": _json_float(np.median(modal_rigid_residual)),
                "modal_rigid_residual_p90": _json_float(np.percentile(modal_rigid_residual, 90)),
            }
        )
    if "modal_fill_enabled" in latent.files and bool(np.asarray(latent["modal_fill_enabled"]).item()):
        modal_fill_degree = latent["modal_fill_degree"].astype(np.int32)
        modal_fill_target_mask = latent["modal_fill_target_mask"].astype(bool)
        modal_fill_connected = latent["modal_fill_connected_to_anchor"].astype(bool)
        target_degree = modal_fill_degree[modal_fill_target_mask]
        stats.update(
            {
                "modal_fill_anchor_count": int(np.asarray(latent["modal_fill_anchor_count"]).item()),
                "modal_fill_target_count": int(np.asarray(latent["modal_fill_target_count"]).item()),
                "modal_fill_filled_count": int(np.asarray(latent["modal_fill_filled_count"]).item()),
                "modal_fill_unfilled_count": int(np.asarray(latent["modal_fill_unfilled_count"]).item()),
                "modal_fill_auto_radius": _json_float(np.asarray(latent["modal_fill_auto_radius"]).item()),
                "modal_fill_degree_p50": _json_float(np.percentile(target_degree, 50)) if target_degree.size else 0.0,
                "modal_fill_degree_p90": _json_float(np.percentile(target_degree, 90)) if target_degree.size else 0.0,
                "modal_fill_degree_max": int(target_degree.max()) if target_degree.size else 0,
                "modal_fill_connected_fraction": _json_float(
                    float((modal_fill_target_mask & modal_fill_connected).sum()) / max(int(modal_fill_target_mask.sum()), 1)
                ),
            }
        )
    return stats


def _print_observation_sanity(obs_path: Path, num_fg: int) -> None:
    observations = np.load(str(obs_path), allow_pickle=False)
    obs_count = observations["obs_count_per_point"].astype(np.int32)
    obs_sample_count = (
        observations["obs_sample_count_per_point"].astype(np.int32)
        if "obs_sample_count_per_point" in observations.files
        else obs_count
    )
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
    view_configs = list(args.view_config)
    modal_npzs = list(args.modal_npz)
    if len(view_configs) != len(modal_npzs):
        raise ValueError("--view-config and --modal-npz must be supplied the same number of times.")
    if float(args.outlier_frac) != 0.0:
        raise ValueError("solve-gaussian-modes preserves foreground Gaussian order and requires --outlier-frac 0.0")

    if args.observation_sampling in {"pixel-candidates", "gaussian-center-contribution"}:
        (
            fg_means,
            fg_scales,
            fg_quats,
            fg_opacities,
            rendered_depths,
            rendered_accs,
        ) = _load_fg_contribution_inputs_from_checkpoint(
            args.input_ckpt,
            view_configs,
            render_views=args.observation_sampling == "pixel-candidates",
        )
        if args.observation_sampling == "gaussian-center-contribution":
            rendered_depths = None
            rendered_accs = None
    else:
        fg_means = _load_fg_means_from_checkpoint(args.input_ckpt)
        fg_scales = None
        fg_quats = None
        fg_opacities = None
        rendered_depths = None
        rendered_accs = None
    gaussian_indices = np.arange(fg_means.shape[0], dtype=np.int32)
    freqs_per_view = _load_modal_freqs(modal_npzs)
    mode_indices = _parse_mode_indices(args.mode_indices, int(freqs_per_view[0].shape[0]))
    out_dir = Path(args.out_dir)
    obs_dir = out_dir / "observations"
    latent_dir = out_dir / "latents"
    vis_dir = out_dir / "vis"
    obs_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    modes: list[dict[str, Any]] = []
    for mode_index in mode_indices:
        reference_freq = float(freqs_per_view[0][mode_index])
        mode_name = f"mode_{mode_index:03d}_{_freq_slug(reference_freq)}hz"
        obs_path = obs_dir / f"{mode_name}.npz"
        latent_path = latent_dir / f"{mode_name}.npz"
        mode_vis_dir = vis_dir / mode_name
        print(f"Solving Gaussian carrier {mode_name}")
        build_points_observation_graph(
            points_world=fg_means,
            view_config_paths=view_configs,
            modal_npz_paths=modal_npzs,
            out_path=obs_path,
            mode_index=mode_index,
            mask_erode_iters=args.mask_erode_iters,
            zbuffer_radius=args.zbuffer_radius,
            zbuffer_mode=args.zbuffer_mode,
            zbuffer_soft_sigma=args.zbuffer_soft_sigma,
            zbuffer_soft_min_weight=args.zbuffer_soft_min_weight,
            front_percentile=args.front_percentile,
            zbuffer_tau=args.zbuffer_tau,
            min_zbuffer_samples=args.min_zbuffer_samples,
            min_observations=args.min_observations,
            freq_tolerance_hz=args.freq_tolerance_hz,
            view_frequency_weighting=args.view_frequency_weighting,
            snr_band_hz=args.snr_band_hz,
            snr_exclude_hz=args.snr_exclude_hz,
            snr_good=args.snr_good,
            view_weight_min=args.view_weight_min,
            depth_weighting=args.depth_weighting,
            depth_weight_power=args.depth_weight_power,
            depth_weight_min=args.depth_weight_min,
            depth_weight_reference_percentile=args.depth_weight_reference_percentile,
            pair_weight_specs=args.pair_weight,
            preserve_all_points=True,
            optional_point_fields={"gaussian_indices": gaussian_indices},
            extra_metadata={
                "point_type": np.array("foreground_gaussian_center"),
                "source_checkpoint": np.array(str(args.input_ckpt)),
            },
            observation_sampling=args.observation_sampling,
            pixel_sample_stride=args.pixel_sample_stride,
            pixel_candidate_k=args.pixel_candidate_k,
            pixel_preselect_k=args.pixel_preselect_k,
            pixel_render_acc_min=args.pixel_render_acc_min,
            pixel_min_contribution=args.pixel_min_contribution,
            pixel_min_mode_amp_percentile=args.pixel_min_mode_amp_percentile,
            pixel_max_samples_per_view=args.pixel_max_samples_per_view,
            gaussian_contribution_radius=args.gaussian_contribution_radius,
            gaussian_contribution_min_share=args.gaussian_contribution_min_share,
            gaussian_contribution_min_score=args.gaussian_contribution_min_score,
            gaussian_contribution_cov_eps_px=args.gaussian_contribution_cov_eps_px,
            gaussian_scales=fg_scales,
            gaussian_quats=fg_quats,
            gaussian_opacities=fg_opacities,
            rendered_depths=rendered_depths,
            rendered_accs=rendered_accs,
        )
        _print_observation_sanity(obs_path, fg_means.shape[0])
        optimize_multi_view(
            observations_path=obs_path,
            out_path=latent_path,
            vis_dir=mode_vis_dir,
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
            modal_rigid_lambda=args.modal_rigid_lambda,
            modal_rigid_k=args.modal_rigid_k,
            modal_rigid_auto_radius_scale=args.modal_rigid_auto_radius_scale,
            modal_rigid_min_shared_views=args.modal_rigid_min_shared_views,
            modal_fill_unobserved=args.modal_fill_unobserved,
            modal_fill_k=args.modal_fill_k,
            modal_fill_auto_radius_scale=args.modal_fill_auto_radius_scale,
            modal_fill_anchor_min_observations=args.modal_fill_anchor_min_observations,
            modal_fill_ridge_mu=args.modal_fill_ridge_mu,
            obs_count_weight_1=args.obs_count_weight_1,
            obs_count_weight_2=args.obs_count_weight_2,
            obs_count_weight_3plus=args.obs_count_weight_3plus,
        )
        latent_stats = _gaussian_latent_stats(latent_path, obs_path, fg_means.shape[0])
        if float(args.modal_rigid_lambda) > 0:
            print(
                "Modal rigidity: "
                f"edges={latent_stats['modal_rigid_edge_count']}, "
                f"degree p50/p90/max={latent_stats['modal_rigid_degree_p50']:.0f}/"
                f"{latent_stats['modal_rigid_degree_p90']:.0f}/{latent_stats['modal_rigid_degree_max']}, "
                f"residual median/p90={latent_stats['modal_rigid_residual_median']:.3g}/"
                f"{latent_stats['modal_rigid_residual_p90']:.3g}"
            )
        if bool(args.modal_fill_unobserved):
            print(
                "Modal fill: "
                f"anchors={latent_stats['modal_fill_anchor_count']}, "
                f"targets={latent_stats['modal_fill_target_count']}, "
                f"filled={latent_stats['modal_fill_filled_count']}, "
                f"unfilled={latent_stats['modal_fill_unfilled_count']}, "
                f"degree p50/p90/max={latent_stats['modal_fill_degree_p50']:.0f}/"
                f"{latent_stats['modal_fill_degree_p90']:.0f}/{latent_stats['modal_fill_degree_max']}"
            )
        freqs_by_view = [float(freqs[mode_index]) for freqs in freqs_per_view]
        modes.append(
            {
                "mode_index": int(mode_index),
                "freq_hz": reference_freq,
                "freqs_hz_by_view": freqs_by_view,
                "label": f"{mode_index}: {reference_freq:.6f} Hz",
                "observation_path": _rel(obs_path, out_dir),
                "latent_path": _rel(latent_path, out_dir),
                "vis_dir": _rel(mode_vis_dir, out_dir),
                "alpha_by_view": _alpha_by_view_diagnostics(latent_path),
                "view_frequency_reliability": _view_frequency_reliability(obs_path),
                "stats": latent_stats,
            }
        )

    manifest = {
        "version": 1,
        "source_checkpoint": str(args.input_ckpt),
        "point_type": "foreground_gaussian_center",
        "source_view_configs": view_configs,
        "source_modal_npzs": modal_npzs,
        "mode_indices": mode_indices,
        "parameters": {
            "mask_erode_iters": int(args.mask_erode_iters),
            "observation_sampling": str(args.observation_sampling),
            "pixel_sample_stride": int(args.pixel_sample_stride),
            "pixel_candidate_k": int(args.pixel_candidate_k),
            "pixel_preselect_k": int(args.pixel_preselect_k),
            "pixel_render_acc_min": float(args.pixel_render_acc_min),
            "pixel_min_contribution": float(args.pixel_min_contribution),
            "pixel_min_mode_amp_percentile": float(args.pixel_min_mode_amp_percentile),
            "pixel_max_samples_per_view": int(args.pixel_max_samples_per_view),
            "gaussian_contribution_radius": int(args.gaussian_contribution_radius),
            "gaussian_contribution_min_share": float(args.gaussian_contribution_min_share),
            "gaussian_contribution_min_score": float(args.gaussian_contribution_min_score),
            "gaussian_contribution_cov_eps_px": float(args.gaussian_contribution_cov_eps_px),
            "zbuffer_radius": int(args.zbuffer_radius),
            "zbuffer_mode": str(args.zbuffer_mode),
            "zbuffer_soft_sigma": float(args.zbuffer_soft_sigma),
            "zbuffer_soft_min_weight": float(args.zbuffer_soft_min_weight),
            "front_percentile": float(args.front_percentile),
            "zbuffer_tau": float(args.zbuffer_tau),
            "min_zbuffer_samples": int(args.min_zbuffer_samples),
            "view_frequency_weighting": str(args.view_frequency_weighting),
            "snr_band_hz": float(args.snr_band_hz),
            "snr_exclude_hz": float(args.snr_exclude_hz),
            "snr_good": float(args.snr_good),
            "view_weight_min": float(args.view_weight_min),
            "depth_weighting": str(args.depth_weighting),
            "depth_weight_power": float(args.depth_weight_power),
            "depth_weight_min": float(args.depth_weight_min),
            "depth_weight_reference_percentile": float(args.depth_weight_reference_percentile),
            "pair_weight": list(args.pair_weight or []),
            "min_observations": int(args.min_observations),
            "freq_tolerance_hz": float(args.freq_tolerance_hz),
            "iterations": int(args.iterations),
            "ridge_mu": float(args.ridge_mu),
            "outlier_frac": float(args.outlier_frac),
            "single_view_smooth_lambda": float(args.single_view_smooth_lambda),
            "single_view_smooth_k": int(args.single_view_smooth_k),
            "single_view_anchor_min_observations": int(args.single_view_anchor_min_observations),
            "graph_smooth_lambda": float(args.graph_smooth_lambda),
            "graph_smooth_k": int(args.graph_smooth_k),
            "graph_auto_radius_scale": float(args.graph_auto_radius_scale),
            "graph_min_shared_views": int(args.graph_min_shared_views),
            "modal_rigid_lambda": float(args.modal_rigid_lambda),
            "modal_rigid_k": int(args.modal_rigid_k),
            "modal_rigid_auto_radius_scale": float(args.modal_rigid_auto_radius_scale),
            "modal_rigid_min_shared_views": int(args.modal_rigid_min_shared_views),
            "modal_fill_unobserved": bool(args.modal_fill_unobserved),
            "modal_fill_k": int(args.modal_fill_k),
            "modal_fill_auto_radius_scale": float(args.modal_fill_auto_radius_scale),
            "modal_fill_anchor_min_observations": int(args.modal_fill_anchor_min_observations),
            "modal_fill_ridge_mu": float(args.modal_fill_ridge_mu),
            "obs_count_weight_1": float(args.obs_count_weight_1),
            "obs_count_weight_2": float(args.obs_count_weight_2),
            "obs_count_weight_3plus": float(args.obs_count_weight_3plus),
            "alpha_model": "per_view_per_mode",
            "alpha_reference_view_index": 0,
        },
        "modes": modes,
    }
    manifest_path = out_dir / "modal_modes_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved Gaussian modal modes manifest -> {manifest_path}")
