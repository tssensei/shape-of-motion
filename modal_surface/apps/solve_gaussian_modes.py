"""Batch solve 3D modal fields directly on foreground 3DGS Gaussians."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modal_surface.carrier import build_points_observation_graph
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


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-ckpt", required=True, help="Static 3DGS checkpoint containing foreground Gaussian centers.")
    parser.add_argument("--view-config", action="append", required=True, help="View JSON config path. Repeat per view.")
    parser.add_argument("--modal-npz", action="append", required=True, help="modal_analysis.npz path. Repeat per view.")
    parser.add_argument("--out-dir", required=True, help="Output directory for observations, latents, vis, and manifest.")
    parser.add_argument("--mode-indices", default="all", help="Comma-separated zero-based mode indices, or 'all'.")
    parser.add_argument(
        "--observation-sampling",
        choices=["gaussian-center", "pixel-candidates"],
        default="gaussian-center",
        help="Observation graph sampling mode.",
    )
    parser.add_argument("--pixel-sample-stride", type=int, default=4, help="Pixel grid stride for pixel-candidates sampling.")
    parser.add_argument("--pixel-candidate-k", type=int, default=4, help="Number of nearby projected Gaussians supervised by each sampled pixel.")
    parser.add_argument("--pixel-search-radius", type=int, default=6, help="Search radius in pixels for pixel-candidates sampling.")
    parser.add_argument("--pixel-weight-sigma", type=float, default=3.0, help="Gaussian pixel-distance weighting sigma for pixel-candidates observations.")
    parser.add_argument("--pixel-min-mode-amp-percentile", type=float, default=50.0, help="Discard sampled modal pixels below this foreground amplitude percentile.")
    parser.add_argument("--pixel-max-samples-per-view", type=int, default=20000, help="Maximum sampled modal pixels per view before candidate expansion.")
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
    parser.add_argument("--modal-rigid-lambda", type=float, default=0.0, help="Edge-stretch modal rigidity weight.")
    parser.add_argument("--modal-rigid-k", type=int, default=8, help="Nearest-neighbor count used for modal rigidity candidate edges.")
    parser.add_argument("--modal-rigid-auto-radius-scale", type=float, default=2.0, help="Multiplier on median kth-neighbor distance for modal rigidity edge pruning.")
    parser.add_argument("--modal-rigid-min-shared-views", type=int, default=1, help="Minimum shared observed views required for a modal rigidity edge.")
    parser.add_argument("--modal-rigid-motion-cos-min", type=float, default=0.3, help="Minimum initial modal motion cosine similarity required for a modal rigidity edge.")
    parser.add_argument("--modal-rigid-bootstrap-iterations", type=int, default=3, help="No-rigidity ALS iterations used to initialize modal rigidity edge compatibility.")
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
                "modal_rigid_motion_cos_p50": _json_float(np.asarray(latent["modal_rigid_motion_cos_p50"]).item()),
                "modal_rigid_motion_cos_p90": _json_float(np.asarray(latent["modal_rigid_motion_cos_p90"]).item()),
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

    fg_means = _load_fg_means_from_checkpoint(args.input_ckpt)
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
            pixel_search_radius=args.pixel_search_radius,
            pixel_weight_sigma=args.pixel_weight_sigma,
            pixel_min_mode_amp_percentile=args.pixel_min_mode_amp_percentile,
            pixel_max_samples_per_view=args.pixel_max_samples_per_view,
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
            modal_rigid_motion_cos_min=args.modal_rigid_motion_cos_min,
            modal_rigid_bootstrap_iterations=args.modal_rigid_bootstrap_iterations,
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
                f"motion_cos p50/p90={latent_stats['modal_rigid_motion_cos_p50']:.3g}/"
                f"{latent_stats['modal_rigid_motion_cos_p90']:.3g}"
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
            "pixel_search_radius": int(args.pixel_search_radius),
            "pixel_weight_sigma": float(args.pixel_weight_sigma),
            "pixel_min_mode_amp_percentile": float(args.pixel_min_mode_amp_percentile),
            "pixel_max_samples_per_view": int(args.pixel_max_samples_per_view),
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
            "modal_rigid_motion_cos_min": float(args.modal_rigid_motion_cos_min),
            "modal_rigid_bootstrap_iterations": int(args.modal_rigid_bootstrap_iterations),
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
