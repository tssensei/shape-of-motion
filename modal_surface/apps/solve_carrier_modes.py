"""Batch solve VGGT-carrier modal fields for multiple frequency indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from modal_surface.apps._shared import (
    _alpha_by_view_diagnostics,
    _freq_slug,
    _latent_stats,
    _load_modal_freqs,
    _parse_mode_indices,
    _rel,
    _view_frequency_reliability,
)
from modal_surface.carrier import build_carrier_observation_graph
from modal_surface.optimization_multi import optimize_multi_view
from modal_surface.solver_cli import (
    add_solver_arguments,
    solver_kwargs,
    staged_solver_manifest_parameters,
    validate_solver_args,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register solve-carrier-modes command-line arguments."""
    parser.add_argument("--carrier-points", required=True, help="VGGT carrier points .npz path.")
    parser.add_argument("--view-config", action="append", required=True, help="View JSON config path. Repeat per view.")
    parser.add_argument("--modal-npz", action="append", required=True, help="modal_analysis.npz path. Repeat per view.")
    parser.add_argument("--out-dir", required=True, help="Output directory for observations, latents, vis, and manifest.")
    parser.add_argument("--mode-indices", default="all", help="Comma-separated zero-based mode indices, or 'all'.")
    parser.add_argument("--mask-erode-iters", type=int, default=1, help="3x3 modal mask erosion iterations.")
    parser.add_argument("--source-mask-erode-iters", type=int, default=1, help="3x3 erosion iterations for filtering VGGT carrier points by their source-view mask.")
    parser.add_argument("--zbuffer-radius", type=int, default=5, help="Local robust z-buffer window radius in pixels.")
    parser.add_argument("--front-percentile", type=float, default=10.0, help="Local depth percentile treated as front surface.")
    parser.add_argument("--zbuffer-tau", type=float, default=0.05, help="Relative depth threshold against local front depth.")
    parser.add_argument("--min-zbuffer-samples", type=int, default=5, help="Minimum local carrier depths for visibility.")
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
    parser.add_argument("--min-observations", type=int, default=1, help="Minimum observed views per carrier point.")
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")
    add_solver_arguments(parser)


def run(args: argparse.Namespace) -> None:
    """Solve all requested frequency indices and write a manifest."""
    validate_solver_args(args)
    view_configs = list(args.view_config)
    modal_npzs = list(args.modal_npz)
    if len(view_configs) != len(modal_npzs):
        raise ValueError("--view-config and --modal-npz must be supplied the same number of times.")

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
        print(f"Solving {mode_name}")
        build_carrier_observation_graph(
            carrier_points_path=args.carrier_points,
            view_config_paths=view_configs,
            modal_npz_paths=modal_npzs,
            out_path=obs_path,
            mode_index=mode_index,
            mask_erode_iters=args.mask_erode_iters,
            source_mask_erode_iters=args.source_mask_erode_iters,
            zbuffer_radius=args.zbuffer_radius,
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
        )
        optimize_multi_view(
            observations_path=obs_path,
            out_path=latent_path,
            vis_dir=mode_vis_dir,
            **solver_kwargs(args),
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
                "stats": _latent_stats(latent_path, obs_path),
            }
        )

    manifest = {
        "version": 1,
        "source_carrier_points": str(args.carrier_points),
        "source_view_configs": view_configs,
        "source_modal_npzs": modal_npzs,
        "mode_indices": mode_indices,
        "parameters": {
            "legacy_solver_parameters_active": bool(args.solver == "legacy-als"),
            "mask_erode_iters": int(args.mask_erode_iters),
            "source_mask_erode_iters": int(args.source_mask_erode_iters),
            "zbuffer_radius": int(args.zbuffer_radius),
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
            "obs_count_weight_1": float(args.obs_count_weight_1),
            "obs_count_weight_2": float(args.obs_count_weight_2),
            "obs_count_weight_3plus": float(args.obs_count_weight_3plus),
            "alpha_model": "per_view_per_mode",
            "alpha_reference_view_index": 0,
            **staged_solver_manifest_parameters(args),
        },
        "modes": modes,
    }
    manifest_path = out_dir / "modal_modes_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved modal modes manifest -> {manifest_path}")
