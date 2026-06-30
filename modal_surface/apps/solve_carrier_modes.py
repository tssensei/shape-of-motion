"""Batch solve VGGT-carrier modal fields for multiple frequency indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from modal_surface.carrier import build_carrier_observation_graph
from modal_surface.optimization_multi import optimize_multi_view


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
    parser.add_argument("--obs-count-weight-1", type=float, default=0.25, help="Data weight multiplier for points observed by one view.")
    parser.add_argument("--obs-count-weight-2", type=float, default=0.75, help="Data weight multiplier for points observed by two views.")
    parser.add_argument("--obs-count-weight-3plus", type=float, default=1.0, help="Data weight multiplier for points observed by three or more views.")


def _load_modal_freqs(paths: list[str]) -> list[np.ndarray]:
    freqs_per_view: list[np.ndarray] = []
    for path in paths:
        z = np.load(str(path), allow_pickle=False)
        required = ["mode_u", "mode_v", "selected_freqs_hz"]
        missing = [key for key in required if key not in z.files]
        if missing:
            raise ValueError(f"{path} missing required modal arrays: {missing}.")
        mode_u = z["mode_u"]
        mode_v = z["mode_v"]
        freqs = z["selected_freqs_hz"].astype(np.float32).reshape(-1)
        if mode_u.ndim != 3 or mode_v.ndim != 3 or mode_u.shape != mode_v.shape:
            raise ValueError(f"{path} must contain mode_u/mode_v with matching shape (K,H,W).")
        if mode_u.shape[0] != freqs.shape[0]:
            raise ValueError(f"{path} selected_freqs_hz length does not match mode_u/mode_v.")
        freqs_per_view.append(freqs)
    if len({freqs.shape[0] for freqs in freqs_per_view}) != 1:
        raise ValueError("All modal npz files must contain the same number of selected frequencies.")
    return freqs_per_view


def _parse_mode_indices(raw: str, num_modes: int) -> list[int]:
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


def _freq_slug(freq_hz: float) -> str:
    text = f"{freq_hz:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _latent_stats(latent_path: Path, observation_path: Path) -> dict[str, Any]:
    latent = np.load(str(latent_path), allow_pickle=False)
    observations = np.load(str(observation_path), allow_pickle=False)
    obs_residual = latent["obs_residual"].astype(np.float32)
    point_residual = latent["point_residual"].astype(np.float32)
    refined_count = None
    if "single_view_refined_mask" in latent.files:
        refined_count = int(latent["single_view_refined_mask"].astype(bool).sum())
    graph_smooth_residual = latent["graph_smooth_residual"].astype(np.float32) if "graph_smooth_residual" in latent.files else None
    return {
        "num_points": int(latent["points_world"].shape[0]),
        "num_observations": int(latent["obs_point_index"].shape[0]),
        "single_view_refined_count": refined_count,
        "graph_edge_count": int(np.asarray(latent["graph_edge_count"]).item()) if "graph_edge_count" in latent.files else None,
        "graph_auto_radius": float(np.asarray(latent["graph_auto_radius"]).item()) if "graph_auto_radius" in latent.files else None,
        "graph_smooth_residual_median": float(np.median(graph_smooth_residual)) if graph_smooth_residual is not None else None,
        "graph_smooth_residual_p90": float(np.percentile(graph_smooth_residual, 90)) if graph_smooth_residual is not None else None,
        "obs_residual_median": float(np.median(obs_residual)),
        "obs_residual_p90": float(np.percentile(obs_residual, 90)),
        "point_residual_median": float(np.median(point_residual)),
        "point_residual_p90": float(np.percentile(point_residual, 90)),
        "observations_per_view": observations["observations_per_view"].astype(int).tolist(),
        "source_mask_candidate_count": int(np.asarray(observations["source_mask_candidate_count"]).item())
        if "source_mask_candidate_count" in observations.files
        else None,
        "source_mask_kept_count": int(np.asarray(observations["source_mask_kept_count"]).item())
        if "source_mask_kept_count" in observations.files
        else None,
    }


def _json_float(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _view_frequency_reliability(observation_path: Path) -> dict[str, Any]:
    observations = np.load(str(observation_path), allow_pickle=False)
    required = [
        "view_ids",
        "view_frequency_weights",
        "view_frequency_snr",
        "view_frequency_signal",
        "view_frequency_noise",
        "view_frequency_bin_hz",
    ]
    missing = [key for key in required if key not in observations.files]
    if missing:
        raise ValueError(f"{observation_path} missing view-frequency reliability fields: {missing}.")
    view_ids = [str(v) for v in observations["view_ids"].tolist()]
    weights = observations["view_frequency_weights"].astype(float)
    snr = observations["view_frequency_snr"].astype(float)
    signal = observations["view_frequency_signal"].astype(float)
    noise = observations["view_frequency_noise"].astype(float)
    bin_hz = observations["view_frequency_bin_hz"].astype(float)
    return {
        "weighting": str(np.asarray(observations["view_frequency_weighting"]).item()),
        "snr_band_hz": float(np.asarray(observations["snr_band_hz"]).item()),
        "snr_exclude_hz": float(np.asarray(observations["snr_exclude_hz"]).item()),
        "snr_good": float(np.asarray(observations["snr_good"]).item()),
        "view_weight_min": float(np.asarray(observations["view_weight_min"]).item()),
        "views": [
            {
                "view_id": view_id,
                "weight": _json_float(weights[idx]),
                "snr": _json_float(snr[idx]),
                "signal": _json_float(signal[idx]),
                "noise": _json_float(noise[idx]),
                "bin_hz": _json_float(bin_hz[idx]),
            }
            for idx, view_id in enumerate(view_ids)
        ],
    }


def run(args: argparse.Namespace) -> None:
    """Solve all requested frequency indices and write a manifest."""
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
        },
        "modes": modes,
    }
    manifest_path = out_dir / "modal_modes_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved modal modes manifest -> {manifest_path}")
