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
    parser.add_argument("--min-observations", type=int, default=1, help="Minimum observed views per carrier point.")
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")
    parser.add_argument("--iterations", type=int, default=8, help="Alternating optimization iterations.")
    parser.add_argument("--ridge-mu", type=float, default=1e-4, help="Per-point ridge regularization.")
    parser.add_argument("--outlier-frac", type=float, default=0.0, help="Fraction of worst residual points dropped per iteration.")


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
    return {
        "num_points": int(latent["points_world"].shape[0]),
        "num_observations": int(latent["obs_point_index"].shape[0]),
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
        )
        optimize_multi_view(
            observations_path=obs_path,
            out_path=latent_path,
            vis_dir=mode_vis_dir,
            iterations=args.iterations,
            ridge_mu=args.ridge_mu,
            outlier_frac=args.outlier_frac,
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
            "min_observations": int(args.min_observations),
            "freq_tolerance_hz": float(args.freq_tolerance_hz),
            "iterations": int(args.iterations),
            "ridge_mu": float(args.ridge_mu),
            "outlier_frac": float(args.outlier_frac),
        },
        "modes": modes,
    }
    manifest_path = out_dir / "modal_modes_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved modal modes manifest -> {manifest_path}")
