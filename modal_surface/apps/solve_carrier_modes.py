"""Batch solve VGGT-carrier modal fields for multiple frequency indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from modal_surface.apps._shared import (
    _alpha_by_view_diagnostics,
    _freq_slug,
    _latent_stats,
    _load_modal_freqs,
    _parse_mode_indices,
    _rel,
)
from modal_surface.carrier import build_carrier_observation_graph
from modal_surface.optimization_staged import optimize_multi_view_staged
from modal_surface.solver_cli import (
    add_staged_solver_arguments,
    staged_solver_config,
    staged_solver_manifest_parameters,
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
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")
    add_staged_solver_arguments(parser)


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
            freq_tolerance_hz=args.freq_tolerance_hz,
        )
        optimize_multi_view_staged(
            observations_path=obs_path,
            out_path=latent_path,
            vis_dir=mode_vis_dir,
            config=staged_solver_config(args),
        )
        with (
            np.load(str(obs_path), allow_pickle=False) as observations,
            np.load(str(latent_path), allow_pickle=False) as latent,
        ):
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
                    "alpha_by_view": _alpha_by_view_diagnostics(latent_path, latent),
                    "stats": _latent_stats(latent_path, latent, observations),
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
            "freq_tolerance_hz": float(args.freq_tolerance_hz),
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
