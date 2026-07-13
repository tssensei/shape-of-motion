"""Batch solve 3D modal fields directly on foreground 3DGS Gaussians."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile

from modal_surface.apps._shared import (
    _alpha_by_view_diagnostics,
    _freq_slug,
    _json_float,
    _latent_stats,
    _load_modal_freqs,
    _parse_mode_indices,
    _rel,
)
from modal_surface.gaussian_observations import build_gaussian_observation_graph
from modal_surface.io import load_view_config
from modal_surface.optimization_staged import optimize_multi_view_staged
from modal_surface.solver_cli import (
    add_staged_solver_arguments,
    staged_solver_config,
    staged_solver_manifest_parameters,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-ckpt", required=True, help="Static 3DGS checkpoint containing foreground Gaussian centers.")
    parser.add_argument("--view-config", action="append", required=True, help="View JSON config path. Repeat per view.")
    parser.add_argument("--modal-npz", action="append", required=True, help="modal_analysis.npz path. Repeat per view.")
    parser.add_argument("--out-dir", required=True, help="Output directory for observations, latents, vis, and manifest.")
    parser.add_argument("--mode-indices", default="all", help="Comma-separated zero-based mode indices, or 'all'.")
    parser.add_argument("--pixel-sample-stride", type=int, default=4, help="Pixel grid stride for pixel-candidates sampling.")
    parser.add_argument("--pixel-candidate-k", type=int, default=4, help="Number of top contribution Gaussians supervised by each sampled pixel.")
    parser.add_argument("--pixel-preselect-k", type=int, default=32, help="Number of 3D nearest Gaussians scored before top-k contribution selection.")
    parser.add_argument("--pixel-render-acc-min", type=float, default=0.05, help="Minimum rendered foreground alpha for sampled modal pixels.")
    parser.add_argument("--pixel-min-contribution", type=float, default=1e-12, help="Minimum unnormalized Gaussian contribution retained for a sampled modal pixel.")
    parser.add_argument("--pixel-max-samples-per-view", type=int, default=20000, help="Maximum sampled modal pixels per view before candidate expansion.")
    parser.add_argument("--mask-erode-iters", type=int, default=1, help="3x3 modal mask erosion iterations.")
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")
    add_staged_solver_arguments(parser)


def _load_fg_pixel_candidate_inputs_from_checkpoint(
    path: str,
    view_config_paths: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
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


def _gaussian_latent_stats(
    latent_path: Path,
    latent: NpzFile,
    observations: NpzFile,
    num_fg: int,
) -> dict[str, Any]:
    stats = _latent_stats(latent_path, latent, observations)
    obs_count = latent["obs_count_per_point"].astype(np.int32)
    obs_sample_count = latent["obs_sample_count_per_point"].astype(np.int32)
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
            "preserved_all_points": bool(np.asarray(observations["preserved_all_points"]).item()),
            "pixel_candidate_method": str(np.asarray(observations["pixel_candidate_method"]).item()),
        }
    )
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
    return stats


def _print_observation_sanity(observations: NpzFile, num_fg: int) -> None:
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
    view_configs = list(args.view_config)
    modal_npzs = list(args.modal_npz)
    if len(view_configs) != len(modal_npzs):
        raise ValueError("--view-config and --modal-npz must be supplied the same number of times.")
    (
        fg_means,
        fg_scales,
        fg_quats,
        fg_opacities,
        rendered_depths,
        rendered_accs,
    ) = _load_fg_pixel_candidate_inputs_from_checkpoint(
        args.input_ckpt,
        view_configs,
    )
    gaussian_indices = np.arange(fg_means.shape[0], dtype=np.int32)
    freqs_per_view = _load_modal_freqs(modal_npzs)
    # e.g. freqs_per_view = [
    # np.array([0.357, 0.714]),  # view 1
    # np.array([0.359, 0.711]),  # view 2
    # np.array([0.356, 0.716]),  # view 3 ]
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
        print(f"Solving Gaussian mode {mode_name}")
        build_gaussian_observation_graph(
            points_world=fg_means,
            view_config_paths=view_configs,
            modal_npz_paths=modal_npzs,
            out_path=obs_path,
            mode_index=mode_index,
            mask_erode_iters=args.mask_erode_iters,
            freq_tolerance_hz=args.freq_tolerance_hz,  #
            preserve_all_points=True,
            optional_point_fields={"gaussian_indices": gaussian_indices},
            extra_metadata={
                "point_type": np.array("foreground_gaussian_center"),
                "source_checkpoint": np.array(str(args.input_ckpt)),
            },
            pixel_sample_stride=args.pixel_sample_stride,
            pixel_candidate_k=args.pixel_candidate_k,
            pixel_preselect_k=args.pixel_preselect_k,
            pixel_render_acc_min=args.pixel_render_acc_min,
            pixel_min_contribution=args.pixel_min_contribution,
            pixel_max_samples_per_view=args.pixel_max_samples_per_view,
            gaussian_scales=fg_scales,
            gaussian_quats=fg_quats,
            gaussian_opacities=fg_opacities,
            rendered_depths=rendered_depths,
            rendered_accs=rendered_accs,
        )
        with np.load(str(obs_path), allow_pickle=False) as observations:
            _print_observation_sanity(observations, fg_means.shape[0])
            optimize_multi_view_staged(
                observations_path=obs_path,
                out_path=latent_path,
                vis_dir=mode_vis_dir,
                config=staged_solver_config(args),
            )
            with np.load(str(latent_path), allow_pickle=False) as latent:
                latent_stats = _gaussian_latent_stats(
                    latent_path,
                    latent,
                    observations,
                    fg_means.shape[0],
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
                        "alpha_by_view": _alpha_by_view_diagnostics(latent_path, latent),
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
            "pixel_sample_stride": int(args.pixel_sample_stride),
            "pixel_candidate_k": int(args.pixel_candidate_k),
            "pixel_preselect_k": int(args.pixel_preselect_k),
            "pixel_render_acc_min": float(args.pixel_render_acc_min),
            "pixel_min_contribution": float(args.pixel_min_contribution),
            "pixel_max_samples_per_view": int(args.pixel_max_samples_per_view),
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
    print(f"Saved Gaussian modal modes manifest -> {manifest_path}")
