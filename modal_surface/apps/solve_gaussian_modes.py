"""Batch solve 3D modal fields directly on foreground 3DGS Gaussians."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
from numpy.lib.npyio import NpzFile

from modal_surface.gaussian_observations import build_gaussian_observation_graph
from modal_surface.gaussian_motion_fill import (
    MOTION_FILL_EPSILON,
    MOTION_FILL_NULLSPACE_RTOL,
    MOTION_FILL_OBSERVATION_DRIFT_RTOL,
    MOTION_FILL_METHOD,
    apply_gaussian_motion_fill,
    rewrite_motion_fill_pointclouds,
    write_motion_fill_diagnostics,
    write_motion_fill_graph,
)
from modal_surface.io import load_modal_freqs, load_view_config
from modal_surface.motion_fill import build_knn_graph, query_knn_candidates
from modal_surface.optimization_staged import optimize_multi_view_staged
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
    latent_path: Path,
    latent: NpzFile,
    observations: NpzFile,
) -> dict[str, Any]:
    required = [
        "points_world",
        "obs_point_index",
        "obs_residual",
        "obs_residual_valid_mask",
        "point_residual",
        "point_residual_valid_mask",
        "solver_method",
        "alpha_identifiable_mask",
        "alpha_optimizer_success",
        "anchor_mask",
        "partial_mask",
        "rejected_mask",
        "alpha_unresolved_mask",
        "no_usable_observation_mask",
    ]
    missing = [key for key in required if key not in latent.files]
    if missing:
        raise ValueError(f"{latent_path} missing staged solver statistic fields: {missing}.")
    obs_residual = latent["obs_residual"].astype(np.float32)
    point_residual = latent["point_residual"].astype(np.float32)
    obs_residual = obs_residual[latent["obs_residual_valid_mask"].astype(bool)]
    point_residual = point_residual[latent["point_residual_valid_mask"].astype(bool)]
    return {
        "num_points": int(latent["points_world"].shape[0]),
        "num_observations": int(latent["obs_point_index"].shape[0]),
        "obs_residual_median": _finite_percentile(obs_residual, 50),
        "obs_residual_p90": _finite_percentile(obs_residual, 90),
        "point_residual_median": _finite_percentile(point_residual, 50),
        "point_residual_p90": _finite_percentile(point_residual, 90),
        "observations_per_view": observations["observations_per_view"].astype(int).tolist(),
        "solver_method": str(np.asarray(latent["solver_method"]).item()),
        "alpha_identifiable_count": int(latent["alpha_identifiable_mask"].astype(bool).sum()),
        "alpha_optimizer_success": bool(np.asarray(latent["alpha_optimizer_success"]).item()),
        "anchor_count": int(latent["anchor_mask"].astype(bool).sum()),
        "partial_unresolved_count": int(latent["partial_mask"].astype(bool).sum()),
        "rejected_count": int(latent["rejected_mask"].astype(bool).sum()),
        "alpha_unresolved_point_count": int(latent["alpha_unresolved_mask"].astype(bool).sum()),
        "no_usable_observation_point_count": int(latent["no_usable_observation_mask"].astype(bool).sum()),
    }


def json_float(value: SupportsFloat) -> float | None:
    result = float(value)
    if not np.isfinite(result):
        return None
    return result


def alpha_by_view_diagnostics(
    latent_path: Path,
    latent: NpzFile,
) -> list[dict[str, Any]]:
    required = [
        "view_ids",
        "alphas",
        "alpha_by_view",
        "alpha_semantics",
        "alpha_reference_view_index",
        "alpha_view_freqs_hz",
        "alpha_identifiable_mask",
        "alpha_exclusion_reason",
        "alpha_phase_std",
        "alpha_gain_std",
        "alpha_log_gain_std",
        "alpha_gain_bound_active_mask",
    ]
    missing = [key for key in required if key not in latent.files]
    if missing:
        raise ValueError(f"{latent_path} missing alpha diagnostic fields: {missing}.")
    semantics = str(np.asarray(latent["alpha_semantics"]).item())
    if semantics != "per_view_per_mode":
        raise ValueError(f"{latent_path} has unexpected alpha_semantics={semantics!r}.")
    reference_view_index = int(np.asarray(latent["alpha_reference_view_index"]).item())
    if reference_view_index != 0:
        raise ValueError(f"{latent_path} has unexpected alpha_reference_view_index={reference_view_index}.")

    view_ids = [str(v) for v in latent["view_ids"].tolist()]
    saved_alphas = latent["alphas"].astype(np.complex64).reshape(-1)
    alphas = latent["alpha_by_view"].astype(np.complex64).reshape(-1)
    freqs_hz = latent["alpha_view_freqs_hz"].astype(np.float32).reshape(-1)
    if saved_alphas.shape != alphas.shape or not np.array_equal(saved_alphas, alphas):
        raise ValueError(f"{latent_path} alphas and alpha_by_view are inconsistent.")
    if alphas.shape[0] != len(view_ids):
        raise ValueError(f"{latent_path} alpha_by_view length does not match view_ids.")
    if freqs_hz.shape[0] != len(view_ids):
        raise ValueError(f"{latent_path} alpha_view_freqs_hz length does not match view_ids.")
    identifiable = latent["alpha_identifiable_mask"].astype(bool).reshape(-1)
    reasons = latent["alpha_exclusion_reason"].astype(str).reshape(-1)
    phase_std = latent["alpha_phase_std"].astype(np.float32).reshape(-1)
    gain_std = latent["alpha_gain_std"].astype(np.float32).reshape(-1)
    log_gain_std = latent["alpha_log_gain_std"].astype(np.float32).reshape(-1)
    gain_bound_active = latent["alpha_gain_bound_active_mask"].astype(bool).reshape(-1)
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
        raise ValueError(f"{latent_path} staged alpha diagnostics have invalid lengths: {invalid}.")
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
    if num_points is not None and int(args.motion_fill_k) >= int(num_points):
        raise ValueError(
            f"--motion-fill-k must be smaller than the foreground Gaussian count ({num_points})."
        )


def _load_fg_pixel_candidate_inputs_from_checkpoint(path: str, view_config_paths: list[str],) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
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
    return fg_means, fg_scales, fg_quats, fg_opacities, rendered_depths, rendered_accs


def _gaussian_latent_stats(
    latent_path: Path,
    latent: NpzFile,
    observations: NpzFile,
    num_fg: int,
) -> dict[str, Any]:
    stats = latent_stats(latent_path, latent, observations)
    obs_count = latent["obs_count_per_point"].astype(np.int32)
    obs_sample_count = latent["obs_sample_count_per_point"].astype(np.int32)
    gaussian_indices = latent["gaussian_indices"].astype(np.int32)
    stats.update(
        {
            "num_foreground_gaussians": int(num_fg),
            "num_output_points": int(latent["points_world"].shape[0]),
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
    if "motion_fill_method" in latent.files:
        completion = latent["completion_mask"].astype(bool)
        partial = latent["partial_mask"].astype(bool)
        unobserved = latent["unobserved_mask"].astype(bool)
        staged_solver_method = str(np.asarray(latent["solver_method"]).item())
        motion_fill_method = str(np.asarray(latent["motion_fill_method"]).item())
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
        rendered_depths,
        rendered_accs,
    ) = _load_fg_pixel_candidate_inputs_from_checkpoint(
        args.input_ckpt,
        view_configs_paths,
    )

    _validate_motion_fill_arguments(args, fg_means.shape[0])
    
    gaussian_indices = np.arange(fg_means.shape[0], dtype=np.int32)
    freqs_per_view = load_modal_freqs(modal_npzs_paths)
    # e.g. freqs_per_view = [
    # np.array([0.357, 0.714]),  # view 1
    # np.array([0.359, 0.711]),  # view 2
    # np.array([0.356, 0.716]),  # view 3 ]
    mode_indices = parse_mode_indices(args.mode_indices, int(freqs_per_view[0].shape[0]))
    out_dir = Path(args.out_dir)
    obs_dir = out_dir / "observations"
    latent_dir = out_dir / "latents"
    vis_dir = out_dir / "vis"
    obs_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    motion_fill_graph = None
    motion_fill_graph_path: Path | None = None
    motion_fill_mode_diagnostics: dict[str, Any] = {}
    if args.motion_fill:
        candidates = query_knn_candidates(fg_means, int(args.motion_fill_k))
        motion_fill_graph = build_knn_graph(
            candidates,
            int(args.motion_fill_k),
            float(args.motion_fill_max_distance),
            MOTION_FILL_EPSILON,
        )
        motion_fill_graph_path = write_motion_fill_graph(
            out_dir / "motion_fill" / "graph.npz",
            fg_means,
            candidates,
            motion_fill_graph,
        )

    modes: list[dict[str, Any]] = []
    for mode_index in mode_indices:
        reference_freq = float(freqs_per_view[0][mode_index])
        mode_name = f"mode_{mode_index:03d}_{freq_slug(reference_freq)}hz"
        obs_path = obs_dir / f"{mode_name}.npz"
        latent_path = latent_dir / f"{mode_name}.npz"
        mode_vis_dir = vis_dir / mode_name
        print(f"Solving Gaussian mode {mode_name}")
        build_gaussian_observation_graph(
            points_world=fg_means,
            view_config_paths=view_configs_paths,
            modal_npz_paths=modal_npzs_paths,
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
            motion_fill_diagnostics = None
            if motion_fill_graph is not None:
                assert motion_fill_graph_path is not None
                motion_fill_diagnostics = apply_gaussian_motion_fill(
                    latent_path,
                    motion_fill_graph,
                    relative_path(motion_fill_graph_path, out_dir),
                )
                rewrite_motion_fill_pointclouds(latent_path, mode_vis_dir)
                motion_fill_mode_diagnostics[mode_name] = motion_fill_diagnostics
            with np.load(str(latent_path), allow_pickle=False) as latent:
                latent_stats = _gaussian_latent_stats(
                    latent_path,
                    latent,
                    observations,
                    fg_means.shape[0],
                )
                if motion_fill_diagnostics is not None:
                    latent_stats["motion_fill"] = motion_fill_diagnostics
                freqs_by_view = [float(freqs[mode_index]) for freqs in freqs_per_view]
                modes.append(
                    {
                        "mode_index": int(mode_index),
                        "freq_hz": reference_freq,
                        "freqs_hz_by_view": freqs_by_view,
                        "label": f"{mode_index}: {reference_freq:.6f} Hz",
                        "observation_path": relative_path(obs_path, out_dir),
                        "latent_path": relative_path(latent_path, out_dir),
                        "vis_dir": relative_path(mode_vis_dir, out_dir),
                        "alpha_by_view": alpha_by_view_diagnostics(latent_path, latent),
                        "stats": latent_stats,
                    }
                )

    manifest_parameters = {
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
        "motion_fill_enabled": bool(args.motion_fill),
        **staged_solver_manifest_parameters(args),
    }
    if motion_fill_graph is not None:
        assert motion_fill_graph_path is not None
        manifest_parameters.update(
            {
                "motion_fill_method": MOTION_FILL_METHOD,
                "motion_fill_k": int(motion_fill_graph.k),
                "motion_fill_max_distance": float(motion_fill_graph.max_distance),
                "motion_fill_epsilon": float(motion_fill_graph.epsilon),
                "motion_fill_nullspace_operator_rtol": MOTION_FILL_NULLSPACE_RTOL,
                "motion_fill_observation_drift_rtol": MOTION_FILL_OBSERVATION_DRIFT_RTOL,
                "motion_fill_graph_path": relative_path(motion_fill_graph_path, out_dir),
                "motion_fill_excluded_policy": "retain_observable_exclude_from_graph",
            }
        )
        write_motion_fill_diagnostics(
            out_dir / "motion_fill" / "diagnostics.json",
            {
                "version": 1,
                "method": MOTION_FILL_METHOD,
                "graph_path": relative_path(motion_fill_graph_path, out_dir),
                "graph": {
                    "k": int(motion_fill_graph.k),
                    "max_distance": float(motion_fill_graph.max_distance),
                    "epsilon": float(motion_fill_graph.epsilon),
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
