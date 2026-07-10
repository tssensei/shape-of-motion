"""Shared helpers for batch modal-solving command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


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


def _finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def _latent_stats(latent_path: Path, observation_path: Path) -> dict[str, Any]:
    latent = np.load(str(latent_path), allow_pickle=False)
    observations = np.load(str(observation_path), allow_pickle=False)
    obs_residual = latent["obs_residual"].astype(np.float32)
    point_residual = latent["point_residual"].astype(np.float32)
    if "obs_residual_valid_mask" in latent.files:
        obs_residual = obs_residual[latent["obs_residual_valid_mask"].astype(bool)]
    if "point_residual_valid_mask" in latent.files:
        point_residual = point_residual[latent["point_residual_valid_mask"].astype(bool)]
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
        "obs_residual_median": _finite_percentile(obs_residual, 50),
        "obs_residual_p90": _finite_percentile(obs_residual, 90),
        "point_residual_median": _finite_percentile(point_residual, 50),
        "point_residual_p90": _finite_percentile(point_residual, 90),
        "observations_per_view": observations["observations_per_view"].astype(int).tolist(),
        "source_mask_candidate_count": int(np.asarray(observations["source_mask_candidate_count"]).item())
        if "source_mask_candidate_count" in observations.files
        else None,
        "source_mask_kept_count": int(np.asarray(observations["source_mask_kept_count"]).item())
        if "source_mask_kept_count" in observations.files
        else None,
        "solver_method": str(np.asarray(latent["solver_method"]).item()) if "solver_method" in latent.files else "legacy-als",
        "alpha_identifiable_count": int(latent["alpha_identifiable_mask"].astype(bool).sum())
        if "alpha_identifiable_mask" in latent.files
        else None,
        "alpha_optimizer_success": bool(np.asarray(latent["alpha_optimizer_success"]).item())
        if "alpha_optimizer_success" in latent.files
        else None,
        "anchor_count": int(latent["anchor_mask"].astype(bool).sum()) if "anchor_mask" in latent.files else None,
        "partial_unresolved_count": int(latent["partial_mask"].astype(bool).sum()) if "partial_mask" in latent.files else None,
        "rejected_count": int(latent["rejected_mask"].astype(bool).sum()) if "rejected_mask" in latent.files else None,
        "alpha_unresolved_point_count": int(latent["alpha_unresolved_mask"].astype(bool).sum())
        if "alpha_unresolved_mask" in latent.files
        else None,
        "no_usable_observation_point_count": int(latent["no_usable_observation_mask"].astype(bool).sum())
        if "no_usable_observation_mask" in latent.files
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


def _alpha_by_view_diagnostics(latent_path: Path) -> list[dict[str, Any]]:
    latent = np.load(str(latent_path), allow_pickle=False)
    required = [
        "view_ids",
        "alphas",
        "alpha_by_view",
        "alpha_semantics",
        "alpha_reference_view_index",
        "alpha_view_freqs_hz",
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
    identifiable = (
        latent["alpha_identifiable_mask"].astype(bool).reshape(-1)
        if "alpha_identifiable_mask" in latent.files
        else np.ones((len(view_ids),), dtype=bool)
    )
    reasons = (
        latent["alpha_exclusion_reason"].astype(str).reshape(-1)
        if "alpha_exclusion_reason" in latent.files
        else np.full((len(view_ids),), "legacy", dtype="<U16")
    )
    phase_std = (
        latent["alpha_phase_std"].astype(np.float32).reshape(-1)
        if "alpha_phase_std" in latent.files
        else np.full((len(view_ids),), np.nan, dtype=np.float32)
    )
    gain_std = (
        latent["alpha_gain_std"].astype(np.float32).reshape(-1)
        if "alpha_gain_std" in latent.files
        else np.full((len(view_ids),), np.nan, dtype=np.float32)
    )
    log_gain_std = (
        latent["alpha_log_gain_std"].astype(np.float32).reshape(-1)
        if "alpha_log_gain_std" in latent.files
        else np.full((len(view_ids),), np.nan, dtype=np.float32)
    )
    gain_bound_active = (
        latent["alpha_gain_bound_active_mask"].astype(bool).reshape(-1)
        if "alpha_gain_bound_active_mask" in latent.files
        else np.zeros((len(view_ids),), dtype=bool)
    )
    return [
        {
            "view_id": view_id,
            "freq_hz": _json_float(freqs_hz[idx]),
            "real": _json_float(np.real(alphas[idx])),
            "imag": _json_float(np.imag(alphas[idx])),
            "abs": _json_float(np.abs(alphas[idx])),
            "phase_rad": _json_float(np.angle(alphas[idx])),
            "identifiable": bool(identifiable[idx]),
            "reason": str(reasons[idx]),
            "phase_std": _json_float(phase_std[idx]),
            "gain_std": _json_float(gain_std[idx]),
            "log_gain_std": _json_float(log_gain_std[idx]),
            "gain_bound_active": bool(gain_bound_active[idx]),
        }
        for idx, view_id in enumerate(view_ids)
    ]
