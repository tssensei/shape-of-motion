"""Shared helpers for batch modal-solving command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile


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


def _latent_stats(
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


def _json_float(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _alpha_by_view_diagnostics(
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
