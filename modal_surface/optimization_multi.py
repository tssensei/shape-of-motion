"""Optimize latent 3D modal displacement from N-view observations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _observations_by_point(num_points: int, obs_point_index: np.ndarray) -> list[np.ndarray]:
    return [np.where(obs_point_index == i)[0] for i in range(num_points)]


def _solve_phi_points(
    obs_y: np.ndarray,
    obs_J: np.ndarray,
    obs_confidence: np.ndarray,
    obs_view_index: np.ndarray,
    obs_by_point: list[np.ndarray],
    active: np.ndarray,
    alphas: np.ndarray,
    ridge_mu: float,
) -> np.ndarray:
    if ridge_mu < 0:
        raise ValueError("ridge_mu must be non-negative.")
    phi = np.zeros((len(obs_by_point), 3), dtype=np.complex64)
    eye = np.eye(3, dtype=np.complex128)
    for point_idx, rows in enumerate(obs_by_point):
        if not active[point_idx]:
            continue
        if rows.size == 0:
            continue
        weights = np.sqrt(np.maximum(obs_confidence[rows].astype(np.float64), 0.0))
        alpha_rows = alphas[obs_view_index[rows]].astype(np.complex128)
        A = (weights[:, None, None] * alpha_rows[:, None, None] * obs_J[rows].astype(np.complex128)).reshape(-1, 3)
        b = (weights[:, None] * obs_y[rows].astype(np.complex128)).reshape(-1)
        lhs = A.conj().T @ A + float(ridge_mu) * eye
        rhs = A.conj().T @ b
        phi[point_idx] = np.linalg.solve(lhs, rhs).astype(np.complex64)
    return phi


def _predict_observations(obs_J: np.ndarray, obs_point_index: np.ndarray, obs_view_index: np.ndarray, phi: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    projected = np.einsum("oij,oj->oi", obs_J.astype(np.float32), phi[obs_point_index].astype(np.complex64))
    return (alphas[obs_view_index][:, None] * projected).astype(np.complex64)


def _solve_alphas(
    obs_y: np.ndarray,
    obs_J: np.ndarray,
    obs_point_index: np.ndarray,
    obs_view_index: np.ndarray,
    obs_confidence: np.ndarray,
    active: np.ndarray,
    phi: np.ndarray,
    num_views: int,
) -> np.ndarray:
    alphas = np.ones((num_views,), dtype=np.complex64)
    active_obs = active[obs_point_index]
    for view_idx in range(1, num_views):
        rows = np.where(active_obs & (obs_view_index == view_idx))[0]
        if rows.size == 0:
            raise ValueError(f"Cannot solve alpha for view index {view_idx}: no active observations.")
        projected = np.einsum("oij,oj->oi", obs_J[rows].astype(np.float32), phi[obs_point_index[rows]].astype(np.complex64))
        weights = np.maximum(obs_confidence[rows].astype(np.float64), 0.0)
        numerator = np.sum(weights[:, None] * np.conj(projected) * obs_y[rows])
        denominator = np.sum(weights[:, None] * np.conj(projected) * projected)
        denom_real = float(np.real(denominator))
        if denom_real <= 1e-12:
            raise ValueError(f"Cannot solve alpha for view index {view_idx}: projected motion energy is too small.")
        alphas[view_idx] = np.complex64(numerator / denominator)
    return alphas


def _obs_residual(obs_y: np.ndarray, pred_y: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(obs_y - pred_y) ** 2, axis=1)).astype(np.float32)


def _point_residuals(num_points: int, obs_point_index: np.ndarray, obs_residual: np.ndarray, active: np.ndarray) -> np.ndarray:
    sums = np.zeros((num_points,), dtype=np.float64)
    counts = np.zeros((num_points,), dtype=np.float64)
    np.add.at(sums, obs_point_index, obs_residual.astype(np.float64) ** 2)
    np.add.at(counts, obs_point_index, 1.0)
    out = np.sqrt(sums / np.maximum(counts, 1.0)).astype(np.float32)
    out[~active] = np.inf
    return out


def _scatter_mode_image(
    out_path: Path,
    pixels_xy: np.ndarray,
    values: np.ndarray,
    width: int,
    height: int,
    title: str,
    cmap: str = "magma",
    normalize: bool = True,
) -> None:
    amp = np.sqrt(np.sum(np.abs(values) ** 2, axis=1)).astype(np.float32)
    if normalize:
        hi = float(np.percentile(amp, 99)) if amp.size else 1.0
        hi = max(hi, 1e-6)
        color_values = np.clip(amp / hi, 0.0, 1.0)
        vmax = 1.0
    else:
        color_values = amp
        vmax = None
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(pixels_xy[:, 0], pixels_xy[:, 1], c=color_values, s=2, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.7g} {p[1]:.7g} {p[2]:.7g} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _amplitude_colors(phi: np.ndarray) -> np.ndarray:
    amp = np.linalg.norm(phi, axis=1)
    hi = float(np.percentile(amp, 99)) if amp.size else 1.0
    val = np.clip(amp / max(hi, 1e-12), 0.0, 1.0)
    rgb = plt.get_cmap("magma")(val)[:, :3]
    return (255.0 * rgb).astype(np.uint8)


def _phase_colors(phi: np.ndarray) -> np.ndarray:
    phase = np.angle(phi[:, 0])
    hue = (phase + np.pi) / (2.0 * np.pi)
    rgb = plt.get_cmap("hsv")(hue)[:, :3]
    return (255.0 * rgb).astype(np.uint8)


def optimize_multi_view(
    observations_path: str | Path,
    out_path: str | Path,
    vis_dir: str | Path | None = None,
    iterations: int = 8,
    ridge_mu: float = 1e-4,
    outlier_frac: float = 0.05,
) -> Path:
    """Optimize shared phi_i and per-view complex alpha_v from observation rows."""
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if not (0.0 <= outlier_frac < 0.5):
        raise ValueError("outlier_frac must be in [0, 0.5).")

    data = np.load(str(observations_path), allow_pickle=False)
    points = data["points_world"].astype(np.float32)
    obs_point_index = data["obs_point_index"].astype(np.int64)
    obs_view_index = data["obs_view_index"].astype(np.int64)
    obs_pixels_xy = data["obs_pixels_xy"].astype(np.float32)
    obs_y = data["obs_y"].astype(np.complex64)
    obs_J = data["obs_J"].astype(np.float32)
    obs_confidence = data["obs_confidence"].astype(np.float32)
    view_ids = data["view_ids"]
    num_views = int(view_ids.shape[0])

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points.shape}.")
    if obs_y.ndim != 2 or obs_y.shape[1] != 2:
        raise ValueError(f"obs_y must have shape (O,2), got {obs_y.shape}.")
    if obs_J.shape != (obs_y.shape[0], 2, 3):
        raise ValueError(f"obs_J must have shape (O,2,3), got {obs_J.shape}.")
    if np.any(obs_point_index < 0) or np.any(obs_point_index >= points.shape[0]):
        raise ValueError("obs_point_index contains invalid point indices.")
    if np.any(obs_view_index < 0) or np.any(obs_view_index >= num_views):
        raise ValueError("obs_view_index contains invalid view indices.")

    obs_by_point = _observations_by_point(points.shape[0], obs_point_index)
    active = np.ones((points.shape[0],), dtype=bool)
    alphas = np.ones((num_views,), dtype=np.complex64)
    history: list[list[float]] = []
    alpha_history: list[np.ndarray] = []

    phi = np.zeros_like(points, dtype=np.complex64)
    pred_y = np.zeros_like(obs_y, dtype=np.complex64)
    obs_residual = np.zeros((obs_y.shape[0],), dtype=np.float32)
    point_residual = np.zeros((points.shape[0],), dtype=np.float32)

    for it in range(iterations):
        if int(active.sum()) < 3:
            raise ValueError("Too few active points remain during optimization.")
        phi = _solve_phi_points(obs_y, obs_J, obs_confidence, obs_view_index, obs_by_point, active, alphas, ridge_mu)
        alphas = _solve_alphas(obs_y, obs_J, obs_point_index, obs_view_index, obs_confidence, active, phi, num_views)
        pred_y = _predict_observations(obs_J, obs_point_index, obs_view_index, phi, alphas)
        obs_residual = _obs_residual(obs_y, pred_y)
        point_residual = _point_residuals(points.shape[0], obs_point_index, obs_residual, active)
        active_residual = point_residual[active]
        history.append(
            [
                float(it),
                float(active.sum()),
                float(active_residual.mean()),
                float(np.median(active_residual)),
            ]
        )
        alpha_history.append(alphas.copy())
        if outlier_frac > 0 and it < iterations - 1:
            active_indices = np.where(active)[0]
            drop_count = int(np.floor(outlier_frac * active_indices.size))
            if drop_count > 0 and active_indices.size - drop_count >= 3:
                drop_indices = active_indices[np.argsort(point_residual[active_indices])[-drop_count:]]
                active[drop_indices] = False

    active_indices = np.where(active)[0]
    old_to_new = np.full((points.shape[0],), -1, dtype=np.int64)
    old_to_new[active_indices] = np.arange(active_indices.size, dtype=np.int64)
    keep_obs = active[obs_point_index]
    optional_point_fields = {}
    if "point_source_view_mask" in data.files:
        point_source_view_mask = data["point_source_view_mask"]
        if point_source_view_mask.shape[0] != points.shape[0]:
            raise ValueError("point_source_view_mask does not match points_world length.")
        optional_point_fields["point_source_view_mask"] = point_source_view_mask[active_indices].astype(bool)
    if "point_source_count" in data.files:
        point_source_count = data["point_source_count"]
        if point_source_count.shape[0] != points.shape[0]:
            raise ValueError("point_source_count does not match points_world length.")
        optional_point_fields["point_source_count"] = point_source_count[active_indices].astype(np.int32)
    if "colors" in data.files:
        colors = data["colors"]
        if colors.shape != (points.shape[0], 3):
            raise ValueError("colors must have shape (N,3) and match points_world length.")
        optional_point_fields["colors"] = colors[active_indices].astype(np.uint8)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points_world=points[active_indices].astype(np.float32),
        phi=phi[active_indices].astype(np.complex64),
        alphas=alphas.astype(np.complex64),
        view_ids=view_ids,
        freq_hz=data["freq_hz"].astype(np.float32),
        mode_index=data["mode_index"].astype(np.int32),
        point_residual=point_residual[active_indices].astype(np.float32),
        obs_point_index=old_to_new[obs_point_index[keep_obs]].astype(np.int32),
        obs_view_index=obs_view_index[keep_obs].astype(np.int32),
        obs_pixels_xy=obs_pixels_xy[keep_obs].astype(np.float32),
        obs_y=obs_y[keep_obs].astype(np.complex64),
        obs_J=obs_J[keep_obs].astype(np.float32),
        obs_confidence=obs_confidence[keep_obs].astype(np.float32),
        obs_pred_y=pred_y[keep_obs].astype(np.complex64),
        obs_residual=obs_residual[keep_obs].astype(np.float32),
        active_indices=active_indices.astype(np.int32),
        optimization_history=np.asarray(history, dtype=np.float32),
        alpha_history=np.asarray(alpha_history, dtype=np.complex64),
        source_observations=np.array(str(observations_path)),
        **optional_point_fields,
    )

    if vis_dir is not None:
        vis = Path(vis_dir)
        vis.mkdir(parents=True, exist_ok=True)
        widths = data["view_image_width"].astype(np.int32)
        heights = data["view_image_height"].astype(np.int32)
        kept_view = obs_view_index[keep_obs]
        kept_pixels = obs_pixels_xy[keep_obs]
        kept_y = obs_y[keep_obs]
        kept_pred = pred_y[keep_obs]
        for view_idx in range(num_views):
            rows = np.where(kept_view == view_idx)[0]
            if rows.size == 0:
                continue
            view_name = str(view_ids[view_idx])
            _scatter_mode_image(
                vis / f"{view_name}_observed.png",
                kept_pixels[rows],
                kept_y[rows],
                int(widths[view_idx]),
                int(heights[view_idx]),
                f"{view_name} observed",
            )
            _scatter_mode_image(
                vis / f"{view_name}_predicted.png",
                kept_pixels[rows],
                kept_pred[rows],
                int(widths[view_idx]),
                int(heights[view_idx]),
                f"{view_name} predicted",
            )
            _scatter_mode_image(
                vis / f"{view_name}_residual.png",
                kept_pixels[rows],
                kept_y[rows] - kept_pred[rows],
                int(widths[view_idx]),
                int(heights[view_idx]),
                f"{view_name} residual",
                cmap="viridis",
                normalize=False,
            )
        _write_ply(vis / "pointcloud_amplitude.ply", points[active_indices], _amplitude_colors(phi[active_indices]))
        _write_ply(vis / "pointcloud_phase_u.ply", points[active_indices], _phase_colors(phi[active_indices]))

    return out
