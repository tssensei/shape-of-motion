"""Visualization helpers for solved multi-view modal fields."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from modal_surface.optimization_staged import StagedSolveResult


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


def write_solve_visualizations(
    staged: StagedSolveResult,
    phi: np.ndarray,
    obs_pred_y: np.ndarray,
    obs_residual_valid_mask: np.ndarray,
    vis_dir: str | Path,
) -> None:
    """Write final per-view observation images and solved-field point clouds."""

    prepared = staged.prepared
    if (
        "view_image_width" not in prepared.arrays
        or "view_image_height" not in prepared.arrays
    ):
        raise ValueError("Visualization requires view_image_width and view_image_height.")
    prediction = np.asarray(obs_pred_y)
    valid = np.asarray(obs_residual_valid_mask)
    if prediction.shape != prepared.obs_y.shape:
        raise ValueError(
            f"obs_pred_y must have shape {prepared.obs_y.shape}, got {prediction.shape}."
        )
    if valid.shape != (prepared.obs_y.shape[0],) or valid.dtype != np.bool_:
        raise ValueError("obs_residual_valid_mask must be boolean and match observations.")

    vis = Path(vis_dir)
    vis.mkdir(parents=True, exist_ok=True)
    widths = prepared.arrays["view_image_width"].astype(np.int32)
    heights = prepared.arrays["view_image_height"].astype(np.int32)
    for view_idx in range(prepared.num_views):
        rows = np.where((prepared.obs_view_index == view_idx) & valid)[0]
        if rows.size == 0:
            continue
        view_name = str(prepared.view_ids[view_idx])
        _scatter_mode_image(
            vis / f"{view_name}_observed.png",
            prepared.obs_pixels_xy[rows],
            prepared.obs_y[rows],
            int(widths[view_idx]),
            int(heights[view_idx]),
            f"{view_name} observed",
        )
        _scatter_mode_image(
            vis / f"{view_name}_predicted.png",
            prepared.obs_pixels_xy[rows],
            prediction[rows],
            int(widths[view_idx]),
            int(heights[view_idx]),
            f"{view_name} predicted",
        )
        _scatter_mode_image(
            vis / f"{view_name}_residual.png",
            prepared.obs_pixels_xy[rows],
            prepared.obs_y[rows] - prediction[rows],
            int(widths[view_idx]),
            int(heights[view_idx]),
            f"{view_name} residual",
            cmap="viridis",
            normalize=False,
        )
    points = prepared.points.astype(np.float32, copy=False)
    final_phi = np.asarray(phi, dtype=np.complex64)
    if final_phi.shape != points.shape:
        raise ValueError(f"phi must have shape {points.shape}, got {final_phi.shape}.")
    _write_ply(vis / "pointcloud_amplitude.ply", points, _amplitude_colors(final_phi))
    _write_ply(vis / "pointcloud_phase_u.ply", points, _phase_colors(final_phi))
