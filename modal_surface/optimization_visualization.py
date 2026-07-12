"""Visualization helpers for solved multi-view modal fields."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
