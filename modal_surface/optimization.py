from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _solve_phi_points(
    y1: np.ndarray,
    y2: np.ndarray,
    J1: np.ndarray,
    J2: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    alpha2: complex,
    ridge_mu: float,
) -> np.ndarray:
    if ridge_mu < 0:
        raise ValueError("ridge_mu must be non-negative.")
    n = y1.shape[0]
    phi = np.zeros((n, 3), dtype=np.complex64)
    eye = np.eye(3, dtype=np.complex128)
    for i in range(n):
        w1 = np.sqrt(max(float(c1[i]), 0.0))
        w2 = np.sqrt(max(float(c2[i]), 0.0))
        A = np.concatenate(
            [
                (w1 * J1[i]).astype(np.complex128),
                (w2 * alpha2 * J2[i]).astype(np.complex128),
            ],
            axis=0,
        )
        b = np.concatenate(
            [
                (w1 * y1[i]).astype(np.complex128),
                (w2 * y2[i]).astype(np.complex128),
            ],
            axis=0,
        )
        lhs = A.conj().T @ A + float(ridge_mu) * eye
        rhs = A.conj().T @ b
        phi[i] = np.linalg.solve(lhs, rhs).astype(np.complex64)
    return phi


def _solve_alpha2(y2: np.ndarray, J2: np.ndarray, phi: np.ndarray, c2: np.ndarray) -> complex:
    projected = np.einsum("nij,nj->ni", J2.astype(np.float32), phi.astype(np.complex64))
    weights = np.maximum(c2.astype(np.float64), 0.0)
    numerator = np.sum(weights[:, None] * np.conj(projected) * y2)
    denominator = np.sum(weights[:, None] * np.conj(projected) * projected)
    denom_real = float(np.real(denominator))
    if denom_real <= 1e-12:
        raise ValueError("Cannot solve alpha2 because projected motion energy is too small.")
    return complex(numerator / denominator)


def _predict(J: np.ndarray, phi: np.ndarray, alpha: complex) -> np.ndarray:
    return (alpha * np.einsum("nij,nj->ni", J.astype(np.float32), phi.astype(np.complex64))).astype(np.complex64)


def _point_residual(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(y - pred) ** 2, axis=1)).astype(np.float32)


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
    sc = ax.scatter(
        pixels_xy[:, 0],
        pixels_xy[:, 1],
        c=color_values,
        s=2,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )
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


def optimize_two_view(
    matches_path: str | Path,
    out_path: str | Path,
    vis_dir: str | Path | None = None,
    iterations: int = 8,
    ridge_mu: float = 1e-4,
    outlier_frac: float = 0.05,
) -> Path:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if not (0.0 <= outlier_frac < 0.5):
        raise ValueError("outlier_frac must be in [0, 0.5).")

    data = np.load(str(matches_path), allow_pickle=False)
    points = data["points_world"].astype(np.float32)
    y1 = data["y1"].astype(np.complex64)
    y2 = data["y2"].astype(np.complex64)
    J1 = data["J1"].astype(np.float32)
    J2 = data["J2"].astype(np.float32)
    c1 = data["c1"].astype(np.float32)
    c2 = data["c2"].astype(np.float32)
    p1_xy = data["p1_xy"].astype(np.float32)
    p2_xy = data["p2_xy"].astype(np.float32)

    active = np.ones(points.shape[0], dtype=bool)
    alpha1 = complex(1.0, 0.0)
    alpha2 = complex(1.0, 0.0)
    history: list[dict[str, float]] = []

    phi_active = np.zeros((int(active.sum()), 3), dtype=np.complex64)
    pred1_active = np.zeros((int(active.sum()), 2), dtype=np.complex64)
    pred2_active = np.zeros((int(active.sum()), 2), dtype=np.complex64)
    r1_active = np.zeros((int(active.sum()),), dtype=np.float32)
    r2_active = np.zeros((int(active.sum()),), dtype=np.float32)

    for it in range(iterations):
        idx = np.where(active)[0]
        if idx.size < 3:
            raise ValueError("Too few active matches remain during optimization.")
        phi_active = _solve_phi_points(
            y1[idx],
            y2[idx],
            J1[idx],
            J2[idx],
            c1[idx],
            c2[idx],
            alpha2=alpha2,
            ridge_mu=ridge_mu,
        )
        alpha2 = _solve_alpha2(y2[idx], J2[idx], phi_active, c2[idx])
        pred1_active = _predict(J1[idx], phi_active, alpha1)
        pred2_active = _predict(J2[idx], phi_active, alpha2)
        r1_active = _point_residual(y1[idx], pred1_active)
        r2_active = _point_residual(y2[idx], pred2_active)
        total = np.sqrt(r1_active**2 + r2_active**2)
        history.append(
            {
                "iteration": float(it),
                "active_count": float(idx.size),
                "alpha2_real": float(np.real(alpha2)),
                "alpha2_imag": float(np.imag(alpha2)),
                "mean_residual": float(total.mean()),
                "median_residual": float(np.median(total)),
            }
        )
        if outlier_frac > 0 and it < iterations - 1:
            drop_count = int(np.floor(outlier_frac * idx.size))
            if drop_count > 0 and idx.size - drop_count >= 3:
                drop_local = np.argsort(total)[-drop_count:]
                active[idx[drop_local]] = False

    idx = np.where(active)[0]
    phi = phi_active
    pred_y1 = pred1_active
    pred_y2 = pred2_active
    residual1 = r1_active
    residual2 = r2_active

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points_world=points[idx].astype(np.float32),
        phi=phi.astype(np.complex64),
        alpha1=np.array(alpha1, dtype=np.complex64),
        alpha2=np.array(alpha2, dtype=np.complex64),
        freq_hz=data["freq_hz"].astype(np.float32),
        residual1=residual1.astype(np.float32),
        residual2=residual2.astype(np.float32),
        p1_xy=p1_xy[idx].astype(np.float32),
        p2_xy=p2_xy[idx].astype(np.float32),
        y1=y1[idx].astype(np.complex64),
        y2=y2[idx].astype(np.complex64),
        pred_y1=pred_y1.astype(np.complex64),
        pred_y2=pred_y2.astype(np.complex64),
        c1=c1[idx].astype(np.float32),
        c2=c2[idx].astype(np.float32),
        active_indices=idx.astype(np.int32),
        optimization_history=np.asarray(
            [
                [
                    h["iteration"],
                    h["active_count"],
                    h["alpha2_real"],
                    h["alpha2_imag"],
                    h["mean_residual"],
                    h["median_residual"],
                ]
                for h in history
            ],
            dtype=np.float32,
        ),
        source_matches=np.array(str(matches_path)),
    )

    if vis_dir is not None:
        vis = Path(vis_dir)
        vis.mkdir(parents=True, exist_ok=True)
        image1_width = int(data["image1_width"])
        image1_height = int(data["image1_height"])
        image2_width = int(data["image2_width"])
        image2_height = int(data["image2_height"])
        _scatter_mode_image(vis / "view1_observed.png", p1_xy[idx], y1[idx], image1_width, image1_height, "View 1 observed")
        _scatter_mode_image(vis / "view1_predicted.png", p1_xy[idx], pred_y1, image1_width, image1_height, "View 1 predicted")
        _scatter_mode_image(
            vis / "view1_residual.png",
            p1_xy[idx],
            (y1[idx] - pred_y1),
            image1_width,
            image1_height,
            "View 1 residual",
            cmap="viridis",
            normalize=False,
        )
        _scatter_mode_image(vis / "view2_observed.png", p2_xy[idx], y2[idx], image2_width, image2_height, "View 2 observed")
        _scatter_mode_image(vis / "view2_predicted.png", p2_xy[idx], pred_y2, image2_width, image2_height, "View 2 predicted")
        _scatter_mode_image(
            vis / "view2_residual.png",
            p2_xy[idx],
            (y2[idx] - pred_y2),
            image2_width,
            image2_height,
            "View 2 residual",
            cmap="viridis",
            normalize=False,
        )
        _write_ply(vis / "pointcloud_amplitude.ply", points[idx], _amplitude_colors(phi))
        _write_ply(vis / "pointcloud_phase_u.ply", points[idx], _phase_colors(phi))

    return out
