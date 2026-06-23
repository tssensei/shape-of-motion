"""Interactive Viser visualization for latent 3D modal fields.

This viewer is intentionally independent of the full SOM/Flow3D viewer. It
only reads the modal_surface optimization output:

    latent_field_12_*.npz

and displays:

    - base 3D surface points;
    - animated harmonic motion X(theta) = X + scale * real(phi * exp(j theta));
    - principal-direction arrows whose length is modal amplitude and whose hue
      is modal phase;
    - view1/view2 camera frustums from modal_surface view configs.

The arrows follow a Davis-style modal visualization convention: geometry
encodes direction and amplitude, while color hue encodes phase. Since a 3D
complex displacement can describe a local ellipse rather than a single line,
the arrow direction is the principal axis of the per-point harmonic motion.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ViewConfig:
    """Minimal camera fields needed to draw a Viser camera frustum."""

    image_width: int
    image_height: int
    K: np.ndarray
    world_to_camera: np.ndarray


def load_view_config(path: Path) -> ViewConfig:
    """Load the subset of a modal_surface view_config needed by the viewer."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    K = np.asarray(raw["K"], dtype=np.float64)
    world_to_camera = np.asarray(raw["world_to_camera"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"{path} K must have shape (3,3), got {K.shape}.")
    if world_to_camera.shape != (4, 4):
        raise ValueError(f"{path} world_to_camera must have shape (4,4), got {world_to_camera.shape}.")
    return ViewConfig(
        image_width=int(raw["image_width"]),
        image_height=int(raw["image_height"]),
        K=K,
        world_to_camera=world_to_camera,
    )


def hsv_colors(phase: np.ndarray) -> np.ndarray:
    """Map phase in radians to RGB hue colors in [0,1]."""
    hue = (phase + np.pi) / (2.0 * np.pi)
    hue = np.mod(hue, 1.0)
    rgb = [colorsys.hsv_to_rgb(float(h), 1.0, 1.0) for h in hue]
    return np.asarray(rgb, dtype=np.float32)


def principal_modal_axes(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute principal axis, amplitude, and phase for each complex 3D phi.

    The harmonic displacement for one point is:

        d(theta) = real(phi * exp(j theta))
                 = real(phi) cos(theta) - imag(phi) sin(theta)

    This traces a line or ellipse in 3D. The principal direction is the first
    left singular vector of M = [real(phi), -imag(phi)], and the principal
    amplitude is the largest singular value. The phase is measured by projecting
    phi onto that principal direction.
    """
    phi = np.asarray(phi, dtype=np.complex64)
    directions = np.zeros((phi.shape[0], 3), dtype=np.float32)
    amplitudes = np.zeros((phi.shape[0],), dtype=np.float32)
    phases = np.zeros((phi.shape[0],), dtype=np.float32)
    real = np.real(phi).astype(np.float64)
    neg_imag = -np.imag(phi).astype(np.float64)

    for i in range(phi.shape[0]):
        M = np.stack([real[i], neg_imag[i]], axis=1)
        if not np.isfinite(M).all():
            continue
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        direction = U[:, 0]
        q = complex(np.dot(direction, phi[i]))
        # The principal axis has a sign ambiguity. Orient it so the projected
        # complex scalar has nonnegative real part, which makes hue stable.
        if np.real(q) < 0:
            direction = -direction
            q = -q
        directions[i] = direction.astype(np.float32)
        amplitudes[i] = np.float32(S[0])
        phases[i] = np.float32(np.angle(q))

    return directions, amplitudes, phases


def select_arrow_indices(amplitudes: np.ndarray, max_arrows: int) -> np.ndarray:
    """Select a deterministic subset of arrows, preferring larger amplitudes."""
    n = amplitudes.shape[0]
    if max_arrows <= 0 or max_arrows >= n:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(amplitudes)[::-1]
    return np.sort(order[:max_arrows]).astype(np.int64)


def animated_points(points: np.ndarray, phi: np.ndarray, phase_degrees: float, disp_scale: float) -> np.ndarray:
    """Evaluate X + scale * real(phi * exp(j theta))."""
    theta = np.deg2rad(float(phase_degrees))
    displacement = np.real(phi * np.exp(1j * theta)).astype(np.float32)
    return points.astype(np.float32) + float(disp_scale) * displacement


def make_arrow_point_cloud(
    points: np.ndarray,
    directions: np.ndarray,
    amplitudes: np.ndarray,
    colors: np.ndarray,
    arrow_scale: float,
    shaft_samples: int,
    head_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Represent arrows as a colored point cloud sampled on shafts and heads."""
    if shaft_samples < 2:
        raise ValueError("shaft_samples must be >= 2.")
    if head_samples < 2:
        raise ValueError("head_samples must be >= 2.")

    amp_ref = float(np.percentile(amplitudes[amplitudes > 0], 95)) if np.any(amplitudes > 0) else 1.0
    amp_ref = max(amp_ref, 1e-8)
    arrow_points: list[np.ndarray] = []
    arrow_colors: list[np.ndarray] = []
    shaft_t = np.linspace(0.0, 1.0, shaft_samples, dtype=np.float32)[:, None]
    head_t = np.linspace(0.0, 1.0, head_samples, dtype=np.float32)[:, None]

    for point, direction, amplitude, color in zip(points, directions, amplitudes, colors):
        length = float(arrow_scale) * float(amplitude) / amp_ref
        if not np.isfinite(length) or length <= 0:
            continue
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            continue
        unit = direction.astype(np.float32) / norm
        start = point.astype(np.float32)
        end = start + np.float32(length) * unit

        shaft = start[None, :] * (1.0 - shaft_t) + end[None, :] * shaft_t
        arrow_points.append(shaft)
        arrow_colors.append(np.repeat(color[None, :], shaft.shape[0], axis=0))

        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(ref, unit))) > 0.95:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        side = np.cross(unit, ref)
        side_norm = float(np.linalg.norm(side))
        if side_norm <= 1e-8:
            continue
        side = side / side_norm
        head_len = np.float32(0.25 * length)
        head_width = np.float32(0.10 * length)
        for sign in (-1.0, 1.0):
            head_end = end - head_len * unit + np.float32(sign) * head_width * side
            head = end[None, :] * (1.0 - head_t) + head_end[None, :] * head_t
            arrow_points.append(head)
            arrow_colors.append(np.repeat(color[None, :], head.shape[0], axis=0))

    if not arrow_points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(arrow_points, axis=0), np.concatenate(arrow_colors, axis=0)


def compute_display_center(points: np.ndarray, mode: str) -> np.ndarray:
    """Return the world-space point that should become the viewer origin."""
    if mode == "none":
        return np.zeros((3,), dtype=np.float32)
    if mode == "centroid":
        return np.mean(points, axis=0).astype(np.float32)
    if mode == "bbox":
        return (0.5 * (np.min(points, axis=0) + np.max(points, axis=0))).astype(np.float32)
    raise ValueError(f"Unknown center mode {mode!r}.")


def add_camera_frustum(
    server,
    name: str,
    cfg: ViewConfig,
    color: tuple[int, int, int],
    display_center: np.ndarray,
) -> None:
    """Add a camera frustum from a modal_surface world_to_camera matrix."""
    import viser.transforms as vtf

    c2w = np.linalg.inv(cfg.world_to_camera)
    fov = float(2.0 * np.arctan(0.5 * cfg.image_height / cfg.K[1, 1]))
    aspect = float(cfg.image_width) / float(cfg.image_height)
    server.scene.add_camera_frustum(
        name,
        fov=fov,
        aspect=aspect,
        scale=0.25,
        color=color,
        wxyz=vtf.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3] - display_center,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize latent 3D modal fields with Viser.")
    parser.add_argument("--latent", required=True, type=Path, help="latent_field_12_*.npz from optimize-two-view.")
    parser.add_argument("--view1-config", required=True, type=Path, help="modal_surface view1_config.json.")
    parser.add_argument("--view2-config", required=True, type=Path, help="modal_surface view2_config.json.")
    parser.add_argument("--port", type=int, default=8891, help="Viser server port.")
    parser.add_argument("--max-arrows", type=int, default=1500, help="Maximum principal-direction arrows to display.")
    parser.add_argument("--arrow-scale", type=float, default=0.15, help="World-space length of the p95 principal arrow.")
    parser.add_argument("--disp-scale", type=float, default=0.5, help="Multiplier for animated full-complex displacement.")
    parser.add_argument("--point-size", type=float, default=0.015, help="Animated/base point cloud point size.")
    parser.add_argument("--arrow-point-size", type=float, default=0.01, help="Point size used to draw sampled arrows.")
    parser.add_argument("--phase-degrees", type=float, default=0.0, help="Initial animation phase in degrees.")
    parser.add_argument("--fps", type=float, default=12.0, help="Playback FPS for the phase animation.")
    parser.add_argument("--shaft-samples", type=int, default=8, help="Number of points sampled along each arrow shaft.")
    parser.add_argument("--head-samples", type=int, default=4, help="Number of points sampled along each arrow head segment.")
    parser.add_argument(
        "--center-mode",
        choices=("bbox", "centroid", "none"),
        default="bbox",
        help="Viewer-only recentering mode. Does not modify latent/config files.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    import viser

    latent = np.load(str(args.latent), allow_pickle=False)
    required = ["points_world", "phi"]
    missing = [key for key in required if key not in latent.files]
    if missing:
        raise ValueError(f"{args.latent} missing required arrays: {missing}")

    points = latent["points_world"].astype(np.float32)
    phi = latent["phi"].astype(np.complex64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points.shape}")
    if phi.shape != points.shape:
        raise ValueError(f"phi must have shape {points.shape}, got {phi.shape}")

    cfg1 = load_view_config(args.view1_config)
    cfg2 = load_view_config(args.view2_config)
    directions, amplitudes, phases = principal_modal_axes(phi)
    arrow_idx = select_arrow_indices(amplitudes, args.max_arrows)
    phase_colors = hsv_colors(phases)
    display_center = compute_display_center(points, args.center_mode)
    points_display = points - display_center[None, :]

    base_color = np.full((points.shape[0], 3), 0.58, dtype=np.float32)
    server = viser.ViserServer(port=args.port, verbose=False)
    add_camera_frustum(server, "/cameras/view1", cfg1, (80, 150, 255), display_center)
    add_camera_frustum(server, "/cameras/view2", cfg2, (255, 130, 70), display_center)

    handles = {"base": None, "animated": None, "arrows": None}

    phase_slider = server.gui.add_slider("Phase (deg)", min=0.0, max=360.0, step=1.0, initial_value=float(args.phase_degrees % 360.0))
    disp_scale_slider = server.gui.add_slider("Displacement scale", min=0.0, max=5.0, step=0.01, initial_value=float(args.disp_scale))
    arrow_scale_slider = server.gui.add_slider("Arrow scale", min=0.0, max=2.0, step=0.01, initial_value=float(args.arrow_scale))
    point_size_slider = server.gui.add_slider("Point size", min=0.001, max=0.08, step=0.001, initial_value=float(args.point_size))
    arrow_point_size_slider = server.gui.add_slider("Arrow point size", min=0.001, max=0.06, step=0.001, initial_value=float(args.arrow_point_size))
    show_base = server.gui.add_checkbox("Show base points", True)
    show_animated = server.gui.add_checkbox("Show animated points", True)
    show_arrows = server.gui.add_checkbox("Show principal arrows", False)
    playing = server.gui.add_checkbox("Play", False)
    fps_slider = server.gui.add_slider("FPS", min=1.0, max=60.0, step=1.0, initial_value=float(args.fps))

    def redraw_base() -> None:
        if handles["base"] is not None:
            handles["base"].remove()
            handles["base"] = None
        if show_base.value:
            handles["base"] = server.scene.add_point_cloud(
                "/modal/base_points",
                points=points_display,
                colors=base_color,
                point_size=float(point_size_slider.value),
            )

    def redraw_dynamic() -> None:
        if handles["animated"] is not None:
            handles["animated"].remove()
            handles["animated"] = None
        if show_animated.value:
            moved = animated_points(points_display, phi, float(phase_slider.value), float(disp_scale_slider.value))
            handles["animated"] = server.scene.add_point_cloud(
                "/modal/animated_points",
                points=moved,
                colors=phase_colors,
                point_size=float(point_size_slider.value),
            )

    def redraw_arrows() -> None:
        if handles["arrows"] is not None:
            handles["arrows"].remove()
            handles["arrows"] = None
        if show_arrows.value:
            arrow_points, arrow_colors = make_arrow_point_cloud(
                points_display[arrow_idx],
                directions[arrow_idx],
                amplitudes[arrow_idx],
                phase_colors[arrow_idx],
                float(arrow_scale_slider.value),
                int(args.shaft_samples),
                int(args.head_samples),
            )
            handles["arrows"] = server.scene.add_point_cloud(
                "/modal/principal_arrows",
                points=arrow_points,
                colors=arrow_colors,
                point_size=float(arrow_point_size_slider.value),
            )

    def redraw_all() -> None:
        redraw_base()
        redraw_dynamic()
        redraw_arrows()

    phase_slider.on_update(lambda _: redraw_dynamic())
    disp_scale_slider.on_update(lambda _: redraw_dynamic())
    point_size_slider.on_update(lambda _: (redraw_base(), redraw_dynamic()))
    arrow_scale_slider.on_update(lambda _: redraw_arrows())
    arrow_point_size_slider.on_update(lambda _: redraw_arrows())
    show_base.on_update(lambda _: redraw_base())
    show_animated.on_update(lambda _: redraw_dynamic())
    show_arrows.on_update(lambda _: redraw_arrows())

    redraw_all()
    print(f"Loaded {points.shape[0]} latent modal points.")
    print(f"Viewer center mode: {args.center_mode}")
    print(f"Viewer display center in original world coordinates: {display_center.tolist()}")
    print(f"Showing {arrow_idx.shape[0]} principal-direction arrows.")
    print(f"Viser server running on http://localhost:{args.port}")

    def playback_loop() -> None:
        while True:
            if playing.value:
                phase_slider.value = (float(phase_slider.value) + 360.0 / max(float(fps_slider.value), 1.0)) % 360.0
            time.sleep(1.0 / max(float(fps_slider.value), 1.0))

    threading.Thread(target=playback_loop, daemon=True).start()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
