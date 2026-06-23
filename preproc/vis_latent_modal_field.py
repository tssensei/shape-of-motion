"""Interactive Viser visualization for latent 3D modal fields.

This viewer is intentionally independent of the full SOM/Flow3D viewer. It
only reads the modal_surface optimization output:

    latent_field_12_*.npz

and displays:

    - base 3D surface points;
    - animated harmonic motion X(theta) = X + scale * real(phi * exp(j theta));
    - phase-colored animated points;
    - view1/view2 camera frustums from modal_surface view configs.

The color hue encodes the current per-point modal phase definition. The viewer
is only a display tool; it does not modify the latent field or camera configs.
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


def principal_modal_phases(phi: np.ndarray) -> np.ndarray:
    """Compute the current principal-axis phase for each complex 3D phi.

    The harmonic displacement for one point is:

        d(theta) = real(phi * exp(j theta))
                 = real(phi) cos(theta) - imag(phi) sin(theta)

    This traces a line or ellipse in 3D. The principal direction is the first
    left singular vector of M = [real(phi), -imag(phi)], and the principal
    amplitude is the largest singular value. The phase is measured by projecting
    phi onto that principal direction.
    """
    phi = np.asarray(phi, dtype=np.complex64)
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
            q = -q
        phases[i] = np.float32(np.angle(q))

    return phases


def animated_points(points: np.ndarray, phi: np.ndarray, phase_degrees: float, disp_scale: float) -> np.ndarray:
    """Evaluate X + scale * real(phi * exp(j theta))."""
    theta = np.deg2rad(float(phase_degrees))
    displacement = np.real(phi * np.exp(1j * theta)).astype(np.float32)
    return points.astype(np.float32) + float(disp_scale) * displacement


def compute_display_center(points: np.ndarray, mode: str) -> np.ndarray:
    """Return the world-space point that should become the viewer origin."""
    if mode == "none":
        return np.zeros((3,), dtype=np.float32)
    if mode == "centroid":
        return np.mean(points, axis=0).astype(np.float32)
    if mode == "bbox":
        return (0.5 * (np.min(points, axis=0) + np.max(points, axis=0))).astype(np.float32)
    raise ValueError(f"Unknown center mode {mode!r}.")


def display_camera_pose(cfg: ViewConfig, display_center: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return Viser camera pose fields in display coordinates."""
    import viser.transforms as vtf

    c2w = np.linalg.inv(cfg.world_to_camera)
    fov = float(2.0 * np.arctan(0.5 * cfg.image_height / cfg.K[1, 1]))
    aspect = float(cfg.image_width) / float(cfg.image_height)
    return (
        vtf.SO3.from_matrix(c2w[:3, :3]).wxyz,
        c2w[:3, 3] - display_center,
        fov,
        aspect,
    )


def add_camera_frustum(
    server,
    name: str,
    cfg: ViewConfig,
    color: tuple[int, int, int],
    display_center: np.ndarray,
) -> None:
    """Add a camera frustum from a modal_surface world_to_camera matrix."""
    wxyz, position, fov, aspect = display_camera_pose(cfg, display_center)
    server.scene.add_camera_frustum(
        name,
        fov=fov,
        aspect=aspect,
        scale=0.25,
        color=color,
        wxyz=wxyz,
        position=position,
    )


def set_client_to_view(event, cfg: ViewConfig, display_center: np.ndarray) -> None:
    """Move the active Viser client camera to one modal_surface camera view."""
    if event.client is None:
        return
    wxyz, position, fov, _ = display_camera_pose(cfg, display_center)
    with event.client.atomic():
        event.client.camera.wxyz = wxyz
        event.client.camera.position = position
        event.client.camera.fov = fov


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize latent 3D modal fields with Viser.")
    parser.add_argument("--latent", required=True, type=Path, help="latent_field_12_*.npz from optimize-two-view.")
    parser.add_argument("--view1-config", required=True, type=Path, help="modal_surface view1_config.json.")
    parser.add_argument("--view2-config", required=True, type=Path, help="modal_surface view2_config.json.")
    parser.add_argument("--port", type=int, default=8891, help="Viser server port.")
    parser.add_argument("--disp-scale", type=float, default=0.5, help="Multiplier for animated full-complex displacement.")
    parser.add_argument("--point-size", type=float, default=0.015, help="Animated/base point cloud point size.")
    parser.add_argument("--phase-degrees", type=float, default=0.0, help="Initial animation phase in degrees.")
    parser.add_argument("--fps", type=float, default=12.0, help="Playback FPS for the phase animation.")
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
    phases = principal_modal_phases(phi)
    phase_colors = hsv_colors(phases)
    display_center = compute_display_center(points, args.center_mode)
    points_display = points - display_center[None, :]

    base_color = np.full((points.shape[0], 3), 0.58, dtype=np.float32)
    server = viser.ViserServer(port=args.port, verbose=False)
    add_camera_frustum(server, "/cameras/view1", cfg1, (80, 150, 255), display_center)
    add_camera_frustum(server, "/cameras/view2", cfg2, (255, 130, 70), display_center)

    handles = {"base": None, "animated": None}

    phase_slider = server.gui.add_slider("Phase (deg)", min=0.0, max=360.0, step=1.0, initial_value=float(args.phase_degrees % 360.0))
    disp_scale_slider = server.gui.add_slider("Displacement scale", min=0.0, max=5.0, step=0.01, initial_value=float(args.disp_scale))
    point_size_slider = server.gui.add_slider("Point size", min=0.001, max=0.08, step=0.001, initial_value=float(args.point_size))
    show_base = server.gui.add_checkbox("Show base points", True)
    show_animated = server.gui.add_checkbox("Show animated points", True)
    go_view1_button = server.gui.add_button("Go to view1")
    go_view2_button = server.gui.add_button("Go to view2")
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

    def redraw_all() -> None:
        redraw_base()
        redraw_dynamic()

    phase_slider.on_update(lambda _: redraw_dynamic())
    disp_scale_slider.on_update(lambda _: redraw_dynamic())
    point_size_slider.on_update(lambda _: (redraw_base(), redraw_dynamic()))
    show_base.on_update(lambda _: redraw_base())
    show_animated.on_update(lambda _: redraw_dynamic())
    go_view1_button.on_click(lambda event: set_client_to_view(event, cfg1, display_center))
    go_view2_button.on_click(lambda event: set_client_to_view(event, cfg2, display_center))

    redraw_all()
    print(f"Loaded {points.shape[0]} latent modal points.")
    print(f"Viewer center mode: {args.center_mode}")
    print(f"Viewer display center in original world coordinates: {display_center.tolist()}")
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
