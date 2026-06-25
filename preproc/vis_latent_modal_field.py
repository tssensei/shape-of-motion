"""Interactive Viser visualization for VGGT scenes and modal motion.

Without --latent, this script only shows the fixed VGGT point cloud and camera
poses. With --latent, it displays optimized latent modal points and animates the
complex displacement field:

    X(t) = X0 + scale * Re(phi * exp(i * phase_t))
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OPENGL_CAMERA_CONVERSION = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


@dataclass(frozen=True)
class Camera:
    """VGGT camera fields needed to draw a Viser camera frustum."""

    label: str
    image_width: int
    image_height: int
    K: np.ndarray
    world_to_camera: np.ndarray


def normalize_depth_array(depth: np.ndarray, num_images: int) -> np.ndarray:
    """Normalize common VGGT depth tensor layouts to (N,H,W)."""
    arr = np.asarray(depth)
    if arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Could not normalize depth array to (N,H,W), got {depth.shape}.")
    if arr.shape[0] != num_images:
        raise ValueError(f"Depth view count {arr.shape[0]} does not match image count {num_images}.")
    return arr.astype(np.float32)


def normalize_matrix_array(array: np.ndarray, num_images: int, name: str) -> np.ndarray:
    """Normalize common VGGT camera matrix layouts to one matrix per image."""
    arr = np.asarray(array)
    if arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape[0] != num_images:
        raise ValueError(f"{name} view count {arr.shape[0]} does not match image count {num_images}.")
    return arr


def extrinsic_to_world_to_camera(extrinsic: np.ndarray) -> np.ndarray:
    """Return a 4x4 world-to-camera matrix from VGGT extrinsics."""
    arr = np.asarray(extrinsic, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = arr
        return out
    raise ValueError(f"Expected extrinsic shape (3,4) or (4,4), got {arr.shape}.")


def load_vggt_outputs(path: Path) -> dict[str, np.ndarray]:
    """Load required VGGT raw output arrays."""
    z = np.load(str(path), allow_pickle=False)
    required = ["image_paths", "processed_hw", "extrinsics", "intrinsics"]
    missing = [key for key in required if key not in z.files]
    if missing:
        raise ValueError(f"{path} missing required VGGT output arrays: {missing}.")
    image_paths = z["image_paths"]
    num_images = int(image_paths.shape[0])
    return {
        "image_paths": image_paths,
        "processed_hw": z["processed_hw"].astype(np.int32),
        "extrinsics": normalize_matrix_array(z["extrinsics"], num_images, "extrinsics"),
        "intrinsics": normalize_matrix_array(z["intrinsics"], num_images, "intrinsics"),
    }


def image_label(path_value: np.ndarray | str | bytes, index: int) -> str:
    """Use image filename stems as camera labels when possible."""
    item = np.asarray(path_value).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    stem = Path(str(item)).stem
    return stem if stem else f"view{index + 1}"


def build_cameras(vggt: dict[str, np.ndarray]) -> list[Camera]:
    """Build camera records in VGGT processed-image coordinates."""
    processed_h, processed_w = (int(v) for v in vggt["processed_hw"])
    cameras: list[Camera] = []
    for i in range(vggt["image_paths"].shape[0]):
        cameras.append(
            Camera(
                label=image_label(vggt["image_paths"][i], i),
                image_width=processed_w,
                image_height=processed_h,
                K=np.asarray(vggt["intrinsics"][i], dtype=np.float64),
                world_to_camera=extrinsic_to_world_to_camera(vggt["extrinsics"][i]),
            )
        )
    return cameras


def load_points_from_export(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a point cloud exported by preproc/run_vggt.py --export-points."""
    z = np.load(str(path), allow_pickle=False)
    required = ["points_world", "colors"]
    missing = [key for key in required if key not in z.files]
    if missing:
        raise ValueError(f"{path} missing required point arrays: {missing}.")
    points = z["points_world"].astype(np.float32)
    colors = z["colors"].astype(np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points.shape}.")
    if colors.shape != (points.shape[0], 3):
        raise ValueError(f"colors must have shape (N,3), got {colors.shape}.")
    valid = np.all(np.isfinite(points), axis=1)
    points = points[valid]
    colors = colors[valid]
    if max_points > 0 and points.shape[0] > max_points:
        keep = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
        points = points[keep]
        colors = colors[keep]
    return points, colors


def load_latent_field(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load optimized latent modal points, complex displacement, and optional RGB."""
    z = np.load(str(path), allow_pickle=False)
    required = ["points_world", "phi"]
    missing = [key for key in required if key not in z.files]
    if missing:
        raise ValueError(f"{path} missing required latent arrays: {missing}.")
    points = z["points_world"].astype(np.float32)
    phi = z["phi"].astype(np.complex64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points.shape}.")
    if phi.shape != points.shape:
        raise ValueError(f"phi must have shape {points.shape}, got {phi.shape}.")

    colors = None
    if "colors" in z.files:
        colors = z["colors"].astype(np.uint8)
        if colors.shape != (points.shape[0], 3):
            raise ValueError(f"colors must have shape (N,3), got {colors.shape}.")

    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.all(np.isfinite(phi.real), axis=1)
        & np.all(np.isfinite(phi.imag), axis=1)
    )
    points = points[valid]
    phi = phi[valid]
    if colors is not None:
        colors = colors[valid]
    if max_points > 0 and points.shape[0] > max_points:
        keep = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
        points = points[keep]
        phi = phi[keep]
        if colors is not None:
            colors = colors[keep]
    return points, phi, colors


def load_latent_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a modal_modes_manifest.json and resolve latent paths."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    modes = payload.get("modes")
    if not isinstance(modes, list) or len(modes) == 0:
        raise ValueError(f"{path} must contain a non-empty modes list.")
    out: list[dict[str, Any]] = []
    labels: set[str] = set()
    for i, item in enumerate(modes):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest mode entry {i} is not an object.")
        latent_value = item.get("latent_path")
        if not isinstance(latent_value, str) or not latent_value:
            raise ValueError(f"Manifest mode entry {i} missing latent_path.")
        latent_path = Path(latent_value)
        if not latent_path.is_absolute():
            latent_path = path.parent / latent_path
        freq = float(item.get("freq_hz", 0.0))
        mode_index = int(item.get("mode_index", i))
        label = str(item.get("label", f"{mode_index}: {freq:.6f} Hz"))
        if label in labels:
            label = f"{label} [{i}]"
        labels.add(label)
        entry = dict(item)
        entry["label"] = label
        entry["latent_path"] = latent_path
        out.append(entry)
    return out


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 column-vector transform to row-major points."""
    return (points.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def transform_vectors(vectors: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply only the linear part of a display transform to row-major vectors."""
    return (vectors.astype(np.complex128) @ transform[:3, :3].T).astype(np.complex64)


def hsv_phase_colors(phi: np.ndarray) -> np.ndarray:
    """Color by phase of the strongest complex displacement component."""
    if phi.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    amp = np.abs(phi)
    component = np.argmax(amp, axis=1)
    phase = np.angle(phi[np.arange(phi.shape[0]), component])
    hue = (phase + np.pi) / (2.0 * np.pi)
    h = (hue * 6.0) % 6.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    q = 1.0 - f
    t = f
    rgb = np.zeros((phi.shape[0], 3), dtype=np.float32)
    masks = [i == k for k in range(6)]
    rgb[masks[0]] = np.stack([np.ones_like(f[masks[0]]), t[masks[0]], np.zeros_like(f[masks[0]])], axis=1)
    rgb[masks[1]] = np.stack([q[masks[1]], np.ones_like(f[masks[1]]), np.zeros_like(f[masks[1]])], axis=1)
    rgb[masks[2]] = np.stack([np.zeros_like(f[masks[2]]), np.ones_like(f[masks[2]]), t[masks[2]]], axis=1)
    rgb[masks[3]] = np.stack([np.zeros_like(f[masks[3]]), q[masks[3]], np.ones_like(f[masks[3]])], axis=1)
    rgb[masks[4]] = np.stack([t[masks[4]], np.zeros_like(f[masks[4]]), np.ones_like(f[masks[4]])], axis=1)
    rgb[masks[5]] = np.stack([np.ones_like(f[masks[5]]), np.zeros_like(f[masks[5]]), q[masks[5]]], axis=1)
    return (255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)


def colors_to_float(colors: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB to Viser float colors."""
    return colors.astype(np.float32) / 255.0


def scene_transform(cameras: list[Camera], alignment: str) -> np.ndarray:
    """Return the display-space transform for points and cameras."""
    if alignment == "none":
        return np.eye(4, dtype=np.float64)
    if alignment == "viser":
        return cameras[0].world_to_camera.copy()
    if alignment == "glb":
        return np.linalg.inv(cameras[0].world_to_camera) @ OPENGL_CAMERA_CONVERSION
    raise ValueError(f"Unknown alignment mode {alignment!r}.")


def scene_scale(points: np.ndarray) -> float:
    """Robust point cloud scale used for frustum size and initial camera distance."""
    if points.shape[0] == 0:
        return 1.0
    lo, hi = np.percentile(points, [5, 95], axis=0)
    return max(float(np.linalg.norm(hi - lo)), 1e-3)


def camera_display_pose(camera: Camera, transform: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return Viser pose fields for one VGGT camera."""
    import viser.transforms as vtf

    c2w = np.linalg.inv(camera.world_to_camera)
    c2w_display = transform @ c2w
    fov = float(2.0 * np.arctan(0.5 * camera.image_height / camera.K[1, 1]))
    aspect = float(camera.image_width) / float(camera.image_height)
    return (
        vtf.SO3.from_matrix(c2w_display[:3, :3]).wxyz,
        c2w_display[:3, 3],
        fov,
        aspect,
    )


def add_camera_frustum(server, camera: Camera, transform: np.ndarray, color: tuple[int, int, int], scale: float):
    """Add one VGGT camera frustum."""
    wxyz, position, fov, aspect = camera_display_pose(camera, transform)
    return server.scene.add_camera_frustum(
        f"/cameras/{camera.label}",
        fov=fov,
        aspect=aspect,
        scale=scale,
        color=color,
        wxyz=wxyz,
        position=position,
    )


def set_client_to_camera(event, camera: Camera, transform: np.ndarray) -> None:
    """Move the active Viser client camera to a VGGT camera pose."""
    if event.client is None:
        return
    wxyz, position, fov, _ = camera_display_pose(camera, transform)
    with event.client.atomic():
        event.client.camera.wxyz = wxyz
        event.client.camera.position = position
        event.client.camera.fov = fov


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the raw VGGT point cloud and camera poses with Viser.")
    parser.add_argument("--vggt-outputs", required=True, type=Path, help="vggt_outputs.npz from preproc/run_vggt.py.")
    parser.add_argument("--points", required=True, type=Path, help="Fixed vggt_points.npz from --export-points.")
    parser.add_argument("--latent", type=Path, default=None, help="Optional latent_field_*.npz to animate modal displacement.")
    parser.add_argument("--latent-manifest", type=Path, default=None, help="Optional modal_modes_manifest.json for frequency selection.")
    parser.add_argument("--port", type=int, default=8891, help="Viser server port.")
    parser.add_argument("--point-size", type=float, default=0.004, help="Point cloud point size.")
    parser.add_argument("--max-points", type=int, default=10000, help="Maximum points to display; 0 keeps all.")
    parser.add_argument("--fps", type=float, default=12.0, help="Initial modal animation FPS.")
    parser.add_argument("--motion-scale", type=float, default=0.02, help="Initial modal displacement scale.")
    parser.add_argument("--max-motion-scale", type=float, default=0.2, help="Maximum GUI modal displacement scale.")
    parser.add_argument(
        "--alignment",
        choices=("viser", "glb", "none"),
        default="viser",
        help="Display transform. 'viser' uses camera-0 coordinates; 'glb' keeps the old VGGT GLB-style axis conversion.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    import viser

    vggt = load_vggt_outputs(args.vggt_outputs.expanduser())
    cameras = build_cameras(vggt)
    raw_points, raw_colors = load_points_from_export(args.points.expanduser(), args.max_points)
    if args.latent is not None and args.latent_manifest is not None:
        raise ValueError("Use either --latent or --latent-manifest, not both.")
    latent_entries = load_latent_manifest(args.latent_manifest.expanduser()) if args.latent_manifest is not None else []
    initial_latent = args.latent.expanduser() if args.latent is not None else None
    if initial_latent is None and latent_entries:
        initial_latent = latent_entries[0]["latent_path"]

    initial_points = raw_points
    initial_colors = raw_colors
    initial_phi = None
    initial_source = str(args.points)
    initial_has_rgb = False
    if initial_latent is not None:
        initial_points, initial_phi, latent_colors = load_latent_field(initial_latent, args.max_points)
        initial_source = str(initial_latent)
        if latent_colors is not None:
            initial_colors = latent_colors
            initial_has_rgb = True
        else:
            initial_colors = hsv_phase_colors(initial_phi)
    if initial_points.shape[0] == 0:
        raise ValueError("No points were loaded.")

    transform = scene_transform(cameras, args.alignment)

    def make_display_state(
        points_in: np.ndarray,
        phi_in: np.ndarray | None,
        colors_in: np.ndarray,
        has_rgb: bool,
        source: str,
    ) -> dict[str, Any]:
        base_points = transform_points(points_in, transform)
        display_phi_local = transform_vectors(phi_in, transform) if phi_in is not None else None
        rgb = colors_to_float(colors_in)
        phase = colors_to_float(hsv_phase_colors(phi_in)) if phi_in is not None else rgb
        return {
            "base_display_points": base_points,
            "display_phi": display_phi_local,
            "phi": phi_in,
            "rgb_colors": rgb,
            "phase_colors": phase,
            "latent_has_rgb": has_rgb,
            "point_source": source,
        }

    state = make_display_state(initial_points, initial_phi, initial_colors, initial_has_rgb, initial_source)
    scale = scene_scale(state["base_display_points"])
    frustum_scale = 0.08 * scale

    server = viser.ViserServer(port=args.port, verbose=False)
    point_handle = {"handle": None}
    point_lock = threading.Lock()
    animation = {"phase": 0.0, "motion_scale": float(args.motion_scale)}

    def current_points() -> np.ndarray:
        if state["display_phi"] is None:
            return state["base_display_points"]
        phase = float(animation["phase"])
        displacement = np.real(state["display_phi"] * np.exp(1j * phase)).astype(np.float32)
        return (state["base_display_points"] + float(animation["motion_scale"]) * displacement).astype(np.float32)

    def current_colors() -> np.ndarray:
        if state["phi"] is not None and "color_scheme" in gui_handles and gui_handles["color_scheme"].value == "phase":
            return state["phase_colors"]
        return state["rgb_colors"]

    def redraw_points(point_size: float) -> None:
        with point_lock:
            if point_handle["handle"] is not None:
                point_handle["handle"].remove()
            point_handle["handle"] = server.scene.add_point_cloud(
                "/vggt/points",
                points=current_points(),
                colors=current_colors(),
                point_size=float(point_size),
            )

    gui_handles = {}
    redraw_points(float(args.point_size))

    camera_colors = [
        (80, 150, 255),
        (255, 130, 70),
        (95, 200, 120),
        (210, 120, 255),
        (255, 210, 80),
        (80, 220, 220),
    ]
    camera_handles = {}
    for i, camera in enumerate(cameras):
        camera_handles[camera.label] = add_camera_frustum(
            server,
            camera,
            transform,
            camera_colors[i % len(camera_colors)],
            frustum_scale,
        )
        button = server.gui.add_button(f"Go to {camera.label}")
        button.on_click(lambda event, camera=camera: set_client_to_camera(event, camera, transform))

    show_cameras = server.gui.add_checkbox("Show cameras", True)
    point_size_slider = server.gui.add_slider("Point size", min=0.0002, max=0.008, step=0.0001, initial_value=float(args.point_size))
    gui_handles["point_size"] = point_size_slider
    if latent_entries:
        labels = tuple(str(entry["label"]) for entry in latent_entries)
        frequency_dropdown = server.gui.add_dropdown("Frequency", labels, initial_value=labels[0])
        gui_handles["frequency"] = frequency_dropdown
        entry_by_label = {str(entry["label"]): entry for entry in latent_entries}
    else:
        entry_by_label = {}
    if state["phi"] is not None:
        play_checkbox = server.gui.add_checkbox("Play", False)
        fps_slider = server.gui.add_slider("FPS", min=1.0, max=60.0, step=1.0, initial_value=float(args.fps))
        max_motion_scale = max(float(args.max_motion_scale), float(args.motion_scale), 1e-6)
        motion_scale_slider = server.gui.add_slider(
            "Motion scale",
            min=0.0,
            max=max_motion_scale,
            step=max_motion_scale / 200.0,
            initial_value=float(args.motion_scale),
        )
        color_options = ("rgb", "phase") if state["latent_has_rgb"] else ("phase",)
        color_scheme = server.gui.add_dropdown("Color scheme", color_options, initial_value=color_options[0])
        gui_handles["play"] = play_checkbox
        gui_handles["fps"] = fps_slider
        gui_handles["motion_scale"] = motion_scale_slider
        gui_handles["color_scheme"] = color_scheme

    def update_camera_visibility(_) -> None:
        for handle in camera_handles.values():
            handle.visible = bool(show_cameras.value)

    def update_point_size(_) -> None:
        redraw_points(float(point_size_slider.value))

    def update_motion_scale(_) -> None:
        if state["phi"] is None:
            return
        animation["motion_scale"] = float(gui_handles["motion_scale"].value)
        redraw_points(float(gui_handles["point_size"].value))

    def update_color_scheme(_) -> None:
        redraw_points(float(gui_handles["point_size"].value))

    def update_frequency(_) -> None:
        nonlocal state
        label = str(gui_handles["frequency"].value)
        entry = entry_by_label[label]
        latent_path = Path(entry["latent_path"]).expanduser()
        points_new, phi_new, colors_new = load_latent_field(latent_path, args.max_points)
        has_rgb = colors_new is not None
        if colors_new is None:
            colors_new = hsv_phase_colors(phi_new)
        state = make_display_state(points_new, phi_new, colors_new, has_rgb, str(latent_path))
        animation["phase"] = 0.0
        redraw_points(float(gui_handles["point_size"].value))

    show_cameras.on_update(update_camera_visibility)
    point_size_slider.on_update(update_point_size)
    if latent_entries:
        gui_handles["frequency"].on_update(update_frequency)
    if state["phi"] is not None:
        gui_handles["motion_scale"].on_update(update_motion_scale)
        gui_handles["color_scheme"].on_update(update_color_scheme)

        def animate_points() -> None:
            while True:
                fps = max(float(gui_handles["fps"].value), 1.0)
                if bool(gui_handles["play"].value):
                    animation["phase"] = (float(animation["phase"]) + 2.0 * np.pi / fps) % (2.0 * np.pi)
                    redraw_points(float(gui_handles["point_size"].value))
                time.sleep(1.0 / fps)

        threading.Thread(target=animate_points, daemon=True).start()

    print(f"Loaded displayed points: {state['base_display_points'].shape[0]} from {state['point_source']}")
    if args.latent is not None or args.latent_manifest is not None:
        if args.latent_manifest is not None:
            print(f"Loaded latent manifest: {args.latent_manifest}")
        print(f"Loaded latent modal displacement: {state['point_source']}")
        print(f"Initial motion scale: {args.motion_scale}")
    print(f"Loaded VGGT cameras: {[camera.label for camera in cameras]}")
    print(f"Alignment: {args.alignment}")
    print(f"Display scene scale: {scale:.6g}")
    for camera in cameras:
        center = np.linalg.inv(camera.world_to_camera)[:3, 3]
        display_center = transform_points(center[None, :], transform)[0]
        print(f"{camera.label} raw center {center.tolist()} display center {display_center.tolist()}")
    print(f"Viser server running on http://localhost:{args.port}")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
