"""Interactive Viser visualization for VGGT scenes and modal motion.

Without --latent, this script only shows the fixed VGGT point cloud and camera
poses. With --latent, it displays optimized latent modal points and animates the
complex displacement field:

    X(t) = X0 + scale * Re(phi * exp(i * phase_t))

With --latent-manifest, it loads all listed modal fields as a Davis-style modal
basis and synthesizes motion through complex modal coordinates q_k(t):

    X(t) = X0 + scale * Re(sum_k phi_k * q_k(t))
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


@dataclass(frozen=True)
class ModalRuntimeData:
    """Stacked latent modal fields sharing one carrier point set."""

    points: np.ndarray
    phi_modes: np.ndarray
    freqs_hz: np.ndarray
    labels: tuple[str, ...]
    colors: np.ndarray | None
    source: str


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


def _load_latent_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load one latent .npz without sampling so manifests can be validated jointly."""
    z = np.load(str(path), allow_pickle=False)
    required = ["points_world", "phi"]
    missing = [key for key in required if key not in z.files]
    if missing:
        raise ValueError(f"{path} missing required latent arrays: {missing}.")
    points = z["points_world"].astype(np.float32)
    phi = z["phi"].astype(np.complex64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path} points_world must have shape (N,3), got {points.shape}.")
    if phi.shape != points.shape:
        raise ValueError(f"{path} phi must have shape {points.shape}, got {phi.shape}.")
    out: dict[str, np.ndarray] = {"points": points, "phi": phi}
    if "freq_hz" in z.files:
        out["freq_hz"] = np.asarray(z["freq_hz"], dtype=np.float32).reshape(())
    if "colors" in z.files:
        colors = z["colors"].astype(np.uint8)
        if colors.shape != (points.shape[0], 3):
            raise ValueError(f"{path} colors must have shape (N,3), got {colors.shape}.")
        out["colors"] = colors
    return out


def load_single_runtime_mode(path: Path, max_points: int) -> ModalRuntimeData:
    """Load one latent field as a one-mode runtime basis."""
    arrays = _load_latent_arrays(path)
    points = arrays["points"]
    phi = arrays["phi"]
    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.all(np.isfinite(phi.real), axis=1)
        & np.all(np.isfinite(phi.imag), axis=1)
    )
    points = points[valid]
    phi = phi[valid]
    colors = arrays.get("colors")
    if colors is not None:
        colors = colors[valid]
    if max_points > 0 and points.shape[0] > max_points:
        keep = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
        points = points[keep]
        phi = phi[keep]
        if colors is not None:
            colors = colors[keep]
    freq = float(np.asarray(arrays.get("freq_hz", np.array(1.0, dtype=np.float32))).item())
    return ModalRuntimeData(
        points=points,
        phi_modes=phi[None, :, :].astype(np.complex64, copy=False),
        freqs_hz=np.asarray([freq], dtype=np.float32),
        labels=(f"0: {freq:.6f} Hz",),
        colors=colors,
        source=str(path),
    )


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


def load_manifest_runtime_modes(path: Path, max_points: int) -> ModalRuntimeData:
    """Load all manifest modes as one Davis-style modal basis."""
    entries = load_latent_manifest(path)
    first: dict[str, np.ndarray] | None = None
    phi_list: list[np.ndarray] = []
    freqs: list[float] = []
    labels: list[str] = []
    valid: np.ndarray | None = None
    colors: np.ndarray | None = None
    for mode_i, entry in enumerate(entries):
        latent_path = Path(entry["latent_path"]).expanduser()
        arrays = _load_latent_arrays(latent_path)
        points = arrays["points"]
        phi = arrays["phi"]
        if first is None:
            first = arrays
            valid = np.all(np.isfinite(points), axis=1)
            if "colors" in arrays:
                colors = arrays["colors"]
        else:
            ref_points = first["points"]
            if points.shape != ref_points.shape:
                raise ValueError(
                    f"Manifest mode {entry['label']} has points_world shape {points.shape}, "
                    f"expected {ref_points.shape}. Re-solve modes with consistent pruning, e.g. --outlier-frac 0.0."
                )
            if not np.allclose(points, ref_points, rtol=1e-5, atol=1e-5):
                raise ValueError(
                    f"Manifest mode {entry['label']} does not share the same points_world as the first mode. "
                    "Davis-style modal superposition requires one common carrier point set."
                )
        assert valid is not None
        valid &= np.all(np.isfinite(phi.real), axis=1) & np.all(np.isfinite(phi.imag), axis=1)
        phi_list.append(phi)
        freq = float(entry.get("freq_hz", np.asarray(arrays.get("freq_hz", np.array(0.0))).item()))
        freqs.append(freq)
        labels.append(str(entry.get("label", f"{mode_i}: {freq:.6f} Hz")))
    if first is None or valid is None:
        raise ValueError(f"{path} did not contain any latent modes.")

    points_out = first["points"][valid]
    phi_out = np.stack([phi[valid] for phi in phi_list], axis=0).astype(np.complex64, copy=False)
    colors_out = colors[valid] if colors is not None else None
    if max_points > 0 and points_out.shape[0] > max_points:
        keep = np.linspace(0, points_out.shape[0] - 1, int(max_points), dtype=np.int64)
        points_out = points_out[keep]
        phi_out = phi_out[:, keep, :]
        if colors_out is not None:
            colors_out = colors_out[keep]
    return ModalRuntimeData(
        points=points_out.astype(np.float32, copy=False),
        phi_modes=phi_out,
        freqs_hz=np.asarray(freqs, dtype=np.float32),
        labels=tuple(labels),
        colors=colors_out.astype(np.uint8, copy=False) if colors_out is not None else None,
        source=str(path),
    )


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


def hsv_phase_colors_for_modes(phi_modes: np.ndarray) -> np.ndarray:
    """Color by the phase of the strongest component across all modes."""
    if phi_modes.shape[0] == 1:
        return hsv_phase_colors(phi_modes[0])
    if phi_modes.shape[1] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    amp = np.abs(phi_modes)
    flat_index = np.argmax(amp.transpose(1, 0, 2).reshape(phi_modes.shape[1], -1), axis=1)
    mode_index = flat_index // 3
    component = flat_index % 3
    phase = np.angle(phi_modes[mode_index, np.arange(phi_modes.shape[1]), component])
    pseudo_phi = np.zeros((phi_modes.shape[1], 3), dtype=np.complex64)
    pseudo_phi[np.arange(phi_modes.shape[1]), component] = np.exp(1j * phase).astype(np.complex64)
    return hsv_phase_colors(pseudo_phi)


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
    parser.add_argument("--latent-manifest", type=Path, default=None, help="Optional modal_modes_manifest.json for Davis-style multi-mode synthesis.")
    parser.add_argument("--port", type=int, default=8891, help="Viser server port.")
    parser.add_argument("--point-size", type=float, default=0.004, help="Point cloud point size.")
    parser.add_argument("--max-points", type=int, default=10000, help="Maximum points to display; 0 keeps all.")
    parser.add_argument("--fps", type=float, default=12.0, help="Initial modal animation FPS.")
    parser.add_argument("--motion-scale", type=float, default=0.02, help="Initial modal displacement scale.")
    parser.add_argument("--max-motion-scale", type=float, default=0.2, help="Maximum GUI modal displacement scale.")
    parser.add_argument("--drive-mode", choices=("oscillator", "free_decay"), default="oscillator", help="Initial modal coordinate drive mode.")
    parser.add_argument("--damping", type=float, default=0.05, help="Initial damping ratio for free-decay modal dynamics.")
    parser.add_argument("--mode-gain", type=float, default=1.0, help="Initial per-mode gain.")
    parser.add_argument("--max-mode-gain", type=float, default=3.0, help="Maximum GUI per-mode gain.")
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
    runtime_data: ModalRuntimeData | None = None
    if args.latent_manifest is not None:
        runtime_data = load_manifest_runtime_modes(args.latent_manifest.expanduser(), args.max_points)
    elif args.latent is not None:
        runtime_data = load_single_runtime_mode(args.latent.expanduser(), args.max_points)

    initial_points = raw_points
    initial_colors = raw_colors
    initial_phi_modes = None
    initial_source = str(args.points)
    initial_has_rgb = False
    if runtime_data is not None:
        initial_points = runtime_data.points
        initial_phi_modes = runtime_data.phi_modes
        initial_source = runtime_data.source
        if runtime_data.colors is not None:
            initial_colors = runtime_data.colors
            initial_has_rgb = True
        else:
            initial_colors = hsv_phase_colors_for_modes(initial_phi_modes)
    if initial_points.shape[0] == 0:
        raise ValueError("No points were loaded.")

    transform = scene_transform(cameras, args.alignment)

    def make_display_state(
        points_in: np.ndarray,
        phi_modes_in: np.ndarray | None,
        colors_in: np.ndarray,
        has_rgb: bool,
        source: str,
    ) -> dict[str, Any]:
        base_points = transform_points(points_in, transform)
        display_phi_local = transform_vectors(phi_modes_in, transform) if phi_modes_in is not None else None
        rgb = colors_to_float(colors_in)
        phase = colors_to_float(hsv_phase_colors_for_modes(phi_modes_in)) if phi_modes_in is not None else rgb
        return {
            "base_display_points": base_points,
            "display_phi": display_phi_local,
            "phi_modes": phi_modes_in,
            "rgb_colors": rgb,
            "phase_colors": phase,
            "latent_has_rgb": has_rgb,
            "point_source": source,
        }

    state = make_display_state(initial_points, initial_phi_modes, initial_colors, initial_has_rgb, initial_source)
    scale = scene_scale(state["base_display_points"])
    frustum_scale = 0.08 * scale

    server = viser.ViserServer(port=args.port, verbose=False)
    point_handle = {"handle": None}
    point_lock = threading.Lock()
    animation = {"motion_scale": float(args.motion_scale)}
    runtime = None
    if runtime_data is not None:
        runtime = {
            "time": 0.0,
            "omega": (2.0 * np.pi * np.maximum(runtime_data.freqs_hz.astype(np.float64), 1e-6)).astype(np.float64),
            "q": np.zeros(runtime_data.freqs_hz.shape[0], dtype=np.complex64),
            "qdot": np.zeros(runtime_data.freqs_hz.shape[0], dtype=np.complex64),
        }

    def current_points() -> np.ndarray:
        if state["display_phi"] is None or runtime is None:
            return state["base_display_points"]
        displacement = np.real(np.einsum("knc,k->nc", state["display_phi"], runtime["q"])).astype(np.float32)
        return (state["base_display_points"] + float(animation["motion_scale"]) * displacement).astype(np.float32)

    def current_colors() -> np.ndarray:
        if state["phi_modes"] is not None and "color_scheme" in gui_handles and gui_handles["color_scheme"].value == "phase":
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
    mode_handles: list[dict[str, Any]] = []
    if runtime_data is not None:
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
        drive_mode = server.gui.add_dropdown("Drive mode", ("oscillator", "free_decay"), initial_value=str(args.drive_mode))
        damping_slider = server.gui.add_slider("Damping", min=0.0, max=0.5, step=0.005, initial_value=float(args.damping))
        color_options = ("rgb", "phase") if state["latent_has_rgb"] else ("phase",)
        color_scheme = server.gui.add_dropdown("Color scheme", color_options, initial_value=color_options[0])
        reset_button = server.gui.add_button("Reset modal state")
        impulse_button = server.gui.add_button("Trigger impulse")
        gui_handles["play"] = play_checkbox
        gui_handles["fps"] = fps_slider
        gui_handles["motion_scale"] = motion_scale_slider
        gui_handles["drive_mode"] = drive_mode
        gui_handles["damping"] = damping_slider
        gui_handles["color_scheme"] = color_scheme
        gui_handles["reset"] = reset_button
        gui_handles["impulse"] = impulse_button

        max_mode_gain = max(float(args.max_mode_gain), float(args.mode_gain), 1e-6)
        for mode_i, label in enumerate(runtime_data.labels):
            mode_handles.append(
                {
                    "enabled": server.gui.add_checkbox(f"Mode {mode_i} enabled {label}", True),
                    "gain": server.gui.add_slider(
                        f"Mode {mode_i} gain",
                        min=0.0,
                        max=max_mode_gain,
                        step=max_mode_gain / 200.0,
                        initial_value=float(args.mode_gain),
                    ),
                    "phase": server.gui.add_slider(
                        f"Mode {mode_i} phase",
                        min=-float(np.pi),
                        max=float(np.pi),
                        step=0.01,
                        initial_value=0.0,
                    ),
                }
            )

    def mode_control_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if runtime_data is None:
            return (
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        enabled = np.asarray([bool(handles["enabled"].value) for handles in mode_handles], dtype=bool)
        gain = np.asarray([float(handles["gain"].value) for handles in mode_handles], dtype=np.float64)
        phase = np.asarray([float(handles["phase"].value) for handles in mode_handles], dtype=np.float64)
        return enabled, gain, phase

    def set_oscillator_state() -> None:
        if runtime is None:
            return
        enabled, gain, phase = mode_control_arrays()
        omega = runtime["omega"]
        q = enabled.astype(np.float64) * gain * np.exp(1j * (omega * float(runtime["time"]) + phase))
        runtime["q"] = q.astype(np.complex64)
        runtime["qdot"] = (1j * omega * q).astype(np.complex64)

    def reset_runtime_state(_) -> None:
        if runtime is None:
            return
        runtime["time"] = 0.0
        runtime["q"].fill(0.0)
        runtime["qdot"].fill(0.0)
        if gui_handles["drive_mode"].value == "oscillator":
            set_oscillator_state()
        redraw_points(float(gui_handles["point_size"].value))

    def trigger_impulse(_) -> None:
        if runtime is None:
            return
        runtime["time"] = 0.0
        enabled, gain, phase = mode_control_arrays()
        omega = runtime["omega"]
        runtime["q"].fill(0.0)
        runtime["qdot"] = (enabled.astype(np.float64) * gain * omega * np.exp(1j * phase)).astype(np.complex64)
        redraw_points(float(gui_handles["point_size"].value))

    def step_runtime(dt: float) -> None:
        if runtime is None:
            return
        runtime["time"] = float(runtime["time"]) + float(dt)
        if gui_handles["drive_mode"].value == "oscillator":
            set_oscillator_state()
            return
        damping = max(float(gui_handles["damping"].value), 0.0)
        omega = runtime["omega"].astype(np.float64)
        n_steps = max(1, int(np.ceil(float(dt) / 0.005)))
        sub_dt = float(dt) / n_steps
        q = runtime["q"].astype(np.complex128)
        qdot = runtime["qdot"].astype(np.complex128)
        for _ in range(n_steps):
            qddot = -(2.0 * damping * omega) * qdot - (omega**2) * q
            qdot = qdot + qddot * sub_dt
            q = q + qdot * sub_dt
        runtime["q"] = q.astype(np.complex64)
        runtime["qdot"] = qdot.astype(np.complex64)

    def update_camera_visibility(_) -> None:
        for handle in camera_handles.values():
            handle.visible = bool(show_cameras.value)

    def update_point_size(_) -> None:
        redraw_points(float(point_size_slider.value))

    def update_motion_scale(_) -> None:
        if runtime_data is None:
            return
        animation["motion_scale"] = float(gui_handles["motion_scale"].value)
        redraw_points(float(gui_handles["point_size"].value))

    def update_color_scheme(_) -> None:
        redraw_points(float(gui_handles["point_size"].value))

    def update_modal_controls(_) -> None:
        if runtime is None:
            return
        if gui_handles["drive_mode"].value == "oscillator":
            set_oscillator_state()
        redraw_points(float(gui_handles["point_size"].value))

    show_cameras.on_update(update_camera_visibility)
    point_size_slider.on_update(update_point_size)
    if runtime_data is not None:
        set_oscillator_state()
        gui_handles["motion_scale"].on_update(update_motion_scale)
        gui_handles["color_scheme"].on_update(update_color_scheme)
        gui_handles["drive_mode"].on_update(reset_runtime_state)
        gui_handles["damping"].on_update(update_modal_controls)
        gui_handles["reset"].on_click(reset_runtime_state)
        gui_handles["impulse"].on_click(trigger_impulse)
        for handles in mode_handles:
            handles["enabled"].on_update(update_modal_controls)
            handles["gain"].on_update(update_modal_controls)
            handles["phase"].on_update(update_modal_controls)

        def animate_points() -> None:
            last_time = time.perf_counter()
            while True:
                try:
                    fps = max(float(gui_handles["fps"].value), 1.0)
                    now = time.perf_counter()
                    dt = min(now - last_time, 0.25)
                    last_time = now
                    if bool(gui_handles["play"].value):
                        step_runtime(dt)
                        redraw_points(float(gui_handles["point_size"].value))
                    time.sleep(1.0 / fps)
                except Exception as exc:
                    print(f"Animation thread failed: {type(exc).__name__}: {exc}", flush=True)
                    raise

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
