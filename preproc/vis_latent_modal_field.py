"""Interactive Viser visualization for the raw VGGT scene.

This script intentionally does not read modal fields, modal observations, or
latent displacement. It is a small diagnostic viewer for checking whether the
VGGT point cloud and camera poses agree before any modal_surface processing is
introduced.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


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
    required = ["image_paths", "processed_hw", "extrinsics", "intrinsics", "depth", "depth_conf"]
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
        "depth": normalize_depth_array(z["depth"], num_images),
        "depth_conf": normalize_depth_array(z["depth_conf"], num_images),
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


def read_resized_image(path: str | bytes | Path, target_hw: tuple[int, int]) -> np.ndarray:
    """Read one RGB image and resize it to VGGT processed resolution."""
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    image = Image.open(Path(str(path)).expanduser()).convert("RGB")
    target_h, target_w = target_hw
    image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def unproject_depth(
    depth: np.ndarray,
    confidence: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject one VGGT depth map into world-space colored points."""
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence >= confidence_threshold)
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
        )

    yy, xx = np.nonzero(valid)
    z = depth[yy, xx].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xx.astype(np.float64) - cx) * z / fx
    y = (yy.astype(np.float64) - cy) * z / fy
    points_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    camera_to_world = np.linalg.inv(world_to_camera)
    points_world_h = points_cam @ camera_to_world.T
    return (
        points_world_h[:, :3].astype(np.float32),
        colors[yy, xx].astype(np.uint8),
        confidence[yy, xx].astype(np.float32),
    )


def load_points_from_vggt_outputs(
    vggt: dict[str, np.ndarray],
    cameras: list[Camera],
    confidence_percentile: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a point cloud from VGGT depth, confidence, intrinsics, and poses."""
    if not (0.0 <= confidence_percentile <= 100.0):
        raise ValueError("--confidence-percentile must be in [0, 100].")
    if max_points < 0:
        raise ValueError("--max-points must be non-negative.")

    depth = vggt["depth"]
    depth_conf = vggt["depth_conf"]
    valid_conf = depth_conf[np.isfinite(depth_conf) & np.isfinite(depth) & (depth > 0)]
    if valid_conf.size == 0:
        raise ValueError("VGGT outputs contain no finite positive depth samples.")
    threshold = float(np.percentile(valid_conf, confidence_percentile))

    processed_hw = (int(vggt["processed_hw"][0]), int(vggt["processed_hw"][1]))
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    confidence_chunks: list[np.ndarray] = []
    for i, camera in enumerate(cameras):
        colors = read_resized_image(vggt["image_paths"][i], processed_hw)
        points_i, colors_i, confidence_i = unproject_depth(
            depth[i],
            depth_conf[i],
            colors,
            camera.K,
            camera.world_to_camera,
            threshold,
        )
        point_chunks.append(points_i)
        color_chunks.append(colors_i)
        confidence_chunks.append(confidence_i)

    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    confidence = np.concatenate(confidence_chunks, axis=0)
    if max_points > 0 and points.shape[0] > max_points:
        # Match the VGGT demo behavior: keep an evenly spaced subset after
        # confidence filtering instead of selecting only the highest scores.
        keep = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
        points = points[keep]
        colors = colors[keep]
        confidence = confidence[keep]
    return points.astype(np.float32), colors.astype(np.uint8), confidence.astype(np.float32)


def load_points_from_export(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a point cloud exported by preproc/run_vggt.py --export-points."""
    z = np.load(str(path), allow_pickle=False)
    required = ["points_world", "colors", "confidence"]
    missing = [key for key in required if key not in z.files]
    if missing:
        raise ValueError(f"{path} missing required point arrays: {missing}.")
    points = z["points_world"].astype(np.float32)
    colors = z["colors"].astype(np.uint8)
    confidence = z["confidence"].astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points.shape}.")
    if colors.shape != (points.shape[0], 3):
        raise ValueError(f"colors must have shape (N,3), got {colors.shape}.")
    if confidence.shape != (points.shape[0],):
        raise ValueError(f"confidence must have shape (N,), got {confidence.shape}.")
    valid = np.all(np.isfinite(points), axis=1) & np.isfinite(confidence)
    points = points[valid]
    colors = colors[valid]
    confidence = confidence[valid]
    if max_points > 0 and points.shape[0] > max_points:
        keep = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
        points = points[keep]
        colors = colors[keep]
        confidence = confidence[keep]
    return points, colors, confidence


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 column-vector transform to row-major points."""
    return (points.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def scene_transform(cameras: list[Camera], alignment: str) -> np.ndarray:
    """Return the display-space transform for points and cameras."""
    if alignment == "none":
        return np.eye(4, dtype=np.float64)
    if alignment == "vggt":
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
    c2w_display = transform @ c2w @ OPENGL_CAMERA_CONVERSION
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
    parser.add_argument("--points", default=None, type=Path, help="Optional vggt_points.npz from --export-points.")
    parser.add_argument("--port", type=int, default=8891, help="Viser server port.")
    parser.add_argument("--point-size", type=float, default=0.004, help="Point cloud point size.")
    parser.add_argument("--max-points", type=int, default=10000, help="Maximum points to display; 0 keeps all.")
    parser.add_argument("--confidence-percentile", type=float, default=30.0, help="Confidence percentile when building points from vggt_outputs.")
    parser.add_argument(
        "--alignment",
        choices=("vggt", "none"),
        default="vggt",
        help="Display transform. 'vggt' matches the VGGT demo scene alignment.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    import viser

    vggt = load_vggt_outputs(args.vggt_outputs.expanduser())
    cameras = build_cameras(vggt)
    if args.points is not None:
        points, colors, confidence = load_points_from_export(args.points.expanduser(), args.max_points)
        point_source = str(args.points)
    else:
        points, colors, confidence = load_points_from_vggt_outputs(vggt, cameras, args.confidence_percentile, args.max_points)
        point_source = "vggt_outputs depth"
    if points.shape[0] == 0:
        raise ValueError("No VGGT points were loaded.")

    transform = scene_transform(cameras, args.alignment)
    display_points = transform_points(points, transform)
    scale = scene_scale(display_points)
    frustum_scale = 0.08 * scale

    server = viser.ViserServer(port=args.port, verbose=False)
    server.scene.add_point_cloud(
        "/vggt/points",
        points=display_points,
        colors=colors.astype(np.float32) / 255.0,
        point_size=float(args.point_size),
    )

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

    def update_camera_visibility(_) -> None:
        for handle in camera_handles.values():
            handle.visible = bool(show_cameras.value)

    show_cameras.on_update(update_camera_visibility)

    print(f"Loaded VGGT points: {points.shape[0]} from {point_source}")
    print(f"Loaded VGGT cameras: {[camera.label for camera in cameras]}")
    print(f"Alignment: {args.alignment}")
    print(f"Display scene scale: {scale:.6g}")
    print(f"Confidence p50/p90: {np.percentile(confidence, [50, 90]).tolist()}")
    for camera in cameras:
        center = np.linalg.inv(camera.world_to_camera)[:3, 3]
        display_center = transform_points(center[None, :], transform)[0]
        print(f"{camera.label} raw center {center.tolist()} display center {display_center.tolist()}")
    print(f"Viser server running on http://localhost:{args.port}")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
