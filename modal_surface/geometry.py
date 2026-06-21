from __future__ import annotations

import cv2
import numpy as np


def erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool, copy=False)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=int(iterations))
    return eroded > 0


def depth_edge_keep_mask(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    edge_tau: float,
    kernel_size: int = 5,
) -> np.ndarray:
    if edge_tau <= 0:
        return valid_mask.astype(bool, copy=False)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    valid_depth = np.where(valid_mask, depth, np.nan).astype(np.float32)
    # Use OpenCV min/max filters on finite-filled values for a fast conservative edge cue.
    finite = np.isfinite(valid_depth)
    if not np.any(finite):
        return np.zeros_like(valid_mask, dtype=bool)
    large = float(np.nanmax(valid_depth[finite]) + 1.0)
    small = float(np.nanmin(valid_depth[finite]) - 1.0)
    depth_for_min = np.where(finite, valid_depth, large).astype(np.float32)
    depth_for_max = np.where(finite, valid_depth, small).astype(np.float32)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    local_min = cv2.erode(depth_for_min, kernel)
    local_max = cv2.dilate(depth_for_max, kernel)
    denom = np.maximum(np.abs(depth), 1e-6)
    local_range = (local_max - local_min) / denom
    return valid_mask & np.isfinite(local_range) & (local_range <= float(edge_tau))


def unproject_pixels(
    pixels_xy: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    pixels_xy = np.asarray(pixels_xy, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (pixels_xy[:, 0] - cx) * depth / fx
    y = (pixels_xy[:, 1] - cy) * depth / fy
    points_cam = np.stack([x, y, depth, np.ones_like(depth)], axis=1)
    camera_to_world = np.linalg.inv(world_to_camera)
    points_world_h = points_cam @ camera_to_world.T
    return points_world_h[:, :3].astype(np.float32)


def world_to_camera_points(points_world: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    ones = np.ones((points_world.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_world, ones], axis=1)
    return (points_h @ world_to_camera.T)[:, :3]


def project_points(
    points_world: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_cam = world_to_camera_points(points_world, world_to_camera)
    z = points_cam[:, 2]
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32), z.astype(np.float32)


def projection_jacobian(
    points_world: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    points_cam = world_to_camera_points(points_world, world_to_camera)
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    if np.any(z <= 0):
        raise ValueError("Projection Jacobian requires positive camera z for all points.")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    R = world_to_camera[:3, :3]
    J_cam = np.zeros((points_world.shape[0], 2, 3), dtype=np.float64)
    J_cam[:, 0, 0] = fx / z
    J_cam[:, 0, 2] = -fx * x / (z * z)
    J_cam[:, 1, 1] = fy / z
    J_cam[:, 1, 2] = -fy * y / (z * z)
    return np.einsum("nij,jk->nik", J_cam, R).astype(np.float32)


def bilinear_sample(image: np.ndarray, pixels_xy: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    pixels_xy = np.asarray(pixels_xy, dtype=np.float64)
    h, w = image.shape[:2]
    x = pixels_xy[:, 0]
    y = pixels_xy[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    if np.any(x0 < 0) or np.any(y0 < 0) or np.any(x1 >= w) or np.any(y1 >= h):
        raise ValueError("bilinear_sample pixels must be inside image with one-pixel margin.")
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    v00 = image[y0, x0]
    v01 = image[y0, x1]
    v10 = image[y1, x0]
    v11 = image[y1, x1]
    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v01
        + (1.0 - wx) * wy * v10
        + wx * wy * v11
    )


def in_image_with_margin(pixels_xy: np.ndarray, width: int, height: int, margin: int = 1) -> np.ndarray:
    x = pixels_xy[:, 0]
    y = pixels_xy[:, 1]
    return (x >= margin) & (x < width - 1 - margin) & (y >= margin) & (y < height - 1 - margin)

