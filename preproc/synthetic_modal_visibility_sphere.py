"""Synthetic sphere test for multi-view modal lifting.

This script builds a noise-free observation graph for a textured sphere with a
known, configurable 3D complex translation mode. It then calls the staged
multi-view solver and writes viewer-compatible latent manifests and diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from modal_surface.optimization_staged import optimize_multi_view_staged
from modal_surface.solver_cli import (
    add_staged_solver_arguments,
    staged_solver_config,
    staged_solver_manifest_parameters,
)


def _world_to_camera_points(points_world: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    ones = np.ones((points_world.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_world, ones], axis=1)
    return (points_h @ world_to_camera.T)[:, :3]


def _project_points(
    points_world: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_cam = _world_to_camera_points(points_world, world_to_camera)
    z = points_cam[:, 2]
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32), z.astype(np.float32)


def _projection_jacobian(
    points_world: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    points_cam = _world_to_camera_points(points_world, world_to_camera)
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


def _fibonacci_sphere(num_points: int, radius: float) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    indices = np.arange(num_points, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / float(num_points)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    theta = golden_angle * indices
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    return (radius * np.stack([x, y, z], axis=1)).astype(np.float32)


def _checker_colors(points: np.ndarray, lat_freq: int = 18, lon_freq: int = 36) -> np.ndarray:
    unit = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-8)
    theta = np.arctan2(unit[:, 1], unit[:, 0])
    phi = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    stripe_a = np.floor((theta + math.pi) / (2.0 * math.pi) * lon_freq).astype(np.int32)
    stripe_b = np.floor(phi / math.pi * lat_freq).astype(np.int32)
    checker = ((stripe_a + stripe_b) % 2).astype(bool)
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    colors[checker] = np.array([235, 235, 235], dtype=np.uint8)
    colors[~checker] = np.array([35, 80, 180], dtype=np.uint8)
    equator = np.abs(unit[:, 2]) < 0.05
    meridian = np.abs(np.sin(4.0 * theta)) < 0.04
    colors[equator] = np.array([240, 80, 40], dtype=np.uint8)
    colors[meridian] = np.array([40, 210, 90], dtype=np.uint8)
    return colors


def _look_at_world_to_camera(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - center
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm <= 1e-8:
        raise ValueError("Camera forward direction is degenerate with world up.")
    right /= right_norm
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    R = np.stack([right, down, forward], axis=0)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = -R @ center
    return T.astype(np.float32)


def _camera_centers_on_arc(num_views: int) -> np.ndarray:
    if num_views < 2:
        raise ValueError("num_views must be at least 2.")
    left = np.array([-2.5, -3.0, 0.0], dtype=np.float64)
    right = np.array([2.5, -3.0, 0.0], dtype=np.float64)
    radius = float(np.linalg.norm(left[:2]))
    half_arc_angle = math.atan2(2.5, 3.0)
    angles = np.linspace(-half_arc_angle, half_arc_angle, num_views, dtype=np.float64)
    centers = np.zeros((num_views, 3), dtype=np.float64)
    centers[:, 0] = radius * np.sin(angles)
    centers[:, 1] = -radius * np.cos(angles)
    centers[0] = left
    centers[-1] = right
    return centers.astype(np.float32)


def _view_alphas(num_views: int, phase_offset_rad: float) -> np.ndarray:
    if num_views < 2:
        raise ValueError("num_views must be at least 2.")
    phases = (
        np.arange(num_views, dtype=np.float64)
        / float(num_views - 1)
        * float(phase_offset_rad)
    )
    return np.exp(1j * phases).astype(np.complex64)


def _camera_intrinsics(width: int, height: int, focal: float) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive.")
    if focal <= 0:
        raise ValueError("focal must be positive.")
    return np.array(
        [
            [focal, 0.0, 0.5 * width],
            [0.0, focal, 0.5 * height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError(f"motion direction must have shape (3,), got {direction.shape}.")
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("motion direction must be finite and non-zero.")
    return (direction / norm).astype(np.float32)


def _build_elliptical_mode(
    major_direction: np.ndarray,
    minor_direction: np.ndarray,
    minor_axis_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    major_axis = _normalize_direction(major_direction)
    minor_candidate = np.asarray(minor_direction, dtype=np.float64)
    if minor_candidate.shape != (3,):
        raise ValueError(
            f"ellipse minor direction must have shape (3,), got {minor_candidate.shape}."
        )
    minor_norm = float(np.linalg.norm(minor_candidate))
    if not np.isfinite(minor_norm) or minor_norm <= 1e-12:
        raise ValueError("ellipse minor direction must be finite and non-zero.")
    minor_candidate /= minor_norm
    minor_orthogonal = minor_candidate - float(np.dot(minor_candidate, major_axis)) * major_axis
    minor_orthogonal_norm = float(np.linalg.norm(minor_orthogonal))
    if not np.isfinite(minor_orthogonal_norm) or minor_orthogonal_norm <= 1e-6:
        raise ValueError("ellipse minor direction must not be parallel to the motion direction.")
    minor_axis = (minor_orthogonal / minor_orthogonal_norm).astype(np.float32)

    minor_axis_ratio = float(minor_axis_ratio)
    if not np.isfinite(minor_axis_ratio) or not (0.0 < minor_axis_ratio <= 1.0):
        raise ValueError("ellipse minor-axis ratio must be finite and in (0, 1].")
    phi_world = (
        major_axis.astype(np.complex64)
        - 1j * np.float32(minor_axis_ratio) * minor_axis.astype(np.complex64)
    ).astype(np.complex64)
    return major_axis, minor_axis, phi_world


def _visibility_by_normal(points: np.ndarray, centers: np.ndarray, margin: float) -> np.ndarray:
    normals = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-8)
    masks = []
    for center in centers:
        direction = center[None, :] - points
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
        masks.append(np.einsum("ij,ij->i", normals, direction) > float(margin))
    return np.stack(masks, axis=1)


def _build_observations(
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    world_to_cameras: np.ndarray,
    camera_centers: np.ndarray,
    image_width: int,
    image_height: int,
    visibility_margin: float,
    phase_offset_rad: float,
    freq_hz: float,
    motion_direction: np.ndarray,
    phi_world: np.ndarray,
    mode_index: int,
) -> dict[str, np.ndarray]:
    motion_direction = _normalize_direction(motion_direction)
    phi_world = np.asarray(phi_world, dtype=np.complex64)
    if phi_world.shape != (3,):
        raise ValueError(f"phi_world must have shape (3,), got {phi_world.shape}.")
    if not np.all(np.isfinite(phi_world.real)) or not np.all(np.isfinite(phi_world.imag)):
        raise ValueError("phi_world must be finite.")
    normal_visibility = _visibility_by_normal(points, camera_centers, visibility_margin)
    point_view_mask = np.zeros((points.shape[0], world_to_cameras.shape[0]), dtype=bool)
    num_views = world_to_cameras.shape[0]
    alphas = _view_alphas(num_views, phase_offset_rad)
    phi_gt = np.broadcast_to(phi_world, points.shape).astype(np.complex64).copy()
    motion_direction_camera = np.einsum(
        "vij,j->vi",
        world_to_cameras[:, :3, :3].astype(np.float64),
        motion_direction.astype(np.float64),
    ).astype(np.float32)

    obs_point_index: list[int] = []
    obs_view_index: list[int] = []
    obs_pixels_xy: list[np.ndarray] = []
    obs_y: list[np.ndarray] = []
    obs_J: list[np.ndarray] = []
    obs_confidence: list[float] = []

    for view_idx, w2c in enumerate(world_to_cameras):
        pixels, z = _project_points(points, K, w2c)
        in_bounds = (
            (z > 0.0)
            & (pixels[:, 0] >= 1.0)
            & (pixels[:, 0] < float(image_width - 2))
            & (pixels[:, 1] >= 1.0)
            & (pixels[:, 1] < float(image_height - 2))
        )
        visible = normal_visibility[:, view_idx] & in_bounds
        indices = np.where(visible)[0]
        if indices.size == 0:
            raise ValueError(f"View {view_idx} has no visible points.")
        point_view_mask[indices, view_idx] = True
        jac = _projection_jacobian(points[indices], K, w2c)
        projected = np.einsum("nij,nj->ni", jac, phi_gt[indices])
        y = (alphas[view_idx] * projected).astype(np.complex64)
        obs_point_index.extend(indices.tolist())
        obs_view_index.extend([view_idx] * int(indices.size))
        obs_pixels_xy.extend(pixels[indices].astype(np.float32))
        obs_y.extend(y)
        obs_J.extend(jac.astype(np.float32))
        obs_confidence.extend([1.0] * int(indices.size))

    obs_point_arr = np.asarray(obs_point_index, dtype=np.int32)
    obs_view_arr = np.asarray(obs_view_index, dtype=np.int32)
    obs_count = np.bincount(obs_point_arr, minlength=points.shape[0]).astype(np.int32)
    data = {
        "points_world": points.astype(np.float32),
        "obs_point_index": obs_point_arr,
        "obs_view_index": obs_view_arr,
        "obs_pixels_xy": np.asarray(obs_pixels_xy, dtype=np.float32),
        "obs_y": np.asarray(obs_y, dtype=np.complex64),
        "obs_J": np.asarray(obs_J, dtype=np.float32),
        "obs_confidence": np.asarray(obs_confidence, dtype=np.float32),
        "obs_count_per_point": obs_count,
        "obs_sample_count_per_point": obs_count.copy(),
        "view_ids": np.array([f"view{view_idx + 1}" for view_idx in range(num_views)]),
        "view_freqs_hz": np.full((num_views,), float(freq_hz), dtype=np.float32),
        "freq_hz": np.array(float(freq_hz), dtype=np.float32),
        "mode_index": np.array(mode_index, dtype=np.int32),
        "colors": colors.astype(np.uint8),
        "phi_gt": phi_gt.astype(np.complex64),
        "motion_direction_world": motion_direction.astype(np.float32),
        "motion_direction_camera": motion_direction_camera,
        "motion_depth_components": motion_direction_camera[:, 2].astype(np.float32),
        "true_alphas": alphas,
        "normal_visibility": normal_visibility.astype(bool),
        "point_view_mask": point_view_mask.astype(bool),
    }
    return data


def _write_npz(path: Path, **arrays: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def _rel(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def _write_manifest(
    path: Path,
    modes: list[dict[str, Any]],
    solver_parameters: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters: dict[str, Any] = {
        "synthetic_test": "visibility_sphere",
        "alpha_model": "per_view_per_mode",
        "alpha_reference_view_index": 0,
    }
    if solver_parameters is not None:
        parameters.update(solver_parameters)
    payload = {
        "version": 1,
        "parameters": parameters,
        "modes": modes,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _tint_colors(colors: np.ndarray, tint: np.ndarray, strength: float = 0.35) -> np.ndarray:
    base = colors.astype(np.float32)
    tinted = (1.0 - float(strength)) * base + float(strength) * tint.astype(np.float32)[None, :]
    return np.clip(tinted, 0.0, 255.0).astype(np.uint8)


def _valid_view_count_colors(valid_view_count: np.ndarray, num_views: int) -> np.ndarray:
    counts = np.asarray(valid_view_count)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("valid_view_count must be a 1-D integer array.")
    if num_views < 2:
        raise ValueError("num_views must be at least 2.")
    counts = counts.astype(np.int32, copy=False)
    if counts.size and (np.any(counts < 0) or np.any(counts > num_views)):
        raise ValueError(f"valid_view_count entries must be in [0, {num_views}].")

    output = np.full((counts.shape[0], 3), 96.0, dtype=np.float32)
    orange = np.array([230.0, 159.0, 0.0], dtype=np.float32)
    blue = np.array([0.0, 114.0, 178.0], dtype=np.float32)
    green = np.array([0.0, 158.0, 115.0], dtype=np.float32)
    purple = np.array([204.0, 121.0, 167.0], dtype=np.float32)
    output[counts == 1] = orange
    output[counts == 2] = blue
    at_least_three = counts >= 3
    if np.any(at_least_three):
        if num_views == 3:
            output[at_least_three] = green
        else:
            blend = (
                (counts[at_least_three].astype(np.float32) - 3.0)
                / float(num_views - 3)
            )[:, None]
            output[at_least_three] = (1.0 - blend) * green + blend * purple
    return np.rint(np.clip(output, 0.0, 255.0)).astype(np.uint8)


def _write_valid_view_count_latent(
    path: Path,
    points: np.ndarray,
    phi: np.ndarray,
    valid_view_count: np.ndarray,
    num_views: int,
    freq_hz: float,
    mode_index: int,
    obs_count_per_point: np.ndarray,
) -> Path:
    points = np.asarray(points, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.complex64)
    valid_view_count = np.asarray(valid_view_count, dtype=np.int32)
    obs_count_per_point = np.asarray(obs_count_per_point, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3 or phi.shape != points.shape:
        raise ValueError(
            f"points and phi must have matching shape (N, 3), got {points.shape} and {phi.shape}."
        )
    if valid_view_count.shape != (points.shape[0],):
        raise ValueError(
            f"valid_view_count must have shape ({points.shape[0]},), got {valid_view_count.shape}."
        )
    if obs_count_per_point.shape != (points.shape[0],):
        raise ValueError(
            f"obs_count_per_point must have shape ({points.shape[0]},), got {obs_count_per_point.shape}."
        )
    group_names = np.array(
        [
            f"{count} valid view" if count == 1 else f"{count} valid views"
            for count in range(num_views + 1)
        ]
    )
    return _write_npz(
        path,
        points_world=points,
        phi=phi,
        colors=_valid_view_count_colors(valid_view_count, num_views),
        freq_hz=np.array(float(freq_hz), dtype=np.float32),
        mode_index=np.array(mode_index, dtype=np.int32),
        obs_count_per_point=obs_count_per_point,
        obs_sample_count_per_point=obs_count_per_point.copy(),
        point_distinct_valid_view_count=valid_view_count,
        point_group=valid_view_count.copy(),
        point_group_names=group_names,
    )


def _write_viewer_inputs(
    out_dir: Path,
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    world_to_cameras: np.ndarray,
    image_width: int,
    image_height: int,
) -> None:
    num_views = world_to_cameras.shape[0]
    _write_npz(out_dir / "toy_points.npz", points_world=points.astype(np.float32), colors=colors.astype(np.uint8))
    np.savez_compressed(
        out_dir / "toy_vggt_outputs.npz",
        image_paths=np.array([f"toy_view{view_idx + 1}.png" for view_idx in range(num_views)]),
        processed_hw=np.array([image_height, image_width], dtype=np.int32),
        extrinsics=world_to_cameras.astype(np.float32),
        intrinsics=np.repeat(K[None, :, :], num_views, axis=0).astype(np.float32),
    )


def _angular_error_deg(phi: np.ndarray, direction: np.ndarray) -> np.ndarray:
    complex_phi = np.asarray(phi, dtype=np.complex128)
    direction = np.asarray(direction, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    component = complex_phi @ direction.astype(np.complex128)
    phase_valid = np.abs(component) > 1e-12
    aligned_phi = np.full(complex_phi.shape, np.nan, dtype=np.float64)
    if np.any(phase_valid):
        phase_alignment = np.exp(-1j * np.angle(component[phase_valid]))
        aligned_phi[phase_valid] = np.real(
            complex_phi[phase_valid] * phase_alignment[:, None]
        )
    phi_norm = np.linalg.norm(aligned_phi, axis=1)
    valid = phase_valid & (phi_norm > 1e-12) & (direction_norm > 1e-12)
    angles = np.full((phi.shape[0],), np.nan, dtype=np.float32)
    cos = np.sum(aligned_phi[valid] * direction[None, :], axis=1) / (
        phi_norm[valid] * direction_norm
    )
    angles[valid] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).astype(np.float32)
    return angles


def _complex_mode_relative_error(phi: np.ndarray, reference_phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.complex128)
    reference_phi = np.asarray(reference_phi, dtype=np.complex128)
    if phi.shape != reference_phi.shape or phi.ndim != 2 or phi.shape[1] != 3:
        raise ValueError(
            "phi and reference_phi must have matching shape (N, 3), got "
            f"{phi.shape} and {reference_phi.shape}."
        )
    reference_norm = np.linalg.norm(reference_phi, axis=1)
    valid = (
        np.all(np.isfinite(phi.real) & np.isfinite(phi.imag), axis=1)
        & np.all(np.isfinite(reference_phi.real) & np.isfinite(reference_phi.imag), axis=1)
        & (reference_norm > 1e-12)
    )
    error = np.full((phi.shape[0],), np.nan, dtype=np.float32)
    if not np.any(valid):
        return error
    inner = np.sum(np.conj(reference_phi[valid]) * phi[valid])
    alignment = np.exp(-1j * np.angle(inner)) if np.abs(inner) > 1e-12 else 1.0 + 0.0j
    aligned_phi = phi[valid] * alignment
    error[valid] = (
        np.linalg.norm(aligned_phi - reference_phi[valid], axis=1) / reference_norm[valid]
    ).astype(np.float32)
    return error


def _mode_error_arrays(phi: np.ndarray, reference_phi: np.ndarray) -> dict[str, np.ndarray]:
    phi = np.asarray(phi, dtype=np.complex128)
    reference_phi = np.asarray(reference_phi, dtype=np.complex128)
    if phi.shape != reference_phi.shape or phi.ndim != 2 or phi.shape[1] != 3:
        raise ValueError(
            "phi and reference_phi must have matching shape (N, 3), got "
            f"{phi.shape} and {reference_phi.shape}."
        )

    recovered_norm = np.linalg.norm(phi, axis=1)
    reference_norm = np.linalg.norm(reference_phi, axis=1)
    finite = (
        np.all(np.isfinite(phi.real) & np.isfinite(phi.imag), axis=1)
        & np.all(np.isfinite(reference_phi.real) & np.isfinite(reference_phi.imag), axis=1)
        & (reference_norm > 1e-12)
    )
    vector_error = np.full((phi.shape[0],), np.nan, dtype=np.float32)
    direction_error = np.full((phi.shape[0],), np.nan, dtype=np.float32)
    amplitude_absolute_error = np.full((phi.shape[0],), np.nan, dtype=np.float32)
    amplitude_relative_error = np.full((phi.shape[0],), np.nan, dtype=np.float32)
    if np.any(finite):
        vector_error[finite] = np.linalg.norm(phi[finite] - reference_phi[finite], axis=1).astype(
            np.float32
        )
        amplitude_absolute_error[finite] = np.abs(
            recovered_norm[finite] - reference_norm[finite]
        ).astype(np.float32)
        amplitude_relative_error[finite] = (
            amplitude_absolute_error[finite] / reference_norm[finite]
        ).astype(np.float32)

    direction_valid = finite & (recovered_norm > 1e-12)
    if np.any(direction_valid):
        inner = np.sum(np.conj(reference_phi[direction_valid]) * phi[direction_valid], axis=1)
        cosine = np.abs(inner) / (
            reference_norm[direction_valid] * recovered_norm[direction_valid]
        )
        direction_error[direction_valid] = np.degrees(
            np.arccos(np.clip(cosine, 0.0, 1.0))
        ).astype(np.float32)

    return {
        "complex_mode_vector_error": vector_error,
        "complex_direction_error_deg": direction_error,
        "amplitude_absolute_error": amplitude_absolute_error,
        "amplitude_relative_error": amplitude_relative_error,
        "trajectory_rmse": (vector_error / np.float32(np.sqrt(2.0))).astype(np.float32),
    }


def _direction_angle_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    cos = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def _phase_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}

    resultant = np.mean(np.exp(1j * finite))
    if abs(resultant) <= 1e-12:
        return {
            "count": int(finite.size),
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }

    center = float(np.angle(resultant))
    unwrapped = center + np.angle(np.exp(1j * (finite - center)))
    stats = _stats(unwrapped)
    stats["mean"] = center
    return stats


def _group_diagnostics(
    name: str,
    mask: np.ndarray,
    phi: np.ndarray,
    direction: np.ndarray,
    point_residual: np.ndarray | None,
    reference_phi: np.ndarray | None = None,
) -> dict[str, Any]:
    if not np.any(mask):
        diagnostics = {
            "name": name,
            "count": 0,
            "angular_error_deg": _stats(np.array([], dtype=np.float32)),
            "phase_rad": _phase_stats(np.array([], dtype=np.float32)),
            "mean_recovered_real_motion": None,
            "mean_recovered_abs_motion": None,
            "point_residual": _stats(np.array([], dtype=np.float32)),
        }
        if reference_phi is not None:
            diagnostics["complex_mode_relative_error"] = _stats(np.array([], dtype=np.float32))
        return diagnostics
    phi_group = phi[mask]
    component = phi_group @ direction.astype(np.complex64)
    phase = np.full((component.shape[0],), np.nan, dtype=np.float32)
    phase_valid = np.abs(component) > 1e-12
    phase[phase_valid] = np.angle(component[phase_valid]).astype(np.float32)
    diagnostics = {
        "name": name,
        "count": int(mask.sum()),
        "angular_error_deg": _stats(_angular_error_deg(phi_group, direction)),
        "phase_rad": _phase_stats(phase),
        "mean_recovered_real_motion": np.mean(np.real(phi_group), axis=0).astype(float).tolist(),
        "mean_recovered_abs_motion": float(np.mean(np.linalg.norm(phi_group, axis=1))),
        "point_residual": _stats(point_residual[mask] if point_residual is not None else np.array([], dtype=np.float32)),
    }
    if reference_phi is not None:
        diagnostics["complex_mode_relative_error"] = _stats(
            _complex_mode_relative_error(phi_group, np.asarray(reference_phi)[mask])
        )
    return diagnostics


def _valid_view_count_group_diagnostics(
    name: str,
    mask: np.ndarray,
    phi: np.ndarray,
    direction: np.ndarray,
    reference_phi: np.ndarray,
    observable_rank: np.ndarray,
    anchor_mask: np.ndarray,
    condition: np.ndarray,
    point_residual: np.ndarray,
    point_residual_valid_mask: np.ndarray,
) -> dict[str, Any]:
    diagnostics = _group_diagnostics(
        name,
        mask,
        phi,
        direction,
        point_residual,
        reference_phi,
    )
    count = int(np.count_nonzero(mask))
    anchor_count = int(np.count_nonzero(mask & anchor_mask))
    error_arrays = _mode_error_arrays(phi[mask], reference_phi[mask])
    diagnostics.update(
        {
            "observable_rank_distribution": {
                f"rank_{rank}": int(np.count_nonzero(mask & (observable_rank == rank)))
                for rank in range(4)
            },
            "anchor_count": anchor_count,
            "anchor_fraction": float(anchor_count / count) if count else None,
            "condition_number": _stats(condition[mask]),
            "observation_residual": _stats(
                point_residual[mask & point_residual_valid_mask]
            ),
        }
    )
    diagnostics.update({key: _stats(values) for key, values in error_arrays.items()})
    return diagnostics


def _write_diagnostics(
    out_path: Path,
    observations: dict[str, np.ndarray],
    solved_latent: dict[str, np.ndarray],
    direction: np.ndarray,
    image_width: int,
    image_height: int,
    visibility_margin: float,
    reference_phi: np.ndarray | None = None,
    trajectory: dict[str, Any] | None = None,
) -> None:
    obs_count = observations["obs_count_per_point"].astype(np.int32)
    visibility = observations["point_view_mask"].astype(bool)
    num_views = int(np.asarray(observations["view_ids"]).reshape(-1).shape[0])
    num_points = int(observations["points_world"].shape[0])
    if visibility.shape != (num_points, num_views):
        raise ValueError(
            f"point_view_mask must have shape ({num_points}, {num_views}), got {visibility.shape}."
        )
    phi = solved_latent["phi"].astype(np.complex64)
    diagnostic_reference_phi = np.asarray(
        observations["phi_gt"] if reference_phi is None else reference_phi,
        dtype=np.complex64,
    )
    if phi.shape != (num_points, 3) or diagnostic_reference_phi.shape != phi.shape:
        raise ValueError(
            "Solved and reference mode fields must have matching shape "
            f"({num_points}, 3), got {phi.shape} and {diagnostic_reference_phi.shape}."
        )

    valid_view_count = np.asarray(
        solved_latent["point_distinct_valid_view_count"], dtype=np.int32
    ).reshape(-1)
    observable_rank = np.asarray(solved_latent["point_observable_rank"], dtype=np.int32).reshape(-1)
    anchor_mask = np.asarray(solved_latent["anchor_mask"], dtype=bool).reshape(-1)
    condition = np.asarray(solved_latent["point_condition"], dtype=np.float32).reshape(-1)
    point_residual = np.asarray(solved_latent["point_residual"], dtype=np.float32).reshape(-1)
    point_residual_valid_mask = np.asarray(
        solved_latent["point_residual_valid_mask"], dtype=bool
    ).reshape(-1)
    point_arrays = {
        "point_distinct_valid_view_count": valid_view_count,
        "point_observable_rank": observable_rank,
        "anchor_mask": anchor_mask,
        "point_condition": condition,
        "point_residual": point_residual,
        "point_residual_valid_mask": point_residual_valid_mask,
    }
    for name, array in point_arrays.items():
        if array.shape != (num_points,):
            raise ValueError(f"{name} must have shape ({num_points},), got {array.shape}.")
    if valid_view_count.size and (
        np.any(valid_view_count < 0) or np.any(valid_view_count > num_views)
    ):
        raise ValueError(f"point_distinct_valid_view_count entries must be in [0, {num_views}].")

    valid_view_count_groups: dict[str, Any] = {}
    valid_view_count_counts: dict[str, int] = {}
    for count in range(num_views + 1):
        name = f"valid_view_count_{count}"
        mask = valid_view_count == count
        valid_view_count_counts[name] = int(np.count_nonzero(mask))
        valid_view_count_groups[name] = _valid_view_count_group_diagnostics(
            name,
            mask,
            phi,
            direction,
            diagnostic_reference_phi,
            observable_rank,
            anchor_mask,
            condition,
            point_residual,
            point_residual_valid_mask,
        )

    pred = solved_latent.get("obs_pred_y")
    residual_stats = None
    if pred is not None:
        obs_y = solved_latent.get("obs_y", observations["obs_y"])
        residual_stats = _stats(np.sqrt(np.sum(np.abs(obs_y - pred) ** 2, axis=1)).astype(np.float32))

    true_alphas = observations["true_alphas"].astype(np.complex64)
    solved_alphas = solved_latent["alpha_by_view"].astype(np.complex64)
    solved_alpha_phase = np.asarray(solved_latent["alpha_phase"], dtype=np.float32).reshape(-1)
    solver_method = str(np.asarray(solved_latent["solver_method"]).item())
    alpha_identifiable = np.asarray(solved_latent["alpha_identifiable_mask"], dtype=bool).reshape(-1)
    alpha_report_mask = alpha_identifiable
    solved_alpha_pairs = [
        [float(np.real(alpha)), float(np.imag(alpha))] if alpha_report_mask[index] else None
        for index, alpha in enumerate(solved_alphas)
    ]
    solved_alpha_phases = [
        float(phase) if alpha_report_mask[index] else None
        for index, phase in enumerate(solved_alpha_phase)
    ]
    alpha_phase_error = np.angle(solved_alphas * np.conj(true_alphas)).astype(np.float32)
    alpha_phase_errors = [
        float(error) if alpha_report_mask[index] else None
        for index, error in enumerate(alpha_phase_error)
    ]
    alpha_exclusion_reason = np.asarray(solved_latent["alpha_exclusion_reason"]).astype(str).reshape(-1).tolist()
    alpha_phase_std_array = solved_latent.get("alpha_phase_std")
    alpha_phase_std = None
    if alpha_phase_std_array is not None:
        alpha_phase_std_values = np.asarray(alpha_phase_std_array, dtype=np.float64).reshape(-1)
        alpha_phase_std = [
            float(value) if np.isfinite(value) else None
            for value in alpha_phase_std_values.tolist()
        ]
    alpha_log_gain_std_array = solved_latent.get("alpha_log_gain_std")
    alpha_log_gain_std = None
    if alpha_log_gain_std_array is not None:
        alpha_log_gain_std_values = np.asarray(alpha_log_gain_std_array, dtype=np.float64).reshape(-1)
        alpha_log_gain_std = [
            float(value) if np.isfinite(value) else None
            for value in alpha_log_gain_std_values.tolist()
        ]
    alpha_identifiability = {
        "available": True,
        "identifiable_mask": alpha_identifiable.tolist(),
        "identifiable_count": int(alpha_identifiable.sum()),
        "exclusion_reason": alpha_exclusion_reason,
        "phase_std_rad": alpha_phase_std,
        "log_gain_std": alpha_log_gain_std,
        "gain_bound_active_mask": (
            np.asarray(solved_latent["alpha_gain_bound_active_mask"], dtype=bool).tolist()
            if "alpha_gain_bound_active_mask" in solved_latent
            else None
        ),
        "optimizer_success": (
            bool(np.asarray(solved_latent["alpha_optimizer_success"]).item())
            if "alpha_optimizer_success" in solved_latent
            else None
        ),
        "optimizer_status": (
            int(np.asarray(solved_latent["alpha_optimizer_status"]).item())
            if "alpha_optimizer_status" in solved_latent
            else None
        ),
        "information_kind": (
            str(np.asarray(solved_latent["alpha_information_kind"]).item())
            if "alpha_information_kind" in solved_latent
            else None
        ),
    }

    point_mask_keys = {
        "anchor": "anchor_mask",
        "partial_unresolved": "partial_mask",
        "rejected": "rejected_mask",
        "unobserved_unresolved": "unobserved_mask",
        "alpha_unresolved": "alpha_unresolved_mask",
        "no_usable_observation": "no_usable_observation_mask",
        "completed": "completion_mask",
    }
    point_class_counts: dict[str, int | None] = {}
    point_classification_available = False
    for label, key in point_mask_keys.items():
        mask_array = solved_latent.get(key)
        if mask_array is None:
            point_class_counts[label] = None
            continue
        point_classification_available = True
        point_class_counts[label] = int(np.asarray(mask_array, dtype=bool).sum())

    status_counts = None
    point_status_array = solved_latent.get("point_solution_status")
    point_status_names_array = solved_latent.get("point_solution_status_names")
    if point_status_array is not None and point_status_names_array is not None:
        point_status = np.asarray(point_status_array, dtype=np.int64).reshape(-1)
        point_status_names = np.asarray(point_status_names_array).astype(str).reshape(-1)
        status_counts = {
            str(name): int(np.count_nonzero(point_status == status_index))
            for status_index, name in enumerate(point_status_names.tolist())
        }
    point_classification = {
        "available": point_classification_available,
        "counts": point_class_counts,
        "status_counts": status_counts,
    }

    payload = {
        "solver": {
            "method": solver_method,
        },
        "num_points": num_points,
        "num_views": num_views,
        "num_observations": int(observations["obs_y"].shape[0]),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "visibility_margin": float(visibility_margin),
        "motion_direction_world": direction.astype(float).tolist(),
        "motion_direction_camera": observations["motion_direction_camera"].astype(float).tolist(),
        "motion_depth_components": observations["motion_depth_components"].astype(float).tolist(),
        "motion_angle_to_camera_normal_deg": np.degrees(
            np.arccos(np.clip(np.abs(observations["motion_depth_components"]), 0.0, 1.0))
        ).astype(float).tolist(),
        "valid_view_count_counts": valid_view_count_counts,
        "valid_view_count_groups": valid_view_count_groups,
        "true_alphas": [[float(np.real(a)), float(np.imag(a))] for a in true_alphas],
        "solved_alphas": solved_alpha_pairs,
        "true_alpha_phases_rad": np.angle(true_alphas).astype(float).tolist(),
        "solved_alpha_phases_rad": solved_alpha_phases,
        "alpha_phase_error_rad": alpha_phase_errors,
        "alpha_identifiability": alpha_identifiability,
        "point_classification": point_classification,
        "residual_stats": residual_stats,
    }
    if num_views == 2:
        view1_only = visibility[:, 0] & ~visibility[:, 1]
        view2_only = visibility[:, 1] & ~visibility[:, 0]
        overlap = visibility[:, 0] & visibility[:, 1]
        observed_all = obs_count > 0
        unobserved = obs_count == 0
        groups = {
            "view1_only": _group_diagnostics(
                "view1_only",
                view1_only,
                phi,
                direction,
                point_residual,
                reference_phi,
            ),
            "view2_only": _group_diagnostics(
                "view2_only",
                view2_only,
                phi,
                direction,
                point_residual,
                reference_phi,
            ),
            "overlap": _group_diagnostics(
                "overlap",
                overlap,
                phi,
                direction,
                point_residual,
                reference_phi,
            ),
            "observed_all": _group_diagnostics(
                "observed_all",
                observed_all,
                phi,
                direction,
                point_residual,
                reference_phi,
            ),
            "unobserved": _group_diagnostics(
                "unobserved",
                unobserved,
                phi,
                direction,
                None,
                reference_phi,
            ),
        }
        v1_mean = groups["view1_only"]["mean_recovered_real_motion"]
        v2_mean = groups["view2_only"]["mean_recovered_real_motion"]
        mean_angle = None
        if v1_mean is not None and v2_mean is not None:
            mean_angle = _direction_angle_deg(
                np.asarray(v1_mean, dtype=np.float64),
                np.asarray(v2_mean, dtype=np.float64),
            )
        payload["visibility_counts"] = {
            "view1_only": int(view1_only.sum()),
            "view2_only": int(view2_only.sum()),
            "overlap": int(overlap.sum()),
            "observed_all": int(observed_all.sum()),
            "unobserved": int(unobserved.sum()),
        }
        payload["groups"] = groups
        payload["view1_only_vs_view2_only_mean_direction_angle_deg"] = mean_angle
    if trajectory is not None:
        payload["trajectory"] = trajectory
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic sphere modal lifting test.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_modal/toy_sphere_solver_staged"))
    parser.add_argument("--num-points", type=int, default=20000)
    parser.add_argument(
        "--num-views",
        type=int,
        default=2,
        help="Number of equally spaced camera views on the fixed endpoint arc (at least 2).",
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--visibility-margin", type=float, default=0.03)
    parser.add_argument("--phase-offset-rad", type=float, default=0.8)
    parser.add_argument("--freq-hz", type=float, default=1.0)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--focal", type=float, default=900.0)
    parser.add_argument(
        "--motion-direction",
        type=float,
        nargs=3,
        default=(0.0, 1.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help=(
            "World-space translation direction. The default +Y direction has a strong "
            "inward/outward component relative to both camera planes."
        ),
    )
    parser.add_argument(
        "--ellipse-minor-direction",
        type=float,
        nargs=3,
        default=(1.0, 0.0, 1.0),
        metavar=("MX", "MY", "MZ"),
        help=(
            "World-space direction used to construct the ellipse minor axis. Its component "
            "along --motion-direction is removed."
        ),
    )
    parser.add_argument(
        "--ellipse-minor-axis-ratio",
        type=float,
        default=0.35,
        help="Ellipse minor-axis amplitude divided by its major-axis amplitude, in (0, 1].",
    )
    add_staged_solver_arguments(parser)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    solver_label = "Solved staged"
    solver_parameters = staged_solver_manifest_parameters(args)

    points = _fibonacci_sphere(args.num_points, args.radius)
    colors = _checker_colors(points)
    K = _camera_intrinsics(args.image_width, args.image_height, args.focal)
    motion_direction, ellipse_minor_direction, ellipse_phi_world = _build_elliptical_mode(
        np.asarray(args.motion_direction, dtype=np.float64),
        np.asarray(args.ellipse_minor_direction, dtype=np.float64),
        args.ellipse_minor_axis_ratio,
    )
    camera_centers = _camera_centers_on_arc(args.num_views)
    world_to_cameras = np.stack(
        [_look_at_world_to_camera(center, np.zeros(3, dtype=np.float64)) for center in camera_centers],
        axis=0,
    )

    observations = _build_observations(
        points=points,
        colors=colors,
        K=K,
        world_to_cameras=world_to_cameras,
        camera_centers=camera_centers,
        image_width=args.image_width,
        image_height=args.image_height,
        visibility_margin=args.visibility_margin,
        phase_offset_rad=args.phase_offset_rad,
        freq_hz=args.freq_hz,
        motion_direction=motion_direction,
        phi_world=motion_direction.astype(np.complex64),
        mode_index=0,
    )
    obs_path = _write_npz(out_dir / "observations" / "toy_sphere_observations.npz", **observations)
    ellipse_observations = _build_observations(
        points=points,
        colors=colors,
        K=K,
        world_to_cameras=world_to_cameras,
        camera_centers=camera_centers,
        image_width=args.image_width,
        image_height=args.image_height,
        visibility_margin=args.visibility_margin,
        phase_offset_rad=args.phase_offset_rad,
        freq_hz=args.freq_hz,
        motion_direction=motion_direction,
        phi_world=ellipse_phi_world,
        mode_index=1,
    )
    ellipse_obs_path = _write_npz(
        out_dir / "observations" / "toy_sphere_tilted_ellipse_observations.npz",
        **ellipse_observations,
    )
    _write_viewer_inputs(out_dir, points, colors, K, world_to_cameras, args.image_width, args.image_height)

    phi_gt = observations["phi_gt"].astype(np.complex64)
    gt_path = _write_npz(
        out_dir / "latents" / "gt_motion.npz",
        points_world=points.astype(np.float32),
        phi=phi_gt,
        colors=colors.astype(np.uint8),
        freq_hz=np.array(float(args.freq_hz), dtype=np.float32),
        mode_index=np.array(0, dtype=np.int32),
        obs_count_per_point=observations["obs_count_per_point"].astype(np.int32),
        obs_sample_count_per_point=observations["obs_sample_count_per_point"].astype(np.int32),
    )
    ellipse_phi_gt = ellipse_observations["phi_gt"].astype(np.complex64)
    ellipse_gt_path = _write_npz(
        out_dir / "latents" / "gt_tilted_ellipse_motion.npz",
        points_world=points.astype(np.float32),
        phi=ellipse_phi_gt,
        colors=colors.astype(np.uint8),
        freq_hz=np.array(float(args.freq_hz), dtype=np.float32),
        mode_index=np.array(1, dtype=np.int32),
        obs_count_per_point=ellipse_observations["obs_count_per_point"].astype(np.int32),
        obs_sample_count_per_point=ellipse_observations["obs_sample_count_per_point"].astype(np.int32),
    )

    solved_path = out_dir / "latents" / "solved_staged.npz"
    optimize_multi_view_staged(
        observations_path=obs_path,
        out_path=solved_path,
        config=staged_solver_config(args),
    )
    solved = _load_npz_dict(solved_path)
    ellipse_solved_path = out_dir / "latents" / "solved_staged_tilted_ellipse.npz"
    optimize_multi_view_staged(
        observations_path=ellipse_obs_path,
        out_path=ellipse_solved_path,
        config=staged_solver_config(args),
    )
    ellipse_solved = _load_npz_dict(ellipse_solved_path)
    num_views = int(np.asarray(observations["view_ids"]).reshape(-1).shape[0])
    valid_view_count = np.asarray(
        solved["point_distinct_valid_view_count"], dtype=np.int32
    )
    ellipse_valid_view_count = np.asarray(
        ellipse_solved["point_distinct_valid_view_count"], dtype=np.int32
    )
    if valid_view_count.shape != (points.shape[0],):
        raise ValueError(
            "Linear point_distinct_valid_view_count must have shape "
            f"({points.shape[0]},), got {valid_view_count.shape}."
        )
    if ellipse_valid_view_count.shape != valid_view_count.shape:
        raise ValueError(
            "Tilted-ellipse point_distinct_valid_view_count must have shape "
            f"{valid_view_count.shape}, got {ellipse_valid_view_count.shape}."
        )
    if not np.array_equal(ellipse_valid_view_count, valid_view_count):
        raise ValueError(
            "Linear and tilted-ellipse solves must have identical valid-view counts "
            "for one shared Viser track manifest."
        )
    valid_view_count_path = _write_valid_view_count_latent(
        out_dir / "latents" / "solved_staged_valid_view_count.npz",
        points,
        solved["phi"],
        valid_view_count,
        num_views,
        args.freq_hz,
        0,
        observations["obs_count_per_point"],
    )
    ellipse_valid_view_count_path = _write_valid_view_count_latent(
        out_dir / "latents" / "solved_staged_tilted_ellipse_valid_view_count.npz",
        points,
        ellipse_solved["phi"],
        ellipse_valid_view_count,
        num_views,
        args.freq_hz,
        1,
        ellipse_observations["obs_count_per_point"],
    )
    overlay_points = np.concatenate([points, points], axis=0).astype(np.float32)
    overlay_phi = np.concatenate([phi_gt, solved["phi"].astype(np.complex64)], axis=0).astype(np.complex64)
    overlay_colors = np.concatenate(
        [
            _tint_colors(colors, np.array([40, 220, 120], dtype=np.uint8)),
            _tint_colors(colors, np.array([240, 80, 220], dtype=np.uint8)),
        ],
        axis=0,
    )
    overlay_obs_count = np.concatenate(
        [
            observations["obs_count_per_point"].astype(np.int32),
            observations["obs_count_per_point"].astype(np.int32),
        ],
        axis=0,
    )
    overlay_path = _write_npz(
        out_dir / "latents" / "gt_vs_solved_staged_overlay.npz",
        points_world=overlay_points,
        phi=overlay_phi,
        colors=overlay_colors,
        freq_hz=np.array(float(args.freq_hz), dtype=np.float32),
        mode_index=np.array(0, dtype=np.int32),
        obs_count_per_point=overlay_obs_count,
        obs_sample_count_per_point=overlay_obs_count.copy(),
        point_group=np.concatenate(
            [
                np.zeros(points.shape[0], dtype=np.int32),
                np.ones(points.shape[0], dtype=np.int32),
            ],
            axis=0,
        ),
        point_group_names=np.array(["GT", solver_label]),
    )
    ellipse_overlay_phi = np.concatenate(
        [ellipse_phi_gt, ellipse_solved["phi"].astype(np.complex64)], axis=0
    ).astype(np.complex64)
    ellipse_overlay_obs_count = np.concatenate(
        [
            ellipse_observations["obs_count_per_point"].astype(np.int32),
            ellipse_observations["obs_count_per_point"].astype(np.int32),
        ],
        axis=0,
    )
    ellipse_overlay_path = _write_npz(
        out_dir / "latents" / "gt_vs_solved_staged_tilted_ellipse_overlay.npz",
        points_world=overlay_points,
        phi=ellipse_overlay_phi,
        colors=overlay_colors,
        freq_hz=np.array(float(args.freq_hz), dtype=np.float32),
        mode_index=np.array(1, dtype=np.int32),
        obs_count_per_point=ellipse_overlay_obs_count,
        obs_sample_count_per_point=ellipse_overlay_obs_count.copy(),
        point_group=np.concatenate(
            [
                np.zeros(points.shape[0], dtype=np.int32),
                np.ones(points.shape[0], dtype=np.int32),
            ],
            axis=0,
        ),
        point_group_names=np.array(["GT", solver_label]),
    )
    _write_diagnostics(
        out_dir / "diagnostics.json",
        observations,
        solved,
        motion_direction,
        args.image_width,
        args.image_height,
        args.visibility_margin,
    )
    ellipse_diagnostics_path = out_dir / "diagnostics_tilted_ellipse.json"
    _write_diagnostics(
        ellipse_diagnostics_path,
        ellipse_observations,
        ellipse_solved,
        motion_direction,
        args.image_width,
        args.image_height,
        args.visibility_margin,
        reference_phi=ellipse_phi_gt,
        trajectory={
            "kind": "tilted_ellipse",
            "major_direction_world": motion_direction.astype(float).tolist(),
            "minor_direction_world": ellipse_minor_direction.astype(float).tolist(),
            "minor_axis_ratio": float(args.ellipse_minor_axis_ratio),
            "complex_mode_convention": "phi = major - i * minor_axis_ratio * minor",
            "complex_error_alignment": "one_shared_phase_per_diagnostic_group",
            "complex_error_alignment_scope": "complex_mode_relative_error",
            "complex_mode_vector_error_alignment": "none",
            "trajectory_rmse_alignment": "none",
            "angular_error_scope": "major_axis_only",
        },
    )

    gt_manifest = out_dir / "manifests" / "gt" / "modal_modes_manifest.json"
    solved_manifest = out_dir / "manifests" / "solved" / "modal_modes_manifest.json"
    compare_manifest = out_dir / "manifests" / "compare" / "modal_modes_manifest.json"
    overlay_manifest = out_dir / "manifests" / "overlay" / "modal_modes_manifest.json"
    valid_view_count_manifest = (
        out_dir / "manifests" / "valid_view_count" / "modal_modes_manifest.json"
    )
    _write_manifest(
        gt_manifest,
        [
            {
                "mode_index": 0,
                "freq_hz": float(args.freq_hz),
                "label": "GT configured motion",
                "latent_path": _rel(gt_path, gt_manifest.parent),
            }
        ],
    )
    _write_manifest(
        solved_manifest,
        [
            {
                "mode_index": 0,
                "freq_hz": float(args.freq_hz),
                "label": solver_label,
                "latent_path": _rel(solved_path, solved_manifest.parent),
            }
        ],
        solver_parameters=solver_parameters,
    )
    _write_manifest(
        compare_manifest,
        [
            {
                "mode_index": 0,
                "freq_hz": float(args.freq_hz),
                "label": "GT configured motion",
                "latent_path": _rel(gt_path, compare_manifest.parent),
            },
            {
                "mode_index": 1,
                "freq_hz": float(args.freq_hz),
                "label": solver_label,
                "latent_path": _rel(solved_path, compare_manifest.parent),
            },
        ],
        solver_parameters=solver_parameters,
    )
    _write_manifest(
        overlay_manifest,
        [
            {
                "mode_index": 0,
                "freq_hz": float(args.freq_hz),
                "label": f"GT vs {solver_label} overlay",
                "latent_path": _rel(overlay_path, overlay_manifest.parent),
                "track_label": "Linear",
            },
            {
                "mode_index": 1,
                "freq_hz": float(args.freq_hz),
                "label": f"GT vs {solver_label} tilted ellipse overlay",
                "latent_path": _rel(ellipse_overlay_path, overlay_manifest.parent),
                "track_label": "Tilted ellipse",
            },
        ],
        solver_parameters=solver_parameters,
    )
    _write_manifest(
        valid_view_count_manifest,
        [
            {
                "mode_index": 0,
                "freq_hz": float(args.freq_hz),
                "label": f"{solver_label} colored by valid-view count",
                "latent_path": _rel(valid_view_count_path, valid_view_count_manifest.parent),
                "track_label": "Linear",
            },
            {
                "mode_index": 1,
                "freq_hz": float(args.freq_hz),
                "label": f"{solver_label} tilted ellipse colored by valid-view count",
                "latent_path": _rel(
                    ellipse_valid_view_count_path, valid_view_count_manifest.parent
                ),
                "track_label": "Tilted ellipse",
            },
        ],
        solver_parameters=solver_parameters,
    )

    visibility = observations["point_view_mask"].astype(bool)
    view_count_per_point = visibility.sum(axis=1)
    visibility_summary = ", ".join(
        f"{view_count}_views={int((view_count_per_point == view_count).sum())}"
        for view_count in range(args.num_views + 1)
    )
    print(f"Wrote synthetic sphere test to {out_dir}")
    print(f"Solver: staged ({solver_label})")
    print(f"Visibility counts: {visibility_summary}")
    print(f"Observation graph: {obs_path}")
    print(f"Tilted ellipse observation graph: {ellipse_obs_path}")
    print(f"Tilted ellipse GT latent: {ellipse_gt_path}")
    print(f"Tilted ellipse solved latent: {ellipse_solved_path}")
    print(f"GT manifest: {gt_manifest}")
    print(f"Solved manifest: {solved_manifest}")
    print(f"Overlay manifest: {overlay_manifest}")
    print(f"Valid-view-count manifest: {valid_view_count_manifest}")
    print(f"Diagnostics: {out_dir / 'diagnostics.json'}")
    print(f"Tilted ellipse diagnostics: {ellipse_diagnostics_path}")


if __name__ == "__main__":
    main()
