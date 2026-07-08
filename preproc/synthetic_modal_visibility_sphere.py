"""Synthetic sphere test for multi-view modal lifting.

This script builds a noise-free observation graph for a textured sphere with a
known horizontal 3D complex mode. It then calls the existing multi-view ALS
solver and writes viewer-compatible latent manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from modal_surface.optimization_multi import optimize_multi_view


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
) -> dict[str, np.ndarray]:
    normal_visibility = _visibility_by_normal(points, camera_centers, visibility_margin)
    point_view_mask = np.zeros((points.shape[0], world_to_cameras.shape[0]), dtype=bool)
    alphas = np.array([1.0 + 0.0j, np.exp(1j * phase_offset_rad)], dtype=np.complex64)
    phi_gt = np.zeros_like(points, dtype=np.complex64)
    phi_gt[:, 0] = np.complex64(1.0 + 0.0j)

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
        "view_ids": np.array(["view1", "view2"]),
        "view_freqs_hz": np.full((2,), float(freq_hz), dtype=np.float32),
        "freq_hz": np.array(float(freq_hz), dtype=np.float32),
        "mode_index": np.array(0, dtype=np.int32),
        "colors": colors.astype(np.uint8),
        "phi_gt": phi_gt.astype(np.complex64),
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


def _write_manifest(path: Path, modes: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "parameters": {
            "synthetic_test": "visibility_sphere",
            "alpha_model": "per_view_per_mode",
            "alpha_reference_view_index": 0,
        },
        "modes": modes,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_viewer_inputs(
    out_dir: Path,
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    world_to_cameras: np.ndarray,
    image_width: int,
    image_height: int,
) -> None:
    _write_npz(out_dir / "toy_points.npz", points_world=points.astype(np.float32), colors=colors.astype(np.uint8))
    np.savez_compressed(
        out_dir / "toy_vggt_outputs.npz",
        image_paths=np.array(["toy_view1.png", "toy_view2.png"]),
        processed_hw=np.array([image_height, image_width], dtype=np.int32),
        extrinsics=world_to_cameras.astype(np.float32),
        intrinsics=np.stack([K, K], axis=0).astype(np.float32),
    )


def _angular_error_deg(phi: np.ndarray, direction: np.ndarray) -> np.ndarray:
    real_phi = np.real(phi).astype(np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    denom = np.linalg.norm(real_phi, axis=1) * max(float(np.linalg.norm(direction)), 1e-12)
    cos = np.sum(real_phi * direction[None, :], axis=1) / np.maximum(denom, 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).astype(np.float32)


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


def _group_diagnostics(
    name: str,
    mask: np.ndarray,
    phi: np.ndarray,
    direction: np.ndarray,
    point_residual: np.ndarray | None,
) -> dict[str, Any]:
    if not np.any(mask):
        return {
            "name": name,
            "count": 0,
            "angular_error_deg": _stats(np.array([], dtype=np.float32)),
            "phase_rad": _stats(np.array([], dtype=np.float32)),
            "mean_recovered_real_motion": None,
            "mean_recovered_abs_motion": None,
            "point_residual": _stats(np.array([], dtype=np.float32)),
        }
    phi_group = phi[mask]
    component = phi_group @ direction.astype(np.complex64)
    return {
        "name": name,
        "count": int(mask.sum()),
        "angular_error_deg": _stats(_angular_error_deg(phi_group, direction)),
        "phase_rad": _stats(np.angle(component).astype(np.float32)),
        "mean_recovered_real_motion": np.mean(np.real(phi_group), axis=0).astype(float).tolist(),
        "mean_recovered_abs_motion": float(np.mean(np.linalg.norm(phi_group, axis=1))),
        "point_residual": _stats(point_residual[mask] if point_residual is not None else np.array([], dtype=np.float32)),
    }


def _write_diagnostics(
    out_path: Path,
    observations: dict[str, np.ndarray],
    solved_latent: dict[str, np.ndarray],
    direction: np.ndarray,
    image_width: int,
    image_height: int,
    visibility_margin: float,
) -> None:
    obs_count = observations["obs_count_per_point"].astype(np.int32)
    visibility = observations["point_view_mask"].astype(bool)
    view1_only = visibility[:, 0] & ~visibility[:, 1]
    view2_only = visibility[:, 1] & ~visibility[:, 0]
    overlap = visibility[:, 0] & visibility[:, 1]
    observed_all = obs_count > 0
    unobserved = obs_count == 0
    phi = solved_latent["phi"].astype(np.complex64)
    point_residual = solved_latent.get("point_residual")
    groups = {
        "view1_only": _group_diagnostics("view1_only", view1_only, phi, direction, point_residual),
        "view2_only": _group_diagnostics("view2_only", view2_only, phi, direction, point_residual),
        "overlap": _group_diagnostics("overlap", overlap, phi, direction, point_residual),
        "observed_all": _group_diagnostics("observed_all", observed_all, phi, direction, point_residual),
        "unobserved": _group_diagnostics("unobserved", unobserved, phi, direction, point_residual),
    }
    v1_mean = groups["view1_only"]["mean_recovered_real_motion"]
    v2_mean = groups["view2_only"]["mean_recovered_real_motion"]
    mean_angle = None
    if v1_mean is not None and v2_mean is not None:
        mean_angle = _direction_angle_deg(np.asarray(v1_mean, dtype=np.float64), np.asarray(v2_mean, dtype=np.float64))

    pred = solved_latent.get("pred_y")
    residual_stats = None
    if pred is not None:
        obs_y = observations["obs_y"]
        residual_stats = _stats(np.sqrt(np.sum(np.abs(obs_y - pred) ** 2, axis=1)).astype(np.float32))

    payload = {
        "num_points": int(observations["points_world"].shape[0]),
        "num_observations": int(observations["obs_y"].shape[0]),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "visibility_margin": float(visibility_margin),
        "visibility_counts": {
            "view1_only": int(view1_only.sum()),
            "view2_only": int(view2_only.sum()),
            "overlap": int(overlap.sum()),
            "observed_all": int(observed_all.sum()),
            "unobserved": int(unobserved.sum()),
        },
        "true_alphas": [[float(np.real(a)), float(np.imag(a))] for a in observations["true_alphas"]],
        "solved_alphas": [[float(np.real(a)), float(np.imag(a))] for a in solved_latent["alphas"]],
        "residual_stats": residual_stats,
        "groups": groups,
        "view1_only_vs_view2_only_mean_direction_angle_deg": mean_angle,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic sphere modal lifting test.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_modal/toy_sphere_solver"))
    parser.add_argument("--num-points", type=int, default=20000)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--visibility-margin", type=float, default=0.03)
    parser.add_argument("--phase-offset-rad", type=float, default=0.8)
    parser.add_argument("--freq-hz", type=float, default=1.0)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--focal", type=float, default=900.0)
    parser.add_argument("--ridge-mu", type=float, default=1e-4)
    parser.add_argument("--iterations", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    points = _fibonacci_sphere(args.num_points, args.radius)
    colors = _checker_colors(points)
    K = _camera_intrinsics(args.image_width, args.image_height, args.focal)
    camera_centers = np.array([[-2.5, -3.0, 0.0], [2.5, -3.0, 0.0]], dtype=np.float32)
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
    )
    obs_path = _write_npz(out_dir / "observations" / "toy_sphere_observations.npz", **observations)
    _write_viewer_inputs(out_dir, points, colors, K, world_to_cameras, args.image_width, args.image_height)

    phi_gt = observations["phi_gt"].astype(np.complex64)
    gt_path = _write_npz(
        out_dir / "latents" / "gt_horizontal.npz",
        points_world=points.astype(np.float32),
        phi=phi_gt,
        colors=colors.astype(np.uint8),
        freq_hz=np.array(float(args.freq_hz), dtype=np.float32),
        mode_index=np.array(0, dtype=np.int32),
        obs_count_per_point=observations["obs_count_per_point"].astype(np.int32),
        obs_sample_count_per_point=observations["obs_sample_count_per_point"].astype(np.int32),
    )

    solved_path = out_dir / "latents" / "solved_als.npz"
    optimize_multi_view(
        observations_path=obs_path,
        out_path=solved_path,
        iterations=args.iterations,
        ridge_mu=args.ridge_mu,
        outlier_frac=0.0,
        graph_smooth_lambda=0.0,
        modal_rigid_lambda=0.0,
        modal_fill_unobserved=False,
    )
    solved = _load_npz_dict(solved_path)
    _write_diagnostics(
        out_dir / "diagnostics.json",
        observations,
        solved,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        args.image_width,
        args.image_height,
        args.visibility_margin,
    )

    gt_manifest = out_dir / "manifests" / "gt" / "modal_modes_manifest.json"
    solved_manifest = out_dir / "manifests" / "solved" / "modal_modes_manifest.json"
    compare_manifest = out_dir / "manifests" / "compare" / "modal_modes_manifest.json"
    _write_manifest(
        gt_manifest,
        [
            {
                "mode_index": 0,
                "freq_hz": float(args.freq_hz),
                "label": "GT horizontal",
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
                "label": "Solved ALS",
                "latent_path": _rel(solved_path, solved_manifest.parent),
            }
        ],
    )
    _write_manifest(
        compare_manifest,
        [
            {
                "mode_index": 0,
                "freq_hz": float(args.freq_hz),
                "label": "GT horizontal",
                "latent_path": _rel(gt_path, compare_manifest.parent),
            },
            {
                "mode_index": 1,
                "freq_hz": float(args.freq_hz),
                "label": "Solved ALS",
                "latent_path": _rel(solved_path, compare_manifest.parent),
            },
        ],
    )

    visibility = observations["point_view_mask"].astype(bool)
    view1_only = int((visibility[:, 0] & ~visibility[:, 1]).sum())
    view2_only = int((visibility[:, 1] & ~visibility[:, 0]).sum())
    overlap = int((visibility[:, 0] & visibility[:, 1]).sum())
    unobserved = int((observations["obs_count_per_point"] == 0).sum())
    print(f"Wrote synthetic sphere test to {out_dir}")
    print(
        "Visibility counts: "
        f"view1_only={view1_only}, view2_only={view2_only}, overlap={overlap}, unobserved={unobserved}"
    )
    print(f"Observation graph: {obs_path}")
    print(f"GT manifest: {gt_manifest}")
    print(f"Solved manifest: {solved_manifest}")
    print(f"Diagnostics: {out_dir / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
