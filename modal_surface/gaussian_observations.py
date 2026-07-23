"""Build modal observation graphs directly on foreground 3DGS Gaussians."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from modal_surface.geometry import erode_mask, project_points, projection_jacobian, unproject_pixels
from modal_surface.io import (
    ViewConfig,
    ensure_modal_shape,
    load_mask,
    load_modal_npz,
    load_view_config,
    save_npz_compressed_atomic,
)

OBSERVATION_TOPOLOGY_FORMAT = "gaussian_observation_topology"
OBSERVATION_MEASUREMENT_FORMAT = "gaussian_observation_measurement"
OBSERVATION_SPLIT_VERSION = 1

def require_scipy_kdtree():
    try:
        from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]
    except ImportError as exc:
        raise ImportError("pixel-candidates contribution sampling requires scipy.spatial.cKDTree.") from exc
    return cKDTree


def quaternion_wxyz_to_rotation_matrices(quats: np.ndarray) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float64)
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"gaussian_quats must have shape (N,4), got {quats.shape}.")
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("gaussian_quats contains invalid zero or non-finite entries.")
    q = quats / norms
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    mats = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    mats[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    mats[:, 0, 1] = 2.0 * (x * y - w * z)
    mats[:, 0, 2] = 2.0 * (x * z + w * y)
    mats[:, 1, 0] = 2.0 * (x * y + w * z)
    mats[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    mats[:, 1, 2] = 2.0 * (y * z - w * x)
    mats[:, 2, 0] = 2.0 * (x * z - w * y)
    mats[:, 2, 1] = 2.0 * (y * z + w * x)
    mats[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return mats


def validate_pixel_candidate_inputs(
    points_world: np.ndarray,
    gaussian_scales: np.ndarray | None,
    gaussian_quats: np.ndarray | None,
    gaussian_opacities: np.ndarray | None,
    rendered_depths: Sequence[np.ndarray] | None,
    rendered_accs: Sequence[np.ndarray] | None,
    configs: Sequence[ViewConfig],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    if gaussian_scales is None:
        raise ValueError("pixel-candidates requires gaussian_scales.")
    if gaussian_quats is None:
        raise ValueError("pixel-candidates requires gaussian_quats.")
    if gaussian_opacities is None:
        raise ValueError("pixel-candidates requires gaussian_opacities.")
    if rendered_depths is None:
        raise ValueError("pixel-candidates requires rendered_depths.")
    if rendered_accs is None:
        raise ValueError("pixel-candidates requires rendered_accs.")

    scales = np.asarray(gaussian_scales, dtype=np.float32)
    quats = np.asarray(gaussian_quats, dtype=np.float32)
    opacities = np.asarray(gaussian_opacities, dtype=np.float32).reshape(-1)
    if scales.shape != points_world.shape:
        raise ValueError(f"gaussian_scales must have shape {points_world.shape}, got {scales.shape}.")
    if quats.shape != (points_world.shape[0], 4):
        raise ValueError(f"gaussian_quats must have shape ({points_world.shape[0]},4), got {quats.shape}.")
    if opacities.shape != (points_world.shape[0],):
        raise ValueError(f"gaussian_opacities must have shape ({points_world.shape[0]},), got {opacities.shape}.")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("gaussian_scales must be finite and positive.")
    if not np.all(np.isfinite(opacities)) or np.any(opacities < 0):
        raise ValueError("gaussian_opacities must be finite and non-negative.")

    depth_arrays: list[np.ndarray] = []
    acc_arrays: list[np.ndarray] = []
    if len(rendered_depths) != len(configs) or len(rendered_accs) != len(configs):
        raise ValueError("rendered_depths/rendered_accs must have one array per view.")
    for cfg, depth, acc in zip(configs, rendered_depths, rendered_accs):
        expected_shape = (cfg.image_height, cfg.image_width)
        depth_arr = np.asarray(depth, dtype=np.float32)
        acc_arr = np.asarray(acc, dtype=np.float32)
        if depth_arr.shape != expected_shape:
            raise ValueError(f"Rendered depth for {cfg.view_id} must have shape {expected_shape}, got {depth_arr.shape}.")
        if acc_arr.shape != expected_shape:
            raise ValueError(f"Rendered alpha for {cfg.view_id} must have shape {expected_shape}, got {acc_arr.shape}.")
        depth_arrays.append(depth_arr)
        acc_arrays.append(acc_arr)
    return scales, quats, opacities, depth_arrays, acc_arrays


@dataclass(frozen=True)
class ScoredPixelGaussianCandidates:
    gaussian_indices: np.ndarray
    contribution_scores: np.ndarray
    positive_camera_z_mask: np.ndarray


def score_pixel_gaussian_candidates(
    *,
    surface_point_world: np.ndarray,
    candidate_indices: np.ndarray,
    points_world: np.ndarray,
    gaussian_scales: np.ndarray,
    gaussian_rotmats: np.ndarray,
    gaussian_opacities: np.ndarray,
    point_camera_z: np.ndarray,
    min_contribution: float,
) -> ScoredPixelGaussianCandidates:
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    indices = indices[(indices >= 0) & (indices < points_world.shape[0])]
    if indices.size == 0:
        return ScoredPixelGaussianCandidates(
            gaussian_indices=np.empty((0,), dtype=np.int64),
            contribution_scores=np.empty((0,), dtype=np.float64),
            positive_camera_z_mask=np.empty((0,), dtype=bool),
        )
    delta = (
        np.asarray(surface_point_world, dtype=np.float64)[None]
        - points_world[indices].astype(np.float64)
    )
    local = np.einsum(
        "nij,nj->ni",
        np.swapaxes(gaussian_rotmats[indices], 1, 2),
        delta,
    )
    scaled = local / np.maximum(
        gaussian_scales[indices].astype(np.float64),
        float(np.float32(1.0e-8)),
    )
    mahalanobis2 = np.sum(scaled * scaled, axis=1)
    scores = gaussian_opacities[indices].astype(np.float64) * np.exp(
        -0.5 * mahalanobis2
    )
    valid = np.isfinite(scores) & (scores >= float(min_contribution))
    indices = indices[valid]
    scores = scores[valid]
    if indices.size == 0:
        return ScoredPixelGaussianCandidates(
            gaussian_indices=np.empty((0,), dtype=np.int64),
            contribution_scores=np.empty((0,), dtype=np.float64),
            positive_camera_z_mask=np.empty((0,), dtype=bool),
        )
    order = np.argsort(scores)[::-1]
    indices = indices[order]
    scores = scores[order]
    positive_z = np.isfinite(point_camera_z[indices]) & (
        point_camera_z[indices] > 0
    )
    return ScoredPixelGaussianCandidates(
        gaussian_indices=indices,
        contribution_scores=scores,
        positive_camera_z_mask=positive_z,
    )


def _append_view_pixel_candidate_topology(
    points_world: np.ndarray,
    gaussian_tree: Any,
    gaussian_scales: np.ndarray,
    gaussian_rotmats: np.ndarray,
    gaussian_opacities: np.ndarray,
    rendered_depth: np.ndarray,
    rendered_acc: np.ndarray,
    view_index: int,
    cfg: ViewConfig,
    mask_erode_iters: int,
    obs_point_indices: list[int],
    obs_view_indices: list[int],
    obs_pixels: list[list[float]],
    obs_j: list[np.ndarray],
    obs_contribution_weight: list[float],
    obs_contribution_score: list[float],
    obs_contribution_sum: list[float],
    obs_surface_pixels: list[list[float]],
    obs_surface_camera_z: list[float],
    pixel_sample_stride: int,
    pixel_candidate_k: int,
    pixel_preselect_k: int,
    pixel_render_acc_min: float,
    pixel_min_contribution: float,
) -> int:
    expected_shape = (cfg.image_height, cfg.image_width)
    mask = load_mask(cfg.mask_path, expected_shape)
    valid_mask = erode_mask(mask, mask_erode_iters)

    valid_depth = np.isfinite(rendered_depth) & (rendered_depth > 0)
    valid_acc = np.isfinite(rendered_acc) & (rendered_acc >= float(pixel_render_acc_min))
    visible_mask = valid_mask & valid_depth & valid_acc
    if not np.any(visible_mask):
        return 0

    ys = np.arange(1, cfg.image_height - 1, int(pixel_sample_stride), dtype=np.int32)
    xs = np.arange(1, cfg.image_width - 1, int(pixel_sample_stride), dtype=np.int32)
    if ys.size == 0 or xs.size == 0:
        return 0
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    keep = visible_mask[flat_y, flat_x]
    flat_x = flat_x[keep]
    flat_y = flat_y[keep]
    if flat_x.size == 0:
        return 0

    depths = rendered_depth[flat_y, flat_x].astype(np.float32)
    surface_pixels = np.stack([flat_x.astype(np.float32), flat_y.astype(np.float32)], axis=1)
    surface_points = unproject_pixels(surface_pixels, depths, cfg.K, cfg.world_to_camera)
    finite_surface = np.all(np.isfinite(surface_points), axis=1) & np.isfinite(depths) & (depths > 0)
    if not np.any(finite_surface):
        return 0
    flat_x = flat_x[finite_surface]
    flat_y = flat_y[finite_surface]
    depths = depths[finite_surface]
    surface_pixels = surface_pixels[finite_surface]
    surface_points = surface_points[finite_surface]

    preselect_k = min(int(pixel_preselect_k), points_world.shape[0])
    _, candidate_rows = gaussian_tree.query(
        surface_points.astype(np.float64), k=preselect_k
    )
    if preselect_k == 1:
        candidate_rows = candidate_rows[:, None]

    _, point_camera_z = project_points(points_world, cfg.K, cfg.world_to_camera)
    added = 0
    min_contribution = float(pixel_min_contribution)

    # for a pixel
    for row, (x, y, depth, surface_point, candidates) in enumerate(
        zip(flat_x.tolist(), flat_y.tolist(), depths.tolist(), surface_points, candidate_rows)
    ):
        scored = score_pixel_gaussian_candidates(
            surface_point_world=surface_point,
            candidate_indices=np.asarray(candidates, dtype=np.int64),
            points_world=points_world,
            gaussian_scales=gaussian_scales,
            gaussian_rotmats=gaussian_rotmats,
            gaussian_opacities=gaussian_opacities,
            point_camera_z=point_camera_z,
            min_contribution=min_contribution,
        )
        if scored.gaussian_indices.size == 0:
            continue
        candidate_indices = scored.gaussian_indices[: int(pixel_candidate_k)]
        scores = scored.contribution_scores[: int(pixel_candidate_k)]
        positive_z = scored.positive_camera_z_mask[: int(pixel_candidate_k)]
        if not np.any(positive_z):
            continue
        candidate_indices = candidate_indices[positive_z]
        scores = scores[positive_z]

        denom = float(np.sum(scores))
        if denom <= 0 or not np.isfinite(denom):
            continue
        weights = (scores / denom).astype(np.float32)
        jacobians = projection_jacobian(points_world[candidate_indices], cfg.K, cfg.world_to_camera)
        for row_idx, (point_idx, score, contribution_weight) in enumerate(
            zip(candidate_indices.tolist(), scores.tolist(), weights.tolist())
        ):
            obs_point_indices.append(int(point_idx))
            obs_view_indices.append(view_index)
            obs_pixels.append([float(x), float(y)])
            obs_j.append(jacobians[row_idx].astype(np.float32))
            obs_contribution_weight.append(float(contribution_weight))
            obs_contribution_score.append(float(score))
            obs_contribution_sum.append(denom)
            obs_surface_pixels.append([float(surface_pixels[row, 0]), float(surface_pixels[row, 1])])
            obs_surface_camera_z.append(float(depth))
            added += 1
    return added


def build_gaussian_observation_graph(
    points_world: np.ndarray,
    view_config_paths: Sequence[str | Path],
    modal_npz_paths: Sequence[str | Path],
    out_path: str | Path,
    source_checkpoint: str | Path,
    mode_index: int = 0,
    mask_erode_iters: int = 1,
    freq_tolerance_hz: float = 0.1,
    pixel_sample_stride: int = 4,
    pixel_candidate_k: int = 4,
    pixel_preselect_k: int = 32,
    pixel_render_acc_min: float = 0.05,
    pixel_min_contribution: float = 1e-12,
    gaussian_scales: np.ndarray | None = None,
    gaussian_quats: np.ndarray | None = None,
    gaussian_opacities: np.ndarray | None = None,
    rendered_depths: Sequence[np.ndarray] | None = None,
    rendered_accs: Sequence[np.ndarray] | None = None,
    gaussian_tree: Any | None = None,
) -> Path:
    """Build an N-view observation graph, optionally reusing a Gaussian KD-tree."""
    topology = build_gaussian_observation_topology(
        points_world=points_world,
        view_config_paths=view_config_paths,
        source_checkpoint=source_checkpoint,
        mask_erode_iters=mask_erode_iters,
        pixel_sample_stride=pixel_sample_stride,
        pixel_candidate_k=pixel_candidate_k,
        pixel_preselect_k=pixel_preselect_k,
        pixel_render_acc_min=pixel_render_acc_min,
        pixel_min_contribution=pixel_min_contribution,
        gaussian_scales=gaussian_scales,
        gaussian_quats=gaussian_quats,
        gaussian_opacities=gaussian_opacities,
        rendered_depths=rendered_depths,
        rendered_accs=rendered_accs,
        gaussian_tree=gaussian_tree,
    )
    measurement = build_gaussian_observation_measurements(
        topology,
        modal_npz_paths,
        [mode_index],
        freq_tolerance_hz,
    )[0]
    observations = compose_gaussian_observations(topology, measurement)
    out = Path(out_path)
    save_npz_compressed_atomic(out, observations)
    return out


def build_gaussian_observation_topology(
    points_world: np.ndarray,
    view_config_paths: Sequence[str | Path],
    source_checkpoint: str | Path,
    mask_erode_iters: int = 1,
    pixel_sample_stride: int = 4,
    pixel_candidate_k: int = 4,
    pixel_preselect_k: int = 32,
    pixel_render_acc_min: float = 0.05,
    pixel_min_contribution: float = 1e-12,
    gaussian_scales: np.ndarray | None = None,
    gaussian_quats: np.ndarray | None = None,
    gaussian_opacities: np.ndarray | None = None,
    rendered_depths: Sequence[np.ndarray] | None = None,
    rendered_accs: Sequence[np.ndarray] | None = None,
    gaussian_tree: Any | None = None,
) -> dict[str, np.ndarray]:
    """Build mode-independent Gaussian/view/pixel observation topology."""
    validate_pixel_candidate_args(
        pixel_sample_stride,
        pixel_candidate_k,
        pixel_preselect_k,
        pixel_render_acc_min,
        pixel_min_contribution,
    )
    if not view_config_paths:
        raise ValueError("At least one view is required.")
    configs = [load_view_config(path) for path in view_config_paths]
    view_ids = [cfg.view_id for cfg in configs]
    if len(set(view_ids)) != len(view_ids):
        raise ValueError("View configs must contain unique view IDs.")

    points_world_all = np.asarray(points_world, dtype=np.float32)
    if points_world_all.ndim != 2 or points_world_all.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points_world_all.shape}.")
    (
        gaussian_scales,
        gaussian_quats,
        gaussian_opacities,
        rendered_depths,
        rendered_accs,
    ) = validate_pixel_candidate_inputs(
        points_world_all,
        gaussian_scales,
        gaussian_quats,
        gaussian_opacities,
        rendered_depths,
        rendered_accs,
        configs,
    )
    if gaussian_tree is None:
        cKDTree = require_scipy_kdtree()
        gaussian_tree = cKDTree(points_world_all.astype(np.float64))
    gaussian_rotmats = quaternion_wxyz_to_rotation_matrices(gaussian_quats)

    obs_point_indices: list[int] = []
    obs_view_indices: list[int] = []
    obs_pixels: list[list[float]] = []
    obs_j: list[np.ndarray] = []
    obs_contribution_weight: list[float] = []
    obs_contribution_score: list[float] = []
    obs_contribution_sum: list[float] = []
    obs_surface_pixels: list[list[float]] = []
    obs_surface_camera_z: list[float] = []
    observations_per_view: list[int] = []
    for view_index, cfg in enumerate(configs):
        count = _append_view_pixel_candidate_topology(
            points_world_all,
            gaussian_tree,
            gaussian_scales,
            gaussian_rotmats,
            gaussian_opacities,
            rendered_depths[view_index],
            rendered_accs[view_index],
            view_index,
            cfg,
            mask_erode_iters,
            obs_point_indices,
            obs_view_indices,
            obs_pixels,
            obs_j,
            obs_contribution_weight,
            obs_contribution_score,
            obs_contribution_sum,
            obs_surface_pixels,
            obs_surface_camera_z,
            pixel_sample_stride,
            pixel_candidate_k,
            pixel_preselect_k,
            pixel_render_acc_min,
            pixel_min_contribution,
        )
        observations_per_view.append(count)

    obs_point_arr = np.asarray(obs_point_indices, dtype=np.int64)
    obs_view_arr = np.asarray(obs_view_indices, dtype=np.int32)
    if obs_point_arr.size == 0:
        raise ValueError("No pixel-candidate observations survived rendered-depth, alpha, and mask checks.")

    sample_counts = np.bincount(obs_point_arr, minlength=points_world_all.shape[0])
    gaus_view_obs_mask = np.zeros((points_world_all.shape[0], len(configs)), dtype=bool)
    gaus_view_obs_mask[obs_point_arr, obs_view_arr] = True
    counts = gaus_view_obs_mask.sum(axis=1).astype(np.int32)
    for key, values in (
        ("obs_contribution_weight", obs_contribution_weight),
        ("obs_contribution_score", obs_contribution_score),
        ("obs_contribution_sum", obs_contribution_sum),
        ("obs_surface_pixels_xy", obs_surface_pixels),
        ("obs_surface_camera_z", obs_surface_camera_z),
    ):
        if len(values) != obs_point_arr.shape[0]:
            raise ValueError(f"Internal error: {key} count does not match observations.")

    topology_source = {
        "points_world": points_world_all,
        "gaussian_indices": np.arange(points_world_all.shape[0], dtype=np.int32),
        "point_type": np.array("foreground_gaussian_center"),
        "source_checkpoint": np.array(str(source_checkpoint)),
        "obs_point_index": obs_point_arr.astype(np.int32),
        "obs_view_index": obs_view_arr.astype(np.int32),
        "obs_pixels_xy": np.asarray(obs_pixels, dtype=np.float32),
        "obs_J": np.asarray(obs_j, dtype=np.float32),
        "obs_count_per_point": counts,
        "obs_sample_count_per_point": sample_counts.astype(np.int32),
        "view_ids": np.asarray(view_ids),
        "view_image_width": np.asarray(
            [cfg.image_width for cfg in configs], dtype=np.int32
        ),
        "view_image_height": np.asarray(
            [cfg.image_height for cfg in configs], dtype=np.int32
        ),
        "mask_erode_iters": np.array(mask_erode_iters, dtype=np.int32),
        "candidate_point_count": np.array(
            points_world_all.shape[0], dtype=np.int32
        ),
        "preserved_all_points": np.array(True),
        "pixel_sample_stride": np.array(pixel_sample_stride, dtype=np.int32),
        "pixel_candidate_k": np.array(pixel_candidate_k, dtype=np.int32),
        "pixel_preselect_k": np.array(pixel_preselect_k, dtype=np.int32),
        "pixel_render_acc_min": np.array(pixel_render_acc_min, dtype=np.float32),
        "pixel_min_contribution": np.array(
            pixel_min_contribution, dtype=np.float32
        ),
        "pixel_candidate_method": np.array(
            "rendered_depth_gaussian_contribution"
        ),
        "observations_per_view": np.asarray(
            observations_per_view, dtype=np.int32
        ),
        "source_view_configs": np.asarray(
            [str(path) for path in view_config_paths]
        ),
        "obs_contribution_weight": np.asarray(
            obs_contribution_weight, dtype=np.float32
        ),
        "obs_contribution_score": np.asarray(
            obs_contribution_score, dtype=np.float32
        ),
        "obs_contribution_sum": np.asarray(
            obs_contribution_sum, dtype=np.float32
        ),
        "obs_surface_pixels_xy": np.asarray(
            obs_surface_pixels, dtype=np.float32
        ),
        "obs_surface_camera_z": np.asarray(
            obs_surface_camera_z, dtype=np.float32
        ),
    }
    return split_gaussian_observation_topology(topology_source)


def _observation_topology_id(topology: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in (
        "points_world",
        "gaussian_indices",
        "obs_point_index",
        "obs_sample_index",
        "obs_J",
        "obs_contribution_weight",
        "obs_contribution_score",
        "sample_view_index",
        "sample_pixels_xy",
        "sample_contribution_sum",
        "sample_surface_pixels_xy",
        "sample_surface_camera_z",
        "view_ids",
        "view_image_width",
        "view_image_height",
    ):
        array = np.ascontiguousarray(topology[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def split_gaussian_observation_topology(
    observations: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Extract mode-independent topology from one full observation artifact."""
    obs_view_index = np.asarray(observations["obs_view_index"], dtype=np.int32)
    obs_pixels_xy = np.asarray(observations["obs_pixels_xy"], dtype=np.float32)
    if obs_view_index.ndim != 1 or obs_pixels_xy.shape != (obs_view_index.size, 2):
        raise ValueError("Observation rows have invalid view/pixel shapes.")
    if obs_view_index.size == 0:
        raise ValueError("Observation topology cannot be empty.")

    sample_start = np.ones(obs_view_index.size, dtype=bool)
    sample_start[1:] = (
        (obs_view_index[1:] != obs_view_index[:-1])
        | np.any(obs_pixels_xy[1:] != obs_pixels_xy[:-1], axis=1)
    )
    sample_rows = np.flatnonzero(sample_start)
    obs_sample_index = np.cumsum(sample_start, dtype=np.int64) - 1

    topology = {
        "format": np.array(OBSERVATION_TOPOLOGY_FORMAT),
        "version": np.array(OBSERVATION_SPLIT_VERSION, dtype=np.int32),
        "points_world": np.asarray(observations["points_world"], dtype=np.float32),
        "gaussian_indices": np.asarray(observations["gaussian_indices"], dtype=np.int32),
        "point_type": np.asarray(observations["point_type"]),
        "source_checkpoint": np.asarray(observations["source_checkpoint"]),
        "obs_point_index": np.asarray(observations["obs_point_index"], dtype=np.int32),
        "obs_sample_index": obs_sample_index.astype(np.int32),
        "obs_J": np.asarray(observations["obs_J"], dtype=np.float32),
        "obs_count_per_point": np.asarray(observations["obs_count_per_point"], dtype=np.int32),
        "obs_sample_count_per_point": np.asarray(
            observations["obs_sample_count_per_point"], dtype=np.int32
        ),
        "view_ids": np.asarray(observations["view_ids"]),
        "view_image_width": np.asarray(observations["view_image_width"], dtype=np.int32),
        "view_image_height": np.asarray(observations["view_image_height"], dtype=np.int32),
        "mask_erode_iters": np.asarray(observations["mask_erode_iters"], dtype=np.int32),
        "candidate_point_count": np.asarray(
            observations["candidate_point_count"], dtype=np.int32
        ),
        "preserved_all_points": np.asarray(observations["preserved_all_points"]),
        "pixel_sample_stride": np.asarray(
            observations["pixel_sample_stride"], dtype=np.int32
        ),
        "pixel_candidate_k": np.asarray(
            observations["pixel_candidate_k"], dtype=np.int32
        ),
        "pixel_preselect_k": np.asarray(
            observations["pixel_preselect_k"], dtype=np.int32
        ),
        "pixel_render_acc_min": np.asarray(
            observations["pixel_render_acc_min"], dtype=np.float32
        ),
        "pixel_min_contribution": np.asarray(
            observations["pixel_min_contribution"], dtype=np.float32
        ),
        "pixel_candidate_method": np.asarray(observations["pixel_candidate_method"]),
        "observations_per_view": np.asarray(
            observations["observations_per_view"], dtype=np.int32
        ),
        "source_view_configs": np.asarray(observations["source_view_configs"]),
        "obs_contribution_weight": np.asarray(
            observations["obs_contribution_weight"], dtype=np.float32
        ),
        "obs_contribution_score": np.asarray(
            observations["obs_contribution_score"], dtype=np.float32
        ),
        "sample_view_index": obs_view_index[sample_rows],
        "sample_pixels_xy": obs_pixels_xy[sample_rows],
        "sample_contribution_sum": np.asarray(
            observations["obs_contribution_sum"], dtype=np.float32
        )[sample_rows],
        "sample_surface_pixels_xy": np.asarray(
            observations["obs_surface_pixels_xy"], dtype=np.float32
        )[sample_rows],
        "sample_surface_camera_z": np.asarray(
            observations["obs_surface_camera_z"], dtype=np.float32
        )[sample_rows],
    }
    topology["samples_per_view"] = np.bincount(
        topology["sample_view_index"],
        minlength=topology["view_ids"].shape[0],
    ).astype(np.int32)
    topology["topology_id"] = np.array(_observation_topology_id(topology))
    return topology


def write_gaussian_observation_topology(
    path: str | Path,
    topology: Mapping[str, np.ndarray],
) -> Path:
    out = Path(path)
    save_npz_compressed_atomic(out, dict(topology))
    return out


def load_gaussian_observation_topology(
    path: str | Path,
) -> dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve(strict=True)
    with np.load(source, allow_pickle=False) as archive:
        topology = {name: archive[name] for name in archive.files}
    if str(np.asarray(topology.get("format")).item()) != OBSERVATION_TOPOLOGY_FORMAT:
        raise ValueError(f"{source} is not a Gaussian observation topology artifact.")
    if int(np.asarray(topology.get("version")).item()) != OBSERVATION_SPLIT_VERSION:
        raise ValueError(f"{source} has an unsupported observation topology version.")
    return topology


def build_gaussian_observation_measurements(
    topology: Mapping[str, np.ndarray],
    modal_npz_paths: Sequence[str | Path],
    mode_indices: Sequence[int],
    freq_tolerance_hz: float,
) -> list[dict[str, np.ndarray]]:
    if not mode_indices:
        raise ValueError("At least one mode index is required.")
    topology_view_ids = np.asarray(topology["view_ids"]).astype(str)
    view_widths = np.asarray(topology["view_image_width"], dtype=np.int32)
    view_heights = np.asarray(topology["view_image_height"], dtype=np.int32)
    if len(modal_npz_paths) != topology_view_ids.size:
        raise ValueError(
            "One --modal-npz must be supplied for each observation topology view."
        )
    if (
        view_widths.shape != topology_view_ids.shape
        or view_heights.shape != topology_view_ids.shape
    ):
        raise ValueError("Observation topology view dimensions have invalid shapes.")
    if freq_tolerance_hz < 0:
        raise ValueError("freq_tolerance_hz must be non-negative.")

    modals: list[dict[str, np.ndarray]] = []
    for view_index, modal_path in enumerate(modal_npz_paths):
        modal = load_modal_npz(modal_path)
        ensure_modal_shape(
            modal,
            (int(view_heights[view_index]), int(view_widths[view_index])),
        )
        modals.append(modal)

    sample_view_index = np.asarray(topology["sample_view_index"], dtype=np.int32)
    sample_pixels_xy = np.asarray(topology["sample_pixels_xy"], dtype=np.float32)
    sample_x = sample_pixels_xy[:, 0].astype(np.int64)
    sample_y = sample_pixels_xy[:, 1].astype(np.int64)
    topology_id = np.asarray(topology["topology_id"])
    measurements: list[dict[str, np.ndarray]] = []
    for raw_mode_index in mode_indices:
        mode_index = int(raw_mode_index)
        if mode_index < 0 or any(
            mode_index >= modal["mode_u"].shape[0] for modal in modals
        ):
            raise ValueError(f"mode_index={mode_index} is out of range.")
        view_freqs_hz = np.asarray(
            [float(modal["selected_freqs_hz"][mode_index]) for modal in modals],
            dtype=np.float32,
        )
        if np.max(np.abs(view_freqs_hz - view_freqs_hz[0])) > freq_tolerance_hz:
            raise ValueError(f"Mode frequency mismatch at mode_index={mode_index}.")
        values = np.empty((sample_view_index.size, 2), dtype=np.complex64)
        for view_index, modal in enumerate(modals):
            rows = sample_view_index == view_index
            values[rows, 0] = modal["mode_u"][
                mode_index, sample_y[rows], sample_x[rows]
            ]
            values[rows, 1] = modal["mode_v"][
                mode_index, sample_y[rows], sample_x[rows]
            ]
        measurements.append(
            {
                "format": np.array(OBSERVATION_MEASUREMENT_FORMAT),
                "version": np.array(OBSERVATION_SPLIT_VERSION, dtype=np.int32),
                "topology_id": topology_id,
                "mode_index": np.array(mode_index, dtype=np.int32),
                "freq_hz": np.array(view_freqs_hz[0], dtype=np.float32),
                "view_freqs_hz": view_freqs_hz,
                "sample_y": values,
                "source_modal_npzs": np.asarray(
                    [str(path) for path in modal_npz_paths]
                ),
            }
        )
    return measurements


def write_gaussian_observation_measurement(
    path: str | Path,
    measurement: Mapping[str, np.ndarray],
) -> Path:
    out = Path(path)
    save_npz_compressed_atomic(out, dict(measurement))
    return out


def load_gaussian_observation_measurement(
    path: str | Path,
) -> dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve(strict=True)
    with np.load(source, allow_pickle=False) as archive:
        measurement = {name: archive[name] for name in archive.files}
    if str(np.asarray(measurement.get("format")).item()) != OBSERVATION_MEASUREMENT_FORMAT:
        raise ValueError(f"{source} is not a Gaussian observation measurement artifact.")
    if int(np.asarray(measurement.get("version")).item()) != OBSERVATION_SPLIT_VERSION:
        raise ValueError(f"{source} has an unsupported observation measurement version.")
    return measurement


def compose_gaussian_observations(
    topology: Mapping[str, np.ndarray],
    measurement: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if str(np.asarray(topology["topology_id"]).item()) != str(
        np.asarray(measurement["topology_id"]).item()
    ):
        raise ValueError("Observation measurement does not belong to this topology.")
    obs_sample_index = np.asarray(topology["obs_sample_index"], dtype=np.int32)
    sample_values = np.asarray(measurement["sample_y"], dtype=np.complex64)
    if sample_values.shape != (
        np.asarray(topology["sample_view_index"]).shape[0],
        2,
    ):
        raise ValueError("Observation measurement sample_y has an invalid shape.")

    observations = {
        name: np.asarray(value)
        for name, value in topology.items()
        if name
        not in {
            "format",
            "version",
            "topology_id",
            "obs_sample_index",
            "sample_view_index",
            "sample_pixels_xy",
            "sample_contribution_sum",
            "sample_surface_pixels_xy",
            "sample_surface_camera_z",
            "samples_per_view",
        }
    }
    observations.update(
        {
            "obs_view_index": np.asarray(topology["sample_view_index"])[obs_sample_index],
            "obs_pixels_xy": np.asarray(topology["sample_pixels_xy"])[obs_sample_index],
            "obs_y": sample_values[obs_sample_index],
            "obs_contribution_sum": np.asarray(
                topology["sample_contribution_sum"]
            )[obs_sample_index],
            "obs_surface_pixels_xy": np.asarray(
                topology["sample_surface_pixels_xy"]
            )[obs_sample_index],
            "obs_surface_camera_z": np.asarray(
                topology["sample_surface_camera_z"]
            )[obs_sample_index],
            "view_freqs_hz": np.asarray(measurement["view_freqs_hz"]),
            "freq_hz": np.asarray(measurement["freq_hz"]),
            "mode_index": np.asarray(measurement["mode_index"]),
            "source_modal_npzs": np.asarray(measurement["source_modal_npzs"]),
        }
    )
    return observations


def validate_pixel_candidate_args(
    pixel_sample_stride: int,
    pixel_candidate_k: int,
    pixel_preselect_k: int,
    pixel_render_acc_min: float,
    pixel_min_contribution: float,
) -> None:
    if pixel_sample_stride < 1:
        raise ValueError("pixel_sample_stride must be at least 1.")
    if pixel_candidate_k < 1:
        raise ValueError("pixel_candidate_k must be at least 1.")
    if pixel_preselect_k < pixel_candidate_k:
        raise ValueError("pixel_preselect_k must be at least pixel_candidate_k.")
    if not (0.0 <= pixel_render_acc_min <= 1.0):
        raise ValueError("pixel_render_acc_min must be in [0, 1].")
    if pixel_min_contribution < 0:
        raise ValueError("pixel_min_contribution must be non-negative.")
