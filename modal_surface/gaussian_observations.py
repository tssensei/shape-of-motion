"""Build modal observation graphs directly on foreground 3DGS Gaussians."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from modal_surface.geometry import erode_mask, project_points, projection_jacobian, unproject_pixels
from modal_surface.io import ViewConfig, ensure_modal_shape, load_mask, load_modal_npz, load_view_config


def _load_view_inputs(
    view_config_paths: Sequence[str | Path],
    modal_npz_paths: Sequence[str | Path],
    mode_index: int,
    freq_tolerance_hz: float,
) -> tuple[list[ViewConfig], list[dict[str, np.ndarray]], np.ndarray, float]:
    if len(view_config_paths) != len(modal_npz_paths):
        raise ValueError("--view-config and --modal-npz must be supplied the same number of times.")
    if len(view_config_paths) < 1:
        raise ValueError("At least one view is required.")
    if mode_index < 0:
        raise ValueError("mode_index must be non-negative.")
    if freq_tolerance_hz < 0:
        raise ValueError("freq_tolerance_hz must be non-negative.")

    configs: list[ViewConfig] = []
    modals: list[dict[str, np.ndarray]] = []
    freqs: list[float] = []
    for cfg_path, modal_path in zip(view_config_paths, modal_npz_paths):
        cfg = load_view_config(cfg_path)
        modal = load_modal_npz(modal_path)
        ensure_modal_shape(modal, (cfg.image_height, cfg.image_width))
        if mode_index >= modal["mode_u"].shape[0]:
            raise ValueError(f"mode_index={mode_index} is out of range for {modal_path}.")
        freq_hz = float(modal["selected_freqs_hz"][mode_index])
        configs.append(cfg)
        modals.append(modal)
        freqs.append(freq_hz)

    reference_freq_hz = freqs[0]
    for cfg, freq_hz in zip(configs[1:], freqs[1:]):
        if abs(reference_freq_hz - freq_hz) > freq_tolerance_hz:
            raise ValueError(
                f"Mode frequency mismatch for {cfg.view_id}: reference={reference_freq_hz:.6f}, "
                f"target={freq_hz:.6f}."
            )
    return configs, modals, np.asarray(freqs, dtype=np.float32), reference_freq_hz


def _require_scipy_kdtree():
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("pixel-candidates contribution sampling requires scipy.spatial.cKDTree.") from exc
    return cKDTree


def _quat_wxyz_to_rotmat(quats: np.ndarray) -> np.ndarray:
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


def _validate_pixel_candidate_inputs(
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


def _append_view_pixel_candidate_observations(
    points_world: np.ndarray,
    gaussian_scales: np.ndarray,
    gaussian_rotmats: np.ndarray,
    gaussian_opacities: np.ndarray,
    rendered_depth: np.ndarray,
    rendered_acc: np.ndarray,
    view_index: int,
    cfg: ViewConfig,
    modal: dict[str, np.ndarray],
    mode_index: int,
    mask_erode_iters: int,
    obs_point_indices: list[int],
    obs_view_indices: list[int],
    obs_pixels: list[list[float]],
    obs_y: list[list[complex]],
    obs_j: list[np.ndarray],
    obs_confidence: list[float],
    obs_camera_z: list[float],
    obs_contribution_weight: list[float],
    obs_contribution_score: list[float],
    obs_contribution_sum: list[float],
    obs_surface_pixels: list[list[float]],
    obs_surface_camera_z: list[float],
    pixel_sample_stride: int,
    pixel_candidate_k: int,
    pixel_preselect_k: int,
    pixel_max_samples_per_view: int,
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

    mode_u = modal["mode_u"][mode_index].astype(np.complex64)
    mode_v = modal["mode_v"][mode_index].astype(np.complex64)

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
    if flat_x.size > int(pixel_max_samples_per_view):
        selected = np.linspace(0, flat_x.size - 1, int(pixel_max_samples_per_view)).astype(np.int64)
        flat_x = flat_x[selected]
        flat_y = flat_y[selected]

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

    cKDTree = _require_scipy_kdtree()
    preselect_k = min(int(pixel_preselect_k), points_world.shape[0])
    tree = cKDTree(points_world.astype(np.float64))
    _, candidate_rows = tree.query(surface_points.astype(np.float64), k=preselect_k)
    if preselect_k == 1:
        candidate_rows = candidate_rows[:, None]

    _, point_camera_z = project_points(points_world, cfg.K, cfg.world_to_camera)
    added = 0
    min_contribution = float(pixel_min_contribution)
    scale_eps = np.float32(1e-8)
    for row, (x, y, depth, surface_point, candidates) in enumerate(
        zip(flat_x.tolist(), flat_y.tolist(), depths.tolist(), surface_points, candidate_rows)
    ):
        candidate_indices = np.asarray(candidates, dtype=np.int64)
        candidate_indices = candidate_indices[(candidate_indices >= 0) & (candidate_indices < points_world.shape[0])]
        if candidate_indices.size == 0:
            continue

        delta = surface_point[None, :].astype(np.float64) - points_world[candidate_indices].astype(np.float64)
        local = np.einsum("nij,nj->ni", np.swapaxes(gaussian_rotmats[candidate_indices], 1, 2), delta)
        scaled = local / np.maximum(gaussian_scales[candidate_indices].astype(np.float64), float(scale_eps))
        mahalanobis2 = np.sum(scaled * scaled, axis=1)
        scores = gaussian_opacities[candidate_indices].astype(np.float64) * np.exp(-0.5 * mahalanobis2)
        valid = np.isfinite(scores) & (scores >= min_contribution)
        if not np.any(valid):
            continue
        candidate_indices = candidate_indices[valid]
        scores = scores[valid]
        order = np.argsort(scores)[::-1][: int(pixel_candidate_k)]
        candidate_indices = candidate_indices[order]
        scores = scores[order]
        positive_z = np.isfinite(point_camera_z[candidate_indices]) & (point_camera_z[candidate_indices] > 0)
        if not np.any(positive_z):
            continue
        candidate_indices = candidate_indices[positive_z]
        scores = scores[positive_z]
        denom = float(np.sum(scores))
        if denom <= 0 or not np.isfinite(denom):
            continue
        weights = (scores / denom).astype(np.float32)
        jacobians = projection_jacobian(points_world[candidate_indices], cfg.K, cfg.world_to_camera)
        y_u = complex(mode_u[int(y), int(x)])
        y_v = complex(mode_v[int(y), int(x)])
        for row_idx, (point_idx, score, contribution_weight) in enumerate(
            zip(candidate_indices.tolist(), scores.tolist(), weights.tolist())
        ):
            candidate_z = float(point_camera_z[int(point_idx)])
            obs_point_indices.append(int(point_idx))
            obs_view_indices.append(view_index)
            obs_pixels.append([float(x), float(y)])
            obs_y.append([y_u, y_v])
            obs_j.append(jacobians[row_idx].astype(np.float32))
            obs_confidence.append(float(contribution_weight))
            obs_camera_z.append(candidate_z)
            obs_contribution_weight.append(float(contribution_weight))
            obs_contribution_score.append(float(score))
            obs_contribution_sum.append(denom)
            obs_surface_pixels.append([float(surface_pixels[row, 0]), float(surface_pixels[row, 1])])
            obs_surface_camera_z.append(float(depth))
            added += 1
    return added


def _write_gaussian_observation_graph(
    points_world_all: np.ndarray,
    configs: Sequence[ViewConfig],
    modals: Sequence[dict[str, np.ndarray]],
    view_freqs_hz: np.ndarray,
    reference_freq_hz: float,
    out_path: str | Path,
    mode_index: int,
    mask_erode_iters: int,
    source_view_config_paths: Sequence[str | Path],
    source_modal_npz_paths: Sequence[str | Path],
    preserve_all_points: bool = False,
    optional_point_fields: dict[str, np.ndarray] | None = None,
    extra_metadata: dict[str, np.ndarray] | None = None,
    pixel_sample_stride: int = 4,
    pixel_candidate_k: int = 4,
    pixel_preselect_k: int = 32,
    pixel_render_acc_min: float = 0.05,
    pixel_min_contribution: float = 1e-12,
    pixel_max_samples_per_view: int = 20000,
    gaussian_scales: np.ndarray | None = None,
    gaussian_quats: np.ndarray | None = None,
    gaussian_opacities: np.ndarray | None = None,
    rendered_depths: Sequence[np.ndarray] | None = None,
    rendered_accs: Sequence[np.ndarray] | None = None,
) -> Path:
    points_world_all = np.asarray(points_world_all, dtype=np.float32)
    if points_world_all.ndim != 2 or points_world_all.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points_world_all.shape}.")
    pixel_candidate_inputs = _validate_pixel_candidate_inputs(
        points_world_all,
        gaussian_scales,
        gaussian_quats,
        gaussian_opacities,
        rendered_depths,
        rendered_accs,
        configs,
    )
    pixel_candidate_rotmats = _quat_wxyz_to_rotmat(pixel_candidate_inputs[1])
    pixel_candidate_scales, _, pixel_candidate_opacities, pixel_candidate_depths, pixel_candidate_accs = (
        pixel_candidate_inputs
    )

    obs_point_indices: list[int] = []
    obs_view_indices: list[int] = []
    obs_pixels: list[list[float]] = []
    obs_y: list[list[complex]] = []
    obs_j: list[np.ndarray] = []
    obs_confidence: list[float] = []
    obs_camera_z: list[float] = []
    obs_contribution_weight: list[float] = []
    obs_contribution_score: list[float] = []
    obs_contribution_sum: list[float] = []
    obs_surface_pixels: list[list[float]] = []
    obs_surface_camera_z: list[float] = []
    observations_per_view: list[int] = []
    for view_index, (cfg, modal) in enumerate(zip(configs, modals)):
        count = _append_view_pixel_candidate_observations(
            points_world_all,
            pixel_candidate_scales,
            pixel_candidate_rotmats,
            pixel_candidate_opacities,
            pixel_candidate_depths[view_index],
            pixel_candidate_accs[view_index],
            view_index,
            cfg,
            modal,
            mode_index,
            mask_erode_iters,
            obs_point_indices,
            obs_view_indices,
            obs_pixels,
            obs_y,
            obs_j,
            obs_confidence,
            obs_camera_z,
            obs_contribution_weight,
            obs_contribution_score,
            obs_contribution_sum,
            obs_surface_pixels,
            obs_surface_camera_z,
            pixel_sample_stride,
            pixel_candidate_k,
            pixel_preselect_k,
            pixel_max_samples_per_view,
            pixel_render_acc_min,
            pixel_min_contribution,
        )
        observations_per_view.append(count)

    obs_point_arr = np.asarray(obs_point_indices, dtype=np.int64)
    obs_view_arr = np.asarray(obs_view_indices, dtype=np.int32)
    if obs_point_arr.size == 0:
        raise ValueError("No pixel-candidate observations survived rendered-depth, alpha, and mask checks.")

    sample_counts = np.bincount(obs_point_arr, minlength=points_world_all.shape[0])
    point_view_mask = np.zeros((points_world_all.shape[0], len(configs)), dtype=bool)
    point_view_mask[obs_point_arr, obs_view_arr] = True
    counts = point_view_mask.sum(axis=1).astype(np.int32)
    observed_points = counts > 0
    obs_confidence_arr = np.asarray(obs_confidence, dtype=np.float32)

    if preserve_all_points:
        active_old_indices = np.arange(points_world_all.shape[0], dtype=np.int64)
        old_to_new = active_old_indices.copy()
    else:
        active_old_indices = np.where(observed_points)[0]
        old_to_new = np.full(points_world_all.shape[0], -1, dtype=np.int64)
        old_to_new[active_old_indices] = np.arange(active_old_indices.size, dtype=np.int64)
    obs_fields_out: dict[str, np.ndarray] = {}
    for key, values in (
        ("obs_contribution_weight", obs_contribution_weight),
        ("obs_contribution_score", obs_contribution_score),
        ("obs_contribution_sum", obs_contribution_sum),
        ("obs_surface_pixels_xy", obs_surface_pixels),
        ("obs_surface_camera_z", obs_surface_camera_z),
    ):
        if len(values) != obs_point_arr.shape[0]:
            raise ValueError(f"Internal error: {key} count does not match observations.")
        obs_fields_out[key] = np.asarray(values, dtype=np.float32)

    optional_point_fields = dict(optional_point_fields or {})
    point_fields_out: dict[str, np.ndarray] = {}
    for key, value in optional_point_fields.items():
        value = np.asarray(value)
        if value.shape[:1] != (points_world_all.shape[0],):
            raise ValueError(
                f"Optional point field {key} must have first dimension "
                f"{points_world_all.shape[0]}, got {value.shape}."
            )
        point_fields_out[key] = value[active_old_indices]
    metadata = dict(extra_metadata or {})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points_world=points_world_all[active_old_indices].astype(np.float32),
        obs_point_index=old_to_new[obs_point_arr].astype(np.int32),
        obs_view_index=obs_view_arr.astype(np.int32),
        obs_pixels_xy=np.asarray(obs_pixels, dtype=np.float32),
        obs_y=np.asarray(obs_y, dtype=np.complex64),
        obs_J=np.asarray(obs_j, dtype=np.float32),
        obs_confidence=obs_confidence_arr,
        obs_camera_z=np.asarray(obs_camera_z, dtype=np.float32),
        obs_count_per_point=counts[active_old_indices].astype(np.int32),
        obs_sample_count_per_point=sample_counts[active_old_indices].astype(np.int32),
        view_ids=np.asarray([cfg.view_id for cfg in configs]),
        view_image_width=np.asarray([cfg.image_width for cfg in configs], dtype=np.int32),
        view_image_height=np.asarray([cfg.image_height for cfg in configs], dtype=np.int32),
        view_freqs_hz=view_freqs_hz.astype(np.float32),
        freq_hz=np.array(reference_freq_hz, dtype=np.float32),
        mode_index=np.array(mode_index, dtype=np.int32),
        mask_erode_iters=np.array(mask_erode_iters, dtype=np.int32),
        candidate_point_count=np.array(points_world_all.shape[0], dtype=np.int32),
        preserved_all_points=np.array(bool(preserve_all_points)),
        pixel_sample_stride=np.array(pixel_sample_stride, dtype=np.int32),
        pixel_candidate_k=np.array(pixel_candidate_k, dtype=np.int32),
        pixel_preselect_k=np.array(pixel_preselect_k, dtype=np.int32),
        pixel_render_acc_min=np.array(pixel_render_acc_min, dtype=np.float32),
        pixel_min_contribution=np.array(pixel_min_contribution, dtype=np.float32),
        pixel_max_samples_per_view=np.array(pixel_max_samples_per_view, dtype=np.int32),
        pixel_candidate_method=np.array("rendered_depth_gaussian_contribution"),
        observations_per_view=np.asarray(observations_per_view, dtype=np.int32),
        source_view_configs=np.asarray([str(path) for path in source_view_config_paths]),
        source_modal_npzs=np.asarray([str(path) for path in source_modal_npz_paths]),
        **obs_fields_out,
        **point_fields_out,
        **metadata,
    )
    return out


def _validate_pixel_candidate_args(
    pixel_sample_stride: int,
    pixel_candidate_k: int,
    pixel_preselect_k: int,
    pixel_render_acc_min: float,
    pixel_min_contribution: float,
    pixel_max_samples_per_view: int,
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
    if pixel_max_samples_per_view < 1:
        raise ValueError("pixel_max_samples_per_view must be at least 1.")


def build_gaussian_observation_graph(
    points_world: np.ndarray,
    view_config_paths: Sequence[str | Path],
    modal_npz_paths: Sequence[str | Path],
    out_path: str | Path,
    mode_index: int = 0,
    mask_erode_iters: int = 1,
    freq_tolerance_hz: float = 0.1,
    preserve_all_points: bool = False,
    optional_point_fields: dict[str, np.ndarray] | None = None,
    extra_metadata: dict[str, np.ndarray] | None = None,
    pixel_sample_stride: int = 4,
    pixel_candidate_k: int = 4,
    pixel_preselect_k: int = 32,
    pixel_render_acc_min: float = 0.05,
    pixel_min_contribution: float = 1e-12,
    pixel_max_samples_per_view: int = 20000,
    gaussian_scales: np.ndarray | None = None,
    gaussian_quats: np.ndarray | None = None,
    gaussian_opacities: np.ndarray | None = None,
    rendered_depths: Sequence[np.ndarray] | None = None,
    rendered_accs: Sequence[np.ndarray] | None = None,
) -> Path:
    """Build an N-view observation graph directly on foreground Gaussians."""
    _validate_pixel_candidate_args(
        pixel_sample_stride,
        pixel_candidate_k,
        pixel_preselect_k,
        pixel_render_acc_min,
        pixel_min_contribution,
        pixel_max_samples_per_view,
    )
    configs, modals, view_freqs_hz, reference_freq_hz = _load_view_inputs(
        view_config_paths,
        modal_npz_paths,
        mode_index,
        freq_tolerance_hz,
    )
    return _write_gaussian_observation_graph(
        points_world,
        configs,
        modals,
        view_freqs_hz,
        reference_freq_hz,
        out_path,
        mode_index,
        mask_erode_iters,
        view_config_paths,
        modal_npz_paths,
        preserve_all_points=preserve_all_points,
        optional_point_fields=optional_point_fields,
        extra_metadata=extra_metadata,
        pixel_sample_stride=pixel_sample_stride,
        pixel_candidate_k=pixel_candidate_k,
        pixel_preselect_k=pixel_preselect_k,
        pixel_render_acc_min=pixel_render_acc_min,
        pixel_min_contribution=pixel_min_contribution,
        pixel_max_samples_per_view=pixel_max_samples_per_view,
        gaussian_scales=gaussian_scales,
        gaussian_quats=gaussian_quats,
        gaussian_opacities=gaussian_opacities,
        rendered_depths=rendered_depths,
        rendered_accs=rendered_accs,
    )
