"""Replay pixel-to-Gaussian selection to diagnose multi-view coverage loss."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from modal_surface.gaussian_observations import (
    quaternion_wxyz_to_rotation_matrices,
    require_scipy_kdtree,
    score_pixel_gaussian_candidates,
    validate_pixel_candidate_args,
    validate_pixel_candidate_inputs,
)
from modal_surface.geometry import erode_mask, project_points, unproject_pixels
from modal_surface.io import (
    load_mask,
    load_view_config,
    save_npz_compressed_atomic,
)


OBSERVATION_COVERAGE_VERSION = 1
OBSERVATION_COVERAGE_CATEGORY_NAMES = (
    "insufficient_preselect_views",
    "multiview_preselect_contribution_lost",
    "multiview_positive_topk_lost",
    "selected_multiview",
)


@dataclass(frozen=True)
class ObservationCoverageDiagnostics:
    points_world: np.ndarray
    view_ids: np.ndarray
    k_values: np.ndarray
    preselect_hit_count_by_view: np.ndarray
    positive_hit_count_by_view: np.ndarray
    selected_hit_count_by_k_view: np.ndarray
    best_positive_rank_by_view: np.ndarray
    best_positive_score_by_view: np.ndarray
    best_positive_score_ratio_by_view: np.ndarray
    preselect_view_count: np.ndarray
    positive_view_count: np.ndarray
    selected_view_count_by_k: np.ndarray
    selected_sample_count_by_k: np.ndarray
    category_by_k: np.ndarray
    category_count_by_k: np.ndarray
    preselect_view_count_histogram: np.ndarray
    positive_view_count_histogram: np.ndarray
    selected_view_count_histogram_by_k: np.ndarray


def parse_coverage_k_values(raw: str, preselect_k: int) -> np.ndarray:
    values = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        value = int(text)
        if value <= 0 or value > preselect_k:
            raise ValueError(
                "Coverage K values must be positive and no greater than "
                f"pixel_preselect_k={preselect_k}"
            )
        values.append(value)
    if not values:
        raise ValueError("Coverage K values must contain at least one integer")
    if len(set(values)) != len(values):
        raise ValueError("Coverage K values must be unique")
    if values != sorted(values):
        raise ValueError("Coverage K values must be strictly increasing")
    return np.asarray(values, dtype=np.int32)


def _view_count_histogram(view_count: np.ndarray, num_views: int) -> np.ndarray:
    return np.bincount(
        np.asarray(view_count, dtype=np.int64),
        minlength=num_views + 1,
    ).astype(np.int64)


def observation_coverage_categories(
    preselect_view_count: np.ndarray,
    positive_view_count: np.ndarray,
    selected_view_count: np.ndarray,
) -> np.ndarray:
    preselect = np.asarray(preselect_view_count)
    positive = np.asarray(positive_view_count)
    selected = np.asarray(selected_view_count)
    if preselect.shape != positive.shape or preselect.shape != selected.shape:
        raise ValueError("Coverage view-count arrays must have matching shapes")
    if preselect.ndim != 1:
        raise ValueError("Coverage view-count arrays must be one-dimensional")
    category = np.zeros(preselect.shape, dtype=np.int8)
    category[(preselect >= 2) & (positive < 2)] = 1
    category[(positive >= 2) & (selected < 2)] = 2
    category[selected >= 2] = 3
    return category


def build_observation_coverage_diagnostics(
    *,
    points_world: np.ndarray,
    gaussian_scales: np.ndarray,
    gaussian_quats: np.ndarray,
    gaussian_opacities: np.ndarray,
    rendered_depths: Sequence[np.ndarray],
    rendered_accs: Sequence[np.ndarray],
    view_config_paths: Sequence[str | Path],
    mask_erode_iters: int,
    pixel_sample_stride: int,
    pixel_preselect_k: int,
    pixel_render_acc_min: float,
    pixel_min_contribution: float,
    k_values: np.ndarray,
    gaussian_tree: Any | None = None,
) -> ObservationCoverageDiagnostics:
    k_values = np.asarray(k_values, dtype=np.int32)
    if k_values.ndim != 1 or k_values.shape[0] == 0:
        raise ValueError("k_values must be a non-empty one-dimensional array")
    if np.any(k_values <= 0) or np.any(k_values > pixel_preselect_k):
        raise ValueError("k_values must lie in [1,pixel_preselect_k]")
    if k_values.shape[0] > 1 and np.any(np.diff(k_values) <= 0):
        raise ValueError("k_values must be strictly increasing")
    validate_pixel_candidate_args(
        pixel_sample_stride,
        int(k_values[-1]),
        pixel_preselect_k,
        pixel_render_acc_min,
        pixel_min_contribution,
    )
    if mask_erode_iters < 0:
        raise ValueError("mask_erode_iters must be non-negative")
    configs = [load_view_config(path) for path in view_config_paths]
    if not configs:
        raise ValueError("At least one view config is required")
    points = np.asarray(points_world, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N,3)")
    (
        scales,
        quats,
        opacities,
        depth_arrays,
        acc_arrays,
    ) = validate_pixel_candidate_inputs(
        points,
        gaussian_scales,
        gaussian_quats,
        gaussian_opacities,
        rendered_depths,
        rendered_accs,
        configs,
    )
    if gaussian_tree is None:
        cKDTree = require_scipy_kdtree()
        gaussian_tree = cKDTree(points.astype(np.float64))
    rotation_matrices = quaternion_wxyz_to_rotation_matrices(quats)
    num_points = int(points.shape[0])
    num_views = len(configs)
    num_k = int(k_values.shape[0])
    preselect_hits = np.zeros((num_points, num_views), dtype=np.int32)
    positive_hits = np.zeros((num_points, num_views), dtype=np.int32)
    selected_hits = np.zeros(
        (num_k, num_points, num_views),
        dtype=np.int32,
    )
    best_rank = np.full((num_points, num_views), -1, dtype=np.int16)
    best_score = np.zeros((num_points, num_views), dtype=np.float32)
    best_score_ratio = np.zeros((num_points, num_views), dtype=np.float32)

    for view_index, (cfg, rendered_depth, rendered_acc) in enumerate(
        zip(configs, depth_arrays, acc_arrays)
    ):
        expected_shape = (cfg.image_height, cfg.image_width)
        mask = load_mask(cfg.mask_path, expected_shape)
        valid_mask = erode_mask(mask, mask_erode_iters)
        visible_mask = (
            valid_mask
            & np.isfinite(rendered_depth)
            & (rendered_depth > 0.0)
            & np.isfinite(rendered_acc)
            & (rendered_acc >= float(pixel_render_acc_min))
        )
        ys = np.arange(
            1,
            cfg.image_height - 1,
            int(pixel_sample_stride),
            dtype=np.int32,
        )
        xs = np.arange(
            1,
            cfg.image_width - 1,
            int(pixel_sample_stride),
            dtype=np.int32,
        )
        if ys.size == 0 or xs.size == 0:
            continue
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        flat_x = xx.reshape(-1)
        flat_y = yy.reshape(-1)
        keep = visible_mask[flat_y, flat_x]
        flat_x = flat_x[keep]
        flat_y = flat_y[keep]
        if flat_x.size == 0:
            continue
        depths = rendered_depth[flat_y, flat_x].astype(np.float32)
        surface_pixels = np.stack(
            [flat_x.astype(np.float32), flat_y.astype(np.float32)],
            axis=1,
        )
        surface_points = unproject_pixels(
            surface_pixels,
            depths,
            cfg.K,
            cfg.world_to_camera,
        )
        finite_surface = (
            np.isfinite(surface_points).all(axis=1)
            & np.isfinite(depths)
            & (depths > 0.0)
        )
        surface_points = surface_points[finite_surface]
        if surface_points.shape[0] == 0:
            continue
        preselect_k = min(int(pixel_preselect_k), num_points)
        _, candidate_rows = gaussian_tree.query(
            surface_points.astype(np.float64),
            k=preselect_k,
        )
        if preselect_k == 1:
            candidate_rows = candidate_rows[:, None]
        _, point_camera_z = project_points(
            points,
            cfg.K,
            cfg.world_to_camera,
        )

        for surface_point, candidates in zip(surface_points, candidate_rows):
            raw_indices = np.asarray(candidates, dtype=np.int64).reshape(-1)
            raw_indices = raw_indices[
                (raw_indices >= 0) & (raw_indices < num_points)
            ]
            if raw_indices.size:
                np.add.at(
                    preselect_hits[:, view_index],
                    raw_indices,
                    1,
                )
            scored = score_pixel_gaussian_candidates(
                surface_point_world=surface_point,
                candidate_indices=raw_indices,
                points_world=points,
                gaussian_scales=scales,
                gaussian_rotmats=rotation_matrices,
                gaussian_opacities=opacities,
                point_camera_z=point_camera_z,
                min_contribution=pixel_min_contribution,
            )
            if scored.gaussian_indices.size == 0:
                continue
            ranks = np.arange(
                1,
                scored.gaussian_indices.shape[0] + 1,
                dtype=np.int16,
            )
            positive = scored.positive_camera_z_mask
            positive_indices = scored.gaussian_indices[positive]
            if positive_indices.size:
                positive_ranks = ranks[positive]
                positive_scores = scored.contribution_scores[positive]
                np.add.at(
                    positive_hits[:, view_index],
                    positive_indices,
                    1,
                )
                previous_rank = best_rank[positive_indices, view_index]
                replace_rank = (previous_rank < 0) | (
                    positive_ranks < previous_rank
                )
                best_rank[
                    positive_indices[replace_rank],
                    view_index,
                ] = positive_ranks[replace_rank]
                np.maximum.at(
                    best_score[:, view_index],
                    positive_indices,
                    positive_scores.astype(np.float32),
                )
                winner_score = float(scored.contribution_scores[0])
                ratios = (
                    positive_scores / winner_score
                    if winner_score > 0.0
                    else np.zeros(positive_scores.shape, dtype=np.float64)
                )
                np.maximum.at(
                    best_score_ratio[:, view_index],
                    positive_indices,
                    ratios.astype(np.float32),
                )
            for k_index, k_value in enumerate(k_values.tolist()):
                top_indices = scored.gaussian_indices[: int(k_value)]
                top_scores = scored.contribution_scores[: int(k_value)]
                top_positive = scored.positive_camera_z_mask[: int(k_value)]
                top_indices = top_indices[top_positive]
                top_scores = top_scores[top_positive]
                if top_indices.size == 0:
                    continue
                denominator = float(np.sum(top_scores))
                if denominator <= 0.0 or not np.isfinite(denominator):
                    continue
                np.add.at(
                    selected_hits[k_index, :, view_index],
                    top_indices,
                    1,
                )

    preselect_view_count = np.count_nonzero(preselect_hits > 0, axis=1).astype(
        np.int8
    )
    positive_view_count = np.count_nonzero(positive_hits > 0, axis=1).astype(
        np.int8
    )
    selected_view_count = np.count_nonzero(selected_hits > 0, axis=2).astype(
        np.int8
    )
    selected_sample_count = selected_hits.sum(axis=2, dtype=np.int64).astype(
        np.int32
    )
    category_by_k = np.stack(
        [
            observation_coverage_categories(
                preselect_view_count,
                positive_view_count,
                selected_view_count[k_index],
            )
            for k_index in range(num_k)
        ],
        axis=0,
    )
    category_count_by_k = np.stack(
        [
            np.bincount(
                category_by_k[k_index],
                minlength=len(OBSERVATION_COVERAGE_CATEGORY_NAMES),
            )
            for k_index in range(num_k)
        ],
        axis=0,
    ).astype(np.int64)
    return ObservationCoverageDiagnostics(
        points_world=points,
        view_ids=np.asarray([cfg.view_id for cfg in configs]),
        k_values=k_values,
        preselect_hit_count_by_view=preselect_hits,
        positive_hit_count_by_view=positive_hits,
        selected_hit_count_by_k_view=selected_hits,
        best_positive_rank_by_view=best_rank,
        best_positive_score_by_view=best_score,
        best_positive_score_ratio_by_view=best_score_ratio,
        preselect_view_count=preselect_view_count,
        positive_view_count=positive_view_count,
        selected_view_count_by_k=selected_view_count,
        selected_sample_count_by_k=selected_sample_count,
        category_by_k=category_by_k,
        category_count_by_k=category_count_by_k,
        preselect_view_count_histogram=_view_count_histogram(
            preselect_view_count,
            num_views,
        ),
        positive_view_count_histogram=_view_count_histogram(
            positive_view_count,
            num_views,
        ),
        selected_view_count_histogram_by_k=np.stack(
            [
                _view_count_histogram(selected_view_count[k], num_views)
                for k in range(num_k)
            ],
            axis=0,
        ),
    )


def validate_reference_observation_replay(
    diagnostics: ObservationCoverageDiagnostics,
    reference: dict[str, np.ndarray],
    baseline_k: int,
    reference_path: Path,
) -> int:
    required = {
        "obs_count_per_point",
        "obs_sample_count_per_point",
        "observations_per_view",
    }
    missing = sorted(required - set(reference))
    if missing:
        raise ValueError(
            f"{reference_path} missing replay-validation fields: {missing}"
        )
    matches = np.flatnonzero(diagnostics.k_values == int(baseline_k))
    if matches.shape[0] != 1:
        raise ValueError(
            f"Coverage K values must contain baseline pixel_candidate_k={baseline_k}"
        )
    baseline_index = int(matches[0])
    expected_view_count = np.asarray(reference["obs_count_per_point"])
    expected_sample_count = np.asarray(reference["obs_sample_count_per_point"])
    if not np.array_equal(
        diagnostics.selected_view_count_by_k[baseline_index],
        expected_view_count,
    ):
        mismatch = int(
            np.count_nonzero(
                diagnostics.selected_view_count_by_k[baseline_index]
                != expected_view_count
            )
        )
        raise ValueError(
            f"Coverage replay disagrees with {reference_path} view counts for "
            f"{mismatch} Gaussians"
        )
    if not np.array_equal(
        diagnostics.selected_sample_count_by_k[baseline_index],
        expected_sample_count,
    ):
        mismatch = int(
            np.count_nonzero(
                diagnostics.selected_sample_count_by_k[baseline_index]
                != expected_sample_count
            )
        )
        raise ValueError(
            f"Coverage replay disagrees with {reference_path} sample counts for "
            f"{mismatch} Gaussians"
        )
    expected_per_view = np.asarray(reference["observations_per_view"])
    actual_per_view = diagnostics.selected_hit_count_by_k_view[
        baseline_index
    ].sum(axis=0, dtype=np.int64)
    if not np.array_equal(actual_per_view, expected_per_view):
        raise ValueError(
            f"Coverage replay disagrees with {reference_path} per-view totals"
        )
    return baseline_index


def write_observation_coverage_diagnostics(
    path: Path,
    diagnostics: ObservationCoverageDiagnostics,
    *,
    source_checkpoint: str,
    reference_observation_path: Path,
    baseline_k: int,
    baseline_k_index: int,
    mask_erode_iters: int,
    pixel_sample_stride: int,
    pixel_preselect_k: int,
    pixel_render_acc_min: float,
    pixel_min_contribution: float,
) -> Path:
    arrays = {
        "version": np.array(OBSERVATION_COVERAGE_VERSION, dtype=np.int32),
        "point_type": np.array("foreground_gaussian_observation_coverage"),
        "source_checkpoint": np.array(source_checkpoint),
        "reference_observation_path": np.array(
            str(reference_observation_path)
        ),
        "num_foreground_gaussians": np.array(
            diagnostics.points_world.shape[0],
            dtype=np.int64,
        ),
        "gaussian_indices": np.arange(
            diagnostics.points_world.shape[0],
            dtype=np.int64,
        ),
        "points_world": diagnostics.points_world.astype(np.float32),
        "view_ids": diagnostics.view_ids,
        "k_values": diagnostics.k_values.astype(np.int32),
        "baseline_k": np.array(baseline_k, dtype=np.int32),
        "baseline_k_index": np.array(baseline_k_index, dtype=np.int32),
        "category_names": np.asarray(OBSERVATION_COVERAGE_CATEGORY_NAMES),
        "preselect_hit_count_by_view": (
            diagnostics.preselect_hit_count_by_view.astype(np.int32)
        ),
        "positive_hit_count_by_view": (
            diagnostics.positive_hit_count_by_view.astype(np.int32)
        ),
        "selected_hit_count_by_k_view": (
            diagnostics.selected_hit_count_by_k_view.astype(np.int32)
        ),
        "best_positive_rank_by_view": (
            diagnostics.best_positive_rank_by_view.astype(np.int16)
        ),
        "best_positive_score_by_view": (
            diagnostics.best_positive_score_by_view.astype(np.float32)
        ),
        "best_positive_score_ratio_by_view": (
            diagnostics.best_positive_score_ratio_by_view.astype(np.float32)
        ),
        "preselect_view_count": diagnostics.preselect_view_count.astype(np.int8),
        "positive_view_count": diagnostics.positive_view_count.astype(np.int8),
        "selected_view_count_by_k": (
            diagnostics.selected_view_count_by_k.astype(np.int8)
        ),
        "selected_sample_count_by_k": (
            diagnostics.selected_sample_count_by_k.astype(np.int32)
        ),
        "category_by_k": diagnostics.category_by_k.astype(np.int8),
        "category_count_by_k": diagnostics.category_count_by_k.astype(np.int64),
        "preselect_view_count_histogram": (
            diagnostics.preselect_view_count_histogram.astype(np.int64)
        ),
        "positive_view_count_histogram": (
            diagnostics.positive_view_count_histogram.astype(np.int64)
        ),
        "selected_view_count_histogram_by_k": (
            diagnostics.selected_view_count_histogram_by_k.astype(np.int64)
        ),
        "mask_erode_iters": np.array(mask_erode_iters, dtype=np.int32),
        "pixel_sample_stride": np.array(pixel_sample_stride, dtype=np.int32),
        "pixel_preselect_k": np.array(pixel_preselect_k, dtype=np.int32),
        "pixel_render_acc_min": np.array(
            pixel_render_acc_min,
            dtype=np.float32,
        ),
        "pixel_min_contribution": np.array(
            pixel_min_contribution,
            dtype=np.float32,
        ),
        "candidate_method": np.array("rendered_depth_gaussian_contribution"),
        "replay_validation": np.array("exact_reference_counts"),
    }
    return save_npz_compressed_atomic(path, arrays)
