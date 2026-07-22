"""Batch solve 3D modal fields directly on foreground 3DGS Gaussians."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any, Mapping, SupportsFloat

import numpy as np

from modal_surface.checkpoint_render_inputs import (
    load_fg_means_from_checkpoint,
    load_fg_pixel_candidate_inputs_from_checkpoint,
)
from modal_surface.gaussian_observations import build_gaussian_observation_graph
from modal_surface.gaussian_motion_fill import (
    MOTION_FILL_EPSILON,
    MOTION_FILL_LSMR_ATOL,
    MOTION_FILL_LSMR_BTOL,
    MOTION_FILL_LSMR_CONLIM,
    MOTION_FILL_NULLSPACE_RTOL,
    MOTION_FILL_NUMERICAL_RANK_POLICY,
    MOTION_FILL_OBSERVATION_DRIFT_RTOL,
    MOTION_FILL_METHOD,
    MOTION_FILL_ROLE_NAMES,
    MOTION_FILL_VERSION,
    RIGID_SEQUENTIAL_MOTION_FILL_METHOD,
    RIGID_SINGLE_VIEW_PARTIAL_FILL_METHOD,
    RIGID_SINGLE_VIEW_PARTIAL_FILL_POLICY,
    GaussianMotionFillResult,
    RigidSeedMotionFillResult,
    apply_gaussian_motion_fill,
    apply_sequential_rigid_motion_fill,
    apply_single_view_component_partial_fill,
    load_motion_fill_graph,
    write_motion_fill_diagnostics,
    write_motion_fill_graph,
)
from modal_surface.incremental_manifest import (
    IncrementalManifestBase,
    load_incremental_manifest_base,
    validate_incremental_observation_topology,
)
from modal_surface.io import (
    load_view_config,
    load_modal_freqs,
    save_npz_compressed_atomic,
)
from modal_surface.motion_fill import KnnGraph, build_knn_graph, query_knn_candidates
from modal_surface.observed_structure_graph import (
    LoadedObservedStructureGraph,
    load_observed_structure_graph,
    validate_observed_structure_graph_sources,
)
from modal_surface.optimization_staged import (
    ANCHOR_CONDITION_MAX,
    POINT_STATUS_NAMES,
    AlphaSyncResult,
    PreparedObservations,
    StagedSolveResult,
    compute_prediction_and_residuals,
    enforce_alpha_failure,
    optimize_multi_view_staged,
    prepare_observations,
    solve_alpha_sync,
)
from modal_surface.optimization_visualization import write_solve_visualizations
from modal_surface.optimization_visualization import write_prepared_solve_visualizations
from modal_surface.rigid_component_solver import (
    RigidComponentSeedSelectionConfig,
    RigidComponentSeedSelectionResult,
    RigidComponentSolveResult,
    RigidComponentSolverConfig,
    select_trusted_rigid_component_seeds,
    solve_rigid_components,
)
from modal_surface.solver_cli import (
    RIGID_COMPONENT_RCOND_DEFAULT,
    RIGID_MOTION_FILL_STAGE_DEFAULT,
    RIGID_SEED_MAX_FINITE_DRIFT_DEFAULT,
    RIGID_SEED_MIN_SECONDARY_VIEW_NODE_RATIO_DEFAULT,
    RIGID_SEED_MIN_SINGULAR_RATIO_DEFAULT,
    RIGID_SEED_MIN_VALID_VIEWS_DEFAULT,
    RIGID_SINGLE_VIEW_OBSERVABLE_RATIO_DEFAULT,
    RIGID_SINGLE_VIEW_RAY_DIRECTION_MIN_FRACTION_DEFAULT,
    STAGED_ANCHOR_RESIDUAL_MAX_DEFAULT,
    STAGED_ANCHOR_SVD_RATIO_DEFAULT,
    add_solve_method_arguments,
    add_staged_solver_arguments,
    rigid_component_manifest_parameters,
    staged_solver_config,
    staged_solver_manifest_parameters,
)


_DEFAULT_MOTION_FILL_K = 8
_DEFAULT_MOTION_FILL_MAX_ANCHOR_HOPS = 8


def parse_mode_indices(raw: str, num_modes: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(num_modes))
    out: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        idx = int(text)
        if idx < 0 or idx >= num_modes:
            raise ValueError(f"mode index {idx} is outside [0,{num_modes - 1}].")
        if idx in out:
            raise ValueError(f"--mode-indices contains duplicate mode index {idx}.")
        out.append(idx)
    if not out:
        raise ValueError("--mode-indices must contain at least one index, or 'all'.")
    return out


def freq_slug(freq_hz: float) -> str:
    text = f"{freq_hz:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def latent_stats(
    staged: StagedSolveResult,
    motion_fill: GaussianMotionFillResult | None,
    observations: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    observable = staged.observable
    alpha = staged.alpha
    if motion_fill is None:
        obs_residual = staged.obs_residual
        obs_residual_valid = staged.obs_residual_valid_mask
        point_residual = staged.point_residual
        point_residual_valid = staged.point_residual_valid_mask
    else:
        obs_residual = motion_fill.obs_residual
        obs_residual_valid = motion_fill.obs_residual_valid_mask
        point_residual = motion_fill.point_residual
        point_residual_valid = motion_fill.point_residual_valid_mask
    obs_residual = np.asarray(obs_residual)[np.asarray(obs_residual_valid, dtype=bool)]
    point_residual = np.asarray(point_residual)[np.asarray(point_residual_valid, dtype=bool)]
    return {
        "num_points": int(staged.prepared.points.shape[0]),
        "num_observations": int(staged.prepared.obs_point_index.shape[0]),
        "obs_residual_median": _finite_percentile(obs_residual, 50),
        "obs_residual_p90": _finite_percentile(obs_residual, 90),
        "point_residual_median": _finite_percentile(point_residual, 50),
        "point_residual_p90": _finite_percentile(point_residual, 90),
        "observations_per_view": observations["observations_per_view"].astype(int).tolist(),
        "solver_method": "staged_overlap_observable",
        "alpha_identifiable_count": int(alpha.identifiable_mask.sum()),
        "alpha_optimizer_success": bool(alpha.optimizer_success),
        "anchor_count": int(observable.anchor_mask.sum()),
        "partial_unresolved_count": int(observable.partial_mask.sum()),
        "rejected_count": int(observable.rejected_mask.sum()),
        "alpha_unresolved_point_count": int(observable.alpha_unresolved_mask.sum()),
        "no_usable_observation_point_count": int(
            observable.no_usable_observation_mask.sum()
        ),
    }


def json_float(value: SupportsFloat) -> float | None:
    result = float(value)
    if not np.isfinite(result):
        return None
    return result


def _alpha_view_frequencies(prepared: PreparedObservations) -> np.ndarray:
    if "view_freqs_hz" in prepared.arrays:
        frequencies = np.asarray(prepared.arrays["view_freqs_hz"], dtype=np.float32)
    else:
        frequencies = np.full(
            (prepared.num_views,),
            float(np.asarray(prepared.arrays["freq_hz"]).item()),
            dtype=np.float32,
        )
    if frequencies.shape != (prepared.num_views,):
        raise ValueError(
            "Observation view_freqs_hz does not match the prepared view count."
        )
    return frequencies


def _unidentifiable_observed_view_indices(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
) -> np.ndarray:
    observed = np.zeros((prepared.num_views,), dtype=bool)
    positive_rows = prepared.obs_weights > 0.0
    observed[np.unique(prepared.obs_view_index[positive_rows])] = True
    return np.where(observed & ~alpha.identifiable_mask)[0].astype(np.int64)


def alpha_by_view_diagnostics(
    staged: StagedSolveResult,
) -> list[dict[str, Any]]:
    return _alpha_by_view_diagnostics(
        staged.prepared,
        staged.alpha,
        staged.alpha_view_freqs_hz,
    )


def _alpha_by_view_diagnostics(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    alpha_view_freqs_hz: np.ndarray,
) -> list[dict[str, Any]]:
    view_ids = [str(v) for v in prepared.view_ids.tolist()]
    alphas = alpha.alphas.astype(np.complex64).reshape(-1)
    freqs_hz = np.asarray(alpha_view_freqs_hz, dtype=np.float32).reshape(-1)
    if alphas.shape[0] != len(view_ids):
        raise ValueError("Alpha count does not match view_ids.")
    if freqs_hz.shape[0] != len(view_ids):
        raise ValueError("alpha_view_freqs_hz length does not match view_ids.")
    identifiable = alpha.identifiable_mask.astype(bool).reshape(-1)
    reasons = alpha.exclusion_reason.astype(str).reshape(-1)
    phase_std = alpha.phase_std.astype(np.float32).reshape(-1)
    log_gain_std = alpha.log_gain_std.astype(np.float32).reshape(-1)
    gain_std = (np.abs(alphas) * log_gain_std).astype(np.float32)
    gain_bound_active = alpha.gain_bound_active_mask.astype(bool).reshape(-1)
    staged_diagnostics = {
        "alpha_identifiable_mask": identifiable,
        "alpha_exclusion_reason": reasons,
        "alpha_phase_std": phase_std,
        "alpha_gain_std": gain_std,
        "alpha_log_gain_std": log_gain_std,
        "alpha_gain_bound_active_mask": gain_bound_active,
    }
    invalid = [name for name, values in staged_diagnostics.items() if values.shape[0] != len(view_ids)]
    if invalid:
        raise ValueError(f"Staged alpha diagnostics have invalid lengths: {invalid}.")
    return [
        {
            "view_id": view_id,
            "freq_hz": json_float(freqs_hz[idx]),
            "real": json_float(np.real(alphas[idx])),
            "imag": json_float(np.imag(alphas[idx])),
            "abs": json_float(np.abs(alphas[idx])),
            "phase_rad": json_float(np.angle(alphas[idx])),
            "identifiable": bool(identifiable[idx]),
            "reason": str(reasons[idx]),
            "phase_std": json_float(phase_std[idx]),
            "gain_std": json_float(gain_std[idx]),
            "log_gain_std": json_float(log_gain_std[idx]),
            "gain_bound_active": bool(gain_bound_active[idx]),
        }
        for idx, view_id in enumerate(view_ids)
    ]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-ckpt", required=True, help="Static 3DGS checkpoint containing foreground Gaussian centers.")
    parser.add_argument("--view-config", action="append", required=True, help="View JSON config path. Repeat per view.")
    parser.add_argument("--modal-npz", action="append", required=True, help="modal_analysis.npz path. Repeat per view.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for observations, diagnostics, latents, vis, and manifest.",
    )
    parser.add_argument("--mode-indices", default="all", help="Comma-separated zero-based mode indices, or 'all'.")
    parser.add_argument(
        "--base-manifest",
        default=None,
        help=(
            "Existing contiguous-prefix Gaussian modal manifest to reuse. Only the "
            "next mode slots named by --mode-indices are solved; the output manifest "
            "references the old artifacts and appends the new modes. Available only "
            "with --solve-method=staged."
        ),
    )
    parser.add_argument("--pixel-sample-stride", type=int, default=4, help="Pixel grid stride for pixel-candidates sampling.")
    parser.add_argument("--pixel-candidate-k", type=int, default=4, help="Number of top contribution Gaussians supervised by each sampled pixel.")
    parser.add_argument("--pixel-preselect-k", type=int, default=32, help="Number of 3D nearest Gaussians scored before top-k contribution selection.")
    parser.add_argument("--pixel-render-acc-min", type=float, default=0.05, help="Minimum rendered foreground alpha for sampled modal pixels.")
    parser.add_argument("--pixel-min-contribution", type=float, default=1e-12, help="Minimum unnormalized Gaussian contribution retained for a sampled modal pixel.")
    parser.add_argument("--mask-erode-iters", type=int, default=1, help="3x3 modal mask erosion iterations.")
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1, help="Allowed selected frequency mismatch.")
    parser.add_argument(
        "--motion-fill",
        action="store_true",
        help=(
            "Run the existing staged nullspace fill, or propagate rigid-component "
            "seeds to every non-seed Gaussian in rigid-components mode."
        ),
    )
    parser.add_argument(
        "--motion-fill-k",
        type=int,
        default=_DEFAULT_MOTION_FILL_K,
        help="Spatial KNN neighborhood size used when --motion-fill is enabled (default: 8).",
    )
    parser.add_argument(
        "--motion-fill-max-distance",
        type=float,
        default=None,
        help="Required maximum KNN edge distance in scene units when --motion-fill is enabled.",
    )
    parser.add_argument(
        "--motion-fill-max-anchor-hops",
        type=int,
        default=_DEFAULT_MOTION_FILL_MAX_ANCHOR_HOPS,
        help=(
            "Maximum KNN hop distance from a fixed anchor included in sequential "
            "pointwise motion fill (default: 8)."
        ),
    )
    add_solve_method_arguments(parser)
    add_staged_solver_arguments(parser)


def _validate_solve_method_arguments(args: argparse.Namespace) -> None:
    graph_paths = list(args.rigid_component_graph)
    if args.solve_method == "rigid-components" and args.base_manifest is not None:
        raise ValueError(
            "--base-manifest is currently supported only with --solve-method=staged."
        )
    rcond = float(args.rigid_component_rcond)
    min_valid_views = int(args.rigid_seed_min_valid_views)
    min_secondary_view_node_ratio = float(
        args.rigid_seed_min_secondary_view_node_ratio
    )
    min_singular_ratio = float(args.rigid_seed_min_singular_ratio)
    max_finite_drift = float(args.rigid_seed_max_finite_drift)
    motion_fill_stage = str(args.rigid_motion_fill_stage)
    observable_ratio = float(args.rigid_single_view_observable_ratio)
    ray_direction_fraction = float(
        args.rigid_single_view_ray_direction_min_fraction
    )
    if not np.isfinite(rcond) or not (0.0 < rcond < 1.0):
        raise ValueError("--rigid-component-rcond must be finite and lie in (0,1).")
    if min_valid_views <= 0:
        raise ValueError("--rigid-seed-min-valid-views must be positive.")
    if (
        not np.isfinite(min_secondary_view_node_ratio)
        or not 0.0 <= min_secondary_view_node_ratio <= 1.0
    ):
        raise ValueError(
            "--rigid-seed-min-secondary-view-node-ratio must be finite and "
            "lie in [0,1]."
        )
    if not np.isfinite(min_singular_ratio) or not (
        0.0 <= min_singular_ratio <= 1.0
    ):
        raise ValueError(
            "--rigid-seed-min-singular-ratio must be finite and lie in [0,1]."
        )
    if not np.isfinite(max_finite_drift) or max_finite_drift < 0.0:
        raise ValueError(
            "--rigid-seed-max-finite-drift must be finite and non-negative."
        )
    if not np.isfinite(observable_ratio) or not 0.0 < observable_ratio <= 1.0:
        raise ValueError(
            "--rigid-single-view-observable-ratio must be finite and lie in (0,1]."
        )
    if (
        not np.isfinite(ray_direction_fraction)
        or not 0.0 <= ray_direction_fraction <= 1.0
    ):
        raise ValueError(
            "--rigid-single-view-ray-direction-min-fraction must be finite and "
            "lie in [0,1]."
        )
    if motion_fill_stage != RIGID_MOTION_FILL_STAGE_DEFAULT and not bool(
        args.motion_fill
    ):
        raise ValueError(
            "--rigid-motion-fill-stage=single-view-components requires --motion-fill."
        )
    if args.solve_method == "staged":
        if graph_paths:
            raise ValueError(
                "--rigid-component-graph requires --solve-method=rigid-components."
            )
        if rcond != RIGID_COMPONENT_RCOND_DEFAULT:
            raise ValueError(
                "A custom --rigid-component-rcond requires "
                "--solve-method=rigid-components."
            )
        if min_valid_views != RIGID_SEED_MIN_VALID_VIEWS_DEFAULT:
            raise ValueError(
                "A custom --rigid-seed-min-valid-views requires "
                "--solve-method=rigid-components."
            )
        if (
            min_secondary_view_node_ratio
            != RIGID_SEED_MIN_SECONDARY_VIEW_NODE_RATIO_DEFAULT
        ):
            raise ValueError(
                "A custom --rigid-seed-min-secondary-view-node-ratio requires "
                "--solve-method=rigid-components."
            )
        if min_singular_ratio != RIGID_SEED_MIN_SINGULAR_RATIO_DEFAULT:
            raise ValueError(
                "A custom --rigid-seed-min-singular-ratio requires "
                "--solve-method=rigid-components."
            )
        if max_finite_drift != RIGID_SEED_MAX_FINITE_DRIFT_DEFAULT:
            raise ValueError(
                "A custom --rigid-seed-max-finite-drift requires "
                "--solve-method=rigid-components."
            )
        if motion_fill_stage != RIGID_MOTION_FILL_STAGE_DEFAULT:
            raise ValueError(
                "A custom --rigid-motion-fill-stage requires "
                "--solve-method=rigid-components."
            )
        if observable_ratio != RIGID_SINGLE_VIEW_OBSERVABLE_RATIO_DEFAULT:
            raise ValueError(
                "A custom --rigid-single-view-observable-ratio requires "
                "--solve-method=rigid-components."
            )
        if (
            ray_direction_fraction
            != RIGID_SINGLE_VIEW_RAY_DIRECTION_MIN_FRACTION_DEFAULT
        ):
            raise ValueError(
                "A custom --rigid-single-view-ray-direction-min-fraction requires "
                "--solve-method=rigid-components."
            )
        return
    if not graph_paths:
        raise ValueError(
            "--solve-method=rigid-components requires --rigid-component-graph."
        )
    if float(args.anchor_svd_ratio_min) != STAGED_ANCHOR_SVD_RATIO_DEFAULT:
        raise ValueError(
            "--anchor-svd-ratio-min is unavailable for rigid component solves; "
            "component rank is diagnostic only."
        )
    if float(args.anchor_residual_max) != STAGED_ANCHOR_RESIDUAL_MAX_DEFAULT:
        raise ValueError(
            "--anchor-residual-max is unavailable for rigid component solves; "
            "component residual is diagnostic only."
        )


def _copy_file_atomic(source: Path, destination: Path) -> Path:
    source_resolved = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.resolve() == source_resolved:
        return destination
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source_resolved, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
        assert temporary is not None
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(str(path), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _scalar_value(
    arrays: Mapping[str, np.ndarray],
    key: str,
    path: Path,
) -> object:
    if key not in arrays:
        raise ValueError(f"{path} is missing required field {key}.")
    value = np.asarray(arrays[key])
    if value.shape != ():
        raise ValueError(f"{path} field {key} must be scalar, got {value.shape}.")
    return value.item()


def _load_rigid_component_graphs(
    graph_paths: list[str],
    mode_indices: list[int],
) -> dict[int, LoadedObservedStructureGraph]:
    loaded_by_mode: dict[int, LoadedObservedStructureGraph] = {}
    for raw_path in graph_paths:
        loaded = load_observed_structure_graph(raw_path)
        if loaded.mode_index in loaded_by_mode:
            previous = loaded_by_mode[loaded.mode_index].graph_path
            raise ValueError(
                "Duplicate rigid component graph for mode "
                f"{loaded.mode_index}: {previous} and {loaded.graph_path}."
            )
        loaded_by_mode[loaded.mode_index] = loaded
    requested = set(mode_indices)
    supplied = set(loaded_by_mode)
    missing = sorted(requested - supplied)
    extra = sorted(supplied - requested)
    if missing or extra:
        raise ValueError(
            "Rigid component graphs must match requested modes exactly: "
            f"missing={missing}, extra={extra}."
        )
    return loaded_by_mode


def _validate_rigid_source_observations(
    loaded_graph: LoadedObservedStructureGraph,
    observations: Mapping[str, np.ndarray],
    *,
    args: argparse.Namespace,
    view_config_paths: list[str],
    modal_npz_paths: list[str],
    freqs_per_view: list[np.ndarray],
    mode_index: int,
    reference_freq: float,
    foreground_points: np.ndarray,
) -> PreparedObservations:
    source_path = Path(loaded_graph.source_observation_path)
    required = {
        "points_world",
        "gaussian_indices",
        "point_type",
        "source_checkpoint",
        "source_view_configs",
        "source_modal_npzs",
        "view_ids",
        "view_freqs_hz",
        "freq_hz",
        "mode_index",
        "candidate_point_count",
        "preserved_all_points",
        "mask_erode_iters",
        "pixel_sample_stride",
        "pixel_candidate_k",
        "pixel_preselect_k",
        "pixel_render_acc_min",
        "pixel_min_contribution",
        "obs_point_index",
        "obs_view_index",
        "obs_pixels_xy",
        "obs_y",
        "obs_J",
        "obs_count_per_point",
        "obs_sample_count_per_point",
        "obs_contribution_weight",
        "obs_contribution_score",
        "obs_contribution_sum",
        "observations_per_view",
        "pixel_candidate_method",
        "view_image_width",
        "view_image_height",
    }
    missing = sorted(required - set(observations))
    if missing:
        raise ValueError(f"{source_path} missing rigid-solve fields: {missing}.")
    if str(_scalar_value(observations, "point_type", source_path)) != (
        "foreground_gaussian_center"
    ):
        raise ValueError(f"{source_path} point_type is incompatible.")
    if str(_scalar_value(observations, "source_checkpoint", source_path)) != str(
        args.input_ckpt
    ):
        raise ValueError(f"{source_path} source_checkpoint does not match --input-ckpt.")
    expected_view_configs = np.asarray(view_config_paths).astype(str)
    if not np.array_equal(
        np.asarray(observations["source_view_configs"]).astype(str),
        expected_view_configs,
    ):
        raise ValueError(
            f"{source_path} source_view_configs do not match --view-config arguments."
        )
    expected_modal_npzs = np.asarray(modal_npz_paths).astype(str)
    if not np.array_equal(
        np.asarray(observations["source_modal_npzs"]).astype(str),
        expected_modal_npzs,
    ):
        raise ValueError(
            f"{source_path} source_modal_npzs do not match --modal-npz arguments."
        )
    expected_view_ids = np.asarray(
        [load_view_config(path).view_id for path in view_config_paths]
    ).astype(str)
    observation_view_ids = np.asarray(observations["view_ids"]).astype(str)
    if not np.array_equal(observation_view_ids, expected_view_ids):
        raise ValueError(f"{source_path} view_ids do not match --view-config order.")
    num_views = int(observation_view_ids.shape[0])
    obs_view_index = np.asarray(observations["obs_view_index"])
    if obs_view_index.ndim != 1 or not np.issubdtype(
        obs_view_index.dtype, np.integer
    ):
        raise ValueError(f"{source_path} obs_view_index must be 1-D integers.")
    if np.any(obs_view_index < 0) or np.any(obs_view_index >= num_views):
        raise ValueError(f"{source_path} obs_view_index is out of range.")
    observations_per_view = np.asarray(observations["observations_per_view"])
    expected_observations_per_view = np.bincount(
        obs_view_index.astype(np.int64), minlength=num_views
    )
    if (
        observations_per_view.shape != (num_views,)
        or not np.issubdtype(observations_per_view.dtype, np.integer)
        or not np.array_equal(
            observations_per_view.astype(np.int64),
            expected_observations_per_view,
        )
    ):
        raise ValueError(f"{source_path} observations_per_view is inconsistent.")
    for key in ("view_image_width", "view_image_height"):
        values = np.asarray(observations[key])
        if (
            values.shape != (num_views,)
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values <= 0)
        ):
            raise ValueError(f"{source_path} {key} must be positive per-view integers.")
    if str(_scalar_value(observations, "pixel_candidate_method", source_path)) != (
        "rendered_depth_gaussian_contribution"
    ):
        raise ValueError(f"{source_path} pixel_candidate_method is incompatible.")
    num_observations = int(obs_view_index.shape[0])
    for key in (
        "obs_contribution_weight",
        "obs_contribution_score",
        "obs_contribution_sum",
    ):
        values = np.asarray(observations[key])
        if (
            values.shape != (num_observations,)
            or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values)
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
        ):
            raise ValueError(
                f"{source_path} {key} must be finite non-negative observation values."
            )
    if int(_scalar_value(observations, "mode_index", source_path)) != mode_index:
        raise ValueError(f"{source_path} mode_index does not match the requested mode.")
    observation_freq = float(_scalar_value(observations, "freq_hz", source_path))
    if not np.isclose(
        observation_freq,
        reference_freq,
        rtol=0.0,
        atol=float(args.freq_tolerance_hz),
    ):
        raise ValueError(
            f"{source_path} frequency does not match the requested modal frequency."
        )
    expected_view_freqs = np.asarray(
        [float(freqs[mode_index]) for freqs in freqs_per_view], dtype=np.float64
    )
    observation_view_freqs = np.asarray(
        observations["view_freqs_hz"], dtype=np.float64
    )
    if observation_view_freqs.shape != expected_view_freqs.shape or not np.allclose(
        observation_view_freqs,
        expected_view_freqs,
        rtol=0.0,
        atol=float(args.freq_tolerance_hz),
    ):
        raise ValueError(
            f"{source_path} view_freqs_hz do not match the selected modal frequencies."
        )
    points = np.asarray(observations["points_world"], dtype=np.float32)
    if points.shape != foreground_points.shape or not np.allclose(
        points,
        foreground_points,
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise ValueError(
            f"{source_path} foreground Gaussian centers do not match the checkpoint."
        )
    indices = np.asarray(observations["gaussian_indices"])
    expected_indices = np.arange(points.shape[0], dtype=np.int64)
    if (
        indices.shape != expected_indices.shape
        or not np.issubdtype(indices.dtype, np.integer)
        or not np.array_equal(indices, expected_indices)
    ):
        raise ValueError(
            f"{source_path} gaussian_indices must preserve checkpoint order."
        )
    if int(_scalar_value(observations, "candidate_point_count", source_path)) != int(
        points.shape[0]
    ):
        raise ValueError(f"{source_path} candidate_point_count is inconsistent.")
    if not bool(_scalar_value(observations, "preserved_all_points", source_path)):
        raise ValueError(f"{source_path} must preserve all foreground Gaussians.")

    integer_parameters = {
        "mask_erode_iters": int(args.mask_erode_iters),
        "pixel_sample_stride": int(args.pixel_sample_stride),
        "pixel_candidate_k": int(args.pixel_candidate_k),
        "pixel_preselect_k": int(args.pixel_preselect_k),
    }
    for key, expected in integer_parameters.items():
        if int(_scalar_value(observations, key, source_path)) != expected:
            raise ValueError(
                f"{source_path} {key} does not match the current CLI value."
            )
    float_parameters = {
        "pixel_render_acc_min": float(args.pixel_render_acc_min),
        "pixel_min_contribution": float(args.pixel_min_contribution),
    }
    for key, expected in float_parameters.items():
        value = float(_scalar_value(observations, key, source_path))
        tolerance = max(abs(expected), 1.0e-12) * 1.0e-6
        if not np.isclose(value, expected, rtol=1.0e-6, atol=tolerance):
            raise ValueError(
                f"{source_path} {key} does not match the current CLI value."
            )

    validate_observed_structure_graph_sources(
        loaded_graph,
        points_world=points,
        gaussian_indices=indices,
        source_checkpoint=str(args.input_ckpt),
        source_observation_path=loaded_graph.source_observation_path,
        mode_index=mode_index,
        freq_hz=observation_freq,
        view_ids=observation_view_ids,
        obs_point_index=observations["obs_point_index"],
        obs_view_index=observations["obs_view_index"],
        obs_weights=observations["obs_contribution_weight"],
        freq_tolerance_hz=float(args.freq_tolerance_hz),
    )
    return prepare_observations(observations)


def _validate_motion_fill_arguments(
    args: argparse.Namespace,
    num_points: int | None = None,
) -> None:
    if not bool(args.motion_fill):
        if args.motion_fill_max_distance is not None:
            raise ValueError("--motion-fill-max-distance requires --motion-fill.")
        if args.motion_fill_k != _DEFAULT_MOTION_FILL_K:
            raise ValueError("A custom --motion-fill-k requires --motion-fill.")
        if (
            args.motion_fill_max_anchor_hops
            != _DEFAULT_MOTION_FILL_MAX_ANCHOR_HOPS
        ):
            raise ValueError(
                "A custom --motion-fill-max-anchor-hops requires --motion-fill."
            )
        return
    if args.motion_fill_max_distance is None:
        raise ValueError("--motion-fill requires --motion-fill-max-distance in scene units.")
    if (
        isinstance(args.motion_fill_k, (bool, np.bool_))
        or not isinstance(args.motion_fill_k, (int, np.integer))
        or int(args.motion_fill_k) <= 0
    ):
        raise ValueError("--motion-fill-k must be a positive integer.")
    max_distance = float(args.motion_fill_max_distance)
    if not np.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("--motion-fill-max-distance must be finite and positive.")
    if (
        isinstance(args.motion_fill_max_anchor_hops, (bool, np.bool_))
        or not isinstance(args.motion_fill_max_anchor_hops, (int, np.integer))
        or int(args.motion_fill_max_anchor_hops) <= 0
    ):
        raise ValueError("--motion-fill-max-anchor-hops must be a positive integer.")
    if num_points is not None and int(args.motion_fill_k) >= int(num_points):
        raise ValueError(
            f"--motion-fill-k must be smaller than the foreground Gaussian count ({num_points})."
        )


def _manifest_parameters(
    args: argparse.Namespace,
    motion_fill_graph_path: str | Path | None = None,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "mask_erode_iters": int(args.mask_erode_iters),
        "pixel_sample_stride": int(args.pixel_sample_stride),
        "pixel_candidate_k": int(args.pixel_candidate_k),
        "pixel_preselect_k": int(args.pixel_preselect_k),
        "pixel_render_acc_min": float(args.pixel_render_acc_min),
        "pixel_min_contribution": float(args.pixel_min_contribution),
        "freq_tolerance_hz": float(args.freq_tolerance_hz),
        "alpha_model": "per_view_per_mode",
        "alpha_reference_view_index": 0,
        "motion_fill_enabled": bool(args.motion_fill),
        **staged_solver_manifest_parameters(args),
    }
    if bool(args.motion_fill):
        parameters.update(
            {
                "motion_fill_method": MOTION_FILL_METHOD,
                "motion_fill_k": int(args.motion_fill_k),
                "motion_fill_max_distance": float(args.motion_fill_max_distance),
                "motion_fill_epsilon": MOTION_FILL_EPSILON,
                "motion_fill_nullspace_operator_rtol": MOTION_FILL_NULLSPACE_RTOL,
                "motion_fill_observation_drift_rtol": MOTION_FILL_OBSERVATION_DRIFT_RTOL,
                "motion_fill_excluded_policy": "retain_observable_exclude_from_graph",
            }
        )
        if motion_fill_graph_path is not None:
            parameters["motion_fill_graph_path"] = str(motion_fill_graph_path)
    return parameters


def _gaussian_latent_stats(
    staged: StagedSolveResult,
    motion_fill: GaussianMotionFillResult | None,
    observations: Mapping[str, np.ndarray],
    num_fg: int,
) -> dict[str, Any]:
    stats = latent_stats(staged, motion_fill, observations)
    prepared = staged.prepared
    observable = staged.observable
    obs_count = prepared.obs_count_per_point.astype(np.int32)
    obs_sample_count = prepared.obs_sample_count_per_point.astype(np.int32)
    if "gaussian_indices" not in prepared.arrays:
        raise ValueError("Gaussian observations are missing gaussian_indices.")
    gaussian_indices = prepared.arrays["gaussian_indices"].astype(np.int32)
    stats.update(
        {
            "num_foreground_gaussians": int(num_fg),
            "num_output_points": int(prepared.points.shape[0]),
            "unobserved_output_count": int((obs_count == 0).sum()),
            "obs_count_p50": json_float(np.percentile(obs_count, 50)),
            "obs_count_p90": json_float(np.percentile(obs_count, 90)),
            "obs_sample_count_p50": json_float(np.percentile(obs_sample_count, 50)),
            "obs_sample_count_p90": json_float(np.percentile(obs_sample_count, 90)),
            "gaussian_indices_contiguous": bool(np.array_equal(gaussian_indices, np.arange(gaussian_indices.shape[0], dtype=np.int32))),
            "preserved_all_points": bool(np.asarray(observations["preserved_all_points"]).item()),
            "pixel_candidate_method": str(np.asarray(observations["pixel_candidate_method"]).item()),
        }
    )
    if motion_fill is not None:
        completion = motion_fill.motion.completion_mask.astype(bool)
        partial = observable.partial_mask.astype(bool)
        unobserved = observable.unobserved_mask.astype(bool)
        staged_solver_method = "staged_overlap_observable"
        motion_fill_method = MOTION_FILL_METHOD
        stats.update(
            {
                "staged_solver_method": staged_solver_method,
                "motion_fill_method": motion_fill_method,
                "effective_field_method": f"{staged_solver_method}+{motion_fill_method}",
                "partial_staged_count": int(partial.sum()),
                "partial_unresolved_count": int((partial & ~completion).sum()),
                "completed_observed_count": int((partial & completion).sum()),
                "unobserved_unresolved_count": int((unobserved & ~completion).sum()),
                "completed_unobserved_count": int((unobserved & completion).sum()),
            }
        )
    contribution_weight = observations["obs_contribution_weight"].astype(np.float32)
    contribution_score = observations["obs_contribution_score"].astype(np.float32)
    contribution_sum = observations["obs_contribution_sum"].astype(np.float32)
    stats.update(
        {
            "contribution_weight_p50": json_float(np.percentile(contribution_weight, 50)),
            "contribution_weight_p90": json_float(np.percentile(contribution_weight, 90)),
            "contribution_weight_max": json_float(contribution_weight.max()),
            "contribution_score_p50": json_float(np.percentile(contribution_score, 50)),
            "contribution_score_p90": json_float(np.percentile(contribution_score, 90)),
            "contribution_score_max": json_float(contribution_score.max()),
            "contribution_sum_p50": json_float(np.percentile(contribution_sum, 50)),
            "contribution_sum_p90": json_float(np.percentile(contribution_sum, 90)),
            "contribution_sum_max": json_float(contribution_sum.max()),
        }
    )
    return stats


def _rigid_gaussian_latent_stats(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    rigid: RigidComponentSolveResult,
    seed_selection: RigidComponentSeedSelectionResult,
    motion_fill: RigidSeedMotionFillResult | None,
    observations: Mapping[str, np.ndarray],
    num_fg: int,
) -> dict[str, Any]:
    if motion_fill is None:
        _, obs_residual, obs_valid, point_residual, point_valid = (
            compute_prediction_and_residuals(prepared, alpha, seed_selection.phi)
        )
    else:
        obs_residual = motion_fill.obs_residual
        obs_valid = motion_fill.obs_residual_valid_mask
        point_residual = motion_fill.point_residual
        point_valid = motion_fill.point_residual_valid_mask
    valid_obs_residual = np.asarray(obs_residual)[np.asarray(obs_valid, dtype=bool)]
    valid_point_residual = np.asarray(point_residual)[
        np.asarray(point_valid, dtype=bool)
    ]
    component_residual = rigid.component_normalized_weighted_residual
    component_drift = rigid.component_finite_drift_max
    model_first_order_relative_max = float(
        max(
            np.max(rigid.edge_model_first_order_relative_real, initial=0.0),
            np.max(rigid.edge_model_first_order_relative_imag, initial=0.0),
        )
    )
    persisted_first_order_relative_max = float(
        max(
            np.max(rigid.edge_first_order_relative_real, initial=0.0),
            np.max(rigid.edge_first_order_relative_imag, initial=0.0),
        )
    )
    quantization_bound_relative_max = float(
        max(
            np.max(
                rigid.edge_first_order_quantization_bound_relative_real,
                initial=0.0,
            ),
            np.max(
                rigid.edge_first_order_quantization_bound_relative_imag,
                initial=0.0,
            ),
        )
    )
    stats: dict[str, Any] = {
        "num_foreground_gaussians": int(num_fg),
        "num_output_points": int(prepared.points.shape[0]),
        "num_observations": int(prepared.obs_point_index.shape[0]),
        "observations_per_view": observations["observations_per_view"].astype(
            int
        ).tolist(),
        "solver_method": "rigid_component_twist",
        "rigidity_model": "complex_infinitesimal_se3",
        "alpha_identifiable_count": int(np.count_nonzero(alpha.identifiable_mask)),
        "alpha_optimizer_success": bool(alpha.optimizer_success),
        "observed_point_count": int(np.count_nonzero(rigid.observed_mask)),
        "rigid_seed_count": int(np.count_nonzero(rigid.rigid_seed_mask)),
        "trusted_rigid_seed_count": int(
            np.count_nonzero(seed_selection.trusted_rigid_seed_mask)
        ),
        "quarantined_rigid_seed_count": int(
            np.count_nonzero(
                rigid.rigid_seed_mask
                & ~seed_selection.trusted_rigid_seed_mask
            )
        ),
        "isolated_observed_count": int(
            np.count_nonzero(rigid.observed_mask & ~rigid.rigid_seed_mask)
        ),
        "fill_target_count": int(np.count_nonzero(rigid.fill_target_mask)),
        "effective_fill_target_count": int(
            np.count_nonzero(seed_selection.effective_fill_target_mask)
        ),
        "unobserved_point_count": int(
            np.count_nonzero(prepared.obs_count_per_point == 0)
        ),
        "rigid_component_count": int(rigid.num_components),
        "rank_deficient_component_count": int(
            np.count_nonzero(rigid.component_rank < 6)
        ),
        "trusted_rigid_component_count": int(
            np.count_nonzero(seed_selection.component_seed_retained_mask)
        ),
        "valid_view_rejected_component_count": int(
            np.count_nonzero(
                seed_selection.component_valid_view_rejected_mask
            )
        ),
        "view_support_downgraded_component_count": int(
            np.count_nonzero(
                rigid.component_distinct_valid_view_count
                > seed_selection.component_supported_valid_view_count
            )
        ),
        "singular_rejected_component_count": int(
            np.count_nonzero(seed_selection.component_singular_rejected_mask)
        ),
        "finite_drift_rejected_component_count": int(
            np.count_nonzero(
                seed_selection.component_finite_drift_rejected_mask
            )
        ),
        "largest_rigid_component_node_count": int(
            np.max(rigid.component_node_count, initial=0)
        ),
        "largest_rigid_component_edge_count": int(
            np.max(rigid.component_edge_count, initial=0)
        ),
        "component_residual_p50": _finite_percentile(component_residual, 50),
        "component_residual_p90": _finite_percentile(component_residual, 90),
        "component_residual_max": _finite_percentile(component_residual, 100),
        "component_finite_drift_p50": _finite_percentile(component_drift, 50),
        "component_finite_drift_p90": _finite_percentile(component_drift, 90),
        "component_finite_drift_max": _finite_percentile(component_drift, 100),
        "model_first_order_relative_max": model_first_order_relative_max,
        "persisted_first_order_relative_max": (
            persisted_first_order_relative_max
        ),
        "first_order_quantization_bound_relative_max": (
            quantization_bound_relative_max
        ),
        "obs_residual_median": _finite_percentile(valid_obs_residual, 50),
        "obs_residual_p90": _finite_percentile(valid_obs_residual, 90),
        "point_residual_median": _finite_percentile(valid_point_residual, 50),
        "point_residual_p90": _finite_percentile(valid_point_residual, 90),
        "gaussian_indices_contiguous": bool(
            np.array_equal(
                np.asarray(prepared.arrays["gaussian_indices"]),
                np.arange(prepared.points.shape[0], dtype=np.int32),
            )
        ),
        "preserved_all_points": bool(
            np.asarray(observations["preserved_all_points"]).item()
        ),
        "pixel_candidate_method": str(
            np.asarray(observations["pixel_candidate_method"]).item()
        ),
    }
    for name in (
        "obs_contribution_weight",
        "obs_contribution_score",
        "obs_contribution_sum",
    ):
        values = np.asarray(observations[name], dtype=np.float32)
        label = name.removeprefix("obs_")
        stats[f"{label}_p50"] = _finite_percentile(values, 50)
        stats[f"{label}_p90"] = _finite_percentile(values, 90)
        stats[f"{label}_max"] = _finite_percentile(values, 100)
    if motion_fill is not None:
        motion_fill_method = str(motion_fill.diagnostics["method"])
        stats.update(
            {
                "motion_fill_method": motion_fill_method,
                "effective_field_method": (
                    "rigid_component_twist+single_view_partial_component_knn_lsmr"
                    if motion_fill_method == RIGID_SINGLE_VIEW_PARTIAL_FILL_METHOD
                    else (
                        "rigid_component_twist+single_view_partial_component_"
                        "knn_lsmr+independent_gaussian_knn_lsmr"
                    )
                ),
                "motion_fill": motion_fill.diagnostics,
            }
        )
    return stats


def _required_scalar_array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    dtype: Any,
) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"Gaussian observations are missing {key}.")
    value = np.asarray(arrays[key])
    if value.shape != ():
        raise ValueError(f"Gaussian observation field {key} must be scalar, got {value.shape}.")
    return value.astype(dtype)


def _write_compact_gaussian_latent(
    out_path: str | Path,
    staged: StagedSolveResult,
    motion_fill: GaussianMotionFillResult | None,
) -> Path:
    phi = staged.observable.phi if motion_fill is None else motion_fill.motion.phi
    roles = None if motion_fill is None else motion_fill.roles.role
    completion = None if motion_fill is None else motion_fill.motion.completion_mask
    return _write_compact_gaussian_latent_arrays(
        out_path,
        staged.prepared,
        phi,
        motion_fill_role=roles,
        completion_mask=completion,
    )


def _write_compact_gaussian_latent_arrays(
    out_path: str | Path,
    prepared: PreparedObservations,
    phi: np.ndarray,
    *,
    motion_fill_role: np.ndarray | None = None,
    completion_mask: np.ndarray | None = None,
) -> Path:
    num_points = int(prepared.points.shape[0])
    if "gaussian_indices" not in prepared.arrays:
        raise ValueError("Gaussian observations are missing gaussian_indices.")
    gaussian_indices = np.asarray(prepared.arrays["gaussian_indices"])
    if gaussian_indices.shape != (num_points,):
        raise ValueError(
            f"gaussian_indices must have shape ({num_points},), got {gaussian_indices.shape}."
        )
    phi = np.asarray(phi)
    if phi.shape != (num_points, 3):
        raise ValueError(f"Final phi must have shape ({num_points},3), got {phi.shape}.")

    arrays: dict[str, np.ndarray] = {
        "points_world": prepared.points.astype(np.float32),
        "phi": phi.astype(np.complex64),
        "gaussian_indices": gaussian_indices.astype(np.int32),
        "freq_hz": _required_scalar_array(prepared.arrays, "freq_hz", np.float32),
        "mode_index": _required_scalar_array(prepared.arrays, "mode_index", np.int32),
        "obs_count_per_point": prepared.obs_count_per_point.astype(np.int32),
        "point_type": _required_scalar_array(prepared.arrays, "point_type", str),
        "source_checkpoint": _required_scalar_array(
            prepared.arrays, "source_checkpoint", str
        ),
    }
    if (motion_fill_role is None) != (completion_mask is None):
        raise ValueError(
            "motion_fill_role and completion_mask must either both be supplied or both be omitted."
        )
    if motion_fill_role is not None and completion_mask is not None:
        role = np.asarray(motion_fill_role)
        completion = np.asarray(completion_mask)
        if role.shape != (num_points,) or not np.issubdtype(role.dtype, np.integer):
            raise ValueError(
                f"motion_fill_role must contain ({num_points},) integers."
            )
        if completion.shape != (num_points,) or completion.dtype != np.bool_:
            raise ValueError(
                f"completion_mask must contain ({num_points},) booleans."
            )
        arrays.update(
            {
                "motion_fill_role": role.astype(np.int8),
                "motion_fill_role_names": np.asarray(MOTION_FILL_ROLE_NAMES),
                "completion_mask": completion.astype(bool),
            }
        )
    return save_npz_compressed_atomic(out_path, arrays)


def _motion_fill_solver_arrays(prefix: str, metadata: Any) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_performed": np.array(bool(metadata.performed)),
        f"{prefix}_converged": np.array(bool(metadata.converged)),
        f"{prefix}_stop_code": np.array(int(metadata.stop_code), dtype=np.int32),
        f"{prefix}_iterations": np.array(int(metadata.iterations), dtype=np.int32),
        f"{prefix}_residual_norm": np.array(float(metadata.residual_norm), dtype=np.float64),
        f"{prefix}_normal_residual_norm": np.array(
            float(metadata.normal_residual_norm), dtype=np.float64
        ),
        f"{prefix}_matrix_norm": np.array(float(metadata.matrix_norm), dtype=np.float64),
        f"{prefix}_condition_estimate": np.array(
            float(metadata.condition_estimate), dtype=np.float64
        ),
        f"{prefix}_solution_norm": np.array(float(metadata.solution_norm), dtype=np.float64),
    }


def _alpha_diagnostic_arrays(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    alpha_view_freqs_hz: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "alphas": alpha.alphas.astype(np.complex64),
        "alpha_reference_view_index": np.array(0, dtype=np.int32),
        "alpha_view_freqs_hz": np.asarray(
            alpha_view_freqs_hz, dtype=np.float32
        ),
        "alpha_identifiable_mask": alpha.identifiable_mask.astype(bool),
        "alpha_reference_connected_mask": alpha.reference_connected_mask.astype(bool),
        "alpha_exclusion_reason": alpha.exclusion_reason,
        "alpha_shared_point_count": alpha.shared_point_count.astype(np.int32),
        "alpha_edge_point_count": alpha.edge_point_count.astype(np.int32),
        "alpha_edge_information": alpha.edge_information.astype(np.float32),
        "alpha_constraint_count_per_view": alpha.constraint_count_per_view.astype(
            np.int32
        ),
        "alpha_information_matrix": alpha.information_matrix.astype(np.complex64),
        "alpha_parameter_information": alpha.parameter_information.astype(np.float32),
        "alpha_parameter_view_indices": alpha.parameter_view_indices.astype(np.int32),
        "alpha_parameter_order": np.array(alpha.parameter_order),
        "alpha_singular_values": alpha.singular_values.astype(np.float32),
        "alpha_rank_ratio": np.array(alpha.rank_ratio, dtype=np.float32),
        "alpha_information_ratio": np.array(alpha.information_ratio, dtype=np.float32),
        "alpha_condition": np.array(alpha.condition, dtype=np.float32),
        "alpha_consistency_residual": np.array(
            alpha.consistency_residual, dtype=np.float32
        ),
        "alpha_phase_std": alpha.phase_std.astype(np.float32),
        "alpha_log_gain_std": alpha.log_gain_std.astype(np.float32),
        "alpha_gain_bound_active_mask": alpha.gain_bound_active_mask.astype(bool),
        "alpha_optimizer_success": np.array(alpha.optimizer_success),
        "alpha_optimizer_status": np.array(alpha.optimizer_status, dtype=np.int32),
        "alpha_optimizer_message": np.array(alpha.optimizer_message),
        "alpha_information_kind": np.array(alpha.information_kind),
    }


def _write_solver_diagnostics(
    out_path: str | Path,
    staged: StagedSolveResult,
    motion_fill: GaussianMotionFillResult | None,
    graph: KnnGraph | None = None,
    graph_path: str | None = None,
) -> Path:
    alpha = staged.alpha
    observable = staged.observable
    final_status = (
        observable.point_status
        if motion_fill is None
        else motion_fill.point_solution_status
    )
    final_point_residual = (
        staged.point_residual
        if motion_fill is None
        else motion_fill.point_residual
    )
    final_point_residual_valid = (
        staged.point_residual_valid_mask
        if motion_fill is None
        else motion_fill.point_residual_valid_mask
    )
    arrays: dict[str, np.ndarray] = {
        **_alpha_diagnostic_arrays(
            staged.prepared,
            alpha,
            staged.alpha_view_freqs_hz,
        ),
        "point_singular_values": observable.singular_values.astype(np.float32),
        "point_observable_rank": observable.observable_rank.astype(np.int8),
        "point_nullity": observable.nullity.astype(np.int8),
        "point_condition": observable.condition.astype(np.float32),
        "point_distinct_view_count": staged.prepared.derived_view_count_per_point.astype(
            np.int32
        ),
        "point_distinct_valid_view_count": observable.distinct_valid_view_count.astype(
            np.int32
        ),
        "point_usable_observation_row_count": observable.usable_observation_row_count.astype(
            np.int32
        ),
        "point_precompletion_residual": observable.precompletion_residual.astype(
            np.float32
        ),
        "staged_point_solution_status": observable.point_status.astype(np.int8),
        "final_point_solution_status": np.asarray(final_status, dtype=np.int8),
        "point_solution_status_names": np.asarray(POINT_STATUS_NAMES),
        "point_residual": np.asarray(final_point_residual, dtype=np.float32),
        "point_residual_valid_mask": np.asarray(
            final_point_residual_valid, dtype=bool
        ),
        "obs_sample_count_per_point": staged.prepared.obs_sample_count_per_point.astype(
            np.int32
        ),
        "anchor_residual_threshold": np.array(
            observable.anchor_residual_threshold, dtype=np.float32
        ),
        "anchor_condition_max": np.array(ANCHOR_CONDITION_MAX, dtype=np.float32),
    }
    if motion_fill is not None:
        if graph is None or graph_path is None:
            raise ValueError("Motion-fill diagnostics require graph metadata.")
        motion = motion_fill.motion
        diagnostics = motion_fill.diagnostics
        system = diagnostics["system"]
        arrays.update(
            {
                "phi_nullspace_correction": motion.phi_nullspace_correction.astype(
                    np.complex64
                ),
                "motion_fill_role": motion_fill.roles.role.astype(np.int8),
                "motion_fill_excluded_reason": motion_fill.roles.excluded_reason.astype(
                    np.int8
                ),
                "motion_fill_point_numerical_nullity": motion_fill.numerical_nullity.astype(
                    np.int8
                ),
                "motion_fill_staged_nullity_refined_mask": motion_fill.staged_nullity_refined_mask.astype(
                    bool
                ),
                "completion_mask": motion.completion_mask.astype(bool),
                "completion_connected_to_anchor": motion.completion_connected_to_anchor.astype(
                    bool
                ),
                "point_active_component_index": motion.connectivity.component_index.astype(
                    np.int32
                ),
                "point_anchor_hop_distance": motion.connectivity.hop_distance.astype(
                    np.int32
                ),
                "active_component_sizes": motion.connectivity.component_sizes.astype(
                    np.int32
                ),
                "active_component_has_anchor": motion.connectivity.component_has_anchor.astype(
                    bool
                ),
                "active_component_anchor_count": motion.connectivity.component_anchor_count.astype(
                    np.int32
                ),
                "motion_fill_method": np.array(MOTION_FILL_METHOD),
                "motion_fill_version": np.array(MOTION_FILL_VERSION, dtype=np.int32),
                "motion_fill_numerical_rank_policy": np.array(
                    MOTION_FILL_NUMERICAL_RANK_POLICY
                ),
                "motion_fill_excluded_policy": np.array(
                    "retain_observable_exclude_from_graph"
                ),
                "motion_fill_graph_path": np.array(graph_path),
                "motion_fill_graph_k": np.array(graph.k, dtype=np.int32),
                "motion_fill_graph_max_distance": np.array(
                    graph.max_distance, dtype=np.float64
                ),
                "motion_fill_graph_epsilon": np.array(
                    graph.epsilon, dtype=np.float64
                ),
                "motion_fill_nullspace_operator_max_relative_error": np.array(
                    diagnostics["nullspace_operator_max_relative_error"],
                    dtype=np.float64,
                ),
                "motion_fill_nullspace_operator_rtol": np.array(
                    MOTION_FILL_NULLSPACE_RTOL, dtype=np.float64
                ),
                "motion_fill_observation_drift_max_relative": np.array(
                    diagnostics["observation_drift_max_relative"], dtype=np.float64
                ),
                "motion_fill_observation_drift_rtol": np.array(
                    MOTION_FILL_OBSERVATION_DRIFT_RTOL, dtype=np.float64
                ),
                "motion_fill_relative_denominator_epsilon": np.array(
                    MOTION_FILL_EPSILON, dtype=np.float64
                ),
                "motion_fill_system_row_count": np.array(
                    motion.system_row_count, dtype=np.int64
                ),
                "motion_fill_system_column_count": np.array(
                    motion.system_column_count, dtype=np.int64
                ),
                "motion_fill_active_edge_count": np.array(
                    motion.active_edge_count, dtype=np.int64
                ),
                "motion_fill_eligible_edge_count": np.array(
                    system["eligible_edge_count"], dtype=np.int64
                ),
                "motion_fill_lsmr_atol": np.array(
                    MOTION_FILL_LSMR_ATOL, dtype=np.float64
                ),
                "motion_fill_lsmr_btol": np.array(
                    MOTION_FILL_LSMR_BTOL, dtype=np.float64
                ),
                "motion_fill_lsmr_conlim": np.array(
                    MOTION_FILL_LSMR_CONLIM, dtype=np.float64
                ),
                "motion_fill_source_solver_method": np.array(
                    "staged_overlap_observable"
                ),
                **_motion_fill_solver_arrays(
                    "motion_fill_lsmr_real", motion.real_solver
                ),
                **_motion_fill_solver_arrays(
                    "motion_fill_lsmr_imaginary", motion.imag_solver
                ),
            }
        )
    return save_npz_compressed_atomic(out_path, arrays)


def _write_rigid_pre_solve_diagnostics(
    out_path: str | Path,
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    alpha_view_freqs_hz: np.ndarray,
    diagnostics_type: str,
) -> Path:
    arrays = {
        **_alpha_diagnostic_arrays(prepared, alpha, alpha_view_freqs_hz),
        "solver_method": np.array("rigid_components"),
        "solver_diagnostics_type": np.array(diagnostics_type),
        "view_ids": prepared.view_ids.astype(str),
    }
    return save_npz_compressed_atomic(out_path, arrays)


def _write_rigid_solver_diagnostics(
    out_path: str | Path,
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    alpha_view_freqs_hz: np.ndarray,
    rigid: RigidComponentSolveResult,
    seed_selection: RigidComponentSeedSelectionResult,
    *,
    rigid_graph_path: str,
    rigid_graph_source_path: str,
    motion_fill: RigidSeedMotionFillResult | None = None,
    motion_fill_graph: KnnGraph | None = None,
    motion_fill_graph_path: str | None = None,
) -> Path:
    if motion_fill is None:
        final_phi = seed_selection.phi
        _, _, _, point_residual, point_residual_valid = (
            compute_prediction_and_residuals(prepared, alpha, final_phi)
        )
        completion_mask = np.zeros(rigid.rigid_seed_mask.shape, dtype=bool)
    else:
        final_phi = motion_fill.motion.phi
        point_residual = motion_fill.point_residual
        point_residual_valid = motion_fill.point_residual_valid_mask
        completion_mask = motion_fill.motion.completion_mask
    arrays: dict[str, np.ndarray] = {
        **_alpha_diagnostic_arrays(prepared, alpha, alpha_view_freqs_hz),
        "solver_method": np.array("rigid_components"),
        "solver_diagnostics_type": np.array("rigid_component_twist_v1"),
        "view_ids": prepared.view_ids.astype(str),
        "rigidity_model": np.array("complex_infinitesimal_se3"),
        "rigid_component_connectivity_policy": np.array(
            "accepted_edge_transitive_components_bridges_merge"
        ),
        "rigid_component_graph_path": np.array(rigid_graph_path),
        "rigid_component_graph_source_path": np.array(rigid_graph_source_path),
        "rigid_component_rcond": np.array(rigid.config.rcond, dtype=np.float64),
        "rigid_component_first_order_rtol": np.array(
            rigid.config.first_order_rtol, dtype=np.float64
        ),
        "rigid_component_first_order_validation": np.array(
            "complex128_model_strict_complex64_quantization_accounted"
        ),
        "rigid_component_persisted_strain_policy": np.array(
            "diagnostic_with_per_edge_actual_cast_error_bound"
        ),
        "rigid_component_phase_samples": np.array(
            rigid.config.phase_samples, dtype=np.int32
        ),
        "rigid_component_seed_policy": np.array(
            "postsolve_valid_view_singular_ratio_and_finite_drift_gate"
        ),
        "rigid_seed_min_valid_views": np.array(
            seed_selection.config.min_valid_views, dtype=np.int32
        ),
        "rigid_seed_min_secondary_view_node_ratio": np.array(
            seed_selection.config.min_secondary_view_node_ratio,
            dtype=np.float64,
        ),
        "rigid_seed_min_singular_ratio": np.array(
            seed_selection.config.min_singular_ratio, dtype=np.float64
        ),
        "rigid_seed_max_finite_drift": np.array(
            seed_selection.config.max_finite_drift, dtype=np.float64
        ),
        "rigid_seed_mask": rigid.rigid_seed_mask.astype(bool),
        "observed_mask": rigid.observed_mask.astype(bool),
        "fill_target_mask": rigid.fill_target_mask.astype(bool),
        "trusted_rigid_seed_mask": (
            seed_selection.trusted_rigid_seed_mask.astype(bool)
        ),
        "effective_fill_target_mask": (
            seed_selection.effective_fill_target_mask.astype(bool)
        ),
        "completion_mask": np.asarray(completion_mask, dtype=bool),
        "point_component_index": rigid.point_component_index.astype(np.int32),
        "rigid_phi_pre_fill": rigid.phi.astype(np.complex64),
        "trusted_rigid_phi_pre_fill": seed_selection.phi.astype(np.complex64),
        "final_phi": np.asarray(final_phi, dtype=np.complex64),
        "point_residual": np.asarray(point_residual, dtype=np.float32),
        "point_residual_valid_mask": np.asarray(point_residual_valid, dtype=bool),
        "obs_count_per_point": prepared.obs_count_per_point.astype(np.int32),
        "obs_sample_count_per_point": prepared.obs_sample_count_per_point.astype(
            np.int32
        ),
        "component_graph_index": rigid.component_graph_index.astype(np.int32),
        "component_node_count": rigid.component_node_count.astype(np.int32),
        "component_edge_count": rigid.component_edge_count.astype(np.int32),
        "component_centroid": rigid.component_centroid.astype(np.float32),
        "component_radius": rigid.component_radius.astype(np.float32),
        "component_usable_observation_row_count": rigid.component_usable_observation_row_count.astype(
            np.int32
        ),
        "component_valid_view_node_count": (
            rigid.component_valid_view_node_count.astype(np.int32)
        ),
        "component_distinct_valid_view_count": rigid.component_distinct_valid_view_count.astype(
            np.int32
        ),
        "component_supported_valid_view_count": (
            seed_selection.component_supported_valid_view_count.astype(np.int32)
        ),
        "component_dominant_valid_view_index": (
            seed_selection.component_dominant_valid_view_index.astype(np.int32)
        ),
        "component_secondary_view_node_ratio": (
            seed_selection.component_secondary_view_node_ratio.astype(np.float32)
        ),
        "component_singular_values": rigid.component_singular_values.astype(
            np.float32
        ),
        "component_singular_ratio": (
            seed_selection.component_singular_ratio.astype(np.float32)
        ),
        "component_rank": rigid.component_rank.astype(np.int8),
        "component_rank_deficient_mask": (rigid.component_rank < 6),
        "component_seed_retained_mask": (
            seed_selection.component_seed_retained_mask.astype(bool)
        ),
        "component_valid_view_rejected_mask": (
            seed_selection.component_valid_view_rejected_mask.astype(bool)
        ),
        "component_singular_rejected_mask": (
            seed_selection.component_singular_rejected_mask.astype(bool)
        ),
        "component_finite_drift_rejected_mask": (
            seed_selection.component_finite_drift_rejected_mask.astype(bool)
        ),
        "component_condition": rigid.component_condition.astype(np.float32),
        "component_weighted_residual_norm": rigid.component_weighted_residual_norm.astype(
            np.float32
        ),
        "component_weighted_measurement_norm": rigid.component_weighted_measurement_norm.astype(
            np.float32
        ),
        "component_normalized_weighted_residual": rigid.component_normalized_weighted_residual.astype(
            np.float32
        ),
        "component_translation": rigid.component_translation.astype(np.complex64),
        "component_rotation": rigid.component_rotation.astype(np.complex64),
        "edge_component_index": rigid.edge_component_index.astype(np.int32),
        "edge_model_first_order_axial_real": rigid.edge_model_first_order_axial_real.astype(
            np.float32
        ),
        "edge_model_first_order_axial_imag": rigid.edge_model_first_order_axial_imag.astype(
            np.float32
        ),
        "edge_model_first_order_relative_real": rigid.edge_model_first_order_relative_real.astype(
            np.float32
        ),
        "edge_model_first_order_relative_imag": rigid.edge_model_first_order_relative_imag.astype(
            np.float32
        ),
        "edge_first_order_axial_real": rigid.edge_first_order_axial_real.astype(
            np.float32
        ),
        "edge_first_order_axial_imag": rigid.edge_first_order_axial_imag.astype(
            np.float32
        ),
        "edge_first_order_relative_real": rigid.edge_first_order_relative_real.astype(
            np.float32
        ),
        "edge_first_order_relative_imag": rigid.edge_first_order_relative_imag.astype(
            np.float32
        ),
        "edge_first_order_quantization_bound_real": rigid.edge_first_order_quantization_bound_real.astype(
            np.float32
        ),
        "edge_first_order_quantization_bound_imag": rigid.edge_first_order_quantization_bound_imag.astype(
            np.float32
        ),
        "edge_first_order_quantization_bound_relative_real": rigid.edge_first_order_quantization_bound_relative_real.astype(
            np.float32
        ),
        "edge_first_order_quantization_bound_relative_imag": rigid.edge_first_order_quantization_bound_relative_imag.astype(
            np.float32
        ),
        "finite_drift_phase_angles": rigid.phase_angles.astype(np.float32),
        "edge_finite_drift_p50": rigid.edge_finite_drift_p50.astype(np.float32),
        "edge_finite_drift_p90": rigid.edge_finite_drift_p90.astype(np.float32),
        "edge_finite_drift_max": rigid.edge_finite_drift_max.astype(np.float32),
        "component_finite_drift_p50": rigid.component_finite_drift_p50.astype(
            np.float32
        ),
        "component_finite_drift_p90": rigid.component_finite_drift_p90.astype(
            np.float32
        ),
        "component_finite_drift_max": rigid.component_finite_drift_max.astype(
            np.float32
        ),
    }
    if motion_fill is not None:
        if motion_fill_graph is None or motion_fill_graph_path is None:
            raise ValueError("Rigid motion-fill diagnostics require graph metadata.")
        motion = motion_fill.motion
        motion_fill_method = str(motion_fill.diagnostics["method"])
        if motion_fill.single_view_partial_diagnostics is None:
            raise ValueError(
                "Rigid motion fill requires single-view partial diagnostics."
            )
        component_fill_policy = RIGID_SINGLE_VIEW_PARTIAL_FILL_POLICY
        arrays.update(
            {
                "motion_fill_method": np.array(motion_fill_method),
                "motion_fill_version": np.array(MOTION_FILL_VERSION, dtype=np.int32),
                "motion_fill_graph_path": np.array(motion_fill_graph_path),
                "motion_fill_graph_k": np.array(
                    motion_fill_graph.k, dtype=np.int32
                ),
                "motion_fill_graph_max_distance": np.array(
                    motion_fill_graph.max_distance, dtype=np.float64
                ),
                "motion_fill_graph_epsilon": np.array(
                    motion_fill_graph.epsilon, dtype=np.float64
                ),
                "motion_fill_role": motion_fill.roles.role.astype(np.int8),
                "motion_fill_excluded_reason": motion_fill.roles.excluded_reason.astype(
                    np.int8
                ),
                "single_view_component_fill_policy": np.array(
                    component_fill_policy
                ),
                "single_view_component_fill_mask": (
                    motion_fill.single_view_component_fill_mask.astype(bool)
                ),
                "single_view_rigid_fill_point_mask": (
                    motion_fill.single_view_rigid_fill_point_mask.astype(bool)
                ),
                "single_view_component_completion_mask": (
                    motion_fill.single_view_component_completion_mask.astype(bool)
                ),
                "single_view_component_anchor_mask": (
                    motion_fill.single_view_component_anchor_mask.astype(bool)
                ),
                "single_view_component_translation": (
                    motion_fill.single_view_component_translation.astype(
                        np.complex64
                    )
                ),
                "single_view_component_rotation": (
                    motion_fill.single_view_component_rotation.astype(np.complex64)
                ),
                "single_view_component_first_order_relative_max": (
                    motion_fill.single_view_component_first_order_relative_max.astype(
                        np.float32
                    )
                ),
                "motion_fill_point_numerical_nullity": motion_fill.numerical_nullity.astype(
                    np.int8
                ),
                "motion_fill_observed_mask": motion_fill.observed_mask.astype(bool),
                "motion_fill_usable_observed_mask": motion_fill.usable_observed_mask.astype(
                    bool
                ),
                "phi_nullspace_correction": motion.phi_nullspace_correction.astype(
                    np.complex64
                ),
                "completion_connected_to_anchor": motion.completion_connected_to_anchor.astype(
                    bool
                ),
                "point_active_component_index": motion.connectivity.component_index.astype(
                    np.int32
                ),
                "point_anchor_hop_distance": motion.connectivity.hop_distance.astype(
                    np.int32
                ),
                "active_component_sizes": motion.connectivity.component_sizes.astype(
                    np.int32
                ),
                "active_component_has_anchor": motion.connectivity.component_has_anchor.astype(
                    bool
                ),
                "active_component_anchor_count": motion.connectivity.component_anchor_count.astype(
                    np.int32
                ),
                "motion_fill_system_row_count": np.array(
                    motion.system_row_count, dtype=np.int64
                ),
                "motion_fill_system_column_count": np.array(
                    motion.system_column_count, dtype=np.int64
                ),
                "motion_fill_active_edge_count": np.array(
                    motion.active_edge_count, dtype=np.int64
                ),
                "motion_fill_lsmr_atol": np.array(
                    MOTION_FILL_LSMR_ATOL, dtype=np.float64
                ),
                "motion_fill_lsmr_btol": np.array(
                    MOTION_FILL_LSMR_BTOL, dtype=np.float64
                ),
                "motion_fill_lsmr_conlim": np.array(
                    MOTION_FILL_LSMR_CONLIM, dtype=np.float64
                ),
                **_motion_fill_solver_arrays(
                    "motion_fill_lsmr_real", motion.real_solver
                ),
                **_motion_fill_solver_arrays(
                    "motion_fill_lsmr_imaginary", motion.imag_solver
                ),
            }
        )
        if motion_fill_method == RIGID_SEQUENTIAL_MOTION_FILL_METHOD:
            arrays["motion_fill_max_anchor_hops"] = np.array(
                int(motion_fill.diagnostics["max_anchor_hops"]),
                dtype=np.int32,
            )
        partial = motion_fill.single_view_partial_diagnostics
        if partial is not None:
            arrays.update(
                {
                    "single_view_observable_singular_ratio_min": np.array(
                        partial.observable_singular_ratio_min, dtype=np.float64
                    ),
                    "single_view_ray_direction_min_fraction": np.array(
                        partial.ray_direction_min_fraction, dtype=np.float64
                    ),
                    "single_view_max_finite_drift": np.array(
                        partial.max_finite_drift, dtype=np.float64
                    ),
                    "single_view_component_observable_rank": (
                        partial.component_observable_rank.astype(np.int8)
                    ),
                    "single_view_component_fill_nullity": (
                        partial.component_fill_nullity.astype(np.int8)
                    ),
                    "single_view_component_ray_dominated_basis_count": (
                        partial.component_ray_dominated_basis_count.astype(np.int8)
                    ),
                    "single_view_component_trusted_knn_edge_count": (
                        partial.component_trusted_knn_edge_count.astype(np.int32)
                    ),
                    "single_view_component_cross_knn_edge_count": (
                        partial.component_cross_knn_edge_count.astype(np.int32)
                    ),
                    "single_view_component_connected_to_trusted_mask": (
                        partial.component_connected_to_trusted_mask.astype(bool)
                    ),
                    "single_view_component_postfill_normalized_residual": (
                        partial.component_postfill_normalized_residual.astype(
                            np.float32
                        )
                    ),
                    "single_view_component_ray_motion_rms": (
                        partial.component_ray_motion_rms.astype(np.float32)
                    ),
                    "single_view_component_tangent_motion_rms": (
                        partial.component_tangent_motion_rms.astype(np.float32)
                    ),
                    "single_view_component_ray_motion_ratio": (
                        partial.component_ray_motion_ratio.astype(np.float32)
                    ),
                    "single_view_component_partial_finite_drift_max": (
                        partial.component_finite_drift_max.astype(np.float32)
                    ),
                    "single_view_component_finite_drift_rejected_mask": (
                        partial.component_finite_drift_rejected_mask.astype(bool)
                    ),
                    "single_view_component_postfill_retained_mask": (
                        partial.component_postfill_retained_mask.astype(bool)
                    ),
                }
            )
    return save_npz_compressed_atomic(out_path, arrays)


def _print_observation_sanity(
    observations: Mapping[str, np.ndarray],
    num_fg: int,
) -> None:
    obs_count = observations["obs_count_per_point"].astype(np.int32)
    obs_sample_count = observations["obs_sample_count_per_point"].astype(np.int32)
    view_ids = observations["view_ids"].astype(str)
    obs_view_index = observations["obs_view_index"].astype(np.int32)
    per_view = np.bincount(obs_view_index, minlength=view_ids.shape[0])
    per_view_text = ", ".join(f"{view_ids[i]}={int(per_view[i])}" for i in range(view_ids.shape[0]))
    obs_percentiles = np.percentile(obs_count, [0, 50, 90, 100])
    obs_sample_percentiles = np.percentile(obs_sample_count, [0, 50, 90, 100])
    print(
        "Observation sanity: "
        f"points={num_fg}, total_obs={obs_view_index.shape[0]}, per_view=[{per_view_text}], "
        f"view_count p0/p50/p90/max="
        f"{obs_percentiles[0]:.0f}/{obs_percentiles[1]:.0f}/{obs_percentiles[2]:.0f}/{obs_percentiles[3]:.0f}, "
        f"sample_count p0/p50/p90/max="
        f"{obs_sample_percentiles[0]:.0f}/{obs_sample_percentiles[1]:.0f}/"
        f"{obs_sample_percentiles[2]:.0f}/{obs_sample_percentiles[3]:.0f}, "
        f"unobserved={int((obs_count == 0).sum())}"
    )


def _write_time_profile(
    out_dir: Path,
    *,
    solve_method: str,
    pipeline_total_seconds: float,
    global_output_seconds: float,
    setup: Mapping[str, float],
    motion_fill_graph: Mapping[str, Any],
    modes: Mapping[str, Mapping[str, Any]],
) -> Path:
    mode_profiles = list(modes.values())
    observation_seconds = float(
        sum(
            float(profile["observation_input_seconds"])
            for profile in mode_profiles
        )
    )
    alpha_seconds = float(
        sum(float(profile["alpha_sync_seconds"]) for profile in mode_profiles)
    )
    rigid_seconds = float(
        sum(float(profile["rigid_component_solve_seconds"]) for profile in mode_profiles)
    )
    seed_seconds = float(
        sum(float(profile["seed_selection_seconds"]) for profile in mode_profiles)
    )
    solve_seconds = alpha_seconds + rigid_seconds + seed_seconds
    motion_fill_mode_seconds = float(
        sum(
            float(profile["motion_fill"]["total_seconds"])
            for profile in mode_profiles
        )
    )
    motion_fill_graph_seconds = float(motion_fill_graph["total_seconds"])
    motion_fill_seconds = motion_fill_graph_seconds + motion_fill_mode_seconds
    mode_output_seconds = float(
        sum(
            float(profile["visualization_and_output_seconds"])
            for profile in mode_profiles
        )
    )
    output_seconds = mode_output_seconds + global_output_seconds
    accounted_mode_seconds = (
        observation_seconds
        + solve_seconds
        + motion_fill_mode_seconds
        + mode_output_seconds
    )
    solve_and_fill_seconds = solve_seconds + motion_fill_seconds
    summary = {
        "mode_count": len(mode_profiles),
        "observation_input_seconds": observation_seconds,
        "alpha_sync_seconds": alpha_seconds,
        "rigid_component_solve_seconds": rigid_seconds,
        "seed_selection_seconds": seed_seconds,
        "solve_total_seconds": solve_seconds,
        "motion_fill_graph_seconds": motion_fill_graph_seconds,
        "motion_fill_per_mode_total_seconds": motion_fill_mode_seconds,
        "motion_fill_total_seconds": motion_fill_seconds,
        "mode_visualization_and_output_seconds": mode_output_seconds,
        "global_output_seconds": global_output_seconds,
        "visualization_and_output_total_seconds": output_seconds,
        "accounted_mode_seconds": accounted_mode_seconds,
        "solve_share_of_accounted_mode": (
            solve_seconds / accounted_mode_seconds if accounted_mode_seconds else 0.0
        ),
        "motion_fill_share_of_accounted_mode": (
            motion_fill_mode_seconds / accounted_mode_seconds
            if accounted_mode_seconds
            else 0.0
        ),
        "solve_and_motion_fill_seconds": solve_and_fill_seconds,
        "solve_share_of_solve_and_motion_fill": (
            solve_seconds / solve_and_fill_seconds if solve_and_fill_seconds else 0.0
        ),
        "motion_fill_share_of_solve_and_motion_fill": (
            motion_fill_seconds / solve_and_fill_seconds
            if solve_and_fill_seconds
            else 0.0
        ),
    }
    payload = {
        "version": 1,
        "clock": "time.perf_counter",
        "units": "wall_seconds",
        "motion_fill_substages_are_inclusive_in_total": True,
        "solve_method": solve_method,
        "pipeline_total_seconds": float(pipeline_total_seconds),
        "global_output_seconds": float(global_output_seconds),
        "setup": dict(setup),
        "motion_fill_graph": dict(motion_fill_graph),
        "modes": dict(modes),
        "summary": summary,
    }
    path = _write_json_atomic(out_dir / "time_profile.json", payload)
    print("Timing summary (wall clock)")
    print(f"  pipeline total              {pipeline_total_seconds:10.3f} s")
    print(f"  setup total                 {float(setup['total_seconds']):10.3f} s")
    print(
        "  motion-fill graph          "
        f"{float(motion_fill_graph['total_seconds']):10.3f} s"
    )
    print(f"  solve total                 {solve_seconds:10.3f} s")
    print(f"  motion fill total           {motion_fill_seconds:10.3f} s")
    print(f"  visualization/output        {output_seconds:10.3f} s")
    print(
        "  solve / fill share          "
        f"{100.0 * float(summary['solve_share_of_solve_and_motion_fill']):.1f}% / "
        f"{100.0 * float(summary['motion_fill_share_of_solve_and_motion_fill']):.1f}%"
    )
    for mode_name, profile in modes.items():
        motion = profile["motion_fill"]
        print(f"  {mode_name} total {float(profile['total_seconds']):.3f} s")
        print(
            "    alpha / rigid / fill      "
            f"{float(profile['alpha_sync_seconds']):.3f} / "
            f"{float(profile['rigid_component_solve_seconds']):.3f} / "
            f"{float(motion['total_seconds']):.3f} s"
        )
        if bool(motion["enabled"]):
            motion_fill_stage = str(motion["stage"])
            if motion_fill_stage == RIGID_MOTION_FILL_STAGE_DEFAULT:
                print(
                    "    component fill total       "
                    f"{float(motion['component_total_seconds']):.3f} s"
                )
                print(
                    "    component LSMR real/imag  "
                    f"{float(motion['component_lsmr_real_seconds']):.3f} / "
                    f"{float(motion['component_lsmr_imaginary_seconds']):.3f} s"
                )
                print(
                    "    point prep/connect/assemble "
                    f"{float(motion['pointwise_preparation_seconds']):.3f} / "
                    f"{float(motion['pointwise_connectivity_seconds']):.3f} / "
                    f"{float(motion['pointwise_system_assembly_seconds']):.3f} s"
                )
                print(
                    "    point max hops / limited   "
                    f"{int(motion['max_anchor_hops'])} / "
                    f"{int(motion['hop_limited_target_count'])}"
                )
                print(
                    "    point LSMR real/imag      "
                    f"{float(motion['pointwise_lsmr_real_seconds']):.3f} / "
                    f"{float(motion['pointwise_lsmr_imaginary_seconds']):.3f} s"
                )
                print(
                    "    point recon / validation  "
                    f"{float(motion['pointwise_reconstruction_seconds']):.3f} / "
                    f"{float(motion['validation_and_residual_seconds']):.3f} s"
                )
            elif motion_fill_stage == "single-view-components":
                print(
                    "    fill prep/connect/assemble "
                    f"{float(motion['preparation_seconds']):.3f} / "
                    f"{float(motion['connectivity_seconds']):.3f} / "
                    f"{float(motion['system_assembly_seconds']):.3f} s"
                )
                print(
                    "    fill LSMR real/imag        "
                    f"{float(motion['lsmr_real_seconds']):.3f} / "
                    f"{float(motion['lsmr_imaginary_seconds']):.3f} s"
                )
                print(
                    "    fill validation/residual  "
                    f"{float(motion['validation_and_residual_seconds']):.3f} s"
                )
            else:
                raise ValueError(
                    f"Unsupported rigid motion-fill profile stage: {motion_fill_stage}"
                )
    print(f"Saved time profile -> {path}")
    return path


def run(args: argparse.Namespace) -> None:
    pipeline_started = perf_counter()
    setup_timings: dict[str, float] = {}
    stage_started = perf_counter()
    _validate_solve_method_arguments(args)
    _validate_motion_fill_arguments(args)
    view_configs_paths = list(args.view_config)
    modal_npzs_paths = list(args.modal_npz)
    if len(view_configs_paths) != len(modal_npzs_paths):
        raise ValueError("--view-config and --modal-npz must be supplied the same number of times.")
    setup_timings["argument_validation_seconds"] = float(
        perf_counter() - stage_started
    )

    stage_started = perf_counter()
    if args.solve_method == "rigid-components":
        fg_means = load_fg_means_from_checkpoint(args.input_ckpt)
        fg_scales = None
        fg_quats = None
        fg_opacities = None
        rendered_depths = None
        rendered_accs = None
    else:
        (
            fg_means,
            fg_scales,
            fg_quats,
            fg_opacities,
            _fg_colors,
            rendered_depths,
            rendered_accs,
        ) = load_fg_pixel_candidate_inputs_from_checkpoint(
            args.input_ckpt,
            view_configs_paths,
        )
    setup_timings["checkpoint_input_seconds"] = float(
        perf_counter() - stage_started
    )

    gaussian_tree = None
    stage_started = perf_counter()
    if args.solve_method == "staged" or args.motion_fill:
        from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

        gaussian_tree = cKDTree(fg_means.astype(np.float64))
    setup_timings["gaussian_tree_seconds"] = float(
        perf_counter() - stage_started
    )

    stage_started = perf_counter()
    _validate_motion_fill_arguments(args, fg_means.shape[0])
    setup_timings["point_count_validation_seconds"] = float(
        perf_counter() - stage_started
    )

    stage_started = perf_counter()
    freqs_per_view = load_modal_freqs(modal_npzs_paths)
    # e.g. freqs_per_view = [
    # np.array([0.357, 0.714]),  # view 1
    # np.array([0.359, 0.711]),  # view 2
    # np.array([0.356, 0.716]),  # view 3 ]
    mode_indices = parse_mode_indices(args.mode_indices, int(freqs_per_view[0].shape[0]))
    rigid_graphs = (
        _load_rigid_component_graphs(
            list(args.rigid_component_graph),
            mode_indices,
        )
        if args.solve_method == "rigid-components"
        else {}
    )
    setup_timings["frequency_and_rigid_graph_loading_seconds"] = float(
        perf_counter() - stage_started
    )
    incremental_base: IncrementalManifestBase | None = None
    if args.base_manifest is not None:
        incremental_base = load_incremental_manifest_base(
            args.base_manifest,
            source_checkpoint=str(args.input_ckpt),
            source_view_configs=view_configs_paths,
            frequencies_by_view=freqs_per_view,
            extension_mode_indices=mode_indices,
            expected_parameters=_manifest_parameters(args),
            frequency_tolerance_hz=float(args.freq_tolerance_hz),
        )
        print(
            "Incremental solve: reusing "
            f"{len(incremental_base.mode_indices)} modes from {incremental_base.path}; "
            f"solving only mode indices {mode_indices}."
        )
    stage_started = perf_counter()
    out_dir = Path(args.out_dir)
    obs_dir = out_dir / "observations"
    latent_dir = out_dir / "latents"
    diagnostics_dir = out_dir / "diagnostics"
    vis_dir = out_dir / "vis"
    obs_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    setup_timings["output_directory_seconds"] = float(
        perf_counter() - stage_started
    )

    # Build KNN graph from gaussian centers
    motion_fill_graph = None
    motion_fill_graph_path: Path | None = None
    motion_fill_mode_diagnostics: dict[str, Any] = {}
    motion_fill_graph_profile: dict[str, Any] = {
        "enabled": bool(args.motion_fill),
        "query_candidates_seconds": 0.0,
        "build_graph_seconds": 0.0,
        "write_graph_seconds": 0.0,
        "total_seconds": 0.0,
    }
    if args.motion_fill:
        graph_started = perf_counter()
        assert gaussian_tree is not None
        if incremental_base is not None:
            if incremental_base.motion_fill_graph_path is None:
                raise ValueError(
                    "The base manifest does not provide the required motion-fill graph"
                )
            motion_fill_graph_path = incremental_base.motion_fill_graph_path
            motion_fill_graph = load_motion_fill_graph(
                motion_fill_graph_path,
                fg_means,
                expected_k=int(args.motion_fill_k),
                expected_max_distance=float(args.motion_fill_max_distance),
            )
            print(f"Reused motion-fill graph -> {motion_fill_graph_path}")
        else:
            stage_started = perf_counter()
            candidates = query_knn_candidates(
                fg_means,
                int(args.motion_fill_k),
                tree=gaussian_tree,
            )
            motion_fill_graph_profile["query_candidates_seconds"] = float(
                perf_counter() - stage_started
            )
            stage_started = perf_counter()
            motion_fill_graph = build_knn_graph(
                candidates,
                int(args.motion_fill_k),
                float(args.motion_fill_max_distance),
                MOTION_FILL_EPSILON,
            )
            motion_fill_graph_profile["build_graph_seconds"] = float(
                perf_counter() - stage_started
            )
            stage_started = perf_counter()
            motion_fill_graph_path = write_motion_fill_graph(
                out_dir / "motion_fill" / "graph.npz",
                fg_means,
                candidates,
                motion_fill_graph,
            )
            motion_fill_graph_profile["write_graph_seconds"] = float(
                perf_counter() - stage_started
            )
        motion_fill_graph_profile["total_seconds"] = float(
            perf_counter() - graph_started
        )

    modes: list[dict[str, Any]] = []
    mode_time_profiles: dict[str, dict[str, Any]] = {}
    for mode_index in mode_indices:
        mode_started = perf_counter()
        reference_freq = float(freqs_per_view[0][mode_index])
        mode_name = f"mode_{mode_index:03d}_{freq_slug(reference_freq)}hz"
        obs_path = obs_dir / f"{mode_name}.npz"
        latent_path = latent_dir / f"{mode_name}.npz"
        diagnostics_path = diagnostics_dir / f"{mode_name}.npz"
        mode_vis_dir = vis_dir / mode_name
        print(f"Solving Gaussian mode {mode_name}")
        latent_path.unlink(missing_ok=True)
        if args.solve_method == "rigid-components":
            mode_profile: dict[str, Any] = {}
            stage_started = perf_counter()
            loaded_graph = rigid_graphs[mode_index]
            source_observation_path = Path(loaded_graph.source_observation_path)
            observations = _load_npz_arrays(source_observation_path)
            prepared = _validate_rigid_source_observations(
                loaded_graph,
                observations,
                args=args,
                view_config_paths=view_configs_paths,
                modal_npz_paths=modal_npzs_paths,
                freqs_per_view=freqs_per_view,
                mode_index=mode_index,
                reference_freq=reference_freq,
                foreground_points=fg_means,
            )
            _print_observation_sanity(observations, fg_means.shape[0])
            _copy_file_atomic(source_observation_path, obs_path)
            rigid_graph_local_path = _copy_file_atomic(
                loaded_graph.graph_path,
                out_dir / "rigid_components" / "graphs" / f"{mode_name}.npz",
            )
            mode_profile["observation_input_seconds"] = float(
                perf_counter() - stage_started
            )
            alpha_config = staged_solver_config(args)
            stage_started = perf_counter()
            alpha = solve_alpha_sync(prepared, alpha_config, enforce_failure=False)
            mode_profile["alpha_sync_seconds"] = float(
                perf_counter() - stage_started
            )
            alpha_view_freqs_hz = _alpha_view_frequencies(prepared)
            invalid_alpha_views = _unidentifiable_observed_view_indices(
                prepared, alpha
            )
            if alpha_config.alpha_failure == "error" and invalid_alpha_views.size:
                _write_rigid_pre_solve_diagnostics(
                    diagnostics_path,
                    prepared,
                    alpha,
                    alpha_view_freqs_hz,
                    "rigid_component_alpha_failure_v1",
                )
                labels = [
                    str(prepared.view_ids[index])
                    for index in invalid_alpha_views.tolist()
                ]
                raise ValueError(
                    "Unidentifiable alpha for observed views after writing "
                    f"diagnostics to {diagnostics_path}: {labels}."
                )
            stage_started = perf_counter()
            try:
                rigid = solve_rigid_components(
                    prepared,
                    alpha,
                    loaded_graph.graph,
                    RigidComponentSolverConfig(
                        rcond=float(args.rigid_component_rcond),
                    ),
                )
            except Exception:
                _write_rigid_pre_solve_diagnostics(
                    diagnostics_path,
                    prepared,
                    alpha,
                    alpha_view_freqs_hz,
                    "rigid_component_solve_failure_v1",
                )
                raise
            mode_profile["rigid_component_solve_seconds"] = float(
                perf_counter() - stage_started
            )

            stage_started = perf_counter()
            seed_selection = select_trusted_rigid_component_seeds(
                rigid,
                RigidComponentSeedSelectionConfig(
                    min_valid_views=int(args.rigid_seed_min_valid_views),
                    min_secondary_view_node_ratio=float(
                        args.rigid_seed_min_secondary_view_node_ratio
                    ),
                    min_singular_ratio=float(
                        args.rigid_seed_min_singular_ratio
                    ),
                    max_finite_drift=float(
                        args.rigid_seed_max_finite_drift
                    ),
                ),
            )
            raw_view_count = rigid.component_distinct_valid_view_count
            supported_view_count = (
                seed_selection.component_supported_valid_view_count
            )
            print(
                "Rigid view support: "
                f"components={rigid.num_components}, "
                f"raw_multiview={int(np.count_nonzero(raw_view_count >= 2))}, "
                f"supported_multiview={int(np.count_nonzero(supported_view_count >= 2))}, "
                f"downgraded={int(np.count_nonzero(raw_view_count > supported_view_count))}, "
                f"effective_single_view={int(np.count_nonzero(supported_view_count == 1))}",
                flush=True,
            )
            mode_profile["seed_selection_seconds"] = float(
                perf_counter() - stage_started
            )

            rigid_motion_fill = None
            motion_fill_timings: dict[str, float] = {}
            motion_fill_relative_path = (
                relative_path(motion_fill_graph_path, out_dir)
                if motion_fill_graph_path is not None
                else None
            )
            if motion_fill_graph is not None:
                assert motion_fill_relative_path is not None
                try:
                    if not np.any(seed_selection.trusted_rigid_seed_mask):
                        raise ValueError(
                            "Rigid seed filtering retained no component for "
                            "motion fill. Relax --rigid-seed-min-valid-views "
                            "or --rigid-seed-min-singular-ratio."
                        )
                    def progress(message: str) -> None:
                        print(f"Motion fill: {message}", flush=True)

                    if args.rigid_motion_fill_stage == "single-view-components":
                        rigid_motion_fill = apply_single_view_component_partial_fill(
                            prepared,
                            alpha,
                            rigid,
                            seed_selection,
                            loaded_graph.graph,
                            motion_fill_graph,
                            motion_fill_relative_path,
                            observable_singular_ratio_min=float(
                                args.rigid_single_view_observable_ratio
                            ),
                            ray_direction_min_fraction=float(
                                args.rigid_single_view_ray_direction_min_fraction
                            ),
                            max_finite_drift=float(
                                args.rigid_seed_max_finite_drift
                            ),
                            timings=motion_fill_timings,
                            progress=progress,
                        )
                    else:
                        rigid_motion_fill = apply_sequential_rigid_motion_fill(
                            prepared,
                            alpha,
                            rigid,
                            seed_selection,
                            loaded_graph.graph,
                            motion_fill_graph,
                            motion_fill_relative_path,
                            observable_singular_ratio_min=float(
                                args.rigid_single_view_observable_ratio
                            ),
                            ray_direction_min_fraction=float(
                                args.rigid_single_view_ray_direction_min_fraction
                            ),
                            max_finite_drift=float(
                                args.rigid_seed_max_finite_drift
                            ),
                            max_anchor_hops=int(
                                args.motion_fill_max_anchor_hops
                            ),
                            timings=motion_fill_timings,
                            progress=progress,
                        )
                except Exception:
                    _write_rigid_solver_diagnostics(
                        diagnostics_path,
                        prepared,
                        alpha,
                        alpha_view_freqs_hz,
                        rigid,
                        seed_selection,
                        rigid_graph_path=relative_path(
                            rigid_graph_local_path, out_dir
                        ),
                        rigid_graph_source_path=str(loaded_graph.graph_path),
                    )
                    raise
                motion_fill_mode_diagnostics[mode_name] = (
                    rigid_motion_fill.diagnostics
                )
            mode_profile["motion_fill"] = {
                "enabled": motion_fill_graph is not None,
                "stage": str(args.rigid_motion_fill_stage),
                "total_seconds": 0.0,
                **motion_fill_timings,
            }
            if (
                rigid_motion_fill is not None
                and args.rigid_motion_fill_stage == "sequential"
            ):
                mode_profile["motion_fill"].update(
                    {
                        "max_anchor_hops": int(
                            rigid_motion_fill.diagnostics["max_anchor_hops"]
                        ),
                        "hop_limited_target_count": int(
                            rigid_motion_fill.diagnostics["completion"][
                                "hop_limited_target_count"
                            ]
                        ),
                    }
                )

            stage_started = perf_counter()
            if rigid_motion_fill is None:
                final_phi = seed_selection.phi
                final_prediction, _, final_residual_valid, _, _ = (
                    compute_prediction_and_residuals(prepared, alpha, final_phi)
                )
                motion_fill_role = None
                completion_mask = None
            else:
                final_phi = rigid_motion_fill.motion.phi
                final_prediction = rigid_motion_fill.obs_pred_y
                final_residual_valid = rigid_motion_fill.obs_residual_valid_mask
                motion_fill_role = rigid_motion_fill.roles.role
                completion_mask = rigid_motion_fill.motion.completion_mask
            write_prepared_solve_visualizations(
                prepared,
                final_phi,
                final_prediction,
                final_residual_valid,
                mode_vis_dir,
            )
            _write_rigid_solver_diagnostics(
                diagnostics_path,
                prepared,
                alpha,
                alpha_view_freqs_hz,
                rigid,
                seed_selection,
                rigid_graph_path=relative_path(rigid_graph_local_path, out_dir),
                rigid_graph_source_path=str(loaded_graph.graph_path),
                motion_fill=rigid_motion_fill,
                motion_fill_graph=motion_fill_graph,
                motion_fill_graph_path=motion_fill_relative_path,
            )
            _write_compact_gaussian_latent_arrays(
                latent_path,
                prepared,
                final_phi,
                motion_fill_role=motion_fill_role,
                completion_mask=completion_mask,
            )
            mode_stats = _rigid_gaussian_latent_stats(
                prepared,
                alpha,
                rigid,
                seed_selection,
                rigid_motion_fill,
                observations,
                fg_means.shape[0],
            )
            freqs_by_view = [
                float(freqs[mode_index]) for freqs in freqs_per_view
            ]
            modes.append(
                {
                    "mode_index": int(mode_index),
                    "freq_hz": reference_freq,
                    "freqs_hz_by_view": freqs_by_view,
                    "label": f"{mode_index}: {reference_freq:.6f} Hz",
                    "observation_path": relative_path(obs_path, out_dir),
                    "source_observation_path": loaded_graph.source_observation_path,
                    "latent_path": relative_path(latent_path, out_dir),
                    "diagnostics_path": relative_path(diagnostics_path, out_dir),
                    "component_diagnostics_path": relative_path(
                        diagnostics_path, out_dir
                    ),
                    "rigid_component_graph_path": relative_path(
                        rigid_graph_local_path, out_dir
                    ),
                    "rigid_component_graph_source_path": str(
                        loaded_graph.graph_path
                    ),
                    "vis_dir": relative_path(mode_vis_dir, out_dir),
                    "alpha_by_view": _alpha_by_view_diagnostics(
                        prepared, alpha, alpha_view_freqs_hz
                    ),
                    "stats": mode_stats,
                }
            )
            mode_profile["visualization_and_output_seconds"] = float(
                perf_counter() - stage_started
            )
            mode_profile["solve_total_seconds"] = float(
                mode_profile["alpha_sync_seconds"]
                + mode_profile["rigid_component_solve_seconds"]
                + mode_profile["seed_selection_seconds"]
            )
            mode_profile["total_seconds"] = float(
                perf_counter() - mode_started
            )
            mode_time_profiles[mode_name] = mode_profile
            continue
        build_gaussian_observation_graph(
            # Staged mode loaded the static render inputs and Gaussian tree above.
            points_world=fg_means,
            view_config_paths=view_configs_paths,
            modal_npz_paths=modal_npzs_paths,
            out_path=obs_path,
            source_checkpoint=args.input_ckpt,
            mode_index=mode_index,
            mask_erode_iters=args.mask_erode_iters,
            freq_tolerance_hz=args.freq_tolerance_hz,  #
            pixel_sample_stride=args.pixel_sample_stride,
            pixel_candidate_k=args.pixel_candidate_k,
            pixel_preselect_k=args.pixel_preselect_k,
            pixel_render_acc_min=args.pixel_render_acc_min,
            pixel_min_contribution=args.pixel_min_contribution,
            gaussian_scales=fg_scales,
            gaussian_quats=fg_quats,
            gaussian_opacities=fg_opacities,
            rendered_depths=rendered_depths,
            rendered_accs=rendered_accs,
            gaussian_tree=gaussian_tree,
        )
        with np.load(str(obs_path), allow_pickle=False) as loaded:
            observations = {key: loaded[key] for key in loaded.files}
        if incremental_base is not None:
            validate_incremental_observation_topology(
                incremental_base,
                observations,
                obs_path,
            )
        _print_observation_sanity(observations, fg_means.shape[0])
        staged = optimize_multi_view_staged(
            observations=observations,
            config=staged_solver_config(args),
        )
        if (
            staged.config.alpha_failure == "error"
            and staged.unidentifiable_observed_view_indices.size
        ):
            _write_solver_diagnostics(diagnostics_path, staged, None)
            enforce_alpha_failure(staged, diagnostics_path)

        motion_fill_result = None
        if motion_fill_graph is not None:
            assert motion_fill_graph_path is not None
            graph_relative_path = relative_path(motion_fill_graph_path, out_dir)
            try:
                motion_fill_result = apply_gaussian_motion_fill(
                    staged,
                    motion_fill_graph,
                    graph_relative_path,
                )
            except Exception:
                _write_solver_diagnostics(diagnostics_path, staged, None)
                raise
            motion_fill_mode_diagnostics[mode_name] = motion_fill_result.diagnostics

        if motion_fill_result is None:
            final_phi = staged.observable.phi
            final_prediction = staged.obs_pred_y
            final_residual_valid = staged.obs_residual_valid_mask
        else:
            final_phi = motion_fill_result.motion.phi
            final_prediction = motion_fill_result.obs_pred_y
            final_residual_valid = motion_fill_result.obs_residual_valid_mask
        write_solve_visualizations(
            staged,
            final_phi,
            final_prediction,
            final_residual_valid,
            mode_vis_dir,
        )
        graph_relative_path = (
            relative_path(motion_fill_graph_path, out_dir)
            if motion_fill_graph_path is not None
            else None
        )
        _write_solver_diagnostics(
            diagnostics_path,
            staged,
            motion_fill_result,
            motion_fill_graph,
            graph_relative_path,
        )
        _write_compact_gaussian_latent(latent_path, staged, motion_fill_result)

        mode_stats = _gaussian_latent_stats(
            staged,
            motion_fill_result,
            observations,
            fg_means.shape[0],
        )
        if motion_fill_result is not None:
            mode_stats["motion_fill"] = motion_fill_result.diagnostics
        freqs_by_view = [float(freqs[mode_index]) for freqs in freqs_per_view]
        mode_entry = {
            "mode_index": int(mode_index),
            "freq_hz": reference_freq,
            "freqs_hz_by_view": freqs_by_view,
            "label": f"{mode_index}: {reference_freq:.6f} Hz",
            "observation_path": relative_path(obs_path, out_dir),
            "latent_path": relative_path(latent_path, out_dir),
            "diagnostics_path": relative_path(diagnostics_path, out_dir),
            "vis_dir": relative_path(mode_vis_dir, out_dir),
            "alpha_by_view": alpha_by_view_diagnostics(staged),
            "stats": mode_stats,
        }
        modes.append(mode_entry)

    global_output_started = perf_counter()
    solver_manifest_parameters = (
        rigid_component_manifest_parameters(args)
        if args.solve_method == "rigid-components"
        else staged_solver_manifest_parameters(args)
    )
    manifest_parameters = {
        "mask_erode_iters": int(args.mask_erode_iters),
        "pixel_sample_stride": int(args.pixel_sample_stride),
        "pixel_candidate_k": int(args.pixel_candidate_k),
        "pixel_preselect_k": int(args.pixel_preselect_k),
        "pixel_render_acc_min": float(args.pixel_render_acc_min),
        "pixel_min_contribution": float(args.pixel_min_contribution),
        "freq_tolerance_hz": float(args.freq_tolerance_hz),
        "alpha_model": "per_view_per_mode",
        "alpha_reference_view_index": 0,
        "motion_fill_enabled": bool(args.motion_fill),
        **solver_manifest_parameters,
    }
    if args.solve_method == "rigid-components":
        manifest_parameters.update(
            {
                "rigid_component_graph_count": len(rigid_graphs),
                "rigid_component_observation_policy": (
                    "reuse_graph_source_observation_no_rebuild"
                ),
            }
        )
    if motion_fill_graph is not None:
        assert motion_fill_graph_path is not None
        motion_fill_method = (
            RIGID_SINGLE_VIEW_PARTIAL_FILL_METHOD
            if args.solve_method == "rigid-components"
            and args.rigid_motion_fill_stage == "single-view-components"
            else RIGID_SEQUENTIAL_MOTION_FILL_METHOD
            if args.solve_method == "rigid-components"
            else MOTION_FILL_METHOD
        )
        manifest_parameters.update(
            {
                "motion_fill_method": motion_fill_method,
                "motion_fill_k": int(motion_fill_graph.k),
                "motion_fill_max_distance": float(motion_fill_graph.max_distance),
                "motion_fill_epsilon": float(motion_fill_graph.epsilon),
                "motion_fill_graph_path": relative_path(motion_fill_graph_path, out_dir),
                "motion_fill_excluded_policy": (
                    "ordinary_gaussians_zero_component_only"
                    if args.solve_method == "rigid-components"
                    and args.rigid_motion_fill_stage == "single-view-components"
                    else (
                        "finite_safe_components_fixed_anchor_hop_limited_"
                        "gaussians_free"
                    )
                    if args.solve_method == "rigid-components"
                    else "retain_observable_exclude_from_graph"
                ),
            }
        )
        if args.solve_method == "staged":
            manifest_parameters.update(
                {
                    "motion_fill_nullspace_operator_rtol": MOTION_FILL_NULLSPACE_RTOL,
                    "motion_fill_observation_drift_rtol": MOTION_FILL_OBSERVATION_DRIFT_RTOL,
                }
            )
        write_motion_fill_diagnostics(
            out_dir / "motion_fill" / "diagnostics.json",
            {
                "version": 1,
                "method": motion_fill_method,
                "graph_path": relative_path(motion_fill_graph_path, out_dir),
                "graph": {
                    "k": int(motion_fill_graph.k),
                    "max_distance": float(motion_fill_graph.max_distance),
                    "epsilon": float(motion_fill_graph.epsilon),
                    "edge_count": int(motion_fill_graph.edge_index.shape[0]),
                    "zero_distance_edge_count": int(
                        np.count_nonzero(motion_fill_graph.edge_distance == 0.0)
                    ),
                    "component_count": int(motion_fill_graph.component_sizes.shape[0]),
                    "isolated_point_count": int(
                        np.count_nonzero(motion_fill_graph.isolated_mask)
                    ),
                    **(
                        {
                            "max_anchor_hops": int(
                                args.motion_fill_max_anchor_hops
                            )
                        }
                        if args.solve_method == "rigid-components"
                        and args.rigid_motion_fill_stage == "sequential"
                        else {}
                    ),
                },
                "modes": motion_fill_mode_diagnostics,
            },
        )

    new_modes = modes
    if incremental_base is None:
        combined_modes = new_modes
        combined_mode_indices = mode_indices
    else:
        combined_modes = [*incremental_base.modes, *new_modes]
        combined_mode_indices = [*incremental_base.mode_indices, *mode_indices]

    manifest: dict[str, Any] = {
        "version": 1,
        "source_checkpoint": str(args.input_ckpt),
        "point_type": "foreground_gaussian_center",
        "source_view_configs": view_configs_paths,
        "source_modal_npzs": modal_npzs_paths,
        "mode_indices": combined_mode_indices,
        "parameters": manifest_parameters,
        "modes": combined_modes,
    }
    if incremental_base is not None:
        manifest["incremental_extension"] = {
            "base_manifest": str(incremental_base.path),
            "reused_mode_indices": list(incremental_base.mode_indices),
            "solved_mode_indices": mode_indices,
            "reused_motion_fill_graph": bool(args.motion_fill),
        }
    manifest_path = out_dir / "modal_modes_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved Gaussian modal modes manifest -> {manifest_path}")
    if args.solve_method == "rigid-components":
        global_output_seconds = float(perf_counter() - global_output_started)
        setup_timings["total_seconds"] = float(sum(setup_timings.values()))
        _write_time_profile(
            out_dir,
            solve_method=str(args.solve_method),
            pipeline_total_seconds=float(perf_counter() - pipeline_started),
            global_output_seconds=global_output_seconds,
            setup=setup_timings,
            motion_fill_graph=motion_fill_graph_profile,
            modes=mode_time_profiles,
        )
