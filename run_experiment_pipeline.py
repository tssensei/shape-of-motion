from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import yaml


CONFIG_FORMAT = "som_modal_experiment_pipeline"
CONFIG_VERSION = 1
STATE_FORMAT = "som_modal_experiment_pipeline_state"
STATE_VERSION = 1
RECEIPT_FORMAT = "som_modal_experiment_stage_receipt"
RECEIPT_VERSION = 1
STATIC_APPROVAL_FORMAT = "som_static_quality_approval"
STATIC_COMPLETION_FORMAT = "som_static_training_completion"
GRAPH_APPROVAL_FORMAT = "som_rigid_graph_approval"
FINAL_READY_FORMAT = "som_modal_visualization_ready"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StaticConfig:
    artifact_id: str
    initial_epochs: int
    num_fg: int
    num_bg: int
    batch_size: int
    num_workers: int
    use_2dgs: bool


@dataclass(frozen=True)
class ModalInputsConfig:
    artifact_id: str
    flow_artifact_id: str
    no_smooth: bool
    sigma_b: float
    sigma_c: float
    analysis_mask_dilate_iters: int


@dataclass(frozen=True)
class ViewConfigsConfig:
    artifact_id: str


@dataclass(frozen=True)
class TopologyConfig:
    artifact_id: str
    pixel_sample_stride: int
    pixel_candidate_k: int
    pixel_preselect_k: int
    pixel_render_acc_min: float
    pixel_min_contribution: float
    mask_erode_iters: int


@dataclass(frozen=True)
class FrequencyConfig:
    artifact_id: str
    basis_id: str
    min_freq_hz: float
    max_freq_hz: float
    frequency_step_hz: float
    mode_counts: tuple[int, ...]
    selected_k: int
    pixel_stride: int


@dataclass(frozen=True)
class SolverConfig:
    artifact_id: str
    method: str
    freq_tolerance_hz: float
    alpha_model: str
    alpha_gain_min: float
    alpha_gain_max: float
    alpha_min_shared_points: int
    alpha_rank_ratio_min: float
    alpha_info_ratio_min: float
    alpha_failure: str
    anchor_svd_ratio_min: float
    anchor_residual_max: float
    rigid_component_rcond: float
    rigid_seed_min_valid_views: int
    rigid_seed_min_secondary_view_node_ratio: float
    rigid_seed_min_singular_ratio: float
    rigid_seed_max_finite_drift: float
    rigid_single_view_observable_ratio: float
    rigid_single_view_ray_direction_min_fraction: float


@dataclass(frozen=True)
class RigidGraphConfig:
    default_candidate_id: str
    max_distance: float
    max_neighbors: int
    color_mad_multiplier: float
    depth_mad_multiplier: float
    depth_samples: int
    min_shared_views: int
    min_component_nodes: int
    min_component_edges: int


@dataclass(frozen=True)
class MotionFillConfig:
    artifact_id: str
    k: int
    max_distance: float
    max_anchor_hops: int


@dataclass(frozen=True)
class RenderedDesignConfig:
    artifact_id: str
    pixel_sample_stride: int
    alpha_min: float
    mask_erode_iters: int
    modes_per_batch: int
    use_2dgs: bool
    write_role_diagnostics: bool


@dataclass(frozen=True)
class FlowCoordinatesConfig:
    artifact_id: str
    ridge_relative: float
    frame_chunk_size: int


@dataclass(frozen=True)
class PhysicsConfig:
    enabled: bool
    artifact_id: str
    damping_ratio: float
    forcing_weight: float
    forcing_difference_weight: float
    assigned_band_half_width_hz: float
    frame_chunk_size: int


@dataclass(frozen=True)
class CheckpointsConfig:
    direct_artifact_id: str
    physics_artifact_id: str


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    config_identity: str
    scene_id: str
    scene_root: Path
    pipeline_id: str
    prestatic_run_id: str
    static: StaticConfig
    modal_inputs: ModalInputsConfig
    view_configs: ViewConfigsConfig
    topology: TopologyConfig
    frequency: FrequencyConfig
    solver: SolverConfig
    rigid_graph: RigidGraphConfig
    motion_fill: MotionFillConfig
    rendered_design: RenderedDesignConfig
    flow_coordinates: FlowCoordinatesConfig
    physics: PhysicsConfig
    checkpoints: CheckpointsConfig


@dataclass(frozen=True)
class SourceView:
    view_id: str
    image_dir: Path
    mask_dir: Path
    fps_hz: float
    width: int
    height: int
    frame_names: tuple[str, ...]
    source_identity: str
    reference_frame_name: str
    reference_frame_stem: str
    reference_local_index: int


@dataclass(frozen=True)
class PrestaticInputs:
    ready_path: Path
    static_dataset: Path
    source_manifest: Path
    reference_cameras: Path
    reference_selection: Path
    views: tuple[SourceView, ...]


@dataclass(frozen=True)
class PipelinePaths:
    pipeline_dir: Path
    reports_dir: Path
    stages_dir: Path
    approvals_dir: Path
    graph_candidates_dir: Path
    state_path: Path
    resolved_config_path: Path
    events_path: Path
    final_ready_path: Path
    static_work_dir: Path
    accepted_static_root: Path
    modal_inputs_root: Path
    dynamic_dataset: Path
    frame_names_dir: Path
    roi_union_dir: Path
    flow_root: Path
    view_configs_dir: Path
    topology_path: Path
    frequency_dir: Path
    modal_analysis_dir: Path
    observations_dir: Path
    motion_fill_graph_path: Path
    gaussian_sidecar_path: Path
    modal_fields_dir: Path
    rendered_design_dir: Path
    flow_coordinates_dir: Path
    physics_coordinates_dir: Path
    direct_checkpoint_dir: Path
    physics_checkpoint_dir: Path


@dataclass(frozen=True)
class GraphCandidateParameters:
    candidate_id: str
    max_distance: float
    max_neighbors: int
    color_mad_multiplier: float
    depth_mad_multiplier: float
    depth_samples: int
    min_shared_views: int
    min_component_nodes: int
    min_component_edges: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _validate_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    label: str,
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing or extra:
        raise ValueError(
            f"{label} fields do not match the strict schema: "
            f"missing={missing}, extra={extra}"
        )


def _parse_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must start with a letter or digit and contain only "
            "letters, digits, underscores, or hyphens"
        )
    return value


def _parse_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _parse_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _parse_float(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            relation = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{label} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


def _config_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _section(payload: Mapping[str, Any], name: str, fields: set[str]) -> Mapping[str, Any]:
    value = _require_mapping(payload.get(name), name)
    _validate_keys(value, required=fields, label=name)
    return value


def load_config(path_value: str | Path) -> PipelineConfig:
    path = Path(path_value).expanduser().resolve(strict=True)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    payload = _require_mapping(loaded, "config")
    root_fields = {
        "format",
        "version",
        "scene_id",
        "scene_root",
        "pipeline_id",
        "prestatic_run_id",
        "static",
        "modal_inputs",
        "view_configs",
        "topology",
        "frequency_selection",
        "solver",
        "rigid_graph",
        "motion_fill",
        "rendered_design",
        "flow_coordinates",
        "physics",
        "checkpoints",
    }
    _validate_keys(payload, required=root_fields, label="config")
    if payload["format"] != CONFIG_FORMAT or payload["version"] != CONFIG_VERSION:
        raise ValueError(
            f"Config must use format={CONFIG_FORMAT!r}, version={CONFIG_VERSION}"
        )
    scene_id = _parse_id(payload["scene_id"], "scene_id")
    scene_root_value = payload["scene_root"]
    if not isinstance(scene_root_value, str) or not scene_root_value:
        raise ValueError("scene_root must be a non-empty path")
    scene_root = Path(scene_root_value).expanduser().resolve()
    if scene_root.name != scene_id:
        raise ValueError(
            f"scene_root basename {scene_root.name!r} must equal scene_id {scene_id!r}"
        )

    static_value = _section(
        payload,
        "static",
        {
            "artifact_id",
            "initial_epochs",
            "num_fg",
            "num_bg",
            "batch_size",
            "num_workers",
            "use_2dgs",
        },
    )
    static = StaticConfig(
        artifact_id=_parse_id(static_value["artifact_id"], "static.artifact_id"),
        initial_epochs=_parse_int(static_value["initial_epochs"], "static.initial_epochs"),
        num_fg=_parse_int(static_value["num_fg"], "static.num_fg"),
        num_bg=_parse_int(static_value["num_bg"], "static.num_bg", minimum=0),
        batch_size=_parse_int(static_value["batch_size"], "static.batch_size"),
        num_workers=_parse_int(static_value["num_workers"], "static.num_workers", minimum=0),
        use_2dgs=_parse_bool(static_value["use_2dgs"], "static.use_2dgs"),
    )

    inputs_value = _section(
        payload,
        "modal_inputs",
        {
            "artifact_id",
            "flow_artifact_id",
            "no_smooth",
            "sigma_b",
            "sigma_c",
            "analysis_mask_dilate_iters",
        },
    )
    modal_inputs = ModalInputsConfig(
        artifact_id=_parse_id(inputs_value["artifact_id"], "modal_inputs.artifact_id"),
        flow_artifact_id=_parse_id(
            inputs_value["flow_artifact_id"], "modal_inputs.flow_artifact_id"
        ),
        no_smooth=_parse_bool(inputs_value["no_smooth"], "modal_inputs.no_smooth"),
        sigma_b=_parse_float(inputs_value["sigma_b"], "modal_inputs.sigma_b", minimum=0.0),
        sigma_c=_parse_float(inputs_value["sigma_c"], "modal_inputs.sigma_c", minimum=0.0),
        analysis_mask_dilate_iters=_parse_int(
            inputs_value["analysis_mask_dilate_iters"],
            "modal_inputs.analysis_mask_dilate_iters",
            minimum=0,
        ),
    )

    view_value = _section(payload, "view_configs", {"artifact_id"})
    view_configs = ViewConfigsConfig(
        artifact_id=_parse_id(view_value["artifact_id"], "view_configs.artifact_id")
    )

    topology_value = _section(
        payload,
        "topology",
        {
            "artifact_id",
            "pixel_sample_stride",
            "pixel_candidate_k",
            "pixel_preselect_k",
            "pixel_render_acc_min",
            "pixel_min_contribution",
            "mask_erode_iters",
        },
    )
    topology = TopologyConfig(
        artifact_id=_parse_id(topology_value["artifact_id"], "topology.artifact_id"),
        pixel_sample_stride=_parse_int(
            topology_value["pixel_sample_stride"], "topology.pixel_sample_stride"
        ),
        pixel_candidate_k=_parse_int(
            topology_value["pixel_candidate_k"], "topology.pixel_candidate_k"
        ),
        pixel_preselect_k=_parse_int(
            topology_value["pixel_preselect_k"], "topology.pixel_preselect_k"
        ),
        pixel_render_acc_min=_parse_float(
            topology_value["pixel_render_acc_min"],
            "topology.pixel_render_acc_min",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        pixel_min_contribution=_parse_float(
            topology_value["pixel_min_contribution"],
            "topology.pixel_min_contribution",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        mask_erode_iters=_parse_int(
            topology_value["mask_erode_iters"],
            "topology.mask_erode_iters",
            minimum=0,
        ),
    )
    if topology.pixel_preselect_k < topology.pixel_candidate_k:
        raise ValueError("topology.pixel_preselect_k must be >= pixel_candidate_k")

    frequency_value = _section(
        payload,
        "frequency_selection",
        {
            "artifact_id",
            "basis_id",
            "min_freq_hz",
            "max_freq_hz",
            "frequency_step_hz",
            "mode_counts",
            "selected_k",
            "pixel_stride",
        },
    )
    mode_counts_raw = frequency_value["mode_counts"]
    if not isinstance(mode_counts_raw, list) or not mode_counts_raw:
        raise ValueError("frequency_selection.mode_counts must be a non-empty list")
    mode_counts = tuple(
        _parse_int(value, f"frequency_selection.mode_counts[{index}]")
        for index, value in enumerate(mode_counts_raw)
    )
    if any(right <= left for left, right in zip(mode_counts, mode_counts[1:])):
        raise ValueError("frequency_selection.mode_counts must be strictly increasing")
    frequency = FrequencyConfig(
        artifact_id=_parse_id(
            frequency_value["artifact_id"], "frequency_selection.artifact_id"
        ),
        basis_id=_parse_id(frequency_value["basis_id"], "frequency_selection.basis_id"),
        min_freq_hz=_parse_float(
            frequency_value["min_freq_hz"],
            "frequency_selection.min_freq_hz",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        max_freq_hz=_parse_float(
            frequency_value["max_freq_hz"],
            "frequency_selection.max_freq_hz",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        frequency_step_hz=_parse_float(
            frequency_value["frequency_step_hz"],
            "frequency_selection.frequency_step_hz",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        mode_counts=mode_counts,
        selected_k=_parse_int(
            frequency_value["selected_k"], "frequency_selection.selected_k"
        ),
        pixel_stride=_parse_int(
            frequency_value["pixel_stride"], "frequency_selection.pixel_stride"
        ),
    )
    if frequency.max_freq_hz <= frequency.min_freq_hz:
        raise ValueError("frequency_selection.max_freq_hz must exceed min_freq_hz")
    if frequency.selected_k not in frequency.mode_counts:
        raise ValueError("frequency_selection.selected_k must occur in mode_counts")
    if frequency.pixel_stride != topology.pixel_sample_stride:
        raise ValueError(
            "frequency_selection.pixel_stride must equal topology.pixel_sample_stride"
        )

    solver_value = _section(
        payload,
        "solver",
        {
            "artifact_id",
            "method",
            "freq_tolerance_hz",
            "alpha_model",
            "alpha_gain_min",
            "alpha_gain_max",
            "alpha_min_shared_points",
            "alpha_rank_ratio_min",
            "alpha_info_ratio_min",
            "alpha_failure",
            "anchor_svd_ratio_min",
            "anchor_residual_max",
            "rigid_component_rcond",
            "rigid_seed_min_valid_views",
            "rigid_seed_min_secondary_view_node_ratio",
            "rigid_seed_min_singular_ratio",
            "rigid_seed_max_finite_drift",
            "rigid_single_view_observable_ratio",
            "rigid_single_view_ray_direction_min_fraction",
        },
    )
    method = solver_value["method"]
    if method not in {"staged", "rigid-components"}:
        raise ValueError("solver.method must be 'staged' or 'rigid-components'")
    alpha_model = solver_value["alpha_model"]
    if alpha_model not in {"phase", "bounded-complex"}:
        raise ValueError("solver.alpha_model must be 'phase' or 'bounded-complex'")
    alpha_failure = solver_value["alpha_failure"]
    if alpha_failure not in {"exclude", "error"}:
        raise ValueError("solver.alpha_failure must be 'exclude' or 'error'")
    solver = SolverConfig(
        artifact_id=_parse_id(solver_value["artifact_id"], "solver.artifact_id"),
        method=str(method),
        freq_tolerance_hz=_parse_float(
            solver_value["freq_tolerance_hz"],
            "solver.freq_tolerance_hz",
            minimum=0.0,
        ),
        alpha_model=str(alpha_model),
        alpha_gain_min=_parse_float(
            solver_value["alpha_gain_min"], "solver.alpha_gain_min", minimum=0.0,
            minimum_inclusive=False,
        ),
        alpha_gain_max=_parse_float(
            solver_value["alpha_gain_max"], "solver.alpha_gain_max", minimum=0.0,
            minimum_inclusive=False,
        ),
        alpha_min_shared_points=_parse_int(
            solver_value["alpha_min_shared_points"], "solver.alpha_min_shared_points"
        ),
        alpha_rank_ratio_min=_parse_float(
            solver_value["alpha_rank_ratio_min"],
            "solver.alpha_rank_ratio_min",
            minimum=0.0,
        ),
        alpha_info_ratio_min=_parse_float(
            solver_value["alpha_info_ratio_min"],
            "solver.alpha_info_ratio_min",
            minimum=0.0,
        ),
        alpha_failure=str(alpha_failure),
        anchor_svd_ratio_min=_parse_float(
            solver_value["anchor_svd_ratio_min"],
            "solver.anchor_svd_ratio_min",
            minimum=0.0,
            maximum=1.0,
        ),
        anchor_residual_max=_parse_float(
            solver_value["anchor_residual_max"],
            "solver.anchor_residual_max",
            minimum=0.0,
        ),
        rigid_component_rcond=_parse_float(
            solver_value["rigid_component_rcond"],
            "solver.rigid_component_rcond",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        rigid_seed_min_valid_views=_parse_int(
            solver_value["rigid_seed_min_valid_views"],
            "solver.rigid_seed_min_valid_views",
        ),
        rigid_seed_min_secondary_view_node_ratio=_parse_float(
            solver_value["rigid_seed_min_secondary_view_node_ratio"],
            "solver.rigid_seed_min_secondary_view_node_ratio",
            minimum=0.0,
            maximum=1.0,
        ),
        rigid_seed_min_singular_ratio=_parse_float(
            solver_value["rigid_seed_min_singular_ratio"],
            "solver.rigid_seed_min_singular_ratio",
            minimum=0.0,
            maximum=1.0,
        ),
        rigid_seed_max_finite_drift=_parse_float(
            solver_value["rigid_seed_max_finite_drift"],
            "solver.rigid_seed_max_finite_drift",
            minimum=0.0,
        ),
        rigid_single_view_observable_ratio=_parse_float(
            solver_value["rigid_single_view_observable_ratio"],
            "solver.rigid_single_view_observable_ratio",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        rigid_single_view_ray_direction_min_fraction=_parse_float(
            solver_value["rigid_single_view_ray_direction_min_fraction"],
            "solver.rigid_single_view_ray_direction_min_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
    )
    if solver.alpha_gain_max < solver.alpha_gain_min:
        raise ValueError("solver.alpha_gain_max must be >= alpha_gain_min")

    graph_value = _section(
        payload,
        "rigid_graph",
        {
            "default_candidate_id",
            "max_distance",
            "max_neighbors",
            "color_mad_multiplier",
            "depth_mad_multiplier",
            "depth_samples",
            "min_shared_views",
            "min_component_nodes",
            "min_component_edges",
        },
    )
    rigid_graph = RigidGraphConfig(
        default_candidate_id=_parse_id(
            graph_value["default_candidate_id"], "rigid_graph.default_candidate_id"
        ),
        max_distance=_parse_float(
            graph_value["max_distance"], "rigid_graph.max_distance", minimum=0.0,
            minimum_inclusive=False,
        ),
        max_neighbors=_parse_int(graph_value["max_neighbors"], "rigid_graph.max_neighbors"),
        color_mad_multiplier=_parse_float(
            graph_value["color_mad_multiplier"],
            "rigid_graph.color_mad_multiplier",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        depth_mad_multiplier=_parse_float(
            graph_value["depth_mad_multiplier"],
            "rigid_graph.depth_mad_multiplier",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        depth_samples=_parse_int(
            graph_value["depth_samples"], "rigid_graph.depth_samples", minimum=2
        ),
        min_shared_views=_parse_int(
            graph_value["min_shared_views"], "rigid_graph.min_shared_views"
        ),
        min_component_nodes=_parse_int(
            graph_value["min_component_nodes"], "rigid_graph.min_component_nodes"
        ),
        min_component_edges=_parse_int(
            graph_value["min_component_edges"], "rigid_graph.min_component_edges"
        ),
    )

    motion_value = _section(
        payload,
        "motion_fill",
        {"artifact_id", "k", "max_distance", "max_anchor_hops"},
    )
    motion_fill = MotionFillConfig(
        artifact_id=_parse_id(motion_value["artifact_id"], "motion_fill.artifact_id"),
        k=_parse_int(motion_value["k"], "motion_fill.k"),
        max_distance=_parse_float(
            motion_value["max_distance"], "motion_fill.max_distance", minimum=0.0,
            minimum_inclusive=False,
        ),
        max_anchor_hops=_parse_int(
            motion_value["max_anchor_hops"], "motion_fill.max_anchor_hops"
        ),
    )

    rendered_value = _section(
        payload,
        "rendered_design",
        {
            "artifact_id",
            "pixel_sample_stride",
            "alpha_min",
            "mask_erode_iters",
            "modes_per_batch",
            "use_2dgs",
            "write_role_diagnostics",
        },
    )
    rendered_design = RenderedDesignConfig(
        artifact_id=_parse_id(
            rendered_value["artifact_id"], "rendered_design.artifact_id"
        ),
        pixel_sample_stride=_parse_int(
            rendered_value["pixel_sample_stride"],
            "rendered_design.pixel_sample_stride",
        ),
        alpha_min=_parse_float(
            rendered_value["alpha_min"],
            "rendered_design.alpha_min",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        mask_erode_iters=_parse_int(
            rendered_value["mask_erode_iters"],
            "rendered_design.mask_erode_iters",
            minimum=0,
        ),
        modes_per_batch=_parse_int(
            rendered_value["modes_per_batch"], "rendered_design.modes_per_batch"
        ),
        use_2dgs=_parse_bool(rendered_value["use_2dgs"], "rendered_design.use_2dgs"),
        write_role_diagnostics=_parse_bool(
            rendered_value["write_role_diagnostics"],
            "rendered_design.write_role_diagnostics",
        ),
    )
    if rendered_design.use_2dgs != static.use_2dgs:
        raise ValueError("rendered_design.use_2dgs must match static.use_2dgs")

    flow_value = _section(
        payload,
        "flow_coordinates",
        {"artifact_id", "ridge_relative", "frame_chunk_size"},
    )
    flow_coordinates = FlowCoordinatesConfig(
        artifact_id=_parse_id(
            flow_value["artifact_id"], "flow_coordinates.artifact_id"
        ),
        ridge_relative=_parse_float(
            flow_value["ridge_relative"],
            "flow_coordinates.ridge_relative",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        frame_chunk_size=_parse_int(
            flow_value["frame_chunk_size"], "flow_coordinates.frame_chunk_size"
        ),
    )

    physics_value = _section(
        payload,
        "physics",
        {
            "enabled",
            "artifact_id",
            "damping_ratio",
            "forcing_weight",
            "forcing_difference_weight",
            "assigned_band_half_width_hz",
            "frame_chunk_size",
        },
    )
    physics = PhysicsConfig(
        enabled=_parse_bool(physics_value["enabled"], "physics.enabled"),
        artifact_id=_parse_id(physics_value["artifact_id"], "physics.artifact_id"),
        damping_ratio=_parse_float(
            physics_value["damping_ratio"], "physics.damping_ratio", minimum=0.0
        ),
        forcing_weight=_parse_float(
            physics_value["forcing_weight"], "physics.forcing_weight", minimum=0.0
        ),
        forcing_difference_weight=_parse_float(
            physics_value["forcing_difference_weight"],
            "physics.forcing_difference_weight",
            minimum=0.0,
        ),
        assigned_band_half_width_hz=_parse_float(
            physics_value["assigned_band_half_width_hz"],
            "physics.assigned_band_half_width_hz",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        frame_chunk_size=_parse_int(
            physics_value["frame_chunk_size"], "physics.frame_chunk_size"
        ),
    )

    checkpoint_value = _section(
        payload,
        "checkpoints",
        {"direct_artifact_id", "physics_artifact_id"},
    )
    checkpoints = CheckpointsConfig(
        direct_artifact_id=_parse_id(
            checkpoint_value["direct_artifact_id"],
            "checkpoints.direct_artifact_id",
        ),
        physics_artifact_id=_parse_id(
            checkpoint_value["physics_artifact_id"],
            "checkpoints.physics_artifact_id",
        ),
    )
    if (
        physics.enabled
        and checkpoints.direct_artifact_id == checkpoints.physics_artifact_id
    ):
        raise ValueError(
            "checkpoints.direct_artifact_id and physics_artifact_id must differ "
            "when physics is enabled"
        )
    if static.use_2dgs:
        raise ValueError(
            "static.use_2dgs=true is not supported by this pipeline because the "
            "COLMAP topology and rigid-graph geometry stages currently use 3DGS"
        )

    return PipelineConfig(
        config_path=path,
        config_identity=_config_hash(payload),
        scene_id=scene_id,
        scene_root=scene_root,
        pipeline_id=_parse_id(payload["pipeline_id"], "pipeline_id"),
        prestatic_run_id=_parse_id(payload["prestatic_run_id"], "prestatic_run_id"),
        static=static,
        modal_inputs=modal_inputs,
        view_configs=view_configs,
        topology=topology,
        frequency=frequency,
        solver=solver,
        rigid_graph=rigid_graph,
        motion_fill=motion_fill,
        rendered_design=rendered_design,
        flow_coordinates=flow_coordinates,
        physics=physics,
        checkpoints=checkpoints,
    )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, payload: Any) -> None:
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
            handle.write("\n")
        assert temporary is not None
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_yaml_atomic(path: Path, payload: Any) -> None:
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
            yaml.dump(payload, handle, default_flow_style=False, sort_keys=False)
        assert temporary is not None
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _path_fingerprint(path: Path) -> dict[str, Any]:
    if path.is_file():
        stat = path.stat()
        payload: dict[str, Any] = {
            "kind": "file",
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        if stat.st_size <= 128 * 1024 * 1024:
            payload["sha256"] = _sha256_file(path)
        return payload
    if not path.is_dir():
        raise FileNotFoundError(path)
    entries: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        stat = item.stat()
        entry: dict[str, Any] = {
            "path": item.relative_to(path).as_posix(),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        if (
            stat.st_size <= 1024 * 1024
            and item.suffix.lower() in {".json", ".yaml", ".yml", ".txt"}
        ):
            entry["sha256"] = _sha256_file(item)
        entries.append(entry)
    return {
        "kind": "directory",
        "file_count": len(entries),
        "entries_identity": _config_hash({"entries": entries}),
    }


def _pipeline_paths(config: PipelineConfig) -> PipelinePaths:
    scene = config.scene_root
    pipeline_dir = scene / "shared" / "pipeline_runs" / config.pipeline_id
    reports = pipeline_dir / "reports"
    inputs_root = scene / "shared" / "inputs" / config.modal_inputs.artifact_id
    basis_root = scene / "bases" / config.frequency.basis_id
    return PipelinePaths(
        pipeline_dir=pipeline_dir,
        reports_dir=reports,
        stages_dir=reports / "stages",
        approvals_dir=reports / "approvals",
        graph_candidates_dir=(
            scene
            / "shared"
            / "preprocessing"
            / "observed_structure_graphs"
            / config.pipeline_id
            / "candidates"
        ),
        state_path=reports / "pipeline_state.json",
        resolved_config_path=reports / "resolved_config.json",
        events_path=reports / "events.jsonl",
        final_ready_path=reports / "FINAL_VISUALIZATION_READY.json",
        static_work_dir=scene / "static_3dgs" / config.static.artifact_id,
        accepted_static_root=(
            scene
            / "shared"
            / "inputs"
            / "accepted_static_checkpoints"
            / config.pipeline_id
        ),
        modal_inputs_root=inputs_root,
        dynamic_dataset=inputs_root / "dynamic_dataset",
        frame_names_dir=inputs_root / "frame_names",
        roi_union_dir=inputs_root / "roi_union",
        flow_root=scene / "modal_2d" / config.modal_inputs.flow_artifact_id,
        view_configs_dir=(
            scene / "shared" / "inputs" / "view_configs" / config.view_configs.artifact_id
        ),
        topology_path=(
            scene
            / "shared"
            / "preprocessing"
            / "gaussian_observation_topology"
            / f"{config.topology.artifact_id}.npz"
        ),
        frequency_dir=(
            scene
            / "shared"
            / "preprocessing"
            / "frequency_selection"
            / config.frequency.artifact_id
        ),
        modal_analysis_dir=basis_root / "modal_analysis",
        observations_dir=(
            scene
            / "shared"
            / "preprocessing"
            / "gaussian_observations"
            / config.frequency.basis_id
        ),
        motion_fill_graph_path=(
            scene
            / "shared"
            / "preprocessing"
            / "motion_fill_graphs"
            / config.motion_fill.artifact_id
            / "motion_fill_graph.npz"
        ),
        gaussian_sidecar_path=(
            scene
            / "shared"
            / "preprocessing"
            / "graph_visualization"
            / config.pipeline_id
            / "static_gaussians_foreground.npz"
        ),
        modal_fields_dir=(
            scene
            / "modal_fields"
            / config.frequency.basis_id
            / config.solver.artifact_id
        ),
        rendered_design_dir=(
            scene
            / "shared"
            / "preprocessing"
            / "rendered_modal_designs"
            / config.frequency.basis_id
            / config.rendered_design.artifact_id
        ),
        flow_coordinates_dir=(
            scene
            / "flow_coordinates"
            / config.frequency.basis_id
            / config.flow_coordinates.artifact_id
        ),
        physics_coordinates_dir=(
            scene
            / "physics_coordinates"
            / config.frequency.basis_id
            / config.physics.artifact_id
        ),
        direct_checkpoint_dir=(
            scene
            / "modal_checkpoints"
            / config.frequency.basis_id
            / config.checkpoints.direct_artifact_id
        ),
        physics_checkpoint_dir=(
            scene
            / "modal_checkpoints"
            / config.frequency.basis_id
            / config.checkpoints.physics_artifact_id
        ),
    )


def _parse_source_view(
    value: Any,
    references: Mapping[str, Mapping[str, Any]],
) -> SourceView:
    record = _require_mapping(value, "source manifest static view")
    required = {
        "view_id",
        "image_dir",
        "mask_dir",
        "fps_hz",
        "camera_group",
        "frame_count",
        "image_width",
        "image_height",
        "image_extension",
        "frame_names",
        "source_identity",
    }
    _validate_keys(record, required=required, label="source manifest static view")
    view_id = _parse_id(record["view_id"], "source view_id")
    if view_id not in references:
        raise ValueError(f"Reference selection is missing static view {view_id!r}")
    reference = references[view_id]
    frame_names_value = record["frame_names"]
    if (
        not isinstance(frame_names_value, list)
        or not frame_names_value
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or Path(name).suffix.lower() != ".png"
            for name in frame_names_value
        )
    ):
        raise ValueError(f"Source view {view_id!r} has invalid PNG frame_names")
    frame_names = tuple(frame_names_value)
    if len(set(frame_names)) != len(frame_names):
        raise ValueError(f"Source view {view_id!r} repeats frame names")
    frame_count = _parse_int(record["frame_count"], f"{view_id}.frame_count")
    if frame_count != len(frame_names):
        raise ValueError(f"Source view {view_id!r} frame_count does not match frame_names")
    reference_name = reference.get("source_frame_name")
    reference_index = reference.get("source_index")
    if reference_name not in frame_names:
        raise ValueError(f"Reference frame for {view_id!r} is not in source frame_names")
    if (
        isinstance(reference_index, bool)
        or not isinstance(reference_index, int)
        or reference_index < 0
        or reference_index >= len(frame_names)
        or frame_names[reference_index] != reference_name
    ):
        raise ValueError(f"Reference index for {view_id!r} is inconsistent")
    image_dir = Path(str(record["image_dir"])).expanduser().resolve(strict=True)
    mask_dir = Path(str(record["mask_dir"])).expanduser().resolve(strict=True)
    source_identity = str(record["source_identity"])
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            {
                "view_id": view_id,
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir),
                "fps_hz": float(record["fps_hz"]),
                "camera_group": record["camera_group"],
                "image_width": int(record["image_width"]),
                "image_height": int(record["image_height"]),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    for frame_name in frame_names:
        image_path = image_dir / frame_name
        mask_path = mask_dir / frame_name
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Canonical RGB/mask pair is missing for {view_id}/{frame_name}"
            )
        image_stat = image_path.stat()
        mask_stat = mask_path.stat()
        hasher.update(frame_name.encode("utf-8"))
        hasher.update(
            (
                f"{image_stat.st_size}:{image_stat.st_mtime_ns}:"
                f"{mask_stat.st_size}:{mask_stat.st_mtime_ns}"
            ).encode("utf-8")
        )
    if hasher.hexdigest() != source_identity:
        raise ValueError(
            f"Canonical source sequence changed after pre-static validation: {view_id}"
        )
    return SourceView(
        view_id=view_id,
        image_dir=image_dir,
        mask_dir=mask_dir,
        fps_hz=_parse_float(
            record["fps_hz"], f"{view_id}.fps_hz", minimum=0.0,
            minimum_inclusive=False,
        ),
        width=_parse_int(record["image_width"], f"{view_id}.image_width"),
        height=_parse_int(record["image_height"], f"{view_id}.image_height"),
        frame_names=frame_names,
        source_identity=source_identity,
        reference_frame_name=str(reference_name),
        reference_frame_stem=Path(str(reference_name)).stem,
        reference_local_index=int(reference_index),
    )


def load_prestatic_inputs(config: PipelineConfig) -> PrestaticInputs:
    run_dir = (
        config.scene_root
        / "shared"
        / "preprocessing"
        / config.prestatic_run_id
    )
    ready_path = run_dir / "reports" / "STATIC_DATASET_READY.json"
    if not ready_path.is_file():
        raise FileNotFoundError(
            f"Pre-static READY marker does not exist: {ready_path}"
        )
    ready = _require_mapping(_read_json(ready_path), "STATIC_DATASET_READY")
    if ready.get("format") != "static_colmap_dataset_ready" or ready.get("version") != 1:
        raise ValueError(f"Unsupported pre-static READY marker: {ready_path}")
    if ready.get("scene_id") != config.scene_id or ready.get("run_id") != config.prestatic_run_id:
        raise ValueError("Pre-static READY scene_id/run_id does not match config")
    static_dataset = Path(str(ready.get("static_dataset"))).expanduser().resolve(strict=True)
    source_manifest = Path(str(ready.get("source_manifest"))).expanduser().resolve(strict=True)
    reference_cameras = Path(str(ready.get("reference_cameras"))).expanduser().resolve(strict=True)
    if static_dataset != run_dir / "sweep_colmap_dataset":
        raise ValueError("Pre-static READY static_dataset is outside its run directory")
    if reference_cameras != run_dir / "references" / "reference_cameras.json":
        raise ValueError("Pre-static READY reference_cameras path is unexpected")
    reference_selection = run_dir / "reports" / "reference_selection.json"
    selection_payload = _require_mapping(
        _read_json(reference_selection), "reference_selection"
    )
    if (
        selection_payload.get("format") != "fixed_reference_selection"
        or selection_payload.get("version") != 1
    ):
        raise ValueError(f"Unsupported reference selection: {reference_selection}")
    selections = selection_payload.get("references")
    if not isinstance(selections, list) or not selections:
        raise ValueError("Reference selection contains no static views")
    selection_by_id: dict[str, Mapping[str, Any]] = {}
    for value in selections:
        record = _require_mapping(value, "reference selection record")
        view_id = record.get("view_id")
        if not isinstance(view_id, str) or view_id in selection_by_id:
            raise ValueError("Reference selection has invalid or duplicate view_id")
        selection_by_id[view_id] = record

    source_payload = _require_mapping(_read_json(source_manifest), "source manifest")
    if (
        source_payload.get("format") != "canonical_source_sequences"
        or source_payload.get("version") != 1
        or source_payload.get("scene_id") != config.scene_id
    ):
        raise ValueError(f"Unsupported source manifest: {source_manifest}")
    static_values = source_payload.get("static_views")
    if not isinstance(static_values, list) or not static_values:
        raise ValueError("Source manifest contains no static views")
    views = tuple(
        _parse_source_view(value, selection_by_id) for value in static_values
    )
    view_ids = tuple(view.view_id for view in views)
    if set(view_ids) != set(selection_by_id) or len(set(view_ids)) != len(view_ids):
        raise ValueError("Source manifest and reference selection view IDs differ")
    dimensions = {(view.width, view.height) for view in views}
    if len(dimensions) != 1:
        raise ValueError(
            f"Native modal pipeline requires one static-view resolution, got {dimensions}"
        )
    requirements = ready.get("static_training_requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("Pre-static READY is missing static training requirements")
    depth_weights = requirements.get("depth_loss_weights")
    if not isinstance(depth_weights, Mapping) or any(
        float(depth_weights.get(name, math.nan)) != 0.0
        for name in ("w_depth_reg", "w_depth_grad", "w_depth_const")
    ):
        raise ValueError("Pre-static READY does not declare zero static depth losses")
    return PrestaticInputs(
        ready_path=ready_path,
        static_dataset=static_dataset,
        source_manifest=source_manifest,
        reference_cameras=reference_cameras,
        reference_selection=reference_selection,
        views=views,
    )


def _resolved_config_payload(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
) -> dict[str, Any]:
    return {
        "format": CONFIG_FORMAT,
        "version": CONFIG_VERSION,
        "source_config": str(config.config_path),
        "config_identity": config.config_identity,
        "scene_id": config.scene_id,
        "scene_root": str(config.scene_root),
        "pipeline_id": config.pipeline_id,
        "prestatic_run_id": config.prestatic_run_id,
        "prestatic_ready": str(inputs.ready_path),
        "source_manifest": str(inputs.source_manifest),
        "reference_cameras": str(inputs.reference_cameras),
        "view_order": [view.view_id for view in inputs.views],
        "views": [
            {
                "view_id": view.view_id,
                "fps_hz": view.fps_hz,
                "width": view.width,
                "height": view.height,
                "frame_count": len(view.frame_names),
                "source_identity": view.source_identity,
                "reference_frame_name": view.reference_frame_name,
                "reference_local_index": view.reference_local_index,
            }
            for view in inputs.views
        ],
        "scientific_config": {
            "static": asdict(config.static),
            "modal_inputs": asdict(config.modal_inputs),
            "view_configs": asdict(config.view_configs),
            "topology": asdict(config.topology),
            "frequency_selection": {
                **asdict(config.frequency),
                "mode_counts": list(config.frequency.mode_counts),
            },
            "solver": asdict(config.solver),
            "rigid_graph": asdict(config.rigid_graph),
            "motion_fill": asdict(config.motion_fill),
            "rendered_design": asdict(config.rendered_design),
            "flow_coordinates": asdict(config.flow_coordinates),
            "physics": asdict(config.physics),
            "checkpoints": asdict(config.checkpoints),
        },
        "pipeline_dir": str(paths.pipeline_dir),
        "transforms": {
            "video_decode": False,
            "rotate": False,
            "crop": False,
            "resize": False,
        },
    }


def _initial_state(config: PipelineConfig) -> dict[str, Any]:
    return {
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        "config_identity": config.config_identity,
        "static_target_epochs": config.static.initial_epochs,
        "gate": None,
        "updated_at": _utc_now(),
    }


def _prepare_controller(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
) -> dict[str, Any]:
    resolved = _resolved_config_payload(config, inputs, paths)
    if paths.pipeline_dir.exists() and not paths.pipeline_dir.is_dir():
        raise ValueError(f"Pipeline output is not a directory: {paths.pipeline_dir}")
    paths.stages_dir.mkdir(parents=True, exist_ok=True)
    paths.approvals_dir.mkdir(parents=True, exist_ok=True)
    if paths.resolved_config_path.exists():
        if _read_json(paths.resolved_config_path) != resolved:
            raise ValueError(
                "Existing resolved pipeline config differs; use a new pipeline_id: "
                f"{paths.resolved_config_path}"
            )
    else:
        _write_json_atomic(paths.resolved_config_path, resolved)
    if paths.state_path.exists():
        state = _require_mapping(_read_json(paths.state_path), "pipeline state")
        if (
            state.get("format") != STATE_FORMAT
            or state.get("version") != STATE_VERSION
            or state.get("config_identity") != config.config_identity
        ):
            raise ValueError(f"Existing pipeline state is incompatible: {paths.state_path}")
        return dict(state)
    state = _initial_state(config)
    _write_json_atomic(paths.state_path, state)
    return state


def _save_state(paths: PipelinePaths, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    _write_json_atomic(paths.state_path, state)


def _append_event(paths: PipelinePaths, payload: Mapping[str, Any]) -> None:
    paths.events_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": _utc_now(), **dict(payload)}
    with paths.events_path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, allow_nan=False)
        handle.write("\n")


def _python_argv(script: str, *arguments: str) -> list[str]:
    return [sys.executable, "-u", str(REPO_ROOT / script), *arguments]


def _run_argv(
    paths: PipelinePaths,
    stage: str,
    argv: Sequence[str],
    *,
    dry_run: bool,
) -> None:
    printable = subprocess.list2cmdline(list(argv))
    if dry_run:
        print(f"[{stage}] {printable}")
        return
    _append_event(paths, {"event": "command_started", "stage": stage, "argv": list(argv)})
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not current_pythonpath
        else str(REPO_ROOT) + os.pathsep + current_pythonpath
    )
    print(f"[{stage}] {printable}", flush=True)
    result = subprocess.run(
        list(argv),
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )
    _append_event(
        paths,
        {
            "event": "command_finished",
            "stage": stage,
            "argv": list(argv),
            "returncode": int(result.returncode),
        },
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, list(argv))


def _receipt_path(paths: PipelinePaths, stage: str) -> Path:
    return paths.stages_dir / f"{stage}.json"


def _validate_receipt(
    config: PipelineConfig,
    receipt_path: Path,
    outputs: Sequence[Path],
    validator: Callable[[], Mapping[str, Any]],
) -> bool:
    if not receipt_path.is_file():
        return False
    receipt = _require_mapping(_read_json(receipt_path), "stage receipt")
    if (
        receipt.get("format") != RECEIPT_FORMAT
        or receipt.get("version") != RECEIPT_VERSION
        or receipt.get("config_identity") != config.config_identity
        or receipt.get("status") != "succeeded"
    ):
        raise ValueError(f"Stage receipt is incompatible: {receipt_path}")
    recorded = receipt.get("output_fingerprints")
    if not isinstance(recorded, list) or len(recorded) != len(outputs):
        raise ValueError(f"Stage receipt has invalid outputs: {receipt_path}")
    actual = [_path_fingerprint(path) for path in outputs]
    if actual != recorded:
        raise ValueError(f"Stage output changed after completion: {receipt_path}")
    validator()
    return True


def _write_receipt(
    config: PipelineConfig,
    paths: PipelinePaths,
    stage: str,
    argv: Sequence[str] | None,
    outputs: Sequence[Path],
    summary: Mapping[str, Any],
) -> None:
    _write_json_atomic(
        _receipt_path(paths, stage),
        {
            "format": RECEIPT_FORMAT,
            "version": RECEIPT_VERSION,
            "config_identity": config.config_identity,
            "stage": stage,
            "status": "succeeded",
            "argv": None if argv is None else list(argv),
            "outputs": [str(path) for path in outputs],
            "output_fingerprints": [_path_fingerprint(path) for path in outputs],
            "summary": dict(summary),
            "completed_at": _utc_now(),
        },
    )


def _run_command_stage(
    config: PipelineConfig,
    paths: PipelinePaths,
    *,
    stage: str,
    argv: Sequence[str],
    outputs: Sequence[Path],
    validator: Callable[[], Mapping[str, Any]],
    dry_run: bool,
    allow_existing_without_receipt: bool = True,
    run_if_existing_incomplete: bool = False,
) -> None:
    receipt = _receipt_path(paths, stage)
    if receipt.exists():
        if dry_run:
            print(f"[{stage}] already completed")
            return
        _validate_receipt(config, receipt, outputs, validator)
        print(f"[{stage}] validated existing output")
        return
    existing = [path for path in outputs if path.exists() or path.is_symlink()]
    if existing and not allow_existing_without_receipt:
        raise FileExistsError(
            f"Stage {stage!r} has output without a valid receipt: "
            + ", ".join(str(path) for path in existing)
        )
    if existing:
        if dry_run:
            print(f"[{stage}] would validate and adopt existing output")
            return
        if len(existing) != len(outputs):
            raise FileExistsError(
                f"Stage {stage!r} has only a subset of expected outputs: "
                + ", ".join(str(path) for path in existing)
            )
        try:
            summary = validator()
        except FileNotFoundError:
            if not run_if_existing_incomplete:
                raise
        else:
            _write_receipt(config, paths, stage, None, outputs, summary)
            print(f"[{stage}] adopted validated output and wrote a receipt")
            return
    _run_argv(paths, stage, argv, dry_run=dry_run)
    if dry_run:
        return
    summary = validator()
    _write_receipt(config, paths, stage, argv, outputs, summary)


def _replace_directory(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to replace existing target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def _run_temp_file_stage(
    config: PipelineConfig,
    paths: PipelinePaths,
    *,
    stage: str,
    build_argv: Callable[[Path], Sequence[str]],
    output: Path,
    validator: Callable[[Path], Mapping[str, Any]],
    dry_run: bool,
) -> None:
    receipt = _receipt_path(paths, stage)
    if receipt.exists():
        if dry_run:
            print(f"[{stage}] already completed")
            return
        _validate_receipt(config, receipt, [output], lambda: validator(output))
        print(f"[{stage}] validated existing output")
        return
    if output.exists() or output.is_symlink():
        if dry_run:
            print(f"[{stage}] would validate and adopt existing output")
            return
        summary = validator(output)
        _write_receipt(config, paths, stage, None, [output], summary)
        print(f"[{stage}] adopted validated output and wrote a receipt")
        return
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / (
        f".{output.stem}.{uuid4().hex}.attempt{output.suffix}"
    )
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Temporary stage target already exists: {temporary}")
    argv = list(build_argv(temporary))
    try:
        _run_argv(paths, stage, argv, dry_run=dry_run)
        if dry_run:
            return
        summary = validator(temporary)
        os.replace(temporary, output)
        validator(output)
        _write_receipt(config, paths, stage, argv, [output], summary)
    except BaseException:
        if not dry_run:
            temporary.unlink(missing_ok=True)
        raise


def _static_checkpoint_path(work_dir: Path) -> Path:
    return work_dir / "checkpoints" / "last.ckpt"


def _static_approval_path(paths: PipelinePaths) -> Path:
    return paths.approvals_dir / "static_quality.json"


def _static_completion_path(paths: PipelinePaths) -> Path:
    return paths.reports_dir / "static_training_completion.json"


def _graph_approval_path(paths: PipelinePaths) -> Path:
    return paths.approvals_dir / "rigid_graph_quality.json"


def _validate_static_candidate(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    work_dir: Path,
    target_epochs: int,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    cfg_path = work_dir / "cfg.yaml"
    checkpoint_path = _static_checkpoint_path(work_dir)
    if not cfg_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Static candidate requires cfg.yaml and checkpoints/last.ckpt: {work_dir}"
        )
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = _require_mapping(
            yaml.load(handle, Loader=yaml.FullLoader), "static cfg.yaml"
        )
    expected = {
        "trajectory_type": "static",
        "num_epochs": target_epochs,
        "num_fg": config.static.num_fg,
        "num_bg": config.static.num_bg,
        "batch_size": config.static.batch_size,
        "num_dl_workers": config.static.num_workers,
        "use_2dgs": config.static.use_2dgs,
    }
    mismatches = {
        key: (cfg.get(key), value)
        for key, value in expected.items()
        if cfg.get(key) != value
    }
    data_cfg = _require_mapping(cfg.get("data"), "static cfg.data")
    if Path(str(data_cfg.get("data_dir"))).resolve() != inputs.static_dataset:
        mismatches["data.data_dir"] = (data_cfg.get("data_dir"), str(inputs.static_dataset))
    for key, value in (("camera_type", "colmap"), ("load_from_cache", True)):
        if data_cfg.get(key) != value:
            mismatches[f"data.{key}"] = (data_cfg.get(key), value)
    loss_cfg = _require_mapping(cfg.get("loss"), "static cfg.loss")
    for key in ("w_mask", "w_depth_reg", "w_depth_grad", "w_depth_const"):
        if float(loss_cfg.get(key, math.nan)) != 0.0:
            mismatches[f"loss.{key}"] = (loss_cfg.get(key), 0.0)
    if mismatches:
        raise ValueError(f"Static candidate config mismatch: {mismatches}")

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError(f"Static checkpoint has no model state: {checkpoint_path}")
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError(f"Static checkpoint has invalid epoch: {checkpoint_path}")
    if require_complete and epoch < target_epochs - 1:
        raise ValueError(
            f"Static checkpoint reached epoch {epoch}, expected at least {target_epochs - 1}"
        )
    metadata = _require_mapping(checkpoint.get("init_metadata"), "static init_metadata")
    if metadata.get("trajectory_type") != "static":
        raise ValueError("Static checkpoint init_metadata trajectory_type changed")
    metadata_data = _require_mapping(metadata.get("data"), "static init_metadata.data")
    if (
        Path(str(metadata_data.get("data_dir"))).resolve() != inputs.static_dataset
        or metadata_data.get("camera_type") != "colmap"
    ):
        raise ValueError("Static checkpoint source dataset/camera does not match pre-static output")
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "cfg": str(cfg_path.resolve()),
        "epoch": int(epoch),
        "global_step": int(checkpoint.get("global_step", 0)),
        "target_epochs": target_epochs,
    }


def _static_training_argv(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    target_epochs: int,
    *,
    resume: bool,
) -> list[str]:
    argv = _python_argv(
        "run_training.py",
        "--work-dir",
        str(paths.static_work_dir),
        "--trajectory-type",
        "static",
        "--num-epochs",
        str(target_epochs),
        "--num-fg",
        str(config.static.num_fg),
        "--num-bg",
        str(config.static.num_bg),
        "--batch-size",
        str(config.static.batch_size),
        "--num-dl-workers",
        str(config.static.num_workers),
        "--loss.w-mask",
        "0",
        "--loss.w-depth-reg",
        "0",
        "--loss.w-depth-grad",
        "0",
        "--loss.w-depth-const",
        "0",
    )
    if config.static.use_2dgs:
        argv.append("--use-2dgs")
    if resume:
        argv.append("--resume")
    argv.extend(
        [
            "data:custom",
            "--data.data-dir",
            str(inputs.static_dataset),
            "--data.camera-type",
            "colmap",
            "--data.load-from-cache",
        ]
    )
    return argv


def _load_static_completion(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    target_epochs: int,
) -> dict[str, Any] | None:
    marker_path = _static_completion_path(paths)
    if not marker_path.is_file():
        return None
    marker = _require_mapping(_read_json(marker_path), "static completion marker")
    if marker.get("target_epochs") != target_epochs:
        return None
    expected = {
        "format": STATIC_COMPLETION_FORMAT,
        "version": 1,
        "config_identity": config.config_identity,
        "work_dir": str(paths.static_work_dir.resolve()),
        "cfg_path": str((paths.static_work_dir / "cfg.yaml").resolve()),
        "checkpoint_path": str(_static_checkpoint_path(paths.static_work_dir).resolve()),
        "source_ready": str(inputs.ready_path.resolve()),
        "source_ready_sha256": _sha256_file(inputs.ready_path),
    }
    mismatches = {
        key: (marker.get(key), value)
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Static completion marker mismatch: {mismatches}")
    summary = _validate_static_candidate(
        config, inputs, paths.static_work_dir, target_epochs
    )
    cfg_path = Path(summary["cfg"])
    checkpoint_path = Path(summary["checkpoint"])
    for key, actual in (
        ("cfg_sha256", _sha256_file(cfg_path)),
        ("checkpoint_sha256", _sha256_file(checkpoint_path)),
        ("epoch", summary["epoch"]),
        ("global_step", summary["global_step"]),
    ):
        if marker.get(key) != actual:
            raise ValueError(
                f"Static output changed after successful training: {key} "
                f"recorded={marker.get(key)!r}, actual={actual!r}"
            )
    return dict(marker)


def _write_static_completion(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    target_epochs: int,
    argv: Sequence[str],
    summary: Mapping[str, Any],
) -> None:
    cfg_path = Path(str(summary["cfg"])).resolve()
    checkpoint_path = Path(str(summary["checkpoint"])).resolve()
    _write_json_atomic(
        _static_completion_path(paths),
        {
            "format": STATIC_COMPLETION_FORMAT,
            "version": 1,
            "config_identity": config.config_identity,
            "work_dir": str(paths.static_work_dir.resolve()),
            "target_epochs": target_epochs,
            "argv": list(argv),
            "source_ready": str(inputs.ready_path.resolve()),
            "source_ready_sha256": _sha256_file(inputs.ready_path),
            "cfg_path": str(cfg_path),
            "cfg_sha256": _sha256_file(cfg_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "epoch": int(summary["epoch"]),
            "global_step": int(summary["global_step"]),
            "completed_at": _utc_now(),
        },
    )


def _run_static_candidate(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    state: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    target_epochs = int(state["static_target_epochs"])
    if _load_static_completion(config, inputs, paths, target_epochs) is not None:
        print(f"[static_train] validated completed {target_epochs}-epoch candidate")
        return
    checkpoint = _static_checkpoint_path(paths.static_work_dir)
    resume = checkpoint.is_file()
    if resume:
        cfg_path = paths.static_work_dir / "cfg.yaml"
        with cfg_path.open("r", encoding="utf-8") as handle:
            existing_cfg = _require_mapping(
                yaml.load(handle, Loader=yaml.FullLoader), "static cfg.yaml"
            )
        existing_target = _parse_int(
            existing_cfg.get("num_epochs"), "existing static num_epochs"
        )
        if existing_target > target_epochs:
            raise ValueError(
                f"Static state target {target_epochs} is below existing cfg target {existing_target}"
            )
        summary = _validate_static_candidate(
            config,
            inputs,
            paths.static_work_dir,
            existing_target,
            require_complete=False,
        )
    argv = _static_training_argv(
        config, inputs, paths, target_epochs, resume=resume
    )
    _run_argv(paths, "static_train", argv, dry_run=dry_run)
    if dry_run:
        return
    summary = _validate_static_candidate(
        config, inputs, paths.static_work_dir, target_epochs
    )
    _write_static_completion(
        config, inputs, paths, target_epochs, argv, summary
    )
    _write_json_atomic(paths.reports_dir / "static_candidate.json", summary)


def _load_static_approval(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
) -> tuple[Path, Path] | None:
    approval_path = _static_approval_path(paths)
    if not approval_path.is_file():
        return None
    approval = _require_mapping(_read_json(approval_path), "static approval")
    if (
        approval.get("format") != STATIC_APPROVAL_FORMAT
        or approval.get("version") != 1
        or approval.get("config_identity") != config.config_identity
    ):
        raise ValueError(f"Static approval is incompatible: {approval_path}")
    checkpoint = Path(str(approval.get("accepted_checkpoint"))).resolve(strict=True)
    cfg_path = Path(str(approval.get("accepted_cfg"))).resolve(strict=True)
    if checkpoint.parent.parent != cfg_path.parent:
        raise ValueError("Static approval checkpoint/cfg are not one accepted bundle")
    if _sha256_file(checkpoint) != approval.get("checkpoint_sha256"):
        raise ValueError("Accepted static checkpoint changed after approval")
    if _sha256_file(cfg_path) != approval.get("cfg_sha256"):
        raise ValueError("Accepted static cfg changed after approval")
    _validate_static_candidate(
        config,
        inputs,
        cfg_path.parent,
        int(approval.get("target_epochs")),
    )
    return checkpoint, cfg_path


def _approve_static(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    state: dict[str, Any],
    note: str | None,
) -> None:
    approval_path = _static_approval_path(paths)
    if approval_path.exists():
        raise FileExistsError(f"Static quality is already approved: {approval_path}")
    target_epochs = int(state["static_target_epochs"])
    if _load_static_completion(config, inputs, paths, target_epochs) is None:
        raise ValueError(
            "Static training has no successful completion marker for the current target; "
            "run the pipeline before approving"
        )
    summary = _validate_static_candidate(
        config, inputs, paths.static_work_dir, target_epochs
    )
    source_checkpoint = Path(summary["checkpoint"])
    source_cfg = Path(summary["cfg"])
    checkpoint_sha = _sha256_file(source_checkpoint)
    bundle = paths.accepted_static_root / checkpoint_sha[:16]
    if bundle.exists():
        raise FileExistsError(f"Accepted static bundle already exists: {bundle}")
    temporary = bundle.parent / f".{bundle.name}.{uuid4().hex}.attempt"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        (temporary / "checkpoints").mkdir()
        shutil.copy2(source_cfg, temporary / "cfg.yaml")
        shutil.copy2(source_checkpoint, temporary / "checkpoints" / "last.ckpt")
        _replace_directory(temporary, bundle)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    accepted_checkpoint = bundle / "checkpoints" / "last.ckpt"
    accepted_cfg = bundle / "cfg.yaml"
    _write_json_atomic(
        approval_path,
        {
            "format": STATIC_APPROVAL_FORMAT,
            "version": 1,
            "config_identity": config.config_identity,
            "source_work_dir": str(paths.static_work_dir),
            "target_epochs": target_epochs,
            "accepted_checkpoint": str(accepted_checkpoint.resolve()),
            "accepted_cfg": str(accepted_cfg.resolve()),
            "checkpoint_sha256": _sha256_file(accepted_checkpoint),
            "cfg_sha256": _sha256_file(accepted_cfg),
            "note": note,
            "approved_at": _utc_now(),
        },
    )
    state["gate"] = None
    _save_state(paths, state)
    _append_event(paths, {"event": "static_quality_approved", "note": note})
    print(f"Approved and froze static checkpoint -> {accepted_checkpoint}")


def _index_files_by_stem(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {
            ".bmp",
            ".jpeg",
            ".jpg",
            ".png",
            ".tif",
            ".tiff",
        }:
            continue
        if path.stem in files:
            raise ValueError(f"Duplicate filename stem {path.stem!r} in {directory}")
        files[path.stem] = path
    return files


def _validate_native_inputs(
    inputs: PrestaticInputs,
    paths: PipelinePaths,
) -> dict[str, Any]:
    from preproc.prepare_modal_dynamic_dataset import validate_prepared_dataset

    validate_prepared_dataset(paths.dynamic_dataset)
    metadata = _require_mapping(
        _read_json(paths.dynamic_dataset / "metadata.json"),
        "modal dynamic metadata",
    )
    spatial = _require_mapping(metadata.get("spatial_processing"), "spatial_processing")
    if spatial.get("mode") != "native_resolution":
        raise ValueError("Modal dynamic dataset was not prepared at native resolution")
    frame_map = _require_mapping(
        _read_json(paths.dynamic_dataset / "modal_frame_map.json"), "modal_frame_map"
    )
    if frame_map.get("views") != [view.view_id for view in inputs.views]:
        raise ValueError("Modal dynamic dataset view order changed")
    for view in inputs.views:
        sidecar = paths.frame_names_dir / f"{view.view_id}.json"
        names = _read_json(sidecar)
        expected = [Path(name).stem for name in view.frame_names]
        if names != expected:
            raise ValueError(f"Frame sidecar changed for {view.view_id}")
        metadata_views = metadata.get("views")
        if not isinstance(metadata_views, list):
            raise ValueError("Modal dynamic metadata views are invalid")
        metadata_record = next(
            (
                record
                for record in metadata_views
                if isinstance(record, Mapping) and record.get("view_id") == view.view_id
            ),
            None,
        )
        if (
            metadata_record is None
            or Path(str(metadata_record.get("frame_names_json"))).resolve()
            != sidecar.resolve()
        ):
            raise ValueError(f"Modal dynamic sidecar provenance changed for {view.view_id}")
        roi_path = paths.roi_union_dir / f"{view.view_id}.png"
        if not roi_path.is_file():
            raise FileNotFoundError(roi_path)
        import cv2

        roi = cv2.imread(str(roi_path), cv2.IMREAD_UNCHANGED)
        if roi is None or roi.shape != (view.height, view.width):
            raise ValueError(f"ROI union shape changed for {view.view_id}")
        if set(int(value) for value in __import__("numpy").unique(roi)).difference({0, 255}):
            raise ValueError(f"ROI union is not binary for {view.view_id}")
        if not bool((roi > 0).any()):
            raise ValueError(f"ROI union is empty for {view.view_id}")
    return {
        "views": [view.view_id for view in inputs.views],
        "frame_count": int(metadata["frame_count"]),
        "resolution": {"width": inputs.views[0].width, "height": inputs.views[0].height},
        "spatial_processing": "native_resolution",
    }


def _prepare_native_inputs(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    *,
    dry_run: bool,
) -> None:
    stage = "native_modal_inputs"
    receipt = _receipt_path(paths, stage)
    if receipt.exists():
        if dry_run:
            print(f"[{stage}] already completed")
            return
        _validate_receipt(
            config,
            receipt,
            [paths.modal_inputs_root],
            lambda: _validate_native_inputs(inputs, paths),
        )
        print(f"[{stage}] validated existing output")
        return
    if paths.modal_inputs_root.exists() or paths.modal_inputs_root.is_symlink():
        if dry_run:
            print(f"[{stage}] would validate and adopt existing output")
            return
        summary = _validate_native_inputs(inputs, paths)
        _write_receipt(
            config, paths, stage, None, [paths.modal_inputs_root], summary
        )
        print(f"[{stage}] adopted validated output and wrote a receipt")
        return
    temporary = paths.modal_inputs_root.parent / (
        f".{paths.modal_inputs_root.name}.{uuid4().hex}.attempt"
    )
    if temporary.exists():
        raise FileExistsError(f"Native modal input attempt exists: {temporary}")
    argv = _python_argv("preproc/prepare_modal_dynamic_dataset.py")
    if dry_run:
        for view in inputs.views:
            sidecar = temporary / "frame_names" / f"{view.view_id}.json"
            argv.extend(
                [
                    "--view",
                    f"{view.view_id}={view.image_dir}={view.mask_dir}={sidecar}={view.fps_hz:.17g}",
                ]
            )
        argv.extend(["--native-resolution", "--out-dir", str(temporary / "dynamic_dataset")])
        _run_argv(paths, stage, argv, dry_run=True)
        return

    import cv2
    import numpy as np

    (temporary / "frame_names").mkdir(parents=True)
    (temporary / "roi_union").mkdir()
    try:
        for view in inputs.views:
            stems = [Path(name).stem for name in view.frame_names]
            sidecar = temporary / "frame_names" / f"{view.view_id}.json"
            _write_json_atomic(sidecar, stems)
            mask_files = _index_files_by_stem(view.mask_dir)
            if set(stems) != set(mask_files):
                missing = sorted(set(stems) - set(mask_files))[:8]
                extra = sorted(set(mask_files) - set(stems))[:8]
                raise ValueError(
                    f"Mask stems differ from canonical frames for {view.view_id}: "
                    f"missing={missing}, extra={extra}"
                )
            union = np.zeros((view.height, view.width), dtype=np.uint8)
            for stem in stems:
                mask = cv2.imread(str(mask_files[stem]), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise ValueError(f"Cannot decode mask: {mask_files[stem]}")
                if mask.ndim == 3:
                    mask = np.max(mask, axis=2)
                if mask.shape != union.shape:
                    raise ValueError(
                        f"Mask {mask_files[stem]} shape {mask.shape} != {union.shape}"
                    )
                union[mask > 0] = 255
            if not union.any():
                raise ValueError(f"ROI union is empty for {view.view_id}")
            if not cv2.imwrite(
                str(temporary / "roi_union" / f"{view.view_id}.png"), union
            ):
                raise OSError(f"Failed to write ROI union for {view.view_id}")
            argv.extend(
                [
                    "--view",
                    f"{view.view_id}={view.image_dir}={view.mask_dir}={sidecar}={view.fps_hz:.17g}",
                ]
            )
        argv.extend(["--native-resolution", "--out-dir", str(temporary / "dynamic_dataset")])
        _run_argv(paths, stage, argv, dry_run=False)
        metadata_path = temporary / "dynamic_dataset" / "metadata.json"
        dynamic_metadata = _require_mapping(
            _read_json(metadata_path), "temporary modal dynamic metadata"
        )
        metadata_views = dynamic_metadata.get("views")
        if not isinstance(metadata_views, list):
            raise ValueError("Temporary modal dynamic metadata views are invalid")
        for record in metadata_views:
            if not isinstance(record, dict):
                raise ValueError("Temporary modal dynamic view metadata is invalid")
            view_id = str(record.get("view_id"))
            if view_id not in {view.view_id for view in inputs.views}:
                raise ValueError(f"Temporary modal dynamic metadata has unknown view {view_id!r}")
            record["frame_names_json"] = str(
                paths.frame_names_dir / f"{view_id}.json"
            )
        _write_json_atomic(metadata_path, dynamic_metadata)
        _replace_directory(temporary, paths.modal_inputs_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    summary = _validate_native_inputs(inputs, paths)
    _write_receipt(config, paths, stage, argv, [paths.modal_inputs_root], summary)


def _flow_cache_path(paths: PipelinePaths, view: SourceView) -> Path:
    return paths.flow_root / view.view_id / "cache_farneback_v1"


def _validate_flow_cache(
    config: PipelineConfig,
    paths: PipelinePaths,
    view: SourceView,
) -> dict[str, Any]:
    from modal_peak_pick.core.cache import load_analysis_cache

    cache = load_analysis_cache(_flow_cache_path(paths, view))
    if cache.flow_u.shape != (len(view.frame_names), view.height, view.width):
        raise ValueError(f"Flow cache shape changed for {view.view_id}: {cache.flow_u.shape}")
    if abs(cache.fps - view.fps_hz) > 1e-9:
        raise ValueError(f"Flow cache FPS changed for {view.view_id}")
    analysis = _require_mapping(cache.metadata.get("analysis"), "flow analysis")
    if (
        analysis.get("reference_frame_index") != view.reference_local_index
        or analysis.get("reference_frame_name") != view.reference_frame_stem
        or analysis.get("flow_method") != "farneback"
    ):
        raise ValueError(f"Flow cache reference changed for {view.view_id}")
    smoothing = _require_mapping(analysis.get("smoothing"), "flow smoothing")
    expected_smoothing = {
        "disabled": config.modal_inputs.no_smooth,
        "sigma_b": config.modal_inputs.sigma_b,
        "sigma_c": config.modal_inputs.sigma_c,
        "analysis_mask_dilate_iters": config.modal_inputs.analysis_mask_dilate_iters,
    }
    if any(smoothing.get(key) != value for key, value in expected_smoothing.items()):
        raise ValueError(f"Flow cache smoothing settings changed for {view.view_id}")
    sources = _require_mapping(cache.metadata.get("sources"), "flow sources")
    if sources.get("input_type") != "image_sequence":
        raise ValueError(f"Flow cache for {view.view_id} is not image-sequence based")
    image_source = _require_mapping(sources.get("image_sequence"), "image_sequence")
    image_dir_source = _require_mapping(image_source.get("image_dir"), "image directory source")
    frame_names_source = _require_mapping(
        image_source.get("frame_names_json"), "frame names source"
    )
    mask_source = _require_mapping(sources.get("mask"), "flow mask source")
    if (
        Path(str(image_dir_source.get("resolved_path"))).resolve() != view.image_dir
        or Path(str(frame_names_source.get("resolved_path"))).resolve()
        != (paths.frame_names_dir / f"{view.view_id}.json").resolve()
        or Path(str(mask_source.get("resolved_path"))).resolve()
        != (paths.roi_union_dir / f"{view.view_id}.png").resolve()
    ):
        raise ValueError(f"Flow cache input paths changed for {view.view_id}")
    if image_source.get("ordered_frame_names") != [
        Path(name).stem for name in view.frame_names
    ]:
        raise ValueError(f"Flow cache frame order changed for {view.view_id}")
    if image_source.get("source_identity") != view.source_identity:
        raise ValueError(f"Flow cache source identity changed for {view.view_id}")
    if cache.mask is None or not bool(cache.mask.any()):
        raise ValueError(f"Flow cache mask is empty for {view.view_id}")
    return {
        "view_id": view.view_id,
        "frames": int(cache.flow_u.shape[0]),
        "width": int(cache.flow_u.shape[2]),
        "height": int(cache.flow_u.shape[1]),
        "fps_hz": cache.fps,
        "reference_frame": view.reference_frame_stem,
    }


def _prepare_flow_caches(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    *,
    dry_run: bool,
) -> None:
    for view in inputs.views:
        cache_dir = _flow_cache_path(paths, view)
        argv = _python_argv(
            "run_modal_peak_pick.py",
            "analyze",
            "--image-dir",
            str(view.image_dir),
            "--frame-names-json",
            str(paths.frame_names_dir / f"{view.view_id}.json"),
            "--fps",
            f"{view.fps_hz:.17g}",
            "--reference-frame",
            view.reference_frame_stem,
            "--source-identity",
            view.source_identity,
            "--cache-dir",
            str(cache_dir),
            "--mask",
            str(paths.roi_union_dir / f"{view.view_id}.png"),
            "--flow-method",
            "farneback",
            "--sigma-b",
            f"{config.modal_inputs.sigma_b:.17g}",
            "--sigma-c",
            f"{config.modal_inputs.sigma_c:.17g}",
            "--analysis-mask-dilate-iters",
            str(config.modal_inputs.analysis_mask_dilate_iters),
        )
        if config.modal_inputs.no_smooth:
            argv.append("--no-smooth")
        _run_command_stage(
            config,
            paths,
            stage=f"flow_cache_{view.view_id}",
            argv=argv,
            outputs=[cache_dir],
            validator=lambda view=view: _validate_flow_cache(config, paths, view),
            dry_run=dry_run,
        )


def _view_config_path(paths: PipelinePaths, view: SourceView) -> Path:
    return paths.view_configs_dir / f"{view.view_id}_config.json"


def _validate_view_configs(
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
) -> dict[str, Any]:
    import numpy as np
    from modal_surface.io import load_view_config

    metadata = _require_mapping(
        _read_json(paths.view_configs_dir / "metadata.json"), "view config metadata"
    )
    if metadata.get("view_order") != [view.view_id for view in inputs.views]:
        raise ValueError("View config order changed")
    if Path(str(metadata.get("source_checkpoint"))).resolve() != accepted_checkpoint:
        raise ValueError("View config static checkpoint changed")
    if Path(str(metadata.get("source_reference_cameras"))).resolve() != inputs.reference_cameras:
        raise ValueError("View config reference-camera source changed")
    spatial = _require_mapping(metadata.get("spatial_processing"), "view spatial processing")
    if spatial != {
        "mode": "native_resolution",
        "intrinsics_scaled": False,
        "roi_resize_allowed": False,
    }:
        raise ValueError("View configs were not prepared at native resolution")
    reference_payload = _require_mapping(
        _read_json(inputs.reference_cameras), "reference cameras"
    )
    references = _require_mapping(
        reference_payload.get("references"), "reference camera records"
    )
    view_metadata = {
        str(record.get("view_id")): record
        for record in metadata.get("views", [])
        if isinstance(record, Mapping)
    }
    for view in inputs.views:
        loaded = load_view_config(_view_config_path(paths, view))
        reference = _require_mapping(
            references.get(view.view_id), f"reference camera {view.view_id}"
        )
        if (
            loaded.view_id != view.view_id
            or loaded.image_width != view.width
            or loaded.image_height != view.height
            or not np.array_equal(loaded.K, np.asarray(reference.get("K")))
            or not np.array_equal(
                loaded.world_to_camera,
                np.asarray(reference.get("normalized_world_to_camera")),
            )
        ):
            raise ValueError(f"View config geometry changed for {view.view_id}")
        if loaded.mask_path != (paths.view_configs_dir / f"{view.view_id}_mask.npy"):
            raise ValueError(f"View config mask path changed for {view.view_id}")
        record = _require_mapping(
            view_metadata.get(view.view_id), f"view metadata {view.view_id}"
        )
        if Path(str(record.get("mask_source"))).resolve() != (
            paths.roi_union_dir / f"{view.view_id}.png"
        ).resolve():
            raise ValueError(f"View config mask source changed for {view.view_id}")
    return {
        "views": [view.view_id for view in inputs.views],
        "resolution": {"width": inputs.views[0].width, "height": inputs.views[0].height},
        "source_checkpoint": str(accepted_checkpoint),
    }


def _prepare_view_configs(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
    *,
    dry_run: bool,
) -> None:
    argv = _python_argv(
        "preproc/prepare_colmap_modal_view_configs.py",
        "--input-ckpt",
        str(accepted_checkpoint),
        "--reference-cameras",
        str(inputs.reference_cameras),
    )
    for view in inputs.views:
        argv.extend(
            [
                "--view",
                f"{view.view_id}={paths.roi_union_dir / f'{view.view_id}.png'}",
            ]
        )
    argv.extend(["--native-resolution", "--out-dir", str(paths.view_configs_dir)])
    _run_command_stage(
        config,
        paths,
        stage="colmap_modal_view_configs",
        argv=argv,
        outputs=[paths.view_configs_dir],
        validator=lambda: _validate_view_configs(inputs, paths, accepted_checkpoint),
        dry_run=dry_run,
    )


def _validate_topology(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
    topology_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from modal_surface.gaussian_observations import load_gaussian_observation_topology

    topology = load_gaussian_observation_topology(topology_path)
    view_ids = np.asarray(topology["view_ids"]).astype(str).tolist()
    if view_ids != [view.view_id for view in inputs.views]:
        raise ValueError("Observation topology view order changed")
    if Path(str(np.asarray(topology["source_checkpoint"]).item())).resolve() != accepted_checkpoint:
        raise ValueError("Observation topology source checkpoint changed")
    expected_view_configs = [
        _view_config_path(paths, view).resolve() for view in inputs.views
    ]
    actual_view_configs = [
        Path(value).resolve()
        for value in np.asarray(topology["source_view_configs"]).astype(str).tolist()
    ]
    if actual_view_configs != expected_view_configs:
        raise ValueError("Observation topology view-config sources changed")
    expected_integer_parameters = {
        "pixel_sample_stride": config.topology.pixel_sample_stride,
        "pixel_candidate_k": config.topology.pixel_candidate_k,
        "pixel_preselect_k": config.topology.pixel_preselect_k,
        "mask_erode_iters": config.topology.mask_erode_iters,
    }
    for name, expected in expected_integer_parameters.items():
        if int(np.asarray(topology[name]).item()) != expected:
            raise ValueError(f"Observation topology {name} changed")
    expected_float_parameters = {
        "pixel_render_acc_min": config.topology.pixel_render_acc_min,
        "pixel_min_contribution": config.topology.pixel_min_contribution,
    }
    for name, expected in expected_float_parameters.items():
        actual = np.float32(np.asarray(topology[name]).item())
        if actual != np.float32(expected):
            raise ValueError(f"Observation topology {name} changed")
    if np.asarray(topology["sample_view_index"]).size == 0:
        raise ValueError("Observation topology has no sampled pixels")
    return {
        "views": view_ids,
        "samples": int(np.asarray(topology["sample_view_index"]).size),
        "observations": int(np.asarray(topology["obs_point_index"]).size),
        "topology_id": str(np.asarray(topology["topology_id"]).item()),
    }


def _prepare_topology(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
    *,
    dry_run: bool,
) -> None:
    def build(output: Path) -> list[str]:
        argv = _python_argv(
            "preproc/build_gaussian_observation_topology.py",
            "--input-ckpt",
            str(accepted_checkpoint),
        )
        for view in inputs.views:
            argv.extend(["--view-config", str(_view_config_path(paths, view))])
        argv.extend(
            [
                "--out",
                str(output),
                "--pixel-sample-stride",
                str(config.topology.pixel_sample_stride),
                "--pixel-candidate-k",
                str(config.topology.pixel_candidate_k),
                "--pixel-preselect-k",
                str(config.topology.pixel_preselect_k),
                "--pixel-render-acc-min",
                f"{config.topology.pixel_render_acc_min:.17g}",
                "--pixel-min-contribution",
                f"{config.topology.pixel_min_contribution:.17g}",
                "--mask-erode-iters",
                str(config.topology.mask_erode_iters),
            ]
        )
        return argv

    _run_temp_file_stage(
        config,
        paths,
        stage="gaussian_observation_topology",
        build_argv=build,
        output=paths.topology_path,
        validator=lambda output: _validate_topology(
            config, inputs, paths, accepted_checkpoint, output
        ),
        dry_run=dry_run,
    )


def _selected_frequency_path(config: PipelineConfig, paths: PipelinePaths) -> Path:
    return paths.frequency_dir / f"selected_frequencies_k{config.frequency.selected_k}.json"


def _load_selected_frequencies(config: PipelineConfig, paths: PipelinePaths) -> list[float]:
    payload = _require_mapping(
        _read_json(_selected_frequency_path(config, paths)), "selected frequencies"
    )
    values = payload.get("selected_peaks_hz")
    if not isinstance(values, list) or len(values) != config.frequency.selected_k:
        raise ValueError("Selected-frequency shortlist has the wrong mode count")
    frequencies = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0.0 for value in frequencies):
        raise ValueError("Selected-frequency shortlist is not finite and positive")
    return frequencies


def _validate_frequency_selection(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
) -> dict[str, Any]:
    import numpy as np
    from flow3d.modal_frequency_selection import _validate_output

    _validate_output(
        paths.frequency_dir,
        np.asarray(config.frequency.mode_counts, dtype=np.int32),
        len(inputs.views),
    )
    summary = _require_mapping(
        _read_json(paths.frequency_dir / "frequency_selection_summary.json"),
        "frequency selection summary",
    )
    settings = _require_mapping(summary.get("settings"), "frequency settings")
    expected_settings = {
        "min_freq_hz": config.frequency.min_freq_hz,
        "max_freq_hz": config.frequency.max_freq_hz,
        "frequency_step_hz": config.frequency.frequency_step_hz,
        "pixel_stride": config.frequency.pixel_stride,
    }
    if any(settings.get(key) != value for key, value in expected_settings.items()):
        raise ValueError("Frequency-selection scientific settings changed")
    source = _require_mapping(summary.get("source"), "frequency source")
    expected_source_paths = {
        "observation_topology": paths.topology_path.resolve(),
        "modal_frame_map": (paths.dynamic_dataset / "modal_frame_map.json").resolve(),
    }
    for key, expected in expected_source_paths.items():
        if Path(str(source.get(key))).resolve() != expected:
            raise ValueError(f"Frequency-selection {key} source changed")
    expected_caches = [
        _flow_cache_path(paths, view).resolve() for view in inputs.views
    ]
    actual_caches = [
        Path(value).resolve() for value in source.get("flow_caches", [])
    ]
    if actual_caches != expected_caches:
        raise ValueError("Frequency-selection flow-cache sources changed")
    if source.get("topology_view_ids") != [view.view_id for view in inputs.views]:
        raise ValueError("Frequency-selection topology view order changed")
    frequencies = _load_selected_frequencies(config, paths)
    return {
        "selected_k": len(frequencies),
        "selected_frequencies_hz": frequencies,
        "views": [view.view_id for view in inputs.views],
    }


def _prepare_frequency_selection(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    *,
    dry_run: bool,
) -> None:
    argv = _python_argv(
        "run_modal_frequency_selection.py",
        "--flow-caches",
        *[
            f"{view.view_id}={_flow_cache_path(paths, view)}"
            for view in inputs.views
        ],
    )
    argv.extend(
        [
            "--observation-topology",
            str(paths.topology_path),
            "--modal-frame-map",
            str(paths.dynamic_dataset / "modal_frame_map.json"),
            "--out-dir",
            str(paths.frequency_dir),
            "--min-freq-hz",
            f"{config.frequency.min_freq_hz:.17g}",
            "--max-freq-hz",
            f"{config.frequency.max_freq_hz:.17g}",
            "--frequency-step-hz",
            f"{config.frequency.frequency_step_hz:.17g}",
            "--mode-counts",
            *[str(value) for value in config.frequency.mode_counts],
            "--pixel-stride",
            str(config.frequency.pixel_stride),
        ]
    )
    _run_command_stage(
        config,
        paths,
        stage="modal_frequency_selection",
        argv=argv,
        outputs=[paths.frequency_dir],
        validator=lambda: _validate_frequency_selection(config, inputs, paths),
        dry_run=dry_run,
    )


def _modal_analysis_path(paths: PipelinePaths, view: SourceView) -> Path:
    return paths.modal_analysis_dir / f"{view.view_id}_modal_analysis.npz"


def _validate_modal_analysis(
    config: PipelineConfig,
    paths: PipelinePaths,
    view: SourceView,
    output: Path,
) -> dict[str, Any]:
    import numpy as np
    from modal_surface.io import ensure_modal_shape, load_modal_npz

    modal = load_modal_npz(output)
    ensure_modal_shape(modal, (view.height, view.width))
    frequencies = np.asarray(modal["selected_freqs_hz"], dtype=np.float32).reshape(-1)
    expected = np.asarray(_load_selected_frequencies(config, paths), dtype=np.float32)
    if not np.array_equal(frequencies, expected):
        raise ValueError(f"Modal export frequencies changed for {view.view_id}")
    return {
        "view_id": view.view_id,
        "modes": int(frequencies.size),
        "width": view.width,
        "height": view.height,
    }


def _prepare_modal_exports(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    *,
    dry_run: bool,
) -> None:
    for view in inputs.views:
        _run_temp_file_stage(
            config,
            paths,
            stage=f"modal_export_{view.view_id}",
            build_argv=lambda output, view=view: _python_argv(
                "run_modal_peak_pick.py",
                "export",
                "--cache-dir",
                str(_flow_cache_path(paths, view)),
                "--peaks-json",
                str(_selected_frequency_path(config, paths)),
                "--out",
                str(output),
                "--mode-amp-clamp",
                "none",
            ),
            output=_modal_analysis_path(paths, view),
            validator=lambda output, view=view: _validate_modal_analysis(
                config, paths, view, output
            ),
            dry_run=dry_run,
        )


def _validate_motion_fill_graph(
    config: PipelineConfig,
    accepted_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    from modal_surface.checkpoint_render_inputs import load_fg_means_from_checkpoint
    from modal_surface.gaussian_motion_fill import load_motion_fill_graph

    points = load_fg_means_from_checkpoint(str(accepted_checkpoint))
    graph = load_motion_fill_graph(
        output,
        points,
        expected_k=config.motion_fill.k,
        expected_max_distance=config.motion_fill.max_distance,
    )
    return {
        "points": int(graph.num_points),
        "edges": int(graph.edge_index.shape[0]),
        "components": int(graph.component_sizes.shape[0]),
        "isolated": int(graph.isolated_mask.sum()),
    }


def _prepare_motion_fill_graph(
    config: PipelineConfig,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
    *,
    dry_run: bool,
) -> None:
    _run_temp_file_stage(
        config,
        paths,
        stage="motion_fill_graph",
        build_argv=lambda output: _python_argv(
            "preproc/build_gaussian_motion_fill_graph.py",
            "--input-ckpt",
            str(accepted_checkpoint),
            "--out-npz",
            str(output),
            "--k",
            str(config.motion_fill.k),
            "--max-distance",
            f"{config.motion_fill.max_distance:.17g}",
        ),
        output=paths.motion_fill_graph_path,
        validator=lambda output: _validate_motion_fill_graph(
            config, accepted_checkpoint, output
        ),
        dry_run=dry_run,
    )


def _validate_gaussian_sidecar(
    accepted_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    import numpy as np

    with np.load(output, allow_pickle=False) as archive:
        required = {
            "version",
            "point_type",
            "source_checkpoint",
            "has_background",
            "num_foreground_gaussians",
            "fg_gaussian_indices",
            "fg_centers",
            "fg_quats_wxyz",
            "fg_scales",
            "fg_rgbs",
            "fg_opacities",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Gaussian viewer sidecar is missing {missing}")
        if bool(np.asarray(archive["has_background"]).item()):
            raise ValueError("Gaussian viewer sidecar unexpectedly includes background")
        if Path(str(np.asarray(archive["source_checkpoint"]).item())).resolve() != accepted_checkpoint:
            raise ValueError("Gaussian viewer sidecar source checkpoint changed")
        count = int(np.asarray(archive["num_foreground_gaussians"]).item())
        centers = np.asarray(archive["fg_centers"])
        if centers.shape != (count, 3) or not np.isfinite(centers).all():
            raise ValueError("Gaussian viewer sidecar has invalid foreground centers")
        if not np.array_equal(
            np.asarray(archive["fg_gaussian_indices"]),
            np.arange(count, dtype=np.int64),
        ):
            raise ValueError("Gaussian viewer sidecar foreground indices changed")
        for name in ("fg_quats_wxyz", "fg_scales", "fg_rgbs", "fg_opacities"):
            if np.asarray(archive[name]).shape[0] != count or not np.isfinite(archive[name]).all():
                raise ValueError(f"Gaussian viewer sidecar has invalid {name}")
    if count <= 0:
        raise ValueError("Gaussian viewer sidecar contains no foreground Gaussians")
    return {"foreground_gaussians": count, "source_checkpoint": str(accepted_checkpoint)}


def _prepare_gaussian_sidecar(
    config: PipelineConfig,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
    *,
    dry_run: bool,
) -> None:
    _run_temp_file_stage(
        config,
        paths,
        stage="static_gaussian_viewer_sidecar",
        build_argv=lambda output: _python_argv(
            "preproc/export_static_gaussians_for_viser.py",
            "--input-ckpt",
            str(accepted_checkpoint),
            "--output-npz",
            str(output),
        ),
        output=paths.gaussian_sidecar_path,
        validator=lambda output: _validate_gaussian_sidecar(accepted_checkpoint, output),
        dry_run=dry_run,
    )


def _default_graph_candidate(config: PipelineConfig) -> GraphCandidateParameters:
    graph = config.rigid_graph
    return GraphCandidateParameters(
        candidate_id=graph.default_candidate_id,
        max_distance=graph.max_distance,
        max_neighbors=graph.max_neighbors,
        color_mad_multiplier=graph.color_mad_multiplier,
        depth_mad_multiplier=graph.depth_mad_multiplier,
        depth_samples=graph.depth_samples,
        min_shared_views=graph.min_shared_views,
        min_component_nodes=graph.min_component_nodes,
        min_component_edges=graph.min_component_edges,
    )


def _graph_candidate_dir(paths: PipelinePaths, candidate_id: str) -> Path:
    return paths.graph_candidates_dir / candidate_id


def _graph_candidate_path(paths: PipelinePaths, candidate_id: str) -> Path:
    return _graph_candidate_dir(paths, candidate_id) / "observed_structure_graph.npz"


def _graph_candidate_payload(
    config: PipelineConfig,
    parameters: GraphCandidateParameters,
    accepted_checkpoint: Path,
    paths: PipelinePaths,
) -> dict[str, Any]:
    return {
        "format": "som_rigid_graph_candidate",
        "version": 1,
        "config_identity": config.config_identity,
        "candidate_id": parameters.candidate_id,
        "parameters": asdict(parameters),
        "source_checkpoint": str(accepted_checkpoint),
        "source_topology": str(paths.topology_path),
    }


def _validate_graph_candidate(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
    parameters: GraphCandidateParameters,
    candidate_dir: Path | None = None,
) -> dict[str, Any]:
    import numpy as np
    from modal_surface.observed_structure_graph import load_observed_structure_graph

    root = (
        _graph_candidate_dir(paths, parameters.candidate_id)
        if candidate_dir is None
        else candidate_dir
    )
    config_path = root / "candidate_config.json"
    graph_path = root / "observed_structure_graph.npz"
    expected_payload = _graph_candidate_payload(
        config, parameters, accepted_checkpoint, paths
    )
    if _read_json(config_path) != expected_payload:
        raise ValueError(f"Rigid graph candidate config changed: {config_path}")
    loaded = load_observed_structure_graph(graph_path)
    if Path(loaded.source_checkpoint).resolve() != accepted_checkpoint:
        raise ValueError("Rigid graph source checkpoint changed")
    if Path(loaded.topology_source_observation_path).resolve() != paths.topology_path:
        raise ValueError("Rigid graph observation topology changed")
    if loaded.view_ids != tuple(view.view_id for view in inputs.views):
        raise ValueError("Rigid graph view order changed")
    with np.load(graph_path, allow_pickle=False) as archive:
        expected_scalars = {
            "max_distance": parameters.max_distance,
            "max_neighbors": parameters.max_neighbors,
            "color_mad_multiplier": parameters.color_mad_multiplier,
            "depth_mad_multiplier": parameters.depth_mad_multiplier,
            "depth_samples": parameters.depth_samples,
            "min_shared_views": parameters.min_shared_views,
            "min_component_nodes": parameters.min_component_nodes,
            "min_component_edges": parameters.min_component_edges,
        }
        for name, expected in expected_scalars.items():
            actual = np.asarray(archive[name]).item()
            if isinstance(expected, int):
                valid = int(actual) == expected
            else:
                valid = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
            if not valid:
                raise ValueError(f"Rigid graph candidate {name} changed")
    counts = loaded.graph.counts
    summary = {
        "candidate_id": parameters.candidate_id,
        "nodes": int(loaded.node_gaussian_indices.size),
        "edges": int(loaded.edge_index.shape[0]),
        "components": int(counts["component_count"]),
        "isolated": int(loaded.isolated_mask.sum()),
        "graph_sha256": _sha256_file(graph_path),
        "candidate_identity": _config_hash(expected_payload),
    }
    if candidate_dir is None:
        receipt = _require_mapping(
            _read_json(root / "receipt.json"), "rigid graph candidate receipt"
        )
        if (
            receipt.get("format") != RECEIPT_FORMAT
            or receipt.get("version") != RECEIPT_VERSION
            or receipt.get("config_identity") != config.config_identity
            or receipt.get("status") != "succeeded"
            or receipt.get("summary") != summary
        ):
            raise ValueError(f"Rigid graph candidate receipt changed: {root}")
    return summary


def _build_graph_candidate(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
    parameters: GraphCandidateParameters,
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    candidate_dir = _graph_candidate_dir(paths, parameters.candidate_id)
    if candidate_dir.exists():
        summary = _validate_graph_candidate(
            config, inputs, paths, accepted_checkpoint, parameters
        )
        print(f"Validated rigid graph candidate -> {candidate_dir}")
        return summary
    temporary = paths.graph_candidates_dir / (
        f".{parameters.candidate_id}.{uuid4().hex}.attempt"
    )
    if temporary.exists():
        raise FileExistsError(f"Rigid graph candidate attempt exists: {temporary}")
    graph_output = temporary / "observed_structure_graph.npz"
    argv = _python_argv(
        "preproc/build_observed_gaussian_structure_graph.py",
        "--input-ckpt",
        str(accepted_checkpoint),
    )
    for view in inputs.views:
        argv.extend(["--view-config", str(_view_config_path(paths, view))])
    argv.extend(
        [
            "--observation-topology",
            str(paths.topology_path),
            "--out-npz",
            str(graph_output),
            "--max-distance",
            f"{parameters.max_distance:.17g}",
            "--max-neighbors",
            str(parameters.max_neighbors),
            "--color-mad-multiplier",
            f"{parameters.color_mad_multiplier:.17g}",
            "--depth-mad-multiplier",
            f"{parameters.depth_mad_multiplier:.17g}",
            "--depth-samples",
            str(parameters.depth_samples),
            "--min-shared-views",
            str(parameters.min_shared_views),
            "--min-component-nodes",
            str(parameters.min_component_nodes),
            "--min-component-edges",
            str(parameters.min_component_edges),
        ]
    )
    if dry_run:
        _run_argv(paths, f"rigid_graph_{parameters.candidate_id}", argv, dry_run=True)
        return None
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        _write_json_atomic(
            temporary / "candidate_config.json",
            _graph_candidate_payload(config, parameters, accepted_checkpoint, paths),
        )
        _run_argv(
            paths,
            f"rigid_graph_{parameters.candidate_id}",
            argv,
            dry_run=False,
        )
        summary = _validate_graph_candidate(
            config,
            inputs,
            paths,
            accepted_checkpoint,
            parameters,
            candidate_dir=temporary,
        )
        _write_json_atomic(
            temporary / "receipt.json",
            {
                "format": RECEIPT_FORMAT,
                "version": RECEIPT_VERSION,
                "config_identity": config.config_identity,
                "stage": "rigid_graph_candidate",
                "status": "succeeded",
                "argv": argv,
                "summary": summary,
                "completed_at": _utc_now(),
            },
        )
        _replace_directory(temporary, candidate_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Built rigid graph candidate -> {candidate_dir}")
    return _validate_graph_candidate(
        config, inputs, paths, accepted_checkpoint, parameters
    )


def _load_graph_candidate_parameters(path: Path) -> GraphCandidateParameters:
    payload = _require_mapping(_read_json(path), "rigid graph candidate config")
    values = _require_mapping(payload.get("parameters"), "rigid graph parameters")
    return GraphCandidateParameters(
        candidate_id=_parse_id(values.get("candidate_id"), "candidate_id"),
        max_distance=_parse_float(values.get("max_distance"), "max_distance", minimum=0.0, minimum_inclusive=False),
        max_neighbors=_parse_int(values.get("max_neighbors"), "max_neighbors"),
        color_mad_multiplier=_parse_float(values.get("color_mad_multiplier"), "color_mad_multiplier", minimum=0.0),
        depth_mad_multiplier=_parse_float(values.get("depth_mad_multiplier"), "depth_mad_multiplier", minimum=0.0),
        depth_samples=_parse_int(values.get("depth_samples"), "depth_samples", minimum=2),
        min_shared_views=_parse_int(values.get("min_shared_views"), "min_shared_views"),
        min_component_nodes=_parse_int(values.get("min_component_nodes"), "min_component_nodes"),
        min_component_edges=_parse_int(values.get("min_component_edges"), "min_component_edges"),
    )


def _load_graph_approval(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
) -> tuple[Path, GraphCandidateParameters] | None:
    approval_path = _graph_approval_path(paths)
    if not approval_path.is_file():
        return None
    approval = _require_mapping(_read_json(approval_path), "rigid graph approval")
    if (
        approval.get("format") != GRAPH_APPROVAL_FORMAT
        or approval.get("version") != 1
        or approval.get("config_identity") != config.config_identity
    ):
        raise ValueError(f"Rigid graph approval is incompatible: {approval_path}")
    candidate_id = _parse_id(approval.get("candidate_id"), "approved candidate_id")
    candidate_dir = _graph_candidate_dir(paths, candidate_id)
    parameters = _load_graph_candidate_parameters(candidate_dir / "candidate_config.json")
    summary = _validate_graph_candidate(
        config, inputs, paths, accepted_checkpoint, parameters
    )
    if (
        approval.get("graph_sha256") != summary["graph_sha256"]
        or approval.get("candidate_identity") != summary["candidate_identity"]
    ):
        raise ValueError("Approved rigid graph candidate changed after approval")
    return candidate_dir / "observed_structure_graph.npz", parameters


def _approve_graph(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
    candidate_id: str,
    note: str | None,
) -> None:
    approval_path = _graph_approval_path(paths)
    if approval_path.exists():
        raise FileExistsError(f"Rigid graph quality is already approved: {approval_path}")
    candidate_dir = _graph_candidate_dir(paths, candidate_id)
    parameters = _load_graph_candidate_parameters(candidate_dir / "candidate_config.json")
    if parameters.candidate_id != candidate_id:
        raise ValueError("Candidate ID does not match candidate directory")
    summary = _validate_graph_candidate(
        config, inputs, paths, accepted_checkpoint, parameters
    )
    _write_json_atomic(
        approval_path,
        {
            "format": GRAPH_APPROVAL_FORMAT,
            "version": 1,
            "config_identity": config.config_identity,
            "candidate_id": candidate_id,
            "candidate_dir": str(candidate_dir.resolve()),
            "candidate_identity": summary["candidate_identity"],
            "graph_sha256": summary["graph_sha256"],
            "note": note,
            "approved_at": _utc_now(),
        },
    )
    _append_event(
        paths, {"event": "rigid_graph_quality_approved", "candidate_id": candidate_id, "note": note}
    )
    print(f"Approved rigid graph candidate -> {candidate_dir}")


def _run_temp_directory_stage(
    config: PipelineConfig,
    paths: PipelinePaths,
    *,
    stage: str,
    build_argv: Callable[[Path], Sequence[str]],
    output: Path,
    validator: Callable[[Path], Mapping[str, Any]],
    dry_run: bool,
) -> None:
    receipt = _receipt_path(paths, stage)
    if receipt.exists():
        if dry_run:
            print(f"[{stage}] already completed")
            return
        _validate_receipt(config, receipt, [output], lambda: validator(output))
        print(f"[{stage}] validated existing output")
        return
    if output.exists() or output.is_symlink():
        if dry_run:
            print(f"[{stage}] would validate and adopt existing output")
            return
        summary = validator(output)
        _write_receipt(config, paths, stage, None, [output], summary)
        print(f"[{stage}] adopted validated output and wrote a receipt")
        return
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid4().hex}.attempt"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Temporary stage target already exists: {temporary}")
    argv = list(build_argv(temporary))
    try:
        _run_argv(paths, stage, argv, dry_run=dry_run)
        if dry_run:
            return
        summary = validator(temporary)
        _replace_directory(temporary, output)
        validator(output)
        _write_receipt(config, paths, stage, argv, [output], summary)
    except BaseException:
        if not dry_run:
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _frequency_slug(freq_hz: float) -> str:
    text = f"{freq_hz:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _measurement_paths(config: PipelineConfig, paths: PipelinePaths) -> list[Path]:
    frequencies = _load_selected_frequencies(config, paths)
    return [
        paths.observations_dir
        / "measurements"
        / f"mode_{index:03d}_{_frequency_slug(frequency)}hz.npz"
        for index, frequency in enumerate(frequencies)
    ]


def _validate_observations(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    output: Path,
) -> dict[str, Any]:
    import numpy as np
    from modal_surface.gaussian_observations import (
        load_gaussian_observation_measurement,
        load_gaussian_observation_topology,
    )

    topology = load_gaussian_observation_topology(paths.topology_path)
    topology_id = str(np.asarray(topology["topology_id"]).item())
    files = sorted((output / "measurements").glob("mode_*.npz"))
    if len(files) != config.frequency.selected_k:
        raise ValueError(
            f"Expected {config.frequency.selected_k} observation measurements, got {len(files)}"
        )
    expected_modal_paths = [
        str(_modal_analysis_path(paths, view).resolve())
        for view in inputs.views
    ]
    mode_indices: list[int] = []
    for path in files:
        measurement = load_gaussian_observation_measurement(path)
        if str(np.asarray(measurement["topology_id"]).item()) != topology_id:
            raise ValueError(f"Observation measurement topology changed: {path}")
        mode_indices.append(int(np.asarray(measurement["mode_index"]).item()))
        sources = np.asarray(measurement["source_modal_npzs"]).astype(str).tolist()
        if [str(Path(value).resolve()) for value in sources] != expected_modal_paths:
            raise ValueError(f"Observation measurement modal sources changed: {path}")
    if mode_indices != list(range(config.frequency.selected_k)):
        raise ValueError("Observation measurement mode order is not contiguous")
    return {
        "modes": len(files),
        "topology_id": topology_id,
        "measurement_files": [path.name for path in files],
    }


def _prepare_observations(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    *,
    dry_run: bool,
) -> None:
    def build(output: Path) -> list[str]:
        argv = _python_argv(
            "preproc/build_gaussian_observations.py",
            "--observation-topology",
            str(paths.topology_path),
        )
        for view in inputs.views:
            argv.extend(["--modal-npz", str(_modal_analysis_path(paths, view))])
        argv.extend(
            [
                "--out-dir",
                str(output),
                "--mode-indices",
                "all",
                "--freq-tolerance-hz",
                f"{config.solver.freq_tolerance_hz:.17g}",
            ]
        )
        return argv

    _run_temp_directory_stage(
        config,
        paths,
        stage="gaussian_observation_measurements",
        build_argv=build,
        output=paths.observations_dir,
        validator=lambda output: _validate_observations(config, inputs, paths, output),
        dry_run=dry_run,
    )


def _validate_modal_fields(
    config: PipelineConfig,
    accepted_checkpoint: Path,
    paths: PipelinePaths,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from flow3d.modal_utils import load_gaussian_modal_fields
    from modal_surface.checkpoint_render_inputs import load_fg_means_from_checkpoint

    manifest = paths.modal_fields_dir / "modal_modes_manifest.json"
    means = torch.from_numpy(load_fg_means_from_checkpoint(str(accepted_checkpoint)))
    fields = load_gaussian_modal_fields(str(manifest), means)
    if fields.source_checkpoint.resolve() != accepted_checkpoint:
        raise ValueError("Modal manifest source checkpoint changed")
    if len(fields.modes) != config.frequency.selected_k:
        raise ValueError("Modal manifest mode count changed")
    expected = np.asarray(_load_selected_frequencies(config, paths), dtype=np.float64)
    actual = fields.freqs_hz.detach().cpu().numpy().astype(np.float64)
    if not np.allclose(actual, expected, rtol=0.0, atol=config.solver.freq_tolerance_hz):
        raise ValueError("Modal manifest frequencies changed")
    if not torch.isfinite(fields.phi_real).all() or not torch.isfinite(fields.phi_imag).all():
        raise ValueError("Modal fields contain non-finite phi")
    return {
        "modes": len(fields.modes),
        "gaussians": int(fields.phi_real.shape[1]),
        "source_checkpoint": str(fields.source_checkpoint),
        "manifest": str(manifest),
    }


def _modal_solver_argv(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
    rigid_graph: Path | None,
    *,
    resume: bool,
) -> list[str]:
    argv = [
        sys.executable,
        "-u",
        "-m",
        "modal_surface",
        "--input-ckpt",
        str(accepted_checkpoint),
    ]
    for view in inputs.views:
        argv.extend(["--view-config", str(_view_config_path(paths, view))])
    for view in inputs.views:
        argv.extend(["--modal-npz", str(_modal_analysis_path(paths, view))])
    argv.extend(
        [
            "--out-dir",
            str(paths.modal_fields_dir),
            "--mode-indices",
            "all",
            "--pixel-sample-stride",
            str(config.topology.pixel_sample_stride),
            "--pixel-candidate-k",
            str(config.topology.pixel_candidate_k),
            "--pixel-preselect-k",
            str(config.topology.pixel_preselect_k),
            "--pixel-render-acc-min",
            f"{config.topology.pixel_render_acc_min:.17g}",
            "--pixel-min-contribution",
            f"{config.topology.pixel_min_contribution:.17g}",
            "--mask-erode-iters",
            str(config.topology.mask_erode_iters),
            "--freq-tolerance-hz",
            f"{config.solver.freq_tolerance_hz:.17g}",
            "--motion-fill",
            "--motion-fill-k",
            str(config.motion_fill.k),
            "--motion-fill-max-distance",
            f"{config.motion_fill.max_distance:.17g}",
            "--motion-fill-graph",
            str(paths.motion_fill_graph_path),
            "--motion-fill-max-anchor-hops",
            str(config.motion_fill.max_anchor_hops),
            "--solve-method",
            config.solver.method,
            "--alpha-model",
            config.solver.alpha_model,
            "--alpha-gain-min",
            f"{config.solver.alpha_gain_min:.17g}",
            "--alpha-gain-max",
            f"{config.solver.alpha_gain_max:.17g}",
            "--alpha-min-shared-points",
            str(config.solver.alpha_min_shared_points),
            "--alpha-rank-ratio-min",
            f"{config.solver.alpha_rank_ratio_min:.17g}",
            "--alpha-info-ratio-min",
            f"{config.solver.alpha_info_ratio_min:.17g}",
            "--alpha-failure",
            config.solver.alpha_failure,
        ]
    )
    if config.solver.method == "staged":
        argv.extend(
            [
                "--anchor-svd-ratio-min",
                f"{config.solver.anchor_svd_ratio_min:.17g}",
                "--anchor-residual-max",
                f"{config.solver.anchor_residual_max:.17g}",
            ]
        )
    else:
        if rigid_graph is None:
            raise ValueError("Rigid-component solver requires an approved graph")
        argv.extend(
            [
                "--rigid-component-graph",
                str(rigid_graph),
                "--rigid-component-observation-topology",
                str(paths.topology_path),
                "--rigid-component-rcond",
                f"{config.solver.rigid_component_rcond:.17g}",
                "--rigid-seed-min-valid-views",
                str(config.solver.rigid_seed_min_valid_views),
                "--rigid-seed-min-secondary-view-node-ratio",
                f"{config.solver.rigid_seed_min_secondary_view_node_ratio:.17g}",
                "--rigid-seed-min-singular-ratio",
                f"{config.solver.rigid_seed_min_singular_ratio:.17g}",
                "--rigid-seed-max-finite-drift",
                f"{config.solver.rigid_seed_max_finite_drift:.17g}",
                "--rigid-motion-fill-stage",
                "sequential",
                "--rigid-single-view-observable-ratio",
                f"{config.solver.rigid_single_view_observable_ratio:.17g}",
                "--rigid-single-view-ray-direction-min-fraction",
                f"{config.solver.rigid_single_view_ray_direction_min_fraction:.17g}",
            ]
        )
        for measurement in _measurement_paths(config, paths):
            argv.extend(
                ["--rigid-component-observation-measurement", str(measurement)]
            )
        if resume:
            argv.append("--resume")
    return argv


def _prepare_modal_fields(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
    rigid_graph: Path | None,
    *,
    dry_run: bool,
) -> None:
    stage = "gaussian_modal_fields"
    receipt = _receipt_path(paths, stage)
    if receipt.exists():
        if dry_run:
            print(f"[{stage}] already completed")
            return
        _validate_receipt(
            config,
            receipt,
            [paths.modal_fields_dir],
            lambda: _validate_modal_fields(config, accepted_checkpoint, paths),
        )
        print(f"[{stage}] validated existing output")
        return
    existing = paths.modal_fields_dir.exists()
    if existing and config.solver.method != "rigid-components":
        raise FileExistsError(
            f"Staged modal output exists without receipt: {paths.modal_fields_dir}"
        )
    argv = _modal_solver_argv(
        config,
        inputs,
        paths,
        accepted_checkpoint,
        rigid_graph,
        resume=existing,
    )
    _run_argv(paths, stage, argv, dry_run=dry_run)
    if dry_run:
        return
    summary = _validate_modal_fields(config, accepted_checkpoint, paths)
    _write_receipt(
        config, paths, stage, argv, [paths.modal_fields_dir], summary
    )


def _validate_rendered_design(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
) -> dict[str, Any]:
    import numpy as np
    from flow3d.modal_rendered_design import (
        ROLE_DIAGNOSTICS_FILENAME,
        ROLE_OVERVIEW_FILENAME,
        load_rendered_modal_design,
    )

    design = load_rendered_modal_design(paths.rendered_design_dir)
    if design.view_ids != tuple(view.view_id for view in inputs.views):
        raise ValueError("Rendered design view order changed")
    if design.source_checkpoint.resolve() != accepted_checkpoint:
        raise ValueError("Rendered design source checkpoint changed")
    if design.source_modal_manifest.resolve() != (
        paths.modal_fields_dir / "modal_modes_manifest.json"
    ).resolve():
        raise ValueError("Rendered design source manifest changed")
    if design.mode_indices.size != config.frequency.selected_k:
        raise ValueError("Rendered design mode count changed")
    if design.pixel_sample_stride != config.rendered_design.pixel_sample_stride:
        raise ValueError("Rendered design pixel stride changed")
    expected_design_parameters = {
        "alpha_min": config.rendered_design.alpha_min,
        "mask_erode_iters": config.rendered_design.mask_erode_iters,
        "modes_per_batch": config.rendered_design.modes_per_batch,
        "rasterizer": (
            "gsplat_2dgs" if config.rendered_design.use_2dgs else "gsplat_3dgs"
        ),
    }
    actual_design_parameters = {
        "alpha_min": design.alpha_min,
        "mask_erode_iters": design.mask_erode_iters,
        "modes_per_batch": design.modes_per_batch,
        "rasterizer": design.rasterizer,
    }
    if actual_design_parameters != expected_design_parameters:
        raise ValueError(
            "Rendered design scientific parameters changed: "
            f"actual={actual_design_parameters}, expected={expected_design_parameters}"
        )
    expected_caches = tuple(_flow_cache_path(paths, view).resolve() for view in inputs.views)
    if tuple(path.resolve() for path in design.source_flow_cache_dirs) != expected_caches:
        raise ValueError("Rendered design flow-cache sources changed")
    if config.rendered_design.write_role_diagnostics:
        role_path = paths.rendered_design_dir / ROLE_DIAGNOSTICS_FILENAME
        overview_path = paths.rendered_design_dir / ROLE_OVERVIEW_FILENAME
        if not role_path.is_file() or not overview_path.is_file():
            raise FileNotFoundError(
                "Rendered design role diagnostics were requested but are incomplete"
            )
        with np.load(role_path, allow_pickle=False) as role_archive:
            if (
                str(np.asarray(role_archive["format"]).item())
                != "rendered_modal_role_diagnostics"
                or int(np.asarray(role_archive["version"]).item()) != 1
                or str(np.asarray(role_archive["rendered_design_identity"]).item())
                != design.artifact_identity
            ):
                raise ValueError("Rendered design role diagnostics identity changed")
    column_count = 2 * config.frequency.selected_k
    per_view: list[dict[str, Any]] = []
    for view_index, view in enumerate(inputs.views):
        rows = design.sample_view_index == view_index
        if not np.any(rows):
            raise ValueError(f"Rendered design has no candidate pixels for {view.view_id}")
        matrix = design.design_matrix[rows].reshape(-1, column_count).astype(np.float64)
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank_tolerance = singular[0] * max(matrix.shape) * np.finfo(np.float64).eps
        rank = int(np.count_nonzero(singular > rank_tolerance))
        if rank != column_count:
            raise ValueError(
                f"Rendered design for {view.view_id} has rank {rank}/{column_count}"
            )
        condition = float(singular[0] / singular[-1])
        per_view.append(
            {
                "view_id": view.view_id,
                "pixels": int(np.count_nonzero(rows)),
                "rank": rank,
                "columns": column_count,
                "condition": condition,
            }
        )
    return {
        "artifact_identity": design.artifact_identity,
        "modes": int(design.mode_indices.size),
        "samples": int(design.sample_view_index.size),
        "views": per_view,
    }


def _prepare_rendered_design(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
    *,
    dry_run: bool,
) -> None:
    argv = _python_argv(
        "preproc/build_rendered_modal_design.py",
        "--input-ckpt",
        str(accepted_checkpoint),
        "--modal-manifest",
        str(paths.modal_fields_dir / "modal_modes_manifest.json"),
    )
    for view in inputs.views:
        argv.extend(["--view-config", str(_view_config_path(paths, view))])
    for view in inputs.views:
        argv.extend(
            ["--flow-cache", f"{view.view_id}={_flow_cache_path(paths, view)}"]
        )
    argv.extend(
        [
            "--out-dir",
            str(paths.rendered_design_dir),
            "--pixel-sample-stride",
            str(config.rendered_design.pixel_sample_stride),
            "--alpha-min",
            f"{config.rendered_design.alpha_min:.17g}",
            "--mask-erode-iters",
            str(config.rendered_design.mask_erode_iters),
            "--modes-per-batch",
            str(config.rendered_design.modes_per_batch),
        ]
    )
    if config.rendered_design.use_2dgs:
        argv.append("--use-2dgs")
    if config.rendered_design.write_role_diagnostics:
        argv.append("--write-role-diagnostics")
    _run_command_stage(
        config,
        paths,
        stage="rendered_modal_design",
        argv=argv,
        outputs=[paths.rendered_design_dir],
        validator=lambda: _validate_rendered_design(
            config, inputs, paths, accepted_checkpoint
        ),
        dry_run=dry_run,
    )


def _validate_coordinates(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    coordinate_dir: Path,
    *,
    physics: bool,
) -> dict[str, Any]:
    from flow3d.modal_flow_coordinates import (
        MODAL_FLOW_COORDINATE_SOLVER,
        MODAL_PHYSICS_COORDINATE_SOLVER,
        load_modal_coordinate_provenance,
        load_modal_flow_coordinates,
    )
    from flow3d.modal_rendered_design import load_rendered_modal_design

    coordinates = load_modal_flow_coordinates(coordinate_dir)
    provenance = load_modal_coordinate_provenance(coordinates)
    expected_solver = (
        MODAL_PHYSICS_COORDINATE_SOLVER if physics else MODAL_FLOW_COORDINATE_SOLVER
    )
    if provenance.get("solver") != expected_solver:
        raise ValueError(f"Coordinate solver is not {expected_solver}")
    design = load_rendered_modal_design(paths.rendered_design_dir)
    if provenance.get("rendered_design_identity") != design.artifact_identity:
        raise ValueError("Coordinate rendered-design identity changed")
    if coordinates.view_ids != tuple(view.view_id for view in inputs.views):
        raise ValueError("Coordinate view order changed")
    if coordinates.mode_indices.size != config.frequency.selected_k:
        raise ValueError("Coordinate mode count changed")
    expected_refs = tuple(view.reference_local_index for view in inputs.views)
    if tuple(int(value) for value in coordinates.reference_local_indices) != expected_refs:
        raise ValueError("Coordinate reference-frame indices changed")
    if not physics and not math.isclose(
        coordinates.ridge_relative,
        config.flow_coordinates.ridge_relative,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Coordinate ridge weight changed")
    if physics:
        settings = _require_mapping(provenance.get("physics"), "physics provenance")
        expected_settings = {
            "damping_ratio": config.physics.damping_ratio,
            "forcing_weight": config.physics.forcing_weight,
            "forcing_difference_weight": config.physics.forcing_difference_weight,
            "assigned_band_half_width_hz": config.physics.assigned_band_half_width_hz,
            "frame_chunk_size": config.physics.frame_chunk_size,
        }
        for name, expected in expected_settings.items():
            actual = settings.get(name)
            if isinstance(expected, int):
                valid = actual == expected
            else:
                valid = isinstance(actual, (int, float)) and math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1e-15
                )
            if not valid:
                raise ValueError(f"Physics coordinate setting changed: {name}")
        if Path(str(provenance.get("source_coordinate"))).resolve() != (
            paths.flow_coordinates_dir / "modal_flow_coordinates.npz"
        ).resolve():
            raise ValueError("Physics source coordinate changed")
    return {
        "solver": expected_solver,
        "views": list(coordinates.view_ids),
        "frames": len(coordinates.frame_names),
        "modes": int(coordinates.mode_indices.size),
        "rendered_design_identity": design.artifact_identity,
    }


def _prepare_flow_coordinates(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    *,
    dry_run: bool,
) -> None:
    argv = _python_argv(
        "run_modal_flow_coordinates.py",
        "--rendered-design",
        str(paths.rendered_design_dir),
        "--modal-frame-map",
        str(paths.dynamic_dataset / "modal_frame_map.json"),
        "--flow-caches",
        *[
            f"{view.view_id}={_flow_cache_path(paths, view)}"
            for view in inputs.views
        ],
        "--out-dir",
        str(paths.flow_coordinates_dir),
        "--ridge-relative",
        f"{config.flow_coordinates.ridge_relative:.17g}",
        "--frame-chunk-size",
        str(config.flow_coordinates.frame_chunk_size),
    )
    _run_command_stage(
        config,
        paths,
        stage="rendered_flow_coordinates",
        argv=argv,
        outputs=[paths.flow_coordinates_dir],
        validator=lambda: _validate_coordinates(
            config, inputs, paths, paths.flow_coordinates_dir, physics=False
        ),
        dry_run=dry_run,
    )


def _prepare_physics_coordinates(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    *,
    dry_run: bool,
) -> None:
    argv = _python_argv(
        "run_modal_physics_coordinates.py",
        "--input-coordinates",
        str(paths.flow_coordinates_dir / "modal_flow_coordinates.npz"),
        "--out-dir",
        str(paths.physics_coordinates_dir),
        "--damping-ratio",
        f"{config.physics.damping_ratio:.17g}",
        "--forcing-weight",
        f"{config.physics.forcing_weight:.17g}",
        "--forcing-difference-weight",
        f"{config.physics.forcing_difference_weight:.17g}",
        "--assigned-band-half-width-hz",
        f"{config.physics.assigned_band_half_width_hz:.17g}",
        "--frame-chunk-size",
        str(config.physics.frame_chunk_size),
    )
    _run_command_stage(
        config,
        paths,
        stage="physics_coordinates",
        argv=argv,
        outputs=[paths.physics_coordinates_dir],
        validator=lambda: _validate_coordinates(
            config, inputs, paths, paths.physics_coordinates_dir, physics=True
        ),
        dry_run=dry_run,
    )


def _validate_materialized_checkpoint(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    work_dir: Path,
    coordinate_dir: Path,
    accepted_checkpoint: Path,
    *,
    configured_work_dir: Path | None = None,
) -> dict[str, Any]:
    import torch
    from flow3d.modal_flow_coordinates import load_modal_coordinate_provenance, load_modal_flow_coordinates
    from flow3d.scene_model import SceneModel

    cfg_path = work_dir / "cfg.yaml"
    checkpoint_path = _static_checkpoint_path(work_dir)
    if not cfg_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Materialized checkpoint is incomplete: {work_dir}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = _require_mapping(
            yaml.load(handle, Loader=yaml.FullLoader), "materialized cfg"
        )
    expected_work_dir = work_dir if configured_work_dir is None else configured_work_dir
    expected = {
        "work_dir": str(expected_work_dir),
        "trajectory_type": "modal_activation",
        "num_epochs": 0,
        "num_fg": config.static.num_fg,
        "num_bg": config.static.num_bg,
        "use_2dgs": config.static.use_2dgs,
        "modal_stage1_init_ckpt": str(accepted_checkpoint),
        "modal_manifest": str(paths.modal_fields_dir / "modal_modes_manifest.json"),
        "modal_frame_map": str(paths.dynamic_dataset / "modal_frame_map.json"),
        "modal_flow_coordinates": str(coordinate_dir / "modal_flow_coordinates.npz"),
    }
    mismatches = {key: (cfg.get(key), value) for key, value in expected.items() if cfg.get(key) != value}
    expected_view_configs = [
        str(_view_config_path(paths, view))
        for view in inputs.views
    ]
    if list(cfg.get("vggt_view_configs", [])) != expected_view_configs:
        mismatches["vggt_view_configs"] = (
            cfg.get("vggt_view_configs"),
            expected_view_configs,
        )
    data_cfg = _require_mapping(cfg.get("data"), "materialized cfg.data")
    if (
        Path(str(data_cfg.get("data_dir"))).resolve() != paths.dynamic_dataset.resolve()
        or data_cfg.get("camera_type") != "vggt"
    ):
        mismatches["data"] = (
            {"data_dir": data_cfg.get("data_dir"), "camera_type": data_cfg.get("camera_type")},
            {"data_dir": str(paths.dynamic_dataset), "camera_type": "vggt"},
        )
    if mismatches:
        raise ValueError(f"Materialized checkpoint config mismatch: {mismatches}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SceneModel.init_from_state_dict(checkpoint["model"])
    if int(model.modal_freqs_hz.shape[0]) != config.frequency.selected_k:
        raise ValueError("Materialized checkpoint mode count changed")
    metadata = _require_mapping(checkpoint.get("init_metadata"), "materialized init_metadata")
    coordinates = load_modal_flow_coordinates(coordinate_dir)
    provenance = load_modal_coordinate_provenance(coordinates)
    metadata_expected = {
        "trajectory_type": "modal_activation",
        "num_fg": config.static.num_fg,
        "num_bg": config.static.num_bg,
        "modal_manifest": str(paths.modal_fields_dir / "modal_modes_manifest.json"),
        "modal_frame_map": str(paths.dynamic_dataset / "modal_frame_map.json"),
        "vggt_view_configs": expected_view_configs,
        "modal_stage1_init_ckpt": str(accepted_checkpoint),
    }
    metadata_mismatches = {
        key: (metadata.get(key), value)
        for key, value in metadata_expected.items()
        if metadata.get(key) != value
    }
    metadata_data = _require_mapping(metadata.get("data"), "materialized init_metadata.data")
    if (
        Path(str(metadata_data.get("data_dir"))).resolve()
        != paths.dynamic_dataset.resolve()
        or metadata_data.get("camera_type") != "vggt"
    ):
        metadata_mismatches["data"] = (
            {
                "data_dir": metadata_data.get("data_dir"),
                "camera_type": metadata_data.get("camera_type"),
            },
            {"data_dir": str(paths.dynamic_dataset), "camera_type": "vggt"},
        )
    if metadata_mismatches:
        raise ValueError(
            f"Materialized checkpoint init_metadata mismatch: {metadata_mismatches}"
        )
    if metadata.get("modal_coordinate_solver") != provenance.get("solver"):
        raise ValueError("Materialized checkpoint coordinate solver provenance changed")
    if metadata.get("modal_coordinate_source") != str(coordinates.path):
        raise ValueError("Materialized checkpoint coordinate source changed")
    rendered_metadata = {
        "modal_rendered_design_source": "rendered_design_source",
        "modal_rendered_design_identity": "rendered_design_identity",
        "modal_rendered_design_normalization": "rendered_design_normalization",
    }
    for metadata_key, provenance_key in rendered_metadata.items():
        if provenance_key in provenance and metadata.get(metadata_key) != provenance[provenance_key]:
            raise ValueError(
                f"Materialized checkpoint {metadata_key} provenance changed"
            )
    return {
        "checkpoint": str(checkpoint_path),
        "coordinate_solver": provenance["solver"],
        "foreground_gaussians": int(model.num_fg_gaussians),
        "background_gaussians": int(model.num_bg_gaussians),
        "modes": int(model.modal_freqs_hz.shape[0]),
    }


def _prepare_materialized_checkpoint(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
    accepted_checkpoint: Path,
    coordinate_dir: Path,
    work_dir: Path,
    stage: str,
    *,
    dry_run: bool,
) -> None:
    receipt = _receipt_path(paths, stage)
    validator = lambda target: _validate_materialized_checkpoint(
        config,
        inputs,
        paths,
        target,
        coordinate_dir,
        accepted_checkpoint,
        configured_work_dir=work_dir,
    )
    if receipt.exists():
        if dry_run:
            print(f"[{stage}] already completed")
            return
        _validate_receipt(config, receipt, [work_dir], lambda: validator(work_dir))
        print(f"[{stage}] validated existing output")
        return
    if work_dir.exists() or work_dir.is_symlink():
        if dry_run:
            print(f"[{stage}] would validate and adopt existing output")
            return
        try:
            summary = validator(work_dir)
        except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
            raise FileExistsError(
                f"Materialized checkpoint target exists but is incomplete or "
                f"incompatible; move or remove it before retrying: {work_dir}"
            ) from error
        _write_receipt(config, paths, stage, None, [work_dir], summary)
        print(f"[{stage}] adopted validated output and wrote a receipt")
        return

    temporary = work_dir.parent / f".{work_dir.name}.{uuid4().hex}.attempt"
    argv = _python_argv(
        "run_training.py",
        "--work-dir",
        str(temporary),
        "--trajectory-type",
        "modal_activation",
        "--modal-stage1-init-ckpt",
        str(accepted_checkpoint),
        "--modal-manifest",
        str(paths.modal_fields_dir / "modal_modes_manifest.json"),
        "--modal-frame-map",
        str(paths.dynamic_dataset / "modal_frame_map.json"),
        "--modal-flow-coordinates",
        str(coordinate_dir / "modal_flow_coordinates.npz"),
        "--vggt-view-configs",
        *[str(_view_config_path(paths, view)) for view in inputs.views],
        "--num-epochs",
        "0",
        "--num-fg",
        str(config.static.num_fg),
        "--num-bg",
        str(config.static.num_bg),
    )
    if config.static.use_2dgs:
        argv.append("--use-2dgs")
    argv.extend(
        [
            "data:custom",
            "--data.data-dir",
            str(paths.dynamic_dataset),
            "--data.camera-type",
            "vggt",
        ]
    )
    try:
        _run_argv(paths, stage, argv, dry_run=dry_run)
        if dry_run:
            return
        cfg_path = temporary / "cfg.yaml"
        with cfg_path.open("r", encoding="utf-8") as handle:
            materialized_cfg = _require_mapping(
                yaml.load(handle, Loader=yaml.FullLoader),
                "temporary materialized cfg",
            )
        materialized_cfg = dict(materialized_cfg)
        materialized_cfg["work_dir"] = str(work_dir)
        _write_yaml_atomic(cfg_path, materialized_cfg)
        summary = validator(temporary)
        _replace_directory(temporary, work_dir)
        summary = validator(work_dir)
        _write_receipt(config, paths, stage, argv, [work_dir], summary)
    except BaseException:
        if not dry_run:
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_final_ready(
    config: PipelineConfig,
    inputs: PrestaticInputs,
    paths: PipelinePaths,
) -> None:
    payload = {
        "format": FINAL_READY_FORMAT,
        "version": 1,
        "config_identity": config.config_identity,
        "scene_id": config.scene_id,
        "pipeline_id": config.pipeline_id,
        "view_order": [view.view_id for view in inputs.views],
        "direct_work_dir": str(paths.direct_checkpoint_dir),
        "direct_checkpoint": str(_static_checkpoint_path(paths.direct_checkpoint_dir)),
        "physics_enabled": config.physics.enabled,
        "physics_work_dir": str(paths.physics_checkpoint_dir) if config.physics.enabled else None,
        "physics_checkpoint": (
            str(_static_checkpoint_path(paths.physics_checkpoint_dir))
            if config.physics.enabled
            else None
        ),
        "original_video_reconstruction_included": False,
        "viewer_targets": ["static"]
        + (["rigid-graph"] if config.solver.method == "rigid-components" else [])
        + ["direct"]
        + (["physics"] if config.physics.enabled else []),
        "completed_at": _utc_now(),
    }
    if paths.final_ready_path.exists():
        existing = _read_json(paths.final_ready_path)
        existing_without_time = dict(existing)
        existing_without_time.pop("completed_at", None)
        payload_without_time = dict(payload)
        payload_without_time.pop("completed_at", None)
        if existing_without_time != payload_without_time:
            raise ValueError("Existing final READY marker differs from current pipeline")
        return
    _write_json_atomic(paths.final_ready_path, payload)
    _append_event(paths, {"event": "final_visualization_ready"})


def _set_gate(paths: PipelinePaths, state: dict[str, Any], gate: Mapping[str, Any] | None) -> None:
    state["gate"] = None if gate is None else dict(gate)
    _save_state(paths, state)


def _print_static_gate(config: PipelineConfig, paths: PipelinePaths) -> None:
    print("\nPaused for manual static-3DGS quality inspection.")
    print(
        "Viewer: "
        + _viewer_command_text(config, paths, "static", None, 8890)
    )
    print(
        "Approve: "
        f"python -u {REPO_ROOT / 'run_experiment_pipeline.py'} approve-static "
        f"--config {config.config_path}"
    )
    print(
        "Extend: "
        f"python -u {REPO_ROOT / 'run_experiment_pipeline.py'} extend-static "
        f"--config {config.config_path} --target-epochs <TOTAL_EPOCHS>"
    )


def _print_graph_gate(
    config: PipelineConfig,
    paths: PipelinePaths,
    candidate_id: str,
) -> None:
    print("\nPaused for manual rigid-component graph inspection.")
    print(
        "Viewer: "
        + _viewer_command_text(config, paths, "rigid-graph", candidate_id, 8890)
    )
    print(
        "Approve: "
        f"python -u {REPO_ROOT / 'run_experiment_pipeline.py'} approve-graph "
        f"--config {config.config_path} --candidate-id {candidate_id}"
    )
    print(
        "Alternative: "
        f"python -u {REPO_ROOT / 'run_experiment_pipeline.py'} build-graph-candidate "
        f"--config {config.config_path} --candidate-id <ID> "
        "--max-distance <DISTANCE>"
    )


def _run_pipeline(config: PipelineConfig, *, dry_run: bool) -> None:
    inputs = load_prestatic_inputs(config)
    paths = _pipeline_paths(config)
    if dry_run:
        state = (
            dict(_require_mapping(_read_json(paths.state_path), "pipeline state"))
            if paths.state_path.is_file()
            else _initial_state(config)
        )
    else:
        state = _prepare_controller(config, inputs, paths)
    static_approval = _load_static_approval(config, inputs, paths)
    if static_approval is None:
        _run_static_candidate(config, inputs, paths, state, dry_run=dry_run)
        if not dry_run:
            _set_gate(
                paths,
                state,
                {
                    "name": "static_quality",
                    "status": "waiting_for_approval",
                    "candidate_work_dir": str(paths.static_work_dir),
                    "target_epochs": int(state["static_target_epochs"]),
                },
            )
        _print_static_gate(config, paths)
        return
    accepted_checkpoint, _accepted_cfg = static_approval

    _prepare_native_inputs(config, inputs, paths, dry_run=dry_run)
    _prepare_flow_caches(config, inputs, paths, dry_run=dry_run)
    _prepare_view_configs(
        config, inputs, paths, accepted_checkpoint, dry_run=dry_run
    )
    _prepare_topology(config, inputs, paths, accepted_checkpoint, dry_run=dry_run)
    _prepare_frequency_selection(config, inputs, paths, dry_run=dry_run)
    _prepare_modal_exports(config, inputs, paths, dry_run=dry_run)
    _prepare_motion_fill_graph(
        config, paths, accepted_checkpoint, dry_run=dry_run
    )
    _prepare_gaussian_sidecar(
        config, paths, accepted_checkpoint, dry_run=dry_run
    )

    rigid_graph: Path | None = None
    if config.solver.method == "rigid-components":
        graph_approval = _load_graph_approval(
            config, inputs, paths, accepted_checkpoint
        )
        if graph_approval is None:
            candidate = _default_graph_candidate(config)
            _build_graph_candidate(
                config,
                inputs,
                paths,
                accepted_checkpoint,
                candidate,
                dry_run=dry_run,
            )
            if not dry_run:
                _set_gate(
                    paths,
                    state,
                    {
                        "name": "rigid_graph_quality",
                        "status": "waiting_for_approval",
                        "candidate_id": candidate.candidate_id,
                        "candidate_dir": str(
                            _graph_candidate_dir(paths, candidate.candidate_id)
                        ),
                    },
                )
            _print_graph_gate(config, paths, candidate.candidate_id)
            return
        rigid_graph, _approved_parameters = graph_approval
        _prepare_observations(config, inputs, paths, dry_run=dry_run)

    _prepare_modal_fields(
        config,
        inputs,
        paths,
        accepted_checkpoint,
        rigid_graph,
        dry_run=dry_run,
    )
    _prepare_rendered_design(
        config, inputs, paths, accepted_checkpoint, dry_run=dry_run
    )
    _prepare_flow_coordinates(config, inputs, paths, dry_run=dry_run)
    if config.physics.enabled:
        _prepare_physics_coordinates(config, inputs, paths, dry_run=dry_run)
    _prepare_materialized_checkpoint(
        config,
        inputs,
        paths,
        accepted_checkpoint,
        paths.flow_coordinates_dir,
        paths.direct_checkpoint_dir,
        "materialize_direct_checkpoint",
        dry_run=dry_run,
    )
    if config.physics.enabled:
        _prepare_materialized_checkpoint(
            config,
            inputs,
            paths,
            accepted_checkpoint,
            paths.physics_coordinates_dir,
            paths.physics_checkpoint_dir,
            "materialize_physics_checkpoint",
            dry_run=dry_run,
        )
    if dry_run:
        print("[final] would write FINAL_VISUALIZATION_READY.json")
        return
    _set_gate(paths, state, None)
    _write_final_ready(config, inputs, paths)
    print(f"Pipeline completed -> {paths.final_ready_path}")
    print("Direct viewer: " + _viewer_command_text(config, paths, "direct", None, 8890))
    if config.physics.enabled:
        print("Physics viewer: " + _viewer_command_text(config, paths, "physics", None, 8891))


def _candidate_parameters_from_args(
    config: PipelineConfig,
    args: argparse.Namespace,
) -> GraphCandidateParameters:
    base = _default_graph_candidate(config)
    values = {
        "max_distance": base.max_distance if args.max_distance is None else float(args.max_distance),
        "max_neighbors": base.max_neighbors if args.max_neighbors is None else int(args.max_neighbors),
        "color_mad_multiplier": base.color_mad_multiplier if args.color_mad_multiplier is None else float(args.color_mad_multiplier),
        "depth_mad_multiplier": base.depth_mad_multiplier if args.depth_mad_multiplier is None else float(args.depth_mad_multiplier),
        "depth_samples": base.depth_samples if args.depth_samples is None else int(args.depth_samples),
        "min_shared_views": base.min_shared_views if args.min_shared_views is None else int(args.min_shared_views),
        "min_component_nodes": base.min_component_nodes if args.min_component_nodes is None else int(args.min_component_nodes),
        "min_component_edges": base.min_component_edges if args.min_component_edges is None else int(args.min_component_edges),
    }
    if values["max_distance"] <= 0.0:
        raise ValueError("--max-distance must be positive")
    if values["max_neighbors"] <= 0 or values["depth_samples"] < 2:
        raise ValueError("--max-neighbors must be positive and --depth-samples at least 2")
    if any(
        values[name] <= 0
        for name in ("min_shared_views", "min_component_nodes", "min_component_edges")
    ):
        raise ValueError("Rigid graph minimum counts must be positive")
    if values["color_mad_multiplier"] < 0.0 or values["depth_mad_multiplier"] < 0.0:
        raise ValueError("Rigid graph MAD multipliers must be non-negative")
    changed = any(
        getattr(base, name) != value for name, value in values.items()
    )
    if args.candidate_id is not None:
        candidate_id = _parse_id(args.candidate_id, "candidate_id")
    elif not changed:
        candidate_id = base.candidate_id
    else:
        candidate_id = "candidate_" + _config_hash(values)[:10]
    return GraphCandidateParameters(candidate_id=candidate_id, **values)


def _viewer_command_text(
    config: PipelineConfig,
    paths: PipelinePaths,
    target: str,
    candidate_id: str | None,
    port: int,
) -> str:
    if target == "static":
        approval = _read_json(_static_approval_path(paths)) if _static_approval_path(paths).is_file() else None
        work_dir = (
            Path(str(approval["accepted_cfg"])).parent
            if isinstance(approval, Mapping)
            else paths.static_work_dir
        )
        argv = ["python", "-u", "run_rendering.py", "--work-dir", str(work_dir), "--port", str(port)]
        return "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. " + " ".join(argv)
    if target == "rigid-graph":
        if config.solver.method != "rigid-components":
            raise ValueError(
                "The rigid-graph viewer is only available for solver.method="
                "rigid-components"
            )
        if candidate_id is None:
            approval_path = _graph_approval_path(paths)
            if approval_path.is_file():
                candidate_id = str(_read_json(approval_path)["candidate_id"])
            else:
                candidate_id = config.rigid_graph.default_candidate_id
        graph = _graph_candidate_path(paths, candidate_id)
        argv = [
            "python",
            "-u",
            "preproc/vis_observed_structure_graph.py",
            "--observed-graph-npz",
            str(graph),
            "--gaussian-npz",
            str(paths.gaussian_sidecar_path),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--max-visible-edges",
            "30000",
        ]
        return "PYTHONPATH=. " + " ".join(argv)
    if target == "direct":
        work_dir = paths.direct_checkpoint_dir
    elif target == "physics":
        if not config.physics.enabled:
            raise ValueError("Physics output is disabled in the pipeline config")
        work_dir = paths.physics_checkpoint_dir
    else:
        raise ValueError(f"Unknown viewer target: {target}")
    argv = ["python", "-u", "run_rendering.py", "--work-dir", str(work_dir), "--port", str(port)]
    return "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. " + " ".join(argv)


def _status_payload(config: PipelineConfig) -> dict[str, Any]:
    paths = _pipeline_paths(config)
    state = _read_json(paths.state_path) if paths.state_path.is_file() else None
    stage_receipts = (
        sorted(path.stem for path in paths.stages_dir.glob("*.json"))
        if paths.stages_dir.is_dir()
        else []
    )
    candidates = (
        sorted(path.name for path in paths.graph_candidates_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
        if paths.graph_candidates_dir.is_dir()
        else []
    )
    return {
        "format": "som_modal_experiment_pipeline_status",
        "version": 1,
        "scene_id": config.scene_id,
        "pipeline_id": config.pipeline_id,
        "started": state is not None,
        "gate": state.get("gate") if isinstance(state, Mapping) else None,
        "static_approved": _static_approval_path(paths).is_file(),
        "rigid_graph_approved": _graph_approval_path(paths).is_file(),
        "completed_stages": stage_receipts,
        "graph_candidates": candidates,
        "final_visualization_ready": paths.final_ready_path.is_file(),
        "final_ready_path": str(paths.final_ready_path),
    }


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the image-first Shape of Motion experiment pipeline after a "
            "completed pre-static joint-COLMAP stage."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run until the next manual gate or completion")
    _add_config_argument(run_parser)
    run_parser.add_argument("--dry-run", action="store_true")

    status_parser = subparsers.add_parser("status", help="Print persistent pipeline status")
    _add_config_argument(status_parser)

    extend_parser = subparsers.add_parser(
        "extend-static", help="Resume the unapproved static candidate to a larger total epoch count"
    )
    _add_config_argument(extend_parser)
    extend_parser.add_argument("--target-epochs", type=int, required=True)

    approve_static_parser = subparsers.add_parser(
        "approve-static", help="Freeze the current static checkpoint after visual inspection"
    )
    _add_config_argument(approve_static_parser)
    approve_static_parser.add_argument("--note", default=None)

    graph_parser = subparsers.add_parser(
        "build-graph-candidate", help="Build an immutable rigid-component graph candidate"
    )
    _add_config_argument(graph_parser)
    graph_parser.add_argument("--candidate-id", default=None)
    graph_parser.add_argument("--max-distance", type=float, default=None)
    graph_parser.add_argument("--max-neighbors", type=int, default=None)
    graph_parser.add_argument("--color-mad-multiplier", type=float, default=None)
    graph_parser.add_argument("--depth-mad-multiplier", type=float, default=None)
    graph_parser.add_argument("--depth-samples", type=int, default=None)
    graph_parser.add_argument("--min-shared-views", type=int, default=None)
    graph_parser.add_argument("--min-component-nodes", type=int, default=None)
    graph_parser.add_argument("--min-component-edges", type=int, default=None)
    graph_parser.add_argument("--dry-run", action="store_true")

    approve_graph_parser = subparsers.add_parser(
        "approve-graph", help="Approve one immutable rigid-component graph candidate"
    )
    _add_config_argument(approve_graph_parser)
    approve_graph_parser.add_argument("--candidate-id", required=True)
    approve_graph_parser.add_argument("--note", default=None)

    viewer_parser = subparsers.add_parser(
        "viewer-command", help="Print a cluster-side Viser command without running it"
    )
    _add_config_argument(viewer_parser)
    viewer_parser.add_argument(
        "--target", choices=["static", "rigid-graph", "direct", "physics"], required=True
    )
    viewer_parser.add_argument("--candidate-id", default=None)
    viewer_parser.add_argument("--port", type=int, default=8890)
    return parser


def _require_pipeline_prerequisites(
    config: PipelineConfig,
    *,
    read_only: bool = False,
) -> tuple[PrestaticInputs, PipelinePaths, dict[str, Any], Path]:
    inputs = load_prestatic_inputs(config)
    paths = _pipeline_paths(config)
    if read_only:
        if not paths.resolved_config_path.is_file() or not paths.state_path.is_file():
            raise FileNotFoundError(
                "Pipeline controller state is absent; run the pipeline before this dry-run"
            )
        if _read_json(paths.resolved_config_path) != _resolved_config_payload(
            config, inputs, paths
        ):
            raise ValueError("Existing resolved pipeline config differs")
        state = _require_mapping(_read_json(paths.state_path), "pipeline state")
        if (
            state.get("format") != STATE_FORMAT
            or state.get("version") != STATE_VERSION
            or state.get("config_identity") != config.config_identity
        ):
            raise ValueError(f"Existing pipeline state is incompatible: {paths.state_path}")
        state = dict(state)
    else:
        state = _prepare_controller(config, inputs, paths)
    approval = _load_static_approval(config, inputs, paths)
    if approval is None:
        raise ValueError("Static quality must be approved before this command")
    return inputs, paths, state, approval[0]


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "run":
        _run_pipeline(config, dry_run=bool(args.dry_run))
        return
    if args.command == "status":
        print(json.dumps(_status_payload(config), indent=2, allow_nan=False))
        return

    if args.command == "extend-static":
        inputs = load_prestatic_inputs(config)
        paths = _pipeline_paths(config)
        state = _prepare_controller(config, inputs, paths)
        if _load_static_approval(config, inputs, paths) is not None:
            raise ValueError("The static checkpoint is already approved and immutable")
        target = _parse_int(args.target_epochs, "--target-epochs")
        current_target = int(state["static_target_epochs"])
        if target <= current_target:
            raise ValueError(
                f"--target-epochs must exceed current target {current_target}"
            )
        state["static_target_epochs"] = target
        _set_gate(paths, state, None)
        _append_event(
            paths,
            {
                "event": "static_target_extended",
                "previous_target_epochs": current_target,
                "target_epochs": target,
            },
        )
        _run_static_candidate(config, inputs, paths, state, dry_run=False)
        _set_gate(
            paths,
            state,
            {
                "name": "static_quality",
                "status": "waiting_for_approval",
                "candidate_work_dir": str(paths.static_work_dir),
                "target_epochs": target,
            },
        )
        _print_static_gate(config, paths)
        return

    if args.command == "approve-static":
        inputs = load_prestatic_inputs(config)
        paths = _pipeline_paths(config)
        state = _prepare_controller(config, inputs, paths)
        _approve_static(config, inputs, paths, state, args.note)
        print(
            "Continue: "
            f"python -u {REPO_ROOT / 'run_experiment_pipeline.py'} run "
            f"--config {config.config_path}"
        )
        return

    if args.command == "build-graph-candidate":
        if config.solver.method != "rigid-components":
            raise ValueError("Graph candidates are only used by solver.method=rigid-components")
        inputs, paths, _state, accepted_checkpoint = _require_pipeline_prerequisites(
            config, read_only=bool(args.dry_run)
        )
        if not paths.topology_path.is_file() or not paths.view_configs_dir.is_dir():
            raise ValueError(
                "Run the pipeline through the rigid-graph gate before building candidates"
            )
        parameters = _candidate_parameters_from_args(config, args)
        _build_graph_candidate(
            config,
            inputs,
            paths,
            accepted_checkpoint,
            parameters,
            dry_run=bool(args.dry_run),
        )
        if not args.dry_run:
            print(
                _viewer_command_text(
                    config, paths, "rigid-graph", parameters.candidate_id, 8890
                )
            )
        return

    if args.command == "approve-graph":
        if config.solver.method != "rigid-components":
            raise ValueError("Graph approval is only used by rigid-components")
        inputs, paths, state, accepted_checkpoint = _require_pipeline_prerequisites(config)
        candidate_id = _parse_id(args.candidate_id, "--candidate-id")
        _approve_graph(
            config,
            inputs,
            paths,
            accepted_checkpoint,
            candidate_id,
            args.note,
        )
        _set_gate(paths, state, None)
        print(
            "Continue: "
            f"python -u {REPO_ROOT / 'run_experiment_pipeline.py'} run "
            f"--config {config.config_path}"
        )
        return

    if args.command == "viewer-command":
        if args.port <= 0 or args.port > 65535:
            raise ValueError("--port must lie in [1,65535]")
        paths = _pipeline_paths(config)
        print(
            "cd " + str(REPO_ROOT) + "\n" + _viewer_command_text(
                config,
                paths,
                args.target,
                args.candidate_id,
                int(args.port),
            )
        )
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
