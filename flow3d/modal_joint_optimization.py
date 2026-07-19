from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from flow3d.modal_bilateral_graph import (
    MODAL_BILATERAL_GRAPH_FORMAT,
    MODAL_BILATERAL_GRAPH_VERSION,
)
from flow3d.modal_flow_coordinates import (
    ModalFlowCoordinates,
    load_modal_flow_coordinates,
    parse_flow_cache_specs,
)
from flow3d.modal_utils import (
    GaussianModalFieldData,
    MOTION_FILL_DISPLAY_ANCHOR,
    MOTION_FILL_DISPLAY_FILLED,
    MOTION_FILL_DISPLAY_PARTIAL,
    stack_modal_motion_fill_display_classes,
)
from modal_peak_pick.core.cache import ModalAnalysisCache, load_analysis_cache


MODAL_JOINT_PARAMETERIZATION = "joint_flow_coordinates_phi_v1"
MODAL_JOINT_OBJECTIVE = "foreground_rgb_reference_flow_bilateral_rigidity_v1"
MODAL_PHI_PARAMETERIZATION = "fixed_flow_coordinates_trainable_phi_v1"
MODAL_PHI_OBJECTIVE = "foreground_rgb_reference_flow_mode_rigidity_v1"


@dataclass(frozen=True)
class ModalJointGraph:
    path: Path
    edge_index: np.ndarray
    edge_weight: np.ndarray
    graph_degree: np.ndarray
    mode_local_degree: np.ndarray
    trainable_phi_mask: np.ndarray
    display_class: np.ndarray


@dataclass(frozen=True)
class ModalJointTrainingContext:
    graph_path: Path
    view_ids: tuple[str, ...]
    flow_caches: tuple[ModalAnalysisCache, ...]
    reference_flows: tuple[np.ndarray, ...]
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    display_class: torch.Tensor
    reference_coordinate_real: torch.Tensor
    reference_coordinate_imag: torch.Tensor
    reference_active_ts: torch.Tensor
    coordinate_scale: torch.Tensor
    mode_rigidity_probe_scale: torch.Tensor
    flow_height: int
    flow_width: int

    def load_flow_batch(
        self,
        frame_view_indices: torch.Tensor,
        frame_local_indices: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        view_indices = frame_view_indices.detach().cpu().numpy().astype(np.int64)
        local_indices = frame_local_indices.detach().cpu().numpy().astype(np.int64)
        flows: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for view_index, local_index in zip(view_indices, local_indices):
            cache = self.flow_caches[int(view_index)]
            flow_u = np.asarray(cache.flow_u[int(local_index)], dtype=np.float32)
            flow_v = np.asarray(cache.flow_v[int(local_index)], dtype=np.float32)
            flow = np.stack([flow_u, flow_v], axis=-1) - self.reference_flows[
                int(view_index)
            ]
            if not np.all(np.isfinite(flow)):
                raise ValueError(
                    f"Flow cache for {self.view_ids[int(view_index)]!r} contains "
                    f"non-finite values at local frame {int(local_index)}"
                )
            cache_mask = cache.mask
            if cache_mask is None:
                raise ValueError(
                    f"Flow cache for {self.view_ids[int(view_index)]!r} has no mask"
                )
            flows.append(flow)
            masks.append(np.asarray(cache_mask, dtype=np.bool_))
        flow_tensor = torch.as_tensor(
            np.stack(flows, axis=0), device=device, dtype=dtype
        )
        mask_tensor = torch.as_tensor(
            np.stack(masks, axis=0), device=device, dtype=torch.bool
        )
        return flow_tensor, mask_tensor


def _load_graph_arrays(
    path: Path,
    gaussian_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as graph:
        required = {
            "format",
            "version",
            "points_world",
            "edge_index",
            "edge_weight",
            "degree",
        }
        missing = sorted(required - set(graph.files))
        if missing:
            raise ValueError(f"{path} is missing bilateral graph fields: {missing}")
        if str(graph["format"].item()) != MODAL_BILATERAL_GRAPH_FORMAT:
            raise ValueError(f"{path} is not a modal bilateral Gaussian graph")
        if int(graph["version"].item()) != MODAL_BILATERAL_GRAPH_VERSION:
            raise ValueError(f"{path} has an unsupported bilateral graph version")
        points = np.asarray(graph["points_world"], dtype=np.float32)
        edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
        edge_weight = np.asarray(graph["edge_weight"], dtype=np.float32)
        degree = np.asarray(graph["degree"], dtype=np.int64)
    if points.shape != gaussian_means.shape:
        raise ValueError(
            f"{path} points_world shape {points.shape} does not match foreground "
            f"Gaussian means {gaussian_means.shape}"
        )
    maximum_delta = float(np.max(np.abs(points - gaussian_means)))
    if maximum_delta > 1e-6:
        raise ValueError(
            f"{path} points_world differs from foreground means by {maximum_delta:.6g}, "
            "above 1e-6"
        )
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError(f"{path} edge_index must have shape (E,2)")
    if edge_weight.shape != (edge_index.shape[0],):
        raise ValueError(f"{path} edge_weight must have shape ({edge_index.shape[0]},)")
    if degree.shape != (points.shape[0],):
        raise ValueError(f"{path} degree must have shape ({points.shape[0]},)")
    if np.any(edge_index < 0) or np.any(edge_index >= points.shape[0]):
        raise ValueError(f"{path} edge_index contains an out-of-range point")
    if not np.all(np.isfinite(edge_weight)) or np.any(edge_weight < 0.0):
        raise ValueError(f"{path} edge_weight must be finite and non-negative")
    if not np.any(edge_weight > 0.0):
        raise ValueError(f"{path} edge_weight must contain a positive value")
    if np.any(degree < 0):
        raise ValueError(f"{path} degree must be non-negative")
    expected_degree = np.bincount(edge_index.reshape(-1), minlength=points.shape[0])
    if not np.array_equal(degree, expected_degree):
        raise ValueError(f"{path} degree does not match edge_index")
    return edge_index, edge_weight, degree


def load_modal_joint_graph(
    graph_path: str | Path,
    modal_fields: GaussianModalFieldData,
    gaussian_means: torch.Tensor,
    minimum_mode_degree: int,
) -> ModalJointGraph:
    if isinstance(minimum_mode_degree, bool) or minimum_mode_degree <= 0:
        raise ValueError("minimum_mode_degree must be a positive integer")
    display_class = stack_modal_motion_fill_display_classes(modal_fields.modes)
    if display_class is None:
        raise ValueError("Joint phi optimization requires motion-fill role metadata")
    means = gaussian_means.detach().cpu().numpy().astype(np.float32)
    path = Path(graph_path).expanduser().resolve()
    edge_index, edge_weight, graph_degree = _load_graph_arrays(path, means)
    allowed = (
        (display_class == MOTION_FILL_DISPLAY_ANCHOR)
        | (display_class == MOTION_FILL_DISPLAY_PARTIAL)
        | (display_class == MOTION_FILL_DISPLAY_FILLED)
    )
    mode_local_degree = np.zeros(display_class.shape, dtype=np.int32)
    for mode_slot, mode_allowed in enumerate(allowed):
        selected = mode_allowed[edge_index[:, 0]] & mode_allowed[edge_index[:, 1]]
        selected_edges = edge_index[selected]
        mode_local_degree[mode_slot] = np.bincount(
            selected_edges.reshape(-1), minlength=means.shape[0]
        ).astype(np.int32)
    trainable = allowed & (mode_local_degree >= int(minimum_mode_degree))
    empty_modes = np.flatnonzero(np.count_nonzero(trainable, axis=1) == 0)
    if empty_modes.size:
        raise ValueError(
            "Bilateral graph leaves no trainable phi points for mode slots "
            f"{empty_modes.tolist()}"
        )
    return ModalJointGraph(
        path=path,
        edge_index=edge_index,
        edge_weight=edge_weight,
        graph_degree=graph_degree,
        mode_local_degree=mode_local_degree,
        trainable_phi_mask=trainable,
        display_class=display_class,
    )


def _validate_flow_caches(
    flow_cache_specs: Sequence[str],
    coordinates: ModalFlowCoordinates,
) -> tuple[ModalAnalysisCache, ...]:
    parsed = parse_flow_cache_specs(flow_cache_specs)
    view_ids = tuple(view_id for view_id, _ in parsed)
    if view_ids != coordinates.view_ids:
        raise ValueError(
            "--modal-flow-caches order must match coordinate view_ids: "
            f"expected {coordinates.view_ids}, got {view_ids}"
        )
    caches: list[ModalAnalysisCache] = []
    for view_index, (view_id, cache_path) in enumerate(parsed):
        cache = load_analysis_cache(cache_path)
        if cache.path.resolve() != coordinates.source_flow_cache_dirs[view_index].resolve():
            raise ValueError(
                f"Flow cache for {view_id!r} does not match the coordinate artifact source"
            )
        rows = np.flatnonzero(coordinates.frame_view_indices == view_index)
        if cache.flow_u.shape[0] != rows.size:
            raise ValueError(
                f"Flow cache for {view_id!r} has {cache.flow_u.shape[0]} frames, "
                f"expected {rows.size}"
            )
        if cache.mask is None:
            raise ValueError(f"Flow cache for {view_id!r} must contain a mask")
        reference_index = int(coordinates.reference_local_indices[view_index])
        if int(cache.metadata["analysis"]["reference_frame_index"]) != reference_index:
            raise ValueError(f"Flow cache reference index for {view_id!r} is inconsistent")
        caches.append(cache)
    spatial_shapes = {cache.flow_u.shape[1:] for cache in caches}
    if len(spatial_shapes) != 1:
        raise ValueError("All modal flow caches must share one spatial resolution")
    return tuple(caches)


def load_modal_joint_training_context(
    *,
    model: Any,
    modal_fields: GaussianModalFieldData,
    graph_path: str | Path,
    flow_cache_specs: Sequence[str],
    coordinate_path: str | Path,
    minimum_mode_degree: int,
    device: torch.device,
) -> ModalJointTrainingContext:
    coordinates = load_modal_flow_coordinates(coordinate_path)
    graph = load_modal_joint_graph(
        graph_path,
        modal_fields,
        model.fg.params["means"],
        minimum_mode_degree,
    )
    expected_mask = model.modal_phi_trainable_mask.detach().cpu().numpy()
    if not np.array_equal(expected_mask, graph.trainable_phi_mask):
        raise ValueError("Checkpoint modal phi trainable mask does not match graph/roles")
    if not np.array_equal(coordinates.mode_indices, np.asarray(
        [mode.mode_index for mode in modal_fields.modes], dtype=np.int64
    )):
        raise ValueError("Coordinate mode order does not match modal manifest")
    if not np.allclose(
        coordinates.frequencies_hz,
        modal_fields.freqs_hz.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("Coordinate frequencies do not match modal manifest")
    caches = _validate_flow_caches(flow_cache_specs, coordinates)
    num_views = len(coordinates.view_ids)
    num_modes = len(modal_fields.modes)
    reference_real = np.empty((num_views, num_modes), dtype=np.float32)
    reference_imag = np.empty_like(reference_real)
    reference_active_ts = np.full((num_views,), -1, dtype=np.int64)
    model_view = model.modal_frame_view_indices.detach().cpu().numpy()
    model_local = model.modal_frame_local_indices.detach().cpu().numpy()
    model_times = model.modal_frame_times_sec.detach().cpu().numpy()
    coordinate_rows = {
        (int(view_index), int(local_index)): row
        for row, (view_index, local_index) in enumerate(
            zip(coordinates.frame_view_indices, coordinates.frame_local_indices)
        )
    }
    selected_rows = np.empty(model_view.shape[0], dtype=np.int64)
    for model_row, key in enumerate(zip(model_view, model_local)):
        normalized_key = (int(key[0]), int(key[1]))
        if normalized_key not in coordinate_rows:
            raise ValueError(
                f"Coordinate artifact has no active frame metadata {normalized_key}"
            )
        selected_rows[model_row] = coordinate_rows[normalized_key]
    expected_times = coordinates.frame_times_sec[selected_rows].astype(
        model_times.dtype,
        copy=False,
    )
    if not np.array_equal(model_times, expected_times):
        raise ValueError("Checkpoint frame times do not match coordinate artifact")
    for name, actual, expected in (
        (
            "real",
            model.modal_coordinate_real.detach().cpu().numpy(),
            coordinates.coordinate_real[selected_rows],
        ),
        (
            "imaginary",
            model.modal_coordinate_imag.detach().cpu().numpy(),
            coordinates.coordinate_imag[selected_rows],
        ),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"Checkpoint base modal coordinate {name} values do not match artifact"
            )
    for view_index, reference_local in enumerate(coordinates.reference_local_indices):
        rows = np.flatnonzero(
            (coordinates.frame_view_indices == view_index)
            & (coordinates.frame_local_indices == int(reference_local))
        )
        if rows.size != 1:
            raise ValueError(
                f"Coordinate artifact does not uniquely contain reference frame for "
                f"{coordinates.view_ids[view_index]!r}"
            )
        row = int(rows[0])
        reference_real[view_index] = coordinates.coordinate_real[row]
        reference_imag[view_index] = coordinates.coordinate_imag[row]
        active = np.flatnonzero(
            (model_view == view_index) & (model_local == int(reference_local))
        )
        if active.size > 1:
            raise ValueError("Active frame map repeats a view reference frame")
        if active.size == 1:
            reference_active_ts[view_index] = int(active[0])

    coordinate_scale = np.empty((num_views, num_modes), dtype=np.float32)
    model_q_real = model.modal_coordinate_real.detach().cpu().numpy()
    model_q_imag = model.modal_coordinate_imag.detach().cpu().numpy()
    for view_index in range(num_views):
        rows = model_view == view_index
        if not np.any(rows):
            raise ValueError(
                "Active training data has no frames for view "
                f"{coordinates.view_ids[view_index]!r}"
            )
        energy = np.mean(model_q_real[rows] ** 2 + model_q_imag[rows] ** 2, axis=0)
        coordinate_scale[view_index] = np.sqrt(np.maximum(energy, 1e-12))

    mode_probe_scale = np.percentile(
        np.sqrt(model_q_real**2 + model_q_imag**2),
        90,
        axis=0,
    ).astype(np.float32)
    positive_probe_scale = mode_probe_scale[mode_probe_scale > 0.0]
    if positive_probe_scale.size == 0:
        raise ValueError("Fixed modal coordinates have zero amplitude for every mode")
    probe_floor = 0.25 * float(np.median(positive_probe_scale))
    mode_probe_scale = np.maximum(mode_probe_scale, probe_floor)

    height, width = caches[0].flow_u.shape[1:]
    reference_flows = tuple(
        np.stack(
            [
                np.asarray(
                    cache.flow_u[int(cache.metadata["analysis"]["reference_frame_index"])],
                    dtype=np.float32,
                ),
                np.asarray(
                    cache.flow_v[int(cache.metadata["analysis"]["reference_frame_index"])],
                    dtype=np.float32,
                ),
            ],
            axis=-1,
        )
        for cache in caches
    )
    return ModalJointTrainingContext(
        graph_path=graph.path,
        view_ids=coordinates.view_ids,
        flow_caches=caches,
        reference_flows=reference_flows,
        edge_index=torch.as_tensor(graph.edge_index, device=device, dtype=torch.long),
        edge_weight=torch.as_tensor(graph.edge_weight, device=device, dtype=torch.float32),
        display_class=torch.as_tensor(graph.display_class, device=device, dtype=torch.long),
        reference_coordinate_real=torch.as_tensor(reference_real, device=device),
        reference_coordinate_imag=torch.as_tensor(reference_imag, device=device),
        reference_active_ts=torch.as_tensor(reference_active_ts, device=device),
        coordinate_scale=torch.as_tensor(coordinate_scale, device=device),
        mode_rigidity_probe_scale=torch.as_tensor(mode_probe_scale, device=device),
        flow_height=int(height),
        flow_width=int(width),
    )


def weighted_rigidity_loss(
    dynamic_means: torch.Tensor,
    canonical_means: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    huber_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = edge_index[:, 0]
    target = edge_index[:, 1]
    rest = torch.linalg.vector_norm(
        canonical_means[source] - canonical_means[target], dim=-1
    ).clamp_min(1e-8)
    dynamic = torch.linalg.vector_norm(
        dynamic_means[source] - dynamic_means[target], dim=-1
    )
    strain = (dynamic - rest[:, None]) / rest[:, None]
    robust = F.smooth_l1_loss(
        strain,
        torch.zeros_like(strain),
        beta=float(huber_beta),
        reduction="none",
    )
    weights = edge_weight[:, None].to(dtype=robust.dtype)
    loss = torch.sum(weights * robust) / (torch.sum(weights) * strain.shape[1])
    return loss, strain.abs().mean(), strain.abs().amax()


def weighted_mode_rigidity_loss(
    phi_real: torch.Tensor,
    phi_imag: torch.Tensor,
    canonical_means: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    probe_scale: torch.Tensor,
    mode_slots: torch.Tensor,
    trainable_mask: torch.Tensor,
    huber_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode_slots.ndim != 1 or mode_slots.numel() == 0:
        raise ValueError("mode_slots must be a non-empty one-dimensional tensor")
    source = edge_index[:, 0]
    target = edge_index[:, 1]
    weights = edge_weight.to(dtype=phi_real.dtype)
    phase_vectors = (
        (1.0, 0.0),
        (0.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
    )
    losses: list[torch.Tensor] = []
    strains: list[torch.Tensor] = []
    for mode_slot in mode_slots.tolist():
        slot = int(mode_slot)
        selected = trainable_mask[slot, source] | trainable_mask[slot, target]
        if not bool(selected.any().item()):
            raise RuntimeError(f"Mode slot {slot} has no trainable bilateral edges")
        selected_source = source[selected]
        selected_target = target[selected]
        rest_vectors = (
            canonical_means[selected_source] - canonical_means[selected_target]
        )
        rest_lengths = torch.linalg.vector_norm(rest_vectors, dim=-1).clamp_min(1e-8)
        selected_weights = weights[selected]
        weight_sum = selected_weights.sum().clamp_min(1e-12)
        real_edge = (
            phi_real[slot, selected_source] - phi_real[slot, selected_target]
        )
        imag_edge = (
            phi_imag[slot, selected_source] - phi_imag[slot, selected_target]
        )
        mode_losses: list[torch.Tensor] = []
        mode_strains: list[torch.Tensor] = []
        amplitude = probe_scale[slot].to(dtype=phi_real.dtype)
        for cosine, negative_sine in phase_vectors:
            displacement = amplitude * (cosine * real_edge + negative_sine * imag_edge)
            deformed_lengths = torch.linalg.vector_norm(
                rest_vectors + displacement,
                dim=-1,
            )
            strain = (deformed_lengths - rest_lengths) / rest_lengths
            robust = F.smooth_l1_loss(
                strain,
                torch.zeros_like(strain),
                beta=float(huber_beta),
                reduction="none",
            )
            mode_losses.append(torch.sum(selected_weights * robust) / weight_sum)
            mode_strains.append(strain.abs())
        losses.append(torch.stack(mode_losses).mean())
        strains.append(torch.stack(mode_strains))
    all_strains = torch.stack(strains)
    return torch.stack(losses).mean(), all_strains.mean(), all_strains.amax()


def weighted_delta_phi_local_loss(
    delta_phi_real: torch.Tensor,
    delta_phi_imag: torch.Tensor,
    canonical_means: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    mode_slots: torch.Tensor,
    trainable_mask: torch.Tensor,
) -> torch.Tensor:
    source = edge_index[:, 0]
    target = edge_index[:, 1]
    rest_squared = (
        (canonical_means[source] - canonical_means[target]).square().sum(dim=-1)
    ).clamp_min(1e-16)
    weights = edge_weight.to(dtype=delta_phi_real.dtype)
    losses: list[torch.Tensor] = []
    for mode_slot in mode_slots.tolist():
        slot = int(mode_slot)
        selected = trainable_mask[slot, source] & trainable_mask[slot, target]
        if not bool(selected.any().item()):
            raise RuntimeError(f"Mode slot {slot} has no trainable bilateral edges")
        real_difference = (
            delta_phi_real[slot, source[selected]]
            - delta_phi_real[slot, target[selected]]
        )
        imag_difference = (
            delta_phi_imag[slot, source[selected]]
            - delta_phi_imag[slot, target[selected]]
        )
        normalized = (
            real_difference.square().sum(dim=-1)
            + imag_difference.square().sum(dim=-1)
        ) / rest_squared[selected]
        selected_weights = weights[selected]
        losses.append(
            torch.sum(selected_weights * normalized)
            / selected_weights.sum().clamp_min(1e-12)
        )
    return torch.stack(losses).mean()


__all__ = [
    "MODAL_JOINT_OBJECTIVE",
    "MODAL_JOINT_PARAMETERIZATION",
    "MODAL_PHI_OBJECTIVE",
    "MODAL_PHI_PARAMETERIZATION",
    "ModalJointGraph",
    "ModalJointTrainingContext",
    "load_modal_joint_graph",
    "load_modal_joint_training_context",
    "weighted_rigidity_loss",
    "weighted_mode_rigidity_loss",
    "weighted_delta_phi_local_loss",
]
