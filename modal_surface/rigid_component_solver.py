"""First-order rigid modal solve over observed-graph components.

Each connected, non-isolated observed-graph component is represented by one
complex infinitesimal SE(3) twist.  The parameterization enforces zero
first-order axial strain for every point pair in the component; finite modal
playback remains the repository's additive harmonic displacement model and is
therefore only rigid to first order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modal_surface.observed_structure_graph import ObservedStructureGraph
from modal_surface.optimization_staged import AlphaSyncResult, PreparedObservations


_EPS = 1.0e-12
_FINITE_DRIFT_EDGE_BATCH_SIZE = 32768


@dataclass(frozen=True)
class RigidComponentSolverConfig:
    rcond: float = 1.0e-8
    first_order_rtol: float = 1.0e-6
    phase_samples: int = 64

    def validate(self) -> None:
        if not np.isfinite(self.rcond) or not 0.0 < self.rcond < 1.0:
            raise ValueError("rigid component rcond must be finite and lie in (0,1)")
        if (
            not np.isfinite(self.first_order_rtol)
            or self.first_order_rtol <= 0.0
        ):
            raise ValueError(
                "rigid component first_order_rtol must be finite and positive"
            )
        if (
            isinstance(self.phase_samples, (bool, np.bool_))
            or not isinstance(self.phase_samples, (int, np.integer))
            or self.phase_samples <= 0
        ):
            raise ValueError("rigid component phase_samples must be a positive integer")


@dataclass(frozen=True)
class RigidComponentSolveResult:
    config: RigidComponentSolverConfig
    phi: np.ndarray
    rigid_seed_mask: np.ndarray
    observed_mask: np.ndarray
    fill_target_mask: np.ndarray
    point_component_index: np.ndarray
    component_graph_index: np.ndarray
    component_node_count: np.ndarray
    component_edge_count: np.ndarray
    component_centroid: np.ndarray
    component_radius: np.ndarray
    component_usable_observation_row_count: np.ndarray
    component_distinct_valid_view_count: np.ndarray
    component_singular_values: np.ndarray
    component_rank: np.ndarray
    component_condition: np.ndarray
    component_weighted_residual_norm: np.ndarray
    component_weighted_measurement_norm: np.ndarray
    component_normalized_weighted_residual: np.ndarray
    component_translation: np.ndarray
    component_rotation: np.ndarray
    edge_component_index: np.ndarray
    edge_first_order_axial_real: np.ndarray
    edge_first_order_axial_imag: np.ndarray
    edge_first_order_relative_real: np.ndarray
    edge_first_order_relative_imag: np.ndarray
    phase_angles: np.ndarray
    edge_finite_drift_p50: np.ndarray
    edge_finite_drift_p90: np.ndarray
    edge_finite_drift_max: np.ndarray
    component_finite_drift_p50: np.ndarray
    component_finite_drift_p90: np.ndarray
    component_finite_drift_max: np.ndarray

    @property
    def num_components(self) -> int:
        return int(self.component_node_count.shape[0])

    @property
    def num_rigid_seeds(self) -> int:
        return int(np.count_nonzero(self.rigid_seed_mask))


def _validate_inputs(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    graph: ObservedStructureGraph,
) -> None:
    num_points = int(prepared.points.shape[0])
    num_views = int(prepared.num_views)
    node_indices = np.asarray(graph.node_gaussian_indices)
    node_points = np.asarray(graph.node_points_world)
    topology = graph.topology
    edge_index = np.asarray(topology.edge_index)
    num_nodes = int(node_indices.shape[0])

    if node_indices.ndim != 1 or not np.issubdtype(node_indices.dtype, np.integer):
        raise ValueError(
            "observed structure graph node_gaussian_indices must be a 1-D integer array"
        )
    if np.any(node_indices < 0) or np.any(node_indices >= num_points):
        raise ValueError(
            "observed structure graph contains an out-of-range Gaussian index"
        )
    if node_indices.size > 1 and np.any(node_indices[1:] <= node_indices[:-1]):
        raise ValueError(
            "observed structure graph Gaussian indices must be strictly increasing"
        )
    if node_points.shape != (num_nodes, 3):
        raise ValueError(
            "observed structure graph node_points_world must have shape (M,3)"
        )
    if not np.array_equal(
        node_points.astype(np.float32, copy=False),
        prepared.points[node_indices].astype(np.float32, copy=False),
    ):
        raise ValueError(
            "observed structure graph static positions do not match observations"
        )
    if edge_index.ndim != 2 or edge_index.shape[1:] != (2,):
        raise ValueError("observed structure graph edge_index must have shape (E,2)")
    if not np.issubdtype(edge_index.dtype, np.integer):
        raise ValueError("observed structure graph edge_index must be integer-valued")
    if np.any(edge_index < 0) or np.any(edge_index >= num_nodes):
        raise ValueError("observed structure graph edge_index is out of range")
    if edge_index.size and np.any(edge_index[:, 0] >= edge_index[:, 1]):
        raise ValueError(
            "observed structure graph edges must be ordered with source < target"
        )
    if edge_index.shape[0] > 1:
        edge_order = np.lexsort((edge_index[:, 1], edge_index[:, 0]))
        if not np.array_equal(edge_order, np.arange(edge_index.shape[0])):
            raise ValueError(
                "observed structure graph edges must be lexicographically sorted"
            )
        if np.any(np.all(edge_index[1:] == edge_index[:-1], axis=1)):
            raise ValueError("observed structure graph contains duplicate edges")

    degree = np.asarray(topology.degree)
    component_index = np.asarray(topology.component_index)
    component_size = np.asarray(topology.component_size)
    isolated_mask = np.asarray(topology.isolated_mask)
    if degree.shape != (num_nodes,) or component_index.shape != (num_nodes,):
        raise ValueError(
            "observed structure graph degree/component arrays must match its nodes"
        )
    if isolated_mask.shape != (num_nodes,) or isolated_mask.dtype != np.bool_:
        raise ValueError(
            "observed structure graph isolated_mask must be a boolean node array"
        )
    if not np.issubdtype(degree.dtype, np.integer) or np.any(degree < 0):
        raise ValueError("observed structure graph degree must be non-negative integers")
    if not np.issubdtype(component_index.dtype, np.integer):
        raise ValueError("observed structure graph component_index must be integer-valued")
    recomputed_degree = np.zeros((num_nodes,), dtype=np.int64)
    if edge_index.size:
        np.add.at(recomputed_degree, edge_index[:, 0], 1)
        np.add.at(recomputed_degree, edge_index[:, 1], 1)
    if not np.array_equal(degree.astype(np.int64), recomputed_degree):
        raise ValueError("observed structure graph degree disagrees with edge_index")
    if not np.array_equal(isolated_mask, recomputed_degree == 0):
        raise ValueError("observed structure graph isolated_mask disagrees with degree")
    if num_nodes:
        if np.any(component_index < 0):
            raise ValueError(
                "observed structure graph component_index must be non-negative"
            )
        num_graph_components = int(component_index.max()) + 1
        if component_size.shape != (num_graph_components,):
            raise ValueError(
                "observed structure graph component_size does not match component labels"
            )
        recomputed_sizes = np.bincount(
            component_index.astype(np.int64), minlength=num_graph_components
        )
        if not np.array_equal(component_size.astype(np.int64), recomputed_sizes):
            raise ValueError(
                "observed structure graph component_size disagrees with component_index"
            )
        if edge_index.size and np.any(
            component_index[edge_index[:, 0]] != component_index[edge_index[:, 1]]
        ):
            raise ValueError("observed structure graph edge crosses component labels")
    elif component_size.shape != (0,):
        raise ValueError(
            "an empty observed structure graph must have no component sizes"
        )

    node_view_mask = np.asarray(graph.node_observed_view_mask)
    if node_view_mask.shape != (num_nodes, num_views) or node_view_mask.dtype != np.bool_:
        raise ValueError(
            "observed structure graph node_observed_view_mask has the wrong shape or dtype"
        )
    if np.asarray(alpha.alphas).shape != (num_views,):
        raise ValueError("alpha values must match the prepared observation views")
    if np.asarray(alpha.identifiable_mask).shape != (num_views,):
        raise ValueError("alpha identifiable mask must match the observation views")
    identifiable_alpha = np.asarray(alpha.alphas)[np.asarray(alpha.identifiable_mask)]
    if np.any(~np.isfinite(identifiable_alpha.real)) or np.any(
        ~np.isfinite(identifiable_alpha.imag)
    ):
        raise ValueError("identifiable alpha values must be finite")


def _batch_skew(vectors: np.ndarray) -> np.ndarray:
    skew = np.zeros((vectors.shape[0], 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]
    return skew


def _minimum_norm_svd(
    design: np.ndarray,
    target: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    U, singular, Vh = np.linalg.svd(design, full_matrices=False)
    if singular.size == 0 or singular[0] <= _EPS:
        return np.zeros((design.shape[1],), dtype=np.complex128), singular, 0
    rank = int(np.count_nonzero(singular > rcond * singular[0]))
    if rank == 0:
        solution = np.zeros((design.shape[1],), dtype=np.complex128)
    else:
        coefficients = (U[:, :rank].conj().T @ target) / singular[:rank]
        solution = Vh[:rank].conj().T @ coefficients
    return solution.astype(np.complex128, copy=False), singular, rank


def _finite_edge_drift_statistics(
    edge_vectors: np.ndarray,
    edge_delta_phi: np.ndarray,
    phase_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_edges = int(edge_vectors.shape[0])
    p50 = np.zeros((num_edges,), dtype=np.float64)
    p90 = np.zeros((num_edges,), dtype=np.float64)
    maximum = np.zeros((num_edges,), dtype=np.float64)
    cos_phase = np.cos(phase_angles)[None, :]
    sin_phase = np.sin(phase_angles)[None, :]

    for start in range(0, num_edges, _FINITE_DRIFT_EDGE_BATCH_SIZE):
        end = min(start + _FINITE_DRIFT_EDGE_BATCH_SIZE, num_edges)
        base = edge_vectors[start:end]
        delta = edge_delta_phi[start:end]
        real = delta.real
        imag = delta.imag
        base_squared = np.einsum("ij,ij->i", base, base)[:, None]
        base_real = np.einsum("ij,ij->i", base, real)[:, None]
        base_imag = np.einsum("ij,ij->i", base, imag)[:, None]
        real_squared = np.einsum("ij,ij->i", real, real)[:, None]
        imag_squared = np.einsum("ij,ij->i", imag, imag)[:, None]
        real_imag = np.einsum("ij,ij->i", real, imag)[:, None]
        deformed_squared = (
            base_squared
            + 2.0 * (base_real * cos_phase - base_imag * sin_phase)
            + real_squared * cos_phase**2
            + imag_squared * sin_phase**2
            - 2.0 * real_imag * cos_phase * sin_phase
        )
        deformed_length = np.sqrt(np.maximum(deformed_squared, 0.0))
        base_length = np.sqrt(np.maximum(base_squared, 0.0))
        drift = np.abs(deformed_length - base_length) / np.maximum(
            base_length, _EPS
        )
        p50[start:end] = np.percentile(drift, 50.0, axis=1)
        p90[start:end] = np.percentile(drift, 90.0, axis=1)
        maximum[start:end] = np.max(drift, axis=1)
    return p50, p90, maximum


def solve_rigid_components(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    graph: ObservedStructureGraph,
    config: RigidComponentSolverConfig | None = None,
) -> RigidComponentSolveResult:
    """Jointly solve one complex first-order rigid twist per graph component."""

    config = config or RigidComponentSolverConfig()
    config.validate()
    _validate_inputs(prepared, alpha, graph)

    points = prepared.points.astype(np.float64)
    num_points = int(points.shape[0])
    node_indices = np.asarray(graph.node_gaussian_indices, dtype=np.int64)
    topology = graph.topology
    degree = np.asarray(topology.degree, dtype=np.int64)
    graph_component = np.asarray(topology.component_index, dtype=np.int64)
    edge_index = np.asarray(topology.edge_index, dtype=np.int64)
    rigid_node_mask = degree > 0
    selected_graph_components = np.unique(graph_component[rigid_node_mask])
    num_components = int(selected_graph_components.shape[0])
    if num_components == 0:
        raise ValueError(
            "observed structure graph has no connected component with an accepted edge"
        )

    graph_component_to_rigid = np.full(
        (int(graph_component.max()) + 1,), -1, dtype=np.int32
    )
    graph_component_to_rigid[selected_graph_components] = np.arange(
        num_components, dtype=np.int32
    )
    node_component = np.full((node_indices.shape[0],), -1, dtype=np.int32)
    node_component[rigid_node_mask] = graph_component_to_rigid[
        graph_component[rigid_node_mask]
    ]
    rigid_seed_indices = node_indices[rigid_node_mask]
    rigid_seed_mask = np.zeros((num_points,), dtype=bool)
    rigid_seed_mask[rigid_seed_indices] = True
    observed_mask = np.zeros((num_points,), dtype=bool)
    observed_mask[node_indices] = True
    point_component_index = np.full((num_points,), -1, dtype=np.int32)
    point_component_index[rigid_seed_indices] = node_component[rigid_node_mask]

    component_node_count = np.bincount(
        node_component[rigid_node_mask], minlength=num_components
    ).astype(np.int32)
    if np.any(component_node_count < 2):
        raise RuntimeError(
            "a selected rigid component contains fewer than two connected nodes"
        )
    component_centroid = np.zeros((num_components, 3), dtype=np.float64)
    np.add.at(
        component_centroid,
        node_component[rigid_node_mask],
        points[rigid_seed_indices],
    )
    component_centroid /= component_node_count[:, None]
    centered_seed_points = (
        points[rigid_seed_indices]
        - component_centroid[node_component[rigid_node_mask]]
    )
    component_squared_radius = np.zeros((num_components,), dtype=np.float64)
    np.add.at(
        component_squared_radius,
        node_component[rigid_node_mask],
        np.einsum("ij,ij->i", centered_seed_points, centered_seed_points),
    )
    component_radius = np.sqrt(
        component_squared_radius / component_node_count.astype(np.float64)
    )

    edge_component_index = node_component[edge_index[:, 0]]
    if np.any(edge_component_index < 0) or np.any(
        edge_component_index != node_component[edge_index[:, 1]]
    ):
        raise RuntimeError("accepted graph edges do not map to one rigid component")
    component_edge_count = np.bincount(
        edge_component_index, minlength=num_components
    ).astype(np.int32)

    usable_row_mask = (
        (prepared.obs_weights > 0.0)
        & alpha.identifiable_mask[prepared.obs_view_index]
        & rigid_seed_mask[prepared.obs_point_index]
    )
    usable_rows = np.flatnonzero(usable_row_mask)
    usable_row_components = point_component_index[
        prepared.obs_point_index[usable_rows]
    ]
    row_order = np.argsort(usable_row_components, kind="stable")
    usable_rows = usable_rows[row_order]
    usable_row_components = usable_row_components[row_order]
    component_row_count = np.bincount(
        usable_row_components, minlength=num_components
    ).astype(np.int32)
    row_offsets = np.concatenate(
        [np.zeros((1,), dtype=np.int64), np.cumsum(component_row_count)]
    )

    phi = np.zeros((num_points, 3), dtype=np.complex128)
    component_view_count = np.zeros((num_components,), dtype=np.int32)
    component_singular = np.zeros((num_components, 6), dtype=np.float64)
    component_rank = np.zeros((num_components,), dtype=np.int8)
    component_condition = np.full((num_components,), np.inf, dtype=np.float64)
    component_residual_norm = np.zeros((num_components,), dtype=np.float64)
    component_measurement_norm = np.zeros((num_components,), dtype=np.float64)
    component_normalized_residual = np.zeros((num_components,), dtype=np.float64)
    component_translation = np.zeros((num_components, 3), dtype=np.complex128)
    component_rotation = np.zeros((num_components, 3), dtype=np.complex128)

    rigid_node_order = np.argsort(node_component[rigid_node_mask], kind="stable")
    ordered_rigid_indices = rigid_seed_indices[rigid_node_order]
    node_offsets = np.concatenate(
        [np.zeros((1,), dtype=np.int64), np.cumsum(component_node_count)]
    )

    for component_idx in range(num_components):
        rows = usable_rows[
            row_offsets[component_idx] : row_offsets[component_idx + 1]
        ]
        if rows.size == 0:
            graph_idx = int(selected_graph_components[component_idx])
            component_nodes = ordered_rigid_indices[
                node_offsets[component_idx] : node_offsets[component_idx + 1]
            ]
            involved_views = np.unique(
                prepared.obs_view_index[
                    np.concatenate(
                        [prepared.rows_by_point[node] for node in component_nodes]
                    )
                ]
            )
            raise ValueError(
                "rigid component has no positive-weight row in an alpha-identifiable "
                f"view: rigid_component={component_idx}, graph_component={graph_idx}, "
                f"nodes={component_nodes.tolist()}, views={involved_views.tolist()}"
            )

        row_points = prepared.obs_point_index[rows]
        centered = points[row_points] - component_centroid[component_idx]
        point_blocks = np.zeros((rows.size, 3, 6), dtype=np.float64)
        point_blocks[:, :, :3] = np.eye(3, dtype=np.float64)[None]
        radius = float(component_radius[component_idx])
        if radius > _EPS:
            point_blocks[:, :, 3:] = -_batch_skew(centered) / radius
        projected_blocks = np.einsum(
            "rij,rjk->rik",
            prepared.obs_J[rows].astype(np.float64),
            point_blocks,
        )
        sqrt_weight = np.sqrt(prepared.obs_weights[rows])
        row_alpha = alpha.alphas[prepared.obs_view_index[rows]].astype(
            np.complex128
        )
        design = (
            sqrt_weight[:, None, None]
            * row_alpha[:, None, None]
            * projected_blocks.astype(np.complex128)
        ).reshape(-1, 6)
        target = (
            sqrt_weight[:, None]
            * prepared.obs_y[rows].astype(np.complex128)
        ).reshape(-1)
        solution, singular, rank = _minimum_norm_svd(
            design, target, rcond=config.rcond
        )
        component_singular[component_idx, : singular.size] = singular
        component_rank[component_idx] = rank
        if rank == 6:
            component_condition[component_idx] = float(
                singular[0] / max(float(singular[5]), _EPS)
            )
        component_translation[component_idx] = solution[:3]
        if radius > _EPS:
            component_rotation[component_idx] = solution[3:] / radius
        component_view_count[component_idx] = int(
            np.unique(prepared.obs_view_index[rows]).size
        )
        residual = design @ solution - target
        residual_norm = float(np.linalg.norm(residual))
        measurement_norm = float(np.linalg.norm(target))
        component_residual_norm[component_idx] = residual_norm
        component_measurement_norm[component_idx] = measurement_norm
        component_normalized_residual[component_idx] = residual_norm / max(
            measurement_norm, _EPS
        )

        component_nodes = ordered_rigid_indices[
            node_offsets[component_idx] : node_offsets[component_idx + 1]
        ]
        centered_nodes = points[component_nodes] - component_centroid[component_idx]
        phi[component_nodes] = (
            component_translation[component_idx]
            + np.cross(
                component_rotation[component_idx][None], centered_nodes
            )
        )

    # Validate the exact complex64 field that downstream motion fill and compact
    # latents consume, rather than only the higher-precision accumulator field.
    persisted_phi = phi.astype(np.complex64)
    global_edges = node_indices[edge_index]
    edge_vectors = points[global_edges[:, 1]] - points[global_edges[:, 0]]
    edge_delta_phi = (
        persisted_phi[global_edges[:, 1]].astype(np.complex128)
        - persisted_phi[global_edges[:, 0]].astype(np.complex128)
    )
    edge_axial = np.einsum("ij,ij->i", edge_vectors, edge_delta_phi)
    edge_squared_length = np.einsum("ij,ij->i", edge_vectors, edge_vectors)
    edge_relative_real = np.abs(edge_axial.real) / np.maximum(
        edge_squared_length, _EPS
    )
    edge_relative_imag = np.abs(edge_axial.imag) / np.maximum(
        edge_squared_length, _EPS
    )
    max_first_order_relative = float(
        max(
            np.max(edge_relative_real, initial=0.0),
            np.max(edge_relative_imag, initial=0.0),
        )
    )
    if max_first_order_relative > config.first_order_rtol:
        worst_real = int(np.argmax(edge_relative_real))
        worst_imag = int(np.argmax(edge_relative_imag))
        raise RuntimeError(
            "rigid component parameterization violated its first-order edge-length "
            f"constraint: max_relative={max_first_order_relative:.9g}, "
            f"rtol={config.first_order_rtol:.9g}, "
            f"worst_real_edge={worst_real}, worst_imag_edge={worst_imag}"
        )

    phase_angles = np.linspace(
        0.0, 2.0 * np.pi, config.phase_samples, endpoint=False, dtype=np.float64
    )
    edge_drift_p50, edge_drift_p90, edge_drift_max = (
        _finite_edge_drift_statistics(
            edge_vectors, edge_delta_phi, phase_angles
        )
    )
    component_drift_p50 = np.zeros((num_components,), dtype=np.float64)
    component_drift_p90 = np.zeros((num_components,), dtype=np.float64)
    component_drift_max = np.zeros((num_components,), dtype=np.float64)
    for component_idx in range(num_components):
        values = edge_drift_max[edge_component_index == component_idx]
        if values.size == 0:
            raise RuntimeError("a rigid component unexpectedly has no accepted edge")
        component_drift_p50[component_idx] = float(np.percentile(values, 50.0))
        component_drift_p90[component_idx] = float(np.percentile(values, 90.0))
        component_drift_max[component_idx] = float(np.max(values))

    return RigidComponentSolveResult(
        config=config,
        phi=persisted_phi,
        rigid_seed_mask=rigid_seed_mask,
        observed_mask=observed_mask,
        fill_target_mask=~rigid_seed_mask,
        point_component_index=point_component_index,
        component_graph_index=selected_graph_components.astype(np.int32),
        component_node_count=component_node_count,
        component_edge_count=component_edge_count,
        component_centroid=component_centroid.astype(np.float32),
        component_radius=component_radius.astype(np.float32),
        component_usable_observation_row_count=component_row_count,
        component_distinct_valid_view_count=component_view_count,
        component_singular_values=component_singular.astype(np.float32),
        component_rank=component_rank,
        component_condition=component_condition.astype(np.float32),
        component_weighted_residual_norm=component_residual_norm.astype(np.float32),
        component_weighted_measurement_norm=component_measurement_norm.astype(
            np.float32
        ),
        component_normalized_weighted_residual=component_normalized_residual.astype(
            np.float32
        ),
        component_translation=component_translation.astype(np.complex64),
        component_rotation=component_rotation.astype(np.complex64),
        edge_component_index=edge_component_index.astype(np.int32),
        edge_first_order_axial_real=edge_axial.real.astype(np.float32),
        edge_first_order_axial_imag=edge_axial.imag.astype(np.float32),
        edge_first_order_relative_real=edge_relative_real.astype(np.float32),
        edge_first_order_relative_imag=edge_relative_imag.astype(np.float32),
        phase_angles=phase_angles.astype(np.float32),
        edge_finite_drift_p50=edge_drift_p50.astype(np.float32),
        edge_finite_drift_p90=edge_drift_p90.astype(np.float32),
        edge_finite_drift_max=edge_drift_max.astype(np.float32),
        component_finite_drift_p50=component_drift_p50.astype(np.float32),
        component_finite_drift_p90=component_drift_p90.astype(np.float32),
        component_finite_drift_max=component_drift_max.astype(np.float32),
    )
