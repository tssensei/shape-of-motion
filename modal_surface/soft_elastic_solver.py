"""Soft elastic modal solve over one observed Gaussian structure graph.

The solver keeps one complex 3-D displacement per observed Gaussian.  Graph
edges impose soft first-order axial-stretch and graph-Laplacian penalties; a
connected component never shares one rigid transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modal_surface.motion_fill import SparseSolveMetadata
from modal_surface.observed_structure_graph import LoadedObservedStructureGraph
from modal_surface.optimization_staged import AlphaSyncResult, PreparedObservations


_EPS = 1.0e-12
_CONVERGED_LSMR_STOP_CODES = frozenset({0, 1, 2, 4, 5})


@dataclass(frozen=True)
class SoftElasticSolverConfig:
    stretch_relative: float = 1.0e-2
    laplacian_relative: float = 1.0e-4
    lsmr_atol: float = 1.0e-6
    lsmr_btol: float = 1.0e-6
    lsmr_conlim: float = 1.0e8
    lsmr_maxiter: int = 2000

    def validate(self) -> None:
        for name, value in (
            ("stretch_relative", self.stretch_relative),
            ("laplacian_relative", self.laplacian_relative),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"soft elastic {name} must be finite and non-negative")
        if self.stretch_relative == 0.0 and self.laplacian_relative == 0.0:
            raise ValueError(
                "soft elastic stretch_relative and laplacian_relative cannot both be zero"
            )
        for name, value in (
            ("lsmr_atol", self.lsmr_atol),
            ("lsmr_btol", self.lsmr_btol),
            ("lsmr_conlim", self.lsmr_conlim),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"soft elastic {name} must be finite and positive")
        if (
            isinstance(self.lsmr_maxiter, (bool, np.bool_))
            or not isinstance(self.lsmr_maxiter, (int, np.integer))
            or self.lsmr_maxiter <= 0
        ):
            raise ValueError("soft elastic lsmr_maxiter must be a positive integer")


@dataclass(frozen=True)
class SoftElasticSolveResult:
    config: SoftElasticSolverConfig
    phi: np.ndarray
    node_phi: np.ndarray
    node_gaussian_indices: np.ndarray
    valid_observation_mask: np.ndarray
    data_scale: float
    system_row_count: int
    system_column_count: int
    supported_column_count: int
    stretch_edge_count: int
    zero_length_edge_count: int
    laplacian_node_count: int
    real_solver: SparseSolveMetadata
    imaginary_solver: SparseSolveMetadata
    edge_axial_real: np.ndarray
    edge_axial_imaginary: np.ndarray
    edge_relative_axial_real: np.ndarray
    edge_relative_axial_imaginary: np.ndarray
    laplacian_real: np.ndarray
    laplacian_imaginary: np.ndarray


@dataclass(frozen=True)
class _SoftElasticSystem:
    operator: Any
    right_hand_side: np.ndarray
    inverse_column_scale: np.ndarray
    valid_observation_mask: np.ndarray
    data_scale: float
    node_gaussian_indices: np.ndarray
    edge_index: np.ndarray
    edge_distance: np.ndarray
    edge_direction: np.ndarray
    active_laplacian_nodes: np.ndarray
    directed_source: np.ndarray
    directed_target: np.ndarray
    directed_weight: np.ndarray
    stretch_edge_count: int
    zero_length_edge_count: int


def _require_scipy_sparse_linalg() -> tuple[Any, Any]:
    try:
        from scipy.sparse.linalg import LinearOperator, lsmr
    except ImportError as exc:
        raise ImportError(
            "Soft elastic solving requires scipy.sparse.linalg.LinearOperator and lsmr"
        ) from exc
    return LinearOperator, lsmr


def _validate_inputs(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    graph: LoadedObservedStructureGraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(prepared.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("soft elastic points must be a finite (N,3) array")
    if graph.num_foreground_gaussians != points.shape[0]:
        raise ValueError(
            "soft elastic graph foreground count does not match observations: "
            f"{graph.num_foreground_gaussians} != {points.shape[0]}"
        )
    node_indices = np.asarray(graph.node_gaussian_indices, dtype=np.int64)
    node_points = np.asarray(graph.node_points_world, dtype=np.float64)
    if node_indices.ndim != 1 or node_points.shape != (node_indices.size, 3):
        raise ValueError("soft elastic graph node arrays are malformed")
    if node_indices.size == 0:
        raise ValueError("soft elastic graph contains no observed Gaussian nodes")
    if not np.array_equal(points[node_indices].astype(np.float32), node_points.astype(np.float32)):
        maximum = float(np.max(np.abs(points[node_indices] - node_points)))
        raise ValueError(
            "soft elastic graph points do not match observation points; "
            f"maximum difference is {maximum:.6g}"
        )
    if tuple(str(value) for value in prepared.view_ids.tolist()) != graph.view_ids:
        raise ValueError("soft elastic graph view order does not match observations")
    if alpha.alphas.shape != (prepared.num_views,):
        raise ValueError("soft elastic alpha count does not match observation views")

    topology = graph.graph.topology
    edges = np.asarray(topology.edge_index, dtype=np.int64)
    edge_weights = np.asarray(topology.edge_combined_weight, dtype=np.float64)
    if edges.ndim != 2 or edges.shape[1:] != (2,):
        raise ValueError("soft elastic graph edge_index must have shape (E,2)")
    if edge_weights.shape != (edges.shape[0],):
        raise ValueError("soft elastic graph edge weights do not match edge_index")
    if np.any(edges < 0) or np.any(edges >= node_indices.size):
        raise ValueError("soft elastic graph edge_index is out of range")
    if np.any(~np.isfinite(edge_weights)) or np.any(edge_weights <= 0.0):
        raise ValueError("soft elastic graph edge weights must be finite and positive")
    return points, node_indices, edges, edge_weights


def _lsmr_metadata(solved: tuple[Any, ...]) -> SparseSolveMetadata:
    stop_code = int(solved[1])
    return SparseSolveMetadata(
        performed=True,
        converged=stop_code in _CONVERGED_LSMR_STOP_CODES,
        stop_code=stop_code,
        iterations=int(solved[2]),
        residual_norm=float(solved[3]),
        normal_residual_norm=float(solved[4]),
        matrix_norm=float(solved[5]),
        condition_estimate=float(solved[6]),
        solution_norm=float(solved[7]),
    )


def _build_system(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    graph: LoadedObservedStructureGraph,
    config: SoftElasticSolverConfig,
) -> _SoftElasticSystem:
    LinearOperator, _ = _require_scipy_sparse_linalg()
    points, node_indices, edges, edge_weights = _validate_inputs(
        prepared, alpha, graph
    )
    num_nodes = int(node_indices.size)
    num_columns = 3 * num_nodes
    global_to_local = np.full((points.shape[0],), -1, dtype=np.int64)
    global_to_local[node_indices] = np.arange(num_nodes, dtype=np.int64)
    observation_local = global_to_local[prepared.obs_point_index]
    valid_observations = (
        (observation_local >= 0)
        & alpha.identifiable_mask[prepared.obs_view_index]
        & (prepared.obs_weights > 0.0)
    )
    valid_rows = np.flatnonzero(valid_observations)
    if valid_rows.size == 0:
        raise ValueError("soft elastic solve has no positive-weight identifiable observations")
    observation_local = observation_local[valid_rows]
    observation_view = prepared.obs_view_index[valid_rows]
    observation_jacobian = prepared.obs_J[valid_rows].astype(np.float64)
    observation_weight = prepared.obs_weights[valid_rows].astype(np.float64)
    observation_alpha = alpha.alphas[observation_view].astype(np.complex128)
    alpha_magnitude = np.abs(observation_alpha)
    if np.any(~np.isfinite(alpha_magnitude)) or np.any(alpha_magnitude <= _EPS):
        raise ValueError("soft elastic identifiable alpha values must be finite and nonzero")

    point_information = np.zeros((num_nodes,), dtype=np.float64)
    row_information = (
        observation_weight
        * alpha_magnitude**2
        * np.sum(observation_jacobian**2, axis=(1, 2))
        / 3.0
    )
    np.add.at(point_information, observation_local, row_information)
    positive_information = point_information[point_information > _EPS]
    if positive_information.size == 0:
        raise ValueError("soft elastic observation normal matrix has zero scale")
    data_scale = float(np.median(positive_information))
    if not np.isfinite(data_scale) or data_scale <= _EPS:
        raise ValueError("soft elastic data scale must be finite and positive")
    sqrt_data_scale = np.sqrt(data_scale)
    data_factor = (
        np.sqrt(observation_weight) * alpha_magnitude / sqrt_data_scale
    )
    # Rotate each complex equation so the matrix remains real; solving its real
    # and imaginary right-hand sides separately is exactly equivalent to
    # minimizing ||alpha_v J_v phi - y_v|| for complex phi.
    target_phase = np.conj(observation_alpha) / alpha_magnitude
    data_target = (
        np.sqrt(observation_weight)[:, None]
        * target_phase[:, None]
        * prepared.obs_y[valid_rows].astype(np.complex128)
        / sqrt_data_scale
    )

    node_points = points[node_indices]
    edge_delta = node_points[edges[:, 1]] - node_points[edges[:, 0]]
    edge_distance = np.linalg.norm(edge_delta, axis=1)
    nonzero_edge = edge_distance > _EPS
    stretch_edges = edges[nonzero_edge]
    stretch_distances = edge_distance[nonzero_edge]
    stretch_directions = edge_delta[nonzero_edge] / stretch_distances[:, None]

    node_weight_sum = np.zeros((num_nodes,), dtype=np.float64)
    if edges.size:
        np.add.at(node_weight_sum, edges[:, 0], edge_weights)
        np.add.at(node_weight_sum, edges[:, 1], edge_weights)
    stretch_weight = edge_weights[nonzero_edge] / np.sqrt(
        node_weight_sum[stretch_edges[:, 0]]
        * node_weight_sum[stretch_edges[:, 1]]
    )
    stretch_factor = np.sqrt(config.stretch_relative * stretch_weight)

    directed_source = np.concatenate([edges[:, 0], edges[:, 1]])
    directed_target = np.concatenate([edges[:, 1], edges[:, 0]])
    directed_raw_weight = np.concatenate([edge_weights, edge_weights])
    directed_weight = directed_raw_weight / node_weight_sum[directed_source]
    active_laplacian_nodes = np.flatnonzero(node_weight_sum > 0.0)
    laplacian_factor = np.sqrt(config.laplacian_relative)

    num_data_rows = 2 * int(valid_rows.size)
    num_stretch_rows = int(stretch_edges.shape[0]) if config.stretch_relative > 0.0 else 0
    num_laplacian_rows = (
        3 * int(active_laplacian_nodes.size)
        if config.laplacian_relative > 0.0
        else 0
    )
    num_rows = num_data_rows + num_stretch_rows + num_laplacian_rows

    def unscaled_matvec(flat_values: np.ndarray) -> np.ndarray:
        values = np.asarray(flat_values, dtype=np.float64).reshape(num_nodes, 3)
        data = np.einsum(
            "oij,oj->oi",
            observation_jacobian,
            values[observation_local],
        )
        output = [data_factor[:, None] * data]
        if num_stretch_rows:
            difference = values[stretch_edges[:, 1]] - values[stretch_edges[:, 0]]
            output.append(
                stretch_factor
                * np.einsum("ei,ei->e", stretch_directions, difference)
            )
        if num_laplacian_rows:
            neighbor_average = np.zeros_like(values)
            np.add.at(
                neighbor_average,
                directed_source,
                directed_weight[:, None] * values[directed_target],
            )
            output.append(
                laplacian_factor
                * (values[active_laplacian_nodes] - neighbor_average[active_laplacian_nodes])
            )
        return np.concatenate([np.asarray(value).reshape(-1) for value in output])

    def unscaled_rmatvec(flat_rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(flat_rows, dtype=np.float64)
        gradient = np.zeros((num_nodes, 3), dtype=np.float64)
        offset = 0
        data_rows = rows[offset : offset + num_data_rows].reshape(-1, 2)
        offset += num_data_rows
        data_gradient = np.einsum(
            "oij,oi->oj",
            observation_jacobian,
            data_factor[:, None] * data_rows,
        )
        np.add.at(gradient, observation_local, data_gradient)
        if num_stretch_rows:
            values = rows[offset : offset + num_stretch_rows]
            offset += num_stretch_rows
            edge_gradient = (
                stretch_factor * values
            )[:, None] * stretch_directions
            np.add.at(gradient, stretch_edges[:, 0], -edge_gradient)
            np.add.at(gradient, stretch_edges[:, 1], edge_gradient)
        if num_laplacian_rows:
            values = rows[offset:].reshape(-1, 3)
            node_values = np.zeros_like(gradient)
            node_values[active_laplacian_nodes] = laplacian_factor * values
            gradient += node_values
            np.add.at(
                gradient,
                directed_target,
                -directed_weight[:, None] * node_values[directed_source],
            )
        return gradient.reshape(-1)

    column_diagonal = np.zeros((num_nodes, 3), dtype=np.float64)
    data_diagonal = (
        data_factor[:, None, None] ** 2 * observation_jacobian**2
    ).sum(axis=1)
    np.add.at(column_diagonal, observation_local, data_diagonal)
    if num_stretch_rows:
        stretch_diagonal = (
            stretch_factor[:, None] ** 2 * stretch_directions**2
        )
        np.add.at(column_diagonal, stretch_edges[:, 0], stretch_diagonal)
        np.add.at(column_diagonal, stretch_edges[:, 1], stretch_diagonal)
    if num_laplacian_rows:
        column_diagonal[active_laplacian_nodes] += config.laplacian_relative
        neighbor_diagonal = np.repeat(
            (config.laplacian_relative * directed_weight**2)[:, None],
            3,
            axis=1,
        )
        np.add.at(
            column_diagonal,
            directed_target,
            neighbor_diagonal,
        )
    flat_diagonal = column_diagonal.reshape(-1)
    inverse_column_scale = np.zeros_like(flat_diagonal)
    supported = flat_diagonal > _EPS
    inverse_column_scale[supported] = 1.0 / np.sqrt(flat_diagonal[supported])

    def matvec(scaled_values: np.ndarray) -> np.ndarray:
        return unscaled_matvec(inverse_column_scale * scaled_values)

    def rmatvec(row_values: np.ndarray) -> np.ndarray:
        return inverse_column_scale * unscaled_rmatvec(row_values)

    operator = LinearOperator(
        shape=(num_rows, num_columns),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )
    right_hand_side = np.zeros((num_rows,), dtype=np.complex128)
    right_hand_side[:num_data_rows] = data_target.reshape(-1)
    return _SoftElasticSystem(
        operator=operator,
        right_hand_side=right_hand_side,
        inverse_column_scale=inverse_column_scale,
        valid_observation_mask=valid_observations,
        data_scale=data_scale,
        node_gaussian_indices=node_indices,
        edge_index=stretch_edges,
        edge_distance=stretch_distances,
        edge_direction=stretch_directions,
        active_laplacian_nodes=active_laplacian_nodes,
        directed_source=directed_source,
        directed_target=directed_target,
        directed_weight=directed_weight,
        stretch_edge_count=num_stretch_rows,
        zero_length_edge_count=int(np.count_nonzero(~nonzero_edge)),
    )


def _laplacian_values(system: _SoftElasticSystem, node_values: np.ndarray) -> np.ndarray:
    values = np.asarray(node_values, dtype=np.float64)
    output = np.zeros_like(values)
    if system.active_laplacian_nodes.size == 0:
        return output
    neighbor_average = np.zeros_like(values)
    np.add.at(
        neighbor_average,
        system.directed_source,
        system.directed_weight[:, None] * values[system.directed_target],
    )
    output[system.active_laplacian_nodes] = (
        values[system.active_laplacian_nodes]
        - neighbor_average[system.active_laplacian_nodes]
    )
    return output


def solve_soft_elastic(
    prepared: PreparedObservations,
    alpha: AlphaSyncResult,
    graph: LoadedObservedStructureGraph,
    config: SoftElasticSolverConfig | None = None,
) -> SoftElasticSolveResult:
    """Solve one complex modal displacement field with soft graph regularization."""

    config = config or SoftElasticSolverConfig()
    config.validate()
    _, lsmr = _require_scipy_sparse_linalg()
    system = _build_system(prepared, alpha, graph, config)

    solved_real = lsmr(
        system.operator,
        system.right_hand_side.real,
        atol=config.lsmr_atol,
        btol=config.lsmr_btol,
        conlim=config.lsmr_conlim,
        maxiter=config.lsmr_maxiter,
    )
    solved_imaginary = lsmr(
        system.operator,
        system.right_hand_side.imag,
        atol=config.lsmr_atol,
        btol=config.lsmr_btol,
        conlim=config.lsmr_conlim,
        maxiter=config.lsmr_maxiter,
    )
    real_metadata = _lsmr_metadata(solved_real)
    imaginary_metadata = _lsmr_metadata(solved_imaginary)
    if not real_metadata.converged or not imaginary_metadata.converged:
        raise RuntimeError(
            "Soft elastic LSMR did not converge: "
            f"real stop={real_metadata.stop_code}, "
            f"imaginary stop={imaginary_metadata.stop_code}"
        )

    node_real = system.inverse_column_scale * np.asarray(solved_real[0])
    node_imaginary = system.inverse_column_scale * np.asarray(solved_imaginary[0])
    node_phi = (
        node_real.reshape(-1, 3) + 1j * node_imaginary.reshape(-1, 3)
    )
    if not np.all(np.isfinite(node_phi.real)) or not np.all(np.isfinite(node_phi.imag)):
        raise RuntimeError("Soft elastic solve produced non-finite modal displacements")
    phi = np.zeros((prepared.points.shape[0], 3), dtype=np.complex64)
    phi[system.node_gaussian_indices] = node_phi.astype(np.complex64)

    edge_difference = node_phi[system.edge_index[:, 1]] - node_phi[
        system.edge_index[:, 0]
    ]
    edge_axial = np.einsum("ei,ei->e", system.edge_direction, edge_difference)
    edge_relative = edge_axial / system.edge_distance
    laplacian_real = _laplacian_values(system, node_phi.real)
    laplacian_imaginary = _laplacian_values(system, node_phi.imag)
    return SoftElasticSolveResult(
        config=config,
        phi=phi,
        node_phi=node_phi.astype(np.complex64),
        node_gaussian_indices=system.node_gaussian_indices.astype(np.int32),
        valid_observation_mask=system.valid_observation_mask,
        data_scale=system.data_scale,
        system_row_count=int(system.operator.shape[0]),
        system_column_count=int(system.operator.shape[1]),
        supported_column_count=int(
            np.count_nonzero(system.inverse_column_scale > 0.0)
        ),
        stretch_edge_count=system.stretch_edge_count,
        zero_length_edge_count=system.zero_length_edge_count,
        laplacian_node_count=int(system.active_laplacian_nodes.size),
        real_solver=real_metadata,
        imaginary_solver=imaginary_metadata,
        edge_axial_real=edge_axial.real.astype(np.float32),
        edge_axial_imaginary=edge_axial.imag.astype(np.float32),
        edge_relative_axial_real=edge_relative.real.astype(np.float32),
        edge_relative_axial_imaginary=edge_relative.imag.astype(np.float32),
        laplacian_real=laplacian_real.astype(np.float32),
        laplacian_imaginary=laplacian_imaginary.astype(np.float32),
    )
