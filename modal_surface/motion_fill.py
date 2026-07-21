"""Spatial nullspace completion for staged modal-field solutions.

This module operates on array-level staged-solver outputs. It deliberately does
not read or write latent files so experiments can preserve their source
artifacts and choose their own output contracts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np


_CONVERGED_LSMR_STOP_CODES = frozenset({0, 1, 2, 4, 5})


def _record_timing(
    timings: dict[str, float] | None,
    name: str,
    started: float,
) -> None:
    if timings is not None:
        timings[name] = float(perf_counter() - started)


def _emit_progress(
    progress: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress is not None:
        progress(message)


@dataclass(frozen=True)
class KnnCandidateSet:
    """Reusable ordered nearest-neighbor candidates up to ``max_k``."""

    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray
    num_points: int
    max_k: int


@dataclass(frozen=True)
class KnnGraph:
    """A distance-pruned, union-symmetrized undirected KNN graph."""

    edge_index: np.ndarray
    edge_distance: np.ndarray
    edge_weight: np.ndarray
    degree: np.ndarray
    component_index: np.ndarray
    component_sizes: np.ndarray
    isolated_mask: np.ndarray
    num_points: int
    k: int
    max_distance: float
    epsilon: float
    candidate_directed_count: int
    retained_directed_count: int
    pruned_directed_count: int

    @property
    def unique_undirected_edge_count(self) -> int:
        return int(self.edge_index.shape[0])


@dataclass(frozen=True)
class AnchorConnectivity:
    """Track-specific graph connectivity relative to fixed anchor points."""

    connected_to_anchor: np.ndarray
    active_mask: np.ndarray
    component_index: np.ndarray
    component_sizes: np.ndarray
    component_has_anchor: np.ndarray
    component_anchor_count: np.ndarray
    hop_distance: np.ndarray


@dataclass(frozen=True)
class SharedGroupMembership:
    """One-time ordered membership and star edges for shared affine groups."""

    ordered_points: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray
    leaders: np.ndarray
    star_edges: np.ndarray


@dataclass(frozen=True)
class SparseSolveMetadata:
    """Convergence information returned by one real-valued LSMR solve."""

    performed: bool
    converged: bool
    stop_code: int
    iterations: int
    residual_norm: float
    normal_residual_norm: float
    matrix_norm: float
    condition_estimate: float
    solution_norm: float


@dataclass(frozen=True)
class ValidatedMotionFillInputs:
    """Validated float64/complex128 arrays used by the sparse solve."""

    phi_observable: np.ndarray
    nullspace_basis: np.ndarray
    point_nullity: np.ndarray
    anchor_mask: np.ndarray
    partial_mask: np.ndarray
    unobserved_mask: np.ndarray
    excluded_mask: np.ndarray
    output_dtype: np.dtype


@dataclass(frozen=True)
class MotionFillResult:
    """Completed field and the graph/solver state needed for diagnostics."""

    phi: np.ndarray
    phi_observable: np.ndarray
    phi_nullspace_correction: np.ndarray
    completion_mask: np.ndarray
    completion_connected_to_anchor: np.ndarray
    coefficient_values: np.ndarray
    coefficient_offsets: np.ndarray
    connectivity: AnchorConnectivity
    real_solver: SparseSolveMetadata
    imag_solver: SparseSolveMetadata
    system_row_count: int
    system_column_count: int
    active_edge_count: int
    point_coefficient_group_index: np.ndarray | None = None
    coefficient_group_dimensions: np.ndarray | None = None


def _require_scipy_kdtree():
    try:
        from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]
    except ImportError as exc:
        raise ImportError("Motion-fill KNN construction requires scipy.spatial.cKDTree.") from exc
    return cKDTree


def _require_scipy_sparse():
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import lsmr
    except ImportError as exc:
        raise ImportError("Motion-fill completion requires scipy.sparse and scipy.sparse.linalg.lsmr.") from exc
    return coo_matrix, lsmr


def _require_positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _validate_points(points_world: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points.shape}.")
    if points.shape[0] < 2:
        raise ValueError("Motion-fill KNN construction requires at least two points.")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_world must contain only finite values.")
    return points


def query_knn_candidates(
    points_world: np.ndarray,
    max_k: int,
    tree: Any | None = None,
) -> KnnCandidateSet:
    """Query ordered neighbors, reusing a supplied spatial tree when available."""

    points = _validate_points(points_world)
    max_k = _require_positive_integer(max_k, "max_k")
    num_points = int(points.shape[0])
    if max_k >= num_points:
        raise ValueError(f"max_k must be smaller than the point count ({num_points}), got {max_k}.")

    if tree is None:
        cKDTree = _require_scipy_kdtree()
        tree = cKDTree(points)
    distances, _ = tree.query(points, k=max_k + 1)
    distances = np.asarray(distances, dtype=np.float64)
    boundaries = np.nextafter(distances[:, -1], np.inf)
    candidate_rows = tree.query_ball_point(
        points,
        boundaries,
        return_sorted=False,
    )
    neighbors = np.empty((num_points, max_k), dtype=np.int64)
    neighbor_distances = np.empty((num_points, max_k), dtype=np.float64)

    for point_index, row in enumerate(candidate_rows):
        row_indices = np.asarray(row, dtype=np.int64)
        row_indices = row_indices[row_indices != point_index]
        row_distances = np.linalg.norm(
            points[row_indices] - points[point_index], axis=1
        )
        if row_indices.size < max_k:
            raise RuntimeError(
                f"KNN query returned only {row_indices.size} non-self candidates for "
                f"point {point_index}; expected at least {max_k}."
            )
        order = np.lexsort((row_indices, row_distances))
        selected = order[:max_k]
        neighbors[point_index] = row_indices[selected]
        neighbor_distances[point_index] = row_distances[selected]

    if not np.all(np.isfinite(neighbor_distances)):
        raise ValueError("KNN query produced non-finite neighbor distances.")
    if np.any(neighbor_distances < 0.0):
        raise ValueError("KNN query produced a negative neighbor distance.")
    return KnnCandidateSet(
        neighbor_indices=neighbors,
        neighbor_distances=neighbor_distances,
        num_points=num_points,
        max_k=max_k,
    )

# tianyi's method fit in?
def _stable_component_labels(num_points: int, edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parent = np.arange(num_points, dtype=np.int64)

    def find(point: int) -> int:
        root = point
        while parent[root] != root:
            root = int(parent[root])
        while parent[point] != point:
            next_point = int(parent[point])
            parent[point] = root
            point = next_point
        return root

    for point_i, point_j in edge_index.tolist():
        root_i = find(int(point_i))
        root_j = find(int(point_j))
        if root_i == root_j:
            continue
        if root_i < root_j:
            parent[root_j] = root_i
        else:
            parent[root_i] = root_j

    roots = np.asarray([find(point) for point in range(num_points)], dtype=np.int64)
    unique_roots = np.unique(roots)
    root_to_component = np.full((num_points,), -1, dtype=np.int64)
    root_to_component[unique_roots] = np.arange(unique_roots.size, dtype=np.int64)
    component_index = root_to_component[roots].astype(np.int32)
    component_sizes = np.bincount(component_index, minlength=unique_roots.size).astype(np.int32)
    return component_index, component_sizes


def _active_component_labels(
    num_points: int,
    edge_index: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    active_points = np.where(active_mask)[0]
    component_index = np.full((num_points,), -1, dtype=np.int32)
    if active_points.size == 0:
        return component_index, np.empty((0,), dtype=np.int32)

    point_to_active = np.full((num_points,), -1, dtype=np.int64)
    point_to_active[active_points] = np.arange(active_points.size, dtype=np.int64)
    active_edge_mask = active_mask[edge_index[:, 0]] & active_mask[edge_index[:, 1]]
    active_edges = point_to_active[edge_index[active_edge_mask]]
    active_component_index, component_sizes = _stable_component_labels(
        int(active_points.size),
        active_edges,
    )
    component_index[active_points] = active_component_index
    return component_index, component_sizes


def build_knn_graph(
    candidates: KnnCandidateSet,
    k: int,
    max_distance: float,
    epsilon: float = 1e-8,
) -> KnnGraph:
    """Build one fixed-cutoff undirected graph from a candidate prefix."""

    # safety check
    if not isinstance(candidates, KnnCandidateSet):
        raise TypeError("candidates must be a KnnCandidateSet returned by query_knn_candidates().")
    k = _require_positive_integer(k, "k")
    if k > candidates.max_k:
        raise ValueError(f"k={k} exceeds the queried max_k={candidates.max_k}.")
    max_distance = float(max_distance)
    epsilon = float(epsilon)
    if not np.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("max_distance must be finite and positive.")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive.")
    num_points = int(candidates.num_points)
    expected_shape = (num_points, int(candidates.max_k))
    if candidates.neighbor_indices.shape != expected_shape:
        raise ValueError(
            f"neighbor_indices must have shape {expected_shape}, got {candidates.neighbor_indices.shape}."
        )
    if candidates.neighbor_distances.shape != expected_shape:
        raise ValueError(
            "neighbor_distances must match neighbor_indices shape, got "
            f"{candidates.neighbor_distances.shape}."
        )

    neighbor_indices = np.asarray(candidates.neighbor_indices[:, :k], dtype=np.int64)
    neighbor_distances = np.asarray(candidates.neighbor_distances[:, :k], dtype=np.float64)
    if np.any(neighbor_indices < 0) or np.any(neighbor_indices >= num_points):
        raise ValueError("KNN candidates contain an out-of-range point index.")
    if np.any(~np.isfinite(neighbor_distances)) or np.any(neighbor_distances < 0.0):
        raise ValueError("KNN candidates must contain finite, non-negative distances.")

    source = np.repeat(np.arange(num_points, dtype=np.int64), k)
    target = neighbor_indices.reshape(-1)
    distance = neighbor_distances.reshape(-1)
    candidate_directed_count = int(source.size)
    retained = distance <= max_distance
    retained_directed_count = int(np.count_nonzero(retained))
    pruned_directed_count = candidate_directed_count - retained_directed_count
    source = source[retained]
    target = target[retained]
    distance = distance[retained]

    if source.size:
        pairs = np.column_stack((np.minimum(source, target), np.maximum(source, target)))
        if np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError("KNN candidates contain a self edge.")
        edge_index, first_indices = np.unique(
            pairs,
            axis=0,
            return_index=True,
        )
        edge_distance = distance[first_indices]
    else:
        edge_index = np.empty((0, 2), dtype=np.int64)
        edge_distance = np.empty((0,), dtype=np.float64)
    edge_weight = 1.0 / (edge_distance + epsilon)

    degree = np.zeros((num_points,), dtype=np.int32)
    if edge_index.size:
        np.add.at(degree, edge_index[:, 0], 1)
        np.add.at(degree, edge_index[:, 1], 1)
    
    # component for gaussian cluster driven by some anchor
    component_index, component_sizes = _stable_component_labels(num_points, edge_index)
    return KnnGraph(
        edge_index=edge_index.astype(np.int64, copy=False),
        edge_distance=edge_distance,
        edge_weight=edge_weight,
        degree=degree,
        component_index=component_index,
        component_sizes=component_sizes,
        isolated_mask=degree == 0,
        num_points=num_points,
        k=k,
        max_distance=max_distance,
        epsilon=epsilon,
        candidate_directed_count=candidate_directed_count,
        retained_directed_count=retained_directed_count,
        pruned_directed_count=pruned_directed_count,
    )


def _validate_boolean_mask(mask: np.ndarray, name: str, num_points: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != (num_points,):
        raise ValueError(f"{name} must have shape ({num_points},), got {array.shape}.")
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must have boolean dtype, got {array.dtype}.")
    return array


def _validate_graph(graph: KnnGraph) -> None:
    if not isinstance(graph, KnnGraph):
        raise TypeError("graph must be a KnnGraph returned by build_knn_graph().")
    num_points = int(graph.num_points)
    num_edges = int(graph.edge_index.shape[0])
    if num_points < 2:
        raise ValueError("graph must contain at least two points.")
    if graph.edge_index.shape != (num_edges, 2):
        raise ValueError(f"graph.edge_index must have shape (E,2), got {graph.edge_index.shape}.")
    if np.any(graph.edge_index < 0) or np.any(graph.edge_index >= num_points):
        raise ValueError("graph.edge_index contains an out-of-range point index.")
    if np.any(graph.edge_index[:, 0] >= graph.edge_index[:, 1]):
        raise ValueError("graph.edge_index must contain unique ordered undirected edges with i < j.")
    if graph.edge_distance.shape != (num_edges,) or graph.edge_weight.shape != (num_edges,):
        raise ValueError("graph edge distance/weight arrays must match graph.edge_index length.")
    if np.any(~np.isfinite(graph.edge_distance)) or np.any(graph.edge_distance < 0.0):
        raise ValueError("graph.edge_distance must contain finite, non-negative values.")
    if np.any(~np.isfinite(graph.edge_weight)) or np.any(graph.edge_weight <= 0.0):
        raise ValueError("graph.edge_weight must contain finite, positive values.")
    if graph.degree.shape != (num_points,):
        raise ValueError("graph.degree must match graph.num_points.")
    if graph.component_index.shape != (num_points,):
        raise ValueError("graph.component_index must match graph.num_points.")


def compute_anchor_connectivity(
    graph: KnnGraph,
    anchor_mask: np.ndarray,
    excluded_mask: np.ndarray | None = None,
) -> AnchorConnectivity:
    """Find anchor connectivity after removing excluded vertices and incident edges."""

    _validate_graph(graph)
    anchor = _validate_boolean_mask(anchor_mask, "anchor_mask", graph.num_points)
    if excluded_mask is None:
        excluded = np.zeros((graph.num_points,), dtype=bool)
    else:
        excluded = _validate_boolean_mask(excluded_mask, "excluded_mask", graph.num_points)
    if np.any(anchor & excluded):
        raise ValueError("anchor_mask and excluded_mask must be mutually exclusive.")
    active_mask = ~excluded
    component_index, component_sizes = _active_component_labels(
        graph.num_points,
        graph.edge_index,
        active_mask,
    )
    num_components = int(component_sizes.shape[0])
    component_anchor_count = np.bincount(
        component_index[active_mask],
        weights=anchor[active_mask].astype(np.int32),
        minlength=num_components,
    ).astype(np.int32)
    component_has_anchor = component_anchor_count > 0
    connected_to_anchor = np.zeros((graph.num_points,), dtype=bool)
    connected_to_anchor[active_mask] = component_has_anchor[component_index[active_mask]]

    adjacency: list[list[int]] = [[] for _ in range(graph.num_points)]
    for point_i, point_j in graph.edge_index.tolist():
        if excluded[int(point_i)] or excluded[int(point_j)]:
            continue
        adjacency[int(point_i)].append(int(point_j))
        adjacency[int(point_j)].append(int(point_i))
    hop_distance = np.full((graph.num_points,), -1, dtype=np.int32)
    queue: deque[int] = deque()
    for point in np.where(anchor)[0].tolist():
        hop_distance[point] = 0
        queue.append(int(point))
    while queue:
        point = queue.popleft()
        next_hop = int(hop_distance[point]) + 1
        for neighbor in adjacency[point]:
            if hop_distance[neighbor] >= 0:
                continue
            hop_distance[neighbor] = next_hop
            queue.append(neighbor)
    if not np.array_equal(hop_distance >= 0, connected_to_anchor):
        raise RuntimeError("Anchor hop-distance traversal disagrees with graph component labels.")
    return AnchorConnectivity(
        connected_to_anchor=connected_to_anchor,
        active_mask=active_mask,
        component_index=component_index,
        component_sizes=component_sizes,
        component_has_anchor=component_has_anchor,
        component_anchor_count=component_anchor_count,
        hop_distance=hop_distance,
    )


def build_shared_group_membership(
    shared_group_index: np.ndarray,
    num_points: int,
) -> SharedGroupMembership:
    """Build ordered group membership once without repeated full point scans."""

    groups = np.asarray(shared_group_index)
    if (
        groups.shape != (num_points,)
        or not np.issubdtype(groups.dtype, np.integer)
        or np.any(groups < -1)
    ):
        raise ValueError(
            "shared_group_index must be an integer point array with values >= -1"
        )
    group_count = int(groups.max(initial=-1)) + 1
    present_groups = np.unique(groups[groups >= 0])
    if not np.array_equal(
        present_groups,
        np.arange(group_count, dtype=present_groups.dtype),
    ):
        raise ValueError("shared_group_index must use contiguous group indices")
    member_points = np.flatnonzero(groups >= 0)
    order = np.argsort(groups[member_points], kind="stable")
    ordered_points = member_points[order].astype(np.int64, copy=False)
    counts = np.bincount(
        groups[ordered_points],
        minlength=group_count,
    ).astype(np.int64, copy=False)
    if np.any(counts < 2):
        invalid_group = int(np.flatnonzero(counts < 2)[0])
        raise ValueError(
            f"shared affine group {invalid_group} must contain at least two points"
        )
    offsets = np.zeros((group_count + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    leaders = ordered_points[offsets[:-1]]
    nonleader_mask = np.ones((ordered_points.size,), dtype=bool)
    nonleader_mask[offsets[:-1]] = False
    star_edges = np.column_stack(
        [
            np.repeat(leaders, counts - 1),
            ordered_points[nonleader_mask],
        ]
    ).astype(np.int64, copy=False)
    return SharedGroupMembership(
        ordered_points=ordered_points,
        counts=counts,
        offsets=offsets,
        leaders=leaders,
        star_edges=star_edges,
    )


def _compute_grouped_anchor_connectivity(
    graph: KnnGraph,
    anchor_mask: np.ndarray,
    membership: SharedGroupMembership,
) -> AnchorConnectivity:
    """Connect every shared affine group before testing anchor reachability."""

    anchor = _validate_boolean_mask(anchor_mask, "anchor_mask", graph.num_points)
    if np.any(anchor[membership.ordered_points]):
        raise ValueError("anchor points cannot belong to a shared affine group")
    augmented_edges = (
        np.concatenate([graph.edge_index, membership.star_edges], axis=0)
        if membership.star_edges.size
        else graph.edge_index
    )
    component_index, component_sizes = _stable_component_labels(
        graph.num_points,
        augmented_edges,
    )
    component_anchor_count = np.bincount(
        component_index,
        weights=anchor.astype(np.int32),
        minlength=component_sizes.shape[0],
    ).astype(np.int32)
    component_has_anchor = component_anchor_count > 0
    connected_to_anchor = component_has_anchor[component_index]

    adjacency: list[list[int]] = [[] for _ in range(graph.num_points)]
    for point_i, point_j in augmented_edges.tolist():
        adjacency[int(point_i)].append(int(point_j))
        adjacency[int(point_j)].append(int(point_i))
    hop_distance = np.full((graph.num_points,), -1, dtype=np.int32)
    queue: deque[int] = deque()
    for point in np.flatnonzero(anchor).tolist():
        hop_distance[point] = 0
        queue.append(int(point))
    while queue:
        point = queue.popleft()
        next_hop = int(hop_distance[point]) + 1
        for neighbor in adjacency[point]:
            if hop_distance[neighbor] >= 0:
                continue
            hop_distance[neighbor] = next_hop
            queue.append(neighbor)
    if not np.array_equal(hop_distance >= 0, connected_to_anchor):
        raise RuntimeError(
            "Grouped anchor hop-distance traversal disagrees with connectivity"
        )
    return AnchorConnectivity(
        connected_to_anchor=connected_to_anchor,
        active_mask=np.ones((graph.num_points,), dtype=bool),
        component_index=component_index,
        component_sizes=component_sizes,
        component_has_anchor=component_has_anchor,
        component_anchor_count=component_anchor_count,
        hop_distance=hop_distance,
    )


def validate_motion_fill_inputs(
    phi_observable: np.ndarray,
    point_nullspace_basis: np.ndarray,
    point_nullity: np.ndarray,
    anchor_mask: np.ndarray,
    partial_mask: np.ndarray,
    unobserved_mask: np.ndarray,
    *,
    excluded_mask: np.ndarray | None = None,
    basis_real_atol: float = 1e-7,
    basis_orthonormal_atol: float = 1e-5,
    observable_orthogonality_atol: float = 1e-5,
) -> ValidatedMotionFillInputs:
    """Validate the staged observable/nullspace decomposition before filling."""

    phi_source = np.asarray(phi_observable)
    if phi_source.ndim != 2 or phi_source.shape[1] != 3:
        raise ValueError(f"phi_observable must have shape (N,3), got {phi_source.shape}.")
    if not np.issubdtype(phi_source.dtype, np.number):
        raise ValueError(f"phi_observable must be numeric, got {phi_source.dtype}.")
    num_points = int(phi_source.shape[0])
    phi = phi_source.astype(np.complex128)
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi_observable must contain only finite values.")

    basis_source = np.asarray(point_nullspace_basis)
    if basis_source.shape != (num_points, 3, 3):
        raise ValueError(
            f"point_nullspace_basis must have shape ({num_points},3,3), got {basis_source.shape}."
        )
    if not np.issubdtype(basis_source.dtype, np.number):
        raise ValueError(f"point_nullspace_basis must be numeric, got {basis_source.dtype}.")
    basis_complex = basis_source.astype(np.complex128)
    if not np.all(np.isfinite(basis_complex)):
        raise ValueError("point_nullspace_basis must contain only finite values.")

    for tolerance, name in (
        (basis_real_atol, "basis_real_atol"),
        (basis_orthonormal_atol, "basis_orthonormal_atol"),
        (observable_orthogonality_atol, "observable_orthogonality_atol"),
    ):
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    max_imaginary = float(np.max(np.abs(basis_complex.imag), initial=0.0))
    if max_imaginary > basis_real_atol:
        raise ValueError(
            "point_nullspace_basis must be effectively real for independent real/imaginary solves; "
            f"maximum imaginary magnitude is {max_imaginary:.6g}."
        )
    basis = basis_complex.real.astype(np.float64, copy=False)

    nullity_source = np.asarray(point_nullity)
    if nullity_source.shape != (num_points,):
        raise ValueError(f"point_nullity must have shape ({num_points},), got {nullity_source.shape}.")
    if not np.issubdtype(nullity_source.dtype, np.integer):
        raise ValueError(f"point_nullity must have integer dtype, got {nullity_source.dtype}.")
    if np.any(nullity_source < 0) or np.any(nullity_source > 3):
        raise ValueError("point_nullity values must lie in [0,3].")
    nullity = nullity_source.astype(np.int8)

    anchor = _validate_boolean_mask(anchor_mask, "anchor_mask", num_points)
    partial = _validate_boolean_mask(partial_mask, "partial_mask", num_points)
    unobserved = _validate_boolean_mask(unobserved_mask, "unobserved_mask", num_points)
    if excluded_mask is None:
        excluded = np.zeros((num_points,), dtype=bool)
    else:
        excluded = _validate_boolean_mask(excluded_mask, "excluded_mask", num_points)
    partition_count = (
        anchor.astype(np.int8)
        + partial.astype(np.int8)
        + unobserved.astype(np.int8)
        + excluded.astype(np.int8)
    )
    if np.any(partition_count != 1):
        raise ValueError(
            "anchor_mask, partial_mask, unobserved_mask, and excluded_mask must be "
            "mutually exclusive and exhaustive."
        )
    if np.any(nullity[anchor] != 0):
        raise ValueError("Every anchor point must have nullity zero.")
    if np.any(nullity[partial] <= 0):
        raise ValueError("Every partial point must have positive nullity.")
    if np.any(nullity[unobserved] != 3):
        raise ValueError("Every unobserved point must have nullity three.")
    if np.any(np.abs(phi[unobserved]) > observable_orthogonality_atol):
        raise ValueError("Every unobserved point must have zero observable motion.")

    identity = np.eye(3, dtype=np.float64)
    for point in range(num_points):
        if excluded[point]:
            continue
        dimension = int(nullity[point])
        if dimension == 0:
            continue
        active_basis = basis[point, :, :dimension]
        gram = active_basis.T @ active_basis
        if not np.allclose(gram, np.eye(dimension), atol=basis_orthonormal_atol, rtol=0.0):
            raise ValueError(f"Active nullspace basis is not orthonormal at point {point}.")
        projection = active_basis.T @ phi[point]
        scale = max(1.0, float(np.linalg.norm(phi[point])))
        if float(np.linalg.norm(projection)) > observable_orthogonality_atol * scale:
            raise ValueError(f"phi_observable is not orthogonal to the nullspace at point {point}.")
        if unobserved[point] and not np.allclose(
            active_basis, identity, atol=basis_orthonormal_atol, rtol=0.0
        ):
            raise ValueError(f"Unobserved point {point} must use the identity nullspace basis.")

    output_dtype = phi_source.dtype if np.issubdtype(phi_source.dtype, np.complexfloating) else np.dtype(np.complex128)
    return ValidatedMotionFillInputs(
        phi_observable=phi,
        nullspace_basis=basis,
        point_nullity=nullity,
        anchor_mask=anchor,
        partial_mask=partial,
        unobserved_mask=unobserved,
        excluded_mask=excluded,
        output_dtype=np.dtype(output_dtype),
    )


def _assemble_sparse_system(
    graph: KnnGraph,
    inputs: ValidatedMotionFillInputs,
    connectivity: AnchorConnectivity,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, int]:
    coo_matrix, _ = _require_scipy_sparse()
    variable_mask = connectivity.connected_to_anchor & ~inputs.anchor_mask
    variable_dimensions = np.where(variable_mask, inputs.point_nullity, 0).astype(np.int64)
    coefficient_offsets = np.zeros((graph.num_points + 1,), dtype=np.int64)
    coefficient_offsets[1:] = np.cumsum(variable_dimensions, dtype=np.int64)
    column_count = int(coefficient_offsets[-1])

    active_edge_mask = (
        connectivity.connected_to_anchor[graph.edge_index[:, 0]]
        & connectivity.connected_to_anchor[graph.edge_index[:, 1]]
    )
    active_edge_indices = np.where(active_edge_mask)[0]
    edges = graph.edge_index[active_edge_indices]
    sqrt_weight = np.sqrt(graph.edge_weight[active_edge_indices])
    row_count = int(edges.shape[0] * 3)
    right_hand_side = (
        sqrt_weight[:, None]
        * (inputs.phi_observable[edges[:, 1]] - inputs.phi_observable[edges[:, 0]])
    ).reshape(-1)

    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    edge_rows = np.arange(edges.shape[0], dtype=np.int64)
    coordinate = np.arange(3, dtype=np.int64)
    for side, sign in ((0, 1.0), (1, -1.0)):
        points = edges[:, side]
        for dimension in (1, 2, 3):
            select = variable_mask[points] & (inputs.point_nullity[points] == dimension)
            if not np.any(select):
                continue
            selected_rows = edge_rows[select]
            selected_points = points[select]
            block = (
                sign
                * sqrt_weight[select, None, None]
                * inputs.nullspace_basis[selected_points, :, :dimension]
            )
            rows = np.broadcast_to(
                selected_rows[:, None, None] * 3 + coordinate[None, :, None],
                block.shape,
            )
            columns = np.broadcast_to(
                coefficient_offsets[selected_points, None, None]
                + np.arange(dimension, dtype=np.int64)[None, None, :],
                block.shape,
            )
            row_parts.append(rows.reshape(-1))
            column_parts.append(columns.reshape(-1))
            value_parts.append(block.reshape(-1))

    if value_parts:
        matrix = coo_matrix(
            (
                np.concatenate(value_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=(row_count, column_count),
            dtype=np.float64,
        ).tocsr()
    else:
        matrix = coo_matrix((row_count, column_count), dtype=np.float64).tocsr()
    return matrix, right_hand_side, coefficient_offsets, variable_mask, int(edges.shape[0])


def _empty_solve_metadata(right_hand_side: np.ndarray) -> SparseSolveMetadata:
    return SparseSolveMetadata(
        performed=False,
        converged=True,
        stop_code=0,
        iterations=0,
        residual_norm=float(np.linalg.norm(right_hand_side)),
        normal_residual_norm=0.0,
        matrix_norm=0.0,
        condition_estimate=1.0,
        solution_norm=0.0,
    )


def _run_lsmr(
    matrix: object,
    right_hand_side: np.ndarray,
    *,
    atol: float,
    btol: float,
    conlim: float,
    maxiter: int | None,
) -> tuple[np.ndarray, SparseSolveMetadata]:
    _, lsmr = _require_scipy_sparse()
    solved = lsmr(
        matrix,
        np.asarray(right_hand_side, dtype=np.float64),
        atol=atol,
        btol=btol,
        conlim=conlim,
        maxiter=maxiter,
    )
    stop_code = int(solved[1])
    metadata = SparseSolveMetadata(
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
    return np.asarray(solved[0], dtype=np.float64), metadata


def fill_grouped_affine_motion(
    graph: KnnGraph,
    phi_fixed: np.ndarray,
    anchor_mask: np.ndarray,
    shared_group_index: np.ndarray,
    shared_group_point_blocks: np.ndarray,
    shared_group_dimensions: np.ndarray,
    *,
    lsmr_atol: float = 1e-10,
    lsmr_btol: float = 1e-10,
    lsmr_conlim: float = 1e8,
    lsmr_maxiter: int | None = None,
    timings: dict[str, float] | None = None,
    shared_group_membership: SharedGroupMembership | None = None,
    progress: Callable[[str], None] | None = None,
) -> MotionFillResult:
    """Fill pointwise motion and shared affine groups in one KNN LSMR system."""

    total_started = perf_counter()
    stage_started = perf_counter()
    _validate_graph(graph)
    phi_source = np.asarray(phi_fixed)
    if (
        phi_source.shape != (graph.num_points, 3)
        or not np.issubdtype(phi_source.dtype, np.complexfloating)
        or not np.all(np.isfinite(phi_source))
    ):
        raise ValueError(
            "phi_fixed must be a finite complex array with shape (num_points,3)"
        )
    anchor = _validate_boolean_mask(anchor_mask, "anchor_mask", graph.num_points)
    if not np.any(anchor):
        raise ValueError("anchor_mask must contain at least one fixed point")
    if np.any(phi_source[~anchor] != 0):
        raise ValueError("phi_fixed must be exactly zero outside anchor_mask")

    shared_groups = np.asarray(shared_group_index)
    if (
        shared_groups.shape != (graph.num_points,)
        or not np.issubdtype(shared_groups.dtype, np.integer)
        or np.any(shared_groups < -1)
    ):
        raise ValueError(
            "shared_group_index must be an integer point array with values >= -1"
        )
    shared_group_count = int(shared_groups.max(initial=-1)) + 1
    dimensions = np.asarray(shared_group_dimensions)
    if (
        dimensions.shape != (shared_group_count,)
        or not np.issubdtype(dimensions.dtype, np.integer)
        or np.any(dimensions <= 0)
        or np.any(dimensions > 6)
    ):
        raise ValueError(
            "shared_group_dimensions must contain one integer in [1,6] per group"
        )
    blocks = np.asarray(shared_group_point_blocks, dtype=np.float64)
    if blocks.shape != (graph.num_points, 3, 6) or not np.isfinite(blocks).all():
        raise ValueError(
            "shared_group_point_blocks must be finite with shape (num_points,3,6)"
        )
    membership = (
        build_shared_group_membership(shared_groups, graph.num_points)
        if shared_group_membership is None
        else shared_group_membership
    )
    expected_counts = np.bincount(
        shared_groups[shared_groups >= 0],
        minlength=shared_group_count,
    ).astype(np.int64, copy=False)
    expected_offsets = np.zeros((shared_group_count + 1,), dtype=np.int64)
    expected_offsets[1:] = np.cumsum(expected_counts, dtype=np.int64)
    if (
        membership.counts.shape != (shared_group_count,)
        or membership.offsets.shape != (shared_group_count + 1,)
        or membership.leaders.shape != (shared_group_count,)
        or membership.ordered_points.shape != (int(expected_offsets[-1]),)
        or membership.star_edges.shape
        != (int(expected_offsets[-1]) - shared_group_count, 2)
        or not np.array_equal(membership.counts, expected_counts)
        or not np.array_equal(membership.offsets, expected_offsets)
        or not np.array_equal(
            np.bincount(
                membership.ordered_points,
                minlength=graph.num_points,
            ),
            (shared_groups >= 0).astype(np.int64),
        )
        or not np.array_equal(
            shared_groups[membership.ordered_points],
            np.repeat(
                np.arange(shared_group_count, dtype=shared_groups.dtype),
                expected_counts,
            ),
        )
        or not np.array_equal(
            membership.leaders,
            membership.ordered_points[membership.offsets[:-1]],
        )
    ):
        raise ValueError(
            "shared_group_membership does not match shared_group_index"
        )
    _record_timing(timings, "input_validation_seconds", stage_started)

    stage_started = perf_counter()
    _emit_progress(progress, "grouped connectivity started")
    connectivity = _compute_grouped_anchor_connectivity(
        graph,
        anchor,
        membership,
    )
    _record_timing(timings, "connectivity_seconds", stage_started)
    _emit_progress(
        progress,
        f"grouped connectivity finished in {perf_counter() - stage_started:.3f} s",
    )

    stage_started = perf_counter()
    ordinary_points = np.flatnonzero((shared_groups < 0) & ~anchor)
    point_group_index = np.full((graph.num_points,), -1, dtype=np.int32)
    point_group_index[shared_groups >= 0] = shared_groups[shared_groups >= 0]
    point_group_index[ordinary_points] = (
        shared_group_count
        + np.arange(ordinary_points.size, dtype=np.int32)
    )
    group_dimensions = np.concatenate(
        [
            dimensions.astype(np.int32),
            np.full((ordinary_points.size,), 3, dtype=np.int32),
        ]
    )
    point_dimensions = np.zeros((graph.num_points,), dtype=np.int32)
    variable_points = ~anchor
    point_dimensions[variable_points] = group_dimensions[
        point_group_index[variable_points]
    ]
    point_blocks = np.zeros((graph.num_points, 3, 6), dtype=np.float64)
    shared_points = shared_groups >= 0
    point_blocks[shared_points] = blocks[shared_points]
    point_blocks[ordinary_points, :, :3] = np.eye(3, dtype=np.float64)[None]

    active_group_mask = np.zeros(group_dimensions.shape, dtype=bool)
    connected_variable = connectivity.connected_to_anchor & variable_points
    active_group_mask[point_group_index[connected_variable]] = True
    active_dimensions = np.where(
        active_group_mask,
        group_dimensions,
        0,
    ).astype(np.int64)
    coefficient_offsets = np.zeros(
        (group_dimensions.shape[0] + 1,),
        dtype=np.int64,
    )
    coefficient_offsets[1:] = np.cumsum(active_dimensions, dtype=np.int64)
    _record_timing(timings, "variable_layout_seconds", stage_started)

    stage_started = perf_counter()
    _emit_progress(progress, "grouped sparse assembly started")
    active_edge_mask = (
        connectivity.connected_to_anchor[graph.edge_index[:, 0]]
        & connectivity.connected_to_anchor[graph.edge_index[:, 1]]
    )
    active_edge_indices = np.flatnonzero(active_edge_mask)
    edges = graph.edge_index[active_edge_indices]
    sqrt_weight = np.sqrt(graph.edge_weight[active_edge_indices])
    right_hand_side = (
        sqrt_weight[:, None]
        * (phi_source[edges[:, 1]] - phi_source[edges[:, 0]])
    ).reshape(-1)

    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    edge_rows = np.arange(edges.shape[0], dtype=np.int64)
    coordinate = np.arange(3, dtype=np.int64)
    for side, sign in ((0, 1.0), (1, -1.0)):
        points = edges[:, side]
        for dimension in np.unique(point_dimensions[points]):
            dimension = int(dimension)
            if dimension == 0:
                continue
            select = connected_variable[points] & (
                point_dimensions[points] == dimension
            )
            if not np.any(select):
                continue
            selected_rows = edge_rows[select]
            selected_points = points[select]
            selected_groups = point_group_index[selected_points]
            block = (
                sign
                * sqrt_weight[select, None, None]
                * point_blocks[selected_points, :, :dimension]
            )
            rows = np.broadcast_to(
                selected_rows[:, None, None] * 3 + coordinate[None, :, None],
                block.shape,
            )
            columns = np.broadcast_to(
                coefficient_offsets[selected_groups, None, None]
                + np.arange(dimension, dtype=np.int64)[None, None, :],
                block.shape,
            )
            row_parts.append(rows.reshape(-1))
            column_parts.append(columns.reshape(-1))
            value_parts.append(block.reshape(-1))

    coo_matrix, _ = _require_scipy_sparse()
    row_count = int(edges.shape[0] * 3)
    column_count = int(coefficient_offsets[-1])
    if value_parts:
        matrix = coo_matrix(
            (
                np.concatenate(value_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=(row_count, column_count),
            dtype=np.float64,
        ).tocsr()
    else:
        matrix = coo_matrix(
            (row_count, column_count), dtype=np.float64
        ).tocsr()
    _record_timing(timings, "system_assembly_seconds", stage_started)
    _emit_progress(
        progress,
        "grouped sparse assembly finished in "
        f"{perf_counter() - stage_started:.3f} s: rows={row_count}, "
        f"cols={column_count}, nnz={int(matrix.nnz)}",
    )

    for tolerance, name in ((lsmr_atol, "lsmr_atol"), (lsmr_btol, "lsmr_btol")):
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not np.isfinite(lsmr_conlim) or lsmr_conlim <= 0.0:
        raise ValueError("lsmr_conlim must be finite and positive")
    if lsmr_maxiter is not None:
        lsmr_maxiter = _require_positive_integer(lsmr_maxiter, "lsmr_maxiter")
    if column_count:
        stage_started = perf_counter()
        _emit_progress(progress, "grouped LSMR real started")
        real_coefficients, real_solver = _run_lsmr(
            matrix,
            right_hand_side.real,
            atol=float(lsmr_atol),
            btol=float(lsmr_btol),
            conlim=float(lsmr_conlim),
            maxiter=lsmr_maxiter,
        )
        _record_timing(timings, "lsmr_real_seconds", stage_started)
        _emit_progress(
            progress,
            "grouped LSMR real finished in "
            f"{perf_counter() - stage_started:.3f} s: "
            f"iterations={real_solver.iterations}, stop_code={real_solver.stop_code}",
        )
        stage_started = perf_counter()
        _emit_progress(progress, "grouped LSMR imaginary started")
        imaginary_coefficients, imag_solver = _run_lsmr(
            matrix,
            right_hand_side.imag,
            atol=float(lsmr_atol),
            btol=float(lsmr_btol),
            conlim=float(lsmr_conlim),
            maxiter=lsmr_maxiter,
        )
        _record_timing(timings, "lsmr_imaginary_seconds", stage_started)
        _emit_progress(
            progress,
            "grouped LSMR imaginary finished in "
            f"{perf_counter() - stage_started:.3f} s: "
            f"iterations={imag_solver.iterations}, stop_code={imag_solver.stop_code}",
        )
        if not real_solver.converged or not imag_solver.converged:
            raise RuntimeError(
                "Grouped motion-fill LSMR did not converge: "
                f"real stop_code={real_solver.stop_code}, "
                f"imaginary stop_code={imag_solver.stop_code}"
            )
        coefficient_values = real_coefficients + 1j * imaginary_coefficients
    else:
        coefficient_values = np.empty((0,), dtype=np.complex128)
        real_solver = _empty_solve_metadata(right_hand_side.real)
        imag_solver = _empty_solve_metadata(right_hand_side.imag)
        if timings is not None:
            timings["lsmr_real_seconds"] = 0.0
            timings["lsmr_imaginary_seconds"] = 0.0

    stage_started = perf_counter()
    _emit_progress(progress, "grouped reconstruction started")
    output_dtype = np.dtype(phi_source.dtype)
    phi_filled = phi_source.astype(output_dtype, copy=True)
    correction = np.zeros((graph.num_points, 3), dtype=output_dtype)
    for dimension in np.unique(point_dimensions[connected_variable]):
        dimension = int(dimension)
        if dimension == 0:
            continue
        selected_points = np.flatnonzero(
            connected_variable & (point_dimensions == dimension)
        )
        selected_groups = point_group_index[selected_points]
        coefficient_indices = (
            coefficient_offsets[selected_groups, None]
            + np.arange(dimension, dtype=np.int64)[None]
        )
        point_correction = np.einsum(
            "nij,nj->ni",
            point_blocks[selected_points, :, :dimension],
            coefficient_values[coefficient_indices],
        )
        correction[selected_points] = point_correction.astype(
            output_dtype,
            copy=False,
        )
        phi_filled[selected_points] = correction[selected_points]
    phi_filled[anchor] = phi_source[anchor]
    completion_mask = connected_variable
    result = MotionFillResult(
        phi=phi_filled,
        phi_observable=phi_source.astype(output_dtype, copy=True),
        phi_nullspace_correction=correction,
        completion_mask=completion_mask,
        completion_connected_to_anchor=(
            connectivity.connected_to_anchor.copy()
        ),
        coefficient_values=coefficient_values,
        coefficient_offsets=coefficient_offsets,
        connectivity=connectivity,
        real_solver=real_solver,
        imag_solver=imag_solver,
        system_row_count=row_count,
        system_column_count=column_count,
        active_edge_count=int(edges.shape[0]),
        point_coefficient_group_index=point_group_index,
        coefficient_group_dimensions=group_dimensions,
    )
    _record_timing(timings, "reconstruction_seconds", stage_started)
    _emit_progress(
        progress,
        f"grouped reconstruction finished in {perf_counter() - stage_started:.3f} s",
    )
    _record_timing(timings, "linear_fill_total_seconds", total_started)
    return result


def fill_nullspace_motion(
    graph: KnnGraph,
    phi_observable: np.ndarray,
    point_nullspace_basis: np.ndarray,
    point_nullity: np.ndarray,
    anchor_mask: np.ndarray,
    partial_mask: np.ndarray,
    unobserved_mask: np.ndarray,
    *,
    excluded_mask: np.ndarray | None = None,
    lsmr_atol: float = 1e-10,
    lsmr_btol: float = 1e-10,
    lsmr_conlim: float = 1e8,
    lsmr_maxiter: int | None = None,
) -> MotionFillResult:
    """Complete nullspace coefficients on anchor-connected graph components."""

    _validate_graph(graph)
    inputs = validate_motion_fill_inputs(
        phi_observable,
        point_nullspace_basis,
        point_nullity,
        anchor_mask,
        partial_mask,
        unobserved_mask,
        excluded_mask=excluded_mask,
    )
    if inputs.phi_observable.shape[0] != graph.num_points:
        raise ValueError(
            f"Motion-fill arrays contain {inputs.phi_observable.shape[0]} points, "
            f"but the graph contains {graph.num_points}."
        )
    for tolerance, name in ((lsmr_atol, "lsmr_atol"), (lsmr_btol, "lsmr_btol")):
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if not np.isfinite(lsmr_conlim) or lsmr_conlim <= 0.0:
        raise ValueError("lsmr_conlim must be finite and positive.")
    if lsmr_maxiter is not None:
        lsmr_maxiter = _require_positive_integer(lsmr_maxiter, "lsmr_maxiter")

    connectivity = compute_anchor_connectivity(
        graph,
        inputs.anchor_mask,
        inputs.excluded_mask,
    )
    matrix, rhs, coefficient_offsets, variable_mask, active_edge_count = _assemble_sparse_system(
        graph, inputs, connectivity
    )
    column_count = int(matrix.shape[1])
    if column_count:
        real_coefficients, real_solver = _run_lsmr(
            matrix,
            rhs.real,
            atol=float(lsmr_atol),
            btol=float(lsmr_btol),
            conlim=float(lsmr_conlim),
            maxiter=lsmr_maxiter,
        )
        imaginary_coefficients, imag_solver = _run_lsmr(
            matrix,
            rhs.imag,
            atol=float(lsmr_atol),
            btol=float(lsmr_btol),
            conlim=float(lsmr_conlim),
            maxiter=lsmr_maxiter,
        )
        if not real_solver.converged or not imag_solver.converged:
            raise RuntimeError(
                "Motion-fill LSMR did not converge: "
                f"real stop_code={real_solver.stop_code}, "
                f"imaginary stop_code={imag_solver.stop_code}."
            )
        coefficient_values = real_coefficients + 1j * imaginary_coefficients
    else:
        coefficient_values = np.empty((0,), dtype=np.complex128)
        real_solver = _empty_solve_metadata(rhs.real)
        imag_solver = _empty_solve_metadata(rhs.imag)

    phi_source = np.asarray(phi_observable)
    phi_filled = phi_source.astype(inputs.output_dtype, copy=True)
    correction = np.zeros((graph.num_points, 3), dtype=inputs.output_dtype)
    for point in np.where(variable_mask)[0].tolist():
        start = int(coefficient_offsets[point])
        end = int(coefficient_offsets[point + 1])
        point_correction = (
            inputs.nullspace_basis[point, :, : end - start] @ coefficient_values[start:end]
        )
        correction[point] = point_correction.astype(inputs.output_dtype, copy=False)
        phi_filled[point] = (
            inputs.phi_observable[point] + point_correction
        ).astype(inputs.output_dtype, copy=False)
    phi_filled[inputs.anchor_mask] = phi_source[inputs.anchor_mask]
    phi_filled[inputs.excluded_mask] = phi_source[inputs.excluded_mask]

    completion_mask = connectivity.connected_to_anchor & ~inputs.anchor_mask
    return MotionFillResult(
        phi=phi_filled,
        phi_observable=phi_source.astype(inputs.output_dtype, copy=True),
        phi_nullspace_correction=correction,
        completion_mask=completion_mask,
        completion_connected_to_anchor=connectivity.connected_to_anchor.copy(),
        coefficient_values=coefficient_values,
        coefficient_offsets=coefficient_offsets,
        connectivity=connectivity,
        real_solver=real_solver,
        imag_solver=imag_solver,
        system_row_count=int(matrix.shape[0]),
        system_column_count=column_count,
        active_edge_count=active_edge_count,
    )
