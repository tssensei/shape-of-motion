"""Optimize one latent 3D modal displacement from N-view observations.

Each call solves one modal frequency. The saved ``alphas`` vector has shape
``(V,)`` and represents the per-view slice ``alpha[:, k]`` for that mode.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _require_scipy_for_graph_smoothing():
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "graph smoothing requires scipy.sparse and scipy.spatial.cKDTree. "
            "Install scipy in the active environment or run with --graph-smooth-lambda 0."
        ) from exc
    return sp, spla, cKDTree


def _observations_by_point(num_points: int, obs_point_index: np.ndarray) -> list[np.ndarray]:
    return [np.where(obs_point_index == i)[0] for i in range(num_points)]


def _solve_phi_points(
    obs_y: np.ndarray,
    obs_J: np.ndarray,
    obs_confidence: np.ndarray,
    obs_view_index: np.ndarray,
    obs_by_point: list[np.ndarray],
    active: np.ndarray,
    alphas: np.ndarray,
    ridge_mu: float,
) -> np.ndarray:
    if ridge_mu < 0:
        raise ValueError("ridge_mu must be non-negative.")
    phi = np.zeros((len(obs_by_point), 3), dtype=np.complex64)
    eye = np.eye(3, dtype=np.complex128)
    for point_idx, rows in enumerate(obs_by_point):
        if not active[point_idx]:
            continue
        if rows.size == 0:
            continue
        weights = np.sqrt(np.maximum(obs_confidence[rows].astype(np.float64), 0.0))
        alpha_rows = alphas[obs_view_index[rows]].astype(np.complex128)
        A = (weights[:, None, None] * alpha_rows[:, None, None] * obs_J[rows].astype(np.complex128)).reshape(-1, 3)
        b = (weights[:, None] * obs_y[rows].astype(np.complex128)).reshape(-1)
        lhs = A.conj().T @ A + float(ridge_mu) * eye
        rhs = A.conj().T @ b
        phi[point_idx] = np.linalg.solve(lhs, rhs).astype(np.complex64)
    return phi


def _point_view_masks(num_points: int, obs_point_index: np.ndarray, obs_view_index: np.ndarray, num_views: int) -> np.ndarray:
    masks = np.zeros((num_points, num_views), dtype=bool)
    masks[obs_point_index, obs_view_index] = True
    return masks


def _obs_count_weights(
    obs_count_per_point: np.ndarray,
    weight_1: float,
    weight_2: float,
    weight_3plus: float,
) -> np.ndarray:
    if weight_1 < 0 or weight_2 < 0 or weight_3plus < 0:
        raise ValueError("obs-count weights must be non-negative.")
    weights = np.full((obs_count_per_point.shape[0],), float(weight_3plus), dtype=np.float32)
    weights[obs_count_per_point <= 1] = float(weight_1)
    weights[obs_count_per_point == 2] = float(weight_2)
    return weights


def _build_graph_edges(
    points: np.ndarray,
    active: np.ndarray,
    point_view_masks: np.ndarray,
    graph_smooth_k: int,
    graph_auto_radius_scale: float,
    graph_min_shared_views: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if graph_smooth_k <= 0:
        raise ValueError("graph_smooth_k must be positive.")
    if graph_auto_radius_scale <= 0:
        raise ValueError("graph_auto_radius_scale must be positive.")
    if graph_min_shared_views < 0:
        raise ValueError("graph_min_shared_views must be non-negative.")

    _, _, cKDTree = _require_scipy_for_graph_smoothing()
    active_indices = np.where(active)[0]
    if active_indices.size < 2:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((points.shape[0],), dtype=np.int32),
            0.0,
        )
    k = min(int(graph_smooth_k) + 1, int(active_indices.size))
    tree = cKDTree(points[active_indices].astype(np.float64))
    distances, neighbor_rows = tree.query(points[active_indices].astype(np.float64), k=k)
    if k == 1:
        distances = distances[:, None]
        neighbor_rows = neighbor_rows[:, None]
    neighbor_distances = distances[:, 1:] if distances.shape[1] > 1 else distances
    finite_neighbor_distances = neighbor_distances[np.isfinite(neighbor_distances) & (neighbor_distances > 0)]
    if finite_neighbor_distances.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((points.shape[0],), dtype=np.int32),
            0.0,
        )
    radius = float(np.median(finite_neighbor_distances) * float(graph_auto_radius_scale))
    if radius <= 0:
        raise ValueError("graph auto radius is non-positive.")

    edge_weights_by_pair: dict[tuple[int, int], float] = {}
    for row, point_idx in enumerate(active_indices.tolist()):
        for dist, neighbor_row in zip(distances[row, 1:], neighbor_rows[row, 1:]):
            if not np.isfinite(dist) or dist <= 0 or dist > radius:
                continue
            neighbor_idx = int(active_indices[int(neighbor_row)])
            if point_idx == neighbor_idx:
                continue
            if graph_min_shared_views > 0:
                shared = int(np.logical_and(point_view_masks[point_idx], point_view_masks[neighbor_idx]).sum())
                if shared < int(graph_min_shared_views):
                    continue
            a, b = sorted((int(point_idx), int(neighbor_idx)))
            if (a, b) in edge_weights_by_pair:
                continue
            edge_weights_by_pair[(a, b)] = 1.0 / max(float(dist), 1e-6)

    if not edge_weights_by_pair:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((points.shape[0],), dtype=np.int32),
            radius,
        )
    sorted_pairs = sorted(edge_weights_by_pair)
    edges = np.asarray(sorted_pairs, dtype=np.int64)
    weights = np.asarray([edge_weights_by_pair[pair] for pair in sorted_pairs], dtype=np.float64)
    weights = weights / max(float(np.median(weights)), 1e-12)
    degree = np.zeros((points.shape[0],), dtype=np.int32)
    np.add.at(degree, edges[:, 0], 1)
    np.add.at(degree, edges[:, 1], 1)
    return edges[:, 0], edges[:, 1], weights, degree, radius


def _solve_phi_graph(
    points: np.ndarray,
    obs_y: np.ndarray,
    obs_J: np.ndarray,
    obs_confidence: np.ndarray,
    obs_point_index: np.ndarray,
    obs_view_index: np.ndarray,
    obs_count_per_point: np.ndarray,
    active: np.ndarray,
    alphas: np.ndarray,
    ridge_mu: float,
    graph_smooth_lambda: float,
    graph_smooth_k: int,
    graph_auto_radius_scale: float,
    graph_min_shared_views: int,
    obs_count_weight_1: float,
    obs_count_weight_2: float,
    obs_count_weight_3plus: float,
    num_views: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if graph_smooth_lambda <= 0:
        raise ValueError("_solve_phi_graph requires positive graph_smooth_lambda.")
    sp, spla, _ = _require_scipy_for_graph_smoothing()
    active_indices = np.where(active)[0]
    if active_indices.size < 3:
        raise ValueError("Too few active points for graph smoothing.")
    active_to_col = np.full((points.shape[0],), -1, dtype=np.int64)
    active_to_col[active_indices] = np.arange(active_indices.size, dtype=np.int64)
    point_view_masks = _point_view_masks(points.shape[0], obs_point_index, obs_view_index, num_views)
    edge_a, edge_b, edge_weights, graph_degree, graph_auto_radius = _build_graph_edges(
        points,
        active,
        point_view_masks,
        graph_smooth_k,
        graph_auto_radius_scale,
        graph_min_shared_views,
    )
    if edge_a.size == 0:
        raise ValueError("Graph smoothing produced no valid edges; relax graph parameters or run with --graph-smooth-lambda 0.")

    obs_count_weight_per_point = _obs_count_weights(
        obs_count_per_point,
        obs_count_weight_1,
        obs_count_weight_2,
        obs_count_weight_3plus,
    )
    keep_obs = active[obs_point_index]
    obs_rows = np.where(keep_obs)[0]
    data_rows = obs_rows.size * 2
    edge_rows = edge_a.size * 3
    ridge_rows = active_indices.size * 3
    total_rows = data_rows + edge_rows + ridge_rows
    total_cols = active_indices.size * 3

    row_idx: list[np.ndarray] = []
    col_idx: list[np.ndarray] = []
    values: list[np.ndarray] = []
    rhs = np.zeros((total_rows,), dtype=np.complex128)

    data_base = np.arange(data_rows, dtype=np.int64).reshape(obs_rows.size, 2)
    data_weights = np.sqrt(
        np.maximum(obs_confidence[obs_rows].astype(np.float64), 0.0)
        * np.maximum(obs_count_weight_per_point[obs_point_index[obs_rows]].astype(np.float64), 0.0)
    )
    alpha_rows = alphas[obs_view_index[obs_rows]].astype(np.complex128)
    point_cols = active_to_col[obs_point_index[obs_rows]]
    for comp in range(2):
        rows = np.repeat(data_base[:, comp], 3)
        cols = np.repeat(point_cols * 3, 3) + np.tile(np.arange(3, dtype=np.int64), obs_rows.size)
        vals = (
            data_weights[:, None]
            * alpha_rows[:, None]
            * obs_J[obs_rows, comp, :].astype(np.complex128)
        ).reshape(-1)
        row_idx.append(rows)
        col_idx.append(cols)
        values.append(vals)
        rhs[data_base[:, comp]] = data_weights * obs_y[obs_rows, comp].astype(np.complex128)

    edge_start = data_rows
    smooth_weight = np.sqrt(float(graph_smooth_lambda) * np.maximum(edge_weights, 0.0))
    edge_local_a = active_to_col[edge_a]
    edge_local_b = active_to_col[edge_b]
    for comp in range(3):
        rows = edge_start + np.arange(edge_a.size, dtype=np.int64) * 3 + comp
        row_idx.append(np.repeat(rows, 2))
        col_idx.append(np.column_stack([edge_local_a * 3 + comp, edge_local_b * 3 + comp]).reshape(-1))
        values.append(np.column_stack([smooth_weight, -smooth_weight]).reshape(-1).astype(np.complex128))

    ridge_start = data_rows + edge_rows
    if ridge_mu > 0:
        ridge_weight = np.sqrt(float(ridge_mu))
        cols = np.arange(ridge_rows, dtype=np.int64)
        rows = ridge_start + cols
        row_idx.append(rows)
        col_idx.append(cols)
        values.append(np.full((ridge_rows,), ridge_weight, dtype=np.complex128))

    matrix = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(row_idx), np.concatenate(col_idx))),
        shape=(total_rows, total_cols),
        dtype=np.complex128,
    ).tocsr()
    solution = spla.lsmr(matrix, rhs, atol=1e-6, btol=1e-6)[0]
    phi = np.zeros((points.shape[0], 3), dtype=np.complex64)
    phi[active_indices] = solution.reshape(active_indices.size, 3).astype(np.complex64)
    return phi, graph_degree, obs_count_weight_per_point, edge_a, edge_b, graph_auto_radius


def _graph_smooth_residual(
    phi: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    residual = np.zeros((phi.shape[0],), dtype=np.float32)
    counts = np.zeros((phi.shape[0],), dtype=np.float64)
    if edge_a.size == 0:
        residual[~active] = np.inf
        return residual
    diff = np.linalg.norm(phi[edge_a] - phi[edge_b], axis=1).astype(np.float64)
    np.add.at(residual, edge_a, diff)
    np.add.at(residual, edge_b, diff)
    np.add.at(counts, edge_a, 1.0)
    np.add.at(counts, edge_b, 1.0)
    residual = (residual.astype(np.float64) / np.maximum(counts, 1.0)).astype(np.float32)
    residual[~active] = np.inf
    return residual


def _build_modal_rigid_edges(
    points: np.ndarray,
    active: np.ndarray,
    point_view_masks: np.ndarray,
    modal_rigid_k: int,
    modal_rigid_auto_radius_scale: float,
    modal_rigid_min_shared_views: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if modal_rigid_k <= 0:
        raise ValueError("modal_rigid_k must be positive.")
    if modal_rigid_auto_radius_scale <= 0:
        raise ValueError("modal_rigid_auto_radius_scale must be positive.")
    if modal_rigid_min_shared_views < 0:
        raise ValueError("modal_rigid_min_shared_views must be non-negative.")

    _, _, cKDTree = _require_scipy_for_graph_smoothing()
    active_indices = np.where(active)[0]
    if active_indices.size < 2:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((points.shape[0],), dtype=np.int32),
            0.0,
        )

    k = min(int(modal_rigid_k) + 1, int(active_indices.size))
    tree = cKDTree(points[active_indices].astype(np.float64))
    distances, neighbor_rows = tree.query(points[active_indices].astype(np.float64), k=k)
    if k == 1:
        distances = distances[:, None]
        neighbor_rows = neighbor_rows[:, None]

    neighbor_distances = distances[:, 1:] if distances.shape[1] > 1 else distances
    finite_neighbor_distances = neighbor_distances[np.isfinite(neighbor_distances) & (neighbor_distances > 0)]
    if finite_neighbor_distances.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((points.shape[0],), dtype=np.int32),
            0.0,
        )
    radius = float(np.median(finite_neighbor_distances) * float(modal_rigid_auto_radius_scale))
    if radius <= 0:
        raise ValueError("modal rigidity auto radius is non-positive.")

    neighbor_sets: list[set[int]] = []
    for row in range(active_indices.size):
        valid = []
        for dist, neighbor_row in zip(distances[row, 1:], neighbor_rows[row, 1:]):
            if np.isfinite(dist) and dist > 0:
                valid.append(int(neighbor_row))
        neighbor_sets.append(set(valid))

    edge_data: dict[tuple[int, int], float] = {}
    for row, point_idx in enumerate(active_indices.tolist()):
        for dist, neighbor_row in zip(distances[row, 1:], neighbor_rows[row, 1:]):
            if not np.isfinite(dist) or dist <= 0 or dist > radius:
                continue
            neighbor_row = int(neighbor_row)
            if row not in neighbor_sets[neighbor_row]:
                continue
            neighbor_idx = int(active_indices[neighbor_row])
            if point_idx == neighbor_idx:
                continue
            if modal_rigid_min_shared_views > 0:
                shared = int(np.logical_and(point_view_masks[point_idx], point_view_masks[neighbor_idx]).sum())
                if shared < int(modal_rigid_min_shared_views):
                    continue
            a, b = sorted((int(point_idx), int(neighbor_idx)))
            if (a, b) in edge_data:
                continue
            edge_data[(a, b)] = 1.0 / max(float(dist), 1e-6)

    if not edge_data:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((points.shape[0],), dtype=np.int32),
            radius,
        )

    sorted_pairs = sorted(edge_data)
    edges = np.asarray(sorted_pairs, dtype=np.int64)
    weights = np.asarray([edge_data[pair] for pair in sorted_pairs], dtype=np.float64)
    weights = weights / max(float(np.median(weights)), 1e-12)
    degree = np.zeros((points.shape[0],), dtype=np.int32)
    np.add.at(degree, edges[:, 0], 1)
    np.add.at(degree, edges[:, 1], 1)
    return edges[:, 0], edges[:, 1], weights, degree, radius


def _solve_phi_modal_rigid(
    points: np.ndarray,
    obs_y: np.ndarray,
    obs_J: np.ndarray,
    obs_confidence: np.ndarray,
    obs_point_index: np.ndarray,
    obs_view_index: np.ndarray,
    active: np.ndarray,
    alphas: np.ndarray,
    ridge_mu: float,
    modal_rigid_lambda: float,
    modal_rigid_edge_a: np.ndarray,
    modal_rigid_edge_b: np.ndarray,
    modal_rigid_edge_weights: np.ndarray,
) -> np.ndarray:
    if modal_rigid_lambda <= 0:
        raise ValueError("_solve_phi_modal_rigid requires positive modal_rigid_lambda.")
    sp, spla, _ = _require_scipy_for_graph_smoothing()
    active_indices = np.where(active)[0]
    if active_indices.size < 3:
        raise ValueError("Too few active points for modal rigidity.")
    if modal_rigid_edge_a.size == 0:
        raise ValueError("Modal rigidity produced no valid edges; relax modal-rigid parameters or run with --modal-rigid-lambda 0.")

    active_to_col = np.full((points.shape[0],), -1, dtype=np.int64)
    active_to_col[active_indices] = np.arange(active_indices.size, dtype=np.int64)
    if np.any(active_to_col[modal_rigid_edge_a] < 0) or np.any(active_to_col[modal_rigid_edge_b] < 0):
        raise ValueError("Modal rigidity edges reference inactive points.")

    keep_obs = active[obs_point_index]
    obs_rows = np.where(keep_obs)[0]
    data_rows = obs_rows.size * 2
    edge_rows = modal_rigid_edge_a.size * 3
    ridge_rows = active_indices.size * 3
    total_rows = data_rows + edge_rows + ridge_rows
    total_cols = active_indices.size * 3

    row_idx: list[np.ndarray] = []
    col_idx: list[np.ndarray] = []
    values: list[np.ndarray] = []
    rhs = np.zeros((total_rows,), dtype=np.complex128)

    data_base = np.arange(data_rows, dtype=np.int64).reshape(obs_rows.size, 2)
    data_weights = np.sqrt(np.maximum(obs_confidence[obs_rows].astype(np.float64), 0.0))
    alpha_rows = alphas[obs_view_index[obs_rows]].astype(np.complex128)
    point_cols = active_to_col[obs_point_index[obs_rows]]
    for comp in range(2):
        rows = np.repeat(data_base[:, comp], 3)
        cols = np.repeat(point_cols * 3, 3) + np.tile(np.arange(3, dtype=np.int64), obs_rows.size)
        vals = (
            data_weights[:, None]
            * alpha_rows[:, None]
            * obs_J[obs_rows, comp, :].astype(np.complex128)
        ).reshape(-1)
        row_idx.append(rows)
        col_idx.append(cols)
        values.append(vals)
        rhs[data_base[:, comp]] = data_weights * obs_y[obs_rows, comp].astype(np.complex128)

    edge_start = data_rows
    rigid_weight = np.sqrt(float(modal_rigid_lambda) * np.maximum(modal_rigid_edge_weights, 0.0))
    edge_local_a = active_to_col[modal_rigid_edge_a]
    edge_local_b = active_to_col[modal_rigid_edge_b]
    edge_rows_idx = (edge_start + np.arange(edge_rows, dtype=np.int64)).reshape(modal_rigid_edge_a.size, 3)
    row_idx.append(np.repeat(edge_rows_idx.reshape(-1), 2))
    col_idx.append(
        np.column_stack(
            [
                edge_local_a * 3,
                edge_local_b * 3,
                edge_local_a * 3 + 1,
                edge_local_b * 3 + 1,
                edge_local_a * 3 + 2,
                edge_local_b * 3 + 2,
            ]
        ).reshape(-1)
    )
    rigid_vals = np.column_stack(
        [
            rigid_weight,
            -rigid_weight,
            rigid_weight,
            -rigid_weight,
            rigid_weight,
            -rigid_weight,
        ]
    )
    values.append(rigid_vals.reshape(-1).astype(np.complex128))

    ridge_start = data_rows + edge_rows
    if ridge_mu > 0:
        ridge_weight = np.sqrt(float(ridge_mu))
        cols = np.arange(ridge_rows, dtype=np.int64)
        rows = ridge_start + cols
        row_idx.append(rows)
        col_idx.append(cols)
        values.append(np.full((ridge_rows,), ridge_weight, dtype=np.complex128))

    matrix = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(row_idx), np.concatenate(col_idx))),
        shape=(total_rows, total_cols),
        dtype=np.complex128,
    ).tocsr()
    solution = spla.lsmr(matrix, rhs, atol=1e-6, btol=1e-6)[0]
    phi = np.zeros((points.shape[0], 3), dtype=np.complex64)
    phi[active_indices] = solution.reshape(active_indices.size, 3).astype(np.complex64)
    return phi


def _modal_rigid_residual(
    phi: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    residual = np.zeros((phi.shape[0],), dtype=np.float32)
    counts = np.zeros((phi.shape[0],), dtype=np.float64)
    if edge_a.size == 0:
        residual[~active] = np.inf
        return residual
    diff = np.linalg.norm(phi[edge_a] - phi[edge_b], axis=1).astype(np.float64)
    np.add.at(residual, edge_a, diff)
    np.add.at(residual, edge_b, diff)
    np.add.at(counts, edge_a, 1.0)
    np.add.at(counts, edge_b, 1.0)
    residual = (residual.astype(np.float64) / np.maximum(counts, 1.0)).astype(np.float32)
    residual[~active] = np.inf
    return residual


def _propagate_modal_motion_to_unobserved(
    points: np.ndarray,
    phi: np.ndarray,
    obs_count_per_point: np.ndarray,
    active: np.ndarray,
    modal_fill_k: int,
    modal_fill_auto_radius_scale: float,
    modal_fill_anchor_min_observations: int,
    modal_fill_ridge_mu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if modal_fill_k <= 0:
        raise ValueError("modal_fill_k must be positive.")
    if modal_fill_auto_radius_scale <= 0:
        raise ValueError("modal_fill_auto_radius_scale must be positive.")
    if modal_fill_anchor_min_observations <= 0:
        raise ValueError("modal_fill_anchor_min_observations must be positive.")
    if modal_fill_ridge_mu < 0:
        raise ValueError("modal_fill_ridge_mu must be non-negative.")

    sp, spla, cKDTree = _require_scipy_for_graph_smoothing()
    filled_phi = phi.copy()
    fill_degree = np.zeros((points.shape[0],), dtype=np.int32)
    target_mask = active & (obs_count_per_point < int(modal_fill_anchor_min_observations))
    anchor_mask = active & (obs_count_per_point >= int(modal_fill_anchor_min_observations))
    connected_to_anchor = np.zeros((points.shape[0],), dtype=bool)
    if not np.any(target_mask):
        return filled_phi, target_mask, connected_to_anchor, fill_degree, 0.0
    if not np.any(anchor_mask):
        raise ValueError("Modal fill has target points but no anchor points; lower --modal-fill-anchor-min-observations.")

    active_indices = np.where(active)[0]
    if active_indices.size < 2:
        raise ValueError("Modal fill requires at least two active points.")
    k = min(int(modal_fill_k) + 1, int(active_indices.size))
    tree = cKDTree(points[active_indices].astype(np.float64))
    distances, neighbor_rows = tree.query(points[active_indices].astype(np.float64), k=k)
    if k == 1:
        distances = distances[:, None]
        neighbor_rows = neighbor_rows[:, None]

    neighbor_distances = distances[:, 1:] if distances.shape[1] > 1 else distances
    finite_neighbor_distances = neighbor_distances[np.isfinite(neighbor_distances) & (neighbor_distances > 0)]
    if finite_neighbor_distances.size == 0:
        raise ValueError("Modal fill could not find finite neighbor distances.")
    radius = float(np.median(finite_neighbor_distances) * float(modal_fill_auto_radius_scale))
    if radius <= 0:
        raise ValueError("modal fill auto radius is non-positive.")

    target_to_col = np.full((points.shape[0],), -1, dtype=np.int64)
    target_indices = np.where(target_mask)[0]
    target_to_col[target_indices] = np.arange(target_indices.size, dtype=np.int64)

    edge_data: dict[tuple[int, int], float] = {}
    for row, point_idx in enumerate(active_indices.tolist()):
        for dist, neighbor_row in zip(distances[row, 1:], neighbor_rows[row, 1:]):
            if not np.isfinite(dist) or dist <= 0 or dist > radius:
                continue
            neighbor_idx = int(active_indices[int(neighbor_row)])
            if point_idx == neighbor_idx:
                continue
            if not (target_mask[point_idx] or target_mask[neighbor_idx]):
                continue
            a, b = sorted((int(point_idx), int(neighbor_idx)))
            edge_data[(a, b)] = min(float(dist), edge_data.get((a, b), float(dist)))

    if not edge_data:
        raise ValueError(
            "Modal fill produced no propagation edges; increase --modal-fill-auto-radius-scale "
            "or --modal-fill-k."
        )

    sorted_edges = sorted(edge_data)
    edges = np.asarray(sorted_edges, dtype=np.int64)
    distances_arr = np.asarray([edge_data[pair] for pair in sorted_edges], dtype=np.float64)
    weights = 1.0 / np.maximum(distances_arr, 1e-6)
    weights = weights / max(float(np.median(weights)), 1e-12)
    np.add.at(fill_degree, edges[:, 0], 1)
    np.add.at(fill_degree, edges[:, 1], 1)

    adjacency: list[list[int]] = [[] for _ in range(points.shape[0])]
    for a, b in edges.tolist():
        adjacency[a].append(b)
        adjacency[b].append(a)
    queue = [int(i) for i in np.where(anchor_mask & (fill_degree > 0))[0]]
    for idx in queue:
        connected_to_anchor[idx] = True
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for neighbor in adjacency[node]:
            if not active[neighbor] or connected_to_anchor[neighbor]:
                continue
            connected_to_anchor[neighbor] = True
            queue.append(neighbor)

    edge_rows = []
    edge_cols = []
    edge_vals = []
    rhs_base = np.zeros((edges.shape[0] + target_indices.size,), dtype=np.float64)
    row_count = 0
    for edge_idx, (a, b) in enumerate(edges.tolist()):
        weight = float(np.sqrt(weights[edge_idx]))
        a_is_target = target_mask[a]
        b_is_target = target_mask[b]
        if a_is_target and b_is_target:
            edge_rows.extend([row_count, row_count])
            edge_cols.extend([int(target_to_col[a]), int(target_to_col[b])])
            edge_vals.extend([weight, -weight])
        elif a_is_target:
            edge_rows.append(row_count)
            edge_cols.append(int(target_to_col[a]))
            edge_vals.append(weight)
            rhs_base[row_count] = weight
        elif b_is_target:
            edge_rows.append(row_count)
            edge_cols.append(int(target_to_col[b]))
            edge_vals.append(weight)
            rhs_base[row_count] = weight
        row_count += 1

    if modal_fill_ridge_mu > 0:
        ridge_weight = float(np.sqrt(modal_fill_ridge_mu))
        for local_idx in range(target_indices.size):
            edge_rows.append(row_count)
            edge_cols.append(local_idx)
            edge_vals.append(ridge_weight)
            row_count += 1

    if row_count == 0:
        raise ValueError("Modal fill produced an empty sparse system.")
    matrix = sp.coo_matrix(
        (
            np.asarray(edge_vals, dtype=np.complex128),
            (np.asarray(edge_rows, dtype=np.int64), np.asarray(edge_cols, dtype=np.int64)),
        ),
        shape=(row_count, target_indices.size),
        dtype=np.complex128,
    ).tocsr()

    solution = np.zeros((target_indices.size, 3), dtype=np.complex128)
    for comp in range(3):
        rhs = np.zeros((row_count,), dtype=np.complex128)
        rhs[: edges.shape[0]] = rhs_base[: edges.shape[0]].astype(np.complex128)
        for edge_idx, (a, b) in enumerate(edges.tolist()):
            weight_rhs = rhs_base[edge_idx]
            if weight_rhs == 0:
                continue
            anchor_idx = b if target_mask[a] else a
            rhs[edge_idx] = weight_rhs * np.complex128(phi[anchor_idx, comp])
        solution[:, comp] = spla.lsmr(matrix, rhs, atol=1e-6, btol=1e-6)[0]

    filled_phi[target_indices] = solution.astype(np.complex64)
    return filled_phi, target_mask, connected_to_anchor, fill_degree, radius


def _predict_observations(obs_J: np.ndarray, obs_point_index: np.ndarray, obs_view_index: np.ndarray, phi: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    projected = np.einsum("oij,oj->oi", obs_J.astype(np.float32), phi[obs_point_index].astype(np.complex64))
    return (alphas[obs_view_index][:, None] * projected).astype(np.complex64)


def _solve_alphas(
    obs_y: np.ndarray,
    obs_J: np.ndarray,
    obs_point_index: np.ndarray,
    obs_view_index: np.ndarray,
    obs_confidence: np.ndarray,
    active: np.ndarray,
    phi: np.ndarray,
    num_views: int,
) -> np.ndarray:
    """Solve per-view, per-current-mode complex offsets with view 0 fixed."""
    alphas = np.ones((num_views,), dtype=np.complex64)
    active_obs = active[obs_point_index]
    for view_idx in range(1, num_views):
        rows = np.where(active_obs & (obs_view_index == view_idx))[0]
        if rows.size == 0:
            raise ValueError(f"Cannot solve alpha for view index {view_idx}: no active observations.")
        projected = np.einsum("oij,oj->oi", obs_J[rows].astype(np.float32), phi[obs_point_index[rows]].astype(np.complex64))
        weights = np.maximum(obs_confidence[rows].astype(np.float64), 0.0)
        numerator = np.sum(weights[:, None] * np.conj(projected) * obs_y[rows])
        denominator = np.sum(weights[:, None] * np.conj(projected) * projected)
        denom_real = float(np.real(denominator))
        if denom_real <= 1e-12:
            raise ValueError(f"Cannot solve alpha for view index {view_idx}: projected motion energy is too small.")
        alphas[view_idx] = np.complex64(numerator / denominator)
    return alphas


def _obs_residual(obs_y: np.ndarray, pred_y: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(obs_y - pred_y) ** 2, axis=1)).astype(np.float32)


def _point_residuals(num_points: int, obs_point_index: np.ndarray, obs_residual: np.ndarray, active: np.ndarray) -> np.ndarray:
    sums = np.zeros((num_points,), dtype=np.float64)
    counts = np.zeros((num_points,), dtype=np.float64)
    np.add.at(sums, obs_point_index, obs_residual.astype(np.float64) ** 2)
    np.add.at(counts, obs_point_index, 1.0)
    out = np.sqrt(sums / np.maximum(counts, 1.0)).astype(np.float32)
    out[~active] = np.inf
    return out


def _refine_single_view_points(
    points: np.ndarray,
    phi: np.ndarray,
    obs_y: np.ndarray,
    obs_J: np.ndarray,
    obs_confidence: np.ndarray,
    obs_point_index: np.ndarray,
    obs_view_index: np.ndarray,
    obs_by_point: list[np.ndarray],
    obs_count_per_point: np.ndarray,
    active: np.ndarray,
    alphas: np.ndarray,
    ridge_mu: float,
    smooth_lambda: float,
    smooth_k: int,
    anchor_min_observations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if smooth_lambda < 0:
        raise ValueError("single_view_smooth_lambda must be non-negative.")
    if smooth_k <= 0:
        raise ValueError("single_view_smooth_k must be positive.")
    if anchor_min_observations < 2:
        raise ValueError("single_view_anchor_min_observations must be at least 2.")
    refined = np.zeros((points.shape[0],), dtype=bool)
    anchor_distance = np.full((points.shape[0],), np.inf, dtype=np.float32)
    if smooth_lambda == 0:
        return phi, refined, anchor_distance

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "single-view smoothing requires scipy.spatial.cKDTree. "
            "Install scipy in the active environment or run with --single-view-smooth-lambda 0."
        ) from exc

    active_counts = np.where(active, obs_count_per_point, 0)
    single_indices = np.where(active & (active_counts == 1))[0]
    anchor_indices = np.where(active & (active_counts >= int(anchor_min_observations)))[0]
    if single_indices.size == 0:
        return phi, refined, anchor_distance
    if anchor_indices.size == 0:
        raise ValueError("single-view smoothing found no active anchor points.")

    k = min(int(smooth_k), int(anchor_indices.size))
    tree = cKDTree(points[anchor_indices].astype(np.float64))
    distances, neighbor_rows = tree.query(points[single_indices].astype(np.float64), k=k)
    if k == 1:
        distances = distances[:, None]
        neighbor_rows = neighbor_rows[:, None]

    eye = np.eye(3, dtype=np.complex128)
    refined_phi = phi.copy()
    smooth_weight = np.sqrt(float(smooth_lambda))
    for local_i, point_idx in enumerate(single_indices.tolist()):
        rows = obs_by_point[point_idx]
        if rows.size != 1:
            continue

        neighbor_indices = anchor_indices[neighbor_rows[local_i]]
        d = distances[local_i].astype(np.float64)
        valid_neighbors = np.isfinite(d)
        if not np.any(valid_neighbors):
            continue
        neighbor_indices = neighbor_indices[valid_neighbors]
        d = d[valid_neighbors]
        weights = 1.0 / np.maximum(d, 1e-6)
        weights = weights / np.sum(weights)
        phi_anchor = np.sum(weights[:, None] * phi[neighbor_indices].astype(np.complex128), axis=0)

        data_weights = np.sqrt(np.maximum(obs_confidence[rows].astype(np.float64), 0.0))
        alpha_rows = alphas[obs_view_index[rows]].astype(np.complex128)
        A_data = (data_weights[:, None, None] * alpha_rows[:, None, None] * obs_J[rows].astype(np.complex128)).reshape(-1, 3)
        b_data = (data_weights[:, None] * obs_y[rows].astype(np.complex128)).reshape(-1)
        A = np.vstack([A_data, smooth_weight * eye])
        b = np.concatenate([b_data, smooth_weight * phi_anchor])
        lhs = A.conj().T @ A + float(ridge_mu) * eye
        rhs = A.conj().T @ b
        refined_phi[point_idx] = np.linalg.solve(lhs, rhs).astype(np.complex64)
        refined[point_idx] = True
        anchor_distance[point_idx] = float(np.sum(weights * d))
    return refined_phi, refined, anchor_distance


def _scatter_mode_image(
    out_path: Path,
    pixels_xy: np.ndarray,
    values: np.ndarray,
    width: int,
    height: int,
    title: str,
    cmap: str = "magma",
    normalize: bool = True,
) -> None:
    amp = np.sqrt(np.sum(np.abs(values) ** 2, axis=1)).astype(np.float32)
    if normalize:
        hi = float(np.percentile(amp, 99)) if amp.size else 1.0
        hi = max(hi, 1e-6)
        color_values = np.clip(amp / hi, 0.0, 1.0)
        vmax = 1.0
    else:
        color_values = amp
        vmax = None
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(pixels_xy[:, 0], pixels_xy[:, 1], c=color_values, s=2, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.7g} {p[1]:.7g} {p[2]:.7g} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _amplitude_colors(phi: np.ndarray) -> np.ndarray:
    amp = np.linalg.norm(phi, axis=1)
    hi = float(np.percentile(amp, 99)) if amp.size else 1.0
    val = np.clip(amp / max(hi, 1e-12), 0.0, 1.0)
    rgb = plt.get_cmap("magma")(val)[:, :3]
    return (255.0 * rgb).astype(np.uint8)


def _phase_colors(phi: np.ndarray) -> np.ndarray:
    phase = np.angle(phi[:, 0])
    hue = (phase + np.pi) / (2.0 * np.pi)
    rgb = plt.get_cmap("hsv")(hue)[:, :3]
    return (255.0 * rgb).astype(np.uint8)


def optimize_multi_view(
    observations_path: str | Path,
    out_path: str | Path,
    vis_dir: str | Path | None = None,
    iterations: int = 8,
    ridge_mu: float = 1e-4,
    outlier_frac: float = 0.05,
    single_view_smooth_lambda: float = 0.0,
    single_view_smooth_k: int = 8,
    single_view_anchor_min_observations: int = 2,
    graph_smooth_lambda: float = 0.0,
    graph_smooth_k: int = 8,
    graph_auto_radius_scale: float = 2.5,
    graph_min_shared_views: int = 1,
    modal_rigid_lambda: float = 0.0,
    modal_rigid_k: int = 8,
    modal_rigid_auto_radius_scale: float = 2.0,
    modal_rigid_min_shared_views: int = 0,
    modal_fill_unobserved: bool = False,
    modal_fill_k: int = 4,
    modal_fill_auto_radius_scale: float = 1.0,
    modal_fill_anchor_min_observations: int = 1,
    modal_fill_ridge_mu: float = 1e-6,
    obs_count_weight_1: float = 0.25,
    obs_count_weight_2: float = 0.75,
    obs_count_weight_3plus: float = 1.0,
) -> Path:
    """Optimize phi_i and per-view alpha_{v,k} for one current mode."""
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if not (0.0 <= outlier_frac < 0.5):
        raise ValueError("outlier_frac must be in [0, 0.5).")
    if single_view_smooth_lambda < 0:
        raise ValueError("single_view_smooth_lambda must be non-negative.")
    if single_view_smooth_k <= 0:
        raise ValueError("single_view_smooth_k must be positive.")
    if single_view_anchor_min_observations < 2:
        raise ValueError("single_view_anchor_min_observations must be at least 2.")
    if graph_smooth_lambda < 0:
        raise ValueError("graph_smooth_lambda must be non-negative.")
    if graph_smooth_k <= 0:
        raise ValueError("graph_smooth_k must be positive.")
    if graph_auto_radius_scale <= 0:
        raise ValueError("graph_auto_radius_scale must be positive.")
    if graph_min_shared_views < 0:
        raise ValueError("graph_min_shared_views must be non-negative.")
    if modal_rigid_lambda < 0:
        raise ValueError("modal_rigid_lambda must be non-negative.")
    if modal_rigid_k <= 0:
        raise ValueError("modal_rigid_k must be positive.")
    if modal_rigid_auto_radius_scale <= 0:
        raise ValueError("modal_rigid_auto_radius_scale must be positive.")
    if modal_rigid_min_shared_views < 0:
        raise ValueError("modal_rigid_min_shared_views must be non-negative.")
    if modal_fill_k <= 0:
        raise ValueError("modal_fill_k must be positive.")
    if modal_fill_auto_radius_scale <= 0:
        raise ValueError("modal_fill_auto_radius_scale must be positive.")
    if modal_fill_anchor_min_observations <= 0:
        raise ValueError("modal_fill_anchor_min_observations must be positive.")
    if modal_fill_ridge_mu < 0:
        raise ValueError("modal_fill_ridge_mu must be non-negative.")
    if graph_smooth_lambda > 0 and single_view_smooth_lambda > 0:
        raise ValueError("Use either graph smoothing or single-view smoothing, not both.")
    if modal_rigid_lambda > 0 and graph_smooth_lambda > 0:
        raise ValueError("Use either graph smoothing or modal rigidity, not both.")
    if modal_rigid_lambda > 0 and single_view_smooth_lambda > 0:
        raise ValueError("Use modal rigidity without single-view smoothing in this v1 solver.")
    if modal_rigid_lambda > 0 and outlier_frac > 0:
        raise ValueError("modal rigidity uses a fixed edge set and requires outlier_frac=0.")
    if modal_fill_unobserved and modal_rigid_lambda > 0:
        raise ValueError("Use modal fill with --modal-rigid-lambda 0 so anchors are solved directly from observations.")

    data = np.load(str(observations_path), allow_pickle=False)
    points = data["points_world"].astype(np.float32)
    obs_point_index = data["obs_point_index"].astype(np.int64)
    obs_view_index = data["obs_view_index"].astype(np.int64)
    obs_pixels_xy = data["obs_pixels_xy"].astype(np.float32)
    obs_y = data["obs_y"].astype(np.complex64)
    obs_J = data["obs_J"].astype(np.float32)
    obs_confidence = data["obs_confidence"].astype(np.float32)
    if "obs_count_per_point" not in data.files:
        raise ValueError("Observation graph is missing obs_count_per_point.")
    obs_count_per_point = data["obs_count_per_point"].astype(np.int32)
    obs_sample_count_per_point = (
        data["obs_sample_count_per_point"].astype(np.int32)
        if "obs_sample_count_per_point" in data.files
        else obs_count_per_point
    )
    view_ids = data["view_ids"]
    num_views = int(view_ids.shape[0])
    if "view_freqs_hz" in data.files:
        alpha_view_freqs_hz = data["view_freqs_hz"].astype(np.float32)
    else:
        alpha_view_freqs_hz = np.full((num_views,), float(np.asarray(data["freq_hz"]).item()), dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points.shape}.")
    if obs_y.ndim != 2 or obs_y.shape[1] != 2:
        raise ValueError(f"obs_y must have shape (O,2), got {obs_y.shape}.")
    if obs_J.shape != (obs_y.shape[0], 2, 3):
        raise ValueError(f"obs_J must have shape (O,2,3), got {obs_J.shape}.")
    if obs_count_per_point.shape != (points.shape[0],):
        raise ValueError(f"obs_count_per_point must have shape ({points.shape[0]},), got {obs_count_per_point.shape}.")
    if obs_sample_count_per_point.shape != (points.shape[0],):
        raise ValueError(
            f"obs_sample_count_per_point must have shape ({points.shape[0]},), "
            f"got {obs_sample_count_per_point.shape}."
        )
    if np.any(obs_point_index < 0) or np.any(obs_point_index >= points.shape[0]):
        raise ValueError("obs_point_index contains invalid point indices.")
    if np.any(obs_view_index < 0) or np.any(obs_view_index >= num_views):
        raise ValueError("obs_view_index contains invalid view indices.")
    if alpha_view_freqs_hz.shape != (num_views,):
        raise ValueError(f"alpha_view_freqs_hz must have shape ({num_views},), got {alpha_view_freqs_hz.shape}.")

    obs_by_point = _observations_by_point(points.shape[0], obs_point_index)
    active = np.ones((points.shape[0],), dtype=bool)
    alphas = np.ones((num_views,), dtype=np.complex64)
    history: list[list[float]] = []
    alpha_history: list[np.ndarray] = []

    phi = np.zeros_like(points, dtype=np.complex64)
    pred_y = np.zeros_like(obs_y, dtype=np.complex64)
    obs_residual = np.zeros((obs_y.shape[0],), dtype=np.float32)
    point_residual = np.zeros((points.shape[0],), dtype=np.float32)
    graph_degree = np.zeros((points.shape[0],), dtype=np.int32)
    graph_smooth_residual = np.zeros((points.shape[0],), dtype=np.float32)
    obs_count_weight_per_point = np.ones((points.shape[0],), dtype=np.float32)
    graph_edge_count = 0
    graph_auto_radius = 0.0
    modal_rigid_degree = np.zeros((points.shape[0],), dtype=np.int32)
    modal_rigid_residual = np.zeros((points.shape[0],), dtype=np.float32)
    modal_rigid_edge_count = 0
    modal_rigid_auto_radius = 0.0
    modal_rigid_edge_a = np.zeros((0,), dtype=np.int64)
    modal_rigid_edge_b = np.zeros((0,), dtype=np.int64)
    modal_rigid_edge_weights = np.zeros((0,), dtype=np.float64)
    modal_fill_target_mask = np.zeros((points.shape[0],), dtype=bool)
    modal_fill_anchor_mask = np.zeros((points.shape[0],), dtype=bool)
    modal_fill_connected_to_anchor = np.zeros((points.shape[0],), dtype=bool)
    modal_fill_degree = np.zeros((points.shape[0],), dtype=np.int32)
    modal_fill_auto_radius = 0.0

    if modal_rigid_lambda > 0:
        point_view_masks = _point_view_masks(points.shape[0], obs_point_index, obs_view_index, num_views)
        (
            modal_rigid_edge_a,
            modal_rigid_edge_b,
            modal_rigid_edge_weights,
            modal_rigid_degree,
            modal_rigid_auto_radius,
        ) = _build_modal_rigid_edges(
            points,
            active,
            point_view_masks,
            modal_rigid_k,
            modal_rigid_auto_radius_scale,
            modal_rigid_min_shared_views,
        )
        if modal_rigid_edge_a.size == 0:
            raise ValueError(
                "Modal rigidity produced no valid local consensus edges; increase "
                "--modal-rigid-auto-radius-scale, increase --modal-rigid-k, lower "
                "--modal-rigid-min-shared-views, or run with --modal-rigid-lambda 0."
            )
        modal_rigid_edge_count = int(modal_rigid_edge_a.size)

    for it in range(iterations):
        if int(active.sum()) < 3:
            raise ValueError("Too few active points remain during optimization.")
        if graph_smooth_lambda > 0:
            phi, graph_degree, obs_count_weight_per_point, graph_edge_a, graph_edge_b, graph_auto_radius = _solve_phi_graph(
                points,
                obs_y,
                obs_J,
                obs_confidence,
                obs_point_index,
                obs_view_index,
                obs_count_per_point,
                active,
                alphas,
                ridge_mu,
                graph_smooth_lambda,
                graph_smooth_k,
                graph_auto_radius_scale,
                graph_min_shared_views,
                obs_count_weight_1,
                obs_count_weight_2,
                obs_count_weight_3plus,
                num_views,
            )
            graph_edge_count = int(graph_edge_a.size)
            graph_smooth_residual = _graph_smooth_residual(phi, graph_edge_a, graph_edge_b, active)
            alpha_confidence = obs_confidence * obs_count_weight_per_point[obs_point_index]
            iteration_regularizer_residual = graph_smooth_residual
        elif modal_rigid_lambda > 0:
            phi = _solve_phi_modal_rigid(
                points,
                obs_y,
                obs_J,
                obs_confidence,
                obs_point_index,
                obs_view_index,
                active,
                alphas,
                ridge_mu,
                modal_rigid_lambda,
                modal_rigid_edge_a,
                modal_rigid_edge_b,
                modal_rigid_edge_weights,
            )
            graph_smooth_residual = np.zeros((points.shape[0],), dtype=np.float32)
            graph_smooth_residual[~active] = np.inf
            modal_rigid_residual = _modal_rigid_residual(phi, modal_rigid_edge_a, modal_rigid_edge_b, active)
            alpha_confidence = obs_confidence
            iteration_regularizer_residual = modal_rigid_residual
        else:
            phi = _solve_phi_points(obs_y, obs_J, obs_confidence, obs_view_index, obs_by_point, active, alphas, ridge_mu)
            graph_smooth_residual = np.zeros((points.shape[0],), dtype=np.float32)
            graph_smooth_residual[~active] = np.inf
            alpha_confidence = obs_confidence
            iteration_regularizer_residual = graph_smooth_residual
        alphas = _solve_alphas(obs_y, obs_J, obs_point_index, obs_view_index, alpha_confidence, active, phi, num_views)
        pred_y = _predict_observations(obs_J, obs_point_index, obs_view_index, phi, alphas)
        obs_residual = _obs_residual(obs_y, pred_y)
        point_residual = _point_residuals(points.shape[0], obs_point_index, obs_residual, active)
        active_residual = point_residual[active]
        active_graph_residual = iteration_regularizer_residual[active]
        history.append(
            [
                float(it),
                float(active.sum()),
                float(active_residual.mean()),
                float(np.median(active_residual)),
                float(active_graph_residual.mean()),
                float(np.median(active_graph_residual)),
            ]
        )
        alpha_history.append(alphas.copy())
        if outlier_frac > 0 and it < iterations - 1:
            active_indices = np.where(active)[0]
            drop_count = int(np.floor(outlier_frac * active_indices.size))
            if drop_count > 0 and active_indices.size - drop_count >= 3:
                drop_indices = active_indices[np.argsort(point_residual[active_indices])[-drop_count:]]
                active[drop_indices] = False

    single_view_refined_mask = np.zeros((points.shape[0],), dtype=bool)
    single_view_anchor_distance = np.full((points.shape[0],), np.inf, dtype=np.float32)
    if single_view_smooth_lambda > 0:
        phi, single_view_refined_mask, single_view_anchor_distance = _refine_single_view_points(
            points,
            phi,
            obs_y,
            obs_J,
            obs_confidence,
            obs_point_index,
            obs_view_index,
            obs_by_point,
            obs_count_per_point,
            active,
            alphas,
            ridge_mu,
            single_view_smooth_lambda,
            single_view_smooth_k,
            single_view_anchor_min_observations,
        )
        pred_y = _predict_observations(obs_J, obs_point_index, obs_view_index, phi, alphas)
        obs_residual = _obs_residual(obs_y, pred_y)
        point_residual = _point_residuals(points.shape[0], obs_point_index, obs_residual, active)

    if modal_fill_unobserved:
        modal_fill_anchor_mask = active & (obs_count_per_point >= int(modal_fill_anchor_min_observations))
        (
            phi,
            modal_fill_target_mask,
            modal_fill_connected_to_anchor,
            modal_fill_degree,
            modal_fill_auto_radius,
        ) = _propagate_modal_motion_to_unobserved(
            points,
            phi,
            obs_count_per_point,
            active,
            modal_fill_k,
            modal_fill_auto_radius_scale,
            modal_fill_anchor_min_observations,
            modal_fill_ridge_mu,
        )
        pred_y = _predict_observations(obs_J, obs_point_index, obs_view_index, phi, alphas)
        obs_residual = _obs_residual(obs_y, pred_y)
        point_residual = _point_residuals(points.shape[0], obs_point_index, obs_residual, active)

    active_indices = np.where(active)[0]
    old_to_new = np.full((points.shape[0],), -1, dtype=np.int64)
    old_to_new[active_indices] = np.arange(active_indices.size, dtype=np.int64)
    keep_obs = active[obs_point_index]
    optional_point_fields = {}
    if "point_source_view_mask" in data.files:
        point_source_view_mask = data["point_source_view_mask"]
        if point_source_view_mask.shape[0] != points.shape[0]:
            raise ValueError("point_source_view_mask does not match points_world length.")
        optional_point_fields["point_source_view_mask"] = point_source_view_mask[active_indices].astype(bool)
    if "point_source_count" in data.files:
        point_source_count = data["point_source_count"]
        if point_source_count.shape[0] != points.shape[0]:
            raise ValueError("point_source_count does not match points_world length.")
        optional_point_fields["point_source_count"] = point_source_count[active_indices].astype(np.int32)
    if "colors" in data.files:
        colors = data["colors"]
        if colors.shape != (points.shape[0], 3):
            raise ValueError("colors must have shape (N,3) and match points_world length.")
        optional_point_fields["colors"] = colors[active_indices].astype(np.uint8)
    if "gaussian_indices" in data.files:
        gaussian_indices = data["gaussian_indices"]
        if gaussian_indices.shape != (points.shape[0],):
            raise ValueError("gaussian_indices must have shape (N,) and match points_world length.")
        optional_point_fields["gaussian_indices"] = gaussian_indices[active_indices].astype(np.int32)
    for scalar_key in ("point_type", "source_checkpoint"):
        if scalar_key in data.files:
            optional_point_fields[scalar_key] = data[scalar_key]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points_world=points[active_indices].astype(np.float32),
        phi=phi[active_indices].astype(np.complex64),
        alphas=alphas.astype(np.complex64),
        alpha_by_view=alphas.astype(np.complex64),
        alpha_semantics=np.array("per_view_per_mode"),
        alpha_reference_view_index=np.array(0, dtype=np.int32),
        alpha_view_freqs_hz=alpha_view_freqs_hz.astype(np.float32),
        view_ids=view_ids,
        freq_hz=data["freq_hz"].astype(np.float32),
        mode_index=data["mode_index"].astype(np.int32),
        point_residual=point_residual[active_indices].astype(np.float32),
        obs_count_per_point=obs_count_per_point[active_indices].astype(np.int32),
        obs_sample_count_per_point=obs_sample_count_per_point[active_indices].astype(np.int32),
        obs_count_weight_per_point=obs_count_weight_per_point[active_indices].astype(np.float32),
        graph_degree=graph_degree[active_indices].astype(np.int32),
        graph_edge_count=np.array(graph_edge_count, dtype=np.int64),
        graph_auto_radius=np.array(graph_auto_radius, dtype=np.float32),
        graph_smooth_residual=graph_smooth_residual[active_indices].astype(np.float32),
        graph_smooth_lambda=np.array(graph_smooth_lambda, dtype=np.float32),
        graph_smooth_k=np.array(graph_smooth_k, dtype=np.int32),
        graph_auto_radius_scale=np.array(graph_auto_radius_scale, dtype=np.float32),
        graph_min_shared_views=np.array(graph_min_shared_views, dtype=np.int32),
        modal_rigid_lambda=np.array(modal_rigid_lambda, dtype=np.float32),
        modal_rigid_k=np.array(modal_rigid_k, dtype=np.int32),
        modal_rigid_auto_radius_scale=np.array(modal_rigid_auto_radius_scale, dtype=np.float32),
        modal_rigid_min_shared_views=np.array(modal_rigid_min_shared_views, dtype=np.int32),
        modal_rigid_edge_count=np.array(modal_rigid_edge_count, dtype=np.int64),
        modal_rigid_degree=modal_rigid_degree[active_indices].astype(np.int32),
        modal_rigid_residual=modal_rigid_residual[active_indices].astype(np.float32),
        modal_rigid_auto_radius=np.array(modal_rigid_auto_radius, dtype=np.float32),
        modal_fill_enabled=np.array(bool(modal_fill_unobserved)),
        modal_fill_k=np.array(modal_fill_k, dtype=np.int32),
        modal_fill_auto_radius_scale=np.array(modal_fill_auto_radius_scale, dtype=np.float32),
        modal_fill_anchor_min_observations=np.array(modal_fill_anchor_min_observations, dtype=np.int32),
        modal_fill_ridge_mu=np.array(modal_fill_ridge_mu, dtype=np.float32),
        modal_fill_auto_radius=np.array(modal_fill_auto_radius, dtype=np.float32),
        modal_fill_anchor_mask=modal_fill_anchor_mask[active_indices].astype(bool),
        modal_fill_target_mask=modal_fill_target_mask[active_indices].astype(bool),
        modal_fill_connected_to_anchor=modal_fill_connected_to_anchor[active_indices].astype(bool),
        modal_fill_degree=modal_fill_degree[active_indices].astype(np.int32),
        modal_fill_anchor_count=np.array(
            int((active & (obs_count_per_point >= int(modal_fill_anchor_min_observations))).sum()) if modal_fill_unobserved else 0,
            dtype=np.int64,
        ),
        modal_fill_target_count=np.array(
            int((active & (obs_count_per_point < int(modal_fill_anchor_min_observations))).sum()) if modal_fill_unobserved else 0,
            dtype=np.int64,
        ),
        modal_fill_filled_count=np.array(int((modal_fill_target_mask & modal_fill_connected_to_anchor).sum()) if modal_fill_unobserved else 0, dtype=np.int64),
        modal_fill_unfilled_count=np.array(int((modal_fill_target_mask & ~modal_fill_connected_to_anchor).sum()) if modal_fill_unobserved else 0, dtype=np.int64),
        obs_count_weight_1=np.array(obs_count_weight_1, dtype=np.float32),
        obs_count_weight_2=np.array(obs_count_weight_2, dtype=np.float32),
        obs_count_weight_3plus=np.array(obs_count_weight_3plus, dtype=np.float32),
        single_view_refined_mask=single_view_refined_mask[active_indices].astype(bool),
        single_view_anchor_distance=single_view_anchor_distance[active_indices].astype(np.float32),
        single_view_smooth_lambda=np.array(single_view_smooth_lambda, dtype=np.float32),
        single_view_smooth_k=np.array(single_view_smooth_k, dtype=np.int32),
        single_view_anchor_min_observations=np.array(single_view_anchor_min_observations, dtype=np.int32),
        obs_point_index=old_to_new[obs_point_index[keep_obs]].astype(np.int32),
        obs_view_index=obs_view_index[keep_obs].astype(np.int32),
        obs_pixels_xy=obs_pixels_xy[keep_obs].astype(np.float32),
        obs_y=obs_y[keep_obs].astype(np.complex64),
        obs_J=obs_J[keep_obs].astype(np.float32),
        obs_confidence=obs_confidence[keep_obs].astype(np.float32),
        obs_pred_y=pred_y[keep_obs].astype(np.complex64),
        obs_residual=obs_residual[keep_obs].astype(np.float32),
        active_indices=active_indices.astype(np.int32),
        optimization_history=np.asarray(history, dtype=np.float32),
        alpha_history=np.asarray(alpha_history, dtype=np.complex64),
        source_observations=np.array(str(observations_path)),
        **optional_point_fields,
    )

    if vis_dir is not None:
        vis = Path(vis_dir)
        vis.mkdir(parents=True, exist_ok=True)
        widths = data["view_image_width"].astype(np.int32)
        heights = data["view_image_height"].astype(np.int32)
        kept_view = obs_view_index[keep_obs]
        kept_pixels = obs_pixels_xy[keep_obs]
        kept_y = obs_y[keep_obs]
        kept_pred = pred_y[keep_obs]
        for view_idx in range(num_views):
            rows = np.where(kept_view == view_idx)[0]
            if rows.size == 0:
                continue
            view_name = str(view_ids[view_idx])
            _scatter_mode_image(
                vis / f"{view_name}_observed.png",
                kept_pixels[rows],
                kept_y[rows],
                int(widths[view_idx]),
                int(heights[view_idx]),
                f"{view_name} observed",
            )
            _scatter_mode_image(
                vis / f"{view_name}_predicted.png",
                kept_pixels[rows],
                kept_pred[rows],
                int(widths[view_idx]),
                int(heights[view_idx]),
                f"{view_name} predicted",
            )
            _scatter_mode_image(
                vis / f"{view_name}_residual.png",
                kept_pixels[rows],
                kept_y[rows] - kept_pred[rows],
                int(widths[view_idx]),
                int(heights[view_idx]),
                f"{view_name} residual",
                cmap="viridis",
                normalize=False,
            )
        _write_ply(vis / "pointcloud_amplitude.ply", points[active_indices], _amplitude_colors(phi[active_indices]))
        _write_ply(vis / "pointcloud_phase_u.ply", points[active_indices], _phase_colors(phi[active_indices]))

    return out
