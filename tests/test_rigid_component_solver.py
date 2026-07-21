from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast

import numpy as np

from modal_surface.observed_structure_graph import ObservedStructureGraph
from modal_surface.optimization_staged import (
    AlphaSyncResult,
    PreparedObservations,
    prepare_observations,
)
from modal_surface.rigid_component_solver import solve_rigid_components


J_BY_VIEW = (
    np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
)


def _rigid_field(
    points: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    centered = points.astype(np.float64) - points.astype(np.float64).mean(axis=0)
    return (
        translation[None]
        + np.cross(rotation[None], centered).astype(np.complex128)
    ).astype(np.complex64)


def _prepare(
    points: np.ndarray,
    phi: np.ndarray,
    point_views: list[list[int]],
    *,
    noise: dict[tuple[int, int], np.ndarray] | None = None,
) -> PreparedObservations:
    point_indices: list[int] = []
    view_indices: list[int] = []
    obs_y: list[np.ndarray] = []
    obs_J: list[np.ndarray] = []
    for point_idx, views in enumerate(point_views):
        for view_idx in views:
            value = J_BY_VIEW[view_idx] @ phi[point_idx]
            if noise is not None and (point_idx, view_idx) in noise:
                value = value + noise[(point_idx, view_idx)]
            point_indices.append(point_idx)
            view_indices.append(view_idx)
            obs_y.append(value)
            obs_J.append(J_BY_VIEW[view_idx])

    point_index = np.asarray(point_indices, dtype=np.int32)
    view_index = np.asarray(view_indices, dtype=np.int32)
    point_view_mask = np.zeros((points.shape[0], len(J_BY_VIEW)), dtype=bool)
    if point_index.size:
        point_view_mask[point_index, view_index] = True
    return prepare_observations(
        {
            "points_world": np.asarray(points, dtype=np.float32),
            "obs_point_index": point_index,
            "obs_view_index": view_index,
            "obs_pixels_xy": np.zeros((point_index.size, 2), dtype=np.float32),
            "obs_y": np.asarray(obs_y, dtype=np.complex64).reshape(-1, 2),
            "obs_J": np.asarray(obs_J, dtype=np.float32).reshape(-1, 2, 3),
            "obs_contribution_weight": np.ones(
                (point_index.size,), dtype=np.float32
            ),
            "obs_count_per_point": point_view_mask.sum(axis=1).astype(np.int32),
            "obs_sample_count_per_point": np.bincount(
                point_index, minlength=points.shape[0]
            ).astype(np.int32),
            "view_ids": np.asarray(["view0", "view1", "view2"]),
            "view_freqs_hz": np.ones((len(J_BY_VIEW),), dtype=np.float32),
            "freq_hz": np.array(1.0, dtype=np.float32),
            "mode_index": np.array(0, dtype=np.int32),
            "gaussian_indices": np.arange(points.shape[0], dtype=np.int32),
        }
    )


def _alpha(identifiable: np.ndarray | None = None) -> AlphaSyncResult:
    if identifiable is None:
        identifiable = np.ones((len(J_BY_VIEW),), dtype=bool)
    return cast(
        AlphaSyncResult,
        SimpleNamespace(
            alphas=np.ones((len(J_BY_VIEW),), dtype=np.complex64),
            identifiable_mask=np.asarray(identifiable, dtype=bool),
        ),
    )


def _stable_components(
    num_nodes: int,
    edge_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.arange(num_nodes, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for source, target in edge_index.tolist():
        source_root = find(int(source))
        target_root = find(int(target))
        if source_root != target_root:
            parent[max(source_root, target_root)] = min(source_root, target_root)
    roots = np.asarray([find(index) for index in range(num_nodes)], dtype=np.int64)
    _, component_index = np.unique(roots, return_inverse=True)
    component_index = component_index.astype(np.int32)
    return (
        component_index,
        np.bincount(component_index).astype(np.int32),
    )


def _graph(
    prepared: PreparedObservations,
    node_indices: np.ndarray,
    edge_index: np.ndarray,
    *,
    edge_weight: float = 1.0,
) -> ObservedStructureGraph:
    nodes = np.asarray(node_indices, dtype=np.int32)
    edges = np.asarray(edge_index, dtype=np.int32).reshape(-1, 2)
    if edges.shape[0] > 1:
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges = edges[order]
    degree = np.zeros((nodes.shape[0],), dtype=np.int32)
    if edges.size:
        np.add.at(degree, edges[:, 0], 1)
        np.add.at(degree, edges[:, 1], 1)
    component_index, component_size = _stable_components(nodes.shape[0], edges)
    node_view_mask = np.zeros(
        (nodes.shape[0], prepared.num_views), dtype=bool
    )
    point_view_mask = np.zeros(
        (prepared.points.shape[0], prepared.num_views), dtype=bool
    )
    positive = prepared.obs_weights > 0.0
    point_view_mask[
        prepared.obs_point_index[positive], prepared.obs_view_index[positive]
    ] = True
    node_view_mask[:] = point_view_mask[nodes]
    topology = SimpleNamespace(
        edge_index=edges,
        edge_combined_weight=np.full(
            (edges.shape[0],), edge_weight, dtype=np.float32
        ),
        degree=degree,
        component_index=component_index,
        component_size=component_size,
        isolated_mask=degree == 0,
    )
    return cast(
        ObservedStructureGraph,
        SimpleNamespace(
            node_gaussian_indices=nodes,
            node_points_world=prepared.points[nodes].copy(),
            node_observed_view_mask=node_view_mask,
            node_observed_view_count=node_view_mask.sum(axis=1).astype(np.int32),
            topology=topology,
        ),
    )


class RigidComponentSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [3.0, 2.0, 4.0],
            ],
            dtype=np.float32,
        )
        self.translation = np.asarray(
            [0.15 + 0.03j, -0.08 + 0.01j, 0.04 - 0.02j],
            dtype=np.complex128,
        )
        self.rotation = np.asarray(
            [0.02 - 0.01j, -0.015 + 0.005j, 0.01 + 0.012j],
            dtype=np.complex128,
        )
        self.component_phi = _rigid_field(
            self.points[:4], self.translation, self.rotation
        )
        self.phi = np.vstack(
            [self.component_phi, np.zeros((1, 3), dtype=np.complex64)]
        )
        self.connected_edges = np.asarray(
            [[0, 1], [0, 2], [0, 3]], dtype=np.int32
        )

    def test_full_rank_component_recovers_complex_twist_and_rigid_seeds(self) -> None:
        prepared = _prepare(
            self.points,
            self.phi,
            [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], []],
        )
        graph = _graph(prepared, np.arange(4), self.connected_edges)
        result = solve_rigid_components(prepared, _alpha(), graph)

        np.testing.assert_allclose(
            result.phi[:4], self.component_phi, atol=2.0e-6, rtol=2.0e-6
        )
        np.testing.assert_allclose(
            result.component_translation[0],
            self.translation,
            atol=2.0e-6,
            rtol=2.0e-6,
        )
        np.testing.assert_allclose(
            result.component_rotation[0],
            self.rotation,
            atol=2.0e-6,
            rtol=2.0e-6,
        )
        np.testing.assert_array_equal(
            result.rigid_seed_mask, [True, True, True, True, False]
        )
        np.testing.assert_array_equal(
            result.fill_target_mask, [False, False, False, False, True]
        )
        np.testing.assert_array_equal(
            result.point_component_index, [0, 0, 0, 0, -1]
        )
        self.assertEqual(int(result.component_rank[0]), 6)
        self.assertLess(
            float(result.edge_first_order_relative_real.max()), 1.0e-6
        )
        self.assertLess(
            float(result.edge_first_order_relative_imag.max()), 1.0e-6
        )
        self.assertEqual(result.phase_angles.shape, (64,))

    def test_single_view_node_is_solved_through_its_component(self) -> None:
        prepared = _prepare(
            self.points[:4],
            self.component_phi,
            [[0], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
        )
        graph = _graph(prepared, np.arange(4), self.connected_edges)
        result = solve_rigid_components(prepared, _alpha(), graph)
        np.testing.assert_allclose(
            result.phi,
            self.component_phi,
            atol=2.0e-6,
            rtol=2.0e-6,
        )
        self.assertTrue(result.rigid_seed_mask[0])
        self.assertEqual(int(prepared.derived_view_count_per_point[0]), 1)

    def test_rank_deficient_component_uses_deterministic_minimum_norm(self) -> None:
        prepared = _prepare(
            self.points[:4],
            self.component_phi,
            [[0], [0], [0], [0]],
        )
        graph = _graph(prepared, np.arange(4), self.connected_edges)
        first = solve_rigid_components(prepared, _alpha(), graph)
        second = solve_rigid_components(prepared, _alpha(), graph)

        self.assertLess(int(first.component_rank[0]), 6)
        self.assertTrue(np.isinf(first.component_condition[0]))
        self.assertTrue(np.all(first.rigid_seed_mask))
        np.testing.assert_array_equal(first.phi, second.phi)
        np.testing.assert_array_equal(
            first.component_translation, second.component_translation
        )
        np.testing.assert_array_equal(
            first.component_rotation, second.component_rotation
        )

    def test_inconsistent_observation_is_diagnostic_only_not_seed_rejection(self) -> None:
        prepared = _prepare(
            self.points[:4],
            self.component_phi,
            [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
            noise={
                (3, 2): np.asarray([0.4 + 0.2j, -0.3 + 0.1j], dtype=np.complex64)
            },
        )
        graph = _graph(prepared, np.arange(4), self.connected_edges)
        result = solve_rigid_components(prepared, _alpha(), graph)

        self.assertTrue(np.all(result.rigid_seed_mask))
        self.assertGreater(
            float(result.component_normalized_weighted_residual[0]), 0.0
        )
        self.assertLess(
            float(result.edge_first_order_relative_real.max()), 1.0e-6
        )
        self.assertLess(
            float(result.edge_first_order_relative_imag.max()), 1.0e-6
        )

    def test_component_without_usable_alpha_row_fails_fast(self) -> None:
        prepared = _prepare(
            self.points[:4],
            self.component_phi,
            [[0], [0], [0], [0]],
        )
        graph = _graph(prepared, np.arange(4), self.connected_edges)
        with self.assertRaisesRegex(ValueError, "no positive-weight row"):
            solve_rigid_components(
                prepared,
                _alpha(np.asarray([False, False, False])),
                graph,
            )

    def test_isolated_observed_node_is_a_free_fill_target(self) -> None:
        prepared = _prepare(
            self.points[:3],
            self.component_phi[:3],
            [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        )
        graph = _graph(
            prepared,
            np.arange(3),
            np.asarray([[0, 1]], dtype=np.int32),
        )
        result = solve_rigid_components(prepared, _alpha(), graph)

        np.testing.assert_array_equal(result.observed_mask, [True, True, True])
        np.testing.assert_array_equal(result.rigid_seed_mask, [True, True, False])
        np.testing.assert_array_equal(result.fill_target_mask, [False, False, True])
        np.testing.assert_array_equal(result.point_component_index, [0, 0, -1])
        np.testing.assert_array_equal(
            result.phi[2], np.zeros((3,), dtype=np.complex64)
        )

    def test_graph_edge_weights_do_not_soften_hard_component_rigidity(self) -> None:
        prepared = _prepare(
            self.points[:4],
            self.component_phi,
            [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
        )
        unit_graph = _graph(
            prepared, np.arange(4), self.connected_edges, edge_weight=1.0
        )
        weak_graph = _graph(
            prepared, np.arange(4), self.connected_edges, edge_weight=1.0e-6
        )
        unit = solve_rigid_components(prepared, _alpha(), unit_graph)
        weak = solve_rigid_components(prepared, _alpha(), weak_graph)
        np.testing.assert_array_equal(unit.phi, weak.phi)

    def test_finite_playback_edge_drift_is_quadratic_at_small_amplitude(self) -> None:
        translation = np.zeros((3,), dtype=np.complex128)
        full_rotation = np.asarray([0.0, 0.0, 0.02], dtype=np.complex128)
        half_rotation = 0.5 * full_rotation
        full_phi = _rigid_field(self.points[:4], translation, full_rotation)
        half_phi = _rigid_field(self.points[:4], translation, half_rotation)
        full_prepared = _prepare(
            self.points[:4],
            full_phi,
            [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
        )
        half_prepared = _prepare(
            self.points[:4],
            half_phi,
            [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
        )
        full = solve_rigid_components(
            full_prepared,
            _alpha(),
            _graph(full_prepared, np.arange(4), self.connected_edges),
        )
        half = solve_rigid_components(
            half_prepared,
            _alpha(),
            _graph(half_prepared, np.arange(4), self.connected_edges),
        )
        ratio = float(
            half.edge_finite_drift_max.max()
            / full.edge_finite_drift_max.max()
        )
        self.assertAlmostEqual(ratio, 0.25, delta=0.01)


if __name__ == "__main__":
    unittest.main()
