from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from modal_surface.observed_structure_graph import (
    LoadedObservedStructureGraph,
    ObservedStructureGraph,
    ObservedStructureGraphConfig,
    ObservedStructureTopology,
)
from modal_surface.optimization_staged import AlphaSyncResult, prepare_observations
from modal_surface.soft_elastic_solver import (
    SoftElasticSolverConfig,
    solve_soft_elastic,
)


def _alpha() -> AlphaSyncResult:
    return AlphaSyncResult(
        alphas=np.ones((2,), dtype=np.complex128),
        identifiable_mask=np.ones((2,), dtype=bool),
        reference_connected_mask=np.ones((2,), dtype=bool),
        exclusion_reason=np.asarray(["reference", "estimated"]),
        shared_point_count=np.zeros((2, 2), dtype=np.int32),
        edge_point_count=np.zeros((2, 2), dtype=np.int32),
        edge_information=np.zeros((2, 2), dtype=np.float64),
        constraint_count_per_view=np.zeros((2,), dtype=np.int32),
        information_matrix=np.eye(2, dtype=np.complex128),
        singular_values=np.ones((1,), dtype=np.float64),
        rank_ratio=1.0,
        information_ratio=1.0,
        condition=1.0,
        consistency_residual=0.0,
        phase_std=np.zeros((2,), dtype=np.float64),
        log_gain_std=np.zeros((2,), dtype=np.float64),
        parameter_information=np.eye(1, dtype=np.float64),
        parameter_view_indices=np.asarray([1], dtype=np.int32),
        parameter_order="phase",
        optimizer_success=True,
        optimizer_status=1,
        optimizer_message="synthetic",
        gain_bound_active_mask=np.zeros((2,), dtype=bool),
        information_kind="synthetic",
    )


def _graph(points: np.ndarray) -> LoadedObservedStructureGraph:
    node_indices = np.asarray([0, 1, 2], dtype=np.int32)
    node_points = points[node_indices].astype(np.float32)
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
    edge_distance = np.ones((2,), dtype=np.float32)
    edge_weight = np.ones((2,), dtype=np.float32)
    degree = np.asarray([1, 2, 1], dtype=np.int32)
    config = ObservedStructureGraphConfig(
        max_neighbors=2,
        max_distance=2.0,
        min_component_nodes=1,
        min_component_edges=1,
    )
    topology = ObservedStructureTopology(
        node_gaussian_indices=node_indices,
        node_points_world=node_points,
        node_colors_rgb=np.ones((3, 3), dtype=np.float32),
        node_observed_view_mask=np.ones((3, 2), dtype=bool),
        edge_index=edges,
        edge_distance=edge_distance,
        edge_distance_weight=edge_weight,
        edge_color_distance=np.zeros((2,), dtype=np.float32),
        edge_color_weight=edge_weight,
        edge_depth_score=edge_weight,
        edge_combined_weight=edge_weight,
        edge_view_support_mask=np.ones((2, 2), dtype=bool),
        edge_view_support_count=np.full((2,), 2, dtype=np.int32),
        edge_endpoint_gap_by_view=np.zeros((2, 2), dtype=np.float32),
        edge_depth_jump_by_view=np.zeros((2, 2), dtype=np.float32),
        degree=degree,
        component_index=np.zeros((3,), dtype=np.int32),
        component_size=np.asarray([3], dtype=np.int32),
        component_pruned_node_mask=np.zeros((3,), dtype=bool),
        isolated_mask=np.zeros((3,), dtype=bool),
        view_ids=np.asarray(["view1", "view2"]),
        endpoint_gap_median_by_view=np.zeros((2,), dtype=np.float32),
        endpoint_gap_mad_by_view=np.zeros((2,), dtype=np.float32),
        endpoint_gap_threshold_by_view=np.ones((2,), dtype=np.float32),
        depth_jump_median_by_view=np.zeros((2,), dtype=np.float32),
        depth_jump_mad_by_view=np.zeros((2,), dtype=np.float32),
        depth_jump_threshold_by_view=np.ones((2,), dtype=np.float32),
        color_distance_median=0.0,
        color_distance_mad=0.0,
        color_distance_threshold=1.0,
        counts={},
        config=config,
    )
    graph = ObservedStructureGraph(
        node_gaussian_indices=node_indices,
        node_points_world=node_points,
        node_colors_rgb=np.ones((3, 3), dtype=np.float32),
        node_observed_view_mask=np.ones((3, 2), dtype=bool),
        node_observed_view_count=np.full((3,), 2, dtype=np.int32),
        topology=topology,
        counts={},
    )
    return LoadedObservedStructureGraph(
        graph_path=Path("synthetic_observed_graph.npz"),
        source_checkpoint="synthetic.ckpt",
        topology_source_observation_path="synthetic_topology.npz",
        num_foreground_gaussians=points.shape[0],
        view_ids=("view1", "view2"),
        graph=graph,
    )


def _prepared(points: np.ndarray, phi: np.ndarray):
    jacobian_by_view = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    point_index = np.repeat(np.arange(3, dtype=np.int32), 2)
    view_index = np.tile(np.arange(2, dtype=np.int32), 3)
    jacobian = jacobian_by_view[view_index]
    observation = np.einsum(
        "oij,oj->oi", jacobian, phi[point_index]
    ).astype(np.complex64)
    return prepare_observations(
        {
            "points_world": points.astype(np.float32),
            "obs_point_index": point_index,
            "obs_view_index": view_index,
            "obs_pixels_xy": np.zeros((6, 2), dtype=np.float32),
            "obs_y": observation,
            "obs_J": jacobian,
            "obs_contribution_weight": np.ones((6,), dtype=np.float32),
            "obs_count_per_point": np.asarray([2, 2, 2, 0], dtype=np.int32),
            "obs_sample_count_per_point": np.asarray(
                [2, 2, 2, 0], dtype=np.int32
            ),
            "view_ids": np.asarray(["view1", "view2"]),
            "freq_hz": np.array(1.0, dtype=np.float32),
            "mode_index": np.array(0, dtype=np.int32),
        }
    )


class SoftElasticSolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        self.config = SoftElasticSolverConfig(
            stretch_relative=1.0e-8,
            laplacian_relative=1.0e-8,
            lsmr_atol=1.0e-10,
            lsmr_btol=1.0e-10,
            lsmr_conlim=1.0e12,
            lsmr_maxiter=500,
        )

    def test_translation_is_recovered_and_unobserved_point_stays_zero(self) -> None:
        translation = np.asarray([0.3 + 0.2j, -0.1 + 0.4j, 0.2 - 0.3j])
        phi = np.zeros((4, 3), dtype=np.complex64)
        phi[:3] = translation
        result = solve_soft_elastic(
            _prepared(self.points, phi),
            _alpha(),
            _graph(self.points),
            self.config,
        )

        np.testing.assert_allclose(result.phi[:3], phi[:3], atol=1.0e-5)
        np.testing.assert_array_equal(result.phi[3], np.zeros((3,), np.complex64))
        np.testing.assert_allclose(result.edge_axial_real, 0.0, atol=1.0e-6)
        np.testing.assert_allclose(result.edge_axial_imaginary, 0.0, atol=1.0e-6)
        np.testing.assert_allclose(result.laplacian_real, 0.0, atol=1.0e-6)
        np.testing.assert_allclose(result.laplacian_imaginary, 0.0, atol=1.0e-6)

    def test_infinitesimal_rotation_has_zero_axial_stretch(self) -> None:
        phi = np.zeros((4, 3), dtype=np.complex64)
        phi[:3, 1] = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        result = solve_soft_elastic(
            _prepared(self.points, phi),
            _alpha(),
            _graph(self.points),
            self.config,
        )

        np.testing.assert_allclose(result.phi[:3], phi[:3], atol=1.0e-4)
        np.testing.assert_allclose(result.edge_axial_real, 0.0, atol=1.0e-6)

    def test_axial_extension_reports_positive_relative_strain(self) -> None:
        phi = np.zeros((4, 3), dtype=np.complex64)
        phi[:3, 0] = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
        result = solve_soft_elastic(
            _prepared(self.points, phi),
            _alpha(),
            _graph(self.points),
            self.config,
        )

        self.assertTrue(np.all(result.edge_relative_axial_real > 0.09))

    def test_config_rejects_zero_regularization(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot both be zero"):
            SoftElasticSolverConfig(
                stretch_relative=0.0,
                laplacian_relative=0.0,
            ).validate()

    def test_graph_checkpoint_identity_mismatch_fails_fast(self) -> None:
        phi = np.zeros((4, 3), dtype=np.complex64)
        graph = _graph(self.points)
        incompatible = LoadedObservedStructureGraph(
            graph_path=graph.graph_path,
            source_checkpoint=graph.source_checkpoint,
            topology_source_observation_path=graph.topology_source_observation_path,
            num_foreground_gaussians=3,
            view_ids=graph.view_ids,
            graph=graph.graph,
        )
        with self.assertRaisesRegex(ValueError, "foreground count"):
            solve_soft_elastic(
                _prepared(self.points, phi),
                _alpha(),
                incompatible,
                self.config,
            )


if __name__ == "__main__":
    unittest.main()
