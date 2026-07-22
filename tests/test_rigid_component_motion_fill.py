from __future__ import annotations

import unittest

import numpy as np

from modal_surface.gaussian_motion_fill import (
    MOTION_FILL_EXCLUDED_NONE,
    MOTION_FILL_ROLE_FIXED_ANCHOR,
    MOTION_FILL_ROLE_FREE_VARIABLE,
    RIGID_SEED_MOTION_FILL_METHOD,
    apply_rigid_seed_motion_fill,
    derive_rigid_seed_motion_fill_roles,
)
from modal_surface.motion_fill import KnnGraph
from modal_surface.optimization_staged import AlphaSyncResult, PreparedObservations


def _graph(edge_index: np.ndarray, num_points: int) -> KnnGraph:
    edges = np.asarray(edge_index, dtype=np.int64)
    degree = np.zeros((num_points,), dtype=np.int32)
    if edges.size:
        np.add.at(degree, edges[:, 0], 1)
        np.add.at(degree, edges[:, 1], 1)
    return KnnGraph(
        edge_index=edges.reshape(-1, 2),
        edge_distance=np.ones((edges.shape[0],), dtype=np.float64),
        edge_weight=np.ones((edges.shape[0],), dtype=np.float64),
        degree=degree,
        component_index=np.zeros((num_points,), dtype=np.int32),
        component_sizes=np.asarray([num_points], dtype=np.int32),
        isolated_mask=degree == 0,
        num_points=num_points,
        k=2,
        max_distance=1.0,
        epsilon=1e-8,
        candidate_directed_count=int(2 * edges.shape[0]),
        retained_directed_count=int(2 * edges.shape[0]),
        pruned_directed_count=0,
    )


def _prepared(num_points: int = 4) -> PreparedObservations:
    points = np.stack(
        [np.arange(num_points), np.zeros(num_points), np.zeros(num_points)],
        axis=1,
    ).astype(np.float32)
    observed_points = np.asarray([0, 1, num_points - 1], dtype=np.int64)
    num_observations = int(observed_points.shape[0])
    jacobian = np.broadcast_to(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (num_observations, 2, 3),
    ).copy()
    measurement = np.zeros((num_observations, 2), dtype=np.complex64)
    view_index = np.zeros((num_observations,), dtype=np.int32)
    weights = np.ones((num_observations,), dtype=np.float32)
    view_count = np.zeros((num_points,), dtype=np.int32)
    sample_count = np.zeros((num_points,), dtype=np.int32)
    view_count[observed_points] = 1
    sample_count[observed_points] = 1
    arrays = {
        "points_world": points,
        "obs_point_index": observed_points,
        "obs_view_index": view_index,
        "obs_J": jacobian,
        "obs_y": measurement,
        "obs_effective_weight": weights,
    }
    return PreparedObservations(
        arrays=arrays,
        points=points,
        obs_point_index=observed_points,
        obs_view_index=view_index,
        obs_pixels_xy=np.zeros((num_observations, 2), dtype=np.float32),
        obs_y=measurement,
        obs_J=jacobian,
        obs_weights=weights,
        obs_count_per_point=view_count,
        obs_sample_count_per_point=sample_count,
        derived_view_count_per_point=view_count.copy(),
        rows_by_point=[
            np.flatnonzero(observed_points == point).astype(np.int64)
            for point in range(num_points)
        ],
        view_ids=np.asarray(["view0"]),
        num_views=1,
    )


def _alpha() -> AlphaSyncResult:
    return AlphaSyncResult(
        alphas=np.asarray([1.0 + 0.0j], dtype=np.complex64),
        identifiable_mask=np.asarray([True]),
        reference_connected_mask=np.asarray([True]),
        exclusion_reason=np.asarray([""]),
        shared_point_count=np.zeros((1, 1), dtype=np.int32),
        edge_point_count=np.zeros((1, 1), dtype=np.int32),
        edge_information=np.zeros((1, 1), dtype=np.float64),
        constraint_count_per_view=np.zeros((1,), dtype=np.int32),
        information_matrix=np.zeros((1, 1), dtype=np.complex128),
        singular_values=np.empty((0,), dtype=np.float64),
        rank_ratio=1.0,
        information_ratio=1.0,
        condition=1.0,
        consistency_residual=0.0,
        phase_std=np.zeros((1,), dtype=np.float64),
        log_gain_std=np.zeros((1,), dtype=np.float64),
        parameter_information=np.empty((0, 0), dtype=np.float64),
        parameter_view_indices=np.empty((0,), dtype=np.int32),
        parameter_order="phase",
        optimizer_success=True,
        optimizer_status=0,
        optimizer_message="",
        gain_bound_active_mask=np.asarray([False]),
        information_kind="phase",
    )


class RigidSeedMotionFillTests(unittest.TestCase):
    def test_roles_are_only_fixed_or_free(self) -> None:
        seed = np.asarray([True, False, False, True])
        roles = derive_rigid_seed_motion_fill_roles(seed)

        np.testing.assert_array_equal(
            roles.role,
            np.asarray(
                [
                    MOTION_FILL_ROLE_FIXED_ANCHOR,
                    MOTION_FILL_ROLE_FREE_VARIABLE,
                    MOTION_FILL_ROLE_FREE_VARIABLE,
                    MOTION_FILL_ROLE_FIXED_ANCHOR,
                ],
                dtype=np.int8,
            ),
        )
        self.assertFalse(np.any(roles.constrained_variable_mask))
        self.assertFalse(np.any(roles.excluded_mask))
        np.testing.assert_array_equal(
            roles.excluded_reason,
            np.full((4,), MOTION_FILL_EXCLUDED_NONE, dtype=np.int8),
        )

    def test_fullspace_fill_preserves_seeds_and_tracks_observability(self) -> None:
        graph = _graph(np.asarray([[0, 1], [1, 2], [2, 3]]), 4)
        seed = np.asarray([True, False, False, True])
        rigid_phi = np.zeros((4, 3), dtype=np.complex64)
        rigid_phi[3] = np.asarray([3.0 + 1.5j, -6.0, 0.75j])

        result = apply_rigid_seed_motion_fill(
            _prepared(), _alpha(), rigid_phi, seed, graph, "motion_fill/graph.npz"
        )

        self.assertEqual(result.diagnostics["method"], RIGID_SEED_MOTION_FILL_METHOD)
        np.testing.assert_array_equal(result.motion.phi[seed], rigid_phi[seed])
        np.testing.assert_array_equal(
            result.numerical_nullity, np.asarray([0, 3, 3, 0], dtype=np.int8)
        )
        np.testing.assert_array_equal(
            result.motion.completion_mask,
            np.asarray([False, True, True, False]),
        )
        np.testing.assert_allclose(result.motion.phi[1], rigid_phi[3] / 3.0)
        np.testing.assert_allclose(result.motion.phi[2], 2.0 * rigid_phi[3] / 3.0)
        np.testing.assert_array_equal(
            result.observed_mask, np.asarray([True, True, False, True])
        )
        np.testing.assert_array_equal(
            result.fill_target_mask, np.asarray([False, True, True, False])
        )
        self.assertEqual(
            result.diagnostics["observability"]["observed_fill_target_count"], 1
        )
        self.assertEqual(
            result.diagnostics["observability"]["unobserved_fill_target_count"], 1
        )
        self.assertEqual(
            result.diagnostics["completion"][
                "completed_observed_fill_target_count"
            ],
            1,
        )
        self.assertEqual(
            result.diagnostics["completion"][
                "completed_unobserved_fill_target_count"
            ],
            1,
        )
        self.assertEqual(result.obs_pred_y.shape, (3, 2))
        self.assertEqual(result.point_residual.shape, (4,))

    def test_seed_free_graph_component_remains_zero_and_unresolved(self) -> None:
        graph = _graph(np.asarray([[0, 1], [2, 3]]), 4)
        seed = np.asarray([True, False, False, False])
        rigid_phi = np.zeros((4, 3), dtype=np.complex64)
        rigid_phi[0] = np.asarray([1.0 + 0.5j, -2.0, 3.0j])

        result = apply_rigid_seed_motion_fill(
            _prepared(), _alpha(), rigid_phi, seed, graph, "motion_fill/graph.npz"
        )

        np.testing.assert_array_equal(
            result.motion.completion_mask,
            np.asarray([False, True, False, False]),
        )
        np.testing.assert_allclose(result.motion.phi[1], rigid_phi[0])
        np.testing.assert_array_equal(
            result.motion.phi[2:], np.zeros((2, 3), dtype=np.complex64)
        )
        self.assertEqual(
            result.diagnostics["connectivity"]["seed_free_component_count"], 1
        )
        self.assertEqual(
            result.diagnostics["completion"][
                "unresolved_observed_fill_target_count"
            ],
            1,
        )
        self.assertEqual(
            result.diagnostics["completion"][
                "unresolved_unobserved_fill_target_count"
            ],
            1,
        )

    def test_rejects_nonzero_nonseed_motion(self) -> None:
        graph = _graph(np.asarray([[0, 1], [1, 2], [2, 3]]), 4)
        seed = np.asarray([True, False, False, True])
        rigid_phi = np.zeros((4, 3), dtype=np.complex64)
        rigid_phi[1, 0] = 1.0

        with self.assertRaisesRegex(ValueError, "exactly zero"):
            apply_rigid_seed_motion_fill(
                _prepared(),
                _alpha(),
                rigid_phi,
                seed,
                graph,
                "motion_fill/graph.npz",
            )


if __name__ == "__main__":
    unittest.main()
