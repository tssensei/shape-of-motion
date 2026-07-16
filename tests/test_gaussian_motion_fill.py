from __future__ import annotations

import unittest

import numpy as np

from modal_surface.gaussian_motion_fill import (
    MOTION_FILL_EXCLUDED_ALPHA_UNRESOLVED,
    MOTION_FILL_EXCLUDED_FULL_RANK_NONANCHOR,
    MOTION_FILL_EXCLUDED_NONE,
    MOTION_FILL_EXCLUDED_NO_USABLE_OBSERVATION,
    MOTION_FILL_EXCLUDED_REJECTED,
    MOTION_FILL_ROLE_CONSTRAINED_VARIABLE,
    MOTION_FILL_ROLE_EXCLUDED,
    MOTION_FILL_ROLE_FIXED_ANCHOR,
    MOTION_FILL_ROLE_FREE_VARIABLE,
    apply_gaussian_motion_fill,
    derive_gaussian_motion_fill_roles,
    weight_gaussian_motion_fill_graph,
)
from modal_surface.motion_fill import (
    build_knn_graph,
    fill_nullspace_motion,
    query_knn_candidates,
)
from modal_surface.optimization_staged import (
    POINT_STATUS_COMPLETED_OBSERVED,
    POINT_STATUS_COMPLETED_UNOBSERVED,
    POINT_STATUS_PARTIAL_UNRESOLVED,
    POINT_STATUS_REJECTED_UNRESOLVED,
    AlphaSyncResult,
    ObservableSolveResult,
    PreparedObservations,
    StagedSolveResult,
    StagedSolverConfig,
)


def _mask(indices: list[int], count: int) -> np.ndarray:
    result = np.zeros((count,), dtype=bool)
    result[indices] = True
    return result


def _observable_from_arrays(arrays: dict[str, np.ndarray]) -> ObservableSolveResult:
    num_points = int(arrays["phi_observable"].shape[0])
    nullity = arrays["point_nullity"]
    return ObservableSolveResult(
        phi=arrays["phi"],
        phi_observable=arrays["phi_observable"],
        nullspace_basis=arrays["point_nullspace_basis"],
        nullity=nullity,
        singular_values=np.zeros((num_points, 3), dtype=np.float32),
        observable_rank=(3 - nullity).astype(np.int8),
        condition=np.ones((num_points,), dtype=np.float32),
        distinct_valid_view_count=np.asarray(
            arrays.get("obs_count_per_point", np.zeros((num_points,), dtype=np.int32))
        ),
        usable_observation_row_count=np.asarray(
            arrays.get(
                "obs_sample_count_per_point", np.zeros((num_points,), dtype=np.int32)
            )
        ),
        precompletion_residual=np.zeros((num_points,), dtype=np.float32),
        anchor_residual_threshold=0.1,
        anchor_mask=arrays["anchor_mask"],
        partial_mask=arrays["partial_mask"],
        rejected_mask=arrays["rejected_mask"],
        unobserved_mask=arrays["unobserved_mask"],
        alpha_unresolved_mask=arrays["alpha_unresolved_mask"],
        no_usable_observation_mask=arrays["no_usable_observation_mask"],
        point_status=arrays["point_solution_status"],
    )


def _role_case() -> tuple[ObservableSolveResult, np.ndarray]:
    count = 7
    nullity = np.asarray([0, 1, 3, 1, 3, 3, 0], dtype=np.int8)
    phi = np.zeros((count, 3), dtype=np.complex64)
    arrays = {
        "phi": phi,
        "phi_observable": phi,
        "point_nullspace_basis": np.zeros((count, 3, 3), dtype=np.complex64),
        "anchor_mask": _mask([0], count),
        "partial_mask": _mask([1, 6], count),
        "unobserved_mask": _mask([2], count),
        "rejected_mask": _mask([3], count),
        "alpha_unresolved_mask": _mask([4], count),
        "no_usable_observation_mask": _mask([5], count),
        "point_nullity": nullity,
        "point_solution_status": np.arange(count, dtype=np.int8),
    }
    return _observable_from_arrays(arrays), nullity


def _formal_staged_case() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
        dtype=np.float32,
    )
    target = np.asarray(
        [1.0 + 0.25j, -2.0 + 0.5j, 3.0 - 0.75j], dtype=np.complex64
    )
    excluded_value = np.asarray([50.0 + 1.0j, -60.0, 70.0j], dtype=np.complex64)
    phi_observable = np.asarray(
        [target, [target[0], target[1], 0.0], np.zeros(3), excluded_value],
        dtype=np.complex64,
    )
    basis = np.zeros((4, 3, 3), dtype=np.complex64)
    basis[1, 2, 0] = 1.0
    basis[2] = np.eye(3, dtype=np.complex64)
    alphas = np.asarray(
        [1.0 + 0.0j, np.exp(0.25j), np.exp(-0.4j)], dtype=np.complex64
    )
    jacobian_xy = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    jacobian_yz = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    jacobian_rotated_xy = np.asarray(
        [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]], dtype=np.float32
    )
    jacobian_excluded_view = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )
    obs_point_index = np.asarray([0, 0, 1, 1, 1, 3], dtype=np.int32)
    obs_view_index = np.asarray([0, 1, 0, 0, 2, 0], dtype=np.int32)
    obs_J = np.asarray(
        [
            jacobian_xy,
            jacobian_yz,
            jacobian_xy,
            jacobian_rotated_xy,
            jacobian_excluded_view,
            jacobian_xy,
        ],
        dtype=np.float32,
    )
    obs_y = np.asarray(
        [
            alphas[0] * (jacobian_xy @ target),
            alphas[1] * (jacobian_yz @ target),
            alphas[0] * (jacobian_xy @ target),
            alphas[0] * (jacobian_rotated_xy @ target),
            [123.0 + 4.0j, -321.0 + 2.0j],
            [100.0 + 0.0j, -100.0 + 0.0j],
        ],
        dtype=np.complex64,
    )
    false_mask = np.zeros((4,), dtype=bool)
    arrays = {
        "points_world": points,
        "phi": phi_observable.copy(),
        "phi_observable": phi_observable.copy(),
        "point_nullspace_basis": basis,
        "point_nullity": np.asarray([0, 1, 3, 0], dtype=np.int8),
        "anchor_mask": _mask([0], 4),
        "partial_mask": _mask([1], 4),
        "unobserved_mask": _mask([2], 4),
        "rejected_mask": _mask([3], 4),
        "alpha_unresolved_mask": false_mask.copy(),
        "no_usable_observation_mask": false_mask.copy(),
        "point_solution_status": np.asarray([0, 3, 4, 5], dtype=np.int8),
        "obs_point_index": obs_point_index,
        "obs_view_index": obs_view_index,
        "obs_J": obs_J,
        "obs_y": obs_y,
        "obs_effective_weight": np.asarray([1.0, 1.0, 0.5, 0.5, 1.0, 1.0], dtype=np.float32),
        "obs_pred_y": np.zeros_like(obs_y),
        "obs_residual": np.zeros((obs_y.shape[0],), dtype=np.float32),
        "obs_residual_valid_mask": np.ones((obs_y.shape[0],), dtype=bool),
        "point_residual": np.zeros((4,), dtype=np.float32),
        "point_residual_valid_mask": np.ones((4,), dtype=bool),
        "alphas": alphas,
        "alpha_identifiable_mask": np.asarray([True, True, False]),
        "obs_count_per_point": np.asarray([2, 2, 0, 1], dtype=np.int32),
        "obs_sample_count_per_point": np.asarray([2, 3, 0, 1], dtype=np.int32),
        "freq_hz": np.array(2.5, dtype=np.float32),
    }
    return arrays, target, excluded_value


def _staged_from_arrays(arrays: dict[str, np.ndarray]) -> StagedSolveResult:
    points = arrays["points_world"]
    num_points = int(points.shape[0])
    point_index = arrays["obs_point_index"].astype(np.int64)
    view_index = arrays["obs_view_index"].astype(np.int32)
    num_views = int(arrays["alphas"].shape[0])
    rows_by_point = [
        np.where(point_index == point)[0].astype(np.int64)
        for point in range(num_points)
    ]
    prepared = PreparedObservations(
        arrays=arrays,
        points=points,
        obs_point_index=point_index,
        obs_view_index=view_index,
        obs_pixels_xy=np.zeros((point_index.size, 2), dtype=np.float32),
        obs_y=arrays["obs_y"],
        obs_J=arrays["obs_J"],
        obs_weights=arrays["obs_effective_weight"],
        obs_count_per_point=arrays["obs_count_per_point"],
        obs_sample_count_per_point=arrays["obs_sample_count_per_point"],
        derived_view_count_per_point=arrays["obs_count_per_point"],
        rows_by_point=rows_by_point,
        view_ids=np.asarray([f"view_{view}" for view in range(num_views)]),
        num_views=num_views,
    )
    identifiable = arrays["alpha_identifiable_mask"]
    alpha = AlphaSyncResult(
        alphas=arrays["alphas"],
        identifiable_mask=identifiable,
        reference_connected_mask=identifiable.copy(),
        exclusion_reason=np.full((num_views,), "", dtype="<U32"),
        shared_point_count=np.zeros((num_views, num_views), dtype=np.int32),
        edge_point_count=np.zeros((num_views, num_views), dtype=np.int32),
        edge_information=np.zeros((num_views, num_views), dtype=np.float64),
        constraint_count_per_view=np.zeros((num_views,), dtype=np.int32),
        information_matrix=np.zeros((num_views, num_views), dtype=np.complex128),
        singular_values=np.empty((0,), dtype=np.float64),
        rank_ratio=1.0,
        information_ratio=1.0,
        condition=1.0,
        consistency_residual=0.0,
        phase_std=np.zeros((num_views,), dtype=np.float64),
        log_gain_std=np.zeros((num_views,), dtype=np.float64),
        parameter_information=np.empty((0, 0), dtype=np.float64),
        parameter_view_indices=np.empty((0,), dtype=np.int32),
        parameter_order="phase",
        optimizer_success=True,
        optimizer_status=1,
        optimizer_message="test",
        gain_bound_active_mask=np.zeros((num_views,), dtype=bool),
        information_kind="test",
    )
    observable = _observable_from_arrays(arrays)
    return StagedSolveResult(
        config=StagedSolverConfig(),
        prepared=prepared,
        alpha=alpha,
        observable=observable,
        alpha_view_freqs_hz=np.full(
            (num_views,), float(np.asarray(arrays["freq_hz"]).item()), dtype=np.float32
        ),
        obs_pred_y=arrays["obs_pred_y"],
        obs_residual=arrays["obs_residual"],
        obs_residual_valid_mask=arrays["obs_residual_valid_mask"],
        point_residual=arrays["point_residual"],
        point_residual_valid_mask=arrays["point_residual_valid_mask"],
        unidentifiable_observed_view_indices=np.asarray([2], dtype=np.int32),
    )


class GaussianMotionFillTests(unittest.TestCase):
    def test_rgb_affinity_reweights_exactly_without_changing_topology(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float32,
        )
        graph = build_knn_graph(
            query_knn_candidates(points, max_k=2),
            k=2,
            max_distance=3.0,
        )
        colors = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        sigma = 2.0

        weighted = weight_gaussian_motion_fill_graph(
            graph,
            colors,
            method="spatial-rgb",
            color_sigma=sigma,
        )
        color_delta = (
            colors[graph.edge_index[:, 0]].astype(np.float64)
            - colors[graph.edge_index[:, 1]].astype(np.float64)
        )
        expected_affinity = np.exp(
            -np.sum(color_delta * color_delta, axis=1) / (2.0 * sigma * sigma)
        )
        np.testing.assert_allclose(
            weighted.graph.edge_weight,
            graph.edge_weight * expected_affinity,
            rtol=1e-12,
            atol=0.0,
        )
        self.assertEqual(weighted.edge_weighting, "spatial-rgb")
        self.assertEqual(weighted.color_metric, "activated_rgb_l2")
        self.assertEqual(weighted.resolved_color_sigma, sigma)
        self.assertEqual(weighted.color_sigma_source, "explicit")

        for field in (
            "edge_index",
            "edge_distance",
            "degree",
            "component_index",
            "component_sizes",
            "isolated_mask",
        ):
            np.testing.assert_array_equal(
                getattr(weighted.graph, field), getattr(graph, field)
            )
        for field in (
            "num_points",
            "k",
            "max_distance",
            "epsilon",
            "candidate_directed_count",
            "retained_directed_count",
            "pruned_directed_count",
        ):
            self.assertEqual(getattr(weighted.graph, field), getattr(graph, field))

    def test_same_color_rgb_weighting_matches_spatial_weights(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        graph = build_knn_graph(
            query_knn_candidates(points, max_k=2),
            k=2,
            max_distance=1.1,
        )
        colors = np.full((points.shape[0], 3), 0.4, dtype=np.float32)

        spatial = weight_gaussian_motion_fill_graph(
            graph,
            colors,
            method="spatial",
            color_sigma=None,
        )
        rgb = weight_gaussian_motion_fill_graph(
            graph,
            colors,
            method="spatial-rgb",
            color_sigma=0.25,
        )

        np.testing.assert_array_equal(spatial.graph.edge_weight, graph.edge_weight)
        np.testing.assert_array_equal(rgb.graph.edge_weight, graph.edge_weight)
        self.assertEqual(spatial.edge_weighting, "spatial")
        self.assertEqual(spatial.color_metric, "none")
        self.assertIsNone(spatial.resolved_color_sigma)
        self.assertEqual(spatial.color_sigma_source, "none")

    def test_auto_color_sigma_uses_edge_median_and_rejects_degenerate_graphs(
        self,
    ) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float32,
        )
        candidates = query_knn_candidates(points, max_k=2)
        graph = build_knn_graph(candidates, k=2, max_distance=3.0)
        colors = np.asarray(
            [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
            dtype=np.float32,
        )

        weighted = weight_gaussian_motion_fill_graph(
            graph,
            colors,
            method="spatial-rgb",
            color_sigma=None,
        )

        edge_color_distance = np.linalg.norm(
            colors[graph.edge_index[:, 0]].astype(np.float64)
            - colors[graph.edge_index[:, 1]].astype(np.float64),
            axis=1,
        )
        self.assertEqual(
            weighted.resolved_color_sigma,
            float(np.median(edge_color_distance)),
        )
        self.assertEqual(weighted.color_sigma_source, "auto_edge_median")

        empty_graph = build_knn_graph(candidates, k=2, max_distance=0.1)
        with self.assertRaises(ValueError):
            weight_gaussian_motion_fill_graph(
                empty_graph,
                colors,
                method="spatial-rgb",
                color_sigma=None,
            )
        with self.assertRaises(ValueError):
            weight_gaussian_motion_fill_graph(
                graph,
                np.zeros_like(colors),
                method="spatial-rgb",
                color_sigma=None,
            )

    def test_rgb_weights_resolve_conflicting_anchors_without_moving_them(
        self,
    ) -> None:
        points = np.asarray(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        graph = build_knn_graph(
            query_knn_candidates(points, max_k=2),
            k=2,
            max_distance=1.1,
        )
        colors = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        spatial_graph = weight_gaussian_motion_fill_graph(
            graph,
            colors,
            method="spatial",
            color_sigma=None,
        ).graph
        rgb_graph = weight_gaussian_motion_fill_graph(
            graph,
            colors,
            method="spatial-rgb",
            color_sigma=0.25,
        ).graph
        phi_observable = np.asarray(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.complex64,
        )
        basis = np.zeros((3, 3, 3), dtype=np.complex64)
        basis[2] = np.eye(3, dtype=np.complex64)
        nullity = np.asarray([0, 0, 3], dtype=np.int8)
        anchors = np.asarray([True, True, False])
        partial = np.zeros((3,), dtype=bool)
        unobserved = np.asarray([False, False, True])

        spatial_fill = fill_nullspace_motion(
            spatial_graph,
            phi_observable,
            basis,
            nullity,
            anchors,
            partial,
            unobserved,
        )
        rgb_fill = fill_nullspace_motion(
            rgb_graph,
            phi_observable,
            basis,
            nullity,
            anchors,
            partial,
            unobserved,
        )

        np.testing.assert_array_equal(spatial_fill.phi[:2], phi_observable[:2])
        np.testing.assert_array_equal(rgb_fill.phi[:2], phi_observable[:2])
        self.assertAlmostEqual(float(spatial_fill.phi[2, 0].real), 5.0, places=5)
        self.assertLess(float(abs(rgb_fill.phi[2, 0])), 1e-4)
        self.assertLess(
            float(abs(rgb_fill.phi[2, 0])),
            float(abs(spatial_fill.phi[2, 0])),
        )

    def test_staged_states_map_to_four_explicit_roles(self) -> None:
        observable, nullity = _role_case()
        roles = derive_gaussian_motion_fill_roles(observable, nullity)

        np.testing.assert_array_equal(
            roles.role,
            [
                MOTION_FILL_ROLE_FIXED_ANCHOR,
                MOTION_FILL_ROLE_CONSTRAINED_VARIABLE,
                MOTION_FILL_ROLE_FREE_VARIABLE,
                MOTION_FILL_ROLE_EXCLUDED,
                MOTION_FILL_ROLE_EXCLUDED,
                MOTION_FILL_ROLE_EXCLUDED,
                MOTION_FILL_ROLE_EXCLUDED,
            ],
        )
        np.testing.assert_array_equal(
            roles.excluded_reason,
            [
                MOTION_FILL_EXCLUDED_NONE,
                MOTION_FILL_EXCLUDED_NONE,
                MOTION_FILL_EXCLUDED_NONE,
                MOTION_FILL_EXCLUDED_REJECTED,
                MOTION_FILL_EXCLUDED_ALPHA_UNRESOLVED,
                MOTION_FILL_EXCLUDED_NO_USABLE_OBSERVATION,
                MOTION_FILL_EXCLUDED_FULL_RANK_NONANCHOR,
            ],
        )

    def test_fill_uses_in_memory_rows_and_quarantines_excluded_points(self) -> None:
        arrays, target, excluded_value = _formal_staged_case()
        points = arrays["points_world"]
        candidates = query_knn_candidates(points, max_k=1)
        graph = build_knn_graph(candidates, k=1, max_distance=0.11)
        filled = apply_gaussian_motion_fill(
            _staged_from_arrays(arrays),
            graph,
            "motion_fill/graph.npz",
        )

        np.testing.assert_allclose(
            filled.motion.phi[0:3],
            np.broadcast_to(target, (3, 3)),
            atol=1e-5,
        )
        np.testing.assert_array_equal(filled.motion.phi[3], excluded_value)
        np.testing.assert_array_equal(
            filled.motion.phi_nullspace_correction[3],
            np.zeros((3,), dtype=np.complex64),
        )
        np.testing.assert_array_equal(
            filled.motion.completion_mask, [False, True, True, False]
        )
        np.testing.assert_array_equal(
            filled.point_solution_status,
            [
                0,
                POINT_STATUS_COMPLETED_OBSERVED,
                POINT_STATUS_COMPLETED_UNOBSERVED,
                POINT_STATUS_REJECTED_UNRESOLVED,
            ],
        )
        np.testing.assert_array_equal(filled.roles.role, [0, 1, 2, 3])
        np.testing.assert_array_equal(
            filled.motion.connectivity.component_index, [0, 0, 0, -1]
        )
        np.testing.assert_array_equal(
            filled.motion.connectivity.hop_distance, [0, 1, 2, -1]
        )
        self.assertFalse(bool(filled.obs_residual_valid_mask[4]))
        self.assertTrue(np.isnan(filled.obs_pred_y[4].real).all())
        self.assertEqual(filled.diagnostics["completion_count"], 2)
        self.assertEqual(filled.diagnostics["role_counts"]["excluded"], 1)
        self.assertEqual(filled.diagnostics["system"]["eligible_edge_count"], 2)
        self.assertEqual(filled.diagnostics["version"], 2)
        self.assertEqual(filled.diagnostics["tolerances"]["lsmr_atol"], 1e-8)
        self.assertEqual(
            filled.diagnostics["tolerances"]["lsmr_maxiter"],
            filled.motion.lsmr_maxiter,
        )
        self.assertEqual(filled.diagnostics["solver_scope"], "componentwise")
        self.assertTrue(filled.diagnostics["parallel_channels"])
        self.assertEqual(len(filled.diagnostics["components"]), 1)
        self.assertEqual(
            filled.diagnostics["lsmr_real"]["iterations_max"],
            filled.motion.real_solver.iterations_max,
        )

    def test_weak_full_rank_partial_is_excluded_without_blocking_exact_fill(self) -> None:
        arrays, target, excluded_value = _formal_staged_case()
        weak_value = excluded_value.copy()
        weak_value[2] = 0.0
        arrays["phi_observable"][3] = weak_value
        arrays["phi"][3] = weak_value
        arrays["rejected_mask"][3] = False
        arrays["partial_mask"][3] = True
        arrays["point_nullity"][3] = 1
        arrays["point_nullspace_basis"][3, 2, 0] = 1.0
        arrays["point_solution_status"][3] = POINT_STATUS_PARTIAL_UNRESOLVED
        arrays["obs_y"][5] = arrays["alphas"][0] * (
            arrays["obs_J"][5] @ weak_value
        )
        weak_jacobian = np.asarray(
            [[0.0, 0.0, 0.008], [0.0, 0.0, 0.0]], dtype=np.float32
        )
        arrays["obs_point_index"] = np.append(arrays["obs_point_index"], 3).astype(
            np.int32
        )
        arrays["obs_view_index"] = np.append(arrays["obs_view_index"], 1).astype(
            np.int32
        )
        arrays["obs_J"] = np.concatenate(
            [arrays["obs_J"], weak_jacobian[None]], axis=0
        )
        arrays["obs_y"] = np.concatenate(
            [
                arrays["obs_y"],
                (
                    arrays["alphas"][1] * (weak_jacobian @ weak_value)
                )[None].astype(np.complex64),
            ],
            axis=0,
        )
        arrays["obs_effective_weight"] = np.append(
            arrays["obs_effective_weight"], np.float32(1.0)
        )
        arrays["obs_pred_y"] = np.concatenate(
            [arrays["obs_pred_y"], np.zeros((1, 2), dtype=np.complex64)], axis=0
        )
        arrays["obs_residual"] = np.append(
            arrays["obs_residual"], np.float32(0.0)
        )
        arrays["obs_residual_valid_mask"] = np.append(
            arrays["obs_residual_valid_mask"], True
        )
        arrays["obs_count_per_point"][3] = 2
        arrays["obs_sample_count_per_point"][3] = 2
        graph = build_knn_graph(
            query_knn_candidates(arrays["points_world"], max_k=1),
            k=1,
            max_distance=0.11,
        )
        staged = _staged_from_arrays(arrays)
        filled = apply_gaussian_motion_fill(
            staged,
            graph,
            "motion_fill/graph.npz",
        )

        np.testing.assert_allclose(
            filled.motion.phi[0:3],
            np.broadcast_to(target, (3, 3)),
            atol=1e-5,
        )
        np.testing.assert_array_equal(filled.motion.phi[3], weak_value)
        np.testing.assert_array_equal(
            filled.motion.phi_nullspace_correction[3],
            np.zeros((3,), dtype=np.complex64),
        )
        np.testing.assert_array_equal(
            filled.motion.completion_mask, [False, True, True, False]
        )
        np.testing.assert_array_equal(filled.roles.role, [0, 1, 2, 3])
        np.testing.assert_array_equal(
            filled.roles.excluded_reason,
            [
                MOTION_FILL_EXCLUDED_NONE,
                MOTION_FILL_EXCLUDED_NONE,
                MOTION_FILL_EXCLUDED_NONE,
                MOTION_FILL_EXCLUDED_FULL_RANK_NONANCHOR,
            ],
        )
        np.testing.assert_array_equal(staged.observable.nullity, [0, 1, 3, 1])
        np.testing.assert_array_equal(filled.numerical_nullity, [0, 1, 3, 0])
        np.testing.assert_array_equal(
            filled.staged_nullity_refined_mask,
            [False, False, False, True],
        )
        np.testing.assert_array_equal(
            filled.point_solution_status,
            [
                0,
                POINT_STATUS_COMPLETED_OBSERVED,
                POINT_STATUS_COMPLETED_UNOBSERVED,
                POINT_STATUS_PARTIAL_UNRESOLVED,
            ],
        )
        self.assertLessEqual(
            float(filled.diagnostics["nullspace_operator_max_relative_error"]),
            1e-4,
        )
        self.assertLessEqual(
            float(filled.diagnostics["observation_drift_max_relative"]),
            1e-4,
        )

        diagnostics = filled.diagnostics
        self.assertEqual(diagnostics["completion_count"], 2)
        self.assertEqual(
            diagnostics["excluded_reason_counts"]["full_rank_nonanchor"], 1
        )
        self.assertEqual(diagnostics["numerical_subspace"]["refined_partial_count"], 1)


if __name__ == "__main__":
    unittest.main()
