from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from modal_surface.optimization_staged import (
    POINT_STATUS_ALPHA_UNRESOLVED,
    POINT_STATUS_NO_USABLE_OBSERVATION,
    StagedSolverConfig,
    enforce_alpha_failure,
    optimize_multi_view_staged,
    prepare_observations,
    solve_alpha_sync,
    solve_observable_points,
    write_staged_debug_npz,
)


J_VIEW_0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
J_VIEW_1 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _observation_dict(
    point_modes: np.ndarray,
    point_views: list[list[int]],
    alpha: np.ndarray,
    duplicate_rows: int = 1,
    num_views: int = 2,
) -> dict[str, np.ndarray]:
    point_modes = np.asarray(point_modes, dtype=np.complex64)
    point_index: list[int] = []
    view_index: list[int] = []
    obs_y: list[np.ndarray] = []
    obs_J: list[np.ndarray] = []
    for point_idx, views in enumerate(point_views):
        for view_idx in views:
            J = J_VIEW_0 if view_idx == 0 else J_VIEW_1
            value = alpha[view_idx] * (J @ point_modes[point_idx])
            for _ in range(duplicate_rows):
                point_index.append(point_idx)
                view_index.append(view_idx)
                obs_y.append(value)
                obs_J.append(J)
    point_arr = np.asarray(point_index, dtype=np.int32)
    view_arr = np.asarray(view_index, dtype=np.int32)
    view_mask = np.zeros((point_modes.shape[0], num_views), dtype=bool)
    if point_arr.size:
        view_mask[point_arr, view_arr] = True
    sample_count = np.bincount(point_arr, minlength=point_modes.shape[0]).astype(np.int32)
    return {
        "points_world": np.column_stack(
            [np.arange(point_modes.shape[0], dtype=np.float32), np.zeros((point_modes.shape[0], 2), dtype=np.float32)]
        ),
        "obs_point_index": point_arr,
        "obs_view_index": view_arr,
        "obs_pixels_xy": np.zeros((point_arr.size, 2), dtype=np.float32),
        "obs_y": np.asarray(obs_y, dtype=np.complex64).reshape(-1, 2),
        "obs_J": np.asarray(obs_J, dtype=np.float32).reshape(-1, 2, 3),
        "obs_contribution_weight": np.ones((point_arr.size,), dtype=np.float32),
        "obs_count_per_point": view_mask.sum(axis=1).astype(np.int32),
        "obs_sample_count_per_point": sample_count,
        "view_ids": np.asarray([f"view{i}" for i in range(num_views)]),
        "view_freqs_hz": np.ones((num_views,), dtype=np.float32),
        "freq_hz": np.array(1.0, dtype=np.float32),
        "mode_index": np.array(0, dtype=np.int32),
        "gaussian_indices": np.arange(point_modes.shape[0], dtype=np.int32),
    }


class StagedModalSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.true_alpha = np.asarray([1.0 + 0.0j, np.exp(0.8j)], dtype=np.complex64)
        self.config = StagedSolverConfig(
            alpha_min_shared_points=1,
            alpha_rank_ratio_min=1e-8,
            alpha_info_ratio_min=1e-8,
        )

    def test_informative_overlap_recovers_phase(self) -> None:
        modes = np.asarray([[1.0, 0.4, 0.2], [0.3, -0.7, 0.8], [-0.2, 1.1, 0.5]], dtype=np.complex64)
        data = _observation_dict(modes, [[0, 1]] * len(modes), self.true_alpha)
        prepared = prepare_observations(data)
        result = solve_alpha_sync(prepared, self.config)
        self.assertTrue(np.all(result.identifiable_mask))
        error = np.angle(result.alphas[1] * np.conj(self.true_alpha[1]))
        self.assertLess(abs(float(error)), 1e-5)

    def test_general_complex_point_motion_is_recovered_at_anchors(self) -> None:
        modes = np.asarray(
            [
                [1.0 + 0.2j, 0.4 - 0.3j, 0.2 + 0.7j],
                [0.3 - 0.1j, -0.7 + 0.5j, 0.8 - 0.2j],
                [-0.2 + 0.6j, 1.1 + 0.1j, 0.5 - 0.4j],
            ],
            dtype=np.complex64,
        )
        prepared = prepare_observations(
            _observation_dict(modes, [[0, 1]] * len(modes), self.true_alpha)
        )
        alpha = solve_alpha_sync(prepared, self.config)
        solved = solve_observable_points(prepared, alpha, self.config)
        self.assertTrue(np.all(solved.anchor_mask))
        np.testing.assert_allclose(solved.phi, modes, atol=1e-5, rtol=1e-5)

    def test_duplicate_rows_do_not_change_alpha(self) -> None:
        modes = np.asarray([[1.0, 0.5, 0.2], [0.3, -0.6, 0.7]], dtype=np.complex64)
        base = solve_alpha_sync(
            prepare_observations(_observation_dict(modes, [[0, 1], [0, 1]], self.true_alpha)),
            self.config,
        )
        duplicated = solve_alpha_sync(
            prepare_observations(
                _observation_dict(modes, [[0, 1], [0, 1]], self.true_alpha, duplicate_rows=4)
            ),
            self.config,
        )
        np.testing.assert_allclose(base.alphas, duplicated.alphas, atol=1e-6, rtol=1e-6)

        base_prepared = prepare_observations(
            _observation_dict(modes, [[0, 1], [0, 1]], self.true_alpha)
        )
        duplicate_prepared = prepare_observations(
            _observation_dict(
                modes,
                [[0, 1], [0, 1]],
                self.true_alpha,
                duplicate_rows=4,
            )
        )
        base_points = solve_observable_points(base_prepared, base, self.config)
        duplicate_points = solve_observable_points(
            duplicate_prepared, duplicated, self.config
        )
        np.testing.assert_allclose(base_points.phi, duplicate_points.phi, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(base_points.anchor_mask, duplicate_points.anchor_mask)

    def test_bounded_complex_recovers_relative_gain(self) -> None:
        true_alpha = np.asarray([1.0 + 0.0j, 0.7 * np.exp(0.8j)], dtype=np.complex64)
        modes = np.asarray([[1.0, 0.5, 0.2], [0.3, -0.6, 0.7], [-0.4, 1.2, 0.3]], dtype=np.complex64)
        config = StagedSolverConfig(
            alpha_model="bounded-complex",
            alpha_min_shared_points=1,
            alpha_rank_ratio_min=1e-8,
            alpha_info_ratio_min=1e-8,
        )
        result = solve_alpha_sync(
            prepare_observations(_observation_dict(modes, [[0, 1]] * len(modes), true_alpha)),
            config,
        )
        self.assertTrue(result.identifiable_mask[1])
        self.assertLess(abs(float(np.angle(result.alphas[1] * np.conj(true_alpha[1])))), 1e-5)
        self.assertLess(abs(float(abs(result.alphas[1]) - abs(true_alpha[1]))), 1e-4)

    def test_block_huber_rejects_ten_percent_corrupt_overlap(self) -> None:
        rng = np.random.default_rng(7)
        modes = rng.normal(size=(40, 3)).astype(np.float32).astype(np.complex64)
        modes[:, 1] += np.complex64(1.5)
        data = _observation_dict(modes, [[0, 1]] * len(modes), self.true_alpha)
        corrupt_points = np.arange(0, 40, 10, dtype=np.int64)
        for point in corrupt_points.tolist():
            row = np.where(
                (data["obs_point_index"] == point)
                & (data["obs_view_index"] == 1)
            )[0][0]
            data["obs_y"][row] += np.asarray([3.0 + 2.0j, -2.0 + 1.0j], dtype=np.complex64)
        prepared = prepare_observations(data)
        alpha = solve_alpha_sync(prepared, self.config)
        phase_error = np.angle(alpha.alphas[1] * np.conj(self.true_alpha[1]))
        self.assertLess(abs(float(phase_error)), 0.02)
        solved = solve_observable_points(prepared, alpha, self.config)
        self.assertGreaterEqual(int(solved.rejected_mask[corrupt_points].sum()), 3)

    def test_zero_cross_view_motion_is_marked_unidentifiable(self) -> None:
        modes = np.asarray([[1.0, 0.0, 0.2], [0.3, 0.0, 0.7]], dtype=np.complex64)
        result = solve_alpha_sync(
            prepare_observations(_observation_dict(modes, [[0, 1], [0, 1]], self.true_alpha)),
            self.config,
        )
        self.assertTrue(result.identifiable_mask[0])
        self.assertFalse(result.identifiable_mask[1])
        self.assertEqual(str(result.exclusion_reason[1]), "insufficient_information")

    def test_single_view_point_stays_at_observable_minimum_norm(self) -> None:
        modes = np.asarray([[1.0, 0.5, 0.25], [0.2, -0.4, 2.0]], dtype=np.complex64)
        prepared = prepare_observations(
            _observation_dict(modes, [[0, 1], [0]], self.true_alpha)
        )
        alpha = solve_alpha_sync(prepared, self.config)
        solved = solve_observable_points(prepared, alpha, self.config)
        self.assertTrue(solved.anchor_mask[0])
        self.assertEqual(int(solved.observable_rank[1]), 2)
        self.assertTrue(solved.partial_mask[1])
        np.testing.assert_allclose(solved.phi[1], np.asarray([0.2, -0.4, 0.0]), atol=1e-6)
        projected = J_VIEW_0 @ solved.phi[1]
        np.testing.assert_allclose(projected, J_VIEW_0 @ modes[1], atol=1e-6)

    def test_zero_overlap_view_is_insufficient_information(self) -> None:
        modes = np.asarray([[1.0, 0.5, 0.25], [0.2, -0.4, 2.0]], dtype=np.complex64)
        alpha = np.asarray([1.0 + 0.0j, np.exp(0.8j), np.exp(-0.3j)], dtype=np.complex64)
        data = _observation_dict(modes, [[0, 1], [2]], alpha, num_views=3)
        result = solve_alpha_sync(prepare_observations(data), self.config)
        self.assertTrue(result.identifiable_mask[0])
        self.assertTrue(result.identifiable_mask[1])
        self.assertFalse(result.identifiable_mask[2])
        np.testing.assert_array_equal(
            result.reference_connected_mask,
            [True, True, False],
        )
        self.assertEqual(str(result.exclusion_reason[2]), "insufficient_information")

    def test_point_seen_only_by_excluded_view_is_not_called_unobserved(self) -> None:
        modes = np.asarray([[1.0, 0.5, 0.25], [0.2, -0.4, 2.0]], dtype=np.complex64)
        prepared = prepare_observations(
            _observation_dict(modes, [[0], [1]], self.true_alpha)
        )
        alpha = solve_alpha_sync(prepared, self.config)
        solved = solve_observable_points(prepared, alpha, self.config)
        self.assertFalse(alpha.identifiable_mask[1])
        self.assertTrue(solved.alpha_unresolved_mask[1])
        self.assertFalse(solved.unobserved_mask[1])
        self.assertFalse(solved.partial_mask[1])
        self.assertEqual(int(solved.point_status[1]), POINT_STATUS_ALPHA_UNRESOLVED)
        np.testing.assert_array_equal(solved.phi[1], np.zeros((3,), dtype=np.complex64))

    def test_empty_reference_is_explicitly_unidentifiable(self) -> None:
        modes = np.asarray([[1.0, 0.5, 0.25]], dtype=np.complex64)
        result = solve_alpha_sync(
            prepare_observations(_observation_dict(modes, [[1]], self.true_alpha)),
            self.config,
        )
        self.assertFalse(result.identifiable_mask[0])
        self.assertEqual(str(result.exclusion_reason[0]), "empty_reference")

    def test_zero_contribution_weight_rows_are_marked_no_usable_observation(self) -> None:
        modes = np.asarray([[1.0, 0.5, 0.25], [0.2, -0.4, 2.0]], dtype=np.complex64)
        data = _observation_dict(modes, [[0], [0]], self.true_alpha)
        data["obs_contribution_weight"][data["obs_point_index"] == 1] = 0.0
        prepared = prepare_observations(data)
        alpha = solve_alpha_sync(prepared, self.config)
        solved = solve_observable_points(prepared, alpha, self.config)
        self.assertTrue(solved.no_usable_observation_mask[1])
        self.assertFalse(solved.alpha_unresolved_mask[1])
        self.assertFalse(solved.unobserved_mask[1])
        self.assertEqual(
            int(solved.point_status[1]), POINT_STATUS_NO_USABLE_OBSERVATION
        )

    def test_poor_baseline_is_partial_instead_of_anchor(self) -> None:
        modes = np.asarray(
            [[1.0, 0.5, 0.25], [0.2, -0.4, 2.0], [-0.3, 0.9, 0.7]],
            dtype=np.complex64,
        )
        data = _observation_dict(modes, [[0, 1]] * len(modes), self.true_alpha)
        weak_view = np.asarray(
            [[1.0, 0.0, 1e-4], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        view_one_rows = np.where(data["obs_view_index"] == 1)[0]
        data["obs_J"][view_one_rows] = weak_view
        for row in view_one_rows.tolist():
            point = int(data["obs_point_index"][row])
            data["obs_y"][row] = self.true_alpha[1] * (weak_view @ modes[point])
        prepared = prepare_observations(data)
        alpha = solve_alpha_sync(prepared, self.config)
        solved = solve_observable_points(prepared, alpha, self.config)
        self.assertTrue(np.all(alpha.identifiable_mask))
        self.assertFalse(np.any(solved.anchor_mask))
        self.assertTrue(np.all(solved.observable_rank == 2))
        self.assertTrue(np.all(solved.partial_mask))

    def test_strict_failure_writes_diagnostics_before_raising(self) -> None:
        modes = np.asarray([[1.0, 0.5, 0.25], [0.2, -0.4, 2.0]], dtype=np.complex64)
        data = _observation_dict(modes, [[0], [1]], self.true_alpha)
        config = StagedSolverConfig(
            alpha_min_shared_points=1,
            alpha_rank_ratio_min=1e-8,
            alpha_info_ratio_min=1e-8,
            alpha_failure="error",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "latent.npz"
            result = optimize_multi_view_staged(data, config=config)
            self.assertFalse(out_path.exists())
            write_staged_debug_npz(result, out_path)
            with self.assertRaises(ValueError):
                enforce_alpha_failure(result, out_path)
            self.assertTrue(out_path.exists())
            with np.load(out_path, allow_pickle=False) as latent:
                self.assertFalse(latent["alpha_identifiable_mask"][1])
                self.assertEqual(
                    str(latent["alpha_exclusion_reason"][1]),
                    "insufficient_information",
                )

    def test_output_preserves_point_order_and_adds_status_fields(self) -> None:
        modes = np.asarray([[1.0, 0.5, 0.25], [0.2, -0.4, 2.0], [0.0, 0.0, 0.0]], dtype=np.complex64)
        data = _observation_dict(modes, [[0, 1], [0], []], self.true_alpha)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "latent.npz"
            result = optimize_multi_view_staged(data, config=self.config)
            self.assertFalse(out_path.exists())
            np.testing.assert_array_equal(
                result.prepared.arrays["gaussian_indices"],
                np.arange(3, dtype=np.int32),
            )
            np.testing.assert_allclose(result.prepared.points, data["points_world"])
            self.assertTrue(result.observable.anchor_mask[0])
            self.assertTrue(result.observable.partial_mask[1])
            self.assertTrue(result.observable.unobserved_mask[2])
            self.assertEqual(result.obs_pred_y.shape, data["obs_y"].shape)
            self.assertEqual(result.obs_residual.shape, (data["obs_y"].shape[0],))
            write_staged_debug_npz(result, out_path)
            with np.load(out_path, allow_pickle=False) as latent:
                np.testing.assert_array_equal(latent["gaussian_indices"], np.arange(3, dtype=np.int32))
                np.testing.assert_allclose(latent["points_world"], data["points_world"])
                self.assertEqual(str(np.asarray(latent["solver_method"]).item()), "staged_overlap_observable")
                self.assertTrue(latent["anchor_mask"][0])
                self.assertTrue(latent["partial_mask"][1])
                self.assertTrue(latent["unobserved_mask"][2])
                self.assertFalse(np.any(latent["completion_mask"]))
                np.testing.assert_array_equal(
                    latent["alpha_reference_connected_mask"],
                    [True, True],
                )
                legacy_fields = {
                    "alpha_component_index",
                    "alpha_information_component_index",
                    "graph_degree",
                    "modal_rigid_edge_count",
                    "modal_fill_enabled",
                    "single_view_refined_mask",
                    "optimization_history",
                    "alpha_history",
                    "active_indices",
                }
                self.assertTrue(legacy_fields.isdisjoint(latent.files))


if __name__ == "__main__":
    unittest.main()
