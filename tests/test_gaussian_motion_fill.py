from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
    write_motion_fill_diagnostics,
    write_motion_fill_graph,
)
from modal_surface.motion_fill import build_knn_graph, query_knn_candidates
from modal_surface.optimization_staged import (
    POINT_STATUS_COMPLETED_OBSERVED,
    POINT_STATUS_COMPLETED_UNOBSERVED,
    POINT_STATUS_REJECTED_UNRESOLVED,
)


def _mask(indices: list[int], count: int) -> np.ndarray:
    result = np.zeros((count,), dtype=bool)
    result[indices] = True
    return result


def _role_case() -> dict[str, np.ndarray]:
    count = 7
    return {
        "points_world": np.column_stack(
            [np.arange(count, dtype=np.float32), np.zeros((count, 2), dtype=np.float32)]
        ),
        "anchor_mask": _mask([0], count),
        "partial_mask": _mask([1, 6], count),
        "unobserved_mask": _mask([2], count),
        "rejected_mask": _mask([3], count),
        "alpha_unresolved_mask": _mask([4], count),
        "no_usable_observation_mask": _mask([5], count),
        "point_nullity": np.asarray([0, 1, 3, 1, 3, 3, 0], dtype=np.int8),
    }


def _formal_latent_case() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
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
        "phi_nullspace_correction": np.zeros_like(phi_observable),
        "point_nullspace_basis": basis,
        "point_nullity": np.asarray([0, 1, 3, 0], dtype=np.int8),
        "anchor_mask": _mask([0], 4),
        "partial_mask": _mask([1], 4),
        "unobserved_mask": _mask([2], 4),
        "rejected_mask": _mask([3], 4),
        "alpha_unresolved_mask": false_mask.copy(),
        "no_usable_observation_mask": false_mask.copy(),
        "completion_mask": false_mask.copy(),
        "completion_connected_to_anchor": false_mask.copy(),
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
        "gaussian_indices": np.arange(4, dtype=np.int32),
        "freq_hz": np.array(2.5, dtype=np.float32),
        "mode_index": np.array(0, dtype=np.int32),
        "point_type": np.array("foreground_gaussian_center"),
        "source_checkpoint": np.array("source.ckpt"),
        "solver_method": np.array("staged_overlap_observable"),
        "preserved_sentinel": np.asarray([7, 11, 13], dtype=np.int32),
    }
    return arrays, target, excluded_value


class GaussianMotionFillTests(unittest.TestCase):
    def test_staged_states_map_to_four_explicit_roles(self) -> None:
        roles = derive_gaussian_motion_fill_roles(_role_case())

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

    def test_fill_preserves_pixel_rows_and_quarantines_excluded_points(self) -> None:
        arrays, target, excluded_value = _formal_latent_case()
        points = arrays["points_world"]
        candidates = query_knn_candidates(points, max_k=1)
        graph = build_knn_graph(candidates, k=1, max_distance=0.11)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latent_path = root / "mode.npz"
            graph_path = root / "motion_fill" / "graph.npz"
            diagnostics_path = root / "motion_fill" / "diagnostics.json"
            np.savez_compressed(latent_path, **arrays)
            write_motion_fill_graph(graph_path, points, candidates, graph)
            diagnostics = apply_gaussian_motion_fill(
                latent_path,
                graph,
                "motion_fill/graph.npz",
            )
            write_motion_fill_diagnostics(diagnostics_path, diagnostics)

            with np.load(latent_path, allow_pickle=False) as filled:
                np.testing.assert_allclose(
                    filled["phi"][0:3],
                    np.broadcast_to(target, (3, 3)),
                    atol=1e-5,
                )
                np.testing.assert_array_equal(filled["phi"][3], excluded_value)
                np.testing.assert_array_equal(
                    filled["phi_nullspace_correction"][3], np.zeros((3,), dtype=np.complex64)
                )
                np.testing.assert_array_equal(filled["completion_mask"], [False, True, True, False])
                np.testing.assert_array_equal(
                    filled["point_solution_status"],
                    [0, POINT_STATUS_COMPLETED_OBSERVED, POINT_STATUS_COMPLETED_UNOBSERVED, POINT_STATUS_REJECTED_UNRESOLVED],
                )
                np.testing.assert_array_equal(filled["motion_fill_role"], [0, 1, 2, 3])
                np.testing.assert_array_equal(
                    filled["point_active_component_index"], [0, 0, 0, -1]
                )
                np.testing.assert_array_equal(filled["point_anchor_hop_distance"], [0, 1, 2, -1])
                np.testing.assert_array_equal(filled["obs_point_index"], arrays["obs_point_index"])
                np.testing.assert_array_equal(filled["obs_view_index"], arrays["obs_view_index"])
                np.testing.assert_array_equal(
                    filled["preserved_sentinel"], arrays["preserved_sentinel"]
                )
                self.assertFalse(bool(filled["obs_residual_valid_mask"][4]))
                self.assertTrue(np.isnan(filled["obs_pred_y"][4].real).all())
                self.assertEqual(
                    str(np.asarray(filled["motion_fill_graph_path"]).item()),
                    "motion_fill/graph.npz",
                )

            with np.load(graph_path, allow_pickle=False) as saved_graph:
                np.testing.assert_array_equal(saved_graph["edge_index"], graph.edge_index)
                np.testing.assert_array_equal(
                    saved_graph["candidate_neighbor_indices"], candidates.neighbor_indices
                )
            with diagnostics_path.open("r", encoding="utf-8") as handle:
                saved_diagnostics = json.load(handle)

        self.assertEqual(diagnostics["completion_count"], 2)
        self.assertEqual(diagnostics["role_counts"]["excluded"], 1)
        self.assertEqual(diagnostics["system"]["eligible_edge_count"], 2)
        self.assertEqual(saved_diagnostics["method"], diagnostics["method"])

    def test_near_nullspace_leakage_fails_for_zero_measurement(self) -> None:
        arrays, _, _ = _formal_latent_case()
        target = np.asarray([0.0, 0.0, 1e4], dtype=np.complex64)
        arrays["phi_observable"][0] = target
        arrays["phi_observable"][1:3] = 0.0
        arrays["phi"] = arrays["phi_observable"].copy()
        leakage = 1e-6
        arrays["point_nullspace_basis"][1] = 0.0
        arrays["point_nullspace_basis"][1, :, 0] = np.asarray(
            [leakage, 0.0, np.sqrt(1.0 - leakage**2)], dtype=np.float32
        )
        for row in (0, 1):
            view = int(arrays["obs_view_index"][row])
            arrays["obs_y"][row] = arrays["alphas"][view] * (
                arrays["obs_J"][row] @ target
            )
        arrays["obs_y"][2:4] = 0.0
        graph = build_knn_graph(
            query_knn_candidates(arrays["points_world"], max_k=1),
            k=1,
            max_distance=0.11,
        )

        with tempfile.TemporaryDirectory() as tmp:
            latent_path = Path(tmp) / "mode.npz"
            np.savez_compressed(latent_path, **arrays)
            with self.assertRaisesRegex(RuntimeError, "relative drift"):
                apply_gaussian_motion_fill(
                    latent_path,
                    graph,
                    "motion_fill/graph.npz",
                )


if __name__ == "__main__":
    unittest.main()
