from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from modal_surface.anchor_residual_diagnostics import (
    build_anchor_residual_diagnostics,
    write_anchor_residual_diagnostics,
)


class AnchorResidualDiagnosticsTests(unittest.TestCase):
    def _observations(self) -> dict[str, np.ndarray]:
        point_index = np.repeat(np.arange(2, dtype=np.int32), 8)
        view_pattern = np.repeat(np.array([0, 1], dtype=np.int32), 4)
        view_index = np.tile(view_pattern, 2)
        jacobian0 = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        jacobian1 = np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        jacobians = np.stack(
            [jacobian0 if view == 0 else jacobian1 for view in view_index],
            axis=0,
        )
        point0 = np.array(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [1.0, 0.0],
                [3.0, 0.0],
                [2.0, 0.0],
                [2.0, 0.0],
                [2.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.complex64,
        )
        point1 = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
            ],
            dtype=np.complex64,
        )
        return {
            "points_world": np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
            ),
            "gaussian_indices": np.arange(2, dtype=np.int32),
            "point_type": np.array("foreground_gaussian_center"),
            "source_checkpoint": np.array("source.ckpt"),
            "obs_point_index": point_index,
            "obs_view_index": view_index,
            "obs_pixels_xy": np.zeros((16, 2), dtype=np.float32),
            "obs_y": np.concatenate([point0, point1], axis=0),
            "obs_J": jacobians,
            "obs_contribution_weight": np.ones((16,), dtype=np.float32),
            "obs_count_per_point": np.array([2, 2], dtype=np.int32),
            "obs_sample_count_per_point": np.array([8, 8], dtype=np.int32),
            "view_ids": np.array(["view1", "view2"]),
            "freq_hz": np.array(0.85, dtype=np.float32),
            "mode_index": np.array(4, dtype=np.int32),
        }

    def _solver_diagnostics(self) -> dict[str, np.ndarray]:
        return {
            "alphas": np.ones((2,), dtype=np.complex64),
            "alpha_identifiable_mask": np.ones((2,), dtype=bool),
            "point_observable_rank": np.array([3, 3], dtype=np.int8),
            "point_condition": np.array([2.0, 2.0], dtype=np.float32),
            "point_distinct_view_count": np.array([2, 2], dtype=np.int32),
            "point_distinct_valid_view_count": np.array([2, 2], dtype=np.int32),
            "point_precompletion_residual": np.array(
                [1.0 / 3.0, np.sqrt(2.0 / 6.0)], dtype=np.float32
            ),
            "staged_point_solution_status": np.array([0, 5], dtype=np.int8),
            "anchor_residual_threshold": np.array(0.4, dtype=np.float32),
            "anchor_condition_max": np.array(100.0, dtype=np.float32),
        }

    def test_exact_decomposition_separates_within_and_cross_view_error(self) -> None:
        result = build_anchor_residual_diagnostics(
            self._observations(),
            self._solver_diagnostics(),
        )
        np.testing.assert_allclose(result.point_within_sse, [1.0, 0.0])
        np.testing.assert_allclose(result.point_cross_sse, [0.0, 2.0])
        np.testing.assert_allclose(result.point_total_sse, [1.0, 2.0])
        np.testing.assert_allclose(result.point_within_fraction, [1.0, 0.0])
        np.testing.assert_array_equal(result.anchor_mask, [True, False])
        np.testing.assert_array_equal(result.residual_rejected_mask, [False, True])
        self.assertEqual(int(result.residual_source_class[1]), 4)

    def test_replay_mismatch_fails_before_artifact_write(self) -> None:
        diagnostics = self._solver_diagnostics()
        diagnostics["point_precompletion_residual"] = np.array(
            [0.2, 0.2], dtype=np.float32
        )
        with self.assertRaisesRegex(ValueError, "Replayed residual disagrees"):
            build_anchor_residual_diagnostics(
                self._observations(),
                diagnostics,
            )

    def test_versioned_artifact_preserves_exact_metrics(self) -> None:
        result = build_anchor_residual_diagnostics(
            self._observations(),
            self._solver_diagnostics(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "residuals.npz"
            write_anchor_residual_diagnostics(
                output,
                result,
                source_checkpoint="source.ckpt",
                source_observation_path=Path("observations/mode.npz"),
                source_solver_diagnostics_path=Path("diagnostics/mode.npz"),
            )
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(int(archive["version"]), 1)
                np.testing.assert_allclose(archive["point_within_sse"], [1.0, 0.0])
                np.testing.assert_array_equal(
                    archive["residual_source_names"],
                    [
                        "other",
                        "accepted_anchor",
                        "low_modal_energy",
                        "within_view_dominated",
                        "cross_view_dominated",
                        "mixed",
                    ],
                )


if __name__ == "__main__":
    unittest.main()
