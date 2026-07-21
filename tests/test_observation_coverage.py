from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from modal_surface.gaussian_observations import (
    quaternion_wxyz_to_rotation_matrices,
    score_pixel_gaussian_candidates,
)
from modal_surface.observation_coverage import (
    ObservationCoverageDiagnostics,
    observation_coverage_categories,
    parse_coverage_k_values,
    validate_reference_observation_replay,
    write_observation_coverage_diagnostics,
)


class ObservationCoverageTests(unittest.TestCase):
    def _diagnostics(self) -> ObservationCoverageDiagnostics:
        return ObservationCoverageDiagnostics(
            points_world=np.zeros((3, 3), dtype=np.float32),
            view_ids=np.array(["view1", "view2"]),
            k_values=np.array([4, 8], dtype=np.int32),
            preselect_hit_count_by_view=np.ones((3, 2), dtype=np.int32),
            positive_hit_count_by_view=np.ones((3, 2), dtype=np.int32),
            selected_hit_count_by_k_view=np.array(
                [
                    [[1, 0], [1, 1], [0, 0]],
                    [[1, 1], [1, 1], [1, 0]],
                ],
                dtype=np.int32,
            ),
            best_positive_rank_by_view=np.ones((3, 2), dtype=np.int16),
            best_positive_score_by_view=np.ones((3, 2), dtype=np.float32),
            best_positive_score_ratio_by_view=np.ones((3, 2), dtype=np.float32),
            preselect_view_count=np.array([2, 2, 1], dtype=np.int8),
            positive_view_count=np.array([2, 2, 1], dtype=np.int8),
            selected_view_count_by_k=np.array(
                [[1, 2, 0], [2, 2, 1]], dtype=np.int8
            ),
            selected_sample_count_by_k=np.array(
                [[1, 2, 0], [2, 2, 1]], dtype=np.int32
            ),
            category_by_k=np.array(
                [[2, 3, 0], [3, 3, 0]], dtype=np.int8
            ),
            category_count_by_k=np.array(
                [[1, 0, 1, 1], [1, 0, 0, 2]], dtype=np.int64
            ),
            preselect_view_count_histogram=np.array([0, 1, 2], dtype=np.int64),
            positive_view_count_histogram=np.array([0, 1, 2], dtype=np.int64),
            selected_view_count_histogram_by_k=np.array(
                [[1, 1, 1], [0, 1, 2]], dtype=np.int64
            ),
        )

    def test_candidate_scoring_preserves_contribution_order(self) -> None:
        points = np.array(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, -1.0]],
            dtype=np.float32,
        )
        rotations = quaternion_wxyz_to_rotation_matrices(
            np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (3, 1),
            )
        )
        scored = score_pixel_gaussian_candidates(
            surface_point_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            candidate_indices=np.array([2, 1, 0], dtype=np.int64),
            points_world=points,
            gaussian_scales=np.ones((3, 3), dtype=np.float32),
            gaussian_rotmats=rotations,
            gaussian_opacities=np.array([0.5, 1.0, 1.0], dtype=np.float32),
            point_camera_z=np.array([1.0, 1.0, -1.0], dtype=np.float64),
            min_contribution=0.0,
        )
        np.testing.assert_array_equal(scored.gaussian_indices, [1, 0, 2])
        np.testing.assert_array_equal(
            scored.positive_camera_z_mask,
            [True, True, False],
        )

    def test_k_parser_and_categories_are_deterministic(self) -> None:
        np.testing.assert_array_equal(
            parse_coverage_k_values("4,8,12,16,32", 32),
            [4, 8, 12, 16, 32],
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_coverage_k_values("8,4", 32)
        categories = observation_coverage_categories(
            np.array([1, 2, 2, 2]),
            np.array([1, 1, 2, 2]),
            np.array([0, 0, 1, 2]),
        )
        np.testing.assert_array_equal(categories, [0, 1, 2, 3])

    def test_reference_replay_validation_and_artifact(self) -> None:
        diagnostics = self._diagnostics()
        reference = {
            "obs_count_per_point": np.array([1, 2, 0], dtype=np.int32),
            "obs_sample_count_per_point": np.array([1, 2, 0], dtype=np.int32),
            "observations_per_view": np.array([2, 1], dtype=np.int32),
        }
        reference_path = Path("reference.npz")
        baseline_index = validate_reference_observation_replay(
            diagnostics,
            reference,
            4,
            reference_path,
        )
        self.assertEqual(baseline_index, 0)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "coverage.npz"
            write_observation_coverage_diagnostics(
                output,
                diagnostics,
                source_checkpoint="source.ckpt",
                reference_observation_path=reference_path,
                baseline_k=4,
                baseline_k_index=0,
                mask_erode_iters=1,
                pixel_sample_stride=2,
                pixel_preselect_k=32,
                pixel_render_acc_min=0.05,
                pixel_min_contribution=1.0e-12,
            )
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(int(archive["version"]), 1)
                np.testing.assert_array_equal(archive["category_by_k"], [[2, 3, 0], [3, 3, 0]])
        bad_reference = dict(reference)
        bad_reference["obs_count_per_point"] = np.array([2, 2, 0])
        with self.assertRaisesRegex(ValueError, "view counts"):
            validate_reference_observation_replay(
                diagnostics,
                bad_reference,
                4,
                reference_path,
            )


if __name__ == "__main__":
    unittest.main()
