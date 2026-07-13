from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preproc.synthetic_modal_visibility_sphere import (
    _build_observations,
    _build_elliptical_mode,
    _camera_intrinsics,
    _camera_centers_on_arc,
    _complex_mode_relative_error,
    _fibonacci_sphere,
    _group_diagnostics,
    _look_at_world_to_camera,
    _mode_error_arrays,
    _phase_stats,
    _valid_view_count_colors,
    _view_alphas,
    _write_diagnostics,
    _write_valid_view_count_latent,
    _write_viewer_inputs,
    build_arg_parser,
)


def _phase_distance(a: float, b: float) -> float:
    return abs(float(np.angle(np.exp(1j * (a - b)))))


class SyntheticModalVisibilitySphereTests(unittest.TestCase):
    def _diagnostic_payload(self, num_views: int) -> dict[str, object]:
        num_points = num_views + 1
        valid_view_count = np.arange(num_points, dtype=np.int32)
        raw_view_count = valid_view_count.copy()
        raw_view_count[0] = 1
        visibility = np.zeros((num_points, num_views), dtype=bool)
        for point_index, count in enumerate(raw_view_count):
            visibility[point_index, :count] = True
        reference_phi = np.zeros((num_points, 3), dtype=np.complex64)
        reference_phi[:, 0] = 1.0 + 0.0j
        recovered_phi = reference_phi.copy()
        if num_views >= 2:
            recovered_phi[2, 0] = 1.0 + np.sqrt(2.0)
        observable_rank = np.where(
            valid_view_count == 0,
            0,
            np.where(valid_view_count == 1, 2, 3),
        ).astype(np.int8)
        anchor_mask = valid_view_count >= 2
        point_residual = valid_view_count.astype(np.float32) * np.float32(0.1)
        point_residual[0] = np.nan
        point_residual_valid = valid_view_count > 0
        condition = np.full((num_points,), np.inf, dtype=np.float32)
        condition[point_residual_valid] = np.arange(1, num_points, dtype=np.float32)
        true_alphas = _view_alphas(num_views, 0.8)
        num_observations = int(raw_view_count.sum())
        observations = {
            "points_world": np.zeros((num_points, 3), dtype=np.float32),
            "obs_count_per_point": raw_view_count,
            "point_view_mask": visibility,
            "view_ids": np.asarray(
                [f"view{view_index + 1}" for view_index in range(num_views)]
            ),
            "obs_y": np.zeros((num_observations, 2), dtype=np.complex64),
            "motion_direction_camera": np.repeat(
                np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), num_views, axis=0
            ),
            "motion_depth_components": np.zeros((num_views,), dtype=np.float32),
            "true_alphas": true_alphas,
            "phi_gt": reference_phi,
        }
        solved = {
            "phi": recovered_phi,
            "point_distinct_valid_view_count": valid_view_count.copy(),
            "point_observable_rank": observable_rank,
            "point_condition": condition,
            "anchor_mask": anchor_mask,
            "point_residual": point_residual,
            "point_residual_valid_mask": point_residual_valid,
            "alpha_by_view": true_alphas.copy(),
            "alpha_phase": np.angle(true_alphas).astype(np.float32),
            "alpha_identifiable_mask": np.ones((num_views,), dtype=bool),
            "alpha_exclusion_reason": np.full((num_views,), "", dtype="<U1"),
            "solver_method": np.asarray("staged_overlap_observable"),
            "obs_pred_y": np.zeros((num_observations, 2), dtype=np.complex64),
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics.json"
            _write_diagnostics(
                path,
                observations,
                solved,
                np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                image_width=1280,
                image_height=720,
                visibility_margin=0.03,
            )
            return json.loads(path.read_text(encoding="utf-8"))

    def test_num_views_cli_default_and_explicit_value(self) -> None:
        parser = build_arg_parser()

        self.assertEqual(parser.parse_args([]).num_views, 2)
        self.assertEqual(parser.parse_args(["--num-views", "3"]).num_views, 3)

    def test_view_geometry_helpers_reject_fewer_than_two_views(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_views"):
            _camera_centers_on_arc(1)
        with self.assertRaisesRegex(ValueError, "num_views"):
            _view_alphas(1, 0.8)

    def test_three_view_diagnostics_group_by_solver_valid_view_count(self) -> None:
        payload = self._diagnostic_payload(num_views=3)

        self.assertEqual(payload["num_views"], 3)
        expected_counts = {f"valid_view_count_{count}": 1 for count in range(4)}
        self.assertEqual(payload["valid_view_count_counts"], expected_counts)
        groups = payload["valid_view_count_groups"]
        self.assertEqual(set(groups), set(expected_counts))

        group_two = groups["valid_view_count_2"]
        self.assertEqual(
            group_two["observable_rank_distribution"],
            {"rank_0": 0, "rank_1": 0, "rank_2": 0, "rank_3": 1},
        )
        self.assertEqual(group_two["anchor_count"], 1)
        self.assertEqual(group_two["anchor_fraction"], 1.0)
        self.assertAlmostEqual(group_two["condition_number"]["mean"], 2.0)
        self.assertAlmostEqual(group_two["observation_residual"]["mean"], 0.2, places=6)
        self.assertAlmostEqual(
            group_two["complex_mode_vector_error"]["mean"], np.sqrt(2.0), places=6
        )
        self.assertAlmostEqual(group_two["complex_direction_error_deg"]["mean"], 0.0)
        self.assertAlmostEqual(
            group_two["amplitude_absolute_error"]["mean"], np.sqrt(2.0), places=6
        )
        self.assertAlmostEqual(
            group_two["amplitude_relative_error"]["mean"], np.sqrt(2.0), places=6
        )
        self.assertAlmostEqual(group_two["trajectory_rmse"]["mean"], 1.0, places=6)

        group_zero = groups["valid_view_count_0"]
        self.assertEqual(group_zero["condition_number"]["count"], 0)
        self.assertEqual(group_zero["observation_residual"]["count"], 0)
        for legacy_key in (
            "visibility_counts",
            "groups",
            "view1_only_vs_view2_only_mean_direction_angle_deg",
        ):
            self.assertNotIn(legacy_key, payload)

    def test_two_view_diagnostics_preserve_legacy_fields(self) -> None:
        payload = self._diagnostic_payload(num_views=2)

        self.assertEqual(payload["num_views"], 2)
        self.assertEqual(
            payload["valid_view_count_counts"],
            {
                "valid_view_count_0": 1,
                "valid_view_count_1": 1,
                "valid_view_count_2": 1,
            },
        )
        self.assertEqual(
            payload["visibility_counts"],
            {
                "view1_only": 2,
                "view2_only": 0,
                "overlap": 1,
                "observed_all": 3,
                "unobserved": 0,
            },
        )
        self.assertEqual(
            set(payload["groups"]),
            {"view1_only", "view2_only", "overlap", "observed_all", "unobserved"},
        )
        self.assertIn("view1_only_vs_view2_only_mean_direction_angle_deg", payload)

    def test_camera_centers_preserve_endpoints_and_insert_center_view(self) -> None:
        expected_endpoints = np.asarray(
            [[-2.5, -3.0, 0.0], [2.5, -3.0, 0.0]], dtype=np.float32
        )

        centers_two = _camera_centers_on_arc(2)
        centers_three = _camera_centers_on_arc(3)

        np.testing.assert_array_equal(centers_two, expected_endpoints)
        np.testing.assert_array_equal(centers_three[[0, -1]], expected_endpoints)
        np.testing.assert_allclose(
            centers_three[1],
            np.asarray([0.0, -np.hypot(2.5, 3.0), 0.0], dtype=np.float32),
            atol=1e-6,
        )
        for center in centers_three:
            world_to_camera = _look_at_world_to_camera(
                center, np.zeros(3, dtype=np.float64)
            )
            sphere_center_camera = world_to_camera[:3, 3]
            np.testing.assert_allclose(sphere_center_camera[:2], np.zeros(2), atol=1e-6)
            self.assertGreater(float(sphere_center_camera[2]), 0.0)

    def test_camera_centers_support_more_than_three_equal_arc_steps(self) -> None:
        centers = _camera_centers_on_arc(5).astype(np.float64)
        radii = np.linalg.norm(centers[:, :2], axis=1)
        angles = np.unwrap(np.arctan2(centers[:, 1], centers[:, 0]))

        np.testing.assert_allclose(radii, np.full(5, np.hypot(2.5, 3.0)), atol=1e-6)
        np.testing.assert_allclose(
            np.diff(angles),
            np.full(4, np.diff(angles)[0]),
            atol=1e-7,
        )

    def test_view_alpha_phases_preserve_span(self) -> None:
        phase_offset = 0.8

        np.testing.assert_allclose(
            _view_alphas(2, phase_offset),
            np.exp(1j * np.asarray([0.0, phase_offset], dtype=np.float64)),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            _view_alphas(3, phase_offset),
            np.exp(1j * np.asarray([0.0, 0.5 * phase_offset, phase_offset], dtype=np.float64)),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            np.unwrap(np.angle(_view_alphas(5, phase_offset))),
            np.linspace(0.0, phase_offset, 5),
            atol=1e-7,
        )

    def test_observation_arrays_expand_to_three_views(self) -> None:
        points = _fibonacci_sphere(512, 1.0)
        colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
        K = _camera_intrinsics(1280, 720, 900.0)
        camera_centers = _camera_centers_on_arc(3)
        world_to_cameras = np.stack(
            [
                _look_at_world_to_camera(center, np.zeros(3, dtype=np.float64))
                for center in camera_centers
            ],
            axis=0,
        )

        observations = _build_observations(
            points=points,
            colors=colors,
            K=K,
            world_to_cameras=world_to_cameras,
            camera_centers=camera_centers,
            image_width=1280,
            image_height=720,
            visibility_margin=0.03,
            phase_offset_rad=0.8,
            freq_hz=1.0,
            motion_direction=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            phi_world=np.asarray([0.0, 1.0, 0.0], dtype=np.complex64),
            mode_index=0,
        )

        np.testing.assert_array_equal(
            observations["view_ids"], np.asarray(["view1", "view2", "view3"])
        )
        np.testing.assert_allclose(observations["view_freqs_hz"], np.ones(3), atol=0.0)
        np.testing.assert_allclose(observations["true_alphas"], _view_alphas(3, 0.8))
        self.assertEqual(observations["point_view_mask"].shape, (points.shape[0], 3))
        self.assertEqual(observations["motion_direction_camera"].shape, (3, 3))
        np.testing.assert_array_equal(np.unique(observations["obs_view_index"]), [0, 1, 2])

    def test_mode_error_arrays_match_full_cycle_trajectory_formula(self) -> None:
        reference = np.asarray(
            [[1.0 + 0.0j, 0.0j, 0.0j], [1.0 + 0.0j, 0.0j, 0.0j]],
            dtype=np.complex64,
        )
        recovered = reference.copy()
        recovered[1, 0] = 3.0 + 0.0j

        errors = _mode_error_arrays(recovered, reference)

        self.assertEqual(
            set(errors),
            {
                "complex_mode_vector_error",
                "complex_direction_error_deg",
                "amplitude_absolute_error",
                "amplitude_relative_error",
                "trajectory_rmse",
            },
        )
        np.testing.assert_allclose(errors["complex_mode_vector_error"], [0.0, 2.0], atol=1e-7)
        np.testing.assert_allclose(errors["complex_direction_error_deg"], [0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(errors["amplitude_absolute_error"], [0.0, 2.0], atol=1e-7)
        np.testing.assert_allclose(errors["amplitude_relative_error"], [0.0, 2.0], atol=1e-7)
        np.testing.assert_allclose(
            errors["trajectory_rmse"],
            errors["complex_mode_vector_error"] / np.sqrt(2.0),
            atol=1e-7,
        )

    def test_valid_view_count_colors_distinguish_two_and_three_views(self) -> None:
        counts = np.asarray([0, 1, 2, 3, 2], dtype=np.int32)

        colors = _valid_view_count_colors(counts, num_views=3)

        self.assertEqual(colors.shape, (5, 3))
        self.assertEqual(colors.dtype, np.uint8)
        np.testing.assert_array_equal(colors[2], colors[4])
        self.assertFalse(np.array_equal(colors[2], colors[3]))
        self.assertFalse(np.array_equal(colors[0], colors[1]))

    def test_valid_view_count_colors_reject_invalid_counts(self) -> None:
        invalid_counts = (
            np.asarray([[0, 1]], dtype=np.int32),
            np.asarray([0.0, 1.0], dtype=np.float32),
            np.asarray([-1, 0], dtype=np.int32),
            np.asarray([0, 4], dtype=np.int32),
        )
        for counts in invalid_counts:
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    _valid_view_count_colors(counts, num_views=3)

    def test_valid_view_count_latent_writes_viewer_group_metadata(self) -> None:
        points = np.zeros((3, 3), dtype=np.float32)
        phi = np.ones((3, 3), dtype=np.complex64)
        valid_view_count = np.asarray([1, 2, 3], dtype=np.int32)
        obs_count = np.asarray([1, 2, 3], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valid_view_count.npz"
            _write_valid_view_count_latent(
                path,
                points,
                phi,
                valid_view_count,
                num_views=3,
                freq_hz=1.0,
                mode_index=0,
                obs_count_per_point=obs_count,
            )
            with np.load(path, allow_pickle=False) as latent:
                np.testing.assert_array_equal(
                    latent["point_distinct_valid_view_count"], valid_view_count
                )
                np.testing.assert_array_equal(latent["point_group"], valid_view_count)
                np.testing.assert_array_equal(
                    latent["point_group_names"],
                    np.asarray(
                        ["0 valid views", "1 valid view", "2 valid views", "3 valid views"]
                    ),
                )
                self.assertFalse(np.array_equal(latent["colors"][1], latent["colors"][2]))

    def test_viewer_inputs_expand_paths_intrinsics_and_extrinsics(self) -> None:
        points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        colors = np.asarray([[10, 20, 30]], dtype=np.uint8)
        K = np.asarray(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _write_viewer_inputs(out_dir, points, colors, K, extrinsics, 1280, 720)
            with np.load(out_dir / "toy_vggt_outputs.npz", allow_pickle=False) as viewer:
                np.testing.assert_array_equal(
                    viewer["image_paths"],
                    np.asarray(["toy_view1.png", "toy_view2.png", "toy_view3.png"]),
                )
                self.assertEqual(viewer["extrinsics"].shape, (3, 4, 4))
                self.assertEqual(viewer["intrinsics"].shape, (3, 3, 3))
                np.testing.assert_allclose(viewer["intrinsics"], np.repeat(K[None], 3, axis=0))

    def test_tilted_ellipse_cli_defaults(self) -> None:
        args = build_arg_parser().parse_args([])

        self.assertEqual(tuple(args.ellipse_minor_direction), (1.0, 0.0, 1.0))
        self.assertEqual(args.ellipse_minor_axis_ratio, 0.35)

    def test_tilted_ellipse_mode_axes_and_quarter_cycle_positions(self) -> None:
        ratio = 0.35
        major, minor, phi = _build_elliptical_mode(
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 1.0], dtype=np.float64),
            ratio,
        )

        self.assertAlmostEqual(float(np.linalg.norm(major)), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(minor)), 1.0, places=6)
        self.assertAlmostEqual(float(np.dot(major, minor)), 0.0, places=6)
        np.testing.assert_allclose(
            minor,
            np.asarray([1.0, 0.0, 1.0], dtype=np.float64) / np.sqrt(2.0),
            atol=1e-7,
        )
        self.assertTrue(np.iscomplexobj(phi))
        np.testing.assert_allclose(np.real(phi), major, atol=1e-7)
        np.testing.assert_allclose(np.imag(phi), -ratio * minor, atol=1e-7)

        expected_positions = [major, ratio * minor, -major, -ratio * minor]
        for quarter, expected in enumerate(expected_positions):
            with self.subTest(quarter=quarter):
                phase = quarter * 0.5 * np.pi
                position = np.real(phi * np.exp(1j * phase))
                np.testing.assert_allclose(position, expected, atol=1e-7)

    def test_tilted_ellipse_mode_rejects_invalid_geometry(self) -> None:
        major = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "must not be parallel"):
            _build_elliptical_mode(major, 2.0 * major, 0.35)
        for invalid_ratio in (0.0, -0.1, 1.1, np.nan):
            with self.subTest(ratio=invalid_ratio):
                with self.assertRaisesRegex(ValueError, "minor-axis ratio"):
                    _build_elliptical_mode(
                        major,
                        np.asarray([1.0, 0.0, 1.0], dtype=np.float64),
                        invalid_ratio,
                    )

    def test_complex_mode_error_detects_missing_ellipse_minor_axis(self) -> None:
        major, _minor, ellipse_phi = _build_elliptical_mode(
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 1.0], dtype=np.float64),
            0.35,
        )
        reference = np.broadcast_to(ellipse_phi, (2, 3)).copy()
        recovered = reference.copy()
        recovered[1] = major.astype(np.complex64)

        error = _complex_mode_relative_error(recovered, reference)

        self.assertAlmostEqual(float(error[0]), 0.0, places=7)
        self.assertGreater(float(error[1]), 0.3)

    def test_complex_mode_error_uses_one_shared_phase_alignment(self) -> None:
        _major, _minor, ellipse_phi = _build_elliptical_mode(
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 1.0], dtype=np.float64),
            0.35,
        )
        reference = np.broadcast_to(ellipse_phi, (2, 3)).copy()
        globally_shifted = reference * np.exp(0.7j)
        incoherent = reference.copy()
        incoherent[0] *= 1j
        incoherent[1] *= -1j

        np.testing.assert_allclose(
            _complex_mode_relative_error(globally_shifted, reference),
            np.zeros(2, dtype=np.float32),
            atol=1e-7,
        )
        self.assertGreater(
            float(np.mean(_complex_mode_relative_error(incoherent, reference))),
            1.0,
        )

    def test_phase_stats_handles_branch_cut_cluster(self) -> None:
        stats = _phase_stats(
            np.asarray([np.pi - 0.1, -np.pi + 0.1], dtype=np.float64)
        )

        self.assertEqual(set(stats), {"count", "mean", "median", "p90", "max"})
        self.assertEqual(stats["count"], 2)
        expected = {
            "mean": np.pi,
            "median": np.pi,
            "p90": np.pi + 0.08,
            "max": np.pi + 0.1,
        }
        for key, expected_value in expected.items():
            with self.subTest(statistic=key):
                value = stats[key]
                self.assertIsNotNone(value)
                self.assertLess(_phase_distance(float(value), expected_value), 1e-12)

    def test_group_diagnostics_uses_circular_phase_stats(self) -> None:
        phases = np.asarray([np.pi - 0.05, -np.pi + 0.05], dtype=np.float64)
        phi = np.zeros((2, 3), dtype=np.complex128)
        phi[:, 0] = np.exp(1j * phases)

        diagnostics = _group_diagnostics(
            "branch_cut",
            np.ones((2,), dtype=bool),
            phi,
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            None,
        )

        phase_stats = diagnostics["phase_rad"]
        self.assertEqual(phase_stats["count"], 2)
        self.assertIsNotNone(phase_stats["mean"])
        self.assertLess(
            _phase_distance(float(phase_stats["mean"]), np.pi),
            1e-6,
        )

    def test_phase_stats_handles_empty_and_undefined_samples(self) -> None:
        expected_empty = {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
        self.assertEqual(
            _phase_stats(np.asarray([np.nan, np.inf, -np.inf], dtype=np.float64)),
            expected_empty,
        )

        antipodal = _phase_stats(np.asarray([0.0, np.pi], dtype=np.float64))
        self.assertEqual(antipodal["count"], 2)
        for key in ("mean", "median", "p90", "max"):
            self.assertIsNone(antipodal[key])


if __name__ == "__main__":
    unittest.main()
