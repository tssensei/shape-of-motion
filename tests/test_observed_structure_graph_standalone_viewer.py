from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from modal_surface.observed_structure_graph import (
    ObservedStructureGraphConfig,
    build_observed_structure_graph,
    write_observed_structure_graph,
)
from preproc.vis_observed_structure_graph import (
    _load_observed_graph_archive,
    _validate_args,
    anchor_residual_fraction_colors,
    anchor_residual_source_colors,
    anchor_residual_view_colors,
    build_parser,
    center_world_points,
    gaussian_covariances,
    graph_component_colors,
    graph_scalar_colors,
    load_anchor_residual_diagnostics,
    load_gaussian_visualization_sidecar,
    load_observation_coverage,
    observation_coverage_colors,
    observed_world_center,
    scale_gaussian_opacities,
    stable_uniform_edge_indices,
    stable_uniform_indices,
    staged_partial_mask,
)


class StandaloneObservedGraphViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.K = np.asarray(
            [[10.0, 0.0, 10.0], [0.0, 10.0, 12.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.points = np.asarray(
            [
                [(u - self.K[0, 2]) * 2.0 / self.K[0, 0], 0.0, 2.0]
                for u in (6.0, 8.0, 10.0, 12.0)
            ],
            dtype=np.float32,
        )
        self.colors = np.full(self.points.shape, 0.75, dtype=np.float32)
        self.depth = np.full((24, 24), 2.0, dtype=np.float32)

    def _write_graph(self, path: Path) -> Path:
        graph = build_observed_structure_graph(
            points_world=self.points,
            colors_rgb=self.colors,
            obs_point_index=np.asarray([0, 1, 1, 2], dtype=np.int32),
            obs_view_index=np.asarray([0, 0, 1, 1], dtype=np.int32),
            obs_weights=np.ones((4,), dtype=np.float32),
            view_ids=np.asarray(["view0", "view1"]),
            Ks=np.repeat(self.K[None], 2, axis=0),
            world_to_cameras=np.repeat(
                np.eye(4, dtype=np.float32)[None],
                2,
                axis=0,
            ),
            rendered_depths=[self.depth, self.depth],
            rendered_accs=[np.ones_like(self.depth), np.ones_like(self.depth)],
            config=ObservedStructureGraphConfig(
                max_neighbors=2,
                max_distance=1.0,
                depth_samples=5,
                min_shared_views=1,
                render_acc_min=0.05,
            ),
        )
        return write_observed_structure_graph(
            path,
            graph,
            mode_index=4,
            freq_hz=0.85,
            source_checkpoint="source.ckpt",
            source_observation_path="observations/mode_004_0p85hz.npz",
            num_foreground_gaussians=self.points.shape[0],
        )

    def _write_sidecar(
        self,
        path: Path,
        *,
        source_checkpoint: str = "source.ckpt",
        offset: float = 0.0,
    ) -> None:
        centers = self.points.copy()
        centers[0, 0] += offset
        half_sqrt = np.float32(np.sqrt(0.5))
        scales = np.full(self.points.shape, 0.01, dtype=np.float32)
        scales[:2] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        quats = np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (self.points.shape[0], 1),
        )
        quats[1] = np.asarray(
            [half_sqrt, 0.0, 0.0, half_sqrt],
            dtype=np.float32,
        )
        np.savez_compressed(
            path,
            version=np.array(1, dtype=np.int32),
            point_type=np.array("static_3dgs_activated_gaussians"),
            source_checkpoint=np.array(source_checkpoint),
            has_background=np.array(False, dtype=bool),
            num_foreground_gaussians=np.array(
                self.points.shape[0], dtype=np.int64
            ),
            fg_gaussian_indices=np.arange(self.points.shape[0], dtype=np.int64),
            fg_centers=centers,
            fg_scales=scales,
            fg_quats_wxyz=quats,
            fg_rgbs=self.colors,
            fg_opacities=np.full(
                (self.points.shape[0], 1),
                0.8,
                dtype=np.float32,
            ),
        )

    def _write_coverage(self, path: Path, *, offset: float = 0.0) -> None:
        points = self.points.copy()
        points[0, 0] += offset
        selected_hits = np.asarray(
            [
                [[1, 0], [1, 1], [0, 1], [0, 0]],
                [[1, 0], [1, 1], [0, 1], [0, 0]],
            ],
            dtype=np.int32,
        )
        point_view_hits = selected_hits[0]
        view_counts = np.asarray([1, 2, 1, 0], dtype=np.int8)
        categories = np.asarray(
            [[0, 3, 0, 0], [0, 3, 0, 0]],
            dtype=np.int8,
        )
        np.savez_compressed(
            path,
            version=np.array(1, dtype=np.int32),
            point_type=np.array("foreground_gaussian_observation_coverage"),
            source_checkpoint=np.array("source.ckpt"),
            reference_observation_path=np.array(
                "observations/mode_004_0p85hz.npz"
            ),
            num_foreground_gaussians=np.array(4, dtype=np.int64),
            gaussian_indices=np.arange(4, dtype=np.int64),
            points_world=points,
            view_ids=np.array(["view0", "view1"]),
            k_values=np.array([4, 8], dtype=np.int32),
            baseline_k=np.array(4, dtype=np.int32),
            baseline_k_index=np.array(0, dtype=np.int32),
            category_names=np.array(
                [
                    "insufficient_preselect_views",
                    "multiview_preselect_contribution_lost",
                    "multiview_positive_topk_lost",
                    "selected_multiview",
                ]
            ),
            preselect_hit_count_by_view=point_view_hits,
            positive_hit_count_by_view=point_view_hits,
            selected_hit_count_by_k_view=selected_hits,
            best_positive_rank_by_view=np.where(
                point_view_hits > 0,
                1,
                -1,
            ).astype(np.int16),
            best_positive_score_by_view=point_view_hits.astype(np.float32),
            best_positive_score_ratio_by_view=point_view_hits.astype(
                np.float32
            ),
            preselect_view_count=view_counts,
            positive_view_count=view_counts,
            selected_view_count_by_k=np.repeat(view_counts[None], 2, axis=0),
            selected_sample_count_by_k=np.repeat(
                view_counts.astype(np.int32)[None],
                2,
                axis=0,
            ),
            category_by_k=categories,
            category_count_by_k=np.array(
                [[3, 0, 0, 1], [3, 0, 0, 1]], dtype=np.int64
            ),
            preselect_view_count_histogram=np.array([1, 2, 1], dtype=np.int64),
            positive_view_count_histogram=np.array([1, 2, 1], dtype=np.int64),
            selected_view_count_histogram_by_k=np.array(
                [[1, 2, 1], [1, 2, 1]], dtype=np.int64
            ),
            mask_erode_iters=np.array(1, dtype=np.int32),
            pixel_sample_stride=np.array(2, dtype=np.int32),
            pixel_preselect_k=np.array(32, dtype=np.int32),
            pixel_render_acc_min=np.array(0.05, dtype=np.float32),
            pixel_min_contribution=np.array(1.0e-12, dtype=np.float32),
            candidate_method=np.array("rendered_depth_gaussian_contribution"),
            replay_validation=np.array("exact_reference_counts"),
        )

    def _write_residual_diagnostics(self, path: Path) -> None:
        num_points = self.points.shape[0]
        num_views = 2
        point_status = np.asarray([3, 0, 3, 4], dtype=np.int8)
        point_view_mask = np.asarray(
            [[True, False], [True, True], [False, True], [False, False]],
            dtype=bool,
        )
        np.savez_compressed(
            path,
            version=np.array(1, dtype=np.int32),
            point_type=np.array(
                "foreground_gaussian_anchor_residual_decomposition"
            ),
            source_checkpoint=np.array("source.ckpt"),
            source_observation_path=np.array(
                "observations/mode_004_0p85hz.npz"
            ),
            source_solver_diagnostics_path=np.array("diagnostics/mode.npz"),
            num_foreground_gaussians=np.array(num_points, dtype=np.int64),
            gaussian_indices=np.arange(num_points, dtype=np.int64),
            points_world=self.points,
            view_ids=np.array(["view0", "view1"]),
            mode_index=np.array(4, dtype=np.int32),
            freq_hz=np.array(0.85, dtype=np.float32),
            alphas=np.ones((num_views,), dtype=np.complex64),
            alpha_identifiable_mask=np.ones((num_views,), dtype=bool),
            point_observable_rank=np.asarray([2, 3, 2, 0], dtype=np.int8),
            point_condition=np.asarray(
                [1.0, 1.0, 1.0, np.inf],
                dtype=np.float32,
            ),
            point_distinct_view_count=np.asarray(
                [1, 2, 1, 0],
                dtype=np.int32,
            ),
            point_distinct_valid_view_count=np.asarray(
                [1, 2, 1, 0],
                dtype=np.int32,
            ),
            point_precompletion_residual=np.asarray(
                [0.2, 0.05, 0.2, np.inf],
                dtype=np.float32,
            ),
            staged_point_solution_status=point_status,
            anchor_residual_threshold=np.array(0.1, dtype=np.float32),
            anchor_condition_max=np.array(100.0, dtype=np.float32),
            view_sample_count=point_view_mask.astype(np.int32),
            view_effective_weight=point_view_mask.astype(np.float32),
            view_mode_mean=np.ones(
                (num_points, num_views, 2), dtype=np.complex64
            ),
            view_signal_energy=np.ones(
                (num_points, num_views), dtype=np.float32
            ),
            view_within_sse=np.full(
                (num_points, num_views), 0.25, dtype=np.float32
            ),
            view_cross_sse=np.full(
                (num_points, num_views), 0.75, dtype=np.float32
            ),
            view_within_residual=np.full(
                (num_points, num_views), 0.5, dtype=np.float32
            ),
            view_cross_residual=np.full(
                (num_points, num_views), np.sqrt(0.75), dtype=np.float32
            ),
            point_effective_weight=np.full((num_points,), 2.0, dtype=np.float32),
            point_signal_energy=np.full((num_points,), 2.0, dtype=np.float32),
            point_signal_rms=np.ones((num_points,), dtype=np.float32),
            point_within_sse=np.full((num_points,), 0.5, dtype=np.float32),
            point_cross_sse=np.full((num_points,), 1.5, dtype=np.float32),
            point_total_sse=np.full((num_points,), 2.0, dtype=np.float32),
            point_within_residual=np.full(
                (num_points,), 0.5, dtype=np.float32
            ),
            point_cross_residual=np.full(
                (num_points,), np.sqrt(0.75), dtype=np.float32
            ),
            point_replayed_total_residual=np.ones(
                (num_points,), dtype=np.float32
            ),
            point_within_fraction=np.linspace(
                0.0, 1.0, num_points, dtype=np.float32
            ),
            point_worst_within_view=np.array([0, 1, -1, 0], dtype=np.int8),
            point_worst_cross_view=np.array([1, 0, -1, 1], dtype=np.int8),
            selected_multiview_mask=np.asarray(
                [False, True, False, False],
                dtype=bool,
            ),
            residual_candidate_mask=np.asarray(
                [False, True, False, False],
                dtype=bool,
            ),
            residual_rejected_mask=np.zeros((num_points,), dtype=bool),
            anchor_mask=point_status == 0,
            low_modal_energy_mask=np.zeros((num_points,), dtype=bool),
            residual_source_class=np.asarray([0, 1, 0, 0], dtype=np.int8),
            residual_source_names=np.array(
                [
                    "other",
                    "accepted_anchor",
                    "low_modal_energy",
                    "within_view_dominated",
                    "cross_view_dominated",
                    "mixed",
                ]
            ),
            low_modal_energy_threshold=np.array(np.nan, dtype=np.float32),
            dominance_ratio=np.array(2.0, dtype=np.float32),
            mad_scale=np.array(1.4826, dtype=np.float32),
            effective_weight_method=np.array(
                "contribution_divided_by_point_view_multiplicity"
            ),
            decomposition_method=np.array(
                "exact_weighted_point_view_mean_sse_identity"
            ),
            replay_validation=np.array("point_precompletion_residual_exact"),
        )

    def test_direct_artifact_and_display_helpers_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = _load_observed_graph_archive(
                self._write_graph(Path(tmp) / "observed_graph.npz")
            )
        self.assertEqual(graph.mode_index, 4)
        self.assertEqual(
            graph.source_observation_path,
            "observations/mode_004_0p85hz.npz",
        )
        self.assertEqual(graph.view_ids, ("view0", "view1"))
        np.testing.assert_array_equal(graph.node_gaussian_indices, [0, 1, 2])
        np.testing.assert_array_equal(
            graph.node_observed_view_mask,
            [[True, False], [True, True], [False, True]],
        )
        np.testing.assert_array_equal(graph.edge_index, [[0, 1], [1, 2]])
        np.testing.assert_array_equal(
            stable_uniform_edge_indices(10, 4),
            [0, 2, 5, 7],
        )
        np.testing.assert_array_equal(stable_uniform_indices(10, 4), [0, 2, 5, 7])
        component_colors = graph_component_colors(
            np.array([0, 1, 0], dtype=np.int32)
        )
        np.testing.assert_array_equal(component_colors[0], component_colors[2])
        np.testing.assert_array_equal(
            graph_scalar_colors(np.array([2.0, 2.0], dtype=np.float32)),
            np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        )

    def test_malformed_indices_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = self._write_graph(Path(tmp) / "observed_graph.npz")
            with np.load(graph_path, allow_pickle=False) as archive:
                malformed = {name: archive[name] for name in archive.files}
            malformed["node_gaussian_indices"] = np.asarray(
                [1, 0, 2],
                dtype=np.int32,
            )
            np.savez_compressed(graph_path, **malformed)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                _load_observed_graph_archive(graph_path)

    def test_sidecar_covariance_center_and_graph_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = _load_observed_graph_archive(
                self._write_graph(root / "observed_graph.npz")
            )
            sidecar_path = root / "gaussians.npz"
            self._write_sidecar(sidecar_path)
            gaussians = load_gaussian_visualization_sidecar(
                sidecar_path,
                (graph,),
            )
        self.assertIsNone(gaussians.background)
        center = observed_world_center((graph,))
        np.testing.assert_allclose(center, [-0.4, 0.0, 2.0])
        centered_nodes = center_world_points(graph.node_points_world, center)
        np.testing.assert_allclose(
            center_world_points(
                gaussians.foreground.centers[graph.node_gaussian_indices],
                center,
            ),
            centered_nodes,
        )
        np.testing.assert_allclose(
            center_world_points(graph.node_points_world[graph.edge_index], center),
            centered_nodes[graph.edge_index],
        )
        np.testing.assert_allclose(
            gaussians.foreground.covariances[0],
            np.diag([1.0, 4.0, 9.0]),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            gaussians.foreground.covariances[1],
            np.diag([4.0, 1.0, 9.0]),
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            gaussian_covariances(
                np.ones((1, 3), dtype=np.float32),
                np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            ),
            np.eye(3, dtype=np.float32)[None],
        )

    def test_sidecar_provenance_and_node_mismatch_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = _load_observed_graph_archive(
                self._write_graph(root / "observed_graph.npz")
            )
            sidecar_path = root / "gaussians.npz"
            self._write_sidecar(sidecar_path, source_checkpoint="other.ckpt")
            with self.assertRaisesRegex(ValueError, "source_checkpoint"):
                load_gaussian_visualization_sidecar(sidecar_path, (graph,))
            self._write_sidecar(sidecar_path, offset=1.0e-3)
            with self.assertRaisesRegex(ValueError, "centers do not match"):
                load_gaussian_visualization_sidecar(sidecar_path, (graph,))

    def test_gaussian_opacity_multiplier_uses_immutable_source_values(self) -> None:
        source = np.array([[0.2], [0.8]], dtype=np.float32)
        scaled = scale_gaussian_opacities(source, 0.25)
        np.testing.assert_allclose(scaled, [[0.05], [0.2]])
        np.testing.assert_allclose(source, [[0.2], [0.8]])
        with self.assertRaisesRegex(ValueError, "multiplier"):
            scale_gaussian_opacities(source, 1.01)
        with self.assertRaisesRegex(ValueError, "shape"):
            scale_gaussian_opacities(source[:, 0], 0.5)

    def test_observation_coverage_loading_colors_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = _load_observed_graph_archive(
                self._write_graph(root / "observed_graph.npz")
            )
            sidecar_path = root / "gaussians.npz"
            coverage_path = root / "coverage.npz"
            self._write_sidecar(sidecar_path)
            self._write_coverage(coverage_path)
            gaussians = load_gaussian_visualization_sidecar(sidecar_path, (graph,))
            coverage = load_observation_coverage(
                coverage_path,
                (graph,),
                gaussians,
            )
            np.testing.assert_array_equal(coverage.k_values, [4, 8])
            np.testing.assert_allclose(
                observation_coverage_colors(coverage.category_by_k[0]),
                [
                    [0.45, 0.45, 0.45],
                    [0.0, 0.45, 1.0],
                    [0.45, 0.45, 0.45],
                    [0.45, 0.45, 0.45],
                ],
            )
            with np.load(coverage_path, allow_pickle=False) as archive:
                valid_coverage = {
                    name: archive[name] for name in archive.files
                }
            malformed = dict(valid_coverage)
            malformed["reference_observation_path"] = np.array(
                "observations/other_mode.npz"
            )
            np.savez_compressed(coverage_path, **malformed)
            with self.assertRaisesRegex(
                ValueError,
                "reference_observation_path",
            ):
                load_observation_coverage(coverage_path, (graph,), gaussians)

            malformed = dict(valid_coverage)
            malformed["view_ids"] = np.array(["view1", "view0"])
            np.savez_compressed(coverage_path, **malformed)
            with self.assertRaisesRegex(ValueError, "view_ids"):
                load_observation_coverage(coverage_path, (graph,), gaussians)

            malformed = dict(valid_coverage)
            malformed_hits = malformed[
                "selected_hit_count_by_k_view"
            ].copy()
            malformed_hits[0, 2] = np.asarray([1, 0], dtype=np.int32)
            malformed["selected_hit_count_by_k_view"] = malformed_hits
            np.savez_compressed(coverage_path, **malformed)
            with self.assertRaisesRegex(ValueError, "observation mask"):
                load_observation_coverage(coverage_path, (graph,), gaussians)

            self._write_coverage(coverage_path, offset=1.0e-3)
            with self.assertRaisesRegex(ValueError, "centers do not match"):
                load_observation_coverage(coverage_path, (graph,), gaussians)

    def test_residual_loading_partial_mask_and_color_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = _load_observed_graph_archive(
                self._write_graph(root / "observed_graph.npz")
            )
            sidecar_path = root / "gaussians.npz"
            residual_path = root / "residuals.npz"
            self._write_sidecar(sidecar_path)
            self._write_residual_diagnostics(residual_path)
            gaussians = load_gaussian_visualization_sidecar(sidecar_path, (graph,))
            residuals = load_anchor_residual_diagnostics(
                residual_path,
                (graph,),
                gaussians,
            )
            with np.load(residual_path, allow_pickle=False) as archive:
                valid_residuals = {
                    name: archive[name] for name in archive.files
                }
            malformed = dict(valid_residuals)
            malformed["source_observation_path"] = np.array(
                "observations/other_mode.npz"
            )
            np.savez_compressed(residual_path, **malformed)
            with self.assertRaisesRegex(ValueError, "source_observation_path"):
                load_anchor_residual_diagnostics(
                    residual_path,
                    (graph,),
                    gaussians,
                )

            malformed = dict(valid_residuals)
            malformed["view_ids"] = np.array(["view1", "view0"])
            np.savez_compressed(residual_path, **malformed)
            with self.assertRaisesRegex(ValueError, "view_ids"):
                load_anchor_residual_diagnostics(
                    residual_path,
                    (graph,),
                    gaussians,
                )

            malformed = dict(valid_residuals)
            malformed["selected_multiview_mask"] = np.zeros(
                (self.points.shape[0],),
                dtype=bool,
            )
            np.savez_compressed(residual_path, **malformed)
            with self.assertRaisesRegex(ValueError, "selected_multiview_mask"):
                load_anchor_residual_diagnostics(
                    residual_path,
                    (graph,),
                    gaussians,
                )
        self.assertEqual(residuals.mode_index, 4)
        np.testing.assert_array_equal(
            residuals.partial_mask,
            [True, False, True, False],
        )
        np.testing.assert_array_equal(
            staged_partial_mask(np.array([0, 3, 5], dtype=np.int8)),
            [False, True, False],
        )
        with self.assertRaisesRegex(ValueError, "unknown value"):
            staged_partial_mask(np.array([8], dtype=np.int8))
        np.testing.assert_allclose(
            anchor_residual_fraction_colors(
                np.array([0.0, 0.5, 1.0], dtype=np.float32)
            ),
            [[0.0, 0.2, 1.0], [0.5, 0.2, 0.5], [1.0, 0.2, 0.0]],
        )
        np.testing.assert_allclose(
            anchor_residual_source_colors(np.array([1, 3, 4], dtype=np.int8)),
            [[0.0, 0.45, 1.0], [1.0, 0.15, 0.0], [0.0, 0.8, 0.2]],
        )
        view_colors = anchor_residual_view_colors(
            np.array([0, 1, -1], dtype=np.int8)
        )
        np.testing.assert_allclose(view_colors[2], [0.45, 0.45, 0.45])

    def test_cli_requires_observed_graph_and_validates_display_ranges(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--observed-graph-npz",
                "observed_graph.npz",
                "--gaussian-npz",
                "gaussians.npz",
                "--coverage-npz",
                "coverage.npz",
                "--residual-diagnostics-npz",
                "residuals.npz",
            ]
        )
        _validate_args(args)
        self.assertEqual(args.observed_graph_npz, Path("observed_graph.npz"))
        self.assertEqual(args.gaussian_npz, Path("gaussians.npz"))
        self.assertEqual(args.observed_point_size, 0.0009)
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--graph-npz",
                    "anchor_graph.npz",
                ]
            )
        args = parser.parse_args(
            [
                "--observed-graph-npz",
                "observed_graph.npz",
                "--line-width",
                "0.0",
            ]
        )
        with self.assertRaisesRegex(ValueError, "line-width"):
            _validate_args(args)
        args = parser.parse_args(
            [
                "--observed-graph-npz",
                "observed_graph.npz",
                "--gaussian-scale",
                "3.1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "gaussian-scale"):
            _validate_args(args)


if __name__ == "__main__":
    unittest.main()
