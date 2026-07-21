from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preproc.vis_anchor_structure_graph import (
    _validate_args,
    anchor_residual_fraction_colors,
    anchor_residual_source_colors,
    anchor_residual_view_colors,
    anchor_graph_component_colors,
    anchor_graph_scalar_colors,
    anchor_world_center,
    build_parser,
    center_world_points,
    gaussian_covariances,
    load_anchor_graph_source,
    load_anchor_graphs_from_manifest,
    load_anchor_residual_diagnostics,
    load_gaussian_visualization_sidecar,
    load_observation_coverage,
    observation_coverage_colors,
    scale_gaussian_opacities,
    staged_partial_mask,
    stable_uniform_edge_indices,
    stable_uniform_indices,
)


class StandaloneAnchorGraphViewerTests(unittest.TestCase):
    def _write_graph(self, path: Path) -> None:
        arrays = {
            "version": np.array(1, dtype=np.int32),
            "mode_index": np.array(4, dtype=np.int32),
            "freq_hz": np.array(0.85, dtype=np.float32),
            "source_checkpoint": np.array("source.ckpt"),
            "num_foreground_gaussians": np.array(3, dtype=np.int32),
            "anchor_gaussian_indices": np.array([0, 1, 2], dtype=np.int32),
            "anchor_points_world": np.array(
                [[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.02, 0.0, 0.0]],
                dtype=np.float32,
            ),
            "anchor_colors_rgb": np.array(
                [[1.0, 1.0, 1.0], [0.9, 0.9, 0.9], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            "anchor_observed_view_mask": np.array(
                [[True, False], [True, True], [False, True]],
                dtype=bool,
            ),
            "edge_index": np.array([[0, 1]], dtype=np.int32),
            "edge_distance": np.array([0.004], dtype=np.float32),
            "edge_distance_weight": np.array([250.0], dtype=np.float32),
            "edge_color_distance": np.array([0.1], dtype=np.float32),
            "edge_color_weight": np.array([0.9], dtype=np.float32),
            "edge_depth_score": np.array([0.8], dtype=np.float32),
            "edge_combined_weight": np.array([180.0], dtype=np.float32),
            "edge_view_support_mask": np.array([[True, False]], dtype=bool),
            "edge_view_support_count": np.array([1], dtype=np.int32),
            "edge_endpoint_gap_by_view": np.array(
                [[0.01, np.nan]],
                dtype=np.float32,
            ),
            "edge_depth_jump_by_view": np.array(
                [[0.02, np.nan]],
                dtype=np.float32,
            ),
            "degree": np.array([1, 1, 0], dtype=np.int32),
            "component_index": np.array([0, 0, 1], dtype=np.int32),
            "component_size": np.array([2, 1], dtype=np.int32),
            "isolated_mask": np.array([False, False, True], dtype=bool),
            "view_ids": np.array(["view1", "view2"]),
            "endpoint_gap_median_by_view": np.array([0.01, np.nan], dtype=np.float32),
            "endpoint_gap_mad_by_view": np.array([0.0, np.nan], dtype=np.float32),
            "endpoint_gap_threshold_by_view": np.array(
                [0.01, np.nan], dtype=np.float32
            ),
            "depth_jump_median_by_view": np.array([0.02, np.nan], dtype=np.float32),
            "depth_jump_mad_by_view": np.array([0.0, np.nan], dtype=np.float32),
            "depth_jump_threshold_by_view": np.array(
                [0.02, np.nan], dtype=np.float32
            ),
            "color_distance_median": np.array(0.1, dtype=np.float32),
            "color_distance_mad": np.array(0.0, dtype=np.float32),
            "color_distance_threshold": np.array(0.1, dtype=np.float32),
            "max_neighbors": np.array(8, dtype=np.int32),
            "max_distance": np.array(0.008, dtype=np.float32),
            "color_mad_multiplier": np.array(3.0, dtype=np.float32),
            "depth_mad_multiplier": np.array(3.0, dtype=np.float32),
            "depth_samples": np.array(5, dtype=np.int32),
            "min_shared_views": np.array(1, dtype=np.int32),
            "render_acc_min": np.array(0.05, dtype=np.float32),
            "epsilon": np.array(1.0e-8, dtype=np.float32),
            "mad_scale": np.array(1.4826, dtype=np.float32),
            "knn_policy": np.array("mutual_knn"),
            "depth_source": np.array("static_3dgs_rendered_depth"),
            "color_space": np.array("opencv_float_rgb_to_lab"),
            "distance_weight_method": np.array("inverse_distance"),
            "color_weight_method": np.array("gaussian_adaptive_threshold"),
            "depth_weight_method": np.array("supporting_view_gaussian_score"),
            "knn_directed_candidate_count": np.array(2, dtype=np.int64),
            "distance_rejected_directed_count": np.array(0, dtype=np.int64),
            "nonmutual_rejected_pair_count": np.array(0, dtype=np.int64),
            "mutual_distance_candidate_count": np.array(1, dtype=np.int64),
            "anchor_count": np.array(3, dtype=np.int64),
            "shared_observed_candidate_view_count": np.array(1, dtype=np.int64),
            "raw_depth_valid_candidate_view_count": np.array(1, dtype=np.int64),
            "endpoint_rejected_candidate_view_count": np.array(0, dtype=np.int64),
            "jump_rejected_candidate_view_count": np.array(0, dtype=np.int64),
            "supporting_candidate_view_count": np.array(1, dtype=np.int64),
            "color_rejected_count": np.array(0, dtype=np.int64),
            "color_retained_count": np.array(1, dtype=np.int64),
            "depth_rejected_count": np.array(0, dtype=np.int64),
            "depth_retained_count": np.array(1, dtype=np.int64),
            "retained_edge_count": np.array(1, dtype=np.int64),
            "isolated_anchor_count": np.array(1, dtype=np.int64),
            "component_count": np.array(2, dtype=np.int64),
        }
        np.savez_compressed(path, **arrays)

    def _write_manifest(self, path: Path, graph_path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "source_checkpoint": "source.ckpt",
                    "parameters": {
                        "anchor_graph_enabled": True,
                        "anchor_graph_version": 1,
                        "anchor_graph_max_neighbors": 8,
                        "anchor_graph_max_distance": 0.008,
                        "anchor_graph_color_mad_multiplier": 3.0,
                        "anchor_graph_depth_mad_multiplier": 3.0,
                        "anchor_graph_depth_samples": 5,
                        "anchor_graph_min_shared_views": 1,
                        "anchor_graph_render_acc_min": 0.05,
                        "anchor_graph_epsilon": 1.0e-8,
                    },
                    "modes": [
                        {
                            "mode_index": 4,
                            "freq_hz": 0.85,
                            "anchor_graph_path": graph_path.name,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def _write_gaussian_sidecar(
        self,
        path: Path,
        *,
        source_checkpoint: str = "source.ckpt",
        center_offset: float = 0.0,
    ) -> None:
        centers = np.array(
            [[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.02, 0.0, 0.0]],
            dtype=np.float32,
        )
        centers[0, 0] += center_offset
        half_sqrt = np.float32(np.sqrt(0.5))
        np.savez_compressed(
            path,
            version=np.array(1, dtype=np.int32),
            point_type=np.array("static_3dgs_activated_gaussians"),
            source_checkpoint=np.array(source_checkpoint),
            has_background=np.array(False, dtype=bool),
            num_foreground_gaussians=np.array(3, dtype=np.int64),
            fg_gaussian_indices=np.arange(3, dtype=np.int64),
            fg_centers=centers,
            fg_scales=np.tile(
                np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                (3, 1),
            ),
            fg_quats_wxyz=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [half_sqrt, 0.0, 0.0, half_sqrt],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            fg_rgbs=np.array(
                [[1.0, 1.0, 1.0], [0.9, 0.9, 0.9], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            fg_opacities=np.full((3, 1), 0.8, dtype=np.float32),
        )

    def _write_coverage(
        self,
        path: Path,
        *,
        source_checkpoint: str = "source.ckpt",
        center_offset: float = 0.0,
    ) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.02, 0.0, 0.0]],
            dtype=np.float32,
        )
        points[0, 0] += center_offset
        categories = np.array([[0, 2, 3], [0, 3, 3]], dtype=np.int8)
        np.savez_compressed(
            path,
            version=np.array(1, dtype=np.int32),
            point_type=np.array("foreground_gaussian_observation_coverage"),
            source_checkpoint=np.array(source_checkpoint),
            reference_observation_path=np.array("observations/mode.npz"),
            num_foreground_gaussians=np.array(3, dtype=np.int64),
            gaussian_indices=np.arange(3, dtype=np.int64),
            points_world=points,
            view_ids=np.array(["view1", "view2"]),
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
            preselect_hit_count_by_view=np.ones((3, 2), dtype=np.int32),
            positive_hit_count_by_view=np.ones((3, 2), dtype=np.int32),
            selected_hit_count_by_k_view=np.ones((2, 3, 2), dtype=np.int32),
            best_positive_rank_by_view=np.ones((3, 2), dtype=np.int16),
            best_positive_score_by_view=np.ones((3, 2), dtype=np.float32),
            best_positive_score_ratio_by_view=np.ones((3, 2), dtype=np.float32),
            preselect_view_count=np.array([1, 2, 2], dtype=np.int8),
            positive_view_count=np.array([1, 2, 2], dtype=np.int8),
            selected_view_count_by_k=np.array(
                [[1, 1, 2], [1, 2, 2]], dtype=np.int8
            ),
            selected_sample_count_by_k=np.ones((2, 3), dtype=np.int32),
            category_by_k=categories,
            category_count_by_k=np.array(
                [[1, 0, 1, 1], [1, 0, 0, 2]], dtype=np.int64
            ),
            preselect_view_count_histogram=np.array([0, 1, 2], dtype=np.int64),
            positive_view_count_histogram=np.array([0, 1, 2], dtype=np.int64),
            selected_view_count_histogram_by_k=np.array(
                [[0, 2, 1], [0, 1, 2]], dtype=np.int64
            ),
            mask_erode_iters=np.array(1, dtype=np.int32),
            pixel_sample_stride=np.array(2, dtype=np.int32),
            pixel_preselect_k=np.array(32, dtype=np.int32),
            pixel_render_acc_min=np.array(0.05, dtype=np.float32),
            pixel_min_contribution=np.array(1.0e-12, dtype=np.float32),
            candidate_method=np.array("rendered_depth_gaussian_contribution"),
            replay_validation=np.array("exact_reference_counts"),
        )

    def _write_residual_diagnostics(
        self,
        path: Path,
        *,
        source_checkpoint: str = "source.ckpt",
    ) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.02, 0.0, 0.0]],
            dtype=np.float32,
        )
        num_points = 3
        num_views = 2
        np.savez_compressed(
            path,
            version=np.array(1, dtype=np.int32),
            point_type=np.array(
                "foreground_gaussian_anchor_residual_decomposition"
            ),
            source_checkpoint=np.array(source_checkpoint),
            source_observation_path=np.array("observations/mode.npz"),
            source_solver_diagnostics_path=np.array("diagnostics/mode.npz"),
            num_foreground_gaussians=np.array(num_points, dtype=np.int64),
            gaussian_indices=np.arange(num_points, dtype=np.int64),
            points_world=points,
            view_ids=np.array(["view1", "view2"]),
            mode_index=np.array(4, dtype=np.int32),
            freq_hz=np.array(0.85, dtype=np.float32),
            alphas=np.ones((num_views,), dtype=np.complex64),
            alpha_identifiable_mask=np.ones((num_views,), dtype=bool),
            point_observable_rank=np.full((num_points,), 3, dtype=np.int8),
            point_condition=np.ones((num_points,), dtype=np.float32),
            point_distinct_view_count=np.full((num_points,), 2, dtype=np.int32),
            point_distinct_valid_view_count=np.full(
                (num_points,), 2, dtype=np.int32
            ),
            point_precompletion_residual=np.full(
                (num_points,), 0.05, dtype=np.float32
            ),
            staged_point_solution_status=np.zeros((num_points,), dtype=np.int8),
            anchor_residual_threshold=np.array(0.1, dtype=np.float32),
            anchor_condition_max=np.array(100.0, dtype=np.float32),
            view_sample_count=np.ones((num_points, num_views), dtype=np.int32),
            view_effective_weight=np.ones(
                (num_points, num_views), dtype=np.float32
            ),
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
            point_within_fraction=np.array(
                [0.0, 0.5, 1.0], dtype=np.float32
            ),
            point_worst_within_view=np.array([0, 1, -1], dtype=np.int8),
            point_worst_cross_view=np.array([1, 0, -1], dtype=np.int8),
            selected_multiview_mask=np.ones((num_points,), dtype=bool),
            residual_candidate_mask=np.ones((num_points,), dtype=bool),
            residual_rejected_mask=np.zeros((num_points,), dtype=bool),
            anchor_mask=np.ones((num_points,), dtype=bool),
            low_modal_energy_mask=np.zeros((num_points,), dtype=bool),
            residual_source_class=np.ones((num_points,), dtype=np.int8),
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
            graph_path = Path(tmp) / "graph.npz"
            self._write_graph(graph_path)
            loaded = load_anchor_graph_source(
                manifest_path=None,
                graph_path=graph_path,
            )
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].mode_index, 4)
        np.testing.assert_array_equal(loaded[0].edge_index, [[0, 1]])
        np.testing.assert_array_equal(
            stable_uniform_edge_indices(10, 4),
            [0, 2, 5, 7],
        )
        np.testing.assert_array_equal(stable_uniform_indices(10, 4), [0, 2, 5, 7])
        component_colors = anchor_graph_component_colors(
            np.array([0, 1, 0], dtype=np.int32)
        )
        np.testing.assert_array_equal(component_colors[0], component_colors[2])
        np.testing.assert_array_equal(
            anchor_graph_scalar_colors(np.array([2.0, 2.0], dtype=np.float32)),
            np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        )

    def test_manifest_linkage_and_topology_validation_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_path = root / "graph.npz"
            manifest_path = root / "modal_modes_manifest.json"
            self._write_graph(graph_path)
            self._write_manifest(manifest_path, graph_path)
            loaded = load_anchor_graphs_from_manifest(manifest_path)
            self.assertEqual(loaded[0].source_checkpoint, "source.ckpt")

            with np.load(graph_path, allow_pickle=False) as archive:
                malformed = {name: archive[name] for name in archive.files}
            malformed["component_index"] = np.array([0, 1, 2], dtype=np.int32)
            np.savez_compressed(graph_path, **malformed)
            with self.assertRaisesRegex(ValueError, "component_index"):
                load_anchor_graphs_from_manifest(manifest_path)

    def test_gaussian_sidecar_covariance_and_graph_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_path = root / "graph.npz"
            sidecar_path = root / "gaussians.npz"
            self._write_graph(graph_path)
            self._write_gaussian_sidecar(sidecar_path)
            graphs = load_anchor_graph_source(
                manifest_path=None,
                graph_path=graph_path,
            )
            gaussians = load_gaussian_visualization_sidecar(
                sidecar_path,
                graphs,
            )
        self.assertIsNone(gaussians.background)
        world_center = anchor_world_center(graphs)
        np.testing.assert_allclose(world_center, [0.01, 0.0, 0.0])
        centered_anchors = center_world_points(
            graphs[0].anchor_points_world,
            world_center,
        )
        centered_gaussians = center_world_points(
            gaussians.foreground.centers,
            world_center,
        )
        np.testing.assert_allclose(
            centered_gaussians[graphs[0].anchor_gaussian_indices],
            centered_anchors,
        )
        np.testing.assert_allclose(
            center_world_points(
                graphs[0].anchor_points_world[graphs[0].edge_index],
                world_center,
            ),
            centered_anchors[graphs[0].edge_index],
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

    def test_gaussian_sidecar_provenance_and_anchor_mismatch_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_path = root / "graph.npz"
            sidecar_path = root / "gaussians.npz"
            self._write_graph(graph_path)
            graphs = load_anchor_graph_source(
                manifest_path=None,
                graph_path=graph_path,
            )
            self._write_gaussian_sidecar(
                sidecar_path,
                source_checkpoint="other.ckpt",
            )
            with self.assertRaisesRegex(ValueError, "source_checkpoint"):
                load_gaussian_visualization_sidecar(sidecar_path, graphs)
            self._write_gaussian_sidecar(sidecar_path, center_offset=1.0e-3)
            with self.assertRaisesRegex(ValueError, "centers do not match"):
                load_gaussian_visualization_sidecar(sidecar_path, graphs)

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
            graph_path = root / "graph.npz"
            sidecar_path = root / "gaussians.npz"
            coverage_path = root / "coverage.npz"
            self._write_graph(graph_path)
            self._write_gaussian_sidecar(sidecar_path)
            self._write_coverage(coverage_path)
            graphs = load_anchor_graph_source(
                manifest_path=None,
                graph_path=graph_path,
            )
            gaussians = load_gaussian_visualization_sidecar(sidecar_path, graphs)
            coverage = load_observation_coverage(
                coverage_path,
                graphs,
                gaussians,
            )
            np.testing.assert_array_equal(coverage.k_values, [4, 8])
            np.testing.assert_allclose(
                observation_coverage_colors(coverage.category_by_k[0]),
                [[0.45, 0.45, 0.45], [1.0, 0.0, 0.0], [0.0, 0.45, 1.0]],
            )
            self._write_coverage(coverage_path, center_offset=1.0e-3)
            with self.assertRaisesRegex(ValueError, "centers do not match"):
                load_observation_coverage(coverage_path, graphs, gaussians)

    def test_anchor_residual_loading_and_color_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_path = root / "graph.npz"
            sidecar_path = root / "gaussians.npz"
            residual_path = root / "residuals.npz"
            self._write_graph(graph_path)
            self._write_gaussian_sidecar(sidecar_path)
            self._write_residual_diagnostics(residual_path)
            graphs = load_anchor_graph_source(
                manifest_path=None,
                graph_path=graph_path,
            )
            gaussians = load_gaussian_visualization_sidecar(sidecar_path, graphs)
            residuals = load_anchor_residual_diagnostics(
                residual_path,
                graphs,
                gaussians,
            )
        self.assertEqual(residuals.mode_index, 4)
        np.testing.assert_array_equal(residuals.partial_mask, [False, False, False])
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

    def test_cli_requires_one_source_and_validates_display_ranges(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--graph-npz",
                "graph.npz",
                "--gaussian-npz",
                "gaussians.npz",
                "--coverage-npz",
                "coverage.npz",
                "--residual-diagnostics-npz",
                "residuals.npz",
            ]
        )
        _validate_args(args)
        self.assertEqual(args.gaussian_npz, Path("gaussians.npz"))
        self.assertEqual(args.coverage_npz, Path("coverage.npz"))
        self.assertEqual(
            args.residual_diagnostics_npz,
            Path("residuals.npz"),
        )
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--manifest",
                    "manifest.json",
                    "--graph-npz",
                    "graph.npz",
                ]
            )
        args = parser.parse_args(
            ["--graph-npz", "graph.npz", "--line-width", "0.0"]
        )
        with self.assertRaisesRegex(ValueError, "line-width"):
            _validate_args(args)
        args = parser.parse_args(
            ["--graph-npz", "graph.npz", "--gaussian-scale", "3.1"]
        )
        with self.assertRaisesRegex(ValueError, "gaussian-scale"):
            _validate_args(args)


if __name__ == "__main__":
    unittest.main()
