from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from modal_surface.observed_structure_graph import (
    LoadedObservedStructureGraph,
    ObservedStructureGraphConfig,
    build_observed_structure_graph,
    load_observed_structure_graph,
    validate_observed_structure_graph_sources,
    write_observed_structure_graph,
)


class ObservedStructureGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.K = np.asarray(
            [[10.0, 0.0, 10.0], [0.0, 10.0, 12.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.world_to_camera = np.eye(4, dtype=np.float32)

    def _points(
        self,
        pixels_u: list[float],
        depths: list[float] | None = None,
    ) -> np.ndarray:
        if depths is None:
            depths = [2.0] * len(pixels_u)
        return np.asarray(
            [
                [(u - self.K[0, 2]) * z / self.K[0, 0], 0.0, z]
                for u, z in zip(pixels_u, depths)
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _observation_rows(
        observed_view_mask: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        point_indices, view_indices = np.nonzero(observed_view_mask)
        return (
            point_indices.astype(np.int32),
            view_indices.astype(np.int32),
            np.ones(point_indices.shape, dtype=np.float32),
        )

    def _build(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        depth_maps: list[np.ndarray],
        observed_view_mask: np.ndarray,
        *,
        max_neighbors: int = 2,
        max_distance: float = 1.0,
    ):
        point_index, view_index, weights = self._observation_rows(
            observed_view_mask
        )
        num_views = observed_view_mask.shape[1]
        return build_observed_structure_graph(
            points_world=points,
            colors_rgb=colors,
            obs_point_index=point_index,
            obs_view_index=view_index,
            obs_weights=weights,
            view_ids=np.asarray([f"view{index}" for index in range(num_views)]),
            Ks=np.repeat(self.K[None], num_views, axis=0),
            world_to_cameras=np.repeat(
                self.world_to_camera[None], num_views, axis=0
            ),
            rendered_depths=depth_maps,
            rendered_accs=[np.ones_like(depth) for depth in depth_maps],
            config=ObservedStructureGraphConfig(
                max_neighbors=max_neighbors,
                max_distance=max_distance,
                depth_samples=5,
                min_shared_views=1,
                render_acc_min=0.05,
            ),
        )

    def test_all_positive_observed_nodes_include_single_view_gaussians(self) -> None:
        points = self._points([6.0, 8.0, 10.0, 12.0, 14.0])
        colors = np.full(points.shape, 0.8, dtype=np.float32)
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        first = build_observed_structure_graph(
            points_world=points,
            colors_rgb=colors,
            obs_point_index=np.asarray([0, 1, 1, 2, 3], dtype=np.int32),
            obs_view_index=np.asarray([0, 0, 1, 1, 0], dtype=np.int32),
            obs_weights=np.asarray([1.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float32),
            view_ids=np.asarray(["view0", "view1"]),
            Ks=np.repeat(self.K[None], 2, axis=0),
            world_to_cameras=np.repeat(self.world_to_camera[None], 2, axis=0),
            rendered_depths=[depth, depth],
            rendered_accs=[np.ones_like(depth), np.ones_like(depth)],
            config=ObservedStructureGraphConfig(
                max_neighbors=2,
                max_distance=1.0,
                depth_samples=5,
                min_shared_views=1,
                render_acc_min=0.05,
            ),
        )
        np.testing.assert_array_equal(first.node_gaussian_indices, [0, 1, 2])
        np.testing.assert_array_equal(
            first.node_observed_view_mask,
            [[True, False], [True, True], [False, True]],
        )
        np.testing.assert_array_equal(first.node_observed_view_count, [1, 2, 1])
        np.testing.assert_array_equal(first.topology.edge_index, [[0, 1], [1, 2]])
        self.assertEqual(first.counts["node_count"], 3)
        self.assertEqual(first.counts["single_view_node_count"], 2)
        self.assertEqual(first.counts["multi_view_node_count"], 1)
        self.assertEqual(first.counts["observation_row_count"], 5)
        self.assertEqual(first.counts["positive_observation_row_count"], 4)
        self.assertEqual(first.counts["zero_weight_observation_row_count"], 1)
        self.assertNotIn("anchor_count", first.counts)
        self.assertNotIn("isolated_anchor_count", first.counts)

    def test_mutual_knn_cutoff_sorting_and_isolation_are_deterministic(self) -> None:
        points = self._points([4.0, 6.0, 8.0, 16.0])
        colors = np.full(points.shape, 0.8, dtype=np.float32)
        observed = np.ones((points.shape[0], 1), dtype=bool)
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        first = self._build(
            points,
            colors,
            [depth],
            observed,
            max_neighbors=2,
            max_distance=0.45,
        )
        second = self._build(
            points,
            colors,
            [depth],
            observed,
            max_neighbors=2,
            max_distance=0.45,
        )
        np.testing.assert_array_equal(first.topology.edge_index, second.topology.edge_index)
        self.assertTrue(
            np.all(first.topology.edge_index[:, 0] < first.topology.edge_index[:, 1])
        )
        self.assertTrue(first.topology.isolated_mask[-1])
        self.assertEqual(first.topology.degree[-1], 0)
        self.assertLessEqual(float(first.topology.edge_distance.max()), 0.45)

    def test_color_rejects_green_leaf_candidate_but_keeps_white_neighbors(self) -> None:
        points = self._points([5.0, 7.0, 9.0, 11.0, 13.0])
        colors = np.ones(points.shape, dtype=np.float32)
        colors[-1] = np.asarray([0.05, 0.5, 0.05], dtype=np.float32)
        observed = np.ones((points.shape[0], 1), dtype=bool)
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        graph = self._build(points, colors, [depth], observed)
        global_edges = graph.node_gaussian_indices[graph.topology.edge_index]
        self.assertGreater(graph.counts["color_rejected_count"], 0)
        self.assertFalse(np.any(global_edges == points.shape[0] - 1))
        self.assertGreater(graph.topology.edge_index.shape[0], 0)

    def test_depth_discontinuity_rejects_same_color_cross_surface_edge(self) -> None:
        points = self._points([6.0, 8.0, 12.0, 14.0])
        colors = np.full(points.shape, 0.7, dtype=np.float32)
        observed = np.ones((points.shape[0], 1), dtype=bool)
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        depth[:, 9:11] = 4.0
        graph = self._build(points, colors, [depth], observed)
        np.testing.assert_array_equal(
            graph.topology.edge_index,
            np.asarray([[0, 1], [2, 3]], dtype=np.int32),
        )
        self.assertEqual(graph.counts["depth_rejected_count"], 1)

    def test_one_shared_view_retains_edge_and_zero_shared_view_rejects(self) -> None:
        points = self._points([6.0, 8.0, 10.0])
        colors = np.full(points.shape, 0.7, dtype=np.float32)
        observed = np.asarray(
            [[True, False], [True, True], [False, True]],
            dtype=bool,
        )
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        graph = self._build(points, colors, [depth, depth], observed)
        np.testing.assert_array_equal(
            graph.topology.edge_index,
            np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            graph.topology.edge_view_support_count,
            np.ones((2,), dtype=np.int32),
        )
        self.assertFalse(
            np.any(np.all(graph.topology.edge_index == [0, 2], axis=1))
        )

    def test_tilted_continuous_surface_and_zero_mad_remain_finite(self) -> None:
        pixels = [6.0, 8.0, 10.0, 12.0]
        depths = [1.8, 1.9, 2.0, 2.1]
        points = self._points(pixels, depths)
        colors = np.full(points.shape, 0.75, dtype=np.float32)
        observed = np.ones((points.shape[0], 1), dtype=bool)
        column_depth = 1.5 + 0.05 * np.arange(24, dtype=np.float32)
        depth = np.repeat(column_depth[None], 24, axis=0)
        graph = self._build(points, colors, [depth], observed)
        self.assertGreater(graph.topology.edge_index.shape[0], 0)
        self.assertTrue(np.isfinite(graph.topology.edge_combined_weight).all())
        self.assertEqual(graph.topology.color_distance_mad, 0.0)

    def test_artifact_uses_independent_node_schema(self) -> None:
        points = self._points([6.0, 8.0, 10.0])
        colors = np.full(points.shape, 0.8, dtype=np.float32)
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        graph = self._build(
            points,
            colors,
            [depth],
            np.ones((points.shape[0], 1), dtype=bool),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_observed_structure_graph(
                Path(tmp) / "observed_graph.npz",
                graph,
                mode_index=4,
                freq_hz=0.85,
                source_checkpoint="source.ckpt",
                source_observation_path="observations/mode.npz",
                num_foreground_gaussians=points.shape[0],
            )
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}

        self.assertEqual(
            arrays["graph_type"].item(),
            "foreground_gaussian_observed_structure_graph",
        )
        self.assertEqual(
            arrays["node_selection"].item(),
            "positive_weight_observation_row",
        )
        self.assertEqual(
            arrays["source_observation_path"].item(),
            "observations/mode.npz",
        )
        self.assertEqual(int(arrays["node_count"]), 3)
        self.assertFalse(any(name.startswith("anchor_") for name in arrays))

    def test_public_loader_and_source_cross_validation(self) -> None:
        points = self._points([6.0, 8.0, 10.0])
        colors = np.full(points.shape, 0.8, dtype=np.float32)
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        obs_point_index = np.arange(points.shape[0], dtype=np.int32)
        obs_view_index = np.zeros(points.shape[0], dtype=np.int32)
        obs_weights = np.ones(points.shape[0], dtype=np.float32)
        graph = self._build(
            points,
            colors,
            [depth],
            np.ones((points.shape[0], 1), dtype=bool),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_observed_structure_graph(
                Path(tmp) / "observed_graph.npz",
                graph,
                mode_index=4,
                freq_hz=0.85,
                source_checkpoint="source.ckpt",
                source_observation_path="observations/mode.npz",
                num_foreground_gaussians=points.shape[0],
            )
            loaded = load_observed_structure_graph(path)
            self.assertIsInstance(loaded, LoadedObservedStructureGraph)
            np.testing.assert_array_equal(
                loaded.graph.topology.edge_index,
                graph.topology.edge_index,
            )
            validate_observed_structure_graph_sources(
                loaded,
                points_world=points,
                gaussian_indices=np.arange(points.shape[0], dtype=np.int32),
                source_checkpoint="source.ckpt",
                source_observation_path="observations/mode.npz",
                mode_index=4,
                freq_hz=0.85,
                view_ids=np.asarray(["view0"]),
                obs_point_index=obs_point_index,
                obs_view_index=obs_view_index,
                obs_weights=obs_weights,
            )
            with np.load(path, allow_pickle=False) as archive:
                graph_arrays = {
                    name: np.asarray(archive[name]) for name in archive.files
                }
            corruptions = {
                "threshold": (
                    "color_distance_threshold",
                    graph_arrays["color_distance_threshold"] + 0.25,
                    "adaptive threshold",
                ),
                "distance_weight": (
                    "edge_distance_weight",
                    graph_arrays["edge_distance_weight"] * 2.0,
                    "edge_distance_weight",
                ),
                "color_weight": (
                    "edge_color_weight",
                    graph_arrays["edge_color_weight"] * 0.5,
                    "edge_color_weight",
                ),
                "depth_score": (
                    "edge_depth_score",
                    graph_arrays["edge_depth_score"] * 0.5,
                    "edge_depth_score",
                ),
            }
            for label, (field, value, message) in corruptions.items():
                with self.subTest(corruption=label):
                    corrupted = dict(graph_arrays)
                    corrupted[field] = value
                    corrupted_path = Path(tmp) / f"corrupted_{label}.npz"
                    np.savez_compressed(corrupted_path, **corrupted)
                    with self.assertRaisesRegex(ValueError, message):
                        load_observed_structure_graph(corrupted_path)
            with self.assertRaisesRegex(ValueError, "nodes do not match"):
                validate_observed_structure_graph_sources(
                    loaded,
                    points_world=points,
                    gaussian_indices=np.arange(points.shape[0], dtype=np.int32),
                    source_checkpoint="source.ckpt",
                    source_observation_path="observations/mode.npz",
                    mode_index=4,
                    freq_hz=0.85,
                    view_ids=np.asarray(["view0"]),
                    obs_point_index=obs_point_index,
                    obs_view_index=obs_view_index,
                    obs_weights=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
                )

    def test_negative_observation_weight_fails_fast(self) -> None:
        points = self._points([8.0, 10.0])
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_observed_structure_graph(
                points_world=points,
                colors_rgb=np.full(points.shape, 0.8, dtype=np.float32),
                obs_point_index=np.asarray([0], dtype=np.int32),
                obs_view_index=np.asarray([0], dtype=np.int32),
                obs_weights=np.asarray([-1.0], dtype=np.float32),
                view_ids=np.asarray(["view0"]),
                Ks=self.K[None],
                world_to_cameras=self.world_to_camera[None],
                rendered_depths=[depth],
                rendered_accs=[np.ones_like(depth)],
                config=ObservedStructureGraphConfig(
                    max_neighbors=2,
                    max_distance=1.0,
                ),
            )


if __name__ == "__main__":
    unittest.main()
