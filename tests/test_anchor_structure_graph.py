from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flow3d.modal_utils import (
    anchor_graph_component_colors,
    load_anchor_structure_graphs,
    load_modal_modes,
    stable_uniform_edge_indices,
)
from modal_surface.anchor_structure_graph import (
    AnchorStructureGraphConfig,
    build_anchor_structure_graph,
    write_anchor_structure_graph,
)


class AnchorStructureGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.K = np.asarray(
            [[10.0, 0.0, 10.0], [0.0, 10.0, 12.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.world_to_camera = np.eye(4, dtype=np.float32)

    def _points(self, pixels_u: list[float], depths: list[float] | None = None) -> np.ndarray:
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
    def _observation_rows(observed_view_mask: np.ndarray) -> tuple[np.ndarray, ...]:
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
        anchor_mask: np.ndarray | None = None,
        max_neighbors: int = 2,
        max_distance: float = 1.0,
    ):
        point_index, view_index, weights = self._observation_rows(observed_view_mask)
        num_views = observed_view_mask.shape[1]
        if anchor_mask is None:
            anchor_mask = np.ones((points.shape[0],), dtype=bool)
        return build_anchor_structure_graph(
            points_world=points,
            colors_rgb=colors,
            anchor_mask=anchor_mask,
            obs_point_index=point_index,
            obs_view_index=view_index,
            obs_weights=weights,
            alpha_identifiable_mask=np.ones((num_views,), dtype=bool),
            view_ids=np.asarray([f"view{index}" for index in range(num_views)]),
            Ks=np.repeat(self.K[None], num_views, axis=0),
            world_to_cameras=np.repeat(
                self.world_to_camera[None], num_views, axis=0
            ),
            rendered_depths=depth_maps,
            rendered_accs=[np.ones_like(depth) for depth in depth_maps],
            config=AnchorStructureGraphConfig(
                max_neighbors=max_neighbors,
                max_distance=max_distance,
                depth_samples=5,
                min_shared_views=1,
                render_acc_min=0.05,
            ),
        )

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
        np.testing.assert_array_equal(first.edge_index, second.edge_index)
        self.assertTrue(np.all(first.edge_index[:, 0] < first.edge_index[:, 1]))
        self.assertTrue(first.isolated_mask[-1])
        self.assertEqual(first.degree[-1], 0)
        self.assertLessEqual(float(first.edge_distance.max()), 0.45)

    def test_color_rejects_green_leaf_candidate_but_keeps_white_neighbors(self) -> None:
        points = self._points([5.0, 7.0, 9.0, 11.0, 13.0])
        colors = np.ones(points.shape, dtype=np.float32)
        colors[-1] = np.asarray([0.05, 0.5, 0.05], dtype=np.float32)
        observed = np.ones((points.shape[0], 1), dtype=bool)
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        graph = self._build(points, colors, [depth], observed)
        global_edges = graph.anchor_gaussian_indices[graph.edge_index]
        self.assertGreater(graph.counts["color_rejected_count"], 0)
        self.assertFalse(np.any(global_edges == points.shape[0] - 1))
        self.assertGreater(graph.edge_index.shape[0], 0)

    def test_depth_discontinuity_rejects_same_color_cross_surface_edge(self) -> None:
        points = self._points([6.0, 8.0, 12.0, 14.0])
        colors = np.full(points.shape, 0.7, dtype=np.float32)
        observed = np.ones((points.shape[0], 1), dtype=bool)
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        depth[:, 9:11] = 4.0
        graph = self._build(points, colors, [depth], observed)
        np.testing.assert_array_equal(
            graph.edge_index,
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
            graph.edge_index,
            np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            graph.edge_view_support_count,
            np.ones((2,), dtype=np.int32),
        )
        self.assertFalse(np.any(np.all(graph.edge_index == [0, 2], axis=1)))

    def test_tilted_continuous_surface_and_zero_mad_remain_finite(self) -> None:
        pixels = [6.0, 8.0, 10.0, 12.0]
        depths = [1.8, 1.9, 2.0, 2.1]
        points = self._points(pixels, depths)
        colors = np.full(points.shape, 0.75, dtype=np.float32)
        observed = np.ones((points.shape[0], 1), dtype=bool)
        column_depth = 1.5 + 0.05 * np.arange(24, dtype=np.float32)
        depth = np.repeat(column_depth[None], 24, axis=0)
        graph = self._build(points, colors, [depth], observed)
        self.assertGreater(graph.edge_index.shape[0], 0)
        self.assertTrue(np.isfinite(graph.edge_combined_weight).all())
        self.assertEqual(graph.color_distance_mad, 0.0)

    def test_modes_induce_separate_anchor_subgraphs(self) -> None:
        points = self._points([6.0, 8.0, 10.0, 12.0])
        colors = np.full(points.shape, 0.75, dtype=np.float32)
        observed = np.ones((points.shape[0], 1), dtype=bool)
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        graph_a = self._build(
            points,
            colors,
            [depth],
            observed,
            anchor_mask=np.asarray([True, True, False, False]),
            max_neighbors=1,
        )
        graph_b = self._build(
            points,
            colors,
            [depth],
            observed,
            anchor_mask=np.asarray([False, False, True, True]),
            max_neighbors=1,
        )
        np.testing.assert_array_equal(graph_a.anchor_gaussian_indices, [0, 1])
        np.testing.assert_array_equal(graph_b.anchor_gaussian_indices, [2, 3])

    def test_artifact_loader_and_viewer_helpers_are_strict_and_deterministic(self) -> None:
        points = self._points([6.0, 8.0, 10.0, 12.0])
        colors = np.full(points.shape, 0.75, dtype=np.float32)
        observed = np.ones((points.shape[0], 1), dtype=bool)
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        graph = self._build(points, colors, [depth], observed)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latent_path = root / "mode.npz"
            np.savez_compressed(
                latent_path,
                points_world=points,
                phi=np.zeros(points.shape, dtype=np.complex64),
            )
            graph_path = write_anchor_structure_graph(
                root / "graph.npz",
                graph,
                mode_index=3,
                freq_hz=2.5,
                source_checkpoint="source.ckpt",
                num_foreground_gaussians=points.shape[0],
            )
            manifest_path = root / "modal_modes_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "source_checkpoint": "source.ckpt",
                        "parameters": {
                            "anchor_graph_enabled": True,
                            "anchor_graph_version": 1,
                            "anchor_graph_max_neighbors": graph.config.max_neighbors,
                            "anchor_graph_max_distance": graph.config.max_distance,
                            "anchor_graph_color_mad_multiplier": (
                                graph.config.color_mad_multiplier
                            ),
                            "anchor_graph_depth_mad_multiplier": (
                                graph.config.depth_mad_multiplier
                            ),
                            "anchor_graph_depth_samples": graph.config.depth_samples,
                            "anchor_graph_min_shared_views": (
                                graph.config.min_shared_views
                            ),
                            "anchor_graph_render_acc_min": graph.config.render_acc_min,
                            "anchor_graph_epsilon": graph.config.epsilon,
                        },
                        "modes": [
                            {
                                "mode_index": 3,
                                "freq_hz": 2.5,
                                "latent_path": latent_path.name,
                                "anchor_graph_path": graph_path.name,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            modes = load_modal_modes(str(manifest_path))
            loaded = load_anchor_structure_graphs(str(manifest_path), modes)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            np.testing.assert_array_equal(loaded[0].edge_index, graph.edge_index)

            with np.load(graph_path, allow_pickle=False) as archive:
                malformed = {name: archive[name] for name in archive.files}
            malformed["anchor_gaussian_indices"] = np.asarray(
                [1, 0, 2, 3], dtype=np.int32
            )
            np.savez_compressed(graph_path, **malformed)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_anchor_structure_graphs(str(manifest_path), modes)
            malformed["anchor_gaussian_indices"] = graph.anchor_gaussian_indices
            malformed["source_checkpoint"] = np.array("different.ckpt")
            np.savez_compressed(graph_path, **malformed)
            with self.assertRaisesRegex(ValueError, "source_checkpoint"):
                load_anchor_structure_graphs(str(manifest_path), modes)

        np.testing.assert_array_equal(
            stable_uniform_edge_indices(10, 4),
            np.asarray([0, 2, 5, 7], dtype=np.int64),
        )
        colors_first = anchor_graph_component_colors(
            np.asarray([0, 1, 0, 2], dtype=np.int32)
        )
        colors_second = anchor_graph_component_colors(
            np.asarray([0, 1, 0, 2], dtype=np.int32)
        )
        np.testing.assert_array_equal(colors_first, colors_second)
        np.testing.assert_array_equal(colors_first[0], colors_first[2])

    def test_legacy_manifest_without_graph_keeps_loader_disabled(self) -> None:
        points = self._points([8.0, 10.0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latent_path = root / "mode.npz"
            np.savez_compressed(
                latent_path,
                points_world=points,
                phi=np.zeros(points.shape, dtype=np.complex64),
            )
            manifest_path = root / "modal_modes_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source_checkpoint": "source.ckpt",
                        "modes": [
                            {
                                "mode_index": 0,
                                "freq_hz": 1.0,
                                "latent_path": latent_path.name,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            modes = load_modal_modes(str(manifest_path))
            self.assertIsNone(
                load_anchor_structure_graphs(str(manifest_path), modes)
            )


if __name__ == "__main__":
    unittest.main()
