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


class ObservedStructureGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.K = np.asarray(
            [[10.0, 0.0, 10.0], [0.0, 10.0, 12.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.points = np.asarray(
            [
                [(u - self.K[0, 2]) * 2.0 / self.K[0, 0], 0.0, 2.0]
                for u in (6.0, 8.0, 10.0, 12.0, 14.0)
            ],
            dtype=np.float32,
        )
        self.colors = np.full(self.points.shape, 0.8, dtype=np.float32)
        self.depth = np.full((24, 24), 2.0, dtype=np.float32)

    def _build(self):
        return build_observed_structure_graph(
            points_world=self.points,
            colors_rgb=self.colors,
            obs_point_index=np.asarray([0, 1, 1, 2, 3], dtype=np.int32),
            obs_view_index=np.asarray([0, 0, 1, 1, 0], dtype=np.int32),
            obs_weights=np.asarray([1.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float32),
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

    def test_all_positive_observed_nodes_include_single_view_gaussians(self) -> None:
        first = self._build()
        second = self._build()
        np.testing.assert_array_equal(first.node_gaussian_indices, [0, 1, 2])
        np.testing.assert_array_equal(
            first.node_observed_view_mask,
            [[True, False], [True, True], [False, True]],
        )
        np.testing.assert_array_equal(first.node_observed_view_count, [1, 2, 1])
        np.testing.assert_array_equal(
            first.topology.edge_index,
            [[0, 1], [1, 2]],
        )
        np.testing.assert_array_equal(
            first.topology.edge_index,
            second.topology.edge_index,
        )
        self.assertEqual(first.counts["node_count"], 3)
        self.assertEqual(first.counts["single_view_node_count"], 2)
        self.assertEqual(first.counts["multi_view_node_count"], 1)
        self.assertEqual(first.counts["observation_row_count"], 5)
        self.assertEqual(first.counts["positive_observation_row_count"], 4)
        self.assertEqual(first.counts["zero_weight_observation_row_count"], 1)
        self.assertNotIn("anchor_count", first.counts)
        self.assertNotIn("isolated_anchor_count", first.counts)

    def test_artifact_uses_independent_node_schema(self) -> None:
        graph = self._build()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_observed_structure_graph(
                Path(tmp) / "observed_graph.npz",
                graph,
                mode_index=4,
                freq_hz=0.85,
                source_checkpoint="source.ckpt",
                source_observation_path="observations/mode.npz",
                num_foreground_gaussians=self.points.shape[0],
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

    def test_negative_observation_weight_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_observed_structure_graph(
                points_world=self.points,
                colors_rgb=self.colors,
                obs_point_index=np.asarray([0], dtype=np.int32),
                obs_view_index=np.asarray([0], dtype=np.int32),
                obs_weights=np.asarray([-1.0], dtype=np.float32),
                view_ids=np.asarray(["view0"]),
                Ks=self.K[None],
                world_to_cameras=np.eye(4, dtype=np.float32)[None],
                rendered_depths=[self.depth],
                rendered_accs=[np.ones_like(self.depth)],
                config=ObservedStructureGraphConfig(
                    max_neighbors=2,
                    max_distance=1.0,
                ),
            )


if __name__ == "__main__":
    unittest.main()
