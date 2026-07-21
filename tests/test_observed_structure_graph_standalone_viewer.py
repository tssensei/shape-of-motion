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
from preproc.vis_anchor_structure_graph import (
    _load_observed_graph_archive,
    _validate_args,
    build_parser,
    center_world_points,
    load_gaussian_visualization_sidecar,
    observed_world_center,
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

    def _write_sidecar(self, path: Path, *, offset: float = 0.0) -> None:
        centers = self.points.copy()
        centers[0, 0] += offset
        np.savez_compressed(
            path,
            version=np.array(1, dtype=np.int32),
            point_type=np.array("static_3dgs_activated_gaussians"),
            source_checkpoint=np.array("source.ckpt"),
            has_background=np.array(False, dtype=bool),
            num_foreground_gaussians=np.array(self.points.shape[0], dtype=np.int64),
            fg_gaussian_indices=np.arange(self.points.shape[0], dtype=np.int64),
            fg_centers=centers,
            fg_scales=np.full(self.points.shape, 0.01, dtype=np.float32),
            fg_quats_wxyz=np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (self.points.shape[0], 1),
            ),
            fg_rgbs=self.colors,
            fg_opacities=np.full(
                (self.points.shape[0], 1),
                0.8,
                dtype=np.float32,
            ),
        )

    def test_direct_artifact_center_and_sidecar_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_path = self._write_graph(root / "observed_graph.npz")
            sidecar_path = root / "gaussians.npz"
            self._write_sidecar(sidecar_path)
            graph = _load_observed_graph_archive(graph_path)
            gaussians = load_gaussian_visualization_sidecar(
                sidecar_path,
                (graph,),
            )

        np.testing.assert_array_equal(graph.node_gaussian_indices, [0, 1, 2])
        np.testing.assert_array_equal(graph.edge_index, [[0, 1], [1, 2]])
        center = observed_world_center((graph,))
        np.testing.assert_allclose(center, [-0.4, 0.0, 2.0])
        np.testing.assert_allclose(
            center_world_points(graph.node_points_world, center),
            gaussians.foreground.centers[graph.node_gaussian_indices] - center,
        )

    def test_malformed_indices_and_sidecar_positions_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_path = self._write_graph(root / "observed_graph.npz")
            with np.load(graph_path, allow_pickle=False) as archive:
                malformed = {name: archive[name] for name in archive.files}
            malformed["node_gaussian_indices"] = np.asarray(
                [1, 0, 2],
                dtype=np.int32,
            )
            np.savez_compressed(graph_path, **malformed)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                _load_observed_graph_archive(graph_path)

            graph_path = self._write_graph(graph_path)
            graph = _load_observed_graph_archive(graph_path)
            sidecar_path = root / "gaussians.npz"
            self._write_sidecar(sidecar_path, offset=1.0e-3)
            with self.assertRaisesRegex(ValueError, "centers do not match"):
                load_gaussian_visualization_sidecar(sidecar_path, (graph,))

    def test_cli_accepts_only_one_direct_or_anchor_source(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--observed-graph-npz",
                "observed_graph.npz",
                "--gaussian-npz",
                "gaussians.npz",
            ]
        )
        _validate_args(args)
        self.assertEqual(args.observed_graph_npz, Path("observed_graph.npz"))
        self.assertEqual(args.observed_point_size, 0.0009)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--graph-npz",
                    "anchor_graph.npz",
                    "--observed-graph-npz",
                    "observed_graph.npz",
                ]
            )


if __name__ == "__main__":
    unittest.main()
