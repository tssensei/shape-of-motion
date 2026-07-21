from __future__ import annotations

import importlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from preproc import build_observed_gaussian_structure_graph as app


class BuildObservedGaussianStructureGraphTests(unittest.TestCase):
    @staticmethod
    def _points() -> np.ndarray:
        return np.asarray(
            [
                [0.0, 0.0, 2.0],
                [0.2, 0.0, 2.0],
                [0.4, 0.0, 2.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _view_configs(paths: list[Path]) -> dict[Path, SimpleNamespace]:
        return {
            path: SimpleNamespace(
                view_id=f"view{index}",
                K=np.asarray(
                    [[10.0, 0.0, 10.0], [0.0, 10.0, 12.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                world_to_camera=np.eye(4, dtype=np.float32),
            )
            for index, path in enumerate(paths)
        }

    def _reference_arrays(
        self,
        *,
        checkpoint_path: Path,
        view_config_paths: list[Path],
    ) -> dict[str, np.ndarray]:
        return {
            "points_world": self._points(),
            "gaussian_indices": np.arange(3, dtype=np.int32),
            "point_type": np.array("foreground_gaussian_center"),
            "source_checkpoint": np.array(str(checkpoint_path)),
            "obs_point_index": np.asarray([0, 0, 1, 2], dtype=np.int32),
            "obs_view_index": np.asarray([0, 1, 0, 1], dtype=np.int32),
            "obs_contribution_weight": np.asarray(
                [1.0, 0.5, 1.0, 0.0],
                dtype=np.float32,
            ),
            "view_ids": np.asarray(["view0", "view1"]),
            "freq_hz": np.array(0.85, dtype=np.float32),
            "mode_index": np.array(4, dtype=np.int32),
            "pixel_render_acc_min": np.array(0.05, dtype=np.float32),
            "source_view_configs": np.asarray(
                [str(path) for path in view_config_paths]
            ),
        }

    @staticmethod
    def _write_reference(path: Path, arrays: dict[str, np.ndarray]) -> None:
        np.savez_compressed(path, **arrays)

    def test_reference_observation_loader_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "static.ckpt"
            view_config_paths = [root / "view0.json", root / "view1.json"]
            configs = self._view_configs(view_config_paths)
            valid_arrays = self._reference_arrays(
                checkpoint_path=checkpoint_path,
                view_config_paths=view_config_paths,
            )
            reference_path = root / "observations.npz"
            self._write_reference(reference_path, valid_arrays)

            with mock.patch.object(
                app,
                "load_view_config",
                side_effect=lambda path: configs[Path(path)],
            ):
                loaded = app.load_reference_observations(
                    reference_path,
                    input_checkpoint=checkpoint_path,
                    view_config_paths=view_config_paths,
                )
            np.testing.assert_array_equal(
                loaded["obs_contribution_weight"],
                valid_arrays["obs_contribution_weight"],
            )

            malformed_cases = (
                (
                    "missing required field",
                    {
                        key: value
                        for key, value in valid_arrays.items()
                        if key != "source_view_configs"
                    },
                    "missing required fields",
                ),
                (
                    "checkpoint mismatch",
                    {
                        **valid_arrays,
                        "source_checkpoint": np.array("different.ckpt"),
                    },
                    "source_checkpoint",
                ),
                (
                    "noncontiguous Gaussian indices",
                    {
                        **valid_arrays,
                        "gaussian_indices": np.asarray([0, 2, 1], dtype=np.int32),
                    },
                    "contiguous checkpoint order",
                ),
                (
                    "view order mismatch",
                    {
                        **valid_arrays,
                        "view_ids": np.asarray(["view1", "view0"]),
                    },
                    "view_ids",
                ),
                (
                    "view-config provenance mismatch",
                    {
                        **valid_arrays,
                        "source_view_configs": np.asarray(
                            [str(view_config_paths[1]), str(view_config_paths[0])]
                        ),
                    },
                    "source_view_configs",
                ),
                (
                    "negative observation weight",
                    {
                        **valid_arrays,
                        "obs_contribution_weight": np.asarray(
                            [1.0, 0.5, -1.0, 0.0],
                            dtype=np.float32,
                        ),
                    },
                    "finite and non-negative",
                ),
                (
                    "observation point out of range",
                    {
                        **valid_arrays,
                        "obs_point_index": np.asarray([0, 0, 1, 3], dtype=np.int32),
                    },
                    "obs_point_index is out of range",
                ),
            )
            for index, (name, arrays, message) in enumerate(malformed_cases):
                with self.subTest(name=name):
                    malformed_path = root / f"malformed_{index}.npz"
                    self._write_reference(malformed_path, arrays)
                    with mock.patch.object(
                        app,
                        "load_view_config",
                        side_effect=lambda path: configs[Path(path)],
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            app.load_reference_observations(
                                malformed_path,
                                input_checkpoint=checkpoint_path,
                                view_config_paths=view_config_paths,
                            )

    def test_cli_registers_required_inputs_and_graph_parameters(self) -> None:
        parser = app.build_parser()
        args = parser.parse_args(
            [
                "--input-ckpt",
                "static.ckpt",
                "--view-config",
                "view0.json",
                "--view-config",
                "view1.json",
                "--reference-observations",
                "observations.npz",
                "--out-npz",
                "observed_graph.npz",
                "--max-distance",
                "0.008",
                "--max-neighbors",
                "12",
                "--color-mad-multiplier",
                "2.5",
                "--depth-mad-multiplier",
                "3.5",
                "--depth-samples",
                "7",
                "--min-shared-views",
                "2",
            ]
        )
        self.assertEqual(args.input_ckpt, Path("static.ckpt"))
        self.assertEqual(
            args.view_config,
            [Path("view0.json"), Path("view1.json")],
        )
        self.assertEqual(args.reference_observations, Path("observations.npz"))
        self.assertEqual(args.out_npz, Path("observed_graph.npz"))
        self.assertEqual(args.max_distance, 0.008)
        self.assertEqual(args.max_neighbors, 12)
        self.assertEqual(args.color_mad_multiplier, 2.5)
        self.assertEqual(args.depth_mad_multiplier, 3.5)
        self.assertEqual(args.depth_samples, 7)
        self.assertEqual(args.min_shared_views, 2)

        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--input-ckpt",
                    "static.ckpt",
                    "--view-config",
                    "view0.json",
                    "--reference-observations",
                    "observations.npz",
                    "--out-npz",
                    "observed_graph.npz",
                ]
            )

    def test_main_only_builds_and_writes_observed_graph(self) -> None:
        solver_app = importlib.import_module(
            "modal_surface.apps.solve_gaussian_modes"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "static.ckpt"
            view_config_paths = [root / "view0.json", root / "view1.json"]
            reference_path = root / "observations.npz"
            output_path = root / "observed_graph.npz"
            configs = self._view_configs(view_config_paths)
            reference_arrays = self._reference_arrays(
                checkpoint_path=checkpoint_path,
                view_config_paths=view_config_paths,
            )
            self._write_reference(reference_path, reference_arrays)

            points = self._points()
            colors = np.full(points.shape, 0.75, dtype=np.float32)
            depth = np.full((24, 24), 2.0, dtype=np.float32)
            acc = np.ones_like(depth)
            fake_graph = SimpleNamespace(
                counts={
                    "node_count": 2,
                    "single_view_node_count": 1,
                    "multi_view_node_count": 1,
                    "retained_edge_count": 1,
                    "isolated_node_count": 0,
                }
            )

            def fake_writer(
                path: Path,
                _graph: object,
                **_kwargs: object,
            ) -> Path:
                Path(path).write_bytes(b"observed graph artifact")
                return Path(path)

            argv = [
                "build_observed_gaussian_structure_graph.py",
                "--input-ckpt",
                str(checkpoint_path),
                "--view-config",
                str(view_config_paths[0]),
                "--view-config",
                str(view_config_paths[1]),
                "--reference-observations",
                str(reference_path),
                "--out-npz",
                str(output_path),
                "--max-distance",
                "0.008",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    app,
                    "load_view_config",
                    side_effect=lambda path: configs[Path(path)],
                ),
                mock.patch.object(
                    app,
                    "load_fg_pixel_candidate_inputs_from_checkpoint",
                    return_value=(
                        points,
                        np.ones_like(points),
                        np.tile(
                            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                            (points.shape[0], 1),
                        ),
                        np.ones((points.shape[0],), dtype=np.float32),
                        colors,
                        [depth, depth],
                        [acc, acc],
                    ),
                ) as checkpoint_loader,
                mock.patch.object(
                    app,
                    "build_observed_structure_graph",
                    return_value=fake_graph,
                ) as graph_builder,
                mock.patch.object(
                    app,
                    "write_observed_structure_graph",
                    side_effect=fake_writer,
                ) as graph_writer,
                mock.patch.object(
                    solver_app,
                    "optimize_multi_view_staged",
                    side_effect=AssertionError("standalone graph builder invoked solve"),
                ) as staged_solver,
                redirect_stdout(io.StringIO()),
            ):
                app.main()

            self.assertTrue(output_path.is_file())
            checkpoint_loader.assert_called_once_with(
                str(checkpoint_path),
                [str(path) for path in view_config_paths],
            )
            graph_builder.assert_called_once()
            build_kwargs = graph_builder.call_args.kwargs
            np.testing.assert_array_equal(
                build_kwargs["obs_point_index"],
                reference_arrays["obs_point_index"],
            )
            np.testing.assert_array_equal(
                build_kwargs["obs_view_index"],
                reference_arrays["obs_view_index"],
            )
            np.testing.assert_array_equal(
                build_kwargs["obs_weights"],
                reference_arrays["obs_contribution_weight"],
            )
            self.assertEqual(build_kwargs["config"].max_distance, 0.008)
            self.assertEqual(build_kwargs["config"].render_acc_min, 0.05)

            graph_writer.assert_called_once()
            write_kwargs = graph_writer.call_args.kwargs
            self.assertEqual(write_kwargs["mode_index"], 4)
            self.assertAlmostEqual(write_kwargs["freq_hz"], 0.85, places=6)
            self.assertEqual(write_kwargs["source_checkpoint"], str(checkpoint_path))
            self.assertEqual(
                write_kwargs["source_observation_path"],
                str(reference_path),
            )
            self.assertEqual(write_kwargs["num_foreground_gaussians"], 3)
            staged_solver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
