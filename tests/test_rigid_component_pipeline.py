from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from flow3d.modal_utils import load_gaussian_modal_fields, load_modal_modes
from modal_surface.apps import solve_gaussian_modes as gaussian_solver_app
from modal_surface.gaussian_motion_fill import (
    MOTION_FILL_ROLE_FIXED_ANCHOR,
    MOTION_FILL_ROLE_FREE_VARIABLE,
    MOTION_FILL_ROLE_NAMES,
)
from modal_surface.io import save_npz_compressed_atomic
from modal_surface.observed_structure_graph import (
    LoadedObservedStructureGraph,
    ObservedStructureGraphConfig,
    build_observed_structure_graph,
    load_observed_structure_graph,
    write_observed_structure_graph,
)
from modal_surface.optimization_staged import optimize_multi_view_staged


CORE_LATENT_KEYS = {
    "points_world",
    "phi",
    "gaussian_indices",
    "freq_hz",
    "mode_index",
    "obs_count_per_point",
    "point_type",
    "source_checkpoint",
}
MOTION_FILL_LATENT_KEYS = {
    "motion_fill_role",
    "motion_fill_role_names",
    "completion_mask",
}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


class _ToyRigidArtifacts:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.checkpoint_path = root / "source.ckpt"
        self.view_config_path = root / "view0.json"
        self.modal_path = root / "view0.npz"
        self.observation_path = root / "source_observations.npz"
        self.graph_path = root / "observed_graph_mode0.npz"
        self.frequency = 0.85

        intrinsic = np.asarray(
            [[10.0, 0.0, 10.0], [0.0, 10.0, 12.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        pixels_u = np.asarray([6.0, 8.0, 10.0, 12.0, 14.0])
        self.points = np.column_stack(
            [
                (pixels_u - intrinsic[0, 2]) * 2.0 / intrinsic[0, 0],
                np.zeros_like(pixels_u),
                np.full_like(pixels_u, 2.0),
            ]
        ).astype(np.float32)
        self.observed_indices = np.arange(4, dtype=np.int32)

        centered = (
            self.points[self.observed_indices].astype(np.float64)
            - self.points[self.observed_indices].astype(np.float64).mean(axis=0)
        )
        translation = np.asarray(
            [0.08 + 0.02j, -0.03 + 0.01j, 0.015 - 0.005j],
            dtype=np.complex128,
        )
        rotation = np.asarray(
            [0.01 - 0.004j, -0.008 + 0.003j, 0.006 + 0.005j],
            dtype=np.complex128,
        )
        phi = translation[None] + np.cross(rotation[None], centered)
        jacobian = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        obs_y = np.asarray(
            [jacobian @ point_phi for point_phi in phi], dtype=np.complex64
        )
        obs_count = np.asarray([1, 1, 1, 1, 0], dtype=np.int32)
        contribution = np.ones((self.observed_indices.size,), dtype=np.float32)
        self.observations = {
            "points_world": self.points,
            "gaussian_indices": np.arange(self.points.shape[0], dtype=np.int32),
            "point_type": np.array("foreground_gaussian_center"),
            "source_checkpoint": np.array(str(self.checkpoint_path)),
            "obs_point_index": self.observed_indices.copy(),
            "obs_view_index": np.zeros(self.observed_indices.shape, dtype=np.int32),
            "obs_pixels_xy": np.column_stack(
                [pixels_u[self.observed_indices], np.full(4, 12.0)]
            ).astype(np.float32),
            "obs_y": obs_y,
            "obs_J": np.broadcast_to(
                jacobian, (self.observed_indices.size, 2, 3)
            ).copy(),
            "obs_camera_z": np.full(
                (self.observed_indices.size,), 2.0, dtype=np.float32
            ),
            "obs_count_per_point": obs_count,
            "obs_sample_count_per_point": obs_count.copy(),
            "view_ids": np.asarray(["view0"]),
            "view_image_width": np.asarray([24], dtype=np.int32),
            "view_image_height": np.asarray([24], dtype=np.int32),
            "view_freqs_hz": np.asarray([self.frequency], dtype=np.float32),
            "freq_hz": np.array(self.frequency, dtype=np.float32),
            "mode_index": np.array(0, dtype=np.int32),
            "mask_erode_iters": np.array(1, dtype=np.int32),
            "candidate_point_count": np.array(
                self.points.shape[0], dtype=np.int32
            ),
            "preserved_all_points": np.array(True),
            "pixel_sample_stride": np.array(4, dtype=np.int32),
            "pixel_candidate_k": np.array(4, dtype=np.int32),
            "pixel_preselect_k": np.array(32, dtype=np.int32),
            "pixel_render_acc_min": np.array(0.05, dtype=np.float32),
            "pixel_min_contribution": np.array(1.0e-12, dtype=np.float32),
            "pixel_candidate_method": np.array(
                "rendered_depth_gaussian_contribution"
            ),
            "observations_per_view": np.asarray([4], dtype=np.int32),
            "source_view_configs": np.asarray([str(self.view_config_path)]),
            "source_modal_npzs": np.asarray([str(self.modal_path)]),
            "obs_contribution_weight": contribution,
            "obs_contribution_score": contribution.copy(),
            "obs_contribution_sum": contribution.copy(),
            "obs_surface_pixels_xy": np.column_stack(
                [pixels_u[self.observed_indices], np.full(4, 12.0)]
            ).astype(np.float32),
            "obs_surface_camera_z": np.full(
                (self.observed_indices.size,), 2.0, dtype=np.float32
            ),
        }
        save_npz_compressed_atomic(self.observation_path, self.observations)

        depth = np.full((24, 24), 2.0, dtype=np.float32)
        self.graph = build_observed_structure_graph(
            points_world=self.points,
            colors_rgb=np.full(self.points.shape, 0.8, dtype=np.float32),
            obs_point_index=self.observations["obs_point_index"],
            obs_view_index=self.observations["obs_view_index"],
            obs_weights=self.observations["obs_contribution_weight"],
            view_ids=self.observations["view_ids"],
            Ks=intrinsic[None],
            world_to_cameras=np.eye(4, dtype=np.float32)[None],
            rendered_depths=[depth],
            rendered_accs=[np.ones_like(depth)],
            config=ObservedStructureGraphConfig(
                max_neighbors=2,
                max_distance=0.41,
                depth_samples=5,
                min_shared_views=1,
                render_acc_min=0.05,
            ),
        )
        if not np.any(self.graph.topology.degree > 0):
            raise RuntimeError("Toy observed structure graph has no connected nodes.")
        self.write_graph(self.graph_path, mode_index=0)

    def write_graph(self, path: Path, *, mode_index: int) -> Path:
        return write_observed_structure_graph(
            path,
            self.graph,
            mode_index=mode_index,
            freq_hz=self.frequency,
            source_checkpoint=str(self.checkpoint_path),
            source_observation_path=str(self.observation_path),
            num_foreground_gaussians=self.points.shape[0],
        )

    def checkpoint_inputs(self) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[np.ndarray],
        list[np.ndarray],
    ]:
        quaternions = np.zeros((self.points.shape[0], 4), dtype=np.float32)
        quaternions[:, 0] = 1.0
        return (
            self.points.copy(),
            np.ones((self.points.shape[0], 3), dtype=np.float32),
            quaternions,
            np.ones((self.points.shape[0],), dtype=np.float32),
            np.full((self.points.shape[0], 3), 0.5, dtype=np.float32),
            [],
            [],
        )

    def args(
        self,
        out_dir: Path,
        *,
        solve_method: str = "rigid-components",
        graph_paths: list[Path] | None = None,
        motion_fill: bool = False,
    ) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        gaussian_solver_app.add_arguments(parser)
        argv = [
            "--input-ckpt",
            str(self.checkpoint_path),
            "--view-config",
            str(self.view_config_path),
            "--modal-npz",
            str(self.modal_path),
            "--out-dir",
            str(out_dir),
            "--mode-indices",
            "0",
        ]
        if solve_method == "rigid-components":
            argv.extend(["--solve-method", "rigid-components"])
            for graph_path in graph_paths or [self.graph_path]:
                argv.extend(["--rigid-component-graph", str(graph_path)])
        if motion_fill:
            argv.extend(
                [
                    "--motion-fill",
                    "--motion-fill-k",
                    "2",
                    "--motion-fill-max-distance",
                    "0.41",
                ]
            )
        return parser.parse_args(argv)


class RigidComponentPipelineTests(unittest.TestCase):
    def test_graphs_map_by_mode_and_reject_duplicate_missing_or_extra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            toy = _ToyRigidArtifacts(Path(tmp))
            graph0 = toy.graph_path
            graph1 = toy.write_graph(Path(tmp) / "graph_mode1.npz", mode_index=1)
            graph1_duplicate = toy.write_graph(
                Path(tmp) / "graph_mode1_duplicate.npz", mode_index=1
            )
            graph2 = toy.write_graph(Path(tmp) / "graph_mode2.npz", mode_index=2)

            mapped = gaussian_solver_app._load_rigid_component_graphs(
                [str(graph1), str(graph0)], [0, 1]
            )
            self.assertEqual(list(mapped), [1, 0])
            self.assertEqual(mapped[0].graph_path, graph0)
            self.assertEqual(mapped[1].graph_path, graph1)

            with self.assertRaisesRegex(ValueError, "Duplicate rigid component graph"):
                gaussian_solver_app._load_rigid_component_graphs(
                    [str(graph1), str(graph1_duplicate)], [1]
                )
            with self.assertRaisesRegex(ValueError, r"missing=\[1\]"):
                gaussian_solver_app._load_rigid_component_graphs(
                    [str(graph0)], [0, 1]
                )
            with self.assertRaisesRegex(ValueError, r"extra=\[2\]"):
                gaussian_solver_app._load_rigid_component_graphs(
                    [str(graph0), str(graph2)], [0]
                )

    def test_source_provenance_pixel_parameters_and_graph_observations_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            toy = _ToyRigidArtifacts(Path(tmp))
            loaded = load_observed_structure_graph(toy.graph_path)
            args = toy.args(Path(tmp) / "out")

            def validate(
                graph: LoadedObservedStructureGraph,
                observations: dict[str, np.ndarray],
            ):
                with patch.object(
                    gaussian_solver_app,
                    "load_view_config",
                    return_value=SimpleNamespace(view_id="view0"),
                ):
                    return gaussian_solver_app._validate_rigid_source_observations(
                        graph,
                        observations,
                        args=args,
                        view_config_paths=[str(toy.view_config_path)],
                        modal_npz_paths=[str(toy.modal_path)],
                        freqs_per_view=[
                            np.asarray([toy.frequency], dtype=np.float32)
                        ],
                        mode_index=0,
                        reference_freq=toy.frequency,
                        foreground_points=toy.points,
                    )

            prepared = validate(loaded, toy.observations)
            np.testing.assert_array_equal(prepared.points, toy.points)

            provenance_mutations = (
                (
                    "source_checkpoint",
                    np.array("different.ckpt"),
                    "source_checkpoint",
                ),
                (
                    "source_view_configs",
                    np.asarray(["different.json"]),
                    "source_view_configs",
                ),
                (
                    "source_modal_npzs",
                    np.asarray(["different.npz"]),
                    "source_modal_npzs",
                ),
                (
                    "view_ids",
                    np.asarray(["different_view"]),
                    "view_ids",
                ),
                (
                    "mask_erode_iters",
                    np.array(2, dtype=np.int32),
                    "mask_erode_iters",
                ),
                (
                    "pixel_sample_stride",
                    np.array(8, dtype=np.int32),
                    "pixel_sample_stride",
                ),
                (
                    "pixel_candidate_k",
                    np.array(8, dtype=np.int32),
                    "pixel_candidate_k",
                ),
                (
                    "pixel_preselect_k",
                    np.array(16, dtype=np.int32),
                    "pixel_preselect_k",
                ),
                (
                    "pixel_render_acc_min",
                    np.array(0.1, dtype=np.float32),
                    "pixel_render_acc_min",
                ),
                (
                    "pixel_min_contribution",
                    np.array(1.0e-6, dtype=np.float32),
                    "pixel_min_contribution",
                ),
            )
            for key, value, message in provenance_mutations:
                with self.subTest(key=key):
                    mismatched = dict(toy.observations)
                    mismatched[key] = value
                    with self.assertRaisesRegex(ValueError, message):
                        validate(loaded, mismatched)

            graph_mismatch = dict(toy.observations)
            graph_mismatch["obs_contribution_weight"] = np.asarray(
                [1.0, 1.0, 1.0, 0.0], dtype=np.float32
            )
            with self.assertRaisesRegex(ValueError, "nodes do not match"):
                validate(loaded, graph_mismatch)

    def test_formal_rigid_run_reuses_observations_and_writes_compatible_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toy = _ToyRigidArtifacts(root)
            out_dir = root / "rigid_output"
            args = toy.args(out_dir, motion_fill=True)
            with (
                patch.object(
                    gaussian_solver_app,
                    "load_fg_means_from_checkpoint",
                    return_value=toy.points.copy(),
                ),
                patch.object(
                    gaussian_solver_app,
                    "load_modal_freqs",
                    return_value=[
                        np.asarray([toy.frequency], dtype=np.float32)
                    ],
                ),
                patch.object(
                    gaussian_solver_app,
                    "load_view_config",
                    return_value=SimpleNamespace(view_id="view0"),
                ),
                patch.object(
                    gaussian_solver_app,
                    "build_gaussian_observation_graph",
                    side_effect=AssertionError(
                        "Rigid solve must reuse the graph source observation."
                    ),
                ) as observation_builder,
                patch.object(gaussian_solver_app, "_print_observation_sanity"),
                patch.object(
                    gaussian_solver_app, "write_prepared_solve_visualizations"
                ) as visualizations,
            ):
                gaussian_solver_app.run(args)

            observation_builder.assert_not_called()
            visualizations.assert_called_once()
            manifest_path = out_dir / "modal_modes_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(manifest["parameters"]["solver"], "rigid_components")
            self.assertEqual(
                manifest["parameters"]["rigid_component_observation_policy"],
                "reuse_graph_source_observation_no_rebuild",
            )
            self.assertTrue(manifest["parameters"]["motion_fill_enabled"])
            self.assertEqual(len(manifest["modes"]), 1)
            mode = manifest["modes"][0]
            self.assertEqual(mode["source_observation_path"], str(toy.observation_path))

            copied_observation = out_dir / mode["observation_path"]
            copied_graph = out_dir / mode["rigid_component_graph_path"]
            self.assertEqual(
                copied_observation.read_bytes(), toy.observation_path.read_bytes()
            )
            self.assertEqual(copied_graph.read_bytes(), toy.graph_path.read_bytes())

            latent = _load_npz(out_dir / mode["latent_path"])
            self.assertEqual(
                set(latent), CORE_LATENT_KEYS | MOTION_FILL_LATENT_KEYS
            )
            self.assertEqual(
                tuple(latent["motion_fill_role_names"].astype(str).tolist()),
                MOTION_FILL_ROLE_NAMES,
            )
            roles = latent["motion_fill_role"]
            self.assertTrue(
                set(roles.tolist()).issubset(
                    {
                        MOTION_FILL_ROLE_FIXED_ANCHOR,
                        MOTION_FILL_ROLE_FREE_VARIABLE,
                    }
                )
            )
            loaded_graph = load_observed_structure_graph(toy.graph_path)
            seed_indices = loaded_graph.node_gaussian_indices[
                loaded_graph.graph.topology.degree > 0
            ]
            expected_seed = np.zeros((toy.points.shape[0],), dtype=bool)
            expected_seed[seed_indices] = True
            np.testing.assert_array_equal(
                roles == MOTION_FILL_ROLE_FIXED_ANCHOR, expected_seed
            )
            np.testing.assert_array_equal(
                roles == MOTION_FILL_ROLE_FREE_VARIABLE, ~expected_seed
            )
            self.assertFalse(np.any(latent["completion_mask"][expected_seed]))

            runtime = load_gaussian_modal_fields(
                str(manifest_path), torch.from_numpy(toy.points.copy())
            )
            viewer_modes = load_modal_modes(str(manifest_path))
            self.assertEqual(tuple(runtime.phi_real.shape), (1, 5, 3))
            self.assertEqual(tuple(runtime.phi_imag.shape), (1, 5, 3))
            self.assertIsNotNone(viewer_modes[0].motion_fill_display_class)

    def test_default_staged_run_still_builds_observations_and_uses_staged_solver(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toy = _ToyRigidArtifacts(root)
            out_dir = root / "staged_output"
            args = toy.args(out_dir, solve_method="staged")
            self.assertEqual(args.solve_method, "staged")
            self.assertEqual(args.rigid_component_graph, [])

            def write_observations(*, out_path: str | Path, **_: object) -> Path:
                return save_npz_compressed_atomic(out_path, toy.observations)

            with (
                patch.object(
                    gaussian_solver_app,
                    "load_fg_pixel_candidate_inputs_from_checkpoint",
                    return_value=toy.checkpoint_inputs(),
                ),
                patch.object(
                    gaussian_solver_app,
                    "load_modal_freqs",
                    return_value=[
                        np.asarray([toy.frequency], dtype=np.float32)
                    ],
                ),
                patch.object(
                    gaussian_solver_app,
                    "build_gaussian_observation_graph",
                    side_effect=write_observations,
                ) as observation_builder,
                patch.object(
                    gaussian_solver_app,
                    "optimize_multi_view_staged",
                    side_effect=optimize_multi_view_staged,
                ) as staged_solver,
                patch.object(
                    gaussian_solver_app,
                    "load_observed_structure_graph",
                    side_effect=AssertionError(
                        "Default staged solve must not load a rigid graph."
                    ),
                ) as graph_loader,
                patch.object(gaussian_solver_app, "_print_observation_sanity"),
                patch.object(gaussian_solver_app, "write_solve_visualizations"),
            ):
                gaussian_solver_app.run(args)

            observation_builder.assert_called_once()
            staged_solver.assert_called_once()
            graph_loader.assert_not_called()
            manifest = json.loads(
                (out_dir / "modal_modes_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(manifest["parameters"]["solver"], "staged")
            self.assertNotIn(
                "rigid_component_observation_policy", manifest["parameters"]
            )
            latent = _load_npz(out_dir / manifest["modes"][0]["latent_path"])
            self.assertEqual(set(latent), CORE_LATENT_KEYS)


if __name__ == "__main__":
    unittest.main()
