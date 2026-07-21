from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from flow3d.modal_utils import (
    classify_motion_fill_display_points,
    load_gaussian_modal_fields,
    load_modal_modes,
)
from modal_surface.apps import solve_gaussian_modes as gaussian_solver_app
from modal_surface.apps.solve_gaussian_modes import (
    _write_compact_gaussian_latent,
    _write_solver_diagnostics,
)
from modal_surface.gaussian_motion_fill import apply_gaussian_motion_fill
from modal_surface.io import save_npz_compressed_atomic
from modal_surface.motion_fill import build_knn_graph, query_knn_candidates
from modal_surface.optimization_staged import (
    StagedSolveResult,
    StagedSolverConfig,
    optimize_multi_view_staged,
)


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
FORBIDDEN_OBSERVATION_ROW_KEYS = {
    "obs_point_index",
    "obs_view_index",
    "obs_pixels_xy",
    "obs_y",
    "obs_J",
    "obs_effective_weight",
    "obs_pred_y",
    "obs_residual",
    "obs_residual_valid_mask",
}
FORBIDDEN_LATENT_KEYS = FORBIDDEN_OBSERVATION_ROW_KEYS | {
    "alphas",
    "alpha_information_matrix",
    "phi_observable",
    "phi_nullspace_correction",
    "point_nullspace_basis",
    "point_residual",
}


J_VIEW_0 = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
)
J_VIEW_1 = np.asarray(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)


def _formal_observations(
    duplicate_rows: int = 1,
    identifiable_alpha: bool = True,
) -> dict[str, np.ndarray]:
    points = np.column_stack(
        [
            0.1 * np.arange(6, dtype=np.float32),
            np.zeros((6, 2), dtype=np.float32),
        ]
    )
    point_modes = np.asarray(
        [
            [1.0, 0.2, 0.3],
            [0.4, 1.0, 0.2],
            [0.2, 0.5, 1.0],
            [1.0, -0.4, 0.6],
            [0.7, 0.3, 0.8],
            [0.1, 0.2, 0.4],
        ],
        dtype=np.complex64,
    )
    point_views = (
        [[0, 1], [0, 1], [0, 1], [0, 1], [0], []]
        if identifiable_alpha
        else [[0], [0], [0], [0], [0], [1]]
    )
    alphas = np.asarray([1.0 + 0.0j, np.exp(0.4j)], dtype=np.complex64)

    point_indices: list[int] = []
    view_indices: list[int] = []
    observation_values: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    for point_index, views in enumerate(point_views):
        for view_index in views:
            jacobian = J_VIEW_0 if view_index == 0 else J_VIEW_1
            observation = alphas[view_index] * (
                jacobian @ point_modes[point_index]
            )
            for _ in range(duplicate_rows):
                point_indices.append(point_index)
                view_indices.append(view_index)
                observation_values.append(observation)
                jacobians.append(jacobian)

    obs_point_index = np.asarray(point_indices, dtype=np.int32)
    obs_view_index = np.asarray(view_indices, dtype=np.int32)
    point_view_mask = np.zeros((points.shape[0], alphas.shape[0]), dtype=bool)
    point_view_mask[obs_point_index, obs_view_index] = True
    sample_count = np.bincount(
        obs_point_index, minlength=points.shape[0]
    ).astype(np.int32)
    return {
        "points_world": points,
        "obs_point_index": obs_point_index,
        "obs_view_index": obs_view_index,
        "obs_pixels_xy": np.zeros((obs_point_index.size, 2), dtype=np.float32),
        "obs_y": np.asarray(observation_values, dtype=np.complex64).reshape(-1, 2),
        "obs_J": np.asarray(jacobians, dtype=np.float32).reshape(-1, 2, 3),
        "obs_contribution_weight": np.ones(
            (obs_point_index.size,), dtype=np.float32
        ),
        "obs_count_per_point": point_view_mask.sum(axis=1).astype(np.int32),
        "obs_sample_count_per_point": sample_count,
        "view_ids": np.asarray(["view0", "view1"]),
        "view_freqs_hz": np.asarray([2.5, 2.5], dtype=np.float32),
        "freq_hz": np.array(2.5, dtype=np.float32),
        "mode_index": np.array(4, dtype=np.int32),
        "gaussian_indices": np.arange(points.shape[0], dtype=np.int32),
        "point_type": np.array("foreground_gaussian_center"),
        "source_checkpoint": np.array("source.ckpt"),
    }


def _solve(duplicate_rows: int = 1) -> StagedSolveResult:
    return optimize_multi_view_staged(
        _formal_observations(duplicate_rows),
        StagedSolverConfig(
            alpha_min_shared_points=1,
            alpha_rank_ratio_min=1e-8,
            alpha_info_ratio_min=1e-8,
        ),
    )


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _formal_run_args(
    out_dir: Path,
    *,
    alpha_failure: str,
    motion_fill: bool = False,
    anchor_graph: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    gaussian_solver_app.add_arguments(parser)
    argv = [
        "--input-ckpt",
        "source.ckpt",
        "--view-config",
        "view0.json",
        "--view-config",
        "view1.json",
        "--modal-npz",
        "view0.npz",
        "--modal-npz",
        "view1.npz",
        "--out-dir",
        str(out_dir),
        "--mode-indices",
        "0",
        "--alpha-min-shared-points",
        "1",
        "--alpha-rank-ratio-min",
        "1e-8",
        "--alpha-info-ratio-min",
        "1e-8",
        "--alpha-failure",
        alpha_failure,
    ]
    if motion_fill:
        argv.extend(
            [
                "--motion-fill",
                "--motion-fill-k",
                "2",
                "--motion-fill-max-distance",
                "0.11",
            ]
        )
    if anchor_graph:
        argv.extend(
            [
                "--anchor-graph",
                "--anchor-graph-max-neighbors",
                "2",
                "--anchor-graph-max-distance",
                "0.11",
            ]
        )
    return parser.parse_args(argv)


def _checkpoint_inputs(points: np.ndarray) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
]:
    num_points = int(points.shape[0])
    quaternions = np.zeros((num_points, 4), dtype=np.float32)
    quaternions[:, 0] = 1.0
    return (
        points,
        np.ones((num_points, 3), dtype=np.float32),
        quaternions,
        np.ones((num_points,), dtype=np.float32),
        np.full((num_points, 3), 0.5, dtype=np.float32),
        [],
        [],
    )


def _observation_writer(
    observations: dict[str, np.ndarray],
) -> Callable[..., Path]:
    def write(*, out_path: str | Path, **_: object) -> Path:
        path = Path(out_path)
        save_npz_compressed_atomic(path, observations)
        return path

    return write


class CompactGaussianArtifactTests(unittest.TestCase):
    def test_no_fill_latent_has_exact_schema_independent_of_row_count(self) -> None:
        base = _solve(duplicate_rows=1)
        duplicated = _solve(duplicate_rows=4)
        self.assertNotEqual(
            base.prepared.obs_y.shape[0], duplicated.prepared.obs_y.shape[0]
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path = _write_compact_gaussian_latent(
                root / "base.npz", base, None
            )
            duplicated_path = _write_compact_gaussian_latent(
                root / "duplicated.npz", duplicated, None
            )
            base_arrays = _load_archive(base_path)
            duplicated_arrays = _load_archive(duplicated_path)

        self.assertEqual(set(base_arrays), CORE_LATENT_KEYS)
        self.assertEqual(set(duplicated_arrays), CORE_LATENT_KEYS)
        self.assertTrue(FORBIDDEN_LATENT_KEYS.isdisjoint(base_arrays))
        self.assertEqual(
            {
                key: (value.shape, value.dtype.str)
                for key, value in base_arrays.items()
            },
            {
                key: (value.shape, value.dtype.str)
                for key, value in duplicated_arrays.items()
            },
        )
        self.assertEqual(
            sum(value.nbytes for value in base_arrays.values()),
            sum(value.nbytes for value in duplicated_arrays.values()),
        )

    def test_formal_diagnostics_exclude_all_observation_row_arrays(self) -> None:
        staged = _solve()
        observation_row_count = int(staged.prepared.obs_y.shape[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_solver_diagnostics(
                Path(tmp) / "diagnostics.npz", staged, None
            )
            diagnostics = _load_archive(path)

        self.assertTrue(FORBIDDEN_OBSERVATION_ROW_KEYS.isdisjoint(diagnostics))
        self.assertFalse(
            any(
                value.ndim > 0 and value.shape[0] == observation_row_count
                for value in diagnostics.values()
            )
        )
        self.assertTrue(
            {
                "alphas",
                "alpha_information_matrix",
                "point_nullity",
                "staged_point_solution_status",
                "final_point_solution_status",
                "point_residual",
                "point_residual_valid_mask",
            }.issubset(diagnostics)
        )
        self.assertTrue(
            {
                "alpha_by_view",
                "alpha_phase",
                "alpha_gain",
                "alpha_gain_std",
                "alpha_semantics",
                "anchor_mask",
                "partial_mask",
                "rejected_mask",
                "unobserved_mask",
                "alpha_unresolved_mask",
                "no_usable_observation_mask",
            }.isdisjoint(diagnostics)
        )
        np.testing.assert_array_equal(
            diagnostics["staged_point_solution_status"],
            staged.observable.point_status,
        )
        np.testing.assert_array_equal(
            diagnostics["final_point_solution_status"],
            staged.observable.point_status,
        )
        np.testing.assert_allclose(diagnostics["point_residual"], staged.point_residual)

    def test_run_alpha_failure_writes_diagnostics_without_final_latent(self) -> None:
        observations = _formal_observations(identifiable_alpha=False)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "solve"
            args = _formal_run_args(
                out_dir,
                alpha_failure="error",
            )
            diagnostics_path = (
                out_dir / "diagnostics" / "mode_000_2p5hz.npz"
            )
            latent_path = out_dir / "latents" / "mode_000_2p5hz.npz"
            with (
                patch.object(
                    gaussian_solver_app,
                    "load_fg_pixel_candidate_inputs_from_checkpoint",
                    return_value=_checkpoint_inputs(observations["points_world"]),
                ),
                patch.object(
                    gaussian_solver_app,
                    "load_modal_freqs",
                    return_value=[
                        np.asarray([2.5], dtype=np.float32),
                        np.asarray([2.5], dtype=np.float32),
                    ],
                ),
                patch.object(
                    gaussian_solver_app,
                    "build_gaussian_observation_graph",
                    side_effect=_observation_writer(observations),
                ),
                patch.object(gaussian_solver_app, "_print_observation_sanity"),
                patch.object(
                    gaussian_solver_app, "write_solve_visualizations"
                ) as visualization,
            ):
                with self.assertRaises(ValueError) as raised:
                    gaussian_solver_app.run(args)

            self.assertIn(str(diagnostics_path), str(raised.exception))
            self.assertTrue(diagnostics_path.is_file())
            self.assertFalse(latent_path.exists())
            self.assertFalse((out_dir / "modal_modes_manifest.json").exists())
            visualization.assert_not_called()
            diagnostics = _load_archive(diagnostics_path)

        self.assertFalse(bool(diagnostics["alpha_identifiable_mask"][1]))
        self.assertTrue(FORBIDDEN_OBSERVATION_ROW_KEYS.isdisjoint(diagnostics))
        self.assertTrue(
            {
                "phi_nullspace_correction",
                "motion_fill_role",
                "completion_mask",
                "motion_fill_graph_path",
            }.isdisjoint(diagnostics)
        )
        np.testing.assert_array_equal(
            diagnostics["final_point_solution_status"],
            diagnostics["staged_point_solution_status"],
        )

    def test_anchor_graph_switch_preserves_existing_solver_artifacts_and_manifest_fields(self) -> None:
        observations = _formal_observations()
        manifests = []
        latents = []
        with tempfile.TemporaryDirectory() as tmp:
            for name, anchor_graph_enabled in (
                ("baseline", False),
                ("anchor_graph", True),
            ):
                out_dir = Path(tmp) / name
                args = _formal_run_args(
                    out_dir,
                    alpha_failure="exclude",
                    anchor_graph=anchor_graph_enabled,
                )
                fake_graph = SimpleNamespace(
                    counts={
                        "retained_edge_count": 2,
                        "isolated_anchor_count": 0,
                    }
                )

                def fake_view_config(path: str) -> SimpleNamespace:
                    return SimpleNamespace(
                        view_id=Path(path).stem,
                        K=np.eye(3, dtype=np.float32),
                        world_to_camera=np.eye(4, dtype=np.float32),
                    )

                with (
                    patch.object(
                        gaussian_solver_app,
                        "load_fg_pixel_candidate_inputs_from_checkpoint",
                        return_value=_checkpoint_inputs(observations["points_world"]),
                    ),
                    patch.object(
                        gaussian_solver_app,
                        "load_modal_freqs",
                        return_value=[
                            np.asarray([2.5], dtype=np.float32),
                            np.asarray([2.5], dtype=np.float32),
                        ],
                    ),
                    patch.object(
                        gaussian_solver_app,
                        "build_gaussian_observation_graph",
                        side_effect=_observation_writer(observations),
                    ),
                    patch.object(gaussian_solver_app, "_print_observation_sanity"),
                    patch.object(gaussian_solver_app, "write_solve_visualizations"),
                    patch.object(
                        gaussian_solver_app,
                        "load_view_config",
                        side_effect=fake_view_config,
                    ),
                    patch.object(
                        gaussian_solver_app,
                        "build_anchor_structure_graph",
                        return_value=fake_graph,
                    ) as graph_builder,
                    patch.object(
                        gaussian_solver_app,
                        "write_anchor_structure_graph",
                        side_effect=lambda path, *_args, **_kwargs: Path(path),
                    ),
                ):
                    gaussian_solver_app.run(args)

                self.assertEqual(
                    graph_builder.call_count,
                    1 if anchor_graph_enabled else 0,
                )
                latents.append(
                    _load_archive(out_dir / "latents" / "mode_000_2p5hz.npz")
                )
                manifests.append(
                    json.loads(
                        (out_dir / "modal_modes_manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )
                )

        self.assertEqual(set(latents[0]), set(latents[1]))
        for key in latents[0]:
            np.testing.assert_array_equal(latents[0][key], latents[1][key])
        baseline_parameters = {
            key: value
            for key, value in manifests[0]["parameters"].items()
            if not key.startswith("anchor_graph")
        }
        graph_parameters = {
            key: value
            for key, value in manifests[1]["parameters"].items()
            if not key.startswith("anchor_graph")
        }
        self.assertEqual(baseline_parameters, graph_parameters)
        baseline_mode = dict(manifests[0]["modes"][0])
        graph_mode = dict(manifests[1]["modes"][0])
        graph_mode.pop("anchor_graph_path")
        self.assertEqual(baseline_mode, graph_mode)

    def test_run_motion_fill_failure_writes_staged_only_diagnostics(self) -> None:
        observations = _formal_observations()
        failure = RuntimeError("mock motion-fill invariant failure")
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "solve"
            args = _formal_run_args(
                out_dir,
                alpha_failure="exclude",
                motion_fill=True,
            )
            diagnostics_path = (
                out_dir / "diagnostics" / "mode_000_2p5hz.npz"
            )
            latent_path = out_dir / "latents" / "mode_000_2p5hz.npz"
            with (
                patch.object(
                    gaussian_solver_app,
                    "load_fg_pixel_candidate_inputs_from_checkpoint",
                    return_value=_checkpoint_inputs(observations["points_world"]),
                ),
                patch.object(
                    gaussian_solver_app,
                    "load_modal_freqs",
                    return_value=[
                        np.asarray([2.5], dtype=np.float32),
                        np.asarray([2.5], dtype=np.float32),
                    ],
                ),
                patch.object(
                    gaussian_solver_app,
                    "build_gaussian_observation_graph",
                    side_effect=_observation_writer(observations),
                ),
                patch.object(gaussian_solver_app, "_print_observation_sanity"),
                patch.object(
                    gaussian_solver_app,
                    "apply_gaussian_motion_fill",
                    side_effect=failure,
                ) as mocked_fill,
                patch.object(
                    gaussian_solver_app, "write_solve_visualizations"
                ) as visualization,
            ):
                with self.assertRaises(RuntimeError) as raised:
                    gaussian_solver_app.run(args)

            self.assertIs(raised.exception, failure)
            mocked_fill.assert_called_once()
            visualization.assert_not_called()
            self.assertTrue(diagnostics_path.is_file())
            self.assertFalse(latent_path.exists())
            self.assertFalse((out_dir / "modal_modes_manifest.json").exists())
            self.assertTrue((out_dir / "motion_fill" / "graph.npz").is_file())
            diagnostics = _load_archive(diagnostics_path)

        self.assertTrue(FORBIDDEN_OBSERVATION_ROW_KEYS.isdisjoint(diagnostics))
        self.assertTrue(
            {
                "phi_nullspace_correction",
                "motion_fill_role",
                "completion_mask",
                "motion_fill_graph_path",
            }.isdisjoint(diagnostics)
        )
        np.testing.assert_array_equal(
            diagnostics["final_point_solution_status"],
            diagnostics["staged_point_solution_status"],
        )

    def test_motion_fill_latent_diagnostics_and_runtime_loaders(self) -> None:
        staged = _solve()
        candidates = query_knn_candidates(staged.prepared.points, max_k=2)
        graph = build_knn_graph(candidates, k=2, max_distance=0.11)
        graph_path = "motion_fill/graph.npz"
        filled = apply_gaussian_motion_fill(staged, graph, graph_path)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latent_path = _write_compact_gaussian_latent(
                root / "latents" / "mode_4.npz", staged, filled
            )
            diagnostics_path = _write_solver_diagnostics(
                root / "diagnostics" / "mode_4.npz",
                staged,
                filled,
                graph,
                graph_path,
            )
            latent = _load_archive(latent_path)
            diagnostics = _load_archive(diagnostics_path)

            manifest_path = root / "modal_modes_manifest.json"
            diagnostics_relative_path = diagnostics_path.relative_to(root).as_posix()
            manifest = {
                "version": 1,
                "point_type": "foreground_gaussian_center",
                "source_checkpoint": "source.ckpt",
                "modes": [
                    {
                        "mode_index": 4,
                        "freq_hz": 2.5,
                        "latent_path": latent_path.relative_to(root).as_posix(),
                        "diagnostics_path": diagnostics_relative_path,
                    }
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(manifest["version"], 1)
            self.assertTrue((root / diagnostics_relative_path).is_file())
            loaded_fields = load_gaussian_modal_fields(
                str(manifest_path),
                torch.from_numpy(staged.prepared.points.copy()),
            )
            viewer_modes = load_modal_modes(str(manifest_path))

        self.assertEqual(
            set(latent), CORE_LATENT_KEYS | MOTION_FILL_LATENT_KEYS
        )
        self.assertTrue(FORBIDDEN_LATENT_KEYS.isdisjoint(latent))
        np.testing.assert_allclose(latent["phi"], filled.motion.phi)
        np.testing.assert_array_equal(latent["motion_fill_role"], filled.roles.role)
        np.testing.assert_array_equal(
            latent["completion_mask"], filled.motion.completion_mask
        )
        self.assertEqual(
            str(latent["point_type"].item()), "foreground_gaussian_center"
        )
        self.assertEqual(str(latent["source_checkpoint"].item()), "source.ckpt")

        self.assertTrue(FORBIDDEN_OBSERVATION_ROW_KEYS.isdisjoint(diagnostics))
        self.assertNotIn("phi_observable", diagnostics)
        np.testing.assert_allclose(
            diagnostics["phi_nullspace_correction"],
            filled.motion.phi_nullspace_correction,
        )
        np.testing.assert_array_equal(
            diagnostics["final_point_solution_status"],
            filled.point_solution_status,
        )
        np.testing.assert_array_equal(
            diagnostics["motion_fill_point_numerical_nullity"],
            filled.numerical_nullity,
        )
        np.testing.assert_allclose(
            latent["phi"] - diagnostics["phi_nullspace_correction"],
            staged.observable.phi,
            atol=1e-6,
        )
        self.assertEqual(
            str(diagnostics["motion_fill_graph_path"].item()), graph_path
        )
        self.assertEqual(int(diagnostics["motion_fill_graph_k"].item()), graph.k)
        self.assertAlmostEqual(
            float(diagnostics["motion_fill_graph_max_distance"].item()),
            graph.max_distance,
        )

        torch.testing.assert_close(
            loaded_fields.phi_real[0], torch.from_numpy(filled.motion.phi.real)
        )
        torch.testing.assert_close(
            loaded_fields.phi_imag[0], torch.from_numpy(filled.motion.phi.imag)
        )
        self.assertIsNotNone(viewer_modes[0].motion_fill_display_class)
        assert viewer_modes[0].motion_fill_display_class is not None
        np.testing.assert_array_equal(
            viewer_modes[0].motion_fill_display_class,
            classify_motion_fill_display_points(
                filled.roles.role, filled.motion.completion_mask
            ),
        )


if __name__ == "__main__":
    unittest.main()
