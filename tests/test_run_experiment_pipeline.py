from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from run_experiment_pipeline import (
    ComparisonPipelineConfig,
    PrestaticInputs,
    REPO_ROOT,
    SourceView,
    _candidate_parameters_from_args,
    _comparison_control_paths,
    _comparison_stage_config,
    _comparison_variant_paths,
    _initial_state,
    _modal_solver_argv,
    _parse_source_view,
    _pipeline_paths,
    _prepare_controller,
    _python_argv,
    _resolved_config_payload,
    _run_argv,
    _set_gate,
    _status_payload,
    _stored_graph_scalar_matches_expected,
    _viewer_command_text,
    build_parser,
    load_comparison_config,
    load_config,
    load_pipeline_configuration,
)


TEMPLATE_PATH = REPO_ROOT / "preproc" / "experiment_pipeline_template.yaml"
COMPARISON_TEMPLATE_PATH = (
    REPO_ROOT / "preproc" / "experiment_comparison_pipeline_template.yaml"
)


def _write_test_config(root: Path) -> tuple[dict[str, object], Path]:
    payload = yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))
    payload["scene_id"] = "scene_v1"
    payload["scene_root"] = str(root / "scene_v1")
    payload["pipeline_id"] = "pipeline_v1"
    payload["prestatic_run_id"] = "prestatic_v1"
    path = root / "pipeline.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload, path


def _write_comparison_config(
    root: Path,
    base_path: Path,
) -> tuple[dict[str, object], Path]:
    payload = yaml.safe_load(
        COMPARISON_TEMPLATE_PATH.read_text(encoding="utf-8")
    )
    payload["base_config"] = str(base_path)
    payload["shared_run_id"] = "comparison_v1"
    path = root / "comparison.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload, path


class ExperimentPipelineTest(unittest.TestCase):
    def test_graph_scalar_validation_uses_artifact_storage_precision(self) -> None:
        import numpy as np

        self.assertTrue(
            _stored_graph_scalar_matches_expected(
                np.array(0.008, dtype=np.float32),
                0.008,
            )
        )
        self.assertFalse(
            _stored_graph_scalar_matches_expected(
                np.array(0.009, dtype=np.float32),
                0.008,
            )
        )
        self.assertTrue(
            _stored_graph_scalar_matches_expected(
                np.array(8, dtype=np.int32),
                8,
            )
        )
        self.assertFalse(
            _stored_graph_scalar_matches_expected(
                np.array(7, dtype=np.int32),
                8,
            )
        )

    def test_comparison_config_reuses_shared_paths_and_isolates_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, base_path = _write_test_config(root)
            _, comparison_path = _write_comparison_config(root, base_path)

            loaded = load_pipeline_configuration(comparison_path)

            self.assertIsInstance(loaded, ComparisonPipelineConfig)
            assert isinstance(loaded, ComparisonPipelineConfig)
            self.assertEqual(
                [variant.config.solver.method for variant in loaded.variants],
                ["staged", "rigid-components", "soft-elastic"],
            )
            base_paths = _pipeline_paths(loaded.base_config)
            control_paths = _comparison_control_paths(loaded)
            self.assertEqual(control_paths.topology_path, base_paths.topology_path)
            self.assertEqual(
                control_paths.observations_dir,
                base_paths.observations_dir,
            )
            variant_paths = [
                _comparison_variant_paths(loaded, variant)
                for variant in loaded.variants
            ]
            self.assertEqual(len({path.modal_fields_dir for path in variant_paths}), 3)
            for variant, paths in zip(loaded.variants, variant_paths):
                self.assertIn(
                    variant.config.solver.artifact_id,
                    paths.rendered_design_dir.parts,
                )
                self.assertEqual(paths.topology_path, base_paths.topology_path)

    def test_comparison_stage_identities_do_not_invalidate_modal_solve_for_physics_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, base_path = _write_test_config(root)
            payload, comparison_path = _write_comparison_config(root, base_path)
            first = load_comparison_config(comparison_path)
            first_variant = first.variants[0]
            first_modal = _comparison_stage_config(
                first,
                first_variant,
                "gaussian_modal_fields",
            ).config_identity
            first_physics = _comparison_stage_config(
                first,
                first_variant,
                "modal_physics_coordinates",
            ).config_identity

            variants = payload["variants"]
            assert isinstance(variants, list)
            variants[0]["physics_artifact_id"] = "different_physics_v2"
            comparison_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            second = load_comparison_config(comparison_path)
            second_variant = second.variants[0]

            self.assertEqual(
                first_modal,
                _comparison_stage_config(
                    second,
                    second_variant,
                    "gaussian_modal_fields",
                ).config_identity,
            )
            self.assertNotEqual(
                first_physics,
                _comparison_stage_config(
                    second,
                    second_variant,
                    "modal_physics_coordinates",
                ).config_identity,
            )
            rigid_variant = second.variants[1]
            self.assertNotEqual(
                _comparison_stage_config(
                    second,
                    rigid_variant,
                    "gaussian_modal_fields",
                    "graph_sha_1",
                ).config_identity,
                _comparison_stage_config(
                    second,
                    rigid_variant,
                    "gaussian_modal_fields",
                    "graph_sha_2",
                ).config_identity,
            )

    def test_staged_modal_command_reuses_shared_observation_bank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config_path = _write_test_config(root)
            config = load_config(config_path)
            paths = _pipeline_paths(config)
            view = SourceView(
                view_id="view1",
                image_dir=root / "images" / "view1",
                mask_dir=root / "masks" / "view1",
                fps_hz=30.0,
                width=960,
                height=540,
                frame_names=("000000.png",),
                source_identity="source_identity",
                reference_frame_name="000000.png",
                reference_frame_stem="000000",
                reference_local_index=0,
            )
            inputs = PrestaticInputs(
                ready_path=root / "ready.json",
                static_dataset=root / "static_dataset",
                source_manifest=root / "source_manifest.json",
                reference_cameras=root / "reference_cameras.json",
                reference_selection=root / "reference_selection.json",
                views=(view,),
            )

            argv = _modal_solver_argv(
                config,
                inputs,
                paths,
                root / "last.ckpt",
                None,
                resume=True,
            )

            self.assertIn("--staged-observation-topology", argv)
            self.assertEqual(
                argv[argv.index("--staged-observation-topology") + 1],
                str(paths.topology_path),
            )
            self.assertIn("--staged-observation-measurement", argv)
            self.assertIn("--resume", argv)
            self.assertNotIn("--rigid-component-observation-topology", argv)

    def test_template_loads_and_derives_formal_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config_path = _write_test_config(root)
            config = load_config(config_path)
            paths = _pipeline_paths(config)

            self.assertEqual(config.scene_id, "scene_v1")
            self.assertEqual(config.static.num_bg, 100000)
            self.assertEqual(config.frequency.mode_counts, (20, 40, 60))
            self.assertEqual(config.frequency.selected_k, 60)
            self.assertEqual(
                paths.pipeline_dir,
                root / "scene_v1" / "shared" / "pipeline_runs" / "pipeline_v1",
            )
            self.assertEqual(
                paths.static_work_dir,
                root / "scene_v1" / "static_3dgs" / "sweep_rgbmask_v1",
            )
            self.assertEqual(
                paths.topology_path,
                root
                / "scene_v1"
                / "shared"
                / "preprocessing"
                / "gaussian_observation_topology"
                / "static_stride2_k4_v1.npz",
            )
            self.assertEqual(
                paths.flow_coordinates_dir,
                root
                / "scene_v1"
                / "flow_coordinates"
                / "greedy_0p2_4p0_step0p025_k60prefix_v1"
                / "rendered_ridge1e4_v1",
            )

    def test_config_schema_rejects_extra_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload, config_path = _write_test_config(root)
            topology = payload["topology"]
            assert isinstance(topology, dict)
            topology["post_export_resize"] = 960
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "strict schema"):
                load_config(config_path)

    def test_soft_elastic_is_a_third_solver_with_explicit_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload, config_path = _write_test_config(root)
            solver = payload["solver"]
            assert isinstance(solver, dict)
            solver["method"] = "soft-elastic"
            solver["artifact_id"] = "soft_elastic_k60_v1"
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )

            config = load_config(config_path)

            self.assertEqual(config.solver.method, "soft-elastic")
            self.assertIsNotNone(config.soft_elastic)
            assert config.soft_elastic is not None
            self.assertEqual(config.soft_elastic.stretch_relative, 0.01)
            self.assertEqual(config.soft_elastic.laplacian_relative, 0.0001)

    def test_legacy_config_without_soft_elastic_section_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload, config_path = _write_test_config(root)
            payload.pop("soft_elastic")
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )

            config = load_config(config_path)

            self.assertIsNone(config.soft_elastic)

    def test_soft_elastic_modal_command_uses_shared_graph_and_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload, config_path = _write_test_config(root)
            solver = payload["solver"]
            assert isinstance(solver, dict)
            solver["method"] = "soft-elastic"
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            config = load_config(config_path)
            paths = _pipeline_paths(config)
            view = SourceView(
                view_id="view1",
                image_dir=root / "images" / "view1",
                mask_dir=root / "masks" / "view1",
                fps_hz=30.0,
                width=960,
                height=540,
                frame_names=("000000.png",),
                source_identity="source_identity",
                reference_frame_name="000000.png",
                reference_frame_stem="000000",
                reference_local_index=0,
            )
            inputs = PrestaticInputs(
                ready_path=root / "ready.json",
                static_dataset=root / "static_dataset",
                source_manifest=root / "source_manifest.json",
                reference_cameras=root / "reference_cameras.json",
                reference_selection=root / "reference_selection.json",
                views=(view,),
            )
            graph = root / "observed_structure_graph.npz"
            argv = _modal_solver_argv(
                config,
                inputs,
                paths,
                root / "last.ckpt",
                graph,
                resume=True,
            )

            self.assertIn("--soft-elastic-graph", argv)
            self.assertEqual(argv[argv.index("--soft-elastic-graph") + 1], str(graph))
            self.assertIn("--soft-elastic-observation-measurement", argv)
            self.assertIn("--soft-elastic-stretch-relative", argv)
            self.assertIn("--resume", argv)
            self.assertNotIn("--rigid-component-graph", argv)

    def test_reference_view_accepts_an_explicit_non_middle_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "images"
            mask_dir = root / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            frame_names = [f"view1_{index:06d}.png" for index in range(4)]
            for frame_name in frame_names:
                (image_dir / frame_name).write_bytes(b"rgb")
                (mask_dir / frame_name).write_bytes(b"mask")
            source_header = {
                "view_id": "view1",
                "image_dir": str(image_dir.resolve()),
                "mask_dir": str(mask_dir.resolve()),
                "fps_hz": 30.0,
                "camera_group": "fixed_camera",
                "image_width": 960,
                "image_height": 540,
            }
            source_hasher = hashlib.sha256()
            source_hasher.update(
                json.dumps(source_header, sort_keys=True).encode("utf-8")
            )
            for frame_name in frame_names:
                image_stat = (image_dir / frame_name).stat()
                mask_stat = (mask_dir / frame_name).stat()
                source_hasher.update(frame_name.encode("utf-8"))
                source_hasher.update(
                    (
                        f"{image_stat.st_size}:{image_stat.st_mtime_ns}:"
                        f"{mask_stat.st_size}:{mask_stat.st_mtime_ns}"
                    ).encode("utf-8")
                )
            view = _parse_source_view(
                {
                    "view_id": "view1",
                    "image_dir": str(image_dir),
                    "mask_dir": str(mask_dir),
                    "fps_hz": 30.0,
                    "camera_group": "fixed_camera",
                    "frame_count": len(frame_names),
                    "image_width": 960,
                    "image_height": 540,
                    "image_extension": ".png",
                    "frame_names": frame_names,
                    "source_identity": source_hasher.hexdigest(),
                },
                {
                    "view1": {
                        "source_frame_name": frame_names[0],
                        "source_index": 0,
                    }
                },
            )

            self.assertEqual(view.reference_local_index, 0)
            self.assertEqual(view.reference_frame_name, frame_names[0])

    def test_candidate_id_is_stable_for_defaults_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config_path = _write_test_config(root)
            config = load_config(config_path)
            parser = build_parser()

            default_args = parser.parse_args(
                ["build-graph-candidate", "--config", str(config_path)]
            )
            default_candidate = _candidate_parameters_from_args(config, default_args)
            self.assertEqual(default_candidate.candidate_id, "knn8_maxdist0p008_v1")

            override_argv = [
                "build-graph-candidate",
                "--config",
                str(config_path),
                "--max-distance",
                "0.01",
                "--max-neighbors",
                "12",
            ]
            first = _candidate_parameters_from_args(
                config, parser.parse_args(override_argv)
            )
            second = _candidate_parameters_from_args(
                config, parser.parse_args(override_argv)
            )
            values = {
                "max_distance": 0.01,
                "max_neighbors": 12,
                "color_mad_multiplier": 3.0,
                "depth_mad_multiplier": 3.0,
                "depth_samples": 5,
                "min_shared_views": 1,
                "min_component_nodes": 4,
                "min_component_edges": 3,
            }
            encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
            expected_id = "candidate_" + hashlib.sha256(
                encoded.encode("utf-8")
            ).hexdigest()[:10]
            self.assertEqual(first.candidate_id, expected_id)
            self.assertEqual(second, first)

            explicit_args = parser.parse_args(
                override_argv + ["--candidate-id", "wider_graph_v2"]
            )
            explicit = _candidate_parameters_from_args(config, explicit_args)
            self.assertEqual(explicit.candidate_id, "wider_graph_v2")

    def test_status_and_gate_state_are_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config_path = _write_test_config(root)
            config = load_config(config_path)
            paths = _pipeline_paths(config)

            status = _status_payload(config)
            self.assertFalse(status["started"])
            self.assertIsNone(status["gate"])
            self.assertEqual(status["graph_candidates"], [])

            state = _initial_state(config)
            gate = {
                "name": "static_quality",
                "status": "waiting_for_approval",
                "target_epochs": 100,
            }
            _set_gate(paths, state, gate)
            saved = json.loads(paths.state_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["gate"], gate)
            self.assertTrue(_status_payload(config)["started"])

            _set_gate(paths, state, None)
            saved = json.loads(paths.state_path.read_text(encoding="utf-8"))
            self.assertIsNone(saved["gate"])

    def test_resolved_config_survives_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config_path = _write_test_config(root)
            config = load_config(config_path)
            paths = _pipeline_paths(config)
            inputs = PrestaticInputs(
                ready_path=root / "ready.json",
                static_dataset=root / "static_dataset",
                source_manifest=root / "source_manifest.json",
                reference_cameras=root / "reference_cameras.json",
                reference_selection=root / "reference_selection.json",
                views=(
                    SourceView(
                        view_id="view1",
                        image_dir=root / "images" / "view1",
                        mask_dir=root / "masks" / "view1",
                        fps_hz=30.0,
                        width=960,
                        height=540,
                        frame_names=("000000.png",),
                        source_identity="source_identity",
                        reference_frame_name="000000.png",
                        reference_frame_stem="000000",
                        reference_local_index=0,
                    ),
                ),
            )

            first_state = _prepare_controller(config, inputs, paths)
            second_state = _prepare_controller(config, inputs, paths)
            resolved = json.loads(
                paths.resolved_config_path.read_text(encoding="utf-8")
            )

            self.assertEqual(second_state, first_state)
            self.assertEqual(
                resolved["scientific_config"]["frequency_selection"]["mode_counts"],
                [20, 40, 60],
            )
            self.assertEqual(resolved, _resolved_config_payload(config, inputs, paths))

    def test_viewer_and_python_commands_use_derived_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config_path = _write_test_config(root)
            config = load_config(config_path)
            paths = _pipeline_paths(config)

            direct = _viewer_command_text(config, paths, "direct", None, 8893)
            self.assertEqual(
                direct,
                "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u run_rendering.py "
                f"--work-dir {paths.direct_checkpoint_dir} --port 8893",
            )
            graph = _viewer_command_text(
                config, paths, "rigid-graph", "wider_graph_v2", 8894
            )
            self.assertIn(
                str(
                    paths.graph_candidates_dir
                    / "wider_graph_v2"
                    / "observed_structure_graph.npz"
                ),
                graph,
            )
            self.assertTrue(graph.startswith("PYTHONPATH=. python -u "))

            argv = _python_argv("run_rendering.py", "--port", "8893")
            self.assertEqual(
                argv,
                [
                    sys.executable,
                    "-u",
                    str(REPO_ROOT / "run_rendering.py"),
                    "--port",
                    "8893",
                ],
            )

    def test_run_argv_dry_run_only_prints_the_stable_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config_path = _write_test_config(root)
            config = load_config(config_path)
            paths = _pipeline_paths(config)
            argv = ["python", "-u", "tool.py", "--label", "two words"]

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                _run_argv(paths, "example_stage", argv, dry_run=True)

            self.assertEqual(
                output.getvalue(),
                f"[example_stage] {subprocess.list2cmdline(argv)}\n",
            )
            self.assertFalse(paths.events_path.exists())
            self.assertFalse(paths.pipeline_dir.exists())


if __name__ == "__main__":
    unittest.main()
    _comparison_control_paths,
    _comparison_stage_config,
    _comparison_variant_paths,
