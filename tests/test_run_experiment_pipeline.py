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
    REPO_ROOT,
    _candidate_parameters_from_args,
    _initial_state,
    _parse_source_view,
    _pipeline_paths,
    _python_argv,
    _run_argv,
    _set_gate,
    _status_payload,
    _viewer_command_text,
    build_parser,
    load_config,
)


TEMPLATE_PATH = REPO_ROOT / "preproc" / "experiment_pipeline_template.yaml"


def _write_test_config(root: Path) -> tuple[dict[str, object], Path]:
    payload = yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))
    payload["scene_id"] = "scene_v1"
    payload["scene_root"] = str(root / "scene_v1")
    payload["pipeline_id"] = "pipeline_v1"
    payload["prestatic_run_id"] = "prestatic_v1"
    path = root / "pipeline.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload, path


class ExperimentPipelineTest(unittest.TestCase):
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
