from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from preproc.register_prestatic_sweep_frames import (
    _validate_existing_package,
    BaseSeedFrame,
    backup_sqlite_database,
    build_bundle_adjuster_command,
    build_feature_extractor_command,
    build_image_registrator_command,
    build_matches_importer_command,
    build_point_triangulator_command,
    build_temporal_match_pairs,
    load_registration_config,
    registration_paths,
    run_registration_pipeline,
    select_registration_targets,
)
from preproc.run_prestatic_pipeline import (
    FrameRecord,
    SequenceConfig,
    ValidatedSequence,
    select_sweep_frames,
)


def _validated_sweep(root: Path, count: int = 1467) -> ValidatedSequence:
    config = SequenceConfig(
        view_id="curtain3_sweep",
        image_dir=root / "images" / "curtain3_sweep",
        mask_dir=root / "masks" / "curtain3_sweep",
        fps_hz=30.0,
        camera_group="sweep_camera",
    )
    frames = tuple(
        FrameRecord(
            frame_name=f"{index:05d}.png",
            frame_stem=f"{index:05d}",
            source_index=index,
            time_sec=index / config.fps_hz,
            image_path=config.image_dir / f"{index:05d}.png",
            mask_path=config.mask_dir / f"{index:05d}.png",
        )
        for index in range(count)
    )
    return ValidatedSequence(
        config=config,
        image_width=1920,
        image_height=1080,
        image_extension=".png",
        frames=frames,
        source_identity="test-sweep-identity",
    )


def _config_payload(root: Path) -> dict[str, object]:
    scene_root = root / "curtain3_2view_v1"
    return {
        "format": "som_prestatic_registration",
        "version": 1,
        "scene_id": "curtain3_2view_v1",
        "scene_root": str(scene_root),
        "run_id": "joint_colmap_frames_10fps_v1",
        "base_prestatic_run_id": "joint_colmap_frames_v1",
        "target_sweep_fps_hz": 10,
        "min_target_registration_ratio": 0.95,
        "seed_neighbors_per_side": 2,
        "target_neighbor_radius": 2,
        "colmap_command": "colmap",
    }


class RegisterPrestaticSweepFramesTest(unittest.TestCase):
    def test_config_is_strict_and_keeps_base_and_derived_runs_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _config_payload(root)
            config_path = root / "registration.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

            config = load_registration_config(config_path)
            self.assertEqual(config.scene_id, "curtain3_2view_v1")
            self.assertEqual(config.run_id, "joint_colmap_frames_10fps_v1")
            self.assertEqual(config.base_prestatic_run_id, "joint_colmap_frames_v1")
            self.assertEqual(config.target_sweep_fps_hz, 10.0)
            self.assertEqual(config.min_target_registration_ratio, 0.95)
            self.assertNotEqual(config.run_id, config.base_prestatic_run_id)
            paths = registration_paths(config)
            self.assertEqual(
                paths.run_dir,
                config.scene_root
                / "shared"
                / "preprocessing"
                / "joint_colmap_frames_10fps_v1",
            )
            self.assertNotEqual(
                paths.run_dir,
                config.scene_root
                / "shared"
                / "preprocessing"
                / config.base_prestatic_run_id,
            )
            self.assertEqual(paths.dataset_dir, paths.run_dir / "sweep_colmap_dataset")

            payload["target_width"] = 960
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported keys"):
                load_registration_config(config_path)

            del payload["target_width"]
            payload["run_id"] = payload["base_prestatic_run_id"]
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must differ"):
                load_registration_config(config_path)

    def test_30fps_to_10fps_selects_489_targets_and_reuses_49_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sweep = _validated_sweep(Path(temporary))
            seed_selection = select_sweep_frames(sweep, 3.0)
            seed_names = {
                selection.source_frame.source_index: (
                    f"sweep/{selection.staged_name}"
                )
                for selection in seed_selection
            }
            targets = select_registration_targets(sweep, 10.0, seed_names)

        self.assertEqual(len(seed_selection), 147)
        self.assertEqual(len(targets), 489)
        self.assertEqual(
            [target.source_frame.source_index for target in targets[:5]],
            [0, 3, 6, 9, 12],
        )
        reused = [target for target in targets if target.reuses_seed]
        added = [target for target in targets if not target.reuses_seed]
        self.assertEqual(len(reused), 49)
        self.assertEqual(len(added), 440)
        self.assertEqual(
            [target.source_frame.source_index for target in reused[:4]],
            [0, 30, 60, 90],
        )
        self.assertEqual(reused[0].staged_relative_name, "sweep/sweep_000000.png")
        self.assertEqual(
            added[0].staged_relative_name,
            "sweep/dense_sweep_src_000003.png",
        )
        self.assertEqual(
            len({target.staged_relative_name for target in targets}), len(targets)
        )
        target_names = {target.staged_relative_name for target in targets}
        self.assertNotIn(seed_names[10], target_names)
        self.assertIn(seed_names[30], target_names)

    def test_temporal_pairs_are_unique_and_cover_every_new_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sweep = _validated_sweep(Path(temporary), count=61)
            seed_selection = select_sweep_frames(sweep, 3.0)
            seed_names = {
                selection.source_frame.source_index: (
                    f"sweep/{selection.staged_name}"
                )
                for selection in seed_selection
            }
            targets = select_registration_targets(sweep, 10.0, seed_names)
            pairs = build_temporal_match_pairs(
                targets,
                seed_names,
                seed_neighbors_per_side=2,
                target_neighbor_radius=2,
            )

        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(left < right for left, right in pairs))
        self.assertTrue(all(left != right for left, right in pairs))
        covered = {name for pair in pairs for name in pair}
        new_names = {
            target.staged_relative_name for target in targets if not target.reuses_seed
        }
        self.assertTrue(new_names)
        self.assertTrue(new_names <= covered)
        seed_name_set = set(seed_names.values())
        for new_name in new_names:
            self.assertTrue(
                any(
                    new_name in pair and bool(set(pair) & seed_name_set)
                    for pair in pairs
                ),
                new_name,
            )

    def test_reused_but_unregistered_seed_is_paired_to_registered_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sweep = _validated_sweep(Path(temporary), count=61)
            seed_selection = select_sweep_frames(sweep, 3.0)
            base_seed_names = {
                selection.source_frame.source_index: (
                    f"sweep/{selection.staged_name}"
                )
                for selection in seed_selection
            }
            targets = select_registration_targets(sweep, 10.0, base_seed_names)
            unregistered_name = base_seed_names[30]
            registered_seeds = {
                index: name
                for index, name in base_seed_names.items()
                if index != 30
            }
            pairs = build_temporal_match_pairs(
                targets,
                registered_seeds,
                seed_neighbors_per_side=2,
                target_neighbor_radius=2,
            )

        self.assertTrue(
            any(
                unregistered_name in pair
                and bool(set(pair) & set(registered_seeds.values()))
                for pair in pairs
            )
        )

    def test_colmap_commands_reuse_seed_camera_and_keep_explicit_paths(self) -> None:
        root = Path("/formal/derived")
        database = root / "database.db"
        images = root / "images"
        image_list = root / "new_images.txt"
        pair_list = root / "temporal_pairs.txt"
        base_model = root / "base_model"
        registered_model = root / "registered_model"
        triangulated_model = root / "triangulated_model"
        final_model = root / "final_model"

        feature = build_feature_extractor_command(
            "colmap", database, images, image_list, 17
        )
        self.assertEqual(feature[:2], ["colmap", "feature_extractor"])
        self.assertEqual(
            feature[feature.index("--ImageReader.existing_camera_id") + 1], "17"
        )
        self.assertEqual(
            feature[feature.index("--image_list_path") + 1], str(image_list)
        )

        importer = build_matches_importer_command("colmap", database, pair_list)
        self.assertEqual(importer[:2], ["colmap", "matches_importer"])
        self.assertEqual(importer[importer.index("--match_type") + 1], "pairs")

        registrator = build_image_registrator_command(
            "colmap", database, base_model, registered_model
        )
        self.assertEqual(registrator[:2], ["colmap", "image_registrator"])
        self.assertEqual(
            registrator[registrator.index("--input_path") + 1], str(base_model)
        )

        triangulator = build_point_triangulator_command(
            "colmap", database, images, registered_model, triangulated_model
        )
        self.assertEqual(triangulator[:2], ["colmap", "point_triangulator"])
        self.assertEqual(
            triangulator[triangulator.index("--image_path") + 1], str(images)
        )

        bundle_adjuster = build_bundle_adjuster_command(
            "colmap", triangulated_model, final_model
        )
        self.assertEqual(bundle_adjuster[:2], ["colmap", "bundle_adjuster"])
        for option in (
            "--BundleAdjustment.refine_focal_length",
            "--BundleAdjustment.refine_principal_point",
            "--BundleAdjustment.refine_extra_params",
        ):
            self.assertEqual(bundle_adjuster[bundle_adjuster.index(option) + 1], "0")

    def test_existing_package_must_contain_only_registered_target_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _config_payload(root)
            config_path = root / "registration.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            paths = registration_paths(load_registration_config(config_path))
            sweep = _validated_sweep(root, count=10)
            targets = select_registration_targets(
                sweep,
                10.0,
                {0: "sweep/sweep_000000.png"},
            )[:2]
            image_dir = paths.dataset_dir / "images"
            mask_dir = paths.dataset_dir / "masks"
            packaged_model = paths.dataset_dir / "colmap" / "sparse" / "0"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            packaged_model.mkdir(parents=True)
            paths.final_model_dir.mkdir(parents=True)
            paths.references_dir.mkdir(parents=True)
            for target in targets:
                target.source_frame.image_path.parent.mkdir(parents=True, exist_ok=True)
                target.source_frame.mask_path.parent.mkdir(parents=True, exist_ok=True)
                target.source_frame.image_path.write_bytes(
                    f"image-{target.source_frame.source_index}".encode()
                )
                target.source_frame.mask_path.write_bytes(
                    f"mask-{target.source_frame.source_index}".encode()
                )
                (image_dir / target.staged_name).write_bytes(
                    target.source_frame.image_path.read_bytes()
                )
                (mask_dir / target.staged_name).write_bytes(
                    target.source_frame.mask_path.read_bytes()
                )
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                (paths.final_model_dir / name).write_bytes(name.encode())
                (packaged_model / name).write_bytes(name.encode())

            extra_name = "sweep_000001.png"
            (image_dir / extra_name).write_bytes(b"extra-image")
            (mask_dir / extra_name).write_bytes(b"extra-mask")
            base = SimpleNamespace(references=())
            with self.assertRaisesRegex(ValueError, "target-only"):
                _validate_existing_package(paths, base, targets)

            (image_dir / extra_name).unlink()
            (mask_dir / extra_name).unlink()
            _validate_existing_package(paths, base, targets)

    def test_dry_run_does_not_write_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _config_payload(root)
            config_path = root / "registration.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            config = load_registration_config(config_path)
            paths = registration_paths(config)
            sweep = _validated_sweep(root)
            seed_selection = select_sweep_frames(sweep, 3.0)
            base_seed_frames = tuple(
                BaseSeedFrame(
                    source_index=selection.source_frame.source_index,
                    staged_relative_name=f"sweep/{selection.staged_name}",
                )
                for selection in seed_selection
            )
            registered_seeds = {
                seed.source_index: seed.staged_relative_name
                for seed in base_seed_frames
            }
            base = SimpleNamespace(
                run_dir=root / "base",
                sweep=sweep,
                base_seed_frames=base_seed_frames,
                registered_seed_by_source_index=registered_seeds,
                resolved={"sweep": {"sample_fps_hz": 3.0}},
                source_manifest={"identity": "test"},
                source_manifest_path=root / "source_sequences.json",
                sweep_camera_id=17,
                artifact_hashes={"database.db": "test-database-hash"},
            )

            output = io.StringIO()
            with patch(
                "preproc.register_prestatic_sweep_frames._load_base_context",
                return_value=base,
            ), contextlib.redirect_stdout(output):
                result = run_registration_pipeline(config, dry_run=True)

            report = json.loads(output.getvalue())
            self.assertIsNone(result)
            self.assertEqual(report["source_sweep_frames"], 1467)
            self.assertEqual(report["base_seed_frames"], 147)
            self.assertEqual(report["target_frames"], 489)
            self.assertEqual(report["reused_seed_targets"], 49)
            self.assertEqual(report["new_images"], 440)
            self.assertEqual(report["registration_targets"], 440)
            self.assertGreater(report["temporal_pairs"], 0)
            self.assertEqual(
                report["transforms"],
                {
                    "video_decode": False,
                    "rotate": False,
                    "crop": False,
                    "resize": False,
                },
            )
            self.assertFalse(paths.run_dir.exists())

    def test_sqlite_backup_is_independent_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "base.db"
            derived = root / "derived" / "database.db"
            with sqlite3.connect(source) as connection:
                connection.execute("CREATE TABLE seeds (value INTEGER)")
                connection.execute("INSERT INTO seeds VALUES (3)")

            backup_sqlite_database(source, derived)
            with sqlite3.connect(derived) as connection:
                connection.execute("INSERT INTO seeds VALUES (10)")
            with sqlite3.connect(source) as connection:
                self.assertEqual(
                    connection.execute("SELECT value FROM seeds").fetchall(), [(3,)]
                )
            with sqlite3.connect(derived) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM seeds ORDER BY value"
                    ).fetchall(),
                    [(3,), (10,)],
                )
            with self.assertRaises(FileExistsError):
                backup_sqlite_database(source, derived)


if __name__ == "__main__":
    unittest.main()
