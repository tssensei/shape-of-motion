from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from preproc.prepare_joint_colmap_video_dataset import ModelCandidate
from preproc.run_prestatic_pipeline import (
    ColmapConfig,
    PipelineConfig,
    SequenceConfig,
    SweepSelection,
    _package_static_dataset,
    _pipeline_paths,
    _stage_colmap_images,
    load_pipeline_config,
    run_pipeline,
    select_reference_frame,
    select_sweep_frames,
    validate_camera_group_assignments,
    validate_sequence,
)


def _write_rgb(path: Path, value: int, size: tuple[int, int] = (8, 6)) -> None:
    width, height = size
    image = np.full((height, width, 3), value, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write test image {path}")


def _write_mask(path: Path, value: int, size: tuple[int, int] = (8, 6)) -> None:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[1:-1, 1:-1] = value
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Could not write test mask {path}")


def _write_sequence(
    root: Path,
    view_id: str,
    count: int,
    *,
    size: tuple[int, int] = (8, 6),
) -> SequenceConfig:
    image_dir = root / "images" / view_id
    mask_dir = root / "masks" / view_id
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index in range(count):
        name = f"{view_id}_{index:06d}.png"
        _write_rgb(image_dir / name, index % 255, size)
        _write_mask(mask_dir / name, 255, size)
    return SequenceConfig(
        view_id=view_id,
        image_dir=image_dir,
        mask_dir=mask_dir,
        fps_hz=15.0,
        camera_group=view_id,
        reference_policy=None,
        reference_frame=None,
    )


def _pipeline_config(
    root: Path,
    sweep: SequenceConfig,
    static_views: tuple[SequenceConfig, ...],
) -> PipelineConfig:
    scene_root = root / "scene_v1"
    return PipelineConfig(
        config_path=root / "config.yaml",
        scene_id="scene_v1",
        scene_root=scene_root,
        run_id="joint_colmap_frames_v1",
        sweep=sweep,
        static_views=static_views,
        colmap=ColmapConfig(
            sweep_sample_fps_hz=3.0,
            camera_model="SIMPLE_RADIAL",
            matcher="exhaustive",
            min_sweep_registration_ratio=0.8,
            command="colmap",
        ),
    )


class PrestaticPipelineTest(unittest.TestCase):
    def test_config_rejects_post_export_resize_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {
                "format": "som_prestatic_pipeline",
                "version": 1,
                "scene_id": "scene_v1",
                "scene_root": str(root / "scene_v1"),
                "run_id": "joint_colmap_frames_v1",
                "sweep": {
                    "view_id": "sweep",
                    "image_dir": str(root / "images" / "sweep"),
                    "mask_dir": str(root / "masks" / "sweep"),
                    "fps_hz": 15,
                    "camera_group": "sweep_camera",
                    "target_width": 960,
                },
                "static_views": [
                    {
                        "view_id": "view1",
                        "image_dir": str(root / "images" / "view1"),
                        "mask_dir": str(root / "masks" / "view1"),
                        "fps_hz": 30,
                        "camera_group": "fixed_camera",
                        "reference_policy": "middle",
                    }
                ],
                "colmap": {
                    "sweep_sample_fps_hz": 3,
                    "camera_model": "SIMPLE_RADIAL",
                    "matcher": "exhaustive",
                    "min_sweep_registration_ratio": 0.8,
                    "command": "colmap",
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported keys"):
                load_pipeline_config(config_path)

            del payload["sweep"]["target_width"]
            payload["static_views"][0]["view_id"] = "sweep"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reserved"):
                load_pipeline_config(config_path)

    def test_validation_rejects_non_rgb_and_non_uint8_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _write_sequence(root, "sweep", 2)
            grayscale_path = config.image_dir / "sweep_000001.png"
            _write_mask(grayscale_path, 255)
            with self.assertRaisesRegex(ValueError, "8-bit three-channel"):
                validate_sequence(config)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _write_sequence(root, "sweep", 2)
            uint16_path = config.image_dir / "sweep_000001.png"
            uint16_image = np.full((6, 8, 3), 1024, dtype=np.uint16)
            self.assertTrue(cv2.imwrite(str(uint16_path), uint16_image))
            with self.assertRaisesRegex(ValueError, "8-bit three-channel"):
                validate_sequence(config)

    def test_time_grid_sampling_and_reference_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sweep_config = _write_sequence(root, "sweep", 31)
            sweep = validate_sequence(sweep_config)
            selections = select_sweep_frames(sweep, 3.0)
            self.assertEqual(
                [selection.source_frame.source_index for selection in selections],
                [0, 5, 10, 15, 20, 25, 30],
            )
            self.assertEqual(
                [selection.staged_name for selection in selections],
                [f"sweep_{index:06d}.png" for index in range(7)],
            )

            static_config = _write_sequence(root, "view1", 10)
            static_config = replace(
                static_config,
                camera_group="fixed_camera",
                reference_policy="middle",
            )
            reference = select_reference_frame(validate_sequence(static_config))
            self.assertEqual(reference.source_frame.source_index, 5)
            self.assertEqual(
                reference.staged_relative_name,
                "static_fixed_camera/view1_ref.png",
            )

    def test_staging_copies_canonical_png_bytes_without_transform(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sweep_config = _write_sequence(root, "sweep", 6, size=(9, 7))
            static_config = _write_sequence(root, "view1", 5, size=(12, 8))
            static_config = replace(
                static_config,
                camera_group="fixed_camera",
                reference_policy="middle",
            )
            config = _pipeline_config(root, sweep_config, (static_config,))
            paths = _pipeline_paths(config)
            paths.reports_dir.mkdir(parents=True)
            sweep = validate_sequence(sweep_config)
            selections = select_sweep_frames(sweep, 3.0)
            reference = select_reference_frame(validate_sequence(static_config))
            _stage_colmap_images(paths, selections, (reference,), sweep, 3.0)
            _stage_colmap_images(paths, selections, (reference,), sweep, 3.0)

            for selection in selections:
                staged = paths.workspace_images / "sweep" / selection.staged_name
                self.assertEqual(
                    staged.read_bytes(), selection.source_frame.image_path.read_bytes()
                )
            staged_reference = paths.workspace_images / reference.staged_relative_name
            self.assertEqual(
                staged_reference.read_bytes(),
                reference.source_frame.image_path.read_bytes(),
            )

    def test_validation_rejects_image_mask_dimension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _write_sequence(root, "sweep", 2)
            _write_mask(config.mask_dir / "sweep_000001.png", 255, size=(4, 4))
            with self.assertRaisesRegex(ValueError, "Image/mask dimensions differ"):
                validate_sequence(config)

    def test_camera_group_validation_supports_shared_fixed_intrinsics(self) -> None:
        records = [
            {"source": "sweep", "camera_id": 1},
            {"source": "sweep", "camera_id": 1},
            {"source": "view1", "camera_id": 2},
            {"source": "view2", "camera_id": 2},
        ]
        result = validate_camera_group_assignments(
            records,
            {
                "sweep": "sweep_camera",
                "view1": "fixed_camera",
                "view2": "fixed_camera",
            },
        )
        self.assertEqual(result, {"sweep_camera": 1, "fixed_camera": 2})

    def test_packaging_keeps_only_registered_sweep_images_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sweep_config = _write_sequence(root, "sweep", 3)
            static_config = _write_sequence(root, "view1", 3)
            static_config = replace(
                static_config,
                camera_group="fixed_camera",
                reference_policy="middle",
            )
            config = _pipeline_config(root, sweep_config, (static_config,))
            paths = _pipeline_paths(config)
            paths.run_dir.mkdir(parents=True)
            paths.reports_dir.mkdir()
            sweep = validate_sequence(sweep_config)
            selections = tuple(
                SweepSelection(
                    staged_name=f"sweep_{index:06d}.png",
                    source_frame=frame,
                    target_time_sec=frame.time_sec,
                )
                for index, frame in enumerate(sweep.frames)
            )
            reference = select_reference_frame(validate_sequence(static_config))
            model_dir = root / "model"
            model_dir.mkdir()
            for filename in ("cameras.bin", "images.bin", "points3D.bin"):
                (model_dir / filename).write_bytes(b"test")
            selected = ModelCandidate(
                path=model_dir,
                registered_names=frozenset(
                    {
                        "sweep/sweep_000000.png",
                        "sweep/sweep_000002.png",
                        reference.staged_relative_name,
                    }
                ),
                registered_sweep_count=2,
                point_count=1,
            )

            _package_static_dataset(paths, selected, selections, (reference,))
            _package_static_dataset(paths, selected, selections, (reference,))

            self.assertEqual(
                sorted(path.name for path in (paths.dataset_dir / "images").iterdir()),
                ["sweep_000000.png", "sweep_000002.png"],
            )
            self.assertEqual(
                sorted(path.name for path in (paths.dataset_dir / "masks").iterdir()),
                ["sweep_000000.png", "sweep_000002.png"],
            )
            self.assertEqual(
                (paths.dataset_dir / "masks" / "sweep_000002.png").read_bytes(),
                selections[2].source_frame.mask_path.read_bytes(),
            )

    def test_dry_run_validates_but_writes_no_formal_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sweep_config = _write_sequence(root, "sweep", 6)
            static_config = _write_sequence(root, "view1", 5)
            static_config = replace(
                static_config,
                camera_group="fixed_camera",
                reference_policy="middle",
            )
            config = _pipeline_config(root, sweep_config, (static_config,))
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_pipeline(config, dry_run=True)
            self.assertIsNone(result)
            self.assertFalse(config.scene_root.exists())


if __name__ == "__main__":
    unittest.main()
