from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from preproc.prepare_modal_dynamic_dataset import (
    ViewSpec,
    parse_view_spec,
    prepare_modal_dynamic_dataset,
    validate_prepared_dataset,
)


class PrepareModalDynamicDatasetTests(unittest.TestCase):
    def _write_view(
        self,
        root: Path,
        view_id: str,
        *,
        height: int = 4,
        width: int = 8,
        fps_hz: float = 30.0,
        frame_names: tuple[str, ...] = ("frame_b", "frame_a"),
    ) -> ViewSpec:
        image_dir = root / view_id / "images"
        mask_dir = root / view_id / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for index, frame_name in enumerate(("frame_a", "frame_b")):
            image = np.full(
                (height, width, 3),
                fill_value=20 + index * 80,
                dtype=np.uint8,
            )
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[:, index * (width // 2) : (index + 1) * (width // 2)] = 7
            self.assertTrue(cv2.imwrite(str(image_dir / f"{frame_name}.png"), image))
            self.assertTrue(cv2.imwrite(str(mask_dir / f"{frame_name}.png"), mask))
        sidecar_path = root / view_id / "frame_names.json"
        with sidecar_path.open("w", encoding="utf-8") as f:
            json.dump(list(frame_names), f)
        return ViewSpec(view_id, image_dir, mask_dir, sidecar_path, fps_hz)

    def test_parse_view_spec(self) -> None:
        parsed = parse_view_spec("view1=images=masks=frames.json=29.97")
        self.assertEqual(parsed.view_id, "view1")
        self.assertEqual(parsed.image_dir, Path("images"))
        self.assertEqual(parsed.mask_dir, Path("masks"))
        self.assertEqual(parsed.frame_names_json, Path("frames.json"))
        self.assertAlmostEqual(parsed.fps_hz, 29.97)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_view_spec("view1=images=masks=frames.json=0")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_view_spec("view/1=images=masks=frames.json=30")

    def test_prepares_three_views_in_sidecar_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            views = [self._write_view(root, f"view{index}") for index in range(1, 4)]
            output = root / "prepared"

            returned = prepare_modal_dynamic_dataset(views, 4, output)
            validate_prepared_dataset(returned)

            with (output / "modal_frame_map.json").open("r", encoding="utf-8") as f:
                frame_map = json.load(f)
            with (output / "metadata.json").open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.assertEqual(returned, output)
            self.assertEqual(frame_map["version"], 1)
            self.assertEqual(frame_map["views"], ["view1", "view2", "view3"])
            self.assertEqual(
                frame_map["view_fps_hz"],
                {"view1": 30.0, "view2": 30.0, "view3": 30.0},
            )
            self.assertEqual(len(frame_map["frames"]), 6)
            self.assertEqual(
                frame_map["frames"][:3],
                [
                    {
                        "frame_name": "view1_000000",
                        "view_id": "view1",
                        "local_index": 0,
                        "time_sec": 0.0,
                        "source_frame_name": "frame_b",
                    },
                    {
                        "frame_name": "view1_000001",
                        "view_id": "view1",
                        "local_index": 1,
                        "time_sec": 1.0 / 30.0,
                        "source_frame_name": "frame_a",
                    },
                    {
                        "frame_name": "view2_000000",
                        "view_id": "view2",
                        "local_index": 0,
                        "time_sec": 0.0,
                        "source_frame_name": "frame_b",
                    },
                ],
            )
            self.assertEqual(metadata["format"], "modal_dynamic_dataset")
            self.assertEqual(metadata["frame_count"], 6)
            self.assertEqual(metadata["target_resolution"], {"width": 4, "height": 2})
            self.assertEqual(metadata["views"][0]["frame_count"], 2)
            self.assertEqual(
                metadata["views"][0]["source_resolution"],
                {"width": 8, "height": 4},
            )
            output_image = cv2.imread(
                str(output / "images" / "view1_000000.png"),
                cv2.IMREAD_COLOR,
            )
            output_mask = cv2.imread(
                str(output / "masks" / "view1_000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertIsNotNone(output_image)
            self.assertIsNotNone(output_mask)
            assert output_image is not None
            assert output_mask is not None
            self.assertEqual(output_image.shape, (2, 4, 3))
            self.assertEqual(output_mask.shape, (2, 4))
            self.assertTrue(np.all(output_image == 100))
            self.assertEqual(set(np.unique(output_mask).tolist()), {0, 255})

    def test_rejects_duplicate_view_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = self._write_view(root, "view1")
            with self.assertRaisesRegex(ValueError, "View IDs must be unique"):
                prepare_modal_dynamic_dataset([view, view], 4, root / "prepared")

    def test_rejects_duplicate_sidecar_frame_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = self._write_view(
                root,
                "view1",
                frame_names=("frame_a", "frame_a"),
            )
            with self.assertRaisesRegex(ValueError, "Duplicate frame name"):
                prepare_modal_dynamic_dataset([view], 4, root / "prepared")

    def test_rejects_source_names_that_do_not_match_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = self._write_view(
                root,
                "view1",
                frame_names=("frame_a", "missing_frame"),
            )
            with self.assertRaisesRegex(ValueError, "names do not match"):
                prepare_modal_dynamic_dataset([view], 4, root / "prepared")

    def test_rejects_missing_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = self._write_view(root, "view1")
            (view.mask_dir / "frame_b.png").unlink()
            with self.assertRaisesRegex(ValueError, "mask names do not match"):
                prepare_modal_dynamic_dataset([view], 4, root / "prepared")

    def test_rejects_image_mask_dimension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = self._write_view(root, "view1")
            self.assertTrue(
                cv2.imwrite(
                    str(view.mask_dir / "frame_a.png"),
                    np.ones((3, 8), dtype=np.uint8),
                )
            )
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                prepare_modal_dynamic_dataset([view], 4, root / "prepared")

    def test_rejects_different_view_aspect_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view1 = self._write_view(root, "view1", height=4, width=8)
            view2 = self._write_view(root, "view2", height=4, width=6)
            with self.assertRaisesRegex(ValueError, "same aspect ratio"):
                prepare_modal_dynamic_dataset([view1, view2], 4, root / "prepared")

    def test_rejects_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = self._write_view(root, "view1")
            output = root / "prepared"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                prepare_modal_dynamic_dataset([view], 4, output)

    def test_validation_rejects_inconsistent_fps_time_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = self._write_view(root, "view1")
            output = prepare_modal_dynamic_dataset([view], 4, root / "prepared")
            frame_map_path = output / "modal_frame_map.json"
            metadata_path = output / "metadata.json"
            frame_map = json.loads(frame_map_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            invalid_fps = json.loads(json.dumps(frame_map))
            invalid_fps["view_fps_hz"]["view1"] = 0.0
            frame_map_path.write_text(json.dumps(invalid_fps), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid FPS"):
                validate_prepared_dataset(output)

            invalid_time = json.loads(json.dumps(frame_map))
            invalid_time["frames"][1]["time_sec"] = 1.0
            frame_map_path.write_text(json.dumps(invalid_time), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inconsistent time_sec"):
                validate_prepared_dataset(output)

            frame_map_path.write_text(json.dumps(frame_map), encoding="utf-8")
            metadata["frame_count"] += 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid frame_count"):
                validate_prepared_dataset(output)

    def test_failed_write_cleans_temporary_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = self._write_view(root, "view1")
            output = root / "prepared"
            with mock.patch(
                "preproc.prepare_modal_dynamic_dataset._write_png",
                side_effect=OSError("injected write failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected write failure"):
                    prepare_modal_dynamic_dataset([view], 4, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".prepared.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
