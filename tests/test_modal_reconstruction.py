from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import run_modal_reconstruction as reconstruction


class _FakeDataset:
    def __init__(self, frame_names: list[str], height: int = 16, width: int = 16):
        self.frame_names = frame_names
        self.num_frames = len(frame_names)
        self._images = [
            torch.full((height, width, 3), 0.2 + 0.1 * index)
            for index in range(self.num_frames)
        ]
        self._masks = [torch.ones((height, width)) for _ in frame_names]
        self._Ks = torch.eye(3)[None].repeat(self.num_frames, 1, 1)
        self._w2cs = torch.eye(4)[None].repeat(self.num_frames, 1, 1)

    def get_image(self, index: int) -> torch.Tensor:
        return self._images[index]

    def get_mask(self, index: int) -> torch.Tensor:
        return self._masks[index]

    def load_image(self, index: int) -> torch.Tensor:
        return self._images[index]

    def load_mask(self, index: int) -> torch.Tensor:
        return self._masks[index]

    def get_Ks(self) -> torch.Tensor:
        return self._Ks

    def get_w2cs(self) -> torch.Tensor:
        return self._w2cs


class _FakeModel:
    def __init__(self, dataset: _FakeDataset):
        self.trajectory_type = "modal_activation"
        self.num_frames = dataset.num_frames
        self.Ks = dataset.get_Ks().clone()
        self.w2cs = dataset.get_w2cs().clone()
        self.modal_phi_real = torch.zeros((1, 2, 3))
        view_ids: list[str] = []
        frame_view_indices = []
        frame_local_indices = []
        for frame_name in dataset.frame_names:
            view_id, local_index_text = frame_name.rsplit("_", maxsplit=1)
            if view_id not in view_ids:
                view_ids.append(view_id)
            frame_view_indices.append(view_ids.index(view_id))
            frame_local_indices.append(int(local_index_text))
        self.modal_frame_view_indices = torch.tensor(frame_view_indices)
        self.modal_frame_local_indices = torch.tensor(frame_local_indices)
        self.modal = SimpleNamespace(
            params={
                "activations": torch.arange(
                    dataset.num_frames * 2,
                    dtype=torch.float32,
                ).reshape(dataset.num_frames, 1, 2)
            }
        )
        self.has_bg = True

    def render(
        self,
        t: int,
        w2cs: torch.Tensor,
        Ks: torch.Tensor,
        img_wh: tuple[int, int],
        bg_color: float,
    ) -> dict[str, torch.Tensor]:
        del t, Ks, bg_color
        width, height = img_wh
        return {
            "img": torch.full(
                (1, height, width, 3),
                0.25,
                device=w2cs.device,
            )
        }


class _FakeVideoWriter:
    def __init__(self, path: str, fps: float):
        self.path = Path(path)
        self.path.touch()
        self.fps = fps
        self.frames: list[np.ndarray] = []
        self.closed = False

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def close(self) -> None:
        self.closed = True


def _write_frame_map(path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 1,
                "views": ["view1", "view2"],
                "frames": [
                    {
                        "frame_name": "view2_000001",
                        "view_id": "view2",
                        "local_index": 1,
                    },
                    {
                        "frame_name": "view1_000001",
                        "view_id": "view1",
                        "local_index": 1,
                    },
                    {
                        "frame_name": "view2_000000",
                        "view_id": "view2",
                        "local_index": 0,
                    },
                    {
                        "frame_name": "view1_000000",
                        "view_id": "view1",
                        "local_index": 0,
                    },
                ],
            },
            f,
        )


def _write_three_view_frame_map(path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 1,
                "views": ["view1", "view2", "view3"],
                "frames": [
                    {
                        "frame_name": f"view{view_index}_{local_index:06d}",
                        "view_id": f"view{view_index}",
                        "local_index": local_index,
                    }
                    for view_index in range(1, 4)
                    for local_index in range(2)
                ],
            },
            f,
        )


class ModalReconstructionFrameTests(unittest.TestCase):
    def test_groups_dataset_frames_in_local_index_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame_map = Path(tmp) / "modal_frame_map.json"
            _write_frame_map(frame_map)
            view_ids, frames_by_view = reconstruction._load_reconstruction_frames(
                frame_map,
                [
                    "view1_000000",
                    "view1_000001",
                    "view2_000000",
                    "view2_000001",
                ],
            )

        self.assertEqual(view_ids, ["view1", "view2"])
        self.assertEqual(
            [frame.frame_name for frame in frames_by_view["view1"]],
            ["view1_000000", "view1_000001"],
        )
        self.assertEqual(
            [frame.ts for frame in frames_by_view["view2"]],
            [2, 3],
        )

    def test_rejects_duplicate_local_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame_map = Path(tmp) / "modal_frame_map.json"
            with frame_map.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "views": ["view1"],
                        "frames": [
                            {
                                "frame_name": "frame0",
                                "view_id": "view1",
                                "local_index": 0,
                            },
                            {
                                "frame_name": "frame1",
                                "view_id": "view1",
                                "local_index": 0,
                            },
                        ],
                    },
                    f,
                )
            with self.assertRaisesRegex(ValueError, "Duplicate local_index"):
                reconstruction._load_reconstruction_frames(
                    frame_map,
                    ["frame0", "frame1"],
                )

    def test_rejects_dataset_frame_missing_from_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame_map = Path(tmp) / "modal_frame_map.json"
            with frame_map.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "views": ["view1"],
                        "frames": [
                            {
                                "frame_name": "frame0",
                                "view_id": "view1",
                                "local_index": 0,
                            }
                        ],
                    },
                    f,
                )
            with self.assertRaisesRegex(ValueError, "missing 1 dataset frames"):
                reconstruction._load_reconstruction_frames(
                    frame_map,
                    ["frame0", "frame1"],
                )


class ModalReconstructionValidationTests(unittest.TestCase):
    def test_restores_matching_top_level_and_data_vggt_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            view_config = root / "view.json"
            view_config.write_text("{}", encoding="utf-8")
            frame_map = root / "frame_map.json"
            frame_map.write_text("{}", encoding="utf-8")
            train_cfg = {
                "trajectory_type": "modal_activation",
                "vggt_view_configs": (str(view_config),),
                "modal_frame_map": str(frame_map),
                "data": {
                    "data_dir": str(data_dir),
                    "camera_type": "vggt",
                    "vggt_view_configs": [str(view_config)],
                    "modal_frame_map": str(frame_map),
                    "load_depths": True,
                    "load_tracks": True,
                },
            }
            kwargs, loaded_frame_map = (
                reconstruction._dataset_kwargs_from_training_config(train_cfg)
            )

        self.assertEqual(kwargs["vggt_view_configs"], (str(view_config),))
        self.assertEqual(loaded_frame_map, frame_map)
        self.assertFalse(kwargs["load_depths"])
        self.assertFalse(kwargs["load_tracks"])

    def test_checkpoint_path_defaults_to_last_and_accepts_init(self) -> None:
        work_dir = Path("/tmp/work")
        self.assertEqual(
            reconstruction._resolve_checkpoint_path(work_dir, None),
            work_dir / "checkpoints" / "last.ckpt",
        )
        self.assertEqual(
            reconstruction._resolve_checkpoint_path(
                work_dir,
                "/tmp/work/checkpoints/init.ckpt",
            ),
            Path("/tmp/work/checkpoints/init.ckpt"),
        )

    def test_rejects_checkpoint_without_model_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "bad.ckpt"
            checkpoint.touch()
            with mock.patch.object(reconstruction.torch, "load", return_value={}):
                with self.assertRaisesRegex(ValueError, "has no model state"):
                    reconstruction._load_checkpoint_model(
                        checkpoint,
                        torch.device("cpu"),
                        False,
                    )

    def test_rejects_checkpoint_frame_count_mismatch(self) -> None:
        dataset = _FakeDataset(
            [
                "view1_000000",
                "view1_000001",
                "view2_000000",
                "view2_000001",
            ]
        )
        model = _FakeModel(dataset)
        model.num_frames = 3
        frames_by_view = {
            "view1": [
                reconstruction.ReconstructionFrame(
                    0, 0, "view1_000000", "view1", 0, 0
                ),
                reconstruction.ReconstructionFrame(
                    1, 1, "view1_000001", "view1", 0, 1
                ),
            ],
            "view2": [
                reconstruction.ReconstructionFrame(
                    2, 2, "view2_000000", "view2", 1, 0
                ),
                reconstruction.ReconstructionFrame(
                    3, 3, "view2_000001", "view2", 1, 1
                ),
            ],
        }
        with self.assertRaisesRegex(ValueError, "Checkpoint has 3 frames"):
            reconstruction._validate_model_alignment(
                model,
                dataset,
                ["view1", "view2"],
                frames_by_view,
            )

    def test_rejects_checkpoint_frame_map_mismatch(self) -> None:
        dataset = _FakeDataset(
            [
                "view1_000000",
                "view1_000001",
                "view2_000000",
                "view2_000001",
            ]
        )
        model = _FakeModel(dataset)
        model.modal_frame_local_indices = torch.tensor([1, 0, 0, 1])
        frames_by_view = {
            "view1": [
                reconstruction.ReconstructionFrame(
                    0, 0, "view1_000000", "view1", 0, 0
                ),
                reconstruction.ReconstructionFrame(
                    1, 1, "view1_000001", "view1", 0, 1
                ),
            ],
            "view2": [
                reconstruction.ReconstructionFrame(
                    2, 2, "view2_000000", "view2", 1, 0
                ),
                reconstruction.ReconstructionFrame(
                    3, 3, "view2_000001", "view2", 1, 1
                ),
            ],
        }
        with self.assertRaisesRegex(ValueError, "local indices do not match"):
            reconstruction._validate_model_alignment(
                model,
                dataset,
                ["view1", "view2"],
                frames_by_view,
            )

    def test_activation_magnitude_statistics(self) -> None:
        activations = torch.tensor(
            [
                [[3.0, 4.0]],
                [[0.0, 0.0]],
            ]
        )
        stats = reconstruction._activation_stats(activations, [0, 1])
        self.assertAlmostEqual(stats["rms"], np.sqrt(12.5))
        self.assertAlmostEqual(stats["p50"], 2.5)
        self.assertAlmostEqual(stats["p90"], 4.5)
        self.assertAlmostEqual(stats["max"], 5.0)

    def test_comparison_frame_pads_without_cropping(self) -> None:
        observed = torch.ones((17, 17, 3))
        rendered = torch.ones((17, 17, 3))
        target = torch.ones((17, 17, 3))

        frame = reconstruction._comparison_frame(observed, rendered, target)

        self.assertEqual(frame.shape, (32, 64, 3))
        self.assertTrue(np.all(frame[:17, :34] == 255))
        self.assertTrue(np.all(frame[:17, 34:51] == 0))
        self.assertTrue(np.all(frame[17:] == 0))
        self.assertTrue(np.all(frame[:, 51:] == 0))


class ModalReconstructionRunTests(unittest.TestCase):
    def test_writes_deterministic_view_videos_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work_dir = root / "work"
            checkpoint = work_dir / "checkpoints" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            frame_map = root / "modal_frame_map.json"
            _write_three_view_frame_map(frame_map)
            output_dir = root / "reconstruction"
            dataset = _FakeDataset(
                [
                    "view1_000000",
                    "view1_000001",
                    "view2_000000",
                    "view2_000001",
                    "view3_000000",
                    "view3_000001",
                ]
            )
            model = _FakeModel(dataset)
            writers: list[_FakeVideoWriter] = []

            def open_writer(path: str, fps: float) -> _FakeVideoWriter:
                writer = _FakeVideoWriter(path, fps)
                writers.append(writer)
                return writer

            with (
                mock.patch.object(
                    reconstruction,
                    "_load_training_config",
                    return_value={"use_2dgs": False},
                ),
                mock.patch.object(
                    reconstruction,
                    "_load_training_dataset",
                    return_value=(dataset, frame_map),
                ),
                mock.patch.object(
                    reconstruction,
                    "_load_checkpoint_model",
                    return_value=model,
                ) as load_model,
                mock.patch.object(
                    reconstruction.imageio,
                    "get_writer",
                    side_effect=open_writer,
                ),
                mock.patch.object(
                    reconstruction.torch.cuda,
                    "is_available",
                    return_value=False,
                ),
            ):
                reconstruction.run(
                    reconstruction.ModalReconstructionConfig(
                        work_dir=str(work_dir),
                        out_dir=str(output_dir),
                    )
                )

            load_model.assert_called_once_with(
                checkpoint,
                torch.device("cpu"),
                False,
            )
            self.assertTrue((output_dir / "view1_comparison.mp4").is_file())
            self.assertTrue((output_dir / "view2_comparison.mp4").is_file())
            self.assertTrue((output_dir / "view3_comparison.mp4").is_file())
            self.assertEqual([len(writer.frames) for writer in writers], [2, 2, 2])
            self.assertTrue(all(writer.closed for writer in writers))
            self.assertTrue(
                all(
                    frame.shape == (16, 48, 3)
                    for writer in writers
                    for frame in writer.frames
                )
            )
            self.assertTrue(
                all(
                    frame.dtype == np.uint8
                    for writer in writers
                    for frame in writer.frames
                )
            )

            with (output_dir / "metrics.json").open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            self.assertEqual(metrics["version"], 1)
            self.assertEqual(metrics["view_order"], ["view1", "view2", "view3"])
            self.assertEqual(metrics["num_modes"], 1)
            self.assertEqual(
                set(metrics["views"]),
                {"view1", "view2", "view3"},
            )
            self.assertEqual(metrics["overall"]["frame_count"], 6)
            self.assertEqual(metrics["overall"]["valid_pixel_count"], 6 * 16 * 16)
            self.assertEqual(
                set(metrics["overall"]["activation_magnitude"]),
                {"rms", "p50", "p90", "max"},
            )
            self.assertEqual(list(root.glob(".reconstruction.tmp-*")), [])

    def test_removes_temporary_output_after_export_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work_dir = root / "work"
            checkpoint = work_dir / "checkpoints" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            frame_map = root / "modal_frame_map.json"
            _write_frame_map(frame_map)
            output_dir = root / "failed_reconstruction"
            dataset = _FakeDataset(
                [
                    "view1_000000",
                    "view1_000001",
                    "view2_000000",
                    "view2_000001",
                ]
            )
            model = _FakeModel(dataset)

            def open_writer(path: str, fps: float) -> _FakeVideoWriter:
                return _FakeVideoWriter(path, fps)

            with (
                mock.patch.object(
                    reconstruction,
                    "_load_training_config",
                    return_value={"use_2dgs": False},
                ),
                mock.patch.object(
                    reconstruction,
                    "_load_training_dataset",
                    return_value=(dataset, frame_map),
                ),
                mock.patch.object(
                    reconstruction,
                    "_load_checkpoint_model",
                    return_value=model,
                ),
                mock.patch.object(
                    reconstruction.imageio,
                    "get_writer",
                    side_effect=open_writer,
                ),
                mock.patch.object(
                    reconstruction,
                    "_write_metrics",
                    side_effect=RuntimeError("write failed"),
                ),
                mock.patch.object(
                    reconstruction.torch.cuda,
                    "is_available",
                    return_value=False,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    reconstruction.run(
                        reconstruction.ModalReconstructionConfig(
                            work_dir=str(work_dir),
                            out_dir=str(output_dir),
                        )
                    )

            self.assertFalse(output_dir.exists())
            self.assertEqual(
                list(root.glob(".failed_reconstruction.tmp-*")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
