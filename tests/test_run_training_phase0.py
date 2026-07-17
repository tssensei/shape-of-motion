from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from flow3d.configs import LossesConfig
from flow3d.data import CustomDataConfig, DavisDataConfig
from run_training import (
    _inject_vggt_static_view_config,
    _log_trainable_parameters,
    _save_new_initial_checkpoints,
    _save_training_completion_checkpoint,
    _validate_checkpoint_policy,
    initialize_and_checkpoint_model,
)


class ModalRgbOnlyDataConfigTests(unittest.TestCase):
    @staticmethod
    def _config(data, losses: LossesConfig):
        return SimpleNamespace(
            trajectory_type="modal_activation",
            data=data,
            loss=losses,
            vggt_view_configs=("view1.json", "view2.json", "view3.json"),
            modal_frame_map="modal_frame_map.json",
            modal_train_view_id=None,
            modal_max_local_frames_per_view=60,
        )

    def test_modal_vggt_disables_tracks_and_zero_weight_depths(self) -> None:
        for config_type in (CustomDataConfig, DavisDataConfig):
            with self.subTest(config_type=config_type.__name__):
                cfg = self._config(
                    config_type(
                        data_dir="dataset",
                        camera_type="vggt",
                        load_tracks=True,
                        load_depths=True,
                    ),
                    LossesConfig(
                        w_depth_reg=0.0,
                        w_depth_grad=0.0,
                        w_depth_const=0.0,
                    ),
                )

                _inject_vggt_static_view_config(cfg)

                self.assertFalse(cfg.data.load_tracks)
                self.assertFalse(cfg.data.load_depths)
                self.assertEqual(cfg.data.vggt_view_configs, cfg.vggt_view_configs)
                self.assertEqual(cfg.data.modal_frame_map, cfg.modal_frame_map)
                self.assertEqual(cfg.data.modal_max_local_frames_per_view, 60)

    def test_any_positive_depth_loss_enables_depth_loading(self) -> None:
        for field in ("w_depth_reg", "w_depth_grad", "w_depth_const"):
            with self.subTest(field=field):
                losses = LossesConfig(
                    w_depth_reg=0.0,
                    w_depth_grad=0.0,
                    w_depth_const=0.0,
                )
                setattr(losses, field, 0.25)
                cfg = self._config(
                    CustomDataConfig(
                        data_dir="dataset",
                        camera_type="vggt",
                        load_tracks=True,
                        load_depths=False,
                    ),
                    losses,
                )

                _inject_vggt_static_view_config(cfg)

                self.assertFalse(cfg.data.load_tracks)
                self.assertTrue(cfg.data.load_depths)

    def test_negative_depth_loss_does_not_enable_depth_loading(self) -> None:
        losses = LossesConfig(
            w_depth_reg=-1.0,
            w_depth_grad=0.0,
            w_depth_const=0.0,
        )
        cfg = self._config(
            CustomDataConfig(
                data_dir="dataset",
                camera_type="vggt",
                load_depths=True,
            ),
            losses,
        )

        _inject_vggt_static_view_config(cfg)

        self.assertFalse(cfg.data.load_depths)


class PhaseZeroCheckpointTests(unittest.TestCase):
    def test_new_initialization_writes_init_and_last_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            last_path = Path(tmp) / "checkpoints" / "last.ckpt"
            checkpoint = {
                "model": {"value": torch.tensor([1.0])},
                "epoch": 0,
                "global_step": 0,
                "init_metadata": {"trajectory_type": "modal_activation"},
            }

            _save_new_initial_checkpoints(checkpoint, str(last_path))

            init_path = last_path.with_name("init.ckpt")
            self.assertTrue(init_path.is_file())
            self.assertTrue(last_path.is_file())
            for path in (init_path, last_path):
                saved = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(saved["epoch"], 0)
                self.assertEqual(saved["global_step"], 0)
                self.assertEqual(saved["init_metadata"], checkpoint["init_metadata"])
                torch.testing.assert_close(
                    saved["model"]["value"], checkpoint["model"]["value"]
                )

    def test_existing_init_checkpoint_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            checkpoint_dir.mkdir()
            last_path = checkpoint_dir / "last.ckpt"
            init_path = checkpoint_dir / "init.ckpt"
            sentinel = b"existing initialization"
            init_path.write_bytes(sentinel)

            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                _save_new_initial_checkpoints(
                    {"model": {}, "epoch": 0, "global_step": 0},
                    str(last_path),
                )

            self.assertEqual(init_path.read_bytes(), sentinel)
            self.assertFalse(last_path.exists())

    def test_existing_training_checkpoint_skips_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            checkpoint_dir.mkdir()
            last_path = checkpoint_dir / "last.ckpt"
            init_path = checkpoint_dir / "init.ckpt"
            last_path.write_bytes(b"existing training")
            init_path.write_bytes(b"existing initialization")

            initialize_and_checkpoint_model(
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                torch.device("cpu"),
                str(last_path),
                {},
            )

            self.assertEqual(last_path.read_bytes(), b"existing training")
            self.assertEqual(init_path.read_bytes(), b"existing initialization")

    def test_orphaned_init_checkpoint_is_rejected_before_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            checkpoint_dir.mkdir()
            last_path = checkpoint_dir / "last.ckpt"
            (checkpoint_dir / "init.ckpt").write_bytes(b"existing initialization")

            with self.assertRaisesRegex(ValueError, "Initialization checkpoint"):
                _validate_checkpoint_policy(
                    str(last_path),
                    False,
                    {"trajectory_type": "modal_activation"},
                )

    def test_training_completion_always_saves_last_checkpoint(self) -> None:
        saved_paths: list[str] = []
        trainer = SimpleNamespace(
            save_checkpoint=lambda path: saved_paths.append(path),
        )

        _save_training_completion_checkpoint(
            trainer,  # type: ignore[arg-type]
            "run/checkpoints/last.ckpt",
        )

        self.assertEqual(saved_paths, ["run/checkpoints/last.ckpt"])

    def test_trainable_parameter_log_reports_only_enabled_names(self) -> None:
        model = torch.nn.Module()
        model.register_parameter("enabled", torch.nn.Parameter(torch.ones(())))
        model.register_parameter("frozen", torch.nn.Parameter(torch.ones(())))
        model.frozen.requires_grad_(False)  # type: ignore[union-attr]

        with patch("run_training.guru.info") as info:
            _log_trainable_parameters(model)  # type: ignore[arg-type]

        message = info.call_args.args[0]
        self.assertIn("enabled", message)
        self.assertNotIn("frozen", message)


if __name__ == "__main__":
    unittest.main()
