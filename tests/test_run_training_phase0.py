from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from flow3d.configs import LossesConfig
from flow3d.data import CustomDataConfig, DavisDataConfig
from flow3d.trainer import Trainer, _harmonic_activation_gradient_norm
from run_training import (
    TrainConfig,
    _inject_vggt_static_view_config,
    _load_stage1_gaussians_from_checkpoint,
    _log_trainable_parameters,
    _make_init_metadata,
    _save_new_initial_checkpoints,
    _save_training_completion_checkpoint,
    _validate_checkpoint_policy,
    _zero_harmonic_activations,
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

    def test_phase_zero_checkpoint_is_rejected_for_harmonic_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            last_path = Path(tmp) / "checkpoints" / "last.ckpt"
            last_path.parent.mkdir()
            torch.save(
                {
                    "init_metadata": {
                        "trajectory_type": "modal_activation",
                    }
                },
                last_path,
            )

            with self.assertRaisesRegex(
                ValueError,
                "incompatible modal parameterization",
            ):
                _validate_checkpoint_policy(
                    str(last_path),
                    True,
                    {
                        "trajectory_type": "modal_activation",
                        "modal_parameterization": "per_view_harmonic_v1",
                    },
                )

    def test_phase_zero_checkpoint_is_rejected_as_static_gaussian_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "phase0.ckpt"
            torch.save(
                {
                    "model": {
                        "modal.params.activations": torch.zeros((3, 1, 2)),
                    }
                },
                checkpoint,
            )

            with self.assertRaisesRegex(ValueError, "original static checkpoint"):
                _load_stage1_gaussians_from_checkpoint(
                    str(checkpoint), torch.device("cpu")
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


class HarmonicTrainingConfigTests(unittest.TestCase):
    def test_fresh_harmonic_activation_is_exact_zero_per_view(self) -> None:
        modal = _zero_harmonic_activations(
            3,
            2,
            torch.device("cpu"),
            torch.float32,
        )

        activations = modal.params["activations"]
        self.assertEqual(tuple(activations.shape), (3, 2, 2))
        self.assertEqual(int(torch.count_nonzero(activations).item()), 0)

    def test_modal_stage2_freezes_all_gaussian_parameters_by_default(self) -> None:
        field_names = (
            "modal_stage2_train_base_means",
            "modal_stage2_train_colors",
            "modal_stage2_train_opacities",
            "modal_stage2_train_scales",
            "modal_stage2_train_quats",
            "modal_stage2_train_bg_means",
            "modal_stage2_train_bg_colors",
            "modal_stage2_train_bg_opacities",
            "modal_stage2_train_bg_scales",
            "modal_stage2_train_bg_quats",
        )

        for field_name in field_names:
            with self.subTest(field_name=field_name):
                self.assertFalse(
                    TrainConfig.__dataclass_fields__[field_name].default
                )

    def test_removed_per_frame_training_options_are_not_config_fields(self) -> None:
        self.assertNotIn("w_act_smooth", LossesConfig.__dataclass_fields__)
        self.assertNotIn("w_act_modal_consistency", LossesConfig.__dataclass_fields__)
        self.assertFalse(
            any(
                field_name.startswith("modal_consistency")
                for field_name in TrainConfig.__dataclass_fields__
            )
        )

    def test_modal_init_metadata_declares_harmonic_parameterization(self) -> None:
        cfg = TrainConfig(
            work_dir="run",
            data=CustomDataConfig(data_dir="dataset"),
            lr=None,  # type: ignore[arg-type]
            loss=LossesConfig(),
            optim=None,  # type: ignore[arg-type]
            trajectory_type="modal_activation",
        )

        metadata = _make_init_metadata(cfg)

        self.assertEqual(
            metadata["modal_parameterization"],
            "per_view_harmonic_v1",
        )

    def test_dynamic_stage_trainability_defaults_to_activation_only(self) -> None:
        model = torch.nn.Module()
        model.trajectory_type = "modal_activation"
        model.fg = torch.nn.Module()
        model.fg.params = torch.nn.ParameterDict(
            {"means": torch.nn.Parameter(torch.zeros(1, 3))}
        )
        model.modal = torch.nn.Module()
        model.modal.params = torch.nn.ParameterDict(
            {"activations": torch.nn.Parameter(torch.zeros(3, 1, 2))}
        )
        model.motion_bases = torch.nn.Module()
        model.motion_bases.params = torch.nn.ParameterDict(
            {"rots": torch.nn.Parameter(torch.zeros(1))}
        )

        trainer = object.__new__(Trainer)
        trainer.model = model
        trainer.epoch = 0
        trainer.modal_warmup_epochs = 0
        for name in (
            "modal_stage2_train_base_means",
            "modal_stage2_train_colors",
            "modal_stage2_train_opacities",
            "modal_stage2_train_scales",
            "modal_stage2_train_quats",
            "modal_stage2_train_bg_means",
            "modal_stage2_train_bg_colors",
            "modal_stage2_train_bg_opacities",
            "modal_stage2_train_bg_scales",
            "modal_stage2_train_bg_quats",
        ):
            setattr(trainer, name, False)

        trainer._apply_modal_trainability()

        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(trainable, {"modal.params.activations"})

        trainer.modal_stage2_train_base_means = True
        trainer._apply_modal_trainability()
        self.assertTrue(model.fg.params["means"].requires_grad)


class HarmonicActivationGradientTests(unittest.TestCase):
    @staticmethod
    def _model_with_gradient(gradient: torch.Tensor | None):
        parameter = torch.nn.Parameter(torch.zeros(3, 1, 2))
        parameter.grad = gradient
        return SimpleNamespace(
            trajectory_type="modal_activation",
            modal=SimpleNamespace(params={"activations": parameter}),
        )

    def test_gradient_norm_is_reported(self) -> None:
        model = self._model_with_gradient(torch.tensor([[[3.0, 4.0]]] * 3))

        norm = _harmonic_activation_gradient_norm(model, 7)  # type: ignore[arg-type]

        self.assertAlmostEqual(norm, 5.0 * (3.0**0.5))

    def test_missing_gradient_is_zero_during_frozen_stage(self) -> None:
        model = self._model_with_gradient(None)

        norm = _harmonic_activation_gradient_norm(model, 7)  # type: ignore[arg-type]

        self.assertEqual(norm, 0.0)

    def test_nonfinite_gradient_is_rejected_before_optimizer_step(self) -> None:
        model = self._model_with_gradient(
            torch.tensor([[[float("nan"), 0.0]]] * 3)
        )

        with self.assertRaisesRegex(FloatingPointError, "step 7"):
            _harmonic_activation_gradient_norm(model, 7)  # type: ignore[arg-type]

    def test_modal_route_skips_control_steps_before_any_stage_check(self) -> None:
        trainer = object.__new__(Trainer)
        trainer.model = SimpleNamespace(trajectory_type="modal_activation")
        trainer._prepare_control_step = Mock(
            side_effect=AssertionError("control preparation must not run")
        )

        trainer.run_control_steps()

        trainer._prepare_control_step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
