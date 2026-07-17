from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import run_rendering
from flow3d.modal_utils import load_modal_frame_map
from flow3d.params import GaussianParams, ModalActivations, MotionBases
from flow3d.scene_model import SceneModel


def _make_modal_model(
    activations: torch.Tensor,
    phi: torch.Tensor,
    freqs_hz: torch.Tensor,
    frame_view_indices: torch.Tensor,
    frame_local_indices: torch.Tensor,
    frame_times_sec: torch.Tensor,
) -> SceneModel:
    num_frames = int(frame_view_indices.shape[0])
    num_gaussians = int(phi.shape[1])
    fg = GaussianParams(
        means=torch.zeros(num_gaussians, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_gaussians, 1),
        scales=torch.zeros(num_gaussians, 3),
        colors=torch.zeros(num_gaussians, 3),
        opacities=torch.zeros(num_gaussians),
    )
    motion_bases = MotionBases(
        rots=torch.zeros(1, num_frames, 6),
        transls=torch.zeros(1, num_frames, 3),
    )
    return SceneModel(
        Ks=torch.eye(3)[None].repeat(num_frames, 1, 1),
        w2cs=torch.eye(4)[None].repeat(num_frames, 1, 1),
        fg_params=fg,
        motion_bases=motion_bases,
        trajectory_type="modal_activation",
        modal=ModalActivations(activations),
        modal_phi_real=phi.real,
        modal_phi_imag=phi.imag,
        modal_freqs_hz=freqs_hz,
        modal_obs_count_per_point=torch.zeros(
            phi.shape[0], num_gaussians, dtype=torch.long
        ),
        modal_frame_view_indices=frame_view_indices,
        modal_frame_local_indices=frame_local_indices,
        modal_frame_times_sec=frame_times_sec,
    )


def _valid_frame_map_payload() -> dict:
    return {
        "version": 1,
        "views": ["view1", "view2"],
        "view_fps_hz": {"view1": 10.0, "view2": 5.0},
        "frames": [
            {
                "frame_name": f"view1_{index}",
                "view_id": "view1",
                "local_index": index,
                "time_sec": index / 10.0,
            }
            for index in range(3)
        ]
        + [
            {
                "frame_name": f"view2_{index}",
                "view_id": "view2",
                "local_index": index,
                "time_sec": index / 5.0,
            }
            for index in range(3)
        ],
    }


class ModalActivationParameterTests(unittest.TestCase):
    def test_parameter_shape_is_per_view(self) -> None:
        modal = ModalActivations(torch.zeros(3, 2, 2))

        self.assertEqual(modal.num_views, 3)
        self.assertEqual(modal.num_modes, 2)

    def test_parameter_rejects_empty_and_nonfinite_values(self) -> None:
        for activations, message in (
            (torch.zeros(0, 1, 2), "at least one view"),
            (torch.zeros(1, 0, 2), "at least one mode"),
            (torch.zeros(1, 1, 2, dtype=torch.long), "floating-point"),
            (torch.tensor([[[float("nan"), 0.0]]]), "finite"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ModalActivations(activations)


class HarmonicSceneModelTests(unittest.TestCase):
    def test_coefficients_and_offsets_match_complex_reference(self) -> None:
        activations = torch.tensor(
            [
                [[1.0, 0.5], [-0.25, 0.75]],
                [[0.2, -0.4], [0.8, 0.1]],
            ]
        )
        phi = torch.tensor(
            [
                [[1.0 + 0.5j, 0.0 + 0.2j, -0.3 + 0.0j],
                 [0.4 - 0.1j, 0.2 + 0.0j, 0.0 + 0.6j]],
                [[0.0 + 0.1j, -0.7 + 0.0j, 0.5 + 0.2j],
                 [0.3 + 0.0j, 0.0 - 0.4j, -0.2 + 0.1j]],
            ],
            dtype=torch.complex64,
        )
        freqs_hz = torch.tensor([1.0, 2.0])
        view_indices = torch.tensor([0, 0, 1, 1])
        local_indices = torch.tensor([0, 1, 0, 1])
        times_sec = torch.tensor([0.0, 0.25, 0.0, 0.125])
        model = _make_modal_model(
            activations,
            phi,
            freqs_hz,
            view_indices,
            local_indices,
            times_sec,
        )
        ts = torch.tensor([3, 0, 3])

        coefficient_real, coefficient_imag = model.compute_modal_coefficients(ts)
        amplitude = torch.complex(
            activations[view_indices[ts], :, 0],
            activations[view_indices[ts], :, 1],
        )
        phase = torch.exp(
            1j * 2.0 * torch.pi * times_sec[ts, None] * freqs_hz[None, :]
        )
        expected_coefficients = amplitude * phase
        torch.testing.assert_close(coefficient_real, expected_coefficients.real)
        torch.testing.assert_close(coefficient_imag, expected_coefficients.imag)

        expected_offsets = torch.einsum(
            "bk,kgc->gbc", expected_coefficients, phi
        ).real
        torch.testing.assert_close(model.compute_modal_offsets(ts), expected_offsets)
        torch.testing.assert_close(
            model.compute_modal_offsets(ts, torch.tensor([1])),
            expected_offsets[[1]],
        )

    def test_zero_initialization_has_finite_nonzero_motion_gradient(self) -> None:
        activations = torch.zeros(2, 1, 2)
        phi = torch.tensor(
            [[[1.0 + 0.5j, -0.25 + 0.75j, 0.4 - 0.1j]]],
            dtype=torch.complex64,
        )
        model = _make_modal_model(
            activations,
            phi,
            torch.tensor([1.0]),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([0.0, 0.25]),
        )

        loss = (model.compute_modal_offsets(torch.tensor([0, 1])) - 1.0).pow(2).mean()
        loss.backward()
        assert model.modal is not None
        gradient = model.modal.params["activations"].grad

        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all().item()))
        self.assertGreater(float(gradient.abs().sum().item()), 0.0)

    def test_invalid_ts_are_rejected(self) -> None:
        model = _make_modal_model(
            torch.zeros(1, 1, 2),
            torch.ones(1, 1, 3, dtype=torch.complex64),
            torch.tensor([1.0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0.0]),
        )

        with self.assertRaisesRegex(ValueError, "1-D"):
            model.compute_modal_coefficients(torch.tensor(0))
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            model.compute_modal_coefficients(torch.tensor([0.0]))
        with self.assertRaisesRegex(ValueError, "must lie"):
            model.compute_modal_coefficients(torch.tensor([1]))
        for dtype in (torch.uint8, torch.int16):
            with self.subTest(dtype=dtype):
                real, imaginary = model.compute_modal_coefficients(
                    torch.tensor([0], dtype=dtype)
                )
                self.assertEqual(real.shape, (1, 1))
                self.assertEqual(imaginary.shape, (1, 1))

    def test_state_round_trip_preserves_harmonic_time_buffer(self) -> None:
        model = _make_modal_model(
            torch.tensor([[[0.5, -0.25]], [[0.2, 0.4]]]),
            torch.ones(1, 1, 3, dtype=torch.complex64),
            torch.tensor([1.5]),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([0.0, 0.2]),
        )

        restored = SceneModel.init_from_state_dict(model.state_dict())

        torch.testing.assert_close(
            restored.modal_frame_times_sec, model.modal_frame_times_sec
        )
        torch.testing.assert_close(
            restored.compute_modal_offsets(torch.tensor([1, 0])),
            model.compute_modal_offsets(torch.tensor([1, 0])),
        )

    def test_legacy_per_frame_checkpoint_is_rejected(self) -> None:
        model = _make_modal_model(
            torch.zeros(1, 1, 2),
            torch.ones(1, 1, 3, dtype=torch.complex64),
            torch.tensor([1.0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0.0]),
        )
        legacy = dict(model.state_dict())
        del legacy["modal_frame_times_sec"]
        legacy["modal_smooth_triplets"] = torch.empty(0, 3, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "Legacy per-frame"):
            SceneModel.init_from_state_dict(legacy)


class ModalFrameMapTests(unittest.TestCase):
    def _load(self, payload: dict, dataset: SimpleNamespace):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "modal_frame_map.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_modal_frame_map(
                str(path), dataset, dataset.num_frames, torch.device("cpu")
            )

    def test_loads_time_from_full_map_for_reordered_smoke_subset(self) -> None:
        dataset = SimpleNamespace(
            frame_names=["view2_1", "view1_0", "view2_0", "view1_1"],
            time_ids=torch.tensor([2, 0, 3, 1]),
            num_frames=4,
        )

        frame_map = self._load(_valid_frame_map_payload(), dataset)

        self.assertEqual(frame_map.view_ids, ["view1", "view2"])
        torch.testing.assert_close(
            frame_map.frame_view_indices, torch.tensor([0, 0, 1, 1])
        )
        torch.testing.assert_close(
            frame_map.frame_local_indices, torch.tensor([0, 1, 1, 0])
        )
        torch.testing.assert_close(
            frame_map.frame_times_sec, torch.tensor([0.0, 0.1, 0.2, 0.0])
        )

    def test_declared_views_may_be_inactive_for_single_view_debug(self) -> None:
        dataset = SimpleNamespace(frame_names=["view2_2"], num_frames=1)

        frame_map = self._load(_valid_frame_map_payload(), dataset)

        torch.testing.assert_close(frame_map.frame_view_indices, torch.tensor([1]))
        torch.testing.assert_close(frame_map.frame_times_sec, torch.tensor([0.4]))

    def test_rejects_invalid_contract_fields(self) -> None:
        dataset = SimpleNamespace(frame_names=["view1_0"], num_frames=1)
        cases = []

        wrong_version = _valid_frame_map_payload()
        wrong_version["version"] = 2
        cases.append((wrong_version, "version 1"))

        duplicate_views = _valid_frame_map_payload()
        duplicate_views["views"] = ["view1", "view1"]
        cases.append((duplicate_views, "unique"))

        wrong_fps = _valid_frame_map_payload()
        wrong_fps["view_fps_hz"]["view1"] = 0.0
        cases.append((wrong_fps, "finite and positive"))

        wrong_time = _valid_frame_map_payload()
        wrong_time["frames"][1]["time_sec"] = 0.11
        cases.append((wrong_time, "does not match"))

        missing_local_index = _valid_frame_map_payload()
        missing_local_index["frames"] = [
            record
            for record in missing_local_index["frames"]
            if record["frame_name"] != "view1_1"
        ]
        cases.append((missing_local_index, "contiguous"))

        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._load(copy.deepcopy(payload), dataset)


class HarmonicRenderingConfigTests(unittest.TestCase):
    def test_view_configs_are_reordered_to_frame_map_view_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame_map = Path(tmp) / "modal_frame_map.json"
            frame_map.write_text(
                json.dumps({"version": 1, "views": ["view2", "view1"]}),
                encoding="utf-8",
            )
            with mock.patch.object(
                run_rendering,
                "load_view_config",
                side_effect=lambda path: SimpleNamespace(
                    view_id=Path(path).stem
                ),
            ):
                ordered = run_rendering._ordered_vggt_view_configs(
                    ("view1", "view2"),
                    str(frame_map),
                )

        self.assertEqual(ordered, ("view2", "view1"))

    def test_view_config_ids_must_match_declared_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame_map = Path(tmp) / "modal_frame_map.json"
            frame_map.write_text(
                json.dumps({"version": 1, "views": ["view1", "view2"]}),
                encoding="utf-8",
            )
            with mock.patch.object(
                run_rendering,
                "load_view_config",
                side_effect=lambda path: SimpleNamespace(
                    view_id=Path(path).stem
                ),
            ):
                with self.assertRaisesRegex(ValueError, "exactly match"):
                    run_rendering._ordered_vggt_view_configs(
                        ("view1",),
                        str(frame_map),
                    )


if __name__ == "__main__":
    unittest.main()
