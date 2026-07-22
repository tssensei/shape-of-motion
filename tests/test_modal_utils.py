from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from flow3d.modal_utils import (
    MOTION_FILL_DISPLAY_ANCHOR,
    MOTION_FILL_DISPLAY_EXCLUDED,
    MOTION_FILL_DISPLAY_FILLED,
    MOTION_FILL_DISPLAY_PARTIAL,
    MOTION_FILL_DISPLAY_UNOBSERVED,
    classify_motion_fill_display_points,
    load_gaussian_modal_fields,
    load_modal_modes,
    motion_fill_display_colors,
    select_motion_fill_display_indices,
    stack_modal_motion_fill_display_classes,
)
from modal_surface.gaussian_motion_fill import MOTION_FILL_ROLE_NAMES


class GaussianModalFieldLoadingTests(unittest.TestCase):
    def _write_mode(
        self,
        root: Path,
        points: np.ndarray,
        phi: np.ndarray,
        gaussian_indices: np.ndarray,
        *,
        point_type: str = "foreground_gaussian_center",
        obs_count: np.ndarray | None = None,
        extra_fields: dict[str, np.ndarray] | None = None,
    ) -> Path:
        latent_path = root / "mode.npz"
        np.savez_compressed(
            latent_path,
            points_world=points.astype(np.float32),
            phi=phi,
            gaussian_indices=gaussian_indices,
            freq_hz=np.array(2.5, dtype=np.float32),
            mode_index=np.array(4, dtype=np.int32),
            obs_count_per_point=(
                np.array([3, 1], dtype=np.int32) if obs_count is None else obs_count
            ),
            point_type=np.array(point_type),
            source_checkpoint=np.array("source.ckpt"),
            **({} if extra_fields is None else extra_fields),
        )
        manifest_path = root / "modal_modes_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "point_type": "foreground_gaussian_center",
                    "source_checkpoint": "source.ckpt",
                    "modes": [
                        {
                            "mode_index": 4,
                            "freq_hz": 2.5,
                            "latent_path": latent_path.name,
                        }
                    ],
                },
                f,
            )
        return manifest_path

    def test_loads_direct_fields_without_interpolation(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        phi = np.array(
            [[1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j], [7.0 + 8.0j, 9.0 + 10.0j, 11.0 + 12.0j]],
            dtype=np.complex64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_mode(
                Path(tmp),
                points,
                phi,
                np.arange(points.shape[0], dtype=np.int32),
            )
            fields = load_gaussian_modal_fields(
                str(manifest_path),
                torch.from_numpy(points.copy()),
            )

        self.assertEqual(len(fields.modes), 1)
        self.assertEqual(fields.modes[0].mode_index, 4)
        torch.testing.assert_close(fields.phi_real[0], torch.from_numpy(phi.real))
        torch.testing.assert_close(fields.phi_imag[0], torch.from_numpy(phi.imag))
        torch.testing.assert_close(fields.freqs_hz, torch.tensor([2.5]))
        torch.testing.assert_close(
            fields.obs_count_per_point,
            torch.tensor([[3, 1]], dtype=torch.long),
        )

    def test_loads_final_phi_and_ignores_motion_fill_provenance_fields(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        phi = np.array(
            [[1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j], [7.0 + 8.0j, 9.0 + 10.0j, 11.0 + 12.0j]],
            dtype=np.complex64,
        )
        phi_observable = np.zeros_like(phi)
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_mode(
                Path(tmp),
                points,
                phi,
                np.arange(points.shape[0], dtype=np.int32),
                extra_fields={
                    "phi_observable": phi_observable,
                    "phi_nullspace_correction": phi - phi_observable,
                    "motion_fill_role": np.asarray([0, 2], dtype=np.int8),
                    "motion_fill_method": np.array("joint_knn_nullspace_lsmr"),
                },
            )
            fields = load_gaussian_modal_fields(
                str(manifest_path),
                torch.from_numpy(points.copy()),
            )

        torch.testing.assert_close(fields.phi_real[0], torch.from_numpy(phi.real))
        torch.testing.assert_close(fields.phi_imag[0], torch.from_numpy(phi.imag))

    def test_rejects_noncontiguous_gaussian_indices(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        phi = np.ones_like(points, dtype=np.complex64)
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_mode(
                Path(tmp),
                points,
                phi,
                np.array([1, 0], dtype=np.int32),
            )
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                load_gaussian_modal_fields(
                    str(manifest_path),
                    torch.from_numpy(points.copy()),
                )

    def test_rejects_position_mismatch(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        phi = np.ones_like(points, dtype=np.complex64)
        gaussian_means = points.copy()
        gaussian_means[0, 0] += 2e-5
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_mode(
                Path(tmp),
                points,
                phi,
                np.arange(points.shape[0], dtype=np.int32),
            )
            with self.assertRaisesRegex(ValueError, "above 1e-05"):
                load_gaussian_modal_fields(
                    str(manifest_path),
                    torch.from_numpy(gaussian_means),
                )

    def test_rejects_gaussian_count_mismatch(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        phi = np.ones_like(points, dtype=np.complex64)
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_mode(
                Path(tmp),
                points,
                phi,
                np.arange(points.shape[0], dtype=np.int32),
            )
            with self.assertRaisesRegex(ValueError, "does not match foreground"):
                load_gaussian_modal_fields(
                    str(manifest_path),
                    torch.from_numpy(points[:1].copy()),
                )

    def test_rejects_manifest_point_type_mismatch(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        phi = np.ones_like(points, dtype=np.complex64)
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_mode(
                Path(tmp),
                points,
                phi,
                np.arange(points.shape[0], dtype=np.int32),
            )
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["point_type"] = "carrier_point"
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaisesRegex(ValueError, "expected foreground_gaussian_center"):
                load_gaussian_modal_fields(
                    str(manifest_path),
                    torch.from_numpy(points.copy()),
                )

    def test_rejects_latent_point_type_mismatch(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        phi = np.ones_like(points, dtype=np.complex64)
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_mode(
                Path(tmp),
                points,
                phi,
                np.arange(points.shape[0], dtype=np.int32),
                point_type="carrier_point",
            )
            with self.assertRaisesRegex(ValueError, "expected foreground_gaussian_center"):
                load_gaussian_modal_fields(
                    str(manifest_path),
                    torch.from_numpy(points.copy()),
                )

    def test_rejects_observation_count_shape_mismatch(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        phi = np.ones_like(points, dtype=np.complex64)
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_mode(
                Path(tmp),
                points,
                phi,
                np.arange(points.shape[0], dtype=np.int32),
                obs_count=np.array([3], dtype=np.int32),
            )
            with self.assertRaisesRegex(ValueError, "obs_count_per_point shape"):
                load_gaussian_modal_fields(
                    str(manifest_path),
                    torch.from_numpy(points.copy()),
                )

    def test_rejects_nonfinite_gaussian_means(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        phi = np.ones_like(points, dtype=np.complex64)
        gaussian_means = points.copy()
        gaussian_means[0, 0] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_mode(
                Path(tmp),
                points,
                phi,
                np.arange(points.shape[0], dtype=np.int32),
            )
            with self.assertRaisesRegex(ValueError, "must be finite"):
                load_gaussian_modal_fields(
                    str(manifest_path),
                    torch.from_numpy(gaussian_means),
                )

    def test_rejects_real_phi(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_mode(
                Path(tmp),
                points,
                np.ones_like(points, dtype=np.float32),
                np.arange(points.shape[0], dtype=np.int32),
            )
            with self.assertRaisesRegex(ValueError, "complex-valued"):
                load_gaussian_modal_fields(
                    str(manifest_path),
                    torch.from_numpy(points.copy()),
                )


class MotionFillRoleDisplayTests(unittest.TestCase):
    def _write_viewer_manifest(
        self,
        root: Path,
        mode_fields: list[dict[str, np.ndarray]],
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        points = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32
        )
        phi = np.ones_like(points, dtype=np.complex64)
        modes = []
        for mode_index, extra_fields in enumerate(mode_fields):
            latent_path = root / f"mode_{mode_index}.npz"
            np.savez_compressed(
                latent_path,
                points_world=points,
                phi=phi,
                freq_hz=np.array(1.5 + mode_index, dtype=np.float32),
                **extra_fields,
            )
            modes.append(
                {
                    "mode_index": mode_index,
                    "freq_hz": 1.5 + mode_index,
                    "latent_path": latent_path.name,
                }
            )
        manifest_path = root / "modal_modes_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump({"version": 1, "modes": modes}, f)
        return manifest_path

    def test_classifies_all_display_roles_and_colors(self) -> None:
        roles = np.asarray([0, 0, 1, 1, 2, 2, 3], dtype=np.int8)
        completion = np.asarray([False, True, False, True, False, True, False])

        classes = classify_motion_fill_display_points(roles, completion)

        np.testing.assert_array_equal(
            classes,
            [
                MOTION_FILL_DISPLAY_ANCHOR,
                MOTION_FILL_DISPLAY_ANCHOR,
                MOTION_FILL_DISPLAY_PARTIAL,
                MOTION_FILL_DISPLAY_FILLED,
                MOTION_FILL_DISPLAY_UNOBSERVED,
                MOTION_FILL_DISPLAY_FILLED,
                MOTION_FILL_DISPLAY_EXCLUDED,
            ],
        )
        np.testing.assert_allclose(
            motion_fill_display_colors(classes),
            [
                [0.05, 0.55, 1.0],
                [0.05, 0.55, 1.0],
                [1.0, 0.55, 0.1],
                [0.1, 0.85, 0.3],
                [0.65, 0.35, 1.0],
                [0.1, 0.85, 0.3],
                [0.45, 0.45, 0.45],
            ],
        )

    def test_rejects_malformed_role_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "1-D"):
            classify_motion_fill_display_points(
                np.zeros((1, 2), dtype=np.int8), np.zeros((1, 2), dtype=bool)
            )
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            classify_motion_fill_display_points(
                np.zeros(2, dtype=np.float32), np.zeros(2, dtype=bool)
            )
        with self.assertRaisesRegex(ValueError, "boolean dtype"):
            classify_motion_fill_display_points(
                np.zeros(2, dtype=np.int8), np.zeros(2, dtype=np.int8)
            )
        with self.assertRaisesRegex(ValueError, "must lie"):
            classify_motion_fill_display_points(
                np.asarray([4], dtype=np.int8), np.asarray([False])
            )

    def test_rejects_completion_for_excluded_role(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-completable"):
            classify_motion_fill_display_points(
                np.asarray([3], dtype=np.int8), np.asarray([True])
            )

    def test_loads_role_metadata_and_preserves_legacy_manifests(self) -> None:
        role_fields = {
            "motion_fill_role": np.asarray([0, 2], dtype=np.int8),
            "completion_mask": np.asarray([False, True]),
            "motion_fill_role_names": np.asarray(MOTION_FILL_ROLE_NAMES),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_modes = load_modal_modes(
                str(self._write_viewer_manifest(root / "legacy", [{}]))
            )
            role_modes = load_modal_modes(
                str(self._write_viewer_manifest(root / "role", [role_fields]))
            )

        self.assertIsNone(legacy_modes[0].motion_fill_display_class)
        self.assertIsNone(stack_modal_motion_fill_display_classes(legacy_modes))
        np.testing.assert_array_equal(
            role_modes[0].motion_fill_display_class,
            [MOTION_FILL_DISPLAY_ANCHOR, MOTION_FILL_DISPLAY_FILLED],
        )
        np.testing.assert_array_equal(
            stack_modal_motion_fill_display_classes(role_modes),
            [[MOTION_FILL_DISPLAY_ANCHOR, MOTION_FILL_DISPLAY_FILLED]],
        )

    def test_selects_enabled_roles_with_one_shared_count_limit(self) -> None:
        classes = np.asarray(
            [
                MOTION_FILL_DISPLAY_ANCHOR,
                MOTION_FILL_DISPLAY_PARTIAL,
                MOTION_FILL_DISPLAY_FILLED,
                MOTION_FILL_DISPLAY_ANCHOR,
                MOTION_FILL_DISPLAY_FILLED,
                MOTION_FILL_DISPLAY_UNOBSERVED,
                MOTION_FILL_DISPLAY_EXCLUDED,
                MOTION_FILL_DISPLAY_ANCHOR,
            ],
            dtype=np.int8,
        )

        anchor_only = np.asarray([True, False, False, False, False])
        np.testing.assert_array_equal(
            select_motion_fill_display_indices(classes, anchor_only, 2),
            [0, 3],
        )

        anchor_and_filled = np.asarray([True, False, True, False, False])
        np.testing.assert_array_equal(
            select_motion_fill_display_indices(classes, anchor_and_filled, 4),
            [0, 2, 3, 4],
        )

        none_enabled = np.zeros(5, dtype=bool)
        np.testing.assert_array_equal(
            select_motion_fill_display_indices(classes, none_enabled, 4),
            [],
        )

        filled_only = np.asarray([False, False, True, False, False])
        np.testing.assert_array_equal(
            select_motion_fill_display_indices(classes, filled_only, 20),
            [2, 4],
        )

    def test_rejects_mixed_multi_mode_role_metadata(self) -> None:
        role_fields = {
            "motion_fill_role": np.asarray([0, 2], dtype=np.int8),
            "completion_mask": np.asarray([False, True]),
            "motion_fill_role_names": np.asarray(MOTION_FILL_ROLE_NAMES),
        }
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_viewer_manifest(
                Path(tmp), [role_fields, {}]
            )
            with self.assertRaisesRegex(ValueError, "either all include"):
                load_modal_modes(str(manifest_path))

    def test_rejects_incomplete_role_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_viewer_manifest(
                Path(tmp),
                [{"motion_fill_role": np.asarray([0, 2], dtype=np.int8)}],
            )
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_modal_modes(str(manifest_path))


if __name__ == "__main__":
    unittest.main()
