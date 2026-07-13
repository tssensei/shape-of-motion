from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from flow3d.modal_utils import load_gaussian_modal_fields


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


if __name__ == "__main__":
    unittest.main()
