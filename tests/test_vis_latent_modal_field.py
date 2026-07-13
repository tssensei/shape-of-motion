from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preproc.vis_latent_modal_field import (
    load_latent_manifest,
    load_manifest_runtime_modes,
    reset_modal_runtime,
    runtime_mode_mask,
    selected_track_phase_component,
)


class VisLatentModalFieldTrackTests(unittest.TestCase):
    def _write_manifest(self, directory: Path, modes: list[dict[str, object]]) -> Path:
        path = directory / "modal_modes_manifest.json"
        path.write_text(json.dumps({"version": 1, "modes": modes}), encoding="utf-8")
        return path

    def test_manifest_track_labels_are_loaded_in_mode_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
            np.savez(
                directory / "linear.npz",
                points_world=points,
                phi=np.ones((2, 3), dtype=np.complex64),
            )
            np.savez(
                directory / "ellipse.npz",
                points_world=points,
                phi=1j * np.ones((2, 3), dtype=np.complex64),
            )
            manifest = self._write_manifest(
                directory,
                [
                    {"latent_path": "linear.npz", "track_label": "Linear"},
                    {"latent_path": "ellipse.npz", "track_label": "Tilted ellipse"},
                ],
            )

            runtime = load_manifest_runtime_modes(manifest, max_points=0)

            self.assertEqual(runtime.track_labels, ("Linear", "Tilted ellipse"))

    def test_manifest_track_labels_are_all_or_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._write_manifest(
                Path(tmp),
                [
                    {"latent_path": "linear.npz", "track_label": "Linear"},
                    {"latent_path": "ellipse.npz"},
                ],
            )

            with self.assertRaisesRegex(ValueError, "all include track_label"):
                load_latent_manifest(manifest)

    def test_manifest_track_labels_must_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._write_manifest(
                Path(tmp),
                [
                    {"latent_path": "linear.npz", "track_label": "Track"},
                    {"latent_path": "ellipse.npz", "track_label": "Track"},
                ],
            )

            with self.assertRaisesRegex(ValueError, "must be unique"):
                load_latent_manifest(manifest)

    def test_runtime_mode_mask_preserves_additive_modes_without_tracks(self) -> None:
        enabled = np.asarray([True, False, True], dtype=bool)

        actual = runtime_mode_mask(enabled, track_labels=None, selected_track=None)

        np.testing.assert_array_equal(actual, enabled)

    def test_runtime_mode_mask_selects_exactly_one_track(self) -> None:
        track_labels = ("Linear", "Tilted ellipse")
        enabled = np.asarray([False, False], dtype=bool)

        linear = runtime_mode_mask(enabled, track_labels, "Linear")
        ellipse = runtime_mode_mask(enabled, track_labels, "Tilted ellipse")

        np.testing.assert_array_equal(linear, np.asarray([True, False]))
        np.testing.assert_array_equal(ellipse, np.asarray([False, True]))

    def test_track_change_helpers_reset_state_and_select_phase_component(self) -> None:
        runtime = {
            "time": 1.5,
            "q": np.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64),
            "qdot": np.asarray([5.0 + 6.0j, 7.0 + 8.0j], dtype=np.complex64),
        }
        track_labels = ("Linear", "Tilted ellipse")
        phase_options = (
            "Linear projected u",
            "Linear projected v",
            "Tilted ellipse projected u",
            "Tilted ellipse projected v",
        )

        reset_modal_runtime(runtime)
        phase_component = selected_track_phase_component(
            track_labels,
            "Tilted ellipse",
            phase_options,
        )

        self.assertEqual(runtime["time"], 0.0)
        np.testing.assert_array_equal(runtime["q"], np.zeros(2, dtype=np.complex64))
        np.testing.assert_array_equal(runtime["qdot"], np.zeros(2, dtype=np.complex64))
        self.assertEqual(phase_component, "Tilted ellipse projected u")


if __name__ == "__main__":
    unittest.main()
