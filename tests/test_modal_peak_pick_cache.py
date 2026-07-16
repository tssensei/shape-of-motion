from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import importlib
import io
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest import mock

import numpy as np

from modal_peak_pick.apps import analyze_cache, export_modes, inspect_modes, pick_ui
from modal_peak_pick.core import cache as cache_module
from modal_peak_pick.core.cache import ModalAnalysisCache, load_analysis_cache, write_analysis_cache
from modal_peak_pick.core.spectrum import dft_at_frequencies


def _analysis_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    flow_u = rng.normal(size=(6, 3, 4)).astype(np.float32)
    flow_v = rng.normal(size=(6, 3, 4)).astype(np.float32)
    spectrum_u = np.fft.rfft(flow_u, axis=0).astype(np.complex64)
    spectrum_v = np.fft.rfft(flow_v, axis=0).astype(np.complex64)
    return {
        "flow_u": flow_u,
        "flow_v": flow_v,
        "spectrum_u": spectrum_u,
        "spectrum_v": spectrum_v,
        "freqs_hz": np.fft.rfftfreq(flow_u.shape[0], d=0.1).astype(np.float32),
        "power_spectrum": np.array([0.0, 5.0, 1.0, 0.25], dtype=np.float32),
        "reference_frame": rng.random((3, 4), dtype=np.float32),
    }


def _metadata() -> dict[str, Any]:
    return {
        "sources": {
            "video": {
                "path": "moved/source.mp4",
                "resolved_path": "/old/location/source.mp4",
                "fingerprint": {
                    "method": "stat",
                    "size_bytes": 123,
                    "mtime_ns": 456,
                },
            },
            "mask": {
                "path": "moved/mask.npy",
                "resolved_path": "/old/location/mask.npy",
                "fingerprint": {
                    "method": "stat",
                    "size_bytes": 12,
                    "mtime_ns": 34,
                },
            },
        },
        "video": {
            "fps": 10.0,
            "frame_range": {
                "t0_s": 0.5,
                "t1_s": 1.1,
                "max_frames": None,
                "decoded_frame_count": 6,
            },
            "resize_max_side": 720,
        },
        "analysis": {
            "flow_method": "farneback",
            "flow_parameters": {},
            "smoothing": {
                "disabled": False,
                "sigma_b": 3.0,
                "sigma_c": 0.0,
                "analysis_mask_dilate_iters": 1,
                "gradient_pyramid_weights": [0.5, 0.3, 0.2],
            },
            "fft": {
                "detrend": True,
                "window": "hann",
                "block_width": 128,
            },
            "spectrum": {
                "method": "mean_image_plane_amplitude",
            },
            "reference_frame_index": 3,
            "reference_time_s": 0.8,
        },
        "timings_seconds": {
            "decode": 1.0,
            "flow": 2.0,
            "smooth": 3.0,
            "fft": 4.0,
            "spectrum": 5.0,
            "analysis_total": 15.0,
        },
    }


def _write_cache(path: Path, with_mask: bool = True) -> ModalAnalysisCache:
    arrays = _analysis_arrays()
    mask: np.ndarray | None = np.array(
        [
            [True, True, True, False],
            [True, True, True, False],
            [True, True, True, False],
        ],
        dtype=bool,
    )
    metadata = _metadata()
    if not with_mask:
        metadata["sources"]["mask"] = None
        mask = None
    return write_analysis_cache(path, mask=mask, metadata=metadata, **arrays)


class ModalAnalysisCacheTests(unittest.TestCase):
    def test_round_trip_uses_read_only_memmaps_without_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache"
            _write_cache(cache_path)
            loaded = load_analysis_cache(cache_path)

            self.assertEqual(
                {path.name for path in cache_path.iterdir()},
                {
                    "metadata.json",
                    "flow_u.npy",
                    "flow_v.npy",
                    "spectrum_u.npy",
                    "spectrum_v.npy",
                    "freqs_hz.npy",
                    "power_spectrum.npy",
                    "reference_frame.npy",
                    "mask.npy",
                },
            )
            self.assertEqual(loaded.metadata["cache_version"], 1)
            self.assertEqual(loaded.metadata["cache_format"], "modal_peak_pick_analysis")
            for array in (
                loaded.flow_u,
                loaded.flow_v,
                loaded.spectrum_u,
                loaded.spectrum_v,
                loaded.freqs_hz,
                loaded.power_spectrum,
                loaded.reference_frame,
                loaded.mask_array,
            ):
                self.assertIsInstance(array, np.memmap)
                self.assertFalse(array.flags.writeable)
            loaded_mask = loaded.mask
            self.assertIsNotNone(loaded_mask)
            assert loaded_mask is not None
            self.assertEqual(loaded_mask.dtype, np.bool_)
            self.assertFalse(loaded_mask.flags.writeable)
            self.assertEqual(loaded.metadata["sources"]["video"]["path"], "moved/source.mp4")

    def test_round_trip_without_mask_keeps_required_zero_mask_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loaded = _write_cache(Path(tmp) / "cache", with_mask=False)

            self.assertFalse(loaded.metadata["has_mask"])
            self.assertIsNone(loaded.mask)
            np.testing.assert_array_equal(loaded.mask_array, np.zeros((3, 4), dtype=np.uint8))

    def test_missing_file_wrong_dtype_and_wrong_shape_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            missing = root / "missing"
            _write_cache(missing)
            (missing / "spectrum_v.npy").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "spectrum_v.npy"):
                load_analysis_cache(missing)

            wrong_dtype = root / "wrong_dtype"
            _write_cache(wrong_dtype)
            np.save(
                wrong_dtype / "flow_u.npy",
                np.zeros((6, 3, 4), dtype=np.float64),
                allow_pickle=False,
            )
            with self.assertRaisesRegex(ValueError, "dtype"):
                load_analysis_cache(wrong_dtype)

            wrong_shape = root / "wrong_shape"
            _write_cache(wrong_shape)
            np.save(
                wrong_shape / "reference_frame.npy",
                np.zeros((2, 4), dtype=np.float32),
                allow_pickle=False,
            )
            with self.assertRaisesRegex(ValueError, "shape"):
                load_analysis_cache(wrong_shape)

    def test_unsupported_version_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache"
            _write_cache(cache_path)
            metadata_path = cache_path / "metadata.json"
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            metadata["cache_version"] = 2
            with metadata_path.open("w", encoding="utf-8") as file:
                json.dump(metadata, file)

            with self.assertRaisesRegex(ValueError, "Unsupported cache version"):
                load_analysis_cache(cache_path)

    def test_existing_target_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache"
            cache_path.mkdir()
            arrays = _analysis_arrays()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_analysis_cache(
                    cache_path,
                    mask=None,
                    metadata=_metadata(),
                    **arrays,
                )

    def test_failed_write_removes_temporary_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "cache"
            arrays = _analysis_arrays()
            metadata = _metadata()
            metadata["sources"]["mask"] = None
            real_save = np.save

            def failing_save(path, array, **kwargs):
                if Path(path).name == "spectrum_u.npy":
                    raise OSError("injected write failure")
                return real_save(path, array, **kwargs)

            with mock.patch.object(cache_module.np, "save", side_effect=failing_save):
                with self.assertRaisesRegex(OSError, "injected write failure"):
                    write_analysis_cache(
                        cache_path,
                        mask=None,
                        metadata=metadata,
                        **arrays,
                    )

            self.assertFalse(cache_path.exists())
            self.assertEqual(list(root.glob(".cache.*.tmp")), [])


class ModalAnalysisStageTests(unittest.TestCase):
    def test_analyze_existing_target_fails_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache"
            cache_path.mkdir()
            args = argparse.Namespace(cache_dir=str(cache_path))
            with mock.patch.object(analyze_cache, "load_video_clip") as decode:
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    analyze_cache.run(args)
            decode.assert_not_called()

    def test_analyze_calls_each_expensive_stage_once_and_records_provenance(self) -> None:
        arrays = _analysis_arrays()
        frames = np.stack(
            [arrays["reference_frame"] + np.float32(i * 0.01) for i in range(6)],
            axis=0,
        ).astype(np.float32)
        mask = np.ones((3, 4), dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "source.mp4"
            mask_path = root / "mask.npy"
            video_path.write_bytes(b"video")
            mask_path.write_bytes(b"mask")
            cache_path = root / "cache"
            args = argparse.Namespace(
                video=str(video_path),
                cache_dir=str(cache_path),
                mask=str(mask_path),
                t0=0.5,
                t1=1.1,
                resize=720,
                max_frames=None,
                flow_method="farneback",
                no_smooth=False,
                sigma_b=3.0,
                sigma_c=0.0,
                analysis_mask_dilate_iters=1,
            )

            with (
                mock.patch.object(analyze_cache, "load_video_clip", return_value=(frames, 10.0)) as decode,
                mock.patch.object(analyze_cache, "load_mask", return_value=mask) as load_mask,
                mock.patch.object(
                    analyze_cache,
                    "compute_dense_flow_to_reference",
                    return_value=(arrays["flow_u"], arrays["flow_v"]),
                ) as flow,
                mock.patch.object(
                    analyze_cache,
                    "contrast_weighted_smooth",
                    return_value=(arrays["flow_u"], arrays["flow_v"]),
                ) as smooth,
                mock.patch.object(
                    analyze_cache,
                    "fft_over_time",
                    return_value=(arrays["freqs_hz"], arrays["spectrum_u"], arrays["spectrum_v"]),
                ) as fft,
                mock.patch.object(
                    analyze_cache,
                    "global_power_spectrum",
                    return_value=arrays["power_spectrum"],
                ) as spectrum,
            ):
                analyze_cache.run(args)

            decode.assert_called_once()
            load_mask.assert_called_once()
            flow.assert_called_once()
            smooth.assert_called_once()
            fft.assert_called_once()
            spectrum.assert_called_once()
            loaded = load_analysis_cache(cache_path)
            self.assertEqual(loaded.metadata["sources"]["video"]["fingerprint"]["size_bytes"], 5)
            self.assertEqual(loaded.metadata["video"]["frame_range"]["decoded_frame_count"], 6)
            self.assertEqual(loaded.metadata["analysis"]["reference_frame_index"], 3)
            self.assertEqual(
                set(("decode", "flow", "smooth", "fft", "spectrum", "cache_write", "total")),
                set(loaded.metadata["timings_seconds"]) - {"analysis_total"},
            )

    def test_inspect_and_pick_only_use_cached_spectra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "cache"
            _write_cache(cache_path)
            inspect_args = argparse.Namespace(
                cache_dir=str(cache_path),
                out_dir=str(root / "inspect"),
                num_peaks=1,
                min_freq_hz=0.1,
                max_freq_hz=None,
                peak_window_hz=0.1,
                preview_percentile=99.0,
            )
            pick_args = argparse.Namespace(
                cache_dir=str(cache_path),
                out_json=str(root / "peaks.json"),
                snap_window_hz=1.0,
            )

            with (
                mock.patch("modal_peak_pick.core.flow.compute_dense_flow_to_reference") as flow,
                mock.patch("modal_peak_pick.core.flow.contrast_weighted_smooth") as smooth,
                mock.patch("modal_peak_pick.core.spectrum.fft_over_time") as fft,
                mock.patch.object(inspect_modes, "_save_spectrum_plot"),
                mock.patch.object(inspect_modes, "_save_mode_preview") as preview,
            ):
                inspect_modes.run(inspect_args)
                flow.assert_not_called()
                smooth.assert_not_called()
                fft.assert_not_called()
                preview.assert_called_once()

            with (
                mock.patch("modal_peak_pick.core.flow.compute_dense_flow_to_reference") as flow,
                mock.patch("modal_peak_pick.core.flow.contrast_weighted_smooth") as smooth,
                mock.patch("modal_peak_pick.core.spectrum.fft_over_time") as fft,
                mock.patch.object(pick_ui, "PeakPickingUI") as ui_type,
                mock.patch.object(pick_ui.plt, "show") as show,
            ):
                pick_ui.run(pick_args)
                flow.assert_not_called()
                smooth.assert_not_called()
                fft.assert_not_called()
                ui_type.assert_called_once()
                self.assertIsInstance(ui_type.call_args.kwargs["U"], np.memmap)
                ui_type.return_value.set_current_frequency.assert_called_once()
                show.assert_called_once()

    def test_export_uses_cached_flow_and_one_exact_off_bin_dft(self) -> None:
        expected_keys = {
            "freqs_hz",
            "power_spectrum",
            "mode_u",
            "mode_v",
            "mask",
            "has_mask",
            "reference_frame",
            "fps",
            "t0",
            "t1",
            "resize",
            "source_video",
            "source_mask",
            "requested_freqs_hz",
            "selected_freqs_hz",
            "frequency_method",
            "flow_method",
            "no_smooth",
            "sigma_b",
            "sigma_c",
            "t_ref_s",
            "mode_amp_clamp_method",
            "mode_amp_local_window",
            "mode_amp_ratio",
            "mode_amp_global_percentile",
            "mode_amp_clamped_fraction",
            "mode_amp_scale_min",
            "mode_amp_original_p95",
            "mode_amp_original_p99",
            "mode_amp_original_max",
            "mode_amp_clamped_p95",
            "mode_amp_clamped_p99",
            "mode_amp_clamped_max",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "cache"
            cache = _write_cache(cache_path)
            output_path = root / "modal_analysis.npz"
            requested = [1.3]
            _, expected_u, expected_v = dft_at_frequencies(
                cache.flow_u,
                cache.flow_v,
                fps=cache.fps,
                freqs_hz=requested,
                detrend=True,
                window="hann",
            )
            args = argparse.Namespace(
                cache_dir=str(cache_path),
                out=str(output_path),
                freqs="1.3",
                peaks_json=None,
                mode_amp_clamp="none",
                mode_amp_local_window=31,
                mode_amp_ratio=5.0,
                mode_amp_global_percentile=99.7,
            )

            with (
                mock.patch("modal_peak_pick.core.video_io.load_video_clip") as decode,
                mock.patch("modal_peak_pick.core.flow.compute_dense_flow_to_reference") as flow,
                mock.patch("modal_peak_pick.core.flow.contrast_weighted_smooth") as smooth,
                mock.patch("modal_peak_pick.core.spectrum.fft_over_time") as fft,
                mock.patch.object(
                    export_modes,
                    "dft_at_frequencies",
                    wraps=dft_at_frequencies,
                ) as exact_dft,
            ):
                export_modes.run(args)
                decode.assert_not_called()
                flow.assert_not_called()
                smooth.assert_not_called()
                fft.assert_not_called()
                exact_dft.assert_called_once()

            with np.load(output_path, allow_pickle=False) as output:
                self.assertEqual(set(output.files), expected_keys)
                self.assertEqual(output["mode_u"].dtype, np.complex64)
                self.assertEqual(output["mode_v"].dtype, np.complex64)
                self.assertEqual(output["selected_freqs_hz"].dtype, np.float32)
                self.assertEqual(output["frequency_method"].item(), "exact_dft")
                np.testing.assert_allclose(output["mode_u"], expected_u, rtol=0.0, atol=0.0)
                np.testing.assert_allclose(output["mode_v"], expected_v, rtol=0.0, atol=0.0)


class ModalPeakPickCliTests(unittest.TestCase):
    def test_cache_commands_require_cache_and_do_not_register_video_analysis_options(self) -> None:
        cli = importlib.import_module("run_modal_peak_pick")
        parser = cli.build_arg_parser()
        analyze = parser.parse_args(
            [
                "analyze",
                "--video",
                "source.mp4",
                "--cache-dir",
                "cache",
            ]
        )
        self.assertEqual(analyze.command, "analyze")
        self.assertEqual(analyze.flow_method, "farneback")

        for command, trailing in (
            ("pick", []),
            ("inspect", ["--out-dir", "inspect"]),
            ("export", ["--freqs", "1.3"]),
        ):
            with self.subTest(command=command):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args([command, "--video", "source.mp4", *trailing])
                args = parser.parse_args([command, "--cache-dir", "cache", *trailing])
                self.assertNotIn("video", vars(args))
                self.assertNotIn("mask", vars(args))
                self.assertNotIn("flow_method", vars(args))
                self.assertNotIn("sigma_b", vars(args))

    def test_main_dispatches_analyze(self) -> None:
        cli = importlib.import_module("run_modal_peak_pick")
        with mock.patch.object(analyze_cache, "run") as runner:
            cli.main(
                [
                    "analyze",
                    "--video",
                    "source.mp4",
                    "--cache-dir",
                    "cache",
                ]
            )
        runner.assert_called_once()
        self.assertEqual(runner.call_args.args[0].command, "analyze")


if __name__ == "__main__":
    unittest.main()
