from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import hsv_to_rgb
from matplotlib.figure import Figure
import numpy as np

from flow3d.modal_flow_coordinates import _load_manifest, _view_pixel_groups
from flow3d.modal_frequency_selection import _temporal_basis
from modal_peak_pick.core.cache import ModalAnalysisCache, load_analysis_cache


_PIXEL_CHUNK_SIZE = 2048


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Modal-analysis cache directory for the first manifest view.",
    )
    parser.add_argument(
        "--modal-manifest",
        required=True,
        help="Solved Gaussian modal manifest used for the reconstruction.",
    )
    parser.add_argument(
        "--preview-percentile",
        type=float,
        default=99.0,
        help="Shared raw/reconstructed magnitude percentile used for phase-HSV brightness.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Gradio server host.")
    parser.add_argument("--port", type=int, default=8894, help="Gradio server port.")


def _view_alphas(manifest_path: Path, manifest: Any, view_index: int) -> np.ndarray:
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    modes = payload.get("modes")
    if not isinstance(modes, list) or len(modes) != manifest.frequencies_hz.size:
        raise ValueError(f"{manifest_path} modes do not match the loaded manifest")

    view_id = manifest.topology.view_ids[view_index]
    alphas = np.empty((len(modes),), dtype=np.complex64)
    for slot, mode in enumerate(modes):
        if not isinstance(mode, dict):
            raise ValueError(f"{manifest_path} mode entries must be objects")
        if mode.get("mode_index") != int(manifest.mode_indices[slot]):
            raise ValueError(f"{manifest_path} mode order changed while loading view alphas")
        if float(mode.get("freq_hz", np.nan)) != float(manifest.frequencies_hz[slot]):
            raise ValueError(f"{manifest_path} frequency order changed while loading view alphas")
        entries = mode.get("alpha_by_view")
        if not isinstance(entries, list) or len(entries) != len(manifest.topology.view_ids):
            raise ValueError(f"{manifest_path} mode {slot} has invalid alpha_by_view")
        entry = entries[view_index]
        if not isinstance(entry, dict) or entry.get("view_id") != view_id:
            raise ValueError(
                f"{manifest_path} mode {slot} alpha order does not match view {view_id!r}"
            )
        if entry.get("identifiable") is not True:
            raise ValueError(
                f"{manifest_path} mode {slot} is not alpha-identifiable in view {view_id!r}"
            )
        alpha = complex(float(entry.get("real", np.nan)), float(entry.get("imag", np.nan)))
        if not np.isfinite(alpha.real) or not np.isfinite(alpha.imag):
            raise ValueError(f"{manifest_path} mode {slot} has a non-finite view alpha")
        alphas[slot] = alpha
    return alphas


def _candidate_power_spectrum(
    cache: ModalAnalysisCache,
    pixels: np.ndarray,
) -> np.ndarray:
    power_sum = np.zeros((cache.freqs_hz.size,), dtype=np.float64)
    for start in range(0, pixels.shape[0], _PIXEL_CHUNK_SIZE):
        current = pixels[start : start + _PIXEL_CHUNK_SIZE]
        x = current[:, 0]
        y = current[:, 1]
        spectrum_u = np.asarray(cache.spectrum_u[:, y, x])
        spectrum_v = np.asarray(cache.spectrum_v[:, y, x])
        amplitude = np.sqrt(np.abs(spectrum_u) ** 2 + np.abs(spectrum_v) ** 2)
        power_sum += np.sum(amplitude, axis=1, dtype=np.float64)
    result = power_sum / float(pixels.shape[0])
    if not np.isfinite(result).all():
        raise ValueError(f"Candidate power spectrum from {cache.path} is non-finite")
    return result.astype(np.float32)


def _exact_candidate_mode(
    cache: ModalAnalysisCache,
    pixels: np.ndarray,
    frequency_hz: float,
) -> np.ndarray:
    fft = cache.metadata["analysis"]["fft"]
    basis, temporal_window = _temporal_basis(
        cache.flow_u.shape[0],
        cache.fps,
        np.asarray([frequency_hz], dtype=np.float64),
        str(fft["window"]),
    )
    mode = np.empty((pixels.shape[0], 2), dtype=np.complex64)
    for start in range(0, pixels.shape[0], _PIXEL_CHUNK_SIZE):
        end = min(start + _PIXEL_CHUNK_SIZE, pixels.shape[0])
        current = pixels[start:end]
        x = current[:, 0]
        y = current[:, 1]
        flow_u = np.asarray(cache.flow_u[:, y, x], dtype=np.float32).copy()
        flow_v = np.asarray(cache.flow_v[:, y, x], dtype=np.float32).copy()
        if bool(fft["detrend"]):
            flow_u -= flow_u.mean(axis=0, keepdims=True, dtype=np.float32)
            flow_v -= flow_v.mean(axis=0, keepdims=True, dtype=np.float32)
        if temporal_window is not None:
            flow_u *= temporal_window[:, None]
            flow_v *= temporal_window[:, None]
        mode[start:end, 0] = (basis @ flow_u)[0]
        mode[start:end, 1] = (basis @ flow_v)[0]
    if not np.isfinite(mode.real).all() or not np.isfinite(mode.imag).all():
        raise ValueError(f"Exact candidate mode from {cache.path} is non-finite")
    return mode


def _project_reconstructed_modes(
    manifest: Any,
    sorted_rows: np.ndarray,
    starts: np.ndarray,
    sorted_weights: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    topology = manifest.topology
    point_indices = topology.obs_point_index[sorted_rows]
    jacobians = topology.obs_J[sorted_rows]
    modes = np.empty((manifest.frequencies_hz.size, starts.size, 2), dtype=np.complex64)
    for slot in range(manifest.frequencies_hz.size):
        point_phi = manifest.phi[slot, point_indices].astype(np.complex128)
        projected = np.einsum("oij,oj->oi", jacobians, point_phi, optimize=True)
        projected *= sorted_weights[:, None]
        modes[slot] = (
            alphas[slot] * np.add.reduceat(projected, starts, axis=0)
        ).astype(np.complex64)
    if not np.isfinite(modes.real).all() or not np.isfinite(modes.imag).all():
        raise ValueError("Projected reconstructed modal images contain non-finite values")
    return modes


def _phase_hsv(values: np.ndarray, magnitude_hi: float) -> np.ndarray:
    hue = (np.angle(values) + np.pi) / (2.0 * np.pi)
    saturation = np.ones_like(hue, dtype=np.float32)
    brightness = np.clip(
        np.abs(values).astype(np.float32) / max(float(magnitude_hi), 1.0e-12),
        0.0,
        1.0,
    )
    return hsv_to_rgb(
        np.stack([hue.astype(np.float32), saturation, brightness], axis=-1)
    ).astype(np.float32)


def _figure_rgb(figure: Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()


class SpectrumComparisonController:
    def __init__(
        self,
        cache_dir: str | Path,
        modal_manifest: str | Path,
        preview_percentile: float,
    ) -> None:
        if not (0.0 < preview_percentile <= 100.0):
            raise ValueError("preview_percentile must be in (0, 100]")
        self.preview_percentile = float(preview_percentile)
        self.cache = load_analysis_cache(cache_dir)
        self.manifest = _load_manifest(modal_manifest)
        self.view_index = 0
        self.view_id = self.manifest.topology.view_ids[self.view_index]

        expected_shape = (
            int(self.manifest.topology.view_image_height[self.view_index]),
            int(self.manifest.topology.view_image_width[self.view_index]),
        )
        if self.cache.flow_u.shape[1:] != expected_shape:
            raise ValueError(
                f"View1 cache shape {self.cache.flow_u.shape[1:]} does not match "
                f"manifest view {self.view_id!r} shape {expected_shape}"
            )
        if self.cache.mask is None:
            raise ValueError("View1 modal-analysis cache must contain a foreground mask")

        (
            self.sorted_rows,
            self.starts,
            self.pixels,
            self.sorted_weights,
            _,
        ) = _view_pixel_groups(self.manifest.topology, self.view_index, self.cache)
        self.raw_power = _candidate_power_spectrum(self.cache, self.pixels)
        self.alphas = _view_alphas(
            self.manifest.path,
            self.manifest,
            self.view_index,
        )
        self.reconstructed_modes = _project_reconstructed_modes(
            self.manifest,
            self.sorted_rows,
            self.starts,
            self.sorted_weights,
            self.alphas,
        )
        self.reconstructed_power = np.mean(
            np.sqrt(
                np.abs(self.reconstructed_modes[:, :, 0]) ** 2
                + np.abs(self.reconstructed_modes[:, :, 1]) ** 2
            ),
            axis=1,
        ).astype(np.float32)
        if not np.isfinite(self.reconstructed_power).all():
            raise ValueError("Reconstructed power spectrum is non-finite")
        if np.unique(self.manifest.frequencies_hz).size != self.manifest.frequencies_hz.size:
            raise ValueError("Reconstructed modal frequencies must be unique")
        raw_frequency_min = float(self.cache.freqs_hz[0])
        raw_frequency_max = float(self.cache.freqs_hz[-1])
        if np.any(self.manifest.frequencies_hz < raw_frequency_min) or np.any(
            self.manifest.frequencies_hz > raw_frequency_max
        ):
            raise ValueError("Reconstructed frequencies fall outside the raw spectrum range")

        selected_min = float(np.min(self.manifest.frequencies_hz))
        selected_max = float(np.max(self.manifest.frequencies_hz))
        selected_span = max(
            selected_max - selected_min,
            float(self.cache.freqs_hz[1] - self.cache.freqs_hz[0]),
        )
        margin = 0.05 * selected_span
        self.frequency_limits = (
            max(raw_frequency_min, selected_min - margin),
            min(raw_frequency_max, selected_max + margin),
        )
        self.component_index = 0
        self.raw_frequency_hz = float(self.manifest.frequencies_hz[0])
        self.reconstructed_index = 0
        self.raw_mode = np.empty((self.pixels.shape[0], 2), dtype=np.complex64)
        self.raw_spectrum_mapping: tuple[float, float, float, float, int] | None = None
        self.reconstructed_spectrum_mapping: tuple[float, float, float, float, int] | None = None
        self._set_frequencies(self.raw_frequency_hz, self.reconstructed_index)

    def _set_frequencies(self, raw_frequency_hz: float, reconstructed_index: int) -> None:
        raw_min = float(self.cache.freqs_hz[0])
        raw_max = float(self.cache.freqs_hz[-1])
        self.raw_frequency_hz = float(np.clip(raw_frequency_hz, raw_min, raw_max))
        self.reconstructed_index = int(reconstructed_index)
        if not (0 <= self.reconstructed_index < self.manifest.frequencies_hz.size):
            raise ValueError("Reconstructed frequency index is outside the manifest")
        self.raw_mode = _exact_candidate_mode(
            self.cache,
            self.pixels,
            self.raw_frequency_hz,
        )
        self._render_all()

    def _render_spectrum(
        self,
        frequencies_hz: np.ndarray,
        power: np.ndarray,
        selected_frequency_hz: float,
        selected_power: float,
        title: str,
        discrete: bool,
    ) -> tuple[np.ndarray, tuple[float, float, float, float, int]]:
        figure = Figure(figsize=(8.0, 3.3), dpi=100)
        axis = figure.add_subplot(111)
        if discrete:
            order = np.argsort(frequencies_hz)
            axis.plot(
                frequencies_hz[order],
                power[order],
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                color="tab:orange",
            )
        else:
            axis.plot(frequencies_hz, power, linewidth=1.2, color="tab:blue")
        axis.axvline(selected_frequency_hz, color="tab:red", linewidth=1.0)
        axis.scatter(
            [selected_frequency_hz],
            [selected_power],
            color="tab:red",
            s=28,
            zorder=3,
        )
        axis.set_xlim(*self.frequency_limits)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Mean image-plane amplitude")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        figure.subplots_adjust(left=0.12, right=0.98, bottom=0.19, top=0.86)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        bounds = axis.get_window_extent()
        mapping = (
            float(bounds.x0),
            float(bounds.x1),
            float(bounds.y0),
            float(bounds.y1),
            int(image.shape[0]),
        )
        return image, mapping

    def _render_modal_image(
        self,
        values: np.ndarray,
        magnitude_hi: float,
        title: str,
    ) -> np.ndarray:
        figure = Figure(figsize=(8.0, 4.5), dpi=100)
        axis = figure.add_subplot(111)
        axis.imshow(
            self.cache.reference_frame,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            alpha=0.35,
        )
        axis.scatter(
            self.pixels[:, 0],
            self.pixels[:, 1],
            c=_phase_hsv(values, magnitude_hi),
            s=7.0,
            marker="s",
            linewidths=0.0,
        )
        axis.set_xlim(-0.5, self.cache.reference_frame.shape[1] - 0.5)
        axis.set_ylim(self.cache.reference_frame.shape[0] - 0.5, -0.5)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.91)
        return _figure_rgb(figure)

    def _render_all(self) -> None:
        raw_power = float(
            np.mean(
                np.sqrt(
                    np.abs(self.raw_mode[:, 0]) ** 2
                    + np.abs(self.raw_mode[:, 1]) ** 2
                )
            )
        )
        reconstructed_frequency = float(
            self.manifest.frequencies_hz[self.reconstructed_index]
        )
        reconstructed_power = float(self.reconstructed_power[self.reconstructed_index])
        self.raw_spectrum_image, self.raw_spectrum_mapping = self._render_spectrum(
            self.cache.freqs_hz,
            self.raw_power,
            self.raw_frequency_hz,
            raw_power,
            f"Original view1 spectrum - {self.raw_frequency_hz:.5f} Hz",
            discrete=False,
        )
        (
            self.reconstructed_spectrum_image,
            self.reconstructed_spectrum_mapping,
        ) = self._render_spectrum(
            self.manifest.frequencies_hz,
            self.reconstructed_power,
            reconstructed_frequency,
            reconstructed_power,
            f"Reconstructed view1 spectrum - {reconstructed_frequency:.5f} Hz",
            discrete=True,
        )

        component_label = "U" if self.component_index == 0 else "V"
        raw_values = self.raw_mode[:, self.component_index]
        reconstructed_values = self.reconstructed_modes[
            self.reconstructed_index, :, self.component_index
        ]
        magnitudes = np.concatenate(
            [np.abs(raw_values), np.abs(reconstructed_values)]
        )
        magnitude_hi = float(np.percentile(magnitudes, self.preview_percentile))
        if not np.isfinite(magnitude_hi) or magnitude_hi <= 0.0:
            magnitude_hi = 1.0
        self.raw_modal_image = self._render_modal_image(
            raw_values,
            magnitude_hi,
            f"Original {component_label} phase - {self.raw_frequency_hz:.5f} Hz",
        )
        self.reconstructed_modal_image = self._render_modal_image(
            reconstructed_values,
            magnitude_hi,
            f"Reconstructed {component_label} phase - {reconstructed_frequency:.5f} Hz",
        )
        self.status = (
            f"**View:** `{self.view_id}` &nbsp; **candidate pixels:** {self.pixels.shape[0]}  \n"
            f"**Original:** {self.raw_frequency_hz:.6f} Hz, power={raw_power:.6g} &nbsp; "
            f"**Reconstructed:** {reconstructed_frequency:.6f} Hz, "
            f"power={reconstructed_power:.6g} &nbsp; **component:** {component_label}"
        )

    def outputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        return (
            self.raw_spectrum_image,
            self.reconstructed_spectrum_image,
            self.raw_modal_image,
            self.reconstructed_modal_image,
            self.status,
        )

    @staticmethod
    def _clicked_frequency(
        index: Any,
        mapping: tuple[float, float, float, float, int] | None,
        limits: tuple[float, float],
    ) -> float | None:
        if mapping is None or not isinstance(index, (tuple, list)) or len(index) < 2:
            return None
        x = float(index[0])
        y_from_top = float(index[1])
        x0, x1, y0, y1, image_height = mapping
        y_from_bottom = float(image_height) - y_from_top
        if x < x0 or x > x1 or y_from_bottom < y0 or y_from_bottom > y1:
            return None
        fraction = (x - x0) / max(x1 - x0, 1.0)
        return float(limits[0] + fraction * (limits[1] - limits[0]))

    def select_raw_spectrum(self, index: Any):
        frequency = self._clicked_frequency(
            index,
            self.raw_spectrum_mapping,
            self.frequency_limits,
        )
        if frequency is None:
            return self.outputs()
        reconstructed_index = int(
            np.argmin(np.abs(self.manifest.frequencies_hz - frequency))
        )
        self._set_frequencies(frequency, reconstructed_index)
        return self.outputs()

    def select_reconstructed_spectrum(self, index: Any):
        frequency = self._clicked_frequency(
            index,
            self.reconstructed_spectrum_mapping,
            self.frequency_limits,
        )
        if frequency is None:
            return self.outputs()
        reconstructed_index = int(
            np.argmin(np.abs(self.manifest.frequencies_hz - frequency))
        )
        snapped_frequency = float(self.manifest.frequencies_hz[reconstructed_index])
        self._set_frequencies(snapped_frequency, reconstructed_index)
        return self.outputs()

    def select_component(self, component: str):
        if component not in ("U", "V"):
            raise ValueError(f"Unknown modal image component {component!r}")
        self.component_index = 0 if component == "U" else 1
        self._render_all()
        return self.outputs()


def make_demo(controller: SpectrumComparisonController):
    import gradio as gr

    with gr.Blocks(title="Modal spectrum reconstruction comparison") as demo:
        gr.Markdown(
            "# View1 modal spectrum comparison\n"
            "Click either spectrum to select a frequency. Original clicks retain the "
            "exact clicked frequency; reconstructed clicks snap both rows to the nearest "
            "solved frequency."
        )
        component = gr.Dropdown(
            choices=("U", "V"),
            value="U",
            label="Modal image component",
            interactive=True,
        )
        status = gr.Markdown(controller.status)
        with gr.Row():
            with gr.Column():
                raw_spectrum = gr.Image(
                    value=controller.raw_spectrum_image,
                    label="Original power spectrum",
                    type="numpy",
                    interactive=True,
                )
                reconstructed_spectrum = gr.Image(
                    value=controller.reconstructed_spectrum_image,
                    label="Reconstructed power spectrum",
                    type="numpy",
                    interactive=True,
                )
            with gr.Column():
                raw_modal = gr.Image(
                    value=controller.raw_modal_image,
                    label="Original modal image",
                    type="numpy",
                    interactive=False,
                )
                reconstructed_modal = gr.Image(
                    value=controller.reconstructed_modal_image,
                    label="Reconstructed modal image",
                    type="numpy",
                    interactive=False,
                )

        outputs = [
            raw_spectrum,
            reconstructed_spectrum,
            raw_modal,
            reconstructed_modal,
            status,
        ]

        def select_raw(evt: gr.SelectData):
            return controller.select_raw_spectrum(evt.index)

        def select_reconstructed(evt: gr.SelectData):
            return controller.select_reconstructed_spectrum(evt.index)

        raw_spectrum.select(select_raw, outputs=outputs)
        reconstructed_spectrum.select(select_reconstructed, outputs=outputs)
        component.change(controller.select_component, inputs=[component], outputs=outputs)
    return demo


def run(args: argparse.Namespace) -> None:
    controller = SpectrumComparisonController(
        cache_dir=args.cache_dir,
        modal_manifest=args.modal_manifest,
        preview_percentile=args.preview_percentile,
    )
    print(
        f"Loaded view1 {controller.view_id!r}: "
        f"candidate_pixels={controller.pixels.shape[0]}, "
        f"reconstructed_modes={controller.manifest.frequencies_hz.size}"
    )
    demo = make_demo(controller)
    demo.launch(server_name=args.host, server_port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare original and projected reconstructed view1 modal spectra."
    )
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
