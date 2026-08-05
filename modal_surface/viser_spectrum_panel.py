from __future__ import annotations

from typing import Callable

import numpy as np
import viser
import viser.uplot

from modal_surface.spectrum_comparison import SpectrumComparisonController


def _spectrum_plot_data(
    frequencies_hz: np.ndarray,
    power: np.ndarray,
    selected_frequency_hz: float,
    marker_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    values = np.asarray(power, dtype=np.float64).reshape(-1)
    if frequencies.shape != values.shape or frequencies.size < 2:
        raise ValueError("Spectrum frequencies and power must have matching length >= 2")
    if not np.isfinite(frequencies).all() or not np.isfinite(values).all():
        raise ValueError("Spectrum frequencies and power must be finite")
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    values = values[order]
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("Spectrum frequencies must be unique")

    frequency = float(selected_frequency_hz)
    if not np.isfinite(frequency):
        raise ValueError("Selected spectrum frequency must be finite")
    span = float(frequencies[-1] - frequencies[0])
    spacing = float(np.min(np.diff(frequencies)))
    half_width = max(1.0e-9 * max(span, 1.0), 1.0e-3 * spacing)
    marker_x = np.asarray(
        [
            frequency - half_width,
            frequency,
            frequency + half_width,
        ],
        dtype=np.float64,
    )
    plot_x = np.unique(np.concatenate([frequencies, marker_x]))
    plot_power = np.interp(plot_x, frequencies, values)
    marker = np.full(plot_x.shape, np.nan, dtype=np.float64)
    marker_indices = [int(np.argmin(np.abs(plot_x - value))) for value in marker_x]
    marker[marker_indices[0]] = 0.0
    marker[marker_indices[1]] = float(marker_height)
    marker[marker_indices[2]] = 0.0
    return plot_x, plot_power, marker


class ModalSpectrumPanel:
    def __init__(
        self,
        server: viser.ViserServer,
        controller: SpectrumComparisonController,
        on_view_selected: Callable[[str], None],
        on_mode_selected: Callable[[int], None],
        on_component_selected: Callable[[str], None],
        on_normalization_selected: Callable[[str], None],
        on_solo_selected: Callable[[int], None],
        on_enable_all: Callable[[], None],
    ) -> None:
        self.server = server
        self.controller = controller
        self._on_view_selected = on_view_selected
        self._on_mode_selected = on_mode_selected
        self._on_component_selected = on_component_selected
        self._on_normalization_selected = on_normalization_selected
        self._on_solo_selected = on_solo_selected
        self._on_enable_all = on_enable_all
        self._updating = False

        num_modes = int(controller.manifest.frequencies_hz.size)
        if num_modes <= 0:
            raise ValueError("Modal spectrum controller contains no reconstructed modes")
        frequency_order = np.argsort(
            controller.manifest.frequencies_hz,
            kind="stable",
        )
        self._frequency_order = tuple(int(index) for index in frequency_order)
        display_indices = np.empty(num_modes, dtype=np.int64)
        display_indices[frequency_order] = np.arange(num_modes, dtype=np.int64)
        self._display_indices = tuple(int(index) for index in display_indices)

        self.view = server.gui.add_dropdown(
            "Spectrum view",
            options=controller.available_view_ids,
            initial_value=controller.view_id,
        )
        self.component = server.gui.add_dropdown(
            "Modal image component",
            options=("U", "V"),
            initial_value="U" if controller.component_index == 0 else "V",
        )
        self.amplitude_normalization = server.gui.add_dropdown(
            "Amplitude normalization",
            options=("per mode", "entire spectrum"),
            initial_value=controller.amplitude_normalization,
        )
        self.mode_index = server.gui.add_slider(
            "Selected mode",
            min=0,
            max=num_modes - 1,
            step=1,
            initial_value=self._display_indices[controller.reconstructed_index],
        )
        self.frequency = server.gui.add_number(
            "Selected frequency (Hz)",
            initial_value=float(
                controller.manifest.frequencies_hz[controller.reconstructed_index]
            ),
            disabled=True,
        )
        solo_selected = server.gui.add_button("Solo selected mode")
        enable_all = server.gui.add_button("Enable all modes")
        self.status = server.gui.add_markdown(controller.status)

        shared_power_max = self._shared_power_max()
        raw_data = _spectrum_plot_data(
            controller.cache.freqs_hz,
            controller.raw_power,
            controller.raw_frequency_hz,
            shared_power_max,
        )
        reconstructed_frequency = float(
            controller.manifest.frequencies_hz[controller.reconstructed_index]
        )
        reconstructed_data = _spectrum_plot_data(
            controller.manifest.frequencies_hz,
            controller.reconstructed_power,
            reconstructed_frequency,
            shared_power_max,
        )
        scales = self._plot_scales(shared_power_max)
        self.raw_plot = server.gui.add_uplot(
            data=raw_data,
            series=self._plot_series("Original power", "#4c9aff"),
            title=f"Original {controller.view_id} spectrum",
            scales=scales,
            legend=viser.uplot.Legend(show=True),
            height=260,
        )
        self.reconstructed_plot = server.gui.add_uplot(
            data=reconstructed_data,
            series=self._plot_series("Reconstructed power", "#ff9f43"),
            title=f"Reconstructed {controller.view_id} spectrum",
            scales=scales,
            legend=viser.uplot.Legend(show=True),
            height=260,
        )
        self.raw_modal_image = server.gui.add_image(
            controller.raw_modal_image,
            label="Original modal image",
            format="jpeg",
            jpeg_quality=90,
        )
        self.reconstructed_modal_image = server.gui.add_image(
            controller.reconstructed_modal_image,
            label="Reconstructed modal image",
            format="jpeg",
            jpeg_quality=90,
        )

        @self.view.on_update
        def _(_) -> None:
            if self._updating:
                return
            self._updating = True
            try:
                self.controller.select_view(str(self.view.value))
                self._refresh()
                self._on_view_selected(str(self.view.value))
            finally:
                self._updating = False

        @self.component.on_update
        def _(_) -> None:
            if self._updating:
                return
            self._on_component_selected(str(self.component.value))

        @self.amplitude_normalization.on_update
        def _(_) -> None:
            if self._updating:
                return
            self._on_normalization_selected(
                str(self.amplitude_normalization.value)
            )

        @self.mode_index.on_update
        def _(_) -> None:
            if self._updating:
                return
            self._on_mode_selected(
                self._frequency_order[int(self.mode_index.value)]
            )

        @solo_selected.on_click
        def _(_) -> None:
            self._on_solo_selected(
                self._frequency_order[int(self.mode_index.value)]
            )

        @enable_all.on_click
        def _(_) -> None:
            self._on_enable_all()

    @staticmethod
    def _plot_series(
        power_label: str,
        power_color: str,
    ) -> tuple[viser.uplot.Series, ...]:
        return (
            viser.uplot.Series(label="Frequency (Hz)"),
            viser.uplot.Series(label=power_label, stroke=power_color, width=2),
            viser.uplot.Series(label="Selected frequency", stroke="#ff3b30", width=1),
        )

    def _shared_power_max(self) -> float:
        lower, upper = self.controller.frequency_limits
        raw_visible = (
            (self.controller.cache.freqs_hz >= lower)
            & (self.controller.cache.freqs_hz <= upper)
        )
        reconstructed_visible = (
            (self.controller.manifest.frequencies_hz >= lower)
            & (self.controller.manifest.frequencies_hz <= upper)
        )
        maxima = [0.0]
        if np.any(raw_visible):
            maxima.append(float(np.max(self.controller.raw_power[raw_visible])))
        if np.any(reconstructed_visible):
            maxima.append(
                float(
                    np.max(
                        self.controller.reconstructed_power[reconstructed_visible]
                    )
                )
            )
        maximum = max(maxima)
        return 1.05 * maximum if maximum > 0.0 else 1.0

    def _plot_scales(self, power_max: float) -> dict[str, viser.uplot.Scale]:
        return {
            "x": viser.uplot.Scale(
                time=False,
                range=tuple(float(value) for value in self.controller.frequency_limits),
            ),
            "y": viser.uplot.Scale(range=(0.0, float(power_max))),
        }

    def _refresh(self) -> None:
        selected_index = int(self.controller.reconstructed_index)
        selected_frequency = float(
            self.controller.manifest.frequencies_hz[selected_index]
        )
        power_max = self._shared_power_max()
        self.mode_index.value = self._display_indices[selected_index]
        self.frequency.value = selected_frequency
        self.amplitude_normalization.value = (
            self.controller.amplitude_normalization
        )
        self.status.content = self.controller.status
        self.raw_plot.data = _spectrum_plot_data(
            self.controller.cache.freqs_hz,
            self.controller.raw_power,
            self.controller.raw_frequency_hz,
            power_max,
        )
        self.raw_plot.title = f"Original {self.controller.view_id} spectrum"
        self.raw_plot.scales = self._plot_scales(power_max)
        self.reconstructed_plot.data = _spectrum_plot_data(
            self.controller.manifest.frequencies_hz,
            self.controller.reconstructed_power,
            selected_frequency,
            power_max,
        )
        self.reconstructed_plot.title = (
            f"Reconstructed {self.controller.view_id} spectrum"
        )
        self.reconstructed_plot.scales = self._plot_scales(power_max)
        self.raw_modal_image.image = self.controller.raw_modal_image
        self.reconstructed_modal_image.image = (
            self.controller.reconstructed_modal_image
        )

    def set_mode_index(self, mode_index: int) -> None:
        index = int(mode_index)
        if not (0 <= index < len(self._display_indices)):
            raise ValueError(f"Unknown modal spectrum mode index: {index}")
        display_index = self._display_indices[index]
        if index == int(self.controller.reconstructed_index):
            if int(self.mode_index.value) != display_index:
                self._updating = True
                try:
                    self.mode_index.value = display_index
                finally:
                    self._updating = False
            return
        self._updating = True
        try:
            self.controller.select_reconstructed_index(index)
            self._refresh()
        finally:
            self._updating = False

    def set_component(self, component: str) -> None:
        value = str(component).upper()
        if value not in ("U", "V"):
            raise ValueError(f"Unknown modal spectrum component: {component!r}")
        target_index = 0 if value == "U" else 1
        if target_index == int(self.controller.component_index):
            if str(self.component.value) != value:
                self._updating = True
                try:
                    self.component.value = value
                finally:
                    self._updating = False
            return
        self._updating = True
        try:
            self.component.value = value
            self.controller.select_component(value)
            self._refresh()
        finally:
            self._updating = False

    def set_amplitude_normalization(self, normalization: str) -> None:
        value = str(normalization)
        if value not in ("per mode", "entire spectrum"):
            raise ValueError(
                f"Unknown modal spectrum amplitude normalization: {value!r}"
            )
        if value == self.controller.amplitude_normalization:
            if str(self.amplitude_normalization.value) != value:
                self._updating = True
                try:
                    self.amplitude_normalization.value = value
                finally:
                    self._updating = False
            return
        self._updating = True
        try:
            self.amplitude_normalization.value = value
            self.controller.select_amplitude_normalization(value)
            self._refresh()
        finally:
            self._updating = False
