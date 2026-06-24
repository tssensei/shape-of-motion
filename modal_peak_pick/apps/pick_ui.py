from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

from modal_peak_pick.core.pipeline import run_modal_analysis_from_video
from modal_peak_pick.core.spectrum import snap_to_local_peak


class PeakPickingUI:
    """
    Interactive spectrum picker.

    Click on the spectrum to select a bin, press "a" to add it, "s" to save,
    "d" to drop the last selected peak, and "q" to quit.
    """

    def __init__(
        self,
        frame_ref: np.ndarray,
        freqs_hz: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        power_spectrum: np.ndarray,
        mask: np.ndarray | None = None,
        out_json: str = "outputs/selected_peaks.json",
        snap_window_hz: float = 1.0,
    ):
        self.frame_ref = frame_ref
        self.freqs_hz = freqs_hz
        self.U = U
        self.V = V
        self.power_spectrum = power_spectrum
        self.mask = mask
        self.out_json = out_json
        self.snap_window_hz = float(snap_window_hz)

        self.selected: list[float] = []
        self.current_f: float | None = None
        self.current_idx: int | None = None
        self.point_yx: tuple[int, int] | None = None
        self.point_source = "auto"

        self.fig = plt.figure(figsize=(15, 8))
        gs = self.fig.add_gridspec(4, 2, width_ratios=[1.9, 1.0], height_ratios=[0.85, 1.0, 1.0, 0.55])
        self.ax_spec = self.fig.add_subplot(gs[:, 0])
        self.ax_ref = self.fig.add_subplot(gs[0, 1])
        self.ax_u = self.fig.add_subplot(gs[1, 1])
        self.ax_v = self.fig.add_subplot(gs[2, 1])
        self.ax_info = self.fig.add_subplot(gs[3, 1])

        self._draw_static()
        self._connect()

    @staticmethod
    def _complex_hsv_image_shared_scale(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
        phase = np.angle(z)
        mag = np.abs(z).astype(np.float32)
        hue = (phase + np.pi) / (2.0 * np.pi)
        sat = np.ones_like(hue, dtype=np.float32)
        val = np.clip((mag - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0).astype(np.float32)
        return hsv_to_rgb(np.stack([hue.astype(np.float32), sat, val], axis=-1))

    def _draw_static(self) -> None:
        self.ax_spec.set_title("Global power spectrum")
        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel("Mean image-plane amplitude")
        self.spec_line, = self.ax_spec.plot(self.freqs_hz, self.power_spectrum, lw=1)
        self.sel_vline = self.ax_spec.axvline(0.0, color="k", lw=1, alpha=0.4)
        self.sel_scatter = self.ax_spec.scatter([], [], s=30)

        self.ax_ref.set_title("Reference frame")
        self.ax_u.set_title("X component U")
        self.ax_v.set_title("Y component V")
        zero_rgb = np.zeros((*self.frame_ref.shape, 3), dtype=np.float32)
        self.ref_im = self.ax_ref.imshow(self.frame_ref, cmap="gray", vmin=0.0, vmax=1.0)
        self.u_im = self.ax_u.imshow(zero_rgb)
        self.v_im = self.ax_v.imshow(zero_rgb)
        self.ref_pt = self.ax_ref.scatter([], [], s=40, c="yellow", marker="x")
        self.u_pt = self.ax_u.scatter([], [], s=40, c="white", marker="x")
        self.v_pt = self.ax_v.scatter([], [], s=40, c="white", marker="x")

        for ax in (self.ax_ref, self.ax_u, self.ax_v):
            ax.set_xticks([])
            ax.set_yticks([])

        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(
            0.01,
            0.95,
            "Click spectrum to select a frequency.",
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )
        self.fig.suptitle(
            "Keys: a=add, d=drop last, s=save, q=quit | "
            "wheel on spectrum=zoom | click images=set readout point"
        )

    def _connect(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax_spec:
            return
        x0, x1 = self.ax_spec.get_xlim()
        full_lo = float(self.freqs_hz[0])
        full_hi = float(self.freqs_hz[-1])
        center = float(event.xdata) if event.xdata is not None else 0.5 * (x0 + x1)
        center = float(np.clip(center, full_lo, full_hi))
        width = max(1e-6, x1 - x0)
        scale = 0.85 if event.button == "up" else 1.18
        new_width = max((full_hi - full_lo) / 400.0, min(full_hi - full_lo, width * scale))
        left = max(full_lo, center - 0.5 * new_width)
        right = min(full_hi, center + 0.5 * new_width)
        if right <= left:
            return
        self.ax_spec.set_xlim(left, right)
        self.fig.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes == self.ax_spec and event.xdata is not None:
            f_selected = snap_to_local_peak(
                self.freqs_hz,
                self.power_spectrum,
                float(event.xdata),
                window_hz=self._effective_snap_window_hz(),
            )
            idx = int(np.argmin(np.abs(self.freqs_hz - f_selected)))
            self.set_current_frequency(float(self.freqs_hz[idx]), keep_point=False)
            return

        if event.inaxes in (self.ax_ref, self.ax_u, self.ax_v) and event.xdata is not None and event.ydata is not None:
            if self.current_idx is None:
                return
            x = int(np.clip(round(event.xdata), 0, self.frame_ref.shape[1] - 1))
            y = int(np.clip(round(event.ydata), 0, self.frame_ref.shape[0] - 1))
            self.point_yx = (y, x)
            self.point_source = "manual"
            self.set_current_frequency(float(self.freqs_hz[self.current_idx]), keep_point=True)

    def _effective_snap_window_hz(self) -> float:
        full_span = float(self.freqs_hz[-1] - self.freqs_hz[0])
        if full_span <= 0:
            return self.snap_window_hz
        x0, x1 = self.ax_spec.get_xlim()
        visible_span = max(0.0, min(float(self.freqs_hz[-1]), float(x1)) - max(float(self.freqs_hz[0]), float(x0)))
        bin_hz = full_span / max(1, len(self.freqs_hz) - 1)
        return float(max(1.5 * bin_hz, self.snap_window_hz * np.clip(visible_span / full_span, 0.0, 1.0)))

    def _pick_representative_point(self, U_slice: np.ndarray, V_slice: np.ndarray) -> tuple[int, int]:
        score = np.abs(U_slice) + np.abs(V_slice)
        if self.mask is not None and np.any(self.mask):
            score = np.where(self.mask, score, -np.inf)
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        return int(y), int(x)

    def _update_info_text(self, U_slice: np.ndarray, V_slice: np.ndarray) -> None:
        if self.point_yx is None or self.current_idx is None or self.current_f is None:
            return
        y, x = self.point_yx
        u0 = U_slice[y, x]
        v0 = V_slice[y, x]
        inside = True if self.mask is None else bool(self.mask[y, x])
        text = "\n".join(
            [
                f"selected = {self.current_f:.4f} Hz (bin {self.current_idx}), P = {self.power_spectrum[self.current_idx]:.6g}",
                f"point = (x={x}, y={y}), source={self.point_source}, in_mask={inside}",
                f"U = {np.real(u0):.6g} + {np.imag(u0):.6g}j, abs={np.abs(u0):.6g}, angle={np.angle(u0):.4f}",
                f"V = {np.real(v0):.6g} + {np.imag(v0):.6g}j, abs={np.abs(v0):.6g}, angle={np.angle(v0):.4f}",
            ]
        )
        self.info_text.set_text(text)

    def set_current_frequency(self, f_hz: float, keep_point: bool = False) -> None:
        idx = int(np.argmin(np.abs(self.freqs_hz - float(f_hz))))
        self.current_idx = idx
        self.current_f = float(self.freqs_hz[idx])
        U_slice = self.U[idx]
        V_slice = self.V[idx]

        mag_u = np.abs(U_slice).astype(np.float32)
        mag_v = np.abs(V_slice).astype(np.float32)
        if self.mask is not None and np.any(self.mask):
            mag_all = np.concatenate([mag_u[self.mask], mag_v[self.mask]])
        else:
            mag_all = np.concatenate([mag_u.ravel(), mag_v.ravel()])
        lo = float(np.percentile(mag_all, 1.0)) if mag_all.size else 0.0
        hi = float(np.percentile(mag_all, 99.0)) if mag_all.size else 1.0
        if hi <= lo:
            hi = lo + 1e-6

        u_rgb = self._complex_hsv_image_shared_scale(U_slice, lo, hi)
        v_rgb = self._complex_hsv_image_shared_scale(V_slice, lo, hi)
        if self.mask is not None:
            mask3 = self.mask[..., None].astype(np.float32)
            u_rgb *= mask3
            v_rgb *= mask3
        self.u_im.set_data(u_rgb)
        self.v_im.set_data(v_rgb)
        self.ax_u.set_title(f"X component U (lo={lo:.3g}, hi={hi:.3g})")
        self.ax_v.set_title(f"Y component V (lo={lo:.3g}, hi={hi:.3g})")

        if not keep_point or self.point_yx is None:
            self.point_yx = self._pick_representative_point(U_slice, V_slice)
            self.point_source = "auto"
        y, x = self.point_yx
        offsets = np.array([[x, y]], dtype=np.float32)
        self.ref_pt.set_offsets(offsets)
        self.u_pt.set_offsets(offsets)
        self.v_pt.set_offsets(offsets)

        self._update_info_text(U_slice, V_slice)
        self.sel_vline.set_xdata([self.current_f, self.current_f])
        self.ax_spec.set_title(f"Global power spectrum - current: {self.current_f:.4f} Hz")
        self._update_selected_scatter()
        self.fig.canvas.draw_idle()

    def _update_selected_scatter(self) -> None:
        xs = np.array(self.selected, dtype=np.float32)
        if xs.size == 0:
            self.sel_scatter.set_offsets(np.zeros((0, 2)))
            return
        ys = np.interp(xs, self.freqs_hz, self.power_spectrum)
        self.sel_scatter.set_offsets(np.stack([xs, ys], axis=1))

    def _on_key(self, event) -> None:
        if event.key == "q":
            plt.close(self.fig)
            return
        if event.key == "a" and self.current_f is not None:
            if all(abs(self.current_f - f) > 1e-3 for f in self.selected):
                self.selected.append(float(self.current_f))
                self.selected.sort()
                self._update_selected_scatter()
                self.fig.canvas.draw_idle()
            return
        if event.key == "d":
            if self.selected:
                self.selected.pop()
                self._update_selected_scatter()
                self.fig.canvas.draw_idle()
            return
        if event.key == "s":
            out_path = Path(self.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump({"selected_peaks_hz": self.selected}, f, indent=2)
            print(f"Saved peaks -> {out_path}: {self.selected}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--mask", default=None, help="Optional binary ROI mask path (.npy or image).")
    parser.add_argument("--t0", type=float, default=0.0, help="Clip start time in seconds.")
    parser.add_argument("--t1", type=float, default=None, help="Clip end time in seconds.")
    parser.add_argument("--resize", type=int, default=None, help="Resize max(H,W) before analysis.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum decoded frames.")
    parser.add_argument("--flow-method", choices=["farneback", "tvl1"], default="farneback")
    parser.add_argument("--no-smooth", action="store_true", help="Disable contrast-weighted flow smoothing.")
    parser.add_argument("--sigma-b", type=float, default=3.0, help="Spatial smoothing sigma.")
    parser.add_argument("--sigma-c", type=float, default=0.0, help="Reference pre-blur sigma.")
    parser.add_argument("--snap-window-hz", type=float, default=1.0, help="Peak snapping window.")
    parser.add_argument("--out-json", default="outputs/selected_peaks.json", help="Selected peaks JSON path.")
    parser.add_argument("--analysis-mask-dilate-iters", type=int, default=0, help="Dilate the analysis mask with a 3x3 kernel before flow smoothing and spectrum computation.")


def run(args: argparse.Namespace) -> None:
    result = run_modal_analysis_from_video(
        video_path=args.video,
        t0=args.t0,
        t1=args.t1,
        resize=args.resize,
        max_frames=args.max_frames,
        flow_method=args.flow_method,
        no_smooth=args.no_smooth,
        sigma_b=args.sigma_b,
        sigma_c=args.sigma_c,
        mask_path=args.mask,
        analysis_mask_dilate_iters=getattr(args, "analysis_mask_dilate_iters", 0),
    )
    ui = PeakPickingUI(
        frame_ref=result.frame_ref,
        freqs_hz=result.freqs_hz,
        U=result.U,
        V=result.V,
        power_spectrum=result.power_spectrum,
        mask=result.mask,
        out_json=args.out_json,
        snap_window_hz=args.snap_window_hz,
    )
    if len(result.freqs_hz) > 2:
        strongest = 1 + int(np.argmax(result.power_spectrum[1:]))
        ui.set_current_frequency(float(result.freqs_hz[strongest]))
    plt.show()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive 2D modal spectrum peak picking.")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
