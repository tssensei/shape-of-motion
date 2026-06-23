from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import hsv_to_rgb
import numpy as np


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--modal-npz", required=True, help="modal_analysis.npz exported by run_modal_peak_pick.py export.")
    parser.add_argument("--out-dir", required=True, help="Output directory for synthesized videos and previews.")
    parser.add_argument("--mode-index", type=int, default=None, help="One-based mode index to synthesize. Omit to synthesize all modes.")
    parser.add_argument("--duration-s", type=float, default=4.0, help="Output video duration in seconds.")
    parser.add_argument("--fps-out", type=float, default=None, help="Output FPS. Default uses the input modal-analysis FPS.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier for the modal oscillation.")
    parser.add_argument("--gain", type=float, default=1.0, help="Extra displacement gain after optional FFT normalization.")
    parser.add_argument("--no-fft-normalize", action="store_true", help="Use raw rFFT mode coefficients without amplitude normalization.")
    parser.add_argument("--reference-video", default=None, help="Optional video path used to load a color reference frame.")
    parser.add_argument("--reference-time-s", type=float, default=None, help="Optional reference time for --reference-video.")
    parser.add_argument("--preview-percentile", type=float, default=99.0, help="Magnitude percentile used for HSV preview images.")


def _scalar_str(value: np.ndarray | str | bytes | object, default: str = "") -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0].item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _scalar_float(value: np.ndarray | float, default: float) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _scalar_int(value: np.ndarray | int, default: int) -> int:
    arr = np.asarray(value)
    if arr.size == 0:
        return int(default)
    return int(arr.reshape(-1)[0])


def _resize_bgr_max_side(frame_bgr: np.ndarray, resize: int) -> np.ndarray:
    if resize <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    side = max(h, w)
    if side == resize:
        return frame_bgr
    scale = float(resize) / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _reference_from_video(video_path: str, t_s: float, resize: int, expected_hw: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open reference video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_idx = max(0, int(round(float(t_s) * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise ValueError(f"Could not read reference frame {frame_idx} from {video_path}")

    frame_bgr = _resize_bgr_max_side(frame_bgr, resize)
    if frame_bgr.shape[:2] != expected_hw:
        raise ValueError(
            f"Reference frame shape {frame_bgr.shape[:2]} does not match modal field shape {expected_hw}. "
            "Pass a matching --reference-video/--reference-time-s or use the grayscale reference from the .npz."
        )
    return frame_bgr


def _reference_from_npz(reference_frame: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_frame)
    if ref.ndim == 2:
        gray = np.clip(ref.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if ref.ndim == 3 and ref.shape[2] == 3:
        return np.clip(ref.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"reference_frame must have shape (H,W) or (H,W,3), got {ref.shape}.")


def _load_reference_frame(z: np.lib.npyio.NpzFile, expected_hw: tuple[int, int], args: argparse.Namespace) -> np.ndarray:
    resize = _scalar_int(z["resize"], -1) if "resize" in z.files else -1
    reference_video = args.reference_video
    if reference_video is None and "source_video" in z.files:
        source_video = _scalar_str(z["source_video"])
        if source_video:
            reference_video = source_video

    if reference_video is not None:
        reference_time = args.reference_time_s
        if reference_time is None:
            reference_time = _scalar_float(z["t_ref_s"], 0.0) if "t_ref_s" in z.files else 0.0
        return _reference_from_video(reference_video, float(reference_time), resize, expected_hw)

    return _reference_from_npz(z["reference_frame"])


def _mask_from_npz(z: np.lib.npyio.NpzFile, expected_hw: tuple[int, int]) -> np.ndarray | None:
    if "has_mask" in z.files and int(np.asarray(z["has_mask"]).reshape(-1)[0]) == 0:
        return None
    if "mask" not in z.files:
        return None
    mask = np.asarray(z["mask"])
    if mask.size == 0:
        return None
    if mask.shape != expected_hw:
        raise ValueError(f"mask shape {mask.shape} does not match modal field shape {expected_hw}.")
    return mask.astype(bool)


def _infer_fft_num_samples(freqs_hz: np.ndarray, fps: float) -> int:
    freqs = np.asarray(freqs_hz, dtype=np.float64).reshape(-1)
    if freqs.shape[0] < 2:
        raise ValueError("Need at least two frequency bins to infer FFT normalization.")
    df = float(freqs[1] - freqs[0])
    if df <= 0:
        raise ValueError(f"Frequency bins must be increasing; got df={df}.")
    n = int(round(float(fps) / df))
    if n <= 0:
        raise ValueError(f"Could not infer positive FFT sample count from fps={fps}, df={df}.")
    if abs((float(fps) / float(n)) - df) > max(1e-6, 1e-4 * df):
        raise ValueError(f"Frequency spacing df={df} is inconsistent with fps={fps} and inferred N={n}.")
    return n


def _temporal_fft_reconstruction_gain(num_samples: int, window: str = "hann") -> float:
    n = int(num_samples)
    if n <= 0:
        raise ValueError("num_samples must be positive.")
    if window.lower() == "hann":
        if n == 1:
            coherent_gain = 1.0
        else:
            w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float32) / float(n - 1))
            coherent_gain = float(np.mean(w))
    else:
        coherent_gain = 1.0
    if coherent_gain <= 0:
        raise ValueError("Window coherent gain must be positive.")
    return float(2.0 / (float(n) * coherent_gain))


def _phase_hsv(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    phase = np.angle(z)
    mag = np.abs(z).astype(np.float32)
    hue = (phase + np.pi) / (2.0 * np.pi)
    sat = np.ones_like(hue, dtype=np.float32)
    val = np.clip((mag - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0).astype(np.float32)
    return hsv_to_rgb(np.stack([hue.astype(np.float32), sat, val], axis=-1))


def _save_mode_visuals(
    out_dir: Path,
    mode_idx: int,
    mode_u: np.ndarray,
    mode_v: np.ndarray,
    mask: np.ndarray | None,
    percentile: float,
) -> None:
    if not (0 < percentile <= 100):
        raise ValueError("preview-percentile must be in (0, 100].")
    amp = np.sqrt((np.abs(mode_u) ** 2 + np.abs(mode_v) ** 2).astype(np.float32))
    vals = amp[mask] if mask is not None and np.any(mask) else amp.ravel()
    hi = float(np.percentile(vals, percentile)) if vals.size else 1.0
    if hi <= 0:
        hi = 1.0
    amp_u8 = np.clip(amp / hi, 0.0, 1.0)
    if mask is not None:
        amp_u8 = amp_u8 * mask.astype(np.float32)
    cv2.imwrite(str(out_dir / f"mode_{mode_idx:02d}_amp.png"), np.clip(amp_u8 * 255.0, 0, 255).astype(np.uint8))

    u_rgb = _phase_hsv(mode_u, 0.0, hi)
    v_rgb = _phase_hsv(mode_v, 0.0, hi)
    if mask is not None:
        m3 = mask[..., None].astype(np.float32)
        u_rgb = u_rgb * m3
        v_rgb = v_rgb * m3
    cv2.imwrite(str(out_dir / f"mode_{mode_idx:02d}_u_hsv.png"), cv2.cvtColor(np.clip(u_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"mode_{mode_idx:02d}_v_hsv.png"), cv2.cvtColor(np.clip(v_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def _backward_warp(frame_bgr: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    map_x = xx.astype(np.float32) - dx.astype(np.float32)
    map_y = yy.astype(np.float32) - dy.astype(np.float32)
    return cv2.remap(
        frame_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _write_mode_video(
    frame_ref_bgr: np.ndarray,
    mode_u: np.ndarray,
    mode_v: np.ndarray,
    freq_hz: float,
    mask: np.ndarray | None,
    out_path: Path,
    fps_out: float,
    duration_s: float,
    speed: float,
    gain: float,
) -> None:
    h, w = frame_ref_bgr.shape[:2]
    if mode_u.shape != (h, w) or mode_v.shape != (h, w):
        raise ValueError(f"mode_u/mode_v shapes {mode_u.shape}/{mode_v.shape} do not match reference frame {(h, w)}.")
    n_frames = max(1, int(round(float(duration_s) * float(fps_out))))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps_out), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {out_path}")

    mask_f = None if mask is None else mask.astype(np.float32)
    try:
        for frame_idx in range(n_frames):
            t = float(frame_idx) / float(fps_out)
            phase = np.exp(1j * 2.0 * np.pi * float(freq_hz) * t * float(speed)).astype(np.complex64)
            phase_rel = (phase - np.complex64(1.0)).astype(np.complex64)
            dx = (float(gain) * np.real(mode_u * phase_rel)).astype(np.float32)
            dy = (float(gain) * np.real(mode_v * phase_rel)).astype(np.float32)
            if mask_f is not None:
                dx *= mask_f
                dy *= mask_f
            writer.write(_backward_warp(frame_ref_bgr, dx, dy))
    finally:
        writer.release()


def _mode_indices(num_modes: int, mode_index: int | None) -> list[int]:
    if mode_index is None:
        return list(range(num_modes))
    idx = int(mode_index) - 1
    if idx < 0 or idx >= num_modes:
        raise ValueError(f"--mode-index must be in [1,{num_modes}], got {mode_index}.")
    return [idx]


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(str(args.modal_npz), allow_pickle=False)
    required = ["mode_u", "mode_v", "selected_freqs_hz", "reference_frame", "fps", "freqs_hz"]
    missing = [key for key in required if key not in z.files]
    if missing:
        raise ValueError(f"{args.modal_npz} missing required arrays: {missing}")

    mode_u = z["mode_u"].astype(np.complex64, copy=False)
    mode_v = z["mode_v"].astype(np.complex64, copy=False)
    selected_freqs = z["selected_freqs_hz"].astype(np.float32, copy=False).reshape(-1)
    if mode_u.ndim != 3 or mode_v.shape != mode_u.shape:
        raise ValueError(f"mode_u/mode_v must have matching shape (K,H,W), got {mode_u.shape}/{mode_v.shape}.")
    if selected_freqs.shape[0] != mode_u.shape[0]:
        raise ValueError(f"selected_freqs_hz length {selected_freqs.shape[0]} does not match number of modes {mode_u.shape[0]}.")

    expected_hw = (int(mode_u.shape[1]), int(mode_u.shape[2]))
    frame_ref_bgr = _load_reference_frame(z, expected_hw, args)
    mask = _mask_from_npz(z, expected_hw)
    fps_in = _scalar_float(z["fps"], 30.0)
    fps_out = float(fps_in if args.fps_out is None else args.fps_out)
    if fps_out <= 0:
        raise ValueError("--fps-out must be positive.")
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive.")

    fft_gain = 1.0
    if not args.no_fft_normalize:
        num_samples = _infer_fft_num_samples(z["freqs_hz"], fps_in)
        fft_gain = _temporal_fft_reconstruction_gain(num_samples, window="hann")
    total_gain = float(args.gain) * float(fft_gain)

    print(f"Loaded modal analysis: {args.modal_npz}")
    print(f"Reference shape: {frame_ref_bgr.shape[:2]}, modes: {mode_u.shape[0]}")
    print(f"FFT normalization gain: {fft_gain:.8g}")
    print(f"User gain: {float(args.gain):.8g}")
    print(f"Total displacement gain: {total_gain:.8g}")

    for mode_i in _mode_indices(mode_u.shape[0], args.mode_index):
        out_idx = mode_i + 1
        freq = float(selected_freqs[mode_i])
        u = (mode_u[mode_i] * np.float32(total_gain)).astype(np.complex64, copy=False)
        v = (mode_v[mode_i] * np.float32(total_gain)).astype(np.complex64, copy=False)
        np.savez_compressed(
            out_dir / f"mode_{out_idx:02d}.npz",
            freq_hz=np.array(freq, dtype=np.float32),
            mode_u=u,
            mode_v=v,
            gain=np.array(total_gain, dtype=np.float32),
        )
        _save_mode_visuals(out_dir, out_idx, u, v, mask, float(args.preview_percentile))
        out_mp4 = out_dir / f"mode_{out_idx:02d}_{freq:.3f}Hz.mp4"
        _write_mode_video(
            frame_ref_bgr=frame_ref_bgr,
            mode_u=u,
            mode_v=v,
            freq_hz=freq,
            mask=mask,
            out_path=out_mp4,
            fps_out=fps_out,
            duration_s=float(args.duration_s),
            speed=float(args.speed),
            gain=1.0,
        )
        print(f"Saved mode {out_idx}: {freq:.6f} Hz -> {out_mp4}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthesize fixed-frequency 2D modal videos from modal_analysis.npz.")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
