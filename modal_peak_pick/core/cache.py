from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping

import numpy as np


CACHE_FORMAT = "modal_peak_pick_analysis"
CACHE_VERSION = 1

_ARRAY_FILES = {
    "flow_u": "flow_u.npy",
    "flow_v": "flow_v.npy",
    "spectrum_u": "spectrum_u.npy",
    "spectrum_v": "spectrum_v.npy",
    "freqs_hz": "freqs_hz.npy",
    "power_spectrum": "power_spectrum.npy",
    "reference_frame": "reference_frame.npy",
    "mask": "mask.npy",
}

_ARRAY_DTYPES = {
    "flow_u": np.dtype(np.float32),
    "flow_v": np.dtype(np.float32),
    "spectrum_u": np.dtype(np.complex64),
    "spectrum_v": np.dtype(np.complex64),
    "freqs_hz": np.dtype(np.float32),
    "power_spectrum": np.dtype(np.float32),
    "reference_frame": np.dtype(np.float32),
    "mask": np.dtype(np.uint8),
}

_REQUIRED_TIMINGS = (
    "decode",
    "flow",
    "smooth",
    "fft",
    "spectrum",
    "cache_write",
    "total",
)


@dataclass(frozen=True)
class ModalAnalysisCache:
    path: Path
    metadata: dict[str, Any]
    flow_u: np.ndarray
    flow_v: np.ndarray
    spectrum_u: np.ndarray
    spectrum_v: np.ndarray
    freqs_hz: np.ndarray
    power_spectrum: np.ndarray
    reference_frame: np.ndarray
    mask_array: np.ndarray

    @property
    def mask(self) -> np.ndarray | None:
        if not bool(self.metadata["has_mask"]):
            return None
        return self.mask_array.view(np.bool_)

    @property
    def fps(self) -> float:
        return float(self.metadata["video"]["fps"])

    @property
    def t_ref_s(self) -> float:
        return float(self.metadata["analysis"]["reference_time_s"])


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Cache metadata field {name!r} must be an object.")
    return value


def _validate_source(source: Mapping[str, Any], name: str) -> None:
    if not isinstance(source.get("path"), str) or not source["path"]:
        raise ValueError(f"Cache metadata {name}.path must be a non-empty string.")
    fingerprint = _require_mapping(source.get("fingerprint"), f"{name}.fingerprint")
    if fingerprint.get("method") != "stat":
        raise ValueError(f"Cache metadata {name}.fingerprint.method must be 'stat'.")
    for field in ("size_bytes", "mtime_ns"):
        value = fingerprint.get(field)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Cache metadata {name}.fingerprint.{field} must be non-negative integer.")


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("cache_format") != CACHE_FORMAT:
        raise ValueError(
            f"Unsupported cache format {metadata.get('cache_format')!r}; expected {CACHE_FORMAT!r}."
        )
    if metadata.get("cache_version") != CACHE_VERSION:
        raise ValueError(
            f"Unsupported cache version {metadata.get('cache_version')!r}; expected {CACHE_VERSION}."
        )
    if not isinstance(metadata.get("has_mask"), bool):
        raise ValueError("Cache metadata field 'has_mask' must be boolean.")

    sources = _require_mapping(metadata.get("sources"), "sources")
    video_source = _require_mapping(sources.get("video"), "sources.video")
    _validate_source(video_source, "sources.video")
    mask_source = sources.get("mask")
    if mask_source is not None:
        mask_source = _require_mapping(mask_source, "sources.mask")
        _validate_source(mask_source, "sources.mask")
    if bool(metadata["has_mask"]) != (mask_source is not None):
        raise ValueError("Cache metadata has_mask must agree with sources.mask.")

    video = _require_mapping(metadata.get("video"), "video")
    fps = float(video.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("Cache metadata video.fps must be finite and positive.")
    frame_range = _require_mapping(video.get("frame_range"), "video.frame_range")
    t0_s = float(frame_range.get("t0_s", np.nan))
    if not np.isfinite(t0_s) or t0_s < 0.0:
        raise ValueError("Cache metadata video.frame_range.t0_s must be finite and non-negative.")
    t1_s = frame_range.get("t1_s")
    if t1_s is not None and (not np.isfinite(float(t1_s)) or float(t1_s) <= t0_s):
        raise ValueError("Cache metadata video.frame_range.t1_s must be greater than t0_s.")
    max_frames = frame_range.get("max_frames")
    if max_frames is not None and (not isinstance(max_frames, int) or max_frames <= 0):
        raise ValueError("Cache metadata video.frame_range.max_frames must be positive when present.")
    decoded_frame_count = frame_range.get("decoded_frame_count")
    if not isinstance(decoded_frame_count, int) or decoded_frame_count < 3:
        raise ValueError("Cache metadata video.frame_range.decoded_frame_count must be at least three.")
    resize = video.get("resize_max_side")
    if resize is not None and (not isinstance(resize, int) or resize <= 0):
        raise ValueError("Cache metadata video.resize_max_side must be positive when present.")

    analysis = _require_mapping(metadata.get("analysis"), "analysis")
    if analysis.get("flow_method") not in {"farneback", "tvl1"}:
        raise ValueError("Cache metadata analysis.flow_method must be 'farneback' or 'tvl1'.")
    _require_mapping(analysis.get("flow_parameters"), "analysis.flow_parameters")
    smoothing = _require_mapping(analysis.get("smoothing"), "analysis.smoothing")
    if not isinstance(smoothing.get("disabled"), bool):
        raise ValueError("Cache metadata analysis.smoothing.disabled must be boolean.")
    for field in ("sigma_b", "sigma_c"):
        value = float(smoothing.get(field, np.nan))
        if not np.isfinite(value):
            raise ValueError(f"Cache metadata analysis.smoothing.{field} must be finite.")
    dilate_iters = smoothing.get("analysis_mask_dilate_iters")
    if not isinstance(dilate_iters, int) or dilate_iters < 0:
        raise ValueError(
            "Cache metadata analysis.smoothing.analysis_mask_dilate_iters must be non-negative."
        )
    if not isinstance(analysis.get("reference_frame_index"), int):
        raise ValueError("Cache metadata analysis.reference_frame_index must be an integer.")
    reference_time_s = float(analysis.get("reference_time_s", np.nan))
    if not np.isfinite(reference_time_s):
        raise ValueError("Cache metadata analysis.reference_time_s must be finite.")
    fft = _require_mapping(analysis.get("fft"), "analysis.fft")
    if not isinstance(fft.get("detrend"), bool):
        raise ValueError("Cache metadata analysis.fft.detrend must be boolean.")
    if not isinstance(fft.get("window"), str):
        raise ValueError("Cache metadata analysis.fft.window must be a string.")
    block_width = fft.get("block_width")
    if not isinstance(block_width, int) or block_width <= 0:
        raise ValueError("Cache metadata analysis.fft.block_width must be positive.")
    _require_mapping(analysis.get("spectrum"), "analysis.spectrum")

    timings = _require_mapping(metadata.get("timings_seconds"), "timings_seconds")
    for name in _REQUIRED_TIMINGS:
        value = float(timings.get(name, np.nan))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Cache timing {name!r} must be finite and non-negative.")

    arrays = _require_mapping(metadata.get("arrays"), "arrays")
    if set(arrays) != set(_ARRAY_FILES):
        raise ValueError(
            f"Cache metadata arrays must contain exactly {sorted(_ARRAY_FILES)}; got {sorted(arrays)}."
        )
    for name, filename in _ARRAY_FILES.items():
        entry = _require_mapping(arrays[name], f"arrays.{name}")
        if entry.get("file") != filename:
            raise ValueError(f"Cache metadata arrays.{name}.file must be {filename!r}.")
        if entry.get("dtype") != _ARRAY_DTYPES[name].name:
            raise ValueError(
                f"Cache metadata arrays.{name}.dtype must be {_ARRAY_DTYPES[name].name!r}."
            )
        shape = entry.get("shape")
        if not isinstance(shape, list) or not all(isinstance(v, int) and v >= 0 for v in shape):
            raise ValueError(f"Cache metadata arrays.{name}.shape must be a list of non-negative integers.")


def _validate_arrays(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    for name, array in arrays.items():
        expected_dtype = _ARRAY_DTYPES[name]
        if array.dtype != expected_dtype:
            raise ValueError(
                f"Cache array {name!r} has dtype {array.dtype}; expected {expected_dtype}."
            )
        expected_shape = tuple(metadata["arrays"][name]["shape"])
        if array.shape != expected_shape:
            raise ValueError(
                f"Cache array {name!r} has shape {array.shape}; metadata declares {expected_shape}."
            )

    flow_u = arrays["flow_u"]
    flow_v = arrays["flow_v"]
    spectrum_u = arrays["spectrum_u"]
    spectrum_v = arrays["spectrum_v"]
    freqs_hz = arrays["freqs_hz"]
    power_spectrum = arrays["power_spectrum"]
    reference_frame = arrays["reference_frame"]
    mask = arrays["mask"]

    if flow_u.ndim != 3 or flow_u.shape != flow_v.shape:
        raise ValueError("Cache flow_u and flow_v must have the same [T,H,W] shape.")
    if spectrum_u.ndim != 3 or spectrum_u.shape != spectrum_v.shape:
        raise ValueError("Cache spectrum_u and spectrum_v must have the same [F,H,W] shape.")
    num_frames, height, width = flow_u.shape
    num_freqs = num_frames // 2 + 1
    if num_frames < 3:
        raise ValueError("Cache flow arrays must contain at least three frames.")
    if spectrum_u.shape != (num_freqs, height, width):
        raise ValueError(
            "Cache spectrum shape must be [T//2+1,H,W] and spatially match the flow arrays."
        )
    if freqs_hz.shape != (num_freqs,) or power_spectrum.shape != (num_freqs,):
        raise ValueError("Cache frequency and power arrays must both have shape [T//2+1].")
    if reference_frame.shape != (height, width) or mask.shape != (height, width):
        raise ValueError("Cache reference frame and mask must spatially match the flow arrays.")
    if int(metadata["video"]["frame_range"]["decoded_frame_count"]) != num_frames:
        raise ValueError("Cache decoded frame count does not match the flow arrays.")
    expected_freqs_hz = np.fft.rfftfreq(num_frames, d=1.0 / float(metadata["video"]["fps"]))
    if not np.allclose(freqs_hz, expected_freqs_hz, rtol=1e-6, atol=1e-6):
        raise ValueError("Cache frequency axis does not match the cached FPS and frame count.")
    if not np.all(np.isfinite(power_spectrum)):
        raise ValueError("Cache power spectrum must be finite.")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Cache mask must contain only 0 and 1 values.")
    if not bool(metadata["has_mask"]) and np.any(mask):
        raise ValueError("Cache mask must be all zeros when has_mask is false.")

    reference_index = int(metadata["analysis"]["reference_frame_index"])
    if reference_index < 0 or reference_index >= num_frames:
        raise ValueError("Cache reference frame index is outside the cached temporal range.")


def load_analysis_cache(cache_dir: str | Path) -> ModalAnalysisCache:
    path = Path(cache_dir).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Modal analysis cache directory does not exist: {path}")
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Modal analysis cache is missing metadata.json: {path}")
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata_value = json.load(file)
    if not isinstance(metadata_value, dict):
        raise ValueError("Cache metadata root must be a JSON object.")
    metadata: dict[str, Any] = metadata_value
    _validate_metadata(metadata)

    arrays: dict[str, np.ndarray] = {}
    for name, filename in _ARRAY_FILES.items():
        array_path = path / filename
        if not array_path.is_file():
            raise FileNotFoundError(f"Modal analysis cache is missing {filename}: {path}")
        loaded = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if not isinstance(loaded, np.ndarray):
            raise ValueError(f"Cache file {filename} must contain one .npy array.")
        arrays[name] = loaded
    _validate_arrays(arrays, metadata)

    return ModalAnalysisCache(
        path=path,
        metadata=metadata,
        flow_u=arrays["flow_u"],
        flow_v=arrays["flow_v"],
        spectrum_u=arrays["spectrum_u"],
        spectrum_v=arrays["spectrum_v"],
        freqs_hz=arrays["freqs_hz"],
        power_spectrum=arrays["power_spectrum"],
        reference_frame=arrays["reference_frame"],
        mask_array=arrays["mask"],
    )


def write_analysis_cache(
    cache_dir: str | Path,
    *,
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    spectrum_u: np.ndarray,
    spectrum_v: np.ndarray,
    freqs_hz: np.ndarray,
    power_spectrum: np.ndarray,
    reference_frame: np.ndarray,
    mask: np.ndarray | None,
    metadata: Mapping[str, Any],
) -> ModalAnalysisCache:
    target = Path(cache_dir).expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Modal analysis cache target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "flow_u": np.asarray(flow_u, dtype=np.float32),
        "flow_v": np.asarray(flow_v, dtype=np.float32),
        "spectrum_u": np.asarray(spectrum_u, dtype=np.complex64),
        "spectrum_v": np.asarray(spectrum_v, dtype=np.complex64),
        "freqs_hz": np.asarray(freqs_hz, dtype=np.float32),
        "power_spectrum": np.asarray(power_spectrum, dtype=np.float32),
        "reference_frame": np.asarray(reference_frame, dtype=np.float32),
        "mask": (
            np.zeros(np.asarray(reference_frame).shape, dtype=np.uint8)
            if mask is None
            else np.asarray(mask, dtype=np.uint8)
        ),
    }

    metadata_out = dict(metadata)
    metadata_out["cache_format"] = CACHE_FORMAT
    metadata_out["cache_version"] = CACHE_VERSION
    metadata_out["has_mask"] = mask is not None
    metadata_out["arrays"] = {
        name: {
            "file": _ARRAY_FILES[name],
            "dtype": _ARRAY_DTYPES[name].name,
            "shape": list(array.shape),
        }
        for name, array in arrays.items()
    }

    temp_path = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
    )
    write_start = time.perf_counter()
    try:
        for name, array in arrays.items():
            np.save(temp_path / _ARRAY_FILES[name], array, allow_pickle=False)

        timings = dict(_require_mapping(metadata_out.get("timings_seconds"), "timings_seconds"))
        timings["cache_write"] = float(time.perf_counter() - write_start)
        timings["total"] = float(timings.get("analysis_total", 0.0)) + timings["cache_write"]
        metadata_out["timings_seconds"] = timings
        with (temp_path / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata_out, file, indent=2, sort_keys=True)

        validated_cache = load_analysis_cache(temp_path)
        del validated_cache
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Modal analysis cache target already exists: {target}")
        os.replace(temp_path, target)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise
    return load_analysis_cache(target)
