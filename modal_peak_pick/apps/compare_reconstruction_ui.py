import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import hsv_to_rgb
from matplotlib.figure import Figure
import numpy as np

from flow3d.modal_flow_coordinates import (
    _load_manifest,
    _view_pixel_groups,
    parse_flow_cache_specs,
)
from flow3d.modal_frequency_selection import _temporal_basis
from modal_peak_pick.core.cache import ModalAnalysisCache, load_analysis_cache
from modal_surface.io import save_npz_compressed_atomic


_PIXEL_CHUNK_SIZE = 2048
_COMPARISON_CACHE_FORMAT = "modal_reconstruction_comparison"
_COMPARISON_CACHE_VERSION = 1


def add_arguments(parser: argparse.ArgumentParser) -> None:
    cache_group = parser.add_mutually_exclusive_group(required=True)
    cache_group.add_argument(
        "--cache-dir",
        help="Modal-analysis cache directory for the first manifest view.",
    )
    cache_group.add_argument(
        "--flow-caches",
        nargs="+",
        metavar="VIEW_ID=PATH",
        help="Modal-analysis cache for every manifest view, in any order.",
    )
    parser.add_argument(
        "--modal-manifest",
        required=True,
        help="Solved Gaussian modal manifest used for the reconstruction.",
    )
    parser.add_argument(
        "--comparison-cache-dir",
        default=None,
        help="Optional persistent cache directory for comparison projections and exact modes.",
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


def _hash_array(hasher: Any, name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    hasher.update(name.encode("utf-8"))
    hasher.update(array.dtype.str.encode("ascii"))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    contiguous = np.ascontiguousarray(array)
    hasher.update(memoryview(contiguous).cast("B"))


def _manifest_projection_identity(manifest: Any) -> str:
    hasher = hashlib.sha256()
    hasher.update(_COMPARISON_CACHE_FORMAT.encode("ascii"))
    hasher.update(str(_COMPARISON_CACHE_VERSION).encode("ascii"))
    hasher.update(str(manifest.path).encode("utf-8"))
    _hash_array(hasher, "mode_indices", manifest.mode_indices)
    _hash_array(hasher, "frequencies_hz", manifest.frequencies_hz)
    _hash_array(hasher, "phi", manifest.phi)
    topology = manifest.topology
    for name in (
        "obs_point_index",
        "obs_view_index",
        "obs_pixels_xy",
        "obs_J",
        "obs_contribution_weight",
        "view_image_width",
        "view_image_height",
    ):
        _hash_array(hasher, name, getattr(topology, name))
    hasher.update(json.dumps(list(topology.view_ids)).encode("utf-8"))
    return hasher.hexdigest()


def _flow_cache_identity(cache: ModalAnalysisCache) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(cache.path.resolve()).encode("utf-8"))
    hasher.update(
        json.dumps(cache.metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    arrays = cache.metadata.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError(f"Modal-analysis cache {cache.path} has invalid array metadata")
    for name in sorted(arrays):
        entry = arrays[name]
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ValueError(
                f"Modal-analysis cache {cache.path} has invalid metadata for array {name!r}"
            )
        array_path = cache.path / entry["file"]
        stat = array_path.stat()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("ascii"))
        hasher.update(str(stat.st_mtime_ns).encode("ascii"))
    return hasher.hexdigest()


def _view_source_identity(
    manifest_identity: str,
    cache: ModalAnalysisCache,
    view_id: str,
    view_index: int,
    alphas: np.ndarray,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(manifest_identity.encode("ascii"))
    hasher.update(_flow_cache_identity(cache).encode("ascii"))
    hasher.update(view_id.encode("utf-8"))
    hasher.update(str(view_index).encode("ascii"))
    _hash_array(hasher, "alphas", alphas)
    return hasher.hexdigest()


def _scalar_string(value: np.ndarray, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{path} {name} must be a scalar string")
    item = array.item()
    if not isinstance(item, str) or not item:
        raise ValueError(f"{path} {name} must be a non-empty scalar string")
    return item


def _scalar_int(value: np.ndarray, name: str, path: Path) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{path} {name} must be an integer scalar")
    return int(array.item())


@dataclass(frozen=True)
class _ViewState:
    cache: ModalAnalysisCache
    view_index: int
    source_identity: str
    disk_directory: Path | None
    pixels: np.ndarray
    raw_power: np.ndarray
    reconstructed_modes: np.ndarray
    reconstructed_power: np.ndarray
    frequency_limits: tuple[float, float]


def _load_view_disk_cache(
    path: Path,
    source_identity: str,
    view_id: str,
    view_index: int,
    cache: ModalAnalysisCache,
    manifest: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "cache_format",
        "cache_version",
        "source_identity",
        "view_id",
        "view_index",
        "mode_indices",
        "frequencies_hz",
        "pixels",
        "raw_power",
        "reconstructed_modes",
        "reconstructed_power",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(
                f"{path} fields must be exactly {sorted(required)}; got {sorted(archive.files)}"
            )
        if _scalar_string(archive["cache_format"], "cache_format", path) != (
            _COMPARISON_CACHE_FORMAT
        ):
            raise ValueError(f"{path} has an unsupported comparison cache format")
        if _scalar_int(archive["cache_version"], "cache_version", path) != (
            _COMPARISON_CACHE_VERSION
        ):
            raise ValueError(f"{path} has an unsupported comparison cache version")
        if _scalar_string(archive["source_identity"], "source_identity", path) != (
            source_identity
        ):
            raise ValueError(f"{path} source identity does not match its cache directory")
        if _scalar_string(archive["view_id"], "view_id", path) != view_id:
            raise ValueError(f"{path} view_id does not match {view_id!r}")
        if _scalar_int(archive["view_index"], "view_index", path) != view_index:
            raise ValueError(f"{path} view_index does not match {view_index}")
        mode_indices = np.asarray(archive["mode_indices"])
        frequencies_hz = np.asarray(archive["frequencies_hz"])
        pixels = np.asarray(archive["pixels"])
        raw_power = np.asarray(archive["raw_power"])
        reconstructed_modes = np.asarray(archive["reconstructed_modes"])
        reconstructed_power = np.asarray(archive["reconstructed_power"])

    if not np.array_equal(mode_indices, manifest.mode_indices):
        raise ValueError(f"{path} mode_indices do not match the modal manifest")
    if not np.array_equal(frequencies_hz, manifest.frequencies_hz):
        raise ValueError(f"{path} frequencies_hz do not match the modal manifest")
    if pixels.dtype != np.int64 or pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"{path} pixels must be int64 with shape [P,2]")
    if pixels.shape[0] == 0:
        raise ValueError(f"{path} pixels must not be empty")
    height, width = cache.reference_frame.shape
    if (
        np.any(pixels[:, 0] < 0)
        or np.any(pixels[:, 0] >= width)
        or np.any(pixels[:, 1] < 0)
        or np.any(pixels[:, 1] >= height)
    ):
        raise ValueError(f"{path} pixels fall outside the view image")
    if raw_power.dtype != np.float32 or raw_power.shape != cache.freqs_hz.shape:
        raise ValueError(f"{path} raw_power must be float32 and match cache frequencies")
    expected_modes_shape = (manifest.frequencies_hz.size, pixels.shape[0], 2)
    if (
        reconstructed_modes.dtype != np.complex64
        or reconstructed_modes.shape != expected_modes_shape
    ):
        raise ValueError(
            f"{path} reconstructed_modes must be complex64 with shape {expected_modes_shape}"
        )
    if (
        reconstructed_power.dtype != np.float32
        or reconstructed_power.shape != manifest.frequencies_hz.shape
    ):
        raise ValueError(
            f"{path} reconstructed_power must be float32 and match manifest frequencies"
        )
    if (
        not np.isfinite(raw_power).all()
        or not np.isfinite(reconstructed_modes.real).all()
        or not np.isfinite(reconstructed_modes.imag).all()
        or not np.isfinite(reconstructed_power).all()
    ):
        raise ValueError(f"{path} contains non-finite comparison values")
    expected_power = np.mean(
        np.sqrt(
            np.abs(reconstructed_modes[:, :, 0]) ** 2
            + np.abs(reconstructed_modes[:, :, 1]) ** 2
        ),
        axis=1,
    ).astype(np.float32)
    if not np.allclose(reconstructed_power, expected_power, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{path} reconstructed_power does not match reconstructed_modes")
    return pixels, raw_power, reconstructed_modes, reconstructed_power


def _write_view_disk_cache(
    path: Path,
    source_identity: str,
    view_id: str,
    view_index: int,
    manifest: Any,
    pixels: np.ndarray,
    raw_power: np.ndarray,
    reconstructed_modes: np.ndarray,
    reconstructed_power: np.ndarray,
) -> None:
    save_npz_compressed_atomic(
        path,
        {
            "cache_format": np.array(_COMPARISON_CACHE_FORMAT),
            "cache_version": np.array(_COMPARISON_CACHE_VERSION, dtype=np.int64),
            "source_identity": np.array(source_identity),
            "view_id": np.array(view_id),
            "view_index": np.array(view_index, dtype=np.int64),
            "mode_indices": np.asarray(manifest.mode_indices),
            "frequencies_hz": np.asarray(manifest.frequencies_hz),
            "pixels": np.asarray(pixels, dtype=np.int64),
            "raw_power": np.asarray(raw_power, dtype=np.float32),
            "reconstructed_modes": np.asarray(reconstructed_modes, dtype=np.complex64),
            "reconstructed_power": np.asarray(reconstructed_power, dtype=np.float32),
        },
    )


def _exact_mode_cache_path(directory: Path, frequency_hz: float) -> Path:
    frequency_bytes = np.float64(frequency_hz).tobytes()
    digest = hashlib.sha256(frequency_bytes).hexdigest()
    return directory / "exact_modes" / f"frequency_{digest}.npz"


def _load_exact_mode_disk_cache(
    path: Path,
    source_identity: str,
    view_id: str,
    frequency_hz: float,
    num_pixels: int,
) -> np.ndarray:
    required = {
        "cache_format",
        "cache_version",
        "source_identity",
        "view_id",
        "frequency_hz",
        "mode",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(
                f"{path} fields must be exactly {sorted(required)}; got {sorted(archive.files)}"
            )
        if _scalar_string(archive["cache_format"], "cache_format", path) != (
            _COMPARISON_CACHE_FORMAT
        ):
            raise ValueError(f"{path} has an unsupported exact-mode cache format")
        if _scalar_int(archive["cache_version"], "cache_version", path) != (
            _COMPARISON_CACHE_VERSION
        ):
            raise ValueError(f"{path} has an unsupported exact-mode cache version")
        if _scalar_string(archive["source_identity"], "source_identity", path) != (
            source_identity
        ):
            raise ValueError(f"{path} source identity does not match its cache directory")
        if _scalar_string(archive["view_id"], "view_id", path) != view_id:
            raise ValueError(f"{path} view_id does not match {view_id!r}")
        stored_frequency = np.asarray(archive["frequency_hz"])
        mode = np.asarray(archive["mode"])
    if stored_frequency.shape != () or stored_frequency.dtype != np.float64:
        raise ValueError(f"{path} frequency_hz must be a float64 scalar")
    if stored_frequency.tobytes() != np.float64(frequency_hz).tobytes():
        raise ValueError(f"{path} frequency_hz does not match its filename key")
    if mode.dtype != np.complex64 or mode.shape != (num_pixels, 2):
        raise ValueError(
            f"{path} mode must be complex64 with shape {(num_pixels, 2)}"
        )
    if not np.isfinite(mode.real).all() or not np.isfinite(mode.imag).all():
        raise ValueError(f"{path} mode contains non-finite values")
    return mode


def _write_exact_mode_disk_cache(
    path: Path,
    source_identity: str,
    view_id: str,
    frequency_hz: float,
    mode: np.ndarray,
) -> None:
    save_npz_compressed_atomic(
        path,
        {
            "cache_format": np.array(_COMPARISON_CACHE_FORMAT),
            "cache_version": np.array(_COMPARISON_CACHE_VERSION, dtype=np.int64),
            "source_identity": np.array(source_identity),
            "view_id": np.array(view_id),
            "frequency_hz": np.array(frequency_hz, dtype=np.float64),
            "mode": np.asarray(mode, dtype=np.complex64),
        },
    )


class SpectrumComparisonController:
    def __init__(
        self,
        modal_manifest: str | Path,
        preview_percentile: float,
        cache_dir: str | Path | None = None,
        flow_caches: Sequence[str] | None = None,
        comparison_cache_dir: str | Path | None = None,
    ) -> None:
        if not (0.0 < preview_percentile <= 100.0):
            raise ValueError("preview_percentile must be in (0, 100]")
        self.preview_percentile = float(preview_percentile)
        self.manifest = _load_manifest(modal_manifest)
        manifest_view_ids = tuple(str(view_id) for view_id in self.manifest.topology.view_ids)
        if not manifest_view_ids:
            raise ValueError("Modal manifest contains no views")
        if flow_caches is not None:
            parsed_caches = parse_flow_cache_specs(flow_caches)
            cache_paths = dict(parsed_caches)
            missing = [view_id for view_id in manifest_view_ids if view_id not in cache_paths]
            extra = [view_id for view_id in cache_paths if view_id not in manifest_view_ids]
            if missing or extra:
                raise ValueError(
                    "--flow-caches must match all manifest views exactly; "
                    f"missing={missing}, extra={extra}"
                )
            self.available_view_ids = manifest_view_ids
            self.cache_paths = cache_paths
        elif cache_dir is not None:
            self.available_view_ids = (manifest_view_ids[0],)
            self.cache_paths = {manifest_view_ids[0]: Path(cache_dir).expanduser()}
        else:
            raise ValueError("Either cache_dir or flow_caches must be provided")

        if np.unique(self.manifest.frequencies_hz).size != self.manifest.frequencies_hz.size:
            raise ValueError("Reconstructed modal frequencies must be unique")
        self.comparison_cache_dir = (
            None
            if comparison_cache_dir is None
            else Path(comparison_cache_dir).expanduser()
        )
        self.manifest_identity = (
            None
            if self.comparison_cache_dir is None
            else _manifest_projection_identity(self.manifest)
        )
        self._view_states: dict[str, _ViewState] = {}
        self._exact_modes: dict[tuple[str, bytes], np.ndarray] = {}
        self.component_index = 0
        self.raw_frequency_hz = float(self.manifest.frequencies_hz[0])
        self.reconstructed_index = 0
        self.raw_spectrum_mapping: tuple[float, float, float, float, int] | None = None
        self.reconstructed_spectrum_mapping: tuple[float, float, float, float, int] | None = None
        self._load_view(self.available_view_ids[0])

    def _load_view(self, view_id: str) -> None:
        if view_id not in self.cache_paths:
            raise ValueError(f"Unknown comparison view {view_id!r}")
        state = self._view_states.get(view_id)
        if state is None:
            state = self._prepare_view_state(view_id)
            self._view_states[view_id] = state
        self.view_id = view_id
        self.view_index = state.view_index
        self.cache = state.cache
        self.source_identity = state.source_identity
        self.disk_directory = state.disk_directory
        self.pixels = state.pixels
        self.raw_power = state.raw_power
        self.reconstructed_modes = state.reconstructed_modes
        self.reconstructed_power = state.reconstructed_power
        self.frequency_limits = state.frequency_limits
        self.raw_mode = np.empty((self.pixels.shape[0], 2), dtype=np.complex64)
        self._set_frequencies(self.raw_frequency_hz, self.reconstructed_index)

    def _prepare_view_state(self, view_id: str) -> _ViewState:
        view_index = list(self.manifest.topology.view_ids).index(view_id)
        cache = load_analysis_cache(self.cache_paths[view_id])

        expected_shape = (
            int(self.manifest.topology.view_image_height[view_index]),
            int(self.manifest.topology.view_image_width[view_index]),
        )
        if cache.flow_u.shape[1:] != expected_shape:
            raise ValueError(
                f"Cache shape {cache.flow_u.shape[1:]} does not match "
                f"manifest view {view_id!r} shape {expected_shape}"
            )
        if cache.mask is None:
            raise ValueError(
                f"Modal-analysis cache for view {view_id!r} must contain a foreground mask"
            )

        alphas = _view_alphas(
            self.manifest.path,
            self.manifest,
            view_index,
        )
        disk_directory: Path | None = None
        if self.comparison_cache_dir is None:
            source_identity = f"memory:{view_id}"
        else:
            assert self.manifest_identity is not None
            source_identity = _view_source_identity(
                self.manifest_identity,
                cache,
                view_id,
                view_index,
                alphas,
            )
            view_digest = hashlib.sha256(view_id.encode("utf-8")).hexdigest()[:12]
            disk_directory = (
                self.comparison_cache_dir
                / f"view_{view_index:03d}_{view_digest}"
                / source_identity
            )

        view_cache_path = (
            None if disk_directory is None else disk_directory / "view_cache.npz"
        )
        if view_cache_path is not None and view_cache_path.is_file():
            (
                pixels,
                raw_power,
                reconstructed_modes,
                reconstructed_power,
            ) = _load_view_disk_cache(
                view_cache_path,
                source_identity,
                view_id,
                view_index,
                cache,
                self.manifest,
            )
            print(f"Loaded comparison view cache -> {view_cache_path}")
        else:
            (
                sorted_rows,
                starts,
                pixels,
                sorted_weights,
                _,
            ) = _view_pixel_groups(self.manifest.topology, view_index, cache)
            pixels = np.asarray(pixels, dtype=np.int64)
            raw_power = _candidate_power_spectrum(cache, pixels)
            reconstructed_modes = _project_reconstructed_modes(
                self.manifest,
                sorted_rows,
                starts,
                sorted_weights,
                alphas,
            )
            reconstructed_power = np.mean(
                np.sqrt(
                    np.abs(reconstructed_modes[:, :, 0]) ** 2
                    + np.abs(reconstructed_modes[:, :, 1]) ** 2
                ),
                axis=1,
            ).astype(np.float32)
            if not np.isfinite(reconstructed_power).all():
                raise ValueError("Reconstructed power spectrum is non-finite")
            if view_cache_path is not None:
                _write_view_disk_cache(
                    view_cache_path,
                    source_identity,
                    view_id,
                    view_index,
                    self.manifest,
                    pixels,
                    raw_power,
                    reconstructed_modes,
                    reconstructed_power,
                )
                print(f"Saved comparison view cache -> {view_cache_path}")

        raw_frequency_min = float(cache.freqs_hz[0])
        raw_frequency_max = float(cache.freqs_hz[-1])
        if np.any(self.manifest.frequencies_hz < raw_frequency_min) or np.any(
            self.manifest.frequencies_hz > raw_frequency_max
        ):
            raise ValueError("Reconstructed frequencies fall outside the raw spectrum range")

        selected_min = float(np.min(self.manifest.frequencies_hz))
        selected_max = float(np.max(self.manifest.frequencies_hz))
        selected_span = max(
            selected_max - selected_min,
            float(cache.freqs_hz[1] - cache.freqs_hz[0]),
        )
        margin = 0.05 * selected_span
        frequency_limits = (
            max(raw_frequency_min, selected_min - margin),
            min(raw_frequency_max, selected_max + margin),
        )
        return _ViewState(
            cache=cache,
            view_index=view_index,
            source_identity=source_identity,
            disk_directory=disk_directory,
            pixels=pixels,
            raw_power=raw_power,
            reconstructed_modes=reconstructed_modes,
            reconstructed_power=reconstructed_power,
            frequency_limits=frequency_limits,
        )

    def _load_exact_mode(self, frequency_hz: float) -> np.ndarray:
        frequency_bytes = np.float64(frequency_hz).tobytes()
        memory_key = (self.source_identity, frequency_bytes)
        cached = self._exact_modes.get(memory_key)
        if cached is not None:
            return cached

        disk_path = (
            None
            if self.disk_directory is None
            else _exact_mode_cache_path(self.disk_directory, frequency_hz)
        )
        if disk_path is not None and disk_path.is_file():
            mode = _load_exact_mode_disk_cache(
                disk_path,
                self.source_identity,
                self.view_id,
                frequency_hz,
                self.pixels.shape[0],
            )
            print(f"Loaded exact-mode cache -> {disk_path}")
        else:
            mode = _exact_candidate_mode(
                self.cache,
                self.pixels,
                frequency_hz,
            )
            if disk_path is not None:
                _write_exact_mode_disk_cache(
                    disk_path,
                    self.source_identity,
                    self.view_id,
                    frequency_hz,
                    mode,
                )
                print(f"Saved exact-mode cache -> {disk_path}")
        self._exact_modes[memory_key] = mode
        return mode

    def _set_frequencies(self, raw_frequency_hz: float, reconstructed_index: int) -> None:
        raw_min = float(self.cache.freqs_hz[0])
        raw_max = float(self.cache.freqs_hz[-1])
        self.raw_frequency_hz = float(np.clip(raw_frequency_hz, raw_min, raw_max))
        self.reconstructed_index = int(reconstructed_index)
        if not (0 <= self.reconstructed_index < self.manifest.frequencies_hz.size):
            raise ValueError("Reconstructed frequency index is outside the manifest")
        self.raw_mode = self._load_exact_mode(self.raw_frequency_hz)
        self._render_all()

    def _render_spectrum(
        self,
        frequencies_hz: np.ndarray,
        power: np.ndarray,
        selected_frequency_hz: float,
        selected_power: float,
        title: str,
        discrete: bool,
        power_limits: tuple[float, float],
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
        axis.set_ylim(*power_limits)
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
        raw_visible = (
            (self.cache.freqs_hz >= self.frequency_limits[0])
            & (self.cache.freqs_hz <= self.frequency_limits[1])
        )
        reconstructed_visible = (
            (self.manifest.frequencies_hz >= self.frequency_limits[0])
            & (self.manifest.frequencies_hz <= self.frequency_limits[1])
        )
        visible_power_maxima = [raw_power, reconstructed_power]
        if np.any(raw_visible):
            visible_power_maxima.append(float(np.max(self.raw_power[raw_visible])))
        if np.any(reconstructed_visible):
            visible_power_maxima.append(
                float(np.max(self.reconstructed_power[reconstructed_visible]))
            )
        shared_power_max = max(visible_power_maxima)
        power_limits = (
            0.0,
            1.05 * shared_power_max if shared_power_max > 0.0 else 1.0,
        )
        self.raw_spectrum_image, self.raw_spectrum_mapping = self._render_spectrum(
            self.cache.freqs_hz,
            self.raw_power,
            self.raw_frequency_hz,
            raw_power,
            f"Original {self.view_id} spectrum - {self.raw_frequency_hz:.5f} Hz",
            discrete=False,
            power_limits=power_limits,
        )
        (
            self.reconstructed_spectrum_image,
            self.reconstructed_spectrum_mapping,
        ) = self._render_spectrum(
            self.manifest.frequencies_hz,
            self.reconstructed_power,
            reconstructed_frequency,
            reconstructed_power,
            f"Reconstructed {self.view_id} spectrum - {reconstructed_frequency:.5f} Hz",
            discrete=True,
            power_limits=power_limits,
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

    def select_view(self, view_id: str):
        self._load_view(view_id)
        return self.outputs()


def make_demo(controller: SpectrumComparisonController):
    import gradio as gr

    with gr.Blocks(title="Modal spectrum reconstruction comparison") as demo:
        gr.Markdown(
            "# Modal spectrum comparison\n"
            "Click either spectrum to select a frequency. Original clicks retain the "
            "exact clicked frequency; reconstructed clicks snap both rows to the nearest "
            "solved frequency."
        )
        view = gr.Dropdown(
            choices=controller.available_view_ids,
            value=controller.view_id,
            label="View",
            interactive=True,
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
        view.change(controller.select_view, inputs=[view], outputs=outputs)
        component.change(controller.select_component, inputs=[component], outputs=outputs)
    return demo


def run(args: argparse.Namespace) -> None:
    controller = SpectrumComparisonController(
        modal_manifest=args.modal_manifest,
        preview_percentile=args.preview_percentile,
        cache_dir=args.cache_dir,
        flow_caches=args.flow_caches,
        comparison_cache_dir=args.comparison_cache_dir,
    )
    print(
        f"Loaded views {list(controller.available_view_ids)!r}; "
        f"active={controller.view_id!r}: "
        f"candidate_pixels={controller.pixels.shape[0]}, "
        f"reconstructed_modes={controller.manifest.frequencies_hz.size}"
    )
    demo = make_demo(controller)
    demo.launch(server_name=args.host, server_port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare original and projected reconstructed modal spectra."
    )
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
