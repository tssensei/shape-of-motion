from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import cv2
import imageio.v3 as iio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow3d.modal_flow_coordinates import (
    _build_design_matrix,
    _load_and_validate_caches,
    _load_frame_map,
    _load_manifest,
    _view_pixel_groups,
)
from modal_peak_pick.core.cache import ModalAnalysisCache


MODAL_BASIS_DIAGNOSTIC_FORMAT = "modal_basis_capacity_diagnostics"
MODAL_BASIS_DIAGNOSTIC_VERSION = 1

SUMMARY_FILENAME = "capacity_summary.json"
DIAGNOSTICS_FILENAME = "capacity_diagnostics.npz"
PLOT_FILENAME = "capacity_curves.png"

_CANDIDATE_METHODS = ("pca_upper_bound", "direct_2d", "lifted_3d")
_FULL_MASK_METHODS = ("pca_upper_bound", "direct_2d")
_RIDGE_METHODS = ("direct_2d", "lifted_3d")
_QUANTILES = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
_FLOW_PIXEL_CHUNK_SIZE = 2048
_PCA_OVERSAMPLE = 16
_PCA_POWER_ITERATIONS = 1
_PCA_SEED = 7182026
_WARP_REFERENCE_RMSE_MAX = 0.05
_FRAME_ENERGY_RELATIVE_FLOOR = 1e-8

__all__ = [
    "DIAGNOSTICS_FILENAME",
    "MODAL_BASIS_DIAGNOSTIC_FORMAT",
    "MODAL_BASIS_DIAGNOSTIC_VERSION",
    "PLOT_FILENAME",
    "SUMMARY_FILENAME",
    "run_modal_basis_diagnostics",
]


@dataclass(frozen=True)
class ModalBasisDiagnosticResult:
    path: Path
    summary_path: Path
    diagnostics_path: Path
    plot_path: Path


@dataclass(frozen=True)
class _ModalView:
    path: Path
    mode_u: np.ndarray
    mode_v: np.ndarray
    export_provenance: dict[str, Any]


@dataclass(frozen=True)
class _Basis:
    raw: np.ndarray
    normalized: np.ndarray
    pair_scales: np.ndarray
    spatial_vectors: np.ndarray
    singular_values: np.ndarray
    numerical_rank: int
    pair_spatial_vectors: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _ProjectionEvaluation:
    energy_per_frame: np.ndarray
    flow_energy: float
    r2: np.ndarray
    sse: np.ndarray
    per_frame_r2_quantiles: np.ndarray
    coefficients: dict[str, np.ndarray]
    standalone_r2: dict[str, np.ndarray]
    ridge_r2: dict[str, float]
    ridge_sse: dict[str, float]


@dataclass(frozen=True)
class _PcaResult:
    spatial_vectors: np.ndarray
    singular_values: np.ndarray
    coefficients: np.ndarray
    numerical_rank: int
    energy_per_frame: np.ndarray
    flow_energy: float
    r2: np.ndarray
    sse: np.ndarray
    per_frame_r2_quantiles: np.ndarray


def _json_float(value: float) -> float | None:
    result = float(value)
    if not np.isfinite(result):
        return None
    return result


def _scalar_string(value: np.ndarray, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{path} {name} must be a scalar string")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str) or not item:
        raise ValueError(f"{path} {name} must be a non-empty scalar string")
    return item


def _scalar_text(value: np.ndarray, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{path} {name} must be a scalar string")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str):
        raise ValueError(f"{path} {name} must be a scalar string")
    return item


def _scalar_number(value: np.ndarray, name: str, path: Path) -> float:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{path} {name} must be a numeric scalar")
    result = float(array.item())
    if not np.isfinite(result):
        raise ValueError(f"{path} {name} must be finite")
    return result


def _parse_view_specs(
    specs: Sequence[str],
    expected_view_ids: Sequence[str],
    argument_name: str,
) -> tuple[Path, ...]:
    if not specs:
        raise ValueError(f"{argument_name} requires at least one VIEW_ID=PATH value")
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, str) or spec.count("=") != 1:
            raise ValueError(f"{argument_name} values must be VIEW_ID=PATH, got {spec!r}")
        view_id, raw_path = spec.split("=", 1)
        if not view_id or not raw_path:
            raise ValueError(f"{argument_name} values must be VIEW_ID=PATH, got {spec!r}")
        if view_id in seen:
            raise ValueError(f"{argument_name} contains duplicate view_id {view_id!r}")
        seen.add(view_id)
        parsed.append((view_id, Path(raw_path).expanduser()))
    parsed_ids = tuple(view_id for view_id, _ in parsed)
    expected = tuple(expected_view_ids)
    if parsed_ids != expected:
        raise ValueError(
            f"{argument_name} view IDs and order must be {expected}, got {parsed_ids}"
        )
    paths = tuple(path for _, path in parsed)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{argument_name} file does not exist: {path}")
    return paths


def _resolve_manifest_path(manifest_path: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{manifest_path} contains an invalid {name}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _load_modal_views(
    paths: Sequence[Path],
    manifest_path: Path,
    manifest: Any,
    caches: Sequence[ModalAnalysisCache],
) -> tuple[_ModalView, ...]:
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest_payload = json.load(file)
    source_modal_npzs = manifest_payload.get("source_modal_npzs")
    if not isinstance(source_modal_npzs, list) or len(source_modal_npzs) != len(paths):
        raise ValueError(f"{manifest_path} source_modal_npzs must match the view count")
    expected_paths = tuple(
        _resolve_manifest_path(manifest_path, value, "source_modal_npzs entry")
        for value in source_modal_npzs
    )
    resolved_paths = tuple(path.resolve() for path in paths)
    if resolved_paths != expected_paths:
        raise ValueError(
            "--modal-npzs must be the exact ordered modal inputs recorded by the manifest"
        )

    modal_views: list[_ModalView] = []
    for view_index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "mode_u",
                "mode_v",
                "selected_freqs_hz",
                "frequency_method",
                "fps",
                "mask",
                "reference_frame",
                "source_video",
                "source_mask",
                "t0",
                "t1",
                "resize",
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
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"{path} is missing required modal fields: {missing}")
            mode_u_all = np.asarray(archive["mode_u"])
            mode_v_all = np.asarray(archive["mode_v"])
            selected_frequencies = np.asarray(
                archive["selected_freqs_hz"], dtype=np.float32
            ).reshape(-1)
            frequency_method = _scalar_string(
                archive["frequency_method"], "frequency_method", path
            )
            fps_array = np.asarray(archive["fps"])
            modal_mask = np.asarray(archive["mask"])
            reference_frame = np.asarray(archive["reference_frame"], dtype=np.float32)
            source_video = _scalar_string(archive["source_video"], "source_video", path)
            source_mask = _scalar_text(archive["source_mask"], "source_mask", path)
            t0 = _scalar_number(archive["t0"], "t0", path)
            t1 = _scalar_number(archive["t1"], "t1", path)
            resize = int(_scalar_number(archive["resize"], "resize", path))
            flow_method = _scalar_string(archive["flow_method"], "flow_method", path)
            no_smooth_value = _scalar_number(archive["no_smooth"], "no_smooth", path)
            if no_smooth_value not in (0.0, 1.0):
                raise ValueError(f"{path} no_smooth must be boolean-valued")
            no_smooth = bool(no_smooth_value)
            sigma_b = _scalar_number(archive["sigma_b"], "sigma_b", path)
            sigma_c = _scalar_number(archive["sigma_c"], "sigma_c", path)
            t_ref_s = _scalar_number(archive["t_ref_s"], "t_ref_s", path)
            clamp_method = _scalar_string(
                archive["mode_amp_clamp_method"], "mode_amp_clamp_method", path
            )
            clamp_window = int(
                _scalar_number(archive["mode_amp_local_window"], "mode_amp_local_window", path)
            )
            clamp_ratio = _scalar_number(archive["mode_amp_ratio"], "mode_amp_ratio", path)
            clamp_percentile = _scalar_number(
                archive["mode_amp_global_percentile"],
                "mode_amp_global_percentile",
                path,
            )
            clamp_arrays = {
                name: np.asarray(archive[name], dtype=np.float32).reshape(-1)
                for name in (
                    "mode_amp_clamped_fraction",
                    "mode_amp_scale_min",
                    "mode_amp_original_p95",
                    "mode_amp_original_p99",
                )
            }

        if frequency_method != "exact_dft":
            raise ValueError(f"{path} frequency_method must be 'exact_dft'")
        if (
            mode_u_all.ndim != 3
            or mode_v_all.shape != mode_u_all.shape
            or not np.iscomplexobj(mode_u_all)
            or not np.iscomplexobj(mode_v_all)
        ):
            raise ValueError(f"{path} mode_u/mode_v must be matching complex [K,H,W] arrays")
        if not np.isfinite(mode_u_all.real).all() or not np.isfinite(mode_u_all.imag).all():
            raise ValueError(f"{path} mode_u contains non-finite values")
        if not np.isfinite(mode_v_all.real).all() or not np.isfinite(mode_v_all.imag).all():
            raise ValueError(f"{path} mode_v contains non-finite values")
        if selected_frequencies.shape != (mode_u_all.shape[0],):
            raise ValueError(f"{path} selected_freqs_hz does not match mode count")
        if not np.isfinite(selected_frequencies).all() or np.any(selected_frequencies <= 0.0):
            raise ValueError(f"{path} selected_freqs_hz must be finite and positive")
        if np.any(manifest.mode_indices < 0) or np.any(
            manifest.mode_indices >= mode_u_all.shape[0]
        ):
            raise ValueError(f"{path} does not contain every manifest mode_index")
        manifest_frequencies_at_modal_precision = manifest.frequencies_hz.astype(np.float32)
        if not np.array_equal(
            selected_frequencies[manifest.mode_indices],
            manifest_frequencies_at_modal_precision,
        ):
            raise ValueError(
                f"{path} selected frequencies at manifest mode_indices do not match the manifest"
            )
        expected_shape = cache_shape = caches[view_index].flow_u.shape[1:]
        if mode_u_all.shape[1:] != cache_shape:
            raise ValueError(
                f"{path} modal shape {mode_u_all.shape[1:]} does not match cache {cache_shape}"
            )
        if fps_array.shape != () or not np.isclose(
            float(fps_array.item()), caches[view_index].fps, rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"{path} FPS does not match its flow cache")
        cache_mask = caches[view_index].mask
        assert cache_mask is not None
        if modal_mask.shape != expected_shape or not np.array_equal(
            modal_mask.astype(bool), cache_mask
        ):
            raise ValueError(f"{path} mask does not match its flow cache mask")
        if reference_frame.shape != expected_shape or not np.array_equal(
            reference_frame, caches[view_index].reference_frame
        ):
            raise ValueError(f"{path} reference_frame does not match its flow cache")
        cache_metadata = caches[view_index].metadata
        cache_sources = cache_metadata["sources"]
        expected_source_mask = (
            "" if cache_sources["mask"] is None else str(cache_sources["mask"]["path"])
        )
        frame_range = cache_metadata["video"]["frame_range"]
        expected_t1 = -1.0 if frame_range["t1_s"] is None else float(frame_range["t1_s"])
        expected_resize = cache_metadata["video"]["resize_max_side"]
        expected_resize = -1 if expected_resize is None else int(expected_resize)
        smoothing = cache_metadata["analysis"]["smoothing"]
        if source_video != str(cache_sources["video"]["path"]):
            raise ValueError(f"{path} source_video does not match its flow cache provenance")
        if source_mask != expected_source_mask:
            raise ValueError(f"{path} source_mask does not match its flow cache provenance")
        scalar_comparisons = {
            "t0": (t0, float(frame_range["t0_s"])),
            "t1": (t1, expected_t1),
            "sigma_b": (sigma_b, float(smoothing["sigma_b"])),
            "sigma_c": (sigma_c, float(smoothing["sigma_c"])),
            "t_ref_s": (t_ref_s, caches[view_index].t_ref_s),
        }
        for name, (actual, expected) in scalar_comparisons.items():
            if np.float32(actual) != np.float32(expected):
                raise ValueError(f"{path} {name} does not match its flow cache")
        if resize != expected_resize:
            raise ValueError(f"{path} resize does not match its flow cache")
        if flow_method != str(cache_metadata["analysis"]["flow_method"]):
            raise ValueError(f"{path} flow_method does not match its flow cache")
        if no_smooth != bool(smoothing["disabled"]):
            raise ValueError(f"{path} no_smooth does not match its flow cache")
        if clamp_method not in {"none", "local-ratio"}:
            raise ValueError(f"{path} has unsupported mode_amp_clamp_method {clamp_method!r}")
        if clamp_window <= 0 or clamp_window % 2 == 0:
            raise ValueError(f"{path} mode_amp_local_window must be a positive odd integer")
        if clamp_ratio <= 0.0 or not (0.0 < clamp_percentile <= 100.0):
            raise ValueError(f"{path} has invalid amplitude-clamp parameters")
        for name, values in clamp_arrays.items():
            if values.shape != (mode_u_all.shape[0],) or not np.isfinite(values).all():
                raise ValueError(f"{path} {name} must be finite with one value per mode")
        modal_views.append(
            _ModalView(
                path=path.resolve(),
                mode_u=mode_u_all[manifest.mode_indices].astype(np.complex64, copy=True),
                mode_v=mode_v_all[manifest.mode_indices].astype(np.complex64, copy=True),
                export_provenance={
                    "source_video": source_video,
                    "source_mask": source_mask,
                    "t0": t0,
                    "t1": t1,
                    "resize": resize,
                    "flow_method": flow_method,
                    "no_smooth": no_smooth,
                    "sigma_b": sigma_b,
                    "sigma_c": sigma_c,
                    "t_ref_s": t_ref_s,
                    "mode_amp_clamp_method": clamp_method,
                    "mode_amp_local_window": clamp_window,
                    "mode_amp_ratio": clamp_ratio,
                    "mode_amp_global_percentile": clamp_percentile,
                    **{
                        name: values[manifest.mode_indices].astype(float).tolist()
                        for name, values in clamp_arrays.items()
                    },
                },
            )
        )
    return tuple(modal_views)


def _validate_observation_modal_values(
    manifest_path: Path,
    manifest: Any,
    modal_views: Sequence[_ModalView],
) -> None:
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    raw_modes = payload.get("modes")
    if not isinstance(raw_modes, list) or len(raw_modes) != manifest.mode_indices.size:
        raise ValueError(f"{manifest_path} modes do not match the loaded manifest")
    topology = manifest.topology
    for mode_slot, raw_mode in enumerate(raw_modes):
        if not isinstance(raw_mode, dict):
            raise ValueError(f"{manifest_path} mode entries must be objects")
        observation_path = _resolve_manifest_path(
            manifest_path, raw_mode.get("observation_path"), "observation_path"
        )
        with np.load(observation_path, allow_pickle=False) as archive:
            if "obs_y" not in archive.files:
                raise ValueError(f"{observation_path} is missing obs_y")
            obs_y = np.asarray(archive["obs_y"])
        if obs_y.shape != (topology.obs_point_index.size, 2) or not np.iscomplexobj(obs_y):
            raise ValueError(f"{observation_path} obs_y must be complex [O,2]")
        if not np.isfinite(obs_y.real).all() or not np.isfinite(obs_y.imag).all():
            raise ValueError(f"{observation_path} obs_y contains non-finite values")
        for view_index, modal in enumerate(modal_views):
            rows = np.flatnonzero(topology.obs_view_index == view_index)
            pixels = topology.obs_pixels_xy[rows]
            expected = np.stack(
                [
                    modal.mode_u[mode_slot, pixels[:, 1], pixels[:, 0]],
                    modal.mode_v[mode_slot, pixels[:, 1], pixels[:, 0]],
                ],
                axis=1,
            )
            if not np.array_equal(obs_y[rows], expected):
                raise ValueError(
                    f"{observation_path} obs_y does not match {modal.path} for "
                    f"view {topology.view_ids[view_index]!r}"
                )


def _validate_dynamic_dataset(
    dynamic_data_dir: Path,
    frame_map: Any,
    caches: Sequence[ModalAnalysisCache],
) -> dict[str, Any]:
    dataset_frame_map_path = dynamic_data_dir / "modal_frame_map.json"
    metadata_path = dynamic_data_dir / "metadata.json"
    if not dataset_frame_map_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Dynamic dataset is missing modal_frame_map.json or metadata.json: {dynamic_data_dir}"
        )
    dataset_frame_map = _load_frame_map(dataset_frame_map_path)
    frame_map_fields = (
        "fps_hz",
        "frame_names",
        "frame_view_indices",
        "frame_local_indices",
        "frame_times_sec",
    )
    if dataset_frame_map.view_ids != frame_map.view_ids:
        raise ValueError("Dynamic dataset view order does not match --modal-frame-map")
    for field in frame_map_fields:
        if not np.array_equal(
            np.asarray(getattr(dataset_frame_map, field)),
            np.asarray(getattr(frame_map, field)),
        ):
            raise ValueError(
                f"Dynamic dataset {field} does not match --modal-frame-map"
            )
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("format") != "modal_dynamic_dataset" or metadata.get("version") != 1:
        raise ValueError(f"{metadata_path} has unsupported format/version")
    if metadata.get("view_count") != len(frame_map.view_ids):
        raise ValueError(f"{metadata_path} view_count does not match frame map")
    if metadata.get("frame_count") != len(frame_map.frame_names):
        raise ValueError(f"{metadata_path} frame_count does not match frame map")
    target = metadata.get("target_resolution")
    if not isinstance(target, dict):
        raise ValueError(f"{metadata_path} is missing target_resolution")
    target_shape = (int(target.get("height", 0)), int(target.get("width", 0)))
    if target_shape[0] <= 0 or target_shape[1] <= 0:
        raise ValueError(f"{metadata_path} target_resolution is invalid")
    for view_index, cache in enumerate(caches):
        if cache.flow_u.shape[1:] != target_shape:
            raise ValueError(
                f"Dynamic target resolution does not match flow cache for "
                f"{frame_map.view_ids[view_index]!r}"
            )
    metadata_views = metadata.get("views")
    if not isinstance(metadata_views, list) or len(metadata_views) != len(frame_map.view_ids):
        raise ValueError(f"{metadata_path} views do not match frame map")
    if [view.get("view_id") if isinstance(view, dict) else None for view in metadata_views] != list(
        frame_map.view_ids
    ):
        raise ValueError(f"{metadata_path} view order does not match frame map")
    for view_index, view in enumerate(metadata_views):
        expected_count = int(np.count_nonzero(frame_map.frame_view_indices == view_index))
        if view.get("frame_count") != expected_count:
            raise ValueError(f"{metadata_path} has an invalid frame count for a view")
        if np.float64(view.get("fps_hz")) != frame_map.fps_hz[view_index]:
            raise ValueError(f"{metadata_path} has an invalid FPS for a view")
    return metadata


def _sampled_mask_pixels(mask: np.ndarray, stride: int) -> np.ndarray:
    y, x = np.nonzero(mask)
    height, width = mask.shape
    selected = (
        (y >= 1)
        & (y < height - 1)
        & (x >= 1)
        & (x < width - 1)
        & ((y - 1) % stride == 0)
        & ((x - 1) % stride == 0)
    )
    pixels = np.stack([x[selected], y[selected]], axis=1).astype(np.int64)
    if pixels.shape[0] == 0:
        raise ValueError("Stride-sampled analysis mask contains no pixels")
    return pixels


def _direct_design(modal: _ModalView, pixels: np.ndarray) -> np.ndarray:
    num_pixels = pixels.shape[0]
    num_modes = modal.mode_u.shape[0]
    x = pixels[:, 0]
    y = pixels[:, 1]
    u = modal.mode_u[:, y, x].T.astype(np.complex128)
    v = modal.mode_v[:, y, x].T.astype(np.complex128)
    design = np.empty((2 * num_pixels, 2 * num_modes), dtype=np.float64)
    design[0::2, 0::2] = u.real
    design[1::2, 0::2] = v.real
    design[0::2, 1::2] = -u.imag
    design[1::2, 1::2] = -v.imag
    if not np.isfinite(design).all():
        raise ValueError(f"Direct modal design from {modal.path} is non-finite")
    return design


def _make_basis(raw: np.ndarray, num_modes: int) -> _Basis:
    if raw.ndim != 2 or raw.shape[1] != 2 * num_modes or raw.shape[0] == 0:
        raise ValueError(f"Modal design must have shape [2P,{2 * num_modes}]")
    pair_scales = np.empty((num_modes,), dtype=np.float64)
    normalized = raw.copy()
    denominator = float(raw.shape[0])
    for mode_slot in range(num_modes):
        pair = raw[:, 2 * mode_slot : 2 * mode_slot + 2]
        scale = np.sqrt(float(np.sum(pair * pair)) / denominator)
        if not np.isfinite(scale):
            raise ValueError(f"Mode pair {mode_slot} has a non-finite scale")
        if scale <= np.finfo(np.float64).eps:
            scale = 1.0
        pair_scales[mode_slot] = scale
        normalized[:, 2 * mode_slot : 2 * mode_slot + 2] /= scale

    spatial_vectors, singular_values, _ = np.linalg.svd(normalized, full_matrices=False)
    tolerance = (
        np.finfo(np.float64).eps
        * max(normalized.shape)
        * float(singular_values[0])
    )
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    spatial_vectors = spatial_vectors[:, :numerical_rank]

    pair_vectors: list[np.ndarray] = []
    for mode_slot in range(num_modes):
        pair = normalized[:, 2 * mode_slot : 2 * mode_slot + 2]
        pair_u, pair_s, _ = np.linalg.svd(pair, full_matrices=False)
        pair_tolerance = (
            np.finfo(np.float64).eps * max(pair.shape) * float(pair_s[0])
        )
        pair_rank = int(np.count_nonzero(pair_s > pair_tolerance))
        pair_vectors.append(pair_u[:, :pair_rank])
    return _Basis(
        raw=raw,
        normalized=normalized,
        pair_scales=pair_scales,
        spatial_vectors=spatial_vectors,
        singular_values=singular_values / np.sqrt(denominator),
        numerical_rank=numerical_rank,
        pair_spatial_vectors=tuple(pair_vectors),
    )


def _flow_feature_chunks(
    cache: ModalAnalysisCache,
    pixels: np.ndarray,
    reference_index: int,
):
    num_pixels = pixels.shape[0]
    for pixel_start in range(0, num_pixels, _FLOW_PIXEL_CHUNK_SIZE):
        pixel_end = min(pixel_start + _FLOW_PIXEL_CHUNK_SIZE, num_pixels)
        current = pixels[pixel_start:pixel_end]
        x = current[:, 0]
        y = current[:, 1]
        flow_u = np.asarray(cache.flow_u[:, y, x], dtype=np.float64)
        flow_v = np.asarray(cache.flow_v[:, y, x], dtype=np.float64)
        if not np.isfinite(flow_u).all() or not np.isfinite(flow_v).all():
            raise ValueError(f"Flow cache {cache.path} contains non-finite sampled flow")
        reference_u = flow_u[reference_index].copy()
        reference_v = flow_v[reference_index].copy()
        flow_u -= reference_u[None, :]
        flow_v -= reference_v[None, :]
        features = np.empty(
            (2 * (pixel_end - pixel_start), cache.flow_u.shape[0]),
            dtype=np.float64,
        )
        features[0::2] = flow_u.T
        features[1::2] = flow_v.T
        yield slice(2 * pixel_start, 2 * pixel_end), features


def _frame_energy_floor(energy_per_frame: np.ndarray) -> float:
    return max(
        np.finfo(np.float64).eps,
        float(np.mean(energy_per_frame)) * _FRAME_ENERGY_RELATIVE_FLOOR,
    )


def _curve_from_coefficients(
    coefficients: np.ndarray,
    energy_per_frame: np.ndarray,
    max_real_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow_energy = float(np.sum(energy_per_frame))
    if flow_energy <= np.finfo(np.float64).eps:
        raise ValueError("Reference-relative sampled flow has zero total energy")
    if coefficients.ndim != 2 or coefficients.shape[1] != energy_per_frame.size:
        raise ValueError("Projection coefficients do not match frame count")
    captured = np.cumsum(coefficients * coefficients, axis=0)
    sse = np.empty((max_real_rank,), dtype=np.float64)
    quantiles = np.empty((max_real_rank, _QUANTILES.size), dtype=np.float64)
    energy_floor = _frame_energy_floor(energy_per_frame)
    positive = energy_per_frame > energy_floor
    if not np.any(positive):
        raise ValueError("No frames exceed the per-frame flow-energy diagnostic floor")
    for rank_index in range(max_real_rank):
        effective = min(rank_index + 1, coefficients.shape[0])
        if effective == 0:
            frame_sse = energy_per_frame.copy()
        else:
            frame_sse = energy_per_frame - captured[effective - 1]
        minimum = float(np.min(frame_sse))
        tolerance = 1e-9 * max(flow_energy, 1.0)
        if minimum < -tolerance:
            raise ValueError("Projection captured more energy than the sampled flow contains")
        sse[rank_index] = float(np.sum(frame_sse))
        frame_r2 = 1.0 - frame_sse[positive] / energy_per_frame[positive]
        quantiles[rank_index] = np.quantile(frame_r2, _QUANTILES)
    return 1.0 - sse / flow_energy, sse, quantiles


def _randomized_pca(
    cache: ModalAnalysisCache,
    pixels: np.ndarray,
    reference_index: int,
    max_real_rank: int,
    seed: int,
) -> _PcaResult:
    num_features = 2 * pixels.shape[0]
    num_frames = cache.flow_u.shape[0]
    sample_rank = min(max_real_rank + _PCA_OVERSAMPLE, num_features, num_frames)
    if sample_rank < max_real_rank:
        raise ValueError(
            f"PCA sample rank {sample_rank} is below requested real rank {max_real_rank}"
        )
    random = np.random.default_rng(seed)
    omega = random.standard_normal((num_frames, sample_rank), dtype=np.float64)
    projected = np.empty((num_features, sample_rank), dtype=np.float64)
    energy_per_frame = np.zeros((num_frames,), dtype=np.float64)
    for rows, flow in _flow_feature_chunks(cache, pixels, reference_index):
        projected[rows] = flow @ omega
        energy_per_frame += np.sum(flow * flow, axis=0)
    q, _ = np.linalg.qr(projected, mode="reduced")
    del projected
    for _ in range(_PCA_POWER_ITERATIONS):
        temporal = np.zeros((sample_rank, num_frames), dtype=np.float64)
        for rows, flow in _flow_feature_chunks(cache, pixels, reference_index):
            temporal += q[rows].T @ flow
        powered = np.empty((num_features, sample_rank), dtype=np.float64)
        for rows, flow in _flow_feature_chunks(cache, pixels, reference_index):
            powered[rows] = flow @ temporal.T
        q, _ = np.linalg.qr(powered, mode="reduced")
        del powered, temporal
    small = np.zeros((sample_rank, num_frames), dtype=np.float64)
    for rows, flow in _flow_feature_chunks(cache, pixels, reference_index):
        small += q[rows].T @ flow
    small_u, singular_values, right_vectors = np.linalg.svd(small, full_matrices=False)
    singular_values = singular_values[:max_real_rank]
    spatial_vectors = (q @ small_u[:, :max_real_rank]).astype(np.float64, copy=False)
    right_vectors = right_vectors[:max_real_rank]
    tolerance = (
        np.finfo(np.float64).eps
        * max(num_features, num_frames)
        * float(singular_values[0])
    )
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    spatial_vectors = spatial_vectors[:, :numerical_rank]
    coefficients = (
        singular_values[:numerical_rank, None] * right_vectors[:numerical_rank]
    )
    r2, sse, quantiles = _curve_from_coefficients(
        coefficients, energy_per_frame, max_real_rank
    )
    return _PcaResult(
        spatial_vectors=spatial_vectors,
        singular_values=singular_values,
        coefficients=coefficients,
        numerical_rank=numerical_rank,
        energy_per_frame=energy_per_frame,
        flow_energy=float(np.sum(energy_per_frame)),
        r2=r2,
        sse=sse,
        per_frame_r2_quantiles=quantiles,
    )


def _ridge_solution(
    basis: _Basis,
    normalized_cross_flow: np.ndarray,
    reference_index: int,
    ridge_relative: float,
) -> np.ndarray:
    normalizer = float(basis.raw.shape[0])
    gram = basis.normalized.T @ basis.normalized / normalizer
    system = gram + ridge_relative * np.eye(gram.shape[0], dtype=np.float64)
    cholesky = np.linalg.cholesky(system)
    rhs = normalized_cross_flow / normalizer
    scaled = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, rhs))
    physical = scaled.copy()
    for mode_slot, scale in enumerate(basis.pair_scales):
        physical[2 * mode_slot : 2 * mode_slot + 2] /= scale
    coordinates = physical[0::2].T + 1j * physical[1::2].T
    coordinates -= coordinates[reference_index : reference_index + 1]
    coordinates -= np.mean(coordinates, axis=0, keepdims=True)
    coordinates = coordinates.astype(np.complex64).astype(np.complex128)
    packed = np.empty_like(physical)
    packed[0::2] = coordinates.real.T
    packed[1::2] = coordinates.imag.T
    packed -= packed[:, reference_index : reference_index + 1]
    return packed


def _evaluate_bases(
    cache: ModalAnalysisCache,
    pixels: np.ndarray,
    reference_index: int,
    bases: Mapping[str, _Basis],
    max_real_rank: int,
    ridge_relative: float,
    expected_energy_per_frame: np.ndarray,
    include_ridge: bool,
) -> _ProjectionEvaluation:
    num_frames = cache.flow_u.shape[0]
    coefficients = {
        name: np.zeros((basis.numerical_rank, num_frames), dtype=np.float64)
        for name, basis in bases.items()
    }
    standalone_coefficients = {
        name: [
            np.zeros((vectors.shape[1], num_frames), dtype=np.float64)
            for vectors in basis.pair_spatial_vectors
        ]
        for name, basis in bases.items()
    }
    ridge_cross = {
        name: np.zeros((basis.normalized.shape[1], num_frames), dtype=np.float64)
        for name, basis in bases.items()
    }
    energy_per_frame = np.zeros((num_frames,), dtype=np.float64)
    for rows, flow in _flow_feature_chunks(cache, pixels, reference_index):
        energy_per_frame += np.sum(flow * flow, axis=0)
        for name, basis in bases.items():
            if basis.numerical_rank:
                coefficients[name] += basis.spatial_vectors[rows].T @ flow
            for mode_slot, vectors in enumerate(basis.pair_spatial_vectors):
                if vectors.shape[1]:
                    standalone_coefficients[name][mode_slot] += vectors[rows].T @ flow
            if include_ridge:
                ridge_cross[name] += basis.normalized[rows].T @ flow
    if not np.allclose(
        energy_per_frame, expected_energy_per_frame, rtol=1e-10, atol=1e-8
    ):
        raise ValueError("Repeated flow pass changed sampled reference-relative energy")

    r2_values: list[np.ndarray] = []
    sse_values: list[np.ndarray] = []
    quantile_values: list[np.ndarray] = []
    standalone: dict[str, np.ndarray] = {}
    ridge_r2: dict[str, float] = {}
    ridge_sse: dict[str, float] = {}
    ridge_solutions: dict[str, np.ndarray] = {}
    flow_energy = float(np.sum(energy_per_frame))
    for name, basis in bases.items():
        r2, sse, quantiles = _curve_from_coefficients(
            coefficients[name], energy_per_frame, max_real_rank
        )
        r2_values.append(r2)
        sse_values.append(sse)
        quantile_values.append(quantiles)
        standalone_r2 = np.empty((len(basis.pair_spatial_vectors),), dtype=np.float64)
        for mode_slot, mode_coefficients in enumerate(standalone_coefficients[name]):
            captured = float(np.sum(mode_coefficients * mode_coefficients))
            standalone_r2[mode_slot] = captured / flow_energy
        standalone[name] = standalone_r2
        if include_ridge:
            ridge_solutions[name] = _ridge_solution(
                basis,
                ridge_cross[name],
                reference_index,
                ridge_relative,
            )
    if include_ridge:
        ridge_frame_sse = {
            name: np.zeros((num_frames,), dtype=np.float64) for name in bases
        }
        for rows, flow in _flow_feature_chunks(cache, pixels, reference_index):
            for name, basis in bases.items():
                residual = basis.raw[rows] @ ridge_solutions[name] - flow
                ridge_frame_sse[name] += np.sum(residual * residual, axis=0)
        for name in bases:
            ridge_sse[name] = float(np.sum(ridge_frame_sse[name]))
            ridge_r2[name] = float(1.0 - ridge_sse[name] / flow_energy)
    return _ProjectionEvaluation(
        energy_per_frame=energy_per_frame,
        flow_energy=flow_energy,
        r2=np.stack(r2_values, axis=0),
        sse=np.stack(sse_values, axis=0),
        per_frame_r2_quantiles=np.stack(quantile_values, axis=0),
        coefficients=coefficients,
        standalone_r2=standalone,
        ridge_r2=ridge_r2,
        ridge_sse=ridge_sse,
    )


def _regional_metrics(
    cache: ModalAnalysisCache,
    pixels: np.ndarray,
    reference_index: int,
    spatial_vectors: Mapping[str, np.ndarray],
    coefficients: Mapping[str, np.ndarray],
    grid_rows: int,
    grid_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = cache.flow_u.shape[1:]
    pixel_regions = (
        np.minimum(pixels[:, 1] * grid_rows // height, grid_rows - 1) * grid_cols
        + np.minimum(pixels[:, 0] * grid_cols // width, grid_cols - 1)
    ).astype(np.int64)
    num_regions = grid_rows * grid_cols
    pixel_count = np.bincount(pixel_regions, minlength=num_regions).astype(np.int64)
    energy = np.zeros((num_regions,), dtype=np.float64)
    residual = {
        name: np.zeros((num_regions,), dtype=np.float64) for name in spatial_vectors
    }
    for rows, flow in _flow_feature_chunks(cache, pixels, reference_index):
        pixel_start = rows.start // 2
        pixel_end = rows.stop // 2
        feature_regions = np.repeat(pixel_regions[pixel_start:pixel_end], 2)
        energy += np.bincount(
            feature_regions,
            weights=np.sum(flow * flow, axis=1),
            minlength=num_regions,
        )
        for name, vectors in spatial_vectors.items():
            prediction = vectors[rows] @ coefficients[name]
            error = prediction - flow
            residual[name] += np.bincount(
                feature_regions,
                weights=np.sum(error * error, axis=1),
                minlength=num_regions,
            )
    valid = (pixel_count > 0) & (energy > np.finfo(np.float64).eps)
    r2 = np.zeros((len(spatial_vectors), num_regions), dtype=np.float64)
    residual_array = np.stack([residual[name] for name in spatial_vectors], axis=0)
    r2[:, valid] = 1.0 - residual_array[:, valid] / energy[None, valid]
    return r2, residual_array, energy, valid


def _load_gray_image(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Dynamic RGB frame does not exist: {path}")
    image = np.asarray(iio.imread(path))
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"{path} must contain a grayscale, RGB, or RGBA image")
    if gray.shape != expected_shape:
        raise ValueError(f"{path} shape {gray.shape} does not match {expected_shape}")
    if np.issubdtype(gray.dtype, np.integer):
        maximum = float(np.iinfo(gray.dtype).max)
        result = gray.astype(np.float32) / maximum
    else:
        result = gray.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{path} contains non-finite image values")
    return result


def _warp_diagnostics(
    cache: ModalAnalysisCache,
    frame_map: Any,
    view_index: int,
    dynamic_data_dir: Path,
    frame_count: int,
) -> dict[str, Any]:
    num_frames, height, width = cache.flow_u.shape
    reference_index = int(cache.metadata["analysis"]["reference_frame_index"])
    if num_frames - 1 < frame_count:
        raise ValueError(
            f"View {frame_map.view_ids[view_index]!r} has only {num_frames - 1} "
            f"non-reference frames, below --warp-frame-count={frame_count}"
        )
    rows = np.flatnonzero(frame_map.frame_view_indices == view_index)
    local_indices = frame_map.frame_local_indices[rows]
    names_by_local = np.empty((num_frames,), dtype=object)
    for row, local_index in zip(rows, local_indices):
        names_by_local[int(local_index)] = frame_map.frame_names[int(row)]
    if any(not isinstance(name, str) for name in names_by_local):
        raise ValueError(f"Frame map is incomplete for view {frame_map.view_ids[view_index]!r}")

    image_dir = dynamic_data_dir / "images"
    reference_path = image_dir / f"{names_by_local[reference_index]}.png"
    dynamic_reference = _load_gray_image(reference_path, (height, width))
    cache_reference = np.asarray(cache.reference_frame, dtype=np.float32)
    cache_mask = cache.mask
    assert cache_mask is not None
    reference_rmse = float(
        np.sqrt(np.mean((dynamic_reference[cache_mask] - cache_reference[cache_mask]) ** 2))
    )
    if reference_rmse > _WARP_REFERENCE_RMSE_MAX:
        raise ValueError(
            f"Dynamic reference image for view {frame_map.view_ids[view_index]!r} "
            f"does not match the flow cache (RMSE={reference_rmse:.6g})"
        )

    candidates = np.delete(np.arange(num_frames, dtype=np.int64), reference_index)
    positions = np.rint(np.linspace(0, candidates.size - 1, frame_count)).astype(np.int64)
    sampled = candidates[positions]
    if np.unique(sampled).size != frame_count:
        raise ValueError("Deterministic warp-frame sampling produced duplicate frames")
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    warp_mae = np.empty((frame_count,), dtype=np.float64)
    zero_mae = np.empty((frame_count,), dtype=np.float64)
    support_fraction = np.empty((frame_count,), dtype=np.float64)
    for sample_index, local_index in enumerate(sampled):
        current_path = image_dir / f"{names_by_local[int(local_index)]}.png"
        current = _load_gray_image(current_path, (height, width))
        flow_u = np.asarray(cache.flow_u[local_index], dtype=np.float32)
        flow_v = np.asarray(cache.flow_v[local_index], dtype=np.float32)
        map_x = grid_x + flow_u
        map_y = grid_y + flow_v
        in_bounds = (
            (map_x >= 0.0)
            & (map_x <= width - 1)
            & (map_y >= 0.0)
            & (map_y <= height - 1)
        )
        support = cache_mask & in_bounds
        if not np.any(support):
            raise ValueError(
                f"Warp support is empty for view {frame_map.view_ids[view_index]!r} "
                f"frame {int(local_index)}"
            )
        warped = cv2.remap(
            current,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        warp_mae[sample_index] = float(
            np.mean(np.abs(warped[support] - cache_reference[support]))
        )
        zero_mae[sample_index] = float(
            np.mean(np.abs(current[support] - cache_reference[support]))
        )
        support_fraction[sample_index] = float(np.mean(support))
    aggregate_warp = float(np.mean(warp_mae))
    aggregate_zero = float(np.mean(zero_mae))
    if aggregate_zero <= np.finfo(np.float64).eps:
        raise ValueError(
            f"Zero-flow photometric error is zero for view {frame_map.view_ids[view_index]!r}"
        )
    ratio = aggregate_warp / aggregate_zero
    return {
        "local_indices": sampled,
        "warp_mae": warp_mae,
        "zero_flow_mae": zero_mae,
        "support_fraction": support_fraction,
        "mean_warp_mae": aggregate_warp,
        "mean_zero_flow_mae": aggregate_zero,
        "warp_to_zero_mae_ratio": ratio,
        "reference_image_rmse": reference_rmse,
        "reference_flow_rms": float(
            np.sqrt(
                np.mean(
                    np.asarray(cache.flow_u[reference_index], dtype=np.float64) ** 2
                    + np.asarray(cache.flow_v[reference_index], dtype=np.float64) ** 2
                )
            )
        ),
    }


def _pad_singular_values(values: np.ndarray, size: int) -> np.ndarray:
    result = np.zeros((size,), dtype=np.float64)
    count = min(size, values.size)
    result[:count] = values[:count]
    return result


def _aggregate_curves(
    sse: np.ndarray,
    energy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    view_r2 = 1.0 - sse / energy[:, None, None]
    pooled = 1.0 - np.sum(sse, axis=0) / float(np.sum(energy))
    macro = np.mean(view_r2, axis=0)
    worst = np.min(view_r2, axis=0)
    std = np.std(view_r2, axis=0)
    return pooled, macro, worst, std


def _plot_curves(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    ranks = arrays["real_ranks"]
    candidate = arrays["candidate_pooled_r2"]
    full = arrays["full_pooled_r2"]
    view_ids = [str(value) for value in arrays["view_ids"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {
        "pca_upper_bound": "black",
        "direct_2d": "tab:blue",
        "lifted_3d": "tab:orange",
    }
    for method_index, method in enumerate(_CANDIDATE_METHODS):
        axes[0, 0].plot(ranks, candidate[method_index], label=method, color=colors[method])
    axes[0, 0].set_title("Common candidate ROI: pooled flow R2")
    axes[0, 0].set_xlabel("real rank")
    axes[0, 0].set_ylabel("reference-relative zero-baseline R2")
    axes[0, 0].axhline(0.0, color="0.7", linewidth=1)
    axes[0, 0].legend()

    for method_index, method in enumerate(_FULL_MASK_METHODS):
        axes[0, 1].plot(ranks, full[method_index], label=method, color=colors[method])
    axes[0, 1].set_title("Stride-sampled full analysis mask: pooled flow R2")
    axes[0, 1].set_xlabel("real rank")
    axes[0, 1].set_ylabel("reference-relative zero-baseline R2")
    axes[0, 1].axhline(0.0, color="0.7", linewidth=1)
    axes[0, 1].legend()

    endpoints = arrays["candidate_view_r2"][:, :, -1]
    x = np.arange(len(view_ids), dtype=np.float64)
    width = 0.24
    for method_index, method in enumerate(_CANDIDATE_METHODS):
        axes[1, 0].bar(
            x + (method_index - 1) * width,
            endpoints[:, method_index],
            width=width,
            label=method,
            color=colors[method],
        )
    axes[1, 0].set_title(f"Candidate ROI endpoints at real rank {int(ranks[-1])}")
    axes[1, 0].set_xticks(x, view_ids)
    axes[1, 0].set_ylabel("R2")
    axes[1, 0].axhline(0.0, color="0.7", linewidth=1)
    axes[1, 0].legend()

    ratios = arrays["warp_to_zero_mae_ratio"]
    axes[1, 1].bar(view_ids, ratios, color="tab:green")
    axes[1, 1].axhline(1.0, color="tab:red", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Farneback photometric warp / zero-flow MAE")
    axes[1, 1].set_ylabel("ratio (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    source: Mapping[str, Any],
    settings: Mapping[str, Any],
    timings: Mapping[str, Any],
) -> None:
    ranks = arrays["real_ranks"]
    final_rank_index = ranks.size - 1
    views: list[dict[str, Any]] = []
    for view_index, raw_view_id in enumerate(arrays["view_ids"]):
        view_id = str(raw_view_id)
        regional_values: list[dict[str, Any]] = []
        for region_index in range(arrays["regional_valid"].shape[1]):
            valid = bool(arrays["regional_valid"][view_index, region_index])
            regional_values.append(
                {
                    "region_index": region_index,
                    "pixel_count": int(arrays["regional_pixel_count"][view_index, region_index]),
                    "flow_energy": float(arrays["regional_flow_energy"][view_index, region_index]),
                    "r2": {
                        method: (
                            float(arrays["regional_r2"][view_index, method_index, region_index])
                            if valid
                            else None
                        )
                        for method_index, method in enumerate(_CANDIDATE_METHODS)
                    },
                }
            )
        views.append(
            {
                "view_id": view_id,
                "candidate_pixel_count": int(arrays["candidate_pixel_count"][view_index]),
                "full_sampled_pixel_count": int(
                    arrays["full_sampled_pixel_count"][view_index]
                ),
                "full_mask_pixel_count": int(arrays["full_mask_pixel_count"][view_index]),
                "candidate_pixel_fraction_of_stride_sampled_mask": float(
                    arrays["candidate_pixel_fraction"][view_index]
                ),
                "candidate_to_full_flow_energy_density_ratio": float(
                    arrays["candidate_flow_energy_density_ratio"][view_index]
                ),
                "per_frame_energy_diagnostics": {
                    "candidate_floor": float(
                        arrays["candidate_frame_energy_floor"][view_index]
                    ),
                    "candidate_excluded_frame_count": int(
                        arrays["candidate_low_energy_frame_count"][view_index]
                    ),
                    "full_mask_floor": float(arrays["full_frame_energy_floor"][view_index]),
                    "full_mask_excluded_frame_count": int(
                        arrays["full_low_energy_frame_count"][view_index]
                    ),
                },
                "candidate_endpoint_r2": {
                    method: float(
                        arrays["candidate_view_r2"][view_index, method_index, final_rank_index]
                    )
                    for method_index, method in enumerate(_CANDIDATE_METHODS)
                },
                "full_mask_endpoint_r2": {
                    method: float(
                        arrays["full_view_r2"][view_index, method_index, final_rank_index]
                    )
                    for method_index, method in enumerate(_FULL_MASK_METHODS)
                },
                "basis_numerical_rank_candidate": {
                    method: int(arrays["candidate_basis_numerical_rank"][view_index, method_index])
                    for method_index, method in enumerate(_CANDIDATE_METHODS)
                },
                "standalone_mode_r2": {
                    method: arrays["standalone_mode_r2"][view_index, method_index].tolist()
                    for method_index, method in enumerate(_RIDGE_METHODS)
                },
                "pair_normalized_ridge_r2": {
                    method: float(arrays["ridge_view_r2"][view_index, method_index])
                    for method_index, method in enumerate(_RIDGE_METHODS)
                },
                "warp": {
                    "mean_warp_mae": float(arrays["warp_mean_mae"][view_index]),
                    "mean_zero_flow_mae": float(arrays["warp_zero_mean_mae"][view_index]),
                    "warp_to_zero_mae_ratio": _json_float(
                        arrays["warp_to_zero_mae_ratio"][view_index]
                    ),
                    "reference_image_rmse": float(
                        arrays["warp_reference_image_rmse"][view_index]
                    ),
                    "reference_flow_rms": float(
                        arrays["warp_reference_flow_rms"][view_index]
                    ),
                },
                "regions": regional_values,
            }
        )
    payload = {
        "format": MODAL_BASIS_DIAGNOSTIC_FORMAT,
        "version": MODAL_BASIS_DIAGNOSTIC_VERSION,
        "metric_semantics": {
            "capacity_r2": "reference_relative_zero_motion_baseline_unclamped",
            "rank_axis": "real_rank; one complex mode contributes at most two real dimensions",
            "capacity_fit": "unregularized_orthogonal_projection",
            "solver_style_ridge_fit": (
                "pair_normalized_reference_flow_ridge_with_runtime_gauge; lifted_3d uses "
                "the production design while direct_2d is a hypothetical comparison"
            ),
            "candidate_roi": "unique_pixel_candidate_observation_pixels",
            "full_mask_roi": "analysis_mask_sampled_on_integer_stride_grid",
            "regional_fit": "global_candidate_fit_then_region_only_residual_evaluation",
            "warp_direction": "sample_current_at_reference_pixel_plus_reference_to_current_flow",
            "mode_pair_normalization": "shared_real_imag_pair_rms",
            "lifted_basis": "candidate_weighted_J_phi_without_alpha",
            "pca_method": "deterministic_randomized_spatial_range_finder_approximation",
            "pca_bound_validation": (
                "must dominate every compared fixed-basis curve at every reported real rank"
            ),
            "stored_singular_values": (
                "PCA entries are flow singular values; fixed-basis entries are normalized-design "
                "singular values divided by sqrt(row_count)"
            ),
        },
        "source": dict(source),
        "settings": dict(settings),
        "view_ids": [str(value) for value in arrays["view_ids"]],
        "mode_indices": arrays["mode_indices"].astype(int).tolist(),
        "frequencies_hz": arrays["frequencies_hz"].astype(float).tolist(),
        "real_ranks": ranks.astype(int).tolist(),
        "overall": {
            "candidate_pooled_r2": {
                method: arrays["candidate_pooled_r2"][method_index].tolist()
                for method_index, method in enumerate(_CANDIDATE_METHODS)
            },
            "full_mask_pooled_r2": {
                method: arrays["full_pooled_r2"][method_index].tolist()
                for method_index, method in enumerate(_FULL_MASK_METHODS)
            },
            "candidate_endpoint_gaps": {
                "pca_minus_direct_2d": float(
                    arrays["candidate_pooled_r2"][0, final_rank_index]
                    - arrays["candidate_pooled_r2"][1, final_rank_index]
                ),
                "direct_2d_minus_lifted_3d": float(
                    arrays["candidate_pooled_r2"][1, final_rank_index]
                    - arrays["candidate_pooled_r2"][2, final_rank_index]
                ),
            },
            "pair_normalized_ridge_pooled_r2": {
                method: float(arrays["ridge_pooled_r2"][method_index])
                for method_index, method in enumerate(_RIDGE_METHODS)
            },
            "standalone_mode_pooled_r2": {
                method: arrays["standalone_mode_pooled_r2"][method_index].tolist()
                for method_index, method in enumerate(_RIDGE_METHODS)
            },
        },
        "views": views,
        "timings_seconds": dict(timings),
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)


def _validate_output(
    directory: Path,
    num_views: int,
    num_modes: int,
    max_rank: int,
    num_regions: int,
    warp_frame_count: int,
) -> None:
    summary_path = directory / SUMMARY_FILENAME
    diagnostics_path = directory / DIAGNOSTICS_FILENAME
    plot_path = directory / PLOT_FILENAME
    if not summary_path.is_file() or not diagnostics_path.is_file() or not plot_path.is_file():
        raise FileNotFoundError(f"Diagnostic output is incomplete: {directory}")
    if plot_path.stat().st_size <= 0:
        raise ValueError(f"{plot_path} is empty")
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if (
        summary.get("format") != MODAL_BASIS_DIAGNOSTIC_FORMAT
        or summary.get("version") != MODAL_BASIS_DIAGNOSTIC_VERSION
    ):
        raise ValueError(f"{summary_path} has invalid format/version")
    if not isinstance(summary.get("views"), list) or len(summary["views"]) != num_views:
        raise ValueError(f"{summary_path} views do not match the expected count")
    with np.load(diagnostics_path, allow_pickle=False) as archive:
        required_shapes = {
            "candidate_flow_energy": (num_views,),
            "candidate_frame_energy_floor": (num_views,),
            "candidate_low_energy_frame_count": (num_views,),
            "candidate_view_r2": (num_views, len(_CANDIDATE_METHODS), max_rank),
            "candidate_view_sse": (num_views, len(_CANDIDATE_METHODS), max_rank),
            "candidate_pooled_r2": (len(_CANDIDATE_METHODS), max_rank),
            "full_flow_energy": (num_views,),
            "full_frame_energy_floor": (num_views,),
            "full_low_energy_frame_count": (num_views,),
            "full_view_r2": (num_views, len(_FULL_MASK_METHODS), max_rank),
            "full_view_sse": (num_views, len(_FULL_MASK_METHODS), max_rank),
            "full_pooled_r2": (len(_FULL_MASK_METHODS), max_rank),
            "standalone_mode_r2": (num_views, len(_RIDGE_METHODS), num_modes),
            "standalone_mode_pooled_r2": (len(_RIDGE_METHODS), num_modes),
            "ridge_view_r2": (num_views, len(_RIDGE_METHODS)),
            "ridge_view_sse": (num_views, len(_RIDGE_METHODS)),
            "ridge_pooled_r2": (len(_RIDGE_METHODS),),
            "regional_r2": (num_views, len(_CANDIDATE_METHODS), num_regions),
            "regional_sse": (num_views, len(_CANDIDATE_METHODS), num_regions),
            "regional_flow_energy": (num_views, num_regions),
            "regional_valid": (num_views, num_regions),
            "regional_pixel_count": (num_views, num_regions),
            "warp_local_indices": (num_views, warp_frame_count),
            "warp_mae": (num_views, warp_frame_count),
            "warp_zero_flow_mae": (num_views, warp_frame_count),
            "warp_support_fraction": (num_views, warp_frame_count),
            "warp_to_zero_mae_ratio": (num_views,),
        }
        for name, shape in required_shapes.items():
            if name not in archive.files or archive[name].shape != shape:
                raise ValueError(f"{diagnostics_path} {name} must have shape {shape}")
            if not np.isfinite(archive[name]).all():
                raise ValueError(f"{diagnostics_path} {name} must be finite")
        if _scalar_string(archive["format"], "format", diagnostics_path) != MODAL_BASIS_DIAGNOSTIC_FORMAT:
            raise ValueError(f"{diagnostics_path} has invalid format")
        if int(np.asarray(archive["version"]).item()) != MODAL_BASIS_DIAGNOSTIC_VERSION:
            raise ValueError(f"{diagnostics_path} has invalid version")
        if archive["view_ids"].shape != (num_views,):
            raise ValueError(f"{diagnostics_path} view_ids has an invalid shape")
        npz_view_ids = [str(value) for value in archive["view_ids"].tolist()]
        if summary.get("view_ids") != npz_view_ids or [
            view.get("view_id") if isinstance(view, dict) else None
            for view in summary["views"]
        ] != npz_view_ids:
            raise ValueError(f"{summary_path} view IDs do not match diagnostics")
        if archive["mode_indices"].shape != (num_modes,):
            raise ValueError(f"{diagnostics_path} mode_indices has an invalid shape")
        if archive["frequencies_hz"].shape != (num_modes,) or not np.isfinite(
            archive["frequencies_hz"]
        ).all():
            raise ValueError(f"{diagnostics_path} frequencies_hz is invalid")
        if not np.array_equal(archive["real_ranks"], np.arange(1, max_rank + 1)):
            raise ValueError(f"{diagnostics_path} real_ranks is invalid")
        method_expectations = {
            "candidate_method_names": _CANDIDATE_METHODS,
            "full_mask_method_names": _FULL_MASK_METHODS,
            "ridge_method_names": _RIDGE_METHODS,
        }
        for name, expected in method_expectations.items():
            actual = tuple(str(value) for value in archive[name].tolist())
            if actual != expected:
                raise ValueError(f"{diagnostics_path} {name} is invalid")
        candidate_energy = np.asarray(archive["candidate_flow_energy"], dtype=np.float64)
        full_energy = np.asarray(archive["full_flow_energy"], dtype=np.float64)
        if np.any(candidate_energy <= 0.0) or np.any(full_energy <= 0.0):
            raise ValueError(f"{diagnostics_path} sampled flow energy must be positive")
        expected_candidate_pooled = 1.0 - np.sum(
            archive["candidate_view_sse"], axis=0
        ) / float(np.sum(candidate_energy))
        expected_full_pooled = 1.0 - np.sum(
            archive["full_view_sse"], axis=0
        ) / float(np.sum(full_energy))
        expected_ridge_pooled = 1.0 - np.sum(
            archive["ridge_view_sse"], axis=0
        ) / float(np.sum(candidate_energy))
        for name, actual, expected in (
            ("candidate_pooled_r2", archive["candidate_pooled_r2"], expected_candidate_pooled),
            ("full_pooled_r2", archive["full_pooled_r2"], expected_full_pooled),
            ("ridge_pooled_r2", archive["ridge_pooled_r2"], expected_ridge_pooled),
        ):
            if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"{diagnostics_path} {name} is inconsistent with SSE/energy")


def run_modal_basis_diagnostics(
    *,
    flow_caches: Sequence[str],
    modal_npzs: Sequence[str],
    modal_manifest: str | Path,
    modal_frame_map: str | Path,
    dynamic_data_dir: str | Path,
    out_dir: str | Path,
    max_real_rank: int = 20,
    pixel_stride: int = 2,
    grid_rows: int = 4,
    grid_cols: int = 4,
    warp_frame_count: int = 32,
    production_ridge_relative: float = 1e-4,
) -> ModalBasisDiagnosticResult:
    if max_real_rank <= 0:
        raise ValueError("max_real_rank must be positive")
    if pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")
    if warp_frame_count <= 0:
        raise ValueError("warp_frame_count must be positive")
    if not np.isfinite(production_ridge_relative) or production_ridge_relative <= 0.0:
        raise ValueError("production_ridge_relative must be finite and positive")

    output_path = Path(out_dir).expanduser()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Diagnostic output directory already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_path = Path(dynamic_data_dir).expanduser()
    if not (dynamic_path / "images").is_dir():
        raise FileNotFoundError(f"Dynamic image directory does not exist: {dynamic_path / 'images'}")

    total_start = time.perf_counter()
    manifest_path = Path(modal_manifest).expanduser().resolve()
    frame_map = _load_frame_map(modal_frame_map)
    manifest = _load_manifest(manifest_path)
    num_modes = int(manifest.mode_indices.size)
    if max_real_rank > 2 * num_modes:
        raise ValueError(
            f"max_real_rank={max_real_rank} exceeds the K={num_modes} real limit {2 * num_modes}"
        )
    caches = _load_and_validate_caches(frame_map, manifest.topology, flow_caches)
    dynamic_metadata = _validate_dynamic_dataset(dynamic_path, frame_map, caches)
    modal_paths = _parse_view_specs(modal_npzs, frame_map.view_ids, "--modal-npzs")
    modal_views = _load_modal_views(modal_paths, manifest_path, manifest, caches)
    _validate_observation_modal_values(manifest_path, manifest, modal_views)
    input_seconds = float(time.perf_counter() - total_start)

    view_results: list[dict[str, Any]] = []
    for view_index, (cache, modal) in enumerate(zip(caches, modal_views)):
        view_start = time.perf_counter()
        reference_index = int(cache.metadata["analysis"]["reference_frame_index"])
        sorted_rows, starts, candidate_pixels, sorted_weights, _ = _view_pixel_groups(
            manifest.topology, view_index, cache
        )
        lifted_raw, lifted_pair_scales = _build_design_matrix(
            manifest, sorted_rows, starts, sorted_weights
        )
        direct_candidate_raw = _direct_design(modal, candidate_pixels)
        candidate_bases = {
            "direct_2d": _make_basis(direct_candidate_raw, num_modes),
            "lifted_3d": _make_basis(lifted_raw, num_modes),
        }
        if not np.allclose(
            candidate_bases["lifted_3d"].pair_scales,
            lifted_pair_scales,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Lifted mode-pair normalization differs from production construction")
        candidate_pca = _randomized_pca(
            cache,
            candidate_pixels,
            reference_index,
            max_real_rank,
            _PCA_SEED + 2 * view_index,
        )
        candidate_evaluation = _evaluate_bases(
            cache,
            candidate_pixels,
            reference_index,
            candidate_bases,
            max_real_rank,
            production_ridge_relative,
            candidate_pca.energy_per_frame,
            include_ridge=True,
        )
        pca_tolerance = 1e-6
        if np.any(
            candidate_evaluation.r2
            > candidate_pca.r2[None, :] + pca_tolerance
        ):
            raise ValueError(
                f"Randomized PCA did not dominate the fixed candidate bases for view "
                f"{frame_map.view_ids[view_index]!r}; increase PCA accuracy before publishing"
            )

        full_mask = cache.mask
        assert full_mask is not None
        full_pixels = _sampled_mask_pixels(full_mask, pixel_stride)
        full_sampled_mask = np.zeros_like(full_mask, dtype=bool)
        full_sampled_mask[full_pixels[:, 1], full_pixels[:, 0]] = True
        if not bool(
            np.all(full_sampled_mask[candidate_pixels[:, 1], candidate_pixels[:, 0]])
        ):
            raise ValueError(
                f"Candidate pixels for view {frame_map.view_ids[view_index]!r} do not "
                f"lie on the --pixel-stride={pixel_stride} analysis-mask sampling grid"
            )
        direct_full_basis = _make_basis(_direct_design(modal, full_pixels), num_modes)
        full_pca = _randomized_pca(
            cache,
            full_pixels,
            reference_index,
            max_real_rank,
            _PCA_SEED + 2 * view_index + 1,
        )
        full_evaluation = _evaluate_bases(
            cache,
            full_pixels,
            reference_index,
            {"direct_2d": direct_full_basis},
            max_real_rank,
            production_ridge_relative,
            full_pca.energy_per_frame,
            include_ridge=False,
        )
        if np.any(full_evaluation.r2 > full_pca.r2[None, :] + pca_tolerance):
            raise ValueError(
                f"Randomized PCA did not dominate the direct full-mask basis for view "
                f"{frame_map.view_ids[view_index]!r}; increase PCA accuracy before publishing"
            )

        regional_vectors = {
            "pca_upper_bound": candidate_pca.spatial_vectors,
            "direct_2d": candidate_bases["direct_2d"].spatial_vectors[
                :, :max_real_rank
            ],
            "lifted_3d": candidate_bases["lifted_3d"].spatial_vectors[
                :, :max_real_rank
            ],
        }
        regional_coefficients = {
            "pca_upper_bound": candidate_pca.coefficients,
            "direct_2d": candidate_evaluation.coefficients["direct_2d"][
                :max_real_rank
            ],
            "lifted_3d": candidate_evaluation.coefficients["lifted_3d"][
                :max_real_rank
            ],
        }
        regional_r2, regional_sse, regional_energy, regional_valid = _regional_metrics(
            cache,
            candidate_pixels,
            reference_index,
            regional_vectors,
            regional_coefficients,
            grid_rows,
            grid_cols,
        )
        warp = _warp_diagnostics(
            cache,
            frame_map,
            view_index,
            dynamic_path,
            warp_frame_count,
        )
        candidate_r2 = np.concatenate(
            [candidate_pca.r2[None], candidate_evaluation.r2], axis=0
        )
        candidate_sse = np.concatenate(
            [candidate_pca.sse[None], candidate_evaluation.sse], axis=0
        )
        candidate_quantiles = np.concatenate(
            [
                candidate_pca.per_frame_r2_quantiles[None],
                candidate_evaluation.per_frame_r2_quantiles,
            ],
            axis=0,
        )
        full_r2 = np.concatenate([full_pca.r2[None], full_evaluation.r2], axis=0)
        full_sse = np.concatenate([full_pca.sse[None], full_evaluation.sse], axis=0)
        full_quantiles = np.concatenate(
            [full_pca.per_frame_r2_quantiles[None], full_evaluation.per_frame_r2_quantiles],
            axis=0,
        )
        candidate_singular_values = np.stack(
            [
                _pad_singular_values(candidate_pca.singular_values, max_real_rank),
                _pad_singular_values(
                    candidate_bases["direct_2d"].singular_values, max_real_rank
                ),
                _pad_singular_values(
                    candidate_bases["lifted_3d"].singular_values, max_real_rank
                ),
            ],
            axis=0,
        )
        full_singular_values = np.stack(
            [
                _pad_singular_values(full_pca.singular_values, max_real_rank),
                _pad_singular_values(direct_full_basis.singular_values, max_real_rank),
            ],
            axis=0,
        )
        regional_pixel_count = np.bincount(
            (
                np.minimum(candidate_pixels[:, 1] * grid_rows // cache.flow_u.shape[1], grid_rows - 1)
                * grid_cols
                + np.minimum(candidate_pixels[:, 0] * grid_cols // cache.flow_u.shape[2], grid_cols - 1)
            ).astype(np.int64),
            minlength=grid_rows * grid_cols,
        ).astype(np.int64)
        candidate_density = candidate_pca.flow_energy / (
            candidate_pixels.shape[0] * cache.flow_u.shape[0]
        )
        full_density = full_pca.flow_energy / (full_pixels.shape[0] * cache.flow_u.shape[0])
        view_results.append(
            {
                "candidate_pixel_count": candidate_pixels.shape[0],
                "full_sampled_pixel_count": full_pixels.shape[0],
                "full_mask_pixel_count": int(np.count_nonzero(full_mask)),
                "candidate_pixel_fraction": candidate_pixels.shape[0] / full_pixels.shape[0],
                "candidate_flow_energy_density_ratio": candidate_density / full_density,
                "candidate_flow_energy": candidate_pca.flow_energy,
                "full_flow_energy": full_pca.flow_energy,
                "candidate_frame_energy_floor": _frame_energy_floor(
                    candidate_pca.energy_per_frame
                ),
                "full_frame_energy_floor": _frame_energy_floor(full_pca.energy_per_frame),
                "candidate_low_energy_frame_count": int(
                    np.count_nonzero(
                        candidate_pca.energy_per_frame
                        <= _frame_energy_floor(candidate_pca.energy_per_frame)
                    )
                ),
                "full_low_energy_frame_count": int(
                    np.count_nonzero(
                        full_pca.energy_per_frame
                        <= _frame_energy_floor(full_pca.energy_per_frame)
                    )
                ),
                "candidate_r2": candidate_r2,
                "candidate_sse": candidate_sse,
                "candidate_quantiles": candidate_quantiles,
                "full_r2": full_r2,
                "full_sse": full_sse,
                "full_quantiles": full_quantiles,
                "candidate_basis_rank": np.asarray(
                    [
                        candidate_pca.numerical_rank,
                        candidate_bases["direct_2d"].numerical_rank,
                        candidate_bases["lifted_3d"].numerical_rank,
                    ],
                    dtype=np.int64,
                ),
                "full_basis_rank": np.asarray(
                    [full_pca.numerical_rank, direct_full_basis.numerical_rank],
                    dtype=np.int64,
                ),
                "candidate_singular_values": candidate_singular_values,
                "full_singular_values": full_singular_values,
                "direct_pair_scales_candidate": candidate_bases["direct_2d"].pair_scales,
                "lifted_pair_scales_candidate": candidate_bases["lifted_3d"].pair_scales,
                "direct_pair_scales_full": direct_full_basis.pair_scales,
                "standalone_mode_r2": np.stack(
                    [
                        candidate_evaluation.standalone_r2["direct_2d"],
                        candidate_evaluation.standalone_r2["lifted_3d"],
                    ],
                    axis=0,
                ),
                "ridge_r2": np.asarray(
                    [
                        candidate_evaluation.ridge_r2["direct_2d"],
                        candidate_evaluation.ridge_r2["lifted_3d"],
                    ],
                    dtype=np.float64,
                ),
                "ridge_sse": np.asarray(
                    [
                        candidate_evaluation.ridge_sse["direct_2d"],
                        candidate_evaluation.ridge_sse["lifted_3d"],
                    ],
                    dtype=np.float64,
                ),
                "regional_r2": regional_r2,
                "regional_sse": regional_sse,
                "regional_energy": regional_energy,
                "regional_valid": regional_valid,
                "regional_pixel_count": regional_pixel_count,
                "warp": warp,
                "seconds": float(time.perf_counter() - view_start),
            }
        )

    view_ids = np.asarray(frame_map.view_ids)
    real_ranks = np.arange(1, max_real_rank + 1, dtype=np.int64)
    candidate_energy = np.asarray(
        [values["candidate_flow_energy"] for values in view_results], dtype=np.float64
    )
    full_energy = np.asarray(
        [values["full_flow_energy"] for values in view_results], dtype=np.float64
    )
    candidate_sse = np.stack([values["candidate_sse"] for values in view_results])
    full_sse = np.stack([values["full_sse"] for values in view_results])
    candidate_pooled, candidate_macro, candidate_worst, candidate_std = _aggregate_curves(
        candidate_sse, candidate_energy
    )
    full_pooled, full_macro, full_worst, full_std = _aggregate_curves(full_sse, full_energy)
    ridge_sse = np.stack([values["ridge_sse"] for values in view_results])
    ridge_pooled = 1.0 - np.sum(ridge_sse, axis=0) / float(np.sum(candidate_energy))
    standalone_view_r2 = np.stack(
        [values["standalone_mode_r2"] for values in view_results]
    )
    standalone_sse = candidate_energy[:, None, None] * (1.0 - standalone_view_r2)
    standalone_pooled_r2 = 1.0 - np.sum(standalone_sse, axis=0) / float(
        np.sum(candidate_energy)
    )

    arrays: dict[str, np.ndarray] = {
        "format": np.array(MODAL_BASIS_DIAGNOSTIC_FORMAT),
        "version": np.array(MODAL_BASIS_DIAGNOSTIC_VERSION, dtype=np.int32),
        "view_ids": view_ids,
        "mode_indices": manifest.mode_indices.astype(np.int64),
        "frequencies_hz": manifest.frequencies_hz.astype(np.float64),
        "real_ranks": real_ranks,
        "candidate_method_names": np.asarray(_CANDIDATE_METHODS),
        "full_mask_method_names": np.asarray(_FULL_MASK_METHODS),
        "ridge_method_names": np.asarray(_RIDGE_METHODS),
        "candidate_pixel_count": np.asarray(
            [values["candidate_pixel_count"] for values in view_results], dtype=np.int64
        ),
        "full_sampled_pixel_count": np.asarray(
            [values["full_sampled_pixel_count"] for values in view_results], dtype=np.int64
        ),
        "full_mask_pixel_count": np.asarray(
            [values["full_mask_pixel_count"] for values in view_results], dtype=np.int64
        ),
        "candidate_pixel_fraction": np.asarray(
            [values["candidate_pixel_fraction"] for values in view_results], dtype=np.float64
        ),
        "candidate_flow_energy_density_ratio": np.asarray(
            [values["candidate_flow_energy_density_ratio"] for values in view_results],
            dtype=np.float64,
        ),
        "candidate_flow_energy": candidate_energy,
        "full_flow_energy": full_energy,
        "candidate_frame_energy_floor": np.asarray(
            [values["candidate_frame_energy_floor"] for values in view_results],
            dtype=np.float64,
        ),
        "full_frame_energy_floor": np.asarray(
            [values["full_frame_energy_floor"] for values in view_results],
            dtype=np.float64,
        ),
        "candidate_low_energy_frame_count": np.asarray(
            [values["candidate_low_energy_frame_count"] for values in view_results],
            dtype=np.int64,
        ),
        "full_low_energy_frame_count": np.asarray(
            [values["full_low_energy_frame_count"] for values in view_results],
            dtype=np.int64,
        ),
        "candidate_view_r2": np.stack([values["candidate_r2"] for values in view_results]),
        "candidate_view_sse": candidate_sse,
        "candidate_per_frame_r2_quantiles": np.stack(
            [values["candidate_quantiles"] for values in view_results]
        ),
        "candidate_pooled_r2": candidate_pooled,
        "candidate_macro_r2": candidate_macro,
        "candidate_worst_view_r2": candidate_worst,
        "candidate_view_r2_std": candidate_std,
        "full_view_r2": np.stack([values["full_r2"] for values in view_results]),
        "full_view_sse": full_sse,
        "full_per_frame_r2_quantiles": np.stack(
            [values["full_quantiles"] for values in view_results]
        ),
        "full_pooled_r2": full_pooled,
        "full_macro_r2": full_macro,
        "full_worst_view_r2": full_worst,
        "full_view_r2_std": full_std,
        "candidate_basis_numerical_rank": np.stack(
            [values["candidate_basis_rank"] for values in view_results]
        ),
        "full_basis_numerical_rank": np.stack(
            [values["full_basis_rank"] for values in view_results]
        ),
        "candidate_basis_singular_values": np.stack(
            [values["candidate_singular_values"] for values in view_results]
        ),
        "full_basis_singular_values": np.stack(
            [values["full_singular_values"] for values in view_results]
        ),
        "direct_pair_scales_candidate": np.stack(
            [values["direct_pair_scales_candidate"] for values in view_results]
        ),
        "lifted_pair_scales_candidate": np.stack(
            [values["lifted_pair_scales_candidate"] for values in view_results]
        ),
        "direct_pair_scales_full": np.stack(
            [values["direct_pair_scales_full"] for values in view_results]
        ),
        "standalone_mode_r2": standalone_view_r2,
        "standalone_mode_pooled_r2": standalone_pooled_r2,
        "ridge_view_r2": np.stack([values["ridge_r2"] for values in view_results]),
        "ridge_view_sse": ridge_sse,
        "ridge_pooled_r2": ridge_pooled,
        "regional_r2": np.stack([values["regional_r2"] for values in view_results]),
        "regional_sse": np.stack([values["regional_sse"] for values in view_results]),
        "regional_flow_energy": np.stack(
            [values["regional_energy"] for values in view_results]
        ),
        "regional_valid": np.stack([values["regional_valid"] for values in view_results]),
        "regional_pixel_count": np.stack(
            [values["regional_pixel_count"] for values in view_results]
        ),
        "warp_local_indices": np.stack(
            [values["warp"]["local_indices"] for values in view_results]
        ),
        "warp_mae": np.stack([values["warp"]["warp_mae"] for values in view_results]),
        "warp_zero_flow_mae": np.stack(
            [values["warp"]["zero_flow_mae"] for values in view_results]
        ),
        "warp_support_fraction": np.stack(
            [values["warp"]["support_fraction"] for values in view_results]
        ),
        "warp_mean_mae": np.asarray(
            [values["warp"]["mean_warp_mae"] for values in view_results], dtype=np.float64
        ),
        "warp_zero_mean_mae": np.asarray(
            [values["warp"]["mean_zero_flow_mae"] for values in view_results],
            dtype=np.float64,
        ),
        "warp_to_zero_mae_ratio": np.asarray(
            [values["warp"]["warp_to_zero_mae_ratio"] for values in view_results],
            dtype=np.float64,
        ),
        "warp_reference_image_rmse": np.asarray(
            [values["warp"]["reference_image_rmse"] for values in view_results],
            dtype=np.float64,
        ),
        "warp_reference_flow_rms": np.asarray(
            [values["warp"]["reference_flow_rms"] for values in view_results],
            dtype=np.float64,
        ),
        "view_seconds": np.asarray(
            [values["seconds"] for values in view_results], dtype=np.float64
        ),
    }

    analysis_seconds = float(time.perf_counter() - total_start)
    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        np.savez_compressed(temp_path / DIAGNOSTICS_FILENAME, **arrays)
        _plot_curves(temp_path / PLOT_FILENAME, arrays)
        settings = {
            "max_real_rank": max_real_rank,
            "pixel_stride": pixel_stride,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "warp_frame_count": warp_frame_count,
            "production_ridge_relative": production_ridge_relative,
            "flow_pixel_chunk_size": _FLOW_PIXEL_CHUNK_SIZE,
            "pca_oversample": _PCA_OVERSAMPLE,
            "pca_power_iterations": _PCA_POWER_ITERATIONS,
            "pca_seed": _PCA_SEED,
            "per_frame_energy_relative_floor": _FRAME_ENERGY_RELATIVE_FLOOR,
        }
        source = {
            "modal_manifest": str(manifest.path),
            "modal_frame_map": str(frame_map.path),
            "flow_caches": [str(cache.path.resolve()) for cache in caches],
            "modal_npzs": [str(modal.path) for modal in modal_views],
            "dynamic_data_dir": str(dynamic_path.resolve()),
            "dynamic_dataset_metadata": dynamic_metadata,
            "modal_export_provenance": [modal.export_provenance for modal in modal_views],
            "flow_cache_video_provenance": [
                cache.metadata["sources"]["video"] for cache in caches
            ],
            "flow_cache_mask_provenance": [
                cache.metadata["sources"]["mask"] for cache in caches
            ],
        }
        timings = {
            "input_validation": input_seconds,
            "views": arrays["view_seconds"].tolist(),
            "analysis": analysis_seconds,
            "binary_artifact_write_before_summary": float(
                time.perf_counter() - total_start - analysis_seconds
            ),
        }
        _write_summary(
            temp_path / SUMMARY_FILENAME,
            arrays,
            source,
            settings,
            timings,
        )
        _validate_output(
            temp_path,
            len(frame_map.view_ids),
            num_modes,
            max_real_rank,
            grid_rows * grid_cols,
            warp_frame_count,
        )
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError(
                f"Diagnostic output directory appeared during analysis: {output_path}"
            )
        os.replace(temp_path, output_path)
    except BaseException:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise
    return ModalBasisDiagnosticResult(
        path=output_path.resolve(),
        summary_path=(output_path / SUMMARY_FILENAME).resolve(),
        diagnostics_path=(output_path / DIAGNOSTICS_FILENAME).resolve(),
        plot_path=(output_path / PLOT_FILENAME).resolve(),
    )
