from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow3d.modal_flow_coordinates import (
    _load_and_validate_caches,
    _load_frame_map,
    _load_manifest,
)
from flow3d.modal_frequency_selection import _candidate_pixels
from modal_peak_pick.core.background_compensation import (
    BackgroundCompensationSettings,
    dilate_foreground_masks,
    estimate_background_transform,
    json_float,
    transform_field_rms,
)
from modal_peak_pick.core.cache import ModalAnalysisCache
from modal_peak_pick.core.video_io import load_video_clip


FORMAT = "modal_flow_stabilization_ab"
VERSION = 1
SUMMARY_FILENAME = "comparison_summary.json"
DIAGNOSTICS_FILENAME = "comparison_diagnostics.npz"
SPECTRUM_PLOT_FILENAME = "spectrum_comparison.png"
FLOW_PLOT_FILENAME = "flow_energy_comparison.png"
REGIONAL_PLOT_FILENAME = "regional_comparison.png"


@dataclass(frozen=True)
class FlowABComparisonResult:
    path: Path
    summary_path: Path
    diagnostics_path: Path
    spectrum_plot_path: Path
    flow_plot_path: Path
    regional_plot_path: Path


def _stats(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "rms": json_float(np.sqrt(np.mean(array * array))),
        "p50": json_float(np.percentile(array, 50)),
        "p90": json_float(np.percentile(array, 90)),
        "p99": json_float(np.percentile(array, 99)),
        "max": json_float(np.max(array)),
    }


def _flow_data(cache: ModalAnalysisCache, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = pixels[:, 0]
    y = pixels[:, 1]
    u = np.asarray(cache.flow_u[:, y, x], dtype=np.float32)
    v = np.asarray(cache.flow_v[:, y, x], dtype=np.float32)
    reference_index = int(cache.metadata["analysis"]["reference_frame_index"])
    u = u - u[reference_index : reference_index + 1]
    v = v - v[reference_index : reference_index + 1]
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError(f"Flow cache {cache.path} has non-finite candidate flow")
    return u, v


def _correlation(a_u: np.ndarray, a_v: np.ndarray, b_u: np.ndarray, b_v: np.ndarray) -> float:
    a = np.concatenate([a_u.reshape(-1), a_v.reshape(-1)]).astype(np.float64)
    b = np.concatenate([b_u.reshape(-1), b_v.reshape(-1)]).astype(np.float64)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= np.finfo(np.float64).eps else float(np.dot(a, b) / denominator)


def _spectrum_power(cache: ModalAnalysisCache, pixels: np.ndarray) -> np.ndarray:
    x = pixels[:, 0]
    y = pixels[:, 1]
    u = np.asarray(cache.spectrum_u[:, y, x])
    v = np.asarray(cache.spectrum_v[:, y, x])
    power = np.mean(np.abs(u) ** 2 + np.abs(v) ** 2, axis=1, dtype=np.float64)
    if not np.isfinite(power).all():
        raise ValueError(f"Spectrum cache {cache.path} has non-finite candidate power")
    return power


def _band_energy(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    selected = (freqs >= low) & (freqs <= high)
    return float(np.sum(power[selected], dtype=np.float64))


def _regional_metrics(
    pixels: np.ndarray,
    height: int,
    width: int,
    a_u: np.ndarray,
    a_v: np.ndarray,
    b_u: np.ndarray,
    b_v: np.ndarray,
    rows: int,
    cols: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            x0 = col * width / cols
            x1 = (col + 1) * width / cols
            y0 = row * height / rows
            y1 = (row + 1) * height / rows
            selected = (
                (pixels[:, 0] >= x0)
                & (pixels[:, 0] < x1)
                & (pixels[:, 1] >= y0)
                & (pixels[:, 1] < y1)
            )
            count = int(np.count_nonzero(selected))
            if count == 0:
                results.append(
                    {"row": row, "col": col, "pixel_count": 0, "correlation": None}
                )
                continue
            a_mag = np.sqrt(a_u[:, selected] ** 2 + a_v[:, selected] ** 2)
            b_mag = np.sqrt(b_u[:, selected] ** 2 + b_v[:, selected] ** 2)
            results.append(
                {
                    "row": row,
                    "col": col,
                    "pixel_count": count,
                    "stabilized_rms": json_float(np.sqrt(np.mean(a_mag * a_mag))),
                    "compensated_rms": json_float(np.sqrt(np.mean(b_mag * b_mag))),
                    "rms_ratio": json_float(
                        np.sqrt(np.mean(b_mag * b_mag))
                        / max(np.sqrt(np.mean(a_mag * a_mag)), 1e-12)
                    ),
                    "correlation": json_float(
                        _correlation(
                            a_u[:, selected],
                            a_v[:, selected],
                            b_u[:, selected],
                            b_v[:, selected],
                        )
                    ),
                }
            )
    return results


def _compensation_diagnostics(cache: ModalAnalysisCache) -> dict[str, np.ndarray]:
    path = cache.path / "camera_compensation_diagnostics.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Compensated cache is missing {path.name}: {cache.path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "residual_background_rms",
            "residual_background_p90",
            "camera_flow_foreground_rms",
            "residual_camera_flow_foreground_rms",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        result = {name: np.asarray(archive[name], dtype=np.float64) for name in required}
    frame_count = cache.flow_u.shape[0]
    for name, values in result.items():
        if values.shape != (frame_count,) or not np.isfinite(values).all():
            raise ValueError(f"{path} {name} must be finite with shape [{frame_count}]")
    return result


def _stabilized_background_motion(
    cache: ModalAnalysisCache,
    settings: BackgroundCompensationSettings,
) -> dict[str, np.ndarray]:
    source = cache.metadata["sources"]["video"]
    video_path = Path(source.get("resolved_path") or source["path"]).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(
            "Stabilized source video is required for background residual diagnostics: "
            f"{video_path}"
        )
    frame_range = cache.metadata["video"]["frame_range"]
    frames, fps = load_video_clip(
        str(video_path),
        t0=float(frame_range["t0_s"]),
        t1=None if frame_range["t1_s"] is None else float(frame_range["t1_s"]),
        resize=cache.metadata["video"]["resize_max_side"],
        grayscale=True,
        max_frames=frame_range["max_frames"],
    )
    if frames.shape != cache.flow_u.shape:
        raise ValueError(
            f"Stabilized source video shape {frames.shape} does not match cache "
            f"{cache.flow_u.shape}"
        )
    if not np.isclose(fps, cache.fps, rtol=0.0, atol=1e-9):
        raise ValueError("Stabilized source video FPS does not match its cache")
    reference_index = int(cache.metadata["analysis"]["reference_frame_index"])
    repeated_mask = np.broadcast_to(
        np.asarray(cache.mask, dtype=np.uint8)[None], frames.shape
    )
    dilated = dilate_foreground_masks(repeated_mask, settings.mask_dilate_px)
    field_rms = np.zeros((frames.shape[0],), dtype=np.float64)
    background_rms = np.zeros_like(field_rms)
    background_p90 = np.zeros_like(field_rms)
    inlier_fraction = np.ones_like(field_rms)
    for frame_index in range(frames.shape[0]):
        if frame_index == reference_index:
            continue
        estimate = estimate_background_transform(
            frames[reference_index],
            frames[frame_index],
            dilated[reference_index],
            dilated[frame_index],
            settings,
            label=f"stabilized residual frame {frame_index}",
        )
        displacement = estimate.target_points - estimate.source_points
        magnitude = np.linalg.norm(displacement, axis=1)
        background_rms[frame_index] = float(np.sqrt(np.mean(magnitude * magnitude)))
        background_p90[frame_index] = float(np.percentile(magnitude, 90))
        field_rms[frame_index] = transform_field_rms(
            estimate.matrix, np.asarray(cache.mask, dtype=bool)
        )
        inlier_fraction[frame_index] = estimate.inlier_fraction
    return {
        "background_rms": background_rms,
        "background_p90": background_p90,
        "camera_flow_foreground_rms": field_rms,
        "inlier_fraction": inlier_fraction,
    }


def _plot_spectra(
    path: Path,
    view_ids: Sequence[str],
    frequencies: Sequence[np.ndarray],
    stabilized_power: Sequence[np.ndarray],
    compensated_power: Sequence[np.ndarray],
) -> None:
    figure, axes = plt.subplots(len(view_ids), 1, figsize=(13, 4 * len(view_ids)), sharex=True)
    axes_array = np.atleast_1d(axes)
    for axis, view_id, freqs, a, b in zip(
        axes_array, view_ids, frequencies, stabilized_power, compensated_power
    ):
        a_scale = max(float(np.max(a)), 1e-12)
        b_scale = max(float(np.max(b)), 1e-12)
        axis.plot(freqs, a / a_scale, label="stabilized")
        axis.plot(freqs, b / b_scale, label="raw + background compensation")
        axis.axvspan(0.05, 0.3, color="gray", alpha=0.15)
        axis.set_ylabel(f"{view_id}\nnormalized power")
        axis.legend()
    axes_array[-1].set_xlabel("frequency (Hz)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_frame_energy(
    path: Path,
    view_ids: Sequence[str],
    stabilized: Sequence[np.ndarray],
    compensated: Sequence[np.ndarray],
) -> None:
    figure, axes = plt.subplots(len(view_ids), 1, figsize=(13, 4 * len(view_ids)), sharex=False)
    axes_array = np.atleast_1d(axes)
    for axis, view_id, a, b in zip(axes_array, view_ids, stabilized, compensated):
        axis.plot(a, label="stabilized")
        axis.plot(b, label="raw + background compensation")
        axis.set_ylabel(f"{view_id}\nflow RMS (px)")
        axis.legend()
    axes_array[-1].set_xlabel("local frame index")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_regions(path: Path, view_ids: Sequence[str], regional: Sequence[list[dict[str, Any]]]) -> None:
    figure, axes = plt.subplots(1, len(view_ids), figsize=(6 * len(view_ids), 5))
    axes_array = np.atleast_1d(axes)
    for axis, view_id, records in zip(axes_array, view_ids, regional):
        rows = max(record["row"] for record in records) + 1
        cols = max(record["col"] for record in records) + 1
        matrix = np.full((rows, cols), np.nan, dtype=np.float64)
        for record in records:
            value = record.get("rms_ratio")
            if value is not None:
                matrix[record["row"], record["col"]] = float(value)
        image = axis.imshow(matrix, vmin=0.5, vmax=1.5, cmap="coolwarm")
        axis.set_title(f"{view_id}: compensated/stabilized RMS")
        axis.set_xlabel("grid column")
        axis.set_ylabel("grid row")
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_modal_flow_ab_comparison(
    *,
    stabilized_cache_specs: Sequence[str],
    compensated_cache_specs: Sequence[str],
    modal_manifest_path: str | Path,
    modal_frame_map_path: str | Path,
    output_dir: str | Path,
    pixel_stride: int = 2,
    grid_rows: int = 4,
    grid_cols: int = 4,
    background_mask_dilate_px: int = 16,
) -> FlowABComparisonResult:
    target = Path(output_dir).expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"A/B comparison output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if pixel_stride <= 0 or grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("pixel_stride, grid_rows, and grid_cols must be positive")
    manifest = _load_manifest(modal_manifest_path)
    frame_map = _load_frame_map(modal_frame_map_path)
    stabilized_caches = _load_and_validate_caches(
        frame_map, manifest.topology, stabilized_cache_specs
    )
    compensated_caches = _load_and_validate_caches(
        frame_map, manifest.topology, compensated_cache_specs
    )
    if len(stabilized_caches) != len(compensated_caches):
        raise ValueError("Stabilized and compensated cache counts differ")
    compensation_models = {
        str(
            cache.metadata["analysis"]["camera_compensation"]["settings"][
                "transform_model"
            ]
        )
        for cache in compensated_caches
    }
    if len(compensation_models) != 1:
        raise ValueError(
            "All compensated caches must use the same background transform model; "
            f"got {sorted(compensation_models)}"
        )
    background_transform_model = next(iter(compensation_models))

    temp = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    )
    try:
        summaries: list[dict[str, Any]] = []
        frequencies: list[np.ndarray] = []
        stable_powers: list[np.ndarray] = []
        compensated_powers: list[np.ndarray] = []
        stable_frame_rms_values: list[np.ndarray] = []
        compensated_frame_rms_values: list[np.ndarray] = []
        regional_values: list[list[dict[str, Any]]] = []
        diagnostic_arrays: dict[str, np.ndarray] = {
            "view_ids": np.asarray(frame_map.view_ids),
        }
        settings = BackgroundCompensationSettings(
            mask_dilate_px=int(background_mask_dilate_px),
            transform_model=background_transform_model,
        )
        pooled_stable_energy = 0.0
        pooled_compensated_energy = 0.0
        pooled_difference_energy = 0.0
        pooled_count = 0

        for view_index, (stable, compensated) in enumerate(
            zip(stabilized_caches, compensated_caches)
        ):
            view_id = frame_map.view_ids[view_index]
            if stable.flow_u.shape != compensated.flow_u.shape:
                raise ValueError(f"{view_id} cache flow shapes differ")
            if not np.array_equal(stable.mask_array, compensated.mask_array):
                raise ValueError(f"{view_id} stabilized and compensated masks differ")
            if not np.array_equal(stable.freqs_hz, compensated.freqs_hz):
                raise ValueError(f"{view_id} stabilized and compensated frequency axes differ")
            if int(stable.metadata["analysis"]["reference_frame_index"]) != int(
                compensated.metadata["analysis"]["reference_frame_index"]
            ):
                raise ValueError(f"{view_id} reference frame indices differ")
            pixels = _candidate_pixels(
                manifest.topology, view_index, stable, int(pixel_stride)
            )
            a_u, a_v = _flow_data(stable, pixels)
            b_u, b_v = _flow_data(compensated, pixels)
            a_mag = np.sqrt(a_u * a_u + a_v * a_v)
            b_mag = np.sqrt(b_u * b_u + b_v * b_v)
            difference = np.sqrt((b_u - a_u) ** 2 + (b_v - a_v) ** 2)
            a_frame_rms = np.sqrt(np.mean(a_mag * a_mag, axis=1))
            b_frame_rms = np.sqrt(np.mean(b_mag * b_mag, axis=1))
            strong = a_frame_rms >= np.percentile(a_frame_rms, 90)
            stable_power = _spectrum_power(stable, pixels)
            compensated_power = _spectrum_power(compensated, pixels)
            low_a = _band_energy(stable.freqs_hz, stable_power, 0.05, 0.3)
            low_b = _band_energy(compensated.freqs_hz, compensated_power, 0.05, 0.3)
            modal_a = _band_energy(stable.freqs_hz, stable_power, 0.3, 2.2)
            modal_b = _band_energy(compensated.freqs_hz, compensated_power, 0.3, 2.2)
            compensated_diag = _compensation_diagnostics(compensated)
            stable_background = _stabilized_background_motion(stable, settings)
            a_rms = float(np.sqrt(np.mean(a_mag * a_mag)))
            b_rms = float(np.sqrt(np.mean(b_mag * b_mag)))
            a_camera_rms = float(
                np.sqrt(
                    np.mean(
                        stable_background["camera_flow_foreground_rms"] ** 2
                    )
                )
            )
            b_camera_rms = float(
                np.sqrt(
                    np.mean(
                        compensated_diag["residual_camera_flow_foreground_rms"] ** 2
                    )
                )
            )
            regional = _regional_metrics(
                pixels,
                stable.flow_u.shape[1],
                stable.flow_u.shape[2],
                a_u,
                a_v,
                b_u,
                b_v,
                grid_rows,
                grid_cols,
            )
            regional_values.append(regional)
            summaries.append(
                {
                    "view_id": view_id,
                    "candidate_pixel_count": int(pixels.shape[0]),
                    "stabilized_flow_magnitude": _stats(a_mag),
                    "compensated_flow_magnitude": _stats(b_mag),
                    "flow_difference_magnitude": _stats(difference),
                    "flow_correlation": json_float(
                        _correlation(a_u, a_v, b_u, b_v)
                    ),
                    "strong_motion_rms_retention": json_float(
                        np.mean(b_frame_rms[strong])
                        / max(float(np.mean(a_frame_rms[strong])), 1e-12)
                    ),
                    "spectrum": {
                        "stabilized_low_0p05_0p3_energy": json_float(low_a),
                        "compensated_low_0p05_0p3_energy": json_float(low_b),
                        "low_energy_ratio": json_float(low_b / max(low_a, 1e-12)),
                        "stabilized_modal_0p3_2p2_energy": json_float(modal_a),
                        "compensated_modal_0p3_2p2_energy": json_float(modal_b),
                        "modal_energy_ratio": json_float(modal_b / max(modal_a, 1e-12)),
                    },
                    "background": {
                        "stabilized_residual_rms_px": _stats(
                            stable_background["background_rms"]
                        ),
                        "compensated_residual_rms_px": _stats(
                            compensated_diag["residual_background_rms"]
                        ),
                        "stabilized_camera_flow_on_foreground_rms_px": json_float(
                            a_camera_rms
                        ),
                        "compensated_residual_camera_flow_on_foreground_rms_px": json_float(
                            b_camera_rms
                        ),
                        "stabilized_contamination_ratio_r": json_float(
                            a_camera_rms / max(a_rms, 1e-12)
                        ),
                        "compensated_contamination_ratio_r": json_float(
                            b_camera_rms / max(b_rms, 1e-12)
                        ),
                    },
                    "regions": regional,
                }
            )
            frequencies.append(np.asarray(stable.freqs_hz, dtype=np.float64))
            stable_powers.append(stable_power)
            compensated_powers.append(compensated_power)
            stable_frame_rms_values.append(a_frame_rms)
            compensated_frame_rms_values.append(b_frame_rms)
            diagnostic_arrays[f"{view_id}_candidate_pixels_xy"] = pixels
            diagnostic_arrays[f"{view_id}_stabilized_frame_rms"] = a_frame_rms
            diagnostic_arrays[f"{view_id}_compensated_frame_rms"] = b_frame_rms
            diagnostic_arrays[f"{view_id}_stabilized_spectrum_power"] = stable_power
            diagnostic_arrays[f"{view_id}_compensated_spectrum_power"] = compensated_power
            diagnostic_arrays[f"{view_id}_stabilized_background_rms"] = stable_background[
                "background_rms"
            ]
            diagnostic_arrays[f"{view_id}_compensated_background_rms"] = compensated_diag[
                "residual_background_rms"
            ]
            pooled_stable_energy += float(np.sum(a_mag * a_mag, dtype=np.float64))
            pooled_compensated_energy += float(np.sum(b_mag * b_mag, dtype=np.float64))
            pooled_difference_energy += float(
                np.sum(difference * difference, dtype=np.float64)
            )
            pooled_count += int(a_mag.size)

        summary = {
            "format": FORMAT,
            "version": VERSION,
            "source": {
                "modal_manifest": str(Path(modal_manifest_path).resolve()),
                "modal_frame_map": str(Path(modal_frame_map_path).resolve()),
                "stabilized_caches": [str(cache.path.resolve()) for cache in stabilized_caches],
                "compensated_caches": [str(cache.path.resolve()) for cache in compensated_caches],
            },
            "settings": {
                "pixel_stride": int(pixel_stride),
                "grid_rows": int(grid_rows),
                "grid_cols": int(grid_cols),
                "background_mask_dilate_px": int(background_mask_dilate_px),
                "background_transform_model": background_transform_model,
            },
            "overall": {
                "candidate_element_count": pooled_count,
                "stabilized_flow_rms": json_float(
                    np.sqrt(pooled_stable_energy / max(1, pooled_count))
                ),
                "compensated_flow_rms": json_float(
                    np.sqrt(pooled_compensated_energy / max(1, pooled_count))
                ),
                "flow_difference_rms": json_float(
                    np.sqrt(pooled_difference_energy / max(1, pooled_count))
                ),
                "compensated_to_stabilized_energy_ratio": json_float(
                    pooled_compensated_energy / max(pooled_stable_energy, 1e-12)
                ),
            },
            "views": summaries,
        }
        with (temp / SUMMARY_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
            file.write("\n")
        np.savez_compressed(
            temp / DIAGNOSTICS_FILENAME,
            format=np.asarray(FORMAT),
            version=np.asarray(VERSION, dtype=np.int64),
            **diagnostic_arrays,
        )
        _plot_spectra(
            temp / SPECTRUM_PLOT_FILENAME,
            frame_map.view_ids,
            frequencies,
            stable_powers,
            compensated_powers,
        )
        _plot_frame_energy(
            temp / FLOW_PLOT_FILENAME,
            frame_map.view_ids,
            stable_frame_rms_values,
            compensated_frame_rms_values,
        )
        _plot_regions(temp / REGIONAL_PLOT_FILENAME, frame_map.view_ids, regional_values)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"A/B comparison output already exists: {target}")
        os.replace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return FlowABComparisonResult(
        path=target,
        summary_path=target / SUMMARY_FILENAME,
        diagnostics_path=target / DIAGNOSTICS_FILENAME,
        spectrum_plot_path=target / SPECTRUM_PLOT_FILENAME,
        flow_plot_path=target / FLOW_PLOT_FILENAME,
        regional_plot_path=target / REGIONAL_PLOT_FILENAME,
    )
