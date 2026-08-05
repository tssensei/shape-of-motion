"""Rendered foreground-Gaussian projection design for modal flow coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np

from modal_peak_pick.core.cache import load_analysis_cache
from modal_surface.geometry import erode_mask, projection_jacobian, world_to_camera_points
from modal_surface.io import load_view_config


RENDERED_MODAL_DESIGN_FORMAT = "rendered_modal_design"
RENDERED_MODAL_DESIGN_VERSION = 1
RENDERED_MODAL_DESIGN_FILENAME = "rendered_modal_design.npz"
RENDERED_MODAL_DESIGN_SUMMARY_FILENAME = "summary.json"
RENDERED_MODAL_DESIGN_NORMALIZATION = "alpha_normalized_foreground_v1"
ROLE_DIAGNOSTICS_FILENAME = "role_diagnostics.npz"
ROLE_OVERVIEW_FILENAME = "role_contribution_overview.png"
_PACKING_MAX_ABS_TOLERANCE = 1e-3
_PACKING_RELATIVE_L2_TOLERANCE = 1e-3

__all__ = [
    "RENDERED_MODAL_DESIGN_FILENAME",
    "RENDERED_MODAL_DESIGN_FORMAT",
    "RENDERED_MODAL_DESIGN_NORMALIZATION",
    "RENDERED_MODAL_DESIGN_SUMMARY_FILENAME",
    "RENDERED_MODAL_DESIGN_VERSION",
    "RenderedModalDesign",
    "build_rendered_modal_design",
    "compute_flow_cache_identity",
    "load_rendered_modal_design",
]


@dataclass(frozen=True)
class RenderedModalDesign:
    """Validated rendered design matrix and immutable source provenance."""

    path: Path
    view_ids: tuple[str, ...]
    view_image_width: np.ndarray
    view_image_height: np.ndarray
    mode_indices: np.ndarray
    frequencies_hz: np.ndarray
    sample_view_index: np.ndarray
    sample_pixels_xy: np.ndarray
    design_matrix: np.ndarray
    sampled_alpha: np.ndarray
    source_checkpoint: Path
    source_modal_manifest: Path
    source_view_configs: tuple[Path, ...]
    source_flow_cache_dirs: tuple[Path, ...]
    source_flow_cache_identities: tuple[str, ...]
    checkpoint_identity: str
    gaussian_identity: str
    phi_identity: str
    camera_identity: str
    artifact_identity: str
    pixel_sample_stride: int
    alpha_min: float
    mask_erode_iters: int
    modes_per_batch: int
    rasterizer: str
    normalization: str


_REQUIRED_FIELDS = {
    "format",
    "version",
    "view_ids",
    "view_image_width",
    "view_image_height",
    "mode_indices",
    "frequencies_hz",
    "sample_view_index",
    "sample_pixels_xy",
    "design_matrix",
    "sampled_alpha",
    "source_checkpoint",
    "source_modal_manifest",
    "source_view_configs",
    "source_flow_cache_dirs",
    "source_flow_cache_identities",
    "checkpoint_identity",
    "gaussian_identity",
    "phi_identity",
    "camera_identity",
    "artifact_identity",
    "pixel_sample_stride",
    "alpha_min",
    "mask_erode_iters",
    "modes_per_batch",
    "rasterizer",
    "normalization",
}


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


def _string_vector(value: np.ndarray, name: str, path: Path) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{path} {name} must be a 1-D string array")
    values: list[str] = []
    for raw in array:
        item = np.asarray(raw).item()
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if not isinstance(item, str) or not item:
            raise ValueError(f"{path} {name} must contain non-empty strings")
        values.append(item)
    return tuple(values)


def _scalar_int(value: np.ndarray, name: str, path: Path) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{path} {name} must be an integer scalar")
    return int(array.item())


def _scalar_float(value: np.ndarray, name: str, path: Path) -> float:
    array = np.asarray(value)
    if (
        array.shape != ()
        or not np.issubdtype(array.dtype, np.number)
        or np.iscomplexobj(array)
    ):
        raise ValueError(f"{path} {name} must be a real scalar")
    result = float(array.item())
    if not np.isfinite(result):
        raise ValueError(f"{path} {name} must be finite")
    return result


def _hash_array(hasher: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    hasher.update(name.encode("utf-8"))
    hasher.update(array.dtype.str.encode("ascii"))
    hasher.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    hasher.update(array.tobytes(order="C"))


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_flow_cache_identity(cache_dir: str | Path) -> str:
    """Fingerprint one validated flow cache without hashing multi-GB flow arrays."""

    cache = load_analysis_cache(cache_dir)
    hasher = hashlib.sha256()
    hasher.update(str(cache.path.resolve()).encode("utf-8"))
    metadata_bytes = json.dumps(
        cache.metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    hasher.update(metadata_bytes)
    arrays = cache.metadata["arrays"]
    for name in sorted(arrays):
        array_path = cache.path / str(arrays[name]["file"])
        stat = array_path.stat()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array_path.resolve()).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("ascii"))
        hasher.update(str(stat.st_mtime_ns).encode("ascii"))
    _hash_array(hasher, "mask", np.asarray(cache.mask_array))
    return hasher.hexdigest()


def _artifact_identity(arrays: dict[str, np.ndarray]) -> str:
    hasher = hashlib.sha256()
    array_fields = (
        "view_image_width",
        "view_image_height",
        "mode_indices",
        "frequencies_hz",
        "sample_view_index",
        "sample_pixels_xy",
        "design_matrix",
        "sampled_alpha",
    )
    scalar_fields = (
        "format",
        "version",
        "view_ids",
        "source_checkpoint",
        "source_modal_manifest",
        "source_view_configs",
        "source_flow_cache_dirs",
        "source_flow_cache_identities",
        "checkpoint_identity",
        "gaussian_identity",
        "phi_identity",
        "camera_identity",
        "pixel_sample_stride",
        "alpha_min",
        "mask_erode_iters",
        "modes_per_batch",
        "rasterizer",
        "normalization",
    )
    for name in scalar_fields:
        _hash_array(hasher, name, np.asarray(arrays[name]))
    for name in array_fields:
        _hash_array(hasher, name, np.asarray(arrays[name]))
    return hasher.hexdigest()


def load_rendered_modal_design(path_value: str | Path) -> RenderedModalDesign:
    """Load and strictly validate a version-1 rendered modal design artifact."""

    path = Path(path_value).expanduser()
    if path.is_dir():
        path = path / RENDERED_MODAL_DESIGN_FILENAME
    path = path.resolve(strict=True)
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(_REQUIRED_FIELDS - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in _REQUIRED_FIELDS}

    if _scalar_string(arrays["format"], "format", path) != RENDERED_MODAL_DESIGN_FORMAT:
        raise ValueError(f"{path} has unsupported rendered modal design format")
    if _scalar_int(arrays["version"], "version", path) != RENDERED_MODAL_DESIGN_VERSION:
        raise ValueError(f"{path} has unsupported rendered modal design version")

    view_ids = _string_vector(arrays["view_ids"], "view_ids", path)
    if not view_ids or len(set(view_ids)) != len(view_ids):
        raise ValueError(f"{path} view_ids must be non-empty and unique")
    num_views = len(view_ids)
    widths = np.asarray(arrays["view_image_width"])
    heights = np.asarray(arrays["view_image_height"])
    if widths.shape != (num_views,) or heights.shape != (num_views,):
        raise ValueError(f"{path} view image dimensions must have shape ({num_views},)")
    if not np.issubdtype(widths.dtype, np.integer) or not np.issubdtype(
        heights.dtype, np.integer
    ):
        raise ValueError(f"{path} view image dimensions must be integer-valued")
    if np.any(widths < 3) or np.any(heights < 3):
        raise ValueError(f"{path} view image dimensions must be at least 3x3")

    mode_indices = np.asarray(arrays["mode_indices"])
    frequencies = np.asarray(arrays["frequencies_hz"], dtype=np.float64)
    if (
        mode_indices.ndim != 1
        or mode_indices.size == 0
        or not np.issubdtype(mode_indices.dtype, np.integer)
    ):
        raise ValueError(f"{path} mode_indices must be a non-empty integer vector")
    if len(set(int(value) for value in mode_indices)) != mode_indices.size:
        raise ValueError(f"{path} mode_indices must be unique")
    if np.any(mode_indices < 0):
        raise ValueError(f"{path} mode_indices must be non-negative")
    if frequencies.shape != mode_indices.shape or not np.all(
        np.isfinite(frequencies) & (frequencies > 0.0)
    ):
        raise ValueError(f"{path} frequencies_hz must be finite and positive per mode")

    sample_view_index = np.asarray(arrays["sample_view_index"])
    pixels = np.asarray(arrays["sample_pixels_xy"])
    design = np.asarray(arrays["design_matrix"])
    sampled_alpha = np.asarray(arrays["sampled_alpha"])
    num_samples = sample_view_index.size
    if (
        sample_view_index.shape != (num_samples,)
        or not np.issubdtype(sample_view_index.dtype, np.integer)
        or np.any(sample_view_index < 0)
        or np.any(sample_view_index >= num_views)
    ):
        raise ValueError(f"{path} sample_view_index must contain valid integer view indices")
    if pixels.shape != (num_samples, 2) or not np.issubdtype(pixels.dtype, np.integer):
        raise ValueError(f"{path} sample_pixels_xy must have integer shape [P,2]")
    if design.shape != (num_samples, 2, 2 * mode_indices.size):
        raise ValueError(
            f"{path} design_matrix must have shape ({num_samples},2,{2 * mode_indices.size})"
        )
    if design.dtype != np.float32 or not np.isfinite(design).all():
        raise ValueError(f"{path} design_matrix must contain finite float32 values")
    if sampled_alpha.shape != (num_samples,) or sampled_alpha.dtype != np.float32:
        raise ValueError(f"{path} sampled_alpha must have float32 shape [P]")
    if not np.isfinite(sampled_alpha).all():
        raise ValueError(f"{path} sampled_alpha must be finite")

    stride = _scalar_int(arrays["pixel_sample_stride"], "pixel_sample_stride", path)
    alpha_min = _scalar_float(arrays["alpha_min"], "alpha_min", path)
    mask_erode_iters = _scalar_int(arrays["mask_erode_iters"], "mask_erode_iters", path)
    modes_per_batch = _scalar_int(arrays["modes_per_batch"], "modes_per_batch", path)
    if stride <= 0 or modes_per_batch <= 0 or mask_erode_iters < 0:
        raise ValueError(f"{path} contains invalid sampling or batching settings")
    if not 0.0 < alpha_min <= 1.0:
        raise ValueError(f"{path} alpha_min must lie in (0,1]")
    if num_samples == 0:
        raise ValueError(f"{path} contains no sampled pixels")
    if np.any(sampled_alpha < alpha_min - 1e-6) or np.any(sampled_alpha > 1.0 + 1e-5):
        raise ValueError(f"{path} sampled_alpha lies outside the declared valid range")

    for view_index in range(num_views):
        rows = np.flatnonzero(sample_view_index == view_index)
        if rows.size == 0:
            raise ValueError(f"{path} view {view_ids[view_index]!r} has no sampled pixels")
        view_pixels = pixels[rows]
        width = int(widths[view_index])
        height = int(heights[view_index])
        if (
            np.any(view_pixels[:, 0] < 1)
            or np.any(view_pixels[:, 0] >= width - 1)
            or np.any(view_pixels[:, 1] < 1)
            or np.any(view_pixels[:, 1] >= height - 1)
        ):
            raise ValueError(f"{path} samples for {view_ids[view_index]!r} leave the internal image area")
        if np.any((view_pixels[:, 0] - 1) % stride) or np.any(
            (view_pixels[:, 1] - 1) % stride
        ):
            raise ValueError(f"{path} samples do not follow the declared stride grid")
        if np.unique(view_pixels, axis=0).shape[0] != rows.size:
            raise ValueError(f"{path} contains duplicate sampled pixels in {view_ids[view_index]!r}")

    source_view_configs = _string_vector(
        arrays["source_view_configs"], "source_view_configs", path
    )
    source_cache_dirs = _string_vector(
        arrays["source_flow_cache_dirs"], "source_flow_cache_dirs", path
    )
    cache_identities = _string_vector(
        arrays["source_flow_cache_identities"], "source_flow_cache_identities", path
    )
    if not (
        len(source_view_configs)
        == len(source_cache_dirs)
        == len(cache_identities)
        == num_views
    ):
        raise ValueError(f"{path} source view/cache provenance must have one entry per view")

    rasterizer = _scalar_string(arrays["rasterizer"], "rasterizer", path)
    if rasterizer not in {"gsplat_3dgs", "gsplat_2dgs"}:
        raise ValueError(f"{path} has unsupported rasterizer {rasterizer!r}")
    normalization = _scalar_string(arrays["normalization"], "normalization", path)
    if normalization != RENDERED_MODAL_DESIGN_NORMALIZATION:
        raise ValueError(f"{path} has unsupported design normalization")
    artifact_identity = _scalar_string(
        arrays["artifact_identity"], "artifact_identity", path
    )
    expected_identity = _artifact_identity(
        {name: value for name, value in arrays.items() if name != "artifact_identity"}
    )
    if artifact_identity != expected_identity:
        raise ValueError(f"{path} artifact_identity does not match its contents")

    return RenderedModalDesign(
        path=path,
        view_ids=view_ids,
        view_image_width=widths.astype(np.int64, copy=False),
        view_image_height=heights.astype(np.int64, copy=False),
        mode_indices=mode_indices.astype(np.int64, copy=False),
        frequencies_hz=frequencies,
        sample_view_index=sample_view_index.astype(np.int64, copy=False),
        sample_pixels_xy=pixels.astype(np.int64, copy=False),
        design_matrix=design,
        sampled_alpha=sampled_alpha,
        source_checkpoint=Path(_scalar_string(arrays["source_checkpoint"], "source_checkpoint", path)),
        source_modal_manifest=Path(
            _scalar_string(arrays["source_modal_manifest"], "source_modal_manifest", path)
        ),
        source_view_configs=tuple(Path(value) for value in source_view_configs),
        source_flow_cache_dirs=tuple(Path(value) for value in source_cache_dirs),
        source_flow_cache_identities=cache_identities,
        checkpoint_identity=_scalar_string(
            arrays["checkpoint_identity"], "checkpoint_identity", path
        ),
        gaussian_identity=_scalar_string(arrays["gaussian_identity"], "gaussian_identity", path),
        phi_identity=_scalar_string(arrays["phi_identity"], "phi_identity", path),
        camera_identity=_scalar_string(arrays["camera_identity"], "camera_identity", path),
        artifact_identity=artifact_identity,
        pixel_sample_stride=stride,
        alpha_min=alpha_min,
        mask_erode_iters=mask_erode_iters,
        modes_per_batch=modes_per_batch,
        rasterizer=rasterizer,
        normalization=normalization,
    )


def _parse_flow_cache_specs(
    specs: Sequence[str | tuple[str, str | Path]],
) -> tuple[tuple[str, Path], ...]:
    if not specs:
        raise ValueError("At least one ordered flow cache is required")
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        if isinstance(spec, tuple):
            if len(spec) != 2:
                raise ValueError(f"Flow cache tuple must be (VIEW_ID, PATH), got {spec!r}")
            view_id, raw_path = spec
        else:
            if not isinstance(spec, str) or spec.count("=") != 1:
                raise ValueError(f"Flow cache specification must be VIEW_ID=PATH, got {spec!r}")
            view_id, raw_path = spec.split("=", 1)
        if not isinstance(view_id, str) or not view_id or not str(raw_path):
            raise ValueError(f"Flow cache specification must be VIEW_ID=PATH, got {spec!r}")
        if view_id in seen:
            raise ValueError(f"Duplicate flow cache view_id {view_id!r}")
        seen.add(view_id)
        parsed.append((view_id, Path(raw_path).expanduser().resolve(strict=True)))
    return tuple(parsed)


def _identity_for_gaussians(scene_model: Any) -> str:
    hasher = hashlib.sha256()
    values = {
        "means": scene_model.fg.params["means"],
        "scales": scene_model.fg.get_scales(),
        "quaternions": scene_model.fg.get_quats(),
        "opacities": scene_model.fg.get_opacities(),
    }
    for name, tensor in values.items():
        _hash_array(hasher, name, tensor.detach().cpu().float().numpy())
    return hasher.hexdigest()


def _identity_for_phi(
    mode_indices: np.ndarray, frequencies_hz: np.ndarray, phi: np.ndarray
) -> str:
    hasher = hashlib.sha256()
    _hash_array(hasher, "mode_indices", mode_indices)
    _hash_array(hasher, "frequencies_hz", frequencies_hz)
    _hash_array(hasher, "phi", phi)
    return hasher.hexdigest()


def _identity_for_cameras(view_ids: tuple[str, ...], configs: Sequence[Any]) -> str:
    hasher = hashlib.sha256()
    _hash_array(hasher, "view_ids", np.asarray(view_ids))
    for view_id, config in zip(view_ids, configs, strict=True):
        _hash_array(hasher, f"{view_id}.image_size", np.asarray([config.image_width, config.image_height], dtype=np.int64))
        _hash_array(hasher, f"{view_id}.K", np.asarray(config.K, dtype=np.float64))
        _hash_array(
            hasher,
            f"{view_id}.world_to_camera",
            np.asarray(config.world_to_camera, dtype=np.float64),
        )
    return hasher.hexdigest()


def _render_feature_samples(
    scene_model: Any,
    features: Any,
    w2c: Any,
    K: Any,
    image_size: tuple[int, int],
    sample_pixels_xy: np.ndarray,
    sampled_alpha: np.ndarray,
) -> np.ndarray:
    import torch

    rendered = scene_model.render(
        None,
        w2c,
        K,
        image_size,
        bg_color=0.0,
        colors_override=features,
        fg_only=True,
    )
    image = rendered["img"][0]
    alpha = rendered["acc"][0]
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    y = torch.as_tensor(sample_pixels_xy[:, 1], device=image.device, dtype=torch.long)
    x = torch.as_tensor(sample_pixels_xy[:, 0], device=image.device, dtype=torch.long)
    rendered_alpha = alpha[y, x].detach().cpu().float().numpy()
    if not np.allclose(rendered_alpha, sampled_alpha, rtol=2e-5, atol=2e-6):
        raise RuntimeError("Feature rendering changed foreground alpha unexpectedly")
    values = image[y, x].detach().cpu().float().numpy()
    return values / sampled_alpha[:, None]


def _design_summary(
    view_ids: tuple[str, ...],
    sample_view_index: np.ndarray,
    sampled_alpha: np.ndarray,
    design_matrix: np.ndarray,
    direct_render_max_abs_error: Sequence[float],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for view_index, view_id in enumerate(view_ids):
        rows = np.flatnonzero(sample_view_index == view_index)
        design = design_matrix[rows].reshape(-1, design_matrix.shape[-1]).astype(np.float64)
        singular_values = np.linalg.svd(design, compute_uv=False)
        tolerance = (
            float(singular_values[0])
            * max(design.shape)
            * np.finfo(np.float64).eps
            if singular_values.size
            else 0.0
        )
        rank = int(np.count_nonzero(singular_values > tolerance))
        condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values.size and singular_values[-1] > tolerance
            else None
        )
        rng = np.random.default_rng(1831 + view_index)
        known_coordinates = rng.standard_normal(design.shape[1])
        synthetic_flow = design @ known_coordinates
        recovered_coordinates, _, recovery_rank, _ = np.linalg.lstsq(
            design, synthetic_flow, rcond=None
        )
        recovery_relative_error = None
        if recovery_rank == design.shape[1]:
            recovery_relative_error = float(
                np.linalg.norm(recovered_coordinates - known_coordinates)
                / np.linalg.norm(known_coordinates)
            )
            if recovery_relative_error > 1e-7:
                raise RuntimeError(
                    f"Synthetic coordinate recovery for {view_id!r} has relative "
                    f"error {recovery_relative_error:.6g}, above 1e-7"
                )
        alpha = sampled_alpha[rows].astype(np.float64)
        summaries.append(
            {
                "view_id": view_id,
                "sample_count": int(rows.size),
                "alpha": {
                    "min": float(alpha.min()),
                    "p10": float(np.percentile(alpha, 10.0)),
                    "median": float(np.median(alpha)),
                    "p90": float(np.percentile(alpha, 90.0)),
                    "max": float(alpha.max()),
                    "mean": float(alpha.mean()),
                },
                "design_column_rms": np.sqrt(np.mean(design * design, axis=0)).tolist(),
                "singular_values": singular_values.tolist(),
                "numerical_rank": rank,
                "condition_number": condition,
                "verification": {
                    "direct_render_max_abs_error": float(
                        direct_render_max_abs_error[view_index]
                    ),
                    "synthetic_coordinate_recovery_rank": int(recovery_rank),
                    "synthetic_coordinate_recovery_relative_error": (
                        recovery_relative_error
                    ),
                },
            }
        )
    return summaries


def _validate_role_fractions(values: np.ndarray, context: str) -> float:
    fractions = np.asarray(values)
    if not np.isfinite(fractions).all():
        raise RuntimeError(f"Role contributor fractions for {context} are not finite")
    tolerance = 5e-4
    minimum = float(np.min(fractions))
    maximum = float(np.max(fractions))
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise RuntimeError(
            f"Role contributor fractions for {context} leave [0,1]: "
            f"min={minimum:.6g}, max={maximum:.6g}"
        )
    sum_error = float(np.max(np.abs(fractions.sum(axis=2) - 1.0)))
    if sum_error > tolerance:
        raise RuntimeError(
            f"Role contributor fractions for {context} do not sum to one; "
            f"max error={sum_error:.6g}"
        )
    return sum_error


def _write_role_overview(
    path: Path,
    view_ids: tuple[str, ...],
    widths: np.ndarray,
    heights: np.ndarray,
    sample_view_index: np.ndarray,
    pixels: np.ndarray,
    role_fraction: np.ndarray,
    role_names: tuple[str, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(view_ids),
        len(role_names),
        figsize=(3.2 * len(role_names), 3.0 * len(view_ids)),
        squeeze=False,
    )
    averaged = role_fraction.mean(axis=1)
    for view_index, view_id in enumerate(view_ids):
        rows = np.flatnonzero(sample_view_index == view_index)
        for role_index, role_name in enumerate(role_names):
            image = np.full(
                (int(heights[view_index]), int(widths[view_index])), np.nan, dtype=np.float32
            )
            view_pixels = pixels[rows]
            image[view_pixels[:, 1], view_pixels[:, 0]] = averaged[rows, role_index]
            axis = axes[view_index, role_index]
            rendered = axis.imshow(image, vmin=0.0, vmax=1.0, cmap="viridis")
            axis.set_title(f"{view_id}: {role_name}")
            axis.set_axis_off()
            fig.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_rendered_modal_design(
    input_ckpt: str | Path,
    modal_manifest: str | Path,
    view_configs: Sequence[str | Path],
    flow_caches: Sequence[str | tuple[str, str | Path]],
    out_dir: str | Path,
    *,
    pixel_sample_stride: int = 2,
    alpha_min: float = 0.05,
    mask_erode_iters: int = 1,
    modes_per_batch: int = 8,
    use_2dgs: bool = False,
    write_role_diagnostics: bool = False,
) -> RenderedModalDesign:
    """Render the complete foreground modal projection into flow-space columns."""

    import torch

    from flow3d.modal_utils import (
        MOTION_FILL_DISPLAY_NAMES,
        load_gaussian_modal_fields,
        stack_modal_motion_fill_display_classes,
    )
    from flow3d.scene_model import SceneModel

    if (
        isinstance(pixel_sample_stride, bool)
        or not isinstance(pixel_sample_stride, (int, np.integer))
        or pixel_sample_stride <= 0
    ):
        raise ValueError("pixel_sample_stride must be a positive integer")
    if (
        isinstance(mask_erode_iters, bool)
        or not isinstance(mask_erode_iters, (int, np.integer))
        or mask_erode_iters < 0
    ):
        raise ValueError("mask_erode_iters must be a non-negative integer")
    if (
        isinstance(modes_per_batch, bool)
        or not isinstance(modes_per_batch, (int, np.integer))
        or modes_per_batch <= 0
    ):
        raise ValueError("modes_per_batch must be a positive integer")
    if not np.isfinite(alpha_min) or not 0.0 < alpha_min <= 1.0:
        raise ValueError("alpha_min must lie in (0,1]")
    if not view_configs:
        raise ValueError("At least one view config is required")
    if not torch.cuda.is_available():
        raise RuntimeError("Rendered modal design construction requires a CUDA device")

    checkpoint_path = Path(input_ckpt).expanduser().resolve(strict=True)
    manifest_path = Path(modal_manifest).expanduser().resolve(strict=True)
    target = Path(out_dir).expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Rendered modal design target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    config_paths = tuple(
        Path(value).expanduser().resolve(strict=True) for value in view_configs
    )
    configs = tuple(load_view_config(path) for path in config_paths)
    view_ids = tuple(config.view_id for config in configs)
    if len(set(view_ids)) != len(view_ids):
        raise ValueError("View configs must contain unique view IDs")
    for config in configs:
        if config.image_width < 3 or config.image_height < 3:
            raise ValueError(f"View {config.view_id!r} image dimensions must be at least 3x3")
        if not np.isfinite(config.K).all() or not np.isfinite(
            config.world_to_camera
        ).all():
            raise ValueError(f"View {config.view_id!r} camera matrices must be finite")
    cache_specs = _parse_flow_cache_specs(flow_caches)
    cache_view_ids = tuple(view_id for view_id, _ in cache_specs)
    if cache_view_ids != view_ids:
        raise ValueError(
            f"Flow cache order must exactly match view configs: expected {view_ids}, got {cache_view_ids}"
        )

    caches = tuple(load_analysis_cache(cache_path) for _, cache_path in cache_specs)
    cache_paths = tuple(cache.path.resolve() for cache in caches)
    if len(set(cache_paths)) != len(cache_paths):
        raise ValueError("A flow cache directory cannot be reused by multiple views")
    cache_identities = tuple(compute_flow_cache_identity(path) for path in cache_paths)
    for config, cache in zip(configs, caches, strict=True):
        expected_shape = (config.image_height, config.image_width)
        if cache.flow_u.shape[1:] != expected_shape:
            raise ValueError(
                f"Flow cache for {config.view_id!r} has shape {cache.flow_u.shape[1:]}, expected {expected_shape}"
            )
        if cache.mask is None:
            raise ValueError(f"Flow cache for {config.view_id!r} must contain a foreground mask")
        if cache.metadata["analysis"]["flow_method"] != "farneback":
            raise ValueError(f"Flow cache for {config.view_id!r} must use Farneback flow")

    with torch.no_grad():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise ValueError(f"{checkpoint_path} does not contain a model state")
        device = torch.device("cuda")
        scene_model = SceneModel.init_from_state_dict(state).to(device)
        scene_model.eval()
        scene_model.use_2dgs = bool(use_2dgs)
        modal_fields = load_gaussian_modal_fields(
            str(manifest_path), scene_model.fg.params["means"]
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_payload = json.load(handle)
    raw_source_checkpoint = manifest_payload.get("source_checkpoint")
    if not isinstance(raw_source_checkpoint, str) or not raw_source_checkpoint:
        raise ValueError(f"{manifest_path} is missing source_checkpoint")
    manifest_checkpoint = Path(raw_source_checkpoint).expanduser()
    if not manifest_checkpoint.is_absolute():
        manifest_checkpoint = manifest_path.parent / manifest_checkpoint
    manifest_checkpoint = manifest_checkpoint.resolve(strict=True)
    if manifest_checkpoint != checkpoint_path:
        raise ValueError(
            f"{manifest_path} source_checkpoint does not identify {checkpoint_path}"
        )

    mode_indices = np.asarray(
        [mode.mode_index for mode in modal_fields.modes], dtype=np.int64
    )
    frequencies_hz = modal_fields.freqs_hz.detach().cpu().double().numpy()
    phi = (
        modal_fields.phi_real.detach().cpu().float().numpy()
        + 1j * modal_fields.phi_imag.detach().cpu().float().numpy()
    ).astype(np.complex64)
    num_modes, num_gaussians, _ = phi.shape
    if num_modes == 0 or num_gaussians != scene_model.num_fg_gaussians:
        raise ValueError("Modal manifest does not provide a complete foreground Gaussian field")

    checkpoint_identity = _hash_file(checkpoint_path)
    gaussian_identity = _identity_for_gaussians(scene_model)
    phi_identity = _identity_for_phi(mode_indices, frequencies_hz, phi)
    camera_identity = _identity_for_cameras(view_ids, configs)

    widths = np.asarray([config.image_width for config in configs], dtype=np.int64)
    heights = np.asarray([config.image_height for config in configs], dtype=np.int64)
    all_view_indices: list[np.ndarray] = []
    all_pixels: list[np.ndarray] = []
    all_alpha: list[np.ndarray] = []
    all_design: list[np.ndarray] = []
    direct_render_max_abs_error: list[float] = []
    role_classes = stack_modal_motion_fill_display_classes(modal_fields.modes)
    if write_role_diagnostics and role_classes is None:
        raise ValueError(
            "--write-role-diagnostics requires motion-fill role metadata in every modal latent"
        )
    all_role_fraction: list[np.ndarray] = []

    for view_index, (config, cache) in enumerate(zip(configs, caches, strict=True)):
        w2c = torch.as_tensor(
            config.world_to_camera, device=device, dtype=torch.float32
        )[None]
        K = torch.as_tensor(config.K, device=device, dtype=torch.float32)[None]
        dummy = torch.zeros(
            (num_gaussians, 1), device=device, dtype=scene_model.fg.params["means"].dtype
        )
        with torch.no_grad():
            alpha_render = scene_model.render(
                None,
                w2c,
                K,
                (config.image_width, config.image_height),
                bg_color=0.0,
                colors_override=dummy,
                fg_only=True,
            )
        alpha_tensor = alpha_render["acc"][0]
        if alpha_tensor.ndim == 3:
            alpha_tensor = alpha_tensor[..., 0]
        alpha_image = alpha_tensor.detach().cpu().float().numpy()
        if alpha_image.shape != (config.image_height, config.image_width):
            raise RuntimeError(
                f"Rendered alpha for {config.view_id!r} has unexpected shape {alpha_image.shape}"
            )
        if not np.isfinite(alpha_image).all():
            raise RuntimeError(f"Rendered alpha for {config.view_id!r} is not finite")
        mask = erode_mask(np.asarray(cache.mask, dtype=bool), mask_erode_iters)
        candidate = mask & (alpha_image >= alpha_min)
        grid = np.zeros(candidate.shape, dtype=bool)
        grid[1:-1:pixel_sample_stride, 1:-1:pixel_sample_stride] = True
        y, x = np.where(candidate & grid)
        if x.size == 0:
            raise ValueError(
                f"View {config.view_id!r} has no pixels after mask, alpha, and stride filtering"
            )
        pixels = np.stack([x, y], axis=1).astype(np.int64)
        sampled_alpha = alpha_image[y, x].astype(np.float32)

        means_np = (
            scene_model.fg.params["means"].detach().cpu().float().numpy()
        )
        camera_z = world_to_camera_points(
            means_np, config.world_to_camera
        )[:, 2]
        visible_depth = camera_z > 1e-8
        if not np.any(visible_depth):
            raise ValueError(f"All foreground Gaussians lie behind view {config.view_id!r}")
        jacobian = np.zeros((num_gaussians, 2, 3), dtype=np.float32)
        jacobian[visible_depth] = projection_jacobian(
            means_np[visible_depth], config.K, config.world_to_camera
        )
        jacobian_t = torch.as_tensor(jacobian, device=device, dtype=torch.float32)
        design = np.empty((pixels.shape[0], 2, 2 * num_modes), dtype=np.float32)
        for start in range(0, num_modes, modes_per_batch):
            stop = min(start + modes_per_batch, num_modes)
            phi_real = modal_fields.phi_real[start:stop]
            phi_imag = modal_fields.phi_imag[start:stop]
            projected_real = torch.einsum("gij,kgj->kgi", jacobian_t, phi_real)
            projected_imag = torch.einsum("gij,kgj->kgi", jacobian_t, phi_imag)
            features = torch.stack(
                (
                    projected_real[..., 0],
                    projected_real[..., 1],
                    -projected_imag[..., 0],
                    -projected_imag[..., 1],
                ),
                dim=-1,
            ).permute(1, 0, 2).reshape(num_gaussians, -1).contiguous()
            with torch.no_grad():
                values = _render_feature_samples(
                    scene_model,
                    features,
                    w2c,
                    K,
                    (config.image_width, config.image_height),
                    pixels,
                    sampled_alpha,
                )
            values = values.reshape(pixels.shape[0], stop - start, 4)
            for local_mode, mode_slot in enumerate(range(start, stop)):
                design[:, 0, 2 * mode_slot] = values[:, local_mode, 0]
                design[:, 1, 2 * mode_slot] = values[:, local_mode, 1]
                design[:, 0, 2 * mode_slot + 1] = values[:, local_mode, 2]
                design[:, 1, 2 * mode_slot + 1] = values[:, local_mode, 3]
            del features, projected_real, projected_imag

        if not np.isfinite(design).all():
            raise RuntimeError(f"Rendered design for {config.view_id!r} contains non-finite values")
        rng = np.random.default_rng(1729 + view_index)
        packed_coordinates = rng.standard_normal(2 * num_modes).astype(np.float32)
        packed_coordinates /= np.sqrt(float(np.mean(packed_coordinates**2)))
        q_real = torch.as_tensor(
            packed_coordinates[0::2], device=device, dtype=torch.float32
        )
        q_imag = torch.as_tensor(
            packed_coordinates[1::2], device=device, dtype=torch.float32
        )
        with torch.no_grad():
            direct_features = torch.einsum(
                "gij,kgj,k->gi", jacobian_t, modal_fields.phi_real, q_real
            ) - torch.einsum(
                "gij,kgj,k->gi", jacobian_t, modal_fields.phi_imag, q_imag
            )
            direct_values = _render_feature_samples(
                scene_model,
                direct_features,
                w2c,
                K,
                (config.image_width, config.image_height),
                pixels,
                sampled_alpha,
            )
        design_values = np.einsum(
            "pdc,c->pd",
            design.astype(np.float64),
            packed_coordinates.astype(np.float64),
            optimize=True,
        )
        direct_values_float64 = direct_values.astype(np.float64)
        direct_difference = design_values - direct_values_float64
        direct_error = float(np.max(np.abs(direct_difference)))
        direct_relative_l2_error = float(
            np.linalg.norm(direct_difference.reshape(-1))
            / max(
                np.linalg.norm(direct_values_float64.reshape(-1)),
                np.finfo(np.float64).eps,
            )
        )
        # Gsplat can accumulate a packed feature render and a direct two-channel
        # render in different float32 orders. Reject only discrepancies that are
        # significant both absolutely and relative to the rendered signal.
        if (
            direct_error > _PACKING_MAX_ABS_TOLERANCE
            and direct_relative_l2_error > _PACKING_RELATIVE_L2_TOLERANCE
        ):
            raise RuntimeError(
                f"Rendered design packing check failed for {config.view_id!r}; "
                f"max absolute error={direct_error:.6g}, "
                f"relative L2 error={direct_relative_l2_error:.6g}"
            )
        direct_render_max_abs_error.append(direct_error)
        del direct_features, direct_values
        all_view_indices.append(np.full(pixels.shape[0], view_index, dtype=np.int64))
        all_pixels.append(pixels)
        all_alpha.append(sampled_alpha)
        all_design.append(design)

        if role_classes is not None and write_role_diagnostics:
            role_fraction = np.empty(
                (pixels.shape[0], num_modes, len(MOTION_FILL_DISPLAY_NAMES)),
                dtype=np.float32,
            )
            for start in range(0, num_modes, modes_per_batch):
                stop = min(start + modes_per_batch, num_modes)
                batch_roles = torch.as_tensor(
                    role_classes[start:stop], device=device, dtype=torch.long
                )
                one_hot = torch.nn.functional.one_hot(
                    batch_roles, num_classes=len(MOTION_FILL_DISPLAY_NAMES)
                ).to(dtype=torch.float32)
                features = one_hot.permute(1, 0, 2).reshape(num_gaussians, -1).contiguous()
                with torch.no_grad():
                    values = _render_feature_samples(
                        scene_model,
                        features,
                        w2c,
                        K,
                        (config.image_width, config.image_height),
                        pixels,
                        sampled_alpha,
                    )
                role_fraction[:, start:stop] = values.reshape(
                    pixels.shape[0], stop - start, len(MOTION_FILL_DISPLAY_NAMES)
                )
            _validate_role_fractions(role_fraction, repr(config.view_id))
            all_role_fraction.append(role_fraction)

    sample_view_index = np.concatenate(all_view_indices, axis=0)
    sample_pixels_xy = np.concatenate(all_pixels, axis=0)
    sampled_alpha = np.concatenate(all_alpha, axis=0)
    design_matrix = np.concatenate(all_design, axis=0)

    arrays: dict[str, np.ndarray] = {
        "format": np.array(RENDERED_MODAL_DESIGN_FORMAT),
        "version": np.array(RENDERED_MODAL_DESIGN_VERSION, dtype=np.int32),
        "view_ids": np.asarray(view_ids),
        "view_image_width": widths,
        "view_image_height": heights,
        "mode_indices": mode_indices,
        "frequencies_hz": frequencies_hz.astype(np.float64),
        "sample_view_index": sample_view_index,
        "sample_pixels_xy": sample_pixels_xy,
        "design_matrix": design_matrix.astype(np.float32, copy=False),
        "sampled_alpha": sampled_alpha.astype(np.float32, copy=False),
        "source_checkpoint": np.array(str(checkpoint_path)),
        "source_modal_manifest": np.array(str(manifest_path)),
        "source_view_configs": np.asarray([str(path) for path in config_paths]),
        "source_flow_cache_dirs": np.asarray([str(path) for path in cache_paths]),
        "source_flow_cache_identities": np.asarray(cache_identities),
        "checkpoint_identity": np.array(checkpoint_identity),
        "gaussian_identity": np.array(gaussian_identity),
        "phi_identity": np.array(phi_identity),
        "camera_identity": np.array(camera_identity),
        "pixel_sample_stride": np.array(pixel_sample_stride, dtype=np.int32),
        "alpha_min": np.array(alpha_min, dtype=np.float64),
        "mask_erode_iters": np.array(mask_erode_iters, dtype=np.int32),
        "modes_per_batch": np.array(modes_per_batch, dtype=np.int32),
        "rasterizer": np.array("gsplat_2dgs" if use_2dgs else "gsplat_3dgs"),
        "normalization": np.array(RENDERED_MODAL_DESIGN_NORMALIZATION),
    }
    artifact_identity = _artifact_identity(arrays)
    arrays["artifact_identity"] = np.array(artifact_identity)

    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    )
    try:
        artifact_path = temp_path / RENDERED_MODAL_DESIGN_FILENAME
        np.savez_compressed(artifact_path, **arrays)
        summary: dict[str, Any] = {
            "format": RENDERED_MODAL_DESIGN_FORMAT,
            "version": RENDERED_MODAL_DESIGN_VERSION,
            "artifact_identity": artifact_identity,
            "artifact": RENDERED_MODAL_DESIGN_FILENAME,
            "mode_count": int(num_modes),
            "foreground_gaussian_count": int(num_gaussians),
            "sample_count": int(sample_view_index.size),
            "settings": {
                "pixel_sample_stride": int(pixel_sample_stride),
                "alpha_min": float(alpha_min),
                "mask_erode_iters": int(mask_erode_iters),
                "modes_per_batch": int(modes_per_batch),
                "rasterizer": "gsplat_2dgs" if use_2dgs else "gsplat_3dgs",
                "normalization": RENDERED_MODAL_DESIGN_NORMALIZATION,
                "feature_packing": "[real_u,real_v,-imag_u,-imag_v]_per_mode",
            },
            "sources": {
                "checkpoint": str(checkpoint_path),
                "modal_manifest": str(manifest_path),
                "view_configs": [str(path) for path in config_paths],
                "flow_cache_dirs": [str(path) for path in cache_paths],
                "flow_cache_identities": list(cache_identities),
            },
            "identities": {
                "checkpoint": checkpoint_identity,
                "gaussian": gaussian_identity,
                "phi": phi_identity,
                "camera": camera_identity,
            },
            "views": _design_summary(
                view_ids,
                sample_view_index,
                sampled_alpha,
                design_matrix,
                direct_render_max_abs_error,
            ),
        }
        if write_role_diagnostics:
            role_fraction = np.concatenate(all_role_fraction, axis=0)
            role_sum_error = _validate_role_fractions(
                role_fraction, "all rendered-design views"
            )
            np.savez_compressed(
                temp_path / ROLE_DIAGNOSTICS_FILENAME,
                format=np.array("rendered_modal_role_diagnostics"),
                version=np.array(1, dtype=np.int32),
                rendered_design_identity=np.array(artifact_identity),
                mode_indices=mode_indices,
                view_ids=np.asarray(view_ids),
                sample_view_index=sample_view_index,
                sample_pixels_xy=sample_pixels_xy,
                role_names=np.asarray(MOTION_FILL_DISPLAY_NAMES),
                role_fraction=role_fraction.astype(np.float32, copy=False),
                role_sum_max_abs_error=np.array(role_sum_error, dtype=np.float64),
            )
            _write_role_overview(
                temp_path / ROLE_OVERVIEW_FILENAME,
                view_ids,
                widths,
                heights,
                sample_view_index,
                sample_pixels_xy,
                role_fraction,
                tuple(MOTION_FILL_DISPLAY_NAMES),
            )
            summary["role_diagnostics"] = {
                "artifact": ROLE_DIAGNOSTICS_FILENAME,
                "overview": ROLE_OVERVIEW_FILENAME,
                "role_names": list(MOTION_FILL_DISPLAY_NAMES),
                "sum_max_abs_error": role_sum_error,
            }
        with (temp_path / RENDERED_MODAL_DESIGN_SUMMARY_FILENAME).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

        validated = load_rendered_modal_design(artifact_path)
        del validated
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Rendered modal design target already exists: {target}")
        os.replace(temp_path, target)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise
    return load_rendered_modal_design(target / RENDERED_MODAL_DESIGN_FILENAME)
