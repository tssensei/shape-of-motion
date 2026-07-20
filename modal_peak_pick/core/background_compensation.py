from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from modal_peak_pick.core.flow import compute_dense_flow_pair


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TRANSFORM_MODELS = ("similarity", "affine", "homography")


@dataclass(frozen=True)
class BackgroundCompensationSettings:
    mask_dilate_px: int = 16
    transform_model: str = "similarity"
    max_corners: int = 4000
    feature_quality: float = 0.01
    feature_min_distance_px: float = 8.0
    lk_window_px: int = 31
    lk_max_level: int = 4
    max_forward_backward_error_px: float = 1.5
    ransac_threshold_px: float = 2.0
    min_inliers: int = 30
    min_inlier_fraction: float = 0.1
    min_background_coverage: float = 0.05
    min_candidate_valid_fraction: float = 0.99


@dataclass(frozen=True)
class TransformEstimate:
    matrix: np.ndarray
    tracked_count: int
    inlier_count: int
    inlier_fraction: float
    forward_backward_rmse: float
    reprojection_rmse: float
    spatial_coverage: float
    source_points: np.ndarray
    target_points: np.ndarray
    inlier_mask: np.ndarray


@dataclass(frozen=True)
class CompensatedFlowResult:
    flow_u: np.ndarray
    flow_v: np.ndarray
    raw_reference_to_stabilized_reference: np.ndarray
    raw_reference_to_raw_frame: np.ndarray
    raw_frame_to_stabilized_reference: np.ndarray
    residual_background_transform: np.ndarray
    tracked_count: np.ndarray
    inlier_count: np.ndarray
    inlier_fraction: np.ndarray
    forward_backward_rmse: np.ndarray
    reprojection_rmse: np.ndarray
    spatial_coverage: np.ndarray
    valid_fraction: np.ndarray
    candidate_valid_fraction: np.ndarray
    residual_background_rms: np.ndarray
    residual_background_p90: np.ndarray
    camera_flow_foreground_rms: np.ndarray
    residual_camera_flow_foreground_rms: np.ndarray
    aligned_preview_indices: np.ndarray
    aligned_previews: np.ndarray


def validate_settings(settings: BackgroundCompensationSettings) -> None:
    if settings.mask_dilate_px < 0:
        raise ValueError("mask_dilate_px must be non-negative")
    if settings.transform_model not in TRANSFORM_MODELS:
        raise ValueError(
            f"transform_model must be one of {TRANSFORM_MODELS}, "
            f"got {settings.transform_model!r}"
        )
    if settings.max_corners < 4:
        raise ValueError("max_corners must be at least four")
    if not 0.0 < settings.feature_quality < 1.0:
        raise ValueError("feature_quality must be in (0,1)")
    if settings.feature_min_distance_px <= 0.0:
        raise ValueError("feature_min_distance_px must be positive")
    if settings.lk_window_px < 3 or settings.lk_window_px % 2 == 0:
        raise ValueError("lk_window_px must be an odd integer at least three")
    if settings.lk_max_level < 0:
        raise ValueError("lk_max_level must be non-negative")
    for name in (
        "max_forward_backward_error_px",
        "ransac_threshold_px",
    ):
        value = float(getattr(settings, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if settings.min_inliers < 4:
        raise ValueError("min_inliers must be at least four")
    for name in (
        "min_inlier_fraction",
        "min_background_coverage",
        "min_candidate_valid_fraction",
    ):
        value = float(getattr(settings, name))
        if not np.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in (0,1]")


def load_frame_names(path: str | Path) -> tuple[str, ...]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Frame-name sidecar does not exist: {source}")
    with source.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Frame-name sidecar must be a non-empty JSON list: {source}")
    names: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(payload):
        if not isinstance(value, str) or not value or Path(value).name != value:
            raise ValueError(
                f"Frame-name sidecar entry {index} must be a filename stem: {source}"
            )
        if Path(value).suffix:
            raise ValueError(
                f"Frame-name sidecar entry must not contain an extension: {value!r}"
            )
        if value in seen:
            raise ValueError(f"Duplicate frame name {value!r} in {source}")
        seen.add(value)
        names.append(value)
    return tuple(names)


def _index_mask_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Foreground mask directory does not exist: {directory}")
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in indexed:
            raise ValueError(
                f"Foreground mask directory repeats stem {path.stem!r}: "
                f"{indexed[path.stem]} and {path}"
            )
        indexed[path.stem] = path
    if not indexed:
        raise ValueError(f"Foreground mask directory contains no images: {directory}")
    return indexed


def load_ordered_foreground_masks(
    mask_dir: str | Path,
    frame_names: Sequence[str],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    directory = Path(mask_dir).expanduser()
    indexed = _index_mask_images(directory)
    expected = set(frame_names)
    actual = set(indexed)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if extra:
            details.append(f"extra={extra[:5]}")
        raise ValueError(
            "Foreground mask names do not match the frame-name sidecar: "
            + ", ".join(details)
        )
    masks = np.empty((len(frame_names), height, width), dtype=np.uint8)
    for index, frame_name in enumerate(frame_names):
        image = cv2.imread(str(indexed[frame_name]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to decode foreground mask: {indexed[frame_name]}")
        if image.ndim == 2:
            binary = image > (0.5 if np.issubdtype(image.dtype, np.floating) else 127)
        else:
            binary = np.any(image > 0, axis=2)
        if binary.shape != (height, width):
            binary = cv2.resize(
                binary.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        masks[index] = binary.astype(np.uint8)
    return masks


def dilate_foreground_masks(masks: np.ndarray, radius_px: int) -> np.ndarray:
    if masks.ndim != 3:
        raise ValueError("Foreground masks must be [T,H,W]")
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    binary = (masks > 0).astype(np.uint8)
    if radius_px == 0:
        return binary
    size = 2 * int(radius_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    result = np.empty_like(binary)
    for index in range(binary.shape[0]):
        result[index] = cv2.dilate(binary[index], kernel, iterations=1)
    return result


def _as_u8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 2:
        raise ValueError("Feature-tracking frames must be [H,W]")
    if not np.isfinite(frame).all():
        raise ValueError("Feature-tracking frames must be finite")
    return np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)


def _sample_mask(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    rounded = np.rint(points).astype(np.int64)
    valid = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    result = np.zeros((points.shape[0],), dtype=bool)
    if np.any(valid):
        result[valid] = mask[rounded[valid, 1], rounded[valid, 0]] > 0
    return result


def _spatial_coverage(points: np.ndarray, height: int, width: int) -> float:
    if points.shape[0] < 2:
        return 0.0
    extent = np.ptp(points, axis=0)
    return float((extent[0] * extent[1]) / max(1.0, float((width - 1) * (height - 1))))


def apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    if matrix.shape != (3, 3):
        raise ValueError("Homography matrix must be [3,3]")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must be [N,2]")
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = homogeneous @ matrix.astype(np.float64).T
    denominator = transformed[:, 2]
    if np.any(np.abs(denominator) <= np.finfo(np.float64).eps):
        raise ValueError("Homography sends points to infinity")
    return transformed[:, :2] / denominator[:, None]


def _fit_transform(
    source: np.ndarray,
    target: np.ndarray,
    model: str,
    ransac_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    if model == "similarity":
        fitted, inliers = cv2.estimateAffinePartial2D(
            source.astype(np.float32),
            target.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_threshold_px),
            maxIters=5000,
            confidence=0.999,
            refineIters=10,
        )
        if fitted is None:
            raise ValueError("RANSAC failed to estimate a similarity transform")
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2] = fitted
    elif model == "affine":
        fitted, inliers = cv2.estimateAffine2D(
            source.astype(np.float32),
            target.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_threshold_px),
            maxIters=5000,
            confidence=0.999,
            refineIters=10,
        )
        if fitted is None:
            raise ValueError("RANSAC failed to estimate an affine transform")
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2] = fitted
    elif model == "homography":
        fitted, inliers = cv2.findHomography(
            source.astype(np.float32),
            target.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_threshold_px),
            maxIters=5000,
            confidence=0.999,
        )
        if fitted is None:
            raise ValueError("RANSAC failed to estimate a homography")
        matrix = fitted.astype(np.float64)
    else:
        raise ValueError(f"Unsupported transform model {model!r}")
    if inliers is None:
        raise ValueError("RANSAC did not return an inlier mask")
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) <= 1e-12:
        raise ValueError("Estimated background transform is non-finite or singular")
    scale = float(matrix[2, 2])
    if abs(scale) <= np.finfo(np.float64).eps:
        scale = float(np.linalg.norm(matrix))
    if not np.isfinite(scale) or abs(scale) <= np.finfo(np.float64).eps:
        raise ValueError("Estimated background transform has invalid homogeneous scale")
    return matrix / scale, inliers.reshape(-1).astype(bool)


def estimate_background_transform(
    source_frame: np.ndarray,
    target_frame: np.ndarray,
    source_foreground: np.ndarray,
    target_foreground: np.ndarray,
    settings: BackgroundCompensationSettings,
    *,
    label: str,
) -> TransformEstimate:
    validate_settings(settings)
    if source_frame.shape != target_frame.shape:
        raise ValueError(f"{label}: source and target frame shapes differ")
    if source_foreground.shape != source_frame.shape or target_foreground.shape != source_frame.shape:
        raise ValueError(f"{label}: foreground mask shape does not match frames")
    height, width = source_frame.shape
    source_background = (source_foreground == 0).astype(np.uint8) * 255
    corners = cv2.goodFeaturesToTrack(
        _as_u8(source_frame),
        maxCorners=int(settings.max_corners),
        qualityLevel=float(settings.feature_quality),
        minDistance=float(settings.feature_min_distance_px),
        mask=source_background,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None or corners.shape[0] < settings.min_inliers:
        count = 0 if corners is None else int(corners.shape[0])
        raise ValueError(
            f"{label}: only {count} background features were detected; "
            f"need at least {settings.min_inliers}"
        )
    source_points = corners.reshape(-1, 2).astype(np.float32)
    lk_parameters = {
        "winSize": (int(settings.lk_window_px), int(settings.lk_window_px)),
        "maxLevel": int(settings.lk_max_level),
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        _as_u8(source_frame),
        _as_u8(target_frame),
        source_points.reshape(-1, 1, 2),
        None,
        **lk_parameters,
    )
    if forward is None or forward_status is None:
        raise ValueError(f"{label}: forward LK tracking failed")
    forward_points = forward.reshape(-1, 2)
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        _as_u8(target_frame),
        _as_u8(source_frame),
        forward_points.reshape(-1, 1, 2),
        None,
        **lk_parameters,
    )
    if backward is None or backward_status is None:
        raise ValueError(f"{label}: backward LK tracking failed")
    backward_points = backward.reshape(-1, 2)
    fb_error = np.linalg.norm(backward_points - source_points, axis=1)
    valid = (
        (forward_status.reshape(-1) > 0)
        & (backward_status.reshape(-1) > 0)
        & np.isfinite(forward_points).all(axis=1)
        & np.isfinite(backward_points).all(axis=1)
        & (fb_error <= float(settings.max_forward_backward_error_px))
        & (~_sample_mask(target_foreground, forward_points))
    )
    tracked_source = source_points[valid].astype(np.float64)
    tracked_target = forward_points[valid].astype(np.float64)
    tracked_fb = fb_error[valid].astype(np.float64)
    tracked_count = int(tracked_source.shape[0])
    if tracked_count < settings.min_inliers:
        raise ValueError(
            f"{label}: only {tracked_count} valid background tracks remain; "
            f"need at least {settings.min_inliers}"
        )
    matrix, inliers = _fit_transform(
        tracked_source,
        tracked_target,
        settings.transform_model,
        settings.ransac_threshold_px,
    )
    inlier_count = int(np.count_nonzero(inliers))
    inlier_fraction = float(inlier_count / tracked_count)
    if inlier_count < settings.min_inliers:
        raise ValueError(
            f"{label}: RANSAC retained {inlier_count} inliers; "
            f"need at least {settings.min_inliers}"
        )
    if inlier_fraction < settings.min_inlier_fraction:
        raise ValueError(
            f"{label}: RANSAC inlier fraction {inlier_fraction:.4f} is below "
            f"{settings.min_inlier_fraction:.4f}"
        )
    inlier_source = tracked_source[inliers]
    predicted = apply_homography(matrix, inlier_source)
    residual = np.linalg.norm(predicted - tracked_target[inliers], axis=1)
    coverage = _spatial_coverage(inlier_source, height, width)
    if coverage < settings.min_background_coverage:
        raise ValueError(
            f"{label}: background inlier coverage {coverage:.4f} is below "
            f"{settings.min_background_coverage:.4f}"
        )
    return TransformEstimate(
        matrix=matrix,
        tracked_count=tracked_count,
        inlier_count=inlier_count,
        inlier_fraction=inlier_fraction,
        forward_backward_rmse=float(np.sqrt(np.mean(tracked_fb * tracked_fb))),
        reprojection_rmse=float(np.sqrt(np.mean(residual * residual))),
        spatial_coverage=coverage,
        source_points=tracked_source,
        target_points=tracked_target,
        inlier_mask=inliers,
    )


def _warp_frame(frame: np.ndarray, matrix: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    height, width = output_shape
    return cv2.warpPerspective(
        frame.astype(np.float32, copy=False),
        matrix.astype(np.float64),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _warp_mask(mask: np.ndarray, matrix: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    height, width = output_shape
    return cv2.warpPerspective(
        (mask > 0).astype(np.uint8),
        matrix.astype(np.float64),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def transform_field_rms(
    matrix: np.ndarray,
    sample_mask: np.ndarray,
    *,
    stride: int = 4,
) -> float:
    y, x = np.nonzero(sample_mask[::stride, ::stride])
    if x.size == 0:
        return 0.0
    points = np.stack([x * stride, y * stride], axis=1).astype(np.float64)
    displacement = apply_homography(matrix, points) - points
    return float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1))))


def _identity_estimate() -> TransformEstimate:
    empty_points = np.empty((0, 2), dtype=np.float64)
    return TransformEstimate(
        matrix=np.eye(3, dtype=np.float64),
        tracked_count=0,
        inlier_count=0,
        inlier_fraction=1.0,
        forward_backward_rmse=0.0,
        reprojection_rmse=0.0,
        spatial_coverage=1.0,
        source_points=empty_points,
        target_points=empty_points,
        inlier_mask=np.empty((0,), dtype=bool),
    )


def compute_background_compensated_flow(
    raw_frames_gray: np.ndarray,
    raw_foreground_masks: np.ndarray,
    stabilized_reference_frame: np.ndarray,
    stabilized_analysis_mask: np.ndarray,
    reference_index: int,
    settings: BackgroundCompensationSettings,
    *,
    flow_method: str,
) -> CompensatedFlowResult:
    validate_settings(settings)
    if raw_frames_gray.ndim != 3 or raw_frames_gray.shape[0] < 3:
        raise ValueError("raw_frames_gray must be [T,H,W] with at least three frames")
    if raw_foreground_masks.shape != raw_frames_gray.shape:
        raise ValueError("raw foreground masks must match raw_frames_gray [T,H,W]")
    if stabilized_reference_frame.ndim != 2:
        raise ValueError("stabilized_reference_frame must be [H,W]")
    if stabilized_analysis_mask.shape != stabilized_reference_frame.shape:
        raise ValueError("stabilized analysis mask must match stabilized reference frame")
    if reference_index < 0 or reference_index >= raw_frames_gray.shape[0]:
        raise ValueError("reference_index is outside the raw video")

    raw_height, raw_width = raw_frames_gray.shape[1:]
    target_height, target_width = stabilized_reference_frame.shape
    if (raw_height, raw_width) != (target_height, target_width):
        raise ValueError(
            "Raw resized frames must exactly match the stabilized reference grid; "
            f"got {(raw_height, raw_width)} versus {(target_height, target_width)}"
        )
    dilated_raw_masks = dilate_foreground_masks(
        raw_foreground_masks, settings.mask_dilate_px
    )
    target_kernel_size = 2 * settings.mask_dilate_px + 1
    target_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (target_kernel_size, target_kernel_size)
    )
    target_foreground = cv2.dilate(
        (stabilized_analysis_mask > 0).astype(np.uint8), target_kernel, iterations=1
    )
    raw_reference = raw_frames_gray[reference_index]
    reference_registration = estimate_background_transform(
        raw_reference,
        stabilized_reference_frame,
        dilated_raw_masks[reference_index],
        target_foreground,
        settings,
        label="raw reference -> stabilized reference registration",
    )
    raw_to_target = reference_registration.matrix.copy()

    frame_count = raw_frames_gray.shape[0]
    flow_u = np.empty((frame_count, target_height, target_width), dtype=np.float32)
    flow_v = np.empty_like(flow_u)
    raw_ref_to_frame = np.empty((frame_count, 3, 3), dtype=np.float64)
    frame_to_target = np.empty_like(raw_ref_to_frame)
    residual_transform = np.empty_like(raw_ref_to_frame)
    tracked_count = np.zeros((frame_count,), dtype=np.int64)
    inlier_count = np.zeros((frame_count,), dtype=np.int64)
    inlier_fraction = np.ones((frame_count,), dtype=np.float64)
    fb_rmse = np.zeros((frame_count,), dtype=np.float64)
    reprojection_rmse = np.zeros((frame_count,), dtype=np.float64)
    spatial_coverage = np.ones((frame_count,), dtype=np.float64)
    valid_fraction = np.zeros((frame_count,), dtype=np.float64)
    candidate_valid_fraction = np.zeros((frame_count,), dtype=np.float64)
    residual_background_rms = np.zeros((frame_count,), dtype=np.float64)
    residual_background_p90 = np.zeros((frame_count,), dtype=np.float64)
    camera_flow_foreground_rms = np.zeros((frame_count,), dtype=np.float64)
    residual_camera_flow_foreground_rms = np.zeros((frame_count,), dtype=np.float64)
    preview_indices = np.unique(
        np.asarray([0, reference_index, frame_count - 1], dtype=np.int64)
    )
    preview_by_index: dict[int, np.ndarray] = {}
    target_background_mask = target_foreground == 0
    target_candidate_mask = stabilized_analysis_mask > 0
    raw_target_inverse = np.linalg.inv(raw_to_target)

    for frame_index in range(frame_count):
        if frame_index == reference_index:
            motion_estimate = _identity_estimate()
        else:
            motion_estimate = estimate_background_transform(
                raw_reference,
                raw_frames_gray[frame_index],
                dilated_raw_masks[reference_index],
                dilated_raw_masks[frame_index],
                settings,
                label=f"raw frame {frame_index}",
            )
        raw_ref_to_frame[frame_index] = motion_estimate.matrix
        current_to_target = raw_to_target @ np.linalg.inv(motion_estimate.matrix)
        frame_to_target[frame_index] = current_to_target
        aligned = _warp_frame(
            raw_frames_gray[frame_index], current_to_target, (target_height, target_width)
        )
        aligned_foreground = _warp_mask(
            dilated_raw_masks[frame_index],
            current_to_target,
            (target_height, target_width),
        )
        valid = _warp_mask(
            np.ones((raw_height, raw_width), dtype=np.uint8),
            current_to_target,
            (target_height, target_width),
        ) > 0
        valid_fraction[frame_index] = float(np.mean(valid))
        candidate_count = int(np.count_nonzero(target_candidate_mask))
        candidate_valid_fraction[frame_index] = float(
            np.count_nonzero(valid & target_candidate_mask) / max(1, candidate_count)
        )
        if candidate_valid_fraction[frame_index] < settings.min_candidate_valid_fraction:
            raise ValueError(
                f"Frame {frame_index} candidate valid fraction "
                f"{candidate_valid_fraction[frame_index]:.6f} is below "
                f"{settings.min_candidate_valid_fraction:.6f}"
            )

        if frame_index == reference_index:
            dense_flow = np.zeros((target_height, target_width, 2), dtype=np.float32)
            residual_estimate = _identity_estimate()
        else:
            dense_flow = compute_dense_flow_pair(
                stabilized_reference_frame, aligned, method=flow_method
            )
            residual_estimate = estimate_background_transform(
                stabilized_reference_frame,
                aligned,
                target_foreground,
                np.maximum(aligned_foreground, target_foreground),
                settings,
                label=f"aligned residual frame {frame_index}",
            )
        flow_u[frame_index] = dense_flow[..., 0]
        flow_v[frame_index] = dense_flow[..., 1]
        residual_transform[frame_index] = residual_estimate.matrix
        tracked_count[frame_index] = motion_estimate.tracked_count
        inlier_count[frame_index] = motion_estimate.inlier_count
        inlier_fraction[frame_index] = motion_estimate.inlier_fraction
        fb_rmse[frame_index] = motion_estimate.forward_backward_rmse
        reprojection_rmse[frame_index] = motion_estimate.reprojection_rmse
        spatial_coverage[frame_index] = motion_estimate.spatial_coverage
        if residual_estimate.source_points.shape[0]:
            residual_displacement = (
                residual_estimate.target_points - residual_estimate.source_points
            )
            residual_magnitude = np.linalg.norm(residual_displacement, axis=1)
            residual_background_rms[frame_index] = float(
                np.sqrt(np.mean(residual_magnitude * residual_magnitude))
            )
            residual_background_p90[frame_index] = float(
                np.percentile(residual_magnitude, 90)
            )
        camera_forward_target = (
            raw_to_target @ motion_estimate.matrix @ raw_target_inverse
        )
        camera_flow_foreground_rms[frame_index] = transform_field_rms(
            camera_forward_target, target_candidate_mask
        )
        residual_camera_flow_foreground_rms[frame_index] = transform_field_rms(
            residual_estimate.matrix, target_candidate_mask
        )
        if frame_index in preview_indices:
            preview_by_index[frame_index] = aligned.astype(np.float32, copy=True)

    return CompensatedFlowResult(
        flow_u=flow_u,
        flow_v=flow_v,
        raw_reference_to_stabilized_reference=raw_to_target,
        raw_reference_to_raw_frame=raw_ref_to_frame,
        raw_frame_to_stabilized_reference=frame_to_target,
        residual_background_transform=residual_transform,
        tracked_count=tracked_count,
        inlier_count=inlier_count,
        inlier_fraction=inlier_fraction,
        forward_backward_rmse=fb_rmse,
        reprojection_rmse=reprojection_rmse,
        spatial_coverage=spatial_coverage,
        valid_fraction=valid_fraction,
        candidate_valid_fraction=candidate_valid_fraction,
        residual_background_rms=residual_background_rms,
        residual_background_p90=residual_background_p90,
        camera_flow_foreground_rms=camera_flow_foreground_rms,
        residual_camera_flow_foreground_rms=residual_camera_flow_foreground_rms,
        aligned_preview_indices=preview_indices,
        aligned_previews=np.stack(
            [preview_by_index[int(index)] for index in preview_indices], axis=0
        ),
    )


def transform_summary(matrix: np.ndarray) -> dict[str, float]:
    if matrix.shape != (3, 3):
        raise ValueError("Transform matrix must be [3,3]")
    linear = matrix[:2, :2]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    return {
        "translation_x_px": float(matrix[0, 2]),
        "translation_y_px": float(matrix[1, 2]),
        "rotation_rad": float(np.arctan2(linear[1, 0], linear[0, 0])),
        "scale_geometric_mean": float(np.sqrt(abs(np.linalg.det(linear)))),
        "anisotropy": float(singular_values[0] / max(singular_values[-1], 1e-12)),
    }


def json_float(value: Any) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None
