"""Run VGGT-Omega and export modal_surface geometry inputs.

This is an optional preprocessing bridge. It assumes the active Python
environment can import `vggt_omega`, usually after:

    cd ~/vggt-omega
    pip install -e .

The script runs VGGT-Omega on an image set, saves the raw camera/depth outputs,
and exports selected reference views to the modal_surface geometry contract:

    view*_config.json
    view*_depth.npy
    view*_mask.npy

Optionally, it can also export a VGGT carrier point cloud:

    vggt_points.npz

The optional point cloud is built directly from VGGT depth, intrinsics, and
extrinsics. It can be used for visualization or Gaussian initialization; modal
observations are solved directly on foreground Gaussians instead.

Example:

    python shape-of-motion/preproc/run_vggt.py \
      --checkpoint ~/vggt-omega/checkpoints/vggt_omega_1b_512.pt \
      --image ~/outputs_modal/bush4/geometry/sweep_colmap/images/refs/view1.png \
      --image ~/outputs_modal/bush4/geometry/sweep_colmap/images/refs/view2.png \
      --image ~/outputs_modal/bush4/geometry/sweep_colmap/images/refs/view3.png \
      --view view1=~/outputs_modal/bush4/geometry/sweep_colmap/images/refs/view1.png=~/outputs_modal/bush4/view1/modal_analysis_0p357hz.npz=~/outputs_modal/bush4/geometry/masks/view1.png \
      --view view2=~/outputs_modal/bush4/geometry/sweep_colmap/images/refs/view2.png=~/outputs_modal/bush4/view2/modal_analysis_0p357hz.npz=~/outputs_modal/bush4/geometry/masks/view2.png \
      --view view3=~/outputs_modal/bush4/geometry/sweep_colmap/images/refs/view3.png=~/outputs_modal/bush4/view3/modal_analysis_0p357hz.npz=~/outputs_modal/bush4/geometry/masks/view3.png \
      --out-dir ~/outputs_modal/bush4/geometry/modal_surface_vggt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ViewSpec:
    label: str
    image_path: Path
    modal_npz: Path
    mask_path: Path


def expand_path(path: str | Path) -> Path:
    """Expand shell-style user paths without requiring shell expansion."""
    return Path(path).expanduser()


def resolved(path: Path) -> Path:
    """Resolve an existing path for path matching."""
    return path.resolve(strict=True)


def parse_view_spec(raw: str) -> ViewSpec:
    """Parse LABEL=IMAGE=MODAL_NPZ=MASK."""
    parts = raw.split("=")
    if len(parts) != 4:
        raise ValueError(f"Expected --view LABEL=IMAGE=MODAL_NPZ=MASK, got: {raw}")
    label, image_path, modal_npz, mask_path = parts
    if not label:
        raise ValueError(f"View label must not be empty: {raw}")
    return ViewSpec(
        label=label,
        image_path=expand_path(image_path),
        modal_npz=expand_path(modal_npz),
        mask_path=expand_path(mask_path),
    )


def load_modal_shape(modal_npz: Path) -> tuple[int, int]:
    z = np.load(str(modal_npz), allow_pickle=False)
    if "mode_u" not in z.files:
        raise KeyError(f"{modal_npz} missing mode_u")
    mode_u = z["mode_u"]
    if mode_u.ndim != 3:
        raise ValueError(f"Expected mode_u shape (K,H,W), got {mode_u.shape}")
    return int(mode_u.shape[1]), int(mode_u.shape[2])


def load_modal_mask_or_resize(modal_npz: Path, fallback_mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Use the modal export mask when present; otherwise resize the supplied mask."""
    z = np.load(str(modal_npz), allow_pickle=False)
    if "has_mask" in z.files and bool(np.asarray(z["has_mask"]).item()) and "mask" in z.files:
        mask = z["mask"]
        if mask.shape == target_hw and mask.size > 0:
            return mask.astype(bool)
    return resize_mask(fallback_mask, target_hw)


def read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(str(path))
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got {mask.shape}: {path}")
        return mask.astype(np.float32) > 0.5 if np.issubdtype(mask.dtype, np.floating) else mask > 0
    image = Image.open(path).convert("L")
    return np.asarray(image) > 0


def resize_depth(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2D float depth map with bilinear interpolation."""
    if depth.shape == target_hw:
        return depth.astype(np.float32, copy=False)
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(depth.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
    return resized[0, 0].numpy().astype(np.float32)


def resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask with nearest-neighbor interpolation."""
    if mask.shape == target_hw:
        return mask.astype(bool, copy=False)
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=target_hw, mode="nearest")
    return resized[0, 0].numpy() > 0.5


def scale_K(K: np.ndarray, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    out = K.astype(np.float64).copy()
    out[0, :] *= float(target_w) / float(source_w)
    out[1, :] *= float(target_h) / float(source_h)
    return out


def normalize_depth_array(depth: np.ndarray, num_images: int) -> np.ndarray:
    """Normalize common VGGT depth tensor layouts to (N,H,W)."""
    arr = np.asarray(depth)
    if arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Could not normalize depth array to (N,H,W), got {depth.shape}.")
    if arr.shape[0] != num_images:
        raise ValueError(f"Depth view count {arr.shape[0]} does not match image count {num_images}.")
    return arr.astype(np.float32)


def normalize_matrix_array(array: np.ndarray, num_images: int, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape[0] != num_images:
        raise ValueError(f"{name} view count {arr.shape[0]} does not match image count {num_images}.")
    return arr


def normalize_image_array(images: np.ndarray, num_images: int) -> np.ndarray:
    """Normalize common VGGT image tensor layouts to uint8 RGB (N,H,W,3)."""
    arr = np.asarray(images)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"Could not normalize image array to (N,H,W,3), got {images.shape}.")
    if arr.shape[0] != num_images:
        raise ValueError(f"Image view count {arr.shape[0]} does not match image count {num_images}.")
    if arr.shape[1] in (1, 3, 4):
        arr = np.moveaxis(arr, 1, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image array, got {arr.shape}.")

    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)
    finite_vals = arr[finite]
    if float(finite_vals.min()) < 0.0:
        lo = float(np.percentile(finite_vals, 1))
        hi = float(np.percentile(finite_vals, 99))
        arr = (arr - lo) / max(hi - lo, 1e-6)
    elif float(finite_vals.max()) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def extrinsic_to_world_to_camera(extrinsic: np.ndarray) -> np.ndarray:
    arr = np.asarray(extrinsic, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = arr
        return out
    raise ValueError(f"Expected extrinsic shape (3,4) or (4,4), got {arr.shape}.")


def depth_edge_mask(depth: np.ndarray, rtol: float) -> np.ndarray:
    """Detect relative depth discontinuities with 4-neighbor comparisons."""
    if rtol < 0:
        raise ValueError("--depth-edge-rtol must be non-negative.")
    depth = depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0)
    edge = np.zeros(depth.shape, dtype=bool)

    def mark_pair(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
        pair_valid = valid[a] & valid[b]
        denom = np.maximum(np.minimum(depth[a], depth[b]), 1e-6)
        jump = np.zeros(pair_valid.shape, dtype=np.float32)
        jump[pair_valid] = np.abs(depth[a][pair_valid] - depth[b][pair_valid]) / denom[pair_valid]
        pair_edge = pair_valid & (jump > rtol)
        out[a] |= pair_edge
        out[b] |= pair_edge

    mark_pair(np.s_[:, 1:], np.s_[:, :-1], edge)
    mark_pair(np.s_[1:, :], np.s_[:-1, :], edge)
    return edge


def unproject_depth_samples(
    depth: np.ndarray,
    confidence: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
    confidence_threshold: float,
    depth_edge: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unproject valid VGGT depth pixels above a VGGT confidence threshold."""
    if depth.shape != confidence.shape:
        raise ValueError(f"Depth/confidence shape mismatch: {depth.shape} vs {confidence.shape}.")
    if colors.shape[:2] != depth.shape:
        raise ValueError(f"Color/depth shape mismatch: {colors.shape[:2]} vs {depth.shape}.")
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence >= confidence_threshold)
    if depth_edge is not None:
        if depth_edge.shape != depth.shape:
            raise ValueError(f"Depth edge mask shape mismatch: {depth_edge.shape} vs {depth.shape}.")
        valid &= ~depth_edge
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    yy, xx = np.nonzero(valid)
    z = depth[yy, xx].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xx.astype(np.float64) - cx) * z / fx
    y = (yy.astype(np.float64) - cy) * z / fy
    points_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    camera_to_world = np.linalg.inv(world_to_camera)
    points_world_h = points_cam @ camera_to_world.T
    pixels_xy = np.stack([xx, yy], axis=1).astype(np.float32)
    return (
        points_world_h[:, :3].astype(np.float32),
        colors[yy, xx].astype(np.uint8),
        confidence[yy, xx].astype(np.float32),
        pixels_xy,
    )


def export_vggt_points(
    out_dir: Path,
    depth: np.ndarray,
    depth_conf: np.ndarray,
    images: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    point_conf_percentile: float,
    max_points: int,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
) -> tuple[Path, dict[str, Any]]:
    """Export a dense VGGT carrier point cloud to vggt_points.npz."""
    if not (0.0 <= point_conf_percentile <= 100.0):
        raise ValueError("--point-conf-percentile must be in [0, 100].")
    if max_points < 0:
        raise ValueError("--max-points must be non-negative.")
    if depth_edge_rtol < 0:
        raise ValueError("--depth-edge-rtol must be non-negative.")

    depth_edge_masks: list[np.ndarray | None] = []
    edge_filtered_points = 0
    for view_idx in range(depth.shape[0]):
        edge = depth_edge_mask(depth[view_idx], depth_edge_rtol) if filter_depth_edges else None
        if edge is not None:
            valid_edge_depth = edge & np.isfinite(depth[view_idx]) & (depth[view_idx] > 0)
            edge_filtered_points += int(valid_edge_depth.sum())
        depth_edge_masks.append(edge)

    valid_conf = depth_conf[np.isfinite(depth_conf) & np.isfinite(depth) & (depth > 0)]
    if valid_conf.size == 0:
        raise ValueError("VGGT depth/confidence produced no valid depth samples.")
    confidence_threshold = float(np.percentile(valid_conf, point_conf_percentile))

    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    confidence_chunks: list[np.ndarray] = []
    source_view_chunks: list[np.ndarray] = []
    source_pixel_chunks: list[np.ndarray] = []
    for view_idx in range(depth.shape[0]):
        world_to_camera = extrinsic_to_world_to_camera(extrinsics[view_idx])
        points, colors, confidence, pixels_xy = unproject_depth_samples(
            depth[view_idx],
            depth_conf[view_idx],
            images[view_idx],
            intrinsics[view_idx],
            world_to_camera,
            confidence_threshold,
            depth_edge=depth_edge_masks[view_idx],
        )
        point_chunks.append(points)
        color_chunks.append(colors)
        confidence_chunks.append(confidence)
        source_view_chunks.append(np.full((points.shape[0],), view_idx, dtype=np.int32))
        source_pixel_chunks.append(pixels_xy)

    points_world = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    confidence = np.concatenate(confidence_chunks, axis=0)
    source_view_index = np.concatenate(source_view_chunks, axis=0)
    source_pixels_xy = np.concatenate(source_pixel_chunks, axis=0)
    candidate_count = int(points_world.shape[0])
    if candidate_count == 0:
        raise ValueError("No VGGT carrier points survived confidence filtering.")

    if max_points > 0 and candidate_count > max_points:
        keep = np.linspace(0, candidate_count - 1, int(max_points), dtype=np.int64)
        points_world = points_world[keep]
        colors = colors[keep]
        confidence = confidence[keep]
        source_view_index = source_view_index[keep]
        source_pixels_xy = source_pixels_xy[keep]

    out_path = out_dir / "vggt_points.npz"
    np.savez_compressed(
        out_path,
        points_world=points_world.astype(np.float32),
        colors=colors.astype(np.uint8),
        confidence=confidence.astype(np.float32),
        source_view_index=source_view_index.astype(np.int32),
        source_pixels_xy=source_pixels_xy.astype(np.float32),
        source_image_height=np.array(depth.shape[1], dtype=np.int32),
        source_image_width=np.array(depth.shape[2], dtype=np.int32),
    )
    stats = {
        "path": out_path.name,
        "candidate_points": candidate_count,
        "saved_points": int(points_world.shape[0]),
        "confidence_percentile": float(point_conf_percentile),
        "confidence_threshold": confidence_threshold,
        "confidence_filtering": True,
        "filter_depth_edges": bool(filter_depth_edges),
        "depth_edge_rtol": float(depth_edge_rtol),
        "edge_filtered_points": int(edge_filtered_points),
        "max_points": int(max_points),
    }
    return out_path, stats


def save_view_config(
    path: Path,
    view_id: str,
    image_width: int,
    image_height: int,
    K: np.ndarray,
    world_to_camera: np.ndarray,
    depth_name: str,
    mask_name: str,
) -> None:
    payload = {
        "view_id": view_id,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "K": K.astype(float).tolist(),
        "world_to_camera": world_to_camera.astype(float).tolist(),
        "depth_path": depth_name,
        "mask_path": mask_name,
        "depth_scale": 1.0,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def to_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VGGT-Omega and export modal_surface geometry inputs.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="VGGT-Omega checkpoint .pt path.")
    parser.add_argument("--image", action="append", required=True, type=Path, help="Input image path. Repeat for every VGGT view.")
    parser.add_argument("--view", action="append", required=True, help="Reference export spec: LABEL=IMAGE=MODAL_NPZ=MASK.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--image-resolution", type=int, default=512, help="VGGT-Omega preprocessing resolution.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Torch device for inference.")
    parser.add_argument("--save-tokens", action="store_true", help="Also save camera_and_register_tokens in raw output.")
    parser.add_argument("--export-points", action="store_true", help="Export VGGT unprojected carrier points to vggt_points.npz.")
    parser.add_argument("--point-conf-percentile", type=float, default=30.0, help="VGGT depth_conf percentile threshold used only while exporting fixed carrier points.")
    parser.add_argument("--filter-depth-edges", action="store_true", help="Drop VGGT carrier points on local depth discontinuities.")
    parser.add_argument("--depth-edge-rtol", type=float, default=0.03, help="Relative depth jump threshold for --filter-depth-edges.")
    parser.add_argument("--max-points", type=int, default=500000, help="Maximum saved carrier points; 0 keeps all points.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        import torch
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except ImportError as exc:
        raise ImportError(
            "Cannot import VGGT-Omega. Install it in the active environment, for example:\n"
            "  cd ~/vggt-omega\n"
            "  pip install -e ."
        ) from exc

    checkpoint = expand_path(args.checkpoint)
    out_dir = expand_path(args.out_dir)
    image_paths = [expand_path(path) for path in args.image]
    view_specs = [parse_view_spec(raw) for raw in args.view]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a PyTorch build compatible with the node NVIDIA driver.")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing input image: {path}")

    resolved_to_index = {resolved(path): i for i, path in enumerate(image_paths)}
    for spec in view_specs:
        if not spec.image_path.exists():
            raise FileNotFoundError(f"Missing view image: {spec.image_path}")
        if not spec.modal_npz.exists():
            raise FileNotFoundError(f"Missing modal npz: {spec.modal_npz}")
        if not spec.mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {spec.mask_path}")
        if resolved(spec.image_path) not in resolved_to_index:
            raise ValueError(f"View image must also be listed with --image: {spec.image_path}")

    device = torch.device(args.device)
    print(f"Loading VGGT-Omega checkpoint: {checkpoint}")
    model = VGGTOmega().to(device).eval()
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))

    image_names = [str(path) for path in image_paths]
    print(f"Running VGGT-Omega on {len(image_names)} images at resolution {args.image_resolution}.")
    images = load_and_preprocess_images(image_names, image_resolution=args.image_resolution).to(device)

    with torch.inference_mode():
        predictions = model(images)

    extrinsics_t, intrinsics_t = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )

    processed_hw = tuple(int(v) for v in predictions["images"].shape[-2:])
    extrinsics = normalize_matrix_array(to_numpy(extrinsics_t), len(image_paths), "extrinsics")
    intrinsics = normalize_matrix_array(to_numpy(intrinsics_t), len(image_paths), "intrinsics")
    depth = normalize_depth_array(to_numpy(predictions["depth"]), len(image_paths))
    depth_conf = normalize_depth_array(to_numpy(predictions["depth_conf"]), len(image_paths))
    processed_images = normalize_image_array(to_numpy(predictions["images"]), len(image_paths))

    raw_payload = {
        "image_paths": np.asarray(image_names),
        "processed_hw": np.asarray(processed_hw, dtype=np.int32),
        "extrinsics": extrinsics.astype(np.float32),
        "intrinsics": intrinsics.astype(np.float32),
        "depth": depth.astype(np.float32),
        "depth_conf": depth_conf.astype(np.float32),
        "pose_enc": to_numpy(predictions["pose_enc"]).astype(np.float32),
    }
    if args.save_tokens:
        raw_payload["camera_and_register_tokens"] = to_numpy(predictions["camera_and_register_tokens"]).astype(np.float32)
    raw_path = out_dir / "vggt_outputs.npz"
    np.savez_compressed(raw_path, **raw_payload)

    stats: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "image_resolution": int(args.image_resolution),
        "device": str(device),
        "processed_hw": list(processed_hw),
        "raw_output": raw_path.name,
        "views": {},
    }
    if args.export_points:
        point_path, point_stats = export_vggt_points(
            out_dir,
            depth,
            depth_conf,
            processed_images,
            intrinsics,
            extrinsics,
            point_conf_percentile=args.point_conf_percentile,
            max_points=args.max_points,
            filter_depth_edges=args.filter_depth_edges,
            depth_edge_rtol=args.depth_edge_rtol,
        )
        stats["carrier_points"] = point_stats
        print(f"Saved VGGT carrier points -> {point_path}")

    for spec in view_specs:
        view_idx = resolved_to_index[resolved(spec.image_path)]
        target_hw = load_modal_shape(spec.modal_npz)
        K_target = scale_K(intrinsics[view_idx], processed_hw, target_hw)
        depth_target = resize_depth(depth[view_idx], target_hw)
        mask_source = read_mask(spec.mask_path)
        mask_target = load_modal_mask_or_resize(spec.modal_npz, mask_source, target_hw)
        world_to_camera = extrinsic_to_world_to_camera(extrinsics[view_idx])

        depth_name = f"{spec.label}_depth.npy"
        mask_name = f"{spec.label}_mask.npy"
        np.save(out_dir / depth_name, depth_target.astype(np.float32))
        np.save(out_dir / mask_name, mask_target.astype(np.uint8))
        save_view_config(
            out_dir / f"{spec.label}_config.json",
            spec.label,
            target_hw[1],
            target_hw[0],
            K_target,
            world_to_camera,
            depth_name,
            mask_name,
        )
        stats["views"][spec.label] = {
            "image_path": str(spec.image_path),
            "image_index": int(view_idx),
            "modal_npz": str(spec.modal_npz),
            "mask_path": str(spec.mask_path),
            "target_height": int(target_hw[0]),
            "target_width": int(target_hw[1]),
            "depth_output": depth_name,
            "mask_output": mask_name,
            "config_output": f"{spec.label}_config.json",
            "depth_conf_median_processed": float(np.median(depth_conf[view_idx])),
        }

    stats_path = out_dir / "vggt_modal_geometry_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved raw VGGT outputs -> {raw_path}")
    print(f"Saved modal_surface geometry -> {out_dir}")


if __name__ == "__main__":
    main()
