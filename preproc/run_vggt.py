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


def extrinsic_to_world_to_camera(extrinsic: np.ndarray) -> np.ndarray:
    arr = np.asarray(extrinsic, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = arr
        return out
    raise ValueError(f"Expected extrinsic shape (3,4) or (4,4), got {arr.shape}.")


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
