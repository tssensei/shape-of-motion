import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flow3d.data.colmap import qvec2rotmat, read_images_binary
from modal_surface.io import load_view_config


def camera_center_from_colmap_image(image) -> np.ndarray:
    R = qvec2rotmat(image.qvec).astype(np.float64)
    t = image.tvec.astype(np.float64)
    return -R.T @ t


def camera_forward_from_colmap_image(image) -> np.ndarray:
    R = qvec2rotmat(image.qvec).astype(np.float64)
    forward = R.T[:, 2]
    return forward / np.linalg.norm(forward)


def fit_sim3_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected src/dst shape (N,3), got {src.shape} and {dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("At least 3 reference cameras are required for Sim(3) alignment")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    if np.linalg.matrix_rank(src_centered) < 2:
        raise ValueError("COLMAP reference camera centers are degenerate")

    cov = (dst_centered.T @ src_centered) / float(src.shape[0])
    U, singular_values, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt
    src_var = np.mean(np.sum(src_centered**2, axis=1))
    if src_var <= 0:
        raise ValueError("COLMAP reference camera centers have zero variance")
    scale = float(np.sum(singular_values * np.diag(D)) / src_var)
    if scale <= 0:
        raise ValueError(f"Estimated non-positive Sim(3) scale: {scale}")
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def load_ref_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} must be a non-empty JSON object mapping view_id to COLMAP image name")
    out = {}
    for view_id, image_name in payload.items():
        if not isinstance(view_id, str) or not isinstance(image_name, str):
            raise ValueError(f"{path} keys and values must be strings")
        out[view_id] = Path(image_name).name
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--res", default="")
    parser.add_argument("--colmap-dir", required=True, type=Path)
    parser.add_argument("--view-config", action="append", required=True, type=Path)
    parser.add_argument("--ref-map", required=True, type=Path)
    parser.add_argument("--out-scene-norm", required=True, type=Path)
    parser.add_argument("--max-center-rmse-ratio", type=float, default=0.05)
    args = parser.parse_args()

    view_configs = [load_view_config(path) for path in args.view_config]
    if len(view_configs) < 3:
        raise ValueError("At least 3 VGGT view configs are required")
    ref_map = load_ref_map(args.ref_map)

    images_path = args.colmap_dir / "images.bin"
    if not images_path.exists():
        raise FileNotFoundError(images_path)
    images = read_images_binary(images_path)
    colmap_by_name = {Path(image.name).name: image for image in images.values()}

    src_centers = []
    dst_centers = []
    src_forwards = []
    dst_forwards = []
    used = []
    for cfg in view_configs:
        if cfg.view_id not in ref_map:
            raise KeyError(f"{args.ref_map} is missing view_id {cfg.view_id!r}")
        image_name = ref_map[cfg.view_id]
        if image_name not in colmap_by_name:
            raise KeyError(f"COLMAP image {image_name!r} is not registered in {images_path}")
        colmap_image = colmap_by_name[image_name]
        vggt_c2w = np.linalg.inv(cfg.world_to_camera).astype(np.float64)
        src_centers.append(camera_center_from_colmap_image(colmap_image))
        dst_centers.append(vggt_c2w[:3, 3])
        src_forwards.append(camera_forward_from_colmap_image(colmap_image))
        dst_forward = vggt_c2w[:3, 2]
        dst_forwards.append(dst_forward / np.linalg.norm(dst_forward))
        used.append({"view_id": cfg.view_id, "colmap_image": image_name})

    src = np.stack(src_centers)
    dst = np.stack(dst_centers)
    scale, R, t = fit_sim3_umeyama(src, dst)
    aligned = scale * (src @ R.T) + t[None]
    residuals = np.linalg.norm(aligned - dst, axis=1)
    dst_diag = np.linalg.norm(dst.max(axis=0) - dst.min(axis=0))
    if dst_diag <= 0:
        raise ValueError("VGGT reference camera centers have zero extent")
    rmse = float(np.sqrt(np.mean(residuals**2)))
    rmse_ratio = rmse / float(dst_diag)
    if rmse_ratio > args.max_center_rmse_ratio:
        raise ValueError(
            f"COLMAP-to-VGGT camera-center alignment RMSE ratio {rmse_ratio:.6g} "
            f"exceeds {args.max_center_rmse_ratio:.6g}"
        )

    src_forwards_np = np.stack(src_forwards)
    dst_forwards_np = np.stack(dst_forwards)
    aligned_forwards = src_forwards_np @ R.T
    dots = np.sum(aligned_forwards * dst_forwards_np, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    forward_angle_deg = np.degrees(np.arccos(dots))

    scene_scale = 1.0 / scale
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R.astype(np.float32)
    transform[:3, 3] = (t / scale).astype(np.float32)

    args.out_scene_norm.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scale": float(scene_scale),
            "transfm": torch.from_numpy(transform),
        },
        args.out_scene_norm,
    )

    diagnostics = {
        "used_refs": used,
        "sim3_scale_colmap_to_vggt": scale,
        "scene_norm_scale": float(scene_scale),
        "center_residuals": residuals.tolist(),
        "center_rmse": rmse,
        "center_rmse_ratio": rmse_ratio,
        "forward_angle_deg": forward_angle_deg.tolist(),
        "transform": transform.tolist(),
    }
    diag_path = args.out_scene_norm.with_suffix(".json")
    with diag_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"Wrote aligned scene norm to {args.out_scene_norm}")
    print(f"Wrote diagnostics to {diag_path}")
    print(f"Sim(3) scale COLMAP->VGGT: {scale:.8g}")
    print(f"center RMSE: {rmse:.6g} ({rmse_ratio:.6g} of reference bbox)")
    print(
        "forward angle deg p50/p95/max: "
        f"{np.percentile(forward_angle_deg, 50):.6g} / "
        f"{np.percentile(forward_angle_deg, 95):.6g} / "
        f"{forward_angle_deg.max():.6g}"
    )


if __name__ == "__main__":
    main()
