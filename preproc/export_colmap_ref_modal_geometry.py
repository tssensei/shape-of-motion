"""Export modal_surface geometry inputs from registered COLMAP reference views.

This script is the bridge between the calibration-sweep geometry stage and the
surface-based modal optimization stage.

The intended workflow is:

1. Record stabilized modal videos for two static-ish viewpoints.
2. Export one reference image for each modal view, usually:

       geometry/images/view1.png
       geometry/images/view2.png

3. Record an additional calibration sweep video around the same scene.
4. Build a multi-view COLMAP sparse reconstruction from sweep frames plus the
   two reference images. The two reference images must be registered in the
   same COLMAP model, for example as:

       refs/view1.png
       refs/view2.png

5. Run UniDepth on the reference images to get per-view disparity maps.
6. Run this script to produce the view configs and aligned depth maps consumed
   by run_modal_surface.py.

Example:

    python shape-of-motion/preproc/export_colmap_ref_modal_geometry.py \
      --model-dir outputs_modal/bush4/geometry/sweep_colmap/sparse_txt \
      --image-dir outputs_modal/bush4/geometry/sweep_colmap/images \
      --mask-dir outputs_modal/bush4/geometry/masks \
      --unidepth-disp-dir outputs_modal/bush4/geometry/unidepth_disp \
      --view1-name refs/view1.png \
      --view2-name refs/view2.png \
      --view1-modal-npz outputs_modal/bush4/view1/modal_analysis_0p357hz.npz \
      --view2-modal-npz outputs_modal/bush4/view2/modal_analysis_0p357hz.npz \
      --out-dir outputs_modal/bush4/geometry/modal_surface_colmap

Outputs:

    view1_config.json
    view2_config.json
    view1_depth.npy
    view2_depth.npy
    view1_mask.npy
    view2_mask.npy
    colmap_ref_pose_stats.json

Why this exists:

Two-view relative pose estimation from only view1/view2 is fragile for this
experiment: background structure can be close to planar, the vibrating bush is
masked out, and monocular depth scale is uncertain. A calibration sweep gives
COLMAP many rigid-background views, producing a more stable sparse model. This
script then reads the already-registered reference view poses from that model
instead of estimating pose from the two references alone.

Depth scale alignment:

UniDepth disparity is converted to a depth-like map by depth = 1 / disparity.
That depth is not guaranteed to share the same scale as the COLMAP model. For
each reference image, COLMAP provides sparse 2D observations with POINT3D_IDs.
For every valid observation, this script compares:

    z_colmap   = camera-space z of the observed sparse 3D point
    z_unidepth = UniDepth depth sampled at the same pixel

and estimates a robust per-view scale:

    scale = median(z_colmap / z_unidepth)
    depth_aligned = depth_unidepth * scale

The resulting aligned depth maps and COLMAP world_to_camera matrices are in the
same coordinate scale, which is required before unprojecting surface points and
projecting them across views in modal_surface.

Assumptions:

- COLMAP TXT model uses PINHOLE cameras.
- Reference image names passed via --view1-name/--view2-name are present in
  images.txt and have corresponding masks/depths keyed by their stem.
- UniDepth outputs disparity .npy files, not metric z-depth directly.
- The modal npz files define the target image size used by modal_surface.
- This script exports only two reference views.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# COLMAP sweep model + view1/view2 reference images
# + UniDepth disparity
# + modal_analysis npz
# -> view1_config.json / view2_config.json / depth / mask

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    if qvec.shape != (4,):
        raise ValueError(f"Expected qvec shape (4,), got {qvec.shape}.")
    qvec = qvec / max(float(np.linalg.norm(qvec)), 1e-12)
    qw, qx, qy, qz = qvec.tolist()
    return np.array(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qw * qz, 2.0 * qz * qx + 2.0 * qw * qy],
            [2.0 * qx * qy + 2.0 * qw * qz, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qw * qx],
            [2.0 * qz * qx - 2.0 * qw * qy, 2.0 * qy * qz + 2.0 * qw * qx, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def read_mask(path: Path, image_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        mask = mask.reshape((*mask.shape[:2], -1)).max(axis=-1)
    mask = mask > 0
    if mask.shape != image_hw:
        mask = cv2.resize(mask.astype(np.uint8), (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def load_modal_shape(modal_npz: Path) -> tuple[int, int]:
    z = np.load(str(modal_npz), allow_pickle=False)
    if "mode_u" not in z:
        raise KeyError(f"{modal_npz} missing mode_u")
    mode_u = z["mode_u"]
    if mode_u.ndim != 3:
        raise ValueError(f"Expected mode_u shape (K,H,W), got {mode_u.shape}")
    return int(mode_u.shape[1]), int(mode_u.shape[2])


def load_modal_mask_or_resize(modal_npz: Path, fallback_mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    z = np.load(str(modal_npz), allow_pickle=False)
    if "has_mask" in z and bool(np.asarray(z["has_mask"]).item()) and "mask" in z:
        mask = z["mask"]
        if mask.shape == target_hw and mask.size > 0:
            return mask.astype(bool)
    return cv2.resize(
        fallback_mask.astype(np.uint8),
        (target_hw[1], target_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    ) > 0


def load_depth_from_unidepth_disp(path: Path, source_hw: tuple[int, int]) -> np.ndarray:
    """Load UniDepth disparity and convert it to a depth-like z map.

    UniDepth outputs used in this project are stored as disparity-like arrays.
    The modal_surface code expects positive depth, so this helper uses
    depth = 1 / disparity and resizes the result to the reference image size.

    The returned depth is not yet guaranteed to be in COLMAP scale. Scale
    alignment is handled later by align_depth_to_colmap().
    """
    disp = np.load(str(path)).astype(np.float32)
    disp = np.squeeze(disp)
    if disp.ndim != 2:
        raise ValueError(f"Expected 2D disparity at {path}, got {disp.shape}")
    if disp.shape != source_hw:
        disp = cv2.resize(disp, (source_hw[1], source_hw[0]), interpolation=cv2.INTER_LINEAR)
    depth = 1.0 / np.clip(disp, 1e-6, 1e6)
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def resize_depth(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if depth.shape == target_hw:
        return depth.astype(np.float32, copy=False)
    return cv2.resize(depth.astype(np.float32), (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)


def scale_K(K: np.ndarray, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    out = K.astype(np.float64).copy()
    out[0, :] *= float(target_w) / float(source_w)
    out[1, :] *= float(target_h) / float(source_h)
    return out


def parse_cameras(path: Path) -> dict[int, dict[str, object]]:
    """Parse COLMAP cameras.txt into camera intrinsics records.

    Only the PINHOLE camera model is supported because modal_surface expects a
    3x3 pinhole intrinsic matrix K. The returned dictionary maps camera_id to
    a record containing image width/height, K, and the raw COLMAP params.
    """
    cameras: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.asarray([float(v) for v in parts[4:]], dtype=np.float64)
            if model != "PINHOLE" or params.shape != (4,):
                raise ValueError(f"Only PINHOLE cameras are supported, got camera {camera_id}: {line}")
            fx, fy, cx, cy = params.tolist()
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "K": K,
                "params": params,
            }
    if not cameras:
        raise ValueError(f"No cameras found in {path}")
    return cameras


def parse_points2d(line: str) -> np.ndarray:
    line = line.strip()
    if not line:
        return np.zeros((0, 3), dtype=np.float64)
    vals = np.fromstring(line, sep=" ", dtype=np.float64)
    if vals.size % 3 != 0:
        raise ValueError("Invalid POINTS2D line; expected triples of X Y POINT3D_ID.")
    return vals.reshape(-1, 3)


def parse_images(path: Path) -> dict[str, dict[str, object]]:
    """Parse COLMAP images.txt into per-image pose and observation records.

    COLMAP stores each image as two non-comment lines:

        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        POINTS2D[] as triples of X Y POINT3D_ID

    The qvec/tvec pair is converted to a 4x4 world_to_camera matrix using
    COLMAP's convention:

        X_cam = R * X_world + t

    The dictionary is keyed by image name, such as "refs/view1.png".
    """
    images: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")]
    if len(lines) % 2 != 0:
        raise ValueError(f"Invalid COLMAP images.txt line count in {path}")
    for i in range(0, len(lines), 2):
        header = lines[i]
        points_line = lines[i + 1]
        parts = header.split(maxsplit=9)
        if len(parts) != 10:
            raise ValueError(f"Invalid image header line: {header}")
        image_id = int(parts[0])
        qvec = np.asarray([float(v) for v in parts[1:5]], dtype=np.float64)
        tvec = np.asarray([float(v) for v in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]
        R = qvec_to_rotmat(qvec)
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = R
        world_to_camera[:3, 3] = tvec
        images[name] = {
            "image_id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
            "world_to_camera": world_to_camera,
            "points2d": parse_points2d(points_line),
        }
    if not images:
        raise ValueError(f"No images found in {path}")
    return images


def parse_points3d(path: Path) -> dict[int, np.ndarray]:
    """Parse COLMAP points3D.txt into POINT3D_ID -> XYZ_world."""
    points: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid points3D line: {line}")
            point_id = int(parts[0])
            points[point_id] = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    if not points:
        raise ValueError(f"No 3D points found in {path}")
    return points


def bilinear_sample(image: np.ndarray, pixels_xy: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = pixels_xy[:, 0].astype(np.float64)
    y = pixels_xy[:, 1].astype(np.float64)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    out = np.full(len(pixels_xy), np.nan, dtype=np.float64)
    if not np.any(valid):
        return out
    wx = x[valid] - x0[valid]
    wy = y[valid] - y0[valid]
    v00 = image[y0[valid], x0[valid]]
    v01 = image[y0[valid], x1[valid]]
    v10 = image[y1[valid], x0[valid]]
    v11 = image[y1[valid], x1[valid]]
    out[valid] = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v01
        + (1.0 - wx) * wy * v10
        + wx * wy * v11
    )
    return out


def align_depth_to_colmap(
    image_record: dict[str, object],
    points3d: dict[int, np.ndarray],
    depth: np.ndarray,
    min_points: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Scale a UniDepth depth map into the COLMAP coordinate scale.

    For one registered reference image, COLMAP gives a sparse set of 2D feature
    observations and their corresponding 3D points. This function computes the
    camera-space z value of each observed sparse 3D point, samples UniDepth at
    the same pixel, and estimates a robust multiplicative scale between them.

    The 10th-90th percentile trimmed median is used to reduce the influence of
    bad sparse points, wrong local depth, and foreground/background boundary
    errors. The function returns the scaled dense depth map and diagnostic
    statistics written later to colmap_ref_pose_stats.json.
    """
    points2d = np.asarray(image_record["points2d"], dtype=np.float64)
    if points2d.size == 0:
        raise ValueError(f"{image_record['name']} has no COLMAP 2D point observations.")
    world_to_camera = np.asarray(image_record["world_to_camera"], dtype=np.float64)
    pixels: list[list[float]] = []
    z_colmap: list[float] = []
    for x, y, point_id_float in points2d:
        point_id = int(round(float(point_id_float)))
        if point_id < 0 or point_id not in points3d:
            continue
        X_world = points3d[point_id]
        X_cam = world_to_camera[:3, :3] @ X_world + world_to_camera[:3, 3]
        if X_cam[2] <= 0:
            continue
        pixels.append([float(x), float(y)])
        z_colmap.append(float(X_cam[2]))
    if len(pixels) < min_points:
        raise RuntimeError(
            f"{image_record['name']} has only {len(pixels)} sparse depth observations; "
            f"need at least {min_points}."
        )

    sampled_depth = bilinear_sample(depth, np.asarray(pixels, dtype=np.float64))
    z_colmap_arr = np.asarray(z_colmap, dtype=np.float64)
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0) & np.isfinite(z_colmap_arr) & (z_colmap_arr > 0)
    # Ratio at each sparse observation: COLMAP camera z divided by UniDepth z.
    # Multiplying UniDepth by the robust median ratio puts dense depth in the
    # same scale as the COLMAP poses.
    ratios = z_colmap_arr[valid] / sampled_depth[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size < min_points:
        raise RuntimeError(
            f"{image_record['name']} has only {int(ratios.size)} valid depth ratios; "
            f"need at least {min_points}."
        )

    p10, p90 = np.percentile(ratios, [10, 90])
    trimmed = ratios[(ratios >= p10) & (ratios <= p90)]
    if trimmed.size < min_points:
        trimmed = ratios
    scale = float(np.median(trimmed))
    aligned = depth.astype(np.float32) * np.float32(scale)
    stats = {
        "image_name": str(image_record["name"]),
        "sparse_observations": int(len(pixels)),
        "valid_ratios": int(ratios.size),
        "scale": scale,
        "ratio_p10": float(np.percentile(ratios, 10)),
        "ratio_p50": float(np.percentile(ratios, 50)),
        "ratio_p90": float(np.percentile(ratios, 90)),
        "trimmed_count": int(trimmed.size),
    }
    return aligned, stats


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
    """Write the JSON view_config contract consumed by modal_surface."""
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


def export_view(
    view_label: str,
    colmap_name: str,
    image_record: dict[str, object],
    camera_record: dict[str, object],
    image_dir: Path,
    mask_dir: Path,
    unidepth_disp_dir: Path,
    modal_npz: Path,
    target_hw: tuple[int, int],
    points3d: dict[int, np.ndarray],
    min_depth_alignment_points: int,
    disable_depth_alignment: bool,
    out_dir: Path,
) -> dict[str, object]:
    """Export one registered reference view to modal_surface inputs.

    This performs all per-view conversion:

    - verify the COLMAP camera image size matches the reference image;
    - load foreground mask and UniDepth disparity by image stem;
    - optionally align dense depth to COLMAP scale;
    - scale K from reference-image coordinates to modal-image coordinates;
    - resize depth and mask to the modal image shape;
    - write depth/mask arrays and view_config JSON.

    The returned dictionary is diagnostic metadata for
    colmap_ref_pose_stats.json, not part of the modal_surface runtime contract.
    """
    source_image = read_image(image_dir / colmap_name)
    source_hw = source_image.shape[:2]
    camera_hw = (int(camera_record["height"]), int(camera_record["width"]))
    if source_hw != camera_hw:
        raise ValueError(f"{colmap_name} image shape {source_hw} does not match COLMAP camera shape {camera_hw}.")

    stem = Path(colmap_name).stem
    source_mask = read_mask(mask_dir / f"{stem}.png", source_hw)
    depth_source = load_depth_from_unidepth_disp(unidepth_disp_dir / f"{stem}.npy", source_hw)

    depth_alignment_stats: dict[str, object]
    if disable_depth_alignment:
        depth_alignment_stats = {"image_name": colmap_name, "scale": 1.0, "disabled": True}
    else:
        depth_source, depth_alignment_stats = align_depth_to_colmap(
            image_record,
            points3d,
            depth_source,
            min_depth_alignment_points,
        )
        depth_alignment_stats["disabled"] = False

    K_target = scale_K(np.asarray(camera_record["K"], dtype=np.float64), source_hw, target_hw)
    depth_target = resize_depth(depth_source, target_hw)
    mask_target = load_modal_mask_or_resize(modal_npz, source_mask, target_hw)

    depth_name = f"{view_label}_depth.npy"
    mask_name = f"{view_label}_mask.npy"
    np.save(out_dir / depth_name, depth_target.astype(np.float32))
    np.save(out_dir / mask_name, mask_target.astype(np.uint8))
    save_view_config(
        out_dir / f"{view_label}_config.json",
        view_label,
        target_hw[1],
        target_hw[0],
        K_target,
        np.asarray(image_record["world_to_camera"], dtype=np.float64),
        depth_name,
        mask_name,
    )

    camera_to_world = np.linalg.inv(np.asarray(image_record["world_to_camera"], dtype=np.float64))
    return {
        "view_label": view_label,
        "colmap_name": colmap_name,
        "image_id": int(image_record["image_id"]),
        "camera_id": int(image_record["camera_id"]),
        "source_height": int(source_hw[0]),
        "source_width": int(source_hw[1]),
        "target_height": int(target_hw[0]),
        "target_width": int(target_hw[1]),
        "qvec": np.asarray(image_record["qvec"], dtype=float).tolist(),
        "tvec": np.asarray(image_record["tvec"], dtype=float).tolist(),
        "camera_center_world": camera_to_world[:3, 3].astype(float).tolist(),
        "depth_alignment": depth_alignment_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export modal_surface inputs from registered COLMAP reference views.")
    parser.add_argument("--model-dir", required=True, type=Path, help="COLMAP TXT model directory containing cameras/images/points3D.txt.")
    parser.add_argument("--image-dir", required=True, type=Path, help="COLMAP image root directory.")
    parser.add_argument("--mask-dir", required=True, type=Path, help="Reference foreground mask directory, keyed by view stem.")
    parser.add_argument("--unidepth-disp-dir", required=True, type=Path, help="UniDepth disparity .npy directory, keyed by view stem.")
    parser.add_argument("--view1-name", required=True, help="View 1 COLMAP image name, e.g. refs/view1.png.")
    parser.add_argument("--view2-name", required=True, help="View 2 COLMAP image name, e.g. refs/view2.png.")
    parser.add_argument("--view1-modal-npz", required=True, type=Path, help="View 1 modal_analysis npz.")
    parser.add_argument("--view2-modal-npz", required=True, type=Path, help="View 2 modal_analysis npz.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for modal_surface geometry files.")
    parser.add_argument("--min-depth-alignment-points", type=int, default=20, help="Minimum sparse observations for per-view depth scale alignment.")
    parser.add_argument("--disable-depth-alignment", action="store_true", help="Keep UniDepth depth in its original inverse-disparity scale.")
    args = parser.parse_args()

    if args.min_depth_alignment_points < 1:
        raise ValueError("--min-depth-alignment-points must be positive.")

    cameras = parse_cameras(args.model_dir / "cameras.txt")
    images = parse_images(args.model_dir / "images.txt")
    points3d = parse_points3d(args.model_dir / "points3D.txt")

    missing = [name for name in [args.view1_name, args.view2_name] if name not in images]
    if missing:
        raise KeyError(f"Missing reference views in images.txt: {missing}. Found names include: {sorted(images.keys())[:20]}")

    target_hw1 = load_modal_shape(args.view1_modal_npz)
    target_hw2 = load_modal_shape(args.view2_modal_npz)
    if target_hw1 != target_hw2:
        raise ValueError(f"View modal shapes must match, got {target_hw1} and {target_hw2}.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    view1 = images[args.view1_name]
    view2 = images[args.view2_name]
    camera1 = cameras[int(view1["camera_id"])]
    camera2 = cameras[int(view2["camera_id"])]
    stats1 = export_view(
        "view1",
        args.view1_name,
        view1,
        camera1,
        args.image_dir,
        args.mask_dir,
        args.unidepth_disp_dir,
        args.view1_modal_npz,
        target_hw1,
        points3d,
        args.min_depth_alignment_points,
        args.disable_depth_alignment,
        args.out_dir,
    )
    stats2 = export_view(
        "view2",
        args.view2_name,
        view2,
        camera2,
        args.image_dir,
        args.mask_dir,
        args.unidepth_disp_dir,
        args.view2_modal_npz,
        target_hw2,
        points3d,
        args.min_depth_alignment_points,
        args.disable_depth_alignment,
        args.out_dir,
    )

    w2c1 = np.asarray(view1["world_to_camera"], dtype=np.float64)
    w2c2 = np.asarray(view2["world_to_camera"], dtype=np.float64)
    view1_to_view2 = w2c2 @ np.linalg.inv(w2c1)
    baseline = float(np.linalg.norm(np.linalg.inv(w2c1)[:3, 3] - np.linalg.inv(w2c2)[:3, 3]))
    payload = {
        "model_dir": str(args.model_dir),
        "image_dir": str(args.image_dir),
        "target_height": int(target_hw1[0]),
        "target_width": int(target_hw1[1]),
        "depth_alignment_disabled": bool(args.disable_depth_alignment),
        "view1": stats1,
        "view2": stats2,
        "view1_to_view2": view1_to_view2.astype(float).tolist(),
        "baseline_colmap_units": baseline,
        "outputs": {
            "view1_config": "view1_config.json",
            "view2_config": "view2_config.json",
            "view1_depth": "view1_depth.npy",
            "view2_depth": "view2_depth.npy",
            "view1_mask": "view1_mask.npy",
            "view2_mask": "view2_mask.npy",
        },
    }
    with (args.out_dir / "colmap_ref_pose_stats.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"view1 registered as {args.view1_name} with image_id={view1['image_id']}")
    print(f"view2 registered as {args.view2_name} with image_id={view2['image_id']}")
    print(f"COLMAP baseline: {baseline:.6g}")
    print(f"saved modal_surface geometry -> {args.out_dir}")


if __name__ == "__main__":
    main()
