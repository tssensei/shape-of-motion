import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

COLMAP_READER_PATH = Path(__file__).resolve().parents[1] / "flow3d" / "data" / "colmap.py"
spec = importlib.util.spec_from_file_location("flow3d_data_colmap", COLMAP_READER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {COLMAP_READER_PATH}")
colmap_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap_reader)
read_cameras_binary = colmap_reader.read_cameras_binary
read_images_binary = colmap_reader.read_images_binary
read_points3d_binary = colmap_reader.read_points3d_binary


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
INVALID_POINT3D = -1


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    return image_paths


def image_map_by_basename(images):
    out = {}
    for image_id, image in images.items():
        name = os.path.basename(image.name)
        if name in out:
            raise ValueError(f"Duplicate COLMAP image basename: {name}")
        out[name] = image_id
    return out


def read_foreground_mask(mask_path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    require_file(mask_path)
    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask.reshape((*mask.shape[:2], -1)).max(axis=-1)
    return mask > 0


def load_colmap_model(colmap_dir: Path):
    cameras_path = colmap_dir / "cameras.bin"
    images_path = colmap_dir / "images.bin"
    points_path = colmap_dir / "points3D.bin"
    require_file(cameras_path)
    require_file(images_path)
    require_file(points_path)
    return (
        read_cameras_binary(cameras_path),
        read_images_binary(images_path),
        read_points3d_binary(points_path),
    )


def fit_disp_alignment(colmap_disp: np.ndarray, mono_disp: np.ndarray):
    ms_colmap_disp = colmap_disp - np.median(colmap_disp) + 1e-8
    ms_mono_disp = mono_disp - np.median(mono_disp) + 1e-8
    valid = np.isfinite(ms_colmap_disp) & np.isfinite(ms_mono_disp)
    valid &= np.abs(ms_mono_disp) > 1e-8
    if not np.any(valid):
        raise ValueError("No valid sparse disparities for scale fitting")
    scale = np.median(ms_colmap_disp[valid] / ms_mono_disp[valid])
    shift = np.median(colmap_disp[valid] - scale * mono_disp[valid])
    if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0:
        raise ValueError(f"Invalid alignment scale/shift: {scale}, {shift}")
    return scale, shift


def align_frame(
    image_path: Path,
    colmap_image,
    camera,
    points3d,
    mono_disp_dir: Path,
    mask_dir: Path,
    min_points: int,
) -> np.ndarray:
    import imageio.v2 as imageio

    image = imageio.imread(image_path)
    height, width = image.shape[:2]
    if camera.width != width or camera.height != height:
        raise ValueError(
            f"COLMAP camera size for {image_path.name} is "
            f"{camera.width}x{camera.height}, but image is {width}x{height}"
        )

    mono_path = mono_disp_dir / f"{image_path.stem}.npy"
    require_file(mono_path)
    mono_disp_map = np.load(mono_path).astype(np.float32)
    if mono_disp_map.shape != (height, width):
        raise ValueError(
            f"Mono disparity shape {mono_disp_map.shape} does not match image "
            f"shape {(height, width)} for {image_path.name}"
        )

    mask_path = mask_dir / f"{image_path.stem}.png"
    fg_mask = read_foreground_mask(mask_path)
    if fg_mask.shape != (height, width):
        raise ValueError(
            f"Mask shape {fg_mask.shape} does not match image shape "
            f"{(height, width)} for {image_path.name}"
        )

    point_ids = colmap_image.point3D_ids
    valid_obs = point_ids != INVALID_POINT3D
    point_ids = point_ids[valid_obs]
    xys = colmap_image.xys[valid_obs].astype(np.float32)
    if point_ids.shape[0] == 0:
        raise ValueError(f"No registered COLMAP points for {image_path.name}")

    xyz_world = []
    for point_id in point_ids:
        if int(point_id) not in points3d:
            raise KeyError(f"Point3D id {point_id} from {image_path.name} is missing")
        xyz_world.append(points3d[int(point_id)].xyz)
    xyz_world = np.asarray(xyz_world, dtype=np.float64)

    xyz_cam = colmap_image.qvec2rotmat().dot(xyz_world.T).T + colmap_image.tvec
    colmap_depth = xyz_cam[:, 2]

    xi = np.rint(xys[:, 0]).astype(np.int64)
    yi = np.rint(xys[:, 1]).astype(np.int64)
    inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    depth_valid = np.isfinite(colmap_depth) & (colmap_depth > 1e-6)
    background = np.zeros_like(inside)
    background[inside] = ~fg_mask[yi[inside], xi[inside]]

    valid = inside & depth_valid & background
    xys_valid = xys[valid]
    colmap_disp = 1.0 / np.clip(colmap_depth[valid], a_min=1e-6, a_max=1e6)
    if xys_valid.shape[0] < min_points:
        raise ValueError(
            f"Only {xys_valid.shape[0]} background sparse points before depth "
            f"sampling for {image_path.name}; require at least {min_points}"
        )
    import cv2

    mono_disp = cv2.remap(
        mono_disp_map,
        xys_valid[None].astype(np.float32),
        None,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )[0]
    sample_valid = np.isfinite(mono_disp) & (mono_disp > 0)
    sample_valid &= np.isfinite(colmap_disp) & (colmap_disp > 0)
    if int(sample_valid.sum()) < min_points:
        raise ValueError(
            f"Only {int(sample_valid.sum())} valid background sparse points for "
            f"{image_path.name}; require at least {min_points}"
        )

    scale, shift = fit_disp_alignment(
        colmap_disp[sample_valid], mono_disp[sample_valid]
    )
    aligned_disp = (scale * mono_disp_map + shift).astype(np.float32)
    min_thre = min(1e-6, float(np.quantile(aligned_disp, 0.01)))
    aligned_disp[aligned_disp < min_thre] = 0.0
    return aligned_disp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mono-disp-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-points", type=int, default=30)
    args = parser.parse_args()

    if not args.mono_disp_dir.is_dir():
        raise NotADirectoryError(args.mono_disp_dir)
    if not args.mask_dir.is_dir():
        raise NotADirectoryError(args.mask_dir)
    if args.min_points <= 0:
        raise ValueError("--min-points must be positive")

    cameras, colmap_images, points3d = load_colmap_model(args.colmap_dir)
    colmap_image_idcs = image_map_by_basename(colmap_images)
    image_paths = list_images(args.image_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    for image_path in tqdm(image_paths, desc="Aligning depth with COLMAP"):
        if image_path.name not in colmap_image_idcs:
            raise KeyError(f"Image {image_path.name} is not registered in COLMAP")
        colmap_image = colmap_images[colmap_image_idcs[image_path.name]]
        camera = cameras[colmap_image.camera_id]
        aligned_disp = align_frame(
            image_path,
            colmap_image,
            camera,
            points3d,
            args.mono_disp_dir,
            args.mask_dir,
            args.min_points,
        )
        np.save(args.out_dir / f"{image_path.stem}.npy", aligned_disp)

    out_files = sorted(args.out_dir.glob("*.npy"))
    if len(out_files) != len(image_paths):
        raise ValueError(
            f"Expected {len(image_paths)} output depth files, found {len(out_files)}"
        )
    print(f"Wrote {len(out_files)} aligned disparities to {args.out_dir}")


if __name__ == "__main__":
    main()
