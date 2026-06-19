import argparse
from pathlib import Path

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    return image_paths


def read_foreground_mask(mask_path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask.reshape((*mask.shape[:2], -1)).max(axis=-1)
    return mask > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dilate", type=int, default=15)
    args = parser.parse_args()

    if not args.mask_dir.is_dir():
        raise NotADirectoryError(args.mask_dir)
    if args.dilate < 0:
        raise ValueError("--dilate must be non-negative")

    image_paths = list_images(args.image_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    kernel = None
    if args.dilate > 0:
        import cv2

        kernel = np.ones((args.dilate, args.dilate), dtype=np.uint8)

    for image_path in image_paths:
        import imageio.v2 as imageio

        image = imageio.imread(image_path)
        mask_path = args.mask_dir / f"{image_path.stem}.png"
        fg_mask = read_foreground_mask(mask_path)
        if fg_mask.shape != image.shape[:2]:
            raise ValueError(
                f"Mask shape {fg_mask.shape} does not match image shape "
                f"{image.shape[:2]} for {image_path.name}"
            )

        if kernel is not None:
            fg_mask = cv2.dilate(fg_mask.astype(np.uint8), kernel, iterations=1) > 0

        colmap_mask = np.full(fg_mask.shape, 255, dtype=np.uint8)
        colmap_mask[fg_mask] = 0
        imageio.imwrite(args.out_dir / f"{image_path.name}.png", colmap_mask)

    out_files = sorted(args.out_dir.glob("*.png"))
    if len(out_files) != len(image_paths):
        raise ValueError(
            f"Expected {len(image_paths)} output masks, found {len(out_files)}"
        )
    print(f"Wrote {len(out_files)} COLMAP masks to {args.out_dir}")


if __name__ == "__main__":
    main()
