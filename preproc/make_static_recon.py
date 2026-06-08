import argparse
import json
import os

import cv2
import imageio.v2 as iio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    img_files = sorted(
        f for f in os.listdir(args.img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )[:: args.stride]

    with open(args.calib, "r") as f:
        calib_dict = json.load(f)

    first = iio.imread(os.path.join(args.img_dir, img_files[0]))
    h, w = first.shape[:2]

    images = []
    points = []
    masks = []
    tstamps = []

    for t, img_file in enumerate(img_files):
        name = os.path.splitext(img_file)[0]
        img = iio.imread(os.path.join(args.img_dir, img_file))
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        disp = np.load(os.path.join(args.depth_dir, name + ".npy")).astype(np.float32)
        if disp.shape != (h, w):
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR)

        depth = 1.0 / np.clip(disp, 1e-6, 1e6)

        fx, fy, cx, cy = calib_dict[name]
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        x = (xs.astype(np.float32) - fx * 0 + (0 - cx)) / fx * depth
        y = (ys.astype(np.float32) - fy * 0 + (0 - cy)) / fy * depth
        z = depth
        pts = np.stack([x, y, z], axis=-1).astype(np.float32)

        valid = np.isfinite(depth) & (depth > 0)

        images.append(img.astype(np.float32) / 255.0)
        points.append(pts)
        masks.append(valid)
        tstamps.append(t)

    n = len(images)
    traj_c2w = np.tile(np.eye(4, dtype=np.float32)[None], (n, 1, 1))
    map_c2w = traj_c2w.copy()

    fx, fy, cx, cy = calib_dict[os.path.splitext(img_files[0])[0]]
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

    save_dict = {
        "tstamps": np.asarray(tstamps, dtype=np.int64),
        "images": np.stack(images, axis=0),
        "points": np.stack(points, axis=0),
        "masks": np.stack(masks, axis=0),
        "map_c2w": map_c2w,
        "traj_c2w": traj_c2w,
        "intrinsics": intrinsics,
        "img_shape": np.array([h, w], dtype=np.int64),
    }

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.save(args.out_path, np.array(save_dict))
    for k, v in save_dict.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
