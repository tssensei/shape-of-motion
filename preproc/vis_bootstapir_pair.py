import argparse
import os

import cv2
import imageio.v2 as iio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_points", type=int, default=1500)
    parser.add_argument("--min_motion", type=float, default=0.0)
    args = parser.parse_args()

    base = os.path.basename(args.track).replace(".npy", "")
    q, t = base.split("_")

    q_img = iio.imread(os.path.join(args.img_dir, q + ".png"))[..., :3]
    t_img = iio.imread(os.path.join(args.img_dir, t + ".png"))[..., :3]

    tracks = np.load(args.track)
    q_pts = tracks[:, :2]
    flow = tracks[:, 2:4]
    t_pts = q_pts + flow

    motion = np.linalg.norm(flow, axis=1)
    valid = np.isfinite(tracks).all(axis=1)
    valid &= motion >= args.min_motion

    idx = np.where(valid)[0]
    if len(idx) > args.max_points:
        step = max(1, len(idx) // args.max_points)
        idx = idx[::step][: args.max_points]

    h, w = q_img.shape[:2]
    canvas = np.concatenate([q_img.copy(), t_img.copy()], axis=1)

    rng = np.random.default_rng(1)
    colors = rng.integers(40, 255, size=(len(idx), 3), dtype=np.uint8)

    for color, i in zip(colors, idx):
        qx, qy = q_pts[i]
        tx, ty = t_pts[i]
        c = tuple(int(v) for v in color.tolist())

        qxy = (int(round(qx)), int(round(qy)))
        txy = (int(round(tx)) + w, int(round(ty)))

        if not (0 <= qxy[0] < w and 0 <= qxy[1] < h):
            continue
        if not (w <= txy[0] < 2 * w and 0 <= txy[1] < h):
            continue

        cv2.circle(canvas, qxy, 2, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, txy, 2, c, -1, cv2.LINE_AA)
        cv2.line(canvas, qxy, txy, c, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    iio.imwrite(args.out, canvas)
    print(f"saved {args.out}")
    print(f"tracks={len(tracks)} drawn={len(idx)} motion min/mean/max={motion.min():.3f}/{motion.mean():.3f}/{motion.max():.3f}")


if __name__ == "__main__":
    main()
