import argparse
import os

import cv2
import imageio.v2 as iio
import numpy as np


def parse_track_pair(track_path, img_dir):
    base = os.path.basename(track_path).replace(".npy", "")
    image_stems = {
        os.path.splitext(name)[0]
        for name in os.listdir(img_dir)
        if name.lower().endswith(".png")
    }

    candidates = []
    for q in image_stems:
        prefix = q + "_"
        if base.startswith(prefix):
            t = base[len(prefix) :]
            if t in image_stems:
                candidates.append((q, t))

    if len(candidates) != 1:
        raise ValueError(
            f"Could not uniquely parse query/target names from {base}. "
            f"Found candidates: {candidates}"
        )
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_points", type=int, default=1500)
    parser.add_argument("--min_motion", type=float, default=0.0)
    parser.add_argument("--min_confidence", type=float, default=0.5)
    args = parser.parse_args()

    q, t = parse_track_pair(args.track, args.img_dir)

    q_img = iio.imread(os.path.join(args.img_dir, q + ".png"))[..., :3]
    t_img = iio.imread(os.path.join(args.img_dir, t + ".png"))[..., :3]

    tracks = np.load(args.track)
    query_track_path = os.path.join(os.path.dirname(args.track), f"{q}_{q}.npy")
    query_tracks = np.load(query_track_path)

    if tracks.shape != query_tracks.shape:
        raise ValueError(
            f"Track shape mismatch: target {tracks.shape}, query {query_tracks.shape}"
        )

    q_pts = query_tracks[:, :2]
    t_pts = tracks[:, :2]
    flow = t_pts - q_pts

    occlusions = tracks[:, 2]
    expected_dist = tracks[:, 3]
    visibility = 1.0 - 1.0 / (1.0 + np.exp(-occlusions))
    confidence = 1.0 - 1.0 / (1.0 + np.exp(-expected_dist))
    tapir_score = visibility * confidence

    motion = np.linalg.norm(flow, axis=1)
    valid = np.isfinite(tracks).all(axis=1) & np.isfinite(query_tracks).all(axis=1)
    valid &= motion >= args.min_motion
    valid &= tapir_score >= args.min_confidence

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

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    iio.imwrite(args.out, canvas)
    print(f"saved {args.out}")
    print(
        f"tracks={len(tracks)} drawn={len(idx)} "
        f"motion min/mean/max={motion.min():.3f}/{motion.mean():.3f}/{motion.max():.3f} "
        f"tapir_score min/mean/max={tapir_score.min():.3f}/{tapir_score.mean():.3f}/{tapir_score.max():.3f}"
    )


if __name__ == "__main__":
    main()
