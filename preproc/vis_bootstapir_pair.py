import argparse
import os

import cv2
import imageio.v2 as iio
import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def list_image_stems(img_dir):
    return {
        os.path.splitext(name)[0]
        for name in os.listdir(img_dir)
        if name.lower().endswith(IMAGE_EXTS)
    }


def find_image_path(img_dir, stem):
    matches = [
        os.path.join(img_dir, name)
        for name in os.listdir(img_dir)
        if os.path.splitext(name)[0] == stem and name.lower().endswith(IMAGE_EXTS)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely find image for {stem} in {img_dir}. "
            f"Found matches: {matches}"
        )
    return matches[0]


def read_rgb(path):
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    return img[..., :3]


def parse_track_pair(track_path, img_dir):
    base = os.path.basename(track_path).replace(".npy", "")
    image_stems = list_image_stems(img_dir)

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


def tapir_score(tracks):
    occlusions = tracks[:, 2]
    expected_dist = tracks[:, 3]
    visibility = 1.0 - sigmoid(occlusions)
    confidence = 1.0 - sigmoid(expected_dist)
    return visibility * confidence


def select_points(query_tracks, target_tracks, max_points, min_motion, min_confidence):
    q_pts = query_tracks[:, :2]
    t_pts = target_tracks[:, :2]
    flow = t_pts - q_pts
    scores = tapir_score(target_tracks)

    motion = np.linalg.norm(flow, axis=1)
    valid = np.isfinite(target_tracks).all(axis=1) & np.isfinite(query_tracks).all(axis=1)
    valid &= motion >= min_motion
    valid &= scores >= min_confidence

    idx = np.where(valid)[0]
    if len(idx) > max_points:
        step = max(1, len(idx) // max_points)
        idx = idx[::step][:max_points]
    return idx, motion, scores


def in_bounds(pt, width, height):
    return (
        np.isfinite(pt).all()
        and 0 <= pt[0] < width
        and 0 <= pt[1] < height
    )


def draw_pair(q_img, t_img, q_pts, t_pts, idx, colors):
    h, w = q_img.shape[:2]
    canvas = np.concatenate([q_img.copy(), t_img.copy()], axis=1)

    for color, i in zip(colors, idx):
        qx, qy = q_pts[i]
        tx, ty = t_pts[i]
        c = tuple(int(v) for v in color.tolist())

        qxy = (int(round(qx)), int(round(qy)))
        txy = (int(round(tx)) + w, int(round(ty)))

        if not in_bounds((qx, qy), w, h):
            continue
        if not (w <= txy[0] < 2 * w and 0 <= txy[1] < h):
            continue

        cv2.circle(canvas, qxy, 2, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, txy, 2, c, -1, cv2.LINE_AA)
        cv2.line(canvas, qxy, txy, c, 1, cv2.LINE_AA)
    return canvas


def load_track_sequence(track_dir, img_dir, query_name, query_tracks, idx):
    target_names = []
    points = []
    scores = []

    for target_name in sorted(list_image_stems(img_dir)):
        track_path = os.path.join(track_dir, f"{query_name}_{target_name}.npy")
        if not os.path.exists(track_path):
            continue
        tracks = np.load(track_path)
        if tracks.shape != query_tracks.shape:
            raise ValueError(
                f"Track shape mismatch for {track_path}: "
                f"target {tracks.shape}, query {query_tracks.shape}"
            )
        target_names.append(target_name)
        points.append(tracks[idx, :2])
        scores.append(tapir_score(tracks)[idx])

    if not target_names:
        raise FileNotFoundError(
            f"No target tracks found for query {query_name} in {track_dir}"
        )
    return target_names, np.stack(points, axis=0), np.stack(scores, axis=0)


def write_track_video(
    img_dir,
    target_names,
    points,
    scores,
    colors,
    out_path,
    fps,
    trail,
    point_radius,
    line_width,
    min_confidence,
):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with iio.get_writer(out_path, fps=fps, macro_block_size=1) as writer:
        for frame_idx, target_name in enumerate(target_names):
            img = read_rgb(find_image_path(img_dir, target_name)).copy()
            h, w = img.shape[:2]
            start = max(0, frame_idx - trail)

            for point_idx, color in enumerate(colors):
                c = tuple(int(v) for v in color.tolist())
                for prev_idx in range(start, frame_idx):
                    if (
                        scores[prev_idx, point_idx] < min_confidence
                        or scores[prev_idx + 1, point_idx] < min_confidence
                    ):
                        continue
                    p0 = points[prev_idx, point_idx]
                    p1 = points[prev_idx + 1, point_idx]
                    if not (in_bounds(p0, w, h) and in_bounds(p1, w, h)):
                        continue
                    cv2.line(
                        img,
                        (int(round(p0[0])), int(round(p0[1]))),
                        (int(round(p1[0])), int(round(p1[1]))),
                        c,
                        line_width,
                        cv2.LINE_AA,
                    )

                if scores[frame_idx, point_idx] < min_confidence:
                    continue
                pt = points[frame_idx, point_idx]
                if not in_bounds(pt, w, h):
                    continue
                cv2.circle(
                    img,
                    (int(round(pt[0])), int(round(pt[1]))),
                    point_radius,
                    c,
                    -1,
                    cv2.LINE_AA,
                )

            writer.append_data(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--video_out", default="")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--trail", type=int, default=20)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--line_width", type=int, default=1)
    parser.add_argument("--max_points", type=int, default=1500)
    parser.add_argument("--min_motion", type=float, default=0.0)
    parser.add_argument("--min_confidence", type=float, default=0.5)
    args = parser.parse_args()

    q, t = parse_track_pair(args.track, args.img_dir)

    q_img = read_rgb(find_image_path(args.img_dir, q))
    t_img = read_rgb(find_image_path(args.img_dir, t))

    tracks = np.load(args.track)
    query_track_path = os.path.join(os.path.dirname(args.track), f"{q}_{q}.npy")
    query_tracks = np.load(query_track_path)

    if tracks.shape != query_tracks.shape:
        raise ValueError(
            f"Track shape mismatch: target {tracks.shape}, query {query_tracks.shape}"
        )

    q_pts = query_tracks[:, :2]
    t_pts = tracks[:, :2]
    idx, motion, scores = select_points(
        query_tracks,
        tracks,
        args.max_points,
        args.min_motion,
        args.min_confidence,
    )

    rng = np.random.default_rng(1)
    colors = rng.integers(40, 255, size=(len(idx), 3), dtype=np.uint8)
    canvas = draw_pair(q_img, t_img, q_pts, t_pts, idx, colors)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    iio.imwrite(args.out, canvas)
    print(f"saved {args.out}")
    print(
        f"tracks={len(tracks)} drawn={len(idx)} "
        f"motion min/mean/max={motion.min():.3f}/{motion.mean():.3f}/{motion.max():.3f} "
        f"tapir_score min/mean/max={scores.min():.3f}/{scores.mean():.3f}/{scores.max():.3f}"
    )

    if args.video_out:
        target_names, points, video_scores = load_track_sequence(
            os.path.dirname(args.track), args.img_dir, q, query_tracks, idx
        )
        write_track_video(
            args.img_dir,
            target_names,
            points,
            video_scores,
            colors,
            args.video_out,
            args.fps,
            args.trail,
            args.point_radius,
            args.line_width,
            args.min_confidence,
        )
        print(
            f"saved {args.video_out} "
            f"frames={len(target_names)} points={len(idx)} query={q}"
        )


if __name__ == "__main__":
    main()
