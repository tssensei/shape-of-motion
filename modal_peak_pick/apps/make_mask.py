from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector
import numpy as np

from modal_peak_pick.core.video_io import load_video_clip, probe_video


class MaskMaker:
    def __init__(self, ax):
        self.verts = None
        self.selector = PolygonSelector(ax, self.onselect, useblit=True)

    def onselect(self, verts):
        self.verts = verts
        plt.close()


def polygon_to_mask(h: int, w: int, verts) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    path = MplPath(verts)
    return path.contains_points(points).reshape(h, w)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--t", type=float, default=0.0, help="Frame time in seconds for drawing the mask.")
    parser.add_argument("--resize", type=int, default=None, help="Resize max(H,W) for mask drawing.")
    parser.add_argument("--out", required=True, help="Output .npy mask path.")


def run(args: argparse.Namespace) -> None:
    info = probe_video(args.video)
    fps = info.fps if info.fps > 0 else 30.0
    frames, _ = load_video_clip(
        args.video,
        t0=args.t,
        t1=args.t + 1.0 / fps,
        resize=args.resize,
        grayscale=True,
        max_frames=1,
    )
    frame = frames[0]
    h, w = frame.shape

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Modal mask drawing")
    ax.set_title("Click polygon vertices around ROI, then close polygon to save.")
    ax.imshow(frame, cmap="gray", vmin=0.0, vmax=1.0)
    mask_maker = MaskMaker(ax)
    plt.show()

    if mask_maker.verts is None or len(mask_maker.verts) < 3:
        raise RuntimeError("No polygon drawn; at least 3 vertices are required.")

    mask = polygon_to_mask(h, w, mask_maker.verts).astype(np.uint8)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), mask)
    print(f"Saved mask: {out_path}")
    print(f"Mask coverage: {mask.mean() * 100.0:.2f}%")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draw a polygon ROI mask for modal peak picking.")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

