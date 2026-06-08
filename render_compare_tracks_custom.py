import argparse
import os
from datetime import datetime

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from flow3d.data.casual_dataset import CasualDataset
from flow3d.renderer import Renderer
from flow3d.vis.utils import draw_tracks_2d, make_video_divisble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--acc-thresh", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CasualDataset(
        data_dir=args.data_dir,
        image_type="images",
        mask_type="masks",
        depth_type="aligned_depth_anything",
        camera_type="droid_recon",
        track_2d_type="bootstapir",
        res="",
        load_from_cache=True,
    )

    ckpt_path = f"{args.work_dir}/checkpoints/last.ckpt"
    assert os.path.exists(ckpt_path), ckpt_path

    cfg_path = f"{args.work_dir}/cfg.yaml"
    use_2dgs = False
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            train_cfg = yaml.safe_load(f)
        use_2dgs = bool(train_cfg.get("use_2dgs", False))

    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        use_2dgs=use_2dgs,
        work_dir=args.work_dir,
        port=None,
    )

    K = dataset.get_Ks()[0].to(device)
    w2cs = dataset.get_w2cs().to(device)
    img_wh = dataset.get_img_wh()
    num_frames = dataset.num_frames
    ts = torch.arange(num_frames, device=device)

    # Select rendered foreground points on frame 0 and ask the model for their full 3D trajectories.
    with torch.inference_mode():
        init_out = renderer.model.render(
            0,
            w2cs[0:1],
            K[None],
            img_wh,
            target_ts=ts,
            return_color=True,
            fg_only=True,
        )

    acc = init_out["acc"][0].squeeze(-1)[:: args.grid, :: args.grid]
    gt_mask = dataset.get_mask(0)[:: args.grid, :: args.grid].to(device)
    mask = (acc > args.acc_thresh) & (gt_mask > 0)

    tracks_3d_map = init_out["tracks_3d"][0][:: args.grid, :: args.grid]
    mask = mask & ~(tracks_3d_map == 0).all(dim=(-1, -2))
    tracks_3d = tracks_3d_map[mask]

    print("selected tracks_3d:", tracks_3d.shape)

    tracks_2d = torch.einsum(
        "ij,bjk,nbk->nbi",
        K,
        w2cs[:, :3],
        F.pad(tracks_3d, (0, 1), value=1.0),
    )
    tracks_2d = tracks_2d[..., :2] / tracks_2d[..., 2:].clamp(min=1e-6)

    video = []
    for i in tqdm(range(num_frames)):
        with torch.inference_mode():
            render_img = renderer.model.render(
                i,
                w2cs[i : i + 1],
                K[None],
                img_wh,
            )["img"][0]

        start = max(0, i - args.window)
        render_tracks = draw_tracks_2d(render_img, tracks_2d[:, start : i + 1])

        input_img = dataset.get_image(i)
        input_img = (input_img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        if render_tracks.shape[0] != input_img.shape[0] or render_tracks.shape[1] != input_img.shape[1]:
            raise ValueError(f"shape mismatch: input {input_img.shape}, render {render_tracks.shape}")

        side_by_side = np.concatenate([input_img[..., :3], render_tracks[..., :3]], axis=1)
        video.append(side_by_side)

    video = np.stack(video, axis=0)

    if args.out:
        out_path = args.out
    else:
        video_dir = f"{args.work_dir}/videos/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        os.makedirs(video_dir, exist_ok=True)
        out_path = f"{video_dir}/input_vs_render_tracks.mp4"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    iio.imwrite(out_path, make_video_divisble(video), fps=args.fps)
    print("saved", out_path)


if __name__ == "__main__":
    main()
