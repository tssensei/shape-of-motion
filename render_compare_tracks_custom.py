import argparse
import os
from datetime import datetime

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from flow3d.renderer import Renderer
from flow3d.vis.utils import draw_tracks_2d, make_video_divisble


def get_data_path(data_dir: str, data_type: str, res: str) -> str:
    return os.path.join(data_dir, data_type, res)


def list_image_paths(img_dir: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg")
    return [
        os.path.join(img_dir, name)
        for name in sorted(os.listdir(img_dir))
        if name.lower().endswith(exts)
    ]


def load_image(path: str) -> torch.Tensor:
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    return torch.from_numpy(img[..., :3]).float() / 255.0


def load_mask(mask_dir: str, stem: str, image_hw: tuple[int, int]) -> torch.Tensor:
    mask_path = None
    for ext in (".png", ".jpg", ".jpeg"):
        cand = os.path.join(mask_dir, stem + ext)
        if os.path.exists(cand):
            mask_path = cand
            break
    if mask_path is None:
        raise FileNotFoundError(f"Missing mask for {stem} in {mask_dir}")

    mask = iio.imread(mask_path)
    fg = mask.reshape((*mask.shape[:2], -1)).max(axis=-1) > 0
    H, W = image_hw
    if fg.shape != (H, W):
        fg = cv2.resize(fg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return torch.from_numpy(fg).float()


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

    ckpt_path = f"{args.work_dir}/checkpoints/last.ckpt"
    assert os.path.exists(ckpt_path), ckpt_path

    cfg_path = f"{args.work_dir}/cfg.yaml"
    use_2dgs = False
    data_cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            train_cfg = yaml.safe_load(f)
        use_2dgs = bool(train_cfg.get("use_2dgs", False))
        data_cfg = train_cfg.get("data", {})

    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        use_2dgs=use_2dgs,
        work_dir=args.work_dir,
        port=None,
    )

    image_type = data_cfg.get("image_type", "images")
    mask_type = data_cfg.get("mask_type", "masks")
    res = data_cfg.get("res", "")
    img_dir = get_data_path(args.data_dir, image_type, res)
    mask_dir = get_data_path(args.data_dir, mask_type, res)
    image_paths = list_image_paths(img_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    first_img = load_image(image_paths[0])
    H, W = first_img.shape[:2]
    img_wh = (W, H)

    Ks = renderer.model.Ks.to(device)
    w2cs = renderer.model.w2cs.to(device)
    num_frames = min(len(image_paths), renderer.model.num_frames, Ks.shape[0], w2cs.shape[0])
    Ks = Ks[:num_frames]
    w2cs = w2cs[:num_frames]
    ts = torch.arange(num_frames, device=device)

    # Select rendered foreground points on frame 0 and ask the model for their full 3D trajectories.
    with torch.inference_mode():
        init_out = renderer.model.render(
            0,
            w2cs[0:1],
            Ks[0:1],
            img_wh,
            target_ts=ts,
            return_color=True,
            fg_only=True,
        )

    acc = init_out["acc"][0].squeeze(-1)[:: args.grid, :: args.grid]
    frame0_stem = os.path.splitext(os.path.basename(image_paths[0]))[0]
    gt_mask = load_mask(mask_dir, frame0_stem, (H, W))[:: args.grid, :: args.grid].to(device)
    mask = (acc > args.acc_thresh) & (gt_mask > 0)

    tracks_3d_map = init_out["tracks_3d"][0][:: args.grid, :: args.grid]
    mask = mask & ~(tracks_3d_map == 0).all(dim=(-1, -2))
    tracks_3d = tracks_3d_map[mask]

    print("selected tracks_3d:", tracks_3d.shape)

    tracks_cam = torch.einsum(
        "bij,bjk,nbk->nbi",
        Ks,
        w2cs[:, :3],
        F.pad(tracks_3d, (0, 1), value=1.0),
    )
    depths = tracks_cam[..., 2:]
    tracks_2d = tracks_cam[..., :2] / depths.clamp(min=1e-6)
    valid_proj = torch.isfinite(tracks_cam).all(dim=-1) & (depths[..., 0] > 1e-6)
    tracks_2d = torch.where(valid_proj[..., None], tracks_2d, torch.full_like(tracks_2d, torch.nan))

    video = []
    for i in tqdm(range(num_frames)):
        with torch.inference_mode():
            render_img = renderer.model.render(
                i,
                w2cs[i : i + 1],
                Ks[i : i + 1],
                img_wh,
            )["img"][0]

        start = max(0, i - args.window)
        render_tracks = draw_tracks_2d(render_img, tracks_2d[:, start : i + 1])

        input_img = load_image(image_paths[i])
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
