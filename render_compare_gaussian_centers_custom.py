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


def project_tracks(
    tracks_3d: torch.Tensor,
    Ks: torch.Tensor,
    w2cs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world-space tracks with per-frame cameras."""
    tracks_cam = torch.einsum(
        "tij,ntj->nti",
        w2cs[:, :3],
        F.pad(tracks_3d, (0, 1), value=1.0),
    )
    tracks_proj = torch.einsum("tij,ntj->nti", Ks, tracks_cam)
    depths = tracks_proj[..., 2]
    tracks_2d = tracks_proj[..., :2] / depths[..., None].clamp(min=1e-6)
    return tracks_2d, depths


def select_gaussian_tracks(
    tracks_3d: torch.Tensor,
    tracks_2d: torch.Tensor,
    depths: torch.Tensor,
    opacities: torch.Tensor,
    query_mask: torch.Tensor,
    query_frame: int,
    img_wh: tuple[int, int],
    max_tracks: int,
    opacity_thresh: float,
    selection: str,
    grid_rows: int,
    grid_cols: int,
    tracks_per_cell: int,
    seed: int,
) -> torch.Tensor:
    W, H = img_wh
    query_xy = tracks_2d[:, query_frame]
    query_depth = depths[:, query_frame]
    finite = torch.isfinite(tracks_3d).all(dim=(1, 2))
    finite &= torch.isfinite(tracks_2d).all(dim=(1, 2))
    finite &= torch.isfinite(depths).all(dim=1)
    in_front = query_depth > 1e-6
    in_frame = (
        (query_xy[:, 0] >= 0)
        & (query_xy[:, 0] <= W - 1)
        & (query_xy[:, 1] >= 0)
        & (query_xy[:, 1] <= H - 1)
    )

    px = query_xy[:, 0].round().long().clamp(0, W - 1)
    py = query_xy[:, 1].round().long().clamp(0, H - 1)
    in_mask = query_mask[py, px] > 0
    opaque = opacities >= opacity_thresh
    valid = finite & in_front & in_frame & in_mask & opaque
    valid_ids = torch.where(valid)[0]
    if len(valid_ids) == 0:
        raise ValueError(
            "No Gaussian centers passed the filters. Try lowering "
            "--opacity-thresh or using a different --query-frame."
        )

    if selection == "uniform":
        selection = "grid-random"

    def score_ids(ids: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "motion":
            deltas = tracks_3d[ids] - tracks_3d[ids, query_frame : query_frame + 1]
            return deltas.norm(dim=-1).amax(dim=-1)
        if mode == "opacity":
            return opacities[ids]
        raise ValueError(f"Selection mode {mode} does not use scores.")

    def rank_ids(ids: torch.Tensor, mode: str, rng) -> torch.Tensor:
        if mode == "random":
            order = rng.permutation(len(ids))
            return ids[torch.from_numpy(order).to(ids.device)]
        scores = score_ids(ids, mode)
        return ids[torch.argsort(scores, descending=True)]

    def take_ranked(ids: torch.Tensor, mode: str, k: int, rng) -> torch.Tensor:
        if k <= 0 or len(ids) <= k:
            return ids
        return rank_ids(ids, mode, rng)[:k]

    grid_prefix = "grid-"
    if selection.startswith(grid_prefix):
        mode = selection[len(grid_prefix) :]
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("--grid-rows and --grid-cols must be positive.")

        valid_xy = query_xy[valid_ids]
        cell_x = ((valid_xy[:, 0] / max(W, 1)) * grid_cols).long()
        cell_y = ((valid_xy[:, 1] / max(H, 1)) * grid_rows).long()
        cell_x = cell_x.clamp(0, grid_cols - 1)
        cell_y = cell_y.clamp(0, grid_rows - 1)
        cell_ids = cell_y * grid_cols + cell_x
        nonempty_cells = torch.unique(cell_ids)
        rng = np.random.default_rng(seed)
        cell_candidates = []
        for cell in nonempty_cells.tolist():
            ids_in_cell = valid_ids[cell_ids == cell]
            ids_in_cell = rank_ids(ids_in_cell, mode, rng)
            if tracks_per_cell > 0:
                ids_in_cell = ids_in_cell[:tracks_per_cell]
            cell_candidates.append(ids_in_cell)

        if max_tracks <= 0:
            selected_ids = torch.cat(cell_candidates, dim=0)
        else:
            selected = []
            offsets = [0 for _ in cell_candidates]
            while len(selected) < max_tracks:
                any_added = False
                for cell_idx, ids_in_cell in enumerate(cell_candidates):
                    if offsets[cell_idx] >= len(ids_in_cell):
                        continue
                    selected.append(ids_in_cell[offsets[cell_idx]])
                    offsets[cell_idx] += 1
                    any_added = True
                    if len(selected) >= max_tracks:
                        break
                if not any_added:
                    break
            selected_ids = torch.stack(selected, dim=0)
        return selected_ids

    if max_tracks <= 0 or len(valid_ids) <= max_tracks:
        return valid_ids

    rng = np.random.default_rng(seed)
    if selection == "motion":
        return take_ranked(valid_ids, "motion", max_tracks, rng)
    if selection == "opacity":
        return take_ranked(valid_ids, "opacity", max_tracks, rng)
    if selection == "random":
        return take_ranked(valid_ids, "random", max_tracks, rng)

    raise ValueError(f"Unknown selection mode: {selection}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--query-frame", type=int, default=0)
    parser.add_argument("--max-tracks", type=int, default=1200)
    parser.add_argument("--opacity-thresh", type=float, default=0.1)
    parser.add_argument(
        "--selection",
        choices=[
            "uniform",
            "grid-motion",
            "grid-opacity",
            "grid-random",
            "motion",
            "opacity",
            "random",
        ],
        default="uniform",
        help=(
            "uniform is spatially balanced grid-random sampling over visible "
            "foreground Gaussian centers; motion/opacity modes bias selection."
        ),
    )
    parser.add_argument("--grid-rows", type=int, default=12)
    parser.add_argument("--grid-cols", type=int, default=16)
    parser.add_argument(
        "--tracks-per-cell",
        type=int,
        default=0,
        help="0 means auto based on --max-tracks and the number of nonempty cells.",
    )
    parser.add_argument(
        "--camera-source",
        choices=["dataset", "model"],
        default="dataset",
        help="Use dataset cameras or camera poses stored in the checkpoint.",
    )
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--seed", type=int, default=42)
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

    num_frames = dataset.num_frames
    query_frame = min(max(args.query_frame, 0), num_frames - 1)
    img_wh = dataset.get_img_wh()
    Ks = dataset.get_Ks().to(device)
    w2cs = dataset.get_w2cs().to(device)
    if args.camera_source == "model":
        if renderer.model.camera_poses is None:
            raise ValueError("Checkpoint does not contain trainable camera poses.")
        w2cs = renderer.model.camera_poses.get_camera_matrix().to(device)

    ts = torch.arange(num_frames, device=device)
    with torch.inference_mode():
        tracks_3d, _ = renderer.model.compute_poses_fg(ts)
        opacities = renderer.model.fg.get_opacities().reshape(-1)
        tracks_2d, depths = project_tracks(tracks_3d, Ks, w2cs)

    query_mask = dataset.get_mask(query_frame).to(device)
    track_ids = select_gaussian_tracks(
        tracks_3d=tracks_3d,
        tracks_2d=tracks_2d,
        depths=depths,
        opacities=opacities,
        query_mask=query_mask,
        query_frame=query_frame,
        img_wh=img_wh,
        max_tracks=args.max_tracks,
        opacity_thresh=args.opacity_thresh,
        selection=args.selection,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        tracks_per_cell=args.tracks_per_cell,
        seed=args.seed,
    )
    tracks_2d = tracks_2d[track_ids]
    print(
        "selected gaussian centers:",
        tracks_2d.shape,
        f"query_frame={query_frame}",
        f"selection={args.selection}",
    )

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

        input_img = dataset.get_image(i)
        input_img = (input_img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        if (
            render_tracks.shape[0] != input_img.shape[0]
            or render_tracks.shape[1] != input_img.shape[1]
        ):
            raise ValueError(
                f"shape mismatch: input {input_img.shape}, render {render_tracks.shape}"
            )

        side_by_side = np.concatenate(
            [input_img[..., :3], render_tracks[..., :3]],
            axis=1,
        )
        video.append(side_by_side)

    video = np.stack(video, axis=0)

    if args.out:
        out_path = args.out
    else:
        video_dir = f"{args.work_dir}/videos/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        os.makedirs(video_dir, exist_ok=True)
        out_path = f"{video_dir}/input_vs_gaussian_center_tracks.mp4"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    iio.imwrite(out_path, make_video_divisble(video), fps=args.fps)
    print("saved", out_path)


if __name__ == "__main__":
    main()
