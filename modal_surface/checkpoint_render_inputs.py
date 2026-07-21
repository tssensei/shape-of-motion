"""Load static foreground Gaussian attributes and rendered camera inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modal_surface.io import load_view_config


def load_fg_means_from_checkpoint(path: str) -> np.ndarray:
    """Load only canonical foreground centers without rendering camera inputs."""
    import torch
    from flow3d.scene_model import SceneModel

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state")
    scene_model = SceneModel.init_from_state_dict(state)
    fg_means = (
        scene_model.fg.params["means"]
        .detach()
        .cpu()
        .float()
        .numpy()
        .astype(np.float32)
    )
    if fg_means.ndim != 2 or fg_means.shape[1] != 3:
        raise ValueError(
            f"fg.params.means must have shape (N,3), got {fg_means.shape}"
        )
    if not np.all(np.isfinite(fg_means)):
        raise ValueError(f"{path} contains non-finite foreground Gaussian centers")
    return fg_means


def load_fg_pixel_candidate_inputs_from_checkpoint(
    path: str,
    view_config_paths: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
]:
    import torch
    from flow3d.scene_model import SceneModel

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sceneModel = SceneModel.init_from_state_dict(state).to(device)
    sceneModel.eval()
    fg_means = sceneModel.fg.params["means"].detach().cpu().float().numpy().astype(np.float32)
    fg_scales = sceneModel.fg.get_scales().detach().cpu().float().numpy().astype(np.float32)
    fg_quats = sceneModel.fg.get_quats().detach().cpu().float().numpy().astype(np.float32)
    fg_opacities = sceneModel.fg.get_opacities().detach().cpu().float().numpy().reshape(-1).astype(np.float32)
    fg_colors = sceneModel.fg.get_colors().detach().cpu().float().numpy().astype(np.float32)
    if fg_means.ndim != 2 or fg_means.shape[1] != 3:
        raise ValueError(f"fg.params.means must have shape (N,3), got {fg_means.shape}")
    if fg_scales.shape != fg_means.shape:
        raise ValueError(f"activated foreground scales must have shape {fg_means.shape}, got {fg_scales.shape}")
    if fg_quats.shape != (fg_means.shape[0], 4):
        raise ValueError(f"activated foreground quats must have shape ({fg_means.shape[0]},4), got {fg_quats.shape}")
    if fg_opacities.shape != (fg_means.shape[0],):
        raise ValueError(f"activated foreground opacities must have shape ({fg_means.shape[0]},), got {fg_opacities.shape}")
    if fg_colors.shape != fg_means.shape:
        raise ValueError(
            f"activated foreground RGB must have shape {fg_means.shape}, got {fg_colors.shape}"
        )
    if not np.all(np.isfinite(fg_means)):
        raise ValueError(f"{path} contains non-finite foreground Gaussian centers")
    if not np.all(np.isfinite(fg_colors)) or np.any(fg_colors < 0.0) or np.any(fg_colors > 1.0):
        raise ValueError(f"{path} contains invalid activated foreground Gaussian RGB")

    rendered_depths: list[np.ndarray] = []
    rendered_accs: list[np.ndarray] = []
    with torch.no_grad():
        for cfg_path in view_config_paths:
            cfg = load_view_config(cfg_path)
            w2c = torch.from_numpy(cfg.world_to_camera).to(device=device, dtype=torch.float32)[None]
            K = torch.from_numpy(cfg.K).to(device=device, dtype=torch.float32)[None]
            rendered = sceneModel.render(
                None,
                w2c,
                K,
                (cfg.image_width, cfg.image_height),
                return_depth=True,
                return_mask=False,
                fg_only=True,
            )
            depth = rendered["depth"][0, ..., 0].detach().cpu().float().numpy().astype(np.float32)
            acc_tensor = rendered["acc"][0]
            if acc_tensor.ndim == 3:
                acc_tensor = acc_tensor[..., 0]
            acc = acc_tensor.detach().cpu().float().numpy().astype(np.float32)
            if depth.shape != (cfg.image_height, cfg.image_width):
                raise ValueError(f"Rendered depth for {cfg.view_id} has unexpected shape {depth.shape}.")
            if acc.shape != (cfg.image_height, cfg.image_width):
                raise ValueError(f"Rendered alpha for {cfg.view_id} has unexpected shape {acc.shape}.")
            rendered_depths.append(depth)
            rendered_accs.append(acc)
    return (
        fg_means,
        fg_scales,
        fg_quats,
        fg_opacities,
        fg_colors,
        rendered_depths,
        rendered_accs,
    )
