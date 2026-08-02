import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
import yaml
from loguru import logger as guru

from flow3d.renderer import Renderer
from modal_surface.io import load_view_config

torch.set_float32_matmul_precision("high")


@dataclass
class RenderConfig:
    work_dir: str
    port: int = 8890
    ckpt_path: str | None = None
    vggt_view_config: tuple[str, ...] = ()
    modal_anchor_manifest: str | None = None
    modal_spectrum_manifest: str | None = None
    modal_spectrum_flow_caches: tuple[str, ...] = ()
    modal_spectrum_comparison_cache_dir: str | None = None
    modal_spectrum_preview_percentile: float = 99.0


def _ordered_vggt_view_configs(
    view_config_paths: tuple[str, ...],
    frame_map_path: str,
) -> tuple[str, ...]:
    path = Path(frame_map_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Modal frame map does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"Modal frame map must use version 1: {path}")
    view_ids = payload.get("views")
    if (
        not isinstance(view_ids, list)
        or not view_ids
        or any(not isinstance(view_id, str) or not view_id for view_id in view_ids)
        or len(set(view_ids)) != len(view_ids)
    ):
        raise ValueError(f"Modal frame map has invalid views: {path}")

    config_by_view_id: dict[str, str] = {}
    for config_path in view_config_paths:
        view_id = load_view_config(config_path).view_id
        if view_id in config_by_view_id:
            raise ValueError(f"Duplicate VGGT view config for view_id={view_id!r}")
        config_by_view_id[view_id] = config_path
    if set(config_by_view_id) != set(view_ids):
        raise ValueError(
            "VGGT view config IDs must exactly match modal frame-map views"
        )
    return tuple(config_by_view_id[view_id] for view_id in view_ids)


def _modal_frame_map_from_training_config(train_cfg: dict) -> str:
    data_cfg = train_cfg.get("data")
    data_value = data_cfg.get("modal_frame_map") if isinstance(data_cfg, dict) else None
    top_value = train_cfg.get("modal_frame_map")
    if top_value and data_value and os.path.normpath(str(top_value)) != os.path.normpath(
        str(data_value)
    ):
        raise ValueError(
            "Training config has conflicting top-level and data.modal_frame_map values"
        )
    frame_map = data_value or top_value
    if not isinstance(frame_map, str) or not frame_map:
        raise ValueError("Modal rendering requires modal_frame_map in cfg.yaml")
    return frame_map


def main(cfg: RenderConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = cfg.ckpt_path or f"{cfg.work_dir}/checkpoints/last.ckpt"
    assert os.path.exists(ckpt_path)

    train_cfg_path = f"{cfg.work_dir}/cfg.yaml"
    with open(train_cfg_path, "r") as file:
        train_cfg = yaml.load(file, Loader=yaml.FullLoader)

    vggt_view_configs = cfg.vggt_view_config or tuple(
        train_cfg.get("vggt_view_configs") or ()
    )
    if train_cfg.get("trajectory_type") == "modal_activation":
        vggt_view_configs = _ordered_vggt_view_configs(
            tuple(vggt_view_configs),
            _modal_frame_map_from_training_config(train_cfg),
        )
    modal_anchor_manifest = cfg.modal_anchor_manifest
    if modal_anchor_manifest is None:
        binding_diag_path = Path(ckpt_path).with_suffix(".modal_binding.json")
        if binding_diag_path.exists():
            with binding_diag_path.open("r", encoding="utf-8") as f:
                binding_diag = json.load(f)
            modal_anchor_manifest = binding_diag["gaussian_modal_manifest"]

    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        use_2dgs=train_cfg["use_2dgs"],
        work_dir=cfg.work_dir,
        port=cfg.port,
        vggt_view_configs=vggt_view_configs,
        modal_anchor_manifest=modal_anchor_manifest,
        modal_spectrum_manifest=cfg.modal_spectrum_manifest,
        modal_spectrum_flow_caches=cfg.modal_spectrum_flow_caches,
        modal_spectrum_comparison_cache_dir=(
            cfg.modal_spectrum_comparison_cache_dir
        ),
        modal_spectrum_preview_percentile=cfg.modal_spectrum_preview_percentile,
    )

    guru.info(f"Starting rendering from {renderer.global_step=}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))
