import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from loguru import logger as guru

from flow3d.renderer import Renderer

import yaml

torch.set_float32_matmul_precision("high")


@dataclass
class RenderConfig:
    work_dir: str
    port: int = 8890
    ckpt_path: str | None = None
    vggt_view_config: tuple[str, ...] = ()
    modal_anchor_manifest: str | None = None


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
    modal_anchor_manifest = cfg.modal_anchor_manifest
    if modal_anchor_manifest is None:
        binding_diag_path = Path(ckpt_path).with_suffix(".modal_binding.json")
        if binding_diag_path.exists():
            with binding_diag_path.open("r", encoding="utf-8") as f:
                binding_diag = json.load(f)
            modal_anchor_manifest = binding_diag.get("modal_manifest")

    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        use_2dgs=train_cfg["use_2dgs"],
        work_dir=cfg.work_dir,
        port=cfg.port,
        vggt_view_configs=vggt_view_configs,
        modal_anchor_manifest=modal_anchor_manifest,
    )

    guru.info(f"Starting rendering from {renderer.global_step=}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))
