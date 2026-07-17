import os
import os.path as osp
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import torch
import tyro
import yaml
from loguru import logger as guru
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.data import (
    BaseDataset,
    DavisDataConfig,
    CustomDataConfig,
    get_train_val_datasets,
    iPhoneDataConfig,
    NvidiaDataConfig,
)
from flow3d.data.colmap import read_points3d_binary
from flow3d.data.utils import to_device
from flow3d.init_utils import (
    init_bg,
    init_fg_from_point_cloud,
    init_fg_from_tracks_3d,
    init_identity_motion_bases,
    init_motion_params_with_dct,
    init_motion_params_with_procrustes,
    run_initial_optim,
    vis_init_params,
    init_trainable_poses,
)
from flow3d.modal_utils import (
    load_gaussian_modal_fields,
    load_modal_frame_map,
    resolve_required_modal_paths,
)
from flow3d.params import CameraScales, GaussianParams, ModalActivations
from flow3d.scene_model import SceneModel, TRAJECTORY_TYPE_TO_ID
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.trainer import Trainer
from flow3d.validator import Validator
from flow3d.vis.utils import get_server

torch.set_float32_matmul_precision("high")


def set_seed(seed):
    # Set the seed for generating random numbers
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)


@dataclass
class TrainConfig:
    work_dir: str
    data: (
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")]
        | Annotated[DavisDataConfig, tyro.conf.subcommand(name="davis")]
        | Annotated[CustomDataConfig, tyro.conf.subcommand(name="custom")]
        | Annotated[NvidiaDataConfig, tyro.conf.subcommand(name="nvidia")]
    )
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig
    num_fg: int = 40_000
    num_bg: int = 100_000
    num_motion_bases: int = 10
    trajectory_type: Literal[
        "som_basis", "dct_center", "modal_activation", "static"
    ] = "som_basis"
    num_dct_bases: int | None = None
    dct_init: Literal["tracks", "zero"] = "tracks"
    modal_manifest: str | None = None
    modal_frame_map: str | None = None
    modal_carrier_points: str | None = None
    vggt_view_configs: tuple[str, ...] = ()
    modal_warmup_epochs: int = 5
    modal_train_base_means: bool = False
    modal_stage1_data_dir: str | None = None
    modal_stage1_frame_map: str | None = None
    modal_stage1_epochs: int = 0
    modal_stage1_init_ckpt: str | None = None
    modal_stage2_train_base_means: bool = False
    modal_stage2_train_colors: bool = False
    modal_stage2_train_opacities: bool = False
    modal_stage2_train_scales: bool = False
    modal_stage2_train_quats: bool = False
    modal_stage2_train_bg_means: bool = False
    modal_stage2_train_bg_colors: bool = False
    modal_stage2_train_bg_opacities: bool = False
    modal_stage2_train_bg_scales: bool = False
    modal_stage2_train_bg_quats: bool = False
    modal_stage2_lr_fg_scales: float | None = None
    modal_stage2_lr_fg_quats: float | None = None
    modal_train_view_id: str | None = None
    modal_max_local_frames_per_view: int | None = None
    num_epochs: int = 200
    port: int | None = None
    vis_debug: bool = False 
    batch_size: int = 8
    num_dl_workers: int = 4
    validate_every: int = 50
    save_videos_every: int = 50
    use_2dgs: bool = False
    resume: bool = False


def main(cfg: TrainConfig):
    _inject_vggt_static_view_config(cfg)
    stage1_data_cfg = _make_modal_stage1_data_config(cfg)
    effective_modal_warmup_epochs = (
        0
        if cfg.modal_stage1_init_ckpt is not None
        else (
            cfg.modal_stage1_epochs
            if stage1_data_cfg is not None
            else cfg.modal_warmup_epochs
        )
    )
    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    init_metadata = _make_init_metadata(cfg)
    _validate_checkpoint_policy(ckpt_path, cfg.resume, init_metadata)

    backup_code(cfg.work_dir)
    train_dataset, train_video_view, val_img_dataset, val_kpt_dataset = (
        get_train_val_datasets(cfg.data, load_val=True)
    )
    if cfg.trajectory_type == "static":
        guru.info(f"Static sweep dataset has {train_dataset.num_frames} frames")
    else:
        guru.info(f"Stage 2 dynamic dataset has {train_dataset.num_frames} frames")
    stage1_dataset = None
    if stage1_data_cfg is not None:
        stage1_dataset, _, _, _ = get_train_val_datasets(
            stage1_data_cfg, load_val=False
        )
        guru.info(
            f"Stage 1 canonical dataset has {stage1_dataset.num_frames} frames"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save config
    os.makedirs(cfg.work_dir, exist_ok=True)
    with open(f"{cfg.work_dir}/cfg.yaml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)

    initialize_and_checkpoint_model(
        cfg,
        train_dataset,
        device,
        ckpt_path,
        init_metadata,
        vis=cfg.vis_debug,
        port=cfg.port,
    )

    trainer, start_epoch = Trainer.init_from_checkpoint(
        ckpt_path,
        device,
        cfg.use_2dgs,
        cfg.lr,
        cfg.loss,
        cfg.optim,
        work_dir=cfg.work_dir,
        port=cfg.port,
        modal_warmup_epochs=effective_modal_warmup_epochs,
        modal_train_base_means=cfg.modal_train_base_means,
        modal_stage2_train_base_means=cfg.modal_stage2_train_base_means,
        modal_stage2_train_colors=cfg.modal_stage2_train_colors,
        modal_stage2_train_opacities=cfg.modal_stage2_train_opacities,
        modal_stage2_train_scales=cfg.modal_stage2_train_scales,
        modal_stage2_train_quats=cfg.modal_stage2_train_quats,
        modal_stage2_train_bg_means=cfg.modal_stage2_train_bg_means,
        modal_stage2_train_bg_colors=cfg.modal_stage2_train_bg_colors,
        modal_stage2_train_bg_opacities=cfg.modal_stage2_train_bg_opacities,
        modal_stage2_train_bg_scales=cfg.modal_stage2_train_bg_scales,
        modal_stage2_train_bg_quats=cfg.modal_stage2_train_bg_quats,
        modal_stage2_lr_fg_scales=cfg.modal_stage2_lr_fg_scales,
        modal_stage2_lr_fg_quats=cfg.modal_stage2_lr_fg_quats,
    )
    if cfg.trajectory_type == "modal_activation":
        _log_trainable_parameters(trainer.model)

    stage1_loader = None
    if stage1_dataset is not None:
        stage1_loader = DataLoader(
            stage1_dataset,
            batch_size=min(cfg.batch_size, stage1_dataset.num_frames),
            num_workers=cfg.num_dl_workers,
            persistent_workers=cfg.num_dl_workers > 0,
            collate_fn=BaseDataset.train_collate_fn,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dl_workers,
        persistent_workers=cfg.num_dl_workers > 0,
        collate_fn=BaseDataset.train_collate_fn,
    )

    validator = None
    if (
        train_video_view is not None
        or val_img_dataset is not None
        or val_kpt_dataset is not None
    ):
        validator = Validator(
            model=trainer.model,
            device=device,
            train_loader=(
                DataLoader(train_video_view, batch_size=1) if train_video_view else None
            ),
            val_img_loader=(
                DataLoader(val_img_dataset, batch_size=1) if val_img_dataset else None
            ),
            val_kpt_loader=(
                DataLoader(val_kpt_dataset, batch_size=1) if val_kpt_dataset else None
            ),
            save_dir=cfg.work_dir,
        )

    guru.info(f"Starting training from {trainer.global_step=}")
    for epoch in (
        pbar := tqdm(
            range(start_epoch, cfg.num_epochs),
            initial=start_epoch,
            total=cfg.num_epochs,
        )
    ):
        trainer.set_epoch(epoch)
        if stage1_loader is not None and epoch < cfg.modal_stage1_epochs:
            active_loader = stage1_loader
            stage_name = "stage1"
        else:
            active_loader = train_loader
            stage_name = "stage2"
        for batch in active_loader:
            batch = to_device(batch, device)
            loss = trainer.train_step(batch)
            pbar.set_description(f"{stage_name} Loss: {loss:.6f}")

        if (
            stage1_loader is not None
            and stage_name == "stage1"
            and epoch == cfg.modal_stage1_epochs - 1
        ):
            trainer.save_checkpoint(f"{cfg.work_dir}/checkpoints/stage1.ckpt")

        if validator is not None:
            if (epoch > 0 and epoch % cfg.validate_every == 0) or (
                epoch == cfg.num_epochs - 1
            ):
                val_logs = validator.validate()
                trainer.log_dict(val_logs)
            if (epoch > 0 and epoch % cfg.save_videos_every == 0) or (
                epoch == cfg.num_epochs - 1
            ):
                validator.save_train_videos(epoch)

    _save_training_completion_checkpoint(trainer, ckpt_path)


def initialize_and_checkpoint_model(
    cfg: TrainConfig,
    train_dataset: BaseDataset,
    device: torch.device,
    ckpt_path: str,
    init_metadata: dict[str, Any],
    vis: bool = False,
    port: int | None = None,
):
    if os.path.exists(ckpt_path):
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return
    init_ckpt_path = os.path.join(os.path.dirname(ckpt_path), "init.ckpt")
    if cfg.trajectory_type == "modal_activation" and os.path.exists(init_ckpt_path):
        raise ValueError(
            f"Initialization checkpoint already exists at {init_ckpt_path}; "
            "refusing to overwrite it."
        )

    if cfg.modal_stage1_init_ckpt is not None:
        if cfg.trajectory_type != "modal_activation":
            raise ValueError("--modal-stage1-init-ckpt requires modal_activation")
        fg_params, bg_params = _load_stage1_gaussians_from_checkpoint(
            cfg.modal_stage1_init_ckpt,
            device,
        )
        motion_bases = init_identity_motion_bases(
            train_dataset.num_frames,
            device,
            fg_params.params["means"].dtype,
        ).to(device)
        tracks_3d = None
        cano_t = 0
    else:
        fg_params, motion_bases, bg_params, tracks_3d, cano_t = init_model_from_tracks(
            train_dataset,
            cfg.num_fg,
            cfg.num_bg,
            cfg.num_motion_bases,
            cfg.trajectory_type,
            cfg.num_dct_bases,
            cfg.dct_init,
            cfg.modal_carrier_points,
            vis=vis,
            port=port,
        )
    # run initial optimization
    Ks = train_dataset.get_Ks().to(device)
    w2cs = train_dataset.get_w2cs().to(device)
    if cfg.trajectory_type == "som_basis":
        run_initial_optim(fg_params, motion_bases, tracks_3d, Ks, w2cs)
    if vis and cfg.port is not None:
        server = get_server(port=cfg.port)
        if cfg.trajectory_type == "som_basis":
            vis_init_params(server, fg_params, motion_bases)


    camera_poses = (
        None
        if cfg.trajectory_type in ("modal_activation", "static")
        else init_trainable_poses(w2cs)
    )
    modal = None
    modal_phi_real = None
    modal_phi_imag = None
    modal_freqs_hz = None
    modal_obs_count_per_point = None
    modal_frame_view_indices = None
    modal_frame_local_indices = None
    modal_frame_times_sec = None
    if cfg.trajectory_type == "modal_activation":
        resolve_required_modal_paths(cfg.modal_manifest, cfg.modal_frame_map)
        modal_fields = load_gaussian_modal_fields(
            cfg.modal_manifest,
            fg_params.params["means"],
        )
        modal_modes = modal_fields.modes
        modal_phi_real = modal_fields.phi_real
        modal_phi_imag = modal_fields.phi_imag
        modal_freqs_hz = modal_fields.freqs_hz
        modal_obs_count_per_point = modal_fields.obs_count_per_point
        frame_map = load_modal_frame_map(
            cfg.modal_frame_map,
            train_dataset,
            motion_bases.num_frames,
            device,
        )
        modal_frame_view_indices = frame_map.frame_view_indices
        modal_frame_local_indices = frame_map.frame_local_indices
        modal_frame_times_sec = frame_map.frame_times_sec
        modal = _zero_harmonic_activations(
            len(frame_map.view_ids),
            len(modal_modes),
            device,
            fg_params.params["means"].dtype,
        )
        view_counts = torch.bincount(
            frame_map.frame_view_indices.detach().cpu(),
            minlength=len(frame_map.view_ids),
        )
        frame_view_indices_cpu = frame_map.frame_view_indices.detach().cpu()
        frame_times_sec_cpu = frame_map.frame_times_sec.detach().cpu()
        mode_summary = ", ".join(
            f"index={mode.mode_index}, freq={mode.freq_hz:.9g}Hz"
            for mode in modal_modes
        )
        view_summaries = []
        active_views = []
        unused_views = []
        for view_index, view_id in enumerate(frame_map.view_ids):
            view_count = int(view_counts[view_index].item())
            if view_count > 0:
                view_times = frame_times_sec_cpu[
                    frame_view_indices_cpu == view_index
                ]
                active_views.append(view_id)
                view_summaries.append(
                    f"{view_id}=count:{view_count},"
                    f"time:[{float(view_times.min().item()):.9g},"
                    f"{float(view_times.max().item()):.9g}]s"
                )
            else:
                unused_views.append(view_id)
                view_summaries.append(f"{view_id}=count:0,time:<unused>")
        if cfg.modal_train_view_id is None and unused_views:
            raise ValueError(
                "Joint modal training requires frames from every declared view; "
                f"views without training frames: {unused_views}"
            )
        guru.info(
            f"Initialized modal_activation: modes={len(modal_modes)} "
            f"[{mode_summary}], per_view_frames=[{', '.join(view_summaries)}], "
            f"active_views={active_views}, unused_views={unused_views}, "
            f"activation_shape={tuple(modal.params['activations'].shape)}, "
            "parameterization=per_view_harmonic_v1"
        )

    model = SceneModel(
        Ks, 
        w2cs, 
        fg_params, 
        motion_bases, 
        camera_poses,
        bg_params,
        cfg.use_2dgs,
        trajectory_type=cfg.trajectory_type,
        cano_t=cano_t,
        num_dct_bases=cfg.num_dct_bases,
        modal=modal,
        modal_phi_real=modal_phi_real,
        modal_phi_imag=modal_phi_imag,
        modal_freqs_hz=modal_freqs_hz,
        modal_obs_count_per_point=modal_obs_count_per_point,
        modal_frame_view_indices=modal_frame_view_indices,
        modal_frame_local_indices=modal_frame_local_indices,
        modal_frame_times_sec=modal_frame_times_sec,
    )

    checkpoint = {
        "model": model.state_dict(),
        "epoch": 0,
        "global_step": 0,
        "init_metadata": init_metadata,
    }
    if cfg.trajectory_type == "modal_activation":
        guru.info(
            "Initialized SceneModel Gaussians: "
            f"fg={model.num_fg_gaussians}, bg={model.num_bg_gaussians}"
        )
        _save_new_initial_checkpoints(checkpoint, ckpt_path)
    else:
        guru.info(f"Saving initialization to {ckpt_path}")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(checkpoint, ckpt_path)


def _save_new_initial_checkpoints(
    checkpoint: dict[str, Any],
    ckpt_path: str,
) -> None:
    init_ckpt_path = os.path.join(os.path.dirname(ckpt_path), "init.ckpt")
    existing_paths = [
        path for path in (init_ckpt_path, ckpt_path) if os.path.exists(path)
    ]
    if existing_paths:
        raise ValueError(
            "Refusing to overwrite existing initialization checkpoint path(s): "
            + ", ".join(existing_paths)
        )
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(checkpoint, init_ckpt_path)
    torch.save(checkpoint, ckpt_path)
    guru.info(
        f"Saved initialization to {init_ckpt_path} and training checkpoint to "
        f"{ckpt_path}"
    )


def _save_training_completion_checkpoint(trainer: Trainer, ckpt_path: str) -> None:
    trainer.save_checkpoint(ckpt_path)


def _log_trainable_parameters(model: SceneModel) -> None:
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    guru.info(
        f"Trainable parameters ({len(trainable_names)}): "
        + (", ".join(trainable_names) if trainable_names else "<none>")
    )


def _zero_harmonic_activations(
    num_views: int,
    num_modes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ModalActivations:
    if num_views <= 0 or num_modes <= 0:
        raise ValueError("Harmonic activation initialization requires views and modes")
    return ModalActivations(
        torch.zeros(num_views, num_modes, 2, device=device, dtype=dtype)
    )


def _load_stage1_gaussians_from_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[GaussianParams, GaussianParams | None]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stage 1 init checkpoint does not exist: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Stage 1 init checkpoint has no model state: {path}")
    if "modal.params.activations" in state_dict:
        raise ValueError(
            "Stage 1 init checkpoint must be the original static checkpoint, "
            "not a Phase 0 modal activation checkpoint"
        )
    trajectory_type_id = state_dict.get("trajectory_type_id")
    if trajectory_type_id is not None and int(trajectory_type_id.item()) != (
        TRAJECTORY_TYPE_TO_ID["static"]
    ):
        raise ValueError(
            "Stage 1 init checkpoint must have trajectory_type='static'"
        )
    try:
        fg_params = GaussianParams.init_from_state_dict(
            state_dict,
            prefix="fg.params.",
        ).to(device)
    except AssertionError as exc:
        raise ValueError(
            f"Stage 1 init checkpoint cannot restore foreground Gaussians: {path}"
        ) from exc

    bg_params = None
    if any(key.startswith("bg.params.") for key in state_dict):
        try:
            bg_params = GaussianParams.init_from_state_dict(
                state_dict,
                prefix="bg.params.",
            ).to(device)
        except AssertionError as exc:
            raise ValueError(
                f"Stage 1 init checkpoint cannot restore background Gaussians: {path}"
            ) from exc

    num_bg = 0 if bg_params is None else bg_params.num_gaussians
    guru.info(
        "Loaded Stage 1 Gaussian init from "
        f"{path}: fg={fg_params.num_gaussians}, bg={num_bg}, "
        f"source_epoch={ckpt.get('epoch')}, source_step={ckpt.get('global_step')}"
    )
    return fg_params, bg_params


def _metadata_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_metadata_value(v) for v in value]
    if isinstance(value, list):
        return [_metadata_value(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def _make_init_metadata(cfg: TrainConfig) -> dict[str, Any]:
    data = cfg.data
    data_metadata = {
        "type": type(data).__name__,
        "data_dir": getattr(data, "data_dir", None),
        "start": getattr(data, "start", None),
        "end": getattr(data, "end", None),
        "res": getattr(data, "res", None),
        "image_type": getattr(data, "image_type", None),
        "mask_type": getattr(data, "mask_type", None),
        "depth_type": getattr(data, "depth_type", None),
        "camera_type": getattr(data, "camera_type", None),
        "modal_train_view_id": getattr(data, "modal_train_view_id", None),
        "modal_max_local_frames_per_view": getattr(
            data, "modal_max_local_frames_per_view", None
        ),
        "load_depths": getattr(data, "load_depths", None),
        "load_tracks": getattr(data, "load_tracks", None),
    }
    stage1_metadata = {
        "data_dir": cfg.modal_stage1_data_dir,
        "frame_map": cfg.modal_stage1_frame_map,
        "epochs": cfg.modal_stage1_epochs,
        "init_ckpt": cfg.modal_stage1_init_ckpt,
        "reuse": cfg.modal_stage1_init_ckpt is not None,
    }
    metadata = {
        "trajectory_type": cfg.trajectory_type,
        "num_fg": cfg.num_fg,
        "num_bg": cfg.num_bg,
        "num_motion_bases": cfg.num_motion_bases,
        "num_dct_bases": cfg.num_dct_bases,
        "dct_init": cfg.dct_init,
        "modal_manifest": cfg.modal_manifest,
        "modal_frame_map": cfg.modal_frame_map,
        "modal_carrier_points": cfg.modal_carrier_points,
        "modal_train_view_id": cfg.modal_train_view_id,
        "modal_max_local_frames_per_view": cfg.modal_max_local_frames_per_view,
        "vggt_view_configs": cfg.vggt_view_configs,
        "modal_warmup_epochs": cfg.modal_warmup_epochs,
        "modal_train_base_means": cfg.modal_train_base_means,
        "modal_stage1": stage1_metadata,
        "modal_stage1_init_ckpt": cfg.modal_stage1_init_ckpt,
        "modal_stage1_reuse": cfg.modal_stage1_init_ckpt is not None,
        "modal_stage2_train_base_means": (
            cfg.modal_stage2_train_base_means or cfg.modal_train_base_means
        ),
        "modal_stage2_train_colors": cfg.modal_stage2_train_colors,
        "modal_stage2_train_opacities": cfg.modal_stage2_train_opacities,
        "modal_stage2_train_scales": cfg.modal_stage2_train_scales,
        "modal_stage2_train_quats": cfg.modal_stage2_train_quats,
        "modal_stage2_train_bg_means": cfg.modal_stage2_train_bg_means,
        "modal_stage2_train_bg_colors": cfg.modal_stage2_train_bg_colors,
        "modal_stage2_train_bg_opacities": cfg.modal_stage2_train_bg_opacities,
        "modal_stage2_train_bg_scales": cfg.modal_stage2_train_bg_scales,
        "modal_stage2_train_bg_quats": cfg.modal_stage2_train_bg_quats,
        "modal_stage2_lr_fg_scales": cfg.modal_stage2_lr_fg_scales,
        "modal_stage2_lr_fg_quats": cfg.modal_stage2_lr_fg_quats,
        "w_act_mag": cfg.loss.w_act_mag,
        "data": data_metadata,
    }
    if cfg.trajectory_type == "modal_activation":
        metadata["modal_parameterization"] = "per_view_harmonic_v1"
    return {key: _metadata_value(value) for key, value in metadata.items()}


def _validate_checkpoint_policy(
    ckpt_path: str,
    resume: bool,
    expected_metadata: dict[str, Any],
):
    if not os.path.exists(ckpt_path):
        if resume:
            raise ValueError(
                f"--resume was set but checkpoint does not exist: {ckpt_path}"
            )
        init_ckpt_path = os.path.join(os.path.dirname(ckpt_path), "init.ckpt")
        if (
            expected_metadata.get("trajectory_type") == "modal_activation"
            and os.path.exists(init_ckpt_path)
        ):
            raise ValueError(
                f"Initialization checkpoint already exists at {init_ckpt_path}; "
                "use a new work_dir rather than overwriting it."
            )
        return

    if not resume:
        raise ValueError(
            f"Checkpoint already exists at {ckpt_path}. "
            "Use a new work_dir or pass --resume."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actual_metadata = ckpt.get("init_metadata")
    if actual_metadata is None:
        raise ValueError(
            f"Checkpoint {ckpt_path} has no init_metadata; "
            "use a new work_dir for this run."
        )
    expected_parameterization = expected_metadata.get("modal_parameterization")
    actual_parameterization = actual_metadata.get("modal_parameterization")
    if (
        expected_parameterization == "per_view_harmonic_v1"
        and actual_parameterization != expected_parameterization
    ):
        raise ValueError(
            "Checkpoint uses an incompatible modal parameterization "
            f"({actual_parameterization!r}); expected "
            "'per_view_harmonic_v1'. Start a new work_dir from the static "
            "checkpoint and staged modal manifest."
        )
    if actual_metadata != expected_metadata:
        keys = sorted(set(actual_metadata) | set(expected_metadata))
        mismatched = [
            key
            for key in keys
            if actual_metadata.get(key) != expected_metadata.get(key)
        ]
        preview = ", ".join(mismatched[:8])
        raise ValueError(
            "Checkpoint init_metadata does not match current config. "
            f"Mismatched keys: {preview}. Use a new work_dir for this run."
        )


def init_model_from_tracks(
    train_dataset,
    num_fg: int,
    num_bg: int,
    num_motion_bases: int,
    trajectory_type: Literal["som_basis", "dct_center", "modal_activation", "static"],
    num_dct_bases: int | None,
    dct_init: Literal["tracks", "zero"],
    modal_carrier_points: str | None,
    vis: bool = False,
    port: int | None = None,
):
    if trajectory_type == "static":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fg_params, bg_params = init_static_gaussians_from_colmap(
            train_dataset,
            num_fg,
            num_bg,
            device,
        )
        motion_bases = init_identity_motion_bases(
            train_dataset.num_frames,
            device,
            fg_params.params["means"].dtype,
        ).to(device)
        return fg_params, motion_bases, bg_params, None, 0

    if trajectory_type == "modal_activation":
        if modal_carrier_points is None:
            raise ValueError("trajectory_type='modal_activation' requires modal_carrier_points")
        carrier = np.load(modal_carrier_points, allow_pickle=False)
        required = {"points_world", "colors"}
        missing = sorted(required - set(carrier.files))
        if missing:
            raise ValueError(f"{modal_carrier_points} missing required fields: {missing}")
        points = carrier["points_world"].astype(np.float32)
        colors = carrier["colors"].astype(np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_world must have shape (N,3), got {points.shape}")
        if colors.shape != points.shape:
            raise ValueError(f"colors must have shape {points.shape}, got {colors.shape}")
        if points.shape[0] > num_fg:
            sel = np.random.choice(points.shape[0], num_fg, replace=False)
            points = points[sel]
            colors = colors[sel]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points_th = torch.from_numpy(points).to(device)
        colors_th = torch.from_numpy(colors).to(device)
        fg_params = init_fg_from_point_cloud(points_th, colors_th).to(device)
        motion_bases = init_identity_motion_bases(
            train_dataset.num_frames, device, points_th.dtype
        ).to(device)
        bg_params = None
        if num_bg > 0:
            bg_points = StaticObservations(*train_dataset.get_bkgd_points(num_bg))
            assert bg_points.check_sizes()
            bg_params = init_bg(bg_points)
            bg_params = bg_params.to(device)
        tracks_3d = None
        cano_t = 0
        return fg_params, motion_bases, bg_params, tracks_3d, cano_t

    tracks_3d = TrackObservations(*train_dataset.get_tracks_3d(num_fg))
    print(
        f"{tracks_3d.xyz.shape=} {tracks_3d.visibles.shape=} "
        f"{tracks_3d.invisibles.shape=} {tracks_3d.confidences.shape} "
        f"{tracks_3d.colors.shape}"
    )
    if not tracks_3d.check_sizes():
        import ipdb

        ipdb.set_trace()

    rot_type = "6d"
    cano_t = int(tracks_3d.visibles.sum(dim=0).argmax().item())

    guru.info(f"{cano_t=} {num_fg=} {num_bg=} {num_motion_bases=}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if trajectory_type == "som_basis":
        motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
            tracks_3d, num_motion_bases, rot_type, cano_t, vis=vis, port=port
        )
        traj_coefs = None
    elif trajectory_type == "dct_center":
        motion_bases, traj_coefs, tracks_3d = init_motion_params_with_dct(
            tracks_3d, num_dct_bases, cano_t, dct_init=dct_init
        )
        motion_coefs = None
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    motion_bases = motion_bases.to(device)

    fg_params = init_fg_from_tracks_3d(
        cano_t, tracks_3d, motion_coefs, traj_coefs=traj_coefs
    )
    fg_params = fg_params.to(device)

    bg_params = None
    if num_bg > 0:
        bg_points = StaticObservations(*train_dataset.get_bkgd_points(num_bg))
        assert bg_points.check_sizes()
        bg_params = init_bg(bg_points)
        bg_params = bg_params.to(device)

    tracks_3d = tracks_3d.to(device)
    return fg_params, motion_bases, bg_params, tracks_3d, cano_t


def init_static_gaussians_from_colmap(
    train_dataset,
    num_fg: int,
    num_bg: int,
    device: torch.device,
) -> tuple[GaussianParams, GaussianParams | None]:
    if getattr(train_dataset, "camera_type", None) != "colmap":
        raise ValueError("trajectory_type='static' currently requires data.camera_type='colmap'")
    data_dir = getattr(train_dataset, "data_dir", None)
    if data_dir is None:
        raise ValueError("COLMAP static initialization requires dataset.data_dir")
    colmap_dir = Path(data_dir) / "colmap" / "sparse" / "0"
    points_path = colmap_dir / "points3D.bin"
    if not points_path.exists():
        raise FileNotFoundError(points_path)

    points3d = read_points3d_binary(points_path)
    if not points3d:
        raise ValueError(f"No COLMAP sparse points found in {points_path}")
    points = np.stack([p.xyz for p in points3d.values()]).astype(np.float32)
    colors = np.stack([p.rgb for p in points3d.values()]).astype(np.float32) / 255.0
    points = _transform_colmap_points_to_dataset_world(train_dataset, points)
    fg_mask, bg_mask = _classify_static_colmap_points(train_dataset, points)

    fg_points, fg_colors = _sample_colmap_points(
        points[fg_mask],
        colors[fg_mask],
        num_fg,
        "foreground",
    )
    fg_params = init_fg_from_point_cloud(
        torch.from_numpy(fg_points).to(device),
        torch.from_numpy(fg_colors).to(device),
    ).to(device)

    bg_params = None
    if num_bg > 0:
        bg_points, bg_colors = _sample_colmap_points(
            points[bg_mask],
            colors[bg_mask],
            num_bg,
            "background",
        )
        bg_params = init_fg_from_point_cloud(
            torch.from_numpy(bg_points).to(device),
            torch.from_numpy(bg_colors).to(device),
        ).to(device)

    guru.info(
        "Initialized static COLMAP Gaussians: "
        f"fg={fg_params.num_gaussians}, "
        f"bg={0 if bg_params is None else bg_params.num_gaussians}"
    )
    return fg_params, bg_params


def _transform_colmap_points_to_dataset_world(train_dataset, points: np.ndarray) -> np.ndarray:
    scene_norm = getattr(train_dataset, "scene_norm_dict", None)
    if scene_norm is None:
        raise ValueError("COLMAP static initialization requires dataset.scene_norm_dict")
    transform = scene_norm["transfm"].detach().cpu().numpy().astype(np.float32)
    scale = float(scene_norm["scale"])
    if scale <= 0:
        raise ValueError(f"scene_norm scale must be positive, got {scale}")
    points_h = np.concatenate(
        [points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    return (points_h @ transform.T)[:, :3] / scale


def _classify_static_colmap_points(train_dataset, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    mask_dir = getattr(train_dataset, "mask_dir", None)
    if mask_dir is None or not os.path.isdir(mask_dir):
        guru.warning("No mask directory found for static COLMAP init; using all points as foreground")
        return np.ones(num_points, dtype=bool), np.zeros(num_points, dtype=bool)

    Ks = train_dataset.get_Ks().detach().cpu().numpy().astype(np.float32)
    w2cs = train_dataset.get_w2cs().detach().cpu().numpy().astype(np.float32)
    fg_counts = np.zeros(num_points, dtype=np.int32)
    bg_counts = np.zeros(num_points, dtype=np.int32)
    chunk = 16384
    for view_idx in range(train_dataset.num_frames):
        mask = train_dataset.get_mask(view_idx).detach().cpu().numpy()
        height, width = mask.shape
        K = Ks[view_idx]
        w2c = w2cs[view_idx]
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        for start in range(0, num_points, chunk):
            end = min(start + chunk, num_points)
            cam = points[start:end] @ R.T + t[None]
            z = cam[:, 2]
            pix_h = cam @ K.T
            valid_z = z > 1e-6
            u = pix_h[:, 0] / np.clip(pix_h[:, 2], 1e-6, None)
            v = pix_h[:, 1] / np.clip(pix_h[:, 2], 1e-6, None)
            ui = np.rint(u).astype(np.int64)
            vi = np.rint(v).astype(np.int64)
            valid = (
                valid_z
                & (ui >= 0)
                & (ui < width)
                & (vi >= 0)
                & (vi < height)
            )
            if not valid.any():
                continue
            vals = mask[vi[valid], ui[valid]]
            local_indices = np.nonzero(valid)[0] + start
            fg_counts[local_indices[vals > 0]] += 1
            bg_counts[local_indices[vals < 0]] += 1

    observed = (fg_counts + bg_counts) > 0
    fg_mask = (fg_counts >= bg_counts) & observed
    fg_mask |= ~observed
    bg_mask = (bg_counts > fg_counts) & observed
    guru.info(
        "COLMAP point mask classification: "
        f"fg={int(fg_mask.sum())}, bg={int(bg_mask.sum())}, "
        f"unobserved={int((~observed).sum())}"
    )
    return fg_mask, bg_mask


def _sample_colmap_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_count: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    if max_count <= 0:
        raise ValueError(f"{label} Gaussian count must be positive")
    if points.shape[0] < 2:
        raise ValueError(f"Need at least 2 {label} COLMAP points, got {points.shape[0]}")
    if points.shape[0] > max_count:
        sel = np.random.choice(points.shape[0], max_count, replace=False)
        points = points[sel]
        colors = colors[sel]
    return points.astype(np.float32), colors.astype(np.float32)


def _make_modal_stage1_data_config(
    cfg: TrainConfig,
) -> DavisDataConfig | CustomDataConfig | None:
    has_stage1_init = cfg.modal_stage1_init_ckpt is not None
    has_stage1_data = (
        cfg.modal_stage1_data_dir is not None
        or cfg.modal_stage1_frame_map is not None
    )
    if cfg.trajectory_type != "modal_activation":
        if has_stage1_init or has_stage1_data or cfg.modal_stage1_epochs != 0:
            raise ValueError("modal stage1 options require modal_activation")
        return None
    if has_stage1_init:
        if has_stage1_data:
            raise ValueError(
                "--modal-stage1-init-ckpt cannot be combined with "
                "--modal-stage1-data-dir or --modal-stage1-frame-map"
            )
        if not os.path.exists(cfg.modal_stage1_init_ckpt):
            raise FileNotFoundError(
                f"Stage 1 init checkpoint does not exist: {cfg.modal_stage1_init_ckpt}"
            )
        guru.info(
            "--modal-stage1-init-ckpt is set; skipping Stage 1 dataset and epochs"
        )
        return None
    if not has_stage1_data:
        if cfg.modal_stage1_epochs != 0:
            raise ValueError(
                "--modal-stage1-epochs requires --modal-stage1-data-dir and "
                "--modal-stage1-frame-map"
            )
        return None
    if cfg.modal_stage1_data_dir is None or cfg.modal_stage1_frame_map is None:
        raise ValueError(
            "Stage 1 requires both --modal-stage1-data-dir and "
            "--modal-stage1-frame-map"
        )
    if cfg.modal_stage1_epochs <= 0:
        raise ValueError("Stage 1 requires --modal-stage1-epochs > 0")
    if not isinstance(cfg.data, (CustomDataConfig, DavisDataConfig)):
        raise ValueError("modal stage1 requires custom or davis data")
    if not cfg.vggt_view_configs:
        raise ValueError("modal stage1 requires --vggt-view-configs")
    if (
        cfg.loss.w_depth_reg != 0.0
        or cfg.loss.w_depth_grad != 0.0
        or cfg.loss.w_depth_const != 0.0
    ):
        raise ValueError(
            "Stage 1 currently disables depth loading, so set "
            "--loss.w-depth-reg 0 --loss.w-depth-grad 0 --loss.w-depth-const 0"
        )

    return replace(
        cfg.data,
        data_dir=cfg.modal_stage1_data_dir,
        start=0,
        end=-1,
        camera_type="vggt",
        vggt_view_configs=cfg.vggt_view_configs,
        modal_frame_map=cfg.modal_stage1_frame_map,
        modal_train_view_id=None,
        modal_max_local_frames_per_view=None,
        load_depths=False,
    )


def _inject_vggt_static_view_config(cfg: TrainConfig):
    depth_losses_enabled = any(
        weight > 0.0
        for weight in (
            cfg.loss.w_depth_reg,
            cfg.loss.w_depth_grad,
            cfg.loss.w_depth_const,
        )
    )
    if cfg.trajectory_type == "static":
        if cfg.modal_manifest is not None or cfg.modal_frame_map is not None:
            raise ValueError("static trajectory does not use modal training inputs")
        if not isinstance(cfg.data, (CustomDataConfig, DavisDataConfig)):
            raise ValueError("static trajectory requires custom or davis data")
        if cfg.data.camera_type != "colmap":
            raise ValueError("static trajectory requires data.camera_type='colmap'")
        if not cfg.data.load_from_cache:
            raise ValueError(
                "static COLMAP sweep training requires --data.load-from-cache "
                "with an aligned scene_norm_dict.pth"
            )
        cache_scene_norm = os.path.join(
            cfg.data.data_dir,
            "flow3d_preprocessed",
            cfg.data.res,
            "scene_norm_dict.pth",
        )
        if not os.path.exists(cache_scene_norm):
            raise FileNotFoundError(cache_scene_norm)
        depth_weight = (
            cfg.loss.w_depth_reg
            + cfg.loss.w_depth_grad
            + cfg.loss.w_depth_const
        )
        cfg.data = replace(
            cfg.data,
            load_tracks=False,
            load_depths=depth_weight > 0,
        )
    if (
        cfg.modal_train_view_id is not None
        or cfg.modal_max_local_frames_per_view is not None
    ):
        if cfg.trajectory_type != "modal_activation":
            raise ValueError("modal frame filtering requires modal_activation")
        if not isinstance(cfg.data, (CustomDataConfig, DavisDataConfig)):
            raise ValueError("modal frame filtering requires custom or davis data")
        if cfg.data.camera_type != "vggt":
            raise ValueError("modal frame filtering requires data.camera_type='vggt'")
        if (
            cfg.modal_max_local_frames_per_view is not None
            and cfg.modal_max_local_frames_per_view <= 0
        ):
            raise ValueError("--modal-max-local-frames-per-view must be positive")
    if cfg.trajectory_type != "modal_activation":
        return
    if not isinstance(cfg.data, (CustomDataConfig, DavisDataConfig)):
        return
    if cfg.data.camera_type != "vggt":
        return
    if not cfg.vggt_view_configs:
        raise ValueError("data.camera_type='vggt' requires --vggt-view-configs")
    if cfg.modal_frame_map is None:
        raise ValueError("data.camera_type='vggt' requires --modal-frame-map")
    cfg.data = replace(
        cfg.data,
        vggt_view_configs=cfg.vggt_view_configs,
        modal_frame_map=cfg.modal_frame_map,
        modal_train_view_id=cfg.modal_train_view_id,
        modal_max_local_frames_per_view=cfg.modal_max_local_frames_per_view,
        load_tracks=False,
        load_depths=depth_losses_enabled,
    )


def backup_code(work_dir):
    root_dir = osp.abspath(osp.join(osp.dirname(__file__)))
    tracked_dirs = [osp.join(root_dir, dirname) for dirname in ["flow3d", "scripts"]]
    dst_dir = osp.join(work_dir, "code", datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    for tracked_dir in tracked_dirs:
        if osp.exists(tracked_dir):
            shutil.copytree(tracked_dir, osp.join(dst_dir, osp.basename(tracked_dir)))


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
