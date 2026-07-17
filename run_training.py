import json
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
    GaussianModalRefinementData,
    ModalEnvelopeLayout,
    build_modal_envelope_layout,
    load_gaussian_modal_refinement_data,
    load_gaussian_modal_fields,
    load_modal_frame_map,
    resolve_required_modal_paths,
)
from flow3d.params import (
    CameraScales,
    GaussianParams,
    ModalHarmonicEnvelope,
    ModalShapeRefinement,
)
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
    modal_warmup_epochs: int = 0
    modal_train_base_means: bool = False
    modal_stage1_data_dir: str | None = None
    modal_stage1_frame_map: str | None = None
    modal_stage1_epochs: int = 0
    modal_stage1_init_ckpt: str | None = None
    modal_shape_refinement: Literal["fixed", "anchor_delta"] = "fixed"
    modal_envelope_init_ckpt: str | None = None
    modal_envelope_knot_interval_sec: float = 0.5
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
    _validate_modal_shape_refinement_config(cfg)
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
    elif cfg.modal_shape_refinement == "anchor_delta":
        guru.info(f"Stage 3A dynamic dataset has {train_dataset.num_frames} frames")
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

    initial_refinement_data = initialize_and_checkpoint_model(
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
    if cfg.modal_shape_refinement == "anchor_delta":
        assert cfg.modal_manifest is not None
        assert cfg.modal_frame_map is not None
        frame_map = load_modal_frame_map(
            cfg.modal_frame_map,
            train_dataset,
            trainer.model.num_frames,
            device,
        )
        refinement_data = initial_refinement_data
        if refinement_data is None:
            refinement_data = load_gaussian_modal_refinement_data(
                cfg.modal_manifest,
                trainer.model.fg.params["means"],
                frame_map.view_ids,
                device,
                trainer.model.fg.params["means"].dtype,
            )
        trainer.set_modal_refinement_data(refinement_data)
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
            stage_name = (
                "stage3a"
                if cfg.modal_shape_refinement == "anchor_delta"
                else "stage2"
            )
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
) -> GaussianModalRefinementData | None:
    if os.path.exists(ckpt_path):
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return None
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
    modal_envelope_knot_offsets = None
    modal_envelope_knot_times_sec = None
    modal_frame_envelope_left = None
    modal_frame_envelope_right = None
    modal_frame_envelope_lerp = None
    modal_refinement = None
    modal_anchor_mask = None
    if cfg.trajectory_type == "modal_activation":
        resolve_required_modal_paths(cfg.modal_manifest, cfg.modal_frame_map)
        assert cfg.modal_manifest is not None
        assert cfg.modal_frame_map is not None
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
        envelope_layout = build_modal_envelope_layout(
            frame_map,
            cfg.modal_envelope_knot_interval_sec,
        )
        modal_envelope_knot_offsets = envelope_layout.knot_offsets
        modal_envelope_knot_times_sec = envelope_layout.knot_times_sec
        modal_frame_envelope_left = envelope_layout.frame_left_indices
        modal_frame_envelope_right = envelope_layout.frame_right_indices
        modal_frame_envelope_lerp = envelope_layout.frame_lerp_weights
        if cfg.modal_shape_refinement == "anchor_delta":
            refinement_data = load_gaussian_modal_refinement_data(
                cfg.modal_manifest,
                fg_params.params["means"],
                frame_map.view_ids,
                device,
                fg_params.params["means"].dtype,
            )
            modal_anchor_mask = refinement_data.anchor_mask
            modal_refinement = ModalShapeRefinement(
                torch.zeros(
                    (*modal_phi_real.shape, 2),
                    device=device,
                    dtype=modal_phi_real.dtype,
                )
            )
            anchor_counts = refinement_data.anchor_mask.sum(dim=1).tolist()
            guru.info(
                "Initialized anchor-only modal shape refinement: "
                f"anchor_counts={anchor_counts}, "
                f"delta_shape={tuple(modal_refinement.params['delta_phi'].shape)}"
            )
            modal = _load_harmonic_envelope_from_checkpoint(
                cfg.modal_envelope_init_ckpt,
                frame_map.view_ids,
                envelope_layout,
                cfg.modal_envelope_knot_interval_sec,
                modal_phi_real,
                modal_phi_imag,
                modal_freqs_hz,
                modal_obs_count_per_point,
                fg_params,
                bg_params,
                device,
            )
        else:
            modal = _zero_harmonic_envelope(
                int(envelope_layout.knot_times_sec.shape[0]),
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
            f"envelope_shape={tuple(modal.params['envelope_knots'].shape)}, "
            f"knot_counts={torch.diff(envelope_layout.knot_offsets).tolist()}, "
            "parameterization=per_view_harmonic_envelope_v1, "
            f"shape_refinement={cfg.modal_shape_refinement}"
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
        modal_envelope_knot_offsets=modal_envelope_knot_offsets,
        modal_envelope_knot_times_sec=modal_envelope_knot_times_sec,
        modal_envelope_knot_interval_sec=(
            cfg.modal_envelope_knot_interval_sec
            if cfg.trajectory_type == "modal_activation"
            else None
        ),
        modal_frame_envelope_left=modal_frame_envelope_left,
        modal_frame_envelope_right=modal_frame_envelope_right,
        modal_frame_envelope_lerp=modal_frame_envelope_lerp,
        modal_refinement=modal_refinement,
        modal_anchor_mask=modal_anchor_mask,
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
    return refinement_data if cfg.modal_shape_refinement == "anchor_delta" else None


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
    trainer.require_modal_delta_gradient_observed()
    trainer.save_checkpoint(ckpt_path)


def _log_trainable_parameters(model: SceneModel) -> None:
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    guru.info(
        f"Trainable parameters ({len(trainable_names)}): "
        + (", ".join(trainable_names) if trainable_names else "<none>")
    )


def _zero_harmonic_envelope(
    num_total_knots: int,
    num_modes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ModalHarmonicEnvelope:
    if num_total_knots <= 0 or num_modes <= 0:
        raise ValueError("Harmonic envelope initialization requires knots and modes")
    return ModalHarmonicEnvelope(
        torch.zeros(num_total_knots, num_modes, 2, device=device, dtype=dtype)
    )


def _require_equal_checkpoint_tensor(
    state_dict: dict[str, torch.Tensor],
    key: str,
    expected: torch.Tensor,
    checkpoint_path: str,
) -> None:
    if key not in state_dict:
        raise ValueError(f"Envelope source checkpoint is missing {key}: {checkpoint_path}")
    actual = state_dict[key].detach().cpu()
    expected_cpu = expected.detach().cpu()
    if actual.shape != expected_cpu.shape or actual.dtype != expected_cpu.dtype:
        raise ValueError(
            f"Envelope source {key} shape/dtype does not match the current "
            f"initialization: {tuple(actual.shape)}/{actual.dtype} versus "
            f"{tuple(expected_cpu.shape)}/{expected_cpu.dtype}"
        )
    if not torch.equal(actual, expected_cpu):
        raise ValueError(
            f"Envelope source {key} differs from the current static checkpoint "
            "or staged manifest"
        )


def _validate_envelope_source_gaussians(
    state_dict: dict[str, torch.Tensor],
    part_name: str,
    params: GaussianParams | None,
    checkpoint_path: str,
) -> None:
    prefix = f"{part_name}.params."
    actual_keys = {key for key in state_dict if key.startswith(prefix)}
    expected_keys = (
        set()
        if params is None
        else {f"{prefix}{name}" for name in params.params.keys()}
    )
    if actual_keys != expected_keys:
        raise ValueError(
            f"Envelope source {part_name} Gaussian fields do not match the "
            f"current static checkpoint: {checkpoint_path}"
        )
    if params is None:
        return
    for name, value in params.params.items():
        _require_equal_checkpoint_tensor(
            state_dict,
            f"{prefix}{name}",
            value,
            checkpoint_path,
        )


def _load_harmonic_envelope_from_checkpoint(
    path: str | None,
    view_ids: list[str],
    envelope_layout: ModalEnvelopeLayout,
    knot_interval_sec: float,
    modal_phi_real: torch.Tensor,
    modal_phi_imag: torch.Tensor,
    modal_freqs_hz: torch.Tensor,
    modal_obs_count_per_point: torch.Tensor,
    fg_params: GaussianParams,
    bg_params: GaussianParams | None,
    device: torch.device,
) -> ModalHarmonicEnvelope:
    if path is None:
        raise ValueError("anchor_delta refinement requires --modal-envelope-init-ckpt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Envelope source checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Envelope source checkpoint has no model state: {path}")
    metadata = checkpoint.get("init_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Envelope source checkpoint has no init_metadata: {path}")
    if (
        metadata.get("modal_parameterization")
        != "per_view_harmonic_envelope_v1"
    ):
        raise ValueError(
            "Envelope source checkpoint must use "
            "per_view_harmonic_envelope_v1"
        )
    if metadata.get("modal_envelope_interpolation") != "linear_complex":
        raise ValueError(
            "Envelope source checkpoint must use linear_complex interpolation"
        )
    source_interval = metadata.get("modal_envelope_knot_interval_sec")
    if (
        isinstance(source_interval, bool)
        or not isinstance(source_interval, (int, float))
        or not np.isfinite(float(source_interval))
        or not np.isclose(
            float(source_interval),
            float(knot_interval_sec),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
    ):
        raise ValueError(
            "Envelope source knot interval does not match the current run"
        )
    if metadata.get("modal_shape_parameterization") is not None:
        raise ValueError(
            "Envelope source checkpoint must be a fixed-shape Stage 2B checkpoint"
        )
    refinement_keys = {
        "modal_refinement.params.delta_phi",
        "modal_anchor_mask",
    }
    present_refinement_keys = sorted(refinement_keys & set(state_dict))
    if present_refinement_keys:
        raise ValueError(
            "Envelope source checkpoint already contains shape refinement: "
            f"{present_refinement_keys}"
        )

    source_frame_map = metadata.get("modal_frame_map")
    if not isinstance(source_frame_map, str) or not source_frame_map:
        raise ValueError("Envelope source metadata is missing modal_frame_map")
    source_frame_map_path = Path(source_frame_map).expanduser()
    if not source_frame_map_path.exists():
        raise FileNotFoundError(
            f"Envelope source frame map does not exist: {source_frame_map_path}"
        )
    with source_frame_map_path.open("r", encoding="utf-8") as f:
        source_frame_payload = json.load(f)
    if not isinstance(source_frame_payload, dict) or source_frame_payload.get(
        "version"
    ) != 1:
        raise ValueError(
            f"Envelope source frame map must use version 1: {source_frame_map_path}"
        )
    source_view_ids = source_frame_payload.get("views")
    if source_view_ids != view_ids:
        raise ValueError(
            "Envelope source view order does not match the current frame map: "
            f"{source_view_ids!r} versus {view_ids!r}"
        )

    _validate_envelope_source_gaussians(state_dict, "fg", fg_params, path)
    _validate_envelope_source_gaussians(state_dict, "bg", bg_params, path)
    for key, expected in (
        ("modal_phi_real", modal_phi_real),
        ("modal_phi_imag", modal_phi_imag),
        ("modal_freqs_hz", modal_freqs_hz),
        ("modal_obs_count_per_point", modal_obs_count_per_point),
        ("modal_envelope_knot_offsets", envelope_layout.knot_offsets),
        ("modal_envelope_knot_times_sec", envelope_layout.knot_times_sec),
        ("modal_frame_envelope_left", envelope_layout.frame_left_indices),
        ("modal_frame_envelope_right", envelope_layout.frame_right_indices),
        ("modal_frame_envelope_lerp", envelope_layout.frame_lerp_weights),
    ):
        _require_equal_checkpoint_tensor(state_dict, key, expected, path)

    interval_key = "modal_envelope_knot_interval_sec"
    if interval_key not in state_dict or not np.isclose(
        float(state_dict[interval_key].item()),
        float(knot_interval_sec),
        rtol=1.0e-6,
        atol=1.0e-8,
    ):
        raise ValueError("Envelope source checkpoint has a mismatched knot interval")

    envelope_key = "modal.params.envelope_knots"
    if envelope_key not in state_dict:
        raise ValueError(f"Envelope source checkpoint is missing {envelope_key}")
    envelope_knots = state_dict[envelope_key]
    expected_shape = (
        int(envelope_layout.knot_times_sec.shape[0]),
        modal_phi_real.shape[0],
        2,
    )
    if envelope_knots.shape != expected_shape:
        raise ValueError(
            f"Envelope source knots must have shape {expected_shape}, "
            f"got {tuple(envelope_knots.shape)}"
        )
    if not torch.is_floating_point(envelope_knots) or not bool(
        torch.isfinite(envelope_knots).all().item()
    ):
        raise ValueError("Envelope source knots must be finite floating point")
    if envelope_knots.dtype != modal_phi_real.dtype:
        raise ValueError(
            "Envelope source knot dtype must match staged modal phi: "
            f"{envelope_knots.dtype} versus {modal_phi_real.dtype}"
        )
    mode_has_envelope = (
        torch.linalg.vector_norm(envelope_knots, dim=-1).gt(0).any(dim=0)
    )
    if not bool(mode_has_envelope.all().item()):
        missing_modes = torch.nonzero(~mode_has_envelope).flatten().tolist()
        raise ValueError(
            "Each mode requires at least one nonzero source envelope knot; zero modes: "
            f"{missing_modes}"
        )
    guru.info(
        f"Loaded fixed harmonic envelope from {path}: "
        f"shape={tuple(envelope_knots.shape)}"
    )
    return ModalHarmonicEnvelope(
        envelope_knots.to(
            device=device,
            dtype=modal_phi_real.dtype,
        ).clone()
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
    if (
        "modal.params.activations" in state_dict
        or "modal.params.envelope_knots" in state_dict
    ):
        raise ValueError(
            "Stage 1 init checkpoint must be the original static checkpoint, "
            "not a modal trajectory checkpoint"
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
        "w_envelope_smooth": cfg.loss.w_envelope_smooth,
        "data": data_metadata,
    }
    if cfg.trajectory_type == "modal_activation":
        metadata.update(
            {
                "modal_parameterization": "per_view_harmonic_envelope_v1",
                "modal_envelope_knot_interval_sec": (
                    cfg.modal_envelope_knot_interval_sec
                ),
                "modal_envelope_interpolation": "linear_complex",
                "lr_modal_envelope_knots": cfg.lr.modal.envelope_knots,
            }
        )
    if cfg.modal_shape_refinement == "anchor_delta":
        metadata.update(
            {
                "modal_shape_parameterization": "anchor_delta_phi_v1",
                "modal_envelope_init_ckpt": cfg.modal_envelope_init_ckpt,
                "modal_envelope_frozen": True,
                "w_modal_2d": cfg.loss.w_modal_2d,
                "w_delta_phi_prior": cfg.loss.w_delta_phi_prior,
                "w_delta_phi_spatial": cfg.loss.w_delta_phi_spatial,
                "lr_modal_refinement_delta_phi": (
                    cfg.lr.modal_refinement.delta_phi
                ),
            }
        )
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
        expected_parameterization == "per_view_harmonic_envelope_v1"
        and actual_parameterization != expected_parameterization
    ):
        raise ValueError(
            "Checkpoint uses an incompatible modal parameterization "
            f"({actual_parameterization!r}); expected "
            "'per_view_harmonic_envelope_v1'. Start a new work_dir from the static "
            "checkpoint and staged modal manifest."
        )
    expected_shape_parameterization = expected_metadata.get(
        "modal_shape_parameterization"
    )
    actual_shape_parameterization = actual_metadata.get(
        "modal_shape_parameterization"
    )
    if actual_shape_parameterization != expected_shape_parameterization:
        raise ValueError(
            "Checkpoint uses an incompatible modal shape parameterization "
            f"({actual_shape_parameterization!r}); expected "
            f"{expected_shape_parameterization!r}. Use a new work_dir."
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


def _validate_modal_shape_refinement_config(cfg: TrainConfig) -> None:
    if cfg.trajectory_type == "modal_activation":
        if cfg.modal_warmup_epochs != 0:
            raise ValueError(
                "Harmonic-envelope training requires --modal-warmup-epochs 0"
            )
        if (
            not np.isfinite(cfg.modal_envelope_knot_interval_sec)
            or cfg.modal_envelope_knot_interval_sec <= 0.0
        ):
            raise ValueError(
                "--modal-envelope-knot-interval-sec must be finite and positive"
            )
        if (
            not np.isfinite(cfg.lr.modal.envelope_knots)
            or cfg.lr.modal.envelope_knots <= 0.0
        ):
            raise ValueError(
                "--lr.modal.envelope-knots must be finite and positive"
            )
        envelope_loss_weights = {
            "w_act_mag": cfg.loss.w_act_mag,
            "w_envelope_smooth": cfg.loss.w_envelope_smooth,
        }
        invalid_envelope_weights = [
            name
            for name, weight in envelope_loss_weights.items()
            if not np.isfinite(weight) or weight < 0.0
        ]
        if invalid_envelope_weights:
            raise ValueError(
                "Modal envelope loss weights must be finite and non-negative: "
                f"{invalid_envelope_weights}"
            )
        gaussian_training_options = {
            "modal_train_base_means": cfg.modal_train_base_means,
            "modal_stage2_train_base_means": cfg.modal_stage2_train_base_means,
            "modal_stage2_train_colors": cfg.modal_stage2_train_colors,
            "modal_stage2_train_opacities": cfg.modal_stage2_train_opacities,
            "modal_stage2_train_scales": cfg.modal_stage2_train_scales,
            "modal_stage2_train_quats": cfg.modal_stage2_train_quats,
            "modal_stage2_train_bg_means": cfg.modal_stage2_train_bg_means,
            "modal_stage2_train_bg_colors": cfg.modal_stage2_train_bg_colors,
            "modal_stage2_train_bg_opacities": cfg.modal_stage2_train_bg_opacities,
            "modal_stage2_train_bg_scales": cfg.modal_stage2_train_bg_scales,
            "modal_stage2_train_bg_quats": cfg.modal_stage2_train_bg_quats,
        }
        enabled_gaussian_options = [
            name for name, enabled in gaussian_training_options.items() if enabled
        ]
        if enabled_gaussian_options:
            raise ValueError(
                "Harmonic-envelope training freezes every Gaussian parameter; "
                f"enabled options: {enabled_gaussian_options}"
            )
        if (
            cfg.modal_stage2_lr_fg_scales is not None
            or cfg.modal_stage2_lr_fg_quats is not None
        ):
            raise ValueError(
                "Harmonic-envelope training does not accept Gaussian LR overrides"
            )
    is_anchor_delta = cfg.modal_shape_refinement == "anchor_delta"
    refinement_weights = (
        cfg.loss.w_modal_2d,
        cfg.loss.w_delta_phi_prior,
        cfg.loss.w_delta_phi_spatial,
    )
    if not is_anchor_delta:
        if cfg.modal_envelope_init_ckpt is not None:
            raise ValueError(
                "--modal-envelope-init-ckpt requires "
                "--modal-shape-refinement anchor_delta"
            )
        if any(weight != 0.0 for weight in refinement_weights):
            raise ValueError(
                "Stage 3A loss weights require "
                "--modal-shape-refinement anchor_delta"
            )
        return

    if cfg.trajectory_type != "modal_activation":
        raise ValueError("anchor_delta refinement requires modal_activation")
    if cfg.modal_stage1_init_ckpt is None:
        raise ValueError(
            "anchor_delta refinement requires --modal-stage1-init-ckpt"
        )
    if cfg.modal_envelope_init_ckpt is None:
        raise ValueError(
            "anchor_delta refinement requires --modal-envelope-init-ckpt"
        )
    if not os.path.exists(cfg.modal_envelope_init_ckpt):
        raise FileNotFoundError(
            "Envelope source checkpoint does not exist: "
            f"{cfg.modal_envelope_init_ckpt}"
        )
    if cfg.modal_manifest is None or cfg.modal_frame_map is None:
        raise ValueError(
            "anchor_delta refinement requires --modal-manifest and "
            "--modal-frame-map"
        )
    if cfg.modal_train_view_id is not None:
        raise ValueError("anchor_delta refinement requires joint all-view training")
    if cfg.port is not None or cfg.vis_debug:
        raise ValueError(
            "Stage 3A Viser integration is deferred; run without --port and "
            "--vis-debug, then use run_modal_reconstruction.py"
        )

    unrelated_loss_weights = {
        "w_depth_reg": cfg.loss.w_depth_reg,
        "w_depth_const": cfg.loss.w_depth_const,
        "w_depth_grad": cfg.loss.w_depth_grad,
        "w_track": cfg.loss.w_track,
        "w_smooth_bases": cfg.loss.w_smooth_bases,
        "w_smooth_tracks": cfg.loss.w_smooth_tracks,
        "w_scale_var": cfg.loss.w_scale_var,
        "w_z_accel": cfg.loss.w_z_accel,
        "w_dct_coef": cfg.loss.w_dct_coef,
        "w_act_mag": cfg.loss.w_act_mag,
        "w_envelope_smooth": cfg.loss.w_envelope_smooth,
        "w_local_iso_ray": cfg.loss.w_local_iso_ray,
        "w_local_iso_perp": cfg.loss.w_local_iso_perp,
        "w_local_iso_dist": cfg.loss.w_local_iso_dist,
    }
    nonzero_unrelated = [
        name for name, weight in unrelated_loss_weights.items() if weight != 0.0
    ]
    if nonzero_unrelated:
        raise ValueError(
            "anchor_delta refinement only uses RGB, mask, and Stage 3A losses; "
            f"set these weights to zero: {nonzero_unrelated}"
        )
    named_refinement_weights = {
        "w_rgb": cfg.loss.w_rgb,
        "w_mask": cfg.loss.w_mask,
        "w_modal_2d": cfg.loss.w_modal_2d,
        "w_delta_phi_prior": cfg.loss.w_delta_phi_prior,
        "w_delta_phi_spatial": cfg.loss.w_delta_phi_spatial,
    }
    invalid_weights = [
        name
        for name, weight in named_refinement_weights.items()
        if not np.isfinite(weight) or weight < 0.0
    ]
    if invalid_weights:
        raise ValueError(
            f"Stage 3A loss weights must be finite and non-negative: {invalid_weights}"
        )
    if cfg.loss.w_rgb <= 0.0:
        raise ValueError("anchor_delta refinement requires --loss.w-rgb > 0")
    if cfg.loss.w_delta_phi_prior <= 0.0:
        raise ValueError(
            "anchor_delta refinement requires --loss.w-delta-phi-prior > 0"
        )
    delta_lr = cfg.lr.modal_refinement.delta_phi
    if not np.isfinite(delta_lr) or delta_lr <= 0.0:
        raise ValueError(
            "--lr.modal-refinement.delta-phi must be finite and positive"
        )


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
