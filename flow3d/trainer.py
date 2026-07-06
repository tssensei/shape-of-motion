import functools
import time
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState
from pytorch_msssim import SSIM
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.loss_utils import (
    compute_gradient_loss,
    compute_ray_local_isometry_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    masked_l1_loss,
)
from flow3d.metrics import PCK, mLPIPS, mPSNR, mSSIM
from flow3d.modal_utils import (
    interpolate_modal_modes_to_gaussians,
    load_modal_consistency_data,
    load_modal_modes,
)
from flow3d.scene_model import SceneModel
from flow3d.vis.utils import get_server
from flow3d.vis.viewer import DynamicViewer
from flow3d.normal_utils import depth_to_normal

class Trainer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        lr_cfg: SceneLRConfig,
        losses_cfg: LossesConfig,
        optim_cfg: OptimizerConfig,
        # Logging.
        work_dir: str,
        port: int | None = None,
        log_every: int = 10,
        checkpoint_every: int = 200,
        validate_every: int = 500,
        validate_video_every: int = 1000,
        validate_viewer_assets_every: int = 100,
        modal_warmup_epochs: int = 0,
        modal_train_base_means: bool = False,
        modal_stage2_train_base_means: bool = False,
        modal_stage2_train_colors: bool = True,
        modal_stage2_train_opacities: bool = True,
        modal_stage2_train_scales: bool = False,
        modal_stage2_train_quats: bool = False,
        modal_stage2_train_bg_means: bool = False,
        modal_stage2_train_bg_colors: bool = True,
        modal_stage2_train_bg_opacities: bool = True,
        modal_stage2_train_bg_scales: bool = False,
        modal_stage2_train_bg_quats: bool = False,
        modal_stage2_lr_fg_scales: float | None = None,
        modal_stage2_lr_fg_quats: float | None = None,
        init_metadata: dict[str, Any] | None = None,
        modal_manifest: str | None = None,
        modal_knn: int = 8,
        modal_interp_power: float = 2.0,
        modal_interp_eps: float = 1e-6,
        modal_consistency_view_configs: tuple[str, ...] = (),
        modal_consistency_modal_npzs: tuple[str, ...] = (),
        modal_consistency_freq_tolerance_hz: float = 0.1,
        modal_consistency_mask_erode_iters: int = 1,
        modal_consistency_zbuffer_radius: int = 5,
        modal_consistency_front_percentile: float = 10.0,
        modal_consistency_zbuffer_tau: float = 0.05,
        modal_consistency_min_zbuffer_samples: int = 5,
    ):
        self.device = device
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.validate_every = validate_every
        self.validate_video_every = validate_video_every
        self.validate_viewer_assets_every = validate_viewer_assets_every

        self.model = model
        self.num_frames = model.num_frames

        self.lr_cfg = lr_cfg
        self.losses_cfg = losses_cfg
        self.optim_cfg = optim_cfg
        self.modal_warmup_epochs = modal_warmup_epochs
        self.modal_train_base_means = modal_train_base_means
        self.modal_stage2_train_base_means = (
            modal_stage2_train_base_means or modal_train_base_means
        )
        self.modal_stage2_train_colors = modal_stage2_train_colors
        self.modal_stage2_train_opacities = modal_stage2_train_opacities
        self.modal_stage2_train_scales = modal_stage2_train_scales
        self.modal_stage2_train_quats = modal_stage2_train_quats
        self.modal_stage2_train_bg_means = modal_stage2_train_bg_means
        self.modal_stage2_train_bg_colors = modal_stage2_train_bg_colors
        self.modal_stage2_train_bg_opacities = modal_stage2_train_bg_opacities
        self.modal_stage2_train_bg_scales = modal_stage2_train_bg_scales
        self.modal_stage2_train_bg_quats = modal_stage2_train_bg_quats
        self.modal_stage2_lr_fg_scales = modal_stage2_lr_fg_scales
        self.modal_stage2_lr_fg_quats = modal_stage2_lr_fg_quats
        self.init_metadata = init_metadata
        self.modal_manifest = modal_manifest
        self.modal_knn = modal_knn
        self.modal_interp_power = modal_interp_power
        self.modal_interp_eps = modal_interp_eps
        self.modal_consistency_view_configs = modal_consistency_view_configs
        self.modal_consistency_modal_npzs = modal_consistency_modal_npzs
        self.modal_consistency_freq_tolerance_hz = modal_consistency_freq_tolerance_hz
        self.modal_consistency_mask_erode_iters = modal_consistency_mask_erode_iters
        self.modal_consistency_zbuffer_radius = modal_consistency_zbuffer_radius
        self.modal_consistency_front_percentile = modal_consistency_front_percentile
        self.modal_consistency_zbuffer_tau = modal_consistency_zbuffer_tau
        self.modal_consistency_min_zbuffer_samples = (
            modal_consistency_min_zbuffer_samples
        )
        self._modal_post_warmup_refreshed = False

        self.reset_opacity_every = (
            self.optim_cfg.reset_opacity_every_n_controls * self.optim_cfg.control_every
        )
        self.optimizers, self.scheduler = self.configure_optimizers()

        # running stats for adaptive density control
        self.running_stats = {
            "xys_grad_norm_acc": torch.zeros(self.model.num_gaussians, device=device),
            "vis_count": torch.zeros(
                self.model.num_gaussians, device=device, dtype=torch.int64
            ),
            "max_radii": torch.zeros(self.model.num_gaussians, device=device),
        }

        self.work_dir = work_dir
        self.writer = SummaryWriter(log_dir=work_dir)
        self.global_step = 0
        self.epoch = 0
        self._apply_modal_trainability()

        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server, self.render_fn, model.num_frames, work_dir, mode="training"
            )

        # metrics
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.psnr_metric = mPSNR()
        self.ssim_metric = mSSIM()
        self.lpips_metric = mLPIPS()
        self.pck_metric = PCK()
        self.bg_psnr_metric = mPSNR()
        self.fg_psnr_metric = mPSNR()
        self.bg_ssim_metric = mSSIM()
        self.fg_ssim_metric = mSSIM()
        self.bg_lpips_metric = mLPIPS()
        self.fg_lpips_metric = mLPIPS()

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._refresh_modal_post_warmup_if_needed()
        self._apply_modal_trainability()
        self._apply_modal_stage2_lr_overrides()

    def _modal_in_dynamic_stage(self) -> bool:
        return (
            self.model.trajectory_type == "modal_activation"
            and self.epoch >= self.modal_warmup_epochs
        )

    def _apply_modal_trainability(self):
        if self.model.trajectory_type != "modal_activation":
            return
        dynamic_stage = self._modal_in_dynamic_stage()
        for name, param in self.model.named_parameters():
            if name.startswith("motion_bases."):
                trainable = False
            elif name == "modal.params.activations":
                trainable = dynamic_stage
            elif dynamic_stage:
                trainable = self._modal_stage2_param_trainable(name)
            else:
                trainable = not name.startswith("modal.")
            param.requires_grad_(trainable)

    def _modal_stage2_param_trainable(self, name: str) -> bool:
        trainable_fg_params = {
            "fg.params.means": self.modal_stage2_train_base_means,
            "fg.params.colors": self.modal_stage2_train_colors,
            "fg.params.opacities": self.modal_stage2_train_opacities,
            "fg.params.scales": self.modal_stage2_train_scales,
            "fg.params.quats": self.modal_stage2_train_quats,
            "bg.params.means": self.modal_stage2_train_bg_means,
            "bg.params.colors": self.modal_stage2_train_bg_colors,
            "bg.params.opacities": self.modal_stage2_train_bg_opacities,
            "bg.params.scales": self.modal_stage2_train_bg_scales,
            "bg.params.quats": self.modal_stage2_train_bg_quats,
        }
        return trainable_fg_params.get(name, False)

    def _apply_modal_stage2_lr_overrides(self):
        if not self._modal_in_dynamic_stage():
            return
        lr_overrides = {
            "fg.params.scales": self.modal_stage2_lr_fg_scales,
            "fg.params.quats": self.modal_stage2_lr_fg_quats,
        }
        for name, lr in lr_overrides.items():
            if lr is None:
                continue
            if name not in self.optimizers:
                raise ValueError(f"Missing optimizer for modal Stage 2 LR override: {name}")
            for group in self.optimizers[name].param_groups:
                group["lr"] = float(lr)

    @torch.no_grad()
    def _refresh_modal_post_warmup_if_needed(self):
        if self.model.trajectory_type != "modal_activation":
            return
        if self._modal_post_warmup_refreshed:
            return
        if not self._modal_in_dynamic_stage():
            return
        if self.modal_manifest is None:
            raise ValueError("modal post-warmup refresh requires modal_manifest")

        guru.info(
            "Refreshing modal fields after warmup from current Gaussian means"
        )
        modal_modes = load_modal_modes(self.modal_manifest)
        modal_phi_real, modal_phi_imag, modal_freqs_hz, _ = (
            interpolate_modal_modes_to_gaussians(
                self.model.fg.params["means"],
                modal_modes,
                self.modal_knn,
                self.modal_interp_power,
                self.modal_interp_eps,
            )
        )
        self.model.set_modal_fields(modal_phi_real, modal_phi_imag, modal_freqs_hz)

        use_modal_consistency = bool(
            self.modal_consistency_view_configs or self.modal_consistency_modal_npzs
        )
        if use_modal_consistency:
            target_view_index = int(
                self.model.modal_consistency_target_view_index.item()
            )
            fps = float(self.model.modal_consistency_fps.item())
            if target_view_index < 0:
                raise ValueError(
                    "modal consistency refresh requires target view index"
                )
            if fps <= 0:
                raise ValueError("modal consistency refresh requires positive fps")
            modal_consistency = load_modal_consistency_data(
                self.model.fg.params["means"],
                modal_modes,
                self.modal_consistency_view_configs,
                self.modal_consistency_modal_npzs,
                self.modal_consistency_freq_tolerance_hz,
                self.modal_consistency_mask_erode_iters,
                self.modal_consistency_zbuffer_radius,
                self.modal_consistency_front_percentile,
                self.modal_consistency_zbuffer_tau,
                self.modal_consistency_min_zbuffer_samples,
            )
            self.model.set_modal_consistency_data(
                modal_consistency.y_real,
                modal_consistency.y_imag,
                modal_consistency.J,
                modal_consistency.gaussian_indices,
                modal_consistency.mode_indices,
                modal_consistency.group_indices,
                modal_consistency.group_count,
                target_view_index=target_view_index,
                fps=fps,
            )
            guru.info(
                "Refreshed modal consistency cache with "
                f"{modal_consistency.y_real.shape[0]} observations across "
                f"{modal_consistency.group_count} view-frequency groups"
            )
        else:
            self.model.set_modal_consistency_data()

        self._modal_post_warmup_refreshed = True

    def save_checkpoint(self, path: str):
        model_dict = self.model.state_dict()
        optimizer_dict = {k: v.state_dict() for k, v in self.optimizers.items()}
        scheduler_dict = {k: v.state_dict() for k, v in self.scheduler.items()}
        ckpt = {
            "model": model_dict,
            "optimizers": optimizer_dict,
            "schedulers": scheduler_dict,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "init_metadata": self.init_metadata,
        }
        torch.save(ckpt, path)
        guru.info(f"Saved checkpoint at {self.global_step=} to {path}")

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, use_2dgs, *args, **kwargs
    ) -> tuple["Trainer", int]:
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)
        print(use_2dgs)
        model.use_2dgs = use_2dgs
        trainer = Trainer(
            model,
            device,
            *args,
            init_metadata=ckpt.get("init_metadata"),
            **kwargs,
        )
        if "optimizers" in ckpt:
            trainer.load_checkpoint_optimizers(ckpt["optimizers"])
        if "schedulers" in ckpt:
            trainer.load_checkpoint_schedulers(ckpt["schedulers"])
        trainer.global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        trainer.set_epoch(start_epoch)
        return trainer, start_epoch

    def load_checkpoint_optimizers(self, opt_ckpt):
        for k, v in self.optimizers.items():
            v.load_state_dict(opt_ckpt[k])

    def load_checkpoint_schedulers(self, sched_ckpt):
        for k, v in self.scheduler.items():
            v.load_state_dict(sched_ckpt[k])

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = 0
        if self.viewer is not None:
            t = (
                int(self.viewer._playback_guis[0].value)
                if not self.viewer._canonical_checkbox.value
                else None
            )
        self.model.training = False
        img = self.model.render(t, w2c[None], K[None], img_wh)["img"][0]
        return (img.cpu().numpy() * 255.0).astype(np.uint8)

    def train_step(self, batch):
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.1)
            self.viewer.lock.acquire()

        loss, stats, num_rays_per_step, num_rays_per_sec = self.compute_losses(batch)
        if loss.isnan():
            guru.info(f"Loss is NaN at step {self.global_step}!!")
            import ipdb

            ipdb.set_trace()
        loss.backward()

        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in self.scheduler.values():
            sched.step()
        self._apply_modal_stage2_lr_overrides()

        self.log_dict(stats)
        self.global_step += 1
        self.run_control_steps()

        if self.viewer is not None:
            self.viewer.lock.release()
            self.viewer.state.num_train_rays_per_sec = num_rays_per_sec
            if self.viewer.mode == "training":
                self.viewer.update(self.global_step, num_rays_per_step)

        if self.global_step % self.checkpoint_every == 0:
            self.save_checkpoint(f"{self.work_dir}/checkpoints/last.ckpt")

        return loss.item()

    def compute_losses(self, batch):
        self.model.training = True
        is_modal_activation = self.model.trajectory_type == "modal_activation"
        use_track_terms = not is_modal_activation

        B = batch["imgs"].shape[0]
        W, H = img_wh = batch["imgs"].shape[2:0:-1]
        N = batch["target_ts"][0].shape[0]

        # (B,).
        ts = batch["ts"]
        # (B, 4, 4).
        w2cs = batch["w2cs"]
        # (B, 3, 3).
        Ks = batch["Ks"]
        # (B, H, W, 3).
        imgs = batch["imgs"]
        # (B, H, W).
        valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0]))
        # (B, H, W).
        masks = batch["masks"]
        masks *= valid_masks
        # (B, H, W).
        depths = batch["depths"]
        # (B, H, W, 3)
        try:
            normals = batch["normals"]
        except:
            pass
        # [(P, 2), ...].
        query_tracks_2d = batch["query_tracks_2d"]
        # [(N,), ...].
        target_ts = batch["target_ts"]
        # [(N, 4, 4), ...].
        target_w2cs = batch["target_w2cs"]
        # [(N, 3, 3), ...].
        target_Ks = batch["target_Ks"]
        # [(N, P, 2), ...].
        target_tracks_2d = batch["target_tracks_2d"]
        # [(N, P), ...].
        target_visibles = batch["target_visibles"]
        # [(N, P), ...].
        target_invisibles = batch["target_invisibles"]
        # [(N, P), ...].
        target_confidences = batch["target_confidences"]
        # [(N, P), ...].
        target_track_depths = batch["target_track_depths"]

        _tic = time.time()
        # (B, G, 3).
        means, quats = self.model.compute_poses_all(ts)  # (G, B, 3), (G, B, 4)
        device = means.device
        means = means.transpose(0, 1)
        quats = quats.transpose(0, 1)
        if use_track_terms:
            # [(N, G, 3), ...].
            target_ts_vec = torch.cat(target_ts)
            # (B * N, G, 3).
            target_means, _ = self.model.compute_poses_all(target_ts_vec)
            target_means = target_means.transpose(0, 1)
            target_mean_list = target_means.split(N)
        else:
            target_ts_vec = None
            target_mean_list = [None] * B
        num_frames = self.model.num_frames

        loss = 0.0

        bg_colors = []
        rendered_all = []
        self._batched_xys = []
        self._batched_radii = []
        self._batched_img_wh = []
        for i in range(B):
            bg_color = torch.ones(1, 3, device=device)
            rendered = self.model.render(
                ts[i].item(),
                w2cs[None, i],
                Ks[None, i],
                img_wh,
                target_ts=target_ts[i] if use_track_terms else None,
                target_w2cs=target_w2cs[i] if use_track_terms else None,
                bg_color=bg_color,
                means=means[i],
                quats=quats[i],
                target_means=(
                    target_mean_list[i].transpose(0, 1)
                    if target_mean_list[i] is not None
                    else None
                ),
                return_depth=True,
                return_mask=self.model.has_bg,
            )
            rendered_all.append(rendered)
            bg_colors.append(bg_color)
            if (
                self.model._current_xys is not None
                and self.model._current_radii is not None
                and self.model._current_img_wh is not None
            ):
                self._batched_xys.append(self.model._current_xys)
                self._batched_radii.append(self.model._current_radii)
                self._batched_img_wh.append(self.model._current_img_wh)

        # Necessary to make viewer work.
        num_rays_per_step = H * W * B
        num_rays_per_sec = num_rays_per_step / (time.time() - _tic)

        # (B, H, W, N, *).
        rendered_all = {
            key: (
                torch.cat([out_dict[key] for out_dict in rendered_all], dim=0)
                if rendered_all[0][key] is not None
                else None
            )
            for key in rendered_all[0]
        }
        bg_colors = torch.cat(bg_colors, dim=0)

        # Compute losses.
        if use_track_terms:
            assert target_ts_vec is not None
            # (B * N).
            frame_intervals = (ts.repeat_interleave(N) - target_ts_vec).abs()
            # (P_all, 2).
            tracks_2d = torch.cat(
                [x.reshape(-1, 2) for x in target_tracks_2d], dim=0
            )
            # (P_all,).
            visibles = torch.cat([x.reshape(-1) for x in target_visibles], dim=0)
            # (P_all,).
            confidences = torch.cat(
                [x.reshape(-1) for x in target_confidences], dim=0
            )
        if not self.model.has_bg:
            imgs = (
                imgs * masks[..., None]
                + (1.0 - masks[..., None]) * bg_colors[:, None, None]
            )
        else:
            imgs = (
                imgs * valid_masks[..., None]
                + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
            )

        if (
            not is_modal_activation
            and rendered_all["rend_normal"] != None
            and rendered_all["surf_normal"] != None
        ):
            # 2DGS normal consistency
            rendered_normals = cast(torch.Tensor, rendered_all["rend_normal"])
            surf_normals = cast(torch.Tensor, rendered_all["surf_normal"])
            surf_normals = surf_normals.reshape(rendered_normals.shape)
            cos_sim = torch.sum(rendered_normals * surf_normals, dim=-1)
            normal_loss = (1 - cos_sim).mean()
            loss += normal_loss * 0.05


        # RGB loss.
        rendered_imgs = cast(torch.Tensor, rendered_all["img"])
        if self.model.has_bg:
            rendered_imgs = (
                rendered_imgs * valid_masks[..., None]
                + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
            )
        rgb_loss = 0.8 * F.l1_loss(rendered_imgs, imgs) + 0.2 * (
            1 - self.ssim(rendered_imgs.permute(0, 3, 1, 2), imgs.permute(0, 3, 1, 2))
        )
        loss += rgb_loss * self.losses_cfg.w_rgb

        # Mask loss.
        if not self.model.has_bg:
            mask_loss = F.mse_loss(rendered_all["acc"], masks[..., None])  # type: ignore
        else:
            mask_loss = F.mse_loss(
                rendered_all["acc"], torch.ones_like(rendered_all["acc"])  # type: ignore
            ) + masked_l1_loss(
                rendered_all["mask"],
                masks[..., None],
                quantile=0.98,  # type: ignore
            )
        loss += mask_loss * self.losses_cfg.w_mask

        if use_track_terms:
            # (B * N, H * W, 3).
            pred_tracks_3d = (
                rendered_all["tracks_3d"]
                .permute(0, 3, 1, 2, 4)
                .reshape(-1, H * W, 3)  # type: ignore
            )
            pred_tracks_2d = torch.einsum(
                "bij,bpj->bpi", torch.cat(target_Ks), pred_tracks_3d
            )
            # (B * N, H * W, 1).
            mapped_depth = torch.clamp(pred_tracks_2d[..., 2:], min=1e-6)
            # (B * N, H * W, 2).
            pred_tracks_2d = pred_tracks_2d[..., :2] / mapped_depth

            # (B * N).
            w_interval = torch.exp(-2 * frame_intervals / num_frames)
            # w_track_loss = min(1, (self.max_steps - self.global_step) / 6000)
            track_weights = confidences[..., None] * w_interval

            # (B, H, W).
            masks_flatten = torch.zeros_like(masks)
            for i in range(B):
                # This takes advantage of the fact that the query 2D tracks are
                # always on the grid.
                query_pixels = query_tracks_2d[i].to(torch.int64)
                masks_flatten[i, query_pixels[:, 1], query_pixels[:, 0]] = 1.0
            # (B * N, H * W).
            masks_flatten = (
                masks_flatten.reshape(-1, H * W).tile(1, N).reshape(-1, H * W) > 0.5
            )

            track_2d_loss = masked_l1_loss(
                pred_tracks_2d[masks_flatten][visibles],
                tracks_2d[visibles],
                mask=track_weights[visibles],
                quantile=0.98,
            ) / max(H, W)
            loss += track_2d_loss * self.losses_cfg.w_track
        else:
            mapped_depth = None
            masks_flatten = None
            track_weights = None
            visibles = None
            track_2d_loss = torch.zeros((), device=device)

        depth_masks = (
            masks[..., None] if not self.model.has_bg else valid_masks[..., None]
        )

        pred_depth = cast(torch.Tensor, rendered_all["depth"])
        pred_disp = 1.0 / (pred_depth + 1e-5)
        valid_depth_masks = torch.isfinite(depths[..., None]) & (depths[..., None] > 0)
        depth_masks = depth_masks * valid_depth_masks.float()
        tgt_disp = 1.0 / (depths[..., None] + 1e-5)
        depth_loss = masked_l1_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks,
            quantile=0.98,
        )
        loss += depth_loss * self.losses_cfg.w_depth_reg

        if use_track_terms:
            # mapped depth loss (using cached depth with EMA)
            #  mapped_depth_loss = 0.0
            assert mapped_depth is not None
            assert masks_flatten is not None
            assert visibles is not None
            assert track_weights is not None
            mapped_depth_gt = torch.cat(
                [x.reshape(-1) for x in target_track_depths], dim=0
            )
            mapped_depth_loss = masked_l1_loss(
                1 / (mapped_depth[masks_flatten][visibles] + 1e-5),
                1 / (mapped_depth_gt[visibles, None] + 1e-5),
                track_weights[visibles],
            )
            loss += mapped_depth_loss * self.losses_cfg.w_depth_const
        else:
            mapped_depth_loss = torch.zeros((), device=device)

        #  depth_gradient_loss = 0.0
        depth_gradient_loss = compute_gradient_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks > 0.5,
            quantile=0.95,
        )
        loss += depth_gradient_loss * self.losses_cfg.w_depth_grad

        # bases should be smooth.
        if self.model.trajectory_type == "som_basis":
            small_accel_loss = compute_se3_smoothness_loss(
                self.model.motion_bases.params["rots"],
                self.model.motion_bases.params["transls"],
            )
            loss += small_accel_loss * self.losses_cfg.w_smooth_bases
            dct_coef_loss = torch.zeros((), device=self.device)
        elif self.model.trajectory_type == "dct_center":
            small_accel_loss = torch.zeros((), device=self.device)
            dct_coef_loss = self.model.fg.params["traj_coefs"].pow(2).mean()
            loss += dct_coef_loss * self.losses_cfg.w_dct_coef
        else:
            small_accel_loss = torch.zeros((), device=self.device)
            dct_coef_loss = torch.zeros((), device=self.device)

        # tracks should be smooth
        ts = torch.clamp(ts, min=1, max=num_frames - 2)
        ts_neighbors = torch.cat((ts - 1, ts, ts + 1))
        means_fg_nbs, _ = self.model.compute_poses_fg(ts_neighbors)
        means_fg_nbs = means_fg_nbs.reshape(
            means_fg_nbs.shape[0], 3, -1, 3
        )  # [G, 3, n, 3]
        if not is_modal_activation and self.losses_cfg.w_smooth_tracks > 0:
            small_accel_loss_tracks = 0.5 * (
                (2 * means_fg_nbs[:, 1:-1] - means_fg_nbs[:, :-2] - means_fg_nbs[:, 2:])
                .norm(dim=-1)
                .mean()
            )
            loss += small_accel_loss_tracks * self.losses_cfg.w_smooth_tracks

        local_iso_ray_loss = torch.zeros((), device=self.device)
        local_iso_perp_loss = torch.zeros((), device=self.device)
        local_iso_dist_loss = torch.zeros((), device=self.device)
        local_iso_num_edges = torch.zeros((), device=self.device)
        local_iso_is_active = (
            not is_modal_activation
            and self.global_step >= self.losses_cfg.local_iso_start_step
            and (
                self.losses_cfg.w_local_iso_ray > 0
                or self.losses_cfg.w_local_iso_perp > 0
                or self.losses_cfg.w_local_iso_dist > 0
            )
        )
        if local_iso_is_active:
            (
                local_iso_ray_loss,
                local_iso_perp_loss,
                local_iso_dist_loss,
                local_iso_num_edges,
            ) = compute_ray_local_isometry_loss(
                means_fg_nbs[:, 1],
                self.model.fg.params["means"],
                w2cs,
                self.losses_cfg.local_iso_knn,
                self.losses_cfg.local_iso_radius_mult,
                self.losses_cfg.local_iso_huber_beta,
                self.losses_cfg.local_iso_edge_weight_temp,
            )
            loss += self.losses_cfg.w_local_iso_ray * local_iso_ray_loss
            loss += self.losses_cfg.w_local_iso_perp * local_iso_perp_loss
            loss += self.losses_cfg.w_local_iso_dist * local_iso_dist_loss


        # Constrain the std of scales.
        # TODO: do we want to penalize before or after exp?
        loss += (
            self.losses_cfg.w_scale_var
            * torch.var(torch.exp(self.model.fg.params["scales"]), dim=-1).mean()
        )
        if self.model.bg is not None:
            loss += (
                self.losses_cfg.w_scale_var
                * torch.var(torch.exp(self.model.bg.params["scales"]), dim=-1).mean()
            )
        
        if self.model.fg.params["means"].isnan().sum() > 0:
            import ipdb
            ipdb.set_trace()
        # # sparsity loss
        # loss += 0.01 * self.opacity_activation(self.opacities).abs().mean()

        # Acceleration along ray direction should be small.
        if is_modal_activation:
            z_accel_loss = torch.zeros((), device=self.device)
        else:
            z_accel_loss = compute_z_acc_loss(means_fg_nbs, w2cs)


        loss += self.losses_cfg.w_z_accel * z_accel_loss
        if is_modal_activation:
            act_smooth_loss = self.model.compute_activation_smoothness_loss()
            loss += self.losses_cfg.w_act_smooth * act_smooth_loss
            act_mag_loss = self.model.compute_activation_magnitude_loss()
            loss += self.losses_cfg.w_act_mag * act_mag_loss
            (
                act_modal_consistency_loss,
                act_modal_consistency_count,
            ) = self.model.compute_activation_modal_consistency_loss(
                self.losses_cfg.modal_consistency_loss_type,
                self.losses_cfg.modal_consistency_beta_abs_max,
                self.losses_cfg.modal_consistency_pred_energy_eps,
            )
            loss += (
                self.losses_cfg.w_act_modal_consistency
                * act_modal_consistency_loss
            )
        else:
            act_smooth_loss = torch.zeros((), device=self.device)
            act_mag_loss = torch.zeros((), device=self.device)
            act_modal_consistency_loss = torch.zeros((), device=self.device)
            act_modal_consistency_count = torch.zeros((), device=self.device)

        # Prepare stats for logging.
        stats = {
            "train/loss": loss.item(),
            "train/rgb_loss": rgb_loss.item(),
            "train/mask_loss": mask_loss.item(),
            "train/depth_loss": depth_loss.item(),
            "train/depth_gradient_loss": depth_gradient_loss.item(),
            "train/mapped_depth_loss": mapped_depth_loss.item(),
            "train/track_2d_loss": track_2d_loss.item(),
            "train/small_accel_loss": small_accel_loss.item(),
            "train/dct_coef_loss": dct_coef_loss.item(),
            "train/act_smooth_loss": act_smooth_loss.item(),
            "train/act_mag_loss": act_mag_loss.item(),
            "train/act_modal_consistency_loss": act_modal_consistency_loss.item(),
            "train/act_modal_consistency_count": act_modal_consistency_count.item(),
            "train/z_acc_loss": z_accel_loss.item(),
            "train/local_iso_ray_loss": local_iso_ray_loss.item(),
            "train/local_iso_perp_loss": local_iso_perp_loss.item(),
            "train/local_iso_dist_loss": local_iso_dist_loss.item(),
            "train/local_iso_num_edges": local_iso_num_edges.item(),
            "train/num_gaussians": self.model.num_gaussians,
            "train/num_fg_gaussians": self.model.num_fg_gaussians,
            "train/num_bg_gaussians": self.model.num_bg_gaussians,
            "train/modal_dynamic_stage": float(self._modal_in_dynamic_stage()),
        }

        # Compute metrics.
        with torch.no_grad():
            psnr = self.psnr_metric(
                rendered_imgs, imgs, masks if not self.model.has_bg else valid_masks
            )
            self.psnr_metric.reset()
            stats["train/psnr"] = psnr
            if self.model.has_bg:
                bg_psnr = self.bg_psnr_metric(rendered_imgs, imgs, 1.0 - masks)
                fg_psnr = self.fg_psnr_metric(rendered_imgs, imgs, masks)
                self.bg_psnr_metric.reset()
                self.fg_psnr_metric.reset()
                stats["train/bg_psnr"] = bg_psnr
                stats["train/fg_psnr"] = fg_psnr

        stats.update(
            **{
                "train/num_rays_per_sec": num_rays_per_sec,
                "train/num_rays_per_step": float(num_rays_per_step),
            }
        )

        return loss, stats, num_rays_per_step, num_rays_per_sec

    def log_dict(self, stats: dict):
        for k, v in stats.items():
            self.writer.add_scalar(k, v, self.global_step)

    def run_control_steps(self):
        if self._modal_in_dynamic_stage():
            return
        global_step = self.global_step
        # Adaptive gaussian control.
        cfg = self.optim_cfg
        num_frames = self.model.num_frames
        ready = self._prepare_control_step()
        if (
            ready
            and global_step > cfg.warmup_steps
            and global_step % cfg.control_every == 0
            and global_step < cfg.stop_control_steps
        ):
            if (
                global_step < cfg.stop_densify_steps
                and global_step % self.reset_opacity_every > num_frames
            ):
                self._densify_control_step(global_step)
            if global_step % self.reset_opacity_every > min(3 * num_frames, 1000):
                self._cull_control_step(global_step)
            if global_step % self.reset_opacity_every == 0:
                self._reset_opacity_control_step()

            # Reset stats after every control.
            for k in self.running_stats:
                self.running_stats[k].zero_()

    @torch.no_grad()
    def _prepare_control_step(self) -> bool:
        # Prepare for adaptive gaussian control based on the current stats.
        if not (
            self.model._current_radii is not None
            and self.model._current_xys is not None
        ):
            guru.warning("Model not training, skipping control step preparation")
            return False

        batch_size = len(self._batched_xys)
        # these quantities are for each rendered view and have shapes (C, G, *)
        # must be aggregated over all views
        for _current_xys, _current_radii, _current_img_wh in zip(
            self._batched_xys, self._batched_radii, self._batched_img_wh
        ):
            sel = _current_radii > 0
            gidcs = torch.where(sel)[1]
            # normalize grads to [-1, 1] screen space
            xys_grad = _current_xys.grad.clone()
            xys_grad[..., 0] *= _current_img_wh[0] / 2.0 * batch_size
            xys_grad[..., 1] *= _current_img_wh[1] / 2.0 * batch_size
            self.running_stats["xys_grad_norm_acc"].index_add_(
                0, gidcs, xys_grad[sel].norm(dim=-1)
            )
            self.running_stats["vis_count"].index_add_(
                0, gidcs, torch.ones_like(gidcs, dtype=torch.int64)
            )
            max_radii = torch.maximum(
                self.running_stats["max_radii"].index_select(0, gidcs),
                _current_radii[sel] / max(_current_img_wh),
            )
            self.running_stats["max_radii"].index_put((gidcs,), max_radii)
        return True

    @torch.no_grad()
    def _densify_control_step(self, global_step):
        assert (self.running_stats["vis_count"] > 0).any()

        cfg = self.optim_cfg
        xys_grad_avg = self.running_stats["xys_grad_norm_acc"] / self.running_stats[
            "vis_count"
        ].clamp_min(1)
        is_grad_too_high = xys_grad_avg > cfg.densify_xys_grad_threshold
        # Split gaussians.
        scales = self.model.get_scales_all()
        is_scale_too_big = scales.amax(dim=-1) > cfg.densify_scale_threshold
        if global_step < cfg.stop_control_by_screen_steps:
            is_radius_too_big = (
                self.running_stats["max_radii"] > cfg.densify_screen_threshold
            )
        else:
            is_radius_too_big = torch.zeros_like(is_grad_too_high, dtype=torch.bool)

        should_split = is_grad_too_high & (is_scale_too_big | is_radius_too_big)
        should_dup = is_grad_too_high & ~is_scale_too_big

        num_fg = self.model.num_fg_gaussians
        should_fg_split = should_split[:num_fg]
        num_fg_splits = int(should_fg_split.sum().item())
        should_fg_dup = should_dup[:num_fg]
        num_fg_dups = int(should_fg_dup.sum().item())

        should_bg_split = should_split[num_fg:]
        num_bg_splits = int(should_bg_split.sum().item())
        should_bg_dup = should_dup[num_fg:]
        num_bg_dups = int(should_bg_dup.sum().item())

        fg_param_map = self.model.fg.densify_params(should_fg_split, should_fg_dup)
        self.model.densify_modal_fields(should_fg_split, should_fg_dup)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            dup_in_optim(
                optimizer,
                [new_params],
                should_fg_split,
                num_fg_splits * 2 + num_fg_dups,
            )

        if self.model.bg is not None:
            bg_param_map = self.model.bg.densify_params(should_bg_split, should_bg_dup)
            for param_name, new_params in bg_param_map.items():
                full_param_name = f"bg.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                dup_in_optim(
                    optimizer,
                    [new_params],
                    should_bg_split,
                    num_bg_splits * 2 + num_bg_dups,
                )

        # update running stats
        for k, v in self.running_stats.items():
            v_fg, v_bg = v[:num_fg], v[num_fg:]
            new_v = torch.cat(
                [
                    v_fg[~should_fg_split],
                    v_fg[should_fg_dup],
                    v_fg[should_fg_split].repeat(2),
                    v_bg[~should_bg_split],
                    v_bg[should_bg_dup],
                    v_bg[should_bg_split].repeat(2),
                ],
                dim=0,
            )
            self.running_stats[k] = new_v
        guru.info(
            f"Split {should_split.sum().item()} gaussians, "
            f"Duplicated {should_dup.sum().item()} gaussians, "
            f"{self.model.num_gaussians} gaussians left"
        )

    @torch.no_grad()
    def _cull_control_step(self, global_step):
        # Cull gaussians.
        cfg = self.optim_cfg
        opacities = self.model.get_opacities_all()
        device = opacities.device
        is_opacity_too_small = opacities < cfg.cull_opacity_threshold
        is_radius_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        is_scale_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        cull_scale_threshold = (
            torch.ones(len(is_scale_too_big), device=device) * cfg.cull_scale_threshold
        )
        num_fg = self.model.num_fg_gaussians
        cull_scale_threshold[num_fg:] *= self.model.bg_scene_scale
        if global_step > self.reset_opacity_every:
            scales = self.model.get_scales_all()
            is_scale_too_big = scales.amax(dim=-1) > cull_scale_threshold
            if global_step < cfg.stop_control_by_screen_steps:
                is_radius_too_big = (
                    self.running_stats["max_radii"] > cfg.cull_screen_threshold
                )
        should_cull = is_opacity_too_small | is_radius_too_big | is_scale_too_big
        should_fg_cull = should_cull[:num_fg]
        should_bg_cull = should_cull[num_fg:]

        fg_param_map = self.model.fg.cull_params(should_fg_cull)
        self.model.cull_modal_fields(should_fg_cull)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            remove_from_optim(optimizer, [new_params], should_fg_cull)

        if self.model.bg is not None:
            bg_param_map = self.model.bg.cull_params(should_bg_cull)
            for param_name, new_params in bg_param_map.items():
                full_param_name = f"bg.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                remove_from_optim(optimizer, [new_params], should_bg_cull)

        # update running stats
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[~should_cull]

        guru.info(
            f"Culled {should_cull.sum().item()} gaussians, "
            f"{self.model.num_gaussians} gaussians left"
        )

    @torch.no_grad()
    def _reset_opacity_control_step(self):
        # Reset gaussian opacities.
        new_val = torch.logit(torch.tensor(0.8 * self.optim_cfg.cull_opacity_threshold))
        for part in ["fg", "bg"]:
            part_module = getattr(self.model, part)
            if part_module is None:
                continue
            part_params = part_module.reset_opacities(new_val)
            # Modify optimizer states by new assignment.
            for param_name, new_params in part_params.items():
                full_param_name = f"{part}.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                reset_in_optim(optimizer, [new_params])
        guru.info("Reset opacities")

    def configure_optimizers(self):
        def _exponential_decay(step, *, lr_init, lr_final):
            t = np.clip(step / self.optim_cfg.max_steps, 0.0, 1.0)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init

        lr_dict = asdict(self.lr_cfg)
        optimizers = {}
        schedulers = {}
        # named parameters will be [part].params.[field]
        # e.g. fg.params.means
        # lr config is a nested dict for each fg/bg part
        for name, params in self.model.named_parameters():
            if (
                self.model.trajectory_type in ("dct_center", "modal_activation")
                and name.startswith("motion_bases.")
            ):
                continue
            part, _, field = name.split(".")
            lr = lr_dict[part][field]
            optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])

            if "scales" in name:
                fnc = functools.partial(_exponential_decay, lr_final=0.1 * lr)
            else:
                fnc = lambda _, **__: 1.0

            optimizers[name] = optim
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optim, functools.partial(fnc, lr_init=lr)
            )
        return optimizers, schedulers


def dup_in_optim(optimizer, new_params: list, should_dup: torch.Tensor, num_dups: int):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            p = param_state[key]
            param_state[key] = torch.cat(
                [p[~should_dup], p.new_zeros(num_dups, *p.shape[1:])],
                dim=0,
            )
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def remove_from_optim(optimizer, new_params: list, _should_cull: torch.Tensor):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            param_state[key] = param_state[key][~_should_cull]
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def reset_in_optim(optimizer, new_params: list):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            param_state[key] = torch.zeros_like(param_state[key])
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()
