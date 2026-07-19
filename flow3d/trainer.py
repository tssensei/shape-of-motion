import functools
import time
from dataclasses import asdict
from typing import Any, Callable, cast

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
from flow3d.modal_joint_optimization import (
    ModalJointTrainingContext,
    weighted_rigidity_loss,
)
from flow3d.modal_utils import (
    MOTION_FILL_DISPLAY_ANCHOR,
    MOTION_FILL_DISPLAY_FILLED,
    MOTION_FILL_DISPLAY_PARTIAL,
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
        init_metadata: dict[str, Any] | None = None,
        modal_joint_context: ModalJointTrainingContext | None = None,
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
        self.init_metadata = init_metadata
        if self.model.trajectory_type == "modal_activation":
            if not self.model.has_modal_joint or modal_joint_context is None:
                raise ValueError(
                    "Trainer only accepts joint q/phi modal checkpoints with a "
                    "validated training context"
                )
            trainable_names = {
                name
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
            expected_names = {
                "modal_joint.params.delta_coordinate_real",
                "modal_joint.params.delta_coordinate_imag",
                "modal_joint.params.delta_phi_real",
                "modal_joint.params.delta_phi_imag",
            }
            if trainable_names != expected_names:
                raise ValueError(
                    "Joint q/phi optimizer received unexpected trainable parameters: "
                    f"{sorted(trainable_names)}"
                )
        elif modal_joint_context is not None:
            raise ValueError("modal joint training context requires modal_activation")
        self.modal_joint_context = modal_joint_context

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

        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server,
                self.render_fn,
                model.num_frames,
                work_dir,
                mode="training",
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
        path: str,
        device: torch.device,
        use_2dgs,
        *args,
        modal_joint_context_loader: (
            Callable[[SceneModel], ModalJointTrainingContext] | None
        ) = None,
        **kwargs,
    ) -> tuple["Trainer", int]:
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, weights_only=False)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)
        if model.has_modal_joint:
            model.requires_grad_(False)
            assert model.modal_joint is not None
            model.modal_joint.requires_grad_(True)
        print(use_2dgs)
        model.use_2dgs = use_2dgs
        modal_joint_context = (
            None
            if modal_joint_context_loader is None
            else modal_joint_context_loader(model)
        )
        trainer = Trainer(
            model,
            device,
            *args,
            init_metadata=ckpt.get("init_metadata"),
            modal_joint_context=modal_joint_context,
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
            t = self.viewer.current_timestep()
        self.model.training = False
        img = self.model.render(
            t,
            w2c[None],
            K[None],
            img_wh,
        )["img"][0]
        return (img.cpu().numpy() * 255.0).astype(np.uint8)

    def train_step(self, batch):
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.1)
            self.viewer.lock.acquire()

        loss, stats, num_rays_per_step, num_rays_per_sec = self.compute_losses(batch)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite loss at step {self.global_step}")
        loss.backward()
        if self.model.has_modal_joint:
            assert self.model.modal_joint is not None
            for name, parameter in self.model.modal_joint.params.items():
                if parameter.grad is None:
                    raise RuntimeError(f"Missing gradient for modal_joint.params.{name}")
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    raise FloatingPointError(
                        f"Non-finite gradient for modal_joint.params.{name}"
                    )
                stats[f"train/modal_{name}_grad_norm"] = float(
                    torch.linalg.vector_norm(parameter.grad).item()
                )
        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in self.scheduler.values():
            sched.step()

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

    def _modal_reference_coordinates(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.modal_joint_context
        if context is None:
            raise RuntimeError("modal joint training context is not initialized")
        all_real, all_imag = self.model.get_all_modal_coefficients()
        real_rows: list[torch.Tensor] = []
        imag_rows: list[torch.Tensor] = []
        for view_index in range(len(context.view_ids)):
            active_ts = int(context.reference_active_ts[view_index].item())
            if active_ts >= 0:
                real_rows.append(all_real[active_ts])
                imag_rows.append(all_imag[active_ts])
            else:
                real_rows.append(context.reference_coordinate_real[view_index])
                imag_rows.append(context.reference_coordinate_imag[view_index])
        return torch.stack(real_rows), torch.stack(imag_rows)

    @staticmethod
    def _project_points(
        points: torch.Tensor,
        w2c: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        camera = torch.einsum(
            "ij,gj->gi",
            w2c[:3],
            F.pad(points, (0, 1), value=1.0),
        )
        pixels_h = torch.einsum("ij,gj->gi", K, camera)
        depth = pixels_h[:, 2:3].clamp_min(1e-6)
        return pixels_h[:, :2] / depth

    def _modal_coordinate_prior_loss(self) -> torch.Tensor:
        context = self.modal_joint_context
        if context is None:
            raise RuntimeError("modal joint training context is not initialized")
        delta_real, delta_imag = self.model.get_centered_modal_coordinate_deltas()
        view_indices = self.model.modal_frame_view_indices
        losses: list[torch.Tensor] = []
        for view_index in range(len(context.view_ids)):
            rows = view_indices == view_index
            normalized = (
                delta_real[rows].square() + delta_imag[rows].square()
            ) / context.coordinate_scale[view_index].square().clamp_min(1e-12)
            losses.append(normalized.mean(dim=0))
        return torch.stack(losses).mean()

    def _modal_coordinate_temporal_loss(self) -> torch.Tensor:
        context = self.modal_joint_context
        if context is None:
            raise RuntimeError("modal joint training context is not initialized")
        real, imag = self.model.get_all_modal_coefficients()
        view_indices = self.model.modal_frame_view_indices
        local_indices = self.model.modal_frame_local_indices
        times = self.model.modal_frame_times_sec
        frequencies = self.model.modal_freqs_hz
        theta = 2.0 * torch.pi * times[:, None] * frequencies[None]
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        envelope_real = real * cosine + imag * sine
        envelope_imag = imag * cosine - real * sine
        losses: list[torch.Tensor] = []
        time_scale = float(self.losses_cfg.modal_coordinate_temporal_scale_sec)
        for view_index in range(len(context.view_ids)):
            rows = torch.nonzero(view_indices == view_index, as_tuple=False).flatten()
            order = torch.argsort(local_indices[rows])
            rows = rows[order]
            if rows.numel() < 2:
                raise RuntimeError(
                    f"View {context.view_ids[view_index]!r} needs at least two frames"
                )
            dt = times[rows[1:]] - times[rows[:-1]]
            if bool((dt <= 0).any().item()):
                raise RuntimeError("modal frame times must increase within each view")
            delta_real = envelope_real[rows[1:]] - envelope_real[rows[:-1]]
            delta_imag = envelope_imag[rows[1:]] - envelope_imag[rows[:-1]]
            normalized = (
                delta_real.square() + delta_imag.square()
            ) / context.coordinate_scale[view_index].square().clamp_min(1e-12)
            normalized = normalized / (dt[:, None] / time_scale).square()
            losses.append(normalized.mean(dim=0))
        return torch.stack(losses).mean()

    def _modal_phi_prior_loss(
        self,
        effective_real: torch.Tensor,
        effective_imag: torch.Tensor,
    ) -> torch.Tensor:
        context = self.modal_joint_context
        if context is None:
            raise RuntimeError("modal joint training context is not initialized")
        delta_energy = (
            effective_real - self.model.modal_phi_real
        ).square() + (effective_imag - self.model.modal_phi_imag).square()
        staged_energy = self.model.modal_phi_real.square() + self.model.modal_phi_imag.square()
        losses: list[torch.Tensor] = []
        for role in (
            MOTION_FILL_DISPLAY_ANCHOR,
            MOTION_FILL_DISPLAY_PARTIAL,
            MOTION_FILL_DISPLAY_FILLED,
        ):
            selected = (
                (context.display_class == role)
                & self.model.modal_phi_trainable_mask
            )
            for mode_slot in range(selected.shape[0]):
                mode_selected = selected[mode_slot]
                if bool(mode_selected.any().item()):
                    denominator = staged_energy[mode_slot, mode_selected].mean()
                    losses.append(
                        delta_energy[mode_slot, mode_selected].mean()
                        / denominator.clamp_min(1e-12)
                    )
        if not losses:
            raise RuntimeError("modal phi prior has no trainable role groups")
        return torch.stack(losses).mean()

    def _compute_modal_joint_losses(self, batch):
        context = self.modal_joint_context
        if context is None or not self.model.has_modal_joint:
            raise RuntimeError("joint q/phi loss requires a validated context")
        started = time.time()
        ts = batch["ts"]
        w2cs = batch["w2cs"]
        Ks = batch["Ks"]
        imgs = batch["imgs"]
        masks = batch["masks"] > 0.5
        valid_masks = batch.get("valid_masks", torch.ones_like(batch["masks"])) > 0.5
        batch_size, height, width = imgs.shape[:3]
        if (height, width) != (context.flow_height, context.flow_width):
            raise ValueError(
                "Training image resolution does not match modal flow cache: "
                f"{(height, width)} versus {(context.flow_height, context.flow_width)}"
            )
        img_wh = (width, height)

        coordinate_real, coordinate_imag = self.model.compute_modal_coefficients(ts)
        effective_phi_real, effective_phi_imag = self.model.get_effective_modal_phi()
        dynamic_offsets = torch.einsum(
            "bk,kgc->gbc", coordinate_real, effective_phi_real
        ) - torch.einsum("bk,kgc->gbc", coordinate_imag, effective_phi_imag)
        canonical_fg = self.model.fg.params["means"]
        dynamic_fg = canonical_fg[:, None] + dynamic_offsets
        foreground_quats = self.model.fg.get_quats()[:, None].expand(
            -1, batch_size, -1
        )
        if self.model.bg is None:
            means_all = dynamic_fg.transpose(0, 1)
            quats_all = foreground_quats.transpose(0, 1)
        else:
            background_means, background_quats = self.model.compute_poses_bg()
            means_all = torch.cat(
                [
                    dynamic_fg,
                    background_means[:, None].expand(-1, batch_size, -1),
                ],
                dim=0,
            ).transpose(0, 1)
            quats_all = torch.cat(
                [
                    foreground_quats,
                    background_quats[:, None].expand(-1, batch_size, -1),
                ],
                dim=0,
            ).transpose(0, 1)

        rgb_losses: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            rendered = self.model.render(
                int(ts[batch_index].item()),
                w2cs[batch_index : batch_index + 1],
                Ks[batch_index : batch_index + 1],
                img_wh,
                means=means_all[batch_index],
                quats=quats_all[batch_index],
            )["img"][0]
            support = masks[batch_index] & valid_masks[batch_index]
            if not bool(support.any().item()):
                raise ValueError("Foreground RGB support is empty")
            rgb_losses.append(torch.abs(rendered[support] - imgs[batch_index][support]).mean())
        rgb_loss = torch.stack(rgb_losses).mean()

        frame_view_indices = self.model.modal_frame_view_indices[ts]
        frame_local_indices = self.model.modal_frame_local_indices[ts]
        observed_flow, cache_masks = context.load_flow_batch(
            frame_view_indices,
            frame_local_indices,
            device=self.device,
            dtype=dynamic_fg.dtype,
        )
        reference_real, reference_imag = self._modal_reference_coordinates()
        batch_reference_real = reference_real[frame_view_indices]
        batch_reference_imag = reference_imag[frame_view_indices]
        reference_offsets = torch.einsum(
            "bk,kgc->gbc", batch_reference_real, effective_phi_real
        ) - torch.einsum(
            "bk,kgc->gbc", batch_reference_imag, effective_phi_imag
        )
        reference_fg = canonical_fg[:, None] + reference_offsets
        canonical_quats = self.model.fg.get_quats()

        flow_losses: list[torch.Tensor] = []
        flow_support_fractions: list[torch.Tensor] = []
        epsilon = float(self.losses_cfg.modal_flow_charbonnier_epsilon_px)
        image_diagonal = float((height * height + width * width) ** 0.5)
        for batch_index in range(batch_size):
            reference_points = reference_fg[:, batch_index]
            current_points = dynamic_fg[:, batch_index]
            uv_reference = self._project_points(
                reference_points, w2cs[batch_index], Ks[batch_index]
            )
            uv_current = self._project_points(
                current_points, w2cs[batch_index], Ks[batch_index]
            )
            gaussian_flow = uv_current - uv_reference
            flow_render = self.model.render(
                None,
                w2cs[batch_index : batch_index + 1],
                Ks[batch_index : batch_index + 1],
                img_wh,
                bg_color=torch.zeros(1, 2, device=self.device),
                colors_override=gaussian_flow,
                means=reference_points,
                quats=canonical_quats,
                fg_only=True,
            )
            accumulation = flow_render["acc"][0, ..., 0]
            predicted_flow = flow_render["img"][0] / accumulation[..., None].clamp_min(1e-6)
            support = cache_masks[batch_index] & (
                accumulation >= self.losses_cfg.modal_flow_render_acc_min
            )
            if not bool(support.any().item()):
                raise ValueError("Rendered flow support is empty")
            residual = predicted_flow[support] - observed_flow[batch_index][support]
            robust = torch.sqrt(residual.square().sum(dim=-1) + epsilon**2) - epsilon
            flow_losses.append(robust.mean() / image_diagonal)
            flow_support_fractions.append(support.float().mean())
        flow_loss = torch.stack(flow_losses).mean()
        flow_support_fraction = torch.stack(flow_support_fractions).mean()

        rigidity_loss, mean_abs_strain, max_abs_strain = weighted_rigidity_loss(
            dynamic_fg,
            canonical_fg,
            context.edge_index,
            context.edge_weight,
            self.losses_cfg.modal_rigidity_huber_beta,
        )
        coordinate_prior_loss = self._modal_coordinate_prior_loss()
        coordinate_temporal_loss = self._modal_coordinate_temporal_loss()
        phi_prior_loss = self._modal_phi_prior_loss(
            effective_phi_real,
            effective_phi_imag,
        )

        loss = (
            self.losses_cfg.w_rgb * rgb_loss
            + self.losses_cfg.w_modal_flow * flow_loss
            + self.losses_cfg.w_modal_rigidity * rigidity_loss
            + self.losses_cfg.w_modal_coordinate_prior * coordinate_prior_loss
            + self.losses_cfg.w_modal_coordinate_temporal * coordinate_temporal_loss
            + self.losses_cfg.w_modal_phi_prior * phi_prior_loss
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Joint modal objective produced a non-finite loss")

        num_rays_per_step = height * width * batch_size
        num_rays_per_sec = num_rays_per_step / (time.time() - started)
        stats = {
            "train/loss": float(loss.detach().item()),
            "train/rgb_loss": float(rgb_loss.detach().item()),
            "train/modal_flow_loss": float(flow_loss.detach().item()),
            "train/modal_flow_support_fraction": float(flow_support_fraction.item()),
            "train/modal_rigidity_loss": float(rigidity_loss.detach().item()),
            "train/modal_rigidity_mean_abs_strain": float(mean_abs_strain.detach().item()),
            "train/modal_rigidity_max_abs_strain": float(max_abs_strain.detach().item()),
            "train/modal_coordinate_prior_loss": float(coordinate_prior_loss.detach().item()),
            "train/modal_coordinate_temporal_loss": float(coordinate_temporal_loss.detach().item()),
            "train/modal_phi_prior_loss": float(phi_prior_loss.detach().item()),
            "train/num_rays_per_sec": num_rays_per_sec,
            "train/num_rays_per_step": float(num_rays_per_step),
        }
        return loss, stats, num_rays_per_step, num_rays_per_sec

    def compute_losses(self, batch):
        self.model.training = True
        if self.model.trajectory_type == "modal_activation":
            return self._compute_modal_joint_losses(batch)
        use_track_terms = self.model.trajectory_type in ("som_basis", "dct_center")

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
            rendered_all["rend_normal"] != None
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
        if use_track_terms and self.losses_cfg.w_smooth_tracks > 0:
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
            use_track_terms
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
        if not use_track_terms:
            z_accel_loss = torch.zeros((), device=self.device)
        else:
            z_accel_loss = compute_z_acc_loss(means_fg_nbs, w2cs)


        loss += self.losses_cfg.w_z_accel * z_accel_loss
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
            "train/z_acc_loss": z_accel_loss.item(),
            "train/local_iso_ray_loss": local_iso_ray_loss.item(),
            "train/local_iso_perp_loss": local_iso_perp_loss.item(),
            "train/local_iso_dist_loss": local_iso_dist_loss.item(),
            "train/local_iso_num_edges": local_iso_num_edges.item(),
            "train/num_gaussians": self.model.num_gaussians,
            "train/num_fg_gaussians": self.model.num_fg_gaussians,
            "train/num_bg_gaussians": self.model.num_bg_gaussians,
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
        if self.model.trajectory_type == "modal_activation":
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
            if not params.requires_grad:
                continue
            if (
                self.model.trajectory_type in ("dct_center", "modal_activation", "static")
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
