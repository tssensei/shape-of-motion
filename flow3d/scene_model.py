import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat.rendering import rasterization
from gsplat.rendering import rasterization_2dgs
from torch import Tensor

from flow3d.params import (
    GaussianParams,
    ModalActivations,
    MotionBases,
    CameraScales,
    CameraPoses,
    build_dct_basis,
)

TRAJECTORY_TYPE_TO_ID = {
    "som_basis": 0,
    "dct_center": 1,
    "modal_activation": 2,
    "static": 3,
}
TRAJECTORY_ID_TO_TYPE = {value: key for key, value in TRAJECTORY_TYPE_TO_ID.items()}


class SceneModel(nn.Module):
    def __init__(
        self,
        Ks: Tensor, # camera intrinsics, [num_frames, 3, 3]
        w2cs: Tensor, # camera extrinsics, [num_frames, 4, 4]
        fg_params: GaussianParams,
        motion_bases: MotionBases,
        camera_poses: CameraPoses | None = None, # currently unused?
        bg_params: GaussianParams | None = None,
        use_2dgs: bool = False,
        trajectory_type: str = "som_basis",
        cano_t: int | None = None,
        num_dct_bases: int | None = None,
        modal: ModalActivations | None = None,
        modal_phi_real: Tensor | None = None,
        modal_phi_imag: Tensor | None = None,
        modal_freqs_hz: Tensor | None = None,
        modal_frame_view_indices: Tensor | None = None,
        modal_frame_local_indices: Tensor | None = None,
        modal_smooth_triplets: Tensor | None = None,
        modal_consistency_y_real: Tensor | None = None,
        modal_consistency_y_imag: Tensor | None = None,
        modal_consistency_J: Tensor | None = None,
        modal_consistency_gaussian_indices: Tensor | None = None,
        modal_consistency_mode_indices: Tensor | None = None,
        modal_consistency_group_indices: Tensor | None = None,
        modal_consistency_group_count: int = 0,
        modal_consistency_target_view_index: int = -1,
        modal_consistency_fps: float = 0.0,
        modal_synthetic_enabled: bool | Tensor = False,
    ):
        super().__init__()
        if trajectory_type not in TRAJECTORY_TYPE_TO_ID:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        self.num_frames = motion_bases.num_frames
        self.trajectory_type = trajectory_type
        self.register_buffer(
            "trajectory_type_id",
            torch.tensor(TRAJECTORY_TYPE_TO_ID[trajectory_type], dtype=torch.long),
        )
        self.fg = fg_params
        self.motion_bases = motion_bases
        self.modal = modal
        self.bg = bg_params
        scene_scale = 1.0 if bg_params is None else bg_params.scene_scale
        self.register_buffer("bg_scene_scale", torch.as_tensor(scene_scale))
        self.register_buffer("Ks", Ks)
        self.register_buffer("w2cs", w2cs)
        self.camera_poses = camera_poses

        self._current_xys = None
        self._current_radii = None
        self._current_img_wh = None

        self.use_2dgs = use_2dgs
        cano_t_value = -1 if cano_t is None else cano_t
        self.register_buffer("cano_t", torch.tensor(cano_t_value, dtype=torch.long))
        if trajectory_type == "dct_center":
            if cano_t is None:
                raise ValueError("cano_t is required for dct_center trajectory")
            if "traj_coefs" not in self.fg.params:
                raise ValueError("fg.params['traj_coefs'] is required for dct_center")
            if num_dct_bases is None:
                num_dct_bases = self.fg.params["traj_coefs"].shape[1]
            dct_basis = build_dct_basis(
                self.num_frames,
                num_dct_bases,
                cano_t,
                device=self.fg.params["means"].device,
                dtype=self.fg.params["means"].dtype,
            )
        else:
            dct_basis = torch.empty(
                self.num_frames, 0, device=self.fg.params["means"].device
            )
        self.register_buffer("dct_basis", dct_basis)

        if trajectory_type == "modal_activation":
            if modal is None:
                raise ValueError("modal_activation requires modal activations")
            if modal_phi_real is None or modal_phi_imag is None:
                raise ValueError("modal_activation requires modal phi real/imag tensors")
            if modal_phi_real.shape != modal_phi_imag.shape:
                raise ValueError("modal phi real/imag tensors must have matching shapes")
            if modal_phi_real.ndim != 3 or modal_phi_real.shape[-1] != 3:
                raise ValueError("modal phi tensors must have shape (K, G, 3)")
            if modal_phi_real.shape[0] == 0:
                modal_phi_real = torch.empty(
                    0,
                    self.num_fg_gaussians,
                    3,
                    device=self.fg.params["means"].device,
                    dtype=self.fg.params["means"].dtype,
                )
                modal_phi_imag = torch.empty_like(modal_phi_real)
            if modal_phi_real.shape[1] != self.num_fg_gaussians:
                raise ValueError("modal phi Gaussian dimension does not match foreground")
            if modal_phi_real.shape[0] != modal.num_modes:
                raise ValueError("modal phi mode count does not match activations")
            if modal.num_frames != self.num_frames:
                raise ValueError("modal activation frame count does not match model")
        else:
            if modal_phi_real is None:
                modal_phi_real = torch.empty(
                    0, self.num_fg_gaussians, 3, device=self.fg.params["means"].device
                )
            if modal_phi_imag is None:
                modal_phi_imag = torch.empty_like(modal_phi_real)
            if modal_phi_real.shape != modal_phi_imag.shape:
                raise ValueError("modal phi real/imag tensors must have matching shapes")
            if modal_phi_real.ndim != 3 or modal_phi_real.shape[-1] != 3:
                raise ValueError("modal phi tensors must have shape (K, G, 3)")
            if modal_phi_real.shape[1] != self.num_fg_gaussians:
                raise ValueError("modal phi Gaussian dimension does not match foreground")

        if modal_freqs_hz is None:
            modal_freqs_hz = torch.empty(
                modal_phi_real.shape[0],
                device=self.fg.params["means"].device,
                dtype=self.fg.params["means"].dtype,
            )
        if modal_frame_view_indices is None:
            modal_frame_view_indices = torch.full(
                (self.num_frames,), -1, device=self.fg.params["means"].device
            )
        if modal_frame_local_indices is None:
            modal_frame_local_indices = torch.full(
                (self.num_frames,), -1, device=self.fg.params["means"].device
            )
        if modal_smooth_triplets is None:
            modal_smooth_triplets = torch.empty(
                0, 3, device=self.fg.params["means"].device, dtype=torch.long
            )
        self.register_buffer("modal_phi_real", modal_phi_real)
        self.register_buffer("modal_phi_imag", modal_phi_imag)
        self.register_buffer("modal_freqs_hz", modal_freqs_hz)
        if isinstance(modal_synthetic_enabled, Tensor):
            modal_synthetic_enabled = bool(modal_synthetic_enabled.item())
        self.register_buffer(
            "modal_synthetic_enabled",
            torch.tensor(bool(modal_synthetic_enabled), dtype=torch.bool),
        )
        self.register_buffer("modal_frame_view_indices", modal_frame_view_indices.long())
        self.register_buffer("modal_frame_local_indices", modal_frame_local_indices.long())
        self.register_buffer("modal_smooth_triplets", modal_smooth_triplets.long())
        if modal_consistency_y_real is None:
            modal_consistency_y_real = torch.empty(
                0, 2, device=self.fg.params["means"].device, dtype=self.fg.params["means"].dtype
            )
        if modal_consistency_y_imag is None:
            modal_consistency_y_imag = torch.empty_like(modal_consistency_y_real)
        if modal_consistency_J is None:
            modal_consistency_J = torch.empty(
                0, 2, 3, device=self.fg.params["means"].device, dtype=self.fg.params["means"].dtype
            )
        if modal_consistency_gaussian_indices is None:
            modal_consistency_gaussian_indices = torch.empty(
                0, device=self.fg.params["means"].device, dtype=torch.long
            )
        if modal_consistency_mode_indices is None:
            modal_consistency_mode_indices = torch.empty(
                0, device=self.fg.params["means"].device, dtype=torch.long
            )
        if modal_consistency_group_indices is None:
            modal_consistency_group_indices = torch.empty(
                0, device=self.fg.params["means"].device, dtype=torch.long
            )
        self.register_buffer("modal_consistency_y_real", modal_consistency_y_real)
        self.register_buffer("modal_consistency_y_imag", modal_consistency_y_imag)
        self.register_buffer("modal_consistency_J", modal_consistency_J)
        self.register_buffer(
            "modal_consistency_gaussian_indices",
            modal_consistency_gaussian_indices.long(),
        )
        self.register_buffer(
            "modal_consistency_mode_indices",
            modal_consistency_mode_indices.long(),
        )
        self.register_buffer(
            "modal_consistency_group_indices",
            modal_consistency_group_indices.long(),
        )
        self.register_buffer(
            "modal_consistency_group_count",
            torch.tensor(
                int(modal_consistency_group_count),
                device=self.fg.params["means"].device,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "modal_consistency_target_view_index",
            torch.tensor(
                int(modal_consistency_target_view_index),
                device=self.fg.params["means"].device,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "modal_consistency_fps",
            torch.tensor(
                float(modal_consistency_fps),
                device=self.fg.params["means"].device,
                dtype=self.fg.params["means"].dtype,
            ),
        )

    @property
    def num_gaussians(self) -> int:
        return self.num_bg_gaussians + self.num_fg_gaussians

    @property
    def num_bg_gaussians(self) -> int:
        return self.bg.num_gaussians if self.bg is not None else 0

    @property
    def num_fg_gaussians(self) -> int:
        return self.fg.num_gaussians

    @property
    def num_motion_bases(self) -> int:
        return self.motion_bases.num_bases

    @property
    def has_bg(self) -> bool:
        return self.bg is not None

    @property
    def has_modal(self) -> bool:
        return self.modal is not None

    @property
    def has_modal_field(self) -> bool:
        return self.modal_phi_real.numel() > 0 and self.modal_phi_imag.numel() > 0

    @property
    def has_modal_consistency(self) -> bool:
        return int(self.modal_consistency_group_count.item()) > 0

    @torch.no_grad()
    def set_modal_fields(
        self,
        modal_phi_real: Tensor,
        modal_phi_imag: Tensor,
        modal_freqs_hz: Tensor,
    ):
        if self.modal is None:
            raise RuntimeError("set_modal_fields requires modal activations")
        device = self.fg.params["means"].device
        dtype = self.fg.params["means"].dtype
        modal_phi_real = modal_phi_real.to(device=device, dtype=dtype)
        modal_phi_imag = modal_phi_imag.to(device=device, dtype=dtype)
        modal_freqs_hz = modal_freqs_hz.to(device=device, dtype=dtype)
        if modal_phi_real.shape != modal_phi_imag.shape:
            raise ValueError("modal phi real/imag tensors must have matching shapes")
        expected_shape = (self.modal.num_modes, self.num_fg_gaussians, 3)
        if tuple(modal_phi_real.shape) != expected_shape:
            raise ValueError(
                f"modal phi tensors must have shape {expected_shape}, "
                f"got {tuple(modal_phi_real.shape)}"
            )
        if tuple(modal_freqs_hz.shape) != (self.modal.num_modes,):
            raise ValueError(
                f"modal_freqs_hz must have shape {(self.modal.num_modes,)}, "
                f"got {tuple(modal_freqs_hz.shape)}"
            )
        self.modal_phi_real = modal_phi_real
        self.modal_phi_imag = modal_phi_imag
        self.modal_freqs_hz = modal_freqs_hz

    @torch.no_grad()
    def set_modal_consistency_data(
        self,
        y_real: Tensor | None = None,
        y_imag: Tensor | None = None,
        J: Tensor | None = None,
        gaussian_indices: Tensor | None = None,
        mode_indices: Tensor | None = None,
        group_indices: Tensor | None = None,
        group_count: int = 0,
        target_view_index: int | None = None,
        fps: float | None = None,
    ):
        device = self.fg.params["means"].device
        dtype = self.fg.params["means"].dtype
        if y_real is None:
            y_real = torch.empty(0, 2, device=device, dtype=dtype)
        if y_imag is None:
            y_imag = torch.empty_like(y_real)
        if J is None:
            J = torch.empty(0, 2, 3, device=device, dtype=dtype)
        if gaussian_indices is None:
            gaussian_indices = torch.empty(0, device=device, dtype=torch.long)
        if mode_indices is None:
            mode_indices = torch.empty(0, device=device, dtype=torch.long)
        if group_indices is None:
            group_indices = torch.empty(0, device=device, dtype=torch.long)

        self.modal_consistency_y_real = y_real.to(device=device, dtype=dtype)
        self.modal_consistency_y_imag = y_imag.to(device=device, dtype=dtype)
        self.modal_consistency_J = J.to(device=device, dtype=dtype)
        self.modal_consistency_gaussian_indices = gaussian_indices.to(
            device=device, dtype=torch.long
        )
        self.modal_consistency_mode_indices = mode_indices.to(
            device=device, dtype=torch.long
        )
        self.modal_consistency_group_indices = group_indices.to(
            device=device, dtype=torch.long
        )
        self.modal_consistency_group_count = torch.tensor(
            int(group_count), device=device, dtype=torch.long
        )
        if target_view_index is not None:
            self.modal_consistency_target_view_index = torch.tensor(
                int(target_view_index), device=device, dtype=torch.long
            )
        if fps is not None:
            self.modal_consistency_fps = torch.tensor(
                float(fps), device=device, dtype=dtype
            )

    def compute_poses_bg(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            means: (G, B, 3)
            quats: (G, B, 4)
        """
        assert self.bg is not None
        return self.bg.params["means"], self.bg.get_quats()

    def compute_transforms(
        self, ts: torch.Tensor, inds: torch.Tensor | None = None
    ) -> torch.Tensor: # (G, B(len(ts)), 3, 4)
        if self.trajectory_type != "som_basis":
            raise RuntimeError("compute_transforms is only valid for som_basis")
        coefs = self.fg.get_coefs()  # (G, K), get_coef() softmax
        if inds is not None:
            coefs = coefs[inds]
        transfms = self.motion_bases.compute_transforms(ts, coefs)  # (G, B, 3, 4)
        return transfms

    def compute_dct_offsets(
        self, ts: torch.Tensor, inds: torch.Tensor | None = None
    ) -> torch.Tensor:
        traj_coefs = self.fg.params["traj_coefs"]
        if inds is not None:
            traj_coefs = traj_coefs[inds]
        basis = self.dct_basis[ts].to(dtype=traj_coefs.dtype, device=traj_coefs.device)
        return torch.einsum("bk,gkc->gbc", basis, traj_coefs)

    def compute_modal_offsets(
        self, ts: torch.Tensor, inds: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.modal is None:
            raise RuntimeError("compute_modal_offsets requires modal activations")
        activations = self.modal.params["activations"][ts]
        phi_real = self.modal_phi_real
        phi_imag = self.modal_phi_imag
        if inds is not None:
            phi_real = phi_real[:, inds]
            phi_imag = phi_imag[:, inds]
        real = activations[..., 0]
        imag = activations[..., 1]
        return torch.einsum("bk,kgc->gbc", real, phi_real) - torch.einsum(
            "bk,kgc->gbc", imag, phi_imag
        )

    def compute_synthetic_modal_offsets(
        self,
        q: torch.Tensor,
        motion_scale: float | torch.Tensor = 1.0,
        inds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.has_modal_field:
            raise RuntimeError("synthetic modal playback requires modal phi fields")
        if q.ndim != 1 or q.shape[0] != self.modal_phi_real.shape[0]:
            raise ValueError(
                f"q must have shape ({self.modal_phi_real.shape[0]},), got {tuple(q.shape)}"
            )
        phi_real = self.modal_phi_real
        phi_imag = self.modal_phi_imag
        if inds is not None:
            phi_real = phi_real[:, inds]
            phi_imag = phi_imag[:, inds]
        q = q.to(device=phi_real.device)
        q_real = q.real.to(dtype=phi_real.dtype)
        q_imag = q.imag.to(dtype=phi_real.dtype)
        offsets = torch.einsum("k,kgc->gc", q_real, phi_real) - torch.einsum(
            "k,kgc->gc", q_imag, phi_imag
        )
        return offsets * torch.as_tensor(
            motion_scale, device=offsets.device, dtype=offsets.dtype
        )

    def compute_activation_smoothness_loss(self) -> torch.Tensor:
        if self.modal is None:
            return self.fg.params["means"].new_zeros(())
        if self.modal_smooth_triplets.numel() == 0:
            return self.modal.params["activations"].sum() * 0.0
        activations = self.modal.params["activations"]
        prev = activations[self.modal_smooth_triplets[:, 0]]
        center = activations[self.modal_smooth_triplets[:, 1]]
        nxt = activations[self.modal_smooth_triplets[:, 2]]
        accel = nxt - 2.0 * center + prev
        return accel.pow(2).sum(dim=-1).mean()

    def compute_activation_magnitude_loss(self) -> torch.Tensor:
        if self.modal is None:
            return self.fg.params["means"].new_zeros(())
        return self.modal.params["activations"].pow(2).sum(dim=-1).mean()

    def compute_activation_modal_consistency_loss(
        self,
        loss_type: str = "aligned_l2",
        beta_abs_max: float = 10.0,
        pred_energy_eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.modal is None or not self.has_modal_consistency:
            zero = self.fg.params["means"].new_zeros(())
            return zero, zero
        if loss_type not in {"corr", "aligned_l2"}:
            raise ValueError(
                f"Unknown modal consistency loss type {loss_type!r}"
            )
        if beta_abs_max <= 0:
            raise ValueError("modal consistency beta_abs_max must be positive")
        if pred_energy_eps <= 0:
            raise ValueError("modal consistency pred_energy_eps must be positive")
        if float(self.modal_consistency_fps.item()) <= 0:
            raise ValueError("modal consistency requires positive fps")

        target_view_index = int(self.modal_consistency_target_view_index.item())
        target_mask = self.modal_frame_view_indices == target_view_index
        if int(target_mask.sum().item()) == 0:
            raise ValueError(
                f"modal consistency target view index {target_view_index} has no frames"
            )
        target_ts = torch.where(target_mask)[0]
        local_indices = self.modal_frame_local_indices[target_ts]
        order = torch.argsort(local_indices)
        target_ts = target_ts[order]
        local_indices = local_indices[order]

        activations = self.modal.params["activations"][target_ts]
        act_complex = torch.complex(activations[..., 0], activations[..., 1])
        dtype = activations.dtype
        times = local_indices.to(device=activations.device, dtype=dtype) / self.modal_consistency_fps
        freqs = self.modal_freqs_hz.to(device=activations.device, dtype=dtype)
        phase_arg = -2.0 * torch.pi * freqs[:, None] * times[None, :]
        phase = torch.complex(torch.cos(phase_arg), torch.sin(phase_arg))
        activation_spectrum = torch.einsum("ft,tm->fm", phase, act_complex)
        activation_spectrum = activation_spectrum / max(int(target_ts.numel()), 1)

        phi = torch.complex(self.modal_phi_real, self.modal_phi_imag)
        xhat = torch.einsum("fm,mgc->fgc", activation_spectrum, phi)
        obs_xhat = xhat[
            self.modal_consistency_mode_indices,
            self.modal_consistency_gaussian_indices,
        ]
        J = self.modal_consistency_J.to(dtype=obs_xhat.dtype)
        pred_y = torch.einsum("oij,oj->oi", J, obs_xhat)
        target_y = torch.complex(
            self.modal_consistency_y_real,
            self.modal_consistency_y_imag,
        )

        eps = torch.as_tensor(
            float(pred_energy_eps), device=activations.device, dtype=dtype
        )
        losses = []
        for group_idx in range(int(self.modal_consistency_group_count.item())):
            rows = self.modal_consistency_group_indices == group_idx
            if int(rows.sum().item()) == 0:
                continue
            pred_group = pred_y[rows].reshape(-1)
            target_group = target_y[rows].reshape(-1)
            target_energy = (target_group.conj() * target_group).real.sum()
            if float(target_energy.detach().item()) <= float(eps):
                continue
            pred_energy = (pred_group.conj() * pred_group).real.sum()
            dot = (pred_group.conj() * target_group).sum()
            if loss_type == "corr":
                denom = (pred_energy + eps) * (target_energy + eps)
                corr = dot.abs().pow(2) / denom
                losses.append(1.0 - corr.clamp(0.0, 1.0))
            else:
                if float(pred_energy.detach().item()) <= float(eps):
                    beta = torch.ones((), device=pred_y.device, dtype=pred_y.dtype)
                else:
                    beta = dot / (pred_energy + eps)
                    beta_abs = beta.abs()
                    max_abs = torch.as_tensor(
                        float(beta_abs_max),
                        device=pred_y.device,
                        dtype=beta_abs.dtype,
                    )
                    scale = torch.clamp(max_abs / beta_abs.clamp_min(eps), max=1.0)
                    beta = beta * scale.to(dtype=beta.dtype)
                residual = beta * pred_group - target_group
                residual_energy = (residual.conj() * residual).real.sum()
                losses.append(residual_energy / (target_energy + eps))

        if not losses:
            zero = self.fg.params["means"].new_zeros(())
            return zero, zero
        return torch.stack(losses).mean(), torch.tensor(
            float(len(losses)), device=activations.device, dtype=dtype
        )

    @torch.no_grad()
    def densify_modal_fields(self, should_split: torch.Tensor, should_dup: torch.Tensor):
        if not self.has_modal:
            return
        for name in ("modal_phi_real", "modal_phi_imag"):
            x = getattr(self, name)
            x_dup = x[:, should_dup]
            x_split = x[:, should_split].repeat(1, 2, 1)
            setattr(self, name, torch.cat([x[:, ~should_split], x_dup, x_split], dim=1))

    @torch.no_grad()
    def cull_modal_fields(self, should_cull: torch.Tensor):
        if not self.has_modal:
            return
        self.modal_phi_real = self.modal_phi_real[:, ~should_cull]
        self.modal_phi_imag = self.modal_phi_imag[:, ~should_cull]

    def compute_poses_fg(
        self, ts: torch.Tensor | None, inds: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :returns means: (G, B, 3), quats: (G, B, 4)
        """
        means = self.fg.params["means"]  # (G, 3)
        quats = self.fg.get_quats()  # (G, 4)
        if inds is not None:
            means = means[inds]
            quats = quats[inds]
        if ts is not None:
            if self.trajectory_type == "static":
                means = means[:, None].expand(-1, ts.shape[0], -1)
                quats = quats[:, None].expand(-1, ts.shape[0], -1)
            elif self.trajectory_type == "dct_center":
                means = means[:, None] + self.compute_dct_offsets(ts, inds)
                quats = quats[:, None].expand(-1, ts.shape[0], -1)
            elif self.trajectory_type == "modal_activation":
                means = means[:, None] + self.compute_modal_offsets(ts, inds)
                quats = quats[:, None].expand(-1, ts.shape[0], -1)
            else:
                transfms = self.compute_transforms(ts, inds)  # (G, B, 3, 4)
                means = torch.einsum(
                    "pnij,pj->pni",
                    transfms,
                    F.pad(means, (0, 1), value=1.0),
                )
                quats = roma.quat_xyzw_to_wxyz(
                    (
                        roma.quat_product(
                            roma.rotmat_to_unitquat(transfms[..., :3, :3]),
                            roma.quat_wxyz_to_xyzw(quats[:, None]),
                        )
                    )
                )
                quats = F.normalize(quats, p=2, dim=-1)
        else:
            means = means[:, None]
            quats = quats[:, None]
        return means, quats

    def compute_poses_all(
        self, ts: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means, quats = self.compute_poses_fg(ts)
        if self.has_bg:
            bg_means, bg_quats = self.compute_poses_bg()
            means = torch.cat(
                [means, bg_means[:, None].expand(-1, means.shape[1], -1)], dim=0
            ).contiguous()
            quats = torch.cat(
                [quats, bg_quats[:, None].expand(-1, means.shape[1], -1)], dim=0
            ).contiguous()
        return means, quats

    def get_colors_all(self) -> torch.Tensor:
        colors = self.fg.get_colors()
        if self.bg is not None:
            colors = torch.cat([colors, self.bg.get_colors()], dim=0).contiguous()
        return colors

    def get_scales_all(self) -> torch.Tensor:
        scales = self.fg.get_scales()
        if self.bg is not None:
            scales = torch.cat([scales, self.bg.get_scales()], dim=0).contiguous()
        return scales

    def get_opacities_all(self) -> torch.Tensor:
        """
        :returns colors: (G, 3), scales: (G, 3), opacities: (G, 1)
        """
        opacities = self.fg.get_opacities()
        if self.bg is not None:
            opacities = torch.cat(
                [opacities, self.bg.get_opacities()], dim=0
            ).contiguous()
        return opacities

    @staticmethod
    def init_from_state_dict(state_dict, prefix=""):
        fg = GaussianParams.init_from_state_dict(
            state_dict, prefix=f"{prefix}fg.params."
        )
        bg = None
        if any("bg." in k for k in state_dict):
            bg = GaussianParams.init_from_state_dict(
                state_dict, prefix=f"{prefix}bg.params."
            )
        motion_bases = MotionBases.init_from_state_dict(
            state_dict, prefix=f"{prefix}motion_bases.params."
        )
        Ks = state_dict[f"{prefix}Ks"]
        w2cs = state_dict[f"{prefix}w2cs"]
        camera_poses = None
        if any("camera_poses." in k for k in state_dict):
            camera_poses = CameraPoses.init_from_state_dict(
                state_dict, prefix=f"{prefix}camera_poses.params."
            )

        if f"{prefix}trajectory_type_id" in state_dict:
            trajectory_type_id = int(state_dict[f"{prefix}trajectory_type_id"].item())
            if trajectory_type_id not in TRAJECTORY_ID_TO_TYPE:
                raise ValueError(f"Unknown trajectory type id: {trajectory_type_id}")
            trajectory_type = TRAJECTORY_ID_TO_TYPE[trajectory_type_id]
        else:
            if f"{prefix}modal.params.activations" in state_dict:
                trajectory_type = "modal_activation"
            elif f"{prefix}fg.params.traj_coefs" in state_dict:
                trajectory_type = "dct_center"
            else:
                trajectory_type = "som_basis"
        cano_t = None
        if f"{prefix}cano_t" in state_dict:
            cano_t_tensor = state_dict[f"{prefix}cano_t"]
            cano_t = int(cano_t_tensor.item()) if cano_t_tensor.item() >= 0 else None
        num_dct_bases = None
        if f"{prefix}fg.params.traj_coefs" in state_dict:
            num_dct_bases = state_dict[f"{prefix}fg.params.traj_coefs"].shape[1]
        modal = None
        modal_phi_real = None
        modal_phi_imag = None
        modal_freqs_hz = None
        modal_frame_view_indices = None
        modal_frame_local_indices = None
        modal_smooth_triplets = None
        modal_consistency_y_real = None
        modal_consistency_y_imag = None
        modal_consistency_J = None
        modal_consistency_gaussian_indices = None
        modal_consistency_mode_indices = None
        modal_consistency_group_indices = None
        modal_consistency_group_count = 0
        modal_consistency_target_view_index = -1
        modal_consistency_fps = 0.0
        if f"{prefix}modal_phi_real" in state_dict:
            modal_phi_real = state_dict[f"{prefix}modal_phi_real"]
            modal_phi_imag = state_dict[f"{prefix}modal_phi_imag"]
            modal_freqs_hz = state_dict[f"{prefix}modal_freqs_hz"]

        if trajectory_type == "modal_activation":
            modal = ModalActivations.init_from_state_dict(
                state_dict, prefix=f"{prefix}modal.params."
            )
            modal_frame_view_indices = state_dict[f"{prefix}modal_frame_view_indices"]
            modal_frame_local_indices = state_dict[f"{prefix}modal_frame_local_indices"]
            modal_smooth_triplets = state_dict[f"{prefix}modal_smooth_triplets"]
            if f"{prefix}modal_consistency_y_real" in state_dict:
                modal_consistency_y_real = state_dict[f"{prefix}modal_consistency_y_real"]
                modal_consistency_y_imag = state_dict[f"{prefix}modal_consistency_y_imag"]
                modal_consistency_J = state_dict[f"{prefix}modal_consistency_J"]
                modal_consistency_gaussian_indices = state_dict[
                    f"{prefix}modal_consistency_gaussian_indices"
                ]
                modal_consistency_mode_indices = state_dict[
                    f"{prefix}modal_consistency_mode_indices"
                ]
                modal_consistency_group_indices = state_dict[
                    f"{prefix}modal_consistency_group_indices"
                ]
                modal_consistency_group_count = int(
                    state_dict[f"{prefix}modal_consistency_group_count"].item()
                )
                modal_consistency_target_view_index = int(
                    state_dict[f"{prefix}modal_consistency_target_view_index"].item()
                )
                modal_consistency_fps = float(
                    state_dict[f"{prefix}modal_consistency_fps"].item()
                )
        modal_synthetic_enabled = state_dict.get(
            f"{prefix}modal_synthetic_enabled",
            torch.tensor(False),
        )

        return SceneModel(
            Ks, 
            w2cs, 
            fg, 
            motion_bases, 
            camera_poses,
            bg,
            trajectory_type=trajectory_type,
            cano_t=cano_t,
            num_dct_bases=num_dct_bases,
            modal=modal,
            modal_phi_real=modal_phi_real,
            modal_phi_imag=modal_phi_imag,
            modal_freqs_hz=modal_freqs_hz,
            modal_frame_view_indices=modal_frame_view_indices,
            modal_frame_local_indices=modal_frame_local_indices,
            modal_smooth_triplets=modal_smooth_triplets,
            modal_consistency_y_real=modal_consistency_y_real,
            modal_consistency_y_imag=modal_consistency_y_imag,
            modal_consistency_J=modal_consistency_J,
            modal_consistency_gaussian_indices=modal_consistency_gaussian_indices,
            modal_consistency_mode_indices=modal_consistency_mode_indices,
            modal_consistency_group_indices=modal_consistency_group_indices,
            modal_consistency_group_count=modal_consistency_group_count,
            modal_consistency_target_view_index=modal_consistency_target_view_index,
            modal_consistency_fps=modal_consistency_fps,
            modal_synthetic_enabled=modal_synthetic_enabled,
        )

    def render(
        self,
        # A single time instance for view rendering.
        t: int | None,
        w2cs: torch.Tensor,  # (C, 4, 4)
        Ks: torch.Tensor,  # (C, 3, 3)
        img_wh: tuple[int, int],
        # Multiple time instances for track rendering: (B,).
        target_ts: torch.Tensor | None = None,  # (B)
        target_w2cs: torch.Tensor | None = None,  # (B, 4, 4)
        bg_color: torch.Tensor | float = 1.0,
        colors_override: torch.Tensor | None = None,
        means: torch.Tensor | None = None,
        quats: torch.Tensor | None = None,
        target_means: torch.Tensor | None = None,
        return_color: bool = True,
        return_depth: bool = False,
        return_mask: bool = False,
        fg_only: bool = False,
        filter_mask: torch.Tensor | None = None,
    ) -> dict:

        curr_w2cs = w2cs
        target_w2cs_clone = None
        if target_w2cs is not None:
            target_w2cs_clone = target_w2cs
        device = w2cs.device
        C = w2cs.shape[0]

        W, H = img_wh
        pose_fnc = self.compute_poses_fg if fg_only else self.compute_poses_all
        N = self.num_fg_gaussians if fg_only else self.num_gaussians

        if means is None or quats is None:
            means, quats = pose_fnc(
                torch.tensor([t], device=device) if t is not None else None
            )
            means = means[:, 0]
            quats = quats[:, 0]

        if colors_override is None:
            if return_color:
                colors_override = (
                    self.fg.get_colors() if fg_only else self.get_colors_all()
                )
            else:
                colors_override = torch.zeros(N, 0, device=device)

        D = colors_override.shape[-1]

        scales = self.fg.get_scales() if fg_only else self.get_scales_all()
        opacities = self.fg.get_opacities() if fg_only else self.get_opacities_all()

        if isinstance(bg_color, float):
            bg_color = torch.full((C, D), bg_color, device=device)
        assert isinstance(bg_color, torch.Tensor)

        mode = "RGB"
        ds_expected = {"img": D}

        if return_mask:
            if self.has_bg and not fg_only:
                mask_values = torch.zeros((self.num_gaussians, 1), device=device)
                mask_values[: self.num_fg_gaussians] = 1.0
            else:
                mask_values = torch.ones((self.num_fg_gaussians, 1), device=device)
            colors_override = torch.cat([colors_override, mask_values], dim=-1)
            bg_color = torch.cat([bg_color, torch.zeros(C, 1, device=device)], dim=-1)
            ds_expected["mask"] = 1

        B = 0
        if target_ts is not None:
            B = target_ts.shape[0]
            if target_means is None:
                target_means, _ = pose_fnc(target_ts)  # [G, B, 3]
            if target_w2cs_clone is not None:
                target_means = torch.einsum(
                    "bij,pbj->pbi",
                    target_w2cs_clone[:, :3],
                    F.pad(target_means, (0, 1), value=1.0),
                )
            track_3d_vals = target_means.flatten(-2)  # (G, B * 3)
            d_track = track_3d_vals.shape[-1]
            colors_override = torch.cat([colors_override, track_3d_vals], dim=-1)
            bg_color = torch.cat(
                [bg_color, torch.zeros(C, track_3d_vals.shape[-1], device=device)],
                dim=-1,
            )
            ds_expected["tracks_3d"] = d_track

        assert colors_override.shape[-1] == sum(ds_expected.values())
        assert bg_color.shape[-1] == sum(ds_expected.values())

        if return_depth:
            mode = "RGB+ED"
            ds_expected["depth"] = 1

        if filter_mask is not None:
            assert filter_mask.shape == (N,)
            means = means[filter_mask]
            quats = quats[filter_mask]
            scales = scales[filter_mask]
            opacities = opacities[filter_mask]
            colors_override = colors_override[filter_mask]

        if self.camera_poses is not None:
            w2cs = self.camera_poses.get_camera_matrix()
            w2cs = w2cs[t].unsqueeze(0)

        if self.use_2dgs:
            colors_override = torch.nan_to_num(colors_override, nan=1e-6)
            backgrounds = torch.nan_to_num(bg_color, nan=1.0)

            outputs = rasterization_2dgs(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors_override,
                backgrounds=bg_color,
                viewmats=curr_w2cs,  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=W,
                height=H,
                packed=False,
                render_mode=mode,
            )

            (
                render_colors,
                alphas,
                render_normals,
                surf_normals,
                _,
                _,
                info,
            ) = outputs
        
        else:
            render_colors, alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors_override,
                backgrounds=bg_color,
                viewmats=curr_w2cs,  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=W,
                height=H,
                packed=False,
                render_mode=mode,
            )
            render_normals = None
            surf_normals = None

        # Populate the current data for adaptive gaussian control.
        if self.training and info["means2d"].requires_grad:
            self._current_xys = info["means2d"]
            radii = info["radii"]
            # Newer gsplat returns per-axis radii with shape (..., G, 2).
            # SOM's density-control code expects one scalar radius per Gaussian.
            if radii.ndim == self._current_xys.ndim and radii.shape[-1] == 2:
                radii = radii.amax(dim=-1)
            self._current_radii = radii
            self._current_img_wh = img_wh
            # We want to be able to access to xys' gradients later in a
            # torch.no_grad context.
            self._current_xys.retain_grad()

        assert render_colors.shape[-1] == sum(ds_expected.values())
        outputs = torch.split(render_colors, list(ds_expected.values()), dim=-1)
        out_dict = {}
        for i, (name, dim) in enumerate(ds_expected.items()):
            x = outputs[i]
            assert x.shape[-1] == dim, f"{x.shape[-1]=} != {dim=}"
            if name == "tracks_3d":
                x = x.reshape(C, H, W, B, 3)
            out_dict[name] = x
        out_dict["acc"] = alphas
        out_dict["rend_normal"] = render_normals
        out_dict["surf_normal"] = surf_normals
        return out_dict
