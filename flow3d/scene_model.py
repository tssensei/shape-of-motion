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
    ModalShapeRefinement,
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
        modal_obs_count_per_point: Tensor | None = None,
        modal_frame_view_indices: Tensor | None = None,
        modal_frame_local_indices: Tensor | None = None,
        modal_frame_times_sec: Tensor | None = None,
        modal_synthetic_enabled: bool | Tensor = False,
        modal_refinement: ModalShapeRefinement | None = None,
        modal_anchor_mask: Tensor | None = None,
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
        self.modal_refinement = modal_refinement
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
            if modal_phi_real.shape[1] != self.num_fg_gaussians:
                raise ValueError("modal phi Gaussian dimension does not match foreground")
            if modal_phi_real.shape[0] != modal.num_modes:
                raise ValueError("modal phi mode count does not match activations")
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

        if not torch.is_floating_point(modal_phi_real) or not torch.is_floating_point(
            modal_phi_imag
        ):
            raise ValueError("modal phi real/imag tensors must have floating-point dtype")
        if modal_phi_real.dtype != modal_phi_imag.dtype:
            raise ValueError("modal phi real/imag tensors must have matching dtypes")
        if not bool(torch.isfinite(modal_phi_real).all().item()) or not bool(
            torch.isfinite(modal_phi_imag).all().item()
        ):
            raise ValueError("modal phi real/imag tensors must contain only finite values")

        if modal_refinement is not None:
            if trajectory_type != "modal_activation":
                raise ValueError(
                    "modal shape refinement requires modal_activation trajectory"
                )
            delta_phi = modal_refinement.params["delta_phi"]
            expected_delta_shape = (*modal_phi_real.shape, 2)
            if delta_phi.shape != expected_delta_shape:
                raise ValueError(
                    "modal delta_phi must have shape "
                    f"{expected_delta_shape}, got {tuple(delta_phi.shape)}"
                )
            if delta_phi.dtype != modal_phi_real.dtype:
                raise ValueError("modal delta_phi dtype must match staged modal phi")
            if delta_phi.device != modal_phi_real.device:
                raise ValueError("modal delta_phi device must match staged modal phi")
            if modal_anchor_mask is None:
                raise ValueError("modal shape refinement requires modal_anchor_mask")
            if modal_anchor_mask.dtype != torch.bool:
                raise ValueError("modal_anchor_mask must have bool dtype")
            if modal_anchor_mask.shape != modal_phi_real.shape[:2]:
                raise ValueError(
                    "modal_anchor_mask must have shape "
                    f"{tuple(modal_phi_real.shape[:2])}, "
                    f"got {tuple(modal_anchor_mask.shape)}"
                )
            if modal_anchor_mask.device != delta_phi.device:
                raise ValueError("modal_anchor_mask device must match modal delta_phi")
            if not bool(modal_anchor_mask.any(dim=1).all().item()):
                raise ValueError(
                    "modal_anchor_mask must contain at least one anchor per mode"
                )
            if bool(torch.count_nonzero(delta_phi[~modal_anchor_mask]).item()):
                raise ValueError("non-anchor modal delta_phi values must be exactly zero")
        elif modal_anchor_mask is not None:
            raise ValueError(
                "modal_anchor_mask cannot be provided without modal shape refinement"
            )

        if modal_freqs_hz is None:
            if modal_phi_real.shape[0] > 0:
                raise ValueError("modal fields require modal freqs_hz")
            modal_freqs_hz = torch.empty(
                modal_phi_real.shape[0],
                device=self.fg.params["means"].device,
                dtype=self.fg.params["means"].dtype,
            )
        if modal_freqs_hz.ndim != 1 or modal_freqs_hz.shape[0] != modal_phi_real.shape[0]:
            raise ValueError(
                "modal freqs_hz must have shape "
                f"({modal_phi_real.shape[0]},), got {tuple(modal_freqs_hz.shape)}"
            )
        if not torch.is_floating_point(modal_freqs_hz):
            raise ValueError("modal freqs_hz must have floating-point dtype")
        if not bool(torch.isfinite(modal_freqs_hz).all().item()):
            raise ValueError("modal freqs_hz must contain only finite values")
        if modal_freqs_hz.numel() > 0 and bool((modal_freqs_hz <= 0).any().item()):
            raise ValueError("modal freqs_hz must be strictly positive")
        if modal_obs_count_per_point is None:
            modal_obs_count_per_point = torch.empty(
                modal_phi_real.shape[0],
                self.num_fg_gaussians,
                device=self.fg.params["means"].device,
                dtype=torch.long,
            )
        if modal_obs_count_per_point.shape != (
            modal_phi_real.shape[0],
            self.num_fg_gaussians,
        ):
            raise ValueError(
                "modal obs_count_per_point must have shape "
                f"({modal_phi_real.shape[0]}, {self.num_fg_gaussians}), "
                f"got {tuple(modal_obs_count_per_point.shape)}"
            )
        if trajectory_type == "modal_activation" and (
            modal_frame_view_indices is None
            or modal_frame_local_indices is None
            or modal_frame_times_sec is None
        ):
            raise ValueError(
                "modal_activation requires frame view, local-index, and time buffers"
            )
        frame_device = self.fg.params["means"].device
        if modal_frame_view_indices is None:
            modal_frame_view_indices = torch.full(
                (self.num_frames,), -1, device=frame_device, dtype=torch.long
            )
        if modal_frame_local_indices is None:
            modal_frame_local_indices = torch.full(
                (self.num_frames,), -1, device=frame_device, dtype=torch.long
            )
        if modal_frame_times_sec is None:
            modal_frame_times_sec = torch.full(
                (self.num_frames,),
                -1.0,
                device=frame_device,
                dtype=self.fg.params["means"].dtype,
            )
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if modal_frame_view_indices.dtype not in integer_dtypes:
            raise ValueError("modal frame view indices must have integer dtype")
        if modal_frame_local_indices.dtype not in integer_dtypes:
            raise ValueError("modal frame local indices must have integer dtype")
        if not torch.is_floating_point(modal_frame_times_sec):
            raise ValueError("modal frame times_sec must have floating-point dtype")
        for name, values in (
            ("modal frame view indices", modal_frame_view_indices),
            ("modal frame local indices", modal_frame_local_indices),
            ("modal frame times_sec", modal_frame_times_sec),
        ):
            if values.ndim != 1 or values.shape[0] != self.num_frames:
                raise ValueError(
                    f"{name} must have shape ({self.num_frames},), "
                    f"got {tuple(values.shape)}"
                )
        if trajectory_type == "modal_activation":
            if modal is None:
                raise ValueError("modal_activation requires modal activations")
            modal_num_views = modal.num_views
            if bool((modal_frame_view_indices < 0).any().item()) or bool(
                (modal_frame_view_indices >= modal_num_views).any().item()
            ):
                raise ValueError(
                    f"modal frame view indices must lie in [0, {modal_num_views})"
                )
            if bool((modal_frame_local_indices < 0).any().item()):
                raise ValueError("modal frame local indices must be non-negative")
            if not bool(torch.isfinite(modal_frame_times_sec).all().item()) or bool(
                (modal_frame_times_sec < 0).any().item()
            ):
                raise ValueError(
                    "modal frame times_sec must be finite and non-negative"
                )
        self.register_buffer("modal_phi_real", modal_phi_real)
        self.register_buffer("modal_phi_imag", modal_phi_imag)
        self.register_buffer("modal_anchor_mask", modal_anchor_mask)
        self.register_buffer("modal_freqs_hz", modal_freqs_hz)
        self.register_buffer("modal_obs_count_per_point", modal_obs_count_per_point.long())
        if isinstance(modal_synthetic_enabled, Tensor):
            modal_synthetic_enabled = bool(modal_synthetic_enabled.item())
        self.register_buffer(
            "modal_synthetic_enabled",
            torch.tensor(bool(modal_synthetic_enabled), dtype=torch.bool),
        )
        self.register_buffer(
            "modal_frame_view_indices",
            modal_frame_view_indices.to(device=frame_device, dtype=torch.long),
        )
        self.register_buffer(
            "modal_frame_local_indices",
            modal_frame_local_indices.to(device=frame_device, dtype=torch.long),
        )
        self.register_buffer(
            "modal_frame_times_sec",
            modal_frame_times_sec.to(
                device=frame_device, dtype=self.fg.params["means"].dtype
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
    def has_modal_refinement(self) -> bool:
        return self.modal_refinement is not None

    @property
    def has_modal_obs_count(self) -> bool:
        return self.modal_obs_count_per_point.numel() > 0

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

    def compute_modal_coefficients(
        self, ts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.modal is None:
            raise RuntimeError("compute_modal_coefficients requires modal activations")
        if ts.ndim != 1:
            raise ValueError(f"ts must be a 1-D tensor, got shape {tuple(ts.shape)}")
        if ts.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("ts must have integer dtype")
        if ts.numel() > 0 and (
            bool((ts < 0).any().item())
            or bool((ts >= self.num_frames).any().item())
        ):
            raise ValueError(f"ts values must lie in [0, {self.num_frames})")
        ts = ts.to(dtype=torch.long)

        view_indices = self.modal_frame_view_indices[ts]
        amplitudes = self.modal.params["activations"][view_indices]
        times_sec = self.modal_frame_times_sec[ts].to(dtype=amplitudes.dtype)
        freqs_hz = self.modal_freqs_hz.to(dtype=amplitudes.dtype)
        theta = 2.0 * torch.pi * times_sec[:, None] * freqs_hz[None, :]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        amplitude_real = amplitudes[..., 0]
        amplitude_imag = amplitudes[..., 1]
        coefficient_real = (
            amplitude_real * cos_theta - amplitude_imag * sin_theta
        )
        coefficient_imag = (
            amplitude_real * sin_theta + amplitude_imag * cos_theta
        )
        return coefficient_real, coefficient_imag

    def compute_modal_offsets(
        self, ts: torch.Tensor, inds: torch.Tensor | None = None
    ) -> torch.Tensor:
        real, imag = self.compute_modal_coefficients(ts)
        phi_real, phi_imag = self.get_effective_modal_phi(inds)
        return torch.einsum("bk,kgc->gbc", real, phi_real) - torch.einsum(
            "bk,kgc->gbc", imag, phi_imag
        )

    def get_effective_modal_phi(
        self, inds: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phi_real = self.modal_phi_real
        phi_imag = self.modal_phi_imag
        if self.modal_refinement is not None:
            delta_phi = self.modal_refinement.params["delta_phi"]
            anchor_mask = self.modal_anchor_mask
            if anchor_mask is None:
                raise RuntimeError(
                    "modal shape refinement is missing modal_anchor_mask"
                )
            if inds is not None:
                delta_phi = delta_phi[:, inds]
                anchor_mask = anchor_mask[:, inds]
            mask = anchor_mask[..., None].to(dtype=delta_phi.dtype)
            phi_real = phi_real if inds is None else phi_real[:, inds]
            phi_imag = phi_imag if inds is None else phi_imag[:, inds]
            phi_real = phi_real + mask * delta_phi[..., 0]
            phi_imag = phi_imag + mask * delta_phi[..., 1]
            return phi_real, phi_imag
        if inds is not None:
            phi_real = phi_real[:, inds]
            phi_imag = phi_imag[:, inds]
        return phi_real, phi_imag

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
        phi_real, phi_imag = self.get_effective_modal_phi(inds)
        q = q.to(device=phi_real.device)
        q_real = q.real.to(dtype=phi_real.dtype)
        q_imag = q.imag.to(dtype=phi_real.dtype)
        offsets = torch.einsum("k,kgc->gc", q_real, phi_real) - torch.einsum(
            "k,kgc->gc", q_imag, phi_imag
        )
        return offsets * torch.as_tensor(
            motion_scale, device=offsets.device, dtype=offsets.dtype
        )

    def compute_activation_magnitude_loss(self) -> torch.Tensor:
        if self.modal is None:
            return self.fg.params["means"].new_zeros(())
        return self.modal.params["activations"].pow(2).sum(dim=-1).mean()

    @torch.no_grad()
    def densify_modal_fields(self, should_split: torch.Tensor, should_dup: torch.Tensor):
        if not self.has_modal_field:
            return
        if self.has_modal_refinement:
            raise RuntimeError("modal shape refinement does not support densification")
        for name in ("modal_phi_real", "modal_phi_imag"):
            x = getattr(self, name)
            x_dup = x[:, should_dup]
            x_split = x[:, should_split].repeat(1, 2, 1)
            setattr(self, name, torch.cat([x[:, ~should_split], x_dup, x_split], dim=1))
        if self.has_modal_obs_count:
            x = self.modal_obs_count_per_point
            x_dup = x[:, should_dup]
            x_split = x[:, should_split].repeat(1, 2)
            self.modal_obs_count_per_point = torch.cat(
                [x[:, ~should_split], x_dup, x_split], dim=1
            )

    @torch.no_grad()
    def cull_modal_fields(self, should_cull: torch.Tensor):
        if not self.has_modal_field:
            return
        if self.has_modal_refinement:
            raise RuntimeError("modal shape refinement does not support culling")
        self.modal_phi_real = self.modal_phi_real[:, ~should_cull]
        self.modal_phi_imag = self.modal_phi_imag[:, ~should_cull]
        if self.has_modal_obs_count:
            self.modal_obs_count_per_point = self.modal_obs_count_per_point[:, ~should_cull]

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
        modal_obs_count_per_point = None
        modal_frame_view_indices = None
        modal_frame_local_indices = None
        modal_frame_times_sec = None
        modal_refinement = None
        modal_anchor_mask = None
        if f"{prefix}modal_phi_real" in state_dict:
            modal_phi_real = state_dict[f"{prefix}modal_phi_real"]
            modal_phi_imag = state_dict[f"{prefix}modal_phi_imag"]
            modal_freqs_hz = state_dict[f"{prefix}modal_freqs_hz"]
            if f"{prefix}modal_obs_count_per_point" in state_dict:
                modal_obs_count_per_point = state_dict[f"{prefix}modal_obs_count_per_point"]

        if trajectory_type == "modal_activation":
            required_frame_keys = (
                f"{prefix}modal_frame_view_indices",
                f"{prefix}modal_frame_local_indices",
                f"{prefix}modal_frame_times_sec",
            )
            missing_frame_keys = [
                key for key in required_frame_keys if key not in state_dict
            ]
            if missing_frame_keys:
                if f"{prefix}modal_smooth_triplets" in state_dict:
                    raise ValueError(
                        "Legacy per-frame modal activation checkpoints are not "
                        "supported; initialize per-view harmonic activation from "
                        "the static checkpoint and staged modal manifest."
                    )
                raise ValueError(
                    "Harmonic modal checkpoint is missing required frame buffers: "
                    f"{missing_frame_keys}"
                )
            modal = ModalActivations.init_from_state_dict(
                state_dict, prefix=f"{prefix}modal.params."
            )
            modal_frame_view_indices = state_dict[f"{prefix}modal_frame_view_indices"]
            modal_frame_local_indices = state_dict[f"{prefix}modal_frame_local_indices"]
            modal_frame_times_sec = state_dict[f"{prefix}modal_frame_times_sec"]
        refinement_key = f"{prefix}modal_refinement.params.delta_phi"
        anchor_mask_key = f"{prefix}modal_anchor_mask"
        if (refinement_key in state_dict) != (anchor_mask_key in state_dict):
            raise ValueError(
                "Modal shape-refinement checkpoint must contain both "
                f"{refinement_key} and {anchor_mask_key}"
            )
        if refinement_key in state_dict:
            modal_refinement = ModalShapeRefinement.init_from_state_dict(
                state_dict, prefix=f"{prefix}modal_refinement.params."
            )
            modal_anchor_mask = state_dict[anchor_mask_key]
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
            modal_obs_count_per_point=modal_obs_count_per_point,
            modal_frame_view_indices=modal_frame_view_indices,
            modal_frame_local_indices=modal_frame_local_indices,
            modal_frame_times_sec=modal_frame_times_sec,
            modal_synthetic_enabled=modal_synthetic_enabled,
            modal_refinement=modal_refinement,
            modal_anchor_mask=modal_anchor_mask,
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
