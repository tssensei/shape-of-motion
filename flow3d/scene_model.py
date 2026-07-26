import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat.rendering import rasterization
from gsplat.rendering import rasterization_2dgs
from torch import Tensor

from flow3d.params import (
    GaussianParams,
    ModalJointParams,
    ModalPhiRefinementParams,
    MotionBases,
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
        modal_coordinate_real: Tensor | None = None,
        modal_coordinate_imag: Tensor | None = None,
        modal_phi_real: Tensor | None = None,
        modal_phi_imag: Tensor | None = None,
        modal_freqs_hz: Tensor | None = None,
        modal_obs_count_per_point: Tensor | None = None,
        modal_frame_view_indices: Tensor | None = None,
        modal_frame_local_indices: Tensor | None = None,
        modal_frame_times_sec: Tensor | None = None,
        modal_synthetic_enabled: bool | Tensor = False,
        modal_joint_params: ModalJointParams | None = None,
        modal_phi_refinement_params: ModalPhiRefinementParams | None = None,
        modal_phi_trainable_mask: Tensor | None = None,
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

        frame_device = self.fg.params["means"].device
        frame_dtype = self.fg.params["means"].dtype

        modal_phi_input_count = sum(
            value is not None for value in (modal_phi_real, modal_phi_imag)
        )
        if modal_phi_input_count == 1:
            raise ValueError("modal phi real/imag tensors must be provided together")
        if modal_phi_real is None:
            modal_phi_real = torch.empty(
                0,
                self.num_fg_gaussians,
                3,
                device=frame_device,
                dtype=frame_dtype,
            )
            modal_phi_imag = torch.empty_like(modal_phi_real)
        assert modal_phi_imag is not None
        if modal_phi_real.shape != modal_phi_imag.shape:
            raise ValueError("modal phi real/imag tensors must have matching shapes")
        if modal_phi_real.ndim != 3 or modal_phi_real.shape[-1] != 3:
            raise ValueError("modal phi tensors must have shape (K, G, 3)")
        num_modes = int(modal_phi_real.shape[0])
        if num_modes == 0:
            # Empty modal buffers still encode the foreground Gaussian axis.
            modal_phi_real = modal_phi_real.new_empty(
                (0, self.num_fg_gaussians, 3)
            )
            modal_phi_imag = modal_phi_imag.new_empty(
                (0, self.num_fg_gaussians, 3)
            )
        elif modal_phi_real.shape[1] != self.num_fg_gaussians:
            raise ValueError("modal phi Gaussian dimension does not match foreground")
        if not torch.is_floating_point(modal_phi_real) or not torch.is_floating_point(
            modal_phi_imag
        ):
            raise ValueError("modal phi real/imag tensors must have floating-point dtype")
        if modal_phi_real.dtype != modal_phi_imag.dtype:
            raise ValueError("modal phi real/imag tensors must have matching dtypes")
        if modal_phi_real.device != frame_device or modal_phi_imag.device != frame_device:
            raise ValueError("modal phi tensors must share the foreground Gaussian device")
        if not bool(torch.isfinite(modal_phi_real).all().item()) or not bool(
            torch.isfinite(modal_phi_imag).all().item()
        ):
            raise ValueError("modal phi real/imag tensors must contain only finite values")

        coordinate_input_count = sum(
            value is not None for value in (modal_coordinate_real, modal_coordinate_imag)
        )
        if coordinate_input_count == 1:
            raise ValueError(
                "modal coordinate real/imag tensors must be provided together"
            )
        if trajectory_type == "modal_activation" and coordinate_input_count == 0:
            raise ValueError(
                "modal_activation requires modal coordinate real/imag tensors"
            )
        if trajectory_type != "modal_activation" and coordinate_input_count != 0:
            raise ValueError(
                "modal coordinates may only be provided for modal_activation trajectory"
            )
        if modal_coordinate_real is None:
            modal_coordinate_real = torch.empty(
                self.num_frames,
                0,
                device=frame_device,
                dtype=frame_dtype,
            )
            modal_coordinate_imag = torch.empty_like(modal_coordinate_real)
        assert modal_coordinate_imag is not None
        expected_coordinate_modes = (
            num_modes if trajectory_type == "modal_activation" else 0
        )
        if modal_coordinate_real.shape != modal_coordinate_imag.shape:
            raise ValueError(
                "modal coordinate real/imag tensors must have matching shapes"
            )
        if modal_coordinate_real.ndim != 2 or modal_coordinate_real.shape != (
            self.num_frames,
            expected_coordinate_modes,
        ):
            raise ValueError(
                "modal coordinate tensors must have shape "
                f"({self.num_frames}, {expected_coordinate_modes}), got "
                f"{tuple(modal_coordinate_real.shape)}"
            )
        if trajectory_type == "modal_activation" and num_modes <= 0:
            raise ValueError("modal_activation requires at least one modal field")
        if not torch.is_floating_point(modal_coordinate_real) or not torch.is_floating_point(
            modal_coordinate_imag
        ):
            raise ValueError(
                "modal coordinate real/imag tensors must have floating-point dtype"
            )
        if modal_coordinate_real.dtype != modal_coordinate_imag.dtype:
            raise ValueError(
                "modal coordinate real/imag tensors must have matching dtypes"
            )
        if modal_coordinate_real.dtype != modal_phi_real.dtype:
            raise ValueError("modal coordinate dtype must match modal phi")
        if (
            modal_coordinate_real.device != frame_device
            or modal_coordinate_imag.device != frame_device
        ):
            raise ValueError(
                "modal coordinate tensors must share the foreground Gaussian device"
            )
        if not bool(torch.isfinite(modal_coordinate_real).all().item()) or not bool(
            torch.isfinite(modal_coordinate_imag).all().item()
        ):
            raise ValueError(
                "modal coordinate real/imag tensors must contain only finite values"
            )

        if modal_freqs_hz is None:
            if num_modes > 0:
                raise ValueError("modal fields require modal freqs_hz")
            modal_freqs_hz = torch.empty(
                0,
                device=frame_device,
                dtype=frame_dtype,
            )
        if modal_freqs_hz.ndim != 1 or modal_freqs_hz.shape[0] != num_modes:
            raise ValueError(
                f"modal freqs_hz must have shape ({num_modes},), "
                f"got {tuple(modal_freqs_hz.shape)}"
            )
        if not torch.is_floating_point(modal_freqs_hz):
            raise ValueError("modal freqs_hz must have floating-point dtype")
        if modal_freqs_hz.device != frame_device:
            raise ValueError("modal freqs_hz must share the foreground Gaussian device")
        if not bool(torch.isfinite(modal_freqs_hz).all().item()):
            raise ValueError("modal freqs_hz must contain only finite values")
        if modal_freqs_hz.numel() > 0 and bool((modal_freqs_hz <= 0).any().item()):
            raise ValueError("modal freqs_hz must be strictly positive")

        if modal_obs_count_per_point is None:
            modal_obs_count_per_point = torch.zeros(
                num_modes,
                self.num_fg_gaussians,
                device=frame_device,
                dtype=torch.long,
            )
        elif (
            num_modes == 0
            and modal_obs_count_per_point.ndim == 2
            and modal_obs_count_per_point.shape[0] == 0
        ):
            modal_obs_count_per_point = modal_obs_count_per_point.new_empty(
                (0, self.num_fg_gaussians)
            )
        if modal_obs_count_per_point.shape != (
            num_modes,
            self.num_fg_gaussians,
        ):
            raise ValueError(
                "modal obs_count_per_point must have shape "
                f"({num_modes}, {self.num_fg_gaussians}), "
                f"got {tuple(modal_obs_count_per_point.shape)}"
            )

        if trajectory_type == "modal_activation" and any(
            value is None
            for value in (
                modal_frame_view_indices,
                modal_frame_local_indices,
                modal_frame_times_sec,
            )
        ):
            raise ValueError("modal_activation requires frame-map buffers")
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
                dtype=frame_dtype,
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
        modal_frame_view_indices = modal_frame_view_indices.to(
            device=frame_device, dtype=torch.long
        )
        modal_frame_local_indices = modal_frame_local_indices.to(
            device=frame_device, dtype=torch.long
        )
        modal_frame_times_sec = modal_frame_times_sec.to(
            device=frame_device, dtype=frame_dtype
        )
        if trajectory_type == "modal_activation":
            if bool((modal_frame_view_indices < 0).any().item()):
                raise ValueError("modal frame view indices must be non-negative")
            if bool((modal_frame_local_indices < 0).any().item()):
                raise ValueError("modal frame local indices must be non-negative")
            if not bool(torch.isfinite(modal_frame_times_sec).all().item()) or bool(
                (modal_frame_times_sec < 0.0).any().item()
            ):
                raise ValueError(
                    "modal frame times_sec must be finite and non-negative"
                )

        if modal_phi_trainable_mask is None:
            modal_phi_trainable_mask = torch.zeros(
                num_modes,
                self.num_fg_gaussians,
                device=frame_device,
                dtype=torch.bool,
            )
        elif (
            num_modes == 0
            and modal_phi_trainable_mask.ndim == 2
            and modal_phi_trainable_mask.shape[0] == 0
        ):
            modal_phi_trainable_mask = modal_phi_trainable_mask.new_empty(
                (0, self.num_fg_gaussians)
            )
        if modal_phi_trainable_mask.shape != (
            num_modes,
            self.num_fg_gaussians,
        ):
            raise ValueError(
                "modal_phi_trainable_mask must have shape "
                f"({num_modes}, {self.num_fg_gaussians})"
            )
        if modal_phi_trainable_mask.dtype != torch.bool:
            raise ValueError("modal_phi_trainable_mask must have boolean dtype")
        modal_phi_trainable_mask = modal_phi_trainable_mask.to(device=frame_device)
        if modal_joint_params is not None and modal_phi_refinement_params is not None:
            raise ValueError("modal joint and phi-only parameters are mutually exclusive")
        if modal_joint_params is not None:
            if trajectory_type != "modal_activation":
                raise ValueError("modal joint parameters require modal_activation")
            expected_coordinate_shape = (self.num_frames, num_modes)
            expected_phi_shape = (num_modes, self.num_fg_gaussians, 3)
            for name in ("delta_coordinate_real", "delta_coordinate_imag"):
                value = modal_joint_params.params[name]
                if value.shape != expected_coordinate_shape:
                    raise ValueError(
                        f"modal_joint.params.{name} must have shape "
                        f"{expected_coordinate_shape}"
                    )
            for name in ("delta_phi_real", "delta_phi_imag"):
                value = modal_joint_params.params[name]
                if value.shape != expected_phi_shape:
                    raise ValueError(
                        f"modal_joint.params.{name} must have shape {expected_phi_shape}"
                    )
                if bool((value[~modal_phi_trainable_mask] != 0).any().item()):
                    raise ValueError(
                        f"modal_joint.params.{name} must be exactly zero outside "
                        "the trainable phi mask"
                    )
            for name, value in modal_joint_params.params.items():
                if value.device != frame_device or value.dtype != frame_dtype:
                    raise ValueError(
                        f"modal_joint.params.{name} must match the foreground "
                        "Gaussian device and dtype"
                    )
            if not bool(modal_phi_trainable_mask.any().item()):
                raise ValueError("modal joint optimization requires trainable phi points")
        elif modal_phi_refinement_params is not None:
            if trajectory_type != "modal_activation":
                raise ValueError("modal phi refinement requires modal_activation")
            expected_phi_shape = (num_modes, self.num_fg_gaussians, 3)
            for name in ("delta_phi_real", "delta_phi_imag"):
                value = modal_phi_refinement_params.params[name]
                if value.shape != expected_phi_shape:
                    raise ValueError(
                        f"modal_phi_refinement.params.{name} must have shape "
                        f"{expected_phi_shape}"
                    )
                if value.device != frame_device or value.dtype != frame_dtype:
                    raise ValueError(
                        f"modal_phi_refinement.params.{name} must match the "
                        "foreground Gaussian device and dtype"
                    )
                if bool((value[~modal_phi_trainable_mask] != 0).any().item()):
                    raise ValueError(
                        f"modal_phi_refinement.params.{name} must be exactly zero "
                        "outside the trainable phi mask"
                    )
            if not bool(modal_phi_trainable_mask.any().item()):
                raise ValueError("modal phi refinement requires trainable phi points")
        elif bool(modal_phi_trainable_mask.any().item()):
            raise ValueError("modal phi trainable mask requires trainable phi parameters")

        self.modal_joint = modal_joint_params
        self.modal_phi_refinement = modal_phi_refinement_params
        self.register_buffer("modal_coordinate_real", modal_coordinate_real)
        self.register_buffer("modal_coordinate_imag", modal_coordinate_imag)
        self.register_buffer("modal_phi_real", modal_phi_real)
        self.register_buffer("modal_phi_imag", modal_phi_imag)
        self.register_buffer("modal_freqs_hz", modal_freqs_hz)
        self.register_buffer("modal_obs_count_per_point", modal_obs_count_per_point.long())
        if isinstance(modal_synthetic_enabled, Tensor):
            modal_synthetic_enabled = bool(modal_synthetic_enabled.item())
        self.register_buffer(
            "modal_synthetic_enabled",
            torch.tensor(bool(modal_synthetic_enabled), dtype=torch.bool),
        )
        self.register_buffer("modal_frame_view_indices", modal_frame_view_indices)
        self.register_buffer("modal_frame_local_indices", modal_frame_local_indices)
        self.register_buffer("modal_frame_times_sec", modal_frame_times_sec)
        self.register_buffer("modal_phi_trainable_mask", modal_phi_trainable_mask)

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
        return (
            self.modal_coordinate_real.numel() > 0
            and self.modal_coordinate_imag.numel() > 0
        )

    @property
    def has_modal_field(self) -> bool:
        return self.modal_phi_real.numel() > 0 and self.modal_phi_imag.numel() > 0

    @property
    def has_modal_joint(self) -> bool:
        return self.modal_joint is not None

    @property
    def has_modal_phi_refinement(self) -> bool:
        return self.modal_phi_refinement is not None

    @property
    def has_trainable_modal_phi(self) -> bool:
        return self.has_modal_joint or self.has_modal_phi_refinement

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
        if self.trajectory_type != "modal_activation":
            raise RuntimeError(
                "compute_modal_coefficients requires modal_activation trajectory"
            )
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
        ts = ts.to(device=self.modal_coordinate_real.device, dtype=torch.long)
        real, imag = self.get_all_modal_coefficients()
        return real[ts], imag[ts]

    def get_centered_modal_coordinate_deltas(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_modal_joint:
            return (
                torch.zeros_like(self.modal_coordinate_real),
                torch.zeros_like(self.modal_coordinate_imag),
            )
        assert self.modal_joint is not None
        view_indices = self.modal_frame_view_indices
        num_views = int(view_indices.max().item()) + 1
        counts = torch.bincount(view_indices, minlength=num_views).to(
            dtype=self.modal_coordinate_real.dtype
        )
        if bool((counts <= 0).any().item()):
            raise RuntimeError("modal joint optimization requires every view to have frames")

        centered: list[torch.Tensor] = []
        for name in ("delta_coordinate_real", "delta_coordinate_imag"):
            delta = self.modal_joint.params[name]
            sums = torch.zeros(
                num_views,
                delta.shape[1],
                device=delta.device,
                dtype=delta.dtype,
            )
            sums.index_add_(0, view_indices, delta)
            means = sums / counts[:, None]
            centered.append(delta - means[view_indices])
        return centered[0], centered[1]

    def get_all_modal_coefficients(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_real, delta_imag = self.get_centered_modal_coordinate_deltas()
        return (
            self.modal_coordinate_real + delta_real,
            self.modal_coordinate_imag + delta_imag,
        )

    def compute_modal_offsets_from_coefficients(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        inds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if real.shape != imag.shape or real.ndim != 2:
            raise ValueError("modal coefficients must be matching (B,K) tensors")
        if real.shape[1] != self.modal_phi_real.shape[0]:
            raise ValueError("modal coefficient mode count does not match phi")
        phi_real, phi_imag = self.get_effective_modal_phi(inds)
        real = real.to(device=phi_real.device, dtype=phi_real.dtype)
        imag = imag.to(device=phi_real.device, dtype=phi_real.dtype)
        return torch.einsum("bk,kgc->gbc", real, phi_real) - torch.einsum(
            "bk,kgc->gbc", imag, phi_imag
        )

    def compute_modal_offsets(
        self, ts: torch.Tensor, inds: torch.Tensor | None = None
    ) -> torch.Tensor:
        real, imag = self.compute_modal_coefficients(ts)
        return self.compute_modal_offsets_from_coefficients(real, imag, inds)

    def get_effective_modal_phi(
        self, inds: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phi_real = self.modal_phi_real
        phi_imag = self.modal_phi_imag
        if self.has_trainable_modal_phi:
            residuals = (
                self.modal_joint.params
                if self.modal_joint is not None
                else self.modal_phi_refinement.params
            )
            mask = self.modal_phi_trainable_mask[..., None]
            raw_real = phi_real + torch.where(
                mask,
                residuals["delta_phi_real"],
                torch.zeros_like(phi_real),
            )
            raw_imag = phi_imag + torch.where(
                mask,
                residuals["delta_phi_imag"],
                torch.zeros_like(phi_imag),
            )

            mask_float = mask.to(dtype=phi_real.dtype)
            inner_real = torch.sum(
                mask_float * (phi_real * raw_real + phi_imag * raw_imag),
                dim=(1, 2),
            )
            inner_imag = torch.sum(
                mask_float * (phi_real * raw_imag - phi_imag * raw_real),
                dim=(1, 2),
            )
            phase = torch.atan2(inner_imag, inner_real)
            cosine = torch.cos(phase)[:, None, None]
            sine = torch.sin(phase)[:, None, None]
            aligned_real = raw_real * cosine + raw_imag * sine
            aligned_imag = raw_imag * cosine - raw_real * sine

            target_energy = torch.sum(
                mask_float * (phi_real.square() + phi_imag.square()), dim=(1, 2)
            )
            aligned_energy = torch.sum(
                mask_float * (aligned_real.square() + aligned_imag.square()),
                dim=(1, 2),
            )
            if bool((target_energy <= 0).any().item()):
                raise RuntimeError("trainable staged phi must have positive energy")
            scale = torch.sqrt(target_energy / aligned_energy.clamp_min(1e-20))
            aligned_real = aligned_real * scale[:, None, None]
            aligned_imag = aligned_imag * scale[:, None, None]
            phi_real = torch.where(mask, aligned_real, phi_real)
            phi_imag = torch.where(mask, aligned_imag, phi_imag)
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

    @torch.no_grad()
    def densify_modal_fields(self, should_split: torch.Tensor, should_dup: torch.Tensor):
        if self.has_trainable_modal_phi:
            raise RuntimeError("modal phi optimization forbids Gaussian densification")
        for name in ("modal_phi_real", "modal_phi_imag"):
            x = getattr(self, name)
            x_dup = x[:, should_dup]
            x_split = x[:, should_split].repeat(1, 2, 1)
            setattr(self, name, torch.cat([x[:, ~should_split], x_dup, x_split], dim=1))
        for name in ("modal_obs_count_per_point", "modal_phi_trainable_mask"):
            x = getattr(self, name)
            x_dup = x[:, should_dup]
            x_split = x[:, should_split].repeat(1, 2)
            setattr(
                self,
                name,
                torch.cat([x[:, ~should_split], x_dup, x_split], dim=1),
            )

    @torch.no_grad()
    def cull_modal_fields(self, should_cull: torch.Tensor):
        if self.has_trainable_modal_phi:
            raise RuntimeError("modal phi optimization forbids Gaussian culling")
        self.modal_phi_real = self.modal_phi_real[:, ~should_cull]
        self.modal_phi_imag = self.modal_phi_imag[:, ~should_cull]
        self.modal_obs_count_per_point = self.modal_obs_count_per_point[:, ~should_cull]
        self.modal_phi_trainable_mask = self.modal_phi_trainable_mask[:, ~should_cull]

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
        legacy_modal_keys = (
            f"{prefix}modal.params.activations",
            f"{prefix}modal.params.envelope_knots",
            f"{prefix}modal_refinement.params.delta_phi",
            f"{prefix}modal_refinement_mask",
            f"{prefix}modal_refinement_role",
            f"{prefix}modal_anchor_mask",
        )
        present_legacy_modal_keys = [
            key for key in legacy_modal_keys if key in state_dict
        ]
        if present_legacy_modal_keys:
            raise ValueError(
                "Legacy harmonic-envelope and modal shape-refinement checkpoints "
                "are incompatible with per-frame flow coordinates; materialize a "
                "new checkpoint from the static checkpoint, staged modal manifest, "
                "frame map, and modal flow-coordinate artifact. Found legacy state: "
                f"{present_legacy_modal_keys}"
            )

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
            if f"{prefix}modal_coordinate_real" in state_dict:
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

        modal_coordinate_real = None
        modal_coordinate_imag = None
        modal_phi_real = None
        modal_phi_imag = None
        modal_freqs_hz = None
        modal_obs_count_per_point = None
        modal_frame_view_indices = None
        modal_frame_local_indices = None
        modal_frame_times_sec = None
        modal_joint_params = None
        modal_phi_refinement_params = None
        modal_phi_trainable_mask = None

        modal_field_keys = (
            f"{prefix}modal_phi_real",
            f"{prefix}modal_phi_imag",
            f"{prefix}modal_freqs_hz",
        )
        modal_field_count = sum(key in state_dict for key in modal_field_keys)
        if modal_field_count not in (0, len(modal_field_keys)):
            missing = [key for key in modal_field_keys if key not in state_dict]
            raise ValueError(f"Modal checkpoint is missing modal field state: {missing}")
        if modal_field_count:
            modal_phi_real = state_dict[f"{prefix}modal_phi_real"]
            modal_phi_imag = state_dict[f"{prefix}modal_phi_imag"]
            modal_freqs_hz = state_dict[f"{prefix}modal_freqs_hz"]
            if f"{prefix}modal_obs_count_per_point" in state_dict:
                modal_obs_count_per_point = state_dict[
                    f"{prefix}modal_obs_count_per_point"
                ]

        if trajectory_type == "modal_activation":
            required_coordinate_keys = (
                f"{prefix}modal_coordinate_real",
                f"{prefix}modal_coordinate_imag",
                f"{prefix}modal_frame_view_indices",
                f"{prefix}modal_frame_local_indices",
                f"{prefix}modal_frame_times_sec",
            )
            missing_coordinate_keys = [
                key for key in required_coordinate_keys if key not in state_dict
            ]
            if missing_coordinate_keys:
                raise ValueError(
                    "Per-frame flow-coordinate checkpoint is missing required state: "
                    f"{missing_coordinate_keys}"
                )
            if modal_field_count != len(modal_field_keys):
                raise ValueError(
                    "Per-frame flow-coordinate checkpoint is missing staged modal fields"
                )
            modal_coordinate_real = state_dict[f"{prefix}modal_coordinate_real"]
            modal_coordinate_imag = state_dict[f"{prefix}modal_coordinate_imag"]
            modal_frame_view_indices = state_dict[
                f"{prefix}modal_frame_view_indices"
            ]
            modal_frame_local_indices = state_dict[
                f"{prefix}modal_frame_local_indices"
            ]
            modal_frame_times_sec = state_dict[f"{prefix}modal_frame_times_sec"]

            joint_prefix = f"{prefix}modal_joint.params."
            phi_refinement_prefix = f"{prefix}modal_phi_refinement.params."
            joint_keys = [key for key in state_dict if key.startswith(joint_prefix)]
            phi_refinement_keys = [
                key for key in state_dict if key.startswith(phi_refinement_prefix)
            ]
            if joint_keys and phi_refinement_keys:
                raise ValueError(
                    "Checkpoint contains both joint and phi-only modal parameters"
                )
            if joint_keys:
                modal_joint_params = ModalJointParams.init_from_state_dict(
                    state_dict,
                    prefix=joint_prefix,
                )
                mask_key = f"{prefix}modal_phi_trainable_mask"
                if mask_key not in state_dict:
                    raise ValueError(
                        "Joint modal checkpoint is missing modal_phi_trainable_mask"
                )
                modal_phi_trainable_mask = state_dict[mask_key]
            elif phi_refinement_keys:
                modal_phi_refinement_params = (
                    ModalPhiRefinementParams.init_from_state_dict(
                        state_dict,
                        prefix=phi_refinement_prefix,
                    )
                )
                mask_key = f"{prefix}modal_phi_trainable_mask"
                if mask_key not in state_dict:
                    raise ValueError(
                        "Phi-only modal checkpoint is missing "
                        "modal_phi_trainable_mask"
                    )
                modal_phi_trainable_mask = state_dict[mask_key]
            elif f"{prefix}modal_phi_trainable_mask" in state_dict:
                stored_mask = state_dict[f"{prefix}modal_phi_trainable_mask"]
                if bool(stored_mask.any().item()):
                    raise ValueError(
                        "Checkpoint has a non-empty modal phi mask without joint parameters"
                    )
                modal_phi_trainable_mask = stored_mask

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
            modal_coordinate_real=modal_coordinate_real,
            modal_coordinate_imag=modal_coordinate_imag,
            modal_phi_real=modal_phi_real,
            modal_phi_imag=modal_phi_imag,
            modal_freqs_hz=modal_freqs_hz,
            modal_obs_count_per_point=modal_obs_count_per_point,
            modal_frame_view_indices=modal_frame_view_indices,
            modal_frame_local_indices=modal_frame_local_indices,
            modal_frame_times_sec=modal_frame_times_sec,
            modal_synthetic_enabled=modal_synthetic_enabled,
            modal_joint_params=modal_joint_params,
            modal_phi_refinement_params=modal_phi_refinement_params,
            modal_phi_trainable_mask=modal_phi_trainable_mask,
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
