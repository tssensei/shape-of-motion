import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat.rendering import rasterization
from gsplat.rendering import rasterization_2dgs
from torch import Tensor

from flow3d.params import (
    GaussianParams,
    ModalHarmonicEnvelope,
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
        modal: ModalHarmonicEnvelope | None = None,
        modal_phi_real: Tensor | None = None,
        modal_phi_imag: Tensor | None = None,
        modal_freqs_hz: Tensor | None = None,
        modal_obs_count_per_point: Tensor | None = None,
        modal_frame_view_indices: Tensor | None = None,
        modal_frame_local_indices: Tensor | None = None,
        modal_frame_times_sec: Tensor | None = None,
        modal_envelope_knot_offsets: Tensor | None = None,
        modal_envelope_knot_times_sec: Tensor | None = None,
        modal_envelope_knot_interval_sec: float | Tensor | None = None,
        modal_frame_envelope_left: Tensor | None = None,
        modal_frame_envelope_right: Tensor | None = None,
        modal_frame_envelope_lerp: Tensor | None = None,
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
                raise ValueError("modal_activation requires a modal envelope")
            if modal_phi_real is None or modal_phi_imag is None:
                raise ValueError("modal_activation requires modal phi real/imag tensors")
            if modal_phi_real.shape != modal_phi_imag.shape:
                raise ValueError("modal phi real/imag tensors must have matching shapes")
            if modal_phi_real.ndim != 3 or modal_phi_real.shape[-1] != 3:
                raise ValueError("modal phi tensors must have shape (K, G, 3)")
            if modal_phi_real.shape[1] != self.num_fg_gaussians:
                raise ValueError("modal phi Gaussian dimension does not match foreground")
            if modal_phi_real.shape[0] != modal.num_modes:
                raise ValueError("modal phi mode count does not match envelope modes")
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
        if trajectory_type == "modal_activation" and any(
            value is None
            for value in (
                modal_frame_view_indices,
                modal_frame_local_indices,
                modal_frame_times_sec,
                modal_envelope_knot_offsets,
                modal_envelope_knot_times_sec,
                modal_envelope_knot_interval_sec,
                modal_frame_envelope_left,
                modal_frame_envelope_right,
                modal_frame_envelope_lerp,
            )
        ):
            raise ValueError(
                "modal_activation requires frame-map and envelope-layout buffers"
            )
        frame_device = self.fg.params["means"].device
        frame_dtype = self.fg.params["means"].dtype
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
        if modal_envelope_knot_offsets is None:
            modal_envelope_knot_offsets = torch.zeros(
                1, device=frame_device, dtype=torch.long
            )
        if modal_envelope_knot_times_sec is None:
            modal_envelope_knot_times_sec = torch.empty(
                0, device=frame_device, dtype=frame_dtype
            )
        if modal_envelope_knot_interval_sec is None:
            modal_envelope_knot_interval_sec = 0.0
        if modal_frame_envelope_left is None:
            modal_frame_envelope_left = torch.full(
                (self.num_frames,), -1, device=frame_device, dtype=torch.long
            )
        if modal_frame_envelope_right is None:
            modal_frame_envelope_right = torch.full(
                (self.num_frames,), -1, device=frame_device, dtype=torch.long
            )
        if modal_frame_envelope_lerp is None:
            modal_frame_envelope_lerp = torch.zeros(
                self.num_frames, device=frame_device, dtype=frame_dtype
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
        for name, values in (
            ("modal envelope knot offsets", modal_envelope_knot_offsets),
            ("modal frame envelope left indices", modal_frame_envelope_left),
            ("modal frame envelope right indices", modal_frame_envelope_right),
        ):
            if values.dtype not in integer_dtypes:
                raise ValueError(f"{name} must have integer dtype")
        if not torch.is_floating_point(modal_frame_times_sec):
            raise ValueError("modal frame times_sec must have floating-point dtype")
        if not torch.is_floating_point(modal_envelope_knot_times_sec):
            raise ValueError("modal envelope knot times must have floating-point dtype")
        if not torch.is_floating_point(modal_frame_envelope_lerp):
            raise ValueError("modal frame envelope lerp must have floating-point dtype")
        for name, values in (
            ("modal frame view indices", modal_frame_view_indices),
            ("modal frame local indices", modal_frame_local_indices),
            ("modal frame times_sec", modal_frame_times_sec),
            ("modal frame envelope left indices", modal_frame_envelope_left),
            ("modal frame envelope right indices", modal_frame_envelope_right),
            ("modal frame envelope lerp", modal_frame_envelope_lerp),
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
        modal_envelope_knot_offsets = modal_envelope_knot_offsets.to(
            device=frame_device, dtype=torch.long
        )
        modal_envelope_knot_times_sec = modal_envelope_knot_times_sec.to(
            device=frame_device, dtype=frame_dtype
        )
        modal_envelope_knot_interval_sec = torch.as_tensor(
            modal_envelope_knot_interval_sec,
            device=frame_device,
            dtype=frame_dtype,
        )
        modal_frame_envelope_left = modal_frame_envelope_left.to(
            device=frame_device, dtype=torch.long
        )
        modal_frame_envelope_right = modal_frame_envelope_right.to(
            device=frame_device, dtype=torch.long
        )
        modal_frame_envelope_lerp = modal_frame_envelope_lerp.to(
            device=frame_device, dtype=frame_dtype
        )
        if trajectory_type == "modal_activation":
            if modal is None:
                raise ValueError("modal_activation requires a modal envelope")
            envelope_knots = modal.params["envelope_knots"]
            if modal_envelope_knot_interval_sec.ndim != 0 or not bool(
                torch.isfinite(modal_envelope_knot_interval_sec).item()
            ) or float(modal_envelope_knot_interval_sec.item()) <= 0.0:
                raise ValueError(
                    "modal envelope knot interval must be a finite positive scalar"
                )
            if envelope_knots.device != frame_device:
                raise ValueError("modal envelope knots must share the Gaussian device")
            if envelope_knots.dtype != modal_phi_real.dtype:
                raise ValueError("modal envelope knot dtype must match staged modal phi")
            if modal_envelope_knot_offsets.ndim != 1 or (
                modal_envelope_knot_offsets.shape[0] < 2
            ):
                raise ValueError(
                    "modal envelope knot offsets must have shape (num_views + 1,)"
                )
            modal_num_views = int(modal_envelope_knot_offsets.shape[0] - 1)
            total_knots = modal.num_total_knots
            if int(modal_envelope_knot_offsets[0].item()) != 0 or int(
                modal_envelope_knot_offsets[-1].item()
            ) != total_knots:
                raise ValueError(
                    "modal envelope knot offsets must start at zero and end at "
                    "the total knot count"
                )
            if bool(
                (modal_envelope_knot_offsets[1:] <= modal_envelope_knot_offsets[:-1])
                .any()
                .item()
            ):
                raise ValueError(
                    "modal envelope knot offsets must allocate at least one knot "
                    "per view"
                )
            if modal_envelope_knot_times_sec.shape != (total_knots,):
                raise ValueError(
                    "modal envelope knot times must have shape "
                    f"({total_knots},), got "
                    f"{tuple(modal_envelope_knot_times_sec.shape)}"
                )
            if not bool(torch.isfinite(modal_envelope_knot_times_sec).all().item()) or bool(
                (modal_envelope_knot_times_sec < 0.0).any().item()
            ):
                raise ValueError(
                    "modal envelope knot times must be finite and non-negative"
                )
            for view_index in range(modal_num_views):
                start = int(modal_envelope_knot_offsets[view_index].item())
                end = int(modal_envelope_knot_offsets[view_index + 1].item())
                view_knot_times = modal_envelope_knot_times_sec[start:end]
                if float(view_knot_times[0].item()) != 0.0:
                    raise ValueError("each modal envelope view must start at t=0")
                if view_knot_times.numel() > 1 and bool(
                    (view_knot_times[1:] <= view_knot_times[:-1]).any().item()
                ):
                    raise ValueError(
                        "modal envelope knot times must be strictly increasing "
                        "within each view"
                    )
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
            if bool((modal_frame_envelope_left < 0).any().item()) or bool(
                (modal_frame_envelope_left >= total_knots).any().item()
            ):
                raise ValueError("modal frame envelope left indices are out of range")
            if bool((modal_frame_envelope_right < 0).any().item()) or bool(
                (modal_frame_envelope_right >= total_knots).any().item()
            ):
                raise ValueError("modal frame envelope right indices are out of range")
            if bool(
                (modal_frame_envelope_right < modal_frame_envelope_left).any().item()
            ):
                raise ValueError("modal frame envelope right index precedes left index")
            if not bool(torch.isfinite(modal_frame_envelope_lerp).all().item()) or bool(
                ((modal_frame_envelope_lerp < 0.0) | (modal_frame_envelope_lerp > 1.0))
                .any()
                .item()
            ):
                raise ValueError("modal frame envelope lerp must lie in [0, 1]")
            view_starts = modal_envelope_knot_offsets[modal_frame_view_indices]
            view_ends = modal_envelope_knot_offsets[modal_frame_view_indices + 1]
            if bool(
                (
                    (modal_frame_envelope_left < view_starts)
                    | (modal_frame_envelope_left >= view_ends)
                    | (modal_frame_envelope_right < view_starts)
                    | (modal_frame_envelope_right >= view_ends)
                )
                .any()
                .item()
            ):
                raise ValueError("modal envelope interpolation crosses view boundaries")
            left_times = modal_envelope_knot_times_sec[modal_frame_envelope_left]
            right_times = modal_envelope_knot_times_sec[modal_frame_envelope_right]
            reconstructed_times = left_times + modal_frame_envelope_lerp * (
                right_times - left_times
            )
            if not torch.allclose(
                reconstructed_times,
                modal_frame_times_sec,
                rtol=1.0e-5,
                atol=1.0e-6,
            ):
                raise ValueError(
                    "modal envelope interpolation does not reproduce frame times"
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
            modal_frame_times_sec,
        )
        self.register_buffer(
            "modal_envelope_knot_offsets",
            modal_envelope_knot_offsets,
        )
        self.register_buffer(
            "modal_envelope_knot_times_sec",
            modal_envelope_knot_times_sec,
        )
        self.register_buffer(
            "modal_envelope_knot_interval_sec",
            modal_envelope_knot_interval_sec,
        )
        self.register_buffer(
            "modal_frame_envelope_left",
            modal_frame_envelope_left,
        )
        self.register_buffer(
            "modal_frame_envelope_right",
            modal_frame_envelope_right,
        )
        self.register_buffer(
            "modal_frame_envelope_lerp",
            modal_frame_envelope_lerp,
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

    def compute_modal_envelopes(
        self, ts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.modal is None:
            raise RuntimeError("compute_modal_envelopes requires a modal envelope")
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

        envelope_knots = self.modal.params["envelope_knots"]
        view_knot_tangents = []
        for view_index in range(self.modal_envelope_knot_offsets.shape[0] - 1):
            start = int(self.modal_envelope_knot_offsets[view_index].item())
            end = int(self.modal_envelope_knot_offsets[view_index + 1].item())
            view_knots = envelope_knots[start:end]
            if end - start <= 1:
                view_knot_tangents.append(torch.zeros_like(view_knots))
                continue
            view_times = self.modal_envelope_knot_times_sec[start:end].to(
                dtype=envelope_knots.dtype
            )
            delta_time = view_times[1:] - view_times[:-1]
            secants = (view_knots[1:] - view_knots[:-1]) / delta_time[
                :, None, None
            ]
            if end - start == 2:
                view_knot_tangents.append(
                    torch.cat([secants[:1], secants[:1]], dim=0)
                )
            else:
                previous_interval = delta_time[:-1]
                next_interval = delta_time[1:]
                interior_tangents = (
                    next_interval[:, None, None] * secants[:-1]
                    + previous_interval[:, None, None] * secants[1:]
                ) / (previous_interval + next_interval)[:, None, None]
                view_knot_tangents.append(
                    torch.cat(
                        [secants[:1], interior_tangents, secants[-1:]],
                        dim=0,
                    )
                )
        knot_tangents = torch.cat(view_knot_tangents, dim=0)

        left_indices = self.modal_frame_envelope_left[ts]
        right_indices = self.modal_frame_envelope_right[ts]
        left = envelope_knots[left_indices]
        right = envelope_knots[right_indices]
        left_tangent = knot_tangents[left_indices]
        right_tangent = knot_tangents[right_indices]
        interpolation_coordinate = self.modal_frame_envelope_lerp[ts].to(
            dtype=envelope_knots.dtype
        )[:, None, None]
        coordinate_square = interpolation_coordinate.square()
        coordinate_cube = coordinate_square * interpolation_coordinate
        h00 = 2.0 * coordinate_cube - 3.0 * coordinate_square + 1.0
        h10 = coordinate_cube - 2.0 * coordinate_square + interpolation_coordinate
        h01 = -2.0 * coordinate_cube + 3.0 * coordinate_square
        h11 = coordinate_cube - coordinate_square
        segment_duration = (
            self.modal_envelope_knot_times_sec[right_indices]
            - self.modal_envelope_knot_times_sec[left_indices]
        ).to(dtype=envelope_knots.dtype)[:, None, None]
        envelopes = (
            h00 * left
            + h10 * segment_duration * left_tangent
            + h01 * right
            + h11 * segment_duration * right_tangent
        )
        same_knot = (left_indices == right_indices)[:, None, None]
        envelopes = torch.where(same_knot, left, envelopes)
        if not bool(torch.isfinite(envelopes).all().item()):
            raise FloatingPointError("interpolated modal envelope is not finite")
        return envelopes[..., 0], envelopes[..., 1]

    def compute_modal_coefficients(
        self, ts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        envelope_real, envelope_imag = self.compute_modal_envelopes(ts)
        times_sec = self.modal_frame_times_sec[ts.to(dtype=torch.long)].to(
            dtype=envelope_real.dtype
        )
        freqs_hz = self.modal_freqs_hz.to(dtype=envelope_real.dtype)
        theta = 2.0 * torch.pi * times_sec[:, None] * freqs_hz[None, :]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        coefficient_real = (
            envelope_real * cos_theta - envelope_imag * sin_theta
        )
        coefficient_imag = (
            envelope_real * sin_theta + envelope_imag * cos_theta
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

    def compute_envelope_magnitude_loss(self) -> torch.Tensor:
        if self.modal is None:
            return self.fg.params["means"].new_zeros(())
        ts = torch.arange(self.num_frames, device=self.modal_phi_real.device)
        envelope_real, envelope_imag = self.compute_modal_envelopes(ts)
        return (envelope_real.square() + envelope_imag.square()).mean()

    def compute_envelope_smoothness_loss(self) -> torch.Tensor:
        if self.modal is None:
            return self.fg.params["means"].new_zeros(())
        knots = self.modal.params["envelope_knots"]
        reference_interval = self.modal_envelope_knot_interval_sec.to(
            dtype=knots.dtype
        )
        group_losses = []
        for view_index in range(self.modal_envelope_knot_offsets.shape[0] - 1):
            start = int(self.modal_envelope_knot_offsets[view_index].item())
            end = int(self.modal_envelope_knot_offsets[view_index + 1].item())
            if end - start <= 1:
                continue
            view_knots = knots[start:end]
            delta = view_knots[1:] - view_knots[:-1]
            delta_time = (
                self.modal_envelope_knot_times_sec[start + 1 : end]
                - self.modal_envelope_knot_times_sec[start : end - 1]
            ).to(dtype=knots.dtype)
            if not bool(torch.isfinite(delta_time).all().item()) or bool(
                (delta_time <= 0.0).any().item()
            ):
                raise ValueError("modal envelope knot intervals must be positive")
            scaled_square = delta.square().sum(dim=-1) / (
                delta_time[:, None] / reference_interval
            ).square()
            group_losses.append(scaled_square.mean(dim=0))
        if not group_losses:
            return knots.new_zeros(())
        loss = torch.stack(group_losses, dim=0).mean()
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("modal envelope smoothness loss is not finite")
        return loss

    def compute_envelope_curvature_loss(self) -> torch.Tensor:
        if self.modal is None:
            return self.fg.params["means"].new_zeros(())
        knots = self.modal.params["envelope_knots"]
        reference_interval = self.modal_envelope_knot_interval_sec.to(
            dtype=knots.dtype
        )
        group_losses = []
        for view_index in range(self.modal_envelope_knot_offsets.shape[0] - 1):
            start = int(self.modal_envelope_knot_offsets[view_index].item())
            end = int(self.modal_envelope_knot_offsets[view_index + 1].item())
            if end - start <= 2:
                continue
            view_knots = knots[start:end]
            delta_time = (
                self.modal_envelope_knot_times_sec[start + 1 : end]
                - self.modal_envelope_knot_times_sec[start : end - 1]
            ).to(dtype=knots.dtype)
            if not bool(torch.isfinite(delta_time).all().item()) or bool(
                (delta_time <= 0.0).any().item()
            ):
                raise ValueError("modal envelope knot intervals must be positive")
            normalized_secants = (view_knots[1:] - view_knots[:-1]) / (
                delta_time[:, None, None] / reference_interval
            )
            curvature = normalized_secants[1:] - normalized_secants[:-1]
            group_losses.append(curvature.square().sum(dim=-1).mean(dim=0))
        if not group_losses:
            return knots.new_zeros(())
        loss = torch.stack(group_losses, dim=0).mean()
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("modal envelope curvature loss is not finite")
        return loss

    def compute_envelope_max_slope_change(self) -> torch.Tensor:
        if self.modal is None:
            return self.fg.params["means"].new_zeros(())
        knots = self.modal.params["envelope_knots"]
        reference_interval = self.modal_envelope_knot_interval_sec.to(
            dtype=knots.dtype
        )
        max_changes = []
        for view_index in range(self.modal_envelope_knot_offsets.shape[0] - 1):
            start = int(self.modal_envelope_knot_offsets[view_index].item())
            end = int(self.modal_envelope_knot_offsets[view_index + 1].item())
            if end - start <= 2:
                continue
            delta_time = (
                self.modal_envelope_knot_times_sec[start + 1 : end]
                - self.modal_envelope_knot_times_sec[start : end - 1]
            ).to(dtype=knots.dtype)
            normalized_secants = (
                knots[start + 1 : end] - knots[start : end - 1]
            ) / (delta_time[:, None, None] / reference_interval)
            slope_change = normalized_secants[1:] - normalized_secants[:-1]
            max_changes.append(
                torch.linalg.vector_norm(slope_change, dim=-1).max()
            )
        if not max_changes:
            return knots.new_zeros(())
        value = torch.stack(max_changes).max()
        if not bool(torch.isfinite(value).item()):
            raise FloatingPointError(
                "modal envelope maximum slope change is not finite"
            )
        return value

    def compute_envelope_max_knot_jump(self) -> torch.Tensor:
        if self.modal is None:
            return self.fg.params["means"].new_zeros(())
        knots = self.modal.params["envelope_knots"]
        max_jumps = []
        for view_index in range(self.modal_envelope_knot_offsets.shape[0] - 1):
            start = int(self.modal_envelope_knot_offsets[view_index].item())
            end = int(self.modal_envelope_knot_offsets[view_index + 1].item())
            if end - start <= 1:
                continue
            delta = knots[start + 1 : end] - knots[start : end - 1]
            max_jumps.append(torch.linalg.vector_norm(delta, dim=-1).max())
        if not max_jumps:
            return knots.new_zeros(())
        value = torch.stack(max_jumps).max()
        if not bool(torch.isfinite(value).item()):
            raise FloatingPointError("modal envelope maximum knot jump is not finite")
        return value

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
        legacy_activation_key = f"{prefix}modal.params.activations"
        if legacy_activation_key in state_dict:
            raise ValueError(
                "Constant per-view harmonic activation checkpoints are not "
                "supported; start a new harmonic-envelope run from the static "
                "checkpoint and staged modal manifest."
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
            if f"{prefix}modal.params.envelope_knots" in state_dict:
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
        modal_envelope_knot_offsets = None
        modal_envelope_knot_times_sec = None
        modal_envelope_knot_interval_sec = None
        modal_frame_envelope_left = None
        modal_frame_envelope_right = None
        modal_frame_envelope_lerp = None
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
                f"{prefix}modal.params.envelope_knots",
                f"{prefix}modal_frame_view_indices",
                f"{prefix}modal_frame_local_indices",
                f"{prefix}modal_frame_times_sec",
                f"{prefix}modal_envelope_knot_offsets",
                f"{prefix}modal_envelope_knot_times_sec",
                f"{prefix}modal_envelope_knot_interval_sec",
                f"{prefix}modal_frame_envelope_left",
                f"{prefix}modal_frame_envelope_right",
                f"{prefix}modal_frame_envelope_lerp",
            )
            missing_frame_keys = [
                key for key in required_frame_keys if key not in state_dict
            ]
            if missing_frame_keys:
                raise ValueError(
                    "Harmonic-envelope checkpoint is missing required state: "
                    f"{missing_frame_keys}"
                )
            modal = ModalHarmonicEnvelope.init_from_state_dict(
                state_dict, prefix=f"{prefix}modal.params."
            )
            modal_frame_view_indices = state_dict[f"{prefix}modal_frame_view_indices"]
            modal_frame_local_indices = state_dict[f"{prefix}modal_frame_local_indices"]
            modal_frame_times_sec = state_dict[f"{prefix}modal_frame_times_sec"]
            modal_envelope_knot_offsets = state_dict[
                f"{prefix}modal_envelope_knot_offsets"
            ]
            modal_envelope_knot_times_sec = state_dict[
                f"{prefix}modal_envelope_knot_times_sec"
            ]
            modal_envelope_knot_interval_sec = state_dict[
                f"{prefix}modal_envelope_knot_interval_sec"
            ]
            modal_frame_envelope_left = state_dict[
                f"{prefix}modal_frame_envelope_left"
            ]
            modal_frame_envelope_right = state_dict[
                f"{prefix}modal_frame_envelope_right"
            ]
            modal_frame_envelope_lerp = state_dict[
                f"{prefix}modal_frame_envelope_lerp"
            ]
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
            modal_envelope_knot_offsets=modal_envelope_knot_offsets,
            modal_envelope_knot_times_sec=modal_envelope_knot_times_sec,
            modal_envelope_knot_interval_sec=modal_envelope_knot_interval_sec,
            modal_frame_envelope_left=modal_frame_envelope_left,
            modal_frame_envelope_right=modal_frame_envelope_right,
            modal_frame_envelope_lerp=modal_frame_envelope_lerp,
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
