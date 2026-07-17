import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState

from flow3d.modal_utils import (
    load_modal_modes,
    stack_modal_motion_fill_display_classes,
)
from flow3d.scene_model import SceneModel
from flow3d.vis.utils import draw_tracks_2d_th, get_server
from flow3d.vis.viewer import (
    DynamicViewer,
    ViewerCamera,
    build_modal_playback_groups,
)
from modal_surface.io import load_view_config


MODAL_SHAPE_PARAMETERIZATION = "role_delta_phi_v1"
MODAL_DELTA_PHI_STATE_KEY = "modal_refinement.params.delta_phi"
MODAL_REFINEMENT_MASK_STATE_KEY = "modal_refinement_mask"
MODAL_REFINEMENT_ROLE_STATE_KEY = "modal_refinement_role"


class Renderer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        # Logging.
        work_dir: str,
        port: int | None = None,
        vggt_view_configs: tuple[str, ...] = (),
        modal_anchor_manifest: str | None = None,
    ):
        self.device = device

        self.model = model
        self.num_frames = model.num_frames

        self.work_dir = work_dir
        self.global_step = 0
        self.epoch = 0
        (
            self.modal_anchor_points,
            self.modal_anchor_phi_real,
            self.modal_anchor_phi_imag,
            modal_anchor_freqs_hz,
            self.modal_anchor_role_classes,
            modal_anchor_role_mode_labels,
        ) = self._load_modal_anchor_data(modal_anchor_manifest)

        self.viewer = None
        if port is not None:
            orbit_center = self._gaussian_orbit_center()
            frustum_scale = 0.08 * self._gaussian_scene_scale(orbit_center)
            viewer_cameras = self._load_viewer_cameras(vggt_view_configs)
            playback_groups = build_modal_playback_groups(
                model.modal_frame_view_indices,
                model.modal_frame_local_indices,
                tuple(camera.label for camera in viewer_cameras),
            )
            modal_freqs_hz = ()
            if model.has_modal_field:
                modal_freqs_hz = tuple(
                    float(x) for x in model.modal_freqs_hz.detach().cpu().numpy()
                )
            elif modal_anchor_freqs_hz:
                modal_freqs_hz = modal_anchor_freqs_hz
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server,
                self.render_fn,
                model.num_frames,
                work_dir,
                mode="rendering",
                viewer_cameras=viewer_cameras,
                orbit_center=orbit_center,
                camera_frustum_scale=frustum_scale,
                playback_groups=playback_groups,
                modal_freqs_hz=modal_freqs_hz,
                has_modal_obs_count=model.has_modal_obs_count,
                modal_anchor_count=(
                    0
                    if self.modal_anchor_points is None
                    else int(self.modal_anchor_points.shape[0])
                ),
                modal_anchor_role_classes=self.modal_anchor_role_classes,
                modal_anchor_role_mode_labels=modal_anchor_role_mode_labels,
            )

        self.tracks_3d = self.model.compute_poses_fg(
            #  torch.arange(max(0, t - 20), max(1, t), device=self.device),
            torch.arange(self.num_frames, device=self.device),
            inds=torch.arange(10, device=self.device),
        )[0]

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, use_2dgs, *args, **kwargs
    ) -> "Renderer":
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, weights_only=False)
        state_dict = ckpt["model"]
        init_metadata = ckpt.get("init_metadata")
        shape_parameterization = (
            init_metadata.get("modal_shape_parameterization")
            if isinstance(init_metadata, dict)
            else None
        )
        has_delta_phi = MODAL_DELTA_PHI_STATE_KEY in state_dict
        has_refinement_mask = MODAL_REFINEMENT_MASK_STATE_KEY in state_dict
        has_refinement_role = MODAL_REFINEMENT_ROLE_STATE_KEY in state_dict
        has_frozen_envelope = (
            isinstance(init_metadata, dict)
            and init_metadata.get("modal_envelope_frozen") is True
        )
        envelope_source = (
            init_metadata.get("modal_envelope_init_ckpt")
            if isinstance(init_metadata, dict)
            else None
        )
        if shape_parameterization is None and not (
            has_delta_phi or has_refinement_mask or has_refinement_role
        ):
            pass
        elif (
            shape_parameterization == MODAL_SHAPE_PARAMETERIZATION
            and has_delta_phi
            and has_refinement_mask
            and has_refinement_role
            and has_frozen_envelope
            and isinstance(envelope_source, str)
            and bool(envelope_source)
        ):
            raise ValueError(
                "Viser rendering does not yet support Stage 3 refined modal "
                "shape checkpoints; inspect this checkpoint with "
                "run_modal_reconstruction.py"
            )
        else:
            raise ValueError(
                "Checkpoint has an incomplete or incompatible modal "
                "shape-refinement contract"
            )
        if "modal.params.activations" in state_dict:
            raise ValueError(
                "Constant per-view harmonic activation checkpoints are not "
                "supported; render a harmonic-envelope checkpoint"
            )
        if "modal.params.envelope_knots" in state_dict:
            if not isinstance(init_metadata, dict):
                raise ValueError("Envelope checkpoint metadata must be a mapping")
            required_envelope_keys = {
                "modal_frame_times_sec",
                "modal_envelope_knot_offsets",
                "modal_envelope_knot_times_sec",
                "modal_envelope_knot_interval_sec",
                "modal_frame_envelope_left",
                "modal_frame_envelope_right",
                "modal_frame_envelope_lerp",
            }
            missing_envelope_keys = sorted(required_envelope_keys - set(state_dict))
            if missing_envelope_keys:
                raise ValueError(
                    "Harmonic-envelope checkpoint is missing required state: "
                    f"{missing_envelope_keys}"
                )
            parameterization = (
                init_metadata.get("modal_parameterization")
                if isinstance(init_metadata, dict)
                else None
            )
            if parameterization != "per_view_harmonic_envelope_v2":
                raise ValueError(
                    "Checkpoint uses an incompatible modal parameterization "
                    f"({parameterization!r}); expected "
                    "'per_view_harmonic_envelope_v2'"
                )
            if (
                init_metadata.get("modal_envelope_interpolation")
                != "cubic_hermite_complex"
            ):
                raise ValueError(
                    "Checkpoint must use cubic_hermite_complex modal envelope "
                    "interpolation"
                )
            metadata_interval = init_metadata.get(
                "modal_envelope_knot_interval_sec"
            )
            state_interval = state_dict["modal_envelope_knot_interval_sec"]
            if (
                isinstance(metadata_interval, bool)
                or not isinstance(metadata_interval, (int, float))
                or not np.isfinite(float(metadata_interval))
                or float(metadata_interval) <= 0.0
                or not isinstance(state_interval, torch.Tensor)
                or state_interval.ndim != 0
                or not np.isclose(
                    float(state_interval.item()),
                    float(metadata_interval),
                    rtol=1.0e-6,
                    atol=1.0e-8,
                )
            ):
                raise ValueError(
                    "Checkpoint envelope interval state/metadata is invalid"
                )
        model = SceneModel.init_from_state_dict(state_dict)
        model.use_2dgs = use_2dgs
        model = model.to(device)
        print(f"num gs: {model.num_gaussians}")
        renderer = Renderer(model, device, *args, **kwargs)
        renderer.global_step = ckpt.get("global_step", 0)
        renderer.epoch = ckpt.get("epoch", 0)
        return renderer

    @torch.inference_mode()
    def _gaussian_orbit_center(self) -> np.ndarray:
        means = self.model.compute_poses_all(None)[0][:, 0]
        finite = torch.isfinite(means).all(dim=-1)
        if not bool(finite.any()):
            return np.zeros(3, dtype=np.float32)
        center = torch.median(means[finite], dim=0).values
        return center.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def _gaussian_scene_scale(self, center: np.ndarray) -> float:
        means = self.model.compute_poses_all(None)[0][:, 0]
        finite = torch.isfinite(means).all(dim=-1)
        if not bool(finite.any()):
            return 1.0
        center_th = torch.as_tensor(center, device=means.device, dtype=means.dtype)
        dists = torch.linalg.norm(means[finite] - center_th, dim=-1)
        if dists.numel() == 0:
            return 1.0
        scale = float(torch.quantile(dists, 0.9).detach().cpu().item())
        return max(scale, 1.0e-3)

    @staticmethod
    def _load_viewer_cameras(vggt_view_configs: tuple[str, ...]) -> tuple[ViewerCamera, ...]:
        viewer_cameras = []
        for path in vggt_view_configs:
            view_cfg = load_view_config(path)
            c2w = np.linalg.inv(view_cfg.world_to_camera)
            fov = float(2.0 * np.arctan(0.5 * view_cfg.image_height / view_cfg.K[1, 1]))
            aspect = float(view_cfg.image_width) / float(view_cfg.image_height)
            viewer_cameras.append(
                ViewerCamera(
                    label=view_cfg.view_id,
                    c2w=c2w.astype(np.float64),
                    fov=fov,
                    aspect=aspect,
                )
            )
        return tuple(viewer_cameras)

    def _load_modal_anchor_data(
        self, modal_anchor_manifest: str | None
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        tuple[float, ...],
        np.ndarray | None,
        tuple[str, ...],
    ]:
        if modal_anchor_manifest is None:
            return None, None, None, (), None, ()

        modes = load_modal_modes(modal_anchor_manifest)
        points = modes[0].points_world.astype(np.float32)
        for mode in modes[1:]:
            if mode.points_world.shape != points.shape or not np.allclose(
                mode.points_world, points, rtol=1e-4, atol=1e-5
            ):
                raise ValueError(
                    f"{modal_anchor_manifest} contains modes with different anchor points"
                )
        if self.model.has_modal_field and len(modes) != self.model.modal_phi_real.shape[0]:
            raise ValueError(
                f"{modal_anchor_manifest} has {len(modes)} modes but checkpoint has "
                f"{self.model.modal_phi_real.shape[0]} modal fields"
            )

        phi = np.stack([mode.phi for mode in modes], axis=0).astype(np.complex64)
        points_th = torch.as_tensor(points, device=self.device, dtype=torch.float32)
        phi_real = torch.as_tensor(phi.real, device=self.device, dtype=torch.float32)
        phi_imag = torch.as_tensor(phi.imag, device=self.device, dtype=torch.float32)
        freqs_hz = tuple(float(mode.freq_hz) for mode in modes)
        role_classes = stack_modal_motion_fill_display_classes(modes)
        role_mode_labels = (
            tuple(
                f"Mode {mode.mode_index}: {mode.freq_hz:.3f} Hz" for mode in modes
            )
            if role_classes is not None
            else ()
        )
        guru.info(
            f"Loaded {points.shape[0]} modal anchor points from {modal_anchor_manifest}"
        )
        return (
            points_th,
            phi_real,
            phi_imag,
            freqs_hz,
            role_classes,
            role_mode_labels,
        )

    def _current_modal_anchor_points(
        self, modal_oscillator: tuple[np.ndarray, float] | None
    ) -> torch.Tensor | None:
        if self.modal_anchor_points is None:
            return None
        points = self.modal_anchor_points
        if modal_oscillator is None:
            return points

        assert self.modal_anchor_phi_real is not None
        assert self.modal_anchor_phi_imag is not None
        q_np, motion_scale = modal_oscillator
        if q_np.shape[0] != self.modal_anchor_phi_real.shape[0]:
            raise ValueError(
                f"Modal oscillator has {q_np.shape[0]} modes but anchors have "
                f"{self.modal_anchor_phi_real.shape[0]}"
            )
        q = torch.from_numpy(q_np).to(self.device)
        q_real = q.real.to(dtype=self.modal_anchor_phi_real.dtype)
        q_imag = q.imag.to(dtype=self.modal_anchor_phi_imag.dtype)
        offsets = torch.einsum("k,knc->nc", q_real, self.modal_anchor_phi_real) - torch.einsum(
            "k,knc->nc", q_imag, self.modal_anchor_phi_imag
        )
        return points + offsets * torch.as_tensor(
            motion_scale, device=points.device, dtype=points.dtype
        )

    @staticmethod
    def _hsv_to_rgb(hue: torch.Tensor, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        hue = torch.remainder(hue, 1.0)
        h6 = hue * 6.0
        sector = torch.floor(h6).long()
        frac = h6 - sector.to(dtype=hue.dtype)
        p = value * (1.0 - saturation)
        q = value * (1.0 - saturation * frac)
        t = value * (1.0 - saturation * (1.0 - frac))
        sector = torch.remainder(sector, 6)

        rgb = torch.empty(hue.shape + (3,), device=hue.device, dtype=hue.dtype)
        rgb0 = torch.stack([value, t, p], dim=-1)
        rgb1 = torch.stack([q, value, p], dim=-1)
        rgb2 = torch.stack([p, value, t], dim=-1)
        rgb3 = torch.stack([p, q, value], dim=-1)
        rgb4 = torch.stack([t, p, value], dim=-1)
        rgb5 = torch.stack([value, p, q], dim=-1)
        for idx, candidate in enumerate((rgb0, rgb1, rgb2, rgb3, rgb4, rgb5)):
            rgb = torch.where((sector == idx)[..., None], candidate, rgb)
        return rgb

    def _modal_phase_colors(
        self,
        mode_index: int,
        component_index: int,
        w2c: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        if component_index not in (0, 1):
            raise ValueError(
                f"Projected phase component must be 0 for u or 1 for v, got {component_index}"
            )
        means = self.model.fg.params["means"]
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        points_cam = means @ R.T + t[None]
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]
        eps = torch.as_tensor(1.0e-6, device=z.device, dtype=z.dtype)
        z_safe = torch.where(
            z.abs() < eps,
            torch.where(z >= 0, eps, -eps),
            z,
        )
        fx = K[0, 0]
        fy = K[1, 1]
        J_cam = torch.zeros(
            (means.shape[0], 2, 3),
            device=means.device,
            dtype=means.dtype,
        )
        J_cam[:, 0, 0] = fx / z_safe
        J_cam[:, 0, 2] = -fx * x / (z_safe * z_safe)
        J_cam[:, 1, 1] = fy / z_safe
        J_cam[:, 1, 2] = -fy * y / (z_safe * z_safe)
        J = torch.einsum("nij,jk->nik", J_cam, R.to(dtype=means.dtype))
        phi_real = self.model.modal_phi_real[mode_index]
        phi_imag = self.model.modal_phi_imag[mode_index]
        projected_real = torch.einsum("nij,nj->ni", J, phi_real)
        projected_imag = torch.einsum("nij,nj->ni", J, phi_imag)
        real = projected_real[:, component_index]
        imag = projected_imag[:, component_index]
        amp = torch.sqrt(real.square() + imag.square())
        phase = torch.atan2(imag, real)
        finite = torch.isfinite(amp) & torch.isfinite(phase)
        finite_amp = amp[finite]
        if finite_amp.numel() == 0:
            value = torch.zeros_like(amp)
        else:
            hi = torch.quantile(finite_amp, 0.95).clamp_min(1.0e-8)
            value = torch.zeros_like(amp)
            value[finite] = (amp[finite] / hi).clamp(0.0, 1.0)
        phase = torch.where(torch.isfinite(phase), phase, torch.zeros_like(phase))
        hue = (phase + torch.pi) / (2.0 * torch.pi)
        saturation = torch.ones_like(value)
        return self._hsv_to_rgb(hue, saturation, value)

    def _modal_obs_count_colors(self, mode_index: int) -> torch.Tensor:
        obs_count = self.model.modal_obs_count_per_point[mode_index]
        colors = torch.full(
            (obs_count.shape[0], 3),
            0.15,
            device=obs_count.device,
            dtype=self.model.modal_phi_real.dtype,
        )
        colors[obs_count <= 1] = torch.tensor(
            [1.0, 0.43, 0.16], device=colors.device, dtype=colors.dtype
        )
        colors[obs_count == 2] = torch.tensor(
            [0.27, 0.55, 1.0], device=colors.device, dtype=colors.dtype
        )
        colors[obs_count >= 3] = torch.tensor(
            [0.27, 0.82, 0.47], device=colors.device, dtype=colors.dtype
        )
        colors[obs_count < 0] = torch.tensor(
            [0.5, 0.5, 0.5], device=colors.device, dtype=colors.dtype
        )
        return colors

    def _current_gaussian_color_override(
        self, w2c: torch.Tensor, K: torch.Tensor
    ) -> torch.Tensor | None:
        if self.viewer is None or not self.model.has_modal_field:
            return None
        color_mode, mode_index = self.viewer.current_gaussian_color_mode()
        if color_mode == "rgb":
            return None
        if mode_index < 0 or mode_index >= self.model.modal_phi_real.shape[0]:
            raise ValueError(
                f"Gaussian color mode index {mode_index} is outside "
                f"[0, {self.model.modal_phi_real.shape[0]})"
        )

        if color_mode == "modal phase":
            phase_mode_index, component_index = (
                self.viewer.current_gaussian_phase_component()
            )
            if phase_mode_index < 0 or phase_mode_index >= self.model.modal_phi_real.shape[0]:
                raise ValueError(
                    f"Gaussian phase mode index {phase_mode_index} is outside "
                    f"[0, {self.model.modal_phi_real.shape[0]})"
                )
            fg_colors = self._modal_phase_colors(
                phase_mode_index,
                component_index,
                w2c,
                K,
            )
        elif color_mode == "obs count":
            if not self.model.has_modal_obs_count:
                raise ValueError("Checkpoint does not contain modal obs_count_per_point")
            fg_colors = self._modal_obs_count_colors(mode_index)
        else:
            raise ValueError(f"Unknown Gaussian render color mode: {color_mode}")

        if not self.model.has_bg:
            return fg_colors
        colors = torch.full(
            (self.model.num_gaussians, 3),
            0.5,
            device=fg_colors.device,
            dtype=fg_colors.dtype,
        )
        colors[: self.model.num_fg_gaussians] = fg_colors
        return colors

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = self.viewer.current_timestep()
        self.model.training = False
        means = None
        quats = None
        render_t = t
        modal_oscillator = self.viewer.current_modal_oscillator()
        debug_points_update_key = ("canonical",) if t is None else ("frame", int(t))
        if modal_oscillator is not None and self.model.has_modal_field:
            q_np, motion_scale = modal_oscillator
            base_means, base_quats = self.model.compute_poses_all(None)
            means = base_means[:, 0].clone()
            quats = base_quats[:, 0]
            q = torch.from_numpy(q_np).to(self.device)
            fg_offsets = self.model.compute_synthetic_modal_offsets(q, motion_scale)
            means[: self.model.num_fg_gaussians] += fg_offsets
            render_t = None
            q_key = tuple(
                np.round(
                    np.stack([q_np.real, q_np.imag], axis=-1).reshape(-1),
                    6,
                ).tolist()
            )
            debug_points_update_key = (
                "oscillator",
                round(float(motion_scale), 6),
                q_key,
            )
        if self.viewer.wants_modal_anchors():
            anchor_points = self._current_modal_anchor_points(modal_oscillator)
            if anchor_points is not None:
                self.viewer.update_modal_anchors(
                    anchor_points.detach().cpu().numpy(),
                    debug_points_update_key,
                )
        if self.viewer.hide_gaussian_render():
            return np.full((H, W, 3), 255, dtype=np.uint8)
        colors_override = self._current_gaussian_color_override(w2c, K)
        img = self.model.render(
            render_t,
            w2c[None],
            K[None],
            img_wh,
            means=means,
            quats=quats,
            colors_override=colors_override,
        )["img"][0]
        render_track_checkbox = getattr(self.viewer, "_render_track_checkbox", None)
        render_tracks = bool(render_track_checkbox.value) if render_track_checkbox is not None else False
        if modal_oscillator is not None:
            render_tracks = False
        if not render_tracks:
            img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        else:
            assert t is not None
            tracks_3d = self.tracks_3d[:, max(0, t - 20) : max(1, t)]
            tracks_2d = torch.einsum(
                "ij,jk,nbk->nbi", K, w2c[:3], F.pad(tracks_3d, (0, 1), value=1.0)
            )
            tracks_2d = tracks_2d[..., :2] / tracks_2d[..., 2:]
            img = draw_tracks_2d_th(img, tracks_2d)
        return img
