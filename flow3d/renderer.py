import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState

from flow3d.scene_model import SceneModel
from flow3d.vis.utils import draw_tracks_2d_th, get_server
from flow3d.vis.viewer import (
    DynamicViewer,
    ViewerCamera,
    build_modal_playback_groups,
)
from modal_surface.io import load_view_config


class Renderer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        # Logging.
        work_dir: str,
        port: int | None = None,
        vggt_view_configs: tuple[str, ...] = (),
    ):
        self.device = device

        self.model = model
        self.num_frames = model.num_frames

        self.work_dir = work_dir
        self.global_step = 0
        self.epoch = 0

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
                gaussian_center_count=model.num_gaussians,
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
        if modal_oscillator is not None:
            q_np, motion_scale = modal_oscillator
            base_means, base_quats = self.model.compute_poses_all(None)
            means = base_means[:, 0].clone()
            quats = base_quats[:, 0]
            q = torch.from_numpy(q_np).to(self.device)
            fg_offsets = self.model.compute_synthetic_modal_offsets(q, motion_scale)
            means[: self.model.num_fg_gaussians] += fg_offsets
            render_t = None
        if self.viewer.wants_gaussian_centers():
            center_means = means
            if center_means is None:
                center_ts = (
                    torch.tensor([t], device=self.device) if t is not None else None
                )
                center_means = self.model.compute_poses_all(center_ts)[0][:, 0]
            self.viewer.update_gaussian_centers(
                center_means.detach().cpu().numpy(),
                self.model.num_fg_gaussians,
            )
        img = self.model.render(
            render_t,
            w2c[None],
            K[None],
            img_wh,
            means=means,
            quats=quats,
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
