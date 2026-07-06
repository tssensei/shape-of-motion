from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from jaxtyping import Float32, UInt8
from nerfview import CameraState, Viewer
from viser import Icon, ViserServer
import viser.transforms as vtf

from flow3d.vis.playback_panel import add_gui_playback_group
from flow3d.vis.render_panel import populate_render_tab


@dataclass(frozen=True)
class ViewerCamera:
    label: str
    c2w: np.ndarray
    fov: float
    aspect: float


@dataclass(frozen=True)
class ViewerPlaybackGroup:
    label: str
    global_timestamps: tuple[int, ...]


def _to_int_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64)


def build_modal_playback_groups(
    frame_view_indices,
    frame_local_indices,
    view_labels: tuple[str, ...] = (),
) -> tuple[ViewerPlaybackGroup, ...]:
    if frame_view_indices is None or frame_local_indices is None:
        return ()
    view_indices = _to_int_numpy(frame_view_indices)
    local_indices = _to_int_numpy(frame_local_indices)
    if view_indices.shape != local_indices.shape:
        raise ValueError("Modal playback view/local index buffers must have matching shape")

    valid_mask = (view_indices >= 0) & (local_indices >= 0)
    if not bool(valid_mask.any()):
        return ()

    groups = []
    for view_index in sorted(int(v) for v in np.unique(view_indices[valid_mask])):
        ts_indices = np.nonzero(valid_mask & (view_indices == view_index))[0]
        items = sorted((int(local_indices[ts]), int(ts)) for ts in ts_indices)
        if not items:
            continue
        label = (
            view_labels[view_index]
            if view_index < len(view_labels)
            else f"view{view_index + 1}"
        )
        groups.append(
            ViewerPlaybackGroup(
                label=label,
                global_timestamps=tuple(ts for _, ts in items),
            )
        )
    return tuple(groups)


class DynamicViewer(Viewer):
    def __init__(
        self,
        server: ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int]],
            Union[
                UInt8[np.ndarray, "H W 3"],
                Tuple[UInt8[np.ndarray, "H W 3"], Optional[Float32[np.ndarray, "H W"]]],
            ],
        ],
        num_frames: int,
        work_dir: str,
        mode: Literal["rendering", "training"] = "rendering",
        viewer_cameras: tuple[ViewerCamera, ...] = (),
        orbit_center: np.ndarray | None = None,
        camera_frustum_scale: float = 1.0,
        playback_groups: tuple[ViewerPlaybackGroup, ...] = (),
    ):
        self.num_frames = num_frames
        self.work_dir = Path(work_dir)
        self.viewer_cameras = tuple(viewer_cameras)
        self.orbit_center = None if orbit_center is None else np.asarray(orbit_center, dtype=np.float32)
        self.camera_frustum_scale = float(camera_frustum_scale)
        self.playback_groups = tuple(playback_groups)
        for group in self.playback_groups:
            if len(group.global_timestamps) == 0:
                raise ValueError(f"Playback group {group.label!r} has no frames")
        self._playback_group_by_label = {
            group.label: group for group in self.playback_groups
        }
        if len(self._playback_group_by_label) != len(self.playback_groups):
            raise ValueError("Playback group labels must be unique")
        self._active_playback_group_label = (
            self.playback_groups[0].label if self.playback_groups else None
        )
        super().__init__(server, render_fn, mode)

    def _define_guis(self):
        super()._define_guis()
        server = self.server
        self._set_default_orbit_center()
        self._time_folder = server.gui.add_folder("Time")
        with self._time_folder:
            self._playback_view_dropdown = None
            if self.playback_groups:
                self._playback_view_dropdown = server.gui.add_dropdown(
                    "Playback view",
                    options=[group.label for group in self.playback_groups],
                    initial_value=self.playback_groups[0].label,
                )
            self._playback_guis = add_gui_playback_group(
                server,
                num_frames=self.num_frames,
                initial_fps=15.0,
                num_frames_getter=self._playback_num_frames,
            )
            self._playback_guis[0].on_update(self.rerender)
            if self._playback_view_dropdown is not None:
                self._playback_view_dropdown.on_update(self._on_playback_view_update)
            self._canonical_checkbox = server.gui.add_checkbox("Canonical", False)
            self._canonical_checkbox.on_update(self.rerender)

            _cached_playback_disabled = []

            def _toggle_gui_playing(event):
                if event.target.value:
                    nonlocal _cached_playback_disabled
                    playback_handles = list(self._playback_guis)
                    if self._playback_view_dropdown is not None:
                        playback_handles.append(self._playback_view_dropdown)
                    _cached_playback_disabled = [gui.disabled for gui in playback_handles]
                    target_disabled = [True] * len(playback_handles)
                else:
                    playback_handles = list(self._playback_guis)
                    if self._playback_view_dropdown is not None:
                        playback_handles.append(self._playback_view_dropdown)
                    target_disabled = _cached_playback_disabled
                for gui, disabled in zip(playback_handles, target_disabled):
                    gui.disabled = disabled

            self._canonical_checkbox.on_update(_toggle_gui_playing)

        self._render_track_checkbox = server.gui.add_checkbox("Render tracks", False)
        self._render_track_checkbox.on_update(self.rerender)
        self._define_camera_guis()

        tabs = server.gui.add_tab_group()
        with tabs.add_tab("Render", Icon.CAMERA):
            self.render_tab_state = populate_render_tab(
                server, Path(self.work_dir) / "camera_paths", self._playback_guis[0]
            )

    def _active_playback_group(self) -> ViewerPlaybackGroup | None:
        if self._active_playback_group_label is None:
            return None
        return self._playback_group_by_label[self._active_playback_group_label]

    def _playback_num_frames(self) -> int:
        group = self._active_playback_group()
        if group is None:
            return self.num_frames
        return len(group.global_timestamps)

    def _sync_timestep_slider_bounds(self) -> None:
        gui_timestep = self._playback_guis[0]
        max_timestep = self._playback_num_frames() - 1
        if hasattr(gui_timestep, "max"):
            gui_timestep.max = max_timestep
        if int(gui_timestep.value) > max_timestep:
            gui_timestep.value = max_timestep

    def _on_playback_view_update(self, event) -> None:
        self._active_playback_group_label = str(event.target.value)
        self._sync_timestep_slider_bounds()
        self.rerender(event)

    def current_timestep(self) -> int | None:
        canonical = (
            hasattr(self, "_canonical_checkbox")
            and bool(self._canonical_checkbox.value)
        )
        if canonical:
            return None
        if not hasattr(self, "_playback_guis"):
            return 0
        local_t = int(self._playback_guis[0].value)
        group = self._active_playback_group()
        if group is None:
            return local_t
        local_t = min(max(local_t, 0), len(group.global_timestamps) - 1)
        return int(group.global_timestamps[local_t])

    def _set_default_orbit_center(self) -> None:
        if self.orbit_center is None or not hasattr(self.server, "on_client_connect"):
            return

        @self.server.on_client_connect
        def _(client) -> None:
            client.camera.look_at = self.orbit_center

    @staticmethod
    def _camera_pose_fields(camera: ViewerCamera) -> tuple[np.ndarray, np.ndarray]:
        return (
            vtf.SO3.from_matrix(camera.c2w[:3, :3]).wxyz,
            camera.c2w[:3, 3],
        )

    def _set_client_to_camera(self, event, camera: ViewerCamera) -> None:
        if event.client is None:
            return
        wxyz, position = self._camera_pose_fields(camera)
        with event.client.atomic():
            event.client.camera.position = position
            if self.orbit_center is not None:
                event.client.camera.look_at = self.orbit_center
            event.client.camera.wxyz = wxyz
            event.client.camera.fov = camera.fov
        self.rerender(event)

    def _define_camera_guis(self) -> None:
        if len(self.viewer_cameras) == 0:
            return

        camera_colors = [
            (80, 150, 255),
            (255, 130, 70),
            (95, 200, 120),
            (210, 120, 255),
            (255, 210, 80),
            (80, 220, 220),
        ]
        camera_handles = {}
        self._camera_folder = self.server.gui.add_folder("VGGT Cameras")
        with self._camera_folder:
            show_cameras = self.server.gui.add_checkbox("Show cameras", True)
            for i, camera in enumerate(self.viewer_cameras):
                wxyz, position = self._camera_pose_fields(camera)
                camera_handles[camera.label] = self.server.scene.add_camera_frustum(
                    f"/vggt_cameras/{camera.label}",
                    fov=camera.fov,
                    aspect=camera.aspect,
                    scale=self.camera_frustum_scale,
                    color=camera_colors[i % len(camera_colors)],
                    wxyz=wxyz,
                    position=position,
                )
                button = self.server.gui.add_button(f"Go to {camera.label}")

                def _go_to_camera(event, camera=camera) -> None:
                    self._set_client_to_camera(event, camera)

                button.on_click(_go_to_camera)

            def _toggle_cameras(event) -> None:
                for handle in camera_handles.values():
                    handle.visible = bool(event.target.value)

            show_cameras.on_update(_toggle_cameras)
