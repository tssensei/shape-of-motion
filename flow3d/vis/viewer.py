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
        modal_freqs_hz: tuple[float, ...] = (),
        has_modal_obs_count: bool = False,
        gaussian_center_count: int = 0,
        modal_anchor_count: int = 0,
    ):
        self.num_frames = num_frames
        self.work_dir = Path(work_dir)
        self.viewer_cameras = tuple(viewer_cameras)
        self.orbit_center = None if orbit_center is None else np.asarray(orbit_center, dtype=np.float32)
        self.camera_frustum_scale = float(camera_frustum_scale)
        self.playback_groups = tuple(playback_groups)
        self.modal_freqs_hz = tuple(float(freq) for freq in modal_freqs_hz)
        self.has_modal_obs_count = bool(has_modal_obs_count)
        self.gaussian_center_count = int(gaussian_center_count)
        self.modal_anchor_count = int(modal_anchor_count)
        self._gaussian_center_handle = None
        self._modal_anchor_handle = None
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
        self._define_modal_playback_guis()
        self._define_gaussian_color_guis()
        self._define_debug_point_guis()
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

    def _define_modal_playback_guis(self) -> None:
        self._modal_playback_handles = None
        if not self.modal_freqs_hz:
            return
        with self.server.gui.add_folder("Modal playback"):
            drive = self.server.gui.add_dropdown(
                "Drive",
                options=("static", "oscillator"),
                initial_value="static",
            )
            motion_scale = self.server.gui.add_slider(
                "Motion scale",
                min=0.0,
                max=1.0,
                step=0.001,
                initial_value=0.1,
            )
            modes = []
            for mode_idx, freq_hz in enumerate(self.modal_freqs_hz):
                enabled = self.server.gui.add_checkbox(f"Mode {mode_idx} enable", True)
                gain = self.server.gui.add_slider(
                    f"Mode {mode_idx} gain",
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    initial_value=1.0,
                )
                phase = self.server.gui.add_slider(
                    f"Mode {mode_idx} phase",
                    min=-np.pi,
                    max=np.pi,
                    step=0.01,
                    initial_value=0.0,
                )
                modes.append(
                    {
                        "enabled": enabled,
                        "gain": gain,
                        "phase": phase,
                        "freq_hz": float(freq_hz),
                    }
                )
                enabled.on_update(self.rerender)
                gain.on_update(self.rerender)
                phase.on_update(self.rerender)
            drive.on_update(self.rerender)
            motion_scale.on_update(self.rerender)
        self._modal_playback_handles = {
            "drive": drive,
            "motion_scale": motion_scale,
            "modes": tuple(modes),
        }

    def _define_gaussian_color_guis(self) -> None:
        self._gaussian_color_handles = None
        if not self.modal_freqs_hz:
            return
        options = ["rgb", "modal phase"]
        if self.has_modal_obs_count:
            options.append("obs count")
        phase_component_options = []
        for mode_index, freq_hz in enumerate(self.modal_freqs_hz):
            phase_component_options.append(
                f"Mode {mode_index} {freq_hz:.3f} Hz projected u"
            )
            phase_component_options.append(
                f"Mode {mode_index} {freq_hz:.3f} Hz projected v"
            )
        with self.server.gui.add_folder("Gaussian color"):
            color_mode = self.server.gui.add_dropdown(
                "Render color mode",
                options=tuple(options),
                initial_value="rgb",
            )
            phase_component = self.server.gui.add_dropdown(
                "Phase component",
                options=tuple(phase_component_options),
                initial_value=phase_component_options[0],
            )
            mode_index = self.server.gui.add_slider(
                "Obs count mode index",
                min=0,
                max=max(len(self.modal_freqs_hz) - 1, 0),
                step=1,
                initial_value=0,
            )
        self._gaussian_color_handles = {
            "color_mode": color_mode,
            "phase_component": phase_component,
            "phase_component_options": tuple(phase_component_options),
            "mode_index": mode_index,
        }
        color_mode.on_update(self.rerender)
        phase_component.on_update(self.rerender)
        mode_index.on_update(self.rerender)

    def _define_debug_point_guis(self) -> None:
        self._debug_point_handles = None
        if self.gaussian_center_count <= 0 and self.modal_anchor_count <= 0:
            return
        max_gaussian_count = max(int(self.gaussian_center_count), 1)
        gaussian_step = max(max_gaussian_count // 200, 1)
        max_anchor_count = max(int(self.modal_anchor_count), 1)
        anchor_step = max(max_anchor_count // 200, 1)
        with self.server.gui.add_folder("Debug points"):
            hide_render = self.server.gui.add_checkbox("Hide Gaussian render", False)
            show_centers = self.server.gui.add_checkbox("Show Gaussian centers", False)
            fg_only = self.server.gui.add_checkbox("Foreground only", True)
            center_count = self.server.gui.add_slider(
                "Center visible count",
                min=0,
                max=max_gaussian_count,
                step=gaussian_step,
                initial_value=min(2000, max_gaussian_count),
            )
            center_point_size = self.server.gui.add_slider(
                "Center point size",
                min=0.0002,
                max=0.008,
                step=0.0001,
                initial_value=0.002,
            )
            show_anchors = None
            anchor_count = None
            anchor_point_size = None
            if self.modal_anchor_count > 0:
                show_anchors = self.server.gui.add_checkbox("Show modal anchors", False)
                anchor_count = self.server.gui.add_slider(
                    "Anchor visible count",
                    min=0,
                    max=max_anchor_count,
                    step=anchor_step,
                    initial_value=min(5000, max_anchor_count),
                )
                anchor_point_size = self.server.gui.add_slider(
                    "Anchor point size",
                    min=0.0002,
                    max=0.008,
                    step=0.0001,
                    initial_value=0.002,
                )
        self._debug_point_handles = {
            "hide_render": hide_render,
            "show_centers": show_centers,
            "fg_only": fg_only,
            "center_count": center_count,
            "center_point_size": center_point_size,
            "show_anchors": show_anchors,
            "anchor_count": anchor_count,
            "anchor_point_size": anchor_point_size,
        }

        def _on_update(event) -> None:
            if not bool(show_centers.value):
                self._remove_gaussian_center_cloud()
            if show_anchors is not None and not bool(show_anchors.value):
                self._remove_modal_anchor_cloud()
            self.rerender(event)

        hide_render.on_update(_on_update)
        show_centers.on_update(_on_update)
        fg_only.on_update(_on_update)
        center_count.on_update(_on_update)
        center_point_size.on_update(_on_update)
        if show_anchors is not None:
            show_anchors.on_update(_on_update)
        if anchor_count is not None:
            anchor_count.on_update(_on_update)
        if anchor_point_size is not None:
            anchor_point_size.on_update(_on_update)

    def _remove_gaussian_center_cloud(self) -> None:
        if self._gaussian_center_handle is not None:
            self._gaussian_center_handle.remove()
            self._gaussian_center_handle = None

    def _remove_modal_anchor_cloud(self) -> None:
        if self._modal_anchor_handle is not None:
            self._modal_anchor_handle.remove()
            self._modal_anchor_handle = None

    def hide_gaussian_render(self) -> bool:
        handles = getattr(self, "_debug_point_handles", None)
        return handles is not None and bool(handles["hide_render"].value)

    def wants_gaussian_centers(self) -> bool:
        handles = getattr(self, "_debug_point_handles", None)
        return handles is not None and bool(handles["show_centers"].value)

    def wants_modal_anchors(self) -> bool:
        handles = getattr(self, "_debug_point_handles", None)
        return (
            handles is not None
            and handles["show_anchors"] is not None
            and bool(handles["show_anchors"].value)
        )

    def current_gaussian_color_mode(self) -> tuple[str, int]:
        handles = getattr(self, "_gaussian_color_handles", None)
        if handles is None:
            return "rgb", 0
        return str(handles["color_mode"].value), int(handles["mode_index"].value)

    def current_gaussian_phase_component(self) -> tuple[int, int]:
        handles = getattr(self, "_gaussian_color_handles", None)
        if handles is None:
            return 0, 0
        options = handles["phase_component_options"]
        selected = str(handles["phase_component"].value)
        if selected not in options:
            raise ValueError(f"Unknown Gaussian phase component: {selected}")
        flat_index = options.index(selected)
        return flat_index // 2, flat_index % 2

    def update_gaussian_centers(self, points: np.ndarray, fg_count: int) -> None:
        if not self.wants_gaussian_centers():
            return
        handles = self._debug_point_handles
        assert handles is not None

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Gaussian centers must have shape (N,3), got {points.shape}"
            )

        fg_count = min(max(int(fg_count), 0), points.shape[0])
        if bool(handles["fg_only"].value):
            selectable = np.arange(fg_count, dtype=np.int64)
        else:
            selectable = np.arange(points.shape[0], dtype=np.int64)

        visible_count = min(
            max(int(handles["center_count"].value), 0), selectable.shape[0]
        )
        self._remove_gaussian_center_cloud()
        if visible_count == 0:
            return

        selected = selectable[:visible_count]
        colors = np.empty((visible_count, 3), dtype=np.float32)
        fg_mask = selected < fg_count
        colors[fg_mask] = np.asarray([0.05, 0.85, 0.20], dtype=np.float32)
        colors[~fg_mask] = np.asarray([0.55, 0.55, 0.55], dtype=np.float32)
        self._gaussian_center_handle = self.server.scene.add_point_cloud(
            "/debug/gaussian_centers",
            points=points[selected],
            colors=colors,
            point_size=float(handles["center_point_size"].value),
        )

    def update_modal_anchors(self, points: np.ndarray) -> None:
        if not self.wants_modal_anchors():
            return
        handles = self._debug_point_handles
        assert handles is not None
        assert handles["anchor_count"] is not None
        assert handles["anchor_point_size"] is not None

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Modal anchors must have shape (N,3), got {points.shape}")

        visible_count = min(max(int(handles["anchor_count"].value), 0), points.shape[0])
        self._remove_modal_anchor_cloud()
        if visible_count == 0:
            return

        selected = np.arange(visible_count, dtype=np.int64)
        colors = np.full((visible_count, 3), [0.05, 0.55, 1.0], dtype=np.float32)
        self._modal_anchor_handle = self.server.scene.add_point_cloud(
            "/debug/modal_anchors",
            points=points[selected],
            colors=colors,
            point_size=float(handles["anchor_point_size"].value),
        )

    def current_modal_oscillator(self) -> tuple[np.ndarray, float] | None:
        handles = getattr(self, "_modal_playback_handles", None)
        if handles is None or str(handles["drive"].value) != "oscillator":
            return None
        if not hasattr(self, "_playback_guis"):
            timestep = 0
            fps = 15.0
        else:
            timestep = int(self._playback_guis[0].value)
            fps = max(float(self._playback_guis[5].value), 1.0e-6)
        time_s = float(timestep) / fps
        q_values = []
        for mode in handles["modes"]:
            if bool(mode["enabled"].value):
                amp = float(mode["gain"].value)
            else:
                amp = 0.0
            phase = 2.0 * np.pi * float(mode["freq_hz"]) * time_s + float(mode["phase"].value)
            q_values.append(amp * np.exp(1j * phase))
        return np.asarray(q_values, dtype=np.complex64), float(handles["motion_scale"].value)

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

    def _reset_client_orbit_center(self, event) -> None:
        if self.orbit_center is None or event.client is None:
            return
        event.client.camera.look_at = self.orbit_center
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
            reset_orbit = self.server.gui.add_button("Reset orbit center")
            reset_orbit.on_click(self._reset_client_orbit_center)
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
