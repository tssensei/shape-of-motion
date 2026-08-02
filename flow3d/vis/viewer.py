from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from jaxtyping import Float32, UInt8
from nerfview import CameraState, RenderTabState, Viewer
from viser import Icon, ViserServer
import viser.transforms as vtf

from flow3d.modal_utils import (
    MOTION_FILL_DISPLAY_NAMES,
    motion_fill_display_colors,
    select_motion_fill_display_indices,
)
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
            [CameraState, RenderTabState],
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
        has_background: bool = False,
        modal_anchor_count: int = 0,
        modal_anchor_role_classes: np.ndarray | None = None,
        modal_anchor_role_mode_labels: tuple[str, ...] = (),
    ):
        self.num_frames = num_frames
        self.work_dir = Path(work_dir)
        self.viewer_cameras = tuple(viewer_cameras)
        self.orbit_center = None if orbit_center is None else np.asarray(orbit_center, dtype=np.float32)
        self.camera_frustum_scale = float(camera_frustum_scale)
        self.playback_groups = tuple(playback_groups)
        self.modal_freqs_hz = tuple(float(freq) for freq in modal_freqs_hz)
        self.has_modal_obs_count = bool(has_modal_obs_count)
        self._enable_hide_gaussian_render = mode == "rendering"
        self._enable_hide_background = mode == "rendering" and bool(has_background)
        self.modal_anchor_count = int(modal_anchor_count)
        self.modal_anchor_role_mode_labels = tuple(modal_anchor_role_mode_labels)
        self.modal_anchor_role_classes = None
        if modal_anchor_role_classes is not None:
            role_classes = np.asarray(modal_anchor_role_classes)
            expected_shape = (
                len(self.modal_anchor_role_mode_labels),
                self.modal_anchor_count,
            )
            if role_classes.shape != expected_shape:
                raise ValueError(
                    "Modal anchor role classes must have shape "
                    f"{expected_shape}, got {role_classes.shape}"
                )
            if not np.issubdtype(role_classes.dtype, np.integer):
                raise ValueError(
                    "Modal anchor role classes must have integer dtype, "
                    f"got {role_classes.dtype}"
                )
            if len(set(self.modal_anchor_role_mode_labels)) != len(
                self.modal_anchor_role_mode_labels
            ):
                raise ValueError("Modal anchor role mode labels must be unique")
            if np.any(role_classes < 0) or np.any(
                role_classes >= len(MOTION_FILL_DISPLAY_NAMES)
            ):
                raise ValueError(
                    "Modal anchor role classes must lie in "
                    f"[0,{len(MOTION_FILL_DISPLAY_NAMES) - 1}]"
                )
            self.modal_anchor_role_classes = role_classes.astype(
                np.int8, copy=False
            )
        elif self.modal_anchor_role_mode_labels:
            raise ValueError(
                "Modal anchor role mode labels require modal anchor role classes"
            )
        self._modal_anchor_handle = None
        self._modal_anchor_cache_key = None
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
        super().__init__(
            server=server,
            render_fn=render_fn,
            output_dir=self.work_dir,
            mode=mode,
        )

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            viewer_res_slider = server.gui.add_slider(
                "Viewer Res",
                min=64,
                max=2048,
                step=1,
                initial_value=self.render_tab_state.viewer_res,
                hint="Maximum resolution of the viewer rendered image.",
            )

            @viewer_res_slider.on_update
            def _(_) -> None:
                self.render_tab_state.viewer_res = int(viewer_res_slider.value)
                self.rerender(_)

        self._rendering_tab_handles["viewer_res_slider"] = viewer_res_slider
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
            self._camera_path_render_tab_state = populate_render_tab(
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
                options=("flow-derived coordinates", "manual oscillator"),
                initial_value="flow-derived coordinates",
            )
            motion_scale = self.server.gui.add_slider(
                "Motion scale",
                min=0.0,
                max=1.0,
                step=0.001,
                initial_value=0.04,
            )
            disable_all_modes = self.server.gui.add_button("Turn off all modes")
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

            def _disable_all_modes(event) -> None:
                for mode in modes:
                    mode["enabled"].value = False
                self.rerender(event)

            disable_all_modes.on_click(_disable_all_modes)
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
        with self.server.gui.add_folder("Gaussian color"):
            color_mode = self.server.gui.add_dropdown(
                "Render color mode",
                options=tuple(options),
                initial_value="rgb",
            )
            phase_mode_index = self.server.gui.add_slider(
                "Phase frequency index",
                min=0,
                max=len(self.modal_freqs_hz) - 1,
                step=1,
                initial_value=0,
            )
            phase_frequency = self.server.gui.add_number(
                "Selected frequency (Hz)",
                initial_value=self.modal_freqs_hz[0],
                disabled=True,
            )
            phase_direction = self.server.gui.add_dropdown(
                "Projection direction",
                options=("u", "v"),
                initial_value="u",
            )
            phase_amplitude_normalization = self.server.gui.add_dropdown(
                "Amplitude normalization",
                options=("per mode", "entire spectrum"),
                initial_value="per mode",
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
            "phase_mode_index": phase_mode_index,
            "phase_frequency": phase_frequency,
            "phase_direction": phase_direction,
            "phase_amplitude_normalization": phase_amplitude_normalization,
            "mode_index": mode_index,
        }

        def update_phase_mode(event) -> None:
            selected_index = int(phase_mode_index.value)
            phase_frequency.value = self.modal_freqs_hz[selected_index]
            self.rerender(event)

        color_mode.on_update(self.rerender)
        phase_mode_index.on_update(update_phase_mode)
        phase_direction.on_update(self.rerender)
        phase_amplitude_normalization.on_update(self.rerender)
        mode_index.on_update(self.rerender)

    def _define_debug_point_guis(self) -> None:
        self._debug_point_handles = None
        if not self._enable_hide_gaussian_render and self.modal_anchor_count <= 0:
            return
        max_anchor_count = max(int(self.modal_anchor_count), 1)
        anchor_step = max(max_anchor_count // 200, 1)
        with self.server.gui.add_folder("Debug points"):
            hide_render = (
                self.server.gui.add_checkbox("Hide Gaussian render", False)
                if self._enable_hide_gaussian_render
                else None
            )
            hide_background = (
                self.server.gui.add_checkbox("Hide background", False)
                if self._enable_hide_background
                else None
            )
            show_anchors = None
            anchor_count = None
            anchor_point_size = None
            anchor_role_mode = None
            anchor_role_filters = ()
            if self.modal_anchor_count > 0:
                role_coloring = self.modal_anchor_role_classes is not None
                show_anchors = self.server.gui.add_checkbox(
                    (
                        "Show modal points by role"
                        if role_coloring
                        else "Show modal anchors"
                    ),
                    False,
                )
                anchor_count = self.server.gui.add_slider(
                    (
                        "Modal point visible count"
                        if role_coloring
                        else "Anchor visible count"
                    ),
                    min=0,
                    max=max_anchor_count,
                    step=anchor_step,
                    initial_value=min(5000, max_anchor_count),
                )
                anchor_point_size = self.server.gui.add_slider(
                    "Modal point size" if role_coloring else "Anchor point size",
                    min=0.0002,
                    max=0.008,
                    step=0.0001,
                    initial_value=0.002,
                )
                if role_coloring:
                    self.server.gui.add_markdown(
                        "**Role colors:** anchor = blue | partial = orange | "
                        "filled = green | unobserved = purple | excluded = gray"
                    )
                    if len(self.modal_anchor_role_mode_labels) > 1:
                        anchor_role_mode = self.server.gui.add_dropdown(
                            "Modal role mode",
                            options=self.modal_anchor_role_mode_labels,
                            initial_value=self.modal_anchor_role_mode_labels[0],
                        )
                    anchor_role_filters = tuple(
                        self.server.gui.add_checkbox(name.capitalize(), True)
                        for name in MOTION_FILL_DISPLAY_NAMES
                    )
        self._debug_point_handles = {
            "hide_render": hide_render,
            "hide_background": hide_background,
            "show_anchors": show_anchors,
            "anchor_count": anchor_count,
            "anchor_point_size": anchor_point_size,
            "anchor_role_mode": anchor_role_mode,
            "anchor_role_filters": anchor_role_filters,
        }

        def _on_update(event) -> None:
            if show_anchors is not None and not bool(show_anchors.value):
                self._remove_modal_anchor_cloud()
            self.rerender(event)

        if hide_render is not None:
            hide_render.on_update(_on_update)
        if hide_background is not None:
            hide_background.on_update(_on_update)
        if show_anchors is not None:
            show_anchors.on_update(_on_update)
        if anchor_count is not None:
            anchor_count.on_update(_on_update)
        if anchor_point_size is not None:
            anchor_point_size.on_update(_on_update)
        if anchor_role_mode is not None:
            anchor_role_mode.on_update(_on_update)
        for role_filter in anchor_role_filters:
            role_filter.on_update(_on_update)

    def _remove_modal_anchor_cloud(self) -> None:
        if self._modal_anchor_handle is not None:
            self._modal_anchor_handle.remove()
            self._modal_anchor_handle = None
        self._modal_anchor_cache_key = None

    def hide_gaussian_render(self) -> bool:
        handles = getattr(self, "_debug_point_handles", None)
        return (
            handles is not None
            and handles["hide_render"] is not None
            and bool(handles["hide_render"].value)
        )

    def hide_background(self) -> bool:
        handles = getattr(self, "_debug_point_handles", None)
        return (
            handles is not None
            and handles["hide_background"] is not None
            and bool(handles["hide_background"].value)
        )

    def wants_modal_anchors(self) -> bool:
        handles = getattr(self, "_debug_point_handles", None)
        return (
            handles is not None
            and handles["show_anchors"] is not None
            and bool(handles["show_anchors"].value)
        )

    def _modal_anchor_role_mode_index(self) -> int:
        if self.modal_anchor_role_classes is None:
            return 0
        handles = self._debug_point_handles
        assert handles is not None
        role_mode = handles["anchor_role_mode"]
        if role_mode is None:
            return 0
        selected = str(role_mode.value)
        if selected not in self.modal_anchor_role_mode_labels:
            raise ValueError(f"Unknown modal role mode: {selected}")
        return self.modal_anchor_role_mode_labels.index(selected)

    def current_gaussian_color_mode(self) -> tuple[str, int]:
        handles = getattr(self, "_gaussian_color_handles", None)
        if handles is None:
            return "rgb", 0
        return str(handles["color_mode"].value), int(handles["mode_index"].value)

    def current_gaussian_phase_component(self) -> tuple[int, int]:
        handles = getattr(self, "_gaussian_color_handles", None)
        if handles is None:
            return 0, 0
        mode_index = int(handles["phase_mode_index"].value)
        direction = str(handles["phase_direction"].value)
        if direction not in ("u", "v"):
            raise ValueError(f"Unknown Gaussian phase projection direction: {direction}")
        return mode_index, (0 if direction == "u" else 1)

    def current_gaussian_phase_amplitude_normalization(self) -> str:
        handles = getattr(self, "_gaussian_color_handles", None)
        if handles is None:
            return "per mode"
        normalization = str(handles["phase_amplitude_normalization"].value)
        if normalization not in ("per mode", "entire spectrum"):
            raise ValueError(
                f"Unknown Gaussian phase amplitude normalization: {normalization}"
            )
        return normalization

    def update_modal_anchors(self, points: np.ndarray, update_key) -> None:
        if not self.wants_modal_anchors():
            return
        handles = self._debug_point_handles
        assert handles is not None
        assert handles["anchor_count"] is not None
        assert handles["anchor_point_size"] is not None

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Modal anchors must have shape (N,3), got {points.shape}")

        requested_count = max(int(handles["anchor_count"].value), 0)
        role_mode_index = self._modal_anchor_role_mode_index()
        role_filter_key = ()
        if self.modal_anchor_role_classes is None:
            visible_count = min(requested_count, points.shape[0])
            selected = np.arange(visible_count, dtype=np.int64)
        else:
            role_filters = handles["anchor_role_filters"]
            enabled_classes = np.asarray(
                [bool(role_filter.value) for role_filter in role_filters],
                dtype=bool,
            )
            role_filter_key = tuple(bool(value) for value in enabled_classes)
            selected = select_motion_fill_display_indices(
                self.modal_anchor_role_classes[role_mode_index],
                enabled_classes,
                requested_count,
            )
            visible_count = int(selected.shape[0])
        if visible_count == 0:
            self._remove_modal_anchor_cloud()
            return

        point_size = float(handles["anchor_point_size"].value)
        cache_key = (
            visible_count,
            point_size,
            role_mode_index,
            role_filter_key,
            update_key,
        )
        if (
            self._modal_anchor_handle is not None
            and self._modal_anchor_cache_key == cache_key
        ):
            return

        self._remove_modal_anchor_cloud()
        if self.modal_anchor_role_classes is None:
            colors = np.full(
                (visible_count, 3), [0.05, 0.55, 1.0], dtype=np.float32
            )
        else:
            colors = motion_fill_display_colors(
                self.modal_anchor_role_classes[role_mode_index, selected]
            )
        self._modal_anchor_handle = self.server.scene.add_point_cloud(
            "/debug/modal_anchors",
            points=points[selected],
            colors=colors,
            point_size=point_size,
        )
        self._modal_anchor_cache_key = cache_key

    def current_modal_oscillator(self) -> tuple[np.ndarray, float] | None:
        handles = getattr(self, "_modal_playback_handles", None)
        if handles is None or str(handles["drive"].value) != "manual oscillator":
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
