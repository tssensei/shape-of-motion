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
    ):
        self.num_frames = num_frames
        self.work_dir = Path(work_dir)
        self.viewer_cameras = tuple(viewer_cameras)
        self.orbit_center = None if orbit_center is None else np.asarray(orbit_center, dtype=np.float32)
        self.camera_frustum_scale = float(camera_frustum_scale)
        super().__init__(server, render_fn, mode)

    def _define_guis(self):
        super()._define_guis()
        server = self.server
        self._set_default_orbit_center()
        self._time_folder = server.gui.add_folder("Time")
        with self._time_folder:
            self._playback_guis = add_gui_playback_group(
                server,
                num_frames=self.num_frames,
                initial_fps=15.0,
            )
            self._playback_guis[0].on_update(self.rerender)
            self._canonical_checkbox = server.gui.add_checkbox("Canonical", False)
            self._canonical_checkbox.on_update(self.rerender)

            _cached_playback_disabled = []

            def _toggle_gui_playing(event):
                if event.target.value:
                    nonlocal _cached_playback_disabled
                    _cached_playback_disabled = [
                        gui.disabled for gui in self._playback_guis
                    ]
                    target_disabled = [True] * len(self._playback_guis)
                else:
                    target_disabled = _cached_playback_disabled
                for gui, disabled in zip(self._playback_guis, target_disabled):
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
