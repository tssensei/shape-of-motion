from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from modal_peak_pick.core.gl_renderer import GLMeshWarpRenderer
from modal_peak_pick.core.mesh import (
    RegularGridMesh,
    build_regular_grid_mesh,
    deform_mesh_vertices,
    sample_displacement_to_vertices,
    sample_field_to_vertices,
)


@dataclass(frozen=True)
class RenderConfig:
    backend: str = "gl_mesh"
    fill_mode: str = "inpaint"
    use_mask: bool = True
    mesh_step: int = 16
    show_mesh: bool = False
    depth_weight: str = "amplitude"


@dataclass
class RenderResources:
    mesh: Optional[RegularGridMesh] = None
    gl_renderer: Optional[GLMeshWarpRenderer] = None
    vertex_depth: Optional[np.ndarray] = None

    def close(self) -> None:
        if self.gl_renderer is not None:
            self.gl_renderer.close()
            self.gl_renderer = None


def prepare_displacement(
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    use_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sanitize displacement fields and optionally zero displacement outside a mask."""
    dx_out = np.asarray(dx, dtype=np.float32).copy()
    dy_out = np.asarray(dy, dtype=np.float32).copy()
    if use_mask and mask is not None:
        mask_f = mask.astype(np.float32)
        dx_out *= mask_f
        dy_out *= mask_f
    return dx_out, dy_out


def backward_warp_frame(frame_bgr_u8: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Fast inverse-map warp for small displacements."""
    h, w = frame_bgr_u8.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    map_x = xx.astype(np.float32) - dx.astype(np.float32)
    map_y = yy.astype(np.float32) - dy.astype(np.float32)
    return cv2.remap(
        frame_bgr_u8,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def forward_splat_frame(
    frame_bgr_u8: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    fill_mode: str = "inpaint",
) -> np.ndarray:
    """Forward splat source pixels to target positions with bilinear weights."""
    h, w = frame_bgr_u8.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]

    x_t = xx.astype(np.float32) + dx.astype(np.float32)
    y_t = yy.astype(np.float32) + dy.astype(np.float32)

    x0 = np.floor(x_t).astype(np.int32)
    y0 = np.floor(y_t).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x_t - x0
    wy = y_t - y0
    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = wx * (1.0 - wy)
    w10 = (1.0 - wx) * wy
    w11 = wx * wy

    out = np.zeros((h, w, 3), dtype=np.float32)
    acc = np.zeros((h, w), dtype=np.float32)
    src = frame_bgr_u8.astype(np.float32)

    def splat(xi: np.ndarray, yi: np.ndarray, ww: np.ndarray) -> None:
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (ww > 0)
        if not np.any(valid):
            return
        xv = xi[valid]
        yv = yi[valid]
        wv = ww[valid].astype(np.float32)
        sv = src[valid]
        np.add.at(acc, (yv, xv), wv)
        np.add.at(out[..., 0], (yv, xv), wv * sv[:, 0])
        np.add.at(out[..., 1], (yv, xv), wv * sv[:, 1])
        np.add.at(out[..., 2], (yv, xv), wv * sv[:, 2])

    splat(x0, y0, w00)
    splat(x1, y0, w01)
    splat(x0, y1, w10)
    splat(x1, y1, w11)

    nz = acc > 1e-8
    out[nz, 0] /= acc[nz]
    out[nz, 1] /= acc[nz]
    out[nz, 2] /= acc[nz]
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)

    if fill_mode == "reference":
        out_u8[~nz] = frame_bgr_u8[~nz]
        return out_u8
    if fill_mode == "inpaint":
        hole_mask = (~nz).astype(np.uint8) * 255
        if np.any(hole_mask):
            c0 = cv2.inpaint(out_u8[..., 0], hole_mask, 3.0, cv2.INPAINT_TELEA)
            c1 = cv2.inpaint(out_u8[..., 1], hole_mask, 3.0, cv2.INPAINT_TELEA)
            c2 = cv2.inpaint(out_u8[..., 2], hole_mask, 3.0, cv2.INPAINT_TELEA)
            out_u8 = np.stack([c0, c1, c2], axis=-1)
        return out_u8
    raise ValueError(f"Unknown fill_mode: {fill_mode}")


def render_reference_frame(
    frame_ref_bgr: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    config: Optional[RenderConfig] = None,
    mask: Optional[np.ndarray] = None,
    resources: Optional[RenderResources] = None,
) -> np.ndarray:
    """Render one synthetic frame from a reference image and 2D displacement."""
    cfg = RenderConfig() if config is None else config
    dx_use, dy_use = prepare_displacement(dx, dy, mask=mask, use_mask=cfg.use_mask)

    backend = cfg.backend.lower()
    if backend == "backward_warp":
        return backward_warp_frame(frame_ref_bgr, dx_use, dy_use)
    if backend == "forward_splat":
        return forward_splat_frame(frame_ref_bgr, dx_use, dy_use, fill_mode=cfg.fill_mode)
    if backend == "gl_mesh":
        own_resources = False
        if resources is None:
            resources = create_render_resources(frame_ref_bgr.shape[:2], cfg)
            own_resources = True
        try:
            mesh = resources.mesh
            renderer = resources.gl_renderer
            if mesh is None or renderer is None:
                raise RuntimeError("gl_mesh backend requires initialized mesh and renderer resources.")
            disp_xy = sample_displacement_to_vertices(dx_use, dy_use, mesh.vertices_xy)
            vertices_xy = deform_mesh_vertices(mesh, disp_xy)
            return renderer.render(frame_ref_bgr, mesh, vertices_xy, vertex_depth=resources.vertex_depth)
        finally:
            if own_resources:
                resources.close()
    raise ValueError(f"Unknown render backend: {cfg.backend}")


def create_render_resources(
    image_shape: tuple[int, int],
    config: Optional[RenderConfig] = None,
    depth_weight_field: Optional[np.ndarray] = None,
) -> RenderResources:
    """Build reusable renderer resources for one image shape/backend."""
    cfg = RenderConfig() if config is None else config
    resources = RenderResources()
    backend = cfg.backend.lower()

    if backend == "gl_mesh":
        h, w = int(image_shape[0]), int(image_shape[1])
        resources.mesh = build_regular_grid_mesh(h, w, int(cfg.mesh_step))
        resources.gl_renderer = GLMeshWarpRenderer(w, h, show_mesh=cfg.show_mesh)
        if cfg.depth_weight.lower() == "amplitude" and depth_weight_field is not None:
            sampled = sample_field_to_vertices(depth_weight_field.astype(np.float32), resources.mesh.vertices_xy)
            resources.vertex_depth = (-sampled.astype(np.float32)).astype(np.float32)
        else:
            resources.vertex_depth = np.zeros((resources.mesh.num_vertices,), dtype=np.float32)
    elif backend not in {"backward_warp", "forward_splat"}:
        raise ValueError(f"Unknown render backend: {cfg.backend}")
    return resources
