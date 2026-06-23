from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegularGridMesh:
    """Regular 2D mesh defined in image coordinates."""

    vertices_xy: np.ndarray
    texcoord_uv: np.ndarray
    triangles: np.ndarray
    image_shape: tuple[int, int]
    step: int
    grid_shape: tuple[int, int]

    @property
    def num_vertices(self) -> int:
        return int(self.vertices_xy.shape[0])

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])


def _grid_axis(length: int, step: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Axis length must be positive.")
    if step <= 0:
        raise ValueError("Grid step must be positive.")

    vals = np.arange(0, length, step, dtype=np.float32)
    if vals.size == 0 or vals[-1] != float(length):
        vals = np.concatenate([vals, np.array([float(length)], dtype=np.float32)])
    return vals


def build_regular_grid_mesh(height: int, width: int, step: int) -> RegularGridMesh:
    """Build a regular image-space mesh for textured warping."""
    if height <= 1 or width <= 1:
        raise ValueError("Image dimensions must be at least 2x2.")

    xs = _grid_axis(int(width), int(step))
    ys = _grid_axis(int(height), int(step))
    xx, yy = np.meshgrid(xs, ys)

    vertices_xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    texcoord_uv = np.stack(
        [
            vertices_xy[:, 0] / max(float(width), 1.0),
            vertices_xy[:, 1] / max(float(height), 1.0),
        ],
        axis=1,
    ).astype(np.float32)

    rows = int(len(ys))
    cols = int(len(xs))
    tris = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i00 = r * cols + c
            i01 = i00 + 1
            i10 = (r + 1) * cols + c
            i11 = i10 + 1
            tris.append((i00, i10, i11))
            tris.append((i00, i11, i01))

    return RegularGridMesh(
        vertices_xy=vertices_xy,
        texcoord_uv=texcoord_uv,
        triangles=np.asarray(tris, dtype=np.int32),
        image_shape=(int(height), int(width)),
        step=int(step),
        grid_shape=(rows, cols),
    )


def sample_field_to_vertices(field: np.ndarray, vertices_xy: np.ndarray) -> np.ndarray:
    """Bilinearly sample one dense image field at mesh vertex locations."""
    if field.ndim != 2:
        raise ValueError("field must be 2D.")
    if vertices_xy.ndim != 2 or vertices_xy.shape[1] != 2:
        raise ValueError("vertices_xy must have shape [N, 2].")

    h, w = field.shape
    x = np.clip(vertices_xy[:, 0].astype(np.float32), 0.0, float(w - 1))
    y = np.clip(vertices_xy[:, 1].astype(np.float32), 0.0, float(h - 1))

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)

    f00 = field[y0, x0].astype(np.float32)
    f01 = field[y0, x1].astype(np.float32)
    f10 = field[y1, x0].astype(np.float32)
    f11 = field[y1, x1].astype(np.float32)

    f0 = f00 * (1.0 - wx) + f01 * wx
    f1 = f10 * (1.0 - wx) + f11 * wx
    return (f0 * (1.0 - wy) + f1 * wy).astype(np.float32)


def sample_displacement_to_vertices(
    dx: np.ndarray,
    dy: np.ndarray,
    vertices_xy: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample a dense displacement field onto mesh vertices."""
    if dx.shape != dy.shape:
        raise ValueError("dx and dy must have the same shape.")
    disp_x = sample_field_to_vertices(dx, vertices_xy)
    disp_y = sample_field_to_vertices(dy, vertices_xy)
    return np.stack([disp_x, disp_y], axis=1).astype(np.float32)


def deform_mesh_vertices(mesh: RegularGridMesh, disp_xy: np.ndarray) -> np.ndarray:
    """Add sampled displacements to the reference mesh vertices."""
    if disp_xy.shape != mesh.vertices_xy.shape:
        raise ValueError(f"disp_xy shape {disp_xy.shape} does not match mesh vertices {mesh.vertices_xy.shape}.")
    return (mesh.vertices_xy + disp_xy.astype(np.float32)).astype(np.float32)
