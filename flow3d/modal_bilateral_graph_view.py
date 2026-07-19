from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from modal_surface.io import load_view_config


@dataclass(frozen=True)
class ModalBilateralGraphViewResult:
    path: Path
    view_id: str
    graph_edge_count: int
    threshold_edge_count: int
    projected_edge_count: int
    displayed_edge_count: int


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_minimum_affinity(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("minimum_affinity must be finite and in [0,1]")
    return result


def _load_graph(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as graph:
        required = {"format", "version", "points_world", "edge_index", "edge_affinity"}
        missing = required - set(graph.files)
        if missing:
            raise ValueError(f"Bilateral graph is missing fields: {sorted(missing)}")
        if str(graph["format"].item()) != "modal_bilateral_gaussian_graph":
            raise ValueError(f"{path} is not a modal bilateral Gaussian graph")
        if int(graph["version"].item()) != 1:
            raise ValueError(f"{path} has an unsupported bilateral graph version")
        points = np.asarray(graph["points_world"], dtype=np.float64)
        edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
        affinity = np.asarray(graph["edge_affinity"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N,3), got {points.shape}")
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError(f"edge_index must have shape (E,2), got {edge_index.shape}")
    if affinity.shape != (edge_index.shape[0],):
        raise ValueError(
            f"edge_affinity must have shape ({edge_index.shape[0]},), got "
            f"{affinity.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("points_world contains non-finite values")
    if np.any(edge_index < 0) or np.any(edge_index >= points.shape[0]):
        raise ValueError("edge_index contains an out-of-range Gaussian index")
    if (
        not np.all(np.isfinite(affinity))
        or np.any(affinity < 0.0)
        or np.any(affinity > 1.0)
    ):
        raise ValueError("edge_affinity must be finite and in [0,1]")
    return points, edge_index, affinity


def _load_background(
    path: Path | None,
    expected_height: int,
    expected_width: int,
) -> np.ndarray | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    background_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if background_bgr is None:
        raise ValueError(f"Unable to read background image: {path}")
    background = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
    if background.shape != (expected_height, expected_width, 3):
        raise ValueError(
            f"Background image must have shape "
            f"({expected_height},{expected_width},3), got {background.shape}"
        )
    return background


def _project_points(
    points: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    camera_points = (
        points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3][None]
    )
    depth = camera_points[:, 2]
    positive_depth = np.isfinite(depth) & (depth > 0.0)
    homogeneous_pixels = camera_points @ intrinsics.T
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    pixels[positive_depth] = (
        homogeneous_pixels[positive_depth, :2] / depth[positive_depth, None]
    )
    in_frame = (
        positive_depth
        & np.all(np.isfinite(pixels), axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= image_width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= image_height - 1)
    )
    return pixels, in_frame


def _uniform_edge_sample(indices: np.ndarray, maximum_count: int) -> np.ndarray:
    if indices.shape[0] <= maximum_count:
        return indices
    positions = np.linspace(
        0, indices.shape[0] - 1, maximum_count, dtype=np.int64
    )
    return indices[positions]


def _write_projection(
    path: Path,
    *,
    background: np.ndarray | None,
    pixels: np.ndarray,
    edge_index: np.ndarray,
    affinity: np.ndarray,
    view_id: str,
    image_width: int,
    image_height: int,
    minimum_affinity: float,
    projected_edge_count: int,
) -> None:
    edge_pixels = pixels[edge_index]
    upper = float(np.percentile(affinity, 99.0))
    if upper <= minimum_affinity:
        upper = minimum_affinity + np.finfo(np.float64).eps
    normalizer = Normalize(
        vmin=minimum_affinity,
        vmax=upper,
        clip=True,
    )
    normalized = normalizer(affinity)
    edge_colors = plt.get_cmap("turbo")(normalized)
    edge_colors[:, 3] = 0.25 + 0.7 * normalized
    line_widths = 0.35 + 1.35 * normalized

    figure_width = 14.0
    figure_height = figure_width * image_height / image_width
    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height), constrained_layout=True
    )
    if background is None:
        axis.set_facecolor((0.03, 0.03, 0.03))
    else:
        axis.imshow(background, origin="upper")
    axis.add_collection(
        LineCollection(
            edge_pixels,
            colors=edge_colors,
            linewidths=line_widths,
            rasterized=True,
        )
    )
    axis.set_xlim(0.0, image_width - 1)
    axis.set_ylim(image_height - 1, 0.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()
    scalar_map = plt.cm.ScalarMappable(norm=normalizer, cmap="turbo")
    scalar_map.set_array([])
    figure.colorbar(
        scalar_map,
        ax=axis,
        label="bilateral edge affinity",
        shrink=0.8,
        pad=0.01,
    )
    figure.suptitle(
        f"{view_id} bilateral Gaussian connectivity | "
        f"affinity >= {minimum_affinity:g} | "
        f"displayed {edge_index.shape[0]:,} / projected "
        f"{projected_edge_count:,} edges"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=".tmp.png",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=180)
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def render_modal_bilateral_graph_view(
    *,
    graph_path: str | Path,
    view_config_path: str | Path,
    output_path: str | Path,
    minimum_affinity: float,
    maximum_edges: int = 100_000,
    background_image_path: str | Path | None = None,
) -> ModalBilateralGraphViewResult:
    minimum = _validate_minimum_affinity(minimum_affinity)
    maximum = _validate_positive_integer(maximum_edges, "maximum_edges")
    graph_file = Path(graph_path).expanduser().resolve()
    config_file = Path(view_config_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    background_path = (
        None
        if background_image_path is None
        else Path(background_image_path).expanduser().resolve()
    )
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output image already exists: {output}")

    points, edge_index, affinity = _load_graph(graph_file)
    config = load_view_config(config_file)
    background = _load_background(
        background_path,
        config.image_height,
        config.image_width,
    )
    pixels, point_in_frame = _project_points(
        points,
        config.world_to_camera,
        config.K,
        config.image_width,
        config.image_height,
    )
    above_threshold = affinity >= minimum
    projected = (
        above_threshold
        & point_in_frame[edge_index[:, 0]]
        & point_in_frame[edge_index[:, 1]]
    )
    projected_indices = np.flatnonzero(projected)
    if projected_indices.shape[0] == 0:
        raise ValueError(
            f"No graph edge with affinity >= {minimum:g} projects fully inside "
            f"view {config.view_id!r}"
        )
    displayed_indices = _uniform_edge_sample(projected_indices, maximum)
    _write_projection(
        output,
        background=background,
        pixels=pixels,
        edge_index=edge_index[displayed_indices],
        affinity=affinity[displayed_indices],
        view_id=config.view_id,
        image_width=config.image_width,
        image_height=config.image_height,
        minimum_affinity=minimum,
        projected_edge_count=int(projected_indices.shape[0]),
    )
    return ModalBilateralGraphViewResult(
        path=output,
        view_id=config.view_id,
        graph_edge_count=int(edge_index.shape[0]),
        threshold_edge_count=int(np.count_nonzero(above_threshold)),
        projected_edge_count=int(projected_indices.shape[0]),
        displayed_edge_count=int(displayed_indices.shape[0]),
    )


__all__ = [
    "ModalBilateralGraphViewResult",
    "render_modal_bilateral_graph_view",
]
