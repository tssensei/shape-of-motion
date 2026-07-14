"""I/O helpers and data contracts for the modal_surface pipeline.

The modal_surface package uses a small JSON + NumPy contract between geometry
preparation and modal optimization:

    view_config.json
        Camera intrinsics, world_to_camera pose, image size, depth path, and
        optional mask path.

    modal_analysis.npz
        2D complex modal response from run_modal_peak_pick.py. The required
        arrays are mode_u, mode_v, and selected_freqs_hz.

    depth.npy / mask.npy
        Dense reference-view depth and foreground mask aligned to the modal
        image resolution.

This module centralizes loading and validation so make-packet, match-two-views,
and optimize-two-view share the same assumptions about shapes and paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ViewConfig:
    """Validated view geometry consumed by modal_surface.

    K and world_to_camera are used to unproject view-local depth pixels into
    world-space surface points and to project those points into another view.
    depth_path and mask_path are resolved relative to the JSON config location
    when the JSON contains relative paths.
    """

    view_id: str
    image_width: int
    image_height: int
    K: np.ndarray
    world_to_camera: np.ndarray
    depth_path: Path
    mask_path: Path | None
    depth_scale: float
    root: Path


def _resolve_path(root: Path, value: str | None) -> Path | None:
    """Resolve optional paths relative to the config file directory."""
    if value is None or value == "":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def load_view_config(path: str | Path) -> ViewConfig:
    """Load and validate a modal_surface view_config JSON file."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    required = ["view_id", "image_width", "image_height", "K", "world_to_camera", "depth_path"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Missing keys in view config {cfg_path}: {missing}")

    root = cfg_path.parent
    K = np.asarray(raw["K"], dtype=np.float64)
    world_to_camera = np.asarray(raw["world_to_camera"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must have shape (3,3), got {K.shape}.")
    if world_to_camera.shape != (4, 4):
        raise ValueError(f"world_to_camera must have shape (4,4), got {world_to_camera.shape}.")

    depth_path = _resolve_path(root, str(raw["depth_path"]))
    if depth_path is None:
        raise ValueError("depth_path must not be empty.")

    return ViewConfig(
        view_id=str(raw["view_id"]),
        image_width=int(raw["image_width"]),
        image_height=int(raw["image_height"]),
        K=K,
        world_to_camera=world_to_camera,
        depth_path=depth_path,
        mask_path=_resolve_path(root, raw.get("mask_path")),
        depth_scale=float(raw.get("depth_scale", 1.0)),
        root=root,
    )


def load_depth(path: str | Path, expected_shape: tuple[int, int], depth_scale: float = 1.0) -> np.ndarray:
    """Load a dense depth map and verify it matches the modal image shape."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        depth = np.load(str(path))
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Cannot read depth file: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32) * float(depth_scale)
    if depth.shape != expected_shape:
        raise ValueError(f"Depth shape {depth.shape} does not match expected {expected_shape}.")
    return depth


def load_mask(path: str | Path | None, expected_shape: tuple[int, int]) -> np.ndarray:
    """Load a binary foreground mask, or return an all-true mask if omitted."""
    if path is None:
        return np.ones(expected_shape, dtype=bool)
    path = Path(path)
    if path.suffix.lower() == ".npy":
        mask = np.load(str(path))
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask file: {path}")
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}.")
    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match expected {expected_shape}.")
    if np.issubdtype(mask.dtype, np.floating):
        return mask > 0.5
    return mask > 0


def load_modal_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a modal peak-pick export and validate required complex mode arrays."""
    z = np.load(str(path), allow_pickle=False)
    required = ["mode_u", "mode_v", "selected_freqs_hz"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise ValueError(f"Modal npz missing required keys: {missing}")
    mode_u = z["mode_u"].astype(np.complex64)
    mode_v = z["mode_v"].astype(np.complex64)
    if mode_u.shape != mode_v.shape or mode_u.ndim != 3:
        raise ValueError(f"mode_u/mode_v must have shape (K,H,W), got {mode_u.shape} and {mode_v.shape}.")
    return {k: z[k] for k in z.files}


def load_modal_freqs(paths: list[str | Path]) -> list[np.ndarray]:
    """Load and validate selected modal frequencies for multiple views."""
    freqs_per_view: list[np.ndarray] = []
    for path in paths:
        modal = load_modal_npz(path)
        mode_u = modal["mode_u"]
        freqs = modal["selected_freqs_hz"].astype(np.float32).reshape(-1)
        if mode_u.shape[0] != freqs.shape[0]:
            raise ValueError(f"{path} selected_freqs_hz length does not match mode_u/mode_v.")
        freqs_per_view.append(freqs)
    if len({freqs.shape[0] for freqs in freqs_per_view}) != 1:
        raise ValueError("All modal npz files must contain the same number of selected frequencies.")
    return freqs_per_view


def ensure_modal_shape(modal: dict[str, np.ndarray], expected_shape: tuple[int, int]) -> None:
    """Ensure modal image arrays are aligned to a view_config image size."""
    mode_shape = tuple(int(v) for v in modal["mode_u"].shape[1:])
    if mode_shape != expected_shape:
        raise ValueError(f"Modal mode image shape {mode_shape} does not match expected {expected_shape}.")
