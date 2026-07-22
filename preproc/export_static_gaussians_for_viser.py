"""Export activated static 3DGS parameters for the standalone Viser viewer."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


_PARAMETER_NAMES = ("means", "quats", "scales", "colors", "opacities")


def _require_tensor(
    state: Mapping[str, Any],
    key: str,
    torch: Any,
) -> Any:
    if key not in state:
        raise ValueError(f"Checkpoint model state is missing {key}")
    value = state[key]
    if not torch.is_tensor(value):
        raise ValueError(f"Checkpoint model field {key} must be a Torch tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"Checkpoint model field {key} must be finite")
    return value.detach().cpu().float()


def _activate_gaussian_group(
    state: Mapping[str, Any],
    prefix: str,
    torch: Any,
) -> dict[str, np.ndarray]:
    raw = {
        name: _require_tensor(state, f"{prefix}.{name}", torch)
        for name in _PARAMETER_NAMES
    }
    means = raw["means"]
    num_gaussians = int(means.shape[0]) if means.ndim == 2 else -1
    expected_shapes = {
        "means": (num_gaussians, 3),
        "quats": (num_gaussians, 4),
        "scales": (num_gaussians, 3),
        "colors": (num_gaussians, 3),
        "opacities": (num_gaussians,),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(raw[name].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Checkpoint model field {prefix}.{name} must have shape "
                f"{expected_shape}, got {actual_shape}"
            )
    if num_gaussians <= 0:
        raise ValueError(f"Checkpoint Gaussian group {prefix} must be non-empty")

    quat_norm = torch.linalg.vector_norm(raw["quats"], dim=-1, keepdim=True)
    if not bool(torch.all(quat_norm > 0.0).item()):
        raise ValueError(f"Checkpoint Gaussian group {prefix} has a zero quaternion")
    activated = {
        "centers": means,
        "quats_wxyz": raw["quats"] / quat_norm,
        "scales": torch.exp(raw["scales"]),
        "rgbs": torch.sigmoid(raw["colors"]),
        "opacities": torch.sigmoid(raw["opacities"]).reshape(-1, 1),
    }
    arrays = {
        name: tensor.numpy().astype(np.float32, copy=False)
        for name, tensor in activated.items()
    }
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(
                f"Activated checkpoint field {prefix}.{name} must be finite"
            )
    if np.any(arrays["scales"] <= 0.0):
        raise ValueError(f"Activated checkpoint scales for {prefix} must be positive")
    return arrays


def _group_arrays(
    group_name: str,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    count = int(arrays["centers"].shape[0])
    short_name = "fg" if group_name == "foreground" else "bg"
    return {
        f"num_{group_name}_gaussians": np.array(count, dtype=np.int64),
        f"{short_name}_gaussian_indices": np.arange(count, dtype=np.int64),
        **{
            f"{short_name}_{name}": np.asarray(value)
            for name, value in arrays.items()
        },
    }


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if not path.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {path.parent}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".npz",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def export_static_gaussians(
    *,
    input_checkpoint: Path,
    output_npz: Path,
    include_background: bool,
) -> None:
    if not input_checkpoint.is_absolute():
        raise ValueError("--input-ckpt must be an absolute path")
    if not input_checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {input_checkpoint}")
    if output_npz.suffix.lower() != ".npz":
        raise ValueError("--output-npz must end in .npz")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Checkpoint export must run in the Shape-of-Motion environment"
        ) from exc

    checkpoint = torch.load(
        input_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{input_checkpoint} must contain a checkpoint mapping")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError(f"{input_checkpoint} does not contain a model state")

    foreground = _activate_gaussian_group(state, "fg.params", torch)
    arrays = {
        "version": np.array(1, dtype=np.int32),
        "point_type": np.array("static_3dgs_activated_gaussians"),
        "source_checkpoint": np.array(str(input_checkpoint)),
        "has_background": np.array(include_background, dtype=bool),
        **_group_arrays("foreground", foreground),
    }
    if include_background:
        background = _activate_gaussian_group(state, "bg.params", torch)
        arrays.update(_group_arrays("background", background))
    _write_npz_atomic(output_npz, arrays)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export activated static 3DGS parameters for the standalone modern "
            "Viser graph viewer"
        )
    )
    parser.add_argument("--input-ckpt", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Also export background Gaussians; fail if the checkpoint has none",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_static_gaussians(
        input_checkpoint=args.input_ckpt,
        output_npz=args.output_npz,
        include_background=args.include_background,
    )
    print(f"Wrote static Gaussian viewer sidecar: {args.output_npz}")


if __name__ == "__main__":
    main()
