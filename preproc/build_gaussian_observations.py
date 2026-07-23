"""Build reusable per-mode Gaussian observation artifacts without solving modes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from modal_surface.checkpoint_render_inputs import (
    load_fg_pixel_candidate_inputs_from_checkpoint,
)
from modal_surface.gaussian_observations import build_gaussian_observation_graph
from modal_surface.io import load_modal_freqs


def _parse_mode_indices(raw: str, num_modes: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(num_modes))
    indices: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        index = int(text)
        if index < 0 or index >= num_modes:
            raise ValueError(
                f"mode index {index} is outside [0,{num_modes - 1}]"
            )
        if index in indices:
            raise ValueError(f"--mode-indices contains duplicate mode index {index}")
        indices.append(index)
    if not indices:
        raise ValueError(
            "--mode-indices must contain at least one index, or 'all'"
        )
    return indices


def _freq_slug(freq_hz: float) -> str:
    text = f"{freq_hz:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build reusable per-mode foreground-Gaussian observations without "
            "running staged or rigid modal optimization"
        )
    )
    parser.add_argument("--input-ckpt", type=Path, required=True)
    parser.add_argument(
        "--view-config",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--modal-npz",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--mode-indices",
        default="all",
        help="Comma-separated zero-based mode indices, or 'all'.",
    )
    parser.add_argument("--pixel-sample-stride", type=int, default=4)
    parser.add_argument("--pixel-candidate-k", type=int, default=4)
    parser.add_argument("--pixel-preselect-k", type=int, default=32)
    parser.add_argument("--pixel-render-acc-min", type=float, default=0.05)
    parser.add_argument("--pixel-min-contribution", type=float, default=1e-12)
    parser.add_argument("--mask-erode-iters", type=int, default=1)
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_checkpoint = args.input_ckpt.expanduser().resolve(strict=True)
    view_configs = [
        path.expanduser().resolve(strict=True) for path in args.view_config
    ]
    modal_npzs = [
        path.expanduser().resolve(strict=True) for path in args.modal_npz
    ]
    if len(view_configs) != len(modal_npzs):
        raise ValueError(
            "--view-config and --modal-npz must be supplied the same number of times"
        )

    view_config_paths = [str(path) for path in view_configs]
    modal_npz_paths = [str(path) for path in modal_npzs]
    frequencies_by_view = load_modal_freqs(modal_npz_paths)
    mode_indices = _parse_mode_indices(
        args.mode_indices,
        int(frequencies_by_view[0].shape[0]),
    )
    out_dir = args.out_dir.expanduser().resolve()
    output_paths = {
        mode_index: out_dir
        / (
            f"mode_{mode_index:03d}_"
            f"{_freq_slug(float(frequencies_by_view[0][mode_index]))}hz.npz"
        )
        for mode_index in mode_indices
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Observation output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        foreground_means,
        foreground_scales,
        foreground_quaternions,
        foreground_opacities,
        _foreground_colors,
        rendered_depths,
        rendered_accumulations,
    ) = load_fg_pixel_candidate_inputs_from_checkpoint(
        str(input_checkpoint),
        view_config_paths,
    )
    from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

    gaussian_tree = cKDTree(foreground_means.astype(np.float64))
    for mode_index in mode_indices:
        output_path = output_paths[mode_index]
        build_gaussian_observation_graph(
            points_world=foreground_means,
            view_config_paths=view_config_paths,
            modal_npz_paths=modal_npz_paths,
            out_path=output_path,
            source_checkpoint=str(input_checkpoint),
            mode_index=mode_index,
            mask_erode_iters=int(args.mask_erode_iters),
            freq_tolerance_hz=float(args.freq_tolerance_hz),
            pixel_sample_stride=int(args.pixel_sample_stride),
            pixel_candidate_k=int(args.pixel_candidate_k),
            pixel_preselect_k=int(args.pixel_preselect_k),
            pixel_render_acc_min=float(args.pixel_render_acc_min),
            pixel_min_contribution=float(args.pixel_min_contribution),
            gaussian_scales=foreground_scales,
            gaussian_quats=foreground_quaternions,
            gaussian_opacities=foreground_opacities,
            rendered_depths=rendered_depths,
            rendered_accs=rendered_accumulations,
            gaussian_tree=gaussian_tree,
        )
        print(f"Wrote Gaussian observations -> {output_path}")


if __name__ == "__main__":
    main()
