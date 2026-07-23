"""Build mode-independent Gaussian observation topology."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from modal_surface.checkpoint_render_inputs import (
    load_fg_pixel_candidate_inputs_from_checkpoint,
)
from modal_surface.gaussian_observations import (
    build_gaussian_observation_topology,
    write_gaussian_observation_topology,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one reusable foreground-Gaussian observation topology "
            "from a static checkpoint and ordered view configs"
        )
    )
    parser.add_argument("--input-ckpt", type=Path, required=True)
    parser.add_argument(
        "--view-config",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pixel-sample-stride", type=int, default=4)
    parser.add_argument("--pixel-candidate-k", type=int, default=4)
    parser.add_argument("--pixel-preselect-k", type=int, default=32)
    parser.add_argument("--pixel-render-acc-min", type=float, default=0.05)
    parser.add_argument("--pixel-min-contribution", type=float, default=1e-12)
    parser.add_argument("--mask-erode-iters", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_checkpoint = args.input_ckpt.expanduser().resolve(strict=True)
    view_configs = [
        path.expanduser().resolve(strict=True) for path in args.view_config
    ]
    output_path = args.out.expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(
            f"Gaussian observation topology already exists: {output_path}"
        )

    view_config_paths = [str(path) for path in view_configs]
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
    topology = build_gaussian_observation_topology(
        points_world=foreground_means,
        view_config_paths=view_config_paths,
        source_checkpoint=str(input_checkpoint),
        mask_erode_iters=int(args.mask_erode_iters),
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
    write_gaussian_observation_topology(output_path, topology)
    print(f"Wrote Gaussian observation topology -> {output_path}")


if __name__ == "__main__":
    main()
