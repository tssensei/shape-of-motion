"""Replay Gaussian candidate selection and diagnose lost multi-view coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from modal_surface.checkpoint_render_inputs import (
    load_fg_pixel_candidate_inputs_from_checkpoint,
)
from modal_surface.gaussian_observations import load_gaussian_observation_topology
from modal_surface.io import load_view_config
from modal_surface.observation_coverage import (
    OBSERVATION_COVERAGE_CATEGORY_NAMES,
    ObservationCoverageDiagnostics,
    build_observation_coverage_diagnostics,
    parse_coverage_k_values,
    validate_reference_observation_replay,
    write_observation_coverage_diagnostics,
)


_REFERENCE_REQUIRED_FIELDS = {
    "points_world",
    "gaussian_indices",
    "point_type",
    "source_checkpoint",
    "obs_count_per_point",
    "obs_sample_count_per_point",
    "view_ids",
    "mask_erode_iters",
    "candidate_point_count",
    "preserved_all_points",
    "pixel_sample_stride",
    "pixel_candidate_k",
    "pixel_preselect_k",
    "pixel_render_acc_min",
    "pixel_min_contribution",
    "pixel_candidate_method",
    "observations_per_view",
    "source_view_configs",
}


def _scalar(array: np.ndarray, name: str, path: Path) -> object:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{path} field {name} must be scalar")
    return value.item()


def load_observation_topology(
    path: Path,
    *,
    input_checkpoint: Path,
    view_config_paths: list[Path],
) -> dict[str, np.ndarray]:
    topology = load_gaussian_observation_topology(path)
    missing = sorted(_REFERENCE_REQUIRED_FIELDS - set(topology))
    if missing:
        raise ValueError(f"{path} missing required fields: {missing}")
    reference = {name: topology[name] for name in _REFERENCE_REQUIRED_FIELDS}
    point_type = str(_scalar(reference["point_type"], "point_type", path))
    if point_type != "foreground_gaussian_center":
        raise ValueError(f"{path} has unsupported point_type={point_type!r}")
    source_checkpoint = str(
        _scalar(reference["source_checkpoint"], "source_checkpoint", path)
    )
    if source_checkpoint != str(input_checkpoint):
        raise ValueError(
            f"{path} source_checkpoint does not match --input-ckpt"
        )
    method = str(
        _scalar(
            reference["pixel_candidate_method"],
            "pixel_candidate_method",
            path,
        )
    )
    if method != "rendered_depth_gaussian_contribution":
        raise ValueError(f"{path} has unsupported pixel_candidate_method={method!r}")
    if not bool(
        _scalar(reference["preserved_all_points"], "preserved_all_points", path)
    ):
        raise ValueError(f"{path} must preserve all foreground Gaussians")
    points = np.asarray(reference["points_world"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path} points_world must have shape (N,3)")
    count = int(
        _scalar(reference["candidate_point_count"], "candidate_point_count", path)
    )
    if count != points.shape[0]:
        raise ValueError(f"{path} candidate_point_count does not match points_world")
    indices = np.asarray(reference["gaussian_indices"])
    expected_indices = np.arange(points.shape[0], dtype=indices.dtype)
    if indices.ndim != 1 or not np.array_equal(indices, expected_indices):
        raise ValueError(f"{path} gaussian_indices must be contiguous checkpoint order")
    configs = [load_view_config(config_path) for config_path in view_config_paths]
    expected_view_ids = np.asarray([config.view_id for config in configs])
    if not np.array_equal(np.asarray(reference["view_ids"]), expected_view_ids):
        raise ValueError(f"{path} view_ids do not match --view-config order")
    expected_config_paths = np.asarray(
        [str(config_path) for config_path in view_config_paths]
    )
    if not np.array_equal(
        np.asarray(reference["source_view_configs"]),
        expected_config_paths,
    ):
        raise ValueError(
            f"{path} source_view_configs do not match --view-config arguments"
        )
    return reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the exact pixel-to-Gaussian candidate selection without "
            "solving modes or running motion fill"
        )
    )
    parser.add_argument("--input-ckpt", type=Path, required=True)
    parser.add_argument(
        "--view-config",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--observation-topology", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--k-values", default="4,8,12,16,32")
    return parser


def _print_summary(diagnostics: ObservationCoverageDiagnostics) -> None:
    k_values = diagnostics.k_values
    selected_histograms = diagnostics.selected_view_count_histogram_by_k
    category_counts = diagnostics.category_count_by_k
    print(
        "K  unobserved  single-view  multi-view  "
        "preselect/contribution-lost  positive/top-K-lost"
    )
    for index, k_value in enumerate(k_values.tolist()):
        histogram = selected_histograms[index]
        print(
            f"{int(k_value):2d} "
            f"{int(histogram[0]):11d} "
            f"{int(histogram[1]):12d} "
            f"{int(histogram[2:].sum()):10d} "
            f"{int(category_counts[index, 1]):27d} "
            f"{int(category_counts[index, 2]):19d}"
        )
    print("Coverage categories:")
    for category_index, name in enumerate(OBSERVATION_COVERAGE_CATEGORY_NAMES):
        print(f"  {category_index}: {name}")


def main() -> None:
    args = build_parser().parse_args()
    reference = load_observation_topology(
        args.observation_topology,
        input_checkpoint=args.input_ckpt,
        view_config_paths=args.view_config,
    )
    mask_erode_iters = int(
        _scalar(
            reference["mask_erode_iters"],
            "mask_erode_iters",
            args.observation_topology,
        )
    )
    pixel_sample_stride = int(
        _scalar(
            reference["pixel_sample_stride"],
            "pixel_sample_stride",
            args.observation_topology,
        )
    )
    baseline_k = int(
        _scalar(
            reference["pixel_candidate_k"],
            "pixel_candidate_k",
            args.observation_topology,
        )
    )
    pixel_preselect_k = int(
        _scalar(
            reference["pixel_preselect_k"],
            "pixel_preselect_k",
            args.observation_topology,
        )
    )
    pixel_render_acc_min = float(
        _scalar(
            reference["pixel_render_acc_min"],
            "pixel_render_acc_min",
            args.observation_topology,
        )
    )
    pixel_min_contribution = float(
        _scalar(
            reference["pixel_min_contribution"],
            "pixel_min_contribution",
            args.observation_topology,
        )
    )
    k_values = parse_coverage_k_values(args.k_values, pixel_preselect_k)
    if baseline_k not in k_values:
        raise ValueError(
            f"--k-values must contain reference pixel_candidate_k={baseline_k}"
        )
    (
        fg_means,
        fg_scales,
        fg_quats,
        fg_opacities,
        _fg_colors,
        rendered_depths,
        rendered_accs,
    ) = load_fg_pixel_candidate_inputs_from_checkpoint(
        str(args.input_ckpt),
        [str(path) for path in args.view_config],
    )
    reference_points = np.asarray(reference["points_world"], dtype=np.float32)
    if fg_means.shape != reference_points.shape or not np.allclose(
        fg_means,
        reference_points,
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise ValueError(
            "Checkpoint foreground Gaussian centers do not match reference observations"
        )
    diagnostics = build_observation_coverage_diagnostics(
        points_world=fg_means,
        gaussian_scales=fg_scales,
        gaussian_quats=fg_quats,
        gaussian_opacities=fg_opacities,
        rendered_depths=rendered_depths,
        rendered_accs=rendered_accs,
        view_config_paths=args.view_config,
        mask_erode_iters=mask_erode_iters,
        pixel_sample_stride=pixel_sample_stride,
        pixel_preselect_k=pixel_preselect_k,
        pixel_render_acc_min=pixel_render_acc_min,
        pixel_min_contribution=pixel_min_contribution,
        k_values=k_values,
    )
    baseline_k_index = validate_reference_observation_replay(
        diagnostics,
        reference,
        baseline_k,
        args.observation_topology,
    )
    output = write_observation_coverage_diagnostics(
        args.out_npz,
        diagnostics,
        source_checkpoint=str(args.input_ckpt),
        reference_observation_path=args.observation_topology,
        baseline_k=baseline_k,
        baseline_k_index=baseline_k_index,
        mask_erode_iters=mask_erode_iters,
        pixel_sample_stride=pixel_sample_stride,
        pixel_preselect_k=pixel_preselect_k,
        pixel_render_acc_min=pixel_render_acc_min,
        pixel_min_contribution=pixel_min_contribution,
    )
    print(f"Exact K={baseline_k} replay matches {args.observation_topology}")
    _print_summary(diagnostics)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
