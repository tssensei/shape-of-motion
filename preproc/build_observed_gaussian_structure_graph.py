"""Build a structure graph from an existing Gaussian observation artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from modal_surface.checkpoint_render_inputs import (
    load_fg_pixel_candidate_inputs_from_checkpoint,
)
from modal_surface.io import load_view_config
from modal_surface.observed_structure_graph import (
    ObservedStructureGraphConfig,
    build_observed_structure_graph,
    write_observed_structure_graph,
)


_REFERENCE_REQUIRED_FIELDS = {
    "points_world",
    "gaussian_indices",
    "point_type",
    "source_checkpoint",
    "obs_point_index",
    "obs_view_index",
    "obs_contribution_weight",
    "view_ids",
    "freq_hz",
    "mode_index",
    "pixel_render_acc_min",
    "source_view_configs",
}


def _scalar(array: np.ndarray, name: str, path: Path) -> object:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{path} field {name} must be scalar")
    return value.item()


def load_reference_observations(
    path: Path,
    *,
    input_checkpoint: Path,
    view_config_paths: list[Path],
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(_REFERENCE_REQUIRED_FIELDS - set(archive.files))
        if missing:
            raise ValueError(f"{path} missing required fields: {missing}")
        reference = {
            name: np.asarray(archive[name])
            for name in _REFERENCE_REQUIRED_FIELDS
        }

    if str(_scalar(reference["point_type"], "point_type", path)) != (
        "foreground_gaussian_center"
    ):
        raise ValueError(f"{path} point_type is incompatible")
    if str(
        _scalar(reference["source_checkpoint"], "source_checkpoint", path)
    ) != str(input_checkpoint):
        raise ValueError(f"{path} source_checkpoint does not match --input-ckpt")

    points = np.asarray(reference["points_world"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{path} points_world must be finite (N,3)")
    indices = np.asarray(reference["gaussian_indices"])
    if (
        indices.shape != (points.shape[0],)
        or not np.issubdtype(indices.dtype, np.integer)
        or not np.array_equal(
            indices,
            np.arange(points.shape[0], dtype=indices.dtype),
        )
    ):
        raise ValueError(
            f"{path} gaussian_indices must preserve contiguous checkpoint order"
        )

    point_index = np.asarray(reference["obs_point_index"])
    view_index = np.asarray(reference["obs_view_index"])
    weights = np.asarray(reference["obs_contribution_weight"])
    if point_index.shape != view_index.shape or point_index.shape != weights.shape:
        raise ValueError(f"{path} observation row arrays must have matching shapes")
    if point_index.ndim != 1 or not np.issubdtype(point_index.dtype, np.integer):
        raise ValueError(f"{path} obs_point_index must be 1-D integers")
    if not np.issubdtype(view_index.dtype, np.integer):
        raise ValueError(f"{path} obs_view_index must be integers")
    if np.any(point_index < 0) or np.any(point_index >= points.shape[0]):
        raise ValueError(f"{path} obs_point_index is out of range")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError(
            f"{path} obs_contribution_weight must be finite and non-negative"
        )

    configs = [load_view_config(config_path) for config_path in view_config_paths]
    expected_view_ids = np.asarray([config.view_id for config in configs]).astype(str)
    view_ids = np.asarray(reference["view_ids"]).astype(str)
    if not np.array_equal(view_ids, expected_view_ids):
        raise ValueError(f"{path} view_ids do not match --view-config order")
    if np.any(view_index < 0) or np.any(view_index >= view_ids.shape[0]):
        raise ValueError(f"{path} obs_view_index is out of range")
    expected_config_paths = np.asarray(
        [str(config_path) for config_path in view_config_paths]
    )
    if not np.array_equal(
        np.asarray(reference["source_view_configs"]).astype(str),
        expected_config_paths,
    ):
        raise ValueError(
            f"{path} source_view_configs do not match --view-config arguments"
        )
    return reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a color-depth structure graph over every Gaussian with a "
            "positive row in an existing observation artifact; no modal solve "
            "or motion fill is run"
        )
    )
    parser.add_argument("--input-ckpt", type=Path, required=True)
    parser.add_argument(
        "--view-config",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--reference-observations", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, required=True)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--color-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--depth-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--depth-samples", type=int, default=5)
    parser.add_argument("--min-shared-views", type=int, default=1)
    parser.add_argument("--min-component-nodes", type=int, default=4)
    parser.add_argument("--min-component-edges", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reference = load_reference_observations(
        args.reference_observations,
        input_checkpoint=args.input_ckpt,
        view_config_paths=args.view_config,
    )
    configs = [load_view_config(path) for path in args.view_config]
    render_acc_min = float(
        _scalar(
            reference["pixel_render_acc_min"],
            "pixel_render_acc_min",
            args.reference_observations,
        )
    )
    config = ObservedStructureGraphConfig(
        max_neighbors=args.max_neighbors,
        max_distance=args.max_distance,
        color_mad_multiplier=args.color_mad_multiplier,
        depth_mad_multiplier=args.depth_mad_multiplier,
        depth_samples=args.depth_samples,
        min_shared_views=args.min_shared_views,
        min_component_nodes=args.min_component_nodes,
        min_component_edges=args.min_component_edges,
        render_acc_min=render_acc_min,
    )
    config.validate(num_nodes=0, num_views=len(configs))

    (
        fg_means,
        _fg_scales,
        _fg_quats,
        _fg_opacities,
        fg_colors,
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
            "Checkpoint foreground Gaussian centers do not match reference "
            "observations"
        )

    graph = build_observed_structure_graph(
        points_world=fg_means,
        colors_rgb=fg_colors,
        obs_point_index=reference["obs_point_index"],
        obs_view_index=reference["obs_view_index"],
        obs_weights=reference["obs_contribution_weight"],
        view_ids=reference["view_ids"],
        Ks=np.stack([config_value.K for config_value in configs], axis=0),
        world_to_cameras=np.stack(
            [config_value.world_to_camera for config_value in configs],
            axis=0,
        ),
        rendered_depths=rendered_depths,
        rendered_accs=rendered_accs,
        config=config,
    )
    output = write_observed_structure_graph(
        args.out_npz,
        graph,
        mode_index=int(
            _scalar(reference["mode_index"], "mode_index", args.reference_observations)
        ),
        freq_hz=float(
            _scalar(reference["freq_hz"], "freq_hz", args.reference_observations)
        ),
        source_checkpoint=str(args.input_ckpt),
        source_observation_path=str(args.reference_observations),
        num_foreground_gaussians=fg_means.shape[0],
    )
    print(
        "Observed structure graph: "
        f"nodes={graph.counts['node_count']}, "
        f"single_view={graph.counts['single_view_node_count']}, "
        f"multi_view={graph.counts['multi_view_node_count']}, "
        f"edges={graph.counts['retained_edge_count']}, "
        f"isolated={graph.counts['isolated_node_count']}, "
        f"pruned_components={graph.counts['component_pruned_component_count']}, "
        f"pruned_nodes={graph.counts['component_pruned_node_count']}, "
        f"pruned_edges={graph.counts['component_pruned_edge_count']}"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
