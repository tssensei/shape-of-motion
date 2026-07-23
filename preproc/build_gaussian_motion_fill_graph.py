"""Build one reusable foreground-Gaussian motion-fill KNN graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from modal_surface.checkpoint_render_inputs import load_fg_means_from_checkpoint
from modal_surface.gaussian_motion_fill import (
    MOTION_FILL_EPSILON,
    write_motion_fill_graph,
)
from modal_surface.motion_fill import build_knn_graph, query_knn_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reusable foreground-Gaussian KNN graph for modal motion fill"
        )
    )
    parser.add_argument("--input-ckpt", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-distance", type=float, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_checkpoint = args.input_ckpt.expanduser().resolve(strict=True)
    output_path = args.out_npz.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    foreground_means = load_fg_means_from_checkpoint(str(input_checkpoint))
    candidates = query_knn_candidates(foreground_means, int(args.k))
    graph = build_knn_graph(
        candidates,
        int(args.k),
        float(args.max_distance),
        MOTION_FILL_EPSILON,
    )
    write_motion_fill_graph(
        output_path,
        foreground_means,
        candidates,
        graph,
    )
    print(
        "Motion-fill graph: "
        f"points={graph.num_points}, "
        f"edges={graph.edge_index.shape[0]}, "
        f"components={graph.component_sizes.shape[0]}, "
        f"isolated={int(graph.isolated_mask.sum())}"
    )
    print(f"Wrote motion-fill graph -> {output_path}")


if __name__ == "__main__":
    main()
