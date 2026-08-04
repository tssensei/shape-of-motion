"""Build a rendered foreground-Gaussian design for modal flow coordinates."""

from __future__ import annotations

import argparse

from flow3d.modal_rendered_design import build_rendered_modal_design


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render complete foreground Gaussian modal fields into an "
            "alpha-normalized flow projection design."
        )
    )
    parser.add_argument("--input-ckpt", required=True)
    parser.add_argument("--modal-manifest", required=True)
    parser.add_argument("--view-config", action="append", required=True)
    parser.add_argument(
        "--flow-cache",
        action="append",
        required=True,
        help="Ordered VIEW_ID=PATH flow cache; repeat once per view.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pixel-sample-stride", type=int, default=2)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--mask-erode-iters", type=int, default=1)
    parser.add_argument("--modes-per-batch", type=int, default=8)
    parser.add_argument("--use-2dgs", action="store_true")
    parser.add_argument("--write-role-diagnostics", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    design = build_rendered_modal_design(
        input_ckpt=args.input_ckpt,
        modal_manifest=args.modal_manifest,
        view_configs=args.view_config,
        flow_caches=args.flow_cache,
        out_dir=args.out_dir,
        pixel_sample_stride=args.pixel_sample_stride,
        alpha_min=args.alpha_min,
        mask_erode_iters=args.mask_erode_iters,
        modes_per_batch=args.modes_per_batch,
        use_2dgs=args.use_2dgs,
        write_role_diagnostics=args.write_role_diagnostics,
    )
    print(f"Saved rendered modal design -> {design.path}")
    print(
        f"Views={len(design.view_ids)}, samples={design.sample_view_index.size}, "
        f"modes={design.mode_indices.size}, identity={design.artifact_identity}"
    )


if __name__ == "__main__":
    main()
