from __future__ import annotations

import argparse
import json

from flow3d.modal_physics_coordinates import (
    PHYSICS_DIAGNOSTICS_JSON_FILENAME,
    postfit_modal_physics_coordinates,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-fit free per-frame modal coordinates with a damped oscillator "
            "and a regularized per-view latent force."
        )
    )
    parser.add_argument(
        "--input-coordinates",
        required=True,
        help="Reference-flow or rendered-projection ridge modal_flow_coordinates.npz.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="New immutable physics-coordinate output directory.",
    )
    parser.add_argument(
        "--damping-ratio",
        type=float,
        default=0.05,
        help="Shared non-negative modal damping ratio zeta.",
    )
    parser.add_argument(
        "--forcing-weight",
        type=float,
        default=0.1,
        help="Non-negative normalized latent-force energy weight.",
    )
    parser.add_argument(
        "--forcing-difference-weight",
        type=float,
        default=0.0,
        help="Non-negative adjacent latent-force difference weight.",
    )
    parser.add_argument(
        "--assigned-band-half-width-hz",
        type=float,
        default=0.1,
        help="Positive half-width used only for assigned-frequency energy diagnostics.",
    )
    parser.add_argument(
        "--frame-chunk-size",
        type=int,
        default=64,
        help="Positive frame chunk size for post-fit flow re-evaluation.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    result = postfit_modal_physics_coordinates(
        input_coordinates=args.input_coordinates,
        out_dir=args.out_dir,
        damping_ratio=args.damping_ratio,
        forcing_weight=args.forcing_weight,
        forcing_difference_weight=args.forcing_difference_weight,
        assigned_band_half_width_hz=args.assigned_band_half_width_hz,
        frame_chunk_size=args.frame_chunk_size,
    )
    diagnostics_path = result.path.parent / PHYSICS_DIAGNOSTICS_JSON_FILENAME
    with diagnostics_path.open("r", encoding="utf-8") as file:
        diagnostics = json.load(file)
    overall = diagnostics["overall"]
    print(f"Saved physics-constrained modal coordinates -> {result.path.parent}")
    print(
        "Coordinate field: "
        f"views={len(result.view_ids)}, frames={len(result.frame_names)}, "
        f"modes={result.mode_indices.size}"
    )
    print(
        "Flow R2: "
        f"{float(overall['input_flow_r2']):.6f} -> "
        f"{float(overall['output_flow_r2']):.6f} "
        f"(delta={float(overall['flow_r2_delta']):+.6f})"
    )
    print(
        "Physics summary: "
        f"fidelity NRMSE={float(overall['mean_fidelity_nrmse']):.6g}, "
        f"p99 retention={float(overall['mean_p99_retention']):.6g}, "
        "assigned-band energy="
        f"{float(overall['mean_input_assigned_frequency_energy_ratio']):.6g} -> "
        f"{float(overall['mean_output_assigned_frequency_energy_ratio']):.6g}"
    )
    for view in diagnostics["views"]:
        print(
            f"  {view['view_id']}: flow R2 "
            f"{float(view['input_flow_r2']):.6f} -> "
            f"{float(view['output_flow_r2']):.6f}; strong-motion R2 "
            f"{float(view['input_strong_motion_flow_r2']):.6f} -> "
            f"{float(view['output_strong_motion_flow_r2']):.6f}"
        )


def main(argv: list[str] | None = None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
