import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flow3d.modal_utils import interpolate_modal_modes_to_gaussians, load_modal_modes


def bbox_stats(points: np.ndarray) -> dict[str, object]:
    points = np.asarray(points, dtype=np.float32)
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    extent = max_xyz - min_xyz
    return {
        "center": ((min_xyz + max_xyz) * 0.5).tolist(),
        "extent": extent.tolist(),
        "diag": float(np.linalg.norm(extent)),
        "min": min_xyz.tolist(),
        "max": max_xyz.tolist(),
    }


def check_bbox_compat(
    gaussian_points: np.ndarray,
    carrier_points: np.ndarray,
    min_diag_ratio: float,
    max_diag_ratio: float,
    max_center_delta_ratio: float,
) -> dict[str, object]:
    gaussian = bbox_stats(gaussian_points)
    carrier = bbox_stats(carrier_points)
    gaussian_diag = float(gaussian["diag"])
    carrier_diag = float(carrier["diag"])
    if gaussian_diag <= 0 or carrier_diag <= 0:
        raise ValueError("Gaussian and carrier bounding boxes must have positive extent")
    diag_ratio = gaussian_diag / carrier_diag
    gaussian_center = np.asarray(gaussian["center"], dtype=np.float32)
    carrier_center = np.asarray(carrier["center"], dtype=np.float32)
    center_delta = gaussian_center - carrier_center
    center_delta_norm = float(np.linalg.norm(center_delta))
    center_delta_ratio = center_delta_norm / carrier_diag
    if diag_ratio < min_diag_ratio or diag_ratio > max_diag_ratio:
        raise ValueError(
            f"Gaussian/carrier bbox diag ratio {diag_ratio:.6g} outside "
            f"[{min_diag_ratio:.6g}, {max_diag_ratio:.6g}]"
        )
    if center_delta_ratio > max_center_delta_ratio:
        raise ValueError(
            f"Gaussian/carrier center delta ratio {center_delta_ratio:.6g} exceeds "
            f"{max_center_delta_ratio:.6g}"
        )
    return {
        "gaussian": gaussian,
        "carrier": carrier,
        "diag_ratio_gaussian_over_carrier": diag_ratio,
        "center_delta": center_delta.tolist(),
        "center_delta_ratio": center_delta_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ckpt", required=True, type=Path)
    parser.add_argument("--output-ckpt", required=True, type=Path)
    parser.add_argument("--modal-manifest", required=True, type=Path)
    parser.add_argument("--modal-carrier-points", required=True, type=Path)
    parser.add_argument("--modal-knn", type=int, default=8)
    parser.add_argument("--modal-interp-power", type=float, default=2.0)
    parser.add_argument("--modal-interp-eps", type=float, default=1e-6)
    parser.add_argument("--min-diag-ratio", type=float, default=0.25)
    parser.add_argument("--max-diag-ratio", type=float, default=4.0)
    parser.add_argument("--max-center-delta-ratio", type=float, default=1.0)
    args = parser.parse_args()

    if not args.input_ckpt.exists():
        raise FileNotFoundError(args.input_ckpt)
    if not args.modal_carrier_points.exists():
        raise FileNotFoundError(args.modal_carrier_points)
    if not args.modal_manifest.exists():
        raise FileNotFoundError(args.modal_manifest)

    ckpt = torch.load(args.input_ckpt, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise ValueError(f"{args.input_ckpt} does not contain a model state")
    model_state = ckpt["model"]
    if "trajectory_type_id" not in model_state:
        raise ValueError(f"{args.input_ckpt} has no trajectory_type_id; expected a static checkpoint")
    trajectory_type_id = int(model_state["trajectory_type_id"].item())
    if trajectory_type_id != 3:
        raise ValueError(
            f"{args.input_ckpt} trajectory_type_id={trajectory_type_id}; expected 3 for static"
        )
    if "modal.params.activations" in model_state:
        raise ValueError("Refusing to bind synthetic modal fields to a learned activation checkpoint")
    for key in ("modal_phi_real", "modal_phi_imag", "modal_freqs_hz"):
        if key in model_state:
            raise ValueError(f"{args.input_ckpt} already contains {key}; use an unbound checkpoint")
    if "fg.params.means" not in model_state:
        raise ValueError(f"{args.input_ckpt} is missing fg.params.means")

    carrier = np.load(args.modal_carrier_points, allow_pickle=False)
    if "points_world" not in carrier.files:
        raise ValueError(f"{args.modal_carrier_points} missing points_world")
    carrier_points = carrier["points_world"].astype(np.float32)
    gaussian_means = model_state["fg.params.means"].detach().cpu().float()
    bbox = check_bbox_compat(
        gaussian_means.numpy(),
        carrier_points,
        args.min_diag_ratio,
        args.max_diag_ratio,
        args.max_center_delta_ratio,
    )

    modes = load_modal_modes(str(args.modal_manifest))
    with torch.no_grad():
        phi_real, phi_imag, freqs_hz, interp_stats = interpolate_modal_modes_to_gaussians(
            gaussian_means,
            modes,
            args.modal_knn,
            args.modal_interp_power,
            args.modal_interp_eps,
        )

    model_state["modal_phi_real"] = phi_real.cpu()
    model_state["modal_phi_imag"] = phi_imag.cpu()
    model_state["modal_freqs_hz"] = freqs_hz.cpu()
    model_state["modal_synthetic_enabled"] = torch.tensor(True)

    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output_ckpt)

    diagnostics = {
        "input_ckpt": str(args.input_ckpt),
        "output_ckpt": str(args.output_ckpt),
        "modal_manifest": str(args.modal_manifest),
        "modal_carrier_points": str(args.modal_carrier_points),
        "num_modes": len(modes),
        "num_fg_gaussians": int(gaussian_means.shape[0]),
        "bbox": bbox,
        "interp_stats": interp_stats,
        "freqs_hz": freqs_hz.cpu().numpy().tolist(),
    }
    diag_path = args.output_ckpt.with_suffix(".modal_binding.json")
    with diag_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"Wrote modal-bound checkpoint to {args.output_ckpt}")
    print(f"Wrote diagnostics to {diag_path}")
    print(
        "carrier-to-Gaussian nearest distance p50/p95/max: "
        f"{interp_stats['nearest_p50']:.6g} / "
        f"{interp_stats['nearest_p95']:.6g} / "
        f"{interp_stats['nearest_max']:.6g}"
    )


if __name__ == "__main__":
    main()
