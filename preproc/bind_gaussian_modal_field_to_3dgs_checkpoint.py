import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if int(manifest.get("version", -1)) != 1:
        raise ValueError(f"{path} must be a version 1 modal manifest")
    modes = manifest.get("modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError(f"{path} contains no modes")
    if manifest.get("point_type") != "foreground_gaussian_center":
        raise ValueError(
            f"{path} point_type={manifest.get('point_type')!r}; expected foreground_gaussian_center"
        )
    return manifest


def _load_checkpoint(path: Path) -> tuple[dict, dict, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state")
    if "trajectory_type_id" not in state:
        raise ValueError(f"{path} has no trajectory_type_id; expected a static checkpoint")
    trajectory_type_id = int(state["trajectory_type_id"].item())
    if trajectory_type_id != 3:
        raise ValueError(f"{path} trajectory_type_id={trajectory_type_id}; expected 3 for static")
    if (
        "modal.params.activations" in state
        or "modal.params.envelope_knots" in state
    ):
        raise ValueError("Refusing to bind synthetic modal fields to a learned activation checkpoint")
    for key in ("modal_phi_real", "modal_phi_imag", "modal_freqs_hz"):
        if key in state and state[key].numel() > 0:
            raise ValueError(f"{path} already contains {key}; use an unbound checkpoint")
    if "modal_obs_count_per_point" in state and state["modal_obs_count_per_point"].numel() > 0:
        raise ValueError(f"{path} already contains modal_obs_count_per_point; use an unbound checkpoint")
    if "fg.params.means" not in state:
        raise ValueError(f"{path} is missing fg.params.means")
    fg_means = state["fg.params.means"].detach().cpu().float()
    if fg_means.ndim != 2 or fg_means.shape[1] != 3:
        raise ValueError(f"fg.params.means must have shape (N,3), got {tuple(fg_means.shape)}")
    return ckpt, state, fg_means


def _latent_path(manifest_path: Path, mode_entry: dict) -> Path:
    latent_rel = mode_entry.get("latent_path")
    if not isinstance(latent_rel, str) or not latent_rel:
        raise ValueError("Each manifest mode entry must contain latent_path")
    latent_path = Path(latent_rel)
    if not latent_path.is_absolute():
        latent_path = manifest_path.parent / latent_path
    if not latent_path.exists():
        raise FileNotFoundError(latent_path)
    return latent_path


def _scalar_string(value: np.ndarray) -> str:
    arr = np.asarray(value)
    if arr.shape != ():
        raise ValueError(f"Expected scalar string field, got shape {arr.shape}")
    return str(arr.item())


def _mode_alpha_by_view(mode_entry: dict) -> list[dict]:
    alpha = mode_entry.get("alpha_by_view")
    if alpha is None:
        return []
    if not isinstance(alpha, list):
        raise ValueError("alpha_by_view must be a list when present")
    return alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ckpt", required=True, type=Path)
    parser.add_argument("--gaussian-modal-manifest", required=True, type=Path)
    parser.add_argument("--output-ckpt", required=True, type=Path)
    parser.add_argument("--position-atol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.output_ckpt.resolve() == args.input_ckpt.resolve():
        raise ValueError("--output-ckpt must be different from --input-ckpt")

    ckpt, state, fg_means = _load_checkpoint(args.input_ckpt)
    manifest = _load_manifest(args.gaussian_modal_manifest)

    fg_points = fg_means.numpy().astype(np.float32)
    num_fg = int(fg_points.shape[0])
    phi_real: list[np.ndarray] = []
    phi_imag: list[np.ndarray] = []
    obs_counts: list[np.ndarray] = []
    missing_obs_count_modes: list[str] = []
    freqs_hz: list[float] = []
    mode_stats: list[dict] = []
    max_position_abs_delta = 0.0

    for mode_entry in manifest["modes"]:
        latent_path = _latent_path(args.gaussian_modal_manifest, mode_entry)
        latent = np.load(str(latent_path), allow_pickle=False)
        required = ("points_world", "phi", "gaussian_indices", "freq_hz")
        missing = [key for key in required if key not in latent.files]
        if missing:
            raise ValueError(f"{latent_path} missing required fields: {missing}")

        if "point_type" in latent.files:
            point_type = _scalar_string(latent["point_type"])
            if point_type != "foreground_gaussian_center":
                raise ValueError(f"{latent_path} point_type={point_type!r}; expected foreground_gaussian_center")

        points = latent["points_world"].astype(np.float32)
        phi = latent["phi"]
        gaussian_indices = latent["gaussian_indices"].astype(np.int32)
        if points.shape != (num_fg, 3):
            raise ValueError(f"{latent_path} points_world shape {points.shape} does not match foreground {(num_fg, 3)}")
        if phi.shape != (num_fg, 3):
            raise ValueError(f"{latent_path} phi shape {phi.shape} does not match foreground {(num_fg, 3)}")
        if gaussian_indices.shape != (num_fg,):
            raise ValueError(f"{latent_path} gaussian_indices shape {gaussian_indices.shape} does not match foreground {(num_fg,)}")
        expected_indices = np.arange(num_fg, dtype=np.int32)
        if not np.array_equal(gaussian_indices, expected_indices):
            raise ValueError(f"{latent_path} gaussian_indices are not contiguous foreground indices")

        position_abs_delta = float(np.max(np.abs(points - fg_points))) if num_fg > 0 else 0.0
        max_position_abs_delta = max(max_position_abs_delta, position_abs_delta)
        if position_abs_delta > args.position_atol:
            raise ValueError(
                f"{latent_path} points_world differs from checkpoint fg.params.means by max "
                f"{position_abs_delta:.6g}, above --position-atol {args.position_atol:.6g}"
            )

        if not np.iscomplexobj(phi):
            phi = phi.astype(np.complex64)
        phi = phi.astype(np.complex64)
        phi_real.append(phi.real.astype(np.float32))
        phi_imag.append(phi.imag.astype(np.float32))
        freqs_hz.append(float(np.asarray(latent["freq_hz"]).item()))

        obs_count = latent["obs_count_per_point"].astype(np.int32) if "obs_count_per_point" in latent.files else None
        if obs_count is not None:
            if obs_count.shape != (num_fg,):
                raise ValueError(
                    f"{latent_path} obs_count_per_point shape {obs_count.shape} "
                    f"does not match foreground {(num_fg,)}"
                )
            obs_counts.append(obs_count)
        else:
            missing_obs_count_modes.append(str(latent_path))
        residual = latent["point_residual"].astype(np.float32) if "point_residual" in latent.files else None
        stats = {
            "mode_index": int(mode_entry.get("mode_index", len(mode_stats))),
            "freq_hz": float(freqs_hz[-1]),
            "latent_path": str(latent_path),
            "alpha_by_view": _mode_alpha_by_view(mode_entry),
        }
        if obs_count is not None:
            stats.update(
                {
                    "obs_count_zero": int((obs_count == 0).sum()),
                    "obs_count_p50": float(np.percentile(obs_count, 50)),
                    "obs_count_p90": float(np.percentile(obs_count, 90)),
                }
            )
        if residual is not None:
            stats.update(
                {
                    "residual_median": float(np.median(residual)),
                    "residual_p90": float(np.percentile(residual, 90)),
                    "residual_max": float(np.max(residual)),
                }
            )
        mode_stats.append(stats)

    state["modal_phi_real"] = torch.from_numpy(np.stack(phi_real, axis=0)).float()
    state["modal_phi_imag"] = torch.from_numpy(np.stack(phi_imag, axis=0)).float()
    state["modal_freqs_hz"] = torch.tensor(freqs_hz, dtype=torch.float32)
    state["modal_phi_trainable_mask"] = torch.zeros(
        (len(phi_real), num_fg), dtype=torch.bool
    )
    if obs_counts:
        if missing_obs_count_modes:
            raise ValueError(
                "Modal obs_count_per_point must be present for every mode or no modes; "
                f"missing: {missing_obs_count_modes}"
            )
        state["modal_obs_count_per_point"] = torch.from_numpy(np.stack(obs_counts, axis=0)).long()
    state["modal_synthetic_enabled"] = torch.tensor(True)

    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output_ckpt)

    diagnostics = {
        "binding_type": "direct_gaussian_solve",
        "input_ckpt": str(args.input_ckpt),
        "output_ckpt": str(args.output_ckpt),
        "gaussian_modal_manifest": str(args.gaussian_modal_manifest),
        "source_checkpoint_in_manifest": manifest.get("source_checkpoint"),
        "num_modes": len(phi_real),
        "num_fg_gaussians": num_fg,
        "has_modal_obs_count_per_point": bool(obs_counts),
        "max_position_abs_delta": max_position_abs_delta,
        "position_atol": float(args.position_atol),
        "freqs_hz": freqs_hz,
        "mode_stats": mode_stats,
    }
    diag_path = args.output_ckpt.with_suffix(".modal_binding.json")
    with diag_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"Wrote direct Gaussian modal-bound checkpoint to {args.output_ckpt}")
    print(f"Wrote diagnostics to {diag_path}")
    print(f"Bound {len(phi_real)} modes directly to {num_fg} foreground Gaussians")
    print(f"max |latent point - fg mean|: {max_position_abs_delta:.6g}")


if __name__ == "__main__":
    main()
