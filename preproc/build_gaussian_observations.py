"""Build per-mode Gaussian observation measurements from shared topology."""

from __future__ import annotations

import argparse
from pathlib import Path

from modal_surface.gaussian_observations import (
    build_gaussian_observation_measurements,
    load_gaussian_observation_topology,
    write_gaussian_observation_measurement,
)
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
            "Build lightweight per-mode measurements from a reusable "
            "foreground-Gaussian observation topology without solving modes"
        )
    )
    parser.add_argument(
        "--observation-topology",
        type=Path,
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
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    topology_path = args.observation_topology.expanduser().resolve(strict=True)
    modal_npzs = [
        path.expanduser().resolve(strict=True) for path in args.modal_npz
    ]

    modal_npz_paths = [str(path) for path in modal_npzs]
    frequencies_by_view = load_modal_freqs(modal_npz_paths)
    mode_indices = _parse_mode_indices(
        args.mode_indices,
        int(frequencies_by_view[0].shape[0]),
    )
    out_dir = args.out_dir.expanduser().resolve()
    measurements_dir = out_dir / "measurements"
    output_paths = {
        mode_index: out_dir
        / "measurements"
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
    measurements_dir.mkdir(parents=True, exist_ok=True)

    topology = load_gaussian_observation_topology(topology_path)
    measurements = build_gaussian_observation_measurements(
        topology,
        modal_npz_paths,
        mode_indices,
        float(args.freq_tolerance_hz),
    )
    for mode_index, measurement in zip(mode_indices, measurements):
        output_path = output_paths[mode_index]
        write_gaussian_observation_measurement(output_path, measurement)
        print(f"Wrote Gaussian observation measurement -> {output_path}")


if __name__ == "__main__":
    main()
