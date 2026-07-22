"""Diagnose within-view and cross-view causes of staged anchor rejection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from modal_surface.anchor_residual_diagnostics import (
    RESIDUAL_SOURCE_NAMES,
    AnchorResidualDiagnostics,
    build_anchor_residual_diagnostics,
    write_anchor_residual_diagnostics,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _scalar_string(array: np.ndarray, name: str, path: Path) -> str:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{path} field {name} must be scalar")
    item = value.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str) or not item:
        raise ValueError(f"{path} field {name} must be a non-empty string")
    return item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exactly decompose staged Gaussian anchor residuals into "
            "within-view pixel dispersion and cross-view disagreement"
        )
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--solver-diagnostics", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    return parser


def _percentiles(values: np.ndarray) -> str:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "no finite values"
    result = np.percentile(finite, [10, 50, 90, 99])
    return " ".join(
        f"p{percentile}={value:.6g}"
        for percentile, value in zip((10, 50, 90, 99), result.tolist())
    )


def print_summary(diagnostics: AnchorResidualDiagnostics) -> None:
    selected = diagnostics.selected_multiview_mask
    candidate = diagnostics.residual_candidate_mask
    rejected = diagnostics.residual_rejected_mask
    anchor = diagnostics.anchor_mask
    print("Anchor residual waterfall")
    print(f"  selected multiview: {int(selected.sum())}")
    print(f"  residual candidates: {int(candidate.sum())}")
    print(f"  residual rejected: {int(rejected.sum())}")
    print(f"  accepted anchors: {int(anchor.sum())}")
    print(
        "  residual threshold: "
        f"{diagnostics.anchor_residual_threshold:.6g}"
    )
    print()
    print("Residual-rejected source classes")
    for class_index, name in enumerate(RESIDUAL_SOURCE_NAMES):
        if class_index < 2:
            continue
        count = int(
            np.count_nonzero(
                rejected
                & (diagnostics.residual_source_class == class_index)
            )
        )
        rate = 100.0 * count / max(int(rejected.sum()), 1)
        print(f"  {name}: {count} ({rate:.2f}%)")

    within_sum = float(np.sum(diagnostics.point_within_sse[rejected]))
    cross_sum = float(np.sum(diagnostics.point_cross_sse[rejected]))
    total = within_sum + cross_sum
    print()
    print("Rejected weighted SSE")
    print(
        f"  within-view: {within_sum:.6g} "
        f"({100.0 * within_sum / max(total, 1.0e-12):.2f}%)"
    )
    print(
        f"  cross-view: {cross_sum:.6g} "
        f"({100.0 * cross_sum / max(total, 1.0e-12):.2f}%)"
    )
    print()
    print("Rejected-point distributions")
    for name, values in (
        ("total residual", diagnostics.point_precompletion_residual[rejected]),
        ("within residual", diagnostics.point_within_residual[rejected]),
        ("cross residual", diagnostics.point_cross_residual[rejected]),
        ("within fraction", diagnostics.point_within_fraction[rejected]),
        ("modal signal RMS", diagnostics.point_signal_rms[rejected]),
    ):
        print(f"  {name}: {_percentiles(values)}")

    print()
    print("Per-view rejected weighted SSE")
    for view_index, view_id in enumerate(diagnostics.view_ids.tolist()):
        within = float(
            np.sum(diagnostics.view_within_sse[rejected, view_index])
        )
        cross = float(
            np.sum(diagnostics.view_cross_sse[rejected, view_index])
        )
        print(f"  {view_id}: within={within:.6g} cross={cross:.6g}")
    print(
        "  robust low-modal-energy threshold: "
        f"{diagnostics.low_modal_energy_threshold:.6g}"
    )


def main() -> None:
    args = build_parser().parse_args()
    observations = _load_npz(args.observations)
    solver_diagnostics = _load_npz(args.solver_diagnostics)
    if "source_checkpoint" not in observations:
        raise ValueError(
            f"{args.observations} missing source_checkpoint provenance"
        )
    source_checkpoint = _scalar_string(
        observations["source_checkpoint"],
        "source_checkpoint",
        args.observations,
    )
    diagnostics = build_anchor_residual_diagnostics(
        observations,
        solver_diagnostics,
    )
    output = write_anchor_residual_diagnostics(
        args.out_npz,
        diagnostics,
        source_checkpoint=source_checkpoint,
        source_observation_path=args.observations,
        source_solver_diagnostics_path=args.solver_diagnostics,
    )
    print(
        "Exact replay matched point_precompletion_residual and weighted SSE "
        "decomposition."
    )
    print_summary(diagnostics)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
