"""Command-line dispatcher for the 3D modal surface pipeline."""

from __future__ import annotations

import argparse
from importlib import import_module
from typing import NamedTuple


class _CommandSpec(NamedTuple):
    module: str
    help: str


COMMANDS: dict[str, _CommandSpec] = {
    "match-carrier-views": _CommandSpec(
        module="modal_surface.apps.match_carrier_views",
        help="Build an N-view observation graph from VGGT carrier points.",
    ),
    "optimize-multi-view": _CommandSpec(
        module="modal_surface.apps.optimize_multi_view",
        help="Optimize latent 3D modal displacement from N-view observations.",
    ),
    "solve-carrier-modes": _CommandSpec(
        module="modal_surface.apps.solve_carrier_modes",
        help="Batch solve VGGT carrier modal fields for multiple mode indices.",
    ),
    "solve-gaussian-modes": _CommandSpec(
        module="modal_surface.apps.solve_gaussian_modes",
        help="Batch solve modal fields directly on foreground 3DGS Gaussian centers.",
    ),
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the canonical command-line parser for modal_surface."""
    parser = argparse.ArgumentParser(description="VGGT-carrier latent 3D modal field tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, spec in COMMANDS.items():
        app = import_module(spec.module)
        command_parser = subparsers.add_parser(command, help=spec.help)
        app.add_arguments(command_parser)
        command_parser.set_defaults(_runner=app.run)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and invoke the selected command adapter."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args._runner(args)
