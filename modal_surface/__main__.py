"""Package entrypoint for ``python -m modal_surface``."""

from __future__ import annotations

import argparse

from modal_surface.apps import solve_gaussian_modes


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and invoke the Gaussian modal solver."""
    parser = argparse.ArgumentParser(description="Foreground-Gaussian latent 3D modal field tools.")
    solve_gaussian_modes.add_arguments(parser)
    args = parser.parse_args(argv)
    solve_gaussian_modes.run(args)


if __name__ == "__main__":
    main()
