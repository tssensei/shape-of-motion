from __future__ import annotations

import argparse

from modal_surface.manifest_combine import combine_modal_manifests


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine two or more solved Gaussian modal manifests without "
            "recomputing latent modes."
        )
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Input modal_modes_manifest.json path. Repeat in any order.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New combined modal_modes_manifest.json path.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    output = combine_modal_manifests(args.manifest, args.output)
    print(f"Saved combined modal manifest -> {output}")


def main(argv: list[str] | None = None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
