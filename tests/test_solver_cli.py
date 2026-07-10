from __future__ import annotations

import argparse
import unittest

from modal_surface.solver_cli import (
    add_solver_arguments,
    solver_kwargs,
    validate_solver_args,
)


class SolverCliTests(unittest.TestCase):
    def _standalone_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        add_solver_arguments(parser, outlier_default=None)
        return parser

    def test_solver_dependent_outlier_default(self) -> None:
        parser = self._standalone_parser()
        staged = parser.parse_args([])
        validate_solver_args(staged, legacy_outlier_default=0.05)
        self.assertEqual(staged.outlier_frac, 0.0)

        legacy = parser.parse_args(["--solver", "legacy-als"])
        validate_solver_args(legacy, legacy_outlier_default=0.05)
        self.assertEqual(legacy.outlier_frac, 0.05)

    def test_staged_rejects_changed_legacy_control(self) -> None:
        args = self._standalone_parser().parse_args(["--ridge-mu", "0"])
        with self.assertRaises(ValueError):
            validate_solver_args(args, legacy_outlier_default=0.05)

    def test_legacy_rejects_changed_staged_control(self) -> None:
        args = self._standalone_parser().parse_args(
            ["--solver", "legacy-als", "--alpha-model", "bounded-complex"]
        )
        with self.assertRaises(ValueError):
            validate_solver_args(args, legacy_outlier_default=0.05)

    def test_solver_kwargs_are_centralized(self) -> None:
        args = self._standalone_parser().parse_args([])
        validate_solver_args(args, legacy_outlier_default=0.05)
        kwargs = solver_kwargs(args)
        self.assertEqual(kwargs["solver"], "staged")
        self.assertEqual(kwargs["iterations"], 8)
        self.assertEqual(kwargs["outlier_frac"], 0.0)
        self.assertEqual(kwargs["alpha_model"], "phase")

    def test_gaussian_legacy_options_are_forwarded(self) -> None:
        parser = argparse.ArgumentParser()
        add_solver_arguments(
            parser,
            include_modal_rigid=True,
            include_modal_fill=True,
        )
        args = parser.parse_args(
            [
                "--solver",
                "legacy-als",
                "--modal-rigid-lambda",
                "0.5",
                "--modal-fill-unobserved",
            ]
        )
        validate_solver_args(args)
        kwargs = solver_kwargs(args)
        self.assertEqual(kwargs["modal_rigid_lambda"], 0.5)
        self.assertTrue(kwargs["modal_fill_unobserved"])


if __name__ == "__main__":
    unittest.main()
