from __future__ import annotations

import argparse
import unittest

from modal_surface.optimization_staged import StagedSolverConfig
from modal_surface.solver_cli import (
    add_staged_solver_arguments,
    staged_solver_config,
    staged_solver_manifest_parameters,
)


class SolverCliTests(unittest.TestCase):
    def _standalone_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        add_staged_solver_arguments(parser)
        return parser

    def test_staged_solver_config_uses_current_defaults(self) -> None:
        args = self._standalone_parser().parse_args([])
        self.assertEqual(
            staged_solver_config(args),
            StagedSolverConfig(),
        )

    def test_staged_solver_options_are_added_to_config(self) -> None:
        args = self._standalone_parser().parse_args(
            ["--alpha-model", "bounded-complex", "--alpha-gain-min", "0.5", "--alpha-failure", "error"]
        )
        config = staged_solver_config(args)
        self.assertEqual(config.alpha_model, "bounded-complex")
        self.assertEqual(config.alpha_gain_min, 0.5)
        self.assertEqual(config.alpha_failure, "error")

    def test_manifest_parameters_identify_staged_solver(self) -> None:
        args = self._standalone_parser().parse_args([])
        parameters = staged_solver_manifest_parameters(args)
        self.assertEqual(parameters["solver"], "staged")
        self.assertEqual(parameters["alpha_solver_model"], "phase")

    def test_legacy_solver_options_are_not_registered(self) -> None:
        parser = self._standalone_parser()
        registered_options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        legacy_options = {
            "--solver",
            "--iterations",
            "--ridge-mu",
            "--outlier-frac",
            "--single-view-smooth-lambda",
            "--single-view-smooth-k",
            "--single-view-anchor-min-observations",
            "--graph-smooth-lambda",
            "--graph-smooth-k",
            "--graph-auto-radius-scale",
            "--graph-min-shared-views",
            "--modal-rigid-lambda",
            "--modal-rigid-k",
            "--modal-rigid-auto-radius-scale",
            "--modal-rigid-min-shared-views",
            "--modal-fill-unobserved",
            "--modal-fill-k",
            "--modal-fill-auto-radius-scale",
            "--modal-fill-anchor-min-observations",
            "--modal-fill-ridge-mu",
            "--obs-count-weight-1",
            "--obs-count-weight-2",
            "--obs-count-weight-3plus",
        }
        self.assertTrue(legacy_options.isdisjoint(registered_options))

if __name__ == "__main__":
    unittest.main()
