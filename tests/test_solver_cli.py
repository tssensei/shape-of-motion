from __future__ import annotations

import argparse
from pathlib import Path
import unittest
from unittest import mock

from modal_surface.optimization_multi import optimize_multi_view
from modal_surface.solver_cli import (
    add_staged_solver_arguments,
    staged_solver_kwargs,
    staged_solver_manifest_parameters,
)


class SolverCliTests(unittest.TestCase):
    def _standalone_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        add_staged_solver_arguments(parser)
        return parser

    def test_staged_solver_kwargs_use_current_defaults(self) -> None:
        args = self._standalone_parser().parse_args([])
        self.assertEqual(
            staged_solver_kwargs(args),
            {
                "alpha_model": "phase",
                "alpha_gain_min": 0.25,
                "alpha_gain_max": 4.0,
                "alpha_min_shared_points": 16,
                "alpha_rank_ratio_min": 1e-4,
                "alpha_info_ratio_min": 1e-4,
                "alpha_failure": "exclude",
                "anchor_svd_ratio_min": 1e-2,
                "anchor_residual_max": 0.1,
            },
        )

    def test_staged_solver_options_are_forwarded(self) -> None:
        args = self._standalone_parser().parse_args(
            ["--alpha-model", "bounded-complex", "--alpha-gain-min", "0.5", "--alpha-failure", "error"]
        )
        kwargs = staged_solver_kwargs(args)
        self.assertEqual(kwargs["alpha_model"], "bounded-complex")
        self.assertEqual(kwargs["alpha_gain_min"], 0.5)
        self.assertEqual(kwargs["alpha_failure"], "error")

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

    def test_public_solver_facade_builds_staged_config(self) -> None:
        expected = Path("latent.npz")
        with mock.patch(
            "modal_surface.optimization_multi.optimize_multi_view_staged",
            return_value=expected,
        ) as staged:
            actual = optimize_multi_view(
                "observations.npz",
                expected,
                alpha_model="bounded-complex",
                alpha_gain_min=0.5,
            )

        self.assertEqual(actual, expected)
        config = staged.call_args.kwargs["config"]
        self.assertEqual(config.alpha_model, "bounded-complex")
        self.assertEqual(config.alpha_gain_min, 0.5)


if __name__ == "__main__":
    unittest.main()
