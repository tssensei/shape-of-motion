from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib
import io
import sys
import unittest
from unittest import mock


EXPECTED_COMMANDS = (
    "match-carrier-views",
    "optimize-multi-view",
    "solve-carrier-modes",
    "solve-gaussian-modes",
)


class ModalSurfaceCliTests(unittest.TestCase):
    def test_registry_contains_exactly_the_public_commands(self) -> None:
        cli = importlib.import_module("modal_surface.cli")
        parser = cli.build_arg_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )

        self.assertEqual(tuple(cli.COMMANDS), EXPECTED_COMMANDS)
        self.assertEqual(tuple(subparsers.choices), EXPECTED_COMMANDS)

    def test_each_command_uses_its_app_parser_and_current_defaults(self) -> None:
        cli = importlib.import_module("modal_surface.cli")
        parser = cli.build_arg_parser()
        cases = (
            (
                "match-carrier-views",
                [
                    "--carrier-points",
                    "carrier.npz",
                    "--view-config",
                    "view.json",
                    "--modal-npz",
                    "modal.npz",
                    "--out",
                    "observations.npz",
                    "--mode-index",
                    "3",
                ],
                "modal_surface.apps.match_carrier_views",
                {
                    "mode_index": 3,
                    "view_config": ["view.json"],
                },
            ),
            (
                "optimize-multi-view",
                [
                    "--observations",
                    "observations.npz",
                    "--out",
                    "latent.npz",
                    "--alpha-min-shared-points",
                    "24",
                ],
                "modal_surface.apps.optimize_multi_view",
                {
                    "alpha_model": "phase",
                    "alpha_min_shared_points": 24,
                },
            ),
            (
                "solve-carrier-modes",
                [
                    "--carrier-points",
                    "carrier.npz",
                    "--view-config",
                    "view.json",
                    "--modal-npz",
                    "modal.npz",
                    "--out-dir",
                    "modes",
                    "--mode-indices",
                    "1,3",
                ],
                "modal_surface.apps.solve_carrier_modes",
                {
                    "mode_indices": "1,3",
                    "alpha_model": "phase",
                },
            ),
            (
                "solve-gaussian-modes",
                [
                    "--input-ckpt",
                    "checkpoint.ckpt",
                    "--view-config",
                    "view.json",
                    "--modal-npz",
                    "modal.npz",
                    "--out-dir",
                    "modes",
                ],
                "modal_surface.apps.solve_gaussian_modes",
                {
                    "pixel_candidate_k": 4,
                    "alpha_model": "phase",
                },
            ),
        )

        for command, command_args, module_name, expected in cases:
            with self.subTest(command=command):
                args = parser.parse_args([command, *command_args])
                app = importlib.import_module(module_name)

                self.assertEqual(args.command, command)
                self.assertIs(args._runner, app.run)
                for name, value in expected.items():
                    self.assertEqual(getattr(args, name), value)

    def test_observation_filter_controls_are_not_registered(self) -> None:
        cli = importlib.import_module("modal_surface.cli")
        parser = cli.build_arg_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )

        for command in (
            "match-carrier-views",
            "solve-carrier-modes",
            "solve-gaussian-modes",
        ):
            with self.subTest(command=command):
                option_strings = {
                    option
                    for action in subparsers.choices[command]._actions
                    for option in action.option_strings
                }
                self.assertNotIn("--min-observations", option_strings)
                self.assertNotIn("--pair-weight", option_strings)

    def test_removed_gaussian_observation_controls_are_not_registered(self) -> None:
        cli = importlib.import_module("modal_surface.cli")
        parser = cli.build_arg_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        gaussian_parser = subparsers.choices["solve-gaussian-modes"]
        option_strings = {
            option
            for action in gaussian_parser._actions
            for option in action.option_strings
        }
        removed_options = {
            "--observation-sampling",
            "--zbuffer-radius",
            "--zbuffer-mode",
            "--zbuffer-soft-sigma",
            "--zbuffer-soft-min-weight",
            "--front-percentile",
            "--zbuffer-tau",
            "--min-zbuffer-samples",
            "--gaussian-contribution-radius",
            "--gaussian-contribution-min-share",
            "--gaussian-contribution-min-score",
            "--gaussian-contribution-cov-eps-px",
        }
        self.assertTrue(removed_options.isdisjoint(option_strings))
        retained_pixel_options = {
            "--pixel-sample-stride",
            "--pixel-candidate-k",
            "--pixel-preselect-k",
            "--pixel-render-acc-min",
            "--pixel-min-contribution",
            "--pixel-min-mode-amp-percentile",
            "--pixel-max-samples-per-view",
        }
        self.assertTrue(retained_pixel_options.issubset(option_strings))

    def test_legacy_solver_controls_are_not_registered(self) -> None:
        cli = importlib.import_module("modal_surface.cli")
        parser = cli.build_arg_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
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

        for command in ("optimize-multi-view", "solve-carrier-modes", "solve-gaussian-modes"):
            with self.subTest(command=command):
                option_strings = {
                    option
                    for action in subparsers.choices[command]._actions
                    for option in action.option_strings
                }
                self.assertTrue(legacy_options.isdisjoint(option_strings))

    def test_main_dispatches_to_the_attached_app_runner(self) -> None:
        cli = importlib.import_module("modal_surface.cli")
        app = importlib.import_module("modal_surface.apps.optimize_multi_view")
        argv = [
            "optimize-multi-view",
            "--observations",
            "observations.npz",
            "--out",
            "latent.npz",
        ]

        with mock.patch.object(app, "run") as runner:
            cli.main(argv)

        runner.assert_called_once()
        args = runner.call_args.args[0]
        self.assertEqual(args.command, "optimize-multi-view")
        self.assertEqual(args.observations, "observations.npz")
        self.assertEqual(args.out, "latent.npz")

    def test_root_wrapper_reexports_the_package_cli(self) -> None:
        cli = importlib.import_module("modal_surface.cli")
        root_cli = importlib.import_module("run_modal_surface")

        self.assertIs(root_cli.build_arg_parser, cli.build_arg_parser)
        self.assertIs(root_cli.main, cli.main)

    def test_parser_and_non_gaussian_help_do_not_import_3dgs_runtime(self) -> None:
        blocked_modules = {
            "torch": None,
            "flow3d": None,
            "flow3d.scene_model": None,
        }
        with mock.patch.dict(sys.modules, blocked_modules):
            sys.modules.pop("modal_surface.cli", None)
            sys.modules.pop("modal_surface.apps.solve_gaussian_modes", None)
            cli = importlib.import_module("modal_surface.cli")
            parser = cli.build_arg_parser()

            stdout = io.StringIO()
            with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
                parser.parse_args(["optimize-multi-view", "--help"])

            self.assertEqual(raised.exception.code, 0)
            self.assertIn("--observations", stdout.getvalue())
            self.assertIsNone(sys.modules["torch"])
            self.assertIsNone(sys.modules["flow3d"])
            self.assertIsNone(sys.modules["flow3d.scene_model"])


if __name__ == "__main__":
    unittest.main()
