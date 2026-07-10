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
                    "min_observations": 1,
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
                    "solver": "staged",
                    "alpha_model": "phase",
                    "alpha_min_shared_points": 24,
                    "outlier_frac": None,
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
                    "solver": "staged",
                    "outlier_frac": 0.0,
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
                    "--observation-sampling",
                    "gaussian-center-contribution",
                ],
                "modal_surface.apps.solve_gaussian_modes",
                {
                    "observation_sampling": "gaussian-center-contribution",
                    "solver": "staged",
                    "modal_fill_unobserved": False,
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
