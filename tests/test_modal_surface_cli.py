from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib
import io
import sys
import unittest
from unittest import mock


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
    }


class ModalSurfaceCliTests(unittest.TestCase):
    def test_parser_has_no_subcommands(self) -> None:
        cli = importlib.import_module("modal_surface.__main__")
        parser = cli.build_arg_parser()

        self.assertFalse(
            any(
                isinstance(action, argparse._SubParsersAction)
                for action in parser._actions
            )
        )

    def test_parser_uses_gaussian_arguments_and_current_defaults(self) -> None:
        cli = importlib.import_module("modal_surface.__main__")
        parser = cli.build_arg_parser()
        args = parser.parse_args(
            [
                "--input-ckpt",
                "checkpoint.ckpt",
                "--view-config",
                "view.json",
                "--modal-npz",
                "modal.npz",
                "--out-dir",
                "modes",
            ]
        )

        self.assertNotIn("command", vars(args))
        self.assertNotIn("_runner", vars(args))
        self.assertEqual(args.pixel_candidate_k, 4)
        self.assertEqual(args.alpha_model, "phase")
        self.assertFalse(args.motion_fill)
        self.assertEqual(args.motion_fill_k, 8)
        self.assertIsNone(args.motion_fill_max_distance)

    def test_observation_filter_controls_are_not_registered(self) -> None:
        cli = importlib.import_module("modal_surface.__main__")
        parser = cli.build_arg_parser()
        option_strings = _option_strings(parser)

        self.assertNotIn("--min-observations", option_strings)
        self.assertNotIn("--pair-weight", option_strings)

    def test_gaussian_motion_fill_controls_are_registered(self) -> None:
        cli = importlib.import_module("modal_surface.__main__")
        parser = cli.build_arg_parser()
        option_strings = _option_strings(parser)

        self.assertTrue(
            {
                "--motion-fill",
                "--motion-fill-k",
                "--motion-fill-max-distance",
            }.issubset(option_strings)
        )

    def test_gaussian_motion_fill_requires_explicit_valid_distance(self) -> None:
        app = importlib.import_module("modal_surface.apps.solve_gaussian_modes")
        base = {
            "motion_fill": True,
            "motion_fill_k": 8,
            "motion_fill_max_distance": 0.05,
        }

        app._validate_motion_fill_arguments(argparse.Namespace(**base), num_points=20)
        for changes, message in (
            ({"motion_fill_max_distance": None}, "requires"),
            ({"motion_fill_max_distance": 0.0}, "finite and positive"),
            ({"motion_fill_k": 20}, "smaller"),
        ):
            with self.subTest(changes=changes):
                values = {**base, **changes}
                with self.assertRaisesRegex(ValueError, message):
                    app._validate_motion_fill_arguments(
                        argparse.Namespace(**values), num_points=20
                    )

        with self.assertRaisesRegex(ValueError, "requires --motion-fill"):
            app._validate_motion_fill_arguments(
                argparse.Namespace(
                    motion_fill=False,
                    motion_fill_k=8,
                    motion_fill_max_distance=0.05,
                )
            )
        with self.assertRaisesRegex(ValueError, "requires --motion-fill"):
            app._validate_motion_fill_arguments(
                argparse.Namespace(
                    motion_fill=False,
                    motion_fill_k=4,
                    motion_fill_max_distance=None,
                )
            )

    def test_snr_weighting_controls_are_not_registered(self) -> None:
        cli = importlib.import_module("modal_surface.__main__")
        parser = cli.build_arg_parser()
        removed_options = {
            "--view-frequency-weighting",
            "--snr-band-hz",
            "--snr-exclude-hz",
            "--snr-good",
            "--view-weight-min",
        }

        self.assertTrue(removed_options.isdisjoint(_option_strings(parser)))

    def test_depth_weighting_controls_are_not_registered(self) -> None:
        cli = importlib.import_module("modal_surface.__main__")
        parser = cli.build_arg_parser()
        removed_options = {
            "--depth-weighting",
            "--depth-weight-power",
            "--depth-weight-min",
            "--depth-weight-reference-percentile",
        }

        self.assertTrue(removed_options.isdisjoint(_option_strings(parser)))

    def test_removed_gaussian_observation_controls_are_not_registered(self) -> None:
        cli = importlib.import_module("modal_surface.__main__")
        parser = cli.build_arg_parser()
        option_strings = _option_strings(parser)
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
            "--pixel-min-mode-amp-percentile",
            "--pixel-max-samples-per-view",
        }
        self.assertTrue(removed_options.isdisjoint(option_strings))
        retained_pixel_options = {
            "--pixel-sample-stride",
            "--pixel-candidate-k",
            "--pixel-preselect-k",
            "--pixel-render-acc-min",
            "--pixel-min-contribution",
        }
        self.assertTrue(retained_pixel_options.issubset(option_strings))

    def test_legacy_solver_controls_are_not_registered(self) -> None:
        cli = importlib.import_module("modal_surface.__main__")
        parser = cli.build_arg_parser()
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

        self.assertTrue(legacy_options.isdisjoint(_option_strings(parser)))

    def test_main_invokes_gaussian_runner(self) -> None:
        cli = importlib.import_module("modal_surface.__main__")
        app = importlib.import_module("modal_surface.apps.solve_gaussian_modes")
        argv = [
            "--input-ckpt",
            "checkpoint.ckpt",
            "--view-config",
            "view.json",
            "--modal-npz",
            "modal.npz",
            "--out-dir",
            "modes",
        ]

        with mock.patch.object(app, "run") as runner:
            cli.main(argv)

        runner.assert_called_once()
        args = runner.call_args.args[0]
        self.assertNotIn("command", vars(args))
        self.assertNotIn("_runner", vars(args))
        self.assertEqual(args.input_ckpt, "checkpoint.ckpt")
        self.assertEqual(args.out_dir, "modes")

    def test_parser_and_help_do_not_import_3dgs_runtime(self) -> None:
        blocked_modules = {
            "torch": None,
            "flow3d": None,
            "flow3d.scene_model": None,
        }
        with mock.patch.dict(sys.modules, blocked_modules):
            sys.modules.pop("modal_surface.__main__", None)
            sys.modules.pop("modal_surface.apps.solve_gaussian_modes", None)
            cli = importlib.import_module("modal_surface.__main__")
            parser = cli.build_arg_parser()

            stdout = io.StringIO()
            with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
                parser.parse_args(["--help"])

            self.assertEqual(raised.exception.code, 0)
            self.assertIn("--input-ckpt", stdout.getvalue())
            self.assertIsNone(sys.modules["torch"])
            self.assertIsNone(sys.modules["flow3d"])
            self.assertIsNone(sys.modules["flow3d.scene_model"])


if __name__ == "__main__":
    unittest.main()
