from __future__ import annotations

import unittest

import numpy as np

from preproc.synthetic_modal_visibility_sphere import _group_diagnostics, _phase_stats


def _phase_distance(a: float, b: float) -> float:
    return abs(float(np.angle(np.exp(1j * (a - b)))))


class SyntheticModalVisibilitySphereTests(unittest.TestCase):
    def test_phase_stats_handles_branch_cut_cluster(self) -> None:
        stats = _phase_stats(
            np.asarray([np.pi - 0.1, -np.pi + 0.1], dtype=np.float64)
        )

        self.assertEqual(set(stats), {"count", "mean", "median", "p90", "max"})
        self.assertEqual(stats["count"], 2)
        expected = {
            "mean": np.pi,
            "median": np.pi,
            "p90": np.pi + 0.08,
            "max": np.pi + 0.1,
        }
        for key, expected_value in expected.items():
            with self.subTest(statistic=key):
                value = stats[key]
                self.assertIsNotNone(value)
                self.assertLess(_phase_distance(float(value), expected_value), 1e-12)

    def test_group_diagnostics_uses_circular_phase_stats(self) -> None:
        phases = np.asarray([np.pi - 0.05, -np.pi + 0.05], dtype=np.float64)
        phi = np.zeros((2, 3), dtype=np.complex128)
        phi[:, 0] = np.exp(1j * phases)

        diagnostics = _group_diagnostics(
            "branch_cut",
            np.ones((2,), dtype=bool),
            phi,
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            None,
        )

        phase_stats = diagnostics["phase_rad"]
        self.assertEqual(phase_stats["count"], 2)
        self.assertIsNotNone(phase_stats["mean"])
        self.assertLess(
            _phase_distance(float(phase_stats["mean"]), np.pi),
            1e-6,
        )

    def test_phase_stats_handles_empty_and_undefined_samples(self) -> None:
        expected_empty = {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
        self.assertEqual(
            _phase_stats(np.asarray([np.nan, np.inf, -np.inf], dtype=np.float64)),
            expected_empty,
        )

        antipodal = _phase_stats(np.asarray([0.0, np.pi], dtype=np.float64))
        self.assertEqual(antipodal["count"], 2)
        for key in ("mean", "median", "p90", "max"):
            self.assertIsNone(antipodal[key])


if __name__ == "__main__":
    unittest.main()
