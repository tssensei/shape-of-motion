from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from modal_surface.motion_fill import (
    build_knn_graph,
    compute_anchor_connectivity,
    fill_nullspace_motion,
    query_knn_candidates,
    validate_motion_fill_inputs,
)
from preproc.toy_modal_motion_fill import (
    _preflight_track,
    _validate_k_values,
    build_arg_parser,
    run_experiment,
)


def _edge_set(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(edge_index)
    }


def _constant_field_case(
    target: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    target = np.asarray(target, dtype=np.complex128)
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=np.float64,
    )
    phi_observable = np.zeros((3, 3), dtype=np.complex128)
    phi_observable[0] = target
    phi_observable[1, :2] = target[:2]

    basis = np.zeros((3, 3, 3), dtype=np.complex128)
    basis[1, 2, 0] = 1.0
    basis[2] = np.eye(3, dtype=np.complex128)
    nullity = np.asarray([0, 1, 3], dtype=np.int8)
    anchor_mask = np.asarray([True, False, False])
    partial_mask = np.asarray([False, True, False])
    unobserved_mask = np.asarray([False, False, True])
    return (
        points,
        phi_observable,
        basis,
        nullity,
        anchor_mask,
        partial_mask,
        unobserved_mask,
    )


def _chain_graph(points: np.ndarray):
    candidates = query_knn_candidates(points, max_k=1)
    return build_knn_graph(candidates, k=1, max_distance=0.11)


def _toy_track_inputs(
    target: np.ndarray,
    mode_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    target = np.asarray(target, dtype=np.complex64)
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.1, 0.0, 0.0], [0.15, 0.0, 0.0]],
        dtype=np.float32,
    )
    colors = np.asarray(
        [[240, 40, 40], [40, 240, 40], [40, 40, 240], [180, 180, 180]],
        dtype=np.uint8,
    )
    view_ids = np.asarray(["view1", "view2", "view3"])
    true_alphas = np.exp(1j * np.asarray([0.0, 0.4, 0.8])).astype(np.complex64)
    point_count = np.asarray([3, 2, 1, 0], dtype=np.int32)
    obs_point_index = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int32)
    obs_view_index = np.asarray([0, 1, 2, 0, 1, 0], dtype=np.int32)
    jacobian = np.broadcast_to(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (obs_point_index.shape[0], 2, 3),
    ).copy()
    phi_gt = np.broadcast_to(target, (points.shape[0], 3)).copy()
    obs_y = (
        true_alphas[obs_view_index, None]
        * np.einsum("nij,nj->ni", jacobian, phi_gt[obs_point_index])
    ).astype(np.complex64)

    anchor_mask = np.asarray([True, True, False, False])
    partial_mask = np.asarray([False, False, True, False])
    unobserved_mask = np.asarray([False, False, False, True])
    phi_observable = np.zeros_like(phi_gt)
    phi_observable[anchor_mask] = phi_gt[anchor_mask]
    phi_observable[partial_mask, :2] = phi_gt[partial_mask, :2]
    basis = np.zeros((points.shape[0], 3, 3), dtype=np.complex64)
    basis[2, 2, 0] = 1.0
    basis[3] = np.eye(3, dtype=np.complex64)
    false_mask = np.zeros((points.shape[0],), dtype=bool)

    observations = {
        "points_world": points,
        "colors": colors,
        "view_ids": view_ids,
        "true_alphas": true_alphas,
        "phi_gt": phi_gt,
        "obs_count_per_point": point_count,
        "obs_sample_count_per_point": point_count.copy(),
        "obs_point_index": obs_point_index.copy(),
        "obs_view_index": obs_view_index.copy(),
        "obs_y": obs_y.copy(),
        "obs_J": jacobian.copy(),
    }
    solved = {
        "points_world": points.copy(),
        "view_ids": view_ids.copy(),
        "alphas": true_alphas.copy(),
        "alpha_identifiable_mask": np.ones((3,), dtype=bool),
        "phi": phi_observable.copy(),
        "phi_observable": phi_observable,
        "phi_nullspace_correction": np.zeros_like(phi_observable),
        "point_nullspace_basis": basis,
        "point_nullity": np.asarray([0, 0, 1, 3], dtype=np.int8),
        "point_observable_rank": np.asarray([3, 3, 2, 0], dtype=np.int8),
        "anchor_mask": anchor_mask,
        "partial_mask": partial_mask,
        "unobserved_mask": unobserved_mask,
        "rejected_mask": false_mask.copy(),
        "alpha_unresolved_mask": false_mask.copy(),
        "no_usable_observation_mask": false_mask.copy(),
        "completion_mask": false_mask.copy(),
        "solver_method": np.array("staged_overlap_observable"),
        "solver_version": np.array(2, dtype=np.int32),
        "obs_count_per_point": point_count.copy(),
        "obs_sample_count_per_point": point_count.copy(),
        "point_distinct_view_count": point_count.copy(),
        "point_distinct_valid_view_count": point_count.copy(),
        "point_usable_observation_row_count": point_count.copy(),
        "obs_point_index": obs_point_index,
        "obs_view_index": obs_view_index,
        "obs_y": obs_y,
        "obs_J": jacobian,
        "obs_effective_weight": np.ones((obs_point_index.shape[0],), dtype=np.float32),
        "point_solution_status": np.asarray([0, 0, 3, 4], dtype=np.int8),
        "freq_hz": np.array(1.0, dtype=np.float32),
        "mode_index": np.array(mode_index, dtype=np.int32),
    }
    ground_truth = {
        "points_world": points.copy(),
        "phi": phi_gt.copy(),
    }
    return observations, solved, ground_truth


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


class MotionFillCoreTests(unittest.TestCase):
    def test_constant_real_field_recovers_partial_and_unobserved_points(self) -> None:
        target = np.asarray([1.0, -2.0, 3.0], dtype=np.complex128)
        (
            points,
            phi_observable,
            basis,
            nullity,
            anchor_mask,
            partial_mask,
            unobserved_mask,
        ) = _constant_field_case(target)

        result = fill_nullspace_motion(
            _chain_graph(points),
            phi_observable,
            basis,
            nullity,
            anchor_mask,
            partial_mask,
            unobserved_mask,
        )

        np.testing.assert_allclose(
            result.phi,
            np.broadcast_to(target, result.phi.shape),
            atol=1e-9,
        )
        np.testing.assert_array_equal(result.phi[anchor_mask], phi_observable[anchor_mask])
        np.testing.assert_array_equal(
            result.phi_nullspace_correction[anchor_mask],
            np.zeros((1, 3), dtype=result.phi_nullspace_correction.dtype),
        )
        self.assertTrue(np.all(result.completion_mask[partial_mask | unobserved_mask]))

    def test_constant_complex_ellipse_uses_both_real_and_imaginary_solves(self) -> None:
        target = np.asarray(
            [1.0 + 0.0j, 0.25 - 0.35j, -0.4 - 0.2j],
            dtype=np.complex128,
        )
        case = _constant_field_case(target)
        result = fill_nullspace_motion(
            _chain_graph(case[0]),
            *case[1:],
        )

        np.testing.assert_allclose(
            result.phi,
            np.broadcast_to(target, result.phi.shape),
            atol=1e-9,
        )
        self.assertGreater(result.real_solver.iterations, 0)
        self.assertGreater(result.imag_solver.iterations, 0)
        self.assertEqual(result.system_column_count, 4)

    def test_partial_correction_stays_in_nullspace_and_preserves_observation(self) -> None:
        target = np.asarray([1.0, 2.0, 3.0], dtype=np.complex128)
        case = _constant_field_case(target)
        result = fill_nullspace_motion(_chain_graph(case[0]), *case[1:])
        observation_operator = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )

        np.testing.assert_allclose(
            observation_operator @ result.phi[1],
            observation_operator @ case[1][1],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.phi_nullspace_correction[1],
            case[2][1, :, :1] @ result.coefficient_values[
                result.coefficient_offsets[1] : result.coefficient_offsets[2]
            ],
            atol=1e-12,
        )

    def test_anchor_free_component_remains_unresolved(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0], [10.1, 0.0, 0.0]],
            dtype=np.float64,
        )
        phi_observable = np.zeros((4, 3), dtype=np.complex128)
        phi_observable[0] = np.asarray([1.0, 2.0, 3.0])
        basis = np.zeros((4, 3, 3), dtype=np.complex128)
        basis[1:] = np.eye(3, dtype=np.complex128)
        nullity = np.asarray([0, 3, 3, 3], dtype=np.int8)
        anchor_mask = np.asarray([True, False, False, False])
        partial_mask = np.zeros((4,), dtype=bool)
        unobserved_mask = ~anchor_mask
        graph = build_knn_graph(
            query_knn_candidates(points, max_k=1),
            k=1,
            max_distance=0.11,
        )

        result = fill_nullspace_motion(
            graph,
            phi_observable,
            basis,
            nullity,
            anchor_mask,
            partial_mask,
            unobserved_mask,
        )

        np.testing.assert_allclose(result.phi[1], phi_observable[0], atol=1e-9)
        np.testing.assert_array_equal(result.phi[2:], phi_observable[2:])
        np.testing.assert_array_equal(result.completion_mask[2:], [False, False])
        np.testing.assert_array_equal(
            result.completion_connected_to_anchor,
            [True, True, False, False],
        )

    def test_variable_nullity_uses_all_active_basis_columns(self) -> None:
        target = np.asarray([1.0, 2.0 - 0.5j, 3.0 + 0.25j], dtype=np.complex128)
        points = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)
        phi_observable = np.asarray([target, [target[0], 0.0, 0.0]], dtype=np.complex128)
        basis = np.zeros((2, 3, 3), dtype=np.complex128)
        basis[1, 1, 0] = 1.0
        basis[1, 2, 1] = 1.0
        nullity = np.asarray([0, 2], dtype=np.int8)
        anchor_mask = np.asarray([True, False])
        partial_mask = np.asarray([False, True])
        unobserved_mask = np.asarray([False, False])

        result = fill_nullspace_motion(
            _chain_graph(points),
            phi_observable,
            basis,
            nullity,
            anchor_mask,
            partial_mask,
            unobserved_mask,
        )

        np.testing.assert_allclose(result.phi[1], target, atol=1e-9)
        self.assertEqual(result.coefficient_offsets.tolist(), [0, 0, 2])

    def test_partial_point_with_zero_nullity_fails_instead_of_becoming_fixed(self) -> None:
        case = list(_constant_field_case(np.asarray([1.0, 2.0, 3.0])))
        case[3][1] = 0

        with self.assertRaisesRegex(ValueError, "partial"):
            validate_motion_fill_inputs(*case[1:])

    def test_wide_out_of_range_nullity_fails_before_narrowing(self) -> None:
        case = list(_constant_field_case(np.asarray([1.0, 2.0, 3.0])))
        case[3] = case[3].astype(np.int16)
        case[3][1] = 257

        with self.assertRaisesRegex(ValueError, r"\[0,3\]"):
            validate_motion_fill_inputs(*case[1:])

    def test_validation_rejects_malformed_nonreal_and_inconsistent_bases(self) -> None:
        case = _constant_field_case(np.asarray([1.0, 2.0, 3.0]))
        malformed = case[2][:, :, :2]
        nonreal = case[2].copy()
        nonreal[1, 2, 0] = 1.0 + 1e-3j
        nonorthonormal = case[2].copy()
        nonorthonormal[1, 2, 0] = 2.0
        nonorthogonal_observable = case[1].copy()
        nonorthogonal_observable[1, 2] = 1.0

        invalid_cases = (
            ("shape", nonorthogonal_observable, malformed),
            ("real", case[1], nonreal),
            ("orthonormal", case[1], nonorthonormal),
            ("orthogonal", nonorthogonal_observable, case[2]),
        )
        for label, phi, basis in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    validate_motion_fill_inputs(
                        phi,
                        basis,
                        case[3],
                        case[4],
                        case[5],
                        case[6],
                    )

    def test_knn_graph_uses_k_prefix_then_threshold_and_deduplicates_edges(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0], [0.7, 0.0, 0.0]],
            dtype=np.float64,
        )
        candidates = query_knn_candidates(points, max_k=3)
        graph_k1 = build_knn_graph(candidates, k=1, max_distance=1.0)
        graph_k2 = build_knn_graph(candidates, k=2, max_distance=1.0)
        thresholded = build_knn_graph(candidates, k=2, max_distance=0.15)

        self.assertEqual(graph_k1.retained_directed_count, 4)
        self.assertEqual(graph_k2.retained_directed_count, 8)
        self.assertTrue(_edge_set(graph_k1.edge_index) <= _edge_set(graph_k2.edge_index))
        self.assertEqual(_edge_set(thresholded.edge_index), {(0, 1)})
        self.assertGreater(thresholded.pruned_directed_count, 0)
        self.assertEqual(
            len(_edge_set(graph_k2.edge_index)),
            graph_k2.edge_index.shape[0],
        )
        np.testing.assert_allclose(
            graph_k2.edge_weight,
            1.0 / (graph_k2.edge_distance + graph_k2.epsilon),
            atol=0.0,
        )
        for first, second in graph_k2.edge_index:
            self.assertLess(int(first), int(second))

    def test_knn_candidates_reject_duplicate_points(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )

        with self.assertRaisesRegex(ValueError, "duplicate|zero-distance"):
            query_knn_candidates(points, max_k=1)

    def test_same_graph_has_track_specific_anchor_connectivity(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0], [10.1, 0.0, 0.0]],
            dtype=np.float64,
        )
        graph = build_knn_graph(
            query_knn_candidates(points, max_k=1),
            k=1,
            max_distance=0.11,
        )

        linear = compute_anchor_connectivity(
            graph, np.asarray([True, False, False, False])
        )
        ellipse = compute_anchor_connectivity(
            graph, np.asarray([True, False, True, False])
        )

        np.testing.assert_array_equal(linear.connected_to_anchor, [True, True, False, False])
        np.testing.assert_array_equal(ellipse.connected_to_anchor, np.ones(4, dtype=bool))
        self.assertEqual(int(np.count_nonzero(~linear.component_has_anchor)), 1)
        self.assertEqual(int(np.count_nonzero(~ellipse.component_has_anchor)), 0)


class ToyMotionFillRunnerTests(unittest.TestCase):
    def test_cli_defaults_and_k_validation(self) -> None:
        args = build_arg_parser().parse_args(["--toy-dir", "toy"])

        self.assertEqual(args.toy_dir, Path("toy"))
        self.assertEqual(args.k_values, [4, 8, 16])
        self.assertEqual(args.max_distance, 0.1)
        self.assertEqual(args.epsilon, 1e-8)
        self.assertEqual(_validate_k_values([4, 8, 16], 20), (4, 8, 16))
        for invalid in ([], [0], [4, 4], [20]):
            with self.subTest(k_values=invalid):
                with self.assertRaises(ValueError):
                    _validate_k_values(invalid, 20)

    def test_preflight_accepts_clean_three_view_staged_track(self) -> None:
        observations, solved, ground_truth = _toy_track_inputs(
            np.asarray([1.0, 0.25, -0.35j], dtype=np.complex64),
            mode_index=1,
        )

        summary = _preflight_track(
            "Tilted ellipse", observations, solved, ground_truth
        )

        self.assertEqual(summary["num_points"], 4)
        self.assertEqual(summary["num_views"], 3)
        self.assertEqual(summary["num_observations"], 6)
        self.assertEqual(
            summary["nullspace_operator_validation"]["max_relative_error"], 0.0
        )
        self.assertEqual(summary["classes"]["anchor"]["point_count"], 2)
        self.assertEqual(
            summary["classes"]["partial"]["nullity_distribution"], {"1": 1}
        )
        self.assertEqual(
            summary["classes"]["unobserved"]["valid_view_count_distribution"],
            {"0": 1},
        )

    def test_preflight_rejects_non_three_view_and_invalid_class_inputs(self) -> None:
        observations, solved, ground_truth = _toy_track_inputs(
            np.asarray([1.0, 2.0, 3.0], dtype=np.complex64),
            mode_index=0,
        )
        two_view_observations = dict(observations)
        two_view_observations["view_ids"] = observations["view_ids"][:2]
        with self.assertRaisesRegex(ValueError, "three-view"):
            _preflight_track(
                "Linear", two_view_observations, solved, ground_truth
            )

        overlapping = dict(solved)
        overlapping["partial_mask"] = solved["partial_mask"].copy()
        overlapping["partial_mask"][0] = True
        with self.assertRaisesRegex(ValueError, "mutually exclusive and exhaustive"):
            _preflight_track("Linear", observations, overlapping, ground_truth)

        failed = dict(solved)
        failed["rejected_mask"] = solved["rejected_mask"].copy()
        failed["rejected_mask"][2] = True
        with self.assertRaisesRegex(ValueError, "unexpected rejected_mask"):
            _preflight_track("Linear", observations, failed, ground_truth)

        mismatched_count = dict(solved)
        mismatched_count["point_distinct_valid_view_count"] = solved[
            "point_distinct_valid_view_count"
        ].copy()
        mismatched_count["point_distinct_valid_view_count"][2] = 2
        with self.assertRaisesRegex(ValueError, "counts to agree"):
            _preflight_track("Linear", observations, mismatched_count, ground_truth)

        inconsistent_rank = dict(solved)
        inconsistent_rank["point_observable_rank"] = solved[
            "point_observable_rank"
        ].copy()
        inconsistent_rank["point_observable_rank"][2] = 1
        with self.assertRaisesRegex(ValueError, "rank.*nullity"):
            _preflight_track("Linear", observations, inconsistent_rank, ground_truth)

        invalid_unobserved_basis = dict(solved)
        invalid_unobserved_basis["point_nullspace_basis"] = solved[
            "point_nullspace_basis"
        ].copy()
        invalid_unobserved_basis["point_nullspace_basis"][3] = np.eye(
            3, dtype=np.complex64
        )[:, [1, 0, 2]]
        with self.assertRaisesRegex(ValueError, "identity nullspace basis"):
            _preflight_track(
                "Linear", observations, invalid_unobserved_basis, ground_truth
            )

        invalid_operator_nullspace = dict(solved)
        invalid_operator_nullspace["point_nullspace_basis"] = solved[
            "point_nullspace_basis"
        ].copy()
        invalid_operator_nullspace["point_nullspace_basis"][2, :, 0] = np.asarray(
            [-2.0, 1.0, 0.0], dtype=np.float32
        ) / np.sqrt(5.0)
        with self.assertRaisesRegex(ValueError, "A_i N_i"):
            _preflight_track(
                "Linear", observations, invalid_operator_nullspace, ground_truth
            )

    def test_run_experiment_writes_k_sweep_artifacts_and_manifests(self) -> None:
        linear = _toy_track_inputs(
            np.asarray([1.0, -2.0, 3.0], dtype=np.complex64),
            mode_index=0,
        )
        ellipse = _toy_track_inputs(
            np.asarray([1.0, 0.25, -0.35j], dtype=np.complex64),
            mode_index=1,
        )

        with tempfile.TemporaryDirectory() as tmp:
            toy_dir = Path(tmp)
            _write_npz(
                toy_dir / "observations" / "toy_sphere_observations.npz",
                linear[0],
            )
            _write_npz(
                toy_dir
                / "observations"
                / "toy_sphere_tilted_ellipse_observations.npz",
                ellipse[0],
            )
            _write_npz(toy_dir / "latents" / "solved_staged.npz", linear[1])
            _write_npz(
                toy_dir / "latents" / "solved_staged_tilted_ellipse.npz",
                ellipse[1],
            )
            _write_npz(toy_dir / "latents" / "gt_motion.npz", linear[2])
            _write_npz(
                toy_dir / "latents" / "gt_tilted_ellipse_motion.npz",
                ellipse[2],
            )

            manifest_paths = run_experiment(
                toy_dir,
                k_values=[1, 2],
                max_distance=0.11,
                epsilon=1e-7,
            )

            self.assertEqual(
                manifest_paths,
                [
                    toy_dir
                    / "manifests"
                    / "motion_fill"
                    / "k1"
                    / "modal_modes_manifest.json",
                    toy_dir
                    / "manifests"
                    / "motion_fill"
                    / "k2"
                    / "modal_modes_manifest.json",
                ],
            )
            for k, manifest_path in zip((1, 2), manifest_paths):
                with self.subTest(k=k):
                    output_dir = toy_dir / "motion_fill" / f"k{k}"
                    expected_outputs = {
                        "graph.npz",
                        "linear_filled.npz",
                        "tilted_ellipse_filled.npz",
                        "linear_overlay.npz",
                        "tilted_ellipse_overlay.npz",
                        "diagnostics.json",
                    }
                    self.assertEqual(
                        {path.name for path in output_dir.iterdir()}, expected_outputs
                    )

                    with np.load(output_dir / "graph.npz", allow_pickle=False) as graph:
                        self.assertEqual(int(graph["k"]), k)
                        self.assertEqual(float(graph["max_distance"]), 0.11)
                        self.assertEqual(float(graph["epsilon"]), 1e-7)
                        self.assertEqual(
                            graph["candidate_neighbor_indices"].shape, (4, k)
                        )
                        self.assertEqual(
                            int(graph["candidate_directed_count"]), 4 * k
                        )
                        self.assertEqual(
                            int(graph["unique_undirected_edge_count"]),
                            graph["edge_index"].shape[0],
                        )

                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    self.assertEqual(manifest["version"], 1)
                    self.assertEqual(manifest["parameters"]["knn_k"], k)
                    self.assertEqual(
                        manifest["parameters"]["motion_fill_method"],
                        "joint_knn_nullspace_lsmr",
                    )
                    self.assertEqual(manifest["parameters"]["lsmr_atol"], 1e-10)
                    self.assertEqual(manifest["parameters"]["lsmr_btol"], 1e-10)
                    self.assertEqual(manifest["parameters"]["lsmr_conlim"], 1e8)
                    self.assertEqual(
                        [mode["track_label"] for mode in manifest["modes"]],
                        ["Linear", "Tilted ellipse"],
                    )
                    for mode in manifest["modes"]:
                        latent_path = manifest_path.parent / mode["latent_path"]
                        self.assertTrue(latent_path.is_file())

                    diagnostics = json.loads(
                        (output_dir / "diagnostics.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(diagnostics["graph"]["k"], k)
                    self.assertEqual(
                        set(diagnostics["tracks"]), {"linear", "tilted_ellipse"}
                    )
                    self.assertEqual(
                        diagnostics["tracks"]["linear"]["anchor_drift"]["max"],
                        0.0,
                    )
                    self.assertLess(
                        diagnostics["tracks"]["linear"][
                            "partial_observation_drift_absolute"
                        ]["max"],
                        1e-8,
                    )
                    self.assertLess(
                        diagnostics["tracks"]["linear"]["classes"]["partial"][
                            "trajectory_rmse_filled"
                        ]["max"],
                        1e-5,
                    )
                    self.assertEqual(
                        diagnostics["tracks"]["linear"]["alpha_error"][
                            "complex_absolute"
                        ]["max"],
                        0.0,
                    )
                    self.assertEqual(
                        diagnostics["tracks"]["linear"]["connectivity"][
                            "anchor_free_hop_distance_sentinel"
                        ],
                        -1,
                    )

                    for track_key, source in (
                        ("linear", linear),
                        ("tilted_ellipse", ellipse),
                    ):
                        with np.load(
                            output_dir / f"{track_key}_filled.npz",
                            allow_pickle=False,
                        ) as filled:
                            required_keys = {
                                "phi",
                                "phi_observable",
                                "phi_nullspace_correction",
                                "completion_mask",
                                "completion_connected_to_anchor",
                                "point_solution_status",
                                "point_connected_component_index",
                                "point_anchor_hop_distance",
                                "coefficient_values",
                                "coefficient_offsets",
                                "graph_k",
                                "graph_max_distance",
                                "graph_epsilon",
                                "motion_fill_method",
                                "motion_fill_lsmr_atol",
                                "motion_fill_lsmr_btol",
                                "motion_fill_lsmr_conlim",
                                "source_solver_method",
                                "source_solver_version",
                                "source_staged_latent",
                                "source_observations",
                            }
                            self.assertTrue(required_keys <= set(filled.files))
                            np.testing.assert_allclose(
                                filled["phi"], source[2]["phi"], atol=1e-5
                            )
                            np.testing.assert_array_equal(
                                filled["phi"][:2], source[1]["phi"][:2]
                            )
                            np.testing.assert_array_equal(
                                filled["completion_mask"],
                                [False, False, True, True],
                            )
                            np.testing.assert_array_equal(
                                filled["point_solution_status"],
                                [0, 0, 1, 2],
                            )
                            self.assertEqual(int(filled["graph_k"]), k)
                            self.assertEqual(
                                str(filled["source_solver_method"]),
                                "staged_overlap_observable",
                            )
                            self.assertEqual(int(filled["source_solver_version"]), 2)

                        with np.load(
                            output_dir / f"{track_key}_overlay.npz",
                            allow_pickle=False,
                        ) as overlay:
                            self.assertEqual(overlay["points_world"].shape, (12, 3))
                            self.assertEqual(overlay["phi"].shape, (12, 3))
                            np.testing.assert_array_equal(
                                overlay["point_group"],
                                np.repeat(np.arange(3, dtype=np.int32), 4),
                            )
                            np.testing.assert_array_equal(
                                overlay["point_group_names"],
                                ["GT", "Observable", f"Filled K={k}"],
                            )

            summary = json.loads(
                (toy_dir / "motion_fill" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["k_values"], [1, 2])
            self.assertEqual(summary["primary_k"], 8)
            self.assertFalse(summary["primary_k_in_sweep"])
            self.assertEqual(set(summary["tracks"]), {"linear", "tilted_ellipse"})
            self.assertEqual(
                set(summary["tracks"]["linear"]["by_k"]), {"1", "2"}
            )
            linear_pair = summary["tracks"]["linear"][
                "pairwise_filled_field_difference"
            ]["k1_vs_k2"]
            self.assertLess(
                linear_pair["partial"]["complex_mode_relative_difference"]["max"],
                1e-5,
            )
            self.assertLess(
                linear_pair["unobserved"]["trajectory_rmse_difference"]["max"],
                1e-5,
            )

            with np.load(
                toy_dir / "latents" / "solved_staged.npz", allow_pickle=False
            ) as baseline:
                np.testing.assert_array_equal(baseline["phi"], linear[1]["phi"])


if __name__ == "__main__":
    unittest.main()
