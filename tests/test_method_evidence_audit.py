from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments import audit_method_evidence as audit


class MethodEvidenceAuditHelpersTest(unittest.TestCase):
    def test_average_ranks_and_spearman_handle_ties(self) -> None:
        values = np.asarray([3.0, 1.0, 1.0, 2.0])
        self.assertTrue(
            np.array_equal(audit.average_ranks(values), np.asarray([4.0, 1.5, 1.5, 3.0]))
        )
        self.assertAlmostEqual(audit.spearman_correlation(values, values), 1.0)
        self.assertAlmostEqual(audit.spearman_correlation(values, -values), -1.0)

    def test_top_choice_uses_available_non_prior_experts_only(self) -> None:
        attention = np.asarray(
            [
                [0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        availability = np.asarray(
            [
                [True, False, True, False, False, False, False, True],
                [False, False, False, False, False, False, False, True],
            ],
            dtype=bool,
        )
        choice, eligible = audit.top_non_prior_choice(attention, availability)
        self.assertTrue(np.array_equal(choice, np.asarray([2, 0])))
        self.assertTrue(np.array_equal(eligible, np.asarray([True, False])))

    def test_single_random_repeat_has_defined_zero_standard_deviation(self) -> None:
        self.assertEqual(audit.sample_standard_deviation([0.2]), 0.0)
        self.assertAlmostEqual(audit.sample_standard_deviation([0.1, 0.3]), 0.1414213562373095)
        with self.assertRaisesRegex(ValueError, "at least one"):
            audit.sample_standard_deviation([])

    def test_expected_random_bootstrap_uses_mean_of_random_errors(self) -> None:
        actual = np.full((2, 24, 4), 10.0, dtype=np.float32)
        original = np.full_like(actual, 11.0)
        random_predictions = np.stack(
            (np.full_like(actual, 12.0), np.full_like(actual, 13.0)), axis=0
        )
        result = audit.cell_cluster_bootstrap_expected_random_delta(
            actual,
            random_predictions,
            original,
            np.asarray(["cell-a", "cell-b"]),
            replicates=100,
            seed=42,
        )
        self.assertAlmostEqual(result["point_original_macro_wape"], 0.1)
        self.assertAlmostEqual(result["point_expected_random_macro_wape"], 0.25)
        self.assertAlmostEqual(
            result["delta_expected_random_minus_original"], 0.15
        )
        self.assertAlmostEqual(result["ci_low"], 0.15)
        self.assertAlmostEqual(result["ci_high"], 0.15)

    def test_reproduction_paths_and_defaults_are_confined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = (root / "artifacts/reproduction/analysis/audit").resolve(strict=False)
            self.assertEqual(
                audit.resolve_reproduction_artifact_path(
                    root, "artifacts/reproduction/analysis/audit", strict=False
                ),
                expected,
            )
            with self.assertRaisesRegex(ValueError, "artifacts/reproduction"):
                audit.resolve_reproduction_artifact_path(
                    root, "artifacts/other/audit", strict=False
                )
        parsed = audit.parser().parse_args([])
        self.assertEqual(parsed.source, str(audit.DEFAULT_SOURCE))
        self.assertEqual(parsed.output, str(audit.DEFAULT_OUTPUT))
        self.assertEqual(parsed.locality_sample_size, 256)

    def test_gpu_device_parser_rejects_ambiguous_or_invalid_lists(self) -> None:
        self.assertEqual(audit.parse_gpu_devices("0,2,3"), [0, 2, 3])
        for value in ("", "0,0", "-1", "gpu0"):
            with self.assertRaises(ValueError):
                audit.parse_gpu_devices(value)

    def test_random_choice_is_deterministic_available_and_never_prior(self) -> None:
        availability = np.asarray(
            [
                [True, False, True, False, False, False, False, True],
                [False, True, False, True, False, True, False, True],
                [False, False, False, False, False, False, False, True],
            ],
            dtype=bool,
        )
        first, first_eligible = audit.matched_random_choice(availability, seed=42)
        second, second_eligible = audit.matched_random_choice(availability, seed=42)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(first_eligible, second_eligible))
        self.assertTrue(np.array_equal(first_eligible, np.asarray([True, True, False])))
        self.assertLess(int(np.max(first)), 7)
        for row, is_eligible in enumerate(first_eligible):
            if is_eligible:
                self.assertTrue(availability[row, first[row]])


if __name__ == "__main__":
    unittest.main()
