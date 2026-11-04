from __future__ import annotations

import unittest

import numpy as np

from experiments import estimate_paired_robustness as revision8
from experiments import wlcr_sea_model as sea


class Revision8RobustnessBootstrapTest(unittest.TestCase):
    def test_point_estimate_averages_paired_macro_wape_deltas(self) -> None:
        actual = np.ones((4, sea.FORECAST_HOURS, sea.TARGET_COUNT), dtype=np.float32)
        actual[..., 1] *= 10.0
        actual[..., 2] *= 100.0
        actual[..., 3] *= 1000.0
        proposed_seed_a = actual + np.asarray([0.1, 2.0, 30.0, 400.0], dtype=np.float32)
        baseline_seed_a = actual + np.asarray([0.2, 1.0, 20.0, 100.0], dtype=np.float32)
        proposed_seed_b = actual + np.asarray([0.15, 1.0, 10.0, 50.0], dtype=np.float32)
        baseline_seed_b = actual + np.asarray([0.10, 2.0, 20.0, 200.0], dtype=np.float32)
        cells = np.asarray(["a", "a", "b", "b"])

        result = revision8.paired_multi_seed_cell_cluster_bootstrap(
            actual,
            [proposed_seed_a, proposed_seed_b],
            [baseline_seed_a, baseline_seed_b],
            cells,
            corruption_seeds=[142, 143],
            replicates=100,
            seed=17,
        )

        def macro_wape(prediction: np.ndarray) -> float:
            numerator = np.sum(np.abs(actual - prediction), axis=(0, 1))
            denominator = np.sum(np.abs(actual), axis=(0, 1))
            return float(np.mean(numerator / denominator))

        expected = np.mean(
            [
                macro_wape(proposed_seed_a) - macro_wape(baseline_seed_a),
                macro_wape(proposed_seed_b) - macro_wape(baseline_seed_b),
            ]
        )
        self.assertEqual(
            result["estimand"],
            "mean_over_corruption_seeds_of_paired_macro_over_indicator_wape_delta",
        )
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["corruption_seed_count"], 2)
        self.assertAlmostEqual(
            result["delta_proposed_minus_baseline_mean"], expected, places=7
        )

    def test_rejects_misaligned_corruption_seed_lists(self) -> None:
        actual = np.ones((1, sea.FORECAST_HOURS, sea.TARGET_COUNT), dtype=np.float32)
        with self.assertRaises(ValueError):
            revision8.paired_multi_seed_cell_cluster_bootstrap(
                actual,
                [actual.copy()],
                [actual.copy()],
                np.asarray(["a"]),
                corruption_seeds=[142, 143],
                replicates=10,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
