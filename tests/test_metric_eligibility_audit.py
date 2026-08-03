from __future__ import annotations

import unittest

import numpy as np

from experiments import audit_metric_eligibility as coverage


class Revision8MetricCoverageTest(unittest.TestCase):
    def test_excludes_only_invalid_or_zero_request_indicator_scales(self) -> None:
        targets = np.ones((2, 24, 4), dtype=np.float32)
        masks = np.ones_like(targets, dtype=np.uint8)
        scales = np.ones((2, 4), dtype=np.float32)
        scales[0, 1] = np.nan
        scales[1, 2] = 0.0
        report = coverage.mase_coverage(targets, masks, scales)
        self.assertEqual(report["request_indicator_groups_with_observed_targets"], 8)
        self.assertEqual(report["mase_eligible_request_indicator_groups"], 6)
        self.assertEqual(report["mase_excluded_request_indicator_groups"], 2)
        self.assertAlmostEqual(report["group_coverage"], 0.75)
        self.assertEqual(report["observed_forecast_values"], 192)
        self.assertEqual(report["mase_eligible_forecast_values"], 144)


if __name__ == "__main__":
    unittest.main()
