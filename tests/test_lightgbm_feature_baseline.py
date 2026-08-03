from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta

from Model.traffic_window_forecasting import BaselineConfig, TrafficRow, seasonal_forecast, split_physical_windows
from Model.lightgbm_feature_baseline import build_feature_row


class LightGBMFeatureTest(unittest.TestCase):
    def test_features_are_window_local_and_exclude_cell_id(self) -> None:
        start = datetime(2024, 1, 1)
        rows = [
            TrafficRow(
                start + timedelta(hours=index),
                "private-cell-id",
                (10.0 + index % 24, 20.0 + index % 24, 30.0 + index % 24, 40.0 + index % 24),
            )
            for index in range(336)
        ]
        rows[100] = TrafficRow(rows[100].timestamp, rows[100].cell, (None, None, None, None))
        window = split_physical_windows(rows)[0]
        baseline = seasonal_forecast(window, BaselineConfig.default())[0]
        names, values = build_feature_row(
            window,
            horizon=0,
            baseline=baseline,
            parameter={"azimuth": 90.0, "scene_code": 2.0, "x": 1.0, "y": 2.0},
            weather={"weather_code": 1.0, "avg_temp": 20.0, "humidity": 60.0, "rain": 0.0, "wind": 5.0},
        )
        self.assertEqual(len(names), len(values))
        self.assertNotIn("cell_id", names)
        self.assertGreater(len(names), 40)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertIn("baseline_m0", names)
        self.assertIn("lag7_m0", names)
        self.assertIn("lag7_mask_m0", names)

if __name__ == "__main__":
    unittest.main()
