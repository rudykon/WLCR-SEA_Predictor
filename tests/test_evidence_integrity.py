from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from experiments import validate_evidence_integrity as evidence
from experiments import train_neural_baselines as neural
from experiments import analyze_routing_and_missingness as comparative


class OriginalWLCREvidenceTest(unittest.TestCase):
    def test_exact_keys_and_actuals_are_required(self) -> None:
        start = datetime(2024, 8, 12)
        start_hour = int((start - datetime(1970, 1, 1)).total_seconds() // 3600)
        actual = np.zeros((1, 24, 4), dtype=np.float32)
        for horizon in range(24):
            actual[0, horizon] = np.asarray(
                [1.0 + horizon, 2.0 + horizon, 3.0 + horizon, 4.0 + horizon]
            )
        dataset = neural.CachedDataset(
            root=Path("<memory>"),
            x_values=np.zeros((1, 336, 4), dtype=np.float32),
            x_masks=np.ones((1, 336, 4), dtype=np.uint8),
            targets=actual,
            target_masks=np.ones_like(actual, dtype=np.uint8),
            mase_scales=np.ones((1, 4), dtype=np.float32),
            cells=np.asarray(["cell-a"]),
            target_start_hours=np.asarray([start_hour], dtype=np.int64),
            history_end_hours=np.asarray([start_hour - 1], dtype=np.int64),
        )
        root = evidence.project_root() / "artifacts/revision7"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "predictions.csv.gz"
            fields = ["cell", "target_timestamp", "horizon"]
            fields.extend(f"actual_{name}" for name in neural.METRIC_NAMES)
            fields.extend(
                f"{evidence.PREDICTION_PREFIX}{name}"
                for name in neural.METRIC_NAMES
            )
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for horizon in range(24):
                    row = {
                        "cell": "cell-a",
                        "target_timestamp": (
                            start + timedelta(hours=horizon)
                        ).isoformat(sep=" "),
                        "horizon": horizon + 1,
                    }
                    for metric, name in enumerate(neural.METRIC_NAMES):
                        row[f"actual_{name}"] = float(actual[0, horizon, metric])
                        row[f"{evidence.PREDICTION_PREFIX}{name}"] = float(
                            actual[0, horizon, metric] + 0.5
                        )
                    writer.writerow(row)
            prediction, report = evidence.load_original_wlcr_prediction(
                path, dataset, np.asarray([0], dtype=np.int64)
            )
        self.assertEqual(prediction.shape, (1, 24, 4))
        self.assertTrue(np.allclose(prediction, actual + 0.5))
        self.assertTrue(report["exact_key_set_match"])
        self.assertEqual(report["forecast_rows"], 24)
        self.assertEqual(report["maximum_actual_absolute_difference"], 0.0)
        self.assertFalse(report["finals_test_opened"])


class CorruptionSeedAggregationTest(unittest.TestCase):
    def test_aggregate_reports_mean_sd_and_t_interval(self) -> None:
        values = (0.20, 0.21, 0.22, 0.23, 0.24)
        rows = [
            {
                "method": "model-a",
                "mechanism": "block",
                "requested_rate": 0.2,
                "corruption_seed": seed,
                "macro_wape": value,
            }
            for seed, value in zip(comparative.CORRUPTION_SEEDS, values)
        ]
        result = comparative.aggregate_seed_rows(
            rows,
            group_fields=("method", "mechanism", "requested_rate"),
            numeric_fields=("macro_wape",),
        )
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["corruption_seed_count"], 5)
        self.assertEqual(
            row["corruption_seeds"],
            ",".join(str(seed) for seed in comparative.CORRUPTION_SEEDS),
        )
        self.assertAlmostEqual(row["macro_wape"], 0.22)
        expected_sd = float(np.std(np.asarray(values), ddof=1))
        self.assertAlmostEqual(row["macro_wape_sd"], expected_sd)
        expected_half_width = comparative.T95_DF4 * expected_sd / np.sqrt(5.0)
        self.assertAlmostEqual(
            row["macro_wape_ci_low"], 0.22 - expected_half_width
        )
        self.assertAlmostEqual(
            row["macro_wape_ci_high"], 0.22 + expected_half_width
        )

    def test_shared_neural_view_matches_reference_preparation(self) -> None:
        base = np.arange(2 * 336 * 4, dtype=np.float32).reshape(2, 336, 4)
        x_values = np.log1p(base / 100.0).astype(np.float32)
        x_masks = np.ones_like(x_values, dtype=np.uint8)
        x_masks[0, :, 3] = 0
        x_masks[1, 10:30, 1] = 0
        dataset = neural.CachedDataset(
            root=Path("<memory>"),
            x_values=x_values,
            x_masks=x_masks,
            targets=np.ones((2, 24, 4), dtype=np.float32),
            target_masks=np.ones((2, 24, 4), dtype=np.uint8),
            mase_scales=np.ones((2, 4), dtype=np.float32),
            cells=np.asarray(["a", "b"]),
            target_start_hours=np.asarray([0, 24], dtype=np.int64),
            history_end_hours=np.asarray([-1, 23], dtype=np.int64),
        )
        extra = np.zeros_like(x_masks, dtype=bool)
        extra[0, 40:100, 0] = True
        extra[1, 200:260, 2] = True
        indices = np.asarray([0, 1], dtype=np.int64)
        values, masks = comparative.shared_neural_request_view(
            dataset, indices, extra
        )
        for shift in (0.0, 0.25):
            normalization = neural.Normalization(
                input_mean=(shift, shift + 0.1, shift + 0.2, shift + 0.3),
                input_std=(1.0, 1.1, 1.2, 1.3),
                target_mean=(0.0, 0.0, 0.0, 0.0),
                target_std=(1.0, 1.0, 1.0, 1.0),
            )
            expected = neural.prepared_inputs(
                dataset, indices, normalization, additional_missing=extra
            ).numpy()
            observed = comparative.normalized_neural_inputs(
                values, masks, normalization
            ).numpy()
            self.assertTrue(np.array_equal(expected, observed))

    def test_clean_replay_tolerance_is_float32_scale_and_reported(self) -> None:
        self.assertEqual(comparative.CLEAN_REPLAY_ABS_TOLERANCE, 2e-4)
        source = Path(
            "experiments/analyze_routing_and_missingness.py"
        ).read_text(encoding="utf-8")
        self.assertIn("clean_replay_max_absolute_difference", source)
        self.assertIn("clean_replay_abs_tolerance", source)


if __name__ == "__main__":
    unittest.main()
