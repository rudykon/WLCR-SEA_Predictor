from __future__ import annotations

import ast
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from experiments import train_neural_baselines as neural


class NeuralBaselineContractTest(unittest.TestCase):
    def test_registered_protocol_dates_and_equal_candidate_budget(self) -> None:
        self.assertEqual(
            [str(value) for value in neural.FIT_DATES],
            [f"2024-08-{day:02d}" for day in range(3, 10)],
        )
        self.assertEqual(
            [str(value) for value in neural.INNER_DATES],
            ["2024-08-10", "2024-08-11"],
        )
        self.assertEqual(
            [str(value) for value in neural.HOLDOUT_DATES],
            [f"2024-08-{day:02d}" for day in range(12, 19)],
        )
        self.assertEqual(
            {name: len(configs) for name, configs in neural.MODEL_CONFIGS.items()},
            {"dlinear": 2, "nlinear": 2, "patchtst": 2, "grud_direct": 2},
        )
        self.assertEqual(neural.parse_model_list("grud_direct"), ["grud_direct"])
        self.assertEqual(
            [config["name"] for config in neural.MODEL_CONFIGS["grud_direct"]],
            ["grud_direct_h32_lr1e3", "grud_direct_h48_lr5e4"],
        )

    def test_model_shapes_and_finite_outputs(self) -> None:
        inputs = torch.randn(2, neural.INPUT_HOURS, neural.MODEL_INPUT_CHANNELS)
        for model_name, configs in neural.MODEL_CONFIGS.items():
            model = neural.build_model(model_name, configs[0])
            with self.subTest(model=model_name):
                output = model(inputs)
                self.assertEqual(
                    tuple(output.shape),
                    (2, neural.FORECAST_HOURS, neural.TARGET_COUNT),
                )
                self.assertTrue(torch.isfinite(output).all())
                self.assertGreater(neural.count_parameters(model), 0)

    def test_grud_elapsed_time_and_masked_values_use_post_corruption_mask(self) -> None:
        x_values = np.full((1, 336, 4), np.log1p(2.0), dtype=np.float32)
        x_masks = np.ones_like(x_values, dtype=np.uint8)
        dataset = neural.CachedDataset(
            root=Path("<memory>"),
            x_values=x_values,
            x_masks=x_masks,
            targets=np.ones((1, 24, 4), dtype=np.float32),
            target_masks=np.ones((1, 24, 4), dtype=np.uint8),
            mase_scales=np.ones((1, 4), dtype=np.float32),
            cells=np.asarray(["cell-a"]),
            target_start_hours=np.asarray([1001], dtype=np.int64),
            history_end_hours=np.asarray([1000], dtype=np.int64),
        )
        normalization = neural.Normalization(
            input_mean=(0.0, 0.0, 0.0, 0.0),
            input_std=(1.0, 1.0, 1.0, 1.0),
            target_mean=(0.0, 0.0, 0.0, 0.0),
            target_std=(1.0, 1.0, 1.0, 1.0),
        )
        extra = np.zeros_like(x_masks, dtype=bool)
        extra[0, 10:12, 0] = True
        inputs = neural.prepared_inputs(
            dataset,
            np.asarray([0]),
            normalization,
            additional_missing=extra,
        )
        model = neural.build_model("grud_direct", neural.MODEL_CONFIGS["grud_direct"][0])
        self.assertIsInstance(model, neural.GRUDDirect)
        elapsed = model.elapsed_since_observation(inputs[:, :, 4:])
        self.assertEqual(float(elapsed[0, 9, 0]), 0.0)
        self.assertEqual(float(elapsed[0, 10, 0]), 1.0)
        self.assertEqual(float(elapsed[0, 11, 0]), 2.0)
        self.assertEqual(float(elapsed[0, 12, 0]), 0.0)
        self.assertEqual(float(elapsed[0, 11, 1]), 0.0)

        with torch.no_grad():
            model.input_decay.weight.fill_(1.0)
            model.input_decay.bias.zero_()
            model.hidden_decay.weight.fill_(1.0)
            model.hidden_decay.bias.zero_()
            short = torch.ones((1, 4), dtype=torch.float32)
            long = 2.0 * short
            self.assertTrue(
                torch.all(
                    model._decay(model.input_decay(short))
                    > model._decay(model.input_decay(long))
                )
            )
            self.assertTrue(
                torch.all(
                    model._decay(model.hidden_decay(short))
                    > model._decay(model.hidden_decay(long))
                )
            )

        model.eval()
        altered = inputs.clone()
        altered[0, 10:12, 0] = torch.tensor([1_000_000.0, -1_000_000.0])
        with torch.no_grad():
            baseline = model(inputs)
            masked_value_changed = model(altered)
        torch.testing.assert_close(baseline, masked_value_changed, rtol=0.0, atol=0.0)
        self.assertTrue(torch.isfinite(baseline).all())

    def test_window_median_fill_preserves_mask(self) -> None:
        history = np.tile(
            np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64),
            (neural.INPUT_HOURS, 1),
        )
        history[0, 0] = np.nan
        values, masks, all_missing = neural.median_fill_and_mask(history)
        self.assertEqual(all_missing, 0)
        self.assertEqual(int(masks[0, 0]), 0)
        self.assertAlmostEqual(float(values[0, 0]), float(np.log1p(1.0)), places=6)
        self.assertEqual(values.shape, (336, 4))
        self.assertEqual(masks.shape, (336, 4))

    def test_input_only_mixed_augmentation_refills_removed_values(self) -> None:
        x_values = np.full((2, 336, 4), np.log1p(2.0), dtype=np.float32)
        x_values[0, 100, 0] = np.log1p(1_000_000.0)
        x_masks = np.ones_like(x_values, dtype=np.uint8)
        targets = np.full((2, 24, 4), 7.0, dtype=np.float32)
        target_masks = np.ones_like(targets, dtype=np.uint8)
        dataset = neural.CachedDataset(
            root=Path("<memory>"),
            x_values=x_values,
            x_masks=x_masks,
            targets=targets,
            target_masks=target_masks,
            mase_scales=np.ones((2, 4), dtype=np.float32),
            cells=np.asarray(["cell-a", "cell-a"]),
            target_start_hours=np.asarray([1001, 1025], dtype=np.int64),
            history_end_hours=np.asarray([1000, 1024], dtype=np.int64),
        )
        normalization = neural.Normalization(
            input_mean=(0.0, 0.0, 0.0, 0.0),
            input_std=(1.0, 1.0, 1.0, 1.0),
            target_mean=(0.0, 0.0, 0.0, 0.0),
            target_std=(1.0, 1.0, 1.0, 1.0),
        )
        extra = np.zeros_like(x_masks, dtype=bool)
        extra[0, 100, 0] = True
        clean = neural.prepared_tensors(dataset, np.asarray([0]), normalization)
        corrupt = neural.prepared_tensors(
            dataset,
            np.asarray([0]),
            normalization,
            additional_missing=extra[:1],
        )
        self.assertEqual(float(corrupt[0][0, 100, 4]), 0.0)
        self.assertLess(float(corrupt[0][0, 100, 0]), 10.0)
        self.assertTrue(torch.equal(clean[1], corrupt[1]))
        self.assertTrue(torch.equal(clean[2], corrupt[2]))

    def test_training_augmentation_reports_unique_and_exposure_denominators(self) -> None:
        x_values = np.full((2, 336, 4), np.log1p(2.0), dtype=np.float32)
        x_masks = np.ones_like(x_values, dtype=np.uint8)
        dataset = neural.CachedDataset(
            root=Path("<memory>"),
            x_values=x_values,
            x_masks=x_masks,
            targets=np.ones((2, 24, 4), dtype=np.float32),
            target_masks=np.ones((2, 24, 4), dtype=np.uint8),
            mase_scales=np.ones((2, 4), dtype=np.float32),
            cells=np.asarray(["cell-a", "cell-a"]),
            target_start_hours=np.asarray([1001, 1025], dtype=np.int64),
            history_end_hours=np.asarray([1000, 1024], dtype=np.int64),
        )
        extra, report = neural.training_augmentation(
            dataset,
            np.asarray([0, 1]),
            augmentation="mixed",
            requested_rate=0.15,
            seed=42,
        )
        self.assertEqual(extra.shape, x_masks.shape)
        self.assertIn("unique_cell_time", report)
        self.assertIn("window_exposure", report)
        self.assertFalse(report["labels_or_target_masks_modified"])
        self.assertTrue(report["normalization_uses_original_observed_training_values"])

    def test_continuous_720_hour_series_forms_sixteen_windows(self) -> None:
        start = datetime(2024, 7, 20)
        timestamps = [start + timedelta(hours=index) for index in range(720)]
        metrics = np.tile(
            np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float64),
            (720, 1),
        )
        arrays, report = neural.build_window_arrays(
            {"cell-a": (timestamps, metrics)}
        )
        self.assertEqual(report["candidate_windows"], 16)
        self.assertEqual(report["continuous_windows"], 16)
        self.assertEqual(report["discontinuous_windows"], [])
        self.assertTrue(
            np.all(arrays["history_end_hours"] + 1 == arrays["target_start_hours"])
        )
        self.assertEqual(arrays["x_values"].shape, (16, 336, 4))
        self.assertEqual(arrays["targets"].shape, (16, 24, 4))

    def test_combined_metric_is_exact_for_perfect_prediction(self) -> None:
        actual = np.full((2, 24, 4), 10.0, dtype=np.float32)
        result = neural.combined_scores(actual, actual.copy())
        self.assertEqual(result["mape_auc"], 1.0)
        self.assertEqual(result["mean_mape"], 0.0)
        self.assertEqual(result["n_hours"], 48)

    def test_output_path_cannot_escape_artifact_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "outputs must remain"):
            neural.resolve_output("/tmp/not-paper-neural-output")

    def test_source_contains_no_duplicate_literal_dictionary_keys(self) -> None:
        source = Path(neural.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        duplicates: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            for key in set(keys):
                if keys.count(key) > 1:
                    duplicates.append((node.lineno, key))
        self.assertEqual(duplicates, [])

    def test_registered_data_split_and_leakage_checks(self) -> None:
        train_path = neural.resolve_train_path()
        before = neural.sha256_file(train_path)
        arrays, report = neural.build_window_arrays(
            neural.read_training_series(train_path)
        )
        self.assertEqual(report["candidate_windows"], 11_686)
        self.assertEqual(report["continuous_windows"], 11_685)
        self.assertEqual(len(report["discontinuous_windows"]), 1)
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            neural.write_dataset_cache(cache, arrays)
            dataset = neural.load_dataset_cache(cache)
            checks = neural.leakage_checks(dataset)
        self.assertEqual(checks["fit_windows"], 5_115)
        self.assertEqual(checks["inner_windows"], 1_460)
        self.assertEqual(checks["holdout_windows"], 5_110)
        self.assertFalse(checks["finals_test_opened"])
        self.assertFalse(checks["explicit_cell_id_feature"])
        self.assertFalse(checks["cross_window_traffic_features"])
        self.assertEqual(neural.sha256_file(train_path), before)


if __name__ == "__main__":
    unittest.main()
