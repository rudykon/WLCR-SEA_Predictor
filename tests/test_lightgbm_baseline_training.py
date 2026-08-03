from __future__ import annotations

import copy
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Model.traffic_window_forecasting import BacktestExample, ForecastRow, TestWindow, TrafficRow
from Model.lightgbm_feature_baseline import MatrixBundle
import experiments.train_lightgbm_baseline as experiments


def tiny_matrix(feature_count: int = 6) -> MatrixBundle:
    timestamp = datetime(2024, 1, 1)
    actuals = tuple(
        TrafficRow(
            timestamp + timedelta(hours=index),
            f"cell-{index}",
            (10.0 + index, 20.0 + index, 30.0 + index, 40.0 + index),
        )
        for index in range(3)
    )
    baselines = tuple(
        ForecastRow(row.timestamp, row.cell, tuple(float(value) for value in row.metrics))
        for row in actuals
    )
    return MatrixBundle(
        np.arange(3 * feature_count, dtype=np.float32).reshape(3, feature_count),
        np.zeros((3, 4), dtype=np.float32),
        actuals,
        baselines,
    )


def synthetic_example(cell: str = "cell-a") -> BacktestExample:
    start = datetime(2024, 1, 1)
    rows = tuple(
        TrafficRow(
            start + timedelta(hours=index),
            cell,
            (float(index + 1),) * 4,
        )
        for index in range(12)
    )
    window = TestWindow(0, cell, rows, ())
    actuals = tuple(
        TrafficRow(
            window.target_start + timedelta(hours=index),
            cell,
            (20.0 + index,) * 4,
        )
        for index in range(2)
    )
    return BacktestExample(window, actuals)


class FakeRoundBooster:
    def __init__(self, best_iteration: int, current_iteration: int) -> None:
        self.best_iteration = best_iteration
        self._current_iteration = current_iteration

    def current_iteration(self) -> int:
        return self._current_iteration


class RoundSelectionTest(unittest.TestCase):
    def test_defaults_use_large_budget_and_explicit_seeds(self) -> None:
        self.assertGreaterEqual(experiments.MAX_BOOST_ROUNDS, 1500)
        self.assertEqual(experiments.MODEL_SEED, 42)
        self.assertEqual(experiments.BOOTSTRAP_SEED, 42)
        self.assertEqual(experiments.CORRUPTION_SEED, 42)
        self.assertEqual(experiments.MODEL_PARAMS["seed"], experiments.MODEL_SEED)
        self.assertEqual(
            experiments.MODEL_PARAMS["feature_fraction_seed"],
            experiments.MODEL_SEED,
        )
        self.assertEqual(
            experiments.MODEL_PARAMS["bagging_seed"], experiments.MODEL_SEED
        )
        self.assertEqual(
            experiments.MODEL_PARAMS["data_random_seed"], experiments.MODEL_SEED
        )

    def test_plain_and_proposed_are_early_stopped_independently(self) -> None:
        matrix = tiny_matrix()
        proposed = [np.arange(6, dtype=np.int64) for _ in range(4)]
        plain = [np.arange(2, dtype=np.int64) for _ in range(4)]
        selected_iterations = [101, 102, 103, 104, 201, 202, 203, 204]
        calls = []

        def fake_train(*args, **kwargs):
            iteration = selected_iterations[len(calls)]
            calls.append(kwargs["num_boost_round"])
            return FakeRoundBooster(iteration, iteration + 60)

        with (
            patch.object(experiments, "configured_gpu_devices", return_value=[0, 1, 2, 3]),
            patch.object(experiments.lgb, "Dataset", return_value=object()),
            patch.object(experiments.lgb, "train", side_effect=fake_train),
            patch.object(experiments.lgb, "early_stopping", return_value=object()),
            patch.object(experiments.lgb, "log_evaluation", return_value=object()),
        ):
            selections = experiments.select_model_rounds(
                matrix,
                matrix,
                {"proposed": proposed, "plain_lgbm": plain},
                max_rounds=1500,
                early_stopping_rounds=60,
            )

        self.assertEqual(len(calls), 8)
        self.assertEqual(selections["proposed"].rounds, (101, 102, 103, 104))
        self.assertEqual(selections["plain_lgbm"].rounds, (201, 202, 203, 204))

    def test_round_diagnostics_record_max_round_contact(self) -> None:
        matrix = tiny_matrix()
        columns = [np.arange(2, dtype=np.int64) for _ in range(4)]

        with (
            patch.object(experiments, "configured_gpu_devices", return_value=[0]),
            patch.object(experiments.lgb, "Dataset", return_value=object()),
            patch.object(
                experiments.lgb,
                "train",
                side_effect=lambda *args, **kwargs: FakeRoundBooster(1500, 1500),
            ),
            patch.object(experiments.lgb, "early_stopping", return_value=object()),
            patch.object(experiments.lgb, "log_evaluation", return_value=object()),
        ):
            selection = experiments.select_rounds_with_diagnostics(
                matrix,
                matrix,
                columns,
                max_rounds=1500,
                early_stopping_rounds=60,
            )

        self.assertEqual(selection.rounds, (1500, 1500, 1500, 1500))
        self.assertTrue(all(item["hit_max_rounds"] for item in selection.diagnostics))
        self.assertTrue(
            all(item["best_iteration_at_max"] for item in selection.diagnostics)
        )


class CacheManifestTest(unittest.TestCase):
    def make_cache(self, directory: str) -> tuple[Path, dict[str, object], list[Path]]:
        cache_dir = Path(directory) / "cache"
        cache_dir.mkdir()
        paths = [cache_dir / f"metric_{metric}.txt" for metric in range(4)]
        for metric, path in enumerate(paths):
            path.write_text(f"model-{metric}\n", encoding="utf-8")
        context = {
            "schema_version": 2,
            "code": {"train_lightgbm_baseline": {"sha256": "code-a"}},
            "inputs": {"train": {"sha256": "data-a"}},
        }
        columns = [np.asarray([0, 1], dtype=np.int64) for _ in range(4)]
        config = experiments.build_cache_config(
            context,
            variant="full",
            training_matrix_sha256="matrix-a",
            feature_names_=("f0", "f1", "f2"),
            columns=columns,
            rounds=(101, 102, 103, 104),
        )
        experiments.write_cache_manifest(cache_dir, config, paths)
        return cache_dir, config, paths

    def test_cache_rejects_data_feature_parameter_round_and_code_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir, config, paths = self.make_cache(directory)
            self.assertEqual(
                experiments.validated_cached_model_paths(cache_dir, config), paths
            )
            mutations = []

            changed = copy.deepcopy(config)
            changed["training_matrix_sha256"] = "matrix-b"
            mutations.append(changed)

            changed = copy.deepcopy(config)
            changed["feature_schema_sha256"] = "feature-b"
            mutations.append(changed)

            changed = copy.deepcopy(config)
            changed["model_params"]["learning_rate"] = 0.5
            mutations.append(changed)

            changed = copy.deepcopy(config)
            changed["rounds"][0] += 1
            mutations.append(changed)

            changed_context = copy.deepcopy(config["experiment_context"])
            changed_context["code"]["train_lightgbm_baseline"]["sha256"] = "code-b"
            changed = copy.deepcopy(config)
            changed["experiment_context"] = changed_context
            changed["experiment_context_sha256"] = experiments.payload_sha256(
                changed_context
            )
            mutations.append(changed)

            for expected in mutations:
                with self.subTest(change=expected):
                    with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                        experiments.validated_cached_model_paths(cache_dir, expected)

    def test_cache_rejects_model_size_and_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir, config, paths = self.make_cache(directory)
            paths[0].write_text("model-0-extra\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                experiments.validated_cached_model_paths(cache_dir, config)

        with tempfile.TemporaryDirectory() as directory:
            cache_dir, config, paths = self.make_cache(directory)
            paths[0].write_text("MODEL-0\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                experiments.validated_cached_model_paths(cache_dir, config)

    def test_model_cache_inventory_exposes_manifest_and_model_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, paths = self.make_cache(directory)
            inventory = experiments.model_cache_inventory(Path(directory))
        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0]["cache"], "cache")
        self.assertEqual(len(inventory[0]["manifest_sha256"]), 64)
        self.assertEqual(
            [item["file"] for item in inventory[0]["models"]],
            [path.name for path in paths],
        )
        self.assertTrue(
            all(len(item["sha256"]) == 64 for item in inventory[0]["models"])
        )

    def test_gpu_workers_are_batched_by_available_device_count(self) -> None:
        matrix = tiny_matrix(feature_count=1)
        columns = [np.asarray([0], dtype=np.int64) for _ in range(4)]
        state = {"active": 0, "maximum": 0}

        class FakeProcess:
            def __init__(self, args) -> None:
                self.args = args
                self.exitcode = None

            def start(self) -> None:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
                Path(self.args[4]).write_text("fake-model\n", encoding="utf-8")
                self.exitcode = 0

            def join(self) -> None:
                state["active"] -= 1

        class FakeContext:
            def Process(self, *, target, args):
                return FakeProcess(args)

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(experiments, "configured_gpu_devices", return_value=[3, 7]),
                patch.object(experiments.mp, "get_context", return_value=FakeContext()),
                patch.object(
                    experiments.lgb,
                    "Booster",
                    side_effect=lambda **kwargs: kwargs["model_file"],
                ),
            ):
                boosters, _, _ = experiments.train_or_load_boosters(
                    matrix,
                    columns,
                    (1, 1, 1, 1),
                    Path(directory) / "models",
                )
            self.assertEqual(len(boosters), 4)
            self.assertEqual(state["maximum"], 2)
            self.assertTrue(
                (Path(directory) / "models" / experiments.CACHE_MANIFEST_NAME).is_file()
            )


class CorruptionAndReportingTest(unittest.TestCase):
    def test_corruption_seed_and_masks_are_deterministic(self) -> None:
        example = synthetic_example()
        first = experiments.corrupt_examples(
            [example], "random", 0.25, base_seed=123
        )[0]
        second = experiments.corrupt_examples(
            [example], "random", 0.25, base_seed=123
        )[0]
        first_mask = [row.metrics[0] is None for row in first.window.rows]
        second_mask = [row.metrics[0] is None for row in second.window.rows]
        self.assertEqual(first_mask, second_mask)
        self.assertEqual(sum(first_mask), 3)
        self.assertTrue(
            all(
                row.metrics == (None, None, None, None)
                for row, masked in zip(first.window.rows, first_mask)
                if masked
            )
        )
        self.assertEqual(
            experiments.corruption_seed(
                example.window.cell,
                example.window.target_start,
                "random",
                0.25,
                base_seed=123,
            ),
            experiments.corruption_seed(
                example.window.cell,
                example.window.target_start,
                "random",
                0.25,
                base_seed=123,
            ),
        )
        self.assertNotEqual(
            experiments.corruption_seed(
                example.window.cell,
                example.window.target_start,
                "random",
                0.25,
                base_seed=123,
            ),
            experiments.corruption_seed(
                example.window.cell,
                example.window.target_start,
                "random",
                0.25,
                base_seed=124,
            ),
        )

    def test_dated_results_are_split_by_target_date(self) -> None:
        start = datetime(2024, 1, 1, 23)
        actuals = [
            TrafficRow(start, "a", (10.0, 20.0, 30.0, 40.0)),
            TrafficRow(start + timedelta(hours=1), "a", (11.0, 21.0, 31.0, 41.0)),
        ]
        predictions = [
            ForecastRow(row.timestamp, row.cell, tuple(float(v) for v in row.metrics))
            for row in actuals
        ]
        rows = experiments.dated_result_rows(
            "proposed", "temporal_lockbox", actuals, predictions
        )
        self.assertEqual([row["date"] for row in rows], ["2024-01-01", "2024-01-02"])
        self.assertTrue(all(row["method"] == "proposed" for row in rows))
        self.assertTrue(all(row["mape_auc"] == 1.0 for row in rows))

    def test_robustness_rows_require_and_report_all_four_methods(self) -> None:
        payloads = {
            "robust_seasonal": {"mape_auc": 0.60, "mean_mape": 0.40},
            "plain_lgbm": {"mape_auc": 0.65, "mean_mape": 0.35},
            "no_missingness": {"mape_auc": 0.67, "mean_mape": 0.33},
            "proposed": {"mape_auc": 0.70, "mean_mape": 0.30},
        }
        row = experiments.robustness_result_row("random", 0.3, payloads)
        for method in payloads:
            self.assertIn(f"{method}_mape_auc", row)
            self.assertIn(f"{method}_mean_mape", row)
            self.assertIn(f"{method}_gain_vs_robust_seasonal", row)
        self.assertAlmostEqual(row["gain"], 0.10)
        with self.assertRaisesRegex(ValueError, "payload methods"):
            experiments.robustness_result_row(
                "random", 0.3, {key: value for key, value in payloads.items() if key != "plain_lgbm"}
            )


class SafetyGateTest(unittest.TestCase):
    def test_output_and_input_gates_reject_legacy_or_test_paths(self) -> None:
        allowed = experiments.resolve_v2_output("artifacts/unit_test_v2_output")
        self.assertTrue(allowed.name.endswith("v2_output"))
        with self.assertRaisesRegex(ValueError, "legacy"):
            experiments.resolve_v2_output("artifacts/paper_experiments_gpu4")
        with self.assertRaisesRegex(ValueError, "registered input"):
            experiments.resolve_registered_input(
                "data/test_data.csv", "data/train_data.csv"
            )

    def test_train_lightgbm_baseline_uses_an_independent_model_namespace(self) -> None:
        output = Path("artifacts/example_v2")
        root = experiments.train_lightgbm_baseline_model_root(output)
        self.assertEqual(
            root,
            output / "models" / experiments.MODEL_CACHE_NAMESPACE,
        )
        self.assertNotEqual(root, output / "models")
        source = Path(experiments.__file__).read_text(encoding="utf-8")
        self.assertNotIn("test_data.csv", source)


if __name__ == "__main__":
    unittest.main()
