from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from Model.traffic_window_forecasting import (
    BacktestExample,
    BaselineConfig,
    ForecastRow,
    TestWindow,
    TrafficRow,
)
from Model.lightgbm_feature_baseline import MatrixBundle
from experiments.train_lightgbm_baseline import cell_fold
from experiments.select_lightgbm_model import (
    TunedCandidate,
    build_outer_fold_plan,
    build_temporal_protocol,
    evaluate_development_candidate,
    example_cells,
    fold_plan_report,
    parse_variants,
    place_oof_predictions,
    select_fold_baseline,
    tune_fold_candidates,
)


def cells_by_fold(count_per_fold: int = 2) -> dict[int, tuple[str, ...]]:
    found: dict[int, list[str]] = {fold: [] for fold in range(5)}
    candidate = 0
    while any(len(values) < count_per_fold for values in found.values()):
        cell = f"cell-{candidate}"
        fold = cell_fold(cell)
        if len(found[fold]) < count_per_fold:
            found[fold].append(cell)
        candidate += 1
    return {fold: tuple(values) for fold, values in found.items()}


def synthetic_examples() -> tuple[list[BacktestExample], dict[int, tuple[str, ...]]]:
    folds = cells_by_fold()
    examples = []
    index = 0
    start = datetime(2024, 1, 1)
    for day in range(16):
        target_start = start + timedelta(days=day)
        for cells in folds.values():
            for cell in cells:
                history = TrafficRow(
                    target_start - timedelta(hours=1),
                    cell,
                    (10.0, 20.0, 30.0, 40.0),
                )
                actual = TrafficRow(
                    target_start,
                    cell,
                    (11.0, 21.0, 31.0, 41.0),
                )
                examples.append(
                    BacktestExample(
                        TestWindow(index, cell, (history,), ()),
                        (actual,),
                    )
                )
                index += 1
    return examples, folds


def tiny_matrix() -> MatrixBundle:
    timestamp = datetime(2024, 1, 1)
    actual = TrafficRow(timestamp, "train-cell", (10.0, 10.0, 10.0, 10.0))
    baseline = ForecastRow(timestamp, "train-cell", (10.0, 10.0, 10.0, 10.0))
    return MatrixBundle(
        np.zeros((1, 32), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        (actual,),
        (baseline,),
    )


class StrictNestedSplitTest(unittest.TestCase):
    def test_outer_validation_cells_are_absent_from_every_selection_layer(self) -> None:
        examples, folds = synthetic_examples()
        protocol = build_temporal_protocol(examples)
        plan = build_outer_fold_plan(examples, protocol, 2)
        expected_validation = set(folds[2])
        self.assertEqual(set(plan.validation_cells), expected_validation)
        self.assertFalse(expected_validation & example_cells(plan.seasonal_selection))
        self.assertFalse(expected_validation & example_cells(plan.selection_train))
        self.assertFalse(expected_validation & example_cells(plan.selection_validation))
        self.assertFalse(expected_validation & example_cells(plan.final_train))
        self.assertEqual(example_cells(plan.outer_validation), expected_validation)
        self.assertEqual(len(plan.outer_validation), 3 * len(expected_validation))
        report = fold_plan_report(plan, protocol)
        self.assertTrue(all(report["leakage_checks"].values()))

    def test_seasonal_configuration_is_selected_only_on_outer_training_cells(self) -> None:
        examples, folds = synthetic_examples()
        protocol = build_temporal_protocol(examples)
        plan = build_outer_fold_plan(examples, protocol, 1)
        validation = set(folds[1])
        candidates = (
            BaselineConfig("candidate-a", (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            BaselineConfig("candidate-b", (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        )
        scorer_calls = []

        def scorer(selected_examples, config):
            cells = example_cells(selected_examples)
            dates = {
                example.window.target_start.date() for example in selected_examples
            }
            scorer_calls.append((config.name, cells, dates))
            auc = 0.8 if config.name == "candidate-b" else 0.7
            return {"mape_auc": auc, "mean_mape": 1.0 - auc, "samples": 1}

        selected, report = select_fold_baseline(
            plan.seasonal_selection,
            candidates=candidates,
            scorer=scorer,
        )
        self.assertEqual(selected.name, "candidate-b")
        self.assertEqual(report["selected"]["name"], "candidate-b")
        self.assertEqual(len(scorer_calls), 2)
        for _, cells, dates in scorer_calls:
            self.assertFalse(validation & cells)
            self.assertEqual(dates, set(protocol.inner_dates))
        with self.assertRaisesRegex(ValueError, "at least one candidate"):
            select_fold_baseline(
                plan.seasonal_selection,
                candidates=(),
                scorer=scorer,
            )

    def test_each_proposed_variant_and_plain_receive_an_independent_tuning_call(self) -> None:
        matrix = tiny_matrix()
        schema = [
            "horizon",
            "target_hour_sin",
            "target_hour_cos",
            "target_dow_sin",
            "target_dow_cos",
            "is_weekend",
            "weather_code",
        ]
        for metric in range(4):
            schema.extend(
                [f"lag1_m{metric}", f"lag7_m{metric}", f"lag14_m{metric}"]
            )
        calls = []

        def evaluator(**kwargs):
            role = kwargs["role"]
            variant = kwargs["variant"]
            calls.append((role, variant))
            auc = {"full": 0.70, "no_weather": 0.72, "plain_lgbm": 0.65}[variant]
            rounds = {
                "full": (101, 102, 103, 104),
                "no_weather": (201, 202, 203, 204),
                "plain_lgbm": (301, 302, 303, 304),
            }[variant]
            return TunedCandidate(
                role=role,
                variant=variant,
                columns=kwargs["columns"],
                rounds=rounds,
                development_metrics={"mape_auc": auc, "mean_mape": 1.0 - auc},
                tuning_seconds=0.0,
                training_seconds=0.0,
                prediction_seconds=0.0,
                model_bytes=1,
                cache_signature=variant,
                cache_hit=False,
            )

        selected, plain, candidates = tune_fold_candidates(
            fold=0,
            proposed_variants=("full", "no_weather"),
            feature_schema=schema,
            selection_train=matrix,
            selection_validation=matrix,
            model_root=Path("unused"),
            training_cells=("train-cell",),
            validation_cells=("validation-cell",),
            training_dates=("2024-01-01",),
            baseline=BaselineConfig.default(),
            source_hashes={},
            implementation_hashes={},
            evaluator=evaluator,
        )
        self.assertEqual(
            calls,
            [
                ("proposed", "full"),
                ("proposed", "no_weather"),
                ("plain_lgbm", "plain_lgbm"),
            ],
        )
        self.assertEqual([item.variant for item in candidates], ["full", "no_weather"])
        self.assertEqual(selected.variant, "no_weather")
        self.assertEqual(selected.rounds, (201, 202, 203, 204))
        self.assertEqual(plain.rounds, (301, 302, 303, 304))


class StrictNestedTuningTest(unittest.TestCase):
    def test_candidate_records_its_own_round_selection_and_cache_audit(self) -> None:
        matrix = tiny_matrix()
        columns = tuple(np.asarray([metric], dtype=np.int64) for metric in range(4))
        round_calls = []

        def round_selector(train, valid, selected_columns):
            round_calls.append((train, valid, selected_columns))
            return [11, 12, 13, 14]

        def trainer(train, selected_columns, rounds, cache_dir):
            self.assertEqual(tuple(rounds), (11, 12, 13, 14))
            return ["booster"], 0.25, 123

        def predictor(boosters, valid, selected_columns):
            return [
                ForecastRow(row.timestamp, row.cell, tuple(float(value) for value in row.metrics))
                for row in valid.actuals
            ], 0.01

        with tempfile.TemporaryDirectory() as directory:
            candidate = evaluate_development_candidate(
                fold=3,
                role="plain_lgbm",
                variant="plain_lgbm",
                selection_train=matrix,
                selection_validation=matrix,
                columns=columns,
                model_root=Path(directory),
                training_cells=("train-cell",),
                validation_cells=("validation-cell",),
                training_dates=("2024-01-01",),
                feature_schema=tuple(f"f{index}" for index in range(32)),
                baseline=BaselineConfig.default(),
                source_hashes={"train": "abc"},
                implementation_hashes={"script": "def"},
                round_selector=round_selector,
                trainer=trainer,
                predictor=predictor,
            )
            audits = list(Path(directory).rglob("cache_audit.json"))
            self.assertEqual(len(audits), 1)
            self.assertIn("strict_nested_cell_disjoint_v2", audits[0].parts)
            audit = json.loads(audits[0].read_text(encoding="utf-8"))
        self.assertEqual(len(round_calls), 1)
        self.assertEqual(candidate.rounds, (11, 12, 13, 14))
        self.assertAlmostEqual(float(candidate.development_metrics["mape_auc"]), 1.0)
        self.assertEqual(audit["signature_payload"]["validation_cells"], ["validation-cell"])
        self.assertEqual(audit["signature_payload"]["role"], "plain_lgbm")

    def test_oof_assignment_is_keyed_and_rejects_duplicate_folds(self) -> None:
        start = datetime(2024, 1, 1)
        actuals = [
            TrafficRow(start, "a", (1.0, 1.0, 1.0, 1.0)),
            TrafficRow(start, "b", (1.0, 1.0, 1.0, 1.0)),
        ]
        predictions = [
            ForecastRow(start, "b", (2.0, 2.0, 2.0, 2.0)),
            ForecastRow(start, "a", (3.0, 3.0, 3.0, 3.0)),
        ]
        destinations = [None, None]
        place_oof_predictions(actuals, destinations, predictions)
        self.assertEqual(destinations[0].cell, "a")
        self.assertEqual(destinations[1].cell, "b")
        with self.assertRaisesRegex(RuntimeError, "duplicate OOF"):
            place_oof_predictions(actuals, destinations, predictions[:1])

    def test_variant_parser_rejects_unknown_and_duplicate_candidates(self) -> None:
        self.assertEqual(parse_variants("full,no_weather"), ("full", "no_weather"))
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_variants("full,not-a-variant")
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_variants("full,full")


if __name__ == "__main__":
    unittest.main()
