from __future__ import annotations

"""Compute revision-5 cell-cluster WAPE intervals and daily descriptions."""

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from Model.traffic_window_forecasting import build_training_backtests, read_traffic
from Model.lightgbm_feature_baseline import build_matrix, load_parameters, load_weather
from experiments.lightgbm_experiment_helpers import (
    build_standard_stat_matrix,
    no_weather_columns,
    standard_stat_feature_names,
    traffic_only_columns,
)
from experiments.train_lightgbm_baseline import feature_columns, feature_names, predict_boosters
from experiments.run_reproducibility_evaluation import (
    PredictionBundle,
    bundle_from_examples,
    load_verified_boosters,
    standard_metric_rows,
)
from experiments.run_feature_ablation_evaluation import (
    DLINEAR_PREDICTIONS,
    FULL_DIR,
    NO_WEATHER_DIR,
    PLAIN_DIR,
    TRAFFIC_ONLY_DIR,
    load_neural_prediction_arrays,
    split_examples,
    write_csv_atomic,
    write_json_atomic,
)
from experiments.run_seasonal_anchor_ablations import registered_inputs, select_baseline_for_inner


OUTPUT_ROOT = Path("artifacts/revision5")
BOOTSTRAP_SEED = 42
DEFAULT_REPLICATES = 5000


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def macro_wape(bundle: PredictionBundle, method: str) -> float:
    row = next(
        item
        for item in standard_metric_rows(bundle)
        if item["method"] == method
        and item["filter"] == "complete_targets_unfiltered"
        and item["indicator"] == "macro_mean"
    )
    return float(row["wape"])


def cell_cluster_wape_bootstrap(
    bundle: PredictionBundle,
    reference: np.ndarray,
    candidate: np.ndarray,
    replicates: int,
) -> dict[str, object]:
    complete = np.all(np.isfinite(bundle.actual), axis=1)
    cells = np.asarray(bundle.cells[complete], dtype=str)
    actual = np.asarray(bundle.actual[complete], dtype=np.float64)
    ref = np.asarray(reference[complete], dtype=np.float64)
    cand = np.asarray(candidate[complete], dtype=np.float64)
    unique_cells = np.unique(cells)
    denominator = np.asarray(
        [np.sum(np.abs(actual[cells == cell]), axis=0) for cell in unique_cells],
        dtype=np.float64,
    )
    ref_error = np.asarray(
        [np.sum(np.abs(actual[cells == cell] - ref[cells == cell]), axis=0) for cell in unique_cells],
        dtype=np.float64,
    )
    cand_error = np.asarray(
        [np.sum(np.abs(actual[cells == cell] - cand[cells == cell]), axis=0) for cell in unique_cells],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(replicates, dtype=np.float64)
    batch_size = 250
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        samples = rng.integers(
            0, len(unique_cells), size=(stop - start, len(unique_cells))
        )
        denom = np.sum(denominator[samples], axis=1)
        ref_wape = np.mean(
            np.sum(ref_error[samples], axis=1) / np.maximum(denom, 1e-12),
            axis=1,
        )
        cand_wape = np.mean(
            np.sum(cand_error[samples], axis=1) / np.maximum(denom, 1e-12),
            axis=1,
        )
        deltas[start:stop] = cand_wape - ref_wape
    return {
        "metric": "unfiltered macro WAPE",
        "delta_direction": "candidate minus reference; negative favors candidate",
        "replicates": int(replicates),
        "seed": BOOTSTRAP_SEED,
        "cluster_unit": "cell",
        "retained_within_cluster": "all seven target dates and all 24 horizons",
        "cells": int(len(unique_cells)),
        "point_delta": float(macro_wape(bundle, "__candidate__") - macro_wape(bundle, "__reference__"))
        if "__candidate__" in bundle.predictions
        else None,
        "bootstrap_mean": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "probability_negative": float(np.mean(deltas < 0.0)),
    }


def daily_wape_deltas(
    bundle: PredictionBundle,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> list[dict[str, object]]:
    complete = np.all(np.isfinite(bundle.actual), axis=1)
    dates = np.asarray([timestamp.date().isoformat() for timestamp in bundle.timestamps])
    rows = []
    for date_value in sorted(np.unique(dates[complete])):
        selected = complete & (dates == date_value)
        actual = bundle.actual[selected]
        ref = reference[selected]
        cand = candidate[selected]
        denom = np.sum(np.abs(actual), axis=0)
        ref_wape = float(
            np.mean(np.sum(np.abs(actual - ref), axis=0) / np.maximum(denom, 1e-12))
        )
        cand_wape = float(
            np.mean(np.sum(np.abs(actual - cand), axis=0) / np.maximum(denom, 1e-12))
        )
        rows.append(
            {
                "target_date": date_value,
                "reference_wape": ref_wape,
                "candidate_wape": cand_wape,
                "delta_candidate_minus_reference": cand_wape - ref_wape,
            }
        )
    return rows


def comparison_row(
    *,
    bundle: PredictionBundle,
    reference: str,
    candidate: str,
    label: str,
    replicates: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    local = PredictionBundle(
        bundle.label,
        bundle.actual,
        {
            "__reference__": bundle.predictions[reference],
            "__candidate__": bundle.predictions[candidate],
        },
        bundle.cells,
        bundle.timestamps,
        bundle.horizons,
        bundle.mase_scales,
    )
    bootstrap = cell_cluster_wape_bootstrap(
        local,
        local.predictions["__reference__"],
        local.predictions["__candidate__"],
        replicates,
    )
    daily = daily_wape_deltas(
        local,
        local.predictions["__reference__"],
        local.predictions["__candidate__"],
    )
    point_delta = macro_wape(local, "__candidate__") - macro_wape(local, "__reference__")
    row = {
        "comparison": label,
        "reference": reference,
        "candidate": candidate,
        "wape_delta_candidate_minus_reference": point_delta,
        "wape_cell_cluster_ci_low": bootstrap["ci_low"],
        "wape_cell_cluster_ci_high": bootstrap["ci_high"],
        "wape_probability_candidate_better": bootstrap["probability_negative"],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cluster_unit": "cell; all dates and horizons retained",
        "daily_better_count": int(
            sum(float(item["delta_candidate_minus_reference"]) < 0.0 for item in daily)
        ),
        "daily_count": len(daily),
        "daily_delta_min": min(float(item["delta_candidate_minus_reference"]) for item in daily),
        "daily_delta_max": max(float(item["delta_candidate_minus_reference"]) for item in daily),
    }
    return row, [{"comparison": label, **item} for item in daily]


def run(round_cap: int, replicates: int) -> dict[str, object]:
    if replicates != DEFAULT_REPLICATES:
        raise ValueError(f"revision5 requires exactly {DEFAULT_REPLICATES} replicates")
    root = project_root()
    inputs = registered_inputs()
    examples = build_training_backtests(read_traffic(inputs["train"]))
    split = split_examples(examples)
    final_examples = split["final_examples"]
    holdout_examples = split["holdout_examples"]
    parameters = load_parameters(inputs["parameter"])
    weather = load_weather(inputs["weather"])
    baseline, _ = select_baseline_for_inner(split["inner_examples"])
    holdout = build_matrix(holdout_examples, baseline, parameters, weather)
    names = tuple(feature_names(final_examples[0], baseline, parameters, weather))
    full_columns = tuple(np.arange(88, dtype=np.int64) for _ in range(4))
    plain_columns = tuple(feature_columns(names, "plain_lgbm", metric) for metric in range(4))
    traffic_columns = traffic_only_columns(names)
    no_weather = no_weather_columns(names)
    predictions_rows: dict[str, Sequence] = {}
    for method, path, columns in (
        ("plain_lgbm", PLAIN_DIR, plain_columns),
        ("wlcr_full_seed42", FULL_DIR, full_columns),
        ("wlcr_no_weather_83d", NO_WEATHER_DIR, no_weather),
        ("wlcr_traffic_only_73d", TRAFFIC_ONLY_DIR, traffic_columns),
    ):
        rows, _ = predict_boosters(
            load_verified_boosters(root / path), holdout, columns
        )
        predictions_rows[method] = rows
    standard_holdout = build_standard_stat_matrix(holdout_examples, baseline, parameters)
    standard_names = standard_stat_feature_names(
        final_examples[0], parameters.get(final_examples[0].window.cell, {})
    )
    standard_columns = tuple(np.arange(len(standard_names), dtype=np.int64) for _ in range(4))
    standard_method = f"standard_stat_lgbm_175d_roundcap{round_cap}"
    standard_dir = OUTPUT_ROOT / "models" / standard_method
    standard_rows, _ = predict_boosters(
        load_verified_boosters(root / standard_dir), standard_holdout, standard_columns
    )
    predictions_rows[standard_method] = standard_rows
    bundle = bundle_from_examples(
        "revision5_fixed_seven_day_holdout", holdout_examples, predictions_rows
    )
    predictions = dict(bundle.predictions)
    predictions.update(
        load_neural_prediction_arrays(root / DLINEAR_PREDICTIONS, bundle, "dlinear")
    )
    bundle = PredictionBundle(
        bundle.label,
        bundle.actual,
        predictions,
        bundle.cells,
        bundle.timestamps,
        bundle.horizons,
        bundle.mase_scales,
    )
    comparisons = []
    daily_rows = []
    for reference, candidate, label in (
        ("plain_lgbm", "wlcr_traffic_only_73d", "Traffic-only WLCR minus sparse-lag LightGBM"),
        ("plain_lgbm", "wlcr_full_seed42", "WLCR Full minus sparse-lag LightGBM"),
        ("dlinear_seed42", "wlcr_traffic_only_73d", "Traffic-only WLCR minus DLinear seed42"),
        (standard_method, "wlcr_no_weather_83d", f"No-weather WLCR minus Standard-stat LightGBM cap {round_cap}"),
    ):
        row, daily = comparison_row(
            bundle=bundle,
            reference=reference,
            candidate=candidate,
            label=label,
            replicates=replicates,
        )
        comparisons.append(row)
        daily_rows.extend(daily)
    output = root / OUTPUT_ROOT
    write_csv_atomic(output / "revision5_cell_cluster_bootstrap.csv", comparisons)
    write_csv_atomic(output / "revision5_daily_wape_deltas.csv", daily_rows)
    report = {
        "schema_version": 1,
        "experiment_version": "manuscript_revision5_v1",
        "bootstrap_definition": (
            "Sample cells with replacement; retain every available holdout date, "
            "horizon, and complete-target observation belonging to each sampled cell."
        ),
        "temporal_inference_boundary": (
            "The seven daily deltas are descriptive and are not interpreted as a "
            "population-level temporal confidence interval."
        ),
        "comparisons": comparisons,
        "finals_test_opened": False,
    }
    write_json_atomic(output / "revision5_statistics_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-cap", type=int, default=5000)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_REPLICATES)
    args = parser.parse_args()
    run(args.round_cap, args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
