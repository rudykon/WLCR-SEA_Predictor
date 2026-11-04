from __future__ import annotations

"""Run repository-isolated experiments requested by manuscript revision 4.

This script uses only the registered training, parameter, and weather files.
It never reads finals test traffic. The outputs add a covariate-class-matched
standard-statistics LightGBM baseline, a compact WLCR variant, history-length
and missingness audits, two-way cell-by-date uncertainty, and an exploratory
quantile provisioning evaluation.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import time
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from Model.traffic_window_forecasting import BacktestExample, ForecastRow, build_training_backtests, read_traffic
from Model.lightgbm_feature_baseline import MatrixBundle, build_matrix, load_parameters, load_weather
from experiments.lightgbm_experiment_helpers import (
    HISTORY_LENGTHS,
    MISSING_MECHANISMS,
    MISSING_RATES,
    build_standard_stat_matrix,
    compact_columns,
    corrupt_example,
    example_history_array,
    example_with_history_array,
    no_weather_columns,
    standard_stat_feature_names,
    traffic_only_columns,
    truncate_example,
)
from experiments.train_lightgbm_baseline import (
    BOOTSTRAP_SEED,
    EARLY_STOPPING_ROUNDS,
    MAX_BOOST_ROUNDS,
    MIN_BOOST_ROUNDS,
    MODEL_PARAMS,
    configured_gpu_devices,
    feature_columns,
    feature_names,
    matrix_fingerprint,
    predict_boosters,
    sha256_file,
    subset_matrix,
    train_or_load_boosters,
)
from experiments.run_reproducibility_evaluation import (
    METRIC_NAMES,
    PredictionBundle,
    bundle_from_examples,
    cluster_bootstrap_combined_delta,
    forecast_array,
    load_verified_boosters,
    standard_metric_rows,
)
from experiments.run_traffic_only_baseline_evaluation import combined_summary
from experiments.run_seasonal_anchor_ablations import registered_inputs, select_baseline_for_inner


SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "manuscript_revision4_v1"
OUTPUT_ROOT = Path("artifacts/revision4")
MODEL_ROOT = OUTPUT_ROOT / "models"
FULL_DIR = Path("artifacts/revision2/models/fixed_seven_day_holdout/proposed")
PLAIN_DIR = Path("artifacts/revision2/models/fixed_seven_day_holdout/plain_lgbm")
NO_WEATHER_DIR = Path(
    "artifacts/revision2/models/fixed_seven_day_holdout/"
    "seven_day_ablations_v1/no_weather"
)
TRAFFIC_ONLY_DIR = Path("artifacts/revision3/models/wlcr_traffic_only_73d_seed42")
DLINEAR_PREDICTIONS = Path(
    "artifacts/paper_neural_baselines_v1/results/predictions/"
    "dlinear_holdout_predictions.csv.gz"
)
PATCHTST_PREDICTIONS = Path(
    "artifacts/paper_neural_baselines_v1/results/predictions/"
    "patchtst_holdout_predictions.csv.gz"
)
REVISION3_PREDICTIONS = Path("artifacts/revision3/revision3_predictions.csv.gz")
SEED = 42
QUANTILES = (0.50, 0.70, 0.80, 0.90)
PRB_METRICS = (2, 3)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def selection_worker(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    valid_features: np.ndarray,
    valid_targets: np.ndarray,
    columns: np.ndarray,
    metric: int,
    device_id: int,
    result_path: str,
) -> None:
    params = dict(MODEL_PARAMS)
    params["gpu_device_id"] = int(device_id)
    params["num_threads"] = max(1, int(MODEL_PARAMS["num_threads"]) // 4)
    train_mask = np.isfinite(train_targets)
    valid_mask = np.isfinite(valid_targets)
    booster = lgb.train(
        params,
        lgb.Dataset(
            train_features[train_mask][:, columns],
            label=train_targets[train_mask],
            free_raw_data=False,
        ),
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[
            lgb.Dataset(
                valid_features[valid_mask][:, columns],
                label=valid_targets[valid_mask],
                free_raw_data=False,
            )
        ],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    raw_best = int(booster.best_iteration or booster.current_iteration())
    write_json_atomic(
        Path(result_path),
        {
            "metric": int(metric),
            "gpu_device_id": int(device_id),
            "best_iteration": raw_best,
            "selected_rounds": max(MIN_BOOST_ROUNDS, raw_best),
            "trained_iterations": int(booster.current_iteration()),
        },
    )


def select_rounds_parallel(
    train: MatrixBundle,
    valid: MatrixBundle,
    columns: Sequence[np.ndarray],
    label: str,
) -> tuple[tuple[int, ...], tuple[dict[str, object], ...], float]:
    devices = configured_gpu_devices()
    if devices[:4] != [0, 1, 2, 3]:
        raise ValueError("set PAPER_GPU_DEVICES=0,1,2,3")
    scratch = project_root() / OUTPUT_ROOT / "scratch" / label
    scratch.mkdir(parents=True, exist_ok=True)
    paths = [scratch / f"metric_{metric}.{os.getpid()}.json" for metric in range(4)]
    context = mp.get_context("spawn")
    processes = []
    started = time.perf_counter()
    for metric in range(4):
        process = context.Process(
            target=selection_worker,
            args=(
                train.features,
                train.targets[:, metric],
                valid.features,
                valid.targets[:, metric],
                columns[metric],
                metric,
                devices[metric],
                str(paths[metric]),
            ),
        )
        process.start()
        processes.append((metric, process))
    failures = []
    for metric, process in processes:
        process.join()
        if process.exitcode != 0 or not paths[metric].exists():
            failures.append((metric, process.exitcode))
    if failures:
        raise RuntimeError(f"round selection failed for {label}: {failures}")
    diagnostics = tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    for path in paths:
        path.unlink(missing_ok=True)
    try:
        scratch.rmdir()
        scratch.parent.rmdir()
    except OSError:
        pass
    return (
        tuple(int(item["selected_rounds"]) for item in diagnostics),
        diagnostics,
        time.perf_counter() - started,
    )


def train_controlled_variant(
    *,
    label: str,
    fit_matrix: MatrixBundle,
    inner_matrix: MatrixBundle,
    final_matrix: MatrixBundle,
    holdout_matrix: MatrixBundle,
    columns: Sequence[np.ndarray],
    schema: Sequence[str],
    context: Mapping[str, object],
) -> tuple[list[ForecastRow], dict[str, object]]:
    rounds, diagnostics, selection_seconds = select_rounds_parallel(
        fit_matrix, inner_matrix, columns, label
    )
    cache_config = {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "variant": label,
        "training_matrix_sha256": matrix_fingerprint(final_matrix),
        "feature_schema_sha256": canonical_sha256(list(schema)),
        "selected_columns": [[int(value) for value in item] for item in columns],
        "selected_feature_names": [
            [schema[int(value)] for value in item] for item in columns
        ],
        "rounds": [int(value) for value in rounds],
        "model_params": dict(MODEL_PARAMS),
        "seed": SEED,
        "context": dict(context),
        "code_sha256": sha256_file(Path(__file__)),
    }
    boosters, training_seconds, model_bytes = train_or_load_boosters(
        final_matrix,
        columns,
        rounds,
        project_root() / MODEL_ROOT / label,
        cache_config,
    )
    rows, prediction_seconds = predict_boosters(boosters, holdout_matrix, columns)
    return rows, {
        "rounds": list(rounds),
        "round_selection": list(diagnostics),
        "round_selection_seconds": selection_seconds,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "model_bytes": int(model_bytes),
        "cache_config_sha256": canonical_sha256(cache_config),
        "feature_count": int(round(np.mean([len(item) for item in columns]))),
    }


def split_examples(examples: Sequence[BacktestExample]) -> dict[str, object]:
    dates = sorted({example.window.target_start.date() for example in examples})
    if len(dates) != 16:
        raise ValueError(f"expected 16 target dates, found {len(dates)}")
    fit_dates = tuple(dates[:7])
    inner_dates = tuple(dates[7:9])
    final_dates = tuple(dates[:9])
    holdout_dates = tuple(dates[9:])

    def choose(wanted: Sequence[date]) -> list[BacktestExample]:
        selected = set(wanted)
        return [
            example
            for example in examples
            if example.window.target_start.date() in selected
        ]

    payload = {
        "dates": dates,
        "fit_dates": fit_dates,
        "inner_dates": inner_dates,
        "final_dates": final_dates,
        "holdout_dates": holdout_dates,
        "fit_examples": choose(fit_dates),
        "inner_examples": choose(inner_dates),
        "final_examples": choose(final_dates),
        "holdout_examples": choose(holdout_dates),
    }
    counts = tuple(
        len(payload[name])
        for name in ("fit_examples", "inner_examples", "final_examples", "holdout_examples")
    )
    if counts != (5115, 1460, 6575, 5110):
        raise ValueError(f"unexpected split counts: {counts}")
    return payload


def matrix_partition(
    final_matrix: MatrixBundle,
    fit_dates: Sequence[date],
    inner_dates: Sequence[date],
) -> tuple[MatrixBundle, MatrixBundle]:
    target_dates = np.asarray(
        [row.timestamp.date() for row in final_matrix.actuals], dtype=object
    )
    fit_set = set(fit_dates)
    inner_set = set(inner_dates)
    fit_mask = np.asarray([value in fit_set for value in target_dates])
    inner_mask = np.asarray([value in inner_set for value in target_dates])
    if np.any(fit_mask & inner_mask) or not np.all(fit_mask | inner_mask):
        raise ValueError("fit and inner masks do not partition final matrix")
    return subset_matrix(final_matrix, fit_mask), subset_matrix(final_matrix, inner_mask)


def load_neural_prediction_arrays(
    path: Path,
    bundle: PredictionBundle,
    prefix: str,
) -> dict[str, np.ndarray]:
    key_to_index = {
        (str(bundle.cells[index]), bundle.timestamps[index], int(bundle.horizons[index])): index
        for index in range(len(bundle.actual))
    }
    arrays = {
        f"{prefix}_seed42": np.full_like(bundle.actual, np.nan, dtype=np.float64),
        f"{prefix}_ensemble": np.full_like(bundle.actual, np.nan, dtype=np.float64),
    }
    labels = ("ul_active_users", "dl_active_users", "dl_prb", "ul_prb")
    seen = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            timestamp = datetime.fromisoformat(row["target_timestamp"])
            key = (row["cell"], timestamp, int(row["horizon"]) - 1)
            if key not in key_to_index:
                raise ValueError(f"unexpected prediction identity at row {row_number}: {key}")
            if key in seen:
                raise ValueError(f"duplicate prediction identity at row {row_number}: {key}")
            seen.add(key)
            index = key_to_index[key]
            for metric, label in enumerate(labels):
                arrays[f"{prefix}_seed42"][index, metric] = float(
                    row[f"prediction_seed42_{label}"]
                )
                arrays[f"{prefix}_ensemble"][index, metric] = float(
                    row[f"prediction_ensemble_{label}"]
                )
    if len(seen) != len(key_to_index):
        raise ValueError(f"{path} is missing {len(key_to_index) - len(seen)} rows")
    if any(np.any(~np.isfinite(array)) for array in arrays.values()):
        raise ValueError(f"{path} produced non-finite aligned predictions")
    return arrays


def load_revision3_array(
    path: Path,
    bundle: PredictionBundle,
    method: str,
) -> np.ndarray:
    key_to_index = {
        (str(bundle.cells[index]), bundle.timestamps[index], int(bundle.horizons[index])): index
        for index in range(len(bundle.actual))
    }
    output = np.full_like(bundle.actual, np.nan, dtype=np.float64)
    labels = ("ul_active_users", "dl_active_users", "dl_prb", "ul_prb")
    seen = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            key = (
                row["cell"],
                datetime.fromisoformat(row["target_timestamp"]),
                int(row["horizon"]) - 1,
            )
            if key not in key_to_index:
                raise ValueError(f"unexpected revision3 row {row_number}: {key}")
            seen.add(key)
            index = key_to_index[key]
            for metric, label in enumerate(labels):
                output[index, metric] = float(row[f"prediction_{method}_{label}"])
    if len(seen) != len(key_to_index) or np.any(~np.isfinite(output)):
        raise ValueError(f"failed to align revision3 method {method}")
    return output


def method_result_rows(
    bundle: PredictionBundle,
    metadata: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics = standard_metric_rows(bundle)
    macro = {
        str(row["method"]): row
        for row in metrics
        if row["filter"] == "complete_targets_unfiltered"
        and row["indicator"] == "macro_mean"
    }
    rows = []
    for method in metadata:
        task = combined_summary(bundle, method, metadata[method].get("features"))
        standard = macro[method]
        rows.append(
            {
                "method": method,
                "status": metadata[method]["status"],
                "information_set": metadata[method]["information_set"],
                "features": metadata[method].get("features"),
                "unfiltered_wape": standard["wape"],
                "unfiltered_mase": standard["mase"],
                "unfiltered_smape": standard["smape"],
                "unfiltered_mae": standard["mae"],
                "unfiltered_rmse": standard["rmse"],
                "ths_mapeauc": task["mape_auc"],
                "filtered_mean_mape": task["mean_mape"],
                "complete_hours": standard["n_hours"],
                "ths_hours": task["filtered_hours"],
            }
        )
    return rows, metrics


def two_way_wape_bootstrap(
    bundle: PredictionBundle,
    reference: np.ndarray,
    candidate: np.ndarray,
    replicates: int,
) -> dict[str, object]:
    complete = np.all(np.isfinite(bundle.actual), axis=1)
    cells = np.asarray(bundle.cells[complete], dtype=str)
    dates = np.asarray(
        [value.date() for value, keep in zip(bundle.timestamps, complete) if keep],
        dtype=object,
    )
    actual = bundle.actual[complete]
    ref = reference[complete]
    cand = candidate[complete]
    unique_cells = np.unique(cells)
    unique_dates = np.unique(dates)
    cell_index = {value: index for index, value in enumerate(unique_cells)}
    date_index = {value: index for index, value in enumerate(unique_dates)}
    shape = (len(unique_cells), len(unique_dates), 4)
    denominator = np.zeros(shape, dtype=np.float64)
    ref_error = np.zeros(shape, dtype=np.float64)
    cand_error = np.zeros(shape, dtype=np.float64)
    for index in range(len(actual)):
        c = cell_index[cells[index]]
        d = date_index[dates[index]]
        denominator[c, d] += np.abs(actual[index])
        ref_error[c, d] += np.abs(actual[index] - ref[index])
        cand_error[c, d] += np.abs(actual[index] - cand[index])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        cell_counts = np.bincount(
            rng.integers(0, len(unique_cells), size=len(unique_cells)),
            minlength=len(unique_cells),
        )
        date_counts = np.bincount(
            rng.integers(0, len(unique_dates), size=len(unique_dates)),
            minlength=len(unique_dates),
        )
        weights = np.outer(cell_counts, date_counts)[:, :, None]
        denom = np.sum(denominator * weights, axis=(0, 1))
        ref_wape = np.mean(
            np.sum(ref_error * weights, axis=(0, 1)) / np.maximum(denom, 1e-12)
        )
        cand_wape = np.mean(
            np.sum(cand_error * weights, axis=(0, 1)) / np.maximum(denom, 1e-12)
        )
        deltas[replicate] = cand_wape - ref_wape
    return {
        "metric": "unfiltered macro WAPE",
        "delta": "candidate minus reference; negative favors candidate",
        "replicates": int(replicates),
        "seed": BOOTSTRAP_SEED,
        "cluster_units": "cell and target date resampled independently",
        "cells": int(len(unique_cells)),
        "dates": int(len(unique_dates)),
        "bootstrap_mean": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "probability_negative": float(np.mean(deltas < 0.0)),
    }


def compact_metric_summary(bundle: PredictionBundle, method: str) -> dict[str, float]:
    rows = standard_metric_rows(bundle)
    macro = next(
        row
        for row in rows
        if row["method"] == method
        and row["filter"] == "complete_targets_unfiltered"
        and row["indicator"] == "macro_mean"
    )
    task = combined_summary(bundle, method, None)
    return {
        "unfiltered_wape": float(macro["wape"]),
        "unfiltered_smape": float(macro["smape"]),
        "ths_mapeauc": float(task["mape_auc"]),
        "filtered_mean_mape": float(task["mean_mape"]),
    }


def quantile_worker(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    inner_features: np.ndarray,
    inner_targets: np.ndarray,
    final_features: np.ndarray,
    final_targets: np.ndarray,
    holdout_features: np.ndarray,
    columns: np.ndarray,
    metric: int,
    tau: float,
    device_id: int,
    model_path: str,
    prediction_path: str,
    report_path: str,
    config: Mapping[str, object],
) -> None:
    params = dict(MODEL_PARAMS)
    params.update(
        {
            "objective": "quantile",
            "metric": "quantile",
            "alpha": float(tau),
            "gpu_device_id": int(device_id),
            "num_threads": max(1, int(MODEL_PARAMS["num_threads"]) // 4),
        }
    )
    fit_mask = np.isfinite(fit_targets)
    inner_mask = np.isfinite(inner_targets)
    final_mask = np.isfinite(final_targets)
    selector = lgb.train(
        params,
        lgb.Dataset(fit_features[fit_mask][:, columns], label=fit_targets[fit_mask]),
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[
            lgb.Dataset(
                inner_features[inner_mask][:, columns],
                label=inner_targets[inner_mask],
            )
        ],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    raw_best = int(selector.best_iteration or selector.current_iteration())
    rounds = max(MIN_BOOST_ROUNDS, raw_best)
    started = time.perf_counter()
    booster = lgb.train(
        params,
        lgb.Dataset(final_features[final_mask][:, columns], label=final_targets[final_mask]),
        num_boost_round=rounds,
    )
    training_seconds = time.perf_counter() - started
    model = Path(model_path)
    prediction = Path(prediction_path)
    report = Path(report_path)
    model.parent.mkdir(parents=True, exist_ok=True)
    prediction.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model))
    raw = booster.predict(holdout_features[:, columns])
    values = np.maximum(np.expm1(raw), 1e-4).astype(np.float64)
    np.save(prediction, values, allow_pickle=False)
    write_json_atomic(
        report,
        {
            "config": dict(config),
            "metric": int(metric),
            "tau": float(tau),
            "gpu_device_id": int(device_id),
            "best_iteration": raw_best,
            "selected_rounds": rounds,
            "training_seconds": training_seconds,
            "model_size_bytes": model.stat().st_size,
            "model_sha256": sha256_file(model),
            "prediction_size_bytes": prediction.stat().st_size,
            "prediction_sha256": sha256_file(prediction),
        },
    )


def run_quantile_jobs(
    fit_matrix: MatrixBundle,
    inner_matrix: MatrixBundle,
    final_matrix: MatrixBundle,
    holdout_matrix: MatrixBundle,
    columns: Sequence[np.ndarray],
) -> tuple[dict[tuple[float, int], np.ndarray], list[dict[str, object]]]:
    devices = configured_gpu_devices()
    jobs = []
    root = project_root() / MODEL_ROOT / "quantile_no_weather"
    for tau in QUANTILES:
        for metric in PRB_METRICS:
            token = f"tau{int(round(tau * 100)):02d}_m{metric}"
            config = {
                "schema_version": 1,
                "experiment_version": EXPERIMENT_VERSION,
                "tau": float(tau),
                "metric": int(metric),
                "fit_matrix_sha256": matrix_fingerprint(fit_matrix),
                "inner_matrix_sha256": matrix_fingerprint(inner_matrix),
                "final_matrix_sha256": matrix_fingerprint(final_matrix),
                "columns": [int(value) for value in columns[metric]],
                "model_params": {
                    **dict(MODEL_PARAMS),
                    "objective": "quantile",
                    "metric": "quantile",
                    "alpha": float(tau),
                },
                "code_sha256": sha256_file(Path(__file__)),
            }
            model = root / f"{token}.txt"
            prediction = root / f"{token}.npy"
            report = root / f"{token}.json"
            cached = False
            if model.exists() and prediction.exists() and report.exists():
                payload = json.loads(report.read_text(encoding="utf-8"))
                cached = (
                    payload.get("config") == config
                    and payload.get("model_sha256") == sha256_file(model)
                    and payload.get("prediction_sha256") == sha256_file(prediction)
                )
                if not cached:
                    raise RuntimeError(f"quantile cache mismatch: {token}")
            jobs.append(
                {
                    "tau": tau,
                    "metric": metric,
                    "token": token,
                    "config": config,
                    "model": model,
                    "prediction": prediction,
                    "report": report,
                    "cached": cached,
                }
            )
    context = mp.get_context("spawn")
    pending = [job for job in jobs if not job["cached"]]
    for batch_start in range(0, len(pending), 4):
        processes = []
        for offset, job in enumerate(pending[batch_start : batch_start + 4]):
            metric = int(job["metric"])
            process = context.Process(
                target=quantile_worker,
                args=(
                    fit_matrix.features,
                    fit_matrix.targets[:, metric],
                    inner_matrix.features,
                    inner_matrix.targets[:, metric],
                    final_matrix.features,
                    final_matrix.targets[:, metric],
                    holdout_matrix.features,
                    columns[metric],
                    metric,
                    float(job["tau"]),
                    devices[offset],
                    str(job["model"]),
                    str(job["prediction"]),
                    str(job["report"]),
                    job["config"],
                ),
            )
            process.start()
            processes.append((job["token"], process))
        failures = []
        for token, process in processes:
            process.join()
            if process.exitcode != 0:
                failures.append((token, process.exitcode))
        if failures:
            raise RuntimeError(f"quantile training failed: {failures}")
    predictions = {}
    reports = []
    for job in jobs:
        payload = json.loads(Path(job["report"]).read_text(encoding="utf-8"))
        predictions[(float(job["tau"]), int(job["metric"]))] = np.load(
            job["prediction"], allow_pickle=False
        )
        reports.append(payload)
    return predictions, reports


def decision_components(
    actual: np.ndarray,
    action: np.ndarray,
    tau: float,
) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(action)
    y = actual[valid]
    a = action[valid]
    under = np.maximum(y - a, 0.0)
    over = np.maximum(a - y, 0.0)
    scale = max(float(np.mean(np.abs(y))), 1e-12)
    return {
        "n_hours": int(len(y)),
        "coverage": float(np.mean(y <= a)),
        "calibration_error": float(np.mean(y <= a) - tau),
        "shortfall_probability": float(np.mean(y > a)),
        "underprovision_ratio": float(np.sum(under) / max(float(np.sum(y)), 1e-12)),
        "overprovision_ratio": float(np.sum(over) / max(float(np.sum(y)), 1e-12)),
        "normalized_decision_cost": float(
            np.mean(tau * under + (1.0 - tau) * over) / scale
        ),
    }


def tune_point_scales(
    inner_actual: np.ndarray,
    inner_prediction: np.ndarray,
) -> dict[tuple[float, int], float]:
    scales = np.linspace(0.50, 2.00, 151)
    selected = {}
    for tau in QUANTILES:
        for metric in PRB_METRICS:
            y = inner_actual[:, metric]
            p = inner_prediction[:, metric]
            costs = [
                decision_components(y, p * float(scale), tau)["normalized_decision_cost"]
                for scale in scales
            ]
            selected[(tau, metric)] = float(scales[int(np.argmin(costs))])
    return selected


def provisioning_rows(
    *,
    bundle: PredictionBundle,
    point_prediction: np.ndarray,
    quantile_predictions: Mapping[tuple[float, int], np.ndarray],
    point_scales: Mapping[tuple[float, int], float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    per_cell = []
    complete = np.all(np.isfinite(bundle.actual), axis=1)
    cells = np.asarray(bundle.cells)
    for tau in QUANTILES:
        for metric in PRB_METRICS:
            actual = bundle.actual[:, metric]
            policies = {
                "direct_quantile_wlcr": quantile_predictions[(tau, metric)],
                "inner_tuned_scaled_point": point_prediction[:, metric]
                * point_scales[(tau, metric)],
            }
            local = []
            for policy, action in policies.items():
                values = decision_components(actual[complete], action[complete], tau)
                row = {
                    "tau": tau,
                    "indicator": METRIC_NAMES[metric],
                    "policy": policy,
                    "point_scale": (
                        point_scales[(tau, metric)]
                        if policy == "inner_tuned_scaled_point"
                        else None
                    ),
                    **values,
                }
                local.append(row)
                rows.append(row)
                for cell in np.unique(cells[complete]):
                    selected = complete & (cells == cell)
                    cell_values = decision_components(
                        actual[selected], action[selected], tau
                    )
                    per_cell.append(
                        {
                            "tau": tau,
                            "indicator": METRIC_NAMES[metric],
                            "policy": policy,
                            "cell": str(cell),
                            **cell_values,
                        }
                    )
            best = min(float(item["normalized_decision_cost"]) for item in local)
            for item in local:
                item["excess_cost_vs_best_evaluated"] = (
                    float(item["normalized_decision_cost"]) - best
                )
    return rows, per_cell


def rounds_from_manifest(path: Path) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(int(value) for value in payload["cache_config"]["rounds"])


def inference_contract_audit(
    holdout_examples: Sequence[BacktestExample],
    baseline,
    parameters,
    weather,
    boosters,
    columns,
) -> dict[str, object]:
    first = holdout_examples[0]
    other = next(
        example for example in holdout_examples if example.window.cell != first.window.cell
    )
    modified = example_history_array(other)
    finite = np.isfinite(modified)
    modified[finite] = modified[finite] * 10.0 + 1.0
    other_modified = example_with_history_array(other, modified)
    alone = build_matrix([first], baseline, parameters, weather)
    mixed = build_matrix([other_modified, first], baseline, parameters, weather)
    mixed_mask = np.arange(len(mixed.actuals)) >= len(mixed.actuals) - 24
    mixed_tail = subset_matrix(mixed, mixed_mask)
    feature_difference = float(np.max(np.abs(alone.features - mixed_tail.features)))
    alone_rows, _ = predict_boosters(boosters, alone, columns)
    mixed_rows, _ = predict_boosters(boosters, mixed_tail, columns)
    prediction_difference = float(
        np.max(np.abs(forecast_array(alone_rows) - forecast_array(mixed_rows)))
    )
    return {
        "request_cell": first.window.cell,
        "modified_other_cell": other.window.cell,
        "feature_max_abs_difference": feature_difference,
        "prediction_max_abs_difference": prediction_difference,
        "pass": feature_difference == 0.0 and prediction_difference == 0.0,
        "interpretation": "adding and modifying an unrelated request does not change the fixed request features or predictions",
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.bootstrap != 5000:
        raise ValueError("revision4 requires exactly 5,000 bootstrap replicates")
    devices = configured_gpu_devices()
    if devices[:4] != [0, 1, 2, 3]:
        raise ValueError("set PAPER_GPU_DEVICES=0,1,2,3")
    started = time.perf_counter()
    root = project_root()
    output = root / OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    inputs = registered_inputs()
    hashes_before = {name: sha256_file(path) for name, path in inputs.items()}

    training_rows = read_traffic(inputs["train"])
    examples = build_training_backtests(training_rows)
    split = split_examples(examples)
    fit_examples = split["fit_examples"]
    inner_examples = split["inner_examples"]
    final_examples = split["final_examples"]
    holdout_examples = split["holdout_examples"]
    if not all(example.window.target_start.hour == 0 for example in examples):
        raise ValueError("the registered backtests are not all midnight-origin")
    parameters = load_parameters(inputs["parameter"])
    weather = load_weather(inputs["weather"])
    baseline, baseline_report = select_baseline_for_inner(inner_examples)

    matrix_started = time.perf_counter()
    final_matrix = build_matrix(final_examples, baseline, parameters, weather)
    holdout_matrix = build_matrix(holdout_examples, baseline, parameters, weather)
    fit_matrix, inner_matrix = matrix_partition(
        final_matrix, split["fit_dates"], split["inner_dates"]
    )
    names = tuple(feature_names(final_examples[0], baseline, parameters, weather))
    if len(names) != 88:
        raise ValueError(f"expected 88 WLCR features, found {len(names)}")
    full_columns = tuple(np.arange(88, dtype=np.int64) for _ in range(4))
    plain_columns = tuple(
        feature_columns(names, "plain_lgbm", metric) for metric in range(4)
    )
    traffic_columns = traffic_only_columns(names)
    no_weather = no_weather_columns(names)
    compact = compact_columns(names)
    matrix_seconds = time.perf_counter() - matrix_started

    predictions_rows = {}
    runtime = {}
    for method, path, columns in (
        ("plain_lgbm", PLAIN_DIR, plain_columns),
        ("wlcr_full_seed42", FULL_DIR, full_columns),
        ("wlcr_no_weather_83d", NO_WEATHER_DIR, no_weather),
        ("wlcr_traffic_only_73d", TRAFFIC_ONLY_DIR, traffic_columns),
    ):
        boosters = load_verified_boosters(root / path)
        rows, seconds = predict_boosters(boosters, holdout_matrix, columns)
        predictions_rows[method] = rows
        runtime[method] = {
            "model_dir": str(path),
            "prediction_seconds": seconds,
            "model_bytes": sum(
                (root / path / f"metric_{metric}.txt").stat().st_size
                for metric in range(4)
            ),
            "cache_reused": True,
        }

    compact_rows, compact_runtime = train_controlled_variant(
        label="wlcr_compact_55d_seed42",
        fit_matrix=fit_matrix,
        inner_matrix=inner_matrix,
        final_matrix=final_matrix,
        holdout_matrix=holdout_matrix,
        columns=compact,
        schema=names,
        context={
            "history_hours": 336,
            "weather": False,
            "seasonal_reference_feature": False,
            "explicit_missingness_descriptors": False,
            "status": "post-hoc exploratory; requires confirmation on untouched data",
        },
    )
    predictions_rows["wlcr_compact_55d_seed42"] = compact_rows
    runtime["wlcr_compact_55d_seed42"] = compact_runtime

    standard_started = time.perf_counter()
    standard_final = build_standard_stat_matrix(final_examples, baseline, parameters)
    standard_holdout = build_standard_stat_matrix(holdout_examples, baseline, parameters)
    standard_fit, standard_inner = matrix_partition(
        standard_final, split["fit_dates"], split["inner_dates"]
    )
    standard_names = standard_stat_feature_names(
        final_examples[0], parameters.get(final_examples[0].window.cell, {})
    )
    if standard_final.features.shape[1] != len(standard_names):
        raise ValueError("standard-stat schema width mismatch")
    standard_columns = tuple(
        np.arange(len(standard_names), dtype=np.int64) for _ in range(4)
    )
    standard_rows, standard_runtime = train_controlled_variant(
        label="standard_stat_lgbm_175d_seed42",
        fit_matrix=standard_fit,
        inner_matrix=standard_inner,
        final_matrix=standard_final,
        holdout_matrix=standard_holdout,
        columns=standard_columns,
        schema=standard_names,
        context={
            "history_hours": 336,
            "information_classes": "traffic, masks, calendar, static; no weather",
            "representation": "origin-relative lags and conventional rolling statistics",
            "tuning_budget": "one LightGBM configuration plus per-target early stopping",
        },
    )
    predictions_rows["standard_stat_lgbm_175d_seed42"] = standard_rows
    standard_runtime["matrix_seconds"] = time.perf_counter() - standard_started
    runtime["standard_stat_lgbm_175d_seed42"] = standard_runtime

    bundle = bundle_from_examples(
        "revision4_fixed_seven_day_holdout", holdout_examples, predictions_rows
    )
    combined_predictions = dict(bundle.predictions)
    combined_predictions.update(
        load_neural_prediction_arrays(root / DLINEAR_PREDICTIONS, bundle, "dlinear")
    )
    combined_predictions.update(
        load_neural_prediction_arrays(root / PATCHTST_PREDICTIONS, bundle, "patchtst")
    )
    combined_predictions["wlcr_full_ensemble"] = load_revision3_array(
        root / REVISION3_PREDICTIONS, bundle, "wlcr_full_ensemble"
    )
    bundle = PredictionBundle(
        bundle.label,
        bundle.actual,
        combined_predictions,
        bundle.cells,
        bundle.timestamps,
        bundle.horizons,
        bundle.mase_scales,
    )

    metadata = {
        "plain_lgbm": {
            "status": "frozen reference",
            "information_set": "target sparse lags + calendar",
            "features": int(round(np.mean([len(item) for item in plain_columns]))),
        },
        "wlcr_full_seed42": {
            "status": "frozen primary instance",
            "information_set": "traffic + masks + calendar + static + task weather",
            "features": 88,
        },
        "standard_stat_lgbm_175d_seed42": {
            "status": "revision4 controlled baseline",
            "information_set": "traffic + masks + calendar + static; no weather",
            "features": len(standard_names),
        },
        "wlcr_traffic_only_73d": {
            "status": "controlled traffic-only",
            "information_set": "traffic + masks + horizon",
            "features": 73,
        },
        "dlinear_seed42": {
            "status": "controlled single model",
            "information_set": "traffic + masks",
            "features": None,
        },
        "patchtst_seed42": {
            "status": "controlled single model",
            "information_set": "traffic + masks",
            "features": None,
        },
        "wlcr_no_weather_83d": {
            "status": "post-hoc exploratory",
            "information_set": "traffic + masks + calendar + static",
            "features": 83,
        },
        "wlcr_compact_55d_seed42": {
            "status": "post-hoc exploratory",
            "information_set": "traffic + calendar + static; no weather/reference/masks",
            "features": 55,
        },
        "wlcr_full_ensemble": {
            "status": "three-seed ensemble",
            "information_set": "traffic + masks + calendar + static + task weather",
            "features": 88,
        },
        "dlinear_ensemble": {
            "status": "three-seed ensemble",
            "information_set": "traffic + masks",
            "features": None,
        },
    }
    single_methods = (
        "plain_lgbm",
        "wlcr_full_seed42",
        "standard_stat_lgbm_175d_seed42",
        "wlcr_traffic_only_73d",
        "dlinear_seed42",
        "patchtst_seed42",
    )
    exploratory_methods = (
        "wlcr_no_weather_83d",
        "wlcr_compact_55d_seed42",
        "wlcr_full_ensemble",
        "dlinear_ensemble",
    )
    result_rows, all_standard_rows = method_result_rows(bundle, metadata)
    by_method = {row["method"]: row for row in result_rows}
    write_csv_atomic(
        output / "revision4_single_model_results.csv",
        [by_method[method] for method in single_methods],
    )
    write_csv_atomic(
        output / "revision4_exploratory_results.csv",
        [by_method[method] for method in exploratory_methods],
    )
    write_csv_atomic(output / "revision4_standard_metrics.csv", all_standard_rows)

    comparisons = []
    for reference, candidate, label in (
        ("plain_lgbm", "wlcr_full_seed42", "WLCR Full minus sparse-lag LightGBM"),
        ("dlinear_seed42", "wlcr_traffic_only_73d", "Traffic-only WLCR minus DLinear seed42"),
        ("standard_stat_lgbm_175d_seed42", "wlcr_no_weather_83d", "No-weather WLCR minus standard-stat LightGBM"),
    ):
        ths = cluster_bootstrap_combined_delta(
            bundle,
            bundle.predictions[reference],
            bundle.predictions[candidate],
            args.bootstrap,
        )
        wape = two_way_wape_bootstrap(
            bundle,
            bundle.predictions[reference],
            bundle.predictions[candidate],
            args.bootstrap,
        )
        comparisons.append(
            {
                "comparison": label,
                "reference": reference,
                "candidate": candidate,
                "ths_delta_candidate_minus_reference": (
                    by_method[candidate]["ths_mapeauc"]
                    - by_method[reference]["ths_mapeauc"]
                ),
                "ths_cell_cluster_ci_low": ths["ci_low"],
                "ths_cell_cluster_ci_high": ths["ci_high"],
                "wape_delta_candidate_minus_reference": (
                    by_method[candidate]["unfiltered_wape"]
                    - by_method[reference]["unfiltered_wape"]
                ),
                "wape_cell_date_ci_low": wape["ci_low"],
                "wape_cell_date_ci_high": wape["ci_high"],
                "wape_probability_candidate_better": wape["probability_negative"],
                "bootstrap_replicates": args.bootstrap,
            }
        )
    write_csv_atomic(output / "revision4_paired_bootstrap.csv", comparisons)

    history_rows = []
    history_runtime = {}
    for hours in HISTORY_LENGTHS:
        if hours == 336:
            summary = compact_metric_summary(bundle, "wlcr_traffic_only_73d")
            history_runtime[str(hours)] = {"cache_reused": True}
        else:
            truncated_final = [truncate_example(example, hours) for example in final_examples]
            truncated_holdout = [truncate_example(example, hours) for example in holdout_examples]
            local_final = build_matrix(truncated_final, baseline, parameters, weather)
            local_holdout = build_matrix(truncated_holdout, baseline, parameters, weather)
            local_fit, local_inner = matrix_partition(
                local_final, split["fit_dates"], split["inner_dates"]
            )
            method = f"wlcr_traffic_only_{hours}h_seed42"
            rows, local_runtime = train_controlled_variant(
                label=method,
                fit_matrix=local_fit,
                inner_matrix=local_inner,
                final_matrix=local_final,
                holdout_matrix=local_holdout,
                columns=traffic_columns,
                schema=names,
                context={
                    "available_history_hours": hours,
                    "physical_window_hours": 336,
                    "prefix_policy": "earlier traffic values masked as unavailable",
                    "information_set": "traffic + masks + horizon",
                },
            )
            local_bundle = bundle_from_examples(
                f"revision4_history_{hours}h", truncated_holdout, {method: rows}
            )
            summary = compact_metric_summary(local_bundle, method)
            history_runtime[str(hours)] = local_runtime
        history_rows.append(
            {
                "model": "wlcr_traffic_only",
                "history_hours": hours,
                **summary,
            }
        )
    write_csv_atomic(output / "revision4_history_lgbm.csv", history_rows)

    stress_rows = []
    clean_summary = compact_metric_summary(bundle, "wlcr_traffic_only_73d")
    traffic_boosters = load_verified_boosters(root / TRAFFIC_ONLY_DIR)
    for mechanism in MISSING_MECHANISMS:
        for rate in MISSING_RATES:
            if rate == 0.0:
                summary = clean_summary
            else:
                corrupted = [
                    corrupt_example(example, mechanism, rate)
                    for example in holdout_examples
                ]
                corrupted_matrix = build_matrix(corrupted, baseline, parameters, weather)
                rows, _ = predict_boosters(
                    traffic_boosters, corrupted_matrix, traffic_columns
                )
                method = "wlcr_traffic_only_73d"
                local_bundle = bundle_from_examples(
                    f"revision4_missing_{mechanism}_{rate}", corrupted, {method: rows}
                )
                summary = compact_metric_summary(local_bundle, method)
            stress_rows.append(
                {
                    "model": "wlcr_traffic_only_seed42",
                    "mechanism": mechanism,
                    "requested_missing_rate": rate,
                    **summary,
                }
            )
    write_csv_atomic(output / "revision4_missingness_lgbm.csv", stress_rows)

    no_weather_rounds = rounds_from_manifest(root / NO_WEATHER_DIR / "cache_manifest.json")
    point_fit_config = {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "variant": "no_weather_fit_only_for_point_scale_selection",
        "training_matrix_sha256": matrix_fingerprint(fit_matrix),
        "selected_columns": [[int(value) for value in item] for item in no_weather],
        "rounds": list(no_weather_rounds),
        "model_params": dict(MODEL_PARAMS),
        "seed": SEED,
        "purpose": "inner-layer selection of multiplicative point-forecast scale for provisioning",
    }
    point_fit_boosters, point_fit_seconds, point_fit_bytes = train_or_load_boosters(
        fit_matrix,
        no_weather,
        no_weather_rounds,
        root / MODEL_ROOT / "no_weather_fit_only_point",
        point_fit_config,
    )
    inner_point_rows, _ = predict_boosters(
        point_fit_boosters, inner_matrix, no_weather
    )
    inner_point = forecast_array(inner_point_rows)
    inner_actual = np.asarray(
        [
            [np.nan if value is None else float(value) for value in row.metrics]
            for row in inner_matrix.actuals
        ],
        dtype=np.float64,
    )
    point_scales = tune_point_scales(inner_actual, inner_point)
    quantile_predictions, quantile_reports = run_quantile_jobs(
        fit_matrix, inner_matrix, final_matrix, holdout_matrix, no_weather
    )
    provisioning, provisioning_per_cell = provisioning_rows(
        bundle=bundle,
        point_prediction=bundle.predictions["wlcr_no_weather_83d"],
        quantile_predictions=quantile_predictions,
        point_scales=point_scales,
    )
    write_csv_atomic(output / "revision4_provisioning.csv", provisioning)
    write_csv_atomic(
        output / "revision4_provisioning_per_cell.csv", provisioning_per_cell
    )

    with inputs["weather"].open("r", encoding="utf-8-sig", newline="") as handle:
        weather_header = next(csv.reader(handle))
    issuance_fields = [
        value
        for value in weather_header
        if "issu" in value.lower()
        or "forecast_time" in value.lower()
        or "发布时间" in value
        or "预报时刻" in value
    ]
    contract = inference_contract_audit(
        holdout_examples,
        baseline,
        parameters,
        weather,
        load_verified_boosters(root / FULL_DIR),
        full_columns,
    )
    if not contract["pass"]:
        raise ValueError("request-local inference contract audit failed")

    write_csv_atomic(
        output / "standard_stat_feature_schema.csv",
        [
            {
                "index": index,
                "feature": name,
                "information_class": (
                    "traffic_or_mask"
                    if "_m" in name
                    else "static"
                    if name in {"azimuth_sin", "azimuth_cos", "scene_code", "x", "y"}
                    else "calendar_or_horizon"
                ),
            }
            for index, name in enumerate(standard_names)
        ],
    )

    hashes_after = {name: sha256_file(path) for name, path in inputs.items()}
    if hashes_after != hashes_before:
        raise ValueError("registered inputs changed during revision4 run")
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
            "gpu_devices": devices,
            "matrix_seconds": matrix_seconds,
            "methods": runtime,
            "history": history_runtime,
            "point_fit_for_scale_selection": {
                "training_seconds": point_fit_seconds,
                "model_bytes": point_fit_bytes,
                "selected_scales": {
                    f"tau{tau}_{METRIC_NAMES[metric]}": scale
                    for (tau, metric), scale in point_scales.items()
                },
            },
            "quantile_models": quantile_reports,
        },
        "protocol": {
            "forecast_origin": "midnight only",
            "history_hours": 336,
            "horizon_hours": 24,
            "fit_dates": [str(value) for value in split["fit_dates"]],
            "inner_dates": [str(value) for value in split["inner_dates"]],
            "final_fit_dates": [str(value) for value in split["final_dates"]],
            "holdout_dates": [str(value) for value in split["holdout_dates"]],
            "windows": {
                "fit": len(fit_examples),
                "inner": len(inner_examples),
                "final_fit": len(final_examples),
                "holdout": len(holdout_examples),
            },
        },
        "baseline_fairness": {
            "standard_stat_features": len(standard_names),
            "wlcr_no_weather_features": 83,
            "shared_information_classes": "traffic, masks, calendar, static; no weather",
            "same_split": True,
            "same_lightgbm_hyperparameters": True,
            "same_per_target_round_selection_layer": True,
            "representation_difference": (
                "standard-stat uses origin-relative lags and conventional rolling statistics; "
                "WLCR uses horizon-relative multi-scale lags, same-hour summaries, and a seasonal reference feature"
            ),
        },
        "weather_availability_audit": {
            "header": weather_header,
            "issuance_time_fields": issuance_fields,
            "online_availability_established": bool(issuance_fields),
            "manuscript_rule": (
                "calendar-day weather descriptors lack issuance-time evidence and are excluded "
                "from deployment claims; no-weather variants remain exploratory"
            ),
        },
        "inference_contract_audit": contract,
        "metric_reporting": {
            "primary": "complete-target unfiltered WAPE, MASE, sMAPE, MAE, and RMSE",
            "secondary": "Tolerance Hit Score, called MAPEAUC in the original task",
            "independent_unit_for_uncertainty": "cell; additional two-way audit resamples cell and target date",
            "multiplicity": "feature removals and stress panels are exploratory; no familywise claim",
        },
        "data_boundary": {
            "registered_rows": len(training_rows),
            "cells": len({row.cell for row in training_rows}),
            "first_timestamp": min(row.timestamp for row in training_rows).isoformat(sep=" "),
            "last_timestamp": max(row.timestamp for row in training_rows).isoformat(sep=" "),
            "untouched_post_holdout_period_available": False,
            "history_672h_compatible_with_fixed_protocol": False,
            "cross_region_or_cross_season_data_registered": False,
        },
        "registered_inputs": {
            name: {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": hashes_after[name],
            }
            for name, path in inputs.items()
        },
        "outputs": {
            "single_model_results": "artifacts/revision4/revision4_single_model_results.csv",
            "exploratory_results": "artifacts/revision4/revision4_exploratory_results.csv",
            "standard_metrics": "artifacts/revision4/revision4_standard_metrics.csv",
            "paired_bootstrap": "artifacts/revision4/revision4_paired_bootstrap.csv",
            "history_lgbm": "artifacts/revision4/revision4_history_lgbm.csv",
            "missingness_lgbm": "artifacts/revision4/revision4_missingness_lgbm.csv",
            "provisioning": "artifacts/revision4/revision4_provisioning.csv",
            "provisioning_per_cell": "artifacts/revision4/revision4_provisioning_per_cell.csv",
            "standard_stat_schema": "artifacts/revision4/standard_stat_feature_schema.csv",
        },
    }
    write_json_atomic(output / "revision4_report.json", report)

    output_files = [
        output / value
        for value in (
            "revision4_single_model_results.csv",
            "revision4_exploratory_results.csv",
            "revision4_standard_metrics.csv",
            "revision4_paired_bootstrap.csv",
            "revision4_history_lgbm.csv",
            "revision4_missingness_lgbm.csv",
            "revision4_provisioning.csv",
            "revision4_provisioning_per_cell.csv",
            "standard_stat_feature_schema.csv",
            "revision4_report.json",
        )
    ]
    manifest = {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "inputs_unchanged": True,
        "code": {
            "path": "experiments/run_feature_ablation_evaluation.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "common_code": {
            "path": "experiments/lightgbm_experiment_helpers.py",
            "sha256": sha256_file(root / "experiments/lightgbm_experiment_helpers.py"),
        },
        "registered_inputs": report["registered_inputs"],
        "outputs": [
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_files
        ],
    }
    write_json_atomic(output / "manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
