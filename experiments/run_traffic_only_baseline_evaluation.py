from __future__ import annotations

"""Run the additional experiments requested by manuscript revision 3.

The script adds (i) a three-seed WLCR Full prediction ensemble using the
already frozen feature schema and target-specific boosting rounds, (ii) a
traffic-only WLCR variant containing the horizon plus all four 18-feature
traffic blocks, and (iii) paired cell-cluster bootstrap comparisons against
the existing three-seed DLinear predictions.  Only registered training,
parameter, and weather inputs are read; finals test traffic is outside the
allowlist.
"""

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import platform
import time
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from Model.traffic_window_forecasting import BacktestExample, build_training_backtests, read_traffic
from Model.lightgbm_feature_baseline import MatrixBundle, build_matrix, load_parameters, load_weather
from experiments.train_lightgbm_baseline import (
    BOOTSTRAP_SEED,
    EARLY_STOPPING_ROUNDS,
    MAX_BOOST_ROUNDS,
    MIN_BOOST_ROUNDS,
    MODEL_PARAMS,
    configured_gpu_devices,
    feature_names,
    matrix_fingerprint,
    predict_boosters,
    rows_from_array,
    sha256_file,
    subset_matrix,
)
from experiments.run_reproducibility_evaluation import (
    OFFICIAL_THRESHOLDS,
    PredictionBundle,
    bundle_from_examples,
    cluster_bootstrap_combined_delta,
    load_verified_boosters,
    official_mask,
    threshold_score,
)
from experiments.run_seasonal_anchor_ablations import method_summary, registered_inputs, select_baseline_for_inner


SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "manuscript_revision3_v1"
OUTPUT_ROOT = Path("artifacts/revision3")
MODEL_ROOT = OUTPUT_ROOT / "models"
FULL_REFERENCE_DIR = Path("artifacts/revision2/models/fixed_seven_day_holdout/proposed")
DLINEAR_PREDICTIONS = Path(
    "artifacts/paper_neural_baselines_v1/results/predictions/"
    "dlinear_holdout_predictions.csv.gz"
)
STRICT_NESTED_REPORT = Path(
    "artifacts/paper_experiments_gpu4_v2/strict_nested_model_selection.json"
)
SEEDS = (42, 43, 44)
EXPECTED_FULL_SEED42 = 0.7839242949385123
EXPECTED_DLINEAR_ENSEMBLE = 0.786768792376949


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
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def seeded_params(seed: int, device_id: int) -> dict[str, object]:
    params = dict(MODEL_PARAMS)
    params.update(
        {
            "seed": int(seed),
            "feature_fraction_seed": int(seed),
            "bagging_seed": int(seed),
            "data_random_seed": int(seed),
            "gpu_device_id": int(device_id),
            "num_threads": max(1, int(MODEL_PARAMS["num_threads"]) // 4),
        }
    )
    return params


def train_metric_worker(
    features: np.ndarray,
    targets: np.ndarray,
    columns: np.ndarray,
    rounds: int,
    seed: int,
    device_id: int,
    output_path: str,
) -> None:
    mask = np.isfinite(targets)
    booster = lgb.train(
        seeded_params(seed, device_id),
        lgb.Dataset(features[mask][:, columns], label=targets[mask]),
        num_boost_round=int(rounds),
    )
    booster.save_model(output_path)


def select_metric_worker(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    valid_features: np.ndarray,
    valid_targets: np.ndarray,
    columns: np.ndarray,
    metric: int,
    seed: int,
    device_id: int,
    result_path: str,
) -> None:
    train_mask = np.isfinite(train_targets)
    valid_mask = np.isfinite(valid_targets)
    booster = lgb.train(
        seeded_params(seed, device_id),
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
    payload = {
        "metric": int(metric),
        "seed": int(seed),
        "gpu_device_id": int(device_id),
        "best_iteration": raw_best,
        "selected_rounds": max(MIN_BOOST_ROUNDS, raw_best),
        "trained_iterations": int(booster.current_iteration()),
    }
    write_json_atomic(Path(result_path), payload)


def parallel_round_selection(
    train: MatrixBundle,
    valid: MatrixBundle,
    columns: Sequence[np.ndarray],
    seed: int,
) -> tuple[tuple[int, ...], tuple[dict[str, object], ...], float]:
    devices = configured_gpu_devices()
    if devices[:4] != [0, 1, 2, 3]:
        raise ValueError("set PAPER_GPU_DEVICES=0,1,2,3")
    scratch = project_root() / MODEL_ROOT / "traffic_only_selection_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    paths = [scratch / f"metric_{metric}.{os.getpid()}.json" for metric in range(4)]
    context = mp.get_context("spawn")
    started = time.perf_counter()
    processes: list[tuple[int, mp.Process]] = []
    for metric in range(4):
        process = context.Process(
            target=select_metric_worker,
            args=(
                train.features,
                train.targets[:, metric],
                valid.features,
                valid.targets[:, metric],
                columns[metric],
                metric,
                seed,
                devices[metric],
                str(paths[metric]),
            ),
        )
        process.start()
        processes.append((metric, process))
    failures: list[tuple[int, int | None]] = []
    for metric, process in processes:
        process.join()
        if process.exitcode != 0 or not paths[metric].exists():
            failures.append((metric, process.exitcode))
    if failures:
        raise RuntimeError(f"traffic-only round selection failed: {failures}")
    diagnostics = tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    for path in paths:
        path.unlink(missing_ok=True)
    try:
        scratch.rmdir()
    except OSError:
        pass
    rounds = tuple(int(item["selected_rounds"]) for item in diagnostics)
    return rounds, diagnostics, time.perf_counter() - started


def model_manifest(
    *,
    seed: int,
    variant: str,
    matrix: MatrixBundle,
    feature_schema: Sequence[str],
    columns: Sequence[np.ndarray],
    rounds: Sequence[int],
) -> dict[str, object]:
    params = seeded_params(seed, 0)
    params["gpu_device_id"] = "assigned per target from PAPER_GPU_DEVICES"
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "variant": variant,
        "seed": int(seed),
        "training_matrix_sha256": matrix_fingerprint(matrix),
        "feature_schema_sha256": canonical_sha256(list(feature_schema)),
        "selected_feature_names": [
            [feature_schema[int(index)] for index in metric_columns]
            for metric_columns in columns
        ],
        "rounds": [int(value) for value in rounds],
        "model_params": params,
    }


def train_or_load_seeded_boosters(
    *,
    matrix: MatrixBundle,
    columns: Sequence[np.ndarray],
    rounds: Sequence[int],
    seed: int,
    variant: str,
    feature_schema: Sequence[str],
) -> tuple[list[lgb.Booster], float, int, dict[str, object]]:
    cache_dir = project_root() / MODEL_ROOT / f"{variant}_seed{seed}"
    expected = model_manifest(
        seed=seed,
        variant=variant,
        matrix=matrix,
        feature_schema=feature_schema,
        columns=columns,
        rounds=rounds,
    )
    manifest_path = cache_dir / "cache_manifest.json"
    paths = [cache_dir / f"metric_{metric}.txt" for metric in range(4)]
    if manifest_path.exists() or any(path.exists() for path in paths):
        if not manifest_path.exists() or not all(path.exists() for path in paths):
            raise RuntimeError(f"incomplete model cache: {cache_dir}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("cache_config") != expected:
            raise RuntimeError(f"cache configuration mismatch: {cache_dir}")
        for path, record in zip(paths, payload.get("models", [])):
            if path.name != record.get("file"):
                raise RuntimeError(f"cache model order mismatch: {cache_dir}")
            if path.stat().st_size != int(record["size_bytes"]):
                raise RuntimeError(f"cache model size mismatch: {path}")
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"cache model SHA256 mismatch: {path}")
        boosters = [lgb.Booster(model_file=str(path)) for path in paths]
        return boosters, 0.0, sum(path.stat().st_size for path in paths), payload

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = [cache_dir / f".metric_{metric}.{os.getpid()}.tmp" for metric in range(4)]
    devices = configured_gpu_devices()
    context = mp.get_context("spawn")
    started = time.perf_counter()
    processes: list[tuple[int, mp.Process]] = []
    try:
        for metric in range(4):
            process = context.Process(
                target=train_metric_worker,
                args=(
                    matrix.features,
                    matrix.targets[:, metric],
                    columns[metric],
                    int(rounds[metric]),
                    seed,
                    devices[metric],
                    str(temporary[metric]),
                ),
            )
            process.start()
            processes.append((metric, process))
        failures: list[tuple[int, int | None]] = []
        for metric, process in processes:
            process.join()
            if process.exitcode != 0 or not temporary[metric].exists():
                failures.append((metric, process.exitcode))
        if failures:
            raise RuntimeError(f"seeded GPU training failed: {failures}")
        for source, destination in zip(temporary, paths):
            os.replace(source, destination)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "cache_config": expected,
            "cache_config_sha256": canonical_sha256(expected),
            "models": [
                {
                    "file": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in paths
            ],
        }
        write_json_atomic(manifest_path, payload)
    except Exception:
        for path in temporary:
            path.unlink(missing_ok=True)
        raise
    elapsed = time.perf_counter() - started
    boosters = [lgb.Booster(model_file=str(path)) for path in paths]
    return boosters, elapsed, sum(path.stat().st_size for path in paths), payload


def traffic_only_columns(names: Sequence[str]) -> tuple[np.ndarray, ...]:
    selected = np.asarray(
        [index for index, name in enumerate(names) if name == "horizon" or index >= 16],
        dtype=np.int64,
    )
    if len(selected) != 73 or selected.tolist() != [0, *range(16, 88)]:
        raise ValueError(f"unexpected traffic-only schema: {selected.tolist()}")
    return tuple(selected.copy() for _ in range(4))


def load_dlinear_arrays(
    path: Path,
    bundle: PredictionBundle,
) -> dict[str, np.ndarray]:
    expected_header = [
        "cell",
        "target_timestamp",
        "horizon",
        "actual_ul_active_users",
        "actual_dl_active_users",
        "actual_dl_prb",
        "actual_ul_prb",
    ]
    key_to_index: dict[tuple[str, datetime, int], int] = {}
    for index, (cell, timestamp, horizon) in enumerate(
        zip(bundle.cells, bundle.timestamps, bundle.horizons)
    ):
        key = (str(cell), timestamp, int(horizon) + 1)
        if key in key_to_index:
            raise ValueError(f"duplicate holdout identity: {key}")
        key_to_index[key] = index
    arrays = {
        "dlinear_seed42": np.full((len(bundle.actual), 4), np.nan, dtype=np.float64),
        "dlinear_seed43": np.full((len(bundle.actual), 4), np.nan, dtype=np.float64),
        "dlinear_seed44": np.full((len(bundle.actual), 4), np.nan, dtype=np.float64),
        "dlinear_ensemble": np.full((len(bundle.actual), 4), np.nan, dtype=np.float64),
    }
    seen: set[tuple[str, datetime, int]] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or reader.fieldnames[:7] != expected_header:
            raise ValueError("unexpected DLinear prediction schema")
        for row_number, row in enumerate(reader, start=2):
            timestamp = datetime.fromisoformat(row["target_timestamp"])
            key = (row["cell"], timestamp, int(row["horizon"]))
            if key not in key_to_index:
                raise ValueError(
                    f"unexpected DLinear identity at CSV row {row_number}: {key}"
                )
            if key in seen:
                raise ValueError(
                    f"duplicate DLinear identity at CSV row {row_number}: {key}"
                )
            seen.add(key)
            index = key_to_index[key]
            actual = np.asarray(
                [
                    np.nan if row["actual_ul_active_users"] == "" else float(row["actual_ul_active_users"]),
                    np.nan if row["actual_dl_active_users"] == "" else float(row["actual_dl_active_users"]),
                    np.nan if row["actual_dl_prb"] == "" else float(row["actual_dl_prb"]),
                    np.nan if row["actual_ul_prb"] == "" else float(row["actual_ul_prb"]),
                ]
            )
            if not np.allclose(actual, bundle.actual[index], equal_nan=True, atol=2e-6):
                raise ValueError(f"DLinear actual mismatch at CSV row {row_number}")
            for seed in SEEDS:
                arrays[f"dlinear_seed{seed}"][index] = np.asarray(
                    (
                        float(row[f"prediction_seed{seed}_ul_active_users"]),
                        float(row[f"prediction_seed{seed}_dl_active_users"]),
                        float(row[f"prediction_seed{seed}_dl_prb"]),
                        float(row[f"prediction_seed{seed}_ul_prb"]),
                    ),
                    dtype=np.float64,
                )
            arrays["dlinear_ensemble"][index] = np.asarray(
                (
                    float(row["prediction_ensemble_ul_active_users"]),
                    float(row["prediction_ensemble_dl_active_users"]),
                    float(row["prediction_ensemble_dl_prb"]),
                    float(row["prediction_ensemble_ul_prb"]),
                ),
                dtype=np.float64,
            )
    missing = set(key_to_index) - seen
    if missing:
        example = sorted(missing, key=lambda item: (item[0], item[1], item[2]))[0]
        raise ValueError(
            f"DLinear prediction file is missing {len(missing)} rows; first={example}"
        )
    if any(np.any(~np.isfinite(values)) for values in arrays.values()):
        raise ValueError("DLinear predictions contain non-finite values after alignment")
    return arrays


def combined_summary(bundle: PredictionBundle, method: str, features: int | None) -> dict[str, object]:
    mask, thresholds = official_mask(bundle)
    prediction = bundle.predictions[method]
    auc, rates = threshold_score(bundle.actual, prediction, mask, OFFICIAL_THRESHOLDS)
    errors = np.mean(np.abs(bundle.actual[mask] - prediction[mask]) / bundle.actual[mask], axis=1)
    return {
        "method": method,
        "features": features,
        "mape_auc": auc,
        "mean_mape": float(np.mean(errors)),
        "hit_020": rates[0],
        "hit_030": rates[1],
        "hit_040": rates[2],
        "hit_050": rates[3],
        "filtered_hours": int(np.sum(mask)),
        "official_q05": [float(value) for value in thresholds],
    }


def unfiltered_summary(bundle: PredictionBundle, method: str) -> dict[str, object]:
    complete = np.all(np.isfinite(bundle.actual), axis=1)
    actual = bundle.actual[complete]
    prediction = bundle.predictions[method][complete]
    absolute = np.abs(actual - prediction)
    per_indicator_wape = np.sum(absolute, axis=0) / np.maximum(
        np.sum(np.abs(actual), axis=0), 1e-12
    )
    per_indicator_mae = np.mean(absolute, axis=0)
    return {
        "complete_target_hours": int(np.sum(complete)),
        "macro_wape": float(np.mean(per_indicator_wape)),
        "macro_mae": float(np.mean(per_indicator_mae)),
        "per_indicator_wape": [float(value) for value in per_indicator_wape],
        "per_indicator_mae": [float(value) for value in per_indicator_mae],
    }


def prediction_rows(bundle: PredictionBundle) -> list[dict[str, object]]:
    methods = tuple(bundle.predictions)
    rows: list[dict[str, object]] = []
    for index in range(len(bundle.actual)):
        row: dict[str, object] = {
            "cell": str(bundle.cells[index]),
            "target_timestamp": bundle.timestamps[index].isoformat(sep=" "),
            "horizon": int(bundle.horizons[index]) + 1,
        }
        for metric, label in enumerate(
            ("ul_active_users", "dl_active_users", "dl_prb", "ul_prb")
        ):
            row[f"actual_{label}"] = float(bundle.actual[index, metric])
            for method in methods:
                row[f"prediction_{method}_{label}"] = float(
                    bundle.predictions[method][index, metric]
                )
        rows.append(row)
    return rows


def write_gzip_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty prediction table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def window_count_audit(
    training_rows: Sequence[object],
    examples: Sequence[BacktestExample],
    final_dates: Sequence[date],
    holdout_dates: Sequence[date],
) -> dict[str, object]:
    cells = sorted({str(row.cell) for row in training_rows})
    by_date: dict[str, set[str]] = {}
    for example in examples:
        key = str(example.window.target_start.date())
        by_date.setdefault(key, set()).add(example.window.cell)
    final_missing = {
        str(value): sorted(set(cells) - by_date.get(str(value), set()))
        for value in final_dates
    }
    holdout_missing = {
        str(value): sorted(set(cells) - by_date.get(str(value), set()))
        for value in holdout_dates
    }
    strict_payload = json.loads((project_root() / STRICT_NESTED_REPORT).read_text(encoding="utf-8"))
    validation_cells = sorted(
        {
            cell
            for fold in strict_payload["audit"]["fold_plans"]
            for cell in fold["validation_cells"]
        }
    )
    return {
        "registered_cells": len(cells),
        "per_target_date_valid_windows": {
            key: len(value) for key, value in sorted(by_date.items())
        },
        "final_fit_expected_cell_dates": len(cells) * len(final_dates),
        "final_fit_observed_windows": sum(len(by_date[str(value)]) for value in final_dates),
        "final_fit_missing_cell_dates": sum(len(value) for value in final_missing.values()),
        "final_fit_missing_cells_by_date": final_missing,
        "holdout_expected_cell_dates": len(cells) * len(holdout_dates),
        "holdout_observed_windows": sum(len(by_date[str(value)]) for value in holdout_dates),
        "holdout_missing_cell_dates": sum(len(value) for value in holdout_missing.values()),
        "holdout_missing_cells_by_date": holdout_missing,
        "strict_nested_eligible_cells": len(validation_cells),
        "strict_nested_excluded_cells": sorted(set(cells) - set(validation_cells)),
        "valid_window_rule": (
            "a cell-date backtest exists only when all 336 history timestamps and all "
            "24 target timestamps are physically present"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.bootstrap != 5000:
        raise ValueError("revision 3 requires exactly 5,000 bootstrap replicates")
    devices = configured_gpu_devices()
    if devices[:4] != [0, 1, 2, 3]:
        raise ValueError("set PAPER_GPU_DEVICES=0,1,2,3")
    started = time.perf_counter()
    inputs = registered_inputs()
    hashes_before = {name: sha256_file(path) for name, path in inputs.items()}

    training_rows = read_traffic(inputs["train"])
    examples = build_training_backtests(training_rows)
    dates = sorted({example.window.target_start.date() for example in examples})
    if len(dates) != 16:
        raise ValueError(f"expected 16 target dates, found {len(dates)}")
    fit_dates = tuple(dates[:7])
    inner_dates = tuple(dates[7:9])
    final_dates = tuple(dates[:9])
    holdout_dates = tuple(dates[9:])

    def by_dates(wanted: Sequence[date]) -> list[BacktestExample]:
        selected = set(wanted)
        return [
            example
            for example in examples
            if example.window.target_start.date() in selected
        ]

    fit_examples = by_dates(fit_dates)
    inner_examples = by_dates(inner_dates)
    final_examples = by_dates(final_dates)
    holdout_examples = by_dates(holdout_dates)
    observed_counts = (
        len(fit_examples),
        len(inner_examples),
        len(final_examples),
        len(holdout_examples),
    )
    if observed_counts != (5115, 1460, 6575, 5110):
        raise ValueError(f"unexpected fixed-seven-day window counts: {observed_counts}")

    parameters = load_parameters(inputs["parameter"])
    weather = load_weather(inputs["weather"])
    baseline, baseline_report = select_baseline_for_inner(inner_examples)
    if baseline_report["selected"]["name"] != "weekly_median_s097":
        raise ValueError("frozen seasonal selection changed")

    matrix_started = time.perf_counter()
    final_matrix = build_matrix(final_examples, baseline, parameters, weather)
    holdout_matrix = build_matrix(holdout_examples, baseline, parameters, weather)
    target_dates = np.asarray(
        [row.timestamp.date() for row in final_matrix.actuals], dtype=object
    )
    fit_mask = np.asarray([value in set(fit_dates) for value in target_dates])
    inner_mask = np.asarray([value in set(inner_dates) for value in target_dates])
    fit_matrix = subset_matrix(final_matrix, fit_mask)
    inner_matrix = subset_matrix(final_matrix, inner_mask)
    names = tuple(feature_names(final_examples[0], baseline, parameters, weather))
    if len(names) != 88:
        raise ValueError(f"expected 88 features, found {len(names)}")
    matrix_seconds = time.perf_counter() - matrix_started

    full_manifest = json.loads(
        (project_root() / FULL_REFERENCE_DIR / "cache_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    full_rounds = tuple(int(value) for value in full_manifest["cache_config"]["rounds"])
    full_columns = tuple(np.arange(88, dtype=np.int64) for _ in range(4))
    predictions_rows: dict[str, Sequence[object]] = {}
    runtime: dict[str, object] = {}

    seed42_boosters = load_verified_boosters(project_root() / FULL_REFERENCE_DIR)
    seed42_rows, seed42_prediction_seconds = predict_boosters(
        seed42_boosters, holdout_matrix, full_columns
    )
    predictions_rows["wlcr_full_seed42"] = seed42_rows
    runtime["wlcr_full_seed42"] = {
        "source": str(FULL_REFERENCE_DIR),
        "frozen_rounds": list(full_rounds),
        "training_seconds": 0.0,
        "prediction_seconds": seed42_prediction_seconds,
        "cache_reused": True,
    }

    for seed in (43, 44):
        boosters, training_seconds, model_bytes, cache_payload = train_or_load_seeded_boosters(
            matrix=final_matrix,
            columns=full_columns,
            rounds=full_rounds,
            seed=seed,
            variant="wlcr_full",
            feature_schema=names,
        )
        rows, prediction_seconds = predict_boosters(boosters, holdout_matrix, full_columns)
        predictions_rows[f"wlcr_full_seed{seed}"] = rows
        runtime[f"wlcr_full_seed{seed}"] = {
            "frozen_rounds": list(full_rounds),
            "training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds,
            "model_bytes": model_bytes,
            "cache_manifest_sha256": canonical_sha256(cache_payload),
            "cache_reused": training_seconds == 0.0,
        }

    traffic_columns = traffic_only_columns(names)
    traffic_rounds, traffic_selection, traffic_selection_seconds = parallel_round_selection(
        fit_matrix, inner_matrix, traffic_columns, seed=42
    )
    traffic_boosters, traffic_training_seconds, traffic_model_bytes, traffic_cache = (
        train_or_load_seeded_boosters(
            matrix=final_matrix,
            columns=traffic_columns,
            rounds=traffic_rounds,
            seed=42,
            variant="wlcr_traffic_only_73d",
            feature_schema=names,
        )
    )
    traffic_rows, traffic_prediction_seconds = predict_boosters(
        traffic_boosters, holdout_matrix, traffic_columns
    )
    predictions_rows["wlcr_traffic_only_73d"] = traffic_rows
    runtime["wlcr_traffic_only_73d"] = {
        "selected_rounds": list(traffic_rounds),
        "round_selection_diagnostics": list(traffic_selection),
        "round_selection_seconds": traffic_selection_seconds,
        "training_seconds": traffic_training_seconds,
        "prediction_seconds": traffic_prediction_seconds,
        "model_bytes": traffic_model_bytes,
        "cache_manifest_sha256": canonical_sha256(traffic_cache),
        "cache_reused": traffic_training_seconds == 0.0,
    }

    preliminary_bundle = bundle_from_examples(
        "revision3_preliminary", holdout_examples, predictions_rows
    )
    wlcr_seed_arrays = [
        preliminary_bundle.predictions[f"wlcr_full_seed{seed}"] for seed in SEEDS
    ]
    predictions_rows["wlcr_full_ensemble"] = rows_from_array(
        holdout_matrix.baselines, np.mean(wlcr_seed_arrays, axis=0)
    )
    bundle = bundle_from_examples("revision3_holdout", holdout_examples, predictions_rows)

    dlinear_path = (project_root() / DLINEAR_PREDICTIONS).resolve(strict=True)
    dlinear_arrays = load_dlinear_arrays(dlinear_path, bundle)
    combined_predictions = dict(bundle.predictions)
    combined_predictions.update(dlinear_arrays)
    bundle = PredictionBundle(
        bundle.label,
        bundle.actual,
        combined_predictions,
        bundle.cells,
        bundle.timestamps,
        bundle.horizons,
        bundle.mase_scales,
    )

    feature_counts = {
        "wlcr_full_seed42": 88,
        "wlcr_full_seed43": 88,
        "wlcr_full_seed44": 88,
        "wlcr_full_ensemble": 88,
        "wlcr_traffic_only_73d": 73,
        "dlinear_seed42": None,
        "dlinear_seed43": None,
        "dlinear_seed44": None,
        "dlinear_ensemble": None,
    }
    summaries = {
        method: combined_summary(bundle, method, feature_counts[method])
        for method in feature_counts
    }
    if not np.isclose(
        summaries["wlcr_full_seed42"]["mape_auc"], EXPECTED_FULL_SEED42, atol=1e-12
    ):
        raise ValueError("frozen WLCR seed-42 score mismatch")
    if not np.isclose(
        summaries["dlinear_ensemble"]["mape_auc"], EXPECTED_DLINEAR_ENSEMBLE, atol=1e-12
    ):
        raise ValueError("DLinear ensemble score mismatch")

    comparison_rows: list[dict[str, object]] = []

    def add_comparison(reference: str, candidate: str, comparison: str) -> None:
        bootstrap = cluster_bootstrap_combined_delta(
            bundle,
            bundle.predictions[reference],
            bundle.predictions[candidate],
            args.bootstrap,
        )
        comparison_rows.append(
            {
                "comparison": comparison,
                "reference": reference,
                "candidate": candidate,
                "reference_mape_auc": summaries[reference]["mape_auc"],
                "candidate_mape_auc": summaries[candidate]["mape_auc"],
                "delta_candidate_minus_reference": float(
                    summaries[candidate]["mape_auc"] - summaries[reference]["mape_auc"]
                ),
                "bootstrap_mean_delta": bootstrap["mean_delta"],
                "ci_low": bootstrap["ci_low"],
                "ci_high": bootstrap["ci_high"],
                "probability_positive": bootstrap["probability_positive"],
                "bootstrap_replicates": bootstrap["replicates"],
                "cluster_unit": bootstrap["cluster_unit"],
            }
        )

    for seed in SEEDS:
        add_comparison(
            f"dlinear_seed{seed}",
            f"wlcr_full_seed{seed}",
            f"paired independent seed {seed}: WLCR Full - DLinear",
        )
    add_comparison(
        "dlinear_ensemble",
        "wlcr_full_ensemble",
        "three-seed prediction ensemble: WLCR Full - DLinear",
    )
    for seed in SEEDS:
        add_comparison(
            f"dlinear_seed{seed}",
            "wlcr_traffic_only_73d",
            f"traffic-only WLCR - DLinear seed {seed}",
        )
    add_comparison(
        "dlinear_ensemble",
        "wlcr_traffic_only_73d",
        "traffic-only WLCR - DLinear ensemble",
    )

    seed_auc = np.asarray(
        [summaries[f"wlcr_full_seed{seed}"]["mape_auc"] for seed in SEEDS]
    )
    seed_mape = np.asarray(
        [summaries[f"wlcr_full_seed{seed}"]["mean_mape"] for seed in SEEDS]
    )
    unfiltered = {
        method: unfiltered_summary(bundle, method)
        for method in (
            "wlcr_full_seed42",
            "wlcr_full_ensemble",
            "wlcr_traffic_only_73d",
            "dlinear_ensemble",
        )
    }
    mask, thresholds = official_mask(bundle)
    count_audit = window_count_audit(training_rows, examples, final_dates, holdout_dates)

    output = project_root() / OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "revision3_metrics.csv"
    comparisons_path = output / "revision3_paired_bootstrap.csv"
    predictions_path = output / "revision3_predictions.csv.gz"
    report_path = output / "revision3_report.json"
    manifest_path = output / "manifest.json"
    write_csv_atomic(metrics_path, list(summaries.values()))
    write_csv_atomic(comparisons_path, comparison_rows)
    write_gzip_csv_atomic(predictions_path, prediction_rows(bundle))

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "scope_boundary": (
            "Only registered train_data.csv, parameter.csv, weather.csv, verified "
            "revision-2 WLCR caches, and verified local DLinear predictions were read; "
            "data/test_data.csv and preliminary reference traffic were not opened."
        ),
        "protocol": {
            "fit_dates": list(map(str, fit_dates)),
            "inner_dates": list(map(str, inner_dates)),
            "final_fit_dates": list(map(str, final_dates)),
            "holdout_dates": list(map(str, holdout_dates)),
            "fit_windows": len(fit_examples),
            "inner_windows": len(inner_examples),
            "final_fit_windows": len(final_examples),
            "holdout_windows": len(holdout_examples),
            "total_forecast_hours": len(bundle.actual),
            "complete_target_hours": int(np.sum(np.all(np.isfinite(bundle.actual), axis=1))),
            "official_filter_hours": int(np.sum(mask)),
            "official_q05_thresholds": [float(value) for value in thresholds],
            "filter_funnel": "122,640 forecast hours -> 106,248 complete-label hours -> 98,963 retained after pooled 5th-percentile filtering",
        },
        "feature_definitions": {
            "wlcr_full": "all 88 frozen features",
            "wlcr_traffic_only_73d": (
                "horizon plus all four contiguous 18-feature traffic blocks; explicit "
                "calendar, static-cell, and weather covariates are excluded"
            ),
            "target_only_34d_correction": (
                "the earlier target-only schema is not covariate-matched to DLinear because "
                "it retains six calendar/horizon, five static, and five weather features"
            ),
        },
        "seed_policy": {
            "seeds": list(SEEDS),
            "full_feature_schema_frozen": True,
            "full_target_specific_rounds_frozen_from_inner_selection": list(full_rounds),
            "varied_parameters": [
                "seed",
                "feature_fraction_seed",
                "bagging_seed",
                "data_random_seed",
            ],
            "no_best_seed_selection": True,
        },
        "seasonal_selection": baseline_report,
        "summaries": summaries,
        "wlcr_full_seed_aggregate": {
            "mape_auc_mean": float(np.mean(seed_auc)),
            "mape_auc_sample_sd": float(np.std(seed_auc, ddof=1)),
            "mean_mape_mean": float(np.mean(seed_mape)),
            "mean_mape_sample_sd": float(np.std(seed_mape, ddof=1)),
            "prediction_ensemble": summaries["wlcr_full_ensemble"],
        },
        "paired_bootstrap": comparison_rows,
        "unfiltered_complete_target_metrics": unfiltered,
        "window_count_audit": count_audit,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
            "gpu_devices": devices,
            "gpu_assignment": "one target regressor per Tesla P40",
            "matrix_construction_seconds": matrix_seconds,
            "models": runtime,
            "total_seconds_before_write": time.perf_counter() - started,
        },
        "registered_input_sha256_before": hashes_before,
        "dlinear_prediction": {
            "path": str(dlinear_path.relative_to(project_root())),
            "size_bytes": dlinear_path.stat().st_size,
            "sha256": sha256_file(dlinear_path),
        },
    }
    write_json_atomic(report_path, report)

    hashes_after = {name: sha256_file(path) for name, path in inputs.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("registered inputs changed during revision-3 experiments")
    outputs = [metrics_path, comparisons_path, predictions_path, report_path]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "inputs_unchanged": True,
        "registered_inputs": {
            name: {
                "path": str(path.relative_to(project_root())),
                "size_bytes": path.stat().st_size,
                "sha256": hashes_after[name],
            }
            for name, path in inputs.items()
        },
        "code": {
            "path": str(Path(__file__).resolve().relative_to(project_root())),
            "sha256": sha256_file(Path(__file__)),
        },
        "outputs": [
            {
                "path": str(path.relative_to(project_root())),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
        "model_manifests": [
            {
                "path": str(path.relative_to(project_root())),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted((project_root() / MODEL_ROOT).glob("*/cache_manifest.json"))
        ],
    }
    write_json_atomic(manifest_path, manifest)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "status": "ok",
                "wlcr_full_seed_aggregate": report["wlcr_full_seed_aggregate"],
                "traffic_only": report["summaries"]["wlcr_traffic_only_73d"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
