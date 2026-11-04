from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from Model.traffic_window_forecasting import (
    BacktestExample,
    BaselineConfig,
    ForecastRow,
    OUTPUT_FLOOR,
    TestWindow,
    TrafficRow,
    build_training_backtests,
    mape_auc,
    read_traffic,
    seasonal_forecast,
)
from Model.lightgbm_feature_baseline import (
    MatrixBundle,
    build_feature_row,
    build_matrix,
    load_parameters,
    load_weather,
)


EXPERIMENT_VERSION = "paper_experiments_v2"
DEFAULT_OUTPUT = "artifacts/paper_experiments_gpu4_v2"
MODEL_SEED = 42
BOOTSTRAP_SEED = 42
CORRUPTION_SEED = 42
# Backward-compatible alias for callers that imported SEED from the V1 script.
SEED = MODEL_SEED
BOOTSTRAP_REPLICATES = 5000
MAX_BOOST_ROUNDS = 1500
EARLY_STOPPING_ROUNDS = 60
MIN_BOOST_ROUNDS = 50
CACHE_MANIFEST_NAME = "cache_manifest.json"
LEGACY_OUTPUT_NAMES = {"paper_experiments", "paper_experiments_gpu4"}
MODEL_CACHE_NAMESPACE = "train_lightgbm_baseline_v2"
MODEL_PARAMS = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.04,
    "num_leaves": 48,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.88,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 0.20,
    "max_bin": 127,
    "verbosity": -1,
    "seed": MODEL_SEED,
    "feature_fraction_seed": MODEL_SEED,
    "bagging_seed": MODEL_SEED,
    "data_random_seed": MODEL_SEED,
    "num_threads": 32,
    "force_col_wise": True,
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
}


@dataclass(frozen=True)
class RoundSelection:
    rounds: tuple[int, ...]
    diagnostics: tuple[dict[str, object], ...]


def configured_gpu_devices() -> list[int]:
    raw = os.environ.get("PAPER_GPU_DEVICES", "0")
    devices = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not devices:
        raise ValueError("PAPER_GPU_DEVICES must contain at least one GPU index")
    return devices


def _train_metric_worker(features, targets, columns, rounds, path: str, device_id: int) -> None:
    params = dict(MODEL_PARAMS)
    params["gpu_device_id"] = int(device_id)
    params["num_threads"] = max(1, int(MODEL_PARAMS["num_threads"]) // len(configured_gpu_devices()))
    mask = np.isfinite(targets)
    booster = lgb.train(
        params,
        lgb.Dataset(features[mask][:, columns], label=targets[mask]),
        num_boost_round=int(rounds),
    )
    booster.save_model(path)


def score_dict(actuals: Sequence[TrafficRow], predictions: Sequence[ForecastRow]) -> dict[str, object]:
    score = mape_auc(actuals, predictions)
    standard = standard_metrics(actuals, predictions)
    return {
        "samples": score.samples,
        "mape_auc": score.mape_auc,
        "mean_mape": score.mean_mape,
        "rates": list(score.rates),
        "mae": standard["mae"],
        "wape": standard["wape"],
        "smape": standard["smape"],
    }


def complete_arrays(
    actuals: Sequence[TrafficRow], predictions: Sequence[ForecastRow]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual_rows = []
    prediction_rows = []
    cells = []
    for actual, prediction in zip(actuals, predictions):
        if any(value is None for value in actual.metrics):
            continue
        actual_rows.append([float(value) for value in actual.metrics])
        prediction_rows.append([float(value) for value in prediction.metrics])
        cells.append(actual.cell)
    return (
        np.asarray(actual_rows, dtype=np.float64),
        np.asarray(prediction_rows, dtype=np.float64),
        np.asarray(cells),
    )


def official_filter_mask(actual: np.ndarray) -> np.ndarray:
    thresholds = np.quantile(actual, 0.05, axis=0, method="linear")
    mask = np.all(actual >= thresholds, axis=1)
    if np.any(actual[mask] <= 0.0):
        raise ValueError("non-positive actual survived the official filter")
    return mask


def standard_metrics(
    actuals: Sequence[TrafficRow], predictions: Sequence[ForecastRow]
) -> dict[str, list[float]]:
    actual, prediction, _ = complete_arrays(actuals, predictions)
    if not len(actual):
        return {"mae": [0.0] * 4, "wape": [0.0] * 4, "smape": [0.0] * 4}
    mask = official_filter_mask(actual)
    actual = actual[mask]
    prediction = prediction[mask]
    absolute = np.abs(actual - prediction)
    mae = np.mean(absolute, axis=0)
    wape = np.sum(absolute, axis=0) / np.maximum(np.sum(np.abs(actual), axis=0), 1e-12)
    smape = np.mean(2.0 * absolute / np.maximum(np.abs(actual) + np.abs(prediction), 1e-12), axis=0)
    return {
        "mae": [float(value) for value in mae],
        "wape": [float(value) for value in wape],
        "smape": [float(value) for value in smape],
    }


def baseline_rows(examples: Sequence[BacktestExample], config: BaselineConfig) -> tuple[list[TrafficRow], list[ForecastRow]]:
    actuals: list[TrafficRow] = []
    predictions: list[ForecastRow] = []
    for example in examples:
        actuals.extend(example.actuals)
        predictions.extend(seasonal_forecast(example.window, config))
    return actuals, predictions


def rows_from_array(templates: Sequence[ForecastRow], values: np.ndarray) -> list[ForecastRow]:
    return [
        ForecastRow(
            template.timestamp,
            template.cell,
            tuple(max(float(value), OUTPUT_FLOOR) for value in values[index]),
        )
        for index, template in enumerate(templates)
    ]


def feature_names(example: BacktestExample, baseline_config: BaselineConfig, parameters, weather) -> list[str]:
    baseline = seasonal_forecast(example.window, baseline_config)[0]
    names, _ = build_feature_row(
        example.window,
        0,
        baseline,
        parameters.get(example.window.cell, {}),
        weather.get(baseline.timestamp.strftime("%Y%m%d"), {}),
    )
    return names


def feature_columns(names: Sequence[str], variant: str, metric: int) -> np.ndarray:
    static = {"azimuth_sin", "azimuth_cos", "scene_code", "x", "y"}
    temporal = {
        "horizon",
        "target_hour_sin",
        "target_hour_cos",
        "target_dow_sin",
        "target_dow_cos",
        "is_weekend",
    }
    selected = []
    for index, name in enumerate(names):
        keep = True
        if variant == "no_baseline" and name.startswith("baseline_"):
            keep = False
        elif variant == "no_missingness" and ("_mask_" in name or name.startswith("missing_ratio_")):
            keep = False
        elif variant == "no_static" and name in static:
            keep = False
        elif variant == "no_weather" and name.startswith("weather_"):
            keep = False
        elif variant == "target_only":
            keep = name in temporal or name in static or name.startswith("weather_") or name.endswith(f"_m{metric}")
        elif variant == "plain_lgbm":
            keep = name in temporal or name.startswith(f"lag1_m{metric}") or name.startswith(f"lag7_m{metric}") or name.startswith(f"lag14_m{metric}")
        if keep:
            selected.append(index)
    if not selected:
        raise ValueError(f"variant {variant} selected no features")
    return np.asarray(selected, dtype=np.int64)


def select_rounds_with_diagnostics(
    train: MatrixBundle,
    valid: MatrixBundle,
    columns: Sequence[np.ndarray],
    *,
    max_rounds: int = MAX_BOOST_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> RoundSelection:
    if max_rounds < MIN_BOOST_ROUNDS:
        raise ValueError(f"max_rounds must be at least {MIN_BOOST_ROUNDS}")
    if early_stopping_rounds <= 0:
        raise ValueError("early_stopping_rounds must be positive")
    devices = configured_gpu_devices()
    rounds: list[int] = []
    diagnostics: list[dict[str, object]] = []
    for metric in range(4):
        train_mask = np.isfinite(train.targets[:, metric])
        valid_mask = np.isfinite(valid.targets[:, metric])
        params = dict(MODEL_PARAMS)
        params["gpu_device_id"] = devices[metric % len(devices)]
        booster = lgb.train(
            params,
            lgb.Dataset(
                train.features[train_mask][:, columns[metric]],
                label=train.targets[train_mask, metric],
                free_raw_data=False,
            ),
            num_boost_round=max_rounds,
            valid_sets=[
                lgb.Dataset(
                    valid.features[valid_mask][:, columns[metric]],
                    label=valid.targets[valid_mask, metric],
                    free_raw_data=False,
                )
            ],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        raw_best = int(booster.best_iteration or booster.current_iteration())
        trained_iterations = int(booster.current_iteration())
        selected = max(MIN_BOOST_ROUNDS, raw_best)
        rounds.append(selected)
        diagnostics.append(
            {
                "metric": metric,
                "selection_metric": "development_log_target_l1",
                "best_iteration": raw_best,
                "selected_rounds": selected,
                "trained_iterations": trained_iterations,
                "max_rounds": max_rounds,
                "early_stopping_rounds": early_stopping_rounds,
                "hit_max_rounds": trained_iterations >= max_rounds,
                "best_iteration_at_max": raw_best >= max_rounds,
                "gpu_device_id": params["gpu_device_id"],
            }
        )
    return RoundSelection(tuple(rounds), tuple(diagnostics))


def select_model_rounds(
    train: MatrixBundle,
    valid: MatrixBundle,
    columns_by_model: Mapping[str, Sequence[np.ndarray]],
    *,
    max_rounds: int = MAX_BOOST_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> dict[str, RoundSelection]:
    return {
        name: select_rounds_with_diagnostics(
            train,
            valid,
            columns,
            max_rounds=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
        )
        for name, columns in columns_by_model.items()
    }


def select_rounds(
    train: MatrixBundle,
    valid: MatrixBundle,
    columns: Sequence[np.ndarray],
) -> list[int]:
    return list(select_rounds_with_diagnostics(train, valid, columns).rounds)


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def payload_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_fingerprint(matrix: MatrixBundle) -> str:
    digest = hashlib.sha256()
    for label, value in (("features", matrix.features), ("targets", matrix.targets)):
        if value is None:
            digest.update(f"{label}:none".encode("utf-8"))
            continue
        array = np.ascontiguousarray(value)
        digest.update(label.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    for row in matrix.actuals:
        digest.update(row.cell.encode("utf-8"))
        digest.update(b"\0")
        digest.update(row.timestamp.isoformat().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def default_cache_config(
    train: MatrixBundle,
    columns: Sequence[np.ndarray],
    rounds: Sequence[int],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "experiment_version": EXPERIMENT_VERSION,
        "training_matrix_sha256": matrix_fingerprint(train),
        "selected_columns": [[int(value) for value in item] for item in columns],
        "rounds": [int(value) for value in rounds],
        "model_params": dict(MODEL_PARAMS),
        "model_seed": MODEL_SEED,
        "code_sha256": sha256_file(Path(__file__)),
    }


def build_cache_config(
    experiment_context: Mapping[str, object],
    *,
    variant: str,
    training_matrix_sha256: str,
    feature_names_: Sequence[str],
    columns: Sequence[np.ndarray],
    rounds: Sequence[int],
) -> dict[str, object]:
    selected_names = [
        [feature_names_[int(index)] for index in metric_columns]
        for metric_columns in columns
    ]
    return {
        "schema_version": 2,
        "experiment_version": EXPERIMENT_VERSION,
        "variant": variant,
        "training_matrix_sha256": training_matrix_sha256,
        "feature_schema_sha256": payload_sha256(list(feature_names_)),
        "selected_feature_names": selected_names,
        "selected_columns": [[int(value) for value in item] for item in columns],
        "rounds": [int(value) for value in rounds],
        "model_params": dict(MODEL_PARAMS),
        "model_seed": MODEL_SEED,
        "experiment_context_sha256": payload_sha256(experiment_context),
        "experiment_context": dict(experiment_context),
    }


def _model_paths(cache_dir: Path) -> list[Path]:
    return [cache_dir / f"metric_{metric}.txt" for metric in range(4)]


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_cache_manifest(
    cache_dir: Path,
    cache_config: Mapping[str, object],
    model_paths: Sequence[Path],
) -> None:
    payload = {
        "schema_version": 2,
        "cache_config": dict(cache_config),
        "cache_config_sha256": payload_sha256(cache_config),
        "models": [
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in model_paths
        ],
    }
    _write_json_atomic(cache_dir / CACHE_MANIFEST_NAME, payload)


def validated_cached_model_paths(
    cache_dir: Path,
    expected_config: Mapping[str, object],
) -> list[Path] | None:
    paths = _model_paths(cache_dir)
    manifest_path = cache_dir / CACHE_MANIFEST_NAME
    existing_models = [path.exists() for path in paths]
    if not any(existing_models) and not manifest_path.exists():
        return None
    if not all(existing_models) or not manifest_path.exists():
        raise RuntimeError(
            f"incomplete or legacy model cache at {cache_dir}; "
            "use a fresh V2 output directory"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise RuntimeError(f"unsupported cache manifest at {manifest_path}")
    if payload.get("cache_config") != dict(expected_config):
        raise RuntimeError(
            f"cache configuration mismatch at {cache_dir}; "
            "use a fresh V2 output directory"
        )
    if payload.get("cache_config_sha256") != payload_sha256(expected_config):
        raise RuntimeError(f"cache configuration hash mismatch at {cache_dir}")
    records = payload.get("models")
    if not isinstance(records, list) or [
        item.get("file") for item in records
    ] != [path.name for path in paths]:
        raise RuntimeError(f"cache model inventory mismatch at {cache_dir}")
    for path, record in zip(paths, records):
        if path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"cached model size mismatch: {path}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"cached model SHA256 mismatch: {path}")
    return paths


def model_cache_inventory(model_root: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for manifest_path in sorted(model_root.rglob(CACHE_MANIFEST_NAME)):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 2:
            raise RuntimeError(f"unsupported cache manifest at {manifest_path}")
        models = payload.get("models")
        if not isinstance(models, list) or len(models) != 4:
            raise RuntimeError(f"invalid model inventory at {manifest_path}")
        inventory.append(
            {
                "cache": str(manifest_path.parent.relative_to(model_root)),
                "manifest_sha256": sha256_file(manifest_path),
                "cache_config_sha256": payload.get("cache_config_sha256"),
                "models": models,
            }
        )
    return inventory


def train_or_load_boosters(
    train: MatrixBundle,
    columns: Sequence[np.ndarray],
    rounds: Sequence[int],
    cache_dir: Path,
    cache_config: Mapping[str, object] | None = None,
) -> tuple[list[lgb.Booster], float, int]:
    expected_config = dict(
        cache_config or default_cache_config(train, columns, rounds)
    )
    cached_paths = validated_cached_model_paths(cache_dir, expected_config)
    if cached_paths is not None:
        boosters = [lgb.Booster(model_file=str(path)) for path in cached_paths]
        return (
            boosters,
            0.0,
            sum(path.stat().st_size for path in cached_paths),
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _model_paths(cache_dir)
    temporary_paths = [
        cache_dir / f".metric_{metric}.{os.getpid()}.tmp"
        for metric in range(4)
    ]
    started = time.perf_counter()
    devices = configured_gpu_devices()
    context = mp.get_context("spawn")
    try:
        for batch_start in range(0, 4, len(devices)):
            processes = []
            batch_metrics = range(batch_start, min(batch_start + len(devices), 4))
            for device_offset, metric in enumerate(batch_metrics):
                process = context.Process(
                    target=_train_metric_worker,
                    args=(
                        train.features,
                        train.targets[:, metric],
                        columns[metric],
                        int(rounds[metric]),
                        str(temporary_paths[metric]),
                        devices[device_offset],
                    ),
                )
                process.start()
                processes.append((metric, process))
            failures = []
            for metric, process in processes:
                process.join()
                if (
                    process.exitcode != 0
                    or not temporary_paths[metric].exists()
                ):
                    failures.append((metric, process.exitcode))
            if failures:
                details = ", ".join(
                    f"metric {metric} exit {exitcode}"
                    for metric, exitcode in failures
                )
                raise RuntimeError(f"GPU training failed: {details}")
        for temporary, path in zip(temporary_paths, paths):
            temporary.replace(path)
        write_cache_manifest(cache_dir, expected_config, paths)
    except Exception:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise
    boosters = [lgb.Booster(model_file=str(path)) for path in paths]
    elapsed = time.perf_counter() - started
    return boosters, elapsed, sum(path.stat().st_size for path in paths)


def predict_boosters(
    boosters: Sequence[lgb.Booster], matrix: MatrixBundle, columns: Sequence[np.ndarray]
) -> tuple[list[ForecastRow], float]:
    started = time.perf_counter()
    values = np.empty((len(matrix.actuals), 4), dtype=np.float64)
    for metric, booster in enumerate(boosters):
        values[:, metric] = np.maximum(
            np.expm1(booster.predict(matrix.features[:, columns[metric]])),
            OUTPUT_FLOOR,
        )
    elapsed = time.perf_counter() - started
    return rows_from_array(matrix.baselines, values), elapsed


def train_ridge(
    train: MatrixBundle,
    columns: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    x = np.asarray(train.features[:, columns], dtype=np.float64)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std < 1e-8] = 1.0
    x = (x - mean) / std
    x = np.column_stack([np.ones(len(x)), x])
    coefficients = []
    for metric in range(4):
        mask = np.isfinite(train.targets[:, metric])
        xm = x[mask]
        ym = train.targets[mask, metric]
        penalty = np.eye(xm.shape[1], dtype=np.float64) * alpha
        penalty[0, 0] = 0.0
        coefficients.append(np.linalg.solve(xm.T @ xm + penalty, xm.T @ ym))
    return {"mean": mean, "std": std, "coef": np.asarray(coefficients)}


def predict_ridge(model: dict[str, np.ndarray], matrix: MatrixBundle, columns: np.ndarray) -> list[ForecastRow]:
    x = np.asarray(matrix.features[:, columns], dtype=np.float64)
    x = (x - model["mean"]) / model["std"]
    x = np.column_stack([np.ones(len(x)), x])
    values = np.maximum(np.expm1(x @ model["coef"].T), OUTPUT_FLOOR)
    return rows_from_array(matrix.baselines, values)


def subset_matrix(matrix: MatrixBundle, mask: np.ndarray) -> MatrixBundle:
    return MatrixBundle(
        matrix.features[mask],
        None if matrix.targets is None else matrix.targets[mask],
        tuple(row for row, keep in zip(matrix.actuals, mask) if keep),
        tuple(row for row, keep in zip(matrix.baselines, mask) if keep),
    )


def cell_fold(cell: str) -> int:
    return hashlib.sha256(cell.encode("utf-8")).digest()[0] % 5


def cluster_bootstrap(
    actuals: Sequence[TrafficRow],
    reference: Sequence[ForecastRow],
    proposed: Sequence[ForecastRow],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    actual, reference_values, cells = complete_arrays(actuals, reference)
    _, proposed_values, _ = complete_arrays(actuals, proposed)
    mask = official_filter_mask(actual)
    actual = actual[mask]
    reference_values = reference_values[mask]
    proposed_values = proposed_values[mask]
    cells = cells[mask]
    reference_error = np.mean(np.abs(actual - reference_values) / actual, axis=1)
    proposed_error = np.mean(np.abs(actual - proposed_values) / actual, axis=1)
    limits = np.asarray([0.2, 0.3, 0.4, 0.5])
    reference_hits = np.sum(reference_error[:, None] < limits[None, :], axis=1)
    proposed_hits = np.sum(proposed_error[:, None] < limits[None, :], axis=1)
    unique_cells = np.unique(cells)
    reference_sum = np.asarray([reference_hits[cells == cell].sum() for cell in unique_cells], dtype=np.float64)
    proposed_sum = np.asarray([proposed_hits[cells == cell].sum() for cell in unique_cells], dtype=np.float64)
    counts = np.asarray([np.sum(cells == cell) * 4 for cell in unique_cells], dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(unique_cells), size=len(unique_cells))
        denominator = counts[sample].sum()
        deltas[index] = (proposed_sum[sample].sum() - reference_sum[sample].sum()) / denominator
    return {
        "replicates": float(replicates),
        "seed": int(seed),
        "seed_policy": "numpy.random.default_rng",
        "mean_delta": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "probability_positive": float(np.mean(deltas > 0.0)),
    }


def corruption_seed(
    cell: str,
    target_start,
    mode: str,
    severity: float | int,
    *,
    base_seed: int = CORRUPTION_SEED,
) -> int:
    seed_text = (
        f"{EXPERIMENT_VERSION}|base_seed={base_seed}|cell={cell}|"
        f"target_start={target_start.isoformat()}|mode={mode}|severity={severity}"
    )
    return int.from_bytes(
        hashlib.sha256(seed_text.encode("utf-8")).digest()[:8],
        "little",
    )


def corrupt_examples(
    examples: Sequence[BacktestExample],
    mode: str,
    severity: float | int,
    *,
    base_seed: int = CORRUPTION_SEED,
) -> list[BacktestExample]:
    corrupted = []
    for example in examples:
        rows = list(example.window.rows)
        seed = corruption_seed(
            example.window.cell,
            example.window.target_start,
            mode,
            severity,
            base_seed=base_seed,
        )
        rng = np.random.default_rng(seed)
        if mode == "random":
            count = int(round(len(rows) * float(severity)))
            indices = rng.choice(len(rows), size=count, replace=False)
        elif mode == "block":
            length = int(severity)
            start = int(rng.integers(0, len(rows) - length + 1))
            indices = np.arange(start, start + length)
        else:
            raise ValueError(f"unknown corruption mode: {mode}")
        for index in indices:
            rows[int(index)] = replace(rows[int(index)], metrics=(None, None, None, None))
        window = replace(example.window, rows=tuple(rows))
        corrupted.append(BacktestExample(window, example.actuals))
    return corrupted


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result(name: str, protocol: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "method": name,
        "protocol": protocol,
        "mape_auc": payload["mape_auc"],
        "mean_mape": payload["mean_mape"],
        "hit_020": payload["rates"][0],
        "hit_030": payload["rates"][1],
        "hit_040": payload["rates"][2],
        "hit_050": payload["rates"][3],
        "samples": payload["samples"],
    }


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_registered_input(requested: str | Path, relative: str) -> Path:
    """Return exactly one registered paper input and reject every other path."""
    root = project_root()
    requested_path = Path(requested)
    if not requested_path.is_absolute():
        requested_path = root / requested_path
    requested_path = requested_path.resolve(strict=False)
    expected = (root / relative).resolve(strict=True)
    if requested_path != expected:
        raise ValueError(
            f"paper experiments require registered input {relative}: "
            f"{requested_path}"
        )
    if not requested_path.is_file():
        raise ValueError(f"registered input is unavailable: {relative}")
    return requested_path


def resolve_v2_output(requested: str | Path) -> Path:
    root = project_root()
    output = Path(requested)
    if not output.is_absolute():
        output = root / output
    output = output.resolve(strict=False)
    artifacts_root = (root / "artifacts").resolve(strict=True)
    if not output.is_relative_to(artifacts_root):
        raise ValueError("paper V2 output must stay under artifacts/")
    relative = output.relative_to(artifacts_root)
    if output.name in LEGACY_OUTPUT_NAMES or not any(
        "v2" in part.lower() for part in relative.parts
    ):
        raise ValueError(
            "refusing to overwrite a legacy paper experiment directory; "
            "choose a fresh V2 output path"
        )
    return output


def train_lightgbm_baseline_model_root(output: Path) -> Path:
    """Keep this script's caches disjoint from select_lightgbm_model.py caches."""
    return output / "models" / MODEL_CACHE_NAMESPACE


def build_experiment_context(
    train_path: Path,
    parameter_path: Path,
    weather_path: Path,
    feature_names_: Sequence[str],
    *,
    max_rounds: int,
    early_stopping_rounds: int,
) -> dict[str, object]:
    root = project_root()
    input_paths = {
        "train": train_path,
        "parameter": parameter_path,
        "weather": weather_path,
    }
    code_paths = {
        "train_lightgbm_baseline": Path(__file__).resolve(),
        "traffic_window_forecasting": (root / "Model/traffic_window_forecasting.py").resolve(strict=True),
        "lightgbm_feature_baseline": (
            root / "Model/lightgbm_feature_baseline.py"
        ).resolve(strict=True),
    }
    return {
        "schema_version": 2,
        "experiment_version": EXPERIMENT_VERSION,
        "model_cache_namespace": MODEL_CACHE_NAMESPACE,
        "inputs": {
            name: {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
        "code": {
            name: {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            }
            for name, path in code_paths.items()
        },
        "feature_schema_sha256": payload_sha256(list(feature_names_)),
        "feature_count": len(feature_names_),
        "model_params": dict(MODEL_PARAMS),
        "round_selection": {
            "metric": "development_log_target_l1",
            "max_rounds": int(max_rounds),
            "early_stopping_rounds": int(early_stopping_rounds),
            "minimum_rounds": MIN_BOOST_ROUNDS,
        },
        "seeds": {
            "model": MODEL_SEED,
            "bootstrap": BOOTSTRAP_SEED,
            "corruption_base": CORRUPTION_SEED,
            "corruption_derivation": (
                "little-endian uint64 from SHA256("
                "experiment_version, base_seed, cell, target_start, mode, severity)"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
            "gpu_devices": configured_gpu_devices(),
        },
    }


def dated_result_rows(
    name: str,
    protocol: str,
    actuals: Sequence[TrafficRow],
    predictions: Sequence[ForecastRow],
) -> list[dict[str, object]]:
    if len(actuals) != len(predictions):
        raise ValueError("actual and prediction lengths differ")
    rows: list[dict[str, object]] = []
    dates = sorted({actual.timestamp.date() for actual in actuals})
    for target_date in dates:
        indices = [
            index
            for index, actual in enumerate(actuals)
            if actual.timestamp.date() == target_date
        ]
        payload = score_dict(
            [actuals[index] for index in indices],
            [predictions[index] for index in indices],
        )
        rows.append(
            {
                "date": str(target_date),
                **flatten_result(name, protocol, payload),
            }
        )
    return rows


def robustness_result_row(
    mode: str,
    severity: float | int,
    payloads: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    required = {
        "robust_seasonal",
        "plain_lgbm",
        "no_missingness",
        "proposed",
    }
    if set(payloads) != required:
        raise ValueError(
            f"robustness payload methods must be {sorted(required)}"
        )
    baseline_auc = float(payloads["robust_seasonal"]["mape_auc"])
    row: dict[str, object] = {
        "mode": mode,
        "severity": severity,
        "corruption_base_seed": CORRUPTION_SEED,
        "baseline_mape_auc": baseline_auc,
    }
    for method in (
        "robust_seasonal",
        "plain_lgbm",
        "no_missingness",
        "proposed",
    ):
        payload = payloads[method]
        auc = float(payload["mape_auc"])
        row[f"{method}_mape_auc"] = auc
        row[f"{method}_mean_mape"] = payload["mean_mape"]
        row[f"{method}_gain_vs_robust_seasonal"] = auc - baseline_auc
    # Preserve the two V1 columns used by the existing plotting script.
    row["proposed_mape_auc"] = payloads["proposed"]["mape_auc"]
    row["proposed_mean_mape"] = payloads["proposed"]["mean_mape"]
    row["gain"] = float(payloads["proposed"]["mape_auc"]) - baseline_auc
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-oriented traffic forecasting experiments")
    parser.add_argument("--train", default="data/train_data.csv")
    parser.add_argument("--parameter", default="data/parameter.csv")
    parser.add_argument("--weather", default="data/weather.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument(
        "--max-boost-rounds",
        type=int,
        default=MAX_BOOST_ROUNDS,
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=EARLY_STOPPING_ROUNDS,
    )
    args = parser.parse_args()
    if args.max_boost_rounds < MAX_BOOST_ROUNDS:
        raise ValueError(
            f"V2 max boosting rounds must be at least {MAX_BOOST_ROUNDS}"
        )
    if args.early_stopping_rounds <= 0:
        raise ValueError("early stopping rounds must be positive")

    train_path = resolve_registered_input(args.train, "data/train_data.csv")
    parameter_path = resolve_registered_input(
        args.parameter, "data/parameter.csv"
    )
    weather_path = resolve_registered_input(args.weather, "data/weather.csv")
    output = resolve_v2_output(args.output)
    model_root = train_lightgbm_baseline_model_root(output)
    output.mkdir(parents=True, exist_ok=True)

    training_rows = read_traffic(train_path)
    examples = build_training_backtests(training_rows)
    dates = sorted({example.window.target_start.date() for example in examples})
    if len(dates) < 16:
        raise ValueError(f"expected at least 16 backtest dates, found {len(dates)}")
    fit_dates = set(dates[:7])
    inner_dates = set(dates[7:10])
    development_dates = set(dates[10:13])
    lock_dates = set(dates[13:16])

    def by_dates(selected) -> list[BacktestExample]:
        return [example for example in examples if example.window.target_start.date() in selected]

    fit_inner = by_dates(fit_dates | inner_dates)
    development = by_dates(development_dates)
    prelock = by_dates(fit_dates | inner_dates | development_dates)
    lock = by_dates(lock_dates)
    parameters = load_parameters(parameter_path)
    weather = load_weather(weather_path)
    baseline_config = BaselineConfig("weekly_median_s097", (0.0, 0.7, 0.2, 0.1, 0.0, 0.0), (0.97,) * 4)

    fit_inner_matrix = build_matrix(fit_inner, baseline_config, parameters, weather)
    development_matrix = build_matrix(development, baseline_config, parameters, weather)
    prelock_matrix = build_matrix(prelock, baseline_config, parameters, weather)
    lock_matrix = build_matrix(lock, baseline_config, parameters, weather)
    names = feature_names(examples[0], baseline_config, parameters, weather)
    full_columns = [feature_columns(names, "full", metric) for metric in range(4)]
    plain_columns = [
        feature_columns(names, "plain_lgbm", metric) for metric in range(4)
    ]
    experiment_context = build_experiment_context(
        train_path,
        parameter_path,
        weather_path,
        names,
        max_rounds=args.max_boost_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    round_selections = select_model_rounds(
        fit_inner_matrix,
        development_matrix,
        {
            "proposed": full_columns,
            "plain_lgbm": plain_columns,
        },
        max_rounds=args.max_boost_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    proposed_rounds = list(round_selections["proposed"].rounds)
    plain_rounds = list(round_selections["plain_lgbm"].rounds)
    round_selection_payload = {
        "schema_version": 2,
        "selection_metric": "development_log_target_l1",
        "models": {
            name: {
                "rounds": list(selection.rounds),
                "diagnostics": list(selection.diagnostics),
            }
            for name, selection in round_selections.items()
        },
    }
    _write_json_atomic(
        output / "round_selection.json",
        round_selection_payload,
    )
    prelock_matrix_sha256 = matrix_fingerprint(prelock_matrix)
    experiment_manifest = {
        **experiment_context,
        "context_sha256": payload_sha256(experiment_context),
        "prelock_matrix_sha256": prelock_matrix_sha256,
        "round_selection": round_selection_payload,
    }
    _write_json_atomic(
        output / "experiment_manifest.json",
        experiment_manifest,
    )

    report: dict[str, object] = {
        "schema_version": 2,
        "experiment_version": EXPERIMENT_VERSION,
        "seeds": experiment_context["seeds"],
        "experiment_context_sha256": payload_sha256(experiment_context),
        "compute": {
            "learner": "LightGBM OpenCL GPU",
            "gpu_devices": configured_gpu_devices(),
            "parallelism": "one target metric per GPU",
            "model_cache_namespace": MODEL_CACHE_NAMESPACE,
        },
        "data": {
            "training_rows": len(training_rows),
            "cells": len({row.cell for row in training_rows}),
            "backtest_windows": len(examples),
            "feature_count": len(names),
        },
        "splits": {
            "fit": [str(value) for value in sorted(fit_dates)],
            "inner": [str(value) for value in sorted(inner_dates)],
            "development": [str(value) for value in sorted(development_dates)],
            "lockbox": [str(value) for value in sorted(lock_dates)],
        },
        "round_selection": round_selection_payload,
    }

    main_results: list[dict[str, object]] = []
    main_results_by_date: list[dict[str, object]] = []
    method_predictions: dict[str, list[ForecastRow]] = {}
    lock_actuals = list(lock_matrix.actuals)
    baseline_configs = {
        "last_day": BaselineConfig("last_day", (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        "last_week": BaselineConfig("last_week", (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        "weekly_80_20": BaselineConfig("weekly_80_20", (0.0, 0.8, 0.2, 0.0, 0.0, 0.0)),
        "robust_seasonal": baseline_config,
    }
    for name, config in baseline_configs.items():
        actuals, predictions = baseline_rows(lock, config)
        payload = score_dict(actuals, predictions)
        method_predictions[name] = predictions
        main_results.append(flatten_result(name, "temporal_lockbox", payload))
        main_results_by_date.extend(
            dated_result_rows(
                name,
                "temporal_lockbox",
                actuals,
                predictions,
            )
        )

    ridge_columns = np.arange(len(names), dtype=np.int64)
    ridge_candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        ridge = train_ridge(fit_inner_matrix, ridge_columns, alpha)
        predictions = predict_ridge(ridge, development_matrix, ridge_columns)
        ridge_candidates.append((score_dict(development_matrix.actuals, predictions)["mape_auc"], alpha))
    ridge_alpha = max(ridge_candidates)[1]
    ridge_started = time.perf_counter()
    ridge = train_ridge(prelock_matrix, ridge_columns, ridge_alpha)
    ridge_training_seconds = time.perf_counter() - ridge_started
    ridge_predictions = predict_ridge(ridge, lock_matrix, ridge_columns)
    method_predictions["ridge"] = ridge_predictions
    main_results.append(flatten_result("ridge", "temporal_lockbox", score_dict(lock_actuals, ridge_predictions)))
    main_results_by_date.extend(
        dated_result_rows(
            "ridge",
            "temporal_lockbox",
            lock_actuals,
            ridge_predictions,
        )
    )

    plain_boosters, plain_train_seconds, plain_bytes = train_or_load_boosters(
        prelock_matrix,
        plain_columns,
        plain_rounds,
        model_root / "plain_lgbm",
        build_cache_config(
            experiment_context,
            variant="plain_lgbm",
            training_matrix_sha256=prelock_matrix_sha256,
            feature_names_=names,
            columns=plain_columns,
            rounds=plain_rounds,
        ),
    )
    plain_predictions, plain_predict_seconds = predict_boosters(plain_boosters, lock_matrix, plain_columns)
    method_predictions["plain_lgbm"] = plain_predictions
    main_results.append(flatten_result("plain_lgbm", "temporal_lockbox", score_dict(lock_actuals, plain_predictions)))
    main_results_by_date.extend(
        dated_result_rows(
            "plain_lgbm",
            "temporal_lockbox",
            lock_actuals,
            plain_predictions,
        )
    )

    full_boosters, full_train_seconds, full_bytes = train_or_load_boosters(
        prelock_matrix,
        full_columns,
        proposed_rounds,
        model_root / "full",
        build_cache_config(
            experiment_context,
            variant="full",
            training_matrix_sha256=prelock_matrix_sha256,
            feature_names_=names,
            columns=full_columns,
            rounds=proposed_rounds,
        ),
    )
    full_predictions, full_predict_seconds = predict_boosters(full_boosters, lock_matrix, full_columns)
    method_predictions["proposed"] = full_predictions
    full_payload = score_dict(lock_actuals, full_predictions)
    main_results.append(flatten_result("proposed", "temporal_lockbox", full_payload))
    main_results_by_date.extend(
        dated_result_rows(
            "proposed",
            "temporal_lockbox",
            lock_actuals,
            full_predictions,
        )
    )

    report["bootstrap"] = {
        "reference": "plain_lgbm",
        "reference_policy": "predeclared primary learned baseline",
        **cluster_bootstrap(
            lock_actuals,
            plain_predictions,
            full_predictions,
            replicates=args.bootstrap,
            seed=BOOTSTRAP_SEED,
        ),
    }
    report["efficiency"] = {
        "ridge_training_seconds": ridge_training_seconds,
        "plain_lgbm_training_seconds": plain_train_seconds,
        "plain_lgbm_prediction_seconds": plain_predict_seconds,
        "plain_lgbm_model_bytes": plain_bytes,
        "proposed_training_seconds": full_train_seconds,
        "proposed_prediction_seconds": full_predict_seconds,
        "proposed_ms_per_window": 1000.0 * full_predict_seconds / len(lock),
        "proposed_model_bytes": full_bytes,
    }

    ablation_results = []
    ablation_boosters: dict[str, list[lgb.Booster]] = {}
    ablation_columns: dict[str, list[np.ndarray]] = {}
    for variant in ("no_baseline", "no_missingness", "no_static", "no_weather", "target_only"):
        columns = [feature_columns(names, variant, metric) for metric in range(4)]
        boosters, training_seconds, model_bytes = train_or_load_boosters(
            prelock_matrix,
            columns,
            proposed_rounds,
            model_root / variant,
            build_cache_config(
                experiment_context,
                variant=variant,
                training_matrix_sha256=prelock_matrix_sha256,
                feature_names_=names,
                columns=columns,
                rounds=proposed_rounds,
            ),
        )
        ablation_boosters[variant] = boosters
        ablation_columns[variant] = columns
        predictions, prediction_seconds = predict_boosters(boosters, lock_matrix, columns)
        payload = score_dict(lock_actuals, predictions)
        ablation_results.append(
            {
                "variant": variant,
                "features": int(sum(len(value) for value in columns) / 4),
                "mape_auc": payload["mape_auc"],
                "delta_vs_full": float(payload["mape_auc"]) - float(full_payload["mape_auc"]),
                "mean_mape": payload["mean_mape"],
                "training_seconds": training_seconds,
                "prediction_seconds": prediction_seconds,
                "model_bytes": model_bytes,
            }
        )
    ablation_results.append(
        {
            "variant": "full",
            "features": len(names),
            "mape_auc": full_payload["mape_auc"],
            "delta_vs_full": 0.0,
            "mean_mape": full_payload["mean_mape"],
            "training_seconds": full_train_seconds,
            "prediction_seconds": full_predict_seconds,
            "model_bytes": full_bytes,
        }
    )

    prelock_cells = np.asarray([row.cell for row in prelock_matrix.actuals])
    lock_cells = np.asarray([row.cell for row in lock_matrix.actuals])
    unseen_predictions: list[ForecastRow | None] = [None] * len(lock_matrix.actuals)
    fold_results = []
    for fold in range(5):
        train_mask = np.asarray([cell_fold(cell) != fold for cell in prelock_cells])
        valid_mask = np.asarray([cell_fold(cell) == fold for cell in lock_cells])
        train_fold = subset_matrix(prelock_matrix, train_mask)
        valid_fold = subset_matrix(lock_matrix, valid_mask)
        train_fold_sha256 = matrix_fingerprint(train_fold)
        boosters, training_seconds, _ = train_or_load_boosters(
            train_fold,
            full_columns,
            proposed_rounds,
            model_root / f"cell_fold_{fold}",
            build_cache_config(
                experiment_context,
                variant=f"cell_fold_{fold}",
                training_matrix_sha256=train_fold_sha256,
                feature_names_=names,
                columns=full_columns,
                rounds=proposed_rounds,
            ),
        )
        predictions, _ = predict_boosters(boosters, valid_fold, full_columns)
        cursor = 0
        for index, keep in enumerate(valid_mask):
            if keep:
                unseen_predictions[index] = predictions[cursor]
                cursor += 1
        fold_payload = score_dict(valid_fold.actuals, predictions)
        fold_results.append(
            {
                "fold": fold,
                "validation_cells": len(set(row.cell for row in valid_fold.actuals)),
                "mape_auc": fold_payload["mape_auc"],
                "mean_mape": fold_payload["mean_mape"],
                "training_seconds": training_seconds,
            }
        )
    if any(value is None for value in unseen_predictions):
        raise RuntimeError("cell-disjoint predictions are incomplete")
    unseen = [value for value in unseen_predictions if value is not None]
    unseen_payload = score_dict(lock_actuals, unseen)
    main_results.append(flatten_result("proposed", "cell_disjoint_lockbox", unseen_payload))
    main_results_by_date.extend(
        dated_result_rows(
            "proposed",
            "cell_disjoint_lockbox",
            lock_actuals,
            unseen,
        )
    )
    report["cell_disjoint_folds"] = fold_results

    robustness_results = []
    no_missingness_boosters = ablation_boosters["no_missingness"]
    no_missingness_columns = ablation_columns["no_missingness"]
    for mode, severities in (("random", (0.10, 0.20, 0.30)), ("block", (24, 48, 96))):
        for severity in severities:
            # Build the corrupted examples exactly once so every method receives
            # the same deterministic history mask.
            corrupted = corrupt_examples(
                lock,
                mode,
                severity,
                base_seed=CORRUPTION_SEED,
            )
            corrupted_matrix = build_matrix(corrupted, baseline_config, parameters, weather)
            proposed_predictions, _ = predict_boosters(
                full_boosters,
                corrupted_matrix,
                full_columns,
            )
            plain_corrupted_predictions, _ = predict_boosters(
                plain_boosters,
                corrupted_matrix,
                plain_columns,
            )
            no_missingness_predictions, _ = predict_boosters(
                no_missingness_boosters,
                corrupted_matrix,
                no_missingness_columns,
            )
            baseline_actuals, corrupted_baseline = baseline_rows(corrupted, baseline_config)
            robustness_results.append(
                robustness_result_row(
                    mode,
                    severity,
                    {
                        "robust_seasonal": score_dict(
                            baseline_actuals,
                            corrupted_baseline,
                        ),
                        "plain_lgbm": score_dict(
                            corrupted_matrix.actuals,
                            plain_corrupted_predictions,
                        ),
                        "no_missingness": score_dict(
                            corrupted_matrix.actuals,
                            no_missingness_predictions,
                        ),
                        "proposed": score_dict(
                            corrupted_matrix.actuals,
                            proposed_predictions,
                        ),
                    },
                )
            )

    report["main_results"] = main_results
    report["main_results_by_date"] = main_results_by_date
    report["ablations"] = ablation_results
    report["robustness"] = robustness_results
    report["ridge_alpha"] = ridge_alpha
    cache_inventory = model_cache_inventory(model_root)
    if not cache_inventory:
        raise RuntimeError("successful V2 run produced no model cache manifests")
    cache_inventory_sha256 = payload_sha256(cache_inventory)
    experiment_manifest = {
        **experiment_manifest,
        "model_cache_inventory": cache_inventory,
        "model_cache_inventory_sha256": cache_inventory_sha256,
    }
    _write_json_atomic(
        output / "experiment_manifest.json",
        experiment_manifest,
    )
    report["model_cache_inventory_sha256"] = cache_inventory_sha256
    report["experiment_manifest_sha256"] = sha256_file(
        output / "experiment_manifest.json"
    )
    write_csv(output / "main_results.csv", main_results)
    write_csv(output / "main_results_by_date.csv", main_results_by_date)
    write_csv(output / "ablation_results.csv", ablation_results)
    write_csv(output / "robustness_results.csv", robustness_results)
    (output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
