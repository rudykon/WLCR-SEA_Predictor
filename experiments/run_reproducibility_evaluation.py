from __future__ import annotations

"""Additional, leakage-aware evidence requested by the second paper review.

The script reads only the registered training, parameter, and weather files.  It
reuses the frozen V2 lockbox models where possible and writes all new evidence
under ``artifacts/revision2``.  Finals test traffic is never opened.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from Model.traffic_window_forecasting import (
    BacktestExample,
    BaselineConfig,
    ForecastRow,
    TrafficRow,
    baseline_candidates,
    build_training_backtests,
    read_traffic,
    seasonal_forecast,
)
from Model.lightgbm_feature_baseline import build_matrix, load_parameters, load_weather
from experiments.train_lightgbm_baseline import (
    BOOTSTRAP_SEED,
    EARLY_STOPPING_ROUNDS,
    MAX_BOOST_ROUNDS,
    MODEL_PARAMS,
    baseline_rows,
    feature_columns,
    feature_names,
    predict_boosters,
    predict_ridge,
    score_dict,
    select_model_rounds,
    sha256_file,
    subset_matrix,
    train_or_load_boosters,
    train_ridge,
)


SCHEMA_VERSION = 1
REVISION2_VERSION = "revision2_evidence_v1"
DEFAULT_FROZEN = "artifacts/paper_experiments_gpu4_v2"
DEFAULT_OUTPUT = "artifacts/revision2"
OFFICIAL_THRESHOLDS = np.asarray((0.2, 0.3, 0.4, 0.5), dtype=np.float64)
THRESHOLD_SETS: dict[str, tuple[float, ...]] = {
    "lower_shift": (0.15, 0.25, 0.35, 0.45),
    "official": (0.20, 0.30, 0.40, 0.50),
    "upper_shift": (0.25, 0.35, 0.45, 0.55),
}
METRIC_NAMES = (
    "ul_active_users",
    "dl_active_users",
    "dl_prb",
    "ul_prb",
)
REGISTERED_INPUTS = {
    "train": (
        "data/train_data.csv",
        "d274407a3db51ba4871851ab447bcc75202bb567337464d85ea280662f3bf1da",
    ),
    "parameter": (
        "data/parameter.csv",
        "d8e02302042e4fd91945a59a53c0c8d730a18f4c6c7b08344a4c8389a866cd77",
    ),
    "weather": (
        "data/weather.csv",
        "92a2d55c44d69e6bcae3001c20ee7a0034e2035423b41a299d2922d17c280a44",
    ),
}
FROZEN_VARIANTS = (
    "no_baseline",
    "no_missingness",
    "no_static",
    "no_weather",
    "target_only",
)


@dataclass(frozen=True)
class PredictionBundle:
    label: str
    actual: np.ndarray
    predictions: Mapping[str, np.ndarray]
    cells: np.ndarray
    timestamps: tuple[datetime, ...]
    horizons: np.ndarray
    mase_scales: np.ndarray

    def subset(self, mask: np.ndarray, label: str | None = None) -> "PredictionBundle":
        return PredictionBundle(
            label or self.label,
            self.actual[mask],
            {name: values[mask] for name, values in self.predictions.items()},
            self.cells[mask],
            tuple(value for value, keep in zip(self.timestamps, mask) if keep),
            self.horizons[mask],
            self.mase_scales[mask],
        )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_registered_inputs() -> dict[str, Path]:
    root = project_root()
    resolved: dict[str, Path] = {}
    for name, (relative, expected_sha256) in REGISTERED_INPUTS.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"invalid registered input: {relative}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"registered input SHA256 mismatch for {relative}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        resolved[name] = path
    return resolved


def resolve_frozen(path_text: str) -> Path:
    root = project_root()
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    expected = (root / DEFAULT_FROZEN).resolve(strict=True)
    if path != expected:
        raise ValueError(f"revision-2 analysis requires frozen artifacts at {DEFAULT_FROZEN}")
    return path


def resolve_output(path_text: str) -> Path:
    root = project_root()
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    allowed = (root / DEFAULT_OUTPUT).resolve(strict=False)
    if path != allowed:
        raise ValueError(f"revision-2 outputs must use {DEFAULT_OUTPUT}")
    return path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def canonical_array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def forecast_array(rows: Sequence[ForecastRow]) -> np.ndarray:
    return np.asarray([row.metrics for row in rows], dtype=np.float64)


def history_mase_scale(example: BacktestExample) -> np.ndarray:
    history = np.asarray(
        [
            [np.nan if value is None else float(value) for value in row.metrics]
            for row in example.window.rows
        ],
        dtype=np.float64,
    )
    if history.shape != (336, 4):
        raise ValueError(f"unexpected history shape: {history.shape}")
    differences = np.abs(history[168:] - history[:-168])
    scales = np.full(4, np.nan, dtype=np.float64)
    for metric in range(4):
        finite = np.isfinite(differences[:, metric])
        if np.any(finite):
            value = float(np.mean(differences[finite, metric]))
            if value > 1e-12:
                scales[metric] = value
    return scales


def example_metadata(
    examples: Sequence[BacktestExample],
) -> tuple[np.ndarray, np.ndarray, tuple[datetime, ...], np.ndarray, np.ndarray]:
    actual_rows: list[list[float]] = []
    cells: list[str] = []
    timestamps: list[datetime] = []
    horizons: list[int] = []
    scales: list[np.ndarray] = []
    for example in examples:
        scale = history_mase_scale(example)
        for horizon, row in enumerate(example.actuals):
            if row.cell != example.window.cell:
                raise ValueError("actual row cell does not match its history window")
            actual_rows.append(
                [np.nan if value is None else float(value) for value in row.metrics]
            )
            cells.append(row.cell)
            timestamps.append(row.timestamp)
            horizons.append(horizon)
            scales.append(scale)
    return (
        np.asarray(actual_rows, dtype=np.float64),
        np.asarray(cells),
        tuple(timestamps),
        np.asarray(horizons, dtype=np.int16),
        np.asarray(scales, dtype=np.float64),
    )


def bundle_from_examples(
    label: str,
    examples: Sequence[BacktestExample],
    predictions: Mapping[str, Sequence[ForecastRow]],
) -> PredictionBundle:
    actual, cells, timestamps, horizons, scales = example_metadata(examples)
    arrays: dict[str, np.ndarray] = {}
    for name, rows in predictions.items():
        if len(rows) != len(actual):
            raise ValueError(f"{label}/{name} prediction length mismatch")
        for index, row in enumerate(rows):
            if row.cell != cells[index] or row.timestamp != timestamps[index]:
                raise ValueError(f"{label}/{name} prediction identity mismatch at row {index}")
        arrays[name] = forecast_array(rows)
    return PredictionBundle(label, actual, arrays, cells, timestamps, horizons, scales)


def concatenate_bundles(label: str, bundles: Sequence[PredictionBundle]) -> PredictionBundle:
    if not bundles:
        raise ValueError("cannot concatenate no prediction bundles")
    methods = tuple(bundles[0].predictions)
    if any(tuple(bundle.predictions) != methods for bundle in bundles[1:]):
        raise ValueError("prediction bundle methods differ")
    return PredictionBundle(
        label,
        np.concatenate([bundle.actual for bundle in bundles]),
        {
            method: np.concatenate([bundle.predictions[method] for bundle in bundles])
            for method in methods
        },
        np.concatenate([bundle.cells for bundle in bundles]),
        tuple(timestamp for bundle in bundles for timestamp in bundle.timestamps),
        np.concatenate([bundle.horizons for bundle in bundles]),
        np.concatenate([bundle.mase_scales for bundle in bundles]),
    )


def complete_mask(bundle: PredictionBundle) -> np.ndarray:
    return np.all(np.isfinite(bundle.actual), axis=1)


def official_mask(bundle: PredictionBundle) -> tuple[np.ndarray, np.ndarray]:
    complete = complete_mask(bundle)
    if not np.any(complete):
        raise ValueError(f"{bundle.label} has no complete targets")
    thresholds = np.quantile(bundle.actual[complete], 0.05, axis=0, method="linear")
    mask = complete & np.all(bundle.actual >= thresholds[None, :], axis=1)
    if np.any(bundle.actual[mask] <= 0.0):
        raise ValueError(f"{bundle.label} official filter retained non-positive targets")
    return mask, thresholds


def threshold_score(
    actual: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    thresholds: Sequence[float],
) -> tuple[float, list[float]]:
    error = np.mean(
        np.abs(actual[mask] - prediction[mask]) / actual[mask],
        axis=1,
    )
    rates = [float(np.mean(error < float(value))) for value in thresholds]
    return float(np.mean(rates)), rates



def metric_values(
    actual: np.ndarray,
    prediction: np.ndarray,
    scales: np.ndarray,
    mask: np.ndarray,
    metric: int,
) -> dict[str, float | int | None]:
    selected = mask & np.isfinite(actual[:, metric]) & np.isfinite(prediction[:, metric])
    y = actual[selected, metric]
    p = prediction[selected, metric]
    if not len(y):
        raise ValueError("metric calculation received no observations")
    absolute = np.abs(y - p)
    denominator = np.maximum(np.abs(y) + np.abs(p), 1e-12)
    valid_scale = selected & np.isfinite(scales[:, metric]) & (scales[:, metric] > 1e-12)
    scaled = np.abs(actual[valid_scale, metric] - prediction[valid_scale, metric]) / scales[
        valid_scale, metric
    ]
    return {
        "n_hours": int(len(y)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(y - p)))),
        "wape": float(np.sum(absolute) / max(float(np.sum(np.abs(y))), 1e-12)),
        "smape": float(np.mean(2.0 * absolute / denominator)),
        "mase": None if not len(scaled) else float(np.mean(scaled)),
        "mase_eligible_hours": int(len(scaled)),
    }


def standard_metric_rows(bundle: PredictionBundle) -> list[dict[str, object]]:
    complete = complete_mask(bundle)
    official, quantiles = official_mask(bundle)
    protocols = {
        "complete_targets_unfiltered": complete,
        "official_5pct_filtered": official,
    }
    rows: list[dict[str, object]] = []
    for method, prediction in bundle.predictions.items():
        for protocol, mask in protocols.items():
            indicator_rows: list[dict[str, object]] = []
            for metric, metric_name in enumerate(METRIC_NAMES):
                values = metric_values(
                    bundle.actual,
                    prediction,
                    bundle.mase_scales,
                    mask,
                    metric,
                )
                row: dict[str, object] = {
                    "dataset": bundle.label,
                    "method": method,
                    "filter": protocol,
                    "indicator": metric_name,
                    **values,
                }
                indicator_rows.append(row)
                rows.append(row)
            rows.append(
                {
                    "dataset": bundle.label,
                    "method": method,
                    "filter": protocol,
                    "indicator": "macro_mean",
                    "n_hours": int(np.sum(mask)),
                    "mae": float(np.mean([float(row["mae"]) for row in indicator_rows])),
                    "rmse": float(np.mean([float(row["rmse"]) for row in indicator_rows])),
                    "wape": float(np.mean([float(row["wape"]) for row in indicator_rows])),
                    "smape": float(np.mean([float(row["smape"]) for row in indicator_rows])),
                    "mase": float(
                        np.mean(
                            [float(row["mase"]) for row in indicator_rows if row["mase"] is not None]
                        )
                    ),
                    "mase_eligible_hours": int(
                        min(int(row["mase_eligible_hours"]) for row in indicator_rows)
                    ),
                }
            )
    rows.append(
        {
            "dataset": bundle.label,
            "method": "__filter_metadata__",
            "filter": "official_5pct_filtered",
            "indicator": "quantile_thresholds",
            "n_hours": int(np.sum(official)),
            "mae": float(quantiles[0]),
            "rmse": float(quantiles[1]),
            "wape": float(quantiles[2]),
            "smape": float(quantiles[3]),
            "mase": None,
            "mase_eligible_hours": 0,
        }
    )
    return rows


def indicator_hit_auc(
    actual: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    metric: int,
) -> tuple[float, list[float]]:
    error = np.abs(actual[mask, metric] - prediction[mask, metric]) / actual[mask, metric]
    rates = [float(np.mean(error < threshold)) for threshold in OFFICIAL_THRESHOLDS]
    return float(np.mean(rates)), rates


def cluster_bootstrap_indicator_delta(
    bundle: PredictionBundle,
    reference: np.ndarray,
    candidate: np.ndarray,
    metric: int,
    mask: np.ndarray,
    samples: np.ndarray,
) -> dict[str, object]:
    cells = bundle.cells[mask]
    unique_cells = np.unique(cells)
    reference_error = np.abs(bundle.actual[mask, metric] - reference[mask, metric]) / bundle.actual[
        mask, metric
    ]
    candidate_error = np.abs(bundle.actual[mask, metric] - candidate[mask, metric]) / bundle.actual[
        mask, metric
    ]
    reference_hits = np.sum(
        reference_error[:, None] < OFFICIAL_THRESHOLDS[None, :], axis=1
    )
    candidate_hits = np.sum(
        candidate_error[:, None] < OFFICIAL_THRESHOLDS[None, :], axis=1
    )
    reference_sum = np.asarray(
        [reference_hits[cells == cell].sum() for cell in unique_cells], dtype=np.float64
    )
    candidate_sum = np.asarray(
        [candidate_hits[cells == cell].sum() for cell in unique_cells], dtype=np.float64
    )
    counts = np.asarray(
        [np.sum(cells == cell) * len(OFFICIAL_THRESHOLDS) for cell in unique_cells],
        dtype=np.float64,
    )
    if samples.shape[1] != len(unique_cells):
        raise ValueError("bootstrap sample width does not match cell count")
    denominator = counts[samples].sum(axis=1)
    deltas = (
        candidate_sum[samples].sum(axis=1) - reference_sum[samples].sum(axis=1)
    ) / denominator
    return {
        "bootstrap_replicates": int(len(samples)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cluster_unit": "cell",
        "mean_delta": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "probability_positive": float(np.mean(deltas > 0.0)),
    }


def cluster_bootstrap_combined_delta(
    bundle: PredictionBundle,
    reference: np.ndarray,
    candidate: np.ndarray,
    replicates: int,
) -> dict[str, object]:
    mask, _ = official_mask(bundle)
    cells = bundle.cells[mask]
    actual = bundle.actual[mask]
    reference_error = np.mean(np.abs(actual - reference[mask]) / actual, axis=1)
    candidate_error = np.mean(np.abs(actual - candidate[mask]) / actual, axis=1)
    reference_hits = np.sum(
        reference_error[:, None] < OFFICIAL_THRESHOLDS[None, :], axis=1
    )
    candidate_hits = np.sum(
        candidate_error[:, None] < OFFICIAL_THRESHOLDS[None, :], axis=1
    )
    unique_cells = np.unique(cells)
    reference_sum = np.asarray(
        [reference_hits[cells == cell].sum() for cell in unique_cells], dtype=np.float64
    )
    candidate_sum = np.asarray(
        [candidate_hits[cells == cell].sum() for cell in unique_cells], dtype=np.float64
    )
    counts = np.asarray(
        [np.sum(cells == cell) * len(OFFICIAL_THRESHOLDS) for cell in unique_cells],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(unique_cells), size=(replicates, len(unique_cells)))
    deltas = (
        candidate_sum[samples].sum(axis=1) - reference_sum[samples].sum(axis=1)
    ) / counts[samples].sum(axis=1)
    return {
        "replicates": int(replicates),
        "seed": BOOTSTRAP_SEED,
        "cluster_unit": "cell (all seven repeated target days remain within cluster)",
        "mean_delta": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "probability_positive": float(np.mean(deltas > 0.0)),
    }


def load_verified_boosters(cache_dir: Path) -> list[lgb.Booster]:
    manifest_path = cache_dir / "cache_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("models")
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError(f"invalid frozen cache manifest: {manifest_path}")
    boosters: list[lgb.Booster] = []
    for metric, record in enumerate(records):
        expected_name = f"metric_{metric}.txt"
        if record.get("file") != expected_name:
            raise ValueError(f"unexpected model order in {manifest_path}")
        path = cache_dir / expected_name
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"frozen model size mismatch: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"frozen model SHA256 mismatch: {path}")
        boosters.append(lgb.Booster(model_file=str(path)))
    return boosters


def load_scene_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) < 3 or header[0] != "标准小区名称" or header[2] != "覆盖场景":
            raise ValueError("parameter scene schema mismatch")
        for row in reader:
            if len(row) >= 3 and row[0].strip():
                labels[row[0].strip()] = row[2].strip() or "unknown"
    return labels



@dataclass(frozen=True)
class FrozenEvidence:
    bundle: PredictionBundle
    lock_examples: tuple[BacktestExample, ...]
    baseline_config: BaselineConfig
    parameters: Mapping[str, Mapping[str, float]]
    weather: Mapping[str, Mapping[str, float]]
    feature_names: tuple[str, ...]
    boosters: Mapping[str, Sequence[lgb.Booster]]
    columns: Mapping[str, Sequence[np.ndarray]]
    feature_counts: Mapping[str, int]
    validation: Mapping[str, object]


def reconstruct_frozen_evidence(
    examples: Sequence[BacktestExample],
    inputs: Mapping[str, Path],
    frozen: Path,
) -> FrozenEvidence:
    dates = sorted({example.window.target_start.date() for example in examples})
    if len(dates) != 16:
        raise ValueError(f"expected 16 backtest dates, found {len(dates)}")
    fit_inner_dates = set(dates[:10])
    development_dates = set(dates[10:13])
    prelock_dates = set(dates[:13])
    lock_dates = set(dates[13:16])

    def selected(wanted) -> list[BacktestExample]:
        return [
            example
            for example in examples
            if example.window.target_start.date() in wanted
        ]

    fit_inner = selected(fit_inner_dates)
    development = selected(development_dates)
    prelock = selected(prelock_dates)
    lock = selected(lock_dates)
    parameters = load_parameters(inputs["parameter"])
    weather = load_weather(inputs["weather"])
    baseline_config = BaselineConfig(
        "weekly_median_s097",
        (0.0, 0.7, 0.2, 0.1, 0.0, 0.0),
        (0.97,) * 4,
    )
    fit_inner_matrix = build_matrix(fit_inner, baseline_config, parameters, weather)
    development_matrix = build_matrix(development, baseline_config, parameters, weather)
    prelock_matrix = build_matrix(prelock, baseline_config, parameters, weather)
    lock_matrix = build_matrix(lock, baseline_config, parameters, weather)
    names = tuple(feature_names(examples[0], baseline_config, parameters, weather))
    columns: dict[str, Sequence[np.ndarray]] = {
        "plain_lgbm": tuple(
            feature_columns(names, "plain_lgbm", metric) for metric in range(4)
        ),
        "proposed": tuple(
            feature_columns(names, "full", metric) for metric in range(4)
        ),
    }
    for variant in FROZEN_VARIANTS:
        columns[variant] = tuple(
            feature_columns(names, variant, metric) for metric in range(4)
        )

    cache_root = frozen / "models" / "train_lightgbm_baseline_v2"
    boosters: dict[str, Sequence[lgb.Booster]] = {
        "plain_lgbm": load_verified_boosters(cache_root / "plain_lgbm"),
        "proposed": load_verified_boosters(cache_root / "full"),
    }
    for variant in FROZEN_VARIANTS:
        boosters[variant] = load_verified_boosters(cache_root / variant)

    row_predictions: dict[str, Sequence[ForecastRow]] = {}
    baseline_configs = {
        "last_day": BaselineConfig(
            "last_day", (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ),
        "last_week": BaselineConfig(
            "last_week", (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        ),
        "weekly_80_20": BaselineConfig(
            "weekly_80_20", (0.0, 0.8, 0.2, 0.0, 0.0, 0.0)
        ),
        "robust_seasonal": baseline_config,
    }
    for name, config in baseline_configs.items():
        _, predictions = baseline_rows(lock, config)
        row_predictions[name] = predictions

    ridge_columns = np.arange(len(names), dtype=np.int64)
    ridge_candidates: list[tuple[float, float]] = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        ridge = train_ridge(fit_inner_matrix, ridge_columns, alpha)
        predictions = predict_ridge(ridge, development_matrix, ridge_columns)
        ridge_candidates.append(
            (
                float(score_dict(development_matrix.actuals, predictions)["mape_auc"]),
                alpha,
            )
        )
    ridge_alpha = max(ridge_candidates)[1]
    ridge = train_ridge(prelock_matrix, ridge_columns, ridge_alpha)
    row_predictions["ridge"] = predict_ridge(ridge, lock_matrix, ridge_columns)

    for name, model in boosters.items():
        predictions, _ = predict_boosters(model, lock_matrix, columns[name])
        row_predictions[name] = predictions

    bundle = bundle_from_examples("frozen_temporal_lockbox_3d", lock, row_predictions)
    expected_results = json.loads((frozen / "results.json").read_text(encoding="utf-8"))
    expected_main = {
        row["method"]: float(row["mape_auc"])
        for row in expected_results["main_results"]
        if row["protocol"] == "temporal_lockbox"
    }
    expected_ablation = {
        row["variant"]: float(row["mape_auc"])
        for row in expected_results["ablations"]
    }
    mask, _ = official_mask(bundle)
    observed: dict[str, float] = {}
    for name, prediction in bundle.predictions.items():
        auc, _ = threshold_score(
            bundle.actual,
            prediction,
            mask,
            OFFICIAL_THRESHOLDS,
        )
        observed[name] = auc
        expected = expected_main.get(name, expected_ablation.get(name))
        if expected is not None and not math.isclose(auc, expected, abs_tol=1e-12):
            raise ValueError(
                f"reconstructed frozen score mismatch for {name}: {auc} != {expected}"
            )
    feature_counts = {
        name: int(round(np.mean([len(item) for item in selected_columns])))
        for name, selected_columns in columns.items()
    }
    return FrozenEvidence(
        bundle=bundle,
        lock_examples=tuple(lock),
        baseline_config=baseline_config,
        parameters=parameters,
        weather=weather,
        feature_names=names,
        boosters=boosters,
        columns=columns,
        feature_counts=feature_counts,
        validation={
            "ridge_alpha": ridge_alpha,
            "reconstructed_mape_auc": observed,
            "frozen_results_sha256": sha256_file(frozen / "results.json"),
            "frozen_prediction_policy": (
                "models loaded only after cache-manifest size and SHA256 verification"
            ),
        },
    )


def build_scale_lookup(
    examples: Sequence[BacktestExample],
) -> dict[tuple[str, datetime], tuple[int, np.ndarray]]:
    lookup: dict[tuple[str, datetime], tuple[int, np.ndarray]] = {}
    for example in examples:
        scale = history_mase_scale(example)
        for horizon, actual in enumerate(example.actuals):
            key = (actual.cell, actual.timestamp)
            if key in lookup:
                raise ValueError(f"duplicate scale lookup key: {key}")
            lookup[key] = (horizon, scale)
    return lookup


def load_strict_nested_bundle(
    examples: Sequence[BacktestExample], frozen: Path
) -> PredictionBundle:
    lookup = build_scale_lookup(examples)
    actual: list[list[float]] = []
    plain: list[list[float]] = []
    proposed: list[list[float]] = []
    cells: list[str] = []
    timestamps: list[datetime] = []
    horizons: list[int] = []
    scales: list[np.ndarray] = []
    path = frozen / "strict_nested_oof_predictions.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = datetime.fromisoformat(row["timestamp"])
            cell = row["cell"]
            key = (cell, timestamp)
            if key not in lookup:
                raise ValueError(f"strict nested prediction lacks history scale: {key}")
            horizon, scale = lookup[key]
            actual.append(
                [
                    np.nan if row[f"actual_m{metric}"].upper() == "NIL" else float(row[f"actual_m{metric}"])
                    for metric in range(4)
                ]
            )
            plain.append([float(row[f"plain_m{metric}"]) for metric in range(4)])
            proposed.append([float(row[f"proposed_m{metric}"]) for metric in range(4)])
            cells.append(cell)
            timestamps.append(timestamp)
            horizons.append(horizon)
            scales.append(scale)
    bundle = PredictionBundle(
        "strict_nested_cell_disjoint_3d",
        np.asarray(actual, dtype=np.float64),
        {
            "plain_lgbm": np.asarray(plain, dtype=np.float64),
            "proposed": np.asarray(proposed, dtype=np.float64),
        },
        np.asarray(cells),
        tuple(timestamps),
        np.asarray(horizons, dtype=np.int16),
        np.asarray(scales, dtype=np.float64),
    )
    if len(bundle.actual) != 52_560:
        raise ValueError(f"unexpected strict nested prediction rows: {len(bundle.actual)}")
    return bundle



def select_baseline_for_inner(
    examples: Sequence[BacktestExample],
) -> tuple[BaselineConfig, dict[str, object]]:
    if not examples:
        raise ValueError("seasonal selection requires at least one prior target day")
    candidates = baseline_candidates()
    reports: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []
    for config in candidates:
        actuals, predictions = baseline_rows(examples, config)
        payload = score_dict(actuals, predictions)
        scores.append(payload)
        reports.append(
            {
                "name": config.name,
                "weights": list(config.weights),
                "scales": list(config.scales),
                "mape_auc": payload["mape_auc"],
                "mean_mape": payload["mean_mape"],
            }
        )

    def key(index: int) -> tuple[float, float, int]:
        score = scores[index]
        return (
            float(score["mape_auc"]),
            -float(score["mean_mape"]),
            -index,
        )

    selected_index = max(range(len(candidates)), key=key)
    return candidates[selected_index], {
        "selection_rule": (
            "highest prior-inner MAPEAUC; lower mean MAPE and candidate order break ties"
        ),
        "selection_dates": sorted(
            {str(example.window.target_start.date()) for example in examples}
        ),
        "selected": reports[selected_index],
        "candidate_count": len(reports),
        "candidates": reports,
    }


def train_prior_only_model(
    *,
    label: str,
    fit_examples: Sequence[BacktestExample],
    inner_examples: Sequence[BacktestExample],
    final_train_examples: Sequence[BacktestExample],
    evaluation_examples: Sequence[BacktestExample],
    parameters: Mapping[str, Mapping[str, float]],
    weather: Mapping[str, Mapping[str, float]],
    cache_root: Path,
    max_rounds: int,
    early_stopping_rounds: int,
) -> tuple[PredictionBundle, dict[str, object]]:
    fit_dates = sorted({example.window.target_start.date() for example in fit_examples})
    inner_dates = sorted({example.window.target_start.date() for example in inner_examples})
    final_dates = sorted(
        {example.window.target_start.date() for example in final_train_examples}
    )
    evaluation_dates = sorted(
        {example.window.target_start.date() for example in evaluation_examples}
    )
    if not fit_dates or not inner_dates or not evaluation_dates:
        raise ValueError(f"{label} has an empty fit, inner, or evaluation layer")
    if max(final_dates) >= min(evaluation_dates):
        raise ValueError(f"{label} violates chronological training/evaluation order")
    if set(fit_dates).intersection(inner_dates):
        raise ValueError(f"{label} fit and inner dates overlap")

    baseline_config, baseline_report = select_baseline_for_inner(inner_examples)
    matrix_started = time.perf_counter()
    final_matrix = build_matrix(
        final_train_examples, baseline_config, parameters, weather
    )
    final_target_dates = np.asarray(
        [row.timestamp.date() for row in final_matrix.actuals], dtype=object
    )
    fit_date_set = set(fit_dates)
    inner_date_set = set(inner_dates)
    fit_mask = np.asarray([value in fit_date_set for value in final_target_dates])
    inner_mask = np.asarray([value in inner_date_set for value in final_target_dates])
    if np.any(fit_mask & inner_mask) or not np.all(fit_mask | inner_mask):
        raise ValueError(f"{label} fit/inner masks do not partition final training rows")
    fit_matrix = subset_matrix(final_matrix, fit_mask)
    inner_matrix = subset_matrix(final_matrix, inner_mask)
    evaluation_matrix = build_matrix(
        evaluation_examples, baseline_config, parameters, weather
    )
    matrix_seconds = time.perf_counter() - matrix_started
    names = tuple(
        feature_names(
            final_train_examples[0], baseline_config, parameters, weather
        )
    )
    columns = {
        "plain_lgbm": tuple(
            feature_columns(names, "plain_lgbm", metric) for metric in range(4)
        ),
        "proposed": tuple(
            feature_columns(names, "full", metric) for metric in range(4)
        ),
    }
    selection_started = time.perf_counter()
    round_selections = select_model_rounds(
        fit_matrix,
        inner_matrix,
        columns,
        max_rounds=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
    )
    selection_seconds = time.perf_counter() - selection_started
    row_predictions: dict[str, Sequence[ForecastRow]] = {}
    _, seasonal_predictions = baseline_rows(evaluation_examples, baseline_config)
    row_predictions["robust_seasonal"] = seasonal_predictions
    fit_reports: dict[str, object] = {}
    for method in ("plain_lgbm", "proposed"):
        selection = round_selections[method]
        boosters, training_seconds, model_bytes = train_or_load_boosters(
            final_matrix,
            columns[method],
            selection.rounds,
            cache_root / method,
        )
        predictions, prediction_seconds = predict_boosters(
            boosters, evaluation_matrix, columns[method]
        )
        row_predictions[method] = predictions
        fit_reports[method] = {
            "selected_rounds": list(selection.rounds),
            "round_selection_diagnostics": list(selection.diagnostics),
            "training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds,
            "model_bytes": model_bytes,
            "feature_count": int(
                round(np.mean([len(item) for item in columns[method]]))
            ),
        }
    bundle = bundle_from_examples(label, evaluation_examples, row_predictions)
    scores: dict[str, object] = {}
    for method, predictions in row_predictions.items():
        actuals = [row for example in evaluation_examples for row in example.actuals]
        scores[method] = score_dict(actuals, predictions)
    return bundle, {
        "label": label,
        "fit_dates": list(map(str, fit_dates)),
        "inner_dates": list(map(str, inner_dates)),
        "final_train_dates": list(map(str, final_dates)),
        "evaluation_dates": list(map(str, evaluation_dates)),
        "fit_windows": len(fit_examples),
        "inner_windows": len(inner_examples),
        "final_train_windows": len(final_train_examples),
        "evaluation_windows": len(evaluation_examples),
        "evaluation_rows": len(bundle.actual),
        "seasonal_selection": baseline_report,
        "matrix_construction_seconds": matrix_seconds,
        "round_selection_seconds": selection_seconds,
        "models": fit_reports,
        "scores": scores,
        "leakage_checks": {
            "all_model_selection_dates_precede_evaluation": max(inner_dates)
            < min(evaluation_dates),
            "all_final_train_dates_precede_evaluation": max(final_dates)
            < min(evaluation_dates),
            "fit_inner_disjoint": not bool(set(fit_dates).intersection(inner_dates)),
        },
    }


def run_fixed_seven_day_holdout(
    examples: Sequence[BacktestExample],
    parameters: Mapping[str, Mapping[str, float]],
    weather: Mapping[str, Mapping[str, float]],
    output: Path,
    max_rounds: int,
    early_stopping_rounds: int,
) -> tuple[PredictionBundle, dict[str, object]]:
    dates = sorted({example.window.target_start.date() for example in examples})
    holdout_dates = set(dates[-7:])
    prior_dates = dates[:-7]
    if len(prior_dates) != 9:
        raise ValueError(f"fixed seven-day holdout expected 9 prior dates, found {len(prior_dates)}")
    fit_dates = set(prior_dates[:-2])
    inner_dates = set(prior_dates[-2:])
    final_dates = set(prior_dates)

    def layer(selected_dates) -> list[BacktestExample]:
        return [
            example
            for example in examples
            if example.window.target_start.date() in selected_dates
        ]

    bundle, report = train_prior_only_model(
        label="fixed_seven_day_holdout",
        fit_examples=layer(fit_dates),
        inner_examples=layer(inner_dates),
        final_train_examples=layer(final_dates),
        evaluation_examples=layer(holdout_dates),
        parameters=parameters,
        weather=weather,
        cache_root=output / "models" / "fixed_seven_day_holdout",
        max_rounds=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
    )
    if report["evaluation_windows"] != 5_110 or len(bundle.actual) != 122_640:
        raise ValueError(
            "fixed seven-day holdout must contain 5,110 windows and 122,640 rows"
        )
    report["single_model_for_all_seven_days"] = True
    return bundle, report


def run_seven_rolling_origins(
    examples: Sequence[BacktestExample],
    parameters: Mapping[str, Mapping[str, float]],
    weather: Mapping[str, Mapping[str, float]],
    output: Path,
    fixed_bundle: PredictionBundle,
    fixed_report: Mapping[str, object],
    max_rounds: int,
    early_stopping_rounds: int,
) -> tuple[PredictionBundle, list[dict[str, object]]]:
    dates = sorted({example.window.target_start.date() for example in examples})
    origins = dates[-7:]
    bundles: list[PredictionBundle] = []
    reports: list[dict[str, object]] = []
    first_origin = origins[0]
    first_mask = np.asarray(
        [timestamp.date() == first_origin for timestamp in fixed_bundle.timestamps]
    )
    bundles.append(
        fixed_bundle.subset(first_mask, f"rolling_origin_{first_origin}")
    )
    reports.append(
        {
            "label": f"rolling_origin_{first_origin}",
            "reused_fixed_seven_day_model": True,
            "origin": str(first_origin),
            "fit_dates": fixed_report["fit_dates"],
            "inner_dates": fixed_report["inner_dates"],
            "final_train_dates": fixed_report["final_train_dates"],
            "evaluation_dates": [str(first_origin)],
            "models": fixed_report["models"],
            "seasonal_selection": fixed_report["seasonal_selection"],
            "leakage_checks": fixed_report["leakage_checks"],
        }
    )

    for origin in origins[1:]:
        prior_dates = [value for value in dates if value < origin]
        fit_dates = set(prior_dates[:-2])
        inner_dates = set(prior_dates[-2:])
        final_dates = set(prior_dates)

        def layer(selected_dates) -> list[BacktestExample]:
            return [
                example
                for example in examples
                if example.window.target_start.date() in selected_dates
            ]

        bundle, report = train_prior_only_model(
            label=f"rolling_origin_{origin}",
            fit_examples=layer(fit_dates),
            inner_examples=layer(inner_dates),
            final_train_examples=layer(final_dates),
            evaluation_examples=layer({origin}),
            parameters=parameters,
            weather=weather,
            cache_root=output / "models" / "rolling_origin" / str(origin),
            max_rounds=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
        )
        report["reused_fixed_seven_day_model"] = False
        report["origin"] = str(origin)
        bundles.append(bundle)
        reports.append(report)
    pooled = concatenate_bundles("seven_rolling_origins", bundles)
    expected_dates = [str(value) for value in origins]
    observed_dates = sorted({str(value.date()) for value in pooled.timestamps})
    if observed_dates != expected_dates or len(pooled.actual) != 122_640:
        raise ValueError("rolling-origin pooled predictions do not cover exactly seven days")
    return pooled, reports



def threshold_sensitivity(
    bundles: Sequence[PredictionBundle],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    grid = np.asarray([0.10 + 0.025 * index for index in range(21)], dtype=np.float64)
    grid_rows: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    for bundle in bundles:
        if not {"plain_lgbm", "proposed"}.issubset(bundle.predictions):
            continue
        mask, _ = official_mask(bundle)
        actual = bundle.actual[mask]
        plain_error = np.mean(
            np.abs(actual - bundle.predictions["plain_lgbm"][mask]) / actual,
            axis=1,
        )
        proposed_error = np.mean(
            np.abs(actual - bundle.predictions["proposed"][mask]) / actual,
            axis=1,
        )
        plain_rates = np.asarray(
            [np.mean(plain_error < threshold) for threshold in grid], dtype=np.float64
        )
        proposed_rates = np.asarray(
            [np.mean(proposed_error < threshold) for threshold in grid], dtype=np.float64
        )
        for threshold, plain_rate, proposed_rate in zip(
            grid, plain_rates, proposed_rates
        ):
            grid_rows.append(
                {
                    "dataset": bundle.label,
                    "threshold": float(threshold),
                    "plain_lgbm_hit_rate": float(plain_rate),
                    "proposed_hit_rate": float(proposed_rate),
                    "delta_proposed_minus_plain": float(proposed_rate - plain_rate),
                    "n_hours": int(len(actual)),
                }
            )
        set_summary: dict[str, object] = {}
        for set_name, thresholds in THRESHOLD_SETS.items():
            plain_score = float(
                np.mean([np.mean(plain_error < threshold) for threshold in thresholds])
            )
            proposed_score = float(
                np.mean([np.mean(proposed_error < threshold) for threshold in thresholds])
            )
            row = {
                "dataset": bundle.label,
                "threshold_set": set_name,
                "thresholds": "|".join(f"{value:.2f}" for value in thresholds),
                "plain_lgbm_score": plain_score,
                "proposed_score": proposed_score,
                "delta_proposed_minus_plain": proposed_score - plain_score,
                "n_hours": int(len(actual)),
            }
            set_rows.append(row)
            set_summary[set_name] = row
        interval = float(grid[-1] - grid[0])
        plain_continuous = float(np.trapezoid(plain_rates, grid) / interval)
        proposed_continuous = float(np.trapezoid(proposed_rates, grid) / interval)
        summaries[bundle.label] = {
            "grid_min": float(grid[0]),
            "grid_max": float(grid[-1]),
            "grid_step": 0.025,
            "normalized_continuous_auc_plain_lgbm": plain_continuous,
            "normalized_continuous_auc_proposed": proposed_continuous,
            "continuous_auc_delta": proposed_continuous - plain_continuous,
            "positive_delta_grid_points": int(np.sum(proposed_rates > plain_rates)),
            "zero_delta_grid_points": int(np.sum(proposed_rates == plain_rates)),
            "negative_delta_grid_points": int(np.sum(proposed_rates < plain_rates)),
            "threshold_sets": set_summary,
        }
    return grid_rows, set_rows, summaries


def ablation_indicator_rows(
    frozen: FrozenEvidence, replicates: int
) -> list[dict[str, object]]:
    bundle = frozen.bundle
    mask, _ = official_mask(bundle)
    unique_cells = np.unique(bundle.cells[mask])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(
        0, len(unique_cells), size=(replicates, len(unique_cells)), dtype=np.int32
    )
    full = bundle.predictions["proposed"]
    variants = ("proposed", *FROZEN_VARIANTS)
    rows: list[dict[str, object]] = []
    for variant in variants:
        prediction = bundle.predictions[variant]
        for metric, metric_name in enumerate(METRIC_NAMES):
            values = metric_values(
                bundle.actual,
                prediction,
                bundle.mase_scales,
                mask,
                metric,
            )
            auc, rates = indicator_hit_auc(
                bundle.actual, prediction, mask, metric
            )
            full_auc, _ = indicator_hit_auc(bundle.actual, full, mask, metric)
            bootstrap = cluster_bootstrap_indicator_delta(
                bundle,
                full,
                prediction,
                metric,
                mask,
                samples,
            )
            rows.append(
                {
                    "variant": "full" if variant == "proposed" else variant,
                    "indicator": metric_name,
                    "features": frozen.feature_counts[
                        "proposed" if variant == "proposed" else variant
                    ],
                    "indicator_hit_auc": auc,
                    "delta_hit_auc_vs_full": auc - full_auc,
                    "hit_020": rates[0],
                    "hit_030": rates[1],
                    "hit_040": rates[2],
                    "hit_050": rates[3],
                    "wape": values["wape"],
                    "smape": values["smape"],
                    "mase": values["mase"],
                    "n_hours": values["n_hours"],
                    "bootstrap_delta_mean": bootstrap["mean_delta"],
                    "bootstrap_ci_low": bootstrap["ci_low"],
                    "bootstrap_ci_high": bootstrap["ci_high"],
                    "bootstrap_probability_positive": bootstrap[
                        "probability_positive"
                    ],
                    "bootstrap_replicates": bootstrap["bootstrap_replicates"],
                    "independent_unit": "cell",
                    "inference_status": "exploratory; no multiplicity-adjusted claim",
                }
            )
    return rows


def scene_stratification_rows(
    bundle: PredictionBundle, scene_labels: Mapping[str, str]
) -> list[dict[str, object]]:
    global_official, _ = official_mask(bundle)
    labels = np.asarray([scene_labels.get(str(cell), "unknown") for cell in bundle.cells])
    rows: list[dict[str, object]] = []
    for scene in sorted(set(labels)):
        scene_mask = global_official & (labels == scene)
        if not np.any(scene_mask):
            continue
        for method in ("robust_seasonal", "plain_lgbm", "proposed"):
            prediction = bundle.predictions[method]
            auc, rates = threshold_score(
                bundle.actual,
                prediction,
                scene_mask,
                OFFICIAL_THRESHOLDS,
            )
            metric_wape = [
                metric_values(
                    bundle.actual,
                    prediction,
                    bundle.mase_scales,
                    scene_mask,
                    metric,
                )["wape"]
                for metric in range(4)
            ]
            rows.append(
                {
                    "dataset": bundle.label,
                    "scene": scene,
                    "method": method,
                    "cells": int(len(np.unique(bundle.cells[scene_mask]))),
                    "hours": int(np.sum(scene_mask)),
                    "mape_auc": auc,
                    "hit_020": rates[0],
                    "hit_030": rates[1],
                    "hit_040": rates[2],
                    "hit_050": rates[3],
                    "wape_ul_active_users": metric_wape[0],
                    "wape_dl_active_users": metric_wape[1],
                    "wape_dl_prb": metric_wape[2],
                    "wape_ul_prb": metric_wape[3],
                    "filter_policy": "global official 5pct thresholds, not re-estimated per scene",
                }
            )
    return rows


def utility_curve_rows(
    bundle: PredictionBundle,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    methods = ("robust_seasonal", "plain_lgbm", "proposed")
    groups = {
        "dl_prb": (2,),
        "ul_prb": (3,),
        "pooled_prb": (2, 3),
    }
    betas = np.asarray([index / 100.0 for index in range(101)], dtype=np.float64)
    curve_rows: list[dict[str, object]] = []
    indexed: dict[tuple[str, str], list[dict[str, object]]] = {}
    complete = complete_mask(bundle)
    for group, metrics in groups.items():
        actual = bundle.actual[complete][:, metrics].reshape(-1)
        demand = max(float(np.sum(actual)), 1e-12)
        for method in methods:
            prediction = bundle.predictions[method][complete][:, metrics].reshape(-1)
            method_rows: list[dict[str, object]] = []
            for beta in betas:
                provision = prediction * (1.0 + beta)
                surplus = np.maximum(provision - actual, 0.0)
                shortage = np.maximum(actual - provision, 0.0)
                row = {
                    "dataset": bundle.label,
                    "resource_group": group,
                    "method": method,
                    "safety_margin_beta": float(beta),
                    "sla_violation_rate": float(np.mean(provision < actual)),
                    "waste_ratio": float(np.sum(surplus) / demand),
                    "shortage_ratio": float(np.sum(shortage) / demand),
                    "provision_to_demand_ratio": float(np.sum(provision) / demand),
                    "n_indicator_hours": int(len(actual)),
                }
                method_rows.append(row)
                curve_rows.append(row)
            indexed[(group, method)] = method_rows
    summary_rows: list[dict[str, object]] = []
    for group in groups:
        for target in (0.10, 0.05, 0.01):
            for method in methods:
                eligible = [
                    row
                    for row in indexed[(group, method)]
                    if float(row["sla_violation_rate"]) <= target
                ]
                selected = eligible[0] if eligible else None
                summary_rows.append(
                    {
                        "dataset": bundle.label,
                        "resource_group": group,
                        "target_sla_violation_rate": target,
                        "method": method,
                        "minimum_beta_on_grid": None
                        if selected is None
                        else selected["safety_margin_beta"],
                        "achieved_sla_violation_rate": None
                        if selected is None
                        else selected["sla_violation_rate"],
                        "waste_ratio_at_selected_beta": None
                        if selected is None
                        else selected["waste_ratio"],
                        "shortage_ratio_at_selected_beta": None
                        if selected is None
                        else selected["shortage_ratio"],
                        "provision_to_demand_ratio": None
                        if selected is None
                        else selected["provision_to_demand_ratio"],
                        "interpretation": (
                            "post-hoc descriptive utility curve; beta is not a deployable tuned policy"
                        ),
                    }
                )
    return curve_rows, summary_rows


def heatmap_rows(
    bundle: PredictionBundle,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    complete = complete_mask(bundle)
    horizon_indicator: list[dict[str, object]] = []
    day_horizon: list[dict[str, object]] = []
    for method in ("plain_lgbm", "proposed"):
        prediction = bundle.predictions[method]
        for horizon in range(24):
            horizon_mask = complete & (bundle.horizons == horizon)
            for metric, metric_name in enumerate(METRIC_NAMES):
                values = metric_values(
                    bundle.actual,
                    prediction,
                    bundle.mase_scales,
                    horizon_mask,
                    metric,
                )
                horizon_indicator.append(
                    {
                        "dataset": bundle.label,
                        "method": method,
                        "horizon": horizon,
                        "indicator": metric_name,
                        "n_hours": values["n_hours"],
                        "mae": values["mae"],
                        "wape": values["wape"],
                        "smape": values["smape"],
                        "mase": values["mase"],
                    }
                )
        weekdays = np.asarray([timestamp.weekday() for timestamp in bundle.timestamps])
        for weekday in range(7):
            for horizon in range(24):
                mask = complete & (weekdays == weekday) & (bundle.horizons == horizon)
                if not np.any(mask):
                    continue
                actual = bundle.actual[mask]
                predicted = prediction[mask]
                absolute = np.abs(actual - predicted)
                day_horizon.append(
                    {
                        "dataset": bundle.label,
                        "method": method,
                        "weekday": weekday,
                        "horizon": horizon,
                        "n_cell_hours": int(np.sum(mask)),
                        "wape_all_indicators": float(
                            np.sum(absolute)
                            / max(float(np.sum(np.abs(actual))), 1e-12)
                        ),
                        "smape_all_indicators": float(
                            np.mean(
                                2.0
                                * absolute
                                / np.maximum(
                                    np.abs(actual) + np.abs(predicted), 1e-12
                                )
                            )
                        ),
                    }
                )
    pairs = sorted(
        {(int(horizon), timestamp.hour) for horizon, timestamp in zip(bundle.horizons, bundle.timestamps)}
    )
    identifiability = {
        "dataset": bundle.label,
        "possible_horizon_hour_cells": 24 * 24,
        "observed_horizon_hour_cells": len(pairs),
        "observed_pairs": [list(pair) for pair in pairs],
        "horizon_equals_hour_of_day_for_all_rows": all(
            horizon == hour for horizon, hour in pairs
        ),
        "conclusion": (
            "All requests start at midnight, so horizon and target hour-of-day are collinear. "
            "A dense horizon-by-hour heatmap would be structurally misleading; use "
            "horizon-by-indicator and weekday-by-horizon instead."
        ),
    }
    return horizon_indicator, day_horizon, identifiability



def asynchronous_corruption_seed(
    example: BacktestExample, mode: str, severity: float | int, metric: int
) -> int:
    text = (
        f"{REVISION2_VERSION}|async|{example.window.cell}|"
        f"{example.window.target_start.isoformat()}|{mode}|{severity}|metric={metric}"
    )
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def corrupt_asynchronous_histories(
    examples: Sequence[BacktestExample], mode: str, severity: float | int
) -> tuple[list[BacktestExample], dict[str, object]]:
    corrupted: list[BacktestExample] = []
    injected_counts = np.zeros(4, dtype=np.int64)
    pairwise_jaccards: list[float] = []
    identical_mask_examples = 0
    for example in examples:
        rows = list(example.window.rows)
        selected_sets: list[set[int]] = []
        for metric in range(4):
            rng = np.random.default_rng(
                asynchronous_corruption_seed(example, mode, severity, metric)
            )
            if mode == "independent_random":
                count = int(round(len(rows) * float(severity)))
                indices = set(
                    map(int, rng.choice(len(rows), size=count, replace=False))
                )
            elif mode == "staggered_block":
                length = int(severity)
                if not 1 <= length <= len(rows):
                    raise ValueError("invalid staggered block length")
                span = len(rows) - length + 1
                start = (int(rng.integers(0, span)) + metric * 53) % span
                indices = set(range(start, start + length))
            else:
                raise ValueError(f"unknown asynchronous corruption mode: {mode}")
            selected_sets.append(indices)
            injected_counts[metric] += len(indices)
            for index in indices:
                values = list(rows[index].metrics)
                values[metric] = None
                rows[index] = replace(rows[index], metrics=tuple(values))
        if len({tuple(sorted(value)) for value in selected_sets}) == 1:
            identical_mask_examples += 1
        for left in range(4):
            for right in range(left + 1, 4):
                union = selected_sets[left] | selected_sets[right]
                intersection = selected_sets[left] & selected_sets[right]
                pairwise_jaccards.append(
                    0.0 if not union else len(intersection) / len(union)
                )
        corrupted.append(
            BacktestExample(replace(example.window, rows=tuple(rows)), example.actuals)
        )
    if identical_mask_examples == len(examples):
        raise ValueError("asynchronous corruption unexpectedly produced identical masks")
    total_history_rows = len(examples) * 336
    return corrupted, {
        "mode": mode,
        "severity": severity,
        "examples": len(examples),
        "injected_fraction_per_indicator": [
            float(value / total_history_rows) for value in injected_counts
        ],
        "mean_pairwise_mask_jaccard": float(np.mean(pairwise_jaccards)),
        "identical_four_mask_examples": identical_mask_examples,
        "seed_policy": "SHA256(revision, cell, target_start, mode, severity, metric)",
    }


def asynchronous_missingness_rows(
    frozen: FrozenEvidence,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    conditions: tuple[tuple[str, float | int], ...] = (
        ("independent_random", 0.10),
        ("independent_random", 0.20),
        ("independent_random", 0.30),
        ("staggered_block", 48),
    )
    result_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for mode, severity in conditions:
        corrupted, diagnostic = corrupt_asynchronous_histories(
            frozen.lock_examples, mode, severity
        )
        diagnostics.append(diagnostic)
        matrix = build_matrix(
            corrupted,
            frozen.baseline_config,
            frozen.parameters,
            frozen.weather,
        )
        row_predictions: dict[str, Sequence[ForecastRow]] = {}
        _, baseline_prediction = baseline_rows(corrupted, frozen.baseline_config)
        row_predictions["robust_seasonal"] = baseline_prediction
        for method in ("plain_lgbm", "no_missingness", "proposed"):
            prediction, _ = predict_boosters(
                frozen.boosters[method], matrix, frozen.columns[method]
            )
            row_predictions[method] = prediction
        bundle = bundle_from_examples(
            f"async_{mode}_{severity}", corrupted, row_predictions
        )
        mask, _ = official_mask(bundle)
        actual_rows = [row for example in corrupted for row in example.actuals]
        for method, predictions in row_predictions.items():
            score = score_dict(actual_rows, predictions)
            per_metric = [
                metric_values(
                    bundle.actual,
                    bundle.predictions[method],
                    bundle.mase_scales,
                    mask,
                    metric,
                )
                for metric in range(4)
            ]
            result_rows.append(
                {
                    "mode": mode,
                    "severity": severity,
                    "method": method,
                    "mape_auc": score["mape_auc"],
                    "mean_mape": score["mean_mape"],
                    "wape_ul_active_users": per_metric[0]["wape"],
                    "wape_dl_active_users": per_metric[1]["wape"],
                    "wape_dl_prb": per_metric[2]["wape"],
                    "wape_ul_prb": per_metric[3]["wape"],
                    "official_filtered_hours": int(np.sum(mask)),
                    "model_refit": False,
                }
            )
    return result_rows, diagnostics



def daily_score_rows(bundle: PredictionBundle) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dates = sorted({timestamp.date() for timestamp in bundle.timestamps})
    for target_date in dates:
        mask = np.asarray(
            [timestamp.date() == target_date for timestamp in bundle.timestamps]
        )
        daily = bundle.subset(mask, f"{bundle.label}_{target_date}")
        official, _ = official_mask(daily)
        for method in ("robust_seasonal", "plain_lgbm", "proposed"):
            auc, rates = threshold_score(
                daily.actual,
                daily.predictions[method],
                official,
                OFFICIAL_THRESHOLDS,
            )
            rows.append(
                {
                    "dataset": bundle.label,
                    "date": str(target_date),
                    "method": method,
                    "mape_auc": auc,
                    "hit_020": rates[0],
                    "hit_030": rates[1],
                    "hit_040": rates[2],
                    "hit_050": rates[3],
                    "official_filtered_hours": int(np.sum(official)),
                    "windows": int(len(daily.actual) // 24),
                }
            )
    return rows


def daily_summary(
    bundle: PredictionBundle, rows: Sequence[Mapping[str, object]], replicates: int
) -> dict[str, object]:
    method_summary: dict[str, object] = {}
    for method in ("robust_seasonal", "plain_lgbm", "proposed"):
        values = [
            float(row["mape_auc"])
            for row in rows
            if row["dataset"] == bundle.label and row["method"] == method
        ]
        if len(values) != 7:
            raise ValueError(f"{bundle.label}/{method} does not have seven daily scores")
        method_summary[method] = {
            "n_target_days": len(values),
            "mean_daily_mape_auc": statistics.mean(values),
            "sample_sd_daily_mape_auc": statistics.stdev(values),
            "minimum_daily_mape_auc": min(values),
            "maximum_daily_mape_auc": max(values),
            "daily_values": values,
        }
    plain_values = method_summary["plain_lgbm"]["daily_values"]
    proposed_values = method_summary["proposed"]["daily_values"]
    deltas = [
        float(proposed) - float(plain)
        for proposed, plain in zip(proposed_values, plain_values)
    ]
    official, _ = official_mask(bundle)
    pooled_plain, _ = threshold_score(
        bundle.actual,
        bundle.predictions["plain_lgbm"],
        official,
        OFFICIAL_THRESHOLDS,
    )
    pooled_proposed, _ = threshold_score(
        bundle.actual,
        bundle.predictions["proposed"],
        official,
        OFFICIAL_THRESHOLDS,
    )
    return {
        "dataset": bundle.label,
        "method_summaries": method_summary,
        "paired_daily_delta_proposed_minus_plain": {
            "n_target_days": 7,
            "mean": statistics.mean(deltas),
            "sample_sd": statistics.stdev(deltas),
            "minimum": min(deltas),
            "maximum": max(deltas),
            "positive_days": sum(value > 0.0 for value in deltas),
            "daily_values": deltas,
        },
        "pooled_official_mape_auc": {
            "plain_lgbm": pooled_plain,
            "proposed": pooled_proposed,
            "delta": pooled_proposed - pooled_plain,
        },
        "paired_cell_cluster_bootstrap": cluster_bootstrap_combined_delta(
            bundle,
            bundle.predictions["plain_lgbm"],
            bundle.predictions["proposed"],
            replicates,
        ),
        "independent_temporal_units": (
            "seven target dates; daily mean and sample SD are descriptive because n=7"
        ),
    }


def write_seven_day_predictions(
    path: Path, fixed: PredictionBundle, rolling: PredictionBundle
) -> None:
    if len(fixed.actual) != len(rolling.actual):
        raise ValueError("fixed and rolling seven-day row counts differ")
    rolling_lookup = {
        (str(cell), timestamp): index
        for index, (cell, timestamp) in enumerate(
            zip(rolling.cells, rolling.timestamps)
        )
    }
    if len(rolling_lookup) != len(rolling.actual):
        raise ValueError("rolling seven-day prediction identities are not unique")
    fieldnames = ["timestamp", "cell", "horizon"]
    fieldnames.extend(f"actual_m{metric}" for metric in range(4))
    fieldnames.extend(f"mase_scale_m{metric}" for metric in range(4))
    for prefix in (
        "fixed_seasonal",
        "fixed_plain",
        "fixed_proposed",
        "rolling_seasonal",
        "rolling_plain",
        "rolling_proposed",
    ):
        fieldnames.extend(f"{prefix}_m{metric}" for metric in range(4))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (cell, timestamp) in enumerate(
            zip(fixed.cells, fixed.timestamps)
        ):
            rolling_index = rolling_lookup[(str(cell), timestamp)]
            if not np.allclose(
                fixed.actual[index], rolling.actual[rolling_index], equal_nan=True
            ):
                raise ValueError("fixed and rolling actual targets disagree")
            row: dict[str, object] = {
                "timestamp": timestamp.isoformat(sep=" "),
                "cell": str(cell),
                "horizon": int(fixed.horizons[index]),
            }
            for metric in range(4):
                actual = fixed.actual[index, metric]
                row[f"actual_m{metric}"] = "NIL" if not np.isfinite(actual) else actual
                row[f"mase_scale_m{metric}"] = fixed.mase_scales[index, metric]
                row[f"fixed_seasonal_m{metric}"] = fixed.predictions[
                    "robust_seasonal"
                ][index, metric]
                row[f"fixed_plain_m{metric}"] = fixed.predictions["plain_lgbm"][
                    index, metric
                ]
                row[f"fixed_proposed_m{metric}"] = fixed.predictions["proposed"][
                    index, metric
                ]
                row[f"rolling_seasonal_m{metric}"] = rolling.predictions[
                    "robust_seasonal"
                ][rolling_index, metric]
                row[f"rolling_plain_m{metric}"] = rolling.predictions["plain_lgbm"][
                    rolling_index, metric
                ]
                row[f"rolling_proposed_m{metric}"] = rolling.predictions["proposed"][
                    rolling_index, metric
                ]
            writer.writerow(row)
    temporary.replace(path)


def output_inventory(output: Path, relative_paths: Sequence[str]) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for relative in relative_paths:
        path = output / relative
        inventory.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def runtime_report(gpu_devices: str) -> dict[str, object]:
    completed = subprocess.run(
        ["nvidia-smi", "-L"],
        check=True,
        text=True,
        capture_output=True,
    )
    gpu_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    requested = [int(value) for value in gpu_devices.split(",") if value.strip()]
    if any(device >= len(gpu_lines) for device in requested):
        raise ValueError(
            f"requested GPU devices {requested} exceed visible GPU inventory {gpu_lines}"
        )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "lightgbm": lgb.__version__,
        "lightgbm_device_type": MODEL_PARAMS["device_type"],
        "requested_gpu_devices": requested,
        "visible_gpus": gpu_lines,
        "final_fit_parallelism": "one target model per requested GPU via spawn workers",
        "round_selection_parallelism": (
            "target-specific early stopping runs sequentially but is assigned across requested GPUs"
        ),
    }



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate all repository-local evidence requested by paper review 2"
    )
    parser.add_argument("--frozen-artifacts", default=DEFAULT_FROZEN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--max-boost-rounds", type=int, default=MAX_BOOST_ROUNDS)
    parser.add_argument(
        "--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS
    )
    parser.add_argument(
        "--skip-asynchronous-missingness",
        action="store_true",
        help="diagnostic-only escape hatch; the complete revision run does not use it",
    )
    args = parser.parse_args()
    if args.bootstrap < 1000:
        raise ValueError("revision-2 bootstrap requires at least 1,000 replicates")
    if args.max_boost_rounds < 50 or args.early_stopping_rounds <= 0:
        raise ValueError("invalid boosting-round selection limits")
    os.environ["PAPER_GPU_DEVICES"] = args.gpu_devices

    inputs = resolve_registered_inputs()
    frozen_path = resolve_frozen(args.frozen_artifacts)
    output = resolve_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = runtime_report(args.gpu_devices)

    training_rows = read_traffic(inputs["train"])
    examples = build_training_backtests(training_rows)
    if len(examples) != 11_685:
        raise ValueError(f"expected 11,685 continuous backtests, found {len(examples)}")
    parameters = load_parameters(inputs["parameter"])
    weather = load_weather(inputs["weather"])
    scene_labels = load_scene_labels(inputs["parameter"])

    frozen = reconstruct_frozen_evidence(examples, inputs, frozen_path)
    strict_nested = load_strict_nested_bundle(examples, frozen_path)
    write_json(output / "frozen_reuse_validation.json", frozen.validation)

    fixed, fixed_report = run_fixed_seven_day_holdout(
        examples,
        parameters,
        weather,
        output,
        args.max_boost_rounds,
        args.early_stopping_rounds,
    )
    write_json(output / "fixed_seven_day_report.json", fixed_report)
    rolling, rolling_reports = run_seven_rolling_origins(
        examples,
        parameters,
        weather,
        output,
        fixed,
        fixed_report,
        args.max_boost_rounds,
        args.early_stopping_rounds,
    )
    write_json(output / "rolling_origin_reports.json", rolling_reports)
    write_seven_day_predictions(output / "seven_day_predictions.csv", fixed, rolling)

    standard_rows: list[dict[str, object]] = []
    for bundle in (fixed, rolling, frozen.bundle, strict_nested):
        standard_rows.extend(standard_metric_rows(bundle))
    write_csv(output / "standard_metrics.csv", standard_rows)

    daily_rows = daily_score_rows(fixed) + daily_score_rows(rolling)
    write_csv(output / "seven_day_daily_metrics.csv", daily_rows)
    fixed_summary = daily_summary(fixed, daily_rows, args.bootstrap)
    rolling_summary = daily_summary(rolling, daily_rows, args.bootstrap)
    write_json(output / "fixed_seven_day_summary.json", fixed_summary)
    write_json(output / "rolling_origin_summary.json", rolling_summary)

    threshold_grid, threshold_sets, threshold_summary = threshold_sensitivity(
        (fixed, rolling, frozen.bundle, strict_nested)
    )
    write_csv(output / "threshold_sensitivity_grid.csv", threshold_grid)
    write_csv(output / "threshold_sensitivity_sets.csv", threshold_sets)
    write_json(output / "threshold_sensitivity_summary.json", threshold_summary)

    ablation_rows = ablation_indicator_rows(frozen, args.bootstrap)
    write_csv(output / "per_indicator_ablation.csv", ablation_rows)
    scene_rows = scene_stratification_rows(fixed, scene_labels)
    write_csv(output / "scene_stratification.csv", scene_rows)

    utility_curve, utility_summary = utility_curve_rows(fixed)
    write_csv(output / "prb_utility_curve.csv", utility_curve)
    write_csv(output / "prb_utility_summary.csv", utility_summary)

    horizon_indicator, day_horizon, identifiability = heatmap_rows(fixed)
    write_csv(output / "error_heatmap_horizon_indicator.csv", horizon_indicator)
    write_csv(output / "error_heatmap_weekday_horizon.csv", day_horizon)
    write_json(output / "error_heatmap_identifiability.json", identifiability)

    async_rows: list[dict[str, object]] = []
    async_diagnostics: list[dict[str, object]] = []
    if not args.skip_asynchronous_missingness:
        async_rows, async_diagnostics = asynchronous_missingness_rows(frozen)
        write_csv(output / "asynchronous_missingness.csv", async_rows)
        write_json(
            output / "asynchronous_missingness_diagnostics.json", async_diagnostics
        )

    model_manifests = [
        {
            "path": str(path.relative_to(output)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted((output / "models").rglob("cache_manifest.json"))
    ]
    results_summary = {
        "schema_version": SCHEMA_VERSION,
        "revision2_version": REVISION2_VERSION,
        "primary_evaluation": fixed_summary,
        "rolling_origin_evaluation": rolling_summary,
        "threshold_sensitivity": threshold_summary,
        "frozen_reuse_validation": frozen.validation,
        "asynchronous_missingness_diagnostics": async_diagnostics,
        "mase_definition": {
            "name": "request-local seasonal MASE",
            "numerator": "absolute forecast error for one target indicator-hour",
            "denominator": (
                "for the same request and indicator, mean absolute difference over all "
                "available pairs x[t]-x[t-168] inside its 336-hour history"
            ),
            "missing_pairs": "excluded pairwise",
            "zero_or_unavailable_scale": (
                "that request-indicator is excluded from MASE only; eligible hours are reported"
            ),
            "aggregation": "mean scaled absolute error across eligible target hours",
        },
        "statistical_units": {
            "paired_bootstrap": "cell; repeated target dates remain within the cell cluster",
            "temporal_stability": "seven target dates, reported descriptively as mean and sample SD",
            "ablation": (
                "exploratory per-indicator cell-cluster intervals without multiplicity-adjusted claims"
            ),
        },
        "training_policy": {
            "fixed_seven_day": (
                "one model selected and fitted only with 2024-08-11 and earlier evidence; "
                "evaluated unchanged on 2024-08-12 through 2024-08-18"
            ),
            "rolling_origin": (
                "seven origins; at each origin, the last two prior target days select the "
                "seasonal candidate and target-specific boosting rounds, and all prior days "
                "fit the final model"
            ),
            "frozen_ablation_and_async_stress": "no refit; verified V2 model caches reused",
        },
        "runtime": runtime,
        "model_cache_manifests": model_manifests,
        "prediction_hashes": {
            "fixed_actual": canonical_array_sha256(fixed.actual),
            "fixed_plain": canonical_array_sha256(fixed.predictions["plain_lgbm"]),
            "fixed_proposed": canonical_array_sha256(fixed.predictions["proposed"]),
            "rolling_plain": canonical_array_sha256(rolling.predictions["plain_lgbm"]),
            "rolling_proposed": canonical_array_sha256(rolling.predictions["proposed"]),
        },
        "scope_boundary": (
            "only registered training, parameter, weather files and frozen V2 artifacts were read; "
            "data/test_data.csv and preliminary reference traffic were not opened"
        ),
    }
    write_json(output / "results_summary.json", results_summary)

    readme = f"""# Revision-2 experiment evidence

This directory is generated by `experiments/run_reproducibility_evaluation.py` and does
not replace or modify `artifacts/paper_experiments_gpu4_v2`.

Primary evidence: one fixed model is selected and fitted using target dates no
later than 2024-08-11, then evaluated unchanged on 5,110 windows from
2024-08-12 through 2024-08-18. Seven leakage-controlled rolling origins are a
separate temporal-stability analysis.

MASE denominator: for each request and indicator, the mean absolute 168-hour
seasonal difference among finite pairs in that request's 336-hour history.
Zero or unavailable denominators are excluded only from MASE, and every CSV
reports the eligible count.

GPU policy: LightGBM final fits train four target models concurrently on devices
{args.gpu_devices}; early-stopping selections are target-specific and assigned
across those devices. Frozen three-day ablations and asynchronous-missingness
stress tests reuse hash-verified V2 models without refitting.

Reproduce from the repository root:

```bash
PYTHONPATH=.runtime/lightgbm:. python3 experiments/run_reproducibility_evaluation.py
```

The horizon-by-hour heatmap requested by the review is not identifiable here:
all forecasts start at midnight, so horizon equals target hour-of-day. The
export therefore provides horizon-by-indicator and weekday-by-horizon data.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    evidence_paths = [
        "README.md",
        "frozen_reuse_validation.json",
        "fixed_seven_day_report.json",
        "fixed_seven_day_summary.json",
        "rolling_origin_reports.json",
        "rolling_origin_summary.json",
        "seven_day_predictions.csv",
        "seven_day_daily_metrics.csv",
        "standard_metrics.csv",
        "threshold_sensitivity_grid.csv",
        "threshold_sensitivity_sets.csv",
        "threshold_sensitivity_summary.json",
        "per_indicator_ablation.csv",
        "scene_stratification.csv",
        "prb_utility_curve.csv",
        "prb_utility_summary.csv",
        "error_heatmap_horizon_indicator.csv",
        "error_heatmap_weekday_horizon.csv",
        "error_heatmap_identifiability.json",
        "results_summary.json",
    ]
    if async_rows:
        evidence_paths.extend(
            [
                "asynchronous_missingness.csv",
                "asynchronous_missingness_diagnostics.json",
            ]
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "revision2_version": REVISION2_VERSION,
        "inputs": {
            name: {
                "path": str(path.relative_to(project_root())),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in inputs.items()
        },
        "frozen_artifacts": {
            "path": str(frozen_path.relative_to(project_root())),
            "results_sha256": sha256_file(frozen_path / "results.json"),
            "strict_nested_predictions_sha256": sha256_file(
                frozen_path / "strict_nested_oof_predictions.csv"
            ),
        },
        "code": {
            "path": str(Path(__file__).resolve().relative_to(project_root())),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime": runtime,
        "outputs": output_inventory(output, evidence_paths),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(results_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
