from __future__ import annotations

"""Train the fixed 73-feature traffic-only LightGBM paper comparator.

The comparator uses only the prediction horizon plus the four 18-feature
traffic/mask blocks derived from a sealed 336-hour request.  It is fitted on
the paper's August 3--11 final-training layer with the predeclared target
rounds, and it writes a self-contained, hash-checked model directory under
``artifacts/reproduction``.  No test traffic, parameter file, weather file,
cell identifier, or external asset is read.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import time
from datetime import date
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np

from Model.lightgbm_feature_baseline import build_matrix
from Model.traffic_window_forecasting import OUTPUT_FLOOR, build_training_backtests, read_traffic
from experiments.run_seasonal_anchor_ablations import select_baseline_for_inner
from experiments.train_lightgbm_baseline import MODEL_PARAMS, feature_names, sha256_file, subset_matrix


SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "traffic_only_73d_reproduction_v1"
OUTPUT_ROOT = Path("artifacts/reproduction")
DEFAULT_OUTPUT = OUTPUT_ROOT / "lightgbm/traffic_only_73d"
TRAIN_FILE = Path("data/train_data.csv")
TRAIN_SHA256 = "d274407a3db51ba4871851ab447bcc75202bb567337464d85ea280662f3bf1da"
TARGET_ROUNDS = (341, 332, 742, 678)
MODEL_SEED = 42
INPUT_HOURS = 336
FORECAST_HOURS = 24
TARGET_COUNT = 4


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_output(value: str | Path) -> Path:
    root = project_root()
    output = Path(value)
    if not output.is_absolute():
        output = root / output
    output = output.resolve(strict=False)
    allowed = (root / OUTPUT_ROOT).resolve(strict=False)
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError(f"output must be a new subdirectory under {OUTPUT_ROOT}")
    return output


def parse_devices(value: str) -> tuple[int, int, int, int]:
    devices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(devices) != TARGET_COUNT or len(set(devices)) != TARGET_COUNT or min(devices) < 0:
        raise ValueError("--gpu-devices must list four distinct non-negative GPU indices")
    return devices  # type: ignore[return-value]


def select_columns(names: Sequence[str]) -> np.ndarray:
    columns = np.asarray(
        [index for index, name in enumerate(names) if name == "horizon" or index >= 16],
        dtype=np.int64,
    )
    if columns.tolist() != [0, *range(16, 88)]:
        raise ValueError("the registered 73-feature traffic-only schema changed")
    return columns


def worker(
    features: np.ndarray,
    targets: np.ndarray,
    columns: np.ndarray,
    metric: int,
    device: int,
    output: str,
) -> None:
    params = dict(MODEL_PARAMS)
    params.update(
        {
            "seed": MODEL_SEED,
            "feature_fraction_seed": MODEL_SEED,
            "bagging_seed": MODEL_SEED,
            "data_random_seed": MODEL_SEED,
            "gpu_device_id": int(device),
            "num_threads": 1,
        }
    )
    observed = np.isfinite(targets)
    booster = lgb.train(
        params,
        lgb.Dataset(features[observed][:, columns], label=targets[observed]),
        num_boost_round=TARGET_ROUNDS[metric],
    )
    booster.save_model(output)


def train_boosters(
    matrix, columns: np.ndarray, devices: tuple[int, int, int, int], output: Path
) -> tuple[list[lgb.Booster], float]:
    temporary = [output / f".metric_{metric}.{os.getpid()}.tmp" for metric in range(TARGET_COUNT)]
    paths = [output / f"metric_{metric}.txt" for metric in range(TARGET_COUNT)]
    context = mp.get_context("spawn")
    started = time.perf_counter()
    processes: list[mp.Process] = []
    try:
        for metric in range(TARGET_COUNT):
            process = context.Process(
                target=worker,
                args=(matrix.features, matrix.targets[:, metric], columns, metric, devices[metric], str(temporary[metric])),
            )
            process.start()
            processes.append(process)
        for metric, process in enumerate(processes):
            process.join()
            if process.exitcode != 0 or not temporary[metric].is_file():
                raise RuntimeError(f"traffic-only LightGBM target {metric} failed: {process.exitcode}")
        for source, target in zip(temporary, paths):
            source.replace(target)
    except Exception:
        for path in temporary:
            path.unlink(missing_ok=True)
        raise
    return [lgb.Booster(model_file=str(path)) for path in paths], time.perf_counter() - started


def raw_predictions(boosters: Sequence[lgb.Booster], matrix, columns: np.ndarray) -> np.ndarray:
    values = np.empty((len(matrix.actuals), TARGET_COUNT), dtype=np.float64)
    selected = np.ascontiguousarray(matrix.features[:, columns], dtype=np.float32)
    for metric, booster in enumerate(boosters):
        values[:, metric] = np.maximum(np.expm1(booster.predict(selected)), OUTPUT_FLOOR)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise FloatingPointError("traffic-only LightGBM produced invalid predictions")
    return values.reshape(-1, FORECAST_HOURS, TARGET_COUNT).astype(np.float32)


def macro_wape(actual: np.ndarray, prediction: np.ndarray) -> tuple[float, list[float]]:
    valid = np.isfinite(actual)
    values = []
    for metric in range(TARGET_COUNT):
        mask = valid[..., metric]
        denominator = float(np.sum(np.abs(actual[..., metric][mask])))
        values.append(float(np.sum(np.abs(actual[..., metric][mask] - prediction[..., metric][mask])) / denominator))
    return float(np.mean(values)), values


def run(args: argparse.Namespace) -> int:
    output = resolve_output(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to mix a new run with existing output: {output}")
    output.mkdir(parents=True)
    devices = parse_devices(args.gpu_devices)
    root = project_root()
    train_path = (root / TRAIN_FILE).resolve(strict=True)
    input_before = sha256_file(train_path)
    if input_before != TRAIN_SHA256:
        raise ValueError("registered training data SHA256 mismatch")

    examples = build_training_backtests(read_traffic(train_path))
    if len(examples) != 11_685:
        raise ValueError(f"expected 11,685 continuous windows, found {len(examples)}")
    fit_dates = tuple(date(2024, 8, day) for day in range(3, 10))
    inner_dates = tuple(date(2024, 8, day) for day in range(10, 12))
    final_dates = (*fit_dates, *inner_dates)
    holdout_dates = tuple(date(2024, 8, day) for day in range(12, 19))

    def select_dates(dates: Sequence[date]):
        included = set(dates)
        return [item for item in examples if item.window.target_start.date() in included]

    fit_examples = select_dates(fit_dates)
    inner_examples = select_dates(inner_dates)
    final_examples = select_dates(final_dates)
    holdout_examples = select_dates(holdout_dates)
    if (len(fit_examples), len(inner_examples), len(final_examples), len(holdout_examples)) != (5115, 1460, 6575, 5110):
        raise ValueError("the registered paper split changed")

    baseline, baseline_report = select_baseline_for_inner(inner_examples)
    selected_baseline = baseline_report["selected"]
    if selected_baseline["name"] != "weekly_median_s097":
        raise ValueError("the paper seasonal baseline selection changed")
    final_matrix = build_matrix(final_examples, baseline, {}, {})
    holdout_matrix = build_matrix(holdout_examples, baseline, {}, {})
    names = feature_names(final_examples[0], baseline, {}, {})
    columns = select_columns(names)
    boosters, training_seconds = train_boosters(final_matrix, columns, devices, output)
    prediction = raw_predictions(boosters, holdout_matrix, columns)
    actual = np.asarray([[row.metrics for row in example.actuals] for example in holdout_examples], dtype=object)
    actual_array = np.asarray(
        [[[np.nan if value is None else float(value) for value in row] for row in request] for request in actual],
        dtype=np.float32,
    )
    score, per_indicator = macro_wape(actual_array, prediction)
    temporary = output / f".holdout_predictions.{os.getpid()}.npy"
    np.save(temporary, prediction, allow_pickle=False)
    temporary.replace(output / "holdout_predictions.npy")
    order_rows = [
        {"window_index": index, "cell": item.window.cell, "target_start": item.window.target_start.isoformat(sep=" ")}
        for index, item in enumerate(holdout_examples)
    ]
    atomic_csv(output / "holdout_order.csv", order_rows)
    config = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "variant": "traffic_only_73d",
        "seed": MODEL_SEED,
        "train_sha256": input_before,
        "rounds": list(TARGET_ROUNDS),
        "selected_feature_names": [list(names[int(index)] for index in columns)] * TARGET_COUNT,
        "selected_feature_indices": columns.tolist(),
        "seasonal_baseline": selected_baseline,
        "model_params": {**MODEL_PARAMS, "gpu_device_id": "one target model per requested GPU"},
    }
    model_paths = [output / f"metric_{metric}.txt" for metric in range(TARGET_COUNT)]
    cache_manifest = {
        "schema_version": SCHEMA_VERSION,
        "cache_config": config,
        "cache_config_sha256": canonical_sha256(config),
        "models": [
            {"file": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in model_paths
        ],
    }
    atomic_json(output / "cache_manifest.json", cache_manifest)
    input_after = sha256_file(train_path)
    if input_after != input_before:
        raise RuntimeError("registered training data changed during training")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "method": "traffic_only_73d",
        "feature_count": int(len(columns)),
        "holdout_windows": len(holdout_examples),
        "rounds": list(TARGET_ROUNDS),
        "macro_wape": score,
        "per_indicator_wape": per_indicator,
        "training_seconds": training_seconds,
        "model_bytes": int(sum(path.stat().st_size for path in model_paths)),
        "registered_train_sha256_before": input_before,
        "registered_train_sha256_after": input_after,
        "seasonal_baseline": selected_baseline,
        "information_class": "target-cell traffic, observation masks, and forecast horizon only",
    }
    atomic_json(output / "summary.json", summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": [
            {"path": str(path.relative_to(output)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    value.add_argument("--gpu-devices", default="0,1,2,3")
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
