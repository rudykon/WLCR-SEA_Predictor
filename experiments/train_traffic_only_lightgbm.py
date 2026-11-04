from __future__ import annotations

"""Four-GPU 165-feature Standard-stat LightGBM paper comparator.

The model uses only target-cell traffic statistics, observation masks, and the
forecast horizon. It never opens finals test traffic, parameter.csv, or
weather.csv. The August holdout is already studied, so results are exploratory.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from Model.traffic_window_forecasting import build_training_backtests, read_traffic
from experiments.lightgbm_experiment_helpers import (
    build_standard_stat_matrix,
    standard_stat_feature_names,
)
from experiments.train_lightgbm_baseline import (
    EARLY_STOPPING_ROUNDS,
    MIN_BOOST_ROUNDS,
    MODEL_PARAMS,
    configured_gpu_devices,
    matrix_fingerprint,
    predict_boosters,
    sha256_file,
    train_or_load_boosters,
)
from experiments.run_reproducibility_evaluation import bundle_from_examples, standard_metric_rows
from experiments.run_traffic_only_baseline_evaluation import combined_summary
from experiments.run_feature_ablation_evaluation import canonical_sha256, matrix_partition, split_examples
from experiments.run_seasonal_anchor_ablations import select_baseline_for_inner

SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "standard_stat_lightgbm_reproduction_v1"
OUTPUT_ROOT = Path("artifacts/reproduction")
DEFAULT_OUTPUT = OUTPUT_ROOT / "lightgbm/standard_stat"
DEFAULT_ROUND_CAP = 10000
METHOD = "standard_stat_traffic_only_165d"
EXPECTED_TRAIN_SHA256 = "d274407a3db51ba4871851ab447bcc75202bb567337464d85ea280662f3bf1da"


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


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def output_manifest(directory: Path) -> dict[str, object]:
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(directory)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"schema_version": SCHEMA_VERSION, "files": files}


def selection_worker(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    columns: np.ndarray,
    metric: int,
    device: int,
    round_cap: int,
    output: str,
) -> None:
    params = dict(MODEL_PARAMS)
    params["gpu_device_id"] = int(device)
    params["num_threads"] = max(1, int(MODEL_PARAMS["num_threads"]) // 4)
    train_mask = np.isfinite(train_y)
    valid_mask = np.isfinite(valid_y)
    booster = lgb.train(
        params,
        lgb.Dataset(train_x[train_mask][:, columns], label=train_y[train_mask]),
        num_boost_round=int(round_cap),
        valid_sets=[lgb.Dataset(valid_x[valid_mask][:, columns], label=valid_y[valid_mask])],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best = int(booster.best_iteration or booster.current_iteration())
    atomic_json(
        Path(output),
        {
            "metric": int(metric),
            "gpu_device_id": int(device),
            "best_iteration": best,
            "selected_rounds": max(MIN_BOOST_ROUNDS, best),
            "round_cap": int(round_cap),
            "fraction_of_cap": float(best / round_cap),
        },
    )


def select_rounds(
    train,
    valid,
    columns: tuple[np.ndarray, ...],
    round_cap: int,
    output: Path,
) -> tuple[tuple[int, ...], list[dict[str, object]]]:
    devices = configured_gpu_devices()
    if devices[:4] != [0, 1, 2, 3]:
        raise ValueError("set PAPER_GPU_DEVICES=0,1,2,3")
    selection_dir = output / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    paths = [selection_dir / f"metric_{metric}.json" for metric in range(4)]
    context = mp.get_context("spawn")
    processes = []
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
                round_cap,
                str(paths[metric]),
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures or any(not path.is_file() for path in paths):
        raise RuntimeError(f"round selection failed: {failures}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return tuple(int(row["selected_rounds"]) for row in reports), reports


def prediction_array(rows, windows: int) -> np.ndarray:
    values = np.asarray([row.metrics for row in rows], dtype=np.float32)
    expected = windows * 24
    if values.shape != (expected, 4):
        raise ValueError(f"unexpected prediction shape: {values.shape}")
    values = values.reshape(windows, 24, 4)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("predictions must be finite and strictly positive")
    return values


def macro_metrics(bundle) -> dict[str, float]:
    row = next(
        value
        for value in standard_metric_rows(bundle)
        if value["method"] == METHOD
        and value["filter"] == "complete_targets_unfiltered"
        and value["indicator"] == "macro_mean"
    )
    task = combined_summary(bundle, METHOD, 165)
    return {
        "macro_wape": float(row["wape"]),
        "macro_mase": float(row["mase"]),
        "macro_smape": float(row["smape"]),
        "macro_mae": float(row["mae"]),
        "macro_rmse": float(row["rmse"]),
        "legacy_holdout_filtered_mapeauc_not_primary": float(task["mape_auc"]),
    }


def run(args: argparse.Namespace) -> int:
    if args.round_cap < DEFAULT_ROUND_CAP:
        raise ValueError(f"round cap must be at least {DEFAULT_ROUND_CAP}")
    root = project_root()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve(strict=False)
    allowed = (root / OUTPUT_ROOT).resolve(strict=False)
    if output != allowed and not output.is_relative_to(allowed):
        raise ValueError("output must remain under artifacts/reproduction")
    output.mkdir(parents=True, exist_ok=True)

    train_path = (root / "data/train_data.csv").resolve(strict=True)
    if not train_path.is_relative_to(root) or sha256_file(train_path) != EXPECTED_TRAIN_SHA256:
        raise ValueError("registered training data SHA256 mismatch")
    input_before = sha256_file(train_path)
    started = time.perf_counter()
    examples = build_training_backtests(read_traffic(train_path))
    split = split_examples(examples)
    final_examples = split["final_examples"]
    holdout_examples = split["holdout_examples"]
    baseline, baseline_report = select_baseline_for_inner(split["inner_examples"])

    final = build_standard_stat_matrix(final_examples, baseline, {})
    holdout = build_standard_stat_matrix(holdout_examples, baseline, {})
    fit, inner = matrix_partition(final, split["fit_dates"], split["inner_dates"])
    names = standard_stat_feature_names(final_examples[0], {})
    suffixes = tuple(f"_m{metric}" for metric in range(4))
    selected = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if name == "horizon" or name.endswith(suffixes)
        ],
        dtype=np.int64,
    )
    if len(selected) != 165:
        raise ValueError(f"expected exactly 165 traffic-only Standard-stat features, found {len(selected)}")
    columns = tuple(selected.copy() for _ in range(4))
    selected_names = [names[int(index)] for index in selected]
    forbidden = [
        name
        for name in selected_names
        if name not in {"horizon"}
        and not any(name.endswith(f"_m{metric}") for metric in range(4))
    ]
    if forbidden:
        raise ValueError(f"forbidden feature names: {forbidden}")

    rounds, selection = select_rounds(fit, inner, columns, args.round_cap, output)
    cache = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "variant": METHOD,
        "training_matrix_sha256": matrix_fingerprint(final),
        "selected_feature_schema_sha256": canonical_sha256(selected_names),
        "rounds": list(rounds),
        "round_cap": int(args.round_cap),
        "model_params": dict(MODEL_PARAMS),
        "code_sha256": sha256_file(Path(__file__)),
    }
    boosters, training_seconds, model_bytes = train_or_load_boosters(
        final, columns, rounds, output / "models", cache
    )
    predictions, prediction_seconds = predict_boosters(boosters, holdout, columns)
    array = prediction_array(predictions, len(holdout_examples))
    temporary = output / f".holdout_predictions.{os.getpid()}.npy"
    np.save(temporary, array, allow_pickle=False)
    temporary.replace(output / "holdout_predictions.npy")

    order_rows = [
        {
            "window_index": index,
            "cell": example.window.cell,
            "target_start": example.window.target_start.isoformat(sep=" "),
        }
        for index, example in enumerate(holdout_examples)
    ]
    atomic_csv(output / "holdout_order.csv", order_rows)
    bundle = bundle_from_examples(
        "revision6_standard_stat_traffic_only", holdout_examples, {METHOD: predictions}
    )
    input_after = sha256_file(train_path)
    if input_before != input_after:
        raise RuntimeError("registered training data changed")
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "method": METHOD,
        "information_class": "target-cell traffic + masks + forecast horizon",
        "feature_count": 165,
        "selected_feature_names": selected_names,
        "selected_rounds": list(rounds),
        "selection": selection,
        "all_targets_early_stopped_before_cap": all(
            int(row["best_iteration"]) < args.round_cap for row in selection
        ),
        "metrics": macro_metrics(bundle),
        "training_seconds": float(training_seconds),
        "prediction_seconds": float(prediction_seconds),
        "model_bytes": int(model_bytes),
        "seasonal_baseline_selection": baseline_report["selected"],
        "holdout_windows": len(holdout_examples),
        "finals_test_opened": False,
        "parameter_features_used": False,
        "weather_features_used": False,
        "calendar_features_used": False,
        "cell_id_or_coordinates_used": False,
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    atomic_json(output / "summary.json", report)
    atomic_json(output / "manifest.json", output_manifest(output))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    value.add_argument("--round-cap", type=int, default=DEFAULT_ROUND_CAP)
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
