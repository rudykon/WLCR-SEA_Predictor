from __future__ import annotations

"""Audit whether the revision-4 standard-stat baseline is round-cap limited."""

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from Model.traffic_window_forecasting import build_training_backtests, read_traffic
from Model.lightgbm_feature_baseline import build_matrix, load_parameters, load_weather
from experiments.lightgbm_experiment_helpers import build_standard_stat_matrix, standard_stat_feature_names
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
from experiments.run_feature_ablation_evaluation import (
    canonical_sha256,
    matrix_partition,
    split_examples,
    write_json_atomic,
)
from experiments.run_seasonal_anchor_ablations import registered_inputs, select_baseline_for_inner


MAX_ROUNDS = 3000
OUTPUT = Path("artifacts/revision4/standard_stat_roundcap_sensitivity.json")
MODEL_DIR = Path("artifacts/revision4/models/standard_stat_lgbm_175d_roundcap3000")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def worker(
    train_x,
    train_y,
    valid_x,
    valid_y,
    metric: int,
    device: int,
    output: str,
) -> None:
    params = dict(MODEL_PARAMS)
    params["gpu_device_id"] = device
    params["num_threads"] = max(1, int(MODEL_PARAMS["num_threads"]) // 4)
    train_mask = np.isfinite(train_y)
    valid_mask = np.isfinite(valid_y)
    booster = lgb.train(
        params,
        lgb.Dataset(train_x[train_mask], label=train_y[train_mask]),
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(valid_x[valid_mask], label=valid_y[valid_mask])],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best = int(booster.best_iteration or booster.current_iteration())
    write_json_atomic(
        Path(output),
        {
            "metric": metric,
            "gpu_device_id": device,
            "best_iteration": best,
            "selected_rounds": max(MIN_BOOST_ROUNDS, best),
            "cap": MAX_ROUNDS,
        },
    )


def select_rounds(train, valid) -> tuple[tuple[int, ...], list[dict[str, object]]]:
    devices = configured_gpu_devices()
    if devices[:4] != [0, 1, 2, 3]:
        raise ValueError("set PAPER_GPU_DEVICES=0,1,2,3")
    scratch = project_root() / "artifacts/revision4/scratch_standard_stat_3000"
    scratch.mkdir(parents=True, exist_ok=True)
    paths = [scratch / f"metric_{metric}.{os.getpid()}.json" for metric in range(4)]
    context = mp.get_context("spawn")
    processes = []
    for metric in range(4):
        process = context.Process(
            target=worker,
            args=(
                train.features,
                train.targets[:, metric],
                valid.features,
                valid.targets[:, metric],
                metric,
                devices[metric],
                str(paths[metric]),
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures or any(not path.exists() for path in paths):
        raise RuntimeError(f"round-cap sensitivity selection failed: {failures}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path in paths:
        path.unlink()
    try:
        scratch.rmdir()
    except OSError:
        pass
    return tuple(int(row["selected_rounds"]) for row in reports), reports


def macro_metrics(bundle, method: str) -> dict[str, float]:
    row = next(
        value
        for value in standard_metric_rows(bundle)
        if value["method"] == method
        and value["filter"] == "complete_targets_unfiltered"
        and value["indicator"] == "macro_mean"
    )
    task = combined_summary(bundle, method, 175)
    return {
        "unfiltered_wape": float(row["wape"]),
        "unfiltered_mase": float(row["mase"]),
        "unfiltered_smape": float(row["smape"]),
        "unfiltered_mae": float(row["mae"]),
        "unfiltered_rmse": float(row["rmse"]),
        "ths_mapeauc": float(task["mape_auc"]),
        "filtered_mean_mape": float(task["mean_mape"]),
    }


def run() -> dict[str, object]:
    started = time.perf_counter()
    root = project_root()
    inputs = registered_inputs()
    rows = read_traffic(inputs["train"])
    examples = build_training_backtests(rows)
    split = split_examples(examples)
    final_examples = split["final_examples"]
    holdout_examples = split["holdout_examples"]
    parameters = load_parameters(inputs["parameter"])
    weather = load_weather(inputs["weather"])
    baseline, _ = select_baseline_for_inner(split["inner_examples"])
    final = build_standard_stat_matrix(final_examples, baseline, parameters)
    holdout = build_standard_stat_matrix(holdout_examples, baseline, parameters)
    fit, inner = matrix_partition(final, split["fit_dates"], split["inner_dates"])
    names = standard_stat_feature_names(
        final_examples[0], parameters.get(final_examples[0].window.cell, {})
    )
    if len(names) != 175:
        raise ValueError("expected 175 standard-stat features")
    columns = tuple(np.arange(175, dtype=np.int64) for _ in range(4))
    rounds, selection = select_rounds(fit, inner)
    cache = {
        "schema_version": 1,
        "variant": "standard_stat_lgbm_175d_roundcap3000",
        "training_matrix_sha256": matrix_fingerprint(final),
        "feature_schema_sha256": canonical_sha256(list(names)),
        "rounds": list(rounds),
        "round_cap": MAX_ROUNDS,
        "model_params": dict(MODEL_PARAMS),
        "code_sha256": sha256_file(Path(__file__)),
    }
    boosters, training_seconds, model_bytes = train_or_load_boosters(
        final, columns, rounds, root / MODEL_DIR, cache
    )
    predictions, prediction_seconds = predict_boosters(boosters, holdout, columns)
    method = "standard_stat_lgbm_175d_roundcap3000"
    bundle = bundle_from_examples(
        "revision4_standard_stat_roundcap_sensitivity",
        holdout_examples,
        {method: predictions},
    )
    primary_path = root / "artifacts/revision4/revision4_single_model_results.csv"
    import csv

    with primary_path.open("r", encoding="utf-8", newline="") as handle:
        primary = next(
            row
            for row in csv.DictReader(handle)
            if row["method"] == "standard_stat_lgbm_175d_seed42"
        )
    metrics = macro_metrics(bundle, method)
    report = {
        "schema_version": 1,
        "round_cap": MAX_ROUNDS,
        "selected_rounds": list(rounds),
        "selection": selection,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "model_bytes": model_bytes,
        "metrics": metrics,
        "original_roundcap1500": {
            key: float(primary[key])
            for key in (
                "unfiltered_wape",
                "unfiltered_mase",
                "unfiltered_smape",
                "unfiltered_mae",
                "unfiltered_rmse",
                "ths_mapeauc",
                "filtered_mean_mape",
            )
        },
        "delta_3000_minus_1500": {
            key: metrics[key] - float(primary[key]) for key in metrics
        },
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_rule": "Use the original equal-budget baseline unless the sensitivity audit materially changes the conclusion; report cap sensitivity as a limitation.",
    }
    write_json_atomic(root / OUTPUT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
