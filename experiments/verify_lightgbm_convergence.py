from __future__ import annotations

"""Run the revision-5 convergence audit for the Standard-stat LightGBM.

Only the registered training, parameter, and weather files are resolved by the
shared paper helpers. Finals test traffic is never opened. Four target models
are selected in parallel on GPUs 0--3 and then refitted on the frozen final-fit
dates before evaluation on the unchanged seven-day holdout.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from Model.traffic_window_forecasting import build_training_backtests, read_traffic
from Model.lightgbm_feature_baseline import load_parameters
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
from experiments.run_feature_ablation_evaluation import (
    canonical_sha256,
    matrix_partition,
    split_examples,
    write_json_atomic,
)
from experiments.run_seasonal_anchor_ablations import registered_inputs, select_baseline_for_inner


DEFAULT_ROUND_CAP = 5000
OUTPUT_ROOT = Path("artifacts/revision5")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def worker(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
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
        lgb.Dataset(train_x[train_mask], label=train_y[train_mask]),
        num_boost_round=int(round_cap),
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
            "metric": int(metric),
            "gpu_device_id": int(device),
            "best_iteration": best,
            "selected_rounds": max(MIN_BOOST_ROUNDS, best),
            "round_cap": int(round_cap),
            "distance_from_cap": int(round_cap - best),
            "fraction_of_cap": float(best / round_cap),
        },
    )


def select_rounds(
    train,
    valid,
    round_cap: int,
) -> tuple[tuple[int, ...], list[dict[str, object]]]:
    devices = configured_gpu_devices()
    if devices[:4] != [0, 1, 2, 3]:
        raise ValueError("set PAPER_GPU_DEVICES=0,1,2,3")
    scratch = project_root() / OUTPUT_ROOT / f"scratch_standard_stat_{round_cap}"
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
                round_cap,
                str(paths[metric]),
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures or any(not path.exists() for path in paths):
        raise RuntimeError(f"round-cap convergence selection failed: {failures}")
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


def run(round_cap: int) -> dict[str, object]:
    if round_cap < DEFAULT_ROUND_CAP:
        raise ValueError(f"round cap must be at least {DEFAULT_ROUND_CAP}")
    started = time.perf_counter()
    root = project_root()
    inputs = registered_inputs()
    rows = read_traffic(inputs["train"])
    examples = build_training_backtests(rows)
    split = split_examples(examples)
    final_examples = split["final_examples"]
    holdout_examples = split["holdout_examples"]
    parameters = load_parameters(inputs["parameter"])
    baseline, baseline_report = select_baseline_for_inner(split["inner_examples"])
    final = build_standard_stat_matrix(final_examples, baseline, parameters)
    holdout = build_standard_stat_matrix(holdout_examples, baseline, parameters)
    fit, inner = matrix_partition(final, split["fit_dates"], split["inner_dates"])
    names = standard_stat_feature_names(
        final_examples[0], parameters.get(final_examples[0].window.cell, {})
    )
    if len(names) != 175:
        raise ValueError("expected 175 standard-stat features")
    columns = tuple(np.arange(175, dtype=np.int64) for _ in range(4))
    rounds, selection = select_rounds(fit, inner, round_cap)
    variant = f"standard_stat_lgbm_175d_roundcap{round_cap}"
    model_dir = OUTPUT_ROOT / "models" / variant
    cache = {
        "schema_version": 1,
        "experiment_version": "manuscript_revision5_v1",
        "variant": variant,
        "training_matrix_sha256": matrix_fingerprint(final),
        "feature_schema_sha256": canonical_sha256(list(names)),
        "rounds": list(rounds),
        "round_cap": int(round_cap),
        "model_params": dict(MODEL_PARAMS),
        "code_sha256": sha256_file(Path(__file__)),
    }
    boosters, training_seconds, model_bytes = train_or_load_boosters(
        final, columns, rounds, root / model_dir, cache
    )
    predictions, prediction_seconds = predict_boosters(boosters, holdout, columns)
    bundle = bundle_from_examples(
        f"revision5_standard_stat_roundcap{round_cap}",
        holdout_examples,
        {variant: predictions},
    )
    metrics = macro_metrics(bundle, variant)
    previous_path = root / "artifacts/revision4/standard_stat_roundcap_sensitivity.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "experiment_version": "manuscript_revision5_v1",
        "round_cap": int(round_cap),
        "selected_rounds": list(rounds),
        "selection": selection,
        "all_targets_early_stopped_before_cap": all(
            int(row["best_iteration"]) < round_cap for row in selection
        ),
        "targets_within_5pct_of_cap": [
            int(row["metric"])
            for row in selection
            if float(row["fraction_of_cap"]) >= 0.95
        ],
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "model_bytes": model_bytes,
        "metrics": metrics,
        "previous_roundcap3000": {
            "selected_rounds": previous["selected_rounds"],
            "metrics": previous["metrics"],
        },
        "delta_vs_roundcap3000": {
            key: metrics[key] - float(previous["metrics"][key]) for key in metrics
        },
        "seasonal_selection": baseline_report["selected"],
        "registered_input_sha256": {
            name: sha256_file(path) for name, path in inputs.items()
        },
        "finals_test_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = root / OUTPUT_ROOT / f"standard_stat_convergence_{round_cap}.json"
    write_json_atomic(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-cap", type=int, default=DEFAULT_ROUND_CAP)
    args = parser.parse_args()
    run(args.round_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
