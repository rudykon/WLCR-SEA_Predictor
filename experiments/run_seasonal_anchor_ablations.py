from __future__ import annotations

"""Fit the five fixed-seven-day WLCR ablations requested by review 2.

Only the registered training, parameter, and weather assets are read.  The
finals test traffic is outside this script's allowlist.  Each ablation selects
its four target-specific boosting rounds independently on 2024-08-10/11, then
fits on 2024-08-03..11 and evaluates one unchanged model on 2024-08-12..18.
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import time
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from Model.traffic_window_forecasting import (
    BacktestExample,
    build_training_backtests,
    read_traffic,
)
from Model.lightgbm_feature_baseline import MatrixBundle, build_matrix, load_parameters, load_weather
from experiments.train_lightgbm_baseline import (
    BOOTSTRAP_SEED,
    EARLY_STOPPING_ROUNDS,
    MAX_BOOST_ROUNDS,
    MIN_BOOST_ROUNDS,
    MODEL_PARAMS,
    MODEL_SEED,
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
    OFFICIAL_THRESHOLDS,
    PredictionBundle,
    bundle_from_examples,
    cluster_bootstrap_combined_delta,
    cluster_bootstrap_indicator_delta,
    indicator_hit_auc,
    load_verified_boosters,
    metric_values,
    official_mask,
    select_baseline_for_inner,
    threshold_score,
    write_csv,
    write_json,
)


SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "fixed_seven_day_ablations_v1"
OUTPUT_ROOT = Path("artifacts/revision2")
REFERENCE_ROOT = OUTPUT_ROOT / "models/fixed_seven_day_holdout"
MODEL_ROOT = REFERENCE_ROOT / "seven_day_ablations_v1"
REGISTERED_INPUTS = {
    "train": (
        Path("data/train_data.csv"),
        "d274407a3db51ba4871851ab447bcc75202bb567337464d85ea280662f3bf1da",
    ),
    "parameter": (
        Path("data/parameter.csv"),
        "d8e02302042e4fd91945a59a53c0c8d730a18f4c6c7b08344a4c8389a866cd77",
    ),
    "weather": (
        Path("data/weather.csv"),
        "92a2d55c44d69e6bcae3001c20ee7a0034e2035423b41a299d2922d17c280a44",
    ),
}
VARIANTS = {
    "no_seasonal_anchor": "no_baseline",
    "no_missingness": "no_missingness",
    "no_static": "no_static",
    "no_weather": "no_weather",
    "target_only": "target_only",
}
EXPECTED_REFERENCE = {
    "plain_lgbm": 0.7623025777310712,
    "full": 0.7839242949385123,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_sha256(payload: object) -> str:
    data = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def registered_inputs() -> dict[str, Path]:
    root = project_root()
    resolved: dict[str, Path] = {}
    for name, (relative, expected) in REGISTERED_INPUTS.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"invalid registered input {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"registered input SHA256 mismatch: {relative}")
        resolved[name] = path
    return resolved


def select_metric_worker(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    valid_features: np.ndarray,
    valid_targets: np.ndarray,
    columns: np.ndarray,
    metric: int,
    device_id: int,
    max_rounds: int,
    early_stopping_rounds: int,
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
        num_boost_round=int(max_rounds),
        valid_sets=[
            lgb.Dataset(
                valid_features[valid_mask][:, columns],
                label=valid_targets[valid_mask],
                free_raw_data=False,
            )
        ],
        callbacks=[
            lgb.early_stopping(int(early_stopping_rounds), verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    raw_best = int(booster.best_iteration or booster.current_iteration())
    selected = max(MIN_BOOST_ROUNDS, raw_best)
    payload = {
        "metric": int(metric),
        "selection_metric": "inner_log_target_l1",
        "best_iteration": raw_best,
        "selected_rounds": selected,
        "trained_iterations": int(booster.current_iteration()),
        "max_rounds": int(max_rounds),
        "early_stopping_rounds": int(early_stopping_rounds),
        "hit_max_rounds": int(booster.current_iteration()) >= int(max_rounds),
        "best_iteration_at_max": raw_best >= int(max_rounds),
        "gpu_device_id": int(device_id),
    }
    Path(result_path).write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_rounds_parallel(
    train: MatrixBundle,
    valid: MatrixBundle,
    columns: Sequence[np.ndarray],
    variant: str,
    max_rounds: int,
    early_stopping_rounds: int,
) -> tuple[tuple[int, ...], tuple[dict[str, object], ...], float]:
    devices = configured_gpu_devices()
    if len(devices) < 4:
        raise ValueError("the seven-day ablation run requires four GPU devices")
    scratch = project_root() / MODEL_ROOT / variant / "selection_scratch"
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
                devices[metric],
                max_rounds,
                early_stopping_rounds,
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
        raise RuntimeError(f"parallel round selection failed: {failures}")
    diagnostics = tuple(
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    )
    for path in paths:
        path.unlink(missing_ok=True)
    try:
        scratch.rmdir()
    except OSError:
        pass
    rounds = tuple(int(item["selected_rounds"]) for item in diagnostics)
    return rounds, diagnostics, time.perf_counter() - started


def method_summary(
    bundle: PredictionBundle,
    method: str,
    feature_count: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    official, thresholds = official_mask(bundle)
    prediction = bundle.predictions[method]
    auc, rates = threshold_score(
        bundle.actual, prediction, official, OFFICIAL_THRESHOLDS
    )
    combined_error = np.mean(
        np.abs(bundle.actual[official] - prediction[official])
        / bundle.actual[official],
        axis=1,
    )
    indicator_rows: list[dict[str, object]] = []
    for metric, metric_name in enumerate(METRIC_NAMES):
        values = metric_values(
            bundle.actual,
            prediction,
            bundle.mase_scales,
            official,
            metric,
        )
        indicator_auc, indicator_rates = indicator_hit_auc(
            bundle.actual, prediction, official, metric
        )
        indicator_rows.append(
            {
                "method": method,
                "indicator": metric_name,
                "features": feature_count,
                "indicator_hit_auc": indicator_auc,
                "hit_020": indicator_rates[0],
                "hit_030": indicator_rates[1],
                "hit_040": indicator_rates[2],
                "hit_050": indicator_rates[3],
                **values,
            }
        )
    macro = {
        key: float(np.mean([float(row[key]) for row in indicator_rows]))
        for key in ("mae", "rmse", "wape", "smape", "mase")
    }
    summary = {
        "method": method,
        "features": int(feature_count),
        "mape_auc": auc,
        "mean_mape": float(np.mean(combined_error)),
        "hit_020": rates[0],
        "hit_030": rates[1],
        "hit_040": rates[2],
        "hit_050": rates[3],
        "filtered_hours": int(np.sum(official)),
        "complete_target_hours": int(np.sum(np.all(np.isfinite(bundle.actual), axis=1))),
        "total_rows": int(len(bundle.actual)),
        "official_q05_m0": float(thresholds[0]),
        "official_q05_m1": float(thresholds[1]),
        "official_q05_m2": float(thresholds[2]),
        "official_q05_m3": float(thresholds[3]),
        "macro_mae": macro["mae"],
        "macro_rmse": macro["rmse"],
        "macro_wape": macro["wape"],
        "macro_smape": macro["smape"],
        "macro_mase": macro["mase"],
        "wape_ul_active_users": indicator_rows[0]["wape"],
        "wape_dl_active_users": indicator_rows[1]["wape"],
        "wape_dl_prb": indicator_rows[2]["wape"],
        "wape_ul_prb": indicator_rows[3]["wape"],
    }
    return summary, indicator_rows


def cache_config(
    *,
    variant: str,
    final_matrix: MatrixBundle,
    feature_schema: Sequence[str],
    columns: Sequence[np.ndarray],
    rounds: Sequence[int],
    inputs: Mapping[str, Path],
    baseline_payload: Mapping[str, object],
    dates: Mapping[str, Sequence[date]],
    devices: Sequence[int],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "variant": variant,
        "training_matrix_sha256": matrix_fingerprint(final_matrix),
        "feature_schema_sha256": canonical_sha256(list(feature_schema)),
        "selected_feature_names": [
            [feature_schema[int(index)] for index in metric_columns]
            for metric_columns in columns
        ],
        "rounds": [int(value) for value in rounds],
        "model_params": dict(MODEL_PARAMS),
        "model_seed": MODEL_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "gpu_devices": [int(value) for value in devices],
        "dates": {
            name: [str(value) for value in values] for name, values in dates.items()
        },
        "baseline": dict(baseline_payload),
        "inputs": {
            name: {
                "path": str(path.relative_to(project_root())),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in inputs.items()
        },
        "code": {
            "script": sha256_file(Path(__file__)),
            "train_lightgbm_baseline": sha256_file(
                project_root() / "experiments/train_lightgbm_baseline.py"
            ),
            "run_reproducibility_evaluation": sha256_file(
                project_root() / "experiments/run_reproducibility_evaluation.py"
            ),
            "traffic_window_forecasting": sha256_file(project_root() / "Model/traffic_window_forecasting.py"),
            "lightgbm_feature_baseline": sha256_file(
                project_root() / "Model/lightgbm_feature_baseline.py"
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    inputs = registered_inputs()
    hashes_before = {name: sha256_file(path) for name, path in inputs.items()}
    output = project_root() / OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    devices = configured_gpu_devices()
    if devices[:4] != [0, 1, 2, 3]:
        raise ValueError("set PAPER_GPU_DEVICES=0,1,2,3 for the audited four-GPU run")

    training_rows = read_traffic(inputs["train"])
    examples = build_training_backtests(training_rows)
    all_dates = sorted({example.window.target_start.date() for example in examples})
    if len(all_dates) != 16:
        raise ValueError(f"expected 16 target dates, found {len(all_dates)}")
    fit_dates = tuple(all_dates[:7])
    inner_dates = tuple(all_dates[7:9])
    final_dates = tuple(all_dates[:9])
    holdout_dates = tuple(all_dates[9:])

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
    expected_counts = (5115, 1460, 6575, 5110)
    observed_counts = (
        len(fit_examples),
        len(inner_examples),
        len(final_examples),
        len(holdout_examples),
    )
    if observed_counts != expected_counts:
        raise ValueError(f"fixed seven-day window counts differ: {observed_counts}")

    parameters = load_parameters(inputs["parameter"])
    weather = load_weather(inputs["weather"])
    baseline, baseline_report = select_baseline_for_inner(inner_examples)
    selected_baseline = baseline_report["selected"]
    if selected_baseline["name"] != "weekly_median_s097":
        raise ValueError(f"unexpected selected seasonal anchor: {selected_baseline}")

    matrix_started = time.perf_counter()
    final_matrix = build_matrix(final_examples, baseline, parameters, weather)
    target_dates = np.asarray(
        [row.timestamp.date() for row in final_matrix.actuals], dtype=object
    )
    fit_mask = np.asarray([value in set(fit_dates) for value in target_dates])
    inner_mask = np.asarray([value in set(inner_dates) for value in target_dates])
    if np.any(fit_mask & inner_mask) or not np.all(fit_mask | inner_mask):
        raise ValueError("fit and inner masks do not partition final training rows")
    fit_matrix = subset_matrix(final_matrix, fit_mask)
    inner_matrix = subset_matrix(final_matrix, inner_mask)
    holdout_matrix = build_matrix(holdout_examples, baseline, parameters, weather)
    matrix_seconds = time.perf_counter() - matrix_started
    names = tuple(feature_names(final_examples[0], baseline, parameters, weather))
    if len(names) != 88:
        raise ValueError(f"expected the full 88-feature schema, found {len(names)}")

    predictions: dict[str, Sequence] = {}
    reference_columns = {
        "plain_lgbm": tuple(
            feature_columns(names, "plain_lgbm", metric) for metric in range(4)
        ),
        "full": tuple(feature_columns(names, "full", metric) for metric in range(4)),
    }
    reference_dirs = {
        "plain_lgbm": project_root() / REFERENCE_ROOT / "plain_lgbm",
        "full": project_root() / REFERENCE_ROOT / "proposed",
    }
    reference_runtime: dict[str, object] = {}
    for method in ("plain_lgbm", "full"):
        boosters = load_verified_boosters(reference_dirs[method])
        rows, prediction_seconds = predict_boosters(
            boosters, holdout_matrix, reference_columns[method]
        )
        predictions[method] = rows
        reference_runtime[method] = {
            "prediction_seconds": prediction_seconds,
            "model_dir": str(reference_dirs[method].relative_to(project_root())),
        }

    variant_runtime: dict[str, object] = {}
    feature_counts = {
        "plain_lgbm": int(np.mean([len(value) for value in reference_columns["plain_lgbm"]])),
        "full": int(np.mean([len(value) for value in reference_columns["full"]])),
    }
    date_payload = {
        "fit": fit_dates,
        "inner": inner_dates,
        "final_fit": final_dates,
        "holdout": holdout_dates,
    }
    for output_name, internal_name in VARIANTS.items():
        columns = tuple(
            feature_columns(names, internal_name, metric) for metric in range(4)
        )
        rounds, diagnostics, selection_seconds = select_rounds_parallel(
            fit_matrix,
            inner_matrix,
            columns,
            output_name,
            args.max_boost_rounds,
            args.early_stopping_rounds,
        )
        config = cache_config(
            variant=output_name,
            final_matrix=final_matrix,
            feature_schema=names,
            columns=columns,
            rounds=rounds,
            inputs=inputs,
            baseline_payload=selected_baseline,
            dates=date_payload,
            devices=devices,
        )
        model_dir = project_root() / MODEL_ROOT / output_name
        boosters, training_seconds, model_bytes = train_or_load_boosters(
            final_matrix,
            columns,
            rounds,
            model_dir,
            config,
        )
        rows, prediction_seconds = predict_boosters(
            boosters, holdout_matrix, columns
        )
        predictions[output_name] = rows
        feature_counts[output_name] = int(
            round(np.mean([len(value) for value in columns]))
        )
        variant_runtime[output_name] = {
            "internal_variant": internal_name,
            "feature_count": feature_counts[output_name],
            "selected_rounds": list(rounds),
            "round_selection_diagnostics": list(diagnostics),
            "round_selection_seconds": selection_seconds,
            "final_training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds,
            "model_bytes": int(model_bytes),
            "model_dir": str(model_dir.relative_to(project_root())),
            "cache_config_sha256": canonical_sha256(config),
        }

    bundle = bundle_from_examples(
        "fixed_seven_day_ablation_holdout", holdout_examples, predictions
    )
    summaries: dict[str, dict[str, object]] = {}
    indicator_rows_by_method: dict[str, list[dict[str, object]]] = {}
    for method in predictions:
        summary, rows = method_summary(bundle, method, feature_counts[method])
        summaries[method] = summary
        indicator_rows_by_method[method] = rows
    for method, expected in EXPECTED_REFERENCE.items():
        observed = float(summaries[method]["mape_auc"])
        if not math.isclose(observed, expected, abs_tol=1e-12):
            raise ValueError(f"reference score mismatch for {method}: {observed} != {expected}")

    official, _ = official_mask(bundle)
    unique_cells = np.unique(bundle.cells[official])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(
        0,
        len(unique_cells),
        size=(args.bootstrap, len(unique_cells)),
        dtype=np.int32,
    )
    overall_rows: list[dict[str, object]] = []
    per_indicator_rows: list[dict[str, object]] = []
    bootstrap_payload: dict[str, object] = {}
    full_auc = float(summaries["full"]["mape_auc"])
    plain_auc = float(summaries["plain_lgbm"]["mape_auc"])
    for method in predictions:
        summary = dict(summaries[method])
        summary["delta_vs_full"] = float(summary["mape_auc"]) - full_auc
        summary["delta_vs_plain"] = float(summary["mape_auc"]) - plain_auc
        summary.update(
            {
                "bootstrap_vs_full_ci_low": None,
                "bootstrap_vs_full_ci_high": None,
                "bootstrap_vs_full_probability_positive": None,
                "bootstrap_vs_plain_ci_low": None,
                "bootstrap_vs_plain_ci_high": None,
                "bootstrap_vs_plain_probability_positive": None,
                "bootstrap_replicates": args.bootstrap,
                "bootstrap_cluster": "cell (all seven target dates retained within cluster)",
            }
        )
        if method in VARIANTS:
            vs_full = cluster_bootstrap_combined_delta(
                bundle,
                bundle.predictions["full"],
                bundle.predictions[method],
                args.bootstrap,
            )
            vs_plain = cluster_bootstrap_combined_delta(
                bundle,
                bundle.predictions["plain_lgbm"],
                bundle.predictions[method],
                args.bootstrap,
            )
            summary.update(
                {
                    "bootstrap_vs_full_ci_low": vs_full["ci_low"],
                    "bootstrap_vs_full_ci_high": vs_full["ci_high"],
                    "bootstrap_vs_full_probability_positive": vs_full[
                        "probability_positive"
                    ],
                    "bootstrap_vs_plain_ci_low": vs_plain["ci_low"],
                    "bootstrap_vs_plain_ci_high": vs_plain["ci_high"],
                    "bootstrap_vs_plain_probability_positive": vs_plain[
                        "probability_positive"
                    ],
                    "bootstrap_replicates": args.bootstrap,
                    "bootstrap_cluster": "cell (all seven target dates retained within cluster)",
                }
            )
            bootstrap_payload[method] = {
                "vs_full": vs_full,
                "vs_plain_lgbm": vs_plain,
            }
        overall_rows.append(summary)

        full_indicator = {
            row["indicator"]: row for row in indicator_rows_by_method["full"]
        }
        plain_indicator = {
            row["indicator"]: row
            for row in indicator_rows_by_method["plain_lgbm"]
        }
        for metric, row in enumerate(indicator_rows_by_method[method]):
            item = dict(row)
            metric_name = str(item["indicator"])
            item["delta_hit_auc_vs_full"] = float(item["indicator_hit_auc"]) - float(
                full_indicator[metric_name]["indicator_hit_auc"]
            )
            item["delta_hit_auc_vs_plain"] = float(item["indicator_hit_auc"]) - float(
                plain_indicator[metric_name]["indicator_hit_auc"]
            )
            item.update(
                {
                    "bootstrap_vs_full_ci_low": None,
                    "bootstrap_vs_full_ci_high": None,
                    "bootstrap_vs_full_probability_positive": None,
                    "bootstrap_vs_plain_ci_low": None,
                    "bootstrap_vs_plain_ci_high": None,
                    "bootstrap_vs_plain_probability_positive": None,
                    "bootstrap_replicates": args.bootstrap,
                    "bootstrap_cluster": "cell",
                }
            )
            if method in VARIANTS:
                boot_full = cluster_bootstrap_indicator_delta(
                    bundle,
                    bundle.predictions["full"],
                    bundle.predictions[method],
                    metric,
                    official,
                    samples,
                )
                boot_plain = cluster_bootstrap_indicator_delta(
                    bundle,
                    bundle.predictions["plain_lgbm"],
                    bundle.predictions[method],
                    metric,
                    official,
                    samples,
                )
                item.update(
                    {
                        "bootstrap_vs_full_ci_low": boot_full["ci_low"],
                        "bootstrap_vs_full_ci_high": boot_full["ci_high"],
                        "bootstrap_vs_full_probability_positive": boot_full[
                            "probability_positive"
                        ],
                        "bootstrap_vs_plain_ci_low": boot_plain["ci_low"],
                        "bootstrap_vs_plain_ci_high": boot_plain["ci_high"],
                        "bootstrap_vs_plain_probability_positive": boot_plain[
                            "probability_positive"
                        ],
                        "bootstrap_replicates": args.bootstrap,
                        "bootstrap_cluster": "cell",
                    }
                )
            per_indicator_rows.append(item)

    target = bundle.predictions["target_only"]
    full = bundle.predictions["full"]
    target_only_diagnostic = {
        "same_feature_count_as_full": feature_counts["target_only"]
        == feature_counts["full"],
        "numerically_identical_predictions": bool(np.array_equal(target, full)),
        "max_absolute_prediction_difference": float(np.max(np.abs(target - full))),
        "mean_absolute_prediction_difference": float(np.mean(np.abs(target - full))),
        "mape_auc_delta_vs_full": float(summaries["target_only"]["mape_auc"])
        - full_auc,
        "equivalence_statement": (
            "Prediction equivalence is not established without a predeclared equivalence margin; "
            "the paired confidence interval is reported for descriptive comparison."
        ),
    }

    overall_path = output / "seven_day_ablation_overall.csv"
    indicator_path = output / "seven_day_ablation_per_indicator.csv"
    report_path = output / "seven_day_ablation_report.json"
    write_csv(overall_path, overall_rows)
    write_csv(indicator_path, per_indicator_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "scope_boundary": (
            "Only registered train_data.csv, parameter.csv, weather.csv and verified "
            "fixed-seven-day Full/Plain model caches were read; data/test_data.csv and "
            "preliminary reference traffic were not opened."
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
            "holdout_rows": len(bundle.actual),
            "filtered_hours": int(np.sum(official)),
            "pooled_filter": "one official linear 5% target quantile mask over all seven days",
            "round_selection": (
                "each ablation independently early-stopped on 2024-08-10/11 log1p-target L1"
            ),
        },
        "seasonal_selection": baseline_report,
        "model_configuration": {
            "params": dict(MODEL_PARAMS),
            "max_boost_rounds": args.max_boost_rounds,
            "early_stopping_rounds": args.early_stopping_rounds,
            "minimum_rounds": MIN_BOOST_ROUNDS,
            "model_seed": MODEL_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": args.bootstrap,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
            "gpu_devices": devices,
            "gpu_assignment": "one target per GPU for both early stopping and final fitting",
            "matrix_construction_seconds": matrix_seconds,
            "reference_prediction": reference_runtime,
            "variants": variant_runtime,
            "total_seconds_before_write": time.perf_counter() - started,
        },
        "overall": {row["method"]: row for row in overall_rows},
        "bootstrap": bootstrap_payload,
        "target_only_vs_full": target_only_diagnostic,
        "registered_input_sha256_before": hashes_before,
    }
    write_json(report_path, report)

    hashes_after = {name: sha256_file(path) for name, path in inputs.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("registered inputs changed during the experiment")
    manifest_path = output / "seven_day_ablation_manifest.json"
    model_manifests = sorted((project_root() / MODEL_ROOT).glob("*/cache_manifest.json"))
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
            for path in (overall_path, indicator_path, report_path)
        ],
        "model_manifests": [
            {
                "path": str(path.relative_to(project_root())),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in model_manifests
        ],
    }
    write_json(manifest_path, manifest)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--max-boost-rounds", type=int, default=MAX_BOOST_ROUNDS)
    parser.add_argument(
        "--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS
    )
    args = parser.parse_args()
    if args.bootstrap != 5000:
        raise ValueError("the audited paper ablation requires exactly 5,000 replicates")
    if args.max_boost_rounds < MAX_BOOST_ROUNDS:
        raise ValueError("max boosting rounds cannot be below the frozen 1,500-round budget")
    if args.early_stopping_rounds != EARLY_STOPPING_ROUNDS:
        raise ValueError("the audited paper ablation requires 60 early-stopping rounds")
    report = run(args)
    print(
        json.dumps(
            {
                "status": "ok",
                "overall": {
                    name: payload["mape_auc"]
                    for name, payload in report["overall"].items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
