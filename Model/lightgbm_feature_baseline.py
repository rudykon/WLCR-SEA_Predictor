from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from Model.traffic_window_forecasting import (
    BaselineConfig,
    BacktestExample,
    ContractError,
    ForecastRow,
    OUTPUT_FLOOR,
    TestWindow,
    TrafficRow,
    build_training_backtests,
    load_baseline_config,
    mape_auc,
    read_traffic,
    seasonal_forecast,
    split_physical_windows,
    validate_results,
    write_results,
)


RANDOM_SEED = 42
LAG_DAYS = (1, 2, 3, 7, 14)


@dataclass(frozen=True)
class MatrixBundle:
    features: object
    targets: object | None
    actuals: tuple[TrafficRow, ...]
    baselines: tuple[ForecastRow, ...]


def _median(values: Sequence[float]) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    return mean, math.sqrt(max(variance, 0.0))


def _prepare_context(window: TestWindow) -> dict[str, object]:
    history = {row.timestamp: row.metrics for row in window.rows}
    log_medians = []
    log_means = []
    log_stds = []
    missing_ratios = []
    for metric in range(4):
        values = [float(row.metrics[metric]) for row in window.rows if row.metrics[metric] is not None]
        logs = [math.log1p(max(value, 0.0)) for value in values]
        mean, std = _mean_std(logs)
        log_medians.append(_median(logs))
        log_means.append(mean)
        log_stds.append(std)
        missing_ratios.append(1.0 - len(values) / len(window.rows))
    return {
        "history": history,
        "log_medians": tuple(log_medians),
        "log_means": tuple(log_means),
        "log_stds": tuple(log_stds),
        "missing_ratios": tuple(missing_ratios),
    }


def _feature_row_with_context(
    window: TestWindow,
    horizon: int,
    baseline: ForecastRow,
    parameter: dict[str, float],
    weather: dict[str, float],
    context: dict[str, object],
) -> tuple[list[str], list[float]]:
    target = window.target_start + timedelta(hours=horizon)
    history = context["history"]
    log_medians = context["log_medians"]
    names = [
        "horizon",
        "target_hour_sin",
        "target_hour_cos",
        "target_dow_sin",
        "target_dow_cos",
        "is_weekend",
        "azimuth_sin",
        "azimuth_cos",
        "scene_code",
        "x",
        "y",
        "weather_code",
        "weather_avg_temp",
        "weather_humidity",
        "weather_rain",
        "weather_wind",
    ]
    azimuth = float(parameter.get("azimuth", 0.0)) % 360.0
    values = [
        float(horizon),
        math.sin(2.0 * math.pi * target.hour / 24.0),
        math.cos(2.0 * math.pi * target.hour / 24.0),
        math.sin(2.0 * math.pi * target.weekday() / 7.0),
        math.cos(2.0 * math.pi * target.weekday() / 7.0),
        float(target.weekday() >= 5),
        math.sin(2.0 * math.pi * azimuth / 360.0),
        math.cos(2.0 * math.pi * azimuth / 360.0),
        float(parameter.get("scene_code", 0.0)),
        float(parameter.get("x", 0.0)),
        float(parameter.get("y", 0.0)),
        float(weather.get("weather_code", 0.0)),
        float(weather.get("avg_temp", 0.0)),
        float(weather.get("humidity", 0.0)),
        float(weather.get("rain", 0.0)),
        float(weather.get("wind", 0.0)),
    ]
    for metric in range(4):
        names.extend(
            [
                f"baseline_m{metric}",
                f"window_log_median_m{metric}",
                f"window_log_mean_m{metric}",
                f"window_log_std_m{metric}",
                f"missing_ratio_m{metric}",
            ]
        )
        values.extend(
            [
                math.log1p(max(baseline.metrics[metric], 0.0)),
                float(log_medians[metric]),
                float(context["log_means"][metric]),
                float(context["log_stds"][metric]),
                float(context["missing_ratios"][metric]),
            ]
        )
        recent7 = []
        recent14 = []
        lag_logs: dict[int, float] = {}
        for day in LAG_DAYS:
            row_values = history.get(target - timedelta(days=day))
            raw = None if row_values is None else row_values[metric]
            present = raw is not None
            log_value = math.log1p(max(float(raw), 0.0)) if present else float(log_medians[metric])
            lag_logs[day] = log_value
            names.extend([f"lag{day}_m{metric}", f"lag{day}_mask_m{metric}"])
            values.extend([log_value, float(present)])
        for day in range(1, 15):
            row_values = history.get(target - timedelta(days=day))
            raw = None if row_values is None else row_values[metric]
            if raw is not None:
                log_value = math.log1p(max(float(raw), 0.0))
                recent14.append(log_value)
                if day <= 7:
                    recent7.append(log_value)
        names.extend([f"same_hour_median7_m{metric}", f"same_hour_median14_m{metric}", f"lag7_minus_lag14_m{metric}"])
        values.extend([_median(recent7), _median(recent14), lag_logs[7] - lag_logs[14]])
    return names, values


def build_feature_row(
    window: TestWindow,
    horizon: int,
    baseline: ForecastRow,
    parameter: dict[str, float],
    weather: dict[str, float],
) -> tuple[list[str], list[float]]:
    return _feature_row_with_context(
        window,
        horizon,
        baseline,
        parameter,
        weather,
        _prepare_context(window),
    )


def load_parameters(path: str | Path) -> dict[str, dict[str, float]]:
    raw_rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or header[0] != "标准小区名称":
            raise ContractError("parameter header mismatch")
        for row in reader:
            if len(row) >= 5 and row[0].strip():
                raw_rows.append(row)
    scenes = {name: index for index, name in enumerate(sorted({row[2].strip() for row in raw_rows}))}
    return {
        row[0].strip(): {
            "azimuth": float(row[1] or 0.0),
            "scene_code": float(scenes[row[2].strip()]),
            "x": float(row[3] or 0.0),
            "y": float(row[4] or 0.0),
        }
        for row in raw_rows
    }


def load_weather(path: str | Path) -> dict[str, dict[str, float]]:
    raw_rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or header[0] != "日期(date)":
            raise ContractError("weather header mismatch")
        raw_rows = [row for row in reader if len(row) >= 10 and row[0].strip()]
    categories = {name: index for index, name in enumerate(sorted({row[1].strip() for row in raw_rows}))}
    return {
        row[0].strip(): {
            "weather_code": float(categories[row[1].strip()]),
            "avg_temp": float(row[4] or 0.0),
            "humidity": float(row[5] or 0.0),
            "rain": float(row[7] or 0.0),
            "wind": float(row[9] or 0.0),
        }
        for row in raw_rows
    }


def build_matrix(
    examples: Sequence[BacktestExample],
    baseline_config: BaselineConfig,
    parameters: dict[str, dict[str, float]],
    weather: dict[str, dict[str, float]],
) -> MatrixBundle:
    import numpy as np

    feature_rows = []
    target_rows = []
    actuals = []
    baselines = []
    expected_names = None
    for example in examples:
        context = _prepare_context(example.window)
        baseline_rows = seasonal_forecast(example.window, baseline_config)
        parameter = parameters.get(example.window.cell, {})
        for horizon, (actual, baseline) in enumerate(zip(example.actuals, baseline_rows)):
            target_key = baseline.timestamp.strftime("%Y%m%d")
            names, values = _feature_row_with_context(
                example.window,
                horizon,
                baseline,
                parameter,
                weather.get(target_key, {}),
                context,
            )
            if expected_names is None:
                expected_names = names
            elif names != expected_names:
                raise ContractError("feature schema changed between windows")
            feature_rows.append(values)
            target_rows.append(
                [float("nan") if value is None else math.log1p(max(float(value), 0.0)) for value in actual.metrics]
            )
            actuals.append(actual)
            baselines.append(baseline)
    return MatrixBundle(
        np.asarray(feature_rows, dtype=np.float32),
        np.asarray(target_rows, dtype=np.float32),
        tuple(actuals),
        tuple(baselines),
    )


def build_test_matrix(
    windows: Sequence[TestWindow],
    baseline_config: BaselineConfig,
    parameters: dict[str, dict[str, float]],
    weather: dict[str, dict[str, float]],
) -> tuple[object, tuple[ForecastRow, ...]]:
    import numpy as np

    feature_rows = []
    baselines = []
    expected_names = None
    for window in windows:
        context = _prepare_context(window)
        baseline_rows = seasonal_forecast(window, baseline_config)
        parameter = parameters.get(window.cell, {})
        for horizon, baseline in enumerate(baseline_rows):
            names, values = _feature_row_with_context(
                window,
                horizon,
                baseline,
                parameter,
                weather.get(baseline.timestamp.strftime("%Y%m%d"), {}),
                context,
            )
            if expected_names is None:
                expected_names = names
            elif names != expected_names:
                raise ContractError("feature schema changed between windows")
            feature_rows.append(values)
            baselines.append(baseline)
    return np.asarray(feature_rows, dtype=np.float32), tuple(baselines)


def _train_models(train: MatrixBundle, valid: MatrixBundle, objective_name: str, output_dir: Path | None = None):
    import lightgbm as lgb
    import numpy as np

    if train.targets is None or valid.targets is None:
        raise ContractError("training matrices require labels")
    if objective_name == "l1":
        objective = "regression_l1"
        alpha = None
    elif objective_name == "q036":
        objective = "quantile"
        alpha = 0.36
    elif objective_name == "q040":
        objective = "quantile"
        alpha = 0.40
    else:
        raise ContractError(f"unknown LightGBM objective {objective_name}")
    predictions = np.empty((len(valid.actuals), 4), dtype=np.float64)
    boosters = []
    best_iterations = []
    for metric in range(4):
        train_mask = np.isfinite(train.targets[:, metric])
        valid_mask = np.isfinite(valid.targets[:, metric])
        params = {
            "objective": objective,
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
            "seed": RANDOM_SEED,
            "feature_fraction_seed": RANDOM_SEED,
            "bagging_seed": RANDOM_SEED,
            "num_threads": 32,
            "force_col_wise": True,
        }
        if alpha is not None:
            params["alpha"] = alpha
        train_set = lgb.Dataset(train.features[train_mask], label=train.targets[train_mask, metric], free_raw_data=False)
        valid_set = lgb.Dataset(valid.features[valid_mask], label=valid.targets[valid_mask, metric], reference=train_set, free_raw_data=False)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(45, verbose=False), lgb.log_evaluation(0)],
        )
        predictions[:, metric] = np.maximum(np.expm1(booster.predict(valid.features, num_iteration=booster.best_iteration)), OUTPUT_FLOOR)
        boosters.append(booster)
        best_iterations.append(int(booster.best_iteration))
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(output_dir / f"metric_{metric}.txt"))
    return boosters, predictions, best_iterations


def _prediction_rows(baselines: Sequence[ForecastRow], values) -> list[ForecastRow]:
    return [
        ForecastRow(row.timestamp, row.cell, tuple(max(float(value), OUTPUT_FLOOR) for value in values[index]))
        for index, row in enumerate(baselines)
    ]


def _blend_rows(baselines: Sequence[ForecastRow], model_rows: Sequence[ForecastRow], weight: float) -> list[ForecastRow]:
    output = []
    for baseline, model in zip(baselines, model_rows):
        values = tuple(
            max(
                math.expm1((1.0 - weight) * math.log1p(baseline.metrics[index]) + weight * math.log1p(model.metrics[index])),
                OUTPUT_FLOOR,
            )
            for index in range(4)
        )
        output.append(ForecastRow(baseline.timestamp, baseline.cell, values))
    return output


def _score_payload(actuals: Sequence[TrafficRow], rows: Sequence[ForecastRow]) -> dict[str, object]:
    score = mape_auc(actuals, rows)
    return {
        "samples": score.samples,
        "mean_mape": score.mean_mape,
        "rates": list(score.rates),
        "mape_auc": score.mape_auc,
        "score": score.score,
    }


def train_and_generate(
    train_path: str | Path,
    test_path: str | Path,
    parameter_path: str | Path,
    weather_path: str | Path,
    baseline_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    import numpy as np

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline_config = load_baseline_config(baseline_config_path)
    train_rows = read_traffic(train_path)
    examples = build_training_backtests(train_rows)
    dates = sorted({example.window.target_start.date() for example in examples})
    if len(dates) < 16:
        raise ContractError(f"expected 16 target dates, found {len(dates)}")
    fit_dates = set(dates[:7])
    inner_dates = set(dates[7:10])
    development_dates = set(dates[10:13])
    lock_dates = set(dates[13:16])
    by_dates = lambda selected: [example for example in examples if example.window.target_start.date() in selected]
    fit_examples = by_dates(fit_dates)
    inner_examples = by_dates(inner_dates)
    development_examples = by_dates(development_dates)
    lock_examples = by_dates(lock_dates)
    parameters = load_parameters(parameter_path)
    weather = load_weather(weather_path)
    fit_matrix = build_matrix(fit_examples, baseline_config, parameters, weather)
    inner_matrix = build_matrix(inner_examples, baseline_config, parameters, weather)
    candidate_records = []
    selected = None
    selected_inner_auc = -1.0
    selected_boosters = None
    selected_iterations = None
    for objective in ("l1", "q036", "q040"):
        boosters, predictions, best_iterations = _train_models(fit_matrix, inner_matrix, objective)
        model_rows = _prediction_rows(inner_matrix.baselines, predictions)
        for weight in (0.25, 0.50, 0.75, 1.0):
            rows = _blend_rows(inner_matrix.baselines, model_rows, weight)
            score = _score_payload(inner_matrix.actuals, rows)
            record = {"objective": objective, "weight": weight, "best_iterations": best_iterations, **score}
            candidate_records.append(record)
            if float(score["mape_auc"]) > selected_inner_auc:
                selected_inner_auc = float(score["mape_auc"])
                selected = (objective, weight)
                selected_boosters = boosters
                selected_iterations = best_iterations
    if selected is None:
        raise ContractError("no LightGBM candidate was selected")

    fit_inner_examples = fit_examples + inner_examples
    fit_inner_matrix = build_matrix(fit_inner_examples, baseline_config, parameters, weather)
    development_matrix = build_matrix(development_examples, baseline_config, parameters, weather)
    dev_boosters, dev_predictions, dev_iterations = _train_models(fit_inner_matrix, development_matrix, selected[0])
    development_model_rows = _prediction_rows(development_matrix.baselines, dev_predictions)
    development_rows = _blend_rows(development_matrix.baselines, development_model_rows, selected[1])
    development_score = _score_payload(development_matrix.actuals, development_rows)
    development_baseline_score = _score_payload(development_matrix.actuals, development_matrix.baselines)

    prelock_examples = fit_inner_examples + development_examples
    prelock_matrix = build_matrix(prelock_examples, baseline_config, parameters, weather)
    lock_matrix = build_matrix(lock_examples, baseline_config, parameters, weather)
    lock_boosters, lock_predictions, lock_iterations = _train_models(prelock_matrix, lock_matrix, selected[0])
    lock_model_rows = _prediction_rows(lock_matrix.baselines, lock_predictions)
    lock_rows = _blend_rows(lock_matrix.baselines, lock_model_rows, selected[1])
    lock_score = _score_payload(lock_matrix.actuals, lock_rows)
    lock_baseline_score = _score_payload(lock_matrix.actuals, lock_matrix.baselines)

    dev_gain = float(development_score["mape_auc"]) - float(development_baseline_score["mape_auc"])
    lock_gain = float(lock_score["mape_auc"]) - float(lock_baseline_score["mape_auc"])
    accepted = dev_gain >= 0.005 and lock_gain >= 0.0
    report = {
        "selected": {"objective": selected[0], "weight": selected[1]},
        "inner_candidates": candidate_records,
        "development": development_score,
        "development_baseline": development_baseline_score,
        "development_gain": dev_gain,
        "lock": lock_score,
        "lock_baseline": lock_baseline_score,
        "lock_gain": lock_gain,
        "accepted": accepted,
        "date_layers": {
            "fit": [str(value) for value in sorted(fit_dates)],
            "inner": [str(value) for value in sorted(inner_dates)],
            "development": [str(value) for value in sorted(development_dates)],
            "lock": [str(value) for value in sorted(lock_dates)],
        },
        "intermediate_best_iterations": {
            "inner": selected_iterations,
            "development": dev_iterations,
            "lock": lock_iterations,
        },
    }
    (output / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not accepted:
        return report

    all_matrix = build_matrix(examples, baseline_config, parameters, weather)
    test_windows = split_physical_windows(read_traffic(test_path))
    test_features, test_baselines = build_test_matrix(test_windows, baseline_config, parameters, weather)
    model_dir = output / "models"
    final_predictions = np.empty((len(test_baselines), 4), dtype=np.float64)
    final_iterations = []
    import lightgbm as lgb

    for metric in range(4):
        mask = np.isfinite(all_matrix.targets[:, metric])
        objective = "regression_l1" if selected[0] == "l1" else "quantile"
        params = {
            "objective": objective,
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
            "seed": RANDOM_SEED,
            "feature_fraction_seed": RANDOM_SEED,
            "bagging_seed": RANDOM_SEED,
            "num_threads": 32,
            "force_col_wise": True,
        }
        if selected[0] == "q036":
            params["alpha"] = 0.36
        elif selected[0] == "q040":
            params["alpha"] = 0.40
        rounds = max(80, int(round(lock_iterations[metric])))
        booster = lgb.train(params, lgb.Dataset(all_matrix.features[mask], label=all_matrix.targets[mask, metric]), num_boost_round=rounds)
        model_dir.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(model_dir / f"metric_{metric}.txt"))
        final_predictions[:, metric] = np.maximum(np.expm1(booster.predict(test_features)), OUTPUT_FLOOR)
        final_iterations.append(rounds)
    model_rows = _prediction_rows(test_baselines, final_predictions)
    final_rows = _blend_rows(test_baselines, model_rows, selected[1])
    validate_results(test_windows, final_rows)
    write_results(output / "results.csv", final_rows)
    report["final_iterations"] = final_iterations
    report["result_rows"] = len(final_rows)
    (output / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and generate a LightGBM comparison baseline")
    parser.add_argument("--train", default="data/train_data.csv")
    parser.add_argument("--test", default="data/test_data.csv")
    parser.add_argument("--parameter", default="data/parameter.csv")
    parser.add_argument("--weather", default="data/weather.csv")
    parser.add_argument("--baseline-config", default="Model/seasonal_baseline_config.json")
    parser.add_argument("--output-dir", default="artifacts/lightgbm_baseline")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = train_and_generate(
        args.train,
        args.test,
        args.parameter,
        args.weather,
        args.baseline_config,
        args.output_dir,
    )
    print(json.dumps({"accepted": report["accepted"], "selected": report["selected"], "output_dir": args.output_dir}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
