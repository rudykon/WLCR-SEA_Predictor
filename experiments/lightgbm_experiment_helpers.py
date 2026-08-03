from __future__ import annotations

"""Shared deterministic helpers for manuscript revision-4 experiments.

The helpers in this module are independent of the finals test path. They
operate only on already constructed training backtests and provide one
deterministic corruption contract for the LightGBM and neural audits.
"""

import hashlib
import math
from datetime import datetime, timedelta
from typing import Mapping, Sequence

import numpy as np

from Model.traffic_window_forecasting import (
    BacktestExample,
    BaselineConfig,
    TestWindow,
    TrafficRow,
    seasonal_forecast,
)


HISTORY_LENGTHS = (72, 168, 336)
MISSING_RATES = (0.0, 0.10, 0.20, 0.30, 0.50)
MISSING_MECHANISMS = ("mcar", "block", "asynchronous", "recent_tail")
STANDARD_LAG_HOURS = (1, 6, 12, 24, 48, 72, 168, 336)
STANDARD_WINDOWS = (24, 72, 168, 336)


def stable_seed(*tokens: object) -> int:
    payload = "|".join(str(token) for token in tokens).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def example_history_array(example: BacktestExample) -> np.ndarray:
    return np.asarray(
        [
            [np.nan if value is None else float(value) for value in row.metrics]
            for row in example.window.rows
        ],
        dtype=np.float64,
    )


def additional_missing_mask(
    *,
    cell: str,
    target_start: datetime,
    mechanism: str,
    rate: float,
    hours: int = 336,
    metrics: int = 4,
) -> np.ndarray:
    if mechanism not in MISSING_MECHANISMS:
        raise ValueError(f"unknown missingness mechanism: {mechanism}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("missingness rate must be in [0, 1]")
    mask = np.zeros((hours, metrics), dtype=bool)
    count = int(round(rate * hours))
    if count <= 0:
        return mask
    count = min(count, hours)
    rng = np.random.default_rng(
        stable_seed("revision4_missingness", cell, target_start.isoformat(), mechanism, rate)
    )
    if mechanism == "mcar":
        selected = rng.choice(hours, size=count, replace=False)
        mask[selected, :] = True
    elif mechanism == "asynchronous":
        for metric in range(metrics):
            selected = rng.choice(hours, size=count, replace=False)
            mask[selected, metric] = True
    elif mechanism == "block":
        start = int(rng.integers(0, hours - count + 1))
        mask[start : start + count, :] = True
    elif mechanism == "recent_tail":
        mask[hours - count :, :] = True
    return mask


def apply_available_history(history: np.ndarray, available_hours: int) -> np.ndarray:
    values = np.asarray(history, dtype=np.float64).copy()
    if values.shape != (336, 4):
        raise ValueError(f"expected a (336, 4) history, found {values.shape}")
    if available_hours not in HISTORY_LENGTHS:
        raise ValueError(f"unsupported history length: {available_hours}")
    values[: 336 - available_hours, :] = np.nan
    return values


def apply_missingness(
    history: np.ndarray,
    *,
    cell: str,
    target_start: datetime,
    mechanism: str,
    rate: float,
) -> np.ndarray:
    values = np.asarray(history, dtype=np.float64).copy()
    if values.shape != (336, 4):
        raise ValueError(f"expected a (336, 4) history, found {values.shape}")
    values[
        additional_missing_mask(
            cell=cell,
            target_start=target_start,
            mechanism=mechanism,
            rate=rate,
        )
    ] = np.nan
    return values


def example_with_history_array(
    example: BacktestExample, history: np.ndarray
) -> BacktestExample:
    values = np.asarray(history, dtype=np.float64)
    if values.shape != (len(example.window.rows), 4):
        raise ValueError("history replacement shape mismatch")
    rows = tuple(
        TrafficRow(
            timestamp=source.timestamp,
            cell=source.cell,
            metrics=tuple(
                None if not math.isfinite(float(value)) else float(value)
                for value in values[index]
            ),
        )
        for index, source in enumerate(example.window.rows)
    )
    return BacktestExample(
        TestWindow(
            index=example.window.index,
            cell=example.window.cell,
            rows=rows,
            gaps=example.window.gaps,
        ),
        example.actuals,
    )


def truncate_example(example: BacktestExample, available_hours: int) -> BacktestExample:
    return example_with_history_array(
        example, apply_available_history(example_history_array(example), available_hours)
    )


def corrupt_example(
    example: BacktestExample, mechanism: str, rate: float
) -> BacktestExample:
    return example_with_history_array(
        example,
        apply_missingness(
            example_history_array(example),
            cell=example.window.cell,
            target_start=example.window.target_start,
            mechanism=mechanism,
            rate=rate,
        ),
    )


def _median(values: Sequence[float]) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return 0.0
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return 0.5 * (clean[middle - 1] + clean[middle])


def _stats(values: Sequence[float], fallback: float) -> tuple[float, ...]:
    clean = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if not len(clean):
        return fallback, 0.0, fallback, fallback, fallback
    return (
        float(np.mean(clean)),
        float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        float(np.median(clean)),
        float(np.min(clean)),
        float(np.max(clean)),
    )


def standard_stat_feature_row(
    example: BacktestExample,
    horizon: int,
    parameter: Mapping[str, float],
) -> tuple[list[str], list[float]]:
    window = example.window
    target = window.target_start + timedelta(hours=horizon)
    origin = window.target_start
    history = {row.timestamp: row.metrics for row in window.rows}
    azimuth = float(parameter.get("azimuth", 0.0)) % 360.0
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
    ]
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
    ]
    for metric in range(4):
        observed = [
            math.log1p(float(row.metrics[metric]))
            for row in window.rows
            if row.metrics[metric] is not None
        ]
        fallback = _median(observed)
        for offset in STANDARD_LAG_HOURS:
            row_values = history.get(origin - timedelta(hours=offset))
            raw = None if row_values is None else row_values[metric]
            present = raw is not None
            names.extend(
                [f"origin_lag{offset}_m{metric}", f"origin_lag{offset}_mask_m{metric}"]
            )
            values.extend(
                [
                    math.log1p(max(float(raw), 0.0)) if present else fallback,
                    float(present),
                ]
            )
        for span in STANDARD_WINDOWS:
            local = window.rows[-span:]
            logs = [
                math.log1p(float(row.metrics[metric]))
                for row in local
                if row.metrics[metric] is not None
            ]
            mean, std, median, minimum, maximum = _stats(logs, fallback)
            names.extend(
                [
                    f"roll{span}_mean_m{metric}",
                    f"roll{span}_std_m{metric}",
                    f"roll{span}_median_m{metric}",
                    f"roll{span}_min_m{metric}",
                    f"roll{span}_max_m{metric}",
                    f"roll{span}_missing_ratio_m{metric}",
                ]
            )
            values.extend(
                [
                    mean,
                    std,
                    median,
                    minimum,
                    maximum,
                    1.0 - len(logs) / span,
                ]
            )
        age = 336.0
        for offset, row in enumerate(reversed(window.rows), start=1):
            if row.metrics[metric] is not None:
                age = float(offset)
                break
        names.append(f"last_observed_age_m{metric}")
        values.append(age)
    return names, values


def build_standard_stat_matrix(
    examples: Sequence[BacktestExample],
    baseline_config: BaselineConfig,
    parameters: Mapping[str, Mapping[str, float]],
):
    from Model.lightgbm_feature_baseline import MatrixBundle

    feature_rows: list[list[float]] = []
    target_rows: list[list[float]] = []
    actuals = []
    baselines = []
    expected_names: list[str] | None = None
    for example in examples:
        baseline_rows = seasonal_forecast(example.window, baseline_config)
        parameter = parameters.get(example.window.cell, {})
        # All traffic statistics and static descriptors are request-level
        # quantities. Only the first six horizon/calendar fields change
        # across the 24 direct-forecast rows, so compute the expensive
        # history summaries once per window rather than 24 times.
        names, request_values = standard_stat_feature_row(example, 0, parameter)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError("standard-stat feature schema changed between rows")
        for horizon, (actual, baseline) in enumerate(
            zip(example.actuals, baseline_rows)
        ):
            target = example.window.target_start + timedelta(hours=horizon)
            values = request_values.copy()
            values[:6] = [
                float(horizon),
                math.sin(2.0 * math.pi * target.hour / 24.0),
                math.cos(2.0 * math.pi * target.hour / 24.0),
                math.sin(2.0 * math.pi * target.weekday() / 7.0),
                math.cos(2.0 * math.pi * target.weekday() / 7.0),
                float(target.weekday() >= 5),
            ]
            feature_rows.append(values)
            target_rows.append(
                [
                    np.nan if value is None else math.log1p(max(float(value), 0.0))
                    for value in actual.metrics
                ]
            )
            actuals.append(actual)
            baselines.append(baseline)
    return MatrixBundle(
        np.asarray(feature_rows, dtype=np.float32),
        np.asarray(target_rows, dtype=np.float32),
        tuple(actuals),
        tuple(baselines),
    )


def standard_stat_feature_names(
    example: BacktestExample, parameter: Mapping[str, float]
) -> tuple[str, ...]:
    names, _ = standard_stat_feature_row(example, 0, parameter)
    return tuple(names)


def traffic_only_columns(names: Sequence[str]) -> tuple[np.ndarray, ...]:
    suffixes = tuple(f"_m{metric}" for metric in range(4))
    selected = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if name == "horizon" or name.endswith(suffixes)
        ],
        dtype=np.int64,
    )
    if len(selected) != 73:
        raise ValueError(f"expected Traffic-only 73D, found {len(selected)}")
    return tuple(selected.copy() for _ in range(4))


def no_weather_columns(names: Sequence[str]) -> tuple[np.ndarray, ...]:
    selected = np.asarray(
        [index for index, name in enumerate(names) if not name.startswith("weather_")],
        dtype=np.int64,
    )
    if len(selected) != 83:
        raise ValueError(f"expected no-weather 83D, found {len(selected)}")
    return tuple(selected.copy() for _ in range(4))


def compact_columns(names: Sequence[str]) -> tuple[np.ndarray, ...]:
    selected = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if not name.startswith("weather_")
            and not name.startswith("baseline_")
            and "_mask_" not in name
            and not name.startswith("missing_ratio_")
        ],
        dtype=np.int64,
    )
    if len(selected) != 55:
        raise ValueError(f"expected compact 55D, found {len(selected)}")
    return tuple(selected.copy() for _ in range(4))
