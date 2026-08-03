from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence


WINDOW_ROWS = 336
FORECAST_ROWS = 24
OUTPUT_FLOOR = 1e-4
TIME_COLUMN = "时间"
CELL_COLUMN = "小区名称"
METRIC_COLUMNS = (
    "小区上行平均激活用户数",
    "小区下行平均激活用户数",
    "下行平均使用的PRB个数",
    "上行平均使用的PRB个数",
)
CSV_HEADER = (TIME_COLUMN, CELL_COLUMN, *METRIC_COLUMNS)

MetricValues = tuple[float | None, float | None, float | None, float | None]
PredictionValues = tuple[float, float, float, float]


class ContractError(ValueError):
    """Raised when a registered data or output contract is violated."""


@dataclass(frozen=True)
class TrafficRow:
    timestamp: datetime
    cell: str
    metrics: MetricValues


@dataclass(frozen=True)
class ForecastRow:
    timestamp: datetime
    cell: str
    metrics: PredictionValues


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    weights: tuple[float, float, float, float, float, float]
    scales: PredictionValues = (1.0, 1.0, 1.0, 1.0)

    @classmethod
    def default(cls) -> "BaselineConfig":
        return cls("robust_weekly", (0.05, 0.45, 0.10, 0.25, 0.10, 0.05))


@dataclass(frozen=True)
class ScoreResult:
    samples: int
    mean_mape: float | None
    rates: tuple[float, float, float, float]
    mape_auc: float

    @property
    def score(self) -> float:
        return max(100.0, 5000.0 * self.mape_auc)


@dataclass(frozen=True)
class BacktestExample:
    window: TestWindow
    actuals: tuple[TrafficRow, ...]


@dataclass(frozen=True)
class WindowGap:
    after_row: int
    previous_timestamp: datetime
    current_timestamp: datetime
    missing_hours: int


@dataclass(frozen=True)
class TestWindow:
    index: int
    cell: str
    rows: tuple[TrafficRow, ...]
    gaps: tuple[WindowGap, ...]

    @property
    def target_start(self) -> datetime:
        return self.rows[-1].timestamp + timedelta(hours=1)


def parse_time(text: str) -> datetime:
    try:
        return datetime.strptime(text.strip(), "%Y/%m/%d %H:%M")
    except ValueError as exc:
        raise ContractError(f"invalid timestamp: {text!r}") from exc


def parse_metric(text: str) -> float | None:
    value_text = text.strip()
    if not value_text or value_text.upper() == "NIL":
        return None
    try:
        value = float(value_text)
    except ValueError as exc:
        raise ContractError(f"invalid metric: {text!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ContractError(f"metric must be finite and non-negative: {text!r}")
    return value


def read_traffic(path: str | Path) -> list[TrafficRow]:
    source = Path(path)
    rows: list[TrafficRow] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ContractError(f"traffic file is empty: {source}")
        if tuple(header[:6]) != CSV_HEADER:
            raise ContractError(f"traffic header mismatch in {source}: {header}")
        for line_number, raw in enumerate(reader, start=2):
            if not raw or all(not item.strip() for item in raw):
                continue
            if len(raw) < 6:
                raise ContractError(f"{source}:{line_number} has {len(raw)} columns")
            cell = raw[1].strip()
            if not cell:
                raise ContractError(f"{source}:{line_number} has empty cell")
            metrics: MetricValues = tuple(parse_metric(item) for item in raw[2:6])  # type: ignore[assignment]
            rows.append(TrafficRow(parse_time(raw[0]), cell, metrics))
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_gaps(rows: Sequence[TrafficRow]) -> tuple[WindowGap, ...]:
    gaps: list[WindowGap] = []
    for index in range(1, len(rows)):
        delta_hours = int((rows[index].timestamp - rows[index - 1].timestamp).total_seconds() // 3600)
        if delta_hours != 1:
            gaps.append(
                WindowGap(
                    after_row=index - 1,
                    previous_timestamp=rows[index - 1].timestamp,
                    current_timestamp=rows[index].timestamp,
                    missing_hours=max(delta_hours - 1, 0),
                )
            )
    return tuple(gaps)


def split_physical_windows(rows: Sequence[TrafficRow], size: int = WINDOW_ROWS) -> list[TestWindow]:
    if size <= 0:
        raise ContractError("window size must be positive")
    if len(rows) % size:
        raise ContractError(f"traffic row count {len(rows)} is not divisible by {size}")
    windows: list[TestWindow] = []
    for index, start in enumerate(range(0, len(rows), size)):
        block = tuple(rows[start : start + size])
        if len(block) != size:
            raise ContractError(f"short window at physical row {start + 2}")
        cell = block[0].cell
        if any(row.cell != cell for row in block):
            raise ContractError(f"mixed cells in physical window {index}")
        windows.append(TestWindow(index=index, cell=cell, rows=block, gaps=_find_gaps(block)))
    return windows


def _median(values: Sequence[float]) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value) and value >= 0.0)
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return 0.5 * (clean[middle - 1] + clean[middle])


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _blend_candidates(candidates: Sequence[float | None], weights: Sequence[float], fallback: float) -> float:
    weighted_logs = 0.0
    total_weight = 0.0
    for value, weight in zip(candidates, weights):
        if value is None or not math.isfinite(value) or value < 0.0 or weight <= 0.0:
            continue
        weighted_logs += weight * math.log1p(value)
        total_weight += weight
    if total_weight <= 0.0:
        return max(fallback, OUTPUT_FLOOR)
    return max(math.expm1(weighted_logs / total_weight), OUTPUT_FLOOR)


def seasonal_forecast(window: TestWindow, config: BaselineConfig) -> list[ForecastRow]:
    history = {row.timestamp: row.metrics for row in window.rows}
    metric_fallbacks: list[float] = []
    for metric_index in range(4):
        values = [float(row.metrics[metric_index]) for row in window.rows if row.metrics[metric_index] is not None]
        metric_fallbacks.append(float(_median(values) or OUTPUT_FLOOR))

    output: list[ForecastRow] = []
    for horizon in range(FORECAST_ROWS):
        target = window.target_start + timedelta(hours=horizon)
        predictions: list[float] = []
        for metric_index in range(4):
            def lookup(days: int) -> float | None:
                values = history.get(target - timedelta(days=days))
                if values is None:
                    return None
                return values[metric_index]

            lag1 = lookup(1)
            lag7 = lookup(7)
            lag14 = lookup(14)
            recent7 = [value for day in range(1, 8) if (value := lookup(day)) is not None]
            recent14 = [value for day in range(1, 15) if (value := lookup(day)) is not None]
            median7 = _median([float(value) for value in recent7])
            median14 = _median([float(value) for value in recent14])
            trend = None
            if lag7 is not None and lag14 is not None and lag14 > 0.0:
                trend_ratio = _clamp((lag7 + OUTPUT_FLOOR) / (lag14 + OUTPUT_FLOOR), 0.75, 1.25)
                trend = max(lag7 * math.sqrt(trend_ratio), 0.0)
            value = _blend_candidates(
                (lag1, lag7, lag14, median7, median14, trend),
                config.weights,
                metric_fallbacks[metric_index],
            )
            value *= config.scales[metric_index]
            predictions.append(max(value, OUTPUT_FLOOR))
        output.append(ForecastRow(target, window.cell, tuple(predictions)))  # type: ignore[arg-type]
    return output


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ContractError("cannot calculate a quantile from no values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mape_auc(actuals: Sequence[TrafficRow], predictions: Sequence[ForecastRow]) -> ScoreResult:
    if len(actuals) != len(predictions):
        raise ContractError("actual and prediction lengths differ")
    complete = [row for row in actuals if all(value is not None for value in row.metrics)]
    if not complete:
        return ScoreResult(0, None, (0.0, 0.0, 0.0, 0.0), 0.0)
    thresholds = tuple(
        _quantile([float(row.metrics[index]) for row in complete], 0.05)
        for index in range(4)
    )
    limits = (0.2, 0.3, 0.4, 0.5)
    hits = [0, 0, 0, 0]
    used = 0
    error_sum = 0.0
    for actual, predicted in zip(actuals, predictions):
        if any(value is None for value in actual.metrics):
            continue
        values = tuple(float(value) for value in actual.metrics)
        if any(values[index] < thresholds[index] for index in range(4)):
            continue
        if any(values[index] <= 0.0 for index in range(4)):
            raise ContractError("a zero actual survived the official 5% filter")
        error = sum(abs(values[index] - predicted.metrics[index]) / values[index] for index in range(4)) / 4.0
        used += 1
        error_sum += error
        for index, limit in enumerate(limits):
            if error < limit:
                hits[index] += 1
    if not used:
        return ScoreResult(0, None, (0.0, 0.0, 0.0, 0.0), 0.0)
    rates = tuple(hit / used for hit in hits)
    return ScoreResult(used, error_sum / used, rates, sum(rates) / 4.0)  # type: ignore[arg-type]


def build_training_backtests(rows: Sequence[TrafficRow]) -> list[BacktestExample]:
    grouped: dict[str, list[TrafficRow]] = defaultdict(list)
    for row in rows:
        grouped[row.cell].append(row)
    examples: list[BacktestExample] = []
    for cell in sorted(grouped):
        cell_rows = sorted(grouped[cell], key=lambda row: row.timestamp)
        by_time = {row.timestamp: row for row in cell_rows}
        earliest = cell_rows[0].timestamp
        latest = cell_rows[-1].timestamp
        target_start = earliest + timedelta(hours=WINDOW_ROWS)
        while target_start + timedelta(hours=FORECAST_ROWS - 1) <= latest:
            history_times = [target_start - timedelta(hours=WINDOW_ROWS - offset) for offset in range(WINDOW_ROWS)]
            actual_times = [target_start + timedelta(hours=offset) for offset in range(FORECAST_ROWS)]
            if all(timestamp in by_time for timestamp in history_times + actual_times):
                history = tuple(by_time[timestamp] for timestamp in history_times)
                actuals = tuple(by_time[timestamp] for timestamp in actual_times)
                window = TestWindow(len(examples), cell, history, _find_gaps(history))
                examples.append(BacktestExample(window, actuals))
            target_start += timedelta(hours=FORECAST_ROWS)
    return examples


def baseline_candidates() -> list[BaselineConfig]:
    weight_sets = (
        ("lag7", (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        ("weekly_80_20", (0.0, 0.80, 0.20, 0.0, 0.0, 0.0)),
        ("weekly_median", (0.0, 0.70, 0.20, 0.10, 0.0, 0.0)),
        ("weekly_trend", (0.0, 0.70, 0.10, 0.10, 0.0, 0.10)),
    )
    configs: list[BaselineConfig] = []
    for name, weights in weight_sets:
        for scale in (0.94, 0.97, 1.0, 1.03, 1.06):
            configs.append(
                BaselineConfig(
                    name=f"{name}_s{int(round(scale * 100)):03d}",
                    weights=weights,
                    scales=(scale, scale, scale, scale),
                )
            )
    return configs


def _score_examples(examples: Sequence[BacktestExample], config: BaselineConfig) -> ScoreResult:
    actuals: list[TrafficRow] = []
    predictions: list[ForecastRow] = []
    for example in examples:
        actuals.extend(example.actuals)
        predictions.extend(seasonal_forecast(example.window, config))
    return mape_auc(actuals, predictions)


def _score_dict(score: ScoreResult) -> dict[str, object]:
    return {
        "samples": score.samples,
        "mean_mape": score.mean_mape,
        "rates": list(score.rates),
        "mape_auc": score.mape_auc,
        "score": score.score,
    }


def _heldout_cell(cell: str) -> bool:
    digest = hashlib.sha256(cell.encode("utf-8")).digest()
    return digest[0] % 5 == 0


def select_baseline(rows: Sequence[TrafficRow]) -> tuple[BaselineConfig, dict[str, object]]:
    examples = build_training_backtests(rows)
    if not examples:
        raise ContractError("training data produced no complete 336-to-24 backtests")
    dates = sorted({example.window.target_start.date() for example in examples})
    if len(dates) >= 9:
        lock_dates = set(dates[-3:])
        development_dates = set(dates[-6:-3])
        inner_dates = set(dates[-9:-6])
    else:
        lock_dates = set(dates)
        development_dates = set(dates)
        inner_dates = set(dates)
    inner = [example for example in examples if example.window.target_start.date() in inner_dates]
    development = [example for example in examples if example.window.target_start.date() in development_dates]
    locked = [example for example in examples if example.window.target_start.date() in lock_dates]
    if not inner:
        inner = list(examples)
    if not development:
        development = list(examples)
    if not locked:
        locked = list(examples)

    selected = baseline_candidates()[0]
    selected_inner_score = _score_examples(inner, selected)
    candidate_scores: list[dict[str, object]] = []
    for config in baseline_candidates():
        score = _score_examples(inner, config)
        candidate_scores.append({"name": config.name, **_score_dict(score)})
        if score.mape_auc > selected_inner_score.mape_auc + 1e-12:
            selected = config
            selected_inner_score = score

    development_score = _score_examples(development, selected)
    lock_score = _score_examples(locked, selected)
    heldout_examples = [example for example in locked if _heldout_cell(example.window.cell)]
    heldout_score = _score_examples(heldout_examples or locked, selected)
    report: dict[str, object] = {
        "selected": {
            "name": selected.name,
            "weights": list(selected.weights),
            "scales": list(selected.scales),
        },
        "training_backtests": len(examples),
        "inner_dates": [str(value) for value in sorted(inner_dates)],
        "development_dates": [str(value) for value in sorted(development_dates)],
        "lock_dates": [str(value) for value in sorted(lock_dates)],
        "inner": _score_dict(selected_inner_score),
        "development": _score_dict(development_score),
        "lock": _score_dict(lock_score),
        "cell_subset_lock_diagnostic": _score_dict(heldout_score),
        "candidate_scores": candidate_scores,
    }
    return selected, report


def validate_results(windows: Sequence[TestWindow], results: Sequence[ForecastRow]) -> None:
    expected_rows = len(windows) * FORECAST_ROWS
    if len(results) != expected_rows:
        raise ContractError("result row count %d != expected %d" % (len(results), expected_rows))
    cursor = 0
    for window in windows:
        for horizon in range(FORECAST_ROWS):
            row = results[cursor]
            expected_timestamp = window.target_start + timedelta(hours=horizon)
            if row.cell != window.cell or row.timestamp != expected_timestamp:
                raise ContractError("prediction row identity does not match its request window")
            if any(not math.isfinite(value) or value <= 0.0 for value in row.metrics):
                raise ContractError("predictions must be finite and strictly positive")
            cursor += 1


def format_time(value: datetime) -> str:
    return "%d/%d/%d %d:00" % (value.year, value.month, value.day, value.hour)


def _format_prediction(value: float) -> str:
    return "%.4f" % max(value, OUTPUT_FLOOR)


def write_results(path: str | Path, results: Sequence[ForecastRow]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for row in results:
            writer.writerow([format_time(row.timestamp), row.cell, *(_format_prediction(value) for value in row.metrics)])


def read_results(path: str | Path) -> list[ForecastRow]:
    rows = read_traffic(path)
    results: list[ForecastRow] = []
    for index, row in enumerate(rows):
        if any(value is None for value in row.metrics):
            raise ContractError("missing prediction at data row %d" % (index + 1))
        metrics: PredictionValues = tuple(float(value) for value in row.metrics)
        results.append(ForecastRow(row.timestamp, row.cell, metrics))
    return results


def load_baseline_config(path: str | Path) -> BaselineConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("model") != "seasonal_baseline":
        raise ContractError(f"unsupported frozen model: {payload.get('model')!r}")
    weights = tuple(float(value) for value in payload.get("weights", ()))
    scales = tuple(float(value) for value in payload.get("scales", ()))
    if len(weights) != 6 or len(scales) != 4:
        raise ContractError("frozen baseline config has invalid weight or scale shape")
    if any(not math.isfinite(value) or value < 0.0 for value in weights + scales):
        raise ContractError("frozen baseline config contains invalid values")
    return BaselineConfig(str(payload.get("name", "frozen")), weights, scales)  # type: ignore[arg-type]


