from __future__ import annotations

"""Revision-7 evidence adapters with strict sample-identity validation."""

import argparse
import csv
import gzip
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from experiments import train_neural_baselines as neural
from experiments import wlcr_sea_model as sea


ORIGINAL_WLCR_PREDICTIONS = Path(
    "artifacts/revision3/revision3_predictions.csv.gz"
)
DEFAULT_OUTPUT = Path("artifacts/revision7/original_wlcr_alignment")
PREDICTION_PREFIX = "prediction_wlcr_traffic_only_73d_"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parsed_number(token: str) -> float:
    value = token.strip()
    if not value:
        return float("nan")
    parsed = float(value)
    return parsed


def load_original_wlcr_prediction(
    path: Path,
    dataset: neural.CachedDataset,
    holdout: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Align the Revision-3 traffic-only WLCR by exact request keys.

    Every ``(cell, target timestamp, horizon)`` key and every available actual
    target is checked against the current registered-training holdout before a
    historical clean prediction is admitted to Revision 7.
    """
    source = path.resolve(strict=True)
    root = project_root()
    if not source.is_relative_to(root):
        raise ValueError("original WLCR evidence must remain inside the project")
    holdout = np.asarray(holdout, dtype=np.int64)
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    prediction = np.full(actual.shape, np.nan, dtype=np.float32)
    expected: dict[tuple[str, str, int], tuple[int, int]] = {}
    for local_window, dataset_index in enumerate(holdout.tolist()):
        target_start = neural.timestamp_from_hour(
            int(dataset.target_start_hours[dataset_index])
        )
        cell = str(dataset.cells[dataset_index])
        for horizon_index in range(neural.FORECAST_HOURS):
            timestamp = (target_start + timedelta(hours=horizon_index)).isoformat(
                sep=" "
            )
            key = (cell, timestamp, horizon_index + 1)
            if key in expected:
                raise ValueError(f"duplicate current holdout key: {key}")
            expected[key] = (local_window, horizon_index)

    actual_columns = [f"actual_{name}" for name in neural.METRIC_NAMES]
    prediction_columns = [
        f"{PREDICTION_PREFIX}{name}" for name in neural.METRIC_NAMES
    ]
    seen: set[tuple[str, str, int]] = set()
    maximum_actual_difference = 0.0
    compared_actual_values = 0
    source_missing_actual_values = 0
    first_key: tuple[str, str, int] | None = None
    last_key: tuple[str, str, int] | None = None

    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "cell",
            "target_timestamp",
            "horizon",
            *actual_columns,
            *prediction_columns,
        }
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(f"Revision-3 prediction columns missing: {missing_columns}")
        for row_number, row in enumerate(reader, start=2):
            timestamp = datetime.fromisoformat(row["target_timestamp"]).isoformat(
                sep=" "
            )
            key = (str(row["cell"]), timestamp, int(row["horizon"]))
            if key not in expected:
                raise ValueError(f"unexpected Revision-3 sample at row {row_number}: {key}")
            if key in seen:
                raise ValueError(f"duplicate Revision-3 sample at row {row_number}: {key}")
            seen.add(key)
            first_key = key if first_key is None else first_key
            last_key = key
            window, horizon = expected[key]
            for metric, (actual_column, prediction_column) in enumerate(
                zip(actual_columns, prediction_columns)
            ):
                current_actual = float(actual[window, horizon, metric])
                source_actual = _parsed_number(row[actual_column])
                if math.isfinite(current_actual):
                    if not math.isfinite(source_actual):
                        raise ValueError(
                            f"Revision-3 actual is missing for an observed target: {key}/{metric}"
                        )
                    difference = abs(current_actual - source_actual)
                    maximum_actual_difference = max(
                        maximum_actual_difference, difference
                    )
                    compared_actual_values += 1
                    if difference > 5e-5:
                        raise ValueError(
                            f"Revision-3 actual mismatch {difference} at {key}/{metric}"
                        )
                else:
                    source_missing_actual_values += int(not math.isfinite(source_actual))
                value = _parsed_number(row[prediction_column])
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"invalid original WLCR prediction at {key}/{metric}: {value}"
                    )
                prediction[window, horizon, metric] = value

    missing_keys = set(expected) - seen
    if missing_keys:
        raise ValueError(
            f"Revision-3 prediction misses {len(missing_keys)} holdout keys; "
            f"examples={sorted(missing_keys)[:3]}"
        )
    if np.any(~np.isfinite(prediction)):
        raise ValueError("aligned original WLCR prediction contains non-finite values")
    report = {
        "schema_version": 1,
        "source": str(source.relative_to(root)),
        "source_sha256": neural.sha256_file(source),
        "holdout_windows": int(len(holdout)),
        "forecast_rows": int(len(seen)),
        "prediction_values": int(prediction.size),
        "exact_key_set_match": True,
        "duplicate_keys": 0,
        "compared_actual_values": compared_actual_values,
        "source_missing_actual_values": source_missing_actual_values,
        "maximum_actual_absolute_difference": maximum_actual_difference,
        "first_key": list(first_key) if first_key else None,
        "last_key": list(last_key) if last_key else None,
        "finals_test_opened": False,
    }
    return prediction, report


def run(args: argparse.Namespace) -> int:
    root = project_root()
    source = root / ORIGINAL_WLCR_PREDICTIONS
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve(strict=False)
    allowed = (root / "artifacts/revision7").resolve(strict=False)
    if not output.is_relative_to(allowed):
        raise ValueError("Revision-7 evidence output must remain under artifacts/revision7")
    output.mkdir(parents=True, exist_ok=True)

    train = neural.resolve_train_path()
    before = neural.sha256_file(train)
    arrays, dataset_report = neural.build_window_arrays(
        neural.read_training_series(train)
    )
    dataset = neural.CachedDataset(root=Path("<memory>"), **arrays)
    holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
    prediction, report = load_original_wlcr_prediction(source, dataset, holdout)
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    scales = np.asarray(dataset.mase_scales[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    metrics = sea.forecast_metrics(actual, prediction, scales, cells)
    expected_wape = 0.1950556749831762
    observed_wape = float(metrics["macro_indicator"]["wape"])
    if not math.isclose(observed_wape, expected_wape, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"original WLCR clean WAPE changed: {observed_wape} != {expected_wape}"
        )
    prediction_path = output / "original_wlcr_holdout_predictions.npy"
    temporary = prediction_path.with_name(f".{prediction_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, prediction, allow_pickle=False)
    temporary.replace(prediction_path)
    after = neural.sha256_file(train)
    if before != after:
        raise RuntimeError("registered training file changed during evidence alignment")
    payload = {
        **report,
        "dataset_report": dataset_report,
        "metrics": metrics,
        "expected_revision4_macro_wape": expected_wape,
        "prediction_file": str(prediction_path.relative_to(root)),
        "prediction_sha256": neural.sha256_file(prediction_path),
        "registered_train_sha256_before": before,
        "registered_train_sha256_after": after,
    }
    report_path = output / "alignment_report.json"
    temporary_report = report_path.with_name(
        f".{report_path.name}.{os.getpid()}.tmp"
    )
    temporary_report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
