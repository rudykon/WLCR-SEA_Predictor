#!/usr/bin/env python3
"""Derive per-day MAPEAUC with one pooled seven-day official filter."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import tempfile
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".runtime/lightgbm"))
import numpy as np

POLICY = "global official 5pct thresholds pooled over seven days"
THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
PROTOCOL_PREFIXES = {
    "fixed_seven_day_holdout": {
        "robust_seasonal": "fixed_seasonal",
        "plain_lgbm": "fixed_plain",
        "proposed": "fixed_proposed",
    },
    "seven_rolling_origins": {
        "robust_seasonal": "rolling_seasonal",
        "plain_lgbm": "rolling_plain",
        "proposed": "rolling_proposed",
    },
}


def parse_value(value: str) -> float:
    if value == "NIL" or value == "":
        return float("nan")
    return float(value)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="artifacts/revision2/seven_day_predictions.csv",
    )
    parser.add_argument(
        "--output",
        default="artifacts/revision2/seven_day_daily_metrics_pooled_filter.csv",
    )
    parser.add_argument(
        "--audit",
        default="artifacts/revision2/seven_day_daily_metrics_pooled_filter_audit.json",
    )
    args = parser.parse_args()

    prediction_path = Path(args.predictions)
    rows = list(csv.DictReader(prediction_path.open(encoding="utf-8", newline="")))
    if len(rows) != 122_640:
        raise RuntimeError(f"expected 122640 prediction rows, found {len(rows)}")

    dates = np.asarray([row["timestamp"][:10] for row in rows], dtype=object)
    cells = np.asarray([row["cell"] for row in rows], dtype=object)
    actual = np.asarray(
        [[parse_value(row[f"actual_m{metric}"]) for metric in range(4)] for row in rows],
        dtype=np.float64,
    )
    complete = np.all(np.isfinite(actual), axis=1)
    quantiles = np.quantile(actual[complete], 0.05, axis=0, method="linear")
    pooled_mask = complete & np.all(actual >= quantiles[None, :], axis=1)
    if int(np.sum(pooled_mask)) != 98_963:
        raise RuntimeError(
            f"pooled official mask expected 98963 hours, found {int(np.sum(pooled_mask))}"
        )

    output_rows: list[dict[str, object]] = []
    pooled_checks: dict[str, dict[str, float]] = {}
    unique_dates = sorted(set(dates.tolist()))
    if len(unique_dates) != 7:
        raise RuntimeError(f"expected seven dates, found {unique_dates}")

    for dataset, methods in PROTOCOL_PREFIXES.items():
        pooled_checks[dataset] = {}
        predictions: dict[str, np.ndarray] = {}
        for method, prefix in methods.items():
            predictions[method] = np.asarray(
                [
                    [parse_value(row[f"{prefix}_m{metric}"]) for metric in range(4)]
                    for row in rows
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(predictions[method])):
                raise RuntimeError(f"{dataset}/{method} contains non-finite predictions")

        for method, prediction in predictions.items():
            pooled_error = np.mean(
                np.abs(actual[pooled_mask] - prediction[pooled_mask])
                / actual[pooled_mask],
                axis=1,
            )
            pooled_checks[dataset][method] = float(
                np.mean([np.mean(pooled_error < threshold) for threshold in THRESHOLDS])
            )

        for date in unique_dates:
            date_all = dates == date
            date_mask = pooled_mask & date_all
            window_count = len(set(zip(cells[date_all].tolist(), dates[date_all].tolist())))
            if window_count != 730:
                raise RuntimeError(f"{dataset}/{date}: expected 730 windows, found {window_count}")
            for method, prediction in predictions.items():
                error = np.mean(
                    np.abs(actual[date_mask] - prediction[date_mask])
                    / actual[date_mask],
                    axis=1,
                )
                rates = [float(np.mean(error < threshold)) for threshold in THRESHOLDS]
                output_rows.append(
                    {
                        "dataset": dataset,
                        "date": date,
                        "method": method,
                        "mape_auc": float(np.mean(rates)),
                        "hit_020": rates[0],
                        "hit_030": rates[1],
                        "hit_040": rates[2],
                        "hit_050": rates[3],
                        "official_filtered_hours": int(np.sum(date_mask)),
                        "windows": window_count,
                        "filter_policy": POLICY,
                    }
                )

        protocol_rows = [row for row in output_rows if row["dataset"] == dataset]
        for date in unique_dates:
            by_method = {
                row["method"]: float(row["mape_auc"])
                for row in protocol_rows
                if row["date"] == date
            }
            if by_method["proposed"] <= by_method["plain_lgbm"]:
                raise RuntimeError(f"{dataset}/{date}: non-positive proposed-minus-plain gain")

    expected = {
        "fixed_seven_day_holdout": {
            "robust_seasonal": 0.7144084152663117,
            "plain_lgbm": 0.7623025777310712,
            "proposed": 0.7839242949385123,
        },
        "seven_rolling_origins": {
            "plain_lgbm": 0.762527409233754,
            "proposed": 0.7894996109657144,
        },
    }
    for dataset, methods in expected.items():
        for method, value in methods.items():
            observed = pooled_checks[dataset][method]
            if abs(observed - value) > 1e-12:
                raise RuntimeError(
                    f"{dataset}/{method}: pooled score mismatch {observed} vs {value}"
                )

    fieldnames = [
        "dataset",
        "date",
        "method",
        "mape_auc",
        "hit_020",
        "hit_030",
        "hit_040",
        "hit_050",
        "official_filtered_hours",
        "windows",
        "filter_policy",
    ]
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(output_rows)
    output_path = Path(args.output)
    atomic_write_text(output_path, buffer.getvalue())

    audit = {
        "source": str(prediction_path),
        "source_rows": len(rows),
        "pooled_filter_hours": int(np.sum(pooled_mask)),
        "pooled_fifth_percentiles": quantiles.tolist(),
        "filter_policy": POLICY,
        "output": str(output_path),
        "output_rows": len(output_rows),
        "pooled_score_checks": pooled_checks,
        "all_daily_proposed_minus_plain_positive": True,
    }
    atomic_write_text(
        Path(args.audit),
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
