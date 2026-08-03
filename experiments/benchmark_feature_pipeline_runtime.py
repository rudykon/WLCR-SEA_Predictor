#!/usr/bin/env python3
"""Benchmark reproducible WLCR feature construction for the paper holdout."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from Model.traffic_window_forecasting import BaselineConfig, build_training_backtests, read_traffic
from Model.lightgbm_feature_baseline import build_matrix, load_parameters, load_weather


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_model_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train_data.csv")
    parser.add_argument("--parameter", default="data/parameter.csv")
    parser.add_argument("--weather", default="data/weather.csv")
    parser.add_argument("--output", default="artifacts/paper_experiments_gpu4_v2/feature_runtime.json")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")

    train = Path(args.train).resolve(strict=True)
    parameter = Path(args.parameter).resolve(strict=True)
    weather_path = Path(args.weather).resolve(strict=True)
    repo = Path(__file__).resolve().parents[1]
    expected = {
        "train": (repo / "data/train_data.csv").resolve(strict=True),
        "parameter": (repo / "data/parameter.csv").resolve(strict=True),
        "weather": (repo / "data/weather.csv").resolve(strict=True),
    }
    actual_paths = {"train": train, "parameter": parameter, "weather": weather_path}
    for name, path in actual_paths.items():
        if path != expected[name]:
            raise ValueError(f"{name} must use registered repository input: {expected[name]}")

    setup_start = time.perf_counter()
    rows = read_traffic(train)
    examples = build_training_backtests(rows)
    dates = sorted({example.window.target_start.date() for example in examples})
    lock_dates = set(dates[13:16])
    lock = [example for example in examples if example.window.target_start.date() in lock_dates]
    parameters = load_parameters(parameter)
    weather = load_weather(weather_path)
    setup_seconds = time.perf_counter() - setup_start

    config = BaselineConfig(
        "weekly_median_s097",
        (0.0, 0.7, 0.2, 0.1, 0.0, 0.0),
        (0.97,) * 4,
    )

    timings: list[float] = []
    rows_per_matrix = None
    feature_count = None
    for _ in range(args.repetitions):
        start = time.perf_counter()
        matrix = build_matrix(lock, config, parameters, weather)
        timings.append(time.perf_counter() - start)
        rows_per_matrix = int(matrix.features.shape[0])
        feature_count = int(matrix.features.shape[1])
        del matrix
        gc.collect()

    report = {
        "schema_version": 1,
        "scope": "88-dimensional feature construction for the 2,190 chronological holdout windows",
        "registered_inputs": {
            name: {
                "path": str(path.relative_to(repo)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in actual_paths.items()
        },
        "holdout_dates": [str(value) for value in sorted(lock_dates)],
        "windows": len(lock),
        "feature_rows": rows_per_matrix,
        "features": feature_count,
        "repetitions": args.repetitions,
        "feature_construction_seconds": timings,
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "median_ms_per_window": 1000.0 * statistics.median(timings) / len(lock),
        "setup_seconds_excluded": setup_seconds,
        "setup_definition": "CSV parsing, backtest-window enumeration, and parameter/weather loading",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_count": __import__("os").cpu_count(),
            "cpu_model": cpu_model_name(),
        },
        "measurement_conditions": {
            "execution": "sequential repetitions after one shared input-loading and window-enumeration phase",
            "inputs": "traffic windows, cell parameters, and weather are preloaded in memory",
            "os_page_cache": "not explicitly flushed between repetitions",
        },
        "notes": [
            "No model fitting or prediction is included.",
            "No finals test traffic is read.",
            "All 2,190 holdout windows are used in every repetition.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
