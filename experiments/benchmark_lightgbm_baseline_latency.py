from __future__ import annotations

"""Single-thread CPU batch-one latency benchmark for frozen WLCR Full."""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from Model.traffic_window_forecasting import build_training_backtests, read_traffic
from Model.lightgbm_feature_baseline import build_matrix, load_parameters, load_weather
from experiments.train_lightgbm_baseline import feature_names
from experiments.run_reproducibility_evaluation import load_verified_boosters
from experiments.run_seasonal_anchor_ablations import registered_inputs, select_baseline_for_inner


OUTPUT = Path("artifacts/revision4/latency_wlcr.json")
MODEL_DIR = Path("artifacts/revision2/models/fixed_seven_day_holdout/proposed")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def identity_hash(examples) -> str:
    payload = "\n".join(
        f"{example.window.cell}|{example.window.target_start.isoformat()}"
        for example in examples
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "mean_ms": float(np.mean(array)),
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(warmup: int, repetitions: int, samples: int) -> dict[str, object]:
    root = project_root()
    inputs = registered_inputs()
    rows = read_traffic(inputs["train"])
    examples = build_training_backtests(rows)
    dates = sorted({example.window.target_start.date() for example in examples})
    inner_dates = set(dates[7:9])
    holdout_dates = set(dates[9:])
    inner = [example for example in examples if example.window.target_start.date() in inner_dates]
    holdout = [example for example in examples if example.window.target_start.date() in holdout_dates]
    if len(inner) != 1460 or len(holdout) != 5110:
        raise ValueError("unexpected fixed-protocol counts")
    selected = sorted(
        holdout,
        key=lambda example: (example.window.cell, example.window.target_start),
    )[:samples]
    baseline, _ = select_baseline_for_inner(inner)
    parameters = load_parameters(inputs["parameter"])
    weather = load_weather(inputs["weather"])
    boosters = load_verified_boosters(root / MODEL_DIR)
    names = feature_names(selected[0], baseline, parameters, weather)
    columns = tuple(np.arange(len(names), dtype=np.int64) for _ in range(4))
    model_bytes = sum((root / MODEL_DIR / f"metric_{metric}.txt").stat().st_size for metric in range(4))
    rss_after_load = rss_bytes()

    def one(example) -> tuple[float, float, float, float]:
        started = time.perf_counter_ns()
        matrix = build_matrix([example], baseline, parameters, weather)
        after_feature = time.perf_counter_ns()
        predictions = []
        for metric, booster in enumerate(boosters):
            raw = booster.predict(
                matrix.features[:, columns[metric]],
                num_threads=1,
            )
            predictions.append(np.maximum(np.expm1(raw), 1e-4))
        after_model = time.perf_counter_ns()
        output = np.stack(predictions, axis=1)
        checksum = float(np.sum(output))
        finished = time.perf_counter_ns()
        if output.shape != (24, 4) or not np.isfinite(checksum) or checksum <= 0.0:
            raise ValueError("invalid WLCR latency output")
        return (
            (finished - started) / 1e6,
            (after_feature - started) / 1e6,
            (after_model - after_feature) / 1e6,
            (finished - after_model) / 1e6,
        )

    for index in range(warmup):
        one(selected[index % len(selected)])
    end_to_end = []
    feature = []
    model = []
    postprocess = []
    for index in range(repetitions):
        values = one(selected[index % len(selected)])
        end_to_end.append(values[0])
        feature.append(values[1])
        model.append(values[2])
        postprocess.append(values[3])
    report = {
        "schema_version": 1,
        "method": "wlcr_full_seed42",
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_affinity": len(os.sched_getaffinity(0)),
        },
        "protocol": {
            "batch_size_requests": 1,
            "outputs_per_request": 96,
            "threads": 1,
            "warmup_requests": warmup,
            "measured_requests": repetitions,
            "unique_request_samples": len(selected),
            "request_identity_sha256": identity_hash(selected),
            "boundary": "in-memory parsed request through features, four model predictions, and inverse transform; transport and payload deserialization excluded",
            "feature_cache": False,
        },
        "latency": {
            "end_to_end": percentiles(end_to_end),
            "feature_construction": percentiles(feature),
            "model_prediction": percentiles(model),
            "postprocess": percentiles(postprocess),
        },
        "memory": {
            "model_size_bytes": model_bytes,
            "process_rss_after_model_and_data_load_bytes": rss_after_load,
        },
    }
    write_json_atomic(root / OUTPUT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    run(args.warmup, args.repetitions, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
