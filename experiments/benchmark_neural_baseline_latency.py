from __future__ import annotations

"""Single-thread CPU batch-one latency benchmark for frozen neural baselines."""

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

import experiments.train_neural_baselines as neural


OUTPUT = Path("artifacts/revision4/latency_neural.json")
SOURCE = Path("artifacts/paper_neural_baselines_v1/results")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def timestamp_from_hour(hour: int) -> datetime:
    return datetime(1970, 1, 1) + timedelta(hours=int(hour))


def identity_hash(cells: np.ndarray, hours: np.ndarray) -> str:
    payload = "\n".join(
        f"{str(cell)}|{timestamp_from_hour(int(hour)).isoformat()}"
        for cell, hour in zip(cells, hours)
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


def normalization(payload) -> neural.Normalization:
    return neural.Normalization(
        input_mean=tuple(float(value) for value in payload["input_mean"]),
        input_std=tuple(float(value) for value in payload["input_std"]),
        target_mean=tuple(float(value) for value in payload["target_mean"]),
        target_std=tuple(float(value) for value in payload["target_std"]),
    )


def run_model(
    model_name: str,
    raw: np.ndarray,
    cells: np.ndarray,
    target_hours: np.ndarray,
    warmup: int,
    repetitions: int,
) -> dict[str, object]:
    root = project_root()
    path = root / SOURCE / "models" / f"{model_name}_seed42.pt"
    checkpoint = torch.load(path, map_location="cpu")
    norm = normalization(checkpoint["normalization"])
    model = neural.build_model(model_name, checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    input_mean = np.asarray(norm.input_mean, dtype=np.float32)
    input_std = np.asarray(norm.input_std, dtype=np.float32)
    rss_after_load = rss_bytes()

    def one(history: np.ndarray) -> tuple[float, float, float, float]:
        started = time.perf_counter_ns()
        values, masks, _ = neural.median_fill_and_mask(history)
        values = (values - input_mean[None, :]) / input_std[None, :]
        inputs = torch.from_numpy(
            np.concatenate((values, masks.astype(np.float32)), axis=1)[None, :, :]
        )
        after_preprocess = time.perf_counter_ns()
        with torch.no_grad():
            normalized = model(inputs)
        after_model = time.perf_counter_ns()
        output = neural.inverse_target(normalized.numpy(), norm)[0]
        checksum = float(np.sum(output))
        finished = time.perf_counter_ns()
        if output.shape != (24, 4) or not np.isfinite(checksum) or checksum <= 0.0:
            raise ValueError("invalid neural latency output")
        return (
            (finished - started) / 1e6,
            (after_preprocess - started) / 1e6,
            (after_model - after_preprocess) / 1e6,
            (finished - after_model) / 1e6,
        )

    for index in range(warmup):
        one(raw[index % len(raw)])
    end_to_end = []
    preprocessing = []
    prediction = []
    postprocess = []
    for index in range(repetitions):
        values = one(raw[index % len(raw)])
        end_to_end.append(values[0])
        preprocessing.append(values[1])
        prediction.append(values[2])
        postprocess.append(values[3])
    return {
        "method": f"{model_name}_seed42",
        "protocol": {
            "batch_size_requests": 1,
            "outputs_per_request": 96,
            "threads": 1,
            "warmup_requests": warmup,
            "measured_requests": repetitions,
            "unique_request_samples": len(raw),
            "request_identity_sha256": identity_hash(cells, target_hours),
            "boundary": "in-memory parsed numeric history through median fill, masks, normalization, CPU model, and inverse transform; transport and payload deserialization excluded",
            "preprocessing_cache": False,
        },
        "latency": {
            "end_to_end": percentiles(end_to_end),
            "preprocessing": percentiles(preprocessing),
            "model_prediction": percentiles(prediction),
            "postprocess": percentiles(postprocess),
        },
        "memory": {
            "model_size_bytes": path.stat().st_size,
            "process_rss_after_model_and_data_load_bytes": rss_after_load,
        },
    }


def run(warmup: int, repetitions: int, samples: int, models: list[str]):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = project_root()
    train = neural.resolve_train_path()
    series = neural.read_training_series(train)
    arrays, report = neural.build_window_arrays(series)
    dataset = neural.CachedDataset(root=Path("."), **arrays)
    holdout_all = dataset.indices_for_dates(neural.HOLDOUT_DATES)
    holdout = np.asarray(
        sorted(
            (int(index) for index in holdout_all),
            key=lambda index: (
                str(arrays["cells"][index]),
                int(arrays["target_start_hours"][index]),
            ),
        )[:samples],
        dtype=np.int64,
    )
    values = np.expm1(np.asarray(arrays["x_values"])[holdout].astype(np.float64))
    observed = np.asarray(arrays["x_masks"])[holdout].astype(bool)
    values[~observed] = np.nan
    cells = np.asarray(arrays["cells"])[holdout]
    target_hours = np.asarray(arrays["target_start_hours"])[holdout]
    results = [
        run_model(
            model,
            values,
            cells,
            target_hours,
            warmup,
            repetitions,
        )
        for model in models
    ]
    output = {
        "schema_version": 1,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_affinity": len(os.sched_getaffinity(0)),
            "torch": torch.__version__,
        },
        "dataset_continuous_windows": report["continuous_windows"],
        "methods": results,
    }
    write_json_atomic(root / OUTPUT, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--models", default="dlinear,patchtst")
    args = parser.parse_args()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    if any(value not in {"dlinear", "patchtst"} for value in models):
        raise ValueError("only dlinear and patchtst are supported")
    run(args.warmup, args.repetitions, args.samples, models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
