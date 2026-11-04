from __future__ import annotations

"""Neural history-length and missingness audits for manuscript revision 4.

The script reads only the registered training trace.  History models are
retrained with the same physical 336-hour request tensor while unavailable
prefixes are masked, matching the WLCR audit.  Missingness tests reuse the
frozen seed-42 DLinear and PatchTST models and the deterministic masks shared
with the LightGBM experiment.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

import experiments.train_neural_baselines as neural
from experiments.lightgbm_experiment_helpers import (
    HISTORY_LENGTHS,
    MISSING_MECHANISMS,
    MISSING_RATES,
    additional_missing_mask,
)


EXPERIMENT_VERSION = "manuscript_revision4_neural_v1"
OUTPUT_ROOT = Path("artifacts/revision4")
SOURCE_ROOT = Path("artifacts/paper_neural_baselines_v1/results")
MODELS = ("dlinear", "patchtst")
SEED = 42


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def target_datetime(hour: int) -> datetime:
    return datetime(1970, 1, 1) + timedelta(hours=int(hour))


def raw_histories(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    values = np.expm1(np.asarray(arrays["x_values"], dtype=np.float64))
    observed = np.asarray(arrays["x_masks"], dtype=bool)
    values[~observed] = np.nan
    return values


def fill_histories(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(values, dtype=np.float64)
    observed = np.isfinite(history)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(history, axis=1)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(observed, history, medians[:, None, :])
    if np.any(~np.isfinite(filled)) or np.any(filled < 0.0):
        raise ValueError("history filling produced invalid traffic values")
    return np.log1p(filled).astype(np.float32), observed.astype(np.uint8)


def arrays_for_history(
    arrays: Mapping[str, np.ndarray], available_hours: int
) -> dict[str, np.ndarray]:
    if available_hours not in HISTORY_LENGTHS:
        raise ValueError(f"unsupported history length: {available_hours}")
    history = raw_histories(arrays)
    history[:, : neural.INPUT_HOURS - available_hours, :] = np.nan
    x_values, x_masks = fill_histories(history)
    return {
        name: (
            x_values
            if name == "x_values"
            else x_masks
            if name == "x_masks"
            else np.asarray(value)
        )
        for name, value in arrays.items()
    }


def normalization_from_payload(payload: Mapping[str, object]) -> neural.Normalization:
    return neural.Normalization(
        input_mean=tuple(float(value) for value in payload["input_mean"]),
        input_std=tuple(float(value) for value in payload["input_std"]),
        target_mean=tuple(float(value) for value in payload["target_mean"]),
        target_std=tuple(float(value) for value in payload["target_std"]),
    )


def metric_summary(
    actual: np.ndarray,
    prediction: np.ndarray,
    mase_scales: np.ndarray,
) -> dict[str, float]:
    task = neural.combined_scores(actual, prediction)
    complete = neural.complete_filter(actual)
    indicator_rows = [
        neural.standard_metric_values(
            actual, prediction, mase_scales, complete, metric
        )
        for metric in range(neural.TARGET_COUNT)
    ]
    mase = [float(row["mase"]) for row in indicator_rows if row["mase"] is not None]
    return {
        "unfiltered_wape": float(
            np.mean([float(row["wape"]) for row in indicator_rows])
        ),
        "unfiltered_smape": float(
            np.mean([float(row["smape"]) for row in indicator_rows])
        ),
        "unfiltered_mae": float(
            np.mean([float(row["mae"]) for row in indicator_rows])
        ),
        "unfiltered_rmse": float(
            np.mean([float(row["rmse"]) for row in indicator_rows])
        ),
        "unfiltered_mase": float(np.mean(mase)) if mase else float("nan"),
        "ths_mapeauc": float(task["mape_auc"]),
        "filtered_mean_mape": float(task["mean_mape"]),
    }


def history_output(hours: int) -> Path:
    return project_root() / OUTPUT_ROOT / "models" / f"neural_history_{hours}h"


def history_cache_valid(hours: int, model: str) -> bool:
    root = history_output(hours)
    report = root / "job_reports" / f"{model}_seed{SEED}.json"
    prediction = root / "worker_predictions" / f"{model}_seed{SEED}.npy"
    checkpoint = root / "models" / f"{model}_seed{SEED}.pt"
    if not (report.exists() and prediction.exists() and checkpoint.exists()):
        return False
    payload = json.loads(report.read_text(encoding="utf-8"))
    return (
        payload.get("model") == model
        and int(payload.get("seed", -1)) == SEED
        and payload.get("holdout_windows") == 5110
        and payload.get("model_sha256") == neural.sha256_file(checkpoint)
        and payload.get("prediction_sha256") == neural.sha256_file(prediction)
    )


def run_history_worker(args: argparse.Namespace) -> int:
    report = neural.run_worker(
        dataset_cache=Path(args.dataset_cache).resolve(strict=True),
        output=history_output(args.history_hours),
        model_name=args.model,
        seed=SEED,
        physical_device=args.physical_device,
        max_epochs=neural.DEFAULT_MAX_EPOCHS,
        patience=neural.DEFAULT_PATIENCE,
        batch_size=neural.DEFAULT_BATCH_SIZE,
        smoke=False,
    )
    report["revision4"] = {
        "experiment_version": EXPERIMENT_VERSION,
        "available_history_hours": args.history_hours,
        "physical_window_hours": neural.INPUT_HOURS,
        "prefix_policy": "earlier traffic values unavailable and median-filled with zero masks",
    }
    path = history_output(args.history_hours) / "job_reports" / f"{args.model}_seed{SEED}.json"
    write_json_atomic(path, report)
    print(json.dumps({"status": "complete", "report": str(path)}, ensure_ascii=False))
    return 0


def launch_history_jobs(cache_paths: Mapping[int, Path]) -> list[dict[str, object]]:
    jobs = []
    device = 0
    for hours in (72, 168):
        for model in MODELS:
            if history_cache_valid(hours, model):
                jobs.append({"hours": hours, "model": model, "cached": True})
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--dataset-cache",
                str(cache_paths[hours]),
                "--history-hours",
                str(hours),
                "--model",
                model,
                "--physical-device",
                str(device),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            environment["OMP_NUM_THREADS"] = "4"
            environment["MKL_NUM_THREADS"] = "4"
            process = subprocess.Popen(
                command,
                cwd=project_root(),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            jobs.append(
                {
                    "hours": hours,
                    "model": model,
                    "device": device,
                    "cached": False,
                    "process": process,
                    "command": command,
                }
            )
            device += 1
    failures = []
    records = []
    for job in jobs:
        if job["cached"]:
            records.append({key: value for key, value in job.items() if key != "process"})
            continue
        process = job["process"]
        stdout, stderr = process.communicate()
        record = {key: value for key, value in job.items() if key not in {"process", "command"}}
        record.update(
            {
                "returncode": process.returncode,
                "stdout": stdout[-2000:],
                "stderr": stderr[-4000:],
            }
        )
        records.append(record)
        if process.returncode != 0:
            failures.append(record)
    if failures:
        raise RuntimeError(f"revision4 neural history jobs failed: {failures}")
    return records


def existing_seed42_prediction(model: str) -> np.ndarray:
    path = project_root() / SOURCE_ROOT / "worker_predictions" / f"{model}_seed{SEED}.npy"
    return np.load(path, allow_pickle=False)


def collect_history_rows(
    arrays: Mapping[str, np.ndarray], holdout: np.ndarray
) -> list[dict[str, object]]:
    actual = np.asarray(arrays["targets"])[holdout]
    scales = np.asarray(arrays["mase_scales"])[holdout]
    rows = []
    for hours in HISTORY_LENGTHS:
        for model in MODELS:
            prediction = (
                existing_seed42_prediction(model)
                if hours == 336
                else np.load(
                    history_output(hours)
                    / "worker_predictions"
                    / f"{model}_seed{SEED}.npy",
                    allow_pickle=False,
                )
            )
            rows.append(
                {
                    "model": model,
                    "seed": SEED,
                    "history_hours": hours,
                    "training_policy": (
                        "frozen registered model"
                        if hours == 336
                        else "reselected and retrained under the same two-configuration protocol"
                    ),
                    **metric_summary(actual, prediction, scales),
                }
            )
    return rows


def load_frozen_model(model_name: str, device: torch.device):
    path = project_root() / SOURCE_ROOT / "models" / f"{model_name}_seed{SEED}.pt"
    checkpoint = torch.load(path, map_location="cpu")
    model = neural.build_model(model_name, checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, normalization_from_payload(checkpoint["normalization"]), path


def prepared_corrupted_inputs(
    base_raw: np.ndarray,
    cells: np.ndarray,
    target_hours: np.ndarray,
    mechanism: str,
    rate: float,
    normalization: neural.Normalization,
) -> torch.Tensor:
    corrupted = np.asarray(base_raw, dtype=np.float64).copy()
    if rate > 0.0:
        for index in range(len(corrupted)):
            corrupted[index][
                additional_missing_mask(
                    cell=str(cells[index]),
                    target_start=target_datetime(int(target_hours[index])),
                    mechanism=mechanism,
                    rate=rate,
                )
            ] = np.nan
    values, masks = fill_histories(corrupted)
    mean = np.asarray(normalization.input_mean, dtype=np.float32)
    std = np.asarray(normalization.input_std, dtype=np.float32)
    values = (values - mean[None, None, :]) / std[None, None, :]
    return torch.from_numpy(
        np.concatenate((values, masks.astype(np.float32)), axis=2)
    )


def missingness_rows(
    arrays: Mapping[str, np.ndarray], holdout: np.ndarray
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    actual = np.asarray(arrays["targets"])[holdout]
    scales = np.asarray(arrays["mase_scales"])[holdout]
    cells = np.asarray(arrays["cells"])[holdout]
    target_hours = np.asarray(arrays["target_start_hours"])[holdout]
    base = raw_histories(arrays)[holdout]
    models = {}
    model_records = []
    for name in MODELS:
        model, normalization, path = load_frozen_model(name, device)
        models[name] = (model, normalization)
        model_records.append(
            {
                "model": name,
                "checkpoint": str(path.relative_to(project_root())),
                "checkpoint_sha256": neural.sha256_file(path),
                "device": str(device),
            }
        )
    rows = []
    for mechanism in MISSING_MECHANISMS:
        for rate in MISSING_RATES:
            for name, (model, normalization) in models.items():
                inputs = prepared_corrupted_inputs(
                    base, cells, target_hours, mechanism, rate, normalization
                )
                prediction_norm = neural.predict_normalized(
                    model,
                    inputs,
                    batch_size=neural.DEFAULT_BATCH_SIZE,
                    device=device,
                )
                prediction = neural.inverse_target(prediction_norm, normalization)
                rows.append(
                    {
                        "model": name,
                        "seed": SEED,
                        "mechanism": mechanism,
                        "requested_missing_rate": rate,
                        "training_policy": "frozen 336-hour seed-42 model",
                        **metric_summary(actual, prediction, scales),
                    }
                )
    return rows, model_records


def run() -> dict[str, object]:
    started = time.perf_counter()
    root = project_root()
    train = neural.resolve_train_path()
    input_hash = neural.sha256_file(train)
    series = neural.read_training_series(train)
    arrays, dataset_report = neural.build_window_arrays(series)
    dataset = neural.CachedDataset(root=Path("."), **arrays)
    leakage = neural.leakage_checks(dataset)
    holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)

    with tempfile.TemporaryDirectory(prefix="revision4-neural-history-") as temporary:
        temporary_root = Path(temporary)
        cache_paths = {}
        for hours in (72, 168):
            transformed = arrays_for_history(arrays, hours)
            cache = temporary_root / f"history_{hours}h"
            neural.write_dataset_cache(cache, transformed)
            cache_paths[hours] = cache
        history_jobs = launch_history_jobs(cache_paths)

    history = collect_history_rows(arrays, holdout)
    missingness, model_records = missingness_rows(arrays, holdout)
    output = root / OUTPUT_ROOT
    write_csv_atomic(output / "revision4_history_neural.csv", history)
    write_csv_atomic(output / "revision4_missingness_neural.csv", missingness)
    if neural.sha256_file(train) != input_hash:
        raise RuntimeError("registered training file changed during neural audit")
    report = {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_visible_count": torch.cuda.device_count(),
        },
        "registered_input": {
            "path": str(train.relative_to(root)),
            "size_bytes": train.stat().st_size,
            "sha256": input_hash,
        },
        "dataset_report": dataset_report,
        "leakage_checks": leakage,
        "history_jobs": history_jobs,
        "frozen_missingness_models": model_records,
        "protocol": {
            "history_hours": list(HISTORY_LENGTHS),
            "physical_tensor_hours": neural.INPUT_HOURS,
            "missing_mechanisms": list(MISSING_MECHANISMS),
            "missing_rates": list(MISSING_RATES),
            "mask_contract": "experiments.lightgbm_experiment_helpers.additional_missing_mask",
            "history_selection": "same two configurations, fit/inner dates, early stopping rule, seed 42",
            "missingness_selection": "none; registered seed-42 checkpoints frozen",
            "finals_test_opened": False,
        },
        "outputs": {
            "history": "artifacts/revision4/revision4_history_neural.csv",
            "missingness": "artifacts/revision4/revision4_missingness_neural.csv",
        },
    }
    write_json_atomic(output / "revision4_neural_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset-cache")
    parser.add_argument("--history-hours", type=int)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--physical-device", type=int)
    args = parser.parse_args()
    if args.worker:
        required = (
            args.dataset_cache,
            args.history_hours,
            args.model,
            args.physical_device,
        )
        if any(value is None for value in required):
            parser.error("worker mode requires dataset, history, model, and device")
        return run_history_worker(args)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
