from __future__ import annotations

"""LightGBM-only subprocess for Revision-7 unseen-cell folds.

This file intentionally runs in the repository's LightGBM/NumPy-2 environment,
separate from the PyTorch/NumPy-1 environment used by the neural models.
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np

from Model.traffic_window_forecasting import OUTPUT_FLOOR
from experiments.train_lightgbm_baseline import MODEL_PARAMS


FORECAST_HOURS = 24
TARGET_COUNT = 4
SEED = 42


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def fold_mapping(cells: Sequence[str]) -> dict[str, int]:
    ordered = sorted(set(str(cell) for cell in cells))
    return {cell: index % 5 for index, cell in enumerate(ordered)}


def train_method(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    train_window_mask: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_window_mask: np.ndarray,
    rounds: Sequence[int],
    physical_device: int,
    model_dir: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    train_rows = np.repeat(np.asarray(train_window_mask, dtype=bool), FORECAST_HOURS)
    evaluation_rows = np.repeat(
        np.asarray(evaluation_window_mask, dtype=bool), FORECAST_HOURS
    )
    train_x = np.asarray(features[train_rows], dtype=np.float32)
    train_y = np.asarray(targets[train_rows], dtype=np.float32)
    evaluate_x = np.asarray(evaluation_features[evaluation_rows], dtype=np.float32)
    output = np.empty((len(evaluate_x), TARGET_COUNT), dtype=np.float32)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_sizes: list[int] = []
    started = time.perf_counter()
    for metric in range(TARGET_COUNT):
        params = dict(MODEL_PARAMS)
        params.update(
            {
                "seed": SEED,
                "feature_fraction_seed": SEED,
                "bagging_seed": SEED,
                "data_random_seed": SEED,
                "gpu_device_id": int(physical_device),
                "num_threads": max(1, int(MODEL_PARAMS["num_threads"]) // 4),
            }
        )
        valid = np.isfinite(train_y[:, metric])
        booster = lgb.train(
            params,
            lgb.Dataset(train_x[valid], label=train_y[valid, metric]),
            num_boost_round=int(rounds[metric]),
        )
        output[:, metric] = np.maximum(
            np.expm1(booster.predict(evaluate_x)), OUTPUT_FLOOR
        ).astype(np.float32)
        destination = model_dir / f"metric_{metric}.txt"
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        booster.save_model(str(temporary))
        os.replace(temporary, destination)
        model_sizes.append(destination.stat().st_size)
    if len(output) % FORECAST_HOURS:
        raise ValueError("LightGBM evaluation rows do not form complete windows")
    prediction = output.reshape(-1, FORECAST_HOURS, TARGET_COUNT)
    if not np.all(np.isfinite(prediction)) or np.any(prediction <= 0.0):
        raise ValueError("LightGBM unseen predictions must be finite and positive")
    return prediction, {
        "rounds": [int(value) for value in rounds],
        "train_windows": int(np.sum(train_window_mask)),
        "evaluation_windows": int(np.sum(evaluation_window_mask)),
        "feature_count": int(features.shape[1]),
        "training_and_prediction_seconds": time.perf_counter() - started,
        "model_size_bytes": int(sum(model_sizes)),
    }


def run(args: argparse.Namespace) -> int:
    cache = Path(args.cache).resolve(strict=True)
    output = Path(args.output).resolve(strict=False)
    matrix_dir = cache / "matrices"
    final_cells = np.load(matrix_dir / "final_cells.npy", allow_pickle=False).astype(str)
    holdout_cells = np.load(
        matrix_dir / "holdout_cells.npy", allow_pickle=False
    ).astype(str)
    mapping = fold_mapping(np.concatenate((final_cells, holdout_cells)).tolist())
    train_window_mask = np.asarray(
        [mapping[str(cell)] != args.fold for cell in final_cells], dtype=bool
    )
    evaluation_window_mask = np.asarray(
        [mapping[str(cell)] == args.fold for cell in holdout_cells], dtype=bool
    )
    if args.smoke:
        train_positions = np.flatnonzero(train_window_mask)[:256]
        evaluation_positions = np.flatnonzero(evaluation_window_mask)[:128]
        train_window_mask[:] = False
        evaluation_window_mask[:] = False
        train_window_mask[train_positions] = True
        evaluation_window_mask[evaluation_positions] = True
    rounds = json.loads((cache / "cache_report.json").read_text(encoding="utf-8"))
    original_rounds = rounds["original_wlcr_rounds"]
    standard_rounds = rounds["standard_stat_rounds"]
    if args.smoke:
        original_rounds = [2, 2, 2, 2]
        standard_rounds = [2, 2, 2, 2]
    original, original_report = train_method(
        features=np.load(
            matrix_dir / "wlcr_final_features.npy", mmap_mode="r", allow_pickle=False
        ),
        targets=np.load(
            matrix_dir / "wlcr_final_targets.npy", mmap_mode="r", allow_pickle=False
        ),
        train_window_mask=train_window_mask,
        evaluation_features=np.load(
            matrix_dir / "wlcr_holdout_features.npy", mmap_mode="r", allow_pickle=False
        ),
        evaluation_window_mask=evaluation_window_mask,
        rounds=original_rounds,
        physical_device=args.physical_device,
        model_dir=output / "worker" / f"fold{args.fold}_original_wlcr_models",
    )
    standard, standard_report = train_method(
        features=np.load(
            matrix_dir / "standard_final_features.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        targets=np.load(
            matrix_dir / "standard_final_targets.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        train_window_mask=train_window_mask,
        evaluation_features=np.load(
            matrix_dir / "standard_holdout_features.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        evaluation_window_mask=evaluation_window_mask,
        rounds=standard_rounds,
        physical_device=args.physical_device,
        model_dir=output / "worker" / f"fold{args.fold}_standard_models",
    )
    worker_dir = output / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = worker_dir / f"fold{args.fold}_lgbm_predictions.npz"
    np.savez_compressed(
        prediction_path,
        original_wlcr_lightgbm=original,
        standard_stat_lightgbm=standard,
    )
    report = {
        "fold": args.fold,
        "physical_gpu": args.physical_device,
        "original_wlcr_lightgbm": original_report,
        "standard_stat_lightgbm": standard_report,
        "prediction_file": str(prediction_path.relative_to(output)),
        "finals_test_opened": False,
    }
    atomic_json(worker_dir / f"fold{args.fold}_lgbm.json", report)
    print(json.dumps({"status": "complete", "fold": args.fold}))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--cache", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--fold", type=int, required=True)
    value.add_argument("--physical-device", type=int, required=True)
    value.add_argument("--smoke", action="store_true")
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except Exception:
        traceback.print_exc()
        raise
