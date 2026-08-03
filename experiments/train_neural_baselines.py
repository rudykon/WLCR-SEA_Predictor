from __future__ import annotations

"""Leakage-controlled neural baselines for the paper revision.

This module implements DLinear, NLinear, a lightweight PatchTST, and a direct
multi-horizon GRU-D control directly with PyTorch.  It reads only the
registered ``data/train_data.csv`` file, constructs continuous 336-hour
histories with 24-hour targets, and evaluates a single model on the fixed
seven-day holdout.  Finals test traffic, archived implementations, cell
identifiers, parameters, and weather are never model inputs.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments import missingness_protocol as missingness


SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "paper_neural_baselines_reproduction_v1"
REGISTERED_TRAIN = Path("data/train_data.csv")
REGISTERED_TRAIN_SHA256 = (
    "d274407a3db51ba4871851ab447bcc75202bb567337464d85ea280662f3bf1da"
)
DEFAULT_OUTPUT_ROOT = Path("artifacts/reproduction/neural_baselines")
INPUT_HOURS = 336
FORECAST_HOURS = 24
TARGET_COUNT = 4
MODEL_INPUT_CHANNELS = 8
OFFICIAL_THRESHOLDS = np.asarray((0.2, 0.3, 0.4, 0.5), dtype=np.float64)
METRIC_NAMES = (
    "ul_active_users",
    "dl_active_users",
    "dl_prb",
    "ul_prb",
)
TRAFFIC_COLUMNS = (
    "小区上行平均激活用户数",
    "小区下行平均激活用户数",
    "下行平均使用的PRB个数",
    "上行平均使用的PRB个数",
)

FIT_DATES = tuple(date(2024, 8, day) for day in range(3, 10))
INNER_DATES = tuple(date(2024, 8, day) for day in range(10, 12))
HOLDOUT_DATES = tuple(date(2024, 8, day) for day in range(12, 19))
SEEDS = (42, 43, 44, 45, 46)
TUNING_CANDIDATES = 2
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_BATCH_SIZE = 128

MODEL_CONFIGS: dict[str, tuple[dict[str, object], ...]] = {
    "dlinear": (
        {
            "name": "dlinear_k25_lr1e3",
            "kernel_size": 25,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
        {
            "name": "dlinear_k49_lr5e4",
            "kernel_size": 49,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
        },
    ),
    "nlinear": (
        {
            "name": "nlinear_lr1e3",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
        {
            "name": "nlinear_lr5e4",
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
        },
    ),
    "patchtst": (
        {
            "name": "patchtst_d32_l1",
            "patch_length": 24,
            "stride": 12,
            "d_model": 32,
            "heads": 4,
            "layers": 1,
            "feedforward": 64,
            "dropout": 0.10,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
        {
            "name": "patchtst_d48_l2",
            "patch_length": 24,
            "stride": 12,
            "d_model": 48,
            "heads": 4,
            "layers": 2,
            "feedforward": 96,
            "dropout": 0.10,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
        },
    ),
    "grud_direct": (
        {
            "name": "grud_direct_h32_lr1e3",
            "hidden_size": 32,
            "dropout": 0.10,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
        {
            "name": "grud_direct_h48_lr5e4",
            "hidden_size": 48,
            "dropout": 0.10,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
        },
    ),
}


@dataclass(frozen=True)
class Normalization:
    input_mean: tuple[float, ...]
    input_std: tuple[float, ...]
    target_mean: tuple[float, ...]
    target_std: tuple[float, ...]


@dataclass(frozen=True)
class CachedDataset:
    root: Path
    x_values: np.ndarray
    x_masks: np.ndarray
    targets: np.ndarray
    target_masks: np.ndarray
    mase_scales: np.ndarray
    cells: np.ndarray
    target_start_hours: np.ndarray
    history_end_hours: np.ndarray

    def indices_for_dates(self, selected: Iterable[date]) -> np.ndarray:
        selected_days = {
            int((value - date(1970, 1, 1)).days) for value in selected
        }
        days = self.target_start_hours // 24
        return np.flatnonzero(np.isin(days, sorted(selected_days)))


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_train_path() -> Path:
    root = project_root()
    path = (root / REGISTERED_TRAIN).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("registered training path escapes the project root")
    actual = sha256_file(path)
    if actual != REGISTERED_TRAIN_SHA256:
        raise ValueError(
            f"registered training SHA256 mismatch: {actual} != "
            f"{REGISTERED_TRAIN_SHA256}"
        )
    return path


def resolve_output(path_text: str) -> Path:
    root = project_root()
    requested = Path(path_text)
    if not requested.is_absolute():
        requested = root / requested
    requested = requested.resolve(strict=False)
    allowed = (root / DEFAULT_OUTPUT_ROOT).resolve(strict=False)
    if requested != allowed and not requested.is_relative_to(allowed):
        raise ValueError(f"outputs must remain under {DEFAULT_OUTPUT_ROOT}")
    return requested


def parse_timestamp(value: str) -> datetime:
    token = value.strip()
    try:
        return datetime.fromisoformat(token)
    except ValueError:
        pass
    for pattern in ("%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(token, pattern)
        except ValueError:
            continue
    raise ValueError(f"invalid timestamp: {value!r}")


def read_training_series(path: Path) -> dict[str, tuple[list[datetime], np.ndarray]]:
    timestamps: dict[str, list[datetime]] = {}
    values: dict[str, list[list[float]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ("时间", "小区名称", *TRAFFIC_COLUMNS)
        if tuple(reader.fieldnames or ()) != required:
            raise ValueError("training CSV header does not match the registered schema")
        for row_number, row in enumerate(reader, start=2):
            cell = row["小区名称"].strip()
            if not cell:
                raise ValueError(f"empty cell identifier at row {row_number}")
            timestamp = parse_timestamp(row["时间"])
            metrics: list[float] = []
            for column in TRAFFIC_COLUMNS:
                token = row[column].strip()
                if token == "NIL":
                    metrics.append(float("nan"))
                    continue
                value = float(token)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"invalid traffic value at row {row_number}/{column}: {token}"
                    )
                metrics.append(value)
            timestamps.setdefault(cell, []).append(timestamp)
            values.setdefault(cell, []).append(metrics)
    return {
        cell: (timestamps[cell], np.asarray(values[cell], dtype=np.float64))
        for cell in timestamps
    }


def history_mase_scale(history: np.ndarray) -> np.ndarray:
    if history.shape != (INPUT_HOURS, TARGET_COUNT):
        raise ValueError(f"unexpected history shape: {history.shape}")
    differences = np.abs(history[168:] - history[:-168])
    scales = np.full(TARGET_COUNT, np.nan, dtype=np.float32)
    for metric in range(TARGET_COUNT):
        finite = np.isfinite(differences[:, metric])
        if np.any(finite):
            value = float(np.mean(differences[finite, metric]))
            if value > 1e-12:
                scales[metric] = value
    return scales


def median_fill_and_mask(history: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if history.shape != (INPUT_HOURS, TARGET_COUNT):
        raise ValueError(f"unexpected history shape: {history.shape}")
    observed = np.isfinite(history)
    filled = history.copy()
    all_missing = 0
    for metric in range(TARGET_COUNT):
        finite = observed[:, metric]
        if np.any(finite):
            fallback = float(np.median(history[finite, metric]))
        else:
            fallback = 0.0
            all_missing += 1
        filled[~finite, metric] = fallback
    return np.log1p(filled).astype(np.float32), observed.astype(np.uint8), all_missing


def build_window_arrays(
    series: Mapping[str, tuple[Sequence[datetime], np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    x_values: list[np.ndarray] = []
    x_masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    mase_scales: list[np.ndarray] = []
    cells: list[str] = []
    target_start_hours: list[int] = []
    history_end_hours: list[int] = []
    candidate_windows = 0
    discontinuous: list[dict[str, object]] = []
    all_missing_fallbacks = 0
    epoch = datetime(1970, 1, 1)

    for cell, (timestamps, metrics) in series.items():
        if len(timestamps) != len(metrics):
            raise ValueError(f"timestamp/value length mismatch for cell {cell}")
        for index in range(INPUT_HOURS, len(timestamps) - FORECAST_HOURS + 1, 24):
            candidate_windows += 1
            start = index - INPUT_HOURS
            stop = index + FORECAST_HOURS
            local_times = timestamps[start:stop]
            gaps = [
                (position, int((right - left).total_seconds() // 3600))
                for position, (left, right) in enumerate(
                    zip(local_times[:-1], local_times[1:])
                )
                if right - left != timedelta(hours=1)
            ]
            if gaps:
                discontinuous.append(
                    {
                        "cell": cell,
                        "target_start": timestamps[index].isoformat(sep=" "),
                        "gaps": [
                            {"after_offset": position, "elapsed_hours": elapsed}
                            for position, elapsed in gaps
                        ],
                    }
                )
                continue
            history = metrics[start:index]
            target = metrics[index:stop]
            filled, mask, missing_count = median_fill_and_mask(history)
            all_missing_fallbacks += missing_count
            x_values.append(filled)
            x_masks.append(mask)
            targets.append(target.astype(np.float32))
            target_masks.append(np.isfinite(target).astype(np.uint8))
            mase_scales.append(history_mase_scale(history))
            cells.append(cell)
            target_start_hours.append(
                int((timestamps[index] - epoch).total_seconds() // 3600)
            )
            history_end_hours.append(
                int((timestamps[index - 1] - epoch).total_seconds() // 3600)
            )

    arrays = {
        "x_values": np.asarray(x_values, dtype=np.float32),
        "x_masks": np.asarray(x_masks, dtype=np.uint8),
        "targets": np.asarray(targets, dtype=np.float32),
        "target_masks": np.asarray(target_masks, dtype=np.uint8),
        "mase_scales": np.asarray(mase_scales, dtype=np.float32),
        "cells": np.asarray(cells, dtype="U64"),
        "target_start_hours": np.asarray(target_start_hours, dtype=np.int64),
        "history_end_hours": np.asarray(history_end_hours, dtype=np.int64),
    }
    report = {
        "candidate_windows": candidate_windows,
        "continuous_windows": len(cells),
        "discontinuous_windows": discontinuous,
        "all_missing_indicator_fallbacks": all_missing_fallbacks,
        "unique_cells": len(series),
        "date_counts": {},
    }
    for hour in arrays["target_start_hours"]:
        value = (epoch + timedelta(hours=int(hour))).date().isoformat()
        report["date_counts"][value] = report["date_counts"].get(value, 0) + 1
    return arrays, report


def write_dataset_cache(root: Path, arrays: Mapping[str, np.ndarray]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(root / f"{name}.npy", np.asarray(array), allow_pickle=False)


def load_dataset_cache(root: Path) -> CachedDataset:
    def loaded(name: str) -> np.ndarray:
        return np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)

    return CachedDataset(
        root=root,
        x_values=loaded("x_values"),
        x_masks=loaded("x_masks"),
        targets=loaded("targets"),
        target_masks=loaded("target_masks"),
        mase_scales=loaded("mase_scales"),
        cells=loaded("cells"),
        target_start_hours=loaded("target_start_hours"),
        history_end_hours=loaded("history_end_hours"),
    )


def leakage_checks(dataset: CachedDataset) -> dict[str, object]:
    fit = dataset.indices_for_dates(FIT_DATES)
    inner = dataset.indices_for_dates(INNER_DATES)
    holdout = dataset.indices_for_dates(HOLDOUT_DATES)
    if len(fit) != 5_115 or len(inner) != 1_460 or len(holdout) != 5_110:
        raise ValueError(
            "fixed split count mismatch: "
            f"fit={len(fit)}, inner={len(inner)}, holdout={len(holdout)}"
        )
    if not np.all(dataset.history_end_hours + 1 == dataset.target_start_hours):
        raise ValueError("one or more histories do not end one hour before their targets")
    if set(fit).intersection(inner) or set(fit).intersection(holdout) or set(inner).intersection(holdout):
        raise ValueError("split indices overlap")
    if np.max(dataset.target_start_hours[np.concatenate((fit, inner))]) >= np.min(
        dataset.target_start_hours[holdout]
    ):
        raise ValueError("training/selection dates do not precede holdout")
    return {
        "train_source": str(REGISTERED_TRAIN),
        "finals_test_opened": False,
        "input_hours": INPUT_HOURS,
        "forecast_hours": FORECAST_HOURS,
        "history_ends_one_hour_before_target": True,
        "fit_inner_holdout_indices_disjoint": True,
        "fit_dates": [str(value) for value in FIT_DATES],
        "inner_dates": [str(value) for value in INNER_DATES],
        "holdout_dates": [str(value) for value in HOLDOUT_DATES],
        "fit_windows": len(fit),
        "inner_windows": len(inner),
        "final_train_windows": len(fit) + len(inner),
        "holdout_windows": len(holdout),
        "explicit_cell_id_feature": False,
        "model_input_channels": [
            *[f"log1p_filled_{name}" for name in METRIC_NAMES],
            *[f"observed_mask_{name}" for name in METRIC_NAMES],
        ],
        "cross_window_traffic_features": False,
        "cell_overlap_is_expected_for_temporal_protocol": int(
            len(
                set(dataset.cells[np.concatenate((fit, inner))].tolist()).intersection(
                    dataset.cells[holdout].tolist()
                )
            )
        ),
    }


def compute_normalization(dataset: CachedDataset, indices: np.ndarray) -> Normalization:
    values = np.asarray(dataset.x_values[indices], dtype=np.float64)
    masks = np.asarray(dataset.x_masks[indices], dtype=bool)
    targets = np.asarray(dataset.targets[indices], dtype=np.float64)
    target_masks = np.asarray(dataset.target_masks[indices], dtype=bool)
    input_mean: list[float] = []
    input_std: list[float] = []
    target_mean: list[float] = []
    target_std: list[float] = []
    for metric in range(TARGET_COUNT):
        observed = values[:, :, metric][masks[:, :, metric]]
        if not len(observed):
            raise ValueError(f"normalization has no input observations for metric {metric}")
        input_mean.append(float(np.mean(observed)))
        input_std.append(max(float(np.std(observed)), 1e-6))
        valid_target = targets[:, :, metric][target_masks[:, :, metric]]
        if not len(valid_target):
            raise ValueError(f"normalization has no targets for metric {metric}")
        transformed = np.log1p(valid_target)
        target_mean.append(float(np.mean(transformed)))
        target_std.append(max(float(np.std(transformed)), 1e-6))
    return Normalization(
        tuple(input_mean), tuple(input_std), tuple(target_mean), tuple(target_std)
    )


def prepared_tensors(
    dataset: CachedDataset,
    indices: np.ndarray,
    normalization: Normalization,
    *,
    additional_missing: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare inputs plus untouched labels and target masks.

    Synthetic corruption affects only the input-history view. The target
    values and target observation mask are copied from the original dataset.
    """
    inputs = prepared_inputs(
        dataset,
        indices,
        normalization,
        additional_missing=additional_missing,
    )
    target_raw = np.asarray(dataset.targets[indices], dtype=np.float32)
    target_mask = np.asarray(dataset.target_masks[indices], dtype=np.float32)
    safe_target = np.where(target_mask > 0.0, target_raw, 0.0)
    target_log = np.log1p(safe_target)
    target_mean = np.asarray(normalization.target_mean, dtype=np.float32)
    target_std = np.asarray(normalization.target_std, dtype=np.float32)
    target_norm = (target_log - target_mean[None, None, :]) / target_std[None, None, :]
    target_norm[target_mask == 0.0] = 0.0
    return (
        inputs,
        torch.from_numpy(target_norm),
        torch.from_numpy(target_mask),
    )


def prepared_inputs(
    dataset: CachedDataset,
    indices: np.ndarray,
    normalization: Normalization,
    *,
    additional_missing: np.ndarray | None = None,
) -> torch.Tensor:
    """Prepare a leakage-safe request view after optional synthetic removal."""
    base_values = np.asarray(dataset.x_values[indices], dtype=np.float32)
    original_masks = np.asarray(dataset.x_masks[indices], dtype=bool)
    if additional_missing is None:
        extra = np.zeros_like(original_masks, dtype=bool)
    else:
        extra = np.asarray(additional_missing, dtype=bool)
        if extra.shape != original_masks.shape:
            raise ValueError("additional missing mask is misaligned")
    masks = original_masks & ~extra

    if np.any(extra):
        raw = np.expm1(base_values.astype(np.float64))
        visible = np.where(masks, raw, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            medians = np.nanmedian(visible, axis=1)
        medians = np.where(np.any(masks, axis=1), medians, 0.0)
        fallback_log = np.log1p(np.maximum(medians, 0.0)).astype(np.float32)
        values = np.where(masks, base_values, fallback_log[:, None, :])
    else:
        values = base_values.copy()

    input_mean = np.asarray(normalization.input_mean, dtype=np.float32)
    input_std = np.asarray(normalization.input_std, dtype=np.float32)
    values = (values - input_mean[None, None, :]) / input_std[None, None, :]
    inputs = np.concatenate((values, masks.astype(np.float32)), axis=2)
    return torch.from_numpy(inputs.astype(np.float32))


def training_augmentation(
    dataset: CachedDataset,
    indices: np.ndarray,
    *,
    augmentation: str,
    requested_rate: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Create one deterministic input-only training corruption view."""
    if augmentation not in {"clean", "mixed"}:
        raise ValueError("augmentation must be clean or mixed")
    if augmentation == "clean" and requested_rate != 0.0:
        raise ValueError("clean augmentation requires a zero requested rate")
    mechanism = "none" if augmentation == "clean" else "mixed"
    extra = missingness.global_corruption_mask(
        np.asarray(dataset.cells[indices]),
        np.asarray(dataset.history_end_hours[indices]),
        mechanism=mechanism,
        requested_rate=requested_rate,
        seed=seed,
    )
    report = missingness.corruption_statistics(
        np.asarray(dataset.x_masks[indices]),
        extra,
        cells=np.asarray(dataset.cells[indices]),
        history_end_hours=np.asarray(dataset.history_end_hours[indices]),
        mechanism=mechanism,
        requested_rate=requested_rate,
        seed=seed,
    )
    report["augmentation"] = augmentation
    report["labels_or_target_masks_modified"] = False
    report["normalization_uses_original_observed_training_values"] = True
    return extra, report


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("moving-average kernel must be odd and greater than one")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        left = values[:, :1, :].repeat(1, padding, 1)
        right = values[:, -1:, :].repeat(1, padding, 1)
        padded = torch.cat((left, values, right), dim=1)
        return self.pool(padded.transpose(1, 2)).transpose(1, 2)


class DLinear(nn.Module):
    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)
        self.seasonal = nn.Linear(INPUT_HOURS, FORECAST_HOURS)
        self.trend = nn.Linear(INPUT_HOURS, FORECAST_HOURS)
        self.channel_mixer = nn.Linear(MODEL_INPUT_CHANNELS, TARGET_COUNT)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        trend = self.moving_average(values)
        seasonal = values - trend
        projected = self.seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        projected = projected + self.trend(trend.transpose(1, 2)).transpose(1, 2)
        return self.channel_mixer(projected)


class NLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.Linear(INPUT_HOURS, FORECAST_HOURS)
        self.channel_mixer = nn.Linear(MODEL_INPUT_CHANNELS, TARGET_COUNT)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        anchor = values[:, -1:, :]
        centered = values - anchor
        projected = self.temporal(centered.transpose(1, 2)).transpose(1, 2)
        return self.channel_mixer(projected + anchor)


class LightweightPatchTST(nn.Module):
    def __init__(
        self,
        *,
        patch_length: int,
        stride: int,
        d_model: int,
        heads: int,
        layers: int,
        feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("PatchTST d_model must be divisible by heads")
        self.patch_length = patch_length
        self.stride = stride
        self.patch_count = 1 + (INPUT_HOURS - patch_length) // stride
        self.patch_embedding = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(MODEL_INPUT_CHANNELS * d_model, FORECAST_HOURS * TARGET_COUNT),
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        patches = values.transpose(1, 2).unfold(
            dimension=2, size=self.patch_length, step=self.stride
        )
        batch, channels, patch_count, _ = patches.shape
        if patch_count != self.patch_count:
            raise ValueError(f"unexpected patch count: {patch_count}")
        encoded = self.patch_embedding(patches)
        encoded = encoded.reshape(batch * channels, patch_count, -1)
        encoded = self.encoder(encoded + self.position)
        pooled = self.norm(encoded).mean(dim=1)
        pooled = pooled.reshape(batch, channels * pooled.shape[-1])
        return self.head(pooled).reshape(batch, FORECAST_HOURS, TARGET_COUNT)


class GRUDDirect(nn.Module):
    """A GRU-D encoder with a direct 24-by-4 normalized-log forecast head.

    The supplied value channels are standardized by ``prepared_inputs`` using
    the applicable training-layer mean. Consequently, zero is the training
    mean in this model's value space. Missing values supplied by the generic
    request-level filler are deliberately ignored: GRU-D reconstructs them
    from the *post-corruption* observation mask, an elapsed-time input decay,
    and the last observed value for each indicator.
    """

    def __init__(self, *, hidden_size: int, dropout: float) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("GRU-D hidden_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("GRU-D dropout must be in [0, 1)")
        self.hidden_size = hidden_size
        # Standardized inputs are centered on the training-layer mean.
        self.register_buffer("train_mean", torch.zeros(TARGET_COUNT))
        self.input_decay = nn.Linear(TARGET_COUNT, TARGET_COUNT)
        self.hidden_decay = nn.Linear(TARGET_COUNT, hidden_size)
        self.gru_cell = nn.GRUCell(2 * TARGET_COUNT, hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, FORECAST_HOURS * TARGET_COUNT),
        )

    @staticmethod
    def elapsed_since_observation(masks: torch.Tensor) -> torch.Tensor:
        """Return per-indicator elapsed hours from the supplied observation mask.

        A present value has elapsed time zero. A first missing value at the
        left boundary has elapsed time one from the model's mean-value prior;
        successive missing hours increase it by one. The caller supplies the
        mask after any synthetic corruption, so augmented removals alter both
        elapsed times and GRU-D imputations.
        """
        if masks.ndim != 3 or masks.shape[1:] != (INPUT_HOURS, TARGET_COUNT):
            raise ValueError(
                "GRU-D masks must have shape (batch, 336, 4); "
                f"got {tuple(masks.shape)}"
            )
        observed = masks > 0.5
        elapsed = torch.empty_like(masks)
        running = torch.zeros(
            (masks.shape[0], TARGET_COUNT), dtype=masks.dtype, device=masks.device
        )
        for step in range(INPUT_HOURS):
            running = torch.where(
                observed[:, step, :], torch.zeros_like(running), running + 1.0
            )
            elapsed[:, step, :] = running
        return elapsed

    @staticmethod
    def _decay(projection: torch.Tensor) -> torch.Tensor:
        """GRU-D's non-increasing positive elapsed-time decay transform."""
        return torch.exp(-torch.relu(projection))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (
            INPUT_HOURS,
            MODEL_INPUT_CHANNELS,
        ):
            raise ValueError(
                "GRU-D inputs must have shape (batch, 336, 8); "
                f"got {tuple(inputs.shape)}"
            )
        values = inputs[:, :, :TARGET_COUNT]
        masks = inputs[:, :, TARGET_COUNT:]
        elapsed = self.elapsed_since_observation(masks)
        batch = inputs.shape[0]
        mean = self.train_mean.to(dtype=inputs.dtype, device=inputs.device).expand(
            batch, -1
        )
        previous = mean
        hidden = inputs.new_zeros((batch, self.hidden_size))

        for step in range(INPUT_HOURS):
            mask_t = masks[:, step, :]
            value_t = values[:, step, :]
            delta_t = elapsed[:, step, :]
            gamma_x = self._decay(self.input_decay(delta_t))
            decayed_previous = gamma_x * previous + (1.0 - gamma_x) * mean
            imputed = mask_t * value_t + (1.0 - mask_t) * decayed_previous
            gamma_h = self._decay(self.hidden_decay(delta_t))
            hidden = self.gru_cell(torch.cat((imputed, mask_t), dim=1), gamma_h * hidden)
            previous = torch.where(mask_t > 0.5, value_t, previous)

        return self.head(hidden).reshape(batch, FORECAST_HOURS, TARGET_COUNT)


def build_model(model_name: str, config: Mapping[str, object]) -> nn.Module:
    if model_name == "dlinear":
        return DLinear(kernel_size=int(config["kernel_size"]))
    if model_name == "nlinear":
        return NLinear()
    if model_name == "patchtst":
        return LightweightPatchTST(
            patch_length=int(config["patch_length"]),
            stride=int(config["stride"]),
            d_model=int(config["d_model"]),
            heads=int(config["heads"]),
            layers=int(config["layers"]),
            feedforward=int(config["feedforward"]),
            dropout=float(config["dropout"]),
        )
    if model_name == "grud_direct":
        return GRUDDirect(
            hidden_size=int(config["hidden_size"]),
            dropout=float(config["dropout"]),
        )
    raise ValueError(f"unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = torch.clamp(mask.sum(), min=1.0)
    return (torch.abs(prediction - target) * mask).sum() / denominator


def inverse_target(
    values: np.ndarray, normalization: Normalization
) -> np.ndarray:
    mean = np.asarray(normalization.target_mean, dtype=np.float32)
    std = np.asarray(normalization.target_std, dtype=np.float32)
    prediction = np.expm1(values * std[None, None, :] + mean[None, None, :])
    return np.maximum(prediction, 1e-4).astype(np.float32)


def official_filter(actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = actual.reshape(-1, TARGET_COUNT).astype(np.float64)
    complete = np.all(np.isfinite(flat), axis=1)
    if not np.any(complete):
        raise ValueError("no complete targets for official metric")
    quantiles = np.quantile(flat[complete], 0.05, axis=0, method="linear")
    selected = complete & np.all(flat >= quantiles[None, :], axis=1)
    if np.any(flat[selected] <= 0.0):
        raise ValueError("official filter retained non-positive targets")
    return selected, quantiles


def combined_scores(actual: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    flat_actual = actual.reshape(-1, TARGET_COUNT).astype(np.float64)
    flat_prediction = prediction.reshape(-1, TARGET_COUNT).astype(np.float64)
    mask, quantiles = official_filter(actual)
    error = np.mean(
        np.abs(flat_actual[mask] - flat_prediction[mask]) / flat_actual[mask], axis=1
    )
    rates = [float(np.mean(error < threshold)) for threshold in OFFICIAL_THRESHOLDS]
    return {
        "n_hours": int(np.sum(mask)),
        "mape_auc": float(np.mean(rates)),
        "mean_mape": float(np.mean(error)),
        "r_0_2": rates[0],
        "r_0_3": rates[1],
        "r_0_4": rates[2],
        "r_0_5": rates[3],
        "quantile_thresholds": quantiles.tolist(),
    }


def standard_metric_values(
    actual: np.ndarray,
    prediction: np.ndarray,
    mase_scales: np.ndarray,
    selected: np.ndarray,
    metric: int,
) -> dict[str, object]:
    flat_actual = actual.reshape(-1, TARGET_COUNT).astype(np.float64)
    flat_prediction = prediction.reshape(-1, TARGET_COUNT).astype(np.float64)
    repeated_scales = np.repeat(mase_scales, FORECAST_HOURS, axis=0).astype(np.float64)
    valid = selected & np.isfinite(flat_actual[:, metric]) & np.isfinite(
        flat_prediction[:, metric]
    )
    y = flat_actual[valid, metric]
    p = flat_prediction[valid, metric]
    if not len(y):
        raise ValueError("standard metric received no valid observations")
    absolute = np.abs(y - p)
    smape_denominator = np.maximum(np.abs(y) + np.abs(p), 1e-12)
    scale_valid = valid & np.isfinite(repeated_scales[:, metric]) & (
        repeated_scales[:, metric] > 1e-12
    )
    scaled = np.abs(
        flat_actual[scale_valid, metric] - flat_prediction[scale_valid, metric]
    ) / repeated_scales[scale_valid, metric]
    return {
        "n_hours": int(len(y)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(y - p)))),
        "wape": float(np.sum(absolute) / max(float(np.sum(np.abs(y))), 1e-12)),
        "smape": float(np.mean(2.0 * absolute / smape_denominator)),
        "mase": None if not len(scaled) else float(np.mean(scaled)),
        "mase_eligible_hours": int(len(scaled)),
    }


def complete_filter(actual: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(actual.reshape(-1, TARGET_COUNT)), axis=1)


def metric_rows(
    *,
    model_name: str,
    run_label: str,
    seed: int | None,
    actual: np.ndarray,
    prediction: np.ndarray,
    mase_scales: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    combined = combined_scores(actual, prediction)
    rows.append(
        {
            "model": model_name,
            "run": run_label,
            "seed": "" if seed is None else seed,
            "filter": "official_5pct_filtered",
            "indicator": "combined",
            **combined,
            "mae": "",
            "rmse": "",
            "wape": "",
            "smape": "",
            "mase": "",
            "mase_eligible_hours": "",
        }
    )
    official, _ = official_filter(actual)
    protocols = {
        "complete_targets_unfiltered": complete_filter(actual),
        "official_5pct_filtered": official,
    }
    for filter_name, selected in protocols.items():
        indicator_values: list[dict[str, object]] = []
        for metric, metric_name in enumerate(METRIC_NAMES):
            values = standard_metric_values(
                actual, prediction, mase_scales, selected, metric
            )
            indicator_values.append(values)
            rows.append(
                {
                    "model": model_name,
                    "run": run_label,
                    "seed": "" if seed is None else seed,
                    "filter": filter_name,
                    "indicator": metric_name,
                    "n_hours": values["n_hours"],
                    "mape_auc": "",
                    "mean_mape": "",
                    "r_0_2": "",
                    "r_0_3": "",
                    "r_0_4": "",
                    "r_0_5": "",
                    "quantile_thresholds": "",
                    **{key: values[key] for key in ("mae", "rmse", "wape", "smape", "mase", "mase_eligible_hours")},
                }
            )
        rows.append(
            {
                "model": model_name,
                "run": run_label,
                "seed": "" if seed is None else seed,
                "filter": filter_name,
                "indicator": "macro_mean",
                "n_hours": int(np.sum(selected)),
                "mape_auc": "",
                "mean_mape": "",
                "r_0_2": "",
                "r_0_3": "",
                "r_0_4": "",
                "r_0_5": "",
                "quantile_thresholds": "",
                "mae": float(np.mean([float(item["mae"]) for item in indicator_values])),
                "rmse": float(np.mean([float(item["rmse"]) for item in indicator_values])),
                "wape": float(np.mean([float(item["wape"]) for item in indicator_values])),
                "smape": float(np.mean([float(item["smape"]) for item in indicator_values])),
                "mase": float(
                    np.mean(
                        [
                            float(item["mase"])
                            for item in indicator_values
                            if item["mase"] is not None
                        ]
                    )
                ),
                "mase_eligible_hours": int(
                    min(int(item["mase_eligible_hours"]) for item in indicator_values)
                ),
            }
        )
    return rows


def make_loader(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def predict_normalized(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    loader = DataLoader(
        TensorDataset(inputs),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    with torch.no_grad():
        for (batch,) in loader:
            prediction = model(batch.to(device, non_blocking=True))
            outputs.append(prediction.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    for inputs, targets, masks in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = masked_l1(prediction, targets, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        weight = float(masks.sum().item())
        total_loss += float(loss.item()) * weight
        total_weight += weight
    return total_loss / max(total_weight, 1.0)


def copy_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def select_configuration(
    *,
    model_name: str,
    seed: int,
    fit_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    inner_inputs: torch.Tensor,
    inner_actual: np.ndarray,
    normalization: Normalization,
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
    smoke: bool,
) -> tuple[dict[str, object], int, list[dict[str, object]], float]:
    candidate_reports: list[dict[str, object]] = []
    best_candidate_index = -1
    best_candidate_key: tuple[float, float, int] | None = None
    best_epoch = 0
    selection_started = time.perf_counter()
    candidates = MODEL_CONFIGS[model_name][: 1 if smoke else TUNING_CANDIDATES]

    for candidate_index, config in enumerate(candidates):
        set_deterministic_seed(seed + candidate_index * 10_000)
        model = build_model(model_name, config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        loader = make_loader(
            fit_tensors,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + candidate_index,
        )
        epoch_reports: list[dict[str, object]] = []
        candidate_best_key: tuple[float, float, int] | None = None
        candidate_best_epoch = 0
        stale_epochs = 0
        for epoch in range(1, max_epochs + 1):
            epoch_started = time.perf_counter()
            train_loss = train_one_epoch(model, loader, optimizer, device)
            prediction_norm = predict_normalized(
                model, inner_inputs, batch_size=batch_size, device=device
            )
            prediction = inverse_target(prediction_norm, normalization)
            scores = combined_scores(inner_actual, prediction)
            key = (
                float(scores["mape_auc"]),
                -float(scores["mean_mape"]),
                -epoch,
            )
            epoch_reports.append(
                {
                    "epoch": epoch,
                    "train_masked_log_l1": train_loss,
                    "inner_mape_auc": scores["mape_auc"],
                    "inner_mean_mape": scores["mean_mape"],
                    "seconds": time.perf_counter() - epoch_started,
                }
            )
            if candidate_best_key is None or key > candidate_best_key:
                candidate_best_key = key
                candidate_best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        if candidate_best_key is None:
            raise RuntimeError(f"{model_name}/{config['name']} produced no checkpoint")
        candidate_report = {
            "candidate_index": candidate_index,
            "config": dict(config),
            "parameter_count": count_parameters(model),
            "best_epoch": candidate_best_epoch,
            "best_inner_mape_auc": candidate_best_key[0],
            "best_inner_mean_mape": -candidate_best_key[1],
            "epochs_run": len(epoch_reports),
            "epoch_reports": epoch_reports,
        }
        candidate_reports.append(candidate_report)
        overall_key = (
            candidate_best_key[0],
            candidate_best_key[1],
            -candidate_index,
        )
        if best_candidate_key is None or overall_key > best_candidate_key:
            best_candidate_key = overall_key
            best_candidate_index = candidate_index
            best_epoch = candidate_best_epoch
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_candidate_index < 0:
        raise RuntimeError(f"no configuration selected for {model_name}")
    return (
        dict(candidates[best_candidate_index]),
        best_epoch,
        candidate_reports,
        time.perf_counter() - selection_started,
    )


def train_final_model(
    *,
    model_name: str,
    config: Mapping[str, object],
    seed: int,
    epochs: int,
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    holdout_inputs: torch.Tensor,
    normalization: Normalization,
    device: torch.device,
    batch_size: int,
) -> tuple[nn.Module, np.ndarray, dict[str, object]]:
    set_deterministic_seed(seed + 1_000_000)
    model = build_model(model_name, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loader = make_loader(
        train_tensors,
        batch_size=batch_size,
        shuffle=True,
        seed=seed + 1_000_000,
    )
    epoch_losses: list[float] = []
    training_started = time.perf_counter()
    for _ in range(epochs):
        epoch_losses.append(train_one_epoch(model, loader, optimizer, device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started

    model.eval()
    with torch.no_grad():
        warmup = holdout_inputs[: min(batch_size, len(holdout_inputs))].to(
            device, non_blocking=True
        )
        _ = model(warmup)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_started = time.perf_counter()
    prediction_norm = predict_normalized(
        model, holdout_inputs, batch_size=batch_size, device=device
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    prediction = inverse_target(prediction_norm, normalization)
    report = {
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_window": 1000.0 * inference_seconds / len(holdout_inputs),
        "epoch_losses": epoch_losses,
        "parameter_count": count_parameters(model),
        "max_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
    }
    return model, prediction, report


def cap_indices(indices: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or len(indices) <= limit:
        return indices
    return indices[:limit]


def run_worker(
    *,
    dataset_cache: Path,
    output: Path,
    model_name: str,
    seed: int,
    physical_device: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    augmentation: str,
    augmentation_rate: float,
    smoke: bool,
) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "worker requires exactly one CUDA device through CUDA_VISIBLE_DEVICES"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    dataset = load_dataset_cache(dataset_cache)
    fit = dataset.indices_for_dates(FIT_DATES)
    inner = dataset.indices_for_dates(INNER_DATES)
    holdout = dataset.indices_for_dates(HOLDOUT_DATES)
    if smoke:
        fit = cap_indices(fit, 256)
        inner = cap_indices(inner, 128)
        holdout = cap_indices(holdout, 128)

    fit_normalization = compute_normalization(dataset, fit)
    fit_rate = 0.0 if augmentation == "clean" else augmentation_rate
    fit_extra, fit_augmentation = training_augmentation(
        dataset,
        fit,
        augmentation=augmentation,
        requested_rate=fit_rate,
        seed=seed,
    )
    fit_tensors = prepared_tensors(
        dataset,
        fit,
        fit_normalization,
        additional_missing=fit_extra,
    )
    inner_tensors = prepared_tensors(dataset, inner, fit_normalization)
    inner_actual = np.asarray(dataset.targets[inner], dtype=np.float32)
    selected_config, selected_epoch, candidate_reports, selection_seconds = (
        select_configuration(
            model_name=model_name,
            seed=seed,
            fit_tensors=fit_tensors,
            inner_inputs=inner_tensors[0],
            inner_actual=inner_actual,
            normalization=fit_normalization,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            smoke=smoke,
        )
    )

    final_train = np.concatenate((fit, inner))
    final_normalization = compute_normalization(dataset, final_train)
    final_extra, final_augmentation = training_augmentation(
        dataset,
        final_train,
        augmentation=augmentation,
        requested_rate=fit_rate,
        seed=seed + 100_000,
    )
    final_tensors = prepared_tensors(
        dataset,
        final_train,
        final_normalization,
        additional_missing=final_extra,
    )
    holdout_inputs = prepared_inputs(dataset, holdout, final_normalization)
    model, predictions, final_report = train_final_model(
        model_name=model_name,
        config=selected_config,
        seed=seed,
        epochs=selected_epoch,
        train_tensors=final_tensors,
        holdout_inputs=holdout_inputs,
        normalization=final_normalization,
        device=device,
        batch_size=batch_size,
    )
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    scores = combined_scores(actual, predictions)

    model_path = output / "models" / f"{model_name}_seed{seed}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_version": EXPERIMENT_VERSION,
            "model": model_name,
            "seed": seed,
            "config": selected_config,
            "selected_epoch": selected_epoch,
            "normalization": asdict(final_normalization),
            "augmentation": augmentation,
            "augmentation_rate": fit_rate,
            "fit_augmentation": fit_augmentation,
            "final_train_augmentation": final_augmentation,
            "state_dict": copy_state_dict(model),
        },
        temporary_model,
    )
    temporary_model.replace(model_path)
    prediction_path = output / "worker_predictions" / f"{model_name}_seed{seed}.npy"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path, predictions, allow_pickle=False)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "model": model_name,
        "seed": seed,
        "physical_gpu": physical_device,
        "visible_gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "fit_windows": len(fit),
        "inner_windows": len(inner),
        "final_train_windows": len(final_train),
        "holdout_windows": len(holdout),
        "augmentation": augmentation,
        "augmentation_rate": fit_rate,
        "fit_augmentation": fit_augmentation,
        "final_train_augmentation": final_augmentation,
        "selection_and_holdout_views_are_clean": True,
        "labels_or_target_masks_modified": False,
        "selection_rule": (
            "highest inner MAPEAUC; lower inner mean MAPE and candidate order "
            "break ties; best epoch selected within the same inner layer"
        ),
        "candidate_budget": len(candidate_reports),
        "max_epochs_per_candidate": max_epochs,
        "patience": patience,
        "selected_config": selected_config,
        "selected_epoch": selected_epoch,
        "candidate_reports": candidate_reports,
        "selection_seconds": selection_seconds,
        "final_training": final_report,
        "holdout_scores": scores,
        "fit_normalization": asdict(fit_normalization),
        "final_normalization": asdict(final_normalization),
        "model_file": str(model_path.relative_to(output)),
        "model_size_bytes": model_path.stat().st_size,
        "model_sha256": sha256_file(model_path),
        "prediction_file": str(prediction_path.relative_to(output)),
        "prediction_sha256": sha256_file(prediction_path),
        "smoke": smoke,
    }
    report_path = output / "job_reports" / f"{model_name}_seed{seed}.json"
    atomic_write_json(report_path, report)
    report["report_file"] = str(report_path.relative_to(output))
    return report


def query_gpu_snapshot() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,pstate",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": int(parts[2]),
                "memory_used_mb": int(parts[3]),
                "utilization_percent": int(parts[4]),
                "pstate": parts[5],
            }
        )
    return rows


def ensure_gpus_available(
    requested: Sequence[int], maximum_existing_memory_mb: int
) -> list[dict[str, object]]:
    snapshot = query_gpu_snapshot()
    by_index = {int(item["index"]): item for item in snapshot}
    for device in requested:
        if device not in by_index:
            raise ValueError(f"requested GPU {device} is not visible")
        used = int(by_index[device]["memory_used_mb"])
        if used > maximum_existing_memory_mb:
            raise RuntimeError(
                f"GPU {device} already uses {used} MB, above the "
                f"{maximum_existing_memory_mb} MB safety threshold"
            )
    return snapshot


def launch_worker_job(
    *,
    script: Path,
    dataset_cache: Path,
    output: Path,
    model_name: str,
    seed: int,
    device: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    augmentation: str,
    augmentation_rate: float,
    smoke: bool,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--dataset-cache",
        str(dataset_cache),
        "--output",
        str(output),
        "--model",
        model_name,
        "--seed",
        str(seed),
        "--physical-device",
        str(device),
        "--max-epochs",
        str(max_epochs),
        "--patience",
        str(patience),
        "--batch-size",
        str(batch_size),
        "--augmentation",
        augmentation,
        "--augmentation-rate",
        str(augmentation_rate),
    ]
    if smoke:
        command.append("--smoke")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["OMP_NUM_THREADS"] = "4"
    environment["MKL_NUM_THREADS"] = "4"
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=project_root(),
        env=environment,
        capture_output=True,
        text=True,
    )
    log_path = output / "logs" / f"{model_name}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND\n"
        + " ".join(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "model": model_name,
            "seed": seed,
            "physical_gpu": device,
            "returncode": completed.returncode,
            "seconds": time.perf_counter() - started,
            "log_file": str(log_path.relative_to(output)),
        }
    report_path = output / "job_reports" / f"{model_name}_seed{seed}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "status": "complete",
        "model": model_name,
        "seed": seed,
        "physical_gpu": device,
        "seconds": time.perf_counter() - started,
        "log_file": str(log_path.relative_to(output)),
        "report": report,
    }


def run_device_queue(
    *,
    device: int,
    jobs: Sequence[tuple[str, int]],
    script: Path,
    dataset_cache: Path,
    output: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    augmentation: str,
    augmentation_rate: float,
    smoke: bool,
) -> list[dict[str, object]]:
    return [
        launch_worker_job(
            script=script,
            dataset_cache=dataset_cache,
            output=output,
            model_name=model_name,
            seed=seed,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            augmentation=augmentation,
            augmentation_rate=augmentation_rate,
            smoke=smoke,
        )
        for model_name, seed in jobs
    ]


def timestamp_from_hour(hour: int) -> datetime:
    return datetime(1970, 1, 1) + timedelta(hours=int(hour))


def daily_metric_rows(
    *,
    model_name: str,
    run_label: str,
    seed: int | None,
    target_start_hours: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dates = np.asarray(
        [timestamp_from_hour(int(value)).date().isoformat() for value in target_start_hours]
    )
    for target_date in sorted(set(dates.tolist())):
        selected = dates == target_date
        scores = combined_scores(actual[selected], prediction[selected])
        rows.append(
            {
                "model": model_name,
                "run": run_label,
                "seed": "" if seed is None else seed,
                "target_date": target_date,
                **scores,
            }
        )
    return rows


def write_prediction_rows(
    *,
    path: Path,
    cells: np.ndarray,
    target_start_hours: np.ndarray,
    actual: np.ndarray,
    target_masks: np.ndarray,
    seed_predictions: Mapping[int, np.ndarray],
    ensemble: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    seeds = sorted(seed_predictions)
    fieldnames = ["cell", "target_timestamp", "horizon"]
    fieldnames.extend(f"actual_{name}" for name in METRIC_NAMES)
    fieldnames.extend(f"observed_{name}" for name in METRIC_NAMES)
    for seed in seeds:
        fieldnames.extend(f"prediction_seed{seed}_{name}" for name in METRIC_NAMES)
    fieldnames.extend(f"prediction_ensemble_{name}" for name in METRIC_NAMES)
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for window_index in range(len(cells)):
            target_start = timestamp_from_hour(int(target_start_hours[window_index]))
            for horizon in range(FORECAST_HOURS):
                row: dict[str, object] = {
                    "cell": str(cells[window_index]),
                    "target_timestamp": (
                        target_start + timedelta(hours=horizon)
                    ).isoformat(sep=" "),
                    "horizon": horizon + 1,
                }
                for metric, metric_name in enumerate(METRIC_NAMES):
                    value = float(actual[window_index, horizon, metric])
                    row[f"actual_{metric_name}"] = "" if not math.isfinite(value) else value
                    row[f"observed_{metric_name}"] = int(
                        target_masks[window_index, horizon, metric]
                    )
                for seed in seeds:
                    for metric, metric_name in enumerate(METRIC_NAMES):
                        row[f"prediction_seed{seed}_{metric_name}"] = float(
                            seed_predictions[seed][window_index, horizon, metric]
                        )
                for metric, metric_name in enumerate(METRIC_NAMES):
                    row[f"prediction_ensemble_{metric_name}"] = float(
                        ensemble[window_index, horizon, metric]
                    )
                writer.writerow(row)
    temporary.replace(path)


def numeric_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate_results(
    *,
    output: Path,
    dataset: CachedDataset,
    job_results: Sequence[Mapping[str, object]],
    requested_models: Sequence[str],
    requested_seeds: Sequence[int],
    smoke: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    holdout = dataset.indices_for_dates(HOLDOUT_DATES)
    if smoke:
        holdout = cap_indices(holdout, 128)
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    target_masks = np.asarray(dataset.target_masks[holdout], dtype=np.uint8)
    mase_scales = np.asarray(dataset.mase_scales[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    target_start_hours = np.asarray(dataset.target_start_hours[holdout], dtype=np.int64)

    successful = [item for item in job_results if item["status"] == "complete"]
    failures = [dict(item) for item in job_results if item["status"] != "complete"]
    metric_output: list[dict[str, object]] = []
    daily_output: list[dict[str, object]] = []
    model_summaries: dict[str, object] = {}

    for model_name in requested_models:
        model_jobs = sorted(
            [item for item in successful if item["model"] == model_name],
            key=lambda item: int(item["seed"]),
        )
        if not model_jobs:
            model_summaries[model_name] = {
                "status": "failed",
                "successful_seeds": [],
            }
            continue
        seed_predictions: dict[int, np.ndarray] = {}
        reports: list[Mapping[str, object]] = []
        for item in model_jobs:
            seed = int(item["seed"])
            report = item["report"]
            prediction_path = output / str(report["prediction_file"])
            prediction = np.load(prediction_path, allow_pickle=False)
            if prediction.shape != actual.shape:
                raise ValueError(
                    f"{model_name}/seed{seed} prediction shape {prediction.shape} "
                    f"does not match {actual.shape}"
                )
            seed_predictions[seed] = prediction
            reports.append(report)
            metric_output.extend(
                metric_rows(
                    model_name=model_name,
                    run_label=f"seed_{seed}",
                    seed=seed,
                    actual=actual,
                    prediction=prediction,
                    mase_scales=mase_scales,
                )
            )
            daily_output.extend(
                daily_metric_rows(
                    model_name=model_name,
                    run_label=f"seed_{seed}",
                    seed=seed,
                    target_start_hours=target_start_hours,
                    actual=actual,
                    prediction=prediction,
                )
            )
        ensemble = np.mean(np.stack(list(seed_predictions.values()), axis=0), axis=0)
        metric_output.extend(
            metric_rows(
                model_name=model_name,
                run_label="seed_ensemble_mean",
                seed=None,
                actual=actual,
                prediction=ensemble,
                mase_scales=mase_scales,
            )
        )
        daily_output.extend(
            daily_metric_rows(
                model_name=model_name,
                run_label="seed_ensemble_mean",
                seed=None,
                target_start_hours=target_start_hours,
                actual=actual,
                prediction=ensemble,
            )
        )
        prediction_path = output / "predictions" / f"{model_name}_holdout_predictions.csv.gz"
        write_prediction_rows(
            path=prediction_path,
            cells=cells,
            target_start_hours=target_start_hours,
            actual=actual,
            target_masks=target_masks,
            seed_predictions=seed_predictions,
            ensemble=ensemble,
        )
        per_seed_scores = [
            combined_scores(actual, seed_predictions[seed])
            for seed in sorted(seed_predictions)
        ]
        ensemble_scores = combined_scores(actual, ensemble)
        model_summaries[model_name] = {
            "status": (
                "complete"
                if len(seed_predictions) == len(requested_seeds)
                else "partial"
            ),
            "successful_seeds": sorted(seed_predictions),
            "per_seed": {
                str(seed): score
                for seed, score in zip(sorted(seed_predictions), per_seed_scores)
            },
            "seed_aggregate": {
                "mape_auc": numeric_summary(
                    [float(score["mape_auc"]) for score in per_seed_scores]
                ),
                "mean_mape": numeric_summary(
                    [float(score["mean_mape"]) for score in per_seed_scores]
                ),
            },
            "ensemble": ensemble_scores,
            "parameter_count": numeric_summary(
                [float(report["final_training"]["parameter_count"]) for report in reports]
            ),
            "selection_seconds": numeric_summary(
                [float(report["selection_seconds"]) for report in reports]
            ),
            "final_training_seconds": numeric_summary(
                [
                    float(report["final_training"]["training_seconds"])
                    for report in reports
                ]
            ),
            "inference_seconds": numeric_summary(
                [
                    float(report["final_training"]["inference_seconds"])
                    for report in reports
                ]
            ),
            "max_cuda_memory_bytes": numeric_summary(
                [
                    float(report["final_training"]["max_cuda_memory_bytes"])
                    for report in reports
                ]
            ),
            "prediction_file": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
        }

    atomic_write_csv(output / "metrics.csv", metric_output)
    atomic_write_csv(output / "daily_metrics.csv", daily_output)
    atomic_write_json(output / "failures.json", failures)
    return (
        {
            "models": model_summaries,
            "failure_count": len(failures),
            "holdout_windows": len(holdout),
            "holdout_rows": len(holdout) * FORECAST_HOURS,
            "aggregation_policy": (
                "Report every independent seed, arithmetic mean and sample "
                "standard deviation across seeds, plus an arithmetic prediction "
                "ensemble. No best-seed selection is used."
            ),
        },
        failures,
    )


def output_manifest(output: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "output_manifest.json":
            continue
        records.append(
            {
                "path": str(path.relative_to(output)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def parse_integer_list(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("at least one integer is required")
    if len(values) != len(set(values)):
        raise ValueError("integer lists must not contain duplicates")
    return values


def parse_model_list(text: str) -> list[str]:
    values = [value.strip().lower() for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("at least one model is required")
    unknown = [value for value in values if value not in MODEL_CONFIGS]
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    if len(values) != len(set(values)):
        raise ValueError("model list must not contain duplicates")
    return values


def run_master(args: argparse.Namespace) -> int:
    output = resolve_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_path = resolve_train_path()
    input_hash_before = sha256_file(train_path)
    requested_models = parse_model_list(args.models)
    requested_seeds = parse_integer_list(args.seeds)
    requested_devices = parse_integer_list(args.gpu_devices)
    if args.augmentation not in {"clean", "mixed"}:
        raise ValueError("augmentation must be clean or mixed")
    if not 0.0 <= args.augmentation_rate <= 1.0:
        raise ValueError("augmentation-rate must be in [0,1]")
    if args.augmentation == "clean" and args.augmentation_rate != 0.0:
        raise ValueError("clean augmentation requires --augmentation-rate 0")
    if not args.smoke and len(requested_seeds) < 3:
        raise ValueError("full evaluation requires at least three independent seeds")
    max_epochs = 2 if args.smoke else args.max_epochs
    patience = 1 if args.smoke else args.patience
    start_snapshot = ensure_gpus_available(
        requested_devices, args.max_existing_gpu_memory_mb
    )
    run_started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="paper-neural-baselines-") as temporary:
        cache_root = Path(temporary)
        series = read_training_series(train_path)
        arrays, dataset_report = build_window_arrays(series)
        if dataset_report["candidate_windows"] != 11_686:
            raise ValueError(
                f"expected 11,686 candidate windows, got "
                f"{dataset_report['candidate_windows']}"
            )
        if dataset_report["continuous_windows"] != 11_685:
            raise ValueError(
                f"expected 11,685 continuous windows, got "
                f"{dataset_report['continuous_windows']}"
            )
        write_dataset_cache(cache_root, arrays)
        dataset = load_dataset_cache(cache_root)
        leakage = leakage_checks(dataset)
        atomic_write_json(output / "dataset_report.json", dataset_report)
        atomic_write_json(output / "leakage_checks.json", leakage)
        protocol = {
            "schema_version": SCHEMA_VERSION,
            "experiment_version": EXPERIMENT_VERSION,
            "train_file": str(REGISTERED_TRAIN),
            "train_sha256": input_hash_before,
            "history_hours": INPUT_HOURS,
            "forecast_hours": FORECAST_HOURS,
            "targets": list(METRIC_NAMES),
            "input_features": (
                "per-window median-filled log1p traffic values and four "
                "observation masks; no cell ID, parameter, weather, or "
                "cross-window traffic"
            ),
            "target_transform": "log1p, normalized using the applicable training layer",
            "loss": "masked L1 in normalized log-target space",
            "training_augmentation": {
                "type": args.augmentation,
                "requested_rate": args.augmentation_rate,
                "scope": "absolute cell-time tensor before overlapping windows",
                "input_history_only": True,
                "labels_and_target_masks_retained": True,
                "normalization_from_original_observed_training_values": True,
                "inner_and_holdout_views": "clean",
            },
            "fit_dates": [str(value) for value in FIT_DATES],
            "inner_dates": [str(value) for value in INNER_DATES],
            "final_train_dates": [
                str(value) for value in (*FIT_DATES, *INNER_DATES)
            ],
            "holdout_dates": [str(value) for value in HOLDOUT_DATES],
            "selection": (
                "Each predeclared candidate is trained only on fit dates. "
                "Configuration and epoch are selected only on inner dates. "
                "The selected configuration is reinitialized and retrained on "
                "all fit+inner dates for the selected epoch count. Holdout is "
                "used only once for final evaluation."
            ),
            "candidate_count_per_model": 1 if args.smoke else TUNING_CANDIDATES,
            "candidate_configurations": {
                model: [
                    dict(config)
                    for config in MODEL_CONFIGS[model][
                        : 1 if args.smoke else TUNING_CANDIDATES
                    ]
                ]
                for model in requested_models
            },
            "max_epochs_per_candidate": max_epochs,
            "patience": patience,
            "batch_size": args.batch_size,
            "seeds": requested_seeds,
            "seed_reporting": "all seeds + mean/sample SD + prediction ensemble",
            "determinism": {
                "torch_deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8",
                "flash_sdp": False,
                "memory_efficient_sdp": False,
                "math_sdp": True,
            },
            "smoke": args.smoke,
        }
        atomic_write_json(output / "protocol.json", protocol)

        jobs = [
            (model_name, seed)
            for model_name in requested_models
            for seed in requested_seeds
        ]
        device_queues: dict[int, list[tuple[str, int]]] = {
            device: [] for device in requested_devices
        }
        for index, job in enumerate(jobs):
            device_queues[requested_devices[index % len(requested_devices)]].append(job)
        gpu_samples: list[dict[str, object]] = [
            {"timestamp": datetime.now().isoformat(), "phase": "before", "gpus": start_snapshot}
        ]
        with ThreadPoolExecutor(max_workers=len(requested_devices)) as executor:
            futures = {
                executor.submit(
                    run_device_queue,
                    device=device,
                    jobs=queue,
                    script=Path(__file__).resolve(),
                    dataset_cache=cache_root,
                    output=output,
                    max_epochs=max_epochs,
                    patience=patience,
                    batch_size=args.batch_size,
                    augmentation=args.augmentation,
                    augmentation_rate=args.augmentation_rate,
                    smoke=args.smoke,
                ): device
                for device, queue in device_queues.items()
                if queue
            }
            pending = set(futures)
            queue_results: list[dict[str, object]] = []
            while pending:
                completed, pending = wait(
                    pending, timeout=5.0, return_when=FIRST_COMPLETED
                )
                gpu_samples.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "phase": "training" if pending else "after_jobs",
                        "gpus": query_gpu_snapshot(),
                    }
                )
                for future in completed:
                    queue_results.extend(future.result())

        summary, failures = aggregate_results(
            output=output,
            dataset=dataset,
            job_results=queue_results,
            requested_models=requested_models,
            requested_seeds=requested_seeds,
            smoke=args.smoke,
        )

    input_hash_after = sha256_file(train_path)
    if input_hash_after != input_hash_before:
        raise RuntimeError("registered training file changed during the run")
    gpu_samples.append(
        {
            "timestamp": datetime.now().isoformat(),
            "phase": "complete",
            "gpus": query_gpu_snapshot(),
        }
    )
    atomic_write_json(
        output / "gpu_evidence.json",
        {
            "requested_devices": requested_devices,
            "safety_threshold_existing_memory_mb": args.max_existing_gpu_memory_mb,
            "samples": gpu_samples,
        },
    )
    complete_summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "status": (
            "complete"
            if not failures
            else "partial_with_recorded_failures"
        ),
        "runtime_seconds": time.perf_counter() - run_started,
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__,
        },
        **summary,
    }
    atomic_write_json(output / "summary.json", complete_summary)
    atomic_write_json(output / "output_manifest.json", output_manifest(output))
    required_models = {"dlinear", "nlinear"}.intersection(requested_models)
    failed_required = [
        model
        for model in required_models
        if complete_summary["models"].get(model, {}).get("status") != "complete"
        and not args.smoke
    ]
    return 1 if failed_required else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train leakage-controlled DLinear/NLinear/PatchTST/GRU-D baselines on "
            "the registered 336-to-24 paper protocol."
        )
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_ROOT / "results"),
        help="Output directory under artifacts/reproduction/neural_baselines/.",
    )
    parser.add_argument("--models", default="dlinear,patchtst")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--augmentation",
        choices=("clean", "mixed"),
        default="clean",
        help="Input-history training augmentation; inner and holdout stay clean.",
    )
    parser.add_argument(
        "--augmentation-rate",
        type=float,
        default=0.0,
        help="Requested unique-axis rate; use 0.15 with --augmentation mixed.",
    )
    parser.add_argument("--max-existing-gpu-memory-mb", type=int, default=1024)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one seed, one candidate, two epochs, and capped split sizes.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-cache", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=tuple(MODEL_CONFIGS), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--physical-device", type=int, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.smoke and not args.worker:
        first_seed = parse_integer_list(args.seeds)[0]
        args.seeds = str(first_seed)
    if args.worker:
        required = {
            "dataset_cache": args.dataset_cache,
            "model": args.model,
            "seed": args.seed,
            "physical_device": args.physical_device,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"worker mode missing arguments: {missing}")
        try:
            report = run_worker(
                dataset_cache=Path(args.dataset_cache).resolve(strict=True),
                output=resolve_output(args.output),
                model_name=str(args.model),
                seed=int(args.seed),
                physical_device=int(args.physical_device),
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                augmentation=args.augmentation,
                augmentation_rate=args.augmentation_rate,
                smoke=args.smoke,
            )
            print(json.dumps({"status": "complete", "report": report["report_file"]}))
            return 0
        except Exception:
            traceback.print_exc()
            return 1
    return run_master(args)


if __name__ == "__main__":
    raise SystemExit(main())
