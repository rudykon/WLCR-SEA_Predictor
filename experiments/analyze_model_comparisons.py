from __future__ import annotations

"""Unified Revision-6 comparisons, missingness stress, and audit plot tables.

All evaluation uses the registered training trace. Finals test traffic is never
opened. Because the August holdout informed earlier revisions, every result is
explicitly exploratory rather than a prospectively frozen confirmation.
"""

import argparse
import csv
import json
import math
import os
import statistics
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments import train_neural_baselines as neural
from experiments import train_wlcr_sea as runner
from experiments import wlcr_sea_model as sea

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path("artifacts/revision6/comparative_analysis")
SEEDS = (42, 43, 44, 45, 46)
METHOD_ORDER = (
    "last_day",
    "last_week",
    "same_hour_median_7d",
    "A0_fixed",
    "standard_stat_traffic_only_165d",
    "DLinear",
    "PatchTST",
    "A4_reliability",
    "A6_mixed_aug",
    "A7_cross_indicator",
)
LABELS = {
    "last_day": "Last-day seasonal",
    "last_week": "Last-week seasonal",
    "same_hour_median_7d": "Same-hour median (7 d)",
    "A0_fixed": "Fixed-weight WLCR",
    "standard_stat_traffic_only_165d": "Standard-stat LightGBM",
    "DLinear": "DLinear",
    "PatchTST": "PatchTST",
    "A4_reliability": "WLCR-SEA-Convex",
    "A6_mixed_aug": "WLCR-SEA-Residual",
    "A7_cross_indicator": "WLCR-SEA + cross-indicator",
}
INFO_CLASSES = {
    "last_day": "target-cell traffic",
    "last_week": "target-cell traffic",
    "same_hour_median_7d": "target-cell traffic + masks",
    "A0_fixed": "target-cell traffic + masks + horizon",
    "standard_stat_traffic_only_165d": "target-cell traffic + masks + horizon",
    "DLinear": "target-cell traffic + masks",
    "PatchTST": "target-cell traffic + masks",
    "A4_reliability": "target-cell traffic + masks + horizon",
    "A6_mixed_aug": "target-cell traffic + masks + horizon",
    "A7_cross_indicator": "target-cell traffic + masks + horizon + cross-indicator context",
}


def resolve(root: Path, value: str, *, strict: bool) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=strict)


def dataset_from_registered_train() -> tuple[neural.CachedDataset, dict[str, object], Path]:
    train_path = neural.resolve_train_path()
    arrays, report = neural.build_window_arrays(neural.read_training_series(train_path))
    dataset = neural.CachedDataset(root=Path("<memory>"), **arrays)
    return dataset, report, train_path


def load_prediction(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.shape != shape:
        raise ValueError(f"prediction shape mismatch at {path}: {values.shape} != {shape}")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"non-finite or non-positive predictions at {path}")
    return np.asarray(values, dtype=np.float32)


def ensemble_files(paths: Sequence[Path], shape: tuple[int, ...]) -> np.ndarray:
    arrays = [load_prediction(path, shape) for path in paths]
    return np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float64).astype(np.float32)


def standard_reorder(path: Path, dataset: neural.CachedDataset, holdout: np.ndarray) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(holdout):
        raise ValueError("Standard-stat holdout order length mismatch")
    epoch = datetime(1970, 1, 1)
    source: dict[tuple[str, int], int] = {}
    for offset, row in enumerate(rows):
        hour = int(
            (datetime.fromisoformat(row["target_start"]) - epoch).total_seconds() // 3600
        )
        key = (str(row["cell"]), hour)
        if key in source:
            raise ValueError(f"duplicate Standard-stat holdout key: {key}")
        source[key] = offset
    expected = [
        (str(dataset.cells[index]), int(dataset.target_start_hours[index]))
        for index in holdout.tolist()
    ]
    if set(source) != set(expected):
        missing = sorted(set(expected) - set(source))[:5]
        extra = sorted(set(source) - set(expected))[:5]
        raise ValueError(f"Standard-stat holdout sample mismatch; missing={missing}, extra={extra}")
    return np.asarray([source[key] for key in expected], dtype=np.int64)


def seed_metric_stats(
    paths: Sequence[Path],
    actual: np.ndarray,
    scales: np.ndarray,
    cells: np.ndarray,
) -> dict[str, float]:
    values = [
        float(sea.forecast_metrics(actual, np.load(path, allow_pickle=False), scales, cells)["macro_indicator"]["wape"])
        for path in paths
    ]
    return {
        "seed_macro_wape_mean": float(statistics.mean(values)),
        "seed_macro_wape_sd": float(statistics.stdev(values)),
        "seed_macro_wape_min": float(min(values)),
        "seed_macro_wape_max": float(max(values)),
    }


def metric_row(
    method: str,
    prediction: np.ndarray,
    actual: np.ndarray,
    scales: np.ndarray,
    cells: np.ndarray,
    thresholds: np.ndarray,
    seed_stats: Mapping[str, float] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    metrics = sea.forecast_metrics(actual, prediction, scales, cells)
    ths = sea.threshold_hit_score(actual, prediction, thresholds)
    row = {
        "method": method,
        "label": LABELS[method],
        "information_class": INFO_CLASSES[method],
        "role": (
            "predeclared_primary"
            if method == runner.PRIMARY_VARIANT
            else "exploratory_ablation"
            if method == "A7_cross_indicator"
            else "baseline_or_ablation"
        ),
        "macro_wape": metrics["macro_indicator"]["wape"],
        "pooled_wape": metrics["pooled_wape"],
        "macro_cell_wape": metrics["macro_cell_wape"],
        "median_cell_wape": metrics["median_cell_wape"],
        "mase": metrics["macro_indicator"]["mase"],
        "smape": metrics["macro_indicator"]["smape"],
        "mae": metrics["macro_indicator"]["mae"],
        "rmse": metrics["macro_indicator"]["rmse"],
        "threshold_hit_score": ths["score"],
        "frozen_thresholds_from_training": True,
    }
    if seed_stats:
        row.update(seed_stats)
    else:
        row.update(
            {
                "seed_macro_wape_mean": "",
                "seed_macro_wape_sd": "",
                "seed_macro_wape_min": "",
                "seed_macro_wape_max": "",
            }
        )
    return row, metrics


def per_cell_wape(actual: np.ndarray, prediction: np.ndarray, cells: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    cells = np.asarray(cells).astype(str)
    for cell in sorted(set(cells.tolist())):
        selected = cells == cell
        valid = np.isfinite(actual[selected]) & np.isfinite(prediction[selected])
        denominator = float(np.sum(np.abs(actual[selected][valid])))
        if denominator > 0.0:
            result[cell] = float(
                np.sum(np.abs(actual[selected][valid] - prediction[selected][valid]))
                / denominator
            )
    return result


def cell_win_row(
    proposed: np.ndarray,
    baseline: np.ndarray,
    actual: np.ndarray,
    cells: np.ndarray,
    baseline_name: str,
) -> dict[str, object]:
    a = per_cell_wape(actual, proposed, cells)
    b = per_cell_wape(actual, baseline, cells)
    common = sorted(set(a).intersection(b))
    differences = np.asarray([a[cell] - b[cell] for cell in common], dtype=np.float64)
    return {
        "proposed": runner.PRIMARY_VARIANT,
        "baseline": baseline_name,
        "cells": len(common),
        "proposed_better_cells": int(np.sum(differences < 0.0)),
        "tie_cells": int(np.sum(np.isclose(differences, 0.0, atol=1e-12, rtol=0.0))),
        "proposed_better_fraction": float(np.mean(differences < 0.0)),
        "median_cell_wape_delta": float(np.median(differences)),
    }


def corrupted_neural_inputs(
    dataset: neural.CachedDataset,
    indices: np.ndarray,
    normalization: neural.Normalization,
    additional_missing: np.ndarray,
) -> torch.Tensor:
    base_values = np.asarray(dataset.x_values[indices], dtype=np.float32)
    original = np.asarray(dataset.x_masks[indices], dtype=bool)
    extra = np.asarray(additional_missing, dtype=bool)
    if extra.shape != original.shape:
        raise ValueError("additional missing mask is misaligned")
    masks = original & ~extra
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
    mean = np.asarray(normalization.input_mean, dtype=np.float32)
    std = np.asarray(normalization.input_std, dtype=np.float32)
    normalized = (values - mean[None, None, :]) / std[None, None, :]
    return torch.from_numpy(
        np.concatenate((normalized, masks.astype(np.float32)), axis=2).astype(np.float32)
    )


def neural_models(
    baseline_root: Path,
    model_name: str,
    device: torch.device,
) -> tuple[list[torch.nn.Module], neural.Normalization]:
    models: list[torch.nn.Module] = []
    normalizations: list[dict[str, object]] = []
    for seed in SEEDS:
        path = baseline_root / "models" / f"{model_name}_seed{seed}.pt"
        payload = torch.load(path, map_location="cpu")
        if payload.get("model") != model_name or int(payload.get("seed")) != seed:
            raise ValueError(f"invalid checkpoint metadata: {path}")
        model = neural.build_model(model_name, payload["config"])
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        models.append(model)
        normalizations.append(payload["normalization"])
    if any(value != normalizations[0] for value in normalizations[1:]):
        raise ValueError(f"{model_name} final normalization differs across seeds")
    return models, neural.Normalization(**normalizations[0])


def predict_neural(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    normalization: neural.Normalization,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            normalized = model(inputs[start : start + batch_size].to(device))
            chunks.append(normalized.detach().cpu().numpy())
    values = np.concatenate(chunks, axis=0)
    return neural.inverse_target(values, normalization)


def neural_missingness_rows(
    model_name: str,
    label: str,
    models: Sequence[torch.nn.Module],
    normalization: neural.Normalization,
    dataset: neural.CachedDataset,
    holdout: np.ndarray,
    actual: np.ndarray,
    scales: np.ndarray,
    cells: np.ndarray,
    thresholds: np.ndarray,
    mechanisms: Sequence[str],
    rates: Sequence[float],
    device: torch.device,
    batch_size: int,
    expected_clean: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    clean_checked = False
    for mechanism in mechanisms:
        for rate in rates:
            extra = sea.global_corruption_mask(
                np.asarray(dataset.cells[holdout]),
                np.asarray(dataset.history_end_hours[holdout]),
                mechanism=mechanism,
                requested_rate=rate,
                seed=42,
            )
            inputs = corrupted_neural_inputs(dataset, holdout, normalization, extra)
            predictions = [
                predict_neural(model, inputs, normalization, device, batch_size)
                for model in models
            ]
            ensemble = np.mean(np.stack(predictions, axis=0), axis=0, dtype=np.float64).astype(np.float32)
            if rate == 0.0 and not clean_checked:
                maximum = float(np.max(np.abs(ensemble - expected_clean)))
                if maximum > 5e-5:
                    raise ValueError(f"{model_name} clean replay mismatch: {maximum}")
                clean_checked = True
            metrics = sea.forecast_metrics(actual, ensemble, scales, cells)
            ths = sea.threshold_hit_score(actual, ensemble, thresholds)
            stats = sea.corruption_statistics(np.asarray(dataset.x_masks[holdout]), extra)
            rows.append(
                {
                    "method": label,
                    "mechanism": mechanism,
                    "requested_rate": rate,
                    **stats,
                    "macro_wape": metrics["macro_indicator"]["wape"],
                    "pooled_wape": metrics["pooled_wape"],
                    "mase": metrics["macro_indicator"]["mase"],
                    "smape": metrics["macro_indicator"]["smape"],
                    "threshold_hit_score": ths["score"],
                    "five_seed_ensemble": True,
                }
            )
    return rows


def sea_missingness_rows(
    source: Path,
    dataset: neural.CachedDataset,
    holdout: np.ndarray,
    actual: np.ndarray,
    scales: np.ndarray,
    cells: np.ndarray,
    thresholds: np.ndarray,
    mechanisms: Sequence[str],
    rates: Sequence[float],
    device: torch.device,
    batch_size: int,
    expected_clean: np.ndarray,
) -> list[dict[str, object]]:
    models: list[sea.WLCRSEA] = []
    priors: list[np.ndarray] = []
    for seed in SEEDS:
        model, payload = runner.load_checkpoint(
            source / "models" / f"{runner.PRIMARY_VARIANT}_seed{seed}.pt", device
        )
        models.append(model)
        priors.append(np.asarray(payload["prior_log"], dtype=np.float32))
    if any(not np.array_equal(prior, priors[0]) for prior in priors[1:]):
        raise ValueError("WLCR-SEA training prior differs across seeds")
    prior = priors[0]
    fixed = sea.WLCRSEA(sea.VARIANTS["A0_fixed"], token_dim=16, hidden_dim=32).to(device).eval()
    rows: list[dict[str, object]] = []
    clean_checked = False
    for mechanism in mechanisms:
        for rate in rates:
            extra = sea.global_corruption_mask(
                np.asarray(dataset.cells[holdout]),
                np.asarray(dataset.history_end_hours[holdout]),
                mechanism=mechanism,
                requested_rate=rate,
                seed=42,
            )
            _, tensors = runner.make_eval_tensors(
                dataset, holdout, prior, additional_missing=extra
            )
            predictions = [
                runner.predict(model, tensors, device=device, batch_size=batch_size)["prediction"]
                for model in models
            ]
            ensemble = np.mean(np.stack(predictions, axis=0), axis=0, dtype=np.float64).astype(np.float32)
            if rate == 0.0 and not clean_checked:
                maximum = float(np.max(np.abs(ensemble - expected_clean)))
                if maximum > 5e-5:
                    raise ValueError(f"WLCR-SEA clean replay mismatch: {maximum}")
                clean_checked = True
            fixed_prediction = runner.predict(
                fixed, tensors, device=device, batch_size=batch_size
            )["prediction"]
            stats = sea.corruption_statistics(np.asarray(dataset.x_masks[holdout]), extra)
            for method, prediction, is_ensemble in (
                (LABELS[runner.PRIMARY_VARIANT], ensemble, True),
                (LABELS["A0_fixed"], fixed_prediction, False),
            ):
                metrics = sea.forecast_metrics(actual, prediction, scales, cells)
                ths = sea.threshold_hit_score(actual, prediction, thresholds)
                rows.append(
                    {
                        "method": method,
                        "mechanism": mechanism,
                        "requested_rate": rate,
                        **stats,
                        "macro_wape": metrics["macro_indicator"]["wape"],
                        "pooled_wape": metrics["pooled_wape"],
                        "mase": metrics["macro_indicator"]["mase"],
                        "smape": metrics["macro_indicator"]["smape"],
                        "threshold_hit_score": ths["score"],
                        "five_seed_ensemble": is_ensemble,
                    }
                )
    return rows


def write_audit_tables(
    source: Path,
    output: Path,
    actual: np.ndarray,
) -> None:
    archive = np.load(
        source / "worker_audit" / f"{runner.PRIMARY_VARIANT}_seed42.npz",
        allow_pickle=False,
    )
    attention = np.asarray(archive["attention"], dtype=np.float64)
    entropy = np.asarray(archive["entropy"], dtype=np.float64)
    availability = np.asarray(archive["availability"], dtype=bool)
    reliability = np.asarray(archive["reliability"], dtype=np.float64)
    prediction = np.load(
        source / "worker_predictions" / f"{runner.PRIMARY_VARIANT}_seed42.npy",
        allow_pickle=False,
    )
    expert_rows = []
    for index, name in enumerate(sea.EXPERT_NAMES):
        available = availability[..., index]
        expert_rows.append(
            {
                "expert_index": index + 1,
                "expert": name,
                "mean_attention": float(np.mean(attention[..., index])),
                "availability_rate": float(np.mean(available)),
                "mean_reliability": float(np.mean(reliability[..., index])),
                "mean_attention_when_available": float(np.mean(attention[..., index][available])),
            }
        )
    runner.atomic_csv(output / "audit_expert_weights.csv", expert_rows)

    support = np.sum(attention > 1e-6, axis=-1).ravel()
    support_rows = [
        {
            "effective_experts": count,
            "requests": int(np.sum(support == count)),
            "fraction": float(np.mean(support == count)),
        }
        for count in range(1, sea.EXPERT_COUNT + 1)
    ]
    runner.atomic_csv(output / "audit_support_distribution.csv", support_rows)

    valid = np.isfinite(actual) & (actual > 0.0) & np.isfinite(prediction)
    flat_entropy = entropy[valid]
    flat_ape = (np.abs(actual - prediction) / np.maximum(actual, 1e-12))[valid]
    order = np.argsort(flat_entropy, kind="mergesort")
    groups = np.empty(len(order), dtype=np.int64)
    groups[order] = np.minimum(np.arange(len(order)) * 10 // len(order), 9)
    decile_rows = []
    for decile in range(10):
        selected = groups == decile
        decile_rows.append(
            {
                "entropy_decile": decile + 1,
                "n": int(np.sum(selected)),
                "mean_entropy": float(np.mean(flat_entropy[selected])),
                "mean_ape": float(np.mean(flat_ape[selected])),
                "median_ape": float(np.median(flat_ape[selected])),
            }
        )
    runner.atomic_csv(output / "audit_entropy_deciles.csv", decile_rows)

    audit = json.loads((source / "auditability.json").read_text(encoding="utf-8"))
    deletion = audit["deletion_fidelity"]
    runner.atomic_csv(
        output / "audit_deletion_fidelity.csv",
        [
            {
                "condition": "Original",
                "macro_wape": deletion["original_macro_wape"],
                "delta_vs_original": 0.0,
            },
            {
                "condition": "Remove random expert",
                "macro_wape": deletion["remove_random_macro_wape"],
                "delta_vs_original": deletion["random_delta"],
            },
            {
                "condition": "Remove top expert",
                "macro_wape": deletion["remove_top_macro_wape"],
                "delta_vs_original": deletion["top_delta"],
            },
        ],
    )


def run(args: argparse.Namespace) -> int:
    root = runner.project_root()
    source = resolve(root, args.source, strict=True)
    baseline_root = resolve(root, args.baseline_root, strict=True)
    standard_root = resolve(root, args.standard_root, strict=True)
    output = resolve(root, args.output, strict=False)
    allowed = (root / runner.OUTPUT_ROOT).resolve(strict=False)
    if not output.is_relative_to(allowed):
        raise ValueError("output must remain under artifacts/revision6")
    output.mkdir(parents=True, exist_ok=True)

    dataset, dataset_report, train_path = dataset_from_registered_train()
    input_before = neural.sha256_file(train_path)
    fit = dataset.indices_for_dates(neural.FIT_DATES)
    inner = dataset.indices_for_dates(neural.INNER_DATES)
    final_train = np.concatenate((fit, inner))
    holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    scales = np.asarray(dataset.mase_scales[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    shape = actual.shape
    thresholds = sea.frozen_low_activity_thresholds(
        dataset.targets, dataset.target_masks, final_train
    )

    predictions: dict[str, np.ndarray] = {
        "last_day": load_prediction(source / "baselines" / "last_day.npy", shape),
        "last_week": load_prediction(source / "baselines" / "last_week.npy", shape),
        "same_hour_median_7d": load_prediction(
            source / "baselines" / "same_hour_median_7d.npy", shape
        ),
        "A0_fixed": load_prediction(source / "baselines" / "A0_fixed.npy", shape),
        "A4_reliability": load_prediction(
            source / "predictions" / "A4_reliability_ensemble.npy", shape
        ),
        "A6_mixed_aug": load_prediction(
            source / "predictions" / "A6_mixed_aug_ensemble.npy", shape
        ),
        "A7_cross_indicator": load_prediction(
            source / "predictions" / "A7_cross_indicator_ensemble.npy", shape
        ),
    }
    neural_seed_paths: dict[str, list[Path]] = {}
    for key, raw in (("DLinear", "dlinear"), ("PatchTST", "patchtst")):
        paths = [
            baseline_root / "worker_predictions" / f"{raw}_seed{seed}.npy"
            for seed in SEEDS
        ]
        neural_seed_paths[key] = paths
        predictions[key] = ensemble_files(paths, shape)

    reorder = standard_reorder(standard_root / "holdout_order.csv", dataset, holdout)
    standard_prediction = load_prediction(
        standard_root / "holdout_predictions.npy", shape
    )
    predictions["standard_stat_traffic_only_165d"] = standard_prediction[reorder]

    sea_seed_paths = [
        source / "worker_predictions" / f"{runner.PRIMARY_VARIANT}_seed{seed}.npy"
        for seed in SEEDS
    ]
    convex_seed_paths = [
        source / "worker_predictions" / f"A4_reliability_seed{seed}.npy"
        for seed in SEEDS
    ]
    seed_paths = {
        runner.PRIMARY_VARIANT: sea_seed_paths,
        "A4_reliability": convex_seed_paths,
        **neural_seed_paths,
    }

    clean_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    indicator_rows: list[dict[str, object]] = []
    metrics_by_method: dict[str, dict[str, object]] = {}
    for method in METHOD_ORDER:
        stats = (
            seed_metric_stats(seed_paths[method], actual, scales, cells)
            if method in seed_paths
            else None
        )
        row, metrics = metric_row(
            method,
            predictions[method],
            actual,
            scales,
            cells,
            thresholds,
            stats,
        )
        clean_rows.append(row)
        metrics_by_method[method] = metrics
        for horizon, value in enumerate(metrics["per_horizon_wape"], start=1):
            horizon_rows.append(
                {
                    "method": method,
                    "label": LABELS[method],
                    "horizon": horizon,
                    "wape": value,
                }
            )
        for item in metrics["per_indicator"]:
            indicator_rows.append(
                {"method": method, "label": LABELS[method], **item}
            )
    runner.atomic_csv(output / "comparative_clean_accuracy.csv", clean_rows)
    runner.atomic_csv(output / "comparative_per_horizon.csv", horizon_rows)
    runner.atomic_csv(output / "comparative_per_indicator.csv", indicator_rows)

    comparison_methods = (
        "A0_fixed",
        "standard_stat_traffic_only_165d",
        "DLinear",
        "PatchTST",
        "A4_reliability",
    )
    bootstrap_rows = []
    win_rows = []
    for baseline in comparison_methods:
        result = sea.cell_cluster_bootstrap_wape_delta(
            actual,
            predictions[runner.PRIMARY_VARIANT],
            predictions[baseline],
            cells,
            replicates=args.bootstrap_replicates,
            seed=42,
        )
        bootstrap_rows.append(
            {
                "comparison": f"{LABELS[runner.PRIMARY_VARIANT]} minus {LABELS[baseline]}",
                "proposed": runner.PRIMARY_VARIANT,
                "baseline": baseline,
                **result,
            }
        )
        win_rows.append(
            cell_win_row(
                predictions[runner.PRIMARY_VARIANT],
                predictions[baseline],
                actual,
                cells,
                baseline,
            )
        )
    runner.atomic_csv(output / "paired_cell_bootstrap.csv", bootstrap_rows)
    runner.atomic_csv(output / "per_cell_win_rates.csv", win_rows)

    mechanisms = runner.ROBUSTNESS_MECHANISMS
    rates = runner.ROBUSTNESS_RATES
    if args.smoke:
        mechanisms = mechanisms[:2]
        rates = rates[:2]
    devices = [int(value) for value in args.gpu_devices.split(",") if value.strip()]
    if not devices or not torch.cuda.is_available():
        raise RuntimeError("missingness evaluation requires CUDA")
    torch.backends.cudnn.benchmark = False
    missing_rows: list[dict[str, object]] = []
    sea_device = torch.device(f"cuda:{devices[0]}")
    missing_rows.extend(
        sea_missingness_rows(
            source,
            dataset,
            holdout,
            actual,
            scales,
            cells,
            thresholds,
            mechanisms,
            rates,
            sea_device,
            args.batch_size,
            predictions[runner.PRIMARY_VARIANT],
        )
    )
    for offset, (model_name, label, clean_key) in enumerate(
        (("dlinear", LABELS["DLinear"], "DLinear"), ("patchtst", LABELS["PatchTST"], "PatchTST")),
        start=1,
    ):
        device = torch.device(f"cuda:{devices[offset % len(devices)]}")
        models, normalization = neural_models(baseline_root, model_name, device)
        missing_rows.extend(
            neural_missingness_rows(
                model_name,
                label,
                models,
                normalization,
                dataset,
                holdout,
                actual,
                scales,
                cells,
                thresholds,
                mechanisms,
                rates,
                device,
                args.batch_size,
                predictions[clean_key],
            )
        )
        for model in models:
            model.to("cpu")
        torch.cuda.empty_cache()
    missing_rows.sort(
        key=lambda row: (
            tuple(LABELS.values()).index(row["method"])
            if row["method"] in tuple(LABELS.values())
            else 99,
            str(row["mechanism"]),
            float(row["requested_rate"]),
        )
    )
    runner.atomic_csv(output / "comparative_missingness.csv", missing_rows)

    write_audit_tables(source, output, actual)
    input_after = neural.sha256_file(train_path)
    if input_before != input_after:
        raise RuntimeError("registered training data changed")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "primary_variant_predeclared": runner.PRIMARY_VARIANT,
        "clean_accuracy": {row["method"]: row for row in clean_rows},
        "paired_bootstrap": bootstrap_rows,
        "per_cell_win_rates": win_rows,
        "missingness_rows": len(missing_rows),
        "missingness_mechanisms": list(mechanisms),
        "missingness_rates": list(rates),
        "frozen_training_thresholds": thresholds.tolist(),
        "dataset_report": dataset_report,
        "finals_test_opened": False,
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
    }
    runner.atomic_json(output / "summary.json", payload)
    runner.atomic_json(output / "manifest.json", runner.output_manifest(output))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", default="artifacts/revision6/wlcr_sea_v2")
    value.add_argument(
        "--baseline-root",
        default="artifacts/paper_neural_baselines_v1/revision6_5seed100/results",
    )
    value.add_argument(
        "--standard-root", default="artifacts/revision6/standard_stat_traffic_only"
    )
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    value.add_argument("--gpu-devices", default="0,1,2")
    value.add_argument("--batch-size", type=int, default=256)
    value.add_argument("--bootstrap-replicates", type=int, default=5000)
    value.add_argument("--smoke", action="store_true")
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
