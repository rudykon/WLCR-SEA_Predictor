from __future__ import annotations

"""Revision-8 clean, fairness-controlled missingness, and routing ablations.

All samples are derived from the registered training trace.  The finals test
traffic file is never opened.  The August holdout was used by earlier project
revisions, so every result remains exploratory rather than confirmatory.
"""

import argparse
import json
import math
import statistics
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from experiments import missingness_protocol as missingness
from experiments import train_neural_baselines as neural
from experiments import analyze_model_comparisons as revision6
from experiments import train_wlcr_sea as runner
from experiments import wlcr_sea_model as sea


SCHEMA_VERSION = 2
MODEL_SEEDS = (42, 43, 44, 45, 46)
CORRUPTION_SEEDS = (142, 143, 144, 145, 146)
T95_DF4 = 2.7764451051977987
CLEAN_REPLAY_ABS_TOLERANCE = 2e-4
DEFAULT_OUTPUT = Path("artifacts/revision8/comparative_analysis")
DEFAULT_ORIGINAL_WLCR = Path(
    "artifacts/revision7/original_wlcr_alignment/original_wlcr_holdout_predictions.npy"
)

METHOD_META: dict[str, tuple[str, str, str]] = {
    "last_day": ("Last-day seasonal", "traffic", "deterministic"),
    "last_week": ("Last-week seasonal", "traffic", "deterministic"),
    "same_hour_median_7d": (
        "Same-hour median (7 d)",
        "traffic + masks",
        "deterministic",
    ),
    "A0_fixed": (
        "Fixed Seasonal Expert Mixture",
        "traffic + masks + horizon",
        "deterministic",
    ),
    "A0_global_static": (
        "Learned global indicator weights",
        "traffic + masks + indicator",
        "clean",
    ),
    "A0_horizon_indicator": (
        "Learned horizon-indicator weights",
        "traffic + masks + horizon",
        "clean",
    ),
    "A1_softmax": (
        "Dynamic Softmax router",
        "traffic + masks + horizon + request context",
        "clean",
    ),
    "A2_entmax": (
        "Dynamic Entmax router",
        "traffic + masks + horizon + request context",
        "clean",
    ),
    "A3_hard_mask": (
        "A3: + hard availability mask",
        "traffic + masks + horizon + request context",
        "clean",
    ),
    "A4_reliability": (
        "A4: + reliability descriptor",
        "traffic + masks + horizon + request context",
        "clean",
    ),
    "A5_residual": (
        "A5: + bounded residual",
        "traffic + masks + horizon + request context",
        "clean",
    ),
    "A6_mixed_aug": (
        "WLCR-SEA (mixed augmentation)",
        "traffic + masks + horizon + request context",
        "mixed-15pct",
    ),
    "original_wlcr": (
        "Original WLCR-LightGBM (traffic-only)",
        "traffic + masks + horizon",
        "clean",
    ),
    "standard_stat": (
        "Standard-stat LightGBM",
        "traffic + masks + horizon",
        "clean",
    ),
    "dlinear_clean": ("DLinear", "traffic + masks", "clean"),
    "dlinear_aug": (
        "DLinear + mixed augmentation",
        "traffic + masks",
        "mixed-15pct",
    ),
    "patchtst_clean": ("PatchTST", "traffic + masks", "clean"),
    "patchtst_aug": (
        "PatchTST + mixed augmentation",
        "traffic + masks",
        "mixed-15pct",
    ),
}


def resolve_inside(root: Path, value: str | Path, *, strict: bool) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=strict)
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes project root: {path}")
    return path


def load_seed_predictions(
    root: Path,
    stem: str,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, list[Path]]:
    paths = [root / "worker_predictions" / f"{stem}_seed{seed}.npy" for seed in MODEL_SEEDS]
    prediction = revision6.ensemble_files(paths, shape)
    return prediction, paths


def seed_wape_summary(
    paths: Sequence[Path],
    actual: np.ndarray,
    scales: np.ndarray,
    cells: np.ndarray,
) -> dict[str, float]:
    values = [
        float(
            sea.forecast_metrics(
                actual,
                revision6.load_prediction(path, actual.shape),
                scales,
                cells,
            )["macro_indicator"]["wape"]
        )
        for path in paths
    ]
    return {
        "seed_macro_wape_mean": float(statistics.mean(values)),
        "seed_macro_wape_sd": float(statistics.stdev(values)),
        "seed_macro_wape_min": float(min(values)),
        "seed_macro_wape_max": float(max(values)),
    }


def clean_metric_row(
    key: str,
    prediction: np.ndarray,
    actual: np.ndarray,
    scales: np.ndarray,
    cells: np.ndarray,
    thresholds: np.ndarray,
    seed_stats: Mapping[str, float] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    label, information, training = METHOD_META[key]
    metrics = sea.forecast_metrics(actual, prediction, scales, cells)
    ths = sea.threshold_hit_score(actual, prediction, thresholds)
    row: dict[str, object] = {
        "method": key,
        "label": label,
        "information_class": information,
        "training_view": training,
        "macro_wape": metrics["macro_indicator"]["wape"],
        "pooled_wape": metrics["pooled_wape"],
        "macro_cell_wape": metrics["macro_cell_wape"],
        "median_cell_wape": metrics["median_cell_wape"],
        "mase": metrics["macro_indicator"]["mase"],
        "smape": metrics["macro_indicator"]["smape"],
        "mae": metrics["macro_indicator"]["mae"],
        "rmse": metrics["macro_indicator"]["rmse"],
        "threshold_hit_score": ths["score"],
    }
    if seed_stats is None:
        row.update(
            {
                "seed_macro_wape_mean": "",
                "seed_macro_wape_sd": "",
                "seed_macro_wape_min": "",
                "seed_macro_wape_max": "",
            }
        )
    else:
        row.update(seed_stats)
    return row, metrics


def load_sea_models(
    source: Path,
    variant: str,
    device: torch.device,
) -> tuple[list[sea.WLCRSEA], np.ndarray]:
    models: list[sea.WLCRSEA] = []
    priors: list[np.ndarray] = []
    for seed in MODEL_SEEDS:
        model, payload = runner.load_checkpoint(
            source / "models" / f"{variant}_seed{seed}.pt", device
        )
        if payload["variant"]["name"] != variant:
            raise ValueError(f"checkpoint variant mismatch for {variant}/seed{seed}")
        models.append(model)
        priors.append(np.asarray(payload["prior_log"], dtype=np.float32))
    if any(not np.array_equal(prior, priors[0]) for prior in priors[1:]):
        raise ValueError(f"training prior differs across {variant} seeds")
    return models, priors[0]


def load_neural_models(
    root: Path,
    model_name: str,
    device: torch.device,
    *,
    expected_augmentation: str,
) -> tuple[list[torch.nn.Module], neural.Normalization]:
    models: list[torch.nn.Module] = []
    normalizations: list[dict[str, object]] = []
    for seed in MODEL_SEEDS:
        path = root / "models" / f"{model_name}_seed{seed}.pt"
        payload = torch.load(path, map_location="cpu")
        if payload.get("model") != model_name or int(payload.get("seed")) != seed:
            raise ValueError(f"invalid neural checkpoint metadata: {path}")
        observed_augmentation = str(payload.get("augmentation", "clean"))
        if observed_augmentation != expected_augmentation:
            raise ValueError(
                f"{path} uses {observed_augmentation}, expected {expected_augmentation}"
            )
        if expected_augmentation == "mixed" and not math.isclose(
            float(payload.get("augmentation_rate", -1.0)), 0.15
        ):
            raise ValueError(f"{path} does not use 15% mixed augmentation")
        model = neural.build_model(model_name, payload["config"])
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        models.append(model)
        normalizations.append(payload["normalization"])
    if any(item != normalizations[0] for item in normalizations[1:]):
        raise ValueError(f"{model_name} normalization differs across seeds")
    return models, neural.Normalization(**normalizations[0])


def predict_neural_ensemble(
    models: Sequence[torch.nn.Module],
    inputs: torch.Tensor,
    normalization: neural.Normalization,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for model in models:
        normalized = neural.predict_normalized(
            model,
            inputs,
            batch_size=batch_size,
            device=device,
        )
        predictions.append(neural.inverse_target(normalized, normalization))
    return np.mean(np.stack(predictions, axis=0), axis=0, dtype=np.float64).astype(
        np.float32
    )


def shared_neural_request_view(
    dataset: neural.CachedDataset,
    indices: np.ndarray,
    additional_missing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one corruption-dependent fill view shared by all normalizations."""
    base_values = np.asarray(dataset.x_values[indices], dtype=np.float32)
    original_masks = np.asarray(dataset.x_masks[indices], dtype=bool)
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
    return np.asarray(values, dtype=np.float32), masks


def normalized_neural_inputs(
    values: np.ndarray,
    masks: np.ndarray,
    normalization: neural.Normalization,
) -> torch.Tensor:
    input_mean = np.asarray(normalization.input_mean, dtype=np.float32)
    input_std = np.asarray(normalization.input_std, dtype=np.float32)
    normalized = (
        np.asarray(values, dtype=np.float32) - input_mean[None, None, :]
    ) / input_std[None, None, :]
    inputs = np.concatenate(
        (normalized, np.asarray(masks, dtype=np.float32)), axis=2
    )
    return torch.from_numpy(inputs.astype(np.float32))


def predict_sea_ensemble(
    models: Sequence[sea.WLCRSEA],
    tensors: tuple[torch.Tensor, ...],
    device: torch.device,
    batch_size: int,
    *,
    include_attention: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    predictions: list[np.ndarray] = []
    attention_sum: np.ndarray | None = None
    for model in models:
        output = runner.predict(
            model,
            tensors,
            device=device,
            batch_size=batch_size,
            include_audit=include_attention,
        )
        predictions.append(output["prediction"])
        if include_attention:
            attention = np.asarray(output["attention"], dtype=np.float64)
            attention_sum = attention if attention_sum is None else attention_sum + attention
    ensemble = np.mean(np.stack(predictions, axis=0), axis=0, dtype=np.float64).astype(
        np.float32
    )
    mean_attention = None
    if attention_sum is not None:
        mean_attention = (attention_sum / len(models)).astype(np.float32)
    return ensemble, mean_attention


def t_interval(values: Sequence[float]) -> tuple[float, float, float, float]:
    mean = float(statistics.mean(values))
    if len(values) == 1:
        return mean, 0.0, mean, mean
    sd = float(statistics.stdev(values))
    half = T95_DF4 * sd / math.sqrt(len(values))
    return mean, sd, mean - half, mean + half


def aggregate_seed_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group_fields: Sequence[str],
    numeric_fields: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        items = grouped[key]
        record = {field: value for field, value in zip(group_fields, key)}
        record["corruption_seed_count"] = len(items)
        record["corruption_seeds"] = ",".join(
            str(int(item["corruption_seed"])) for item in items
        )
        for field in numeric_fields:
            values = [float(item[field]) for item in items]
            mean, sd, low, high = t_interval(values)
            record[field] = mean
            record[f"{field}_sd"] = sd
            record[f"{field}_ci_low"] = low
            record[f"{field}_ci_high"] = high
        output.append(record)
    return output


def metric_payload(
    prediction: np.ndarray,
    actual: np.ndarray,
    scales: np.ndarray,
    cells: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    metrics = sea.forecast_metrics(actual, prediction, scales, cells)
    ths = sea.threshold_hit_score(actual, prediction, thresholds)
    values = {
        "macro_wape": metrics["macro_indicator"]["wape"],
        "pooled_wape": metrics["pooled_wape"],
        "mase": metrics["macro_indicator"]["mase"],
        "smape": metrics["macro_indicator"]["smape"],
        "threshold_hit_score": ths["score"],
    }
    return values, metrics


def scenario_name(mechanism: str, rate: float) -> str:
    if rate == 0.0:
        return "clean"
    if mechanism == "recent_tail":
        return f"timeline_tail_{rate:.2f}"
    return f"{mechanism}_{rate:.2f}"


def run(args: argparse.Namespace) -> int:
    root = runner.project_root()
    source = resolve_inside(root, args.source, strict=True)
    clean_neural_root = resolve_inside(root, args.clean_neural_root, strict=True)
    augmented_neural_root = resolve_inside(
        root, args.augmented_neural_root, strict=True
    )
    standard_root = resolve_inside(root, args.standard_root, strict=True)
    original_wlcr = resolve_inside(root, args.original_wlcr, strict=True)
    output = resolve_inside(root, args.output, strict=False)
    allowed = (root / "artifacts/revision8").resolve(strict=False)
    if not output.is_relative_to(allowed):
        raise ValueError("Revision-8 analysis output must remain under artifacts/revision8")
    output.mkdir(parents=True, exist_ok=True)

    dataset, dataset_report, train_path = revision6.dataset_from_registered_train()
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
        "last_day": revision6.load_prediction(source / "baselines/last_day.npy", shape),
        "last_week": revision6.load_prediction(source / "baselines/last_week.npy", shape),
        "same_hour_median_7d": revision6.load_prediction(
            source / "baselines/same_hour_median_7d.npy", shape
        ),
        "A0_fixed": revision6.load_prediction(source / "baselines/A0_fixed.npy", shape),
        "original_wlcr": revision6.load_prediction(original_wlcr, shape),
    }
    seed_paths: dict[str, list[Path]] = {}
    for variant in (
        "A0_global_static",
        "A0_horizon_indicator",
        "A1_softmax",
        "A2_entmax",
        "A3_hard_mask",
        "A4_reliability",
        "A5_residual",
        "A6_mixed_aug",
    ):
        prediction, paths = load_seed_predictions(source, variant, shape)
        predictions[variant] = prediction
        seed_paths[variant] = paths
    for key, stem, checkpoint_root in (
        ("dlinear_clean", "dlinear", clean_neural_root),
        ("patchtst_clean", "patchtst", clean_neural_root),
        ("dlinear_aug", "dlinear", augmented_neural_root),
        ("patchtst_aug", "patchtst", augmented_neural_root),
    ):
        prediction, paths = load_seed_predictions(checkpoint_root, stem, shape)
        predictions[key] = prediction
        seed_paths[key] = paths

    reorder = revision6.standard_reorder(
        standard_root / "holdout_order.csv", dataset, holdout
    )
    standard_prediction = revision6.load_prediction(
        standard_root / "holdout_predictions.npy", shape
    )
    predictions["standard_stat"] = standard_prediction[reorder]

    clean_order = tuple(METHOD_META)
    clean_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    indicator_rows: list[dict[str, object]] = []
    clean_metrics: dict[str, dict[str, object]] = {}
    for key in clean_order:
        stats = (
            seed_wape_summary(seed_paths[key], actual, scales, cells)
            if key in seed_paths
            else None
        )
        row, metrics = clean_metric_row(
            key,
            predictions[key],
            actual,
            scales,
            cells,
            thresholds,
            stats,
        )
        clean_rows.append(row)
        clean_metrics[key] = metrics
        for horizon, value in enumerate(metrics["per_horizon_wape"], start=1):
            horizon_rows.append(
                {
                    "method": key,
                    "label": METHOD_META[key][0],
                    "horizon": horizon,
                    "wape": value,
                }
            )
        for item in metrics["per_indicator"]:
            indicator_rows.append(
                {"method": key, "label": METHOD_META[key][0], **item}
            )
    runner.atomic_csv(output / "comparative_clean_accuracy.csv", clean_rows)
    runner.atomic_csv(output / "comparative_per_horizon.csv", horizon_rows)
    runner.atomic_csv(output / "comparative_per_indicator.csv", indicator_rows)

    bootstrap_rows: list[dict[str, object]] = []
    for baseline in (
        "A0_fixed",
        "A0_global_static",
        "A0_horizon_indicator",
        "original_wlcr",
        "standard_stat",
        "dlinear_clean",
        "patchtst_clean",
        "dlinear_aug",
        "patchtst_aug",
        "A5_residual",
    ):
        result = sea.cell_cluster_bootstrap_wape_delta(
            actual,
            predictions["A6_mixed_aug"],
            predictions[baseline],
            cells,
            replicates=args.bootstrap_replicates,
            seed=42,
        )
        direct = float(
            clean_metrics["A6_mixed_aug"]["macro_indicator"]["wape"]
            - clean_metrics[baseline]["macro_indicator"]["wape"]
        )
        if not math.isclose(
            float(result["delta_proposed_minus_baseline"]),
            direct,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"bootstrap point mismatch for {baseline}")
        bootstrap_rows.append(
            {
                "comparison": (
                    f"{METHOD_META['A6_mixed_aug'][0]} minus {METHOD_META[baseline][0]}"
                ),
                "proposed": "A6_mixed_aug",
                "baseline": baseline,
                **result,
            }
        )
    runner.atomic_csv(output / "paired_cell_bootstrap.csv", bootstrap_rows)

    devices = [int(item) for item in args.gpu_devices.split(",") if item.strip()]
    if len(devices) < 3 or not torch.cuda.is_available():
        raise RuntimeError("Revision-8 missingness analysis requires at least three GPUs")
    sea_device = torch.device(f"cuda:{devices[0]}")
    dlinear_device = torch.device(f"cuda:{devices[1]}")
    patchtst_device = torch.device(f"cuda:{devices[2]}")
    a6_models, prior = load_sea_models(source, "A6_mixed_aug", sea_device)
    dlinear_clean_models, dlinear_clean_norm = load_neural_models(
        clean_neural_root,
        "dlinear",
        dlinear_device,
        expected_augmentation="clean",
    )
    dlinear_aug_models, dlinear_aug_norm = load_neural_models(
        augmented_neural_root,
        "dlinear",
        dlinear_device,
        expected_augmentation="mixed",
    )
    patch_clean_models, patch_clean_norm = load_neural_models(
        clean_neural_root,
        "patchtst",
        patchtst_device,
        expected_augmentation="clean",
    )
    patch_aug_models, patch_aug_norm = load_neural_models(
        augmented_neural_root,
        "patchtst",
        patchtst_device,
        expected_augmentation="mixed",
    )

    mechanisms = runner.ROBUSTNESS_MECHANISMS
    rates = runner.ROBUSTNESS_RATES
    corruption_seeds = CORRUPTION_SEEDS
    if args.smoke:
        mechanisms = ("block",)
        rates = (0.0, 0.2)
        corruption_seeds = (CORRUPTION_SEEDS[0],)

    missing_rows: list[dict[str, object]] = []
    missing_indicator_rows: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []
    horizon_weight_rows: list[dict[str, object]] = []
    a6_ablation_cache: dict[tuple[str, float, int], dict[str, object]] = {}
    clean_replay_checked: set[str] = set()
    clean_replay_max_absolute_difference: dict[str, float] = {}
    selected_weight_scenarios = {
        ("mcar", 0.0),
        ("block", 0.2),
        ("recent_tail", 0.2),
        ("asynchronous", 0.2),
        ("block", 0.5),
    }
    for mechanism in mechanisms:
        for rate in rates:
            if rate == 0.0 and mechanism != mechanisms[0]:
                continue
            for corruption_seed in corruption_seeds:
                extra = missingness.global_corruption_mask(
                    np.asarray(dataset.cells[holdout]),
                    np.asarray(dataset.history_end_hours[holdout]),
                    mechanism=mechanism,
                    requested_rate=rate,
                    seed=corruption_seed,
                )
                stats = missingness.corruption_statistics(
                    np.asarray(dataset.x_masks[holdout]),
                    extra,
                    cells=np.asarray(dataset.cells[holdout]),
                    history_end_hours=np.asarray(dataset.history_end_hours[holdout]),
                    mechanism=mechanism,
                    requested_rate=rate,
                    seed=corruption_seed,
                )
                flat_stats = missingness.flatten_statistics(stats)
                _, sea_tensors = runner.make_eval_tensors(
                    dataset, holdout, prior, additional_missing=extra
                )
                include_attention = (mechanism, rate) in selected_weight_scenarios
                sea_prediction, mean_attention = predict_sea_ensemble(
                    a6_models,
                    sea_tensors,
                    sea_device,
                    args.batch_size,
                    include_attention=include_attention,
                )

                shared_values, shared_masks = shared_neural_request_view(
                    dataset, holdout, extra
                )
                dclean_inputs = normalized_neural_inputs(
                    shared_values, shared_masks, dlinear_clean_norm
                )
                daug_inputs = normalized_neural_inputs(
                    shared_values, shared_masks, dlinear_aug_norm
                )
                pclean_inputs = normalized_neural_inputs(
                    shared_values, shared_masks, patch_clean_norm
                )
                paug_inputs = normalized_neural_inputs(
                    shared_values, shared_masks, patch_aug_norm
                )
                condition_predictions = {
                    "A6_mixed_aug": sea_prediction,
                    "dlinear_clean": predict_neural_ensemble(
                        dlinear_clean_models,
                        dclean_inputs,
                        dlinear_clean_norm,
                        dlinear_device,
                        args.batch_size,
                    ),
                    "dlinear_aug": predict_neural_ensemble(
                        dlinear_aug_models,
                        daug_inputs,
                        dlinear_aug_norm,
                        dlinear_device,
                        args.batch_size,
                    ),
                    "patchtst_clean": predict_neural_ensemble(
                        patch_clean_models,
                        pclean_inputs,
                        patch_clean_norm,
                        patchtst_device,
                        args.batch_size,
                    ),
                    "patchtst_aug": predict_neural_ensemble(
                        patch_aug_models,
                        paug_inputs,
                        patch_aug_norm,
                        patchtst_device,
                        args.batch_size,
                    ),
                }
                for key, prediction in condition_predictions.items():
                    if rate == 0.0 and key not in clean_replay_checked:
                        maximum = float(np.max(np.abs(prediction - predictions[key])))
                        clean_replay_max_absolute_difference[key] = maximum
                        if maximum > CLEAN_REPLAY_ABS_TOLERANCE:
                            raise ValueError(
                                f"{key} clean replay mismatch: {maximum} exceeds "
                                f"{CLEAN_REPLAY_ABS_TOLERANCE}"
                            )
                        clean_replay_checked.add(key)
                    values, metrics = metric_payload(
                        prediction, actual, scales, cells, thresholds
                    )
                    row = {
                        "method": key,
                        "label": METHOD_META[key][0],
                        "training_view": METHOD_META[key][2],
                        "mechanism": mechanism,
                        "mechanism_display": (
                            "timeline_tail" if mechanism == "recent_tail" else mechanism
                        ),
                        "requested_rate": rate,
                        "corruption_seed": corruption_seed,
                        **flat_stats,
                        **values,
                    }
                    missing_rows.append(row)
                    if key == "A6_mixed_aug":
                        a6_ablation_cache[(mechanism, rate, corruption_seed)] = row
                    for item in metrics["per_indicator"]:
                        missing_indicator_rows.append(
                            {
                                "method": key,
                                "label": METHOD_META[key][0],
                                "training_view": METHOD_META[key][2],
                                "mechanism": mechanism,
                                "mechanism_display": (
                                    "timeline_tail"
                                    if mechanism == "recent_tail"
                                    else mechanism
                                ),
                                "requested_rate": rate,
                                "corruption_seed": corruption_seed,
                                "indicator": item["indicator"],
                                "wape": item["wape"],
                                "mase": item["mase"],
                                "smape": item["smape"],
                            }
                        )
                if mean_attention is not None:
                    scenario = scenario_name(mechanism, rate)
                    prior_rows.append(
                        {
                            "scenario": scenario,
                            "mechanism": mechanism,
                            "requested_rate": rate,
                            "corruption_seed": corruption_seed,
                            "mean_prior_mass": float(np.mean(mean_attention[..., 7])),
                            "p90_prior_mass": float(
                                np.quantile(mean_attention[..., 7], 0.9)
                            ),
                            "mean_effective_support": float(
                                np.mean(np.sum(mean_attention > 1e-6, axis=-1))
                            ),
                        }
                    )
                    for metric, metric_name in enumerate(sea.METRIC_NAMES):
                        for horizon in range(sea.FORECAST_HOURS):
                            for expert, expert_name in enumerate(sea.EXPERT_NAMES):
                                horizon_weight_rows.append(
                                    {
                                        "scenario": scenario,
                                        "mechanism": mechanism,
                                        "requested_rate": rate,
                                        "corruption_seed": corruption_seed,
                                        "indicator": metric_name,
                                        "horizon": horizon + 1,
                                        "expert": expert_name,
                                        "mean_weight": float(
                                            np.mean(
                                                mean_attention[
                                                    :, horizon, metric, expert
                                                ]
                                            )
                                        ),
                                    }
                                )

    runner.atomic_csv(output / "comparative_missingness_by_seed.csv", missing_rows)
    runner.atomic_csv(
        output / "comparative_missingness_per_indicator_by_seed.csv",
        missing_indicator_rows,
    )
    missing_numeric = (
        "macro_wape",
        "pooled_wape",
        "mase",
        "smape",
        "threshold_hit_score",
        "unique_original_missing_rate",
        "unique_selected_for_corruption_rate",
        "unique_newly_removed_rate",
        "unique_newly_removed_fraction_of_observed",
        "unique_final_total_missing_rate",
        "exposure_original_missing_rate",
        "exposure_selected_for_corruption_rate",
        "exposure_newly_removed_rate",
        "exposure_newly_removed_fraction_of_observed",
        "exposure_final_total_missing_rate",
    )
    missing_aggregate = aggregate_seed_rows(
        missing_rows,
        group_fields=(
            "method",
            "label",
            "training_view",
            "mechanism",
            "mechanism_display",
            "requested_rate",
        ),
        numeric_fields=missing_numeric,
    )
    runner.atomic_csv(output / "comparative_missingness.csv", missing_aggregate)
    indicator_aggregate = aggregate_seed_rows(
        missing_indicator_rows,
        group_fields=(
            "method",
            "label",
            "training_view",
            "mechanism",
            "mechanism_display",
            "requested_rate",
            "indicator",
        ),
        numeric_fields=("wape", "mase", "smape"),
    )
    runner.atomic_csv(
        output / "comparative_missingness_per_indicator.csv", indicator_aggregate
    )

    ablation_conditions = (
        ("mcar", 0.0),
        ("block", 0.2),
        ("recent_tail", 0.2),
        ("asynchronous", 0.2),
        ("block", 0.5),
    )
    if args.smoke:
        ablation_conditions = (("mcar", 0.0), ("block", 0.2))
    ablation_raw: list[dict[str, object]] = []
    for (mechanism, rate), corruption_seed in (
        (condition, seed)
        for condition in ablation_conditions
        for seed in corruption_seeds
    ):
        cached = a6_ablation_cache.get((mechanism, rate, corruption_seed))
        if cached is None and rate == 0.0:
            zero_rate_matches = [
                item
                for (_, cached_rate, cached_seed), item in a6_ablation_cache.items()
                if cached_rate == 0.0 and cached_seed == corruption_seed
            ]
            if len(zero_rate_matches) == 1:
                cached = zero_rate_matches[0]
        if cached is None:
            raise RuntimeError(f"missing A6 ablation cache for {mechanism}/{rate}")
        ablation_raw.append({**cached, "variant": "A6_mixed_aug"})

    ablation_device = torch.device(
        f"cuda:{devices[3] if len(devices) > 3 else devices[0]}"
    )
    for variant in ("A3_hard_mask", "A4_reliability", "A5_residual"):
        models, variant_prior = load_sea_models(source, variant, ablation_device)
        for mechanism, rate in ablation_conditions:
            for corruption_seed in corruption_seeds:
                mask_mechanism = "mcar" if rate == 0.0 else mechanism
                extra = missingness.global_corruption_mask(
                    np.asarray(dataset.cells[holdout]),
                    np.asarray(dataset.history_end_hours[holdout]),
                    mechanism=mask_mechanism,
                    requested_rate=rate,
                    seed=corruption_seed,
                )
                stats = missingness.corruption_statistics(
                    np.asarray(dataset.x_masks[holdout]),
                    extra,
                    cells=np.asarray(dataset.cells[holdout]),
                    history_end_hours=np.asarray(dataset.history_end_hours[holdout]),
                    mechanism=mask_mechanism,
                    requested_rate=rate,
                    seed=corruption_seed,
                )
                _, tensors = runner.make_eval_tensors(
                    dataset, holdout, variant_prior, additional_missing=extra
                )
                prediction, _ = predict_sea_ensemble(
                    models,
                    tensors,
                    ablation_device,
                    args.batch_size,
                    include_attention=False,
                )
                values, _ = metric_payload(
                    prediction, actual, scales, cells, thresholds
                )
                ablation_raw.append(
                    {
                        "variant": variant,
                        "method": variant,
                        "label": METHOD_META[variant][0],
                        "training_view": METHOD_META[variant][2],
                        "mechanism": mechanism,
                        "mechanism_display": (
                            "timeline_tail"
                            if mechanism == "recent_tail"
                            else mechanism
                        ),
                        "requested_rate": rate,
                        "corruption_seed": corruption_seed,
                        **missingness.flatten_statistics(stats),
                        **values,
                    }
                )
        for model in models:
            model.to("cpu")
        torch.cuda.empty_cache()
    runner.atomic_csv(output / "structured_ablation_by_seed.csv", ablation_raw)
    ablation_aggregate = aggregate_seed_rows(
        ablation_raw,
        group_fields=(
            "variant",
            "label",
            "training_view",
            "mechanism",
            "mechanism_display",
            "requested_rate",
        ),
        numeric_fields=(
            "macro_wape",
            "pooled_wape",
            "mase",
            "smape",
            "threshold_hit_score",
        ),
    )
    runner.atomic_csv(output / "structured_ablation.csv", ablation_aggregate)

    if prior_rows:
        runner.atomic_csv(output / "prior_mass_by_seed.csv", prior_rows)
        prior_aggregate = aggregate_seed_rows(
            prior_rows,
            group_fields=("scenario", "mechanism", "requested_rate"),
            numeric_fields=(
                "mean_prior_mass",
                "p90_prior_mass",
                "mean_effective_support",
            ),
        )
        runner.atomic_csv(output / "prior_mass.csv", prior_aggregate)
    if horizon_weight_rows:
        runner.atomic_csv(
            output / "expert_horizon_weights_by_seed.csv", horizon_weight_rows
        )
        horizon_aggregate = aggregate_seed_rows(
            horizon_weight_rows,
            group_fields=(
                "scenario",
                "mechanism",
                "requested_rate",
                "indicator",
                "horizon",
                "expert",
            ),
            numeric_fields=("mean_weight",),
        )
        runner.atomic_csv(output / "expert_horizon_weights.csv", horizon_aggregate)

    input_after = neural.sha256_file(train_path)
    if input_before != input_after:
        raise RuntimeError("registered training file changed during Revision-8 analysis")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "primary_variant": "A6_mixed_aug",
        "original_wlcr_in_main_table": True,
        "fair_missingness_training_matrix": {
            "A6_mixed_aug": "15% mixed augmentation",
            "dlinear_aug": "15% mixed augmentation",
            "patchtst_aug": "15% mixed augmentation",
            "clean_trained_counterparts_retained": True,
        },
        "corruption_protocol": {
            "scope": "absolute cell-time timeline before overlapping windows",
            "recent_tail_semantics": "tail of each cell's unique evaluated timeline",
            "request_relative_tail_used": False,
            "corruption_seeds": list(corruption_seeds),
            "unique_cell_time_and_window_exposure_rates_reported": True,
        },
        "bootstrap_estimand": "macro_over_indicator_wape",
        "bootstrap_replicates": args.bootstrap_replicates,
        "inference_batch_size": args.batch_size,
        "shared_corruption_fill_view_across_neural_normalizations": True,
        "clean_replay_abs_tolerance": CLEAN_REPLAY_ABS_TOLERANCE,
        "clean_replay_max_absolute_difference": (
            clean_replay_max_absolute_difference
        ),
        "clean_accuracy": {row["method"]: row for row in clean_rows},
        "paired_bootstrap": bootstrap_rows,
        "missingness_raw_rows": len(missing_rows),
        "missingness_aggregate_rows": len(missing_aggregate),
        "per_indicator_missingness_rows": len(indicator_aggregate),
        "structured_ablation_rows": len(ablation_aggregate),
        "dataset_report": dataset_report,
        "frozen_training_thresholds": thresholds.tolist(),
        "finals_test_opened": False,
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
    }
    runner.atomic_json(output / "summary.json", payload)
    runner.atomic_json(output / "manifest.json", runner.output_manifest(output))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", default="artifacts/revision8/wlcr_sea")
    value.add_argument(
        "--clean-neural-root",
        default="artifacts/paper_neural_baselines_v1/revision6_5seed100/results",
    )
    value.add_argument(
        "--augmented-neural-root",
        default="artifacts/paper_neural_baselines_v1/revision7_mixed/results",
    )
    value.add_argument(
        "--standard-root", default="artifacts/revision6/standard_stat_traffic_only"
    )
    value.add_argument("--original-wlcr", default=str(DEFAULT_ORIGINAL_WLCR))
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    value.add_argument("--gpu-devices", default="0,1,2,3")
    value.add_argument("--batch-size", type=int, default=256)
    value.add_argument("--bootstrap-replicates", type=int, default=5000)
    value.add_argument("--smoke", action="store_true")
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
