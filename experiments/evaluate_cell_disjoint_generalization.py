from __future__ import annotations

"""Revision-7 strict cell-disjoint comparison with all feasible strong baselines.

Each fold excludes its evaluation cells from every fitted weight, prior,
normalization statistic, and booster. Temporal hyperparameters, epochs, and
boosting rounds are frozen from the previously completed temporal inner layer.
All evaluation uses the registered training trace; finals inference traffic is
never opened.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from Model.traffic_window_forecasting import (
    BaselineConfig,
    build_training_backtests,
    read_traffic,
)
from Model.lightgbm_feature_baseline import build_matrix
from experiments import train_neural_baselines as neural
from experiments import train_wlcr_sea as runner
from experiments import wlcr_sea_model as sea
from experiments.lightgbm_experiment_helpers import (
    build_standard_stat_matrix,
    standard_stat_feature_names,
)


SCHEMA_VERSION = 1
FOLDS = 5
MODEL_SEED = 42
BOOTSTRAP_SEED = 42
AUGMENTATION_RATE = 0.15
FINAL_AUGMENTATION_SEED = (
    MODEL_SEED + neural.FINAL_AUGMENTATION_SEED_OFFSET
)
DEFAULT_WLCR_BATCH_SIZE = runner.DEFAULT_BATCH_SIZE
DEFAULT_NEURAL_BATCH_SIZE = neural.DEFAULT_BATCH_SIZE
REPRODUCTION_ROOT = Path("artifacts/reproduction")
DEFAULT_OUTPUT = REPRODUCTION_ROOT / "cell_disjoint_protocol_matched"
DEFAULT_NEURAL_ROOT = REPRODUCTION_ROOT / "neural_baselines/mixed"
TRAFFIC_ONLY_MANIFEST = REPRODUCTION_ROOT / "lightgbm/traffic_only_73d/cache_manifest.json"
STANDARD_STAT_SUMMARY = REPRODUCTION_ROOT / "lightgbm/standard_stat/summary.json"
METHODS = (
    "wlcr_sea",
    "fixed_seasonal_mixture",
    "same_hour_median_7d",
    "original_wlcr_lightgbm",
    "standard_stat_lightgbm",
    "dlinear_aug",
    "patchtst_aug",
)


def refit_training_protocol(
    wlcr_batch_size: int, neural_batch_size: int
) -> dict[str, object]:
    """Return the auditable final-refit settings used in every outer fold."""
    if wlcr_batch_size <= 0 or neural_batch_size <= 0:
        raise ValueError("cell-disjoint batch sizes must be positive")
    batch_size_by_model = {
        "wlcr_sea": int(wlcr_batch_size),
        "dlinear_aug": int(neural_batch_size),
        "patchtst_aug": int(neural_batch_size),
    }
    augmentation_seed_by_model = {
        method: FINAL_AUGMENTATION_SEED
        for method in ("wlcr_sea", "dlinear_aug", "patchtst_aug")
    }
    return {
        "single_model_seed": True,
        "model_seed": MODEL_SEED,
        "batch_size_by_model": batch_size_by_model,
        "augmentation_seed_by_model": augmentation_seed_by_model,
        "augmentation_rate": AUGMENTATION_RATE,
        "matched_final_refit_augmentation_view": (
            len(set(augmentation_seed_by_model.values())) == 1
        ),
        "matches_temporal_batch_size_defaults": (
            wlcr_batch_size == DEFAULT_WLCR_BATCH_SIZE
            and neural_batch_size == DEFAULT_NEURAL_BATCH_SIZE
        ),
        "configuration_and_epoch": (
            "frozen from the seed-42 temporal inner-selection checkpoints; "
            "no fold-specific retuning"
        ),
    }


def validate_worker_refit_report(
    report: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    """Fail if declared refit settings diverge from recorded training reports."""
    if int(report.get("model_seed", -1)) != MODEL_SEED:
        raise ValueError("cell-disjoint worker used an unexpected model seed")
    if report.get("refit_training_protocol") != expected:
        raise ValueError("cell-disjoint worker protocol declaration changed")
    batches = expected["batch_size_by_model"]
    augmentation_seeds = expected["augmentation_seed_by_model"]
    if not isinstance(batches, Mapping) or not isinstance(augmentation_seeds, Mapping):
        raise TypeError("cell-disjoint refit protocol mappings are malformed")
    training = report.get("training_reports")
    if not isinstance(training, Mapping):
        raise TypeError("cell-disjoint worker training reports are missing")
    signature_fields = (
        "mechanism",
        "requested_rate",
        "seed",
        "scope",
        "newly_removed_rate",
        "final_total_missing_rate",
        "unique_cell_time",
        "window_exposure",
    )
    signatures: list[dict[str, object]] = []
    for method in ("wlcr_sea", "dlinear_aug", "patchtst_aug"):
        method_report = training.get(method)
        if not isinstance(method_report, Mapping):
            raise TypeError(f"missing training report for {method}")
        if int(method_report.get("batch_size", -1)) != int(batches[method]):
            raise ValueError(f"{method} batch size diverged from the refit protocol")
        expected_seed = int(augmentation_seeds[method])
        if int(method_report.get("augmentation_seed", -1)) != expected_seed:
            raise ValueError(f"{method} augmentation seed diverged from the protocol")
        augmentation = method_report.get("augmentation")
        if not isinstance(augmentation, Mapping):
            raise TypeError(f"missing augmentation report for {method}")
        if int(augmentation.get("seed", -1)) != expected_seed:
            raise ValueError(f"{method} recorded a different augmentation view")
        signatures.append({field: augmentation[field] for field in signature_fields})
    if not signatures[0] == signatures[1] == signatures[2]:
        raise ValueError("trainable methods did not receive the same augmentation view")


def fold_mapping(cells: Sequence[str]) -> dict[str, int]:
    ordered = sorted(set(str(cell) for cell in cells))
    return {cell: index % FOLDS for index, cell in enumerate(ordered)}


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def prepare_fresh_output(output: Path) -> None:
    """Create an output directory without permitting stale-result reuse."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"cell-disjoint output must be new or empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)


def example_key(cell: str, target_start: datetime) -> tuple[str, str]:
    return str(cell), target_start.isoformat(sep=" ")


def dataset_key(dataset: neural.CachedDataset, index: int) -> tuple[str, str]:
    return (
        str(dataset.cells[index]),
        neural.timestamp_from_hour(int(dataset.target_start_hours[index])).isoformat(
            sep=" "
        ),
    )


def align_examples(
    dataset: neural.CachedDataset,
    indices: np.ndarray,
    examples: Sequence[object],
    label: str,
) -> np.ndarray:
    lookup: dict[tuple[str, str], int] = {}
    for index in np.asarray(indices, dtype=np.int64).tolist():
        key = dataset_key(dataset, int(index))
        if key in lookup:
            raise ValueError(f"duplicate dataset {label} key: {key}")
        lookup[key] = int(index)
    ordered: list[int] = []
    seen: set[tuple[str, str]] = set()
    for example in examples:
        key = example_key(example.window.cell, example.window.target_start)
        if key not in lookup:
            raise ValueError(f"{label} example missing from neural cache: {key}")
        if key in seen:
            raise ValueError(f"duplicate {label} example key: {key}")
        seen.add(key)
        ordered.append(lookup[key])
    if seen != set(lookup):
        raise ValueError(
            f"{label} key-set mismatch: examples={len(seen)} cache={len(lookup)}"
        )
    return np.asarray(ordered, dtype=np.int64)


def split_examples_locally(examples: Sequence[object]) -> dict[str, object]:
    dates = sorted({example.window.target_start.date() for example in examples})
    if len(dates) != 16:
        raise ValueError(f"expected 16 target dates, found {len(dates)}")
    final_dates = set(dates[:9])
    holdout_dates = set(dates[9:])
    final_examples = [
        example
        for example in examples
        if example.window.target_start.date() in final_dates
    ]
    holdout_examples = [
        example
        for example in examples
        if example.window.target_start.date() in holdout_dates
    ]
    if (len(final_examples), len(holdout_examples)) != (6575, 5110):
        raise ValueError(
            f"unexpected local split counts: {(len(final_examples), len(holdout_examples))}"
        )
    return {
        "final_examples": final_examples,
        "holdout_examples": holdout_examples,
    }


def load_frozen_rounds(root: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    original = json.loads((root / TRAFFIC_ONLY_MANIFEST).read_text(encoding="utf-8"))
    standard = json.loads((root / STANDARD_STAT_SUMMARY).read_text(encoding="utf-8"))
    original_rounds = tuple(
        int(value) for value in original["cache_config"]["rounds"]
    )
    standard_rounds = tuple(int(value) for value in standard["selected_rounds"])
    if len(original_rounds) != 4 or len(standard_rounds) != 4:
        raise ValueError("frozen LightGBM rounds must have four targets")
    return original_rounds, standard_rounds


def build_shared_cache(root: Path, cache: Path) -> dict[str, object]:
    train_path = neural.resolve_train_path()
    arrays, dataset_report = neural.build_window_arrays(
        neural.read_training_series(train_path)
    )
    dataset_dir = cache / "dataset"
    neural.write_dataset_cache(dataset_dir, arrays)
    dataset = neural.load_dataset_cache(dataset_dir)
    fit = dataset.indices_for_dates(neural.FIT_DATES)
    inner = dataset.indices_for_dates(neural.INNER_DATES)
    final_indices = np.concatenate((fit, inner))
    holdout_indices = dataset.indices_for_dates(neural.HOLDOUT_DATES)

    examples = build_training_backtests(read_traffic(train_path))
    split = split_examples_locally(examples)
    final_examples = split["final_examples"]
    holdout_examples = split["holdout_examples"]
    final_order = align_examples(dataset, final_indices, final_examples, "final")
    holdout_order = align_examples(dataset, holdout_indices, holdout_examples, "holdout")

    standard_summary = json.loads(
        (root / STANDARD_STAT_SUMMARY).read_text(encoding="utf-8")
    )
    frozen_baseline = standard_summary["seasonal_baseline_selection"]
    baseline = BaselineConfig(
        str(frozen_baseline["name"]),
        tuple(float(value) for value in frozen_baseline["weights"]),
        tuple(float(value) for value in frozen_baseline["scales"]),
    )
    wlcr_final = build_matrix(final_examples, baseline, {}, {})
    wlcr_holdout = build_matrix(holdout_examples, baseline, {}, {})
    if wlcr_final.features.shape[1] != 88 or wlcr_holdout.features.shape[1] != 88:
        raise ValueError("original WLCR full matrix must contain 88 features")
    wlcr_columns = np.asarray([0, *range(16, 88)], dtype=np.int64)
    original_manifest = json.loads(
        (root / TRAFFIC_ONLY_MANIFEST).read_text(encoding="utf-8")
    )
    selected_wlcr_names = list(
        original_manifest["cache_config"]["selected_feature_names"][0]
    )
    if len(selected_wlcr_names) != 73:
        raise ValueError("original WLCR manifest must contain 73 traffic-only features")

    standard_final = build_standard_stat_matrix(final_examples, baseline, {})
    standard_holdout = build_standard_stat_matrix(holdout_examples, baseline, {})
    standard_names = standard_stat_feature_names(final_examples[0], {})
    suffixes = tuple(f"_m{metric}" for metric in range(4))
    standard_columns = np.asarray(
        [
            index
            for index, name in enumerate(standard_names)
            if name == "horizon" or name.endswith(suffixes)
        ],
        dtype=np.int64,
    )
    selected_standard_names = [
        standard_names[int(index)] for index in standard_columns
    ]
    if len(selected_standard_names) != 165:
        raise ValueError("Standard-stat cache must contain 165 traffic-only features")

    matrix_dir = cache / "matrices"
    atomic_npy(
        matrix_dir / "wlcr_final_features.npy",
        np.asarray(wlcr_final.features[:, wlcr_columns], dtype=np.float32),
    )
    atomic_npy(
        matrix_dir / "wlcr_holdout_features.npy",
        np.asarray(wlcr_holdout.features[:, wlcr_columns], dtype=np.float32),
    )
    atomic_npy(
        matrix_dir / "wlcr_final_targets.npy",
        np.asarray(wlcr_final.targets, dtype=np.float32),
    )
    atomic_npy(
        matrix_dir / "standard_final_features.npy",
        np.asarray(standard_final.features[:, standard_columns], dtype=np.float32),
    )
    atomic_npy(
        matrix_dir / "standard_holdout_features.npy",
        np.asarray(standard_holdout.features[:, standard_columns], dtype=np.float32),
    )
    atomic_npy(
        matrix_dir / "standard_final_targets.npy",
        np.asarray(standard_final.targets, dtype=np.float32),
    )
    atomic_npy(matrix_dir / "final_dataset_indices.npy", final_order)
    atomic_npy(matrix_dir / "holdout_dataset_indices.npy", holdout_order)
    atomic_npy(
        matrix_dir / "final_cells.npy",
        np.asarray([example.window.cell for example in final_examples], dtype=str),
    )
    atomic_npy(
        matrix_dir / "holdout_cells.npy",
        np.asarray([example.window.cell for example in holdout_examples], dtype=str),
    )
    original_rounds, standard_rounds = load_frozen_rounds(root)
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_report": dataset_report,
        "final_windows": len(final_examples),
        "holdout_windows": len(holdout_examples),
        "exact_final_key_alignment": True,
        "exact_holdout_key_alignment": True,
        "seasonal_baseline_configuration_frozen": frozen_baseline,
        "original_wlcr_feature_count": len(selected_wlcr_names),
        "standard_stat_feature_count": len(selected_standard_names),
        "original_wlcr_feature_names": selected_wlcr_names,
        "standard_stat_feature_names": selected_standard_names,
        "original_wlcr_rounds": list(original_rounds),
        "standard_stat_rounds": list(standard_rounds),
        "parameter_features_selected": False,
        "weather_features_selected": False,
        "parameter_or_weather_files_opened": False,
        "canonical_input_file": str(train_path.relative_to(root)),
        "calendar_features_selected": False,
        "cell_id_or_coordinate_features_selected": False,
    }
    runner.atomic_json(cache / "cache_report.json", report)
    return report


def load_neural_protocol(
    root: Path, model_name: str
) -> tuple[dict[str, object], int, Path]:
    path = root / "models" / f"{model_name}_seed{MODEL_SEED}.pt"
    payload = torch.load(path, map_location="cpu")
    if payload.get("model") != model_name or int(payload.get("seed")) != MODEL_SEED:
        raise ValueError(f"invalid frozen neural protocol checkpoint: {path}")
    if payload.get("augmentation") != "mixed":
        raise ValueError(f"unseen {model_name} must use the fair mixed augmentation protocol")
    if not np.isclose(float(payload.get("augmentation_rate", -1.0)), AUGMENTATION_RATE):
        raise ValueError(f"unseen {model_name} must use 15% input-only augmentation")
    return dict(payload["config"]), int(payload["selected_epoch"]), path


def worker(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("unseen worker requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cache = Path(args.cache).resolve(strict=True)
    output = Path(args.output).resolve(strict=False)
    source = Path(args.source).resolve(strict=True)
    neural_root = Path(args.neural_root).resolve(strict=True)
    refit_protocol = refit_training_protocol(
        args.wlcr_batch_size, args.neural_batch_size
    )
    if not refit_protocol["matches_temporal_batch_size_defaults"]:
        raise ValueError(
            "cell-disjoint refits must retain WLCR batch 256 and neural batch 128"
        )
    dataset = neural.load_dataset_cache(cache / "dataset")
    matrix_dir = cache / "matrices"
    final_indices = np.load(
        matrix_dir / "final_dataset_indices.npy", allow_pickle=False
    )
    holdout_indices = np.load(
        matrix_dir / "holdout_dataset_indices.npy", allow_pickle=False
    )
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
    train = np.asarray(final_indices[train_window_mask], dtype=np.int64)
    evaluate = np.asarray(holdout_indices[evaluation_window_mask], dtype=np.int64)
    if not len(train) or not len(evaluate):
        raise ValueError(f"empty unseen fold {args.fold}")
    train_cells = set(str(dataset.cells[index]) for index in train)
    evaluation_cells = set(str(dataset.cells[index]) for index in evaluate)
    overlap = train_cells.intersection(evaluation_cells)
    if overlap:
        raise ValueError(f"cell leakage in fold {args.fold}: {sorted(overlap)[:3]}")

    source_payload = torch.load(
        source / "models" / f"{runner.PRIMARY_VARIANT}_seed{MODEL_SEED}.pt",
        map_location="cpu",
    )
    variant = sea.VariantConfig(**source_payload["variant"])
    if variant.name != runner.PRIMARY_VARIANT:
        raise ValueError("frozen temporal SEA checkpoint is not A6_mixed_aug")
    sea_config = dict(source_payload["selected_config"])
    sea_epochs = int(source_payload["selected_epoch"])
    prior = sea.training_prior_log(dataset.targets, dataset.target_masks, train)
    sea_train_tensors, sea_augmentation = runner.make_training_tensors(
        dataset,
        train,
        prior,
        variant,
        seed=FINAL_AUGMENTATION_SEED,
    )
    expert_batch, sea_eval_tensors = runner.make_eval_tensors(dataset, evaluate, prior)
    sea_model, sea_output, sea_training = runner.train_final(
        variant=variant,
        config=sea_config,
        seed=MODEL_SEED,
        epochs=sea_epochs,
        train_tensors=sea_train_tensors,
        holdout_tensors=sea_eval_tensors,
        device=device,
        batch_size=args.wlcr_batch_size,
        include_audit=True,
    )
    fixed_model = sea.WLCRSEA(
        sea.VARIANTS["A0_fixed"], token_dim=16, hidden_dim=32
    ).eval()
    fixed_prediction = runner.predict(
        fixed_model,
        sea_eval_tensors,
        device=torch.device("cpu"),
        batch_size=args.wlcr_batch_size,
    )["prediction"]
    same_hour_prediction = np.asarray(
        sea.prediction_from_log(expert_batch.values[..., 3]), dtype=np.float32
    )

    predictions: dict[str, np.ndarray] = {
        "wlcr_sea": sea_output["prediction"],
        "fixed_seasonal_mixture": fixed_prediction,
        "same_hour_median_7d": same_hour_prediction,
    }
    training_reports: dict[str, object] = {
        "wlcr_sea": {
            "config": sea_config,
            "epochs": sea_epochs,
            "augmentation": sea_augmentation,
            "augmentation_seed": FINAL_AUGMENTATION_SEED,
            "batch_size": args.wlcr_batch_size,
            "training": sea_training,
            "mean_prior_mass": float(np.mean(sea_output["attention"][..., -1])),
        }
    }

    for model_name in ("dlinear", "patchtst"):
        config, epochs, protocol_checkpoint = load_neural_protocol(
            neural_root, model_name
        )
        normalization = neural.compute_normalization(dataset, train)
        extra, augmentation_report = neural.training_augmentation(
            dataset,
            train,
            augmentation="mixed",
            requested_rate=AUGMENTATION_RATE,
            seed=FINAL_AUGMENTATION_SEED,
        )
        train_tensors = neural.prepared_tensors(
            dataset,
            train,
            normalization,
            additional_missing=extra,
        )
        evaluate_inputs = neural.prepared_inputs(dataset, evaluate, normalization)
        model, prediction, report = neural.train_final_model(
            model_name=model_name,
            config=config,
            seed=MODEL_SEED,
            epochs=epochs,
            train_tensors=train_tensors,
            holdout_inputs=evaluate_inputs,
            normalization=normalization,
            device=device,
            batch_size=args.neural_batch_size,
        )
        key = f"{model_name}_aug"
        predictions[key] = prediction
        training_reports[key] = {
            "config": config,
            "epochs": epochs,
            "augmentation": augmentation_report,
            "augmentation_seed": FINAL_AUGMENTATION_SEED,
            "batch_size": args.neural_batch_size,
            "normalization": normalization.__dict__,
            "training": report,
            "frozen_protocol_checkpoint": str(protocol_checkpoint),
        }
        model_path = output / "worker" / f"fold{args.fold}_{model_name}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "fold": args.fold,
                "model": model_name,
                "seed": MODEL_SEED,
                "batch_size": args.neural_batch_size,
                "config": config,
                "selected_epoch": epochs,
                "augmentation": "mixed",
                "augmentation_rate": AUGMENTATION_RATE,
                "augmentation_seed": FINAL_AUGMENTATION_SEED,
                "normalization": normalization.__dict__,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
            },
            model_path,
        )
        del model
        torch.cuda.empty_cache()

    project_root = runner.project_root()
    lgbm_script = project_root / "experiments/evaluate_cell_disjoint_lightgbm_worker.py"
    lgbm_command = [
        sys.executable,
        str(lgbm_script),
        "--cache",
        str(cache),
        "--output",
        str(output),
        "--fold",
        str(args.fold),
        "--physical-device",
        str(args.physical_device),
    ]
    if args.smoke:
        lgbm_command.append("--smoke")
    lgbm_environment = os.environ.copy()
    lgbm_environment["PYTHONPATH"] = os.pathsep.join(
        (str(project_root / ".runtime/lightgbm"), str(project_root))
    )
    lgbm_environment["OMP_NUM_THREADS"] = "4"
    lgbm_environment["MKL_NUM_THREADS"] = "4"
    completed_lgbm = subprocess.run(
        lgbm_command,
        cwd=project_root,
        env=lgbm_environment,
        capture_output=True,
        text=True,
    )
    lgbm_log = output / "logs" / f"fold{args.fold}_lgbm.log"
    lgbm_log.parent.mkdir(parents=True, exist_ok=True)
    lgbm_log.write_text(
        "COMMAND\n"
        + " ".join(lgbm_command)
        + "\n\nSTDOUT\n"
        + completed_lgbm.stdout
        + "\nSTDERR\n"
        + completed_lgbm.stderr,
        encoding="utf-8",
    )
    if completed_lgbm.returncode != 0:
        raise RuntimeError(
            f"fold {args.fold} LightGBM worker failed; see {lgbm_log}"
        )
    lgbm_report = json.loads(
        (output / "worker" / f"fold{args.fold}_lgbm.json").read_text(
            encoding="utf-8"
        )
    )
    with np.load(
        output / str(lgbm_report["prediction_file"]), allow_pickle=False
    ) as lgbm_arrays:
        predictions["original_wlcr_lightgbm"] = np.asarray(
            lgbm_arrays["original_wlcr_lightgbm"], dtype=np.float32
        )
        predictions["standard_stat_lightgbm"] = np.asarray(
            lgbm_arrays["standard_stat_lightgbm"], dtype=np.float32
        )
    training_reports["original_wlcr_lightgbm"] = lgbm_report[
        "original_wlcr_lightgbm"
    ]
    training_reports["standard_stat_lightgbm"] = lgbm_report[
        "standard_stat_lightgbm"
    ]
    training_reports["lightgbm_environment"] = {
        "isolated_numpy2_runtime": True,
        "log": str(lgbm_log.relative_to(output)),
    }

    actual = np.asarray(dataset.targets[evaluate], dtype=np.float32)
    scales = np.asarray(dataset.mase_scales[evaluate], dtype=np.float32)
    cells = np.asarray(dataset.cells[evaluate])
    thresholds = sea.frozen_low_activity_thresholds(
        dataset.targets, dataset.target_masks, train
    )
    metrics: dict[str, object] = {}
    for method in METHODS:
        metrics[method] = {
            "forecast": sea.forecast_metrics(
                actual, predictions[method], scales, cells
            ),
            "threshold_hit_score": sea.threshold_hit_score(
                actual, predictions[method], thresholds
            ),
        }
    worker_dir = output / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = worker_dir / f"fold{args.fold}_predictions.npz"
    np.savez_compressed(
        prediction_path,
        indices=evaluate,
        **predictions,
    )
    sea_model_path = worker_dir / f"fold{args.fold}_wlcr_sea.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "fold": args.fold,
            "seed": MODEL_SEED,
            "batch_size": args.wlcr_batch_size,
            "augmentation_seed": FINAL_AUGMENTATION_SEED,
            "variant": source_payload["variant"],
            "config": sea_config,
            "selected_epoch": sea_epochs,
            "prior_log": prior,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in sea_model.state_dict().items()
            },
        },
        sea_model_path,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "fold": args.fold,
        "physical_gpu": args.physical_device,
        "train_windows": len(train),
        "evaluation_windows": len(evaluate),
        "train_cells": len(train_cells),
        "evaluation_cells": len(evaluation_cells),
        "cell_overlap": 0,
        "all_evaluation_cells_excluded_from_every_fit": True,
        "temporal_configuration_frozen": True,
        "model_seed": MODEL_SEED,
        "refit_training_protocol": refit_protocol,
        "metrics": metrics,
        "training_reports": training_reports,
        "prediction_file": str(prediction_path.relative_to(output)),
        "sea_model_file": str(sea_model_path.relative_to(output)),
        "finals_test_opened": False,
    }
    validate_worker_refit_report(report, refit_protocol)
    runner.atomic_json(worker_dir / f"fold{args.fold}.json", report)
    print(json.dumps({"status": "complete", "fold": args.fold}))
    return 0


def launch_worker(
    fold: int,
    device: int,
    script: Path,
    cache: Path,
    source: Path,
    neural_root: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--fold",
        str(fold),
        "--physical-device",
        str(device),
        "--cache",
        str(cache),
        "--source",
        str(source),
        "--neural-root",
        str(neural_root),
        "--output",
        str(output),
        "--wlcr-batch-size",
        str(args.wlcr_batch_size),
        "--neural-batch-size",
        str(args.neural_batch_size),
    ]
    if args.smoke:
        command.append("--smoke")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["OMP_NUM_THREADS"] = "4"
    environment["MKL_NUM_THREADS"] = "4"
    completed = subprocess.run(
        command,
        cwd=runner.project_root(),
        env=environment,
        capture_output=True,
        text=True,
    )
    log = output / "logs" / f"fold{fold}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "COMMAND\n"
        + " ".join(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    return {
        "fold": fold,
        "device": device,
        "returncode": completed.returncode,
        "log": str(log.relative_to(output)),
    }


def aggregate(
    output: Path,
    dataset: neural.CachedDataset,
    holdout: np.ndarray,
    bootstrap_replicates: int,
) -> dict[str, object]:
    shape = np.asarray(dataset.targets[holdout]).shape
    predictions = {
        method: np.full(shape, np.nan, dtype=np.float32) for method in METHODS
    }
    position = {int(index): offset for offset, index in enumerate(holdout.tolist())}
    fold_rows: list[dict[str, object]] = []
    prior_mass: list[float] = []
    refit_protocol: dict[str, object] | None = None
    for fold in range(FOLDS):
        report = json.loads(
            (output / "worker" / f"fold{fold}.json").read_text(encoding="utf-8")
        )
        if int(report["cell_overlap"]) != 0:
            raise ValueError(f"fold {fold} reports cell overlap")
        reported_refit_protocol = dict(report["refit_training_protocol"])
        validate_worker_refit_report(report, reported_refit_protocol)
        if refit_protocol is None:
            refit_protocol = reported_refit_protocol
        elif reported_refit_protocol != refit_protocol:
            raise ValueError("cell-disjoint folds used inconsistent refit protocols")
        prior_mass.append(
            float(report["training_reports"]["wlcr_sea"]["mean_prior_mass"])
        )
        row: dict[str, object] = {
            "fold": fold,
            "train_cells": report["train_cells"],
            "evaluation_cells": report["evaluation_cells"],
            "evaluation_windows": report["evaluation_windows"],
            "cell_overlap": report["cell_overlap"],
            "mean_prior_mass": prior_mass[-1],
        }
        for method in METHODS:
            row[f"{method}_macro_wape"] = report["metrics"][method]["forecast"][
                "macro_indicator"
            ]["wape"]
        fold_rows.append(row)
        with np.load(
            output / str(report["prediction_file"]), allow_pickle=False
        ) as arrays:
            indices = arrays["indices"].tolist()
            for local, index in enumerate(indices):
                offset = position[int(index)]
                for method in METHODS:
                    predictions[method][offset] = arrays[method][local]
    for method, values in predictions.items():
        if np.any(~np.isfinite(values)):
            raise ValueError(f"unseen folds did not cover every {method} prediction")
        atomic_npy(output / f"unseen_{method}.npy", values)

    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    scales = np.asarray(dataset.mase_scales[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    metrics = {
        method: sea.forecast_metrics(actual, prediction, scales, cells)
        for method, prediction in predictions.items()
    }
    comparisons: dict[str, object] = {}
    for baseline in METHODS[1:]:
        result = sea.cell_cluster_bootstrap_wape_delta(
            actual,
            predictions["wlcr_sea"],
            predictions[baseline],
            cells,
            replicates=bootstrap_replicates,
            seed=BOOTSTRAP_SEED,
        )
        direct = float(
            metrics["wlcr_sea"]["macro_indicator"]["wape"]
            - metrics[baseline]["macro_indicator"]["wape"]
        )
        if not np.isclose(
            float(result["delta_proposed_minus_baseline"]), direct, atol=1e-12
        ):
            raise RuntimeError(f"unseen bootstrap point mismatch for {baseline}")
        comparisons[f"wlcr_sea_minus_{baseline}"] = result
    if refit_protocol is None:
        raise RuntimeError("cell-disjoint aggregation found no fold protocols")
    cluster_counts = {
        int(result["clusters"]) for result in comparisons.values()
    }
    if len(cluster_counts) != 1:
        raise RuntimeError("cell-cluster comparisons used inconsistent cluster counts")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": (
            "five deterministic cell-disjoint folds; evaluation cells excluded from "
            "all fitted priors, normalizations, neural weights, SEA weights, and boosters; "
            "temporal configurations and epochs frozen; temporal refit batch sizes "
            "and final-refit augmentation view matched; one fixed model seed"
        ),
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "methods": list(METHODS),
        "refit_training_protocol": refit_protocol,
        "uncertainty_scope": (
            "descriptive paired cell-cluster bootstrap conditional on the frozen "
            "configuration, epoch, model seed, and augmentation view; it excludes "
            "optimization-seed, tuning, temporal, and regional uncertainty"
        ),
        "statistical_unit": {
            "evaluation_cells": int(len(set(cells.tolist()))),
            "evaluable_cell_clusters": cluster_counts.pop(),
            "evaluation_windows": int(len(holdout)),
        },
        "folds": fold_rows,
        "all_fold_cell_overlaps_zero": all(
            int(row["cell_overlap"]) == 0 for row in fold_rows
        ),
        "metrics": metrics,
        "paired_cell_cluster_bootstrap": comparisons,
        "unseen_mean_prior_mass": float(np.mean(prior_mass)),
        "unseen_prior_mass_fold_sd": float(np.std(prior_mass, ddof=1)),
        "finals_test_opened": False,
    }
    runner.atomic_csv(output / "fold_results.csv", fold_rows)
    runner.atomic_json(output / "summary.json", payload)
    return payload


def run_master(args: argparse.Namespace) -> int:
    root = runner.project_root()
    source = runner.resolve_output(args.source).resolve(strict=True)
    neural_root = Path(args.neural_root)
    if not neural_root.is_absolute():
        neural_root = root / neural_root
    neural_root = neural_root.resolve(strict=True)
    if not neural_root.is_relative_to(root):
        raise ValueError("neural protocol root must remain inside the project")
    output = runner.resolve_output(args.output)
    allowed = (root / REPRODUCTION_ROOT).resolve(strict=False)
    if not output.is_relative_to(allowed):
        raise ValueError("cell-disjoint output must remain under artifacts/reproduction")
    prepare_fresh_output(output)
    train_path = neural.resolve_train_path()
    before = neural.sha256_file(train_path)
    devices = [int(item) for item in args.gpu_devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("at least one GPU is required")
    refit_protocol = refit_training_protocol(
        args.wlcr_batch_size, args.neural_batch_size
    )
    if not refit_protocol["matches_temporal_batch_size_defaults"]:
        raise ValueError(
            "cell-disjoint refits must retain WLCR batch 256 and neural batch 128"
        )
    with tempfile.TemporaryDirectory(prefix="revision7-unseen-") as temporary:
        cache = Path(temporary)
        cache_report = build_shared_cache(root, cache)
        dataset = neural.load_dataset_cache(cache / "dataset")
        holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
        active_folds = range(1) if args.smoke else range(FOLDS)
        fold_list = list(active_folds)
        groups = [
            (device, fold_list[offset :: len(devices)])
            for offset, device in enumerate(devices)
            if fold_list[offset :: len(devices)]
        ]

        def run_group(group: tuple[int, list[int]]) -> list[dict[str, object]]:
            device, folds = group
            return [
                launch_worker(
                    fold,
                    device,
                    Path(__file__).resolve(),
                    cache,
                    source,
                    neural_root,
                    output,
                    args,
                )
                for fold in folds
            ]

        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            nested = list(executor.map(run_group, groups))
        statuses = [item for group in nested for item in group]
        runner.atomic_json(output / "worker_status.json", statuses)
        failures = [item for item in statuses if int(item["returncode"]) != 0]
        if failures:
            return 1
        if args.smoke:
            runner.atomic_json(
                output / "smoke_summary.json",
                json.loads(
                    (output / "worker/fold0.json").read_text(encoding="utf-8")
                ),
            )
        else:
            aggregate(output, dataset, holdout, args.bootstrap_replicates)
    after = neural.sha256_file(train_path)
    if before != after:
        raise RuntimeError("registered training data changed during unseen experiment")
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source.relative_to(root)),
        "neural_protocol_root": str(neural_root.relative_to(root)),
        "folds": FOLDS,
        "model_seed": MODEL_SEED,
        "single_model_seed": True,
        "refit_training_protocol": refit_protocol,
        "augmentation": "mixed input-history only",
        "augmentation_rate": AUGMENTATION_RATE,
        "gpu_devices": devices,
        "bootstrap_replicates": args.bootstrap_replicates,
        "smoke": args.smoke,
        "shared_cache_report": cache_report,
        "canonical_input_files_opened": [str(train_path.relative_to(root))],
        "parameter_or_weather_files_opened": False,
        "frozen_configuration_and_epoch_sources": {
            "wlcr_sea": str(
                (
                    source
                    / "models"
                    / f"{runner.PRIMARY_VARIANT}_seed{MODEL_SEED}.pt"
                ).relative_to(root)
            ),
            "dlinear_aug": str(
                (
                    neural_root / "models" / f"dlinear_seed{MODEL_SEED}.pt"
                ).relative_to(root)
            ),
            "patchtst_aug": str(
                (
                    neural_root / "models" / f"patchtst_seed{MODEL_SEED}.pt"
                ).relative_to(root)
            ),
        },
        "registered_train_sha256_before": before,
        "registered_train_sha256_after": after,
        "finals_test_opened": False,
    }
    runner.atomic_json(output / "protocol.json", protocol)
    runner.atomic_json(output / "manifest.json", runner.output_manifest(output))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", default=str(runner.DEFAULT_OUTPUT))
    value.add_argument("--neural-root", default=str(DEFAULT_NEURAL_ROOT))
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    value.add_argument("--gpu-devices", default="0,1,2,3")
    value.add_argument(
        "--wlcr-batch-size", type=int, default=DEFAULT_WLCR_BATCH_SIZE
    )
    value.add_argument(
        "--neural-batch-size", type=int, default=DEFAULT_NEURAL_BATCH_SIZE
    )
    value.add_argument("--bootstrap-replicates", type=int, default=5000)
    value.add_argument("--smoke", action="store_true")
    value.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--fold", type=int, help=argparse.SUPPRESS)
    value.add_argument("--physical-device", type=int, help=argparse.SUPPRESS)
    value.add_argument("--cache", help=argparse.SUPPRESS)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.worker:
        try:
            return worker(args)
        except Exception:
            traceback.print_exc()
            return 1
    return run_master(args)


if __name__ == "__main__":
    raise SystemExit(main())
