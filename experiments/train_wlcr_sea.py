from __future__ import annotations

"""Four-GPU WLCR-SEA reproduction on registered training data.

The August holdout has already informed prior manuscript revisions. Therefore
this script labels every result exploratory and does not treat the holdout as a
new confirmatory test. Finals test traffic is never opened.
"""

import argparse
import csv
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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments import train_neural_baselines as neural
from experiments import wlcr_sea_model as sea

SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "wlcr_sea_paper_reproduction_v1"
OUTPUT_ROOT = Path("artifacts/reproduction")
DEFAULT_OUTPUT = OUTPUT_ROOT / "wlcr_sea"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_VARIANTS = (
    "A0_global_static",
    "A0_horizon_indicator",
    "A1_softmax",
    "A2_entmax",
    "A3_hard_mask",
    "A4_reliability",
    "A5_residual",
    "A6_mixed_aug",
)
PRIMARY_VARIANT = "A6_mixed_aug"
AUGMENTATION_RATE = 0.15
DEFAULT_BATCH_SIZE = 256
ROBUSTNESS_RATES = (0.0, 0.10, 0.20, 0.30, 0.50)
ROBUSTNESS_MECHANISMS = ("mcar", "block", "recent_tail", "asynchronous")
CORRUPTION_SEEDS = (42, 43, 44, 45, 46)
MODEL_CANDIDATES = (
    {
        "name": "d16_h32_lr1e3_delta025",
        "token_dim": 16,
        "hidden_dim": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "residual_bound": 0.25,
    },
    {
        "name": "d32_h64_lr5e4_delta050",
        "token_dim": 32,
        "hidden_dim": 64,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "residual_bound": 0.50,
    },
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_output(text: str) -> Path:
    root = project_root()
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    allowed = (root / OUTPUT_ROOT).resolve(strict=False)
    if path != allowed and not path.is_relative_to(allowed):
        raise ValueError(f"outputs must remain under {OUTPUT_ROOT}")
    return path


def parse_list(text: str, *, integer: bool = False) -> list[object]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("list argument cannot be empty")
    parsed: list[object] = [int(item) for item in values] if integer else values
    if len(parsed) != len(set(parsed)):
        raise ValueError("list arguments cannot contain duplicates")
    return parsed


def set_seed(seed: int) -> None:
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


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def batch_to_tensors(batch: sea.ExpertBatch) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(np.asarray(batch.values, dtype=np.float32)),
        torch.from_numpy(np.asarray(batch.availability, dtype=np.uint8)),
        torch.from_numpy(np.asarray(batch.reliability, dtype=np.float32)),
        torch.from_numpy(np.asarray(batch.context, dtype=np.float32)),
    )


def make_training_tensors(
    dataset: neural.CachedDataset,
    indices: np.ndarray,
    prior: np.ndarray,
    variant: sea.VariantConfig,
    *,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    clean = sea.build_expert_batch(
        np.asarray(dataset.x_values[indices]),
        np.asarray(dataset.x_masks[indices]),
        prior,
    )
    mechanism = "none" if variant.augmentation == "clean" else variant.augmentation
    requested_rate = 0.0 if variant.augmentation == "clean" else AUGMENTATION_RATE
    extra = sea.global_corruption_mask(
        np.asarray(dataset.cells[indices]),
        np.asarray(dataset.history_end_hours[indices]),
        mechanism=mechanism,
        requested_rate=requested_rate,
        seed=seed,
    )
    augmented = sea.build_expert_batch(
        np.asarray(dataset.x_values[indices]),
        np.asarray(dataset.x_masks[indices]),
        prior,
        additional_missing=extra,
    )
    targets = np.asarray(dataset.targets[indices], dtype=np.float32)
    target_masks = np.asarray(dataset.target_masks[indices], dtype=np.float32)
    target_log = np.log1p(np.where(target_masks > 0.0, targets, 0.0)).astype(np.float32)
    tensors = (
        *batch_to_tensors(augmented),
        torch.from_numpy(target_log),
        torch.from_numpy(target_masks),
        *batch_to_tensors(clean),
    )
    stats = sea.corruption_statistics(
        np.asarray(dataset.x_masks[indices]),
        extra,
        cells=np.asarray(dataset.cells[indices]),
        history_end_hours=np.asarray(dataset.history_end_hours[indices]),
        mechanism=mechanism,
        requested_rate=requested_rate,
        seed=seed,
    )
    stats["labels_or_target_masks_modified"] = False
    return tensors, stats


def make_eval_tensors(
    dataset: neural.CachedDataset,
    indices: np.ndarray,
    prior: np.ndarray,
    *,
    additional_missing: np.ndarray | None = None,
) -> tuple[sea.ExpertBatch, tuple[torch.Tensor, ...]]:
    batch = sea.build_expert_batch(
        np.asarray(dataset.x_values[indices]),
        np.asarray(dataset.x_masks[indices]),
        prior,
        additional_missing=additional_missing,
    )
    return batch, batch_to_tensors(batch)


def make_loader(tensors: tuple[torch.Tensor, ...], batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def train_epoch(
    model: sea.WLCRSEA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    variant: sea.VariantConfig,
) -> dict[str, float]:
    model.train()
    totals = {key: 0.0 for key in ("total", "prediction", "residual", "reliability", "consistency")}
    batches = 0
    for fields in loader:
        (
            values,
            availability,
            reliability,
            context,
            target_log,
            target_mask,
            clean_values,
            clean_availability,
            clean_reliability,
            clean_context,
        ) = [field.to(device, non_blocking=True) for field in fields]
        optimizer.zero_grad(set_to_none=True)
        augmented_output = model(values, availability.bool(), reliability, context)
        clean_output = None
        if variant.consistency_weight > 0.0:
            clean_output = model(
                clean_values,
                clean_availability.bool(),
                clean_reliability,
                clean_context,
            )
            loss, pieces = sea.sea_loss(
                clean_output,
                target_log,
                target_mask,
                clean_reliability,
                consistency_output=augmented_output,
                consistency_weight=variant.consistency_weight,
                reliability_weight=1e-3 if variant.reliability else 0.0,
            )
        else:
            loss, pieces = sea.sea_loss(
                augmented_output,
                target_log,
                target_mask,
                reliability,
                reliability_weight=1e-3 if variant.reliability else 0.0,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss became non-finite")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"non-finite gradient in {name}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        totals["total"] += float(loss.detach().cpu())
        for key, value in pieces.items():
            totals[key] += value
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def predict(
    model: sea.WLCRSEA,
    tensors: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
    batch_size: int,
    include_audit: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    output: dict[str, list[np.ndarray]] = {
        "prediction_log": [],
        "baseline_log": [],
        "residual": [],
        "entropy": [],
        "attention": [],
    }
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    with torch.no_grad():
        for fields in loader:
            values, availability, reliability, context = [
                field.to(device, non_blocking=True) for field in fields
            ]
            batch_output = model(values, availability.bool(), reliability, context)
            for key in output:
                if key == "attention" and not include_audit:
                    continue
                if key in {"baseline_log", "residual", "entropy"} and not include_audit:
                    continue
                output[key].append(batch_output[key].detach().cpu().numpy())
    result = {
        key: np.concatenate(parts, axis=0)
        for key, parts in output.items()
        if parts
    }
    if not np.all(np.isfinite(result["prediction_log"])):
        raise FloatingPointError("model produced non-finite log predictions")
    result["prediction"] = np.asarray(
        sea.prediction_from_log(result["prediction_log"]), dtype=np.float32
    )
    return result


def variant_candidates(variant: sea.VariantConfig, smoke: bool) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for raw in MODEL_CANDIDATES[:1 if smoke else len(MODEL_CANDIDATES)]:
        config = dict(raw)
        if variant.residual_bound == 0.0:
            config["residual_bound"] = 0.0
        else:
            config["residual_bound"] = min(
                float(config["residual_bound"]), variant.residual_bound
            )
        candidates.append(config)
    return candidates


def model_from_config(variant: sea.VariantConfig, config: Mapping[str, object]) -> sea.WLCRSEA:
    return sea.WLCRSEA(
        variant,
        token_dim=int(config["token_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        residual_bound=float(config["residual_bound"]),
    )


def select_configuration(
    *,
    variant: sea.VariantConfig,
    seed: int,
    fit_tensors: tuple[torch.Tensor, ...],
    inner_tensors: tuple[torch.Tensor, ...],
    inner_actual: np.ndarray,
    inner_scales: np.ndarray,
    inner_cells: np.ndarray,
    frozen_thresholds: np.ndarray,
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
    smoke: bool,
) -> tuple[dict[str, object], int, list[dict[str, object]], float]:
    reports: list[dict[str, object]] = []
    best_key: tuple[float, float, int] | None = None
    best_config: dict[str, object] | None = None
    best_epoch = 0
    started = time.perf_counter()
    for candidate_index, config in enumerate(variant_candidates(variant, smoke)):
        set_seed(seed + candidate_index * 10_000)
        model = model_from_config(variant, config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        loader = make_loader(fit_tensors, batch_size, seed + candidate_index)
        epochs: list[dict[str, object]] = []
        local_best: tuple[float, float, int] | None = None
        local_epoch = 0
        stale = 0
        for epoch in range(1, max_epochs + 1):
            epoch_started = time.perf_counter()
            losses = train_epoch(model, loader, optimizer, device, variant)
            inner_output = predict(
                model,
                inner_tensors,
                device=device,
                batch_size=batch_size,
            )
            metrics = sea.forecast_metrics(
                inner_actual, inner_output["prediction"], inner_scales, inner_cells
            )
            ths = sea.threshold_hit_score(
                inner_actual, inner_output["prediction"], frozen_thresholds
            )
            macro_wape = float(metrics["macro_indicator"]["wape"])
            key = (-macro_wape, float(ths["score"]), -epoch)
            epochs.append(
                {
                    "epoch": epoch,
                    "train_loss": losses,
                    "inner_macro_indicator_wape": macro_wape,
                    "inner_pooled_wape": metrics["pooled_wape"],
                    "inner_threshold_hit_score": ths["score"],
                    "seconds": time.perf_counter() - epoch_started,
                }
            )
            if local_best is None or key > local_best:
                local_best = key
                local_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if local_best is None:
            raise RuntimeError(f"no checkpoint for {variant.name}/{config['name']}")
        reports.append(
            {
                "candidate_index": candidate_index,
                "config": config,
                "parameter_count": count_parameters(model),
                "best_epoch": local_epoch,
                "best_inner_macro_indicator_wape": -local_best[0],
                "best_inner_threshold_hit_score": local_best[1],
                "epochs_run": len(epochs),
                "epoch_reports": epochs,
            }
        )
        global_key = (local_best[0], local_best[1], -candidate_index)
        if best_key is None or global_key > best_key:
            best_key = global_key
            best_config = config
            best_epoch = local_epoch
        del model
        torch.cuda.empty_cache()
    if best_config is None:
        raise RuntimeError(f"no configuration selected for {variant.name}")
    return best_config, best_epoch, reports, time.perf_counter() - started


def train_final(
    *,
    variant: sea.VariantConfig,
    config: Mapping[str, object],
    seed: int,
    epochs: int,
    train_tensors: tuple[torch.Tensor, ...],
    holdout_tensors: tuple[torch.Tensor, ...],
    device: torch.device,
    batch_size: int,
    include_audit: bool,
) -> tuple[sea.WLCRSEA, dict[str, np.ndarray], dict[str, object]]:
    set_seed(seed + 1_000_000)
    model = model_from_config(variant, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loader = make_loader(train_tensors, batch_size, seed + 1_000_000)
    curves: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        losses = train_epoch(model, loader, optimizer, device, variant)
        curves.append({"epoch": epoch, **losses})
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    warmup = tuple(tensor[: min(batch_size, len(tensor))] for tensor in holdout_tensors)
    _ = predict(model, warmup, device=device, batch_size=batch_size)
    torch.cuda.synchronize()
    inference_start = time.perf_counter()
    outputs = predict(
        model,
        holdout_tensors,
        device=device,
        batch_size=batch_size,
        include_audit=include_audit,
    )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    return model, outputs, {
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_window": 1000.0 * inference_seconds / len(holdout_tensors[0]),
        "epoch_curve": curves,
        "parameter_count": count_parameters(model),
        "max_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def run_worker(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker requires one CUDA device through CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    dataset = neural.load_dataset_cache(Path(args.dataset_cache).resolve(strict=True))
    fit = dataset.indices_for_dates(neural.FIT_DATES)
    inner = dataset.indices_for_dates(neural.INNER_DATES)
    holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
    if args.smoke:
        fit, inner, holdout = fit[:256], inner[:128], holdout[:128]
    variant = sea.VARIANTS[args.variant]
    fit_prior = sea.training_prior_log(dataset.targets, dataset.target_masks, fit)
    fit_tensors, fit_corruption = make_training_tensors(
        dataset, fit, fit_prior, variant, seed=args.seed
    )
    _, inner_tensors = make_eval_tensors(dataset, inner, fit_prior)
    fit_thresholds = sea.frozen_low_activity_thresholds(
        dataset.targets, dataset.target_masks, fit
    )
    effective_max_epochs = 2 if args.smoke else args.max_epochs
    effective_patience = 1 if args.smoke else args.patience
    selected_config, selected_epoch, candidate_reports, selection_seconds = select_configuration(
        variant=variant,
        seed=args.seed,
        fit_tensors=fit_tensors,
        inner_tensors=inner_tensors,
        inner_actual=np.asarray(dataset.targets[inner], dtype=np.float32),
        inner_scales=np.asarray(dataset.mase_scales[inner], dtype=np.float32),
        inner_cells=np.asarray(dataset.cells[inner]),
        frozen_thresholds=fit_thresholds,
        device=device,
        max_epochs=effective_max_epochs,
        patience=effective_patience,
        batch_size=args.batch_size,
        smoke=args.smoke,
    )
    final_train = np.concatenate((fit, inner))
    final_prior = sea.training_prior_log(dataset.targets, dataset.target_masks, final_train)
    final_tensors, final_corruption = make_training_tensors(
        dataset,
        final_train,
        final_prior,
        variant,
        seed=args.seed + neural.FINAL_AUGMENTATION_SEED_OFFSET,
    )
    holdout_batch, holdout_tensors = make_eval_tensors(dataset, holdout, final_prior)
    model, outputs, final_report = train_final(
        variant=variant,
        config=selected_config,
        seed=args.seed,
        epochs=selected_epoch,
        train_tensors=final_tensors,
        holdout_tensors=holdout_tensors,
        device=device,
        batch_size=args.batch_size,
        include_audit=args.seed == 42,
    )
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    scales = np.asarray(dataset.mase_scales[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    frozen_thresholds = sea.frozen_low_activity_thresholds(
        dataset.targets, dataset.target_masks, final_train
    )
    metrics = sea.forecast_metrics(actual, outputs["prediction"], scales, cells)
    ths = sea.threshold_hit_score(actual, outputs["prediction"], frozen_thresholds)
    output = resolve_output(args.output)
    model_path = output / "models" / f"{variant.name}_seed{args.seed}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_version": EXPERIMENT_VERSION,
            "variant": asdict(variant),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "fit_augmentation_seed": args.seed,
            "final_augmentation_seed": (
                args.seed + neural.FINAL_AUGMENTATION_SEED_OFFSET
            ),
            "selected_config": selected_config,
            "selected_epoch": selected_epoch,
            "prior_log": final_prior,
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(model_path)
    prediction_path = output / "worker_predictions" / f"{variant.name}_seed{args.seed}.npy"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path, outputs["prediction"], allow_pickle=False)
    audit_path = ""
    if args.seed == 42:
        audit_file = output / "worker_audit" / f"{variant.name}_seed42.npz"
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            audit_file,
            attention=outputs["attention"].astype(np.float16),
            entropy=outputs["entropy"].astype(np.float16),
            baseline_log=outputs["baseline_log"].astype(np.float32),
            residual=outputs["residual"].astype(np.float32),
            availability=holdout_batch.availability.astype(np.uint8),
            reliability=holdout_batch.reliability.astype(np.float16),
        )
        audit_path = str(audit_file.relative_to(output))
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "variant": variant.name,
        "variant_config": asdict(variant),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "fit_augmentation_seed": args.seed,
        "final_augmentation_seed": (
            args.seed + neural.FINAL_AUGMENTATION_SEED_OFFSET
        ),
        "physical_gpu": args.physical_device,
        "visible_gpu_name": torch.cuda.get_device_name(0),
        "fit_windows": len(fit),
        "inner_windows": len(inner),
        "final_train_windows": len(final_train),
        "holdout_windows": len(holdout),
        "fit_corruption": fit_corruption,
        "final_train_corruption": final_corruption,
        "candidate_reports": candidate_reports,
        "selection_seconds": selection_seconds,
        "selected_config": selected_config,
        "selected_epoch": selected_epoch,
        "final_training": final_report,
        "holdout_metrics": metrics,
        "holdout_threshold_hit_score": ths,
        "availability_consistency_max_unavailable_weight": (
            float(np.max(outputs["attention"][~holdout_batch.availability]))
            if args.seed == 42 and variant.hard_mask and np.any(~holdout_batch.availability)
            else None
        ),
        "model_file": str(model_path.relative_to(output)),
        "model_size_bytes": model_path.stat().st_size,
        "model_sha256": sha256_file(model_path),
        "prediction_file": str(prediction_path.relative_to(output)),
        "prediction_sha256": sha256_file(prediction_path),
        "audit_file": audit_path,
        "smoke": args.smoke,
    }
    report_path = output / "job_reports" / f"{variant.name}_seed{args.seed}.json"
    atomic_json(report_path, report)
    print(json.dumps({"status": "complete", "report": str(report_path.relative_to(output))}))
    return 0


def gpu_snapshot() -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
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


def launch_job(
    *,
    script: Path,
    dataset_cache: Path,
    output: Path,
    variant: str,
    seed: int,
    device: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--dataset-cache",
        str(dataset_cache),
        "--output",
        str(output),
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--physical-device",
        str(device),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.smoke:
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
    log_path = output / "logs" / f"{variant}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND\n" + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "variant": variant,
            "seed": seed,
            "physical_gpu": device,
            "returncode": completed.returncode,
            "seconds": time.perf_counter() - started,
            "log_file": str(log_path.relative_to(output)),
        }
    report_path = output / "job_reports" / f"{variant}_seed{seed}.json"
    return {
        "status": "complete",
        "variant": variant,
        "seed": seed,
        "physical_gpu": device,
        "seconds": time.perf_counter() - started,
        "log_file": str(log_path.relative_to(output)),
        "report": json.loads(report_path.read_text(encoding="utf-8")),
    }


def run_queue(
    device: int,
    jobs: Sequence[tuple[str, int]],
    script: Path,
    cache: Path,
    output: Path,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    return [
        launch_job(
            script=script,
            dataset_cache=cache,
            output=output,
            variant=variant,
            seed=seed,
            device=device,
            args=args,
        )
        for variant, seed in jobs
    ]


def numeric_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def deterministic_baselines(
    dataset: neural.CachedDataset, holdout: np.ndarray, final_train: np.ndarray, output: Path
) -> tuple[dict[str, object], dict[str, np.ndarray], sea.ExpertBatch]:
    prior = sea.training_prior_log(dataset.targets, dataset.target_masks, final_train)
    batch, tensors = make_eval_tensors(dataset, holdout, prior)
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    scales = np.asarray(dataset.mase_scales[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    thresholds = sea.frozen_low_activity_thresholds(dataset.targets, dataset.target_masks, final_train)
    baseline_dir = output / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    asset_path = baseline_dir / "fixed_seasonal_assets.npz"
    temporary_asset = asset_path.with_name(f".{asset_path.name}.{os.getpid()}.tmp")
    with temporary_asset.open("wb") as handle:
        np.savez_compressed(
            handle,
            fixed_weights=np.asarray(sea.FIXED_SEASONAL_WEIGHTS, dtype=np.float32),
            prior_log=prior,
            trend_clip=np.asarray([sea.TREND_CLIP], dtype=np.float32),
        )
    temporary_asset.replace(asset_path)
    methods: dict[str, np.ndarray] = {
        "last_day": np.asarray(sea.prediction_from_log(batch.values[..., 0])),
        "last_week": np.asarray(sea.prediction_from_log(batch.values[..., 1])),
        "same_hour_median_7d": np.asarray(sea.prediction_from_log(batch.values[..., 3])),
    }
    fixed_model = sea.WLCRSEA(sea.VARIANTS["A0_fixed"], token_dim=16, hidden_dim=32)
    fixed_output = predict(fixed_model, tensors, device=torch.device("cpu"), batch_size=512)
    methods["A0_fixed"] = fixed_output["prediction"]
    summary: dict[str, object] = {}
    for name, prediction in methods.items():
        summary[name] = {
            "metrics": sea.forecast_metrics(actual, prediction, scales, cells),
            "threshold_hit_score": sea.threshold_hit_score(actual, prediction, thresholds),
        }
        np.save(baseline_dir / f"{name}.npy", prediction, allow_pickle=False)
    summary["A0_fixed"]["trainable_parameters"] = 0
    summary["A0_fixed"]["frozen_scalar_assets"] = int(
        len(sea.FIXED_SEASONAL_WEIGHTS) + prior.size + 1
    )
    summary["A0_fixed"]["serialized_asset_file"] = str(
        asset_path.relative_to(output)
    )
    summary["A0_fixed"]["serialized_asset_size_bytes"] = int(asset_path.stat().st_size)
    atomic_json(output / "deterministic_baselines.json", summary)
    return summary, methods, batch


def aggregate_jobs(
    output: Path,
    dataset: neural.CachedDataset,
    holdout: np.ndarray,
    final_train: np.ndarray,
    variants: Sequence[str],
    seeds: Sequence[int],
    results: Sequence[Mapping[str, object]],
    baseline_predictions: Mapping[str, np.ndarray],
) -> dict[str, object]:
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    scales = np.asarray(dataset.mase_scales[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    thresholds = sea.frozen_low_activity_thresholds(dataset.targets, dataset.target_masks, final_train)
    successful = [item for item in results if item["status"] == "complete"]
    failures = [dict(item) for item in results if item["status"] != "complete"]
    rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    ensemble_predictions: dict[str, np.ndarray] = {}
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        jobs = sorted(
            [item for item in successful if item["variant"] == variant],
            key=lambda item: int(item["seed"]),
        )
        if not jobs:
            summaries[variant] = {"status": "failed"}
            continue
        predictions: dict[int, np.ndarray] = {}
        reports: list[Mapping[str, object]] = []
        for item in jobs:
            seed = int(item["seed"])
            report = item["report"]
            prediction = np.load(output / str(report["prediction_file"]), allow_pickle=False)
            predictions[seed] = prediction
            reports.append(report)
            metric = report["holdout_metrics"]
            rows.append(
                {
                    "variant": variant,
                    "run": f"seed_{seed}",
                    "seed": seed,
                    "macro_wape": metric["macro_indicator"]["wape"],
                    "pooled_wape": metric["pooled_wape"],
                    "macro_cell_wape": metric["macro_cell_wape"],
                    "median_cell_wape": metric["median_cell_wape"],
                    "mase": metric["macro_indicator"]["mase"],
                    "smape": metric["macro_indicator"]["smape"],
                    "threshold_hit_score": report["holdout_threshold_hit_score"]["score"],
                }
            )
        ensemble = np.mean(np.stack(list(predictions.values()), axis=0), axis=0)
        ensemble_predictions[variant] = ensemble.astype(np.float32)
        np.save(prediction_dir / f"{variant}_ensemble.npy", ensemble, allow_pickle=False)
        metrics = sea.forecast_metrics(actual, ensemble, scales, cells)
        ths = sea.threshold_hit_score(actual, ensemble, thresholds)
        rows.append(
            {
                "variant": variant,
                "run": "five_seed_ensemble",
                "seed": "",
                "macro_wape": metrics["macro_indicator"]["wape"],
                "pooled_wape": metrics["pooled_wape"],
                "macro_cell_wape": metrics["macro_cell_wape"],
                "median_cell_wape": metrics["median_cell_wape"],
                "mase": metrics["macro_indicator"]["mase"],
                "smape": metrics["macro_indicator"]["smape"],
                "threshold_hit_score": ths["score"],
            }
        )
        for horizon, value in enumerate(metrics["per_horizon_wape"], start=1):
            horizon_rows.append(
                {"variant": variant, "run": "five_seed_ensemble", "horizon": horizon, "wape": value}
            )
        per_seed_macro = [float(report["holdout_metrics"]["macro_indicator"]["wape"]) for report in reports]
        summaries[variant] = {
            "status": "complete" if len(predictions) == len(seeds) else "partial",
            "successful_seeds": sorted(predictions),
            "macro_wape_across_seeds": numeric_summary(per_seed_macro),
            "ensemble_metrics": metrics,
            "ensemble_threshold_hit_score": ths,
            "parameter_count": numeric_summary([float(report["final_training"]["parameter_count"]) for report in reports]),
            "selected_epochs": {str(report["seed"]): report["selected_epoch"] for report in reports},
            "training_seconds": numeric_summary([float(report["final_training"]["training_seconds"]) for report in reports]),
            "inference_ms_per_window": numeric_summary([float(report["final_training"]["inference_ms_per_window"]) for report in reports]),
            "model_size_bytes": numeric_summary([float(report["model_size_bytes"]) for report in reports]),
        }
    baseline = baseline_predictions["A0_fixed"]
    for variant, prediction in ensemble_predictions.items():
        summaries[variant]["paired_bootstrap_vs_fixed"] = sea.cell_cluster_bootstrap_wape_delta(
            actual, prediction, baseline, cells, replicates=5000, seed=42
        )
    atomic_csv(output / "clean_accuracy.csv", rows)
    atomic_csv(output / "per_horizon_wape.csv", horizon_rows)
    atomic_json(output / "failures.json", failures)
    payload = {
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "primary_variant_predeclared": PRIMARY_VARIANT,
        "variants": summaries,
        "failure_count": len(failures),
    }
    atomic_json(output / "summary.json", payload)
    return {**payload, "ensemble_predictions": ensemble_predictions}


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result
    rx, ry = ranks(np.asarray(x).ravel()), ranks(np.asarray(y).ravel())
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def load_checkpoint(path: Path, device: torch.device) -> tuple[sea.WLCRSEA, dict[str, object]]:
    # Registered project checkpoints contain configuration dictionaries and a
    # NumPy frozen prior in addition to tensor weights. Callers must verify the
    # trusted checkpoint hash before opting into the complete payload format.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    variant = sea.VariantConfig(**payload["variant"])
    model = model_from_config(variant, payload["selected_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def audit_and_robustness(
    output: Path,
    dataset: neural.CachedDataset,
    holdout: np.ndarray,
    final_train: np.ndarray,
    baseline_predictions: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    smoke: bool,
) -> None:
    checkpoint = output / "models" / f"{PRIMARY_VARIANT}_seed42.pt"
    if not checkpoint.is_file():
        return
    device = torch.device("cuda:0")
    model, payload = load_checkpoint(checkpoint, device)
    prior = np.asarray(payload["prior_log"], dtype=np.float32)
    clean_batch, clean_tensors = make_eval_tensors(dataset, holdout, prior)
    clean_output = predict(
        model, clean_tensors, device=device, batch_size=batch_size, include_audit=True
    )
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    scales = np.asarray(dataset.mase_scales[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    thresholds = sea.frozen_low_activity_thresholds(dataset.targets, dataset.target_masks, final_train)
    robustness_rows: list[dict[str, object]] = []
    avoidance_rows: list[dict[str, object]] = []
    rates = ROBUSTNESS_RATES[:2] if smoke else ROBUSTNESS_RATES
    mechanisms = ROBUSTNESS_MECHANISMS[:2] if smoke else ROBUSTNESS_MECHANISMS
    fixed_model = sea.WLCRSEA(sea.VARIANTS["A0_fixed"], token_dim=16, hidden_dim=32)
    for mechanism in mechanisms:
        for rate in rates:
            extra = sea.global_corruption_mask(
                np.asarray(dataset.cells[holdout]),
                np.asarray(dataset.history_end_hours[holdout]),
                mechanism=mechanism,
                requested_rate=rate,
                seed=42,
            )
            corrupt_batch, corrupt_tensors = make_eval_tensors(
                dataset, holdout, prior, additional_missing=extra
            )
            stats = sea.corruption_statistics(np.asarray(dataset.x_masks[holdout]), extra)
            for method, method_model, method_device in (
                (PRIMARY_VARIANT, model, device),
                ("A0_fixed", fixed_model, torch.device("cpu")),
            ):
                result = predict(
                    method_model,
                    corrupt_tensors,
                    device=method_device,
                    batch_size=batch_size,
                    include_audit=method == PRIMARY_VARIANT,
                )
                metrics = sea.forecast_metrics(actual, result["prediction"], scales, cells)
                ths = sea.threshold_hit_score(actual, result["prediction"], thresholds)
                robustness_rows.append(
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
                    }
                )
                if method == PRIMARY_VARIANT and rate > 0.0:
                    changed = (
                        (corrupt_batch.availability != clean_batch.availability)
                        | (np.abs(corrupt_batch.reliability - clean_batch.reliability) > 1e-6)
                        | (np.abs(corrupt_batch.values - clean_batch.values) > 1e-6)
                    )
                    corrupted_mass = np.sum(result["attention"] * changed, axis=-1)
                    avoidance_rows.append(
                        {
                            "mechanism": mechanism,
                            "requested_rate": rate,
                            "mean_corrupted_expert_mass": float(np.mean(corrupted_mass)),
                            "mean_avoidance_rate": float(np.mean(1.0 - corrupted_mass)),
                        }
                    )
    atomic_csv(output / "missingness_robustness.csv", robustness_rows)
    atomic_csv(output / "corrupted_expert_avoidance.csv", avoidance_rows)

    attention = clean_output["attention"]
    entropy = clean_output["entropy"]
    residual = clean_output["residual"]
    baseline_log = clean_output["baseline_log"]
    support = np.sum(attention > 1e-6, axis=-1)
    hard_violation = float(
        np.max(attention[~clean_batch.availability])
        if np.any(~clean_batch.availability)
        else 0.0
    )
    y = actual
    p = clean_output["prediction"]
    valid = np.isfinite(y) & (y > 0.0)
    ape = np.zeros_like(y, dtype=np.float64)
    ape[valid] = np.abs(y[valid] - p[valid]) / y[valid]
    residual_ratio = np.abs(residual) / (np.abs(baseline_log) + 1e-6)

    values_t, availability_t, reliability_t, context_t = clean_tensors
    top = np.argmax(attention[..., :7], axis=-1)
    random_choice = np.zeros_like(top)
    rng = np.random.default_rng(42)
    for index in np.ndindex(top.shape):
        available = np.flatnonzero(clean_batch.availability[index][:7])
        candidates = available[available != top[index]]
        random_choice[index] = int(rng.choice(candidates)) if len(candidates) else int(top[index])

    def deletion_prediction(choice: np.ndarray) -> np.ndarray:
        outputs: list[np.ndarray] = []
        loader = DataLoader(
            TensorDataset(values_t, availability_t, reliability_t, context_t, torch.from_numpy(choice)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        with torch.no_grad():
            for values, availability, reliability, context, selected in loader:
                availability = availability.bool()
                selected = selected.long()
                availability.scatter_(-1, selected.unsqueeze(-1), False)
                reliability.scatter_(-1, selected.unsqueeze(-1), 0.0)
                result = model(
                    values.to(device),
                    availability.to(device),
                    reliability.to(device),
                    context.to(device),
                )
                outputs.append(result["prediction_log"].cpu().numpy())
        return np.asarray(sea.prediction_from_log(np.concatenate(outputs, axis=0)))

    top_prediction = deletion_prediction(top)
    random_prediction = deletion_prediction(random_choice)
    original_metrics = sea.forecast_metrics(actual, p, scales, cells)
    top_metrics = sea.forecast_metrics(actual, top_prediction, scales, cells)
    random_metrics = sea.forecast_metrics(actual, random_prediction, scales, cells)
    audit = {
        "attention_is_internal_allocation_not_causal_explanation": True,
        "hard_availability_max_unavailable_weight": hard_violation,
        "effective_experts": {
            "mean": float(np.mean(support)),
            "p50": float(np.quantile(support, 0.5)),
            "p90": float(np.quantile(support, 0.9)),
        },
        "attention_entropy": {
            "mean": float(np.mean(entropy)),
            "p50": float(np.quantile(entropy, 0.5)),
            "p90": float(np.quantile(entropy, 0.9)),
            "spearman_with_absolute_percentage_error": rank_correlation(entropy[valid], ape[valid]),
        },
        "deletion_fidelity": {
            "original_macro_wape": original_metrics["macro_indicator"]["wape"],
            "remove_top_macro_wape": top_metrics["macro_indicator"]["wape"],
            "remove_random_macro_wape": random_metrics["macro_indicator"]["wape"],
            "top_delta": top_metrics["macro_indicator"]["wape"] - original_metrics["macro_indicator"]["wape"],
            "random_delta": random_metrics["macro_indicator"]["wape"] - original_metrics["macro_indicator"]["wape"],
        },
        "bounded_residual_ratio": {
            "mean": float(np.mean(residual_ratio)),
            "p50": float(np.quantile(residual_ratio, 0.5)),
            "p90": float(np.quantile(residual_ratio, 0.9)),
            "max_absolute_residual": float(np.max(np.abs(residual))),
            "configured_bound": float(payload["selected_config"]["residual_bound"]),
        },
    }
    atomic_json(output / "auditability.json", audit)

    expert_rows: list[dict[str, object]] = []
    for metric, metric_name in enumerate(sea.METRIC_NAMES):
        for horizon in range(sea.FORECAST_HOURS):
            for expert, expert_name in enumerate(sea.EXPERT_NAMES):
                expert_rows.append(
                    {
                        "metric": metric_name,
                        "horizon": horizon + 1,
                        "expert": expert_name,
                        "mean_weight": float(np.mean(attention[:, horizon, metric, expert])),
                        "availability_rate": float(np.mean(clean_batch.availability[:, horizon, metric, expert])),
                        "mean_reliability": float(np.mean(clean_batch.reliability[:, horizon, metric, expert])),
                    }
                )
    atomic_csv(output / "attention_by_expert_horizon.csv", expert_rows)

    sample_rows: list[dict[str, object]] = []
    sample_windows = min(200, len(holdout))
    for window in range(sample_windows):
        for horizon in range(sea.FORECAST_HOURS):
            for metric, metric_name in enumerate(sea.METRIC_NAMES):
                row: dict[str, object] = {
                    "window_index": int(holdout[window]),
                    "cell": str(dataset.cells[holdout[window]]),
                    "horizon": horizon + 1,
                    "metric": metric_name,
                    "baseline_log": float(baseline_log[window, horizon, metric]),
                    "residual": float(residual[window, horizon, metric]),
                    "prediction": float(p[window, horizon, metric]),
                    "entropy": float(entropy[window, horizon, metric]),
                }
                for expert, expert_name in enumerate(sea.EXPERT_NAMES):
                    row[f"weight_{expert_name}"] = float(attention[window, horizon, metric, expert])
                sample_rows.append(row)
    atomic_csv(output / "request_audit_sample.csv", sample_rows)


def write_access_audit(output: Path) -> None:
    sources = [Path("experiments/wlcr_sea_model.py"), Path("experiments/train_wlcr_sea.py")]
    forbidden = (
        "/".join(("data", "test_data.csv")),
        "/".join(("data", "reference", "preliminary", "test_data.csv")),
        "requests" + ".",
        "socket" + ".",
    )
    scans = []
    for relative in sources:
        text = (project_root() / relative).read_text(encoding="utf-8")
        scans.append(
            {
                "path": str(relative),
                "sha256": sha256_file(project_root() / relative),
                "forbidden_tokens_found": [token for token in forbidden if token in text],
            }
        )
    atomic_json(
        output / "request_local_access_audit.json",
        {
            "schema_version": 1,
            "feature_allowlist": [
                "target-cell 336-hour log1p traffic",
                "target-cell observation masks",
                "forecast horizon",
                "metric identity",
                "frozen training-derived horizon/metric prior",
            ],
            "feature_lineage": {
                "seasonal_experts": "current request history only",
                "availability_and_reliability": "current request masks only",
                "training_prior": "model parameter estimated from the applicable training layer",
            },
            "explicit_cell_id_feature": False,
            "parameter_or_coordinate_feature": False,
            "weather_feature": False,
            "cross_request_cache": False,
            "external_network": False,
            "finals_test_opened": False,
            "source_scans": scans,
        },
    )


def output_manifest(output: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        records.append(
            {
                "path": str(path.relative_to(output)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def run_master(args: argparse.Namespace) -> int:
    output = resolve_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    variants = [str(item) for item in parse_list(args.variants)]
    unknown = [name for name in variants if name not in sea.VARIANTS or name == "A0_fixed"]
    if unknown:
        raise ValueError(f"unknown trainable variants: {unknown}")
    seeds = [int(item) for item in parse_list(args.seeds, integer=True)]
    devices = [int(item) for item in parse_list(args.gpu_devices, integer=True)]
    if not args.smoke and len(seeds) < 5:
        raise ValueError("full WLCR-SEA evaluation requires at least five seeds")
    available = {int(item["index"]): item for item in gpu_snapshot()}
    for device in devices:
        if device not in available:
            raise ValueError(f"GPU {device} is not visible")
        if int(available[device]["memory_used_mb"]) > args.max_existing_gpu_memory_mb:
            raise RuntimeError(f"GPU {device} is already occupied")
    train_path = neural.resolve_train_path()
    train_hash_before = neural.sha256_file(train_path)
    started = time.perf_counter()
    gpu_samples = [{"timestamp": datetime.now().isoformat(), "phase": "before", "gpus": gpu_snapshot()}]
    with tempfile.TemporaryDirectory(prefix="wlcr-sea-") as temporary:
        cache = Path(temporary)
        series = neural.read_training_series(train_path)
        arrays, dataset_report = neural.build_window_arrays(series)
        if dataset_report["candidate_windows"] != 11_686 or dataset_report["continuous_windows"] != 11_685:
            raise ValueError("registered supervised-window counts changed")
        neural.write_dataset_cache(cache, arrays)
        dataset = neural.load_dataset_cache(cache)
        leakage = neural.leakage_checks(dataset)
        fit = dataset.indices_for_dates(neural.FIT_DATES)
        inner = dataset.indices_for_dates(neural.INNER_DATES)
        holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
        if args.smoke:
            fit, inner, holdout = fit[:256], inner[:128], holdout[:128]
        final_train = np.concatenate((fit, inner))
        atomic_json(output / "dataset_report.json", dataset_report)
        atomic_json(output / "leakage_checks.json", leakage)
        protocol = {
            "schema_version": SCHEMA_VERSION,
            "experiment_version": EXPERIMENT_VERSION,
            "evidence_status": "exploratory_redesign_on_existing_trace",
            "confirmatory_test_available": False,
            "reason": "No new untouched period exists in the repository; the August holdout informed earlier revisions.",
            "registered_train_file": str(neural.REGISTERED_TRAIN),
            "registered_train_sha256": train_hash_before,
            "finals_test_opened": False,
            "request_local_information_class": "target-cell traffic + masks + horizon; frozen training prior is a model parameter",
            "fit_dates": [str(value) for value in neural.FIT_DATES],
            "inner_dates": [str(value) for value in neural.INNER_DATES],
            "holdout_dates": [str(value) for value in neural.HOLDOUT_DATES],
            "primary_variant_predeclared": PRIMARY_VARIANT,
            "variants": {name: asdict(sea.VARIANTS[name]) for name in variants},
            "model_candidates": list(MODEL_CANDIDATES),
            "selection_metric": "lowest inner macro-over-indicator WAPE; frozen-threshold hit score is a tie break",
            "max_epochs": 2 if args.smoke else args.max_epochs,
            "patience": 1 if args.smoke else args.patience,
            "batch_size": args.batch_size,
            "seeds": seeds,
            "gpu_devices": devices,
            "augmentation_rate": AUGMENTATION_RATE,
            "augmentation_input_history_only": True,
            "augmentation_labels_and_target_masks_retained": True,
            "final_augmentation_seed_offset": (
                neural.FINAL_AUGMENTATION_SEED_OFFSET
            ),
            "training_prior": {
                "estimator": "median of observed log1p targets",
                "shape": [sea.FORECAST_HOURS, sea.TARGET_COUNT],
                "selection_scope": "fit dates",
                "final_evaluation_scope": "fit plus inner dates",
            },
            "expert_definition_constants": {
                "weekly_trend_clip_kappa": sea.TREND_CLIP,
                "weekly_trend_domain": "log1p",
                "nominal_recency_hours": list(sea.EXPERT_DISTANCE_HOURS),
                "nominal_recency_denominator_hours": sea.INPUT_HOURS,
                "nominal_recency_interpretation": (
                    "fixed expert-type descriptor under complete data; not the "
                    "realized mean age after request-specific missingness"
                ),
                "single_observation_standard_deviation": 0.0,
                "context_trend": (
                    "median(observed last 24 h)-median(observed preceding 24 h); "
                    "zero if either day has no observation"
                ),
                "fixed_mixture_missingness": (
                    "remove unavailable experts and renormalize fixed weights; "
                    "fall back to the frozen prior if their total mass is zero"
                ),
            },
            "attention_interpretation": "inspectable internal expert allocation, not causal explanation",
        }
        atomic_json(output / "protocol.json", protocol)
        jobs = [(variant, seed) for variant in variants for seed in seeds]
        queues = {device: [] for device in devices}
        for index, job in enumerate(jobs):
            queues[devices[index % len(devices)]].append(job)
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = {
                executor.submit(
                    run_queue,
                    device,
                    queue,
                    Path(__file__).resolve(),
                    cache,
                    output,
                    args,
                ): device
                for device, queue in queues.items()
                if queue
            }
            pending = set(futures)
            results: list[dict[str, object]] = []
            while pending:
                complete, pending = wait(pending, timeout=5.0, return_when=FIRST_COMPLETED)
                gpu_samples.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "phase": "training" if pending else "after_jobs",
                        "gpus": gpu_snapshot(),
                    }
                )
                for future in complete:
                    results.extend(future.result())
        baseline_summary, baseline_predictions, _ = deterministic_baselines(
            dataset, holdout, final_train, output
        )
        aggregate = aggregate_jobs(
            output,
            dataset,
            holdout,
            final_train,
            variants,
            seeds,
            results,
            baseline_predictions,
        )
        if args.legacy_seed42_audit and PRIMARY_VARIANT in variants and any(
            item["status"] == "complete"
            and item["variant"] == PRIMARY_VARIANT
            and int(item["seed"]) == 42
            for item in results
        ):
            audit_and_robustness(
                output,
                dataset,
                holdout,
                final_train,
                baseline_predictions,
                batch_size=args.batch_size,
                smoke=args.smoke,
            )
    train_hash_after = neural.sha256_file(train_path)
    if train_hash_after != train_hash_before:
        raise RuntimeError("registered training file changed during WLCR-SEA experiments")
    write_access_audit(output)
    gpu_samples.append({"timestamp": datetime.now().isoformat(), "phase": "complete", "gpus": gpu_snapshot()})
    atomic_json(output / "gpu_evidence.json", {"requested_devices": devices, "samples": gpu_samples})
    atomic_json(
        output / "run_metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_version": EXPERIMENT_VERSION,
            "runtime_seconds": time.perf_counter() - started,
            "input_sha256_before": train_hash_before,
            "input_sha256_after": train_hash_after,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "numpy": np.__version__,
            },
        },
    )
    atomic_json(output / "manifest.json", output_manifest(output))
    failures = json.loads((output / "failures.json").read_text(encoding="utf-8"))
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the WLCR-SEA paper protocol on four GPUs.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-existing-gpu-memory-mb", type=int, default=1024)
    parser.add_argument(
        "--legacy-seed42-audit",
        action="store_true",
        help="Also emit the superseded single-seed Revision-6 audit files.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-cache", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=tuple(sea.VARIANTS), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--physical-device", type=int, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.smoke and not args.worker:
        args.seeds = args.seeds.split(",")[0]
    if args.worker:
        missing = [name for name in ("dataset_cache", "variant", "seed", "physical_device") if getattr(args, name) is None]
        if missing:
            parser.error(f"worker missing arguments: {missing}")
        try:
            return run_worker(args)
        except Exception:
            traceback.print_exc()
            return 1
    return run_master(args)


if __name__ == "__main__":
    raise SystemExit(main())
