from __future__ import annotations

"""Full multi-seed auditability evidence for WLCR-SEA.

This version is independent of Revision 7 and evaluates the five frozen A6
checkpoints under ``artifacts/revision8/wlcr_sea`` on the registered training
trace holdout only.  Expert deletion hard-masks the selected expert, zeros its
reliability, and reruns both routing and the bounded residual; cached attention
is never renormalized.  The audit also performs the Revision-8 multi-request,
bitwise request-local invariance check, including other requests from the same
cell.  It deliberately never opens the finals traffic input.
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments import train_neural_baselines as neural
from experiments import analyze_matched_missingness as comparative
from experiments import audit_request_locality as locality
from experiments import train_wlcr_sea as runner
from experiments import wlcr_sea_model as sea


SCHEMA_VERSION = 2
MODEL_SEEDS = comparative.MODEL_SEEDS
NON_PRIOR_EXPERTS = tuple(range(sea.EXPERT_COUNT - 1))
DEFAULT_SOURCE = Path("artifacts/reproduction/wlcr_sea")
DEFAULT_OUTPUT = Path("artifacts/reproduction/analysis/audit")


def resolve_reproduction_artifact_path(root: Path, text: str | Path, *, strict: bool) -> Path:
    """Resolve one audit path while confining it to the reproduction directory."""
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=strict)
    allowed = (root / "artifacts/reproduction").resolve(strict=False)
    if candidate != allowed and not candidate.is_relative_to(allowed):
        raise ValueError("full-audit paths must remain under artifacts/reproduction")
    return candidate


def parse_gpu_devices(text: str) -> list[int]:
    """Parse a unique, ordered list of physical CUDA device identifiers."""
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one GPU device is required")
    devices = [int(item) for item in values]
    if len(devices) != len(set(devices)):
        raise ValueError("--gpu-devices cannot contain duplicate physical devices")
    if any(device < 0 for device in devices):
        raise ValueError("--gpu-devices must contain non-negative identifiers")
    return devices


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with exact tie handling."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    if not len(array):
        return np.empty(0, dtype=np.float64)
    starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.flatnonzero(np.diff(sorted_values) != 0) + 1)
    )
    ends = np.concatenate((starts[1:], np.asarray([len(array)], dtype=np.int64)))
    average = 0.5 * (starts + 1 + ends)
    sorted_ranks = np.repeat(average, ends - starts)
    result = np.empty(len(array), dtype=np.float64)
    result[order] = sorted_ranks
    return result


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 2:
        return 0.0
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def top_non_prior_choice(
    attention: np.ndarray, availability: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(attention, dtype=np.float64)[..., : len(NON_PRIOR_EXPERTS)]
    available = np.asarray(availability, dtype=bool)[..., : len(NON_PRIOR_EXPERTS)]
    eligible = np.any(available, axis=-1)
    scores = np.where(available, weights, -np.inf)
    choice = np.argmax(scores, axis=-1).astype(np.int64)
    choice = np.where(eligible, choice, 0).astype(np.int64)
    return choice, eligible


def matched_random_choice(
    availability: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    available = np.asarray(availability, dtype=bool)[..., : len(NON_PRIOR_EXPERTS)]
    shape = available.shape[:-1]
    flat = available.reshape(-1, len(NON_PRIOR_EXPERTS))
    counts = np.sum(flat, axis=1)
    eligible = counts > 0
    rng = np.random.default_rng(seed)
    rank = np.zeros(len(flat), dtype=np.int64)
    rank[eligible] = np.floor(rng.random(np.sum(eligible)) * counts[eligible]).astype(
        np.int64
    )
    cumulative = np.cumsum(flat, axis=1)
    choice = np.argmax(cumulative > rank[:, None], axis=1).astype(np.int64)
    choice[~eligible] = 0
    return choice.reshape(shape), eligible.reshape(shape)


def rerouted_prediction(
    model: sea.WLCRSEA,
    tensors: tuple[torch.Tensor, ...],
    choice: np.ndarray,
    eligible: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    include_attention: bool = False,
) -> dict[str, np.ndarray]:
    values, availability, reliability, context = tensors
    choice_tensor = torch.from_numpy(np.asarray(choice, dtype=np.int64))
    eligible_tensor = torch.from_numpy(np.asarray(eligible, dtype=np.uint8))
    loader = DataLoader(
        TensorDataset(
            values,
            availability,
            reliability,
            context,
            choice_tensor,
            eligible_tensor,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    prediction_logs: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    deleted_weights: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for fields in loader:
            batch_values, batch_availability, batch_reliability, batch_context = [
                field.to(device, non_blocking=True) for field in fields[:4]
            ]
            selected = fields[4].to(device, non_blocking=True).long()
            selected_valid = fields[5].to(device, non_blocking=True).bool()
            available = batch_availability.bool().clone()
            reliable = batch_reliability.clone()
            one_hot = torch.nn.functional.one_hot(
                selected, num_classes=sea.EXPERT_COUNT
            ).bool()
            delete_mask = one_hot & selected_valid.unsqueeze(-1)
            available = available & ~delete_mask
            reliable = reliable.masked_fill(delete_mask, 0.0)
            output = model(batch_values, available, reliable, batch_context)
            prediction_logs.append(output["prediction_log"].cpu().numpy())
            residuals.append(output["residual"].cpu().numpy())
            selected_weight = torch.gather(
                output["attention"], -1, selected.unsqueeze(-1)
            ).squeeze(-1)
            deleted_weights.append(
                torch.where(
                    selected_valid,
                    selected_weight,
                    torch.zeros_like(selected_weight),
                )
                .cpu()
                .numpy()
            )
            if include_attention:
                attentions.append(output["attention"].cpu().numpy())
    prediction_log = np.concatenate(prediction_logs, axis=0)
    result: dict[str, np.ndarray] = {
        "prediction_log": prediction_log,
        "prediction": np.asarray(sea.prediction_from_log(prediction_log)),
        "residual": np.concatenate(residuals, axis=0),
        "deleted_weight": np.concatenate(deleted_weights, axis=0),
    }
    if include_attention:
        result["attention"] = np.concatenate(attentions, axis=0)
    return result


def metric_wape(
    actual: np.ndarray,
    prediction: np.ndarray,
    scales: np.ndarray,
    cells: np.ndarray,
) -> float:
    return float(
        sea.forecast_metrics(actual, prediction, scales, cells)["macro_indicator"][
            "wape"
        ]
    )


def sample_standard_deviation(values: Sequence[float]) -> float:
    """Return a defined sample spread for one or more deterministic repeats."""
    if not values:
        raise ValueError("at least one random-deletion repeat is required")
    return 0.0 if len(values) == 1 else float(statistics.stdev(values))


def cell_cluster_bootstrap_expected_random_delta(
    actual: np.ndarray,
    random_predictions: np.ndarray,
    original_prediction: np.ndarray,
    cells: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    y = np.asarray(actual, dtype=np.float64)
    random_values = np.asarray(random_predictions, dtype=np.float64)
    original = np.asarray(original_prediction, dtype=np.float64)
    if random_values.ndim != y.ndim + 1 or random_values.shape[1:] != y.shape:
        raise ValueError("random deletion predictions are misaligned")
    if original.shape != y.shape:
        raise ValueError("original deletion reference is misaligned")
    cell_names, inverse = np.unique(np.asarray(cells).astype(str), return_inverse=True)
    row_cells = np.repeat(inverse, sea.FORECAST_HOURS)
    clusters = len(cell_names)
    denominator = np.zeros((clusters, sea.TARGET_COUNT), dtype=np.float64)
    original_numerator = np.zeros_like(denominator)
    random_numerator = np.zeros(
        (random_values.shape[0], clusters, sea.TARGET_COUNT), dtype=np.float64
    )
    for metric in range(sea.TARGET_COUNT):
        target = y[..., metric].reshape(-1)
        valid = np.isfinite(target)
        denominator[:, metric] = np.bincount(
            row_cells,
            weights=np.where(valid, np.abs(target), 0.0),
            minlength=clusters,
        )
        original_flat = original[..., metric].reshape(-1)
        original_numerator[:, metric] = np.bincount(
            row_cells,
            weights=np.where(valid, np.abs(target - original_flat), 0.0),
            minlength=clusters,
        )
        for repeat in range(random_values.shape[0]):
            prediction = random_values[repeat, ..., metric].reshape(-1)
            random_numerator[repeat, :, metric] = np.bincount(
                row_cells,
                weights=np.where(valid, np.abs(target - prediction), 0.0),
                minlength=clusters,
            )
    eligible_clusters = np.any(denominator > 0.0, axis=1)
    denominator = denominator[eligible_clusters]
    original_numerator = original_numerator[eligible_clusters]
    random_numerator = random_numerator[:, eligible_clusters, :]
    clusters = int(np.sum(eligible_clusters))
    denominator_sum = np.sum(denominator, axis=0)
    if np.any(denominator_sum <= 0.0):
        raise ValueError("every indicator needs a positive random-deletion denominator")
    original_macro = float(
        np.mean(np.sum(original_numerator, axis=0) / denominator_sum)
    )
    random_macro_by_repeat = np.mean(
        np.sum(random_numerator, axis=1) / denominator_sum[None, :], axis=1
    )
    expected_random_macro = float(np.mean(random_macro_by_repeat))
    point_delta = expected_random_macro - original_macro
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, clusters, size=clusters)
        denominator_b = np.sum(denominator[sampled], axis=0)
        original_b = float(
            np.mean(np.sum(original_numerator[sampled], axis=0) / denominator_b)
        )
        random_b = np.mean(
            np.sum(random_numerator[:, sampled, :], axis=1)
            / denominator_b[None, :],
            axis=1,
        )
        deltas[replicate] = float(np.mean(random_b) - original_b)
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "estimand": "expected_macro_wape_over_matched_random_deletions",
        "random_repeats": int(random_values.shape[0]),
        "clusters": clusters,
        "replicates": replicates,
        "seed": seed,
        "point_original_macro_wape": original_macro,
        "point_expected_random_macro_wape": expected_random_macro,
        "delta_expected_random_minus_original": point_delta,
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_delta_below_zero": float(np.mean(deltas < 0.0)),
    }


def worker(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("audit worker requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    runner.set_seed(args.seed)
    root = runner.project_root()
    source = resolve_reproduction_artifact_path(root, args.source, strict=True)
    output = resolve_reproduction_artifact_path(root, args.output, strict=False)
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("full-audit source and output must be disjoint")
    worker_dir = output / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    dataset = neural.load_dataset_cache(Path(args.dataset_cache).resolve(strict=True))
    holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    scales = np.asarray(dataset.mase_scales[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    model, payload = runner.load_checkpoint(
        source / "models" / f"{runner.PRIMARY_VARIANT}_seed{args.seed}.pt", device
    )
    prior = np.asarray(payload["prior_log"], dtype=np.float32)
    expert_batch, tensors = runner.make_eval_tensors(dataset, holdout, prior)
    original = runner.predict(
        model,
        tensors,
        device=device,
        batch_size=args.batch_size,
        include_audit=True,
    )
    original_wape = metric_wape(actual, original["prediction"], scales, cells)

    top_choice, top_eligible = top_non_prior_choice(
        original["attention"], expert_batch.availability
    )
    top = rerouted_prediction(
        model,
        tensors,
        top_choice,
        top_eligible,
        device=device,
        batch_size=args.batch_size,
        include_attention=True,
    )
    top_wape = metric_wape(actual, top["prediction"], scales, cells)

    random_rows: list[dict[str, object]] = []
    random_prediction_sum = np.zeros_like(original["prediction"], dtype=np.float64)
    random_predictions: list[np.ndarray] = []
    random_wapes: list[float] = []
    maximum_random_deleted_weight = 0.0
    for repeat in range(args.random_repeats):
        random_choice, random_eligible = matched_random_choice(
            expert_batch.availability,
            seed=args.random_seed + 1000 * args.seed + repeat,
        )
        random_output = rerouted_prediction(
            model,
            tensors,
            random_choice,
            random_eligible,
            device=device,
            batch_size=args.batch_size,
            include_attention=False,
        )
        random_prediction_sum += random_output["prediction"]
        random_predictions.append(np.asarray(random_output["prediction"], dtype=np.float32))
        random_wape = metric_wape(
            actual, random_output["prediction"], scales, cells
        )
        random_wapes.append(random_wape)
        maximum_random_deleted_weight = max(
            maximum_random_deleted_weight,
            float(np.max(np.abs(random_output["deleted_weight"]))),
        )
        random_rows.append(
            {
                "seed": args.seed,
                "repeat": repeat,
                "random_seed": args.random_seed + 1000 * args.seed + repeat,
                "available_non_prior_only": True,
                "eligible_fraction": float(np.mean(random_eligible)),
                "macro_wape": random_wape,
                "delta_vs_original": random_wape - original_wape,
            }
        )
    random_mean_prediction = (
        random_prediction_sum / float(args.random_repeats)
    ).astype(np.float32)
    random_mean_prediction_wape = metric_wape(
        actual, random_mean_prediction, scales, cells
    )

    loo_rows: list[dict[str, object]] = []
    pooled_attention: list[np.ndarray] = []
    pooled_influence: list[np.ndarray] = []
    for expert in NON_PRIOR_EXPERTS:
        choice = np.full(original["prediction"].shape, expert, dtype=np.int64)
        eligible = np.asarray(expert_batch.availability[..., expert], dtype=bool)
        deleted = rerouted_prediction(
            model,
            tensors,
            choice,
            eligible,
            device=device,
            batch_size=args.batch_size,
            include_attention=False,
        )
        influence = np.abs(original["prediction"] - deleted["prediction"])
        weights = np.asarray(original["attention"][..., expert])
        pooled_attention.append(weights[eligible])
        pooled_influence.append(influence[eligible])
        deleted_wape = metric_wape(actual, deleted["prediction"], scales, cells)
        loo_rows.append(
            {
                "seed": args.seed,
                "expert_index": expert,
                "expert": sea.EXPERT_NAMES[expert],
                "availability_rate": float(np.mean(eligible)),
                "mean_attention_when_available": float(np.mean(weights[eligible])),
                "mean_absolute_prediction_change": float(np.mean(influence[eligible])),
                "p90_absolute_prediction_change": float(
                    np.quantile(influence[eligible], 0.9)
                ),
                "spearman_attention_influence": spearman_correlation(
                    weights[eligible], influence[eligible]
                ),
                "macro_wape": deleted_wape,
                "delta_vs_original": deleted_wape - original_wape,
                "maximum_deleted_weight_after_reroute": float(
                    np.max(np.abs(deleted["deleted_weight"]))
                ),
            }
        )
    pooled_spearman = spearman_correlation(
        np.concatenate(pooled_attention), np.concatenate(pooled_influence)
    )

    unavailable = ~np.asarray(expert_batch.availability, dtype=bool)
    hard_violation = (
        float(np.max(np.abs(original["attention"][unavailable])))
        if np.any(unavailable)
        else 0.0
    )
    lower, upper = sea.bounded_audit_envelope(
        expert_batch.values,
        expert_batch.availability,
        float(payload["selected_config"]["residual_bound"]),
    )
    prediction_log = np.asarray(original["prediction_log"], dtype=np.float64)
    lower_violation = float(np.max(np.maximum(lower - prediction_log, 0.0)))
    upper_violation = float(np.max(np.maximum(prediction_log - upper, 0.0)))
    support = np.sum(original["attention"] > 1e-6, axis=-1)
    residual_ratio = np.abs(original["residual"]) / (
        np.abs(original["baseline_log"]) + 1e-6
    )
    valid = np.isfinite(actual) & (actual > 0.0)
    ape = np.zeros_like(actual, dtype=np.float64)
    ape[valid] = (
        np.abs(actual[valid] - original["prediction"][valid]) / actual[valid]
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "physical_gpu": args.physical_device,
        "deletion_semantics": (
            "mask selected expert, zero its reliability, rerun attention and bounded "
            "residual; cached weights are never renormalized"
        ),
        "original_macro_wape": original_wape,
        "top_macro_wape": top_wape,
        "top_delta": top_wape - original_wape,
        "top_eligible_fraction": float(np.mean(top_eligible)),
        "top_maximum_deleted_weight_after_reroute": float(
            np.max(np.abs(top["deleted_weight"]))
        ),
        "random_repeat_count": args.random_repeats,
        "random_macro_wape_mean": float(statistics.mean(random_wapes)),
        "random_macro_wape_sd": sample_standard_deviation(random_wapes),
        "random_delta_mean": float(statistics.mean(random_wapes) - original_wape),
        "random_mean_prediction_macro_wape": random_mean_prediction_wape,
        "random_mean_prediction_delta": random_mean_prediction_wape - original_wape,
        "random_maximum_deleted_weight_after_reroute": maximum_random_deleted_weight,
        "leave_one_out_pooled_spearman": pooled_spearman,
        "hard_availability_max_unavailable_weight": hard_violation,
        "effective_support_mean": float(np.mean(support)),
        "effective_support_p50": float(np.quantile(support, 0.5)),
        "effective_support_p90": float(np.quantile(support, 0.9)),
        "attention_entropy_ape_spearman": spearman_correlation(
            original["entropy"][valid], ape[valid]
        ),
        "residual_ratio_mean": float(np.mean(residual_ratio)),
        "residual_ratio_p50": float(np.quantile(residual_ratio, 0.5)),
        "residual_ratio_p90": float(np.quantile(residual_ratio, 0.9)),
        "maximum_absolute_residual": float(np.max(np.abs(original["residual"]))),
        "configured_residual_bound": float(
            payload["selected_config"]["residual_bound"]
        ),
        "mean_prior_mass": float(np.mean(original["attention"][..., -1])),
        "bounded_envelope_maximum_lower_violation": lower_violation,
        "bounded_envelope_maximum_upper_violation": upper_violation,
        "bounded_envelope_pass": lower_violation <= 5e-6 and upper_violation <= 5e-6,
        "finals_test_opened": False,
    }
    prediction_file = worker_dir / f"seed{args.seed}_predictions.npz"
    np.savez_compressed(
        prediction_file,
        original=original["prediction"],
        top=top["prediction"],
        random_mean=random_mean_prediction,
        random_repeats=np.stack(random_predictions, axis=0),
    )
    runner.atomic_csv(worker_dir / f"seed{args.seed}_random.csv", random_rows)
    runner.atomic_csv(worker_dir / f"seed{args.seed}_loo.csv", loo_rows)
    runner.atomic_json(worker_dir / f"seed{args.seed}.json", report)
    print(json.dumps({"status": "complete", "seed": args.seed}))
    return 0


def launch_worker(
    seed: int,
    device: int,
    script: Path,
    cache: Path,
    source: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--seed",
        str(seed),
        "--physical-device",
        str(device),
        "--dataset-cache",
        str(cache),
        "--source",
        str(source),
        "--output",
        str(output),
        "--batch-size",
        str(args.batch_size),
        "--random-repeats",
        str(args.random_repeats),
        "--random-seed",
        str(args.random_seed),
    ]
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
    log = output / "logs" / f"seed{seed}.log"
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
        "seed": seed,
        "device": device,
        "returncode": completed.returncode,
        "log": str(log.relative_to(output)),
    }


def request_local_invariance(
    dataset: neural.CachedDataset,
    holdout: np.ndarray,
    source: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Audit 256 deterministic requests against perturbations of their complements.

    This invokes the Revision-8 locality primitives rather than the former
    the former single-request diagnostic.  For each selected request, all other
    requests are mutated, including the remaining requests of the same cell;
    expert tensors and every tensor returned by the frozen seed-42 model must
    remain bitwise identical.
    """
    checkpoint = source / "models" / f"{runner.PRIMARY_VARIANT}_seed{args.locality_model_seed}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing locality checkpoint: {checkpoint}")
    source_manifest = source / "manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"missing Revision-8 source manifest: {source_manifest}")

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.locality_cpu_threads)
        runner.set_seed(args.locality_model_seed)
        population = locality.build_audit_requests(dataset, holdout)
        selected, quotas = locality.select_stratified_requests(
            population, args.locality_sample_size, args.locality_audit_seed
        )
        model, payload = runner.load_checkpoint(checkpoint, torch.device("cpu"))
        prior = np.asarray(payload["prior_log"], dtype=np.float32)
        details, perturbation = locality.audit_selected_requests(
            dataset, selected, model, prior
        )
    finally:
        torch.set_num_threads(previous_threads)

    targets = locality.target_list_payload(
        selected, quotas, population, audit_seed=args.locality_audit_seed
    )
    violations = [
        item for item in details if not bool(item["bitwise_request_local_invariance_pass"])
    ]
    fields = locality.source_dataset_fields()
    expected_fields = ["x_masks", "x_values"]
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "registered_train_path": str(neural.REGISTERED_TRAIN),
        "source": str(source.relative_to(runner.project_root())),
        "source_manifest_sha256": runner.sha256_file(source_manifest),
        "checkpoint": str(checkpoint.relative_to(runner.project_root())),
        "checkpoint_sha256": runner.sha256_file(checkpoint),
        "variant": runner.PRIMARY_VARIANT,
        "model_seed": args.locality_model_seed,
        "device": "cpu",
        "cpu_threads": args.locality_cpu_threads,
        "script_sha256": runner.sha256_file(Path(__file__).resolve()),
        "locality_primitives_source_sha256": runner.sha256_file(
            runner.project_root() / "experiments/audit_request_locality.py"
        ),
        "make_eval_tensors_source_sha256": runner.sha256_file(
            runner.project_root() / "experiments/train_wlcr_sea.py"
        ),
        "expert_builder_source_sha256": runner.sha256_file(
            runner.project_root() / "experiments/wlcr_sea_model.py"
        ),
        "make_eval_tensors_dataset_fields": fields,
        "make_eval_tensors_allowlist_verified": fields == expected_fields,
        "inference_field_allowlist": [
            "target request x_values[336,4]",
            "target request x_masks[336,4]",
            "frozen prior_log[24,4]",
            "globally shared checkpoint parameters",
        ],
        "finals_test_opened": False,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "n_tested": len(details),
        "n_violations": len(violations),
        "bitwise_request_local_invariance_pass": len(violations) == 0,
        "target_list_sha256": targets["target_list_sha256"],
        "tested_cells": sorted({request.cell for request in selected}),
        "n_tested_cells": len({request.cell for request in selected}),
        "tested_dates": sorted({request.target_date for request in selected}),
        "tested_missingness_bins": sorted(
            {request.missingness_bin for request in selected}
        ),
        "strata_counts": targets["sampling"]["strata"],
        "input_perturbation": perturbation,
        "same_cell_noncurrent_requests_are_included": all(
            int(item.get("same_cell_noncurrent_request_count", 0)) > 0
            for item in details
            if "error_type" not in item
        ),
        "provenance": protocol,
        "finals_test_opened": False,
    }
    return summary, targets, details, protocol

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def seed_interval(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    mean, sd, low, high = comparative.t_interval(values)
    return {"mean": mean, "sd": sd, "ci_low": low, "ci_high": high}


def aggregate_master(
    output: Path,
    dataset: neural.CachedDataset,
    holdout: np.ndarray,
    source: Path,
    args: argparse.Namespace,
) -> None:
    reports = [
        json.loads(
            (output / "worker" / f"seed{seed}.json").read_text(encoding="utf-8")
        )
        for seed in MODEL_SEEDS
    ]
    random_rows: list[dict[str, object]] = []
    loo_rows: list[dict[str, object]] = []
    predictions: dict[str, list[np.ndarray]] = {
        "original": [],
        "top": [],
        "random_mean": [],
    }
    random_repeat_predictions: list[np.ndarray] = []
    for seed in MODEL_SEEDS:
        random_rows.extend(
            read_csv_rows(output / "worker" / f"seed{seed}_random.csv")
        )
        loo_rows.extend(
            read_csv_rows(output / "worker" / f"seed{seed}_loo.csv")
        )
        with np.load(
            output / "worker" / f"seed{seed}_predictions.npz", allow_pickle=False
        ) as arrays:
            for key in predictions:
                predictions[key].append(np.asarray(arrays[key], dtype=np.float32))
            random_repeat_predictions.append(
                np.asarray(arrays["random_repeats"], dtype=np.float32)
            )
    ensembles = {
        key: np.mean(np.stack(items, axis=0), axis=0, dtype=np.float64).astype(
            np.float32
        )
        for key, items in predictions.items()
    }
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    random_repeat_ensemble = np.mean(
        np.stack(random_repeat_predictions, axis=0),
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    bootstrap = {
        "top_deletion_minus_original": sea.cell_cluster_bootstrap_wape_delta(
            actual,
            ensembles["top"],
            ensembles["original"],
            cells,
            replicates=args.bootstrap_replicates,
            seed=42,
        ),
        "expected_matched_random_deletion_minus_original": cell_cluster_bootstrap_expected_random_delta(
            actual,
            random_repeat_ensemble,
            ensembles["original"],
            cells,
            replicates=args.bootstrap_replicates,
            seed=43,
        ),
        "mean_random_prediction_minus_original_diagnostic": sea.cell_cluster_bootstrap_wape_delta(
            actual,
            ensembles["random_mean"],
            ensembles["original"],
            cells,
            replicates=args.bootstrap_replicates,
            seed=44,
        ),
    }
    runner.atomic_json(output / "deletion_bootstrap.json", bootstrap)
    runner.atomic_csv(output / "seed_results.csv", reports)
    runner.atomic_csv(output / "random_deletion_repeats.csv", random_rows)
    runner.atomic_csv(output / "leave_one_out_by_seed.csv", loo_rows)

    loo_summary: list[dict[str, object]] = []
    for expert in sea.EXPERT_NAMES[:-1]:
        selected = [row for row in loo_rows if row["expert"] == expert]
        record: dict[str, object] = {
            "expert": expert,
            "seed_count": len(selected),
        }
        for field in (
            "availability_rate",
            "mean_attention_when_available",
            "mean_absolute_prediction_change",
            "p90_absolute_prediction_change",
            "spearman_attention_influence",
            "delta_vs_original",
        ):
            values = [float(row[field]) for row in selected]
            mean, sd, low, high = comparative.t_interval(values)
            record[field] = mean
            record[f"{field}_sd"] = sd
            record[f"{field}_ci_low"] = low
            record[f"{field}_ci_high"] = high
        loo_summary.append(record)
    runner.atomic_csv(output / "leave_one_out_summary.csv", loo_summary)

    invariance, locality_targets, locality_details, locality_protocol = request_local_invariance(
        dataset, holdout, source, args
    )
    runner.atomic_json(output / "request_local_targets.json", locality_targets)
    runner.atomic_json(output / "request_local_details.json", {"requests": locality_details})
    runner.atomic_json(output / "request_local_protocol.json", locality_protocol)
    runner.atomic_json(output / "request_local_invariance.json", invariance)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "model_seeds": list(MODEL_SEEDS),
        "random_repeats_per_model_seed": args.random_repeats,
        "deletion_semantics": (
            "selected available non-prior expert is hard-masked; reliability is zeroed; "
            "attention and bounded residual are recomputed"
        ),
        "seed_intervals": {
            field: seed_interval(reports, field)
            for field in (
                "original_macro_wape",
                "top_delta",
                "random_delta_mean",
                "random_mean_prediction_delta",
                "leave_one_out_pooled_spearman",
                "hard_availability_max_unavailable_weight",
                "effective_support_mean",
                "attention_entropy_ape_spearman",
                "residual_ratio_p50",
                "residual_ratio_p90",
                "maximum_absolute_residual",
                "mean_prior_mass",
            )
        },
        "paired_cell_cluster_bootstrap": bootstrap,
        "all_bounded_envelope_checks_pass": all(
            bool(row["bounded_envelope_pass"]) for row in reports
        ),
        "maximum_bounded_envelope_violation": max(
            max(
                float(row["bounded_envelope_maximum_lower_violation"]),
                float(row["bounded_envelope_maximum_upper_violation"]),
            )
            for row in reports
        ),
        "request_local_invariance": invariance,
        "attention_is_internal_allocation_not_causal_explanation": True,
        "finals_test_opened": False,
    }
    runner.atomic_json(output / "summary.json", summary)
    runner.atomic_json(output / "manifest.json", runner.output_manifest(output))


def run_master(args: argparse.Namespace) -> int:
    root = runner.project_root()
    source = resolve_reproduction_artifact_path(root, args.source, strict=True)
    output = resolve_reproduction_artifact_path(root, args.output, strict=False)
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("full-audit source and output must be disjoint")
    if not (source / "models").is_dir():
        raise FileNotFoundError(f"missing source model directory: {source / 'models'}")
    if not (source / "manifest.json").is_file():
        raise FileNotFoundError(f"missing source manifest: {source / 'manifest.json'}")
    output.mkdir(parents=True, exist_ok=True)
    train_path = neural.resolve_train_path()
    before = neural.sha256_file(train_path)
    devices = parse_gpu_devices(args.gpu_devices)
    with tempfile.TemporaryDirectory(prefix="revision8-full-audit-") as temporary:
        cache = Path(temporary)
        arrays, _ = neural.build_window_arrays(neural.read_training_series(train_path))
        neural.write_dataset_cache(cache, arrays)
        dataset = neural.load_dataset_cache(cache)
        holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
        if args.locality_sample_size > len(holdout):
            raise ValueError("--locality-sample-size exceeds the registered holdout requests")
        groups = [
            (device, list(MODEL_SEEDS[offset :: len(devices)]))
            for offset, device in enumerate(devices)
            if MODEL_SEEDS[offset :: len(devices)]
        ]

        def run_group(group: tuple[int, list[int]]) -> list[dict[str, object]]:
            device, seeds = group
            return [
                launch_worker(
                    seed,
                    device,
                    Path(__file__).resolve(),
                    cache,
                    source,
                    output,
                    args,
                )
                for seed in seeds
            ]

        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            nested = list(executor.map(run_group, groups))
        statuses = [item for group in nested for item in group]
        runner.atomic_json(output / "worker_status.json", statuses)
        failures = [item for item in statuses if int(item["returncode"]) != 0]
        if failures:
            return 1
        aggregate_master(output, dataset, holdout, source, args)
    after = neural.sha256_file(train_path)
    if before != after:
        raise RuntimeError("registered training data changed during Revision-8 full audit")
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "source": str(source.relative_to(root)),
        "source_manifest_sha256": runner.sha256_file(source / "manifest.json"),
        "model_seeds": list(MODEL_SEEDS),
        "gpu_devices": devices,
        "random_repeats_per_seed": args.random_repeats,
        "bootstrap_replicates": args.bootstrap_replicates,
        "locality_model_seed": args.locality_model_seed,
        "locality_sample_size": args.locality_sample_size,
        "locality_audit_seed": args.locality_audit_seed,
        "locality_cpu_threads": args.locality_cpu_threads,
        "registered_train_sha256_before": before,
        "registered_train_sha256_after": after,
        "finals_test_opened": False,
    }
    runner.atomic_json(output / "protocol.json", protocol)
    runner.atomic_json(output / "manifest.json", runner.output_manifest(output))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", default=str(DEFAULT_SOURCE))
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    value.add_argument("--gpu-devices", default="0,1,2,3")
    value.add_argument("--batch-size", type=int, default=256)
    value.add_argument("--random-repeats", type=int, default=10)
    value.add_argument("--random-seed", type=int, default=7000)
    value.add_argument("--bootstrap-replicates", type=int, default=5000)
    value.add_argument(
        "--locality-model-seed", type=int, default=locality.DEFAULT_MODEL_SEED
    )
    value.add_argument(
        "--locality-sample-size", type=int, default=locality.DEFAULT_SAMPLE_SIZE
    )
    value.add_argument(
        "--locality-audit-seed", type=int, default=locality.DEFAULT_AUDIT_SEED
    )
    value.add_argument("--locality-cpu-threads", type=int, default=1)
    value.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    value.add_argument("--physical-device", type=int, help=argparse.SUPPRESS)
    value.add_argument("--dataset-cache", help=argparse.SUPPRESS)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.random_repeats <= 0:
        raise ValueError("--random-repeats must be positive")
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    if args.locality_sample_size <= 0:
        raise ValueError("--locality-sample-size must be positive")
    if args.locality_cpu_threads <= 0:
        raise ValueError("--locality-cpu-threads must be positive")
    if args.locality_model_seed not in MODEL_SEEDS:
        raise ValueError("--locality-model-seed must be one of the frozen model seeds")
    if args.worker:
        try:
            return worker(args)
        except Exception:
            traceback.print_exc()
            return 1
    return run_master(args)


if __name__ == "__main__":
    raise SystemExit(main())
