from __future__ import annotations

"""Independent Revision-8 paired robustness statistics.

This script recomputes the six Revision-7 robustness comparisons directly from
the registered training trace and the frozen A6, DLinear-Aug, and PatchTST-Aug
checkpoints.  It never opens the finals inference input.  For each condition,
the five model checkpoints are averaged first; each of the five deterministic
corruption masks is then evaluated separately.  The paired cell-cluster
bootstrap resamples cells once per replicate, keeps every request, horizon, and
indicator belonging to a sampled cell together, computes the macro-over-
indicator WAPE difference for each corruption seed, and averages those five
differences inside the replicate.

The result is exploratory evidence on the existing August holdout, not a
prospective confirmation.
"""

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments import missingness_protocol as missingness
from experiments import train_neural_baselines as neural
from experiments import analyze_model_comparisons as revision6
from experiments import train_wlcr_sea as runner
from experiments import wlcr_sea_model as sea


SCHEMA_VERSION = 1
MODEL_SEEDS = (42, 43, 44, 45, 46)
CORRUPTION_SEEDS = (142, 143, 144, 145, 146)
DEFAULT_OUTPUT = Path("artifacts/revision8/robustness_statistics")
DEFAULT_SEA_ROOT = Path("artifacts/revision7/wlcr_sea")
DEFAULT_NEURAL_ROOT = Path(
    "artifacts/paper_neural_baselines_v1/revision7_mixed/results"
)
CONDITIONS: tuple[tuple[str, float, str], ...] = (
    ("block", 0.20, "block_20pct"),
    ("recent_tail", 0.20, "timeline_tail_20pct"),
    ("asynchronous", 0.30, "asynchronous_30pct"),
    ("block", 0.50, "block_50pct"),
    ("recent_tail", 0.50, "timeline_tail_50pct"),
    ("asynchronous", 0.50, "asynchronous_50pct"),
)
BASELINES: tuple[tuple[str, str], ...] = (
    ("dlinear_aug", "DLinear-Aug"),
    ("patchtst_aug", "PatchTST-Aug"),
)


def resolve_inside(root: Path, value: str | Path, *, strict: bool) -> Path:
    """Resolve a repository-relative path and reject escapes."""
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=strict)
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes project root: {path}")
    return path


def resolve_output(root: Path, value: str | Path) -> Path:
    path = resolve_inside(root, value, strict=False)
    allowed = (root / "artifacts/revision8").resolve(strict=False)
    if not path.is_relative_to(allowed):
        raise ValueError("Revision-8 outputs must remain under artifacts/revision8")
    return path


def checkpoint_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": int(path.stat().st_size),
        "sha256": runner.sha256_file(path),
    }


def load_sea_ensemble(
    root: Path, checkpoint_root: Path, device: torch.device
) -> tuple[list[sea.WLCRSEA], np.ndarray, list[dict[str, object]]]:
    """Load the five frozen A6 checkpoints and their common prior."""
    models: list[sea.WLCRSEA] = []
    priors: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for seed in MODEL_SEEDS:
        path = checkpoint_root / "models" / f"A6_mixed_aug_seed{seed}.pt"
        model, payload = runner.load_checkpoint(path, device)
        if payload.get("variant", {}).get("name") != "A6_mixed_aug":
            raise ValueError(f"A6 checkpoint metadata mismatch: {path}")
        if int(payload.get("seed", -1)) != seed:
            raise ValueError(f"A6 checkpoint seed mismatch: {path}")
        models.append(model)
        priors.append(np.asarray(payload["prior_log"], dtype=np.float32))
        records.append(checkpoint_record(root, path))
    if any(not np.array_equal(prior, priors[0]) for prior in priors[1:]):
        raise ValueError("A6 training priors differ across model seeds")
    return models, priors[0], records


def load_neural_ensemble(
    root: Path,
    checkpoint_root: Path,
    model_name: str,
    device: torch.device,
) -> tuple[list[torch.nn.Module], neural.Normalization, list[dict[str, object]]]:
    """Load one five-seed, 15%-mixed-augmentation neural ensemble."""
    models: list[torch.nn.Module] = []
    normalizations: list[Mapping[str, object]] = []
    records: list[dict[str, object]] = []
    for seed in MODEL_SEEDS:
        path = checkpoint_root / "models" / f"{model_name}_seed{seed}.pt"
        payload = torch.load(path, map_location="cpu")
        if payload.get("model") != model_name or int(payload.get("seed", -1)) != seed:
            raise ValueError(f"neural checkpoint metadata mismatch: {path}")
        if payload.get("augmentation") != "mixed" or not math.isclose(
            float(payload.get("augmentation_rate", -1.0)), 0.15, abs_tol=1e-12
        ):
            raise ValueError(f"neural checkpoint is not a 15% mixed model: {path}")
        model = neural.build_model(model_name, payload["config"])
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        models.append(model)
        normalizations.append(payload["normalization"])
        records.append(checkpoint_record(root, path))
    if any(item != normalizations[0] for item in normalizations[1:]):
        raise ValueError(f"{model_name} normalizations differ across model seeds")
    return models, neural.Normalization(**normalizations[0]), records


def shared_neural_request_view(
    dataset: neural.CachedDataset,
    indices: np.ndarray,
    additional_missing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the common corruption-dependent input view for neural baselines."""
    base_values = np.asarray(dataset.x_values[indices], dtype=np.float32)
    original_masks = np.asarray(dataset.x_masks[indices], dtype=bool)
    extra = np.asarray(additional_missing, dtype=bool)
    if extra.shape != original_masks.shape:
        raise ValueError("additional missingness is not aligned to neural inputs")
    masks = original_masks & ~extra
    if not np.any(extra):
        return base_values.copy(), masks
    raw = np.expm1(base_values.astype(np.float64))
    visible = np.where(masks, raw, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(visible, axis=1)
    medians = np.where(np.any(masks, axis=1), medians, 0.0)
    fallback_log = np.log1p(np.maximum(medians, 0.0)).astype(np.float32)
    values = np.where(masks, base_values, fallback_log[:, None, :])
    return np.asarray(values, dtype=np.float32), masks


def normalized_neural_inputs(
    values: np.ndarray, masks: np.ndarray, normalization: neural.Normalization
) -> torch.Tensor:
    input_mean = np.asarray(normalization.input_mean, dtype=np.float32)
    input_std = np.asarray(normalization.input_std, dtype=np.float32)
    normalized = (
        np.asarray(values, dtype=np.float32) - input_mean[None, None, :]
    ) / input_std[None, None, :]
    inputs = np.concatenate((normalized, np.asarray(masks, dtype=np.float32)), axis=2)
    return torch.from_numpy(inputs.astype(np.float32))


def predict_sea_ensemble(
    models: Sequence[sea.WLCRSEA],
    tensors: tuple[torch.Tensor, ...],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for model in models:
        output = runner.predict(
            model,
            tensors,
            device=device,
            batch_size=batch_size,
            include_audit=False,
        )
        predictions.append(np.asarray(output["prediction"], dtype=np.float32))
    return np.mean(np.stack(predictions, axis=0), axis=0, dtype=np.float64).astype(
        np.float32
    )


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
            model, inputs, batch_size=batch_size, device=device
        )
        predictions.append(neural.inverse_target(normalized, normalization))
    return np.mean(np.stack(predictions, axis=0), axis=0, dtype=np.float64).astype(
        np.float32
    )


def validate_prediction(name: str, prediction: np.ndarray, shape: tuple[int, ...]) -> None:
    values = np.asarray(prediction)
    if values.shape != shape:
        raise ValueError(f"{name} prediction shape {values.shape} != {shape}")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} emitted non-finite or non-positive predictions")


def macro_wape_from_components(
    numerator: np.ndarray, denominator: np.ndarray
) -> float:
    totals = np.sum(numerator, axis=0)
    scales = np.sum(denominator, axis=0)
    if np.any(scales <= 0.0):
        raise ValueError("each indicator requires a positive WAPE denominator")
    return float(np.mean(totals / scales))


def paired_multi_seed_cell_cluster_bootstrap(
    actual: np.ndarray,
    proposed_by_corruption_seed: Sequence[np.ndarray],
    baseline_by_corruption_seed: Sequence[np.ndarray],
    cells: np.ndarray,
    *,
    corruption_seeds: Sequence[int],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Paired cell-cluster CI after averaging deltas over corruption seeds.

    A corruption seed is not treated as an independent cell.  Each bootstrap
    draw samples the eligible cells once and reuses that same draw for every
    corruption seed and both methods.  Therefore all requests, horizons, and
    indicators for a sampled cell remain together, and the reported CI targets
    the mean macro-WAPE difference over the five predetermined masks.
    """
    y = np.asarray(actual, dtype=np.float64)
    proposed = [np.asarray(item, dtype=np.float64) for item in proposed_by_corruption_seed]
    baseline = [np.asarray(item, dtype=np.float64) for item in baseline_by_corruption_seed]
    cell_ids = np.asarray(cells).astype(str)
    if len(proposed) != len(baseline) or len(proposed) != len(corruption_seeds):
        raise ValueError("corruption-seed prediction lists are misaligned")
    if not proposed:
        raise ValueError("at least one corruption seed is required")
    if y.ndim != 3 or y.shape[1:] != (sea.FORECAST_HOURS, sea.TARGET_COUNT):
        raise ValueError("actual array does not match the forecast contract")
    if len(cell_ids) != len(y):
        raise ValueError("cell identities do not align with actual array")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    for index, (a, b) in enumerate(zip(proposed, baseline)):
        if a.shape != y.shape or b.shape != y.shape:
            raise ValueError(f"corruption index {index} has an invalid prediction shape")

    unique_cells = np.asarray(sorted(set(cell_ids.tolist())))
    seed_count = len(proposed)
    numerator_a = np.zeros((seed_count, len(unique_cells), sea.TARGET_COUNT), dtype=np.float64)
    numerator_b = np.zeros_like(numerator_a)
    denominator = np.zeros_like(numerator_a)
    for cell_index, cell in enumerate(unique_cells):
        selected = cell_ids == cell
        for corruption_index, (a, b) in enumerate(zip(proposed, baseline)):
            for indicator in range(sea.TARGET_COUNT):
                yy = y[selected, :, indicator]
                aa = a[selected, :, indicator]
                bb = b[selected, :, indicator]
                valid = np.isfinite(yy) & np.isfinite(aa) & np.isfinite(bb)
                observed = yy[valid]
                denominator[corruption_index, cell_index, indicator] = np.sum(
                    np.abs(observed)
                )
                numerator_a[corruption_index, cell_index, indicator] = np.sum(
                    np.abs(observed - aa[valid])
                )
                numerator_b[corruption_index, cell_index, indicator] = np.sum(
                    np.abs(observed - bb[valid])
                )

    eligible = np.any(denominator > 0.0, axis=(0, 2))
    numerator_a = numerator_a[:, eligible]
    numerator_b = numerator_b[:, eligible]
    denominator = denominator[:, eligible]
    if len(denominator[0]) == 0:
        raise ValueError("no cell has an evaluable target denominator")
    if np.any(np.sum(denominator, axis=1) <= 0.0):
        raise ValueError("every corruption seed needs positive denominators")

    proposed_wape_by_seed = np.asarray(
        [macro_wape_from_components(numerator_a[index], denominator[index]) for index in range(seed_count)],
        dtype=np.float64,
    )
    baseline_wape_by_seed = np.asarray(
        [macro_wape_from_components(numerator_b[index], denominator[index]) for index in range(seed_count)],
        dtype=np.float64,
    )
    delta_by_seed = proposed_wape_by_seed - baseline_wape_by_seed

    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sample = rng.integers(0, denominator.shape[1], size=denominator.shape[1])
        deltas = []
        for corruption_index in range(seed_count):
            sampled_denominator = np.sum(denominator[corruption_index, sample], axis=0)
            if np.any(sampled_denominator <= 0.0):
                raise RuntimeError("bootstrap sample has a zero indicator denominator")
            proposed_macro = float(
                np.mean(
                    np.sum(numerator_a[corruption_index, sample], axis=0)
                    / sampled_denominator
                )
            )
            baseline_macro = float(
                np.mean(
                    np.sum(numerator_b[corruption_index, sample], axis=0)
                    / sampled_denominator
                )
            )
            deltas.append(proposed_macro - baseline_macro)
        draws[replicate] = float(np.mean(deltas))

    return {
        "estimand": (
            "mean_over_corruption_seeds_of_paired_macro_over_indicator_wape_delta"
        ),
        "corruption_seed_aggregation": "mean delta within every cell-bootstrap replicate",
        "corruption_seed_count": int(seed_count),
        "corruption_seeds": [int(item) for item in corruption_seeds],
        "replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "clusters": int(denominator.shape[1]),
        "point_proposed_macro_wape_mean": float(np.mean(proposed_wape_by_seed)),
        "point_baseline_macro_wape_mean": float(np.mean(baseline_wape_by_seed)),
        "delta_proposed_minus_baseline_mean": float(np.mean(delta_by_seed)),
        "point_proposed_macro_wape_by_corruption_seed": proposed_wape_by_seed.tolist(),
        "point_baseline_macro_wape_by_corruption_seed": baseline_wape_by_seed.tolist(),
        "delta_proposed_minus_baseline_by_corruption_seed": delta_by_seed.tolist(),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_delta_below_zero": float(np.mean(draws < 0.0)),
    }


def stable_bootstrap_seed(base_seed: int, condition: str, baseline: str) -> int:
    payload = f"{base_seed}|{condition}|{baseline}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return int((base_seed + offset) % (2**32 - 1))


def condition_stats_row(
    mechanism: str,
    requested_rate: float,
    condition: str,
    baseline_key: str,
    baseline_label: str,
    result: Mapping[str, object],
) -> dict[str, object]:
    return {
        "condition": condition,
        "mechanism": mechanism,
        "mechanism_display": "timeline_tail" if mechanism == "recent_tail" else mechanism,
        "requested_rate": requested_rate,
        "proposed": "A6_mixed_aug",
        "proposed_label": "WLCR-SEA",
        "baseline": baseline_key,
        "baseline_label": baseline_label,
        "estimand": result["estimand"],
        "corruption_seed_aggregation": result["corruption_seed_aggregation"],
        "corruption_seed_count": result["corruption_seed_count"],
        "corruption_seeds": ",".join(str(item) for item in result["corruption_seeds"]),
        "bootstrap_replicates": result["replicates"],
        "bootstrap_seed": result["bootstrap_seed"],
        "clusters": result["clusters"],
        "a6_macro_wape_mean": result["point_proposed_macro_wape_mean"],
        "baseline_macro_wape_mean": result["point_baseline_macro_wape_mean"],
        "delta_a6_minus_baseline": result["delta_proposed_minus_baseline_mean"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "probability_delta_below_zero": result["probability_delta_below_zero"],
    }


def run(args: argparse.Namespace) -> int:
    root = runner.project_root()
    sea_root = resolve_inside(root, args.sea_root, strict=True)
    neural_root = resolve_inside(root, args.neural_root, strict=True)
    output = resolve_output(root, args.output)
    output.mkdir(parents=True, exist_ok=True)

    devices = [int(item) for item in args.gpu_devices.split(",") if item.strip()]
    if len(devices) != 3 or len(set(devices)) != 3:
        raise ValueError("provide exactly three distinct GPU device ids")
    if not torch.cuda.is_available():
        raise RuntimeError("Revision-8 robustness statistics require CUDA")
    if max(devices) >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested GPU ids {devices} exceed visible devices {torch.cuda.device_count()}"
        )
    sea_device = torch.device(f"cuda:{devices[0]}")
    dlinear_device = torch.device(f"cuda:{devices[1]}")
    patchtst_device = torch.device(f"cuda:{devices[2]}")

    dataset, dataset_report, train_path = revision6.dataset_from_registered_train()
    expected_train = (root / "data/train_data.csv").resolve(strict=True)
    if train_path.resolve(strict=True) != expected_train:
        raise RuntimeError("Revision-8 loader did not resolve the registered training trace")
    input_before = neural.sha256_file(train_path)
    fit = dataset.indices_for_dates(neural.FIT_DATES)
    inner = dataset.indices_for_dates(neural.INNER_DATES)
    holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
    if args.smoke:
        holdout = holdout[:128]
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    shape = actual.shape

    a6_models, prior, a6_records = load_sea_ensemble(root, sea_root, sea_device)
    dlinear_models, dlinear_norm, dlinear_records = load_neural_ensemble(
        root, neural_root, "dlinear", dlinear_device
    )
    patchtst_models, patchtst_norm, patchtst_records = load_neural_ensemble(
        root, neural_root, "patchtst", patchtst_device
    )

    conditions = CONDITIONS if not args.smoke else (CONDITIONS[0],)
    corruption_seeds = CORRUPTION_SEEDS if not args.smoke else (CORRUPTION_SEEDS[0],)
    paired_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    detailed_results: list[dict[str, object]] = []

    for mechanism, requested_rate, condition in conditions:
        predictions: dict[str, list[np.ndarray]] = {
            "A6_mixed_aug": [],
            "dlinear_aug": [],
            "patchtst_aug": [],
        }
        condition_statistics: list[dict[str, object]] = []
        for corruption_seed in corruption_seeds:
            extra = missingness.global_corruption_mask(
                np.asarray(dataset.cells[holdout]),
                np.asarray(dataset.history_end_hours[holdout]),
                mechanism=mechanism,
                requested_rate=requested_rate,
                seed=corruption_seed,
            )
            corruption_statistics = missingness.corruption_statistics(
                np.asarray(dataset.x_masks[holdout]),
                extra,
                cells=np.asarray(dataset.cells[holdout]),
                history_end_hours=np.asarray(dataset.history_end_hours[holdout]),
                mechanism=mechanism,
                requested_rate=requested_rate,
                seed=corruption_seed,
            )
            condition_statistics.append(corruption_statistics)
            _, sea_tensors = runner.make_eval_tensors(
                dataset, holdout, prior, additional_missing=extra
            )
            sea_prediction = predict_sea_ensemble(
                a6_models, sea_tensors, sea_device, args.batch_size
            )
            neural_values, neural_masks = shared_neural_request_view(
                dataset, holdout, extra
            )
            dlinear_prediction = predict_neural_ensemble(
                dlinear_models,
                normalized_neural_inputs(neural_values, neural_masks, dlinear_norm),
                dlinear_norm,
                dlinear_device,
                args.batch_size,
            )
            patchtst_prediction = predict_neural_ensemble(
                patchtst_models,
                normalized_neural_inputs(neural_values, neural_masks, patchtst_norm),
                patchtst_norm,
                patchtst_device,
                args.batch_size,
            )
            for name, prediction in (
                ("A6_mixed_aug", sea_prediction),
                ("dlinear_aug", dlinear_prediction),
                ("patchtst_aug", patchtst_prediction),
            ):
                validate_prediction(f"{condition}/{corruption_seed}/{name}", prediction, shape)
                predictions[name].append(prediction)

        for baseline_key, baseline_label in BASELINES:
            bootstrap_seed = stable_bootstrap_seed(
                args.bootstrap_seed, condition, baseline_key
            )
            result = paired_multi_seed_cell_cluster_bootstrap(
                actual,
                predictions["A6_mixed_aug"],
                predictions[baseline_key],
                cells,
                corruption_seeds=corruption_seeds,
                replicates=args.bootstrap_replicates,
                seed=bootstrap_seed,
            )
            paired_rows.append(
                condition_stats_row(
                    mechanism,
                    requested_rate,
                    condition,
                    baseline_key,
                    baseline_label,
                    result,
                )
            )
            for index, corruption_seed in enumerate(corruption_seeds):
                seed_rows.append(
                    {
                        "condition": condition,
                        "mechanism": mechanism,
                        "mechanism_display": (
                            "timeline_tail" if mechanism == "recent_tail" else mechanism
                        ),
                        "requested_rate": requested_rate,
                        "corruption_seed": corruption_seed,
                        "proposed": "A6_mixed_aug",
                        "baseline": baseline_key,
                        "a6_macro_wape": result[
                            "point_proposed_macro_wape_by_corruption_seed"
                        ][index],
                        "baseline_macro_wape": result[
                            "point_baseline_macro_wape_by_corruption_seed"
                        ][index],
                        "delta_a6_minus_baseline": result[
                            "delta_proposed_minus_baseline_by_corruption_seed"
                        ][index],
                    }
                )
            detailed_results.append(
                {
                    "condition": condition,
                    "mechanism": mechanism,
                    "requested_rate": requested_rate,
                    "baseline": baseline_key,
                    "baseline_label": baseline_label,
                    **result,
                }
            )

        if len(condition_statistics) != len(corruption_seeds):
            raise RuntimeError("one corruption report is required for every seed")

    runner.atomic_csv(output / "paired_cell_bootstrap.csv", paired_rows)
    runner.atomic_csv(output / "corruption_seed_wape.csv", seed_rows)

    input_after = neural.sha256_file(train_path)
    if input_before != input_after:
        raise RuntimeError("registered training trace changed during Revision-8 analysis")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "purpose": "paired robustness CIs for Revision-7 P0#9 crossover and severe-loss claims",
        "registered_train_file": "data/train_data.csv",
        "registered_train_sha256_before": input_before,
        "registered_train_sha256_after": input_after,
        "finals_test_opened": False,
        "models": {
            "A6_mixed_aug": a6_records,
            "dlinear_aug": dlinear_records,
            "patchtst_aug": patchtst_records,
        },
        "model_seeds": list(MODEL_SEEDS),
        "corruption_seeds": list(corruption_seeds),
        "conditions": [
            {"mechanism": mechanism, "requested_rate": rate, "name": name}
            for mechanism, rate, name in conditions
        ],
        "inference": {
            "gpu_devices": devices,
            "batch_size": args.batch_size,
            "five_model_seed_predictions_averaged_before_scoring": True,
            "shared_corruption_dependent_fill_view_for_neural_baselines": True,
        },
        "bootstrap_protocol": {
            "metric": "macro-over-indicator WAPE",
            "replicates": args.bootstrap_replicates,
            "base_seed": args.bootstrap_seed,
            "cluster": "cell",
            "within_cluster_preserved": "all requests, horizons, and indicators",
            "corruption_seed_handling": (
                "compute a paired A6-minus-baseline delta per corruption seed and "
                "average those deltas inside every bootstrap replicate"
            ),
            "interval": "percentile 95% interval",
        },
        "dataset_report": dataset_report,
        "results": detailed_results,
    }
    runner.atomic_json(output / "summary.json", summary)
    runner.atomic_json(output / "manifest.json", runner.output_manifest(output))
    print(json.dumps({"status": "complete", "output": str(output.relative_to(root))}))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--sea-root", default=str(DEFAULT_SEA_ROOT))
    value.add_argument("--neural-root", default=str(DEFAULT_NEURAL_ROOT))
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    value.add_argument("--gpu-devices", default="0,1,2")
    value.add_argument("--batch-size", type=int, default=256)
    value.add_argument("--bootstrap-replicates", type=int, default=5000)
    value.add_argument("--bootstrap-seed", type=int, default=20260729)
    value.add_argument("--smoke", action="store_true")
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
