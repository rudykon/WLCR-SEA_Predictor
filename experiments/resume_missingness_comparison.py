from __future__ import annotations

"""Complete an interrupted Revision-8 comparative-analysis run safely.

The original full run atomically emitted clean and full missingness tables
before its execution session was externally interrupted.  This helper never
recomputes or overwrites those completed tables.  It reruns only the missing
structured A3--A5 ablation and attention-routing summaries from the registered
training trace, writes them to an independent retry directory, and promotes a
fully verified staged directory only when explicitly requested.
"""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments import missingness_protocol as missingness
from experiments import train_neural_baselines as neural
from experiments import analyze_model_comparisons as revision6
from experiments import analyze_matched_missingness as comparative
from experiments import train_wlcr_sea as runner
from experiments import wlcr_sea_model as sea


RECOVERY_SCHEMA_VERSION = 1
PARTIAL_FILES = (
    "comparative_clean_accuracy.csv",
    "comparative_missingness.csv",
    "comparative_missingness_by_seed.csv",
    "comparative_missingness_per_indicator.csv",
    "comparative_missingness_per_indicator_by_seed.csv",
    "comparative_per_horizon.csv",
    "comparative_per_indicator.csv",
    "paired_cell_bootstrap.csv",
)
RETRY_FILES = (
    "structured_ablation.csv",
    "structured_ablation_by_seed.csv",
    "prior_mass.csv",
    "prior_mass_by_seed.csv",
    "expert_horizon_weights.csv",
    "expert_horizon_weights_by_seed.csv",
    "recovery_summary.json",
)
ABLATION_CONDITIONS = (
    ("mcar", 0.0),
    ("block", 0.2),
    ("recent_tail", 0.2),
    ("asynchronous", 0.2),
    ("block", 0.5),
)
ROUTING_CONDITIONS = ABLATION_CONDITIONS


def resolve_revision8_path(root: Path, value: str | Path, *, strict: bool) -> Path:
    path = comparative.resolve_inside(root, value, strict=strict)
    allowed = (root / "artifacts/revision8").resolve(strict=False)
    if not path.is_relative_to(allowed):
        raise ValueError("Revision-8 recovery paths must remain under artifacts/revision8")
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def required_files(directory: Path, names: Sequence[str]) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files in {directory}: {missing}")


def a6_ablation_rows(partial: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected = {
        (mechanism, rate, seed)
        for mechanism, rate in ABLATION_CONDITIONS
        for seed in comparative.CORRUPTION_SEEDS
    }
    seen: set[tuple[str, float, int]] = set()
    for row in read_csv_rows(partial / "comparative_missingness_by_seed.csv"):
        key = (
            str(row["mechanism"]),
            float(row["requested_rate"]),
            int(row["corruption_seed"]),
        )
        if row["method"] != "A6_mixed_aug" or key not in expected:
            continue
        if key in seen:
            raise ValueError(f"duplicate A6 ablation row: {key}")
        seen.add(key)
        rows.append({**row, "variant": "A6_mixed_aug"})
    if seen != expected:
        raise ValueError(
            "partial missingness table does not contain exactly the required "
            f"A6 ablation scenarios; missing={sorted(expected - seen)}"
        )
    return rows


def ablation_rows(
    *,
    dataset: neural.CachedDataset,
    holdout: np.ndarray,
    actual: np.ndarray,
    scales: np.ndarray,
    cells: np.ndarray,
    thresholds: np.ndarray,
    source: Path,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in ("A3_hard_mask", "A4_reliability", "A5_residual"):
        models, prior = comparative.load_sea_models(source, variant, device)
        for mechanism, rate in ABLATION_CONDITIONS:
            for corruption_seed in comparative.CORRUPTION_SEEDS:
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
                _, tensors = runner.make_eval_tensors(
                    dataset, holdout, prior, additional_missing=extra
                )
                prediction, _ = comparative.predict_sea_ensemble(
                    models,
                    tensors,
                    device,
                    batch_size,
                    include_attention=False,
                )
                values, _ = comparative.metric_payload(
                    prediction, actual, scales, cells, thresholds
                )
                rows.append(
                    {
                        "variant": variant,
                        "method": variant,
                        "label": comparative.METHOD_META[variant][0],
                        "training_view": comparative.METHOD_META[variant][2],
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
    return rows


def routing_rows(
    *,
    dataset: neural.CachedDataset,
    holdout: np.ndarray,
    source: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    models, prior = comparative.load_sea_models(source, "A6_mixed_aug", device)
    prior_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for mechanism, rate in ROUTING_CONDITIONS:
        for corruption_seed in comparative.CORRUPTION_SEEDS:
            extra = missingness.global_corruption_mask(
                np.asarray(dataset.cells[holdout]),
                np.asarray(dataset.history_end_hours[holdout]),
                mechanism=mechanism,
                requested_rate=rate,
                seed=corruption_seed,
            )
            _, tensors = runner.make_eval_tensors(
                dataset, holdout, prior, additional_missing=extra
            )
            _, mean_attention = comparative.predict_sea_ensemble(
                models,
                tensors,
                device,
                batch_size,
                include_attention=True,
            )
            if mean_attention is None:
                raise RuntimeError("A6 routing replay did not return attention")
            scenario = comparative.scenario_name(mechanism, rate)
            prior_rows.append(
                {
                    "scenario": scenario,
                    "mechanism": mechanism,
                    "requested_rate": rate,
                    "corruption_seed": corruption_seed,
                    "mean_prior_mass": float(np.mean(mean_attention[..., 7])),
                    "p90_prior_mass": float(np.quantile(mean_attention[..., 7], 0.9)),
                    "mean_effective_support": float(
                        np.mean(np.sum(mean_attention > 1e-6, axis=-1))
                    ),
                }
            )
            for metric, metric_name in enumerate(sea.METRIC_NAMES):
                for horizon in range(sea.FORECAST_HOURS):
                    for expert, expert_name in enumerate(sea.EXPERT_NAMES):
                        weight_rows.append(
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
                                        mean_attention[:, horizon, metric, expert]
                                    )
                                ),
                            }
                        )
    for model in models:
        model.to("cpu")
    torch.cuda.empty_cache()
    return prior_rows, weight_rows


def aggregate_ablation(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return comparative.aggregate_seed_rows(
        rows,
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


def recovery_payload(
    *,
    partial: Path,
    dataset_report: Mapping[str, object],
    thresholds: np.ndarray,
    input_before: str,
    input_after: str,
    structured_rows: int,
    prior_rows: int,
    weight_rows: int,
) -> dict[str, object]:
    smoke_path = partial.parent / "comparative_analysis_smoke" / "summary.json"
    if not smoke_path.is_file():
        raise FileNotFoundError(f"missing validated smoke summary: {smoke_path}")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("finals_test_opened") is not False:
        raise ValueError("smoke run does not prove finals-test isolation")
    clean = read_csv_rows(partial / "comparative_clean_accuracy.csv")
    bootstrap = read_csv_rows(partial / "paired_cell_bootstrap.csv")
    missing_raw = read_csv_rows(partial / "comparative_missingness_by_seed.csv")
    missing_aggregate = read_csv_rows(partial / "comparative_missingness.csv")
    indicator_aggregate = read_csv_rows(
        partial / "comparative_missingness_per_indicator.csv"
    )
    return {
        "schema_version": comparative.SCHEMA_VERSION,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "primary_variant": "A6_mixed_aug",
        "original_wlcr_in_main_table": True,
        "recovery": {
            "recovery_schema_version": RECOVERY_SCHEMA_VERSION,
            "reason": "execution session interrupted after full missingness tables were atomically written",
            "partial_files_reused": list(PARTIAL_FILES),
            "recomputed_outputs": list(RETRY_FILES[:-1]),
            "independent_retry_required": True,
        },
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
            "corruption_seeds": list(comparative.CORRUPTION_SEEDS),
            "unique_cell_time_and_window_exposure_rates_reported": True,
        },
        "bootstrap_estimand": "macro_over_indicator_wape",
        "bootstrap_replicates": 5000,
        "inference_batch_size": 256,
        "shared_corruption_fill_view_across_neural_normalizations": True,
        "clean_replay_abs_tolerance": comparative.CLEAN_REPLAY_ABS_TOLERANCE,
        "clean_replay_max_absolute_difference": smoke[
            "clean_replay_max_absolute_difference"
        ],
        "clean_replay_evidence_source": str(smoke_path),
        "clean_accuracy": {row["method"]: row for row in clean},
        "paired_bootstrap": bootstrap,
        "missingness_raw_rows": len(missing_raw),
        "missingness_aggregate_rows": len(missing_aggregate),
        "per_indicator_missingness_rows": len(indicator_aggregate),
        "structured_ablation_rows": structured_rows,
        "prior_mass_raw_rows": prior_rows,
        "expert_horizon_weight_raw_rows": weight_rows,
        "dataset_report": dataset_report,
        "frozen_training_thresholds": thresholds.tolist(),
        "finals_test_opened": False,
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
    }


def run_recovery(args: argparse.Namespace) -> int:
    root = runner.project_root()
    partial = resolve_revision8_path(root, args.partial, strict=True)
    source = resolve_revision8_path(root, args.source, strict=True)
    output = resolve_revision8_path(root, args.output, strict=False)
    required_files(partial, PARTIAL_FILES)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"retry output already exists and is non-empty: {output}")
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
    thresholds = sea.frozen_low_activity_thresholds(
        dataset.targets, dataset.target_masks, final_train
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Revision-8 recovery requires one CUDA device")
    device = torch.device(f"cuda:{args.gpu_device}")

    raw = a6_ablation_rows(partial)
    raw.extend(
        ablation_rows(
            dataset=dataset,
            holdout=holdout,
            actual=actual,
            scales=scales,
            cells=cells,
            thresholds=thresholds,
            source=source,
            device=device,
            batch_size=args.batch_size,
        )
    )
    aggregate = aggregate_ablation(raw)
    runner.atomic_csv(output / "structured_ablation_by_seed.csv", raw)
    runner.atomic_csv(output / "structured_ablation.csv", aggregate)

    prior_rows, weight_rows = routing_rows(
        dataset=dataset,
        holdout=holdout,
        source=source,
        device=device,
        batch_size=args.batch_size,
    )
    runner.atomic_csv(output / "prior_mass_by_seed.csv", prior_rows)
    runner.atomic_csv(
        output / "prior_mass.csv",
        comparative.aggregate_seed_rows(
            prior_rows,
            group_fields=("scenario", "mechanism", "requested_rate"),
            numeric_fields=(
                "mean_prior_mass",
                "p90_prior_mass",
                "mean_effective_support",
            ),
        ),
    )
    runner.atomic_csv(output / "expert_horizon_weights_by_seed.csv", weight_rows)
    runner.atomic_csv(
        output / "expert_horizon_weights.csv",
        comparative.aggregate_seed_rows(
            weight_rows,
            group_fields=(
                "scenario",
                "mechanism",
                "requested_rate",
                "indicator",
                "horizon",
                "expert",
            ),
            numeric_fields=("mean_weight",),
        ),
    )
    input_after = neural.sha256_file(train_path)
    if input_before != input_after:
        raise RuntimeError("registered training file changed during Revision-8 recovery")
    payload = recovery_payload(
        partial=partial,
        dataset_report=dataset_report,
        thresholds=thresholds,
        input_before=input_before,
        input_after=input_after,
        structured_rows=len(aggregate),
        prior_rows=len(prior_rows),
        weight_rows=len(weight_rows),
    )
    runner.atomic_json(output / "recovery_summary.json", payload)
    required_files(output, RETRY_FILES)
    return 0


def promote(args: argparse.Namespace) -> int:
    root = runner.project_root()
    partial = resolve_revision8_path(root, args.partial, strict=True)
    retry = resolve_revision8_path(root, args.output, strict=True)
    backup = resolve_revision8_path(root, args.backup, strict=False)
    required_files(partial, PARTIAL_FILES)
    required_files(retry, RETRY_FILES)
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite recovery backup: {backup}")
    summary = json.loads((retry / "recovery_summary.json").read_text(encoding="utf-8"))
    if summary["input_sha256_before"] != summary["input_sha256_after"]:
        raise ValueError("recovery summary reports a training-input mutation")
    if summary.get("finals_test_opened") is not False:
        raise ValueError("recovery summary does not prove finals-test isolation")
    staging = partial.parent / "comparative_analysis_recovery_staging"
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite recovery staging: {staging}")
    staging.mkdir(parents=False)
    for name in PARTIAL_FILES:
        shutil.copy2(partial / name, staging / name)
    for name in RETRY_FILES[:-1]:
        shutil.copy2(retry / name, staging / name)
    runner.atomic_json(staging / "summary.json", summary)
    runner.atomic_json(staging / "manifest.json", runner.output_manifest(staging))
    required_files(staging, PARTIAL_FILES + RETRY_FILES[:-1] + ("summary.json", "manifest.json"))
    partial.rename(backup)
    staging.rename(partial)
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", default="artifacts/revision8/wlcr_sea")
    value.add_argument("--partial", default="artifacts/revision8/comparative_analysis")
    value.add_argument("--output", default="artifacts/revision8/comparative_analysis_retry")
    value.add_argument(
        "--backup",
        default="artifacts/revision8/comparative_analysis_interrupted_20260729",
    )
    value.add_argument("--gpu-device", type=int, default=3)
    value.add_argument("--batch-size", type=int, default=256)
    value.add_argument("--promote", action="store_true")
    return value


if __name__ == "__main__":
    parsed = parser().parse_args()
    raise SystemExit(promote(parsed) if parsed.promote else run_recovery(parsed))
