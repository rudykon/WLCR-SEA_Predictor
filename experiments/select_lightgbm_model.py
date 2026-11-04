from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from Model.traffic_window_forecasting import (
    BacktestExample,
    BaselineConfig,
    ForecastRow,
    TrafficRow,
    baseline_candidates,
    build_training_backtests,
    read_traffic,
    seasonal_forecast,
)
from Model.lightgbm_feature_baseline import MatrixBundle, build_matrix, load_parameters, load_weather
from experiments.train_lightgbm_baseline import (
    MODEL_PARAMS,
    SEED,
    cell_fold,
    cluster_bootstrap,
    feature_columns,
    feature_names,
    predict_boosters,
    score_dict,
    select_rounds,
    train_or_load_boosters,
)


SCHEMA_VERSION = 2
DEFAULT_OUTPUT = "artifacts/paper_strict_nested_gpu4_v2"
OUTER_FOLDS = tuple(range(5))
PROPOSED_VARIANTS = (
    "full",
    "no_weather",
    "no_baseline",
    "no_missingness",
    "no_static",
    "target_only",
)
IMPLEMENTATION_PATHS = (
    "experiments/select_lightgbm_model.py",
    "experiments/train_lightgbm_baseline.py",
    "Model/lightgbm_feature_baseline.py",
    "Model/traffic_window_forecasting.py",
)
SOURCE_PATHS = (
    "data/train_data.csv",
    "data/parameter.csv",
    "data/weather.csv",
)


@dataclass(frozen=True)
class TemporalProtocol:
    fit_dates: frozenset[date]
    inner_dates: frozenset[date]
    development_dates: frozenset[date]
    lockbox_dates: frozenset[date]

    @property
    def selection_train_dates(self) -> frozenset[date]:
        return self.fit_dates | self.inner_dates

    @property
    def prelock_dates(self) -> frozenset[date]:
        return self.selection_train_dates | self.development_dates

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "fit": sorted(map(str, self.fit_dates)),
            "inner": sorted(map(str, self.inner_dates)),
            "selection_train": sorted(map(str, self.selection_train_dates)),
            "development": sorted(map(str, self.development_dates)),
            "prelock": sorted(map(str, self.prelock_dates)),
            "lockbox": sorted(map(str, self.lockbox_dates)),
        }


@dataclass(frozen=True)
class OuterFoldPlan:
    fold: int
    validation_cells: tuple[str, ...]
    outer_training_cells: tuple[str, ...]
    seasonal_selection: tuple[BacktestExample, ...]
    selection_train: tuple[BacktestExample, ...]
    selection_validation: tuple[BacktestExample, ...]
    final_train: tuple[BacktestExample, ...]
    outer_validation: tuple[BacktestExample, ...]


@dataclass(frozen=True)
class TunedCandidate:
    role: str
    variant: str
    columns: tuple[np.ndarray, ...]
    rounds: tuple[int, ...]
    development_metrics: dict[str, object]
    tuning_seconds: float
    training_seconds: float
    prediction_seconds: float
    model_bytes: int
    cache_signature: str
    cache_hit: bool

    def report(self) -> dict[str, object]:
        return {
            "role": self.role,
            "variant": self.variant,
            "rounds": list(self.rounds),
            "feature_counts": [len(value) for value in self.columns],
            "development": self.development_metrics,
            "round_selection_seconds": self.tuning_seconds,
            "development_training_seconds": self.training_seconds,
            "development_prediction_seconds": self.prediction_seconds,
            "development_model_bytes": self.model_bytes,
            "cache_signature": self.cache_signature,
            "cache_hit": self.cache_hit,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered_hashes(root: Path, paths: Sequence[str]) -> dict[str, str]:
    return {relative: sha256_file(root / relative) for relative in paths}


def stable_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_temporal_protocol(examples: Sequence[BacktestExample]) -> TemporalProtocol:
    dates = sorted({example.window.target_start.date() for example in examples})
    if len(dates) < 16:
        raise ValueError(f"strict nested evaluation needs 16 target dates, found {len(dates)}")
    return TemporalProtocol(
        fit_dates=frozenset(dates[:7]),
        inner_dates=frozenset(dates[7:10]),
        development_dates=frozenset(dates[10:13]),
        lockbox_dates=frozenset(dates[13:16]),
    )


def example_cells(examples: Sequence[BacktestExample]) -> set[str]:
    return {example.window.cell for example in examples}


def build_outer_fold_plan(
    examples: Sequence[BacktestExample],
    protocol: TemporalProtocol,
    fold: int,
) -> OuterFoldPlan:
    if fold not in OUTER_FOLDS:
        raise ValueError(f"invalid outer fold: {fold}")
    lockbox = tuple(
        example
        for example in examples
        if example.window.target_start.date() in protocol.lockbox_dates
    )
    validation_cells = tuple(
        sorted({item.window.cell for item in lockbox if cell_fold(item.window.cell) == fold})
    )
    if not validation_cells:
        raise ValueError(f"outer fold {fold} has no validation cells")
    validation = set(validation_cells)

    def training_subset(selected_dates: frozenset[date]) -> tuple[BacktestExample, ...]:
        return tuple(
            example
            for example in examples
            if example.window.target_start.date() in selected_dates
            and example.window.cell not in validation
        )

    seasonal_selection = training_subset(protocol.inner_dates)
    selection_train = training_subset(protocol.selection_train_dates)
    selection_validation = training_subset(protocol.development_dates)
    final_train = training_subset(protocol.prelock_dates)
    outer_validation = tuple(item for item in lockbox if item.window.cell in validation)
    plan = OuterFoldPlan(
        fold=fold,
        validation_cells=validation_cells,
        outer_training_cells=tuple(sorted(example_cells(final_train))),
        seasonal_selection=seasonal_selection,
        selection_train=selection_train,
        selection_validation=selection_validation,
        final_train=final_train,
        outer_validation=outer_validation,
    )
    validate_outer_fold_plan(plan)
    return plan


def validate_outer_fold_plan(plan: OuterFoldPlan) -> None:
    validation = set(plan.validation_cells)
    if not validation or not plan.outer_validation:
        raise ValueError(f"outer fold {plan.fold} has an empty validation layer")
    for stage, examples in (
        ("seasonal_selection", plan.seasonal_selection),
        ("selection_train", plan.selection_train),
        ("selection_validation", plan.selection_validation),
        ("final_train", plan.final_train),
    ):
        if not examples:
            raise ValueError(f"outer fold {plan.fold} has empty {stage}")
        overlap = validation & example_cells(examples)
        if overlap:
            raise RuntimeError(
                f"outer fold {plan.fold} leaks validation cells into {stage}: {sorted(overlap)}"
            )
    if example_cells(plan.outer_validation) != validation:
        raise RuntimeError(f"outer fold {plan.fold} validation cell list is incomplete")
    if validation & set(plan.outer_training_cells):
        raise RuntimeError(f"outer fold {plan.fold} training/validation overlap")


def fold_plan_report(plan: OuterFoldPlan, protocol: TemporalProtocol) -> dict[str, object]:
    validation = set(plan.validation_cells)
    stages = {}
    for name, examples in (
        ("seasonal_selection", plan.seasonal_selection),
        ("selection_train", plan.selection_train),
        ("selection_validation", plan.selection_validation),
        ("final_train", plan.final_train),
        ("outer_validation", plan.outer_validation),
    ):
        cells = tuple(sorted(example_cells(examples)))
        stages[name] = {
            "windows": len(examples),
            "forecast_rows": len(examples) * 24,
            "cells": len(cells),
            "cell_list_sha256": stable_digest(cells),
            "dates": sorted({str(item.window.target_start.date()) for item in examples}),
            "validation_cell_overlap": sorted(validation & set(cells)),
        }
    return {
        "fold": plan.fold,
        "fold_assignment": "sha256(cell_id).digest()[0] % 5",
        "validation_cells": list(plan.validation_cells),
        "validation_cell_count": len(plan.validation_cells),
        "validation_cells_sha256": stable_digest(plan.validation_cells),
        "outer_training_cells": list(plan.outer_training_cells),
        "outer_training_cell_count": len(plan.outer_training_cells),
        "outer_training_cells_sha256": stable_digest(plan.outer_training_cells),
        "temporal_protocol": protocol.as_dict(),
        "stages": stages,
        "leakage_checks": {
            "seasonal_selection_disjoint": not validation.intersection(
                example_cells(plan.seasonal_selection)
            ),
            "selection_train_disjoint": not validation.intersection(
                example_cells(plan.selection_train)
            ),
            "selection_validation_disjoint": not validation.intersection(
                example_cells(plan.selection_validation)
            ),
            "final_train_disjoint": not validation.intersection(example_cells(plan.final_train)),
            "outer_validation_exact": example_cells(plan.outer_validation) == validation,
        },
    }


def score_baseline_examples(
    examples: Sequence[BacktestExample], config: BaselineConfig
) -> dict[str, object]:
    actuals: list[TrafficRow] = []
    predictions: list[ForecastRow] = []
    for example in examples:
        actuals.extend(example.actuals)
        predictions.extend(seasonal_forecast(example.window, config))
    return score_dict(actuals, predictions)


def baseline_config_report(config: BaselineConfig) -> dict[str, object]:
    return {
        "name": config.name,
        "weights": list(config.weights),
        "scales": list(config.scales),
    }


def select_fold_baseline(
    examples: Sequence[BacktestExample],
    candidates: Sequence[BaselineConfig] | None = None,
    scorer: Callable[
        [Sequence[BacktestExample], BaselineConfig], dict[str, object]
    ] = score_baseline_examples,
) -> tuple[BaselineConfig, dict[str, object]]:
    if not examples:
        raise ValueError("seasonal selection requires outer-training inner examples")
    candidate_list = tuple(
        baseline_candidates() if candidates is None else candidates
    )
    if not candidate_list:
        raise ValueError("seasonal selection requires at least one candidate")
    reports: list[dict[str, object]] = []
    metrics_by_candidate: list[dict[str, object]] = []
    for config in candidate_list:
        metrics = scorer(examples, config)
        metrics_by_candidate.append(metrics)
        reports.append(
            {
                "config": baseline_config_report(config),
                "inner_metrics": metrics,
            }
        )

    def key(index: int) -> tuple[float, float, int]:
        metrics = metrics_by_candidate[index]
        mean_mape = metrics.get("mean_mape")
        return (
            float(metrics["mape_auc"]),
            -float("inf") if mean_mape is None else -float(mean_mape),
            -index,
        )

    selected_index = max(range(len(candidate_list)), key=key)
    selected = candidate_list[selected_index]
    return selected, {
        "selection_rule": (
            "highest inner-layer MAPEAUC among seasonal candidates using only "
            "outer-training cells; lower mean MAPE and candidate order break ties"
        ),
        "selection_cells": sorted(example_cells(examples)),
        "selection_cells_sha256": stable_digest(
            tuple(sorted(example_cells(examples)))
        ),
        "selection_dates": sorted(
            {str(item.window.target_start.date()) for item in examples}
        ),
        "selected": baseline_config_report(selected),
        "selected_inner_metrics": metrics_by_candidate[selected_index],
        "candidates": reports,
    }


def cache_payload(
    *,
    fold: int,
    phase: str,
    role: str,
    variant: str,
    rounds: Sequence[int],
    columns: Sequence[np.ndarray],
    training_cells: Sequence[str],
    validation_cells: Sequence[str],
    training_dates: Sequence[str],
    feature_schema: Sequence[str],
    baseline: BaselineConfig,
    source_hashes: dict[str, str],
    implementation_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fold": fold,
        "phase": phase,
        "role": role,
        "variant": variant,
        "rounds": list(map(int, rounds)),
        "feature_columns": [[int(index) for index in value] for value in columns],
        "feature_schema": list(feature_schema),
        "training_cells": list(training_cells),
        "validation_cells": list(validation_cells),
        "training_dates": list(training_dates),
        "baseline": {
            "name": baseline.name,
            "weights": list(baseline.weights),
            "scales": list(baseline.scales),
        },
        "model_params": MODEL_PARAMS,
        "source_sha256": source_hashes,
        "implementation_sha256": implementation_hashes,
    }


def cache_directory(
    model_root: Path,
    fold: int,
    phase: str,
    role: str,
    variant: str,
    signature: str,
) -> Path:
    return (
        model_root
        / "strict_nested_cell_disjoint_v2"
        / f"fold_{fold}"
        / phase
        / f"{role}_{variant}_{signature[:16]}"
    )


def evaluate_development_candidate(
    *,
    fold: int,
    role: str,
    variant: str,
    selection_train: MatrixBundle,
    selection_validation: MatrixBundle,
    columns: tuple[np.ndarray, ...],
    model_root: Path,
    training_cells: Sequence[str],
    validation_cells: Sequence[str],
    training_dates: Sequence[str],
    feature_schema: Sequence[str],
    baseline: BaselineConfig,
    source_hashes: dict[str, str],
    implementation_hashes: dict[str, str],
    round_selector: Callable[
        [MatrixBundle, MatrixBundle, Sequence[np.ndarray]], list[int]
    ] = select_rounds,
    trainer: Callable = train_or_load_boosters,
    predictor: Callable = predict_boosters,
) -> TunedCandidate:
    started = time.perf_counter()
    rounds = tuple(
        int(value) for value in round_selector(selection_train, selection_validation, columns)
    )
    tuning_seconds = time.perf_counter() - started
    if len(rounds) != 4 or any(value <= 0 for value in rounds):
        raise RuntimeError(f"invalid independently selected rounds for {role}/{variant}: {rounds}")
    signature_payload = cache_payload(
        fold=fold,
        phase="development_selection",
        role=role,
        variant=variant,
        rounds=rounds,
        columns=columns,
        training_cells=training_cells,
        validation_cells=validation_cells,
        training_dates=training_dates,
        feature_schema=feature_schema,
        baseline=baseline,
        source_hashes=source_hashes,
        implementation_hashes=implementation_hashes,
    )
    signature = stable_digest(signature_payload)
    cache_dir = cache_directory(
        model_root, fold, "development_selection", role, variant, signature
    )
    cache_hit = all((cache_dir / f"metric_{metric}.txt").exists() for metric in range(4))
    boosters, training_seconds, model_bytes = trainer(
        selection_train, columns, rounds, cache_dir
    )
    predictions, prediction_seconds = predictor(
        boosters, selection_validation, columns
    )
    metrics = score_dict(selection_validation.actuals, predictions)
    write_json(
        cache_dir / "cache_audit.json",
        {
            "signature": signature,
            "cache_hit_before_run": cache_hit,
            "signature_payload": signature_payload,
        },
    )
    return TunedCandidate(
        role=role,
        variant=variant,
        columns=columns,
        rounds=rounds,
        development_metrics=metrics,
        tuning_seconds=tuning_seconds,
        training_seconds=float(training_seconds),
        prediction_seconds=float(prediction_seconds),
        model_bytes=int(model_bytes),
        cache_signature=signature,
        cache_hit=cache_hit,
    )


def choose_proposed_candidate(candidates: Sequence[TunedCandidate]) -> TunedCandidate:
    if not candidates:
        raise ValueError("at least one proposed candidate is required")

    def key(index: int) -> tuple[float, float, int]:
        metrics = candidates[index].development_metrics
        mean_mape = metrics.get("mean_mape")
        return (
            float(metrics["mape_auc"]),
            -float("inf") if mean_mape is None else -float(mean_mape),
            -index,
        )

    return candidates[max(range(len(candidates)), key=key)]


def tune_fold_candidates(
    *,
    fold: int,
    proposed_variants: Sequence[str],
    feature_schema: Sequence[str],
    selection_train: MatrixBundle,
    selection_validation: MatrixBundle,
    model_root: Path,
    training_cells: Sequence[str],
    validation_cells: Sequence[str],
    training_dates: Sequence[str],
    baseline: BaselineConfig,
    source_hashes: dict[str, str],
    implementation_hashes: dict[str, str],
    evaluator: Callable[..., TunedCandidate] = evaluate_development_candidate,
) -> tuple[TunedCandidate, TunedCandidate, list[TunedCandidate]]:
    candidates = []
    shared = {
        "fold": fold,
        "selection_train": selection_train,
        "selection_validation": selection_validation,
        "model_root": model_root,
        "training_cells": training_cells,
        "validation_cells": validation_cells,
        "training_dates": training_dates,
        "feature_schema": feature_schema,
        "baseline": baseline,
        "source_hashes": source_hashes,
        "implementation_hashes": implementation_hashes,
    }
    for variant in proposed_variants:
        columns = tuple(
            feature_columns(feature_schema, variant, metric) for metric in range(4)
        )
        candidates.append(
            evaluator(
                **shared,
                role="proposed",
                variant=variant,
                columns=columns,
            )
        )
    selected = choose_proposed_candidate(candidates)
    plain_columns = tuple(
        feature_columns(feature_schema, "plain_lgbm", metric) for metric in range(4)
    )
    plain = evaluator(
        **shared,
        role="plain_lgbm",
        variant="plain_lgbm",
        columns=plain_columns,
    )
    return selected, plain, candidates


def fit_outer_model(
    *,
    fold: int,
    candidate: TunedCandidate,
    final_train: MatrixBundle,
    outer_validation: MatrixBundle,
    model_root: Path,
    training_cells: Sequence[str],
    validation_cells: Sequence[str],
    training_dates: Sequence[str],
    feature_schema: Sequence[str],
    baseline: BaselineConfig,
    source_hashes: dict[str, str],
    implementation_hashes: dict[str, str],
    trainer: Callable = train_or_load_boosters,
    predictor: Callable = predict_boosters,
) -> tuple[list[ForecastRow], dict[str, object]]:
    signature_payload = cache_payload(
        fold=fold,
        phase="outer_final_fit",
        role=candidate.role,
        variant=candidate.variant,
        rounds=candidate.rounds,
        columns=candidate.columns,
        training_cells=training_cells,
        validation_cells=validation_cells,
        training_dates=training_dates,
        feature_schema=feature_schema,
        baseline=baseline,
        source_hashes=source_hashes,
        implementation_hashes=implementation_hashes,
    )
    signature = stable_digest(signature_payload)
    cache_dir = cache_directory(
        model_root,
        fold,
        "outer_final_fit",
        candidate.role,
        candidate.variant,
        signature,
    )
    cache_hit = all((cache_dir / f"metric_{metric}.txt").exists() for metric in range(4))
    boosters, training_seconds, model_bytes = trainer(
        final_train, candidate.columns, candidate.rounds, cache_dir
    )
    predictions, prediction_seconds = predictor(
        boosters, outer_validation, candidate.columns
    )
    write_json(
        cache_dir / "cache_audit.json",
        {
            "signature": signature,
            "cache_hit_before_run": cache_hit,
            "signature_payload": signature_payload,
        },
    )
    return predictions, {
        "role": candidate.role,
        "variant": candidate.variant,
        "rounds": list(candidate.rounds),
        "feature_counts": [len(value) for value in candidate.columns],
        "training_seconds": float(training_seconds),
        "prediction_seconds": float(prediction_seconds),
        "model_bytes": int(model_bytes),
        "cache_signature": signature,
        "cache_hit": cache_hit,
    }


def write_fold_predictions(
    path: Path,
    fold: int | None,
    actuals: Sequence[TrafficRow],
    plain_predictions: Sequence[ForecastRow],
    proposed_predictions: Sequence[ForecastRow],
) -> None:
    if not (len(actuals) == len(plain_predictions) == len(proposed_predictions)):
        raise RuntimeError("fold prediction lengths do not match")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = ["fold", "timestamp", "cell"]
    fields += [f"actual_m{metric}" for metric in range(4)]
    fields += [f"plain_m{metric}" for metric in range(4)]
    fields += [f"proposed_m{metric}" for metric in range(4)]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for actual, plain, proposed in zip(actuals, plain_predictions, proposed_predictions):
            key = (actual.cell, actual.timestamp)
            if key != (plain.cell, plain.timestamp) or key != (
                proposed.cell,
                proposed.timestamp,
            ):
                raise RuntimeError("prediction key does not match its actual row")
            row: dict[str, object] = {
                "fold": cell_fold(actual.cell) if fold is None else fold,
                "timestamp": actual.timestamp.isoformat(sep=" "),
                "cell": actual.cell,
            }
            row.update(
                {
                    f"actual_m{metric}": (
                        "NIL" if actual.metrics[metric] is None else float(actual.metrics[metric])
                    )
                    for metric in range(4)
                }
            )
            row.update(
                {f"plain_m{metric}": float(plain.metrics[metric]) for metric in range(4)}
            )
            row.update(
                {
                    f"proposed_m{metric}": float(proposed.metrics[metric])
                    for metric in range(4)
                }
            )
            writer.writerow(row)
    temporary.replace(path)


def place_oof_predictions(
    actuals: Sequence[TrafficRow],
    destinations: list[ForecastRow | None],
    predictions: Sequence[ForecastRow],
) -> None:
    if len(actuals) != len(destinations):
        raise ValueError("OOF destination length mismatch")
    positions = {}
    for index, actual in enumerate(actuals):
        key = (actual.cell, actual.timestamp)
        if key in positions:
            raise RuntimeError(f"duplicate outer-evaluation key: {key}")
        positions[key] = index
    for prediction in predictions:
        key = (prediction.cell, prediction.timestamp)
        if key not in positions:
            raise RuntimeError(f"prediction outside pooled outer evaluation: {key}")
        index = positions[key]
        if destinations[index] is not None:
            raise RuntimeError(f"duplicate OOF prediction assignment: {key}")
        destinations[index] = prediction


def run_strict_outer_fold(
    *,
    plan: OuterFoldPlan,
    protocol: TemporalProtocol,
    parameters: dict[str, dict[str, float]],
    weather: dict[str, dict[str, float]],
    proposed_variants: Sequence[str],
    output: Path,
    model_root: Path,
    source_hashes: dict[str, str],
    implementation_hashes: dict[str, str],
) -> tuple[list[ForecastRow], list[ForecastRow], dict[str, object]]:
    validate_outer_fold_plan(plan)
    baseline, seasonal_report = select_fold_baseline(plan.seasonal_selection)
    feature_schema = feature_names(
        plan.selection_train[0], baseline, parameters, weather
    )
    selection_train = build_matrix(plan.selection_train, baseline, parameters, weather)
    selection_validation = build_matrix(
        plan.selection_validation, baseline, parameters, weather
    )
    final_train = build_matrix(plan.final_train, baseline, parameters, weather)
    outer_validation = build_matrix(plan.outer_validation, baseline, parameters, weather)
    selected, plain, candidates = tune_fold_candidates(
        fold=plan.fold,
        proposed_variants=proposed_variants,
        feature_schema=feature_schema,
        selection_train=selection_train,
        selection_validation=selection_validation,
        model_root=model_root,
        training_cells=tuple(sorted(example_cells(plan.selection_train))),
        validation_cells=plan.validation_cells,
        training_dates=sorted(map(str, protocol.selection_train_dates)),
        baseline=baseline,
        source_hashes=source_hashes,
        implementation_hashes=implementation_hashes,
    )
    final_shared = {
        "fold": plan.fold,
        "final_train": final_train,
        "outer_validation": outer_validation,
        "model_root": model_root,
        "training_cells": tuple(sorted(example_cells(plan.final_train))),
        "validation_cells": plan.validation_cells,
        "training_dates": sorted(map(str, protocol.prelock_dates)),
        "feature_schema": feature_schema,
        "baseline": baseline,
        "source_hashes": source_hashes,
        "implementation_hashes": implementation_hashes,
    }
    proposed_predictions, proposed_fit = fit_outer_model(
        **final_shared, candidate=selected
    )
    plain_predictions, plain_fit = fit_outer_model(**final_shared, candidate=plain)
    proposed_metrics = score_dict(outer_validation.actuals, proposed_predictions)
    plain_metrics = score_dict(outer_validation.actuals, plain_predictions)
    prediction_path = (
        output
        / "strict_nested_cell_disjoint"
        / f"fold_{plan.fold}"
        / "strict_nested_predictions.csv"
    )
    write_fold_predictions(
        prediction_path,
        plan.fold,
        outer_validation.actuals,
        plain_predictions,
        proposed_predictions,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "strict_nested": True,
        **fold_plan_report(plan, protocol),
        "selection_rule": (
            "within this outer-training-cell subset, select the seasonal configuration "
            "on the chronological inner layer; independently early-stop every proposed "
            "feature variant and plain LightGBM on the chronological development layer; "
            "select the proposed variant by development MAPEAUC; refit both models on "
            "outer-training prelock windows before outer validation"
        ),
        "seasonal_selection": seasonal_report,
        "proposed_candidates": [candidate.report() for candidate in candidates],
        "selected_variant": selected.variant,
        "selected_rounds": list(selected.rounds),
        "plain_rounds": list(plain.rounds),
        "proposed_final_fit": proposed_fit,
        "plain_final_fit": plain_fit,
        "selected_mape_auc": proposed_metrics["mape_auc"],
        "plain_mape_auc": plain_metrics["mape_auc"],
        "gain": float(proposed_metrics["mape_auc"])
        - float(plain_metrics["mape_auc"]),
        "selected_metrics": proposed_metrics,
        "plain_metrics": plain_metrics,
        "per_fold_metric_scope": (
            "diagnostic only; authoritative pooled metrics concatenate all five OOF folds "
            "before complete-case and 5th-percentile filtering"
        ),
        "prediction_file": str(prediction_path.relative_to(output)),
        "prediction_rows": len(proposed_predictions),
    }
    write_json(
        output
        / "strict_nested_cell_disjoint"
        / f"fold_{plan.fold}"
        / "strict_nested_fold_report.json",
        report,
    )
    return proposed_predictions, plain_predictions, report


def run_temporal_selection(
    *,
    output: Path,
    model_root: Path,
    baseline: BaselineConfig,
    parameters: dict[str, dict[str, float]],
    weather: dict[str, dict[str, float]],
    feature_schema: Sequence[str],
    selection_train_examples: Sequence[BacktestExample],
    development_examples: Sequence[BacktestExample],
    prelock_examples: Sequence[BacktestExample],
    lockbox_examples: Sequence[BacktestExample],
) -> dict[str, object]:
    selection_train = build_matrix(
        selection_train_examples, baseline, parameters, weather
    )
    development = build_matrix(development_examples, baseline, parameters, weather)
    prelock = build_matrix(prelock_examples, baseline, parameters, weather)
    lockbox = build_matrix(lockbox_examples, baseline, parameters, weather)
    full_columns = tuple(
        feature_columns(feature_schema, "full", metric) for metric in range(4)
    )
    rounds = select_rounds(selection_train, development, full_columns)
    write_json(
        output / "strict_nested_temporal_round_selection.json",
        {
            "schema_version": SCHEMA_VERSION,
            "selection_scope": "self_contained_temporal_development",
            "rounds": rounds,
            "legacy_selected_rounds_json_used": False,
        },
    )
    development_results = []
    columns_by_variant = {}
    for variant in PROPOSED_VARIANTS:
        columns = tuple(
            feature_columns(feature_schema, variant, metric) for metric in range(4)
        )
        columns_by_variant[variant] = columns
        boosters, training_seconds, model_bytes = train_or_load_boosters(
            selection_train,
            columns,
            rounds,
            model_root / "temporal_selection" / f"development_{variant}",
        )
        predictions, prediction_seconds = predict_boosters(
            boosters, development, columns
        )
        metrics = score_dict(development.actuals, predictions)
        development_results.append(
            {
                "variant": variant,
                "mape_auc": metrics["mape_auc"],
                "mean_mape": metrics["mean_mape"],
                "training_seconds": training_seconds,
                "prediction_seconds": prediction_seconds,
                "model_bytes": model_bytes,
            }
        )
    selected_index = max(
        range(len(development_results)),
        key=lambda index: (
            float(development_results[index]["mape_auc"]),
            -float(development_results[index]["mean_mape"]),
            -index,
        ),
    )
    selected_variant = str(development_results[selected_index]["variant"])
    selected_columns = columns_by_variant[selected_variant]
    boosters, _, model_bytes = train_or_load_boosters(
        prelock,
        selected_columns,
        rounds,
        model_root / "temporal_selection" / f"selected_{selected_variant}",
    )
    predictions, prediction_seconds = predict_boosters(
        boosters, lockbox, selected_columns
    )
    return {
        "selected_variant": selected_variant,
        "selection_rule": (
            "highest temporal-development MAPEAUC; temporal lockbox excluded"
        ),
        "development_results": development_results,
        "temporal_lockbox": {
            "selected": score_dict(lockbox.actuals, predictions),
            "model_bytes": model_bytes,
            "prediction_seconds": prediction_seconds,
        },
    }


def parse_variants(text: str) -> tuple[str, ...]:
    variants = tuple(value.strip() for value in text.split(",") if value.strip())
    if not variants:
        raise ValueError("at least one proposed feature variant is required")
    unknown = sorted(set(variants) - set(PROPOSED_VARIANTS))
    if unknown:
        raise ValueError(f"unknown proposed variants: {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("proposed feature variants must be unique")
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run temporal selection and strict nested cell-disjoint evaluation"
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--variants", default=",".join(PROPOSED_VARIANTS))
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print and validate the outer-cell/inner-time split without GPU training",
    )
    args = parser.parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    variants = parse_variants(args.variants)
    project_root = Path(".").resolve()
    output = Path(args.output)
    model_root = output / "strict_nested_models"
    rows = read_traffic("data/train_data.csv")
    examples = build_training_backtests(rows)
    protocol = build_temporal_protocol(examples)
    plans = [build_outer_fold_plan(examples, protocol, fold) for fold in OUTER_FOLDS]
    source_hashes = registered_hashes(project_root, SOURCE_PATHS)
    implementation_hashes = registered_hashes(project_root, IMPLEMENTATION_PATHS)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "strict_nested_cell_disjoint",
        "seed": SEED,
        "outer_folds": len(OUTER_FOLDS),
        "outer_fold_assignment": "sha256(cell_id).digest()[0] % 5",
        "temporal_protocol": protocol.as_dict(),
        "seasonal_tuning": (
            "each outer fold selects its seasonal configuration on inner dates using "
            "only outer-training cells"
        ),
        "proposed_variants": list(variants),
        "plain_tuning": (
            "independent early stopping within each outer-training-cell subset"
        ),
        "proposed_tuning": (
            "each feature variant independently early-stopped within each outer-training-cell "
            "subset; variant selected by inner chronological development MAPEAUC"
        ),
        "pooled_metric_policy": (
            "concatenate all five outer-fold OOF predictions, then apply complete-case and "
            "per-metric 5th-percentile filtering once"
        ),
        "bootstrap_replicates": args.bootstrap,
        "legacy_selected_rounds_json_used": False,
        "output_files": {
            "summary_json": "strict_nested_model_selection.json",
            "pooled_oof_csv": "strict_nested_oof_predictions.csv",
            "fold_directory": "strict_nested_cell_disjoint",
        },
        "source_sha256": source_hashes,
        "implementation_sha256": implementation_hashes,
        "fold_plans": [fold_plan_report(plan, protocol) for plan in plans],
    }
    if args.plan_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    parameters = load_parameters("data/parameter.csv")
    weather = load_weather("data/weather.csv")
    baseline = BaselineConfig(
        "weekly_median_s097",
        (0.0, 0.7, 0.2, 0.1, 0.0, 0.0),
        (0.97,) * 4,
    )
    feature_schema = feature_names(examples[0], baseline, parameters, weather)

    def layer(selected_dates: frozenset[date]) -> tuple[BacktestExample, ...]:
        return tuple(
            example
            for example in examples
            if example.window.target_start.date() in selected_dates
        )

    temporal = run_temporal_selection(
        output=output,
        model_root=model_root,
        baseline=baseline,
        parameters=parameters,
        weather=weather,
        feature_schema=feature_schema,
        selection_train_examples=layer(protocol.selection_train_dates),
        development_examples=layer(protocol.development_dates),
        prelock_examples=layer(protocol.prelock_dates),
        lockbox_examples=layer(protocol.lockbox_dates),
    )
    lockbox_actuals = tuple(
        actual
        for example in layer(protocol.lockbox_dates)
        for actual in example.actuals
    )
    proposed_oof: list[ForecastRow | None] = [None] * len(lockbox_actuals)
    plain_oof: list[ForecastRow | None] = [None] * len(lockbox_actuals)
    fold_reports = []
    for plan in plans:
        proposed, plain, report = run_strict_outer_fold(
            plan=plan,
            protocol=protocol,
            parameters=parameters,
            weather=weather,
            proposed_variants=variants,
            output=output,
            model_root=model_root,
            source_hashes=source_hashes,
            implementation_hashes=implementation_hashes,
        )
        place_oof_predictions(lockbox_actuals, proposed_oof, proposed)
        place_oof_predictions(lockbox_actuals, plain_oof, plain)
        fold_reports.append(report)
    if any(value is None for value in proposed_oof) or any(
        value is None for value in plain_oof
    ):
        raise RuntimeError("strict nested OOF predictions are incomplete")
    proposed_rows = [value for value in proposed_oof if value is not None]
    plain_rows = [value for value in plain_oof if value is not None]
    proposed_metrics = score_dict(lockbox_actuals, proposed_rows)
    plain_metrics = score_dict(lockbox_actuals, plain_rows)
    pooled_prediction_path = output / "strict_nested_oof_predictions.csv"
    write_fold_predictions(
        pooled_prediction_path,
        None,
        lockbox_actuals,
        plain_rows,
        proposed_rows,
    )
    cell_disjoint = {
        "strict_nested": True,
        "selection_rule": (
            "outer validation cells are excluded from seasonal selection, round selection, "
            "feature-variant selection, and booster fitting; plain and proposed are "
            "independently tuned"
        ),
        "selected": proposed_metrics,
        "plain_lgbm": plain_metrics,
        "folds": fold_reports,
        "bootstrap": cluster_bootstrap(
            lockbox_actuals,
            plain_rows,
            proposed_rows,
            replicates=args.bootstrap,
        ),
        "pooled_metric_scope": (
            "authoritative pooled OOF score with one pooled filtering pass"
        ),
        "pooled_prediction_file": str(pooled_prediction_path.relative_to(output)),
        "pooled_prediction_rows": len(proposed_rows),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "strict_nested": True,
        **temporal,
        "cell_disjoint": cell_disjoint,
        "audit": audit,
    }
    write_json(output / "strict_nested_model_selection.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
