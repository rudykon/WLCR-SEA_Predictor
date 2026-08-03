from __future__ import annotations

"""Assemble the paper's clean-result table from freshly generated assets."""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments import train_neural_baselines as neural
from experiments import wlcr_sea_model as sea


OUTPUT_ROOT = Path("artifacts/reproduction")
SEEDS = (42, 43, 44, 45, 46)
PRIMARY = "A6_mixed_aug"

METHODS = (
    ("A0_fixed", "Fixed seasonal mixture", "deterministic"),
    ("A0_global_static", "Global indicator weights", "clean"),
    ("A0_horizon_indicator", "Horizon-indicator weights", "clean"),
    ("A1_softmax", "Dynamic Softmax router", "clean"),
    ("A2_entmax", "Dynamic Entmax router", "clean"),
    ("A3_hard_mask", "A3: + hard availability mask", "clean"),
    ("A4_reliability", "A4: + reliability descriptor", "clean"),
    ("A5_residual", "A5: + bounded residual", "clean"),
    ("standard_stat", "Standard-stat LightGBM", "clean"),
    ("original_wlcr", "Traffic-only LightGBM (73D)", "clean"),
    ("dlinear_clean", "DLinear", "clean"),
    ("dlinear_aug", "DLinear-Aug", "mixed-15pct"),
    ("patchtst_clean", "PatchTST", "clean"),
    ("patchtst_aug", "PatchTST-Aug", "mixed-15pct"),
    ("grud_direct_aug", "GRU-D-Direct-Aug", "mixed-15pct"),
    (PRIMARY, "WLCR-SEA", "mixed-15pct"),
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_inside(root: Path, value: str | Path, *, strict: bool) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve(strict=strict)
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes the repository: {value}")
    return resolved


def resolve_output(root: Path, value: str | Path) -> Path:
    output = resolve_inside(root, value, strict=False)
    allowed = (root / OUTPUT_ROOT).resolve(strict=False)
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("output must be a new child of artifacts/reproduction")
    return output


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_prediction(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if values.shape != shape:
        raise ValueError(f"prediction shape mismatch: {path} has {values.shape}, expected {shape}")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"invalid prediction values: {path}")
    return values


def ensemble(root: Path, model: str, shape: tuple[int, ...]) -> tuple[np.ndarray, list[Path]]:
    paths = [root / "worker_predictions" / f"{model}_seed{seed}.npy" for seed in SEEDS]
    arrays = [load_prediction(path, shape) for path in paths]
    return np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float64).astype(np.float32), paths


def reorder_tree_prediction(root: Path, dataset: neural.CachedDataset, holdout: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, Path, Path]:
    prediction_path = root / "holdout_predictions.npy"
    order_path = root / "holdout_order.csv"
    raw = load_prediction(prediction_path, shape)
    with order_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(holdout):
        raise ValueError(f"tree holdout order is incomplete: {order_path}")
    epoch = datetime(1970, 1, 1)
    positions: dict[tuple[str, int], int] = {}
    for offset, row in enumerate(rows):
        hour = int((datetime.fromisoformat(row["target_start"]) - epoch).total_seconds() // 3600)
        key = (str(row["cell"]), hour)
        if key in positions:
            raise ValueError(f"duplicate tree holdout key: {key}")
        positions[key] = offset
    expected = [(str(dataset.cells[index]), int(dataset.target_start_hours[index])) for index in holdout.tolist()]
    if set(positions) != set(expected):
        raise ValueError("tree holdout keys do not match the paper evaluation set")
    return raw[np.asarray([positions[key] for key in expected], dtype=np.int64)], prediction_path, order_path


def metric_row(
    *,
    method: str,
    label: str,
    training_view: str,
    prediction: np.ndarray,
    actual: np.ndarray,
    scales: np.ndarray,
    cells: np.ndarray,
    thresholds: np.ndarray,
    seed_paths: Sequence[Path] | None,
) -> dict[str, object]:
    metrics = sea.forecast_metrics(actual, prediction, scales, cells)
    ths = sea.threshold_hit_score(actual, prediction, thresholds)
    row: dict[str, object] = {
        "method": method,
        "label": label,
        "training_view": training_view,
        "macro_wape": float(metrics["macro_indicator"]["wape"]),
        "mase": float(metrics["macro_indicator"]["mase"]),
        "smape": float(metrics["macro_indicator"]["smape"]),
        "threshold_hit_score": float(ths["score"]),
        "pooled_wape": float(metrics["pooled_wape"]),
        "macro_cell_wape": float(metrics["macro_cell_wape"]),
        "median_cell_wape": float(metrics["median_cell_wape"]),
    }
    if seed_paths:
        values = [float(sea.forecast_metrics(actual, load_prediction(path, prediction.shape), scales, cells)["macro_indicator"]["wape"]) for path in seed_paths]
        row.update(
            {
                "seed_count": len(values),
                "seed_macro_wape_mean": float(np.mean(values)),
                "seed_macro_wape_sd": float(np.std(values, ddof=1)),
                "seed_macro_wape_min": float(np.min(values)),
                "seed_macro_wape_max": float(np.max(values)),
            }
        )
    else:
        row.update({"seed_count": 0, "seed_macro_wape_mean": "", "seed_macro_wape_sd": "", "seed_macro_wape_min": "", "seed_macro_wape_max": ""})
    return row


def output_manifest(output: Path) -> list[dict[str, object]]:
    return [
        {"path": str(path.relative_to(output)), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def run(args: argparse.Namespace) -> int:
    root = project_root()
    output = resolve_output(root, args.output)
    if output.exists():
        raise FileExistsError(f"refusing to mix results in existing output: {output}")
    output.mkdir(parents=True)
    sea_root = resolve_inside(root, args.sea_root, strict=True)
    neural_clean = resolve_inside(root, args.neural_clean_root, strict=True)
    neural_mixed = resolve_inside(root, args.neural_mixed_root, strict=True)
    grud_root = resolve_inside(root, args.grud_root, strict=True)
    standard_root = resolve_inside(root, args.standard_root, strict=True)
    traffic_root = resolve_inside(root, args.traffic_root, strict=True)

    train_path = neural.resolve_train_path()
    series = neural.read_training_series(train_path)
    arrays, dataset_report = neural.build_window_arrays(series)
    dataset = neural.CachedDataset(root=Path("<memory>"), **arrays)
    fit = dataset.indices_for_dates(neural.FIT_DATES)
    inner = dataset.indices_for_dates(neural.INNER_DATES)
    holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
    final_train = np.concatenate((fit, inner))
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    scales = np.asarray(dataset.mase_scales[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    shape = actual.shape
    thresholds = sea.frozen_low_activity_thresholds(dataset.targets, dataset.target_masks, final_train)

    predictions: dict[str, np.ndarray] = {}
    seed_paths: dict[str, list[Path]] = {}
    predictions["A0_fixed"] = load_prediction(sea_root / "baselines" / "A0_fixed.npy", shape)
    for model in (
        "A0_global_static",
        "A0_horizon_indicator",
        "A1_softmax",
        "A2_entmax",
        "A3_hard_mask",
        "A4_reliability",
        "A5_residual",
        PRIMARY,
    ):
        predictions[model], seed_paths[model] = ensemble(sea_root, model, shape)
    predictions["dlinear_clean"], seed_paths["dlinear_clean"] = ensemble(neural_clean, "dlinear", shape)
    predictions["patchtst_clean"], seed_paths["patchtst_clean"] = ensemble(neural_clean, "patchtst", shape)
    predictions["dlinear_aug"], seed_paths["dlinear_aug"] = ensemble(neural_mixed, "dlinear", shape)
    predictions["patchtst_aug"], seed_paths["patchtst_aug"] = ensemble(neural_mixed, "patchtst", shape)
    predictions["grud_direct_aug"], seed_paths["grud_direct_aug"] = ensemble(grud_root, "grud_direct", shape)
    predictions["standard_stat"], standard_prediction, standard_order = reorder_tree_prediction(standard_root, dataset, holdout, shape)
    predictions["original_wlcr"], traffic_prediction, traffic_order = reorder_tree_prediction(traffic_root, dataset, holdout, shape)

    rows = [
        metric_row(
            method=method,
            label=label,
            training_view=training_view,
            prediction=predictions[method],
            actual=actual,
            scales=scales,
            cells=cells,
            thresholds=thresholds,
            seed_paths=seed_paths.get(method),
        )
        for method, label, training_view in METHODS
    ]
    atomic_csv(output / "comparative_clean_accuracy.csv", rows)
    bootstrap_rows = []
    for method, _label, _view in METHODS:
        if method in {PRIMARY, "A0_fixed"}:
            continue
        result = sea.cell_cluster_bootstrap_wape_delta(actual, predictions[PRIMARY], predictions[method], cells, replicates=args.bootstrap_replicates, seed=42)
        bootstrap_rows.append({"proposed": PRIMARY, "baseline": method, **result})
    atomic_csv(output / "paired_cell_bootstrap.csv", bootstrap_rows)
    summary = {
        "schema_version": 1,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "finals_test_opened": False,
        "registered_train_file": "data/train_data.csv",
        "registered_train_sha256": neural.sha256_file(train_path),
        "dataset_report": dataset_report,
        "holdout_windows": int(len(holdout)),
        "model_seeds": list(SEEDS),
        "frozen_thresholds": thresholds.tolist(),
        "sources": {
            "sea": str(sea_root.relative_to(root)),
            "neural_clean": str(neural_clean.relative_to(root)),
            "neural_mixed": str(neural_mixed.relative_to(root)),
            "grud": str(grud_root.relative_to(root)),
            "standard_stat": str(standard_root.relative_to(root)),
            "traffic_only_73d": str(traffic_root.relative_to(root)),
            "standard_prediction": str(standard_prediction.relative_to(root)),
            "standard_order": str(standard_order.relative_to(root)),
            "traffic_prediction": str(traffic_prediction.relative_to(root)),
            "traffic_order": str(traffic_order.relative_to(root)),
        },
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "manifest.json", output_manifest(output))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--sea-root", default="artifacts/reproduction/wlcr_sea")
    value.add_argument("--neural-clean-root", default="artifacts/reproduction/neural_baselines/clean")
    value.add_argument("--neural-mixed-root", default="artifacts/reproduction/neural_baselines/mixed")
    value.add_argument("--grud-root", default="artifacts/reproduction/neural_baselines/grud_mixed")
    value.add_argument("--standard-root", default="artifacts/reproduction/lightgbm/standard_stat")
    value.add_argument("--traffic-root", default="artifacts/reproduction/lightgbm/traffic_only_73d")
    value.add_argument("--output", default="artifacts/reproduction/analysis/clean")
    value.add_argument("--bootstrap-replicates", type=int, default=5000)
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
