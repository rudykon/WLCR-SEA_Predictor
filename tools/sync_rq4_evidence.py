#!/usr/bin/env python3
from __future__ import annotations

"""Validate the retained RQ4 evidence and synchronize manuscript macros."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


DEFAULT_EVIDENCE_ROOT = Path(
    "artifacts/reproduction/cell_disjoint_protocol_matched"
)
DEFAULT_OUTPUT = Path("paper/rq4_evidence.tex")
EXPECTED_METHODS = (
    "wlcr_sea",
    "fixed_seasonal_mixture",
    "same_hour_median_7d",
    "original_wlcr_lightgbm",
    "standard_stat_lightgbm",
    "dlinear_aug",
    "patchtst_aug",
)
EXPECTED_BATCHES = {
    "wlcr_sea": 256,
    "dlinear_aug": 128,
    "patchtst_aug": 128,
}
EXPECTED_AUGMENTATION_SEEDS = {
    "wlcr_sea": 100_042,
    "dlinear_aug": 100_042,
    "patchtst_aug": 100_042,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(root: Path) -> dict[str, str]:
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    records = load_json(manifest_path)
    require(isinstance(records, list), "manifest must be a list")
    seen: set[str] = set()
    hashes: dict[str, str] = {}
    root_resolved = root.resolve(strict=True)
    for record in records:
        require(isinstance(record, dict), "manifest entries must be objects")
        relative_text = str(record.get("path", ""))
        relative = Path(relative_text)
        require(
            relative_text
            and not relative.is_absolute()
            and ".." not in relative.parts,
            f"unsafe manifest path: {relative_text!r}",
        )
        require(relative_text not in seen, f"duplicate manifest path: {relative_text}")
        seen.add(relative_text)
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root_resolved), f"manifest path escapes root: {path}")
        require(path.is_file(), f"manifest entry is not a file: {path}")
        require(
            path.stat().st_size == int(record["size_bytes"]),
            f"size mismatch: {relative_text}",
        )
        actual_hash = sha256_file(path)
        require(actual_hash == str(record["sha256"]), f"hash mismatch: {relative_text}")
        hashes[relative_text] = actual_hash
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    require(actual_files == seen, "manifest file set does not match evidence directory")
    hashes["manifest.json"] = sha256_file(manifest_path)
    return hashes


def validate_evidence(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    hashes = validate_manifest(root)
    summary = load_json(root / "summary.json")
    protocol = load_json(root / "protocol.json")
    statuses = load_json(root / "worker_status.json")

    require(summary.get("finals_test_opened") is False, "summary opened finals test")
    require(protocol.get("finals_test_opened") is False, "protocol opened finals test")
    require(protocol.get("smoke") is False, "RQ4 evidence cannot be a smoke run")
    require(int(protocol.get("folds", -1)) == 5, "RQ4 evidence must use five folds")
    require(protocol.get("single_model_seed") is True, "RQ4 must declare one model seed")
    require(int(protocol.get("model_seed", -1)) == 42, "unexpected model seed")
    require(
        protocol.get("parameter_or_weather_files_opened") is False,
        "noncanonical context files were opened",
    )
    require(
        protocol.get("canonical_input_files_opened") == ["data/train_data.csv"],
        "unexpected canonical input boundary",
    )
    require(
        protocol.get("registered_train_sha256_before")
        == protocol.get("registered_train_sha256_after"),
        "registered training trace changed during RQ4",
    )

    refit = protocol.get("refit_training_protocol", {})
    require(refit.get("single_model_seed") is True, "refit is not single-seed")
    require(int(refit.get("model_seed", -1)) == 42, "unexpected refit model seed")
    require(
        refit.get("batch_size_by_model") == EXPECTED_BATCHES,
        "RQ4 batch sizes do not match the temporal refit protocols",
    )
    require(
        refit.get("augmentation_seed_by_model") == EXPECTED_AUGMENTATION_SEEDS,
        "RQ4 final-refit augmentation views are not matched",
    )
    require(
        refit.get("matched_final_refit_augmentation_view") is True,
        "RQ4 does not declare a matched final-refit augmentation view",
    )
    require(
        refit.get("matches_temporal_batch_size_defaults") is True,
        "RQ4 does not match temporal batch-size defaults",
    )
    require(
        summary.get("refit_training_protocol") == refit,
        "summary and protocol disagree about refitting",
    )

    require(isinstance(statuses, list) and len(statuses) == 5, "invalid worker status")
    require(
        sorted(int(row["fold"]) for row in statuses) == list(range(5)),
        "worker status does not cover folds 0--4",
    )
    require(
        all(int(row["returncode"]) == 0 for row in statuses),
        "at least one RQ4 worker failed",
    )
    augmentation_keys = (
        "mechanism",
        "requested_rate",
        "seed",
        "scope",
        "newly_removed_rate",
        "final_total_missing_rate",
        "unique_cell_time",
        "window_exposure",
    )
    for fold in range(5):
        worker = load_json(root / "worker" / f"fold{fold}.json")
        require(worker.get("finals_test_opened") is False, f"fold {fold} opened finals")
        require(int(worker.get("cell_overlap", -1)) == 0, f"fold {fold} has overlap")
        require(int(worker.get("model_seed", -1)) == 42, f"fold {fold} seed changed")
        require(
            worker.get("refit_training_protocol") == refit,
            f"fold {fold} refit protocol mismatch",
        )
        training = worker["training_reports"]
        signatures = []
        for method in ("wlcr_sea", "dlinear_aug", "patchtst_aug"):
            method_report = training[method]
            require(
                int(method_report["batch_size"]) == EXPECTED_BATCHES[method],
                f"fold {fold} {method} batch mismatch",
            )
            require(
                int(method_report["augmentation_seed"])
                == EXPECTED_AUGMENTATION_SEEDS[method],
                f"fold {fold} {method} augmentation seed mismatch",
            )
            augmentation = method_report["augmentation"]
            require(
                int(augmentation["seed"]) == EXPECTED_AUGMENTATION_SEEDS[method],
                f"fold {fold} {method} reported view seed mismatch",
            )
            signatures.append({key: augmentation[key] for key in augmentation_keys})
        require(
            signatures[0] == signatures[1] == signatures[2],
            f"fold {fold} trainable methods did not receive the same view",
        )
    folds = summary.get("folds", [])
    require(isinstance(folds, list) and len(folds) == 5, "summary must contain five folds")
    require(
        sorted(int(row["fold"]) for row in folds) == list(range(5)),
        "summary does not cover folds 0--4",
    )
    require(
        summary.get("all_fold_cell_overlaps_zero") is True
        and all(int(row["cell_overlap"]) == 0 for row in folds),
        "cell-disjoint evidence contains cell overlap",
    )
    require(
        sum(int(row["evaluation_windows"]) for row in folds) == 5_110,
        "cell-disjoint folds do not cover 5,110 windows",
    )

    unit = summary.get("statistical_unit", {})
    require(int(unit.get("evaluation_cells", -1)) == 730, "unexpected evaluation cells")
    require(
        int(unit.get("evaluable_cell_clusters", -1)) == 727,
        "unexpected evaluable cell-cluster count",
    )
    require(int(unit.get("evaluation_windows", -1)) == 5_110, "unexpected windows")
    require(tuple(summary.get("methods", ())) == EXPECTED_METHODS, "method set changed")

    metrics = summary.get("metrics", {})
    comparisons = summary.get("paired_cell_cluster_bootstrap", {})
    proposed = metrics["wlcr_sea"]["macro_indicator"]["wape"]
    for baseline in EXPECTED_METHODS[1:]:
        key = f"wlcr_sea_minus_{baseline}"
        result = comparisons[key]
        baseline_wape = metrics[baseline]["macro_indicator"]["wape"]
        require(int(result["replicates"]) == 5_000, f"{key}: bootstrap count changed")
        require(int(result["seed"]) == 42, f"{key}: bootstrap seed changed")
        require(int(result["clusters"]) == 727, f"{key}: cluster count changed")
        require(
            result["point_proposed_macro_wape"] == proposed,
            f"{key}: proposed WAPE mismatch",
        )
        require(
            result["point_baseline_macro_wape"] == baseline_wape,
            f"{key}: baseline WAPE mismatch",
        )
        require(
            abs(
                result["delta_proposed_minus_baseline"]
                - (proposed - baseline_wape)
            )
            <= Decimal("1e-15"),
            f"{key}: delta is not proposed minus baseline",
        )

    prior_mean = sum(
        (row["mean_prior_mass"] for row in folds), start=Decimal("0")
    ) / Decimal(len(folds))
    require(
        abs(prior_mean - summary["unseen_mean_prior_mass"]) <= Decimal("1e-15"),
        "unseen-cell prior mass is not the fold mean",
    )
    return {
        "summary": summary,
        "protocol": protocol,
        "hashes": hashes,
    }


def format_fixed(value: Decimal | int, places: int, *, signed: bool = False) -> str:
    decimal_value = value if isinstance(value, Decimal) else Decimal(value)
    quantum = Decimal(1).scaleb(-places)
    rendered = format(decimal_value.quantize(quantum, rounding=ROUND_HALF_UP), f".{places}f")
    if signed and decimal_value >= 0:
        return "+" + rendered
    return rendered


def render_tex(evidence: dict[str, Any]) -> str:
    summary = evidence["summary"]
    protocol = evidence["protocol"]
    hashes = evidence["hashes"]
    metrics = summary["metrics"]
    comparisons = summary["paired_cell_cluster_bootstrap"]

    def wape(method: str) -> Decimal:
        return metrics[method]["macro_indicator"]["wape"]

    def comparison(method: str) -> dict[str, Any]:
        return comparisons[f"wlcr_sea_minus_{method}"]

    dlinear = comparison("dlinear_aug")
    original = comparison("original_wlcr_lightgbm")
    patch = comparison("patchtst_aug")
    refit = protocol["refit_training_protocol"]
    macros = {
        "RQFourWLCRWAPE": format_fixed(wape("wlcr_sea"), 4),
        "RQFourDLinearWAPE": format_fixed(wape("dlinear_aug"), 4),
        "RQFourDLinearDelta": format_fixed(dlinear["delta_proposed_minus_baseline"], 5, signed=True),
        "RQFourDLinearCILow": format_fixed(dlinear["ci_low"], 5),
        "RQFourDLinearCIHigh": format_fixed(dlinear["ci_high"], 5),
        "RQFourOriginalWAPE": format_fixed(wape("original_wlcr_lightgbm"), 4),
        "RQFourOriginalDelta": format_fixed(original["delta_proposed_minus_baseline"], 5, signed=True),
        "RQFourOriginalCILow": format_fixed(original["ci_low"], 5),
        "RQFourOriginalCIHigh": format_fixed(original["ci_high"], 5),
        "RQFourPatchWAPE": format_fixed(wape("patchtst_aug"), 4),
        "RQFourPatchDelta": format_fixed(patch["delta_proposed_minus_baseline"], 5, signed=True),
        "RQFourPatchCILow": format_fixed(patch["ci_low"], 5),
        "RQFourPatchCIHigh": format_fixed(patch["ci_high"], 5),
        "RQFourSameHourWAPE": format_fixed(wape("same_hour_median_7d"), 4),
        "RQFourFixedMixtureWAPE": format_fixed(wape("fixed_seasonal_mixture"), 4),
        "RQFourStandardStatWAPE": format_fixed(wape("standard_stat_lightgbm"), 4),
        "RQFourPriorMassPercent": format_fixed(summary["unseen_mean_prior_mass"] * Decimal(100), 2),
        "RQFourClusterCount": str(summary["statistical_unit"]["evaluable_cell_clusters"]),
        "RQFourEvaluationWindowCount": "5,110",
        "RQFourModelSeed": str(refit["model_seed"]),
        "RQFourAugmentationSeed": str(refit["augmentation_seed_by_model"]["wlcr_sea"]),
        "RQFourWLCRBatchSize": str(refit["batch_size_by_model"]["wlcr_sea"]),
        "RQFourNeuralBatchSize": str(refit["batch_size_by_model"]["dlinear_aug"]),
    }
    lines = [
        "% Generated by tools/sync_rq4_evidence.py; do not edit by hand.",
        f"% manifest sha256: {hashes['manifest.json']}",
        f"% summary sha256: {hashes['summary.json']}",
        f"% protocol sha256: {hashes['protocol.json']}",
    ]
    lines.extend(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
    )
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    action = value.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="atomically update the TeX macros")
    action.add_argument("--check", action="store_true", help="fail if the TeX macros are stale")
    action.add_argument("--stdout", action="store_true", help="print validated TeX macros")
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        rendered = render_tex(validate_evidence(args.evidence_root))
        if args.stdout:
            sys.stdout.write(rendered)
            return 0
        if args.write:
            atomic_write(args.output, rendered)
            return 0
        if not args.output.is_file():
            print(f"missing synchronized macro file: {args.output}", file=sys.stderr)
            return 1
        if args.output.read_text(encoding="utf-8") != rendered:
            print(f"stale synchronized macro file: {args.output}", file=sys.stderr)
            return 1
        return 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"RQ4 evidence validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
