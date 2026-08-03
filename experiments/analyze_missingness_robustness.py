from __future__ import annotations

"""Revision-9 missingness curves and matched-inference robustness statistics.

This driver is intentionally separate from the frozen Revision-8 scripts.  It
uses only the registered training trace to recreate the existing August
holdout and the exact global cell-time missingness protocol:

* one clean scenario plus four mechanisms at four positive requested rates
  (17 scenarios), each under the five fixed corruption seeds 142--146;
* frozen five-seed WLCR-SEA A6, DLinear-Aug, PatchTST-Aug, and
  GRU-D-Direct-Aug ensembles evaluated under the same masks;
* a separately generated, clean-trained Original WLCR-LightGBM stress asset,
  ingested only after request identity, targets, masks, hashes, and clean
  replay evidence pass.

The Original WLCR curve is deliberately labelled as a descriptive,
non-augmentation-matched stress control.  It is never included in the matched
inference bootstrap table or in claims about the causal contribution of the
WLCR-SEA routing architecture.

All outputs are exploratory evidence on an existing holdout.  This script
never opens ``data/test_data.csv`` or the preliminary reference test input.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from experiments import missingness_protocol as missingness
from experiments import train_neural_baselines as neural
from experiments import analyze_model_comparisons as revision6
from experiments import estimate_paired_robustness as robustness
from experiments import train_wlcr_sea as runner
from experiments import wlcr_sea_model as sea


SCHEMA_VERSION = 1
MODEL_SEEDS = (42, 43, 44, 45, 46)
CORRUPTION_SEEDS = (142, 143, 144, 145, 146)
CLEAN_REPLAY_ABS_TOLERANCE = 2e-4
ORIGINAL_METRIC_ABS_TOLERANCE = 3e-6
ORIGINAL_ARRAY_ABS_TOLERANCE = 1e-7

DEFAULT_OUTPUT = Path("artifacts/reproduction/analysis/missingness")
DEFAULT_SEA_ROOT = Path("artifacts/reproduction/wlcr_sea")
DEFAULT_AUGMENTED_NEURAL_ROOT = Path("artifacts/reproduction/neural_baselines/mixed")
DEFAULT_GRUD_ROOT = Path("artifacts/reproduction/neural_baselines/grud_mixed")
DEFAULT_ORIGINAL_WLCR_ROOT = Path("artifacts/reproduction/analysis/traffic_only_73d_missingness")
DEFAULT_ORIGINAL_WLCR_CLEAN = Path("artifacts/reproduction/lightgbm/traffic_only_73d/holdout_predictions.npy")

IDENTITY_COLUMNS = (
    "holdout_position",
    "dataset_index",
    "cell",
    "target_start",
    "history_end",
    "target_start_hour",
    "history_end_hour",
)
ORIGINAL_MASK_COLUMNS = (
    "mechanism",
    "requested_rate",
    "corruption_seed",
    "scenario",
    "mask_file",
    "mask_sha256",
    "mask_shape",
    "mask_dtype",
)
ORIGINAL_RAW_REQUIRED_COLUMNS = (
    "method",
    "label",
    "training_view",
    "mechanism",
    "mechanism_display",
    "requested_rate",
    "scenario",
    "corruption_seed",
    "mask_file",
    "mask_sha256",
    "prediction_file",
    "request_identity_sha256",
    "prediction_sha256",
    "macro_wape",
    "pooled_wape",
    "mase",
    "smape",
    "threshold_hit_score",
)
ORIGINAL_INDICATOR_REQUIRED_COLUMNS = (
    "method",
    "mechanism",
    "requested_rate",
    "scenario",
    "corruption_seed",
    "mask_sha256",
    "indicator",
    "wape",
    "mase",
    "smape",
)

METHOD_META: dict[str, dict[str, str]] = {
    "A6_mixed_aug": {
        "label": "WLCR-SEA (mixed augmentation)",
        "training_view": "mixed-15pct",
        "comparison_class": "augmentation_matched",
        "comparison_note": "15% mixed-augmentation matched inference control",
    },
    "dlinear_aug": {
        "label": "DLinear-Aug",
        "training_view": "mixed-15pct",
        "comparison_class": "augmentation_matched",
        "comparison_note": "15% mixed-augmentation matched inference control",
    },
    "patchtst_aug": {
        "label": "PatchTST-Aug",
        "training_view": "mixed-15pct",
        "comparison_class": "augmentation_matched",
        "comparison_note": "15% mixed-augmentation matched inference control",
    },
    "grud_direct_aug": {
        "label": "GRU-D-Direct-Aug",
        "training_view": "mixed-15pct",
        "comparison_class": "augmentation_matched",
        "comparison_note": "15% mixed-augmentation matched inference control",
    },
    "original_wlcr": {
        "label": "Original WLCR-LightGBM (clean-trained)",
        "training_view": "clean-trained frozen model",
        "comparison_class": "descriptive_clean_trained_not_augmentation_matched",
        "comparison_note": (
            "clean-trained frozen stress control; descriptive only and not "
            "eligible for augmentation-matched routing claims"
        ),
    },
}

MATCHED_BASELINES: tuple[tuple[str, str], ...] = (
    ("dlinear_aug", "DLinear-Aug"),
    ("patchtst_aug", "PatchTST-Aug"),
    ("grud_direct_aug", "GRU-D-Direct-Aug"),
)
DESCRIPTIVE_BASELINE = ("original_wlcr", "Original WLCR-LightGBM (clean-trained)")


@dataclass(frozen=True)
class Scenario:
    mechanism: str
    requested_rate: float

    @property
    def scenario(self) -> str:
        return scenario_name(self.mechanism, self.requested_rate)

    @property
    def mechanism_display(self) -> str:
        return "timeline_tail" if self.mechanism == "recent_tail" else self.mechanism

    @property
    def key(self) -> tuple[str, float]:
        return (self.mechanism, canonical_rate(self.requested_rate))


@dataclass(frozen=True)
class OriginalWLCRArtifact:
    root: Path
    raw_rows: Mapping[tuple[str, float, int], Mapping[str, str]]
    indicator_rows: Mapping[tuple[str, float, int, str], Mapping[str, str]]
    mask_rows: Mapping[tuple[str, float, int], Mapping[str, str]]
    clean_replay: Mapping[str, object]
    request_identity_sha256: str


def canonical_rate(value: float | str) -> float:
    """Normalize a protocol rate without allowing accidental near-duplicates."""
    rate = float(value)
    if not math.isfinite(rate):
        raise ValueError(f"non-finite requested rate: {value!r}")
    return round(rate, 2)


def scenario_name(mechanism: str, requested_rate: float) -> str:
    """Return the canonical on-disk scenario name shared with the WLCR worker."""
    rate = canonical_rate(requested_rate)
    if rate == 0.0:
        if mechanism != "mcar":
            raise ValueError("the clean scenario must be represented once as mcar/0.00")
        return "clean"
    if mechanism not in runner.ROBUSTNESS_MECHANISMS:
        raise ValueError(f"unknown missingness mechanism: {mechanism}")
    prefix = "timeline_tail" if mechanism == "recent_tail" else mechanism
    return f"{prefix}_{rate:.2f}"


def full_scenarios() -> tuple[Scenario, ...]:
    """The exact Revision-8 protocol: 1 clean + 4 mechanisms x 4 rates."""
    scenarios: list[Scenario] = []
    for mechanism in runner.ROBUSTNESS_MECHANISMS:
        for rate in runner.ROBUSTNESS_RATES:
            if rate == 0.0:
                if mechanism == "mcar":
                    scenarios.append(Scenario(mechanism, rate))
            else:
                scenarios.append(Scenario(mechanism, rate))
    if len(scenarios) != 17:
        raise RuntimeError(f"expected 17 revision-8 scenarios, found {len(scenarios)}")
    if len({scenario.key for scenario in scenarios}) != len(scenarios):
        raise RuntimeError("revision-8 scenarios are not unique")
    return tuple(scenarios)


def smoke_scenarios() -> tuple[Scenario, ...]:
    """A small explicit subset for post-asset plumbing checks, not publication."""
    return (Scenario("mcar", 0.0), Scenario("block", 0.20))


def resolve_inside(root: Path, value: str | Path, *, strict: bool) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve(strict=strict)
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes project root: {resolved}")
    return resolved


def resolve_output(root: Path, value: str | Path) -> Path:
    output = resolve_inside(root, value, strict=False)
    allowed = (root / "artifacts/reproduction").resolve(strict=False)
    if not output.is_relative_to(allowed):
        raise ValueError("outputs must remain under artifacts/reproduction")
    return output


def parse_gpu_devices(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if len(values) < 3:
        raise ValueError("provide at least three distinct GPU device ids")
    if len(values) != len(set(values)):
        raise ValueError("GPU device ids must be distinct")
    if any(item < 0 for item in values):
        raise ValueError("GPU device ids must be non-negative")
    return values


def atomic_output_directory(path: Path) -> None:
    """Prevent a rerun from silently mixing files from distinct model assets."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"refusing to mix a new Revision-9 run into non-empty output: {path}; "
            "choose a fresh --output path"
        )
    path.mkdir(parents=True, exist_ok=True)


def hash_mask(mask: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def expected_identity_rows(
    dataset: neural.CachedDataset, holdout: np.ndarray
) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for position, dataset_index in enumerate(holdout.tolist()):
        target_start_hour = int(dataset.target_start_hours[dataset_index])
        history_end_hour = int(dataset.history_end_hours[dataset_index])
        rows.append(
            {
                "holdout_position": str(position),
                "dataset_index": str(int(dataset_index)),
                "cell": str(dataset.cells[dataset_index]),
                "target_start": neural.timestamp_from_hour(target_start_hour).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "history_end": neural.timestamp_from_hour(history_end_hour).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "target_start_hour": str(target_start_hour),
                "history_end_hour": str(history_end_hour),
            }
        )
    return tuple(rows)


def align_original_clean_reference(
    root: Path,
    reference: np.ndarray,
    expected_identity: Sequence[Mapping[str, str]],
) -> np.ndarray:
    """Reorder the persisted clean reference to the request identity order."""
    order_rows, _ = read_csv_rows(
        root / "holdout_order.csv",
        required_columns=("cell", "target_start"),
    )
    if len(order_rows) != len(reference) or len(order_rows) != len(expected_identity):
        raise ValueError("Original WLCR holdout order does not match the reference shape")
    positions: dict[tuple[str, str], int] = {}
    for position, row in enumerate(order_rows):
        key = (str(row["cell"]), str(row["target_start"]))
        if key in positions:
            raise ValueError(f"duplicate Original WLCR holdout key: {key}")
        positions[key] = position
    expected_keys = [
        (str(row["cell"]), str(row["target_start"]))
        for row in expected_identity
    ]
    if len(set(expected_keys)) != len(expected_keys) or set(positions) != set(expected_keys):
        raise ValueError("Original WLCR holdout order keys differ from the request identity")
    indices = np.asarray([positions[key] for key in expected_keys], dtype=np.int64)
    return np.ascontiguousarray(
        np.asarray(reference, dtype=np.float32)[indices], dtype=np.float32
    )


def read_csv_rows(path: Path, *, required_columns: Iterable[str]) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV lacks a header: {path}")
        fields = tuple(reader.fieldnames)
        missing = [name for name in required_columns if name not in fields]
        if missing:
            raise ValueError(f"CSV lacks required columns at {path}: {missing}")
        rows = [dict(row) for row in reader]
    return rows, fields


def validate_identity_rows(
    rows: Sequence[Mapping[str, str]], expected: Sequence[Mapping[str, str]]
) -> None:
    if len(rows) != len(expected):
        raise ValueError(
            f"Original WLCR request identity has {len(rows)} rows, expected {len(expected)}"
        )
    for position, (observed, target) in enumerate(zip(rows, expected)):
        for column in IDENTITY_COLUMNS:
            if str(observed.get(column, "")) != str(target[column]):
                raise ValueError(
                    "Original WLCR request identity mismatch at position "
                    f"{position}, column {column}: {observed.get(column)!r} != "
                    f"{target[column]!r}"
                )


def assert_array_alignment(
    name: str,
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    require_dtype: np.dtype | None = None,
    absolute_tolerance: float = 0.0,
) -> None:
    values = np.asarray(observed)
    target = np.asarray(expected)
    if values.shape != target.shape:
        raise ValueError(f"{name} shape {values.shape} != expected {target.shape}")
    if require_dtype is not None and values.dtype != require_dtype:
        raise ValueError(f"{name} dtype {values.dtype} != required {require_dtype}")
    if not np.allclose(
        values,
        target,
        rtol=0.0,
        atol=absolute_tolerance,
        equal_nan=True,
    ):
        maximum = float(np.nanmax(np.abs(values.astype(np.float64) - target.astype(np.float64))))
        raise ValueError(
            f"{name} differs from the registered-train holdout; maximum absolute "
            f"difference={maximum}"
        )


def validate_manifest(root: Path) -> list[dict[str, object]]:
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        records = payload.get("files")
    else:
        records = payload
    if not isinstance(records, list) or not records:
        raise ValueError("Original WLCR manifest must contain a non-empty files list")
    checked: list[dict[str, object]] = []
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Original WLCR manifest has a non-object record")
        relative = Path(str(item.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe Original WLCR manifest path: {relative}")
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_relative_to(root):
            raise ValueError(f"Original WLCR manifest path escapes root: {relative}")
        observed_size = candidate.stat().st_size
        observed_hash = runner.sha256_file(candidate)
        if observed_size != int(item.get("size_bytes", -1)):
            raise ValueError(f"Original WLCR manifest size mismatch: {relative}")
        if observed_hash != str(item.get("sha256", "")):
            raise ValueError(f"Original WLCR manifest SHA256 mismatch: {relative}")
        checked.append(
            {
                "path": str(relative),
                "size_bytes": observed_size,
                "sha256": observed_hash,
            }
        )
    return checked


def keyed_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    name: str,
    key_columns: Sequence[str],
) -> dict[tuple[object, ...], Mapping[str, str]]:
    result: dict[tuple[object, ...], Mapping[str, str]] = {}
    for row in rows:
        pieces: list[object] = []
        for column in key_columns:
            value = row[column]
            if column == "requested_rate":
                pieces.append(canonical_rate(value))
            elif column == "corruption_seed":
                pieces.append(int(value))
            else:
                pieces.append(str(value))
        key = tuple(pieces)
        if key in result:
            raise ValueError(f"duplicate {name} key: {key}")
        result[key] = row
    return result


def expected_mask_keys(scenarios: Sequence[Scenario], seeds: Sequence[int]) -> set[tuple[str, float, int]]:
    return {
        (scenario.mechanism, canonical_rate(scenario.requested_rate), int(seed))
        for scenario in scenarios
        for seed in seeds
    }


def validate_original_summary(
    root: Path,
    summary: Mapping[str, object],
    *,
    train_hash: str,
    expected_scenarios: Sequence[Scenario],
    expected_corruption_seeds: Sequence[int],
) -> None:
    if summary.get("registered_train_file") != "data/train_data.csv":
        raise ValueError("Original WLCR asset did not declare data/train_data.csv")
    if summary.get("registered_train_sha256_before") != train_hash:
        raise ValueError("Original WLCR asset train hash before execution is invalid")
    if summary.get("registered_train_sha256_after") != train_hash:
        raise ValueError("Original WLCR asset train hash after execution is invalid")
    if summary.get("finals_test_opened") is not False:
        raise ValueError("Original WLCR asset must declare finals_test_opened=false")
    if tuple(int(value) for value in summary.get("corruption_seeds", ())) != tuple(
        int(value) for value in expected_corruption_seeds
    ):
        raise ValueError("Original WLCR asset corruption seeds do not match this run")
    scenarios = summary.get("scenarios")
    if not isinstance(scenarios, list) or not all(
        isinstance(item, Mapping) for item in scenarios
    ):
        raise ValueError("Original WLCR scenarios must be machine-readable mappings")
    observed = {
        (
            str(item["mechanism"]),
            canonical_rate(item["requested_rate"]),
            str(item["scenario"]),
        )
        for item in scenarios
    }
    expected = {
        (item.mechanism, canonical_rate(item.requested_rate), item.scenario)
        for item in expected_scenarios
    }
    if len(scenarios) != len(expected) or observed != expected:
        raise ValueError("Original WLCR summary scenarios do not match this run")
    if int(summary.get("scenario_count", -1)) != len(expected):
        raise ValueError("Original WLCR summary scenario count is inconsistent")
    expected_seed_scenarios = len(expected) * len(expected_corruption_seeds)
    if int(summary.get("seed_scenario_count", -1)) != expected_seed_scenarios:
        raise ValueError("Original WLCR summary seed-scenario count is inconsistent")
    del root


def load_original_artifact(
    root: Path,
    *,
    dataset: neural.CachedDataset,
    holdout: np.ndarray,
    actual: np.ndarray,
    scales: np.ndarray,
    thresholds: np.ndarray,
    train_hash: str,
    scenarios: Sequence[Scenario],
    corruption_seeds: Sequence[int],
) -> tuple[OriginalWLCRArtifact, list[dict[str, object]]]:
    """Load schema/provenance only; per-mask arrays are checked at inference time."""
    expected_identity = expected_identity_rows(dataset, holdout)
    identity_path = root / "request_identity.csv"
    identity_rows, identity_fields = read_csv_rows(
        identity_path, required_columns=IDENTITY_COLUMNS
    )
    if identity_fields != IDENTITY_COLUMNS:
        raise ValueError(
            "Original WLCR request identity columns must exactly match the shared contract"
        )
    validate_identity_rows(identity_rows, expected_identity)
    identity_hash = runner.sha256_file(identity_path)

    artifact_actual = np.load(root / "actual.npy", allow_pickle=False)
    artifact_scales = np.load(root / "mase_scales.npy", allow_pickle=False)
    artifact_thresholds = np.load(root / "frozen_thresholds.npy", allow_pickle=False)
    assert_array_alignment(
        "Original WLCR actual.npy",
        artifact_actual,
        actual,
        require_dtype=np.dtype(np.float32),
        absolute_tolerance=ORIGINAL_ARRAY_ABS_TOLERANCE,
    )
    assert_array_alignment(
        "Original WLCR mase_scales.npy",
        artifact_scales,
        scales,
        require_dtype=np.dtype(np.float32),
        absolute_tolerance=ORIGINAL_ARRAY_ABS_TOLERANCE,
    )
    assert_array_alignment(
        "Original WLCR frozen_thresholds.npy",
        artifact_thresholds,
        thresholds,
        require_dtype=np.dtype(np.float64),
        absolute_tolerance=1e-12,
    )

    raw_rows, _ = read_csv_rows(
        root / "missingness_by_seed.csv", required_columns=ORIGINAL_RAW_REQUIRED_COLUMNS
    )
    indicator_rows, _ = read_csv_rows(
        root / "missingness_per_indicator_by_seed.csv",
        required_columns=ORIGINAL_INDICATOR_REQUIRED_COLUMNS,
    )
    mask_rows, mask_fields = read_csv_rows(
        root / "mask_hashes.csv", required_columns=ORIGINAL_MASK_COLUMNS
    )
    if mask_fields != ORIGINAL_MASK_COLUMNS:
        raise ValueError("Original WLCR mask_hashes.csv columns do not match the contract")
    raw_by_key = keyed_rows(
        raw_rows,
        name="Original WLCR missingness row",
        key_columns=("mechanism", "requested_rate", "corruption_seed"),
    )
    indicator_by_key = keyed_rows(
        indicator_rows,
        name="Original WLCR per-indicator row",
        key_columns=("mechanism", "requested_rate", "corruption_seed", "indicator"),
    )
    mask_by_key = keyed_rows(
        mask_rows,
        name="Original WLCR mask row",
        key_columns=("mechanism", "requested_rate", "corruption_seed"),
    )
    # The full run requires the exact 17-by-5 protocol.  The explicit smoke
    # mode is a non-publication plumbing check and validates its own two-by-one
    # artifact with the same identity, target, mask, and replay gates.
    required_keys = expected_mask_keys(scenarios, corruption_seeds)
    if set(raw_by_key) != required_keys:
        raise ValueError(
            "Original WLCR raw stress rows do not exactly cover the requested masks"
        )
    if set(mask_by_key) != required_keys:
        raise ValueError(
            "Original WLCR mask records do not exactly cover the requested masks"
        )
    expected_indicator_keys = {
        (*key, indicator)
        for key in required_keys
        for indicator in sea.METRIC_NAMES
    }
    if set(indicator_by_key) != expected_indicator_keys:
        raise ValueError(
            "Original WLCR per-indicator rows do not exactly cover the requested masks"
        )
    for key, row in raw_by_key.items():
        mechanism, rate, seed = key
        expected_scenario = scenario_name(mechanism, rate)
        if row["method"] != "original_wlcr":
            raise ValueError(f"Original WLCR method field mismatch: {key}")
        if row["scenario"] != expected_scenario:
            raise ValueError(f"Original WLCR raw scenario mismatch: {key}")
        if row["mask_file"] != f"masks/{expected_scenario}_seed{seed}.npy":
            raise ValueError(f"Original WLCR raw mask filename mismatch: {key}")
        if row["request_identity_sha256"] != identity_hash:
            raise ValueError(f"Original WLCR raw identity hash mismatch: {key}")
        if row["training_view"] != "clean-trained frozen model":
            raise ValueError(f"Original WLCR must be clean-trained and frozen: {key}")
        expected_display = "timeline_tail" if mechanism == "recent_tail" else mechanism
        if row["mechanism_display"] != expected_display:
            raise ValueError(f"Original WLCR mechanism display mismatch: {key}")
        if int(row["corruption_seed"]) != seed or canonical_rate(row["requested_rate"]) != rate:
            raise ValueError(f"Original WLCR raw key is internally inconsistent: {key}")
    clean_replay = json.loads((root / "clean_replay.json").read_text(encoding="utf-8"))
    if clean_replay.get("all_passed") is not True:
        raise ValueError("Original WLCR clean replay was not marked passed")
    if clean_replay.get("reference_path") != str(DEFAULT_ORIGINAL_WLCR_CLEAN):
        raise ValueError("Original WLCR clean replay uses an unexpected reference path")
    if clean_replay.get("request_identity_sha256") != identity_hash:
        raise ValueError("Original WLCR clean replay identity hash does not match")
    if set(str(key) for key in clean_replay.get("per_seed", {})) != {
        str(seed) for seed in corruption_seeds
    }:
        raise ValueError("Original WLCR clean replay lacks one or more required seeds")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    validate_original_summary(
        root,
        summary,
        train_hash=train_hash,
        expected_scenarios=scenarios,
        expected_corruption_seeds=corruption_seeds,
    )
    manifest_records = validate_manifest(root)
    manifest_paths = {str(item["path"]) for item in manifest_records}
    required_manifest_paths = {
        "request_identity.csv",
        "actual.npy",
        "mase_scales.npy",
        "frozen_thresholds.npy",
        "missingness_by_seed.csv",
        "missingness_per_indicator_by_seed.csv",
        "mask_hashes.csv",
        "clean_replay.json",
        "summary.json",
    }
    required_manifest_paths.update(str(row["prediction_file"]) for row in raw_by_key.values())
    required_manifest_paths.update(str(row["mask_file"]) for row in mask_by_key.values())
    missing_manifest_paths = sorted(required_manifest_paths - manifest_paths)
    if missing_manifest_paths:
        raise ValueError(
            "Original WLCR manifest omits required provenance files: "
            f"{missing_manifest_paths[:5]}"
        )
    return (
        OriginalWLCRArtifact(
            root=root,
            raw_rows=raw_by_key,
            indicator_rows=indicator_by_key,
            mask_rows=mask_by_key,
            clean_replay=clean_replay,
            request_identity_sha256=identity_hash,
        ),
        manifest_records,
    )


def load_artifact_mask_and_prediction(
    artifact: OriginalWLCRArtifact,
    scenario: Scenario,
    corruption_seed: int,
    *,
    expected_mask: np.ndarray,
    prediction_shape: tuple[int, ...],
) -> tuple[np.ndarray, Mapping[str, str]]:
    key = (scenario.mechanism, canonical_rate(scenario.requested_rate), int(corruption_seed))
    raw = artifact.raw_rows[key]
    mask_row = artifact.mask_rows[key]
    expected_scenario = scenario.scenario
    expected_mask_file = f"masks/{expected_scenario}_seed{corruption_seed}.npy"
    expected_prediction_file = (
        f"predictions/original_wlcr_{expected_scenario}_seed{corruption_seed}.npy"
    )
    if mask_row["scenario"] != expected_scenario:
        raise ValueError(f"Original WLCR mask scenario mismatch: {key}")
    if mask_row["mask_file"] != expected_mask_file:
        raise ValueError(f"Original WLCR mask filename mismatch: {key}")
    if mask_row["mask_shape"] != "5110x336x4" or mask_row["mask_dtype"] != "uint8":
        raise ValueError(f"Original WLCR mask representation mismatch: {key}")
    mask_path = artifact.root / mask_row["mask_file"]
    observed_mask = np.load(mask_path, allow_pickle=False)
    if observed_mask.dtype != np.uint8 or observed_mask.shape != expected_mask.shape:
        raise ValueError(f"Original WLCR mask array shape/dtype mismatch: {key}")
    actual_mask_hash = hash_mask(observed_mask)
    if actual_mask_hash != mask_row["mask_sha256"] or actual_mask_hash != raw["mask_sha256"]:
        raise ValueError(f"Original WLCR mask SHA256 mismatch: {key}")
    if not np.array_equal(observed_mask.astype(bool), np.asarray(expected_mask, dtype=bool)):
        raise ValueError(f"Original WLCR mask is not the Revision-8 global mask: {key}")
    if raw["prediction_file"] != expected_prediction_file:
        raise ValueError(f"Original WLCR prediction filename mismatch: {key}")
    prediction_path = artifact.root / raw["prediction_file"]
    if runner.sha256_file(prediction_path) != raw["prediction_sha256"]:
        raise ValueError(f"Original WLCR prediction SHA256 mismatch: {key}")
    prediction = np.load(prediction_path, allow_pickle=False)
    if prediction.dtype != np.float32:
        raise ValueError(f"Original WLCR prediction is not float32: {key}")
    robustness.validate_prediction("/".join(map(str, key)), prediction, prediction_shape)
    return np.asarray(prediction, dtype=np.float32), raw


def validate_original_clean_replay(
    artifact: OriginalWLCRArtifact,
    *,
    corruption_seed: int,
    prediction: np.ndarray,
    clean_reference: np.ndarray,
) -> float:
    entry = artifact.clean_replay["per_seed"][str(corruption_seed)]
    if not isinstance(entry, Mapping) or entry.get("passed") is not True:
        raise ValueError(f"Original WLCR clean replay failed for seed {corruption_seed}")
    difference = float(np.max(np.abs(prediction.astype(np.float64) - clean_reference.astype(np.float64))))
    tolerance = float(artifact.clean_replay.get("absolute_tolerance", CLEAN_REPLAY_ABS_TOLERANCE))
    if tolerance > CLEAN_REPLAY_ABS_TOLERANCE:
        raise ValueError("Original WLCR artifact uses a looser clean replay tolerance")
    if difference > tolerance:
        raise ValueError(
            f"Original WLCR clean replay mismatch for seed {corruption_seed}: {difference}"
        )
    expected_file = f"predictions/original_wlcr_clean_seed{corruption_seed}.npy"
    if entry.get("prediction_file") != expected_file:
        raise ValueError(f"Original WLCR clean replay filename mismatch: {corruption_seed}")
    actual_file_hash = runner.sha256_file(artifact.root / expected_file)
    if entry.get("prediction_sha256") != actual_file_hash:
        raise ValueError(f"Original WLCR clean replay SHA256 mismatch: {corruption_seed}")
    reported = float(entry.get("maximum_absolute_difference", math.inf))
    if reported > tolerance:
        raise ValueError(f"Original WLCR worker reported a failed clean replay: {corruption_seed}")
    return difference


def checkpoint_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": int(path.stat().st_size),
        "sha256": runner.sha256_file(path),
    }


def load_a6_ensemble(
    root: Path, checkpoint_root: Path, device: torch.device
) -> tuple[list[sea.WLCRSEA], np.ndarray, list[dict[str, object]]]:
    models: list[sea.WLCRSEA] = []
    priors: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for seed in MODEL_SEEDS:
        path = checkpoint_root / "models" / f"A6_mixed_aug_seed{seed}.pt"
        model, payload = runner.load_checkpoint(path, device)
        if payload.get("variant", {}).get("name") != "A6_mixed_aug":
            raise ValueError(f"A6 checkpoint variant mismatch: {path}")
        if int(payload.get("seed", -1)) != seed:
            raise ValueError(f"A6 checkpoint seed mismatch: {path}")
        models.append(model)
        priors.append(np.asarray(payload["prior_log"], dtype=np.float32))
        records.append(checkpoint_record(root, path))
    if any(not np.array_equal(prior, priors[0]) for prior in priors[1:]):
        raise ValueError("A6 model seeds have inconsistent training priors")
    return models, priors[0], records


def load_augmented_neural_ensemble(
    root: Path,
    checkpoint_root: Path,
    model_name: str,
    device: torch.device,
) -> tuple[list[torch.nn.Module], neural.Normalization, list[dict[str, object]]]:
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
            raise ValueError(f"neural checkpoint is not 15% mixed augmentation: {path}")
        model = neural.build_model(model_name, payload["config"])
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        models.append(model)
        normalizations.append(payload["normalization"])
        records.append(checkpoint_record(root, path))
    if any(item != normalizations[0] for item in normalizations[1:]):
        raise ValueError(f"{model_name} normalizations differ across model seeds")
    return models, neural.Normalization(**normalizations[0]), records


def predict_sea_ensemble(
    models: Sequence[sea.WLCRSEA],
    tensors: tuple[torch.Tensor, ...],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for model in models:
        output = runner.predict(
            model, tensors, device=device, batch_size=batch_size, include_audit=False
        )
        predictions.append(np.asarray(output["prediction"], dtype=np.float32))
    return np.mean(np.stack(predictions, axis=0), axis=0, dtype=np.float64).astype(
        np.float32
    )


def shared_neural_request_view(
    dataset: neural.CachedDataset,
    indices: np.ndarray,
    additional_missing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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


def load_clean_ensemble(root: Path, model_name: str, shape: tuple[int, ...]) -> np.ndarray:
    paths = [root / "worker_predictions" / f"{model_name}_seed{seed}.npy" for seed in MODEL_SEEDS]
    return revision6.ensemble_files(paths, shape)


def metric_payload(
    prediction: np.ndarray,
    actual: np.ndarray,
    scales: np.ndarray,
    cells: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[dict[str, float], Mapping[str, object]]:
    metrics = sea.forecast_metrics(actual, prediction, scales, cells)
    threshold = sea.threshold_hit_score(actual, prediction, thresholds)
    values = {
        "macro_wape": float(metrics["macro_indicator"]["wape"]),
        "pooled_wape": float(metrics["pooled_wape"]),
        "mase": float(metrics["macro_indicator"]["mase"]),
        "smape": float(metrics["macro_indicator"]["smape"]),
        "threshold_hit_score": float(threshold["score"]),
    }
    return values, metrics


def aggregate_seed_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group_fields: Sequence[str],
    numeric_fields: Sequence[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        items = groups[key]
        record: dict[str, object] = {
            field: value for field, value in zip(group_fields, key)
        }
        record["corruption_seed_count"] = len(items)
        record["corruption_seeds"] = ",".join(
            str(int(item["corruption_seed"])) for item in items
        )
        for field in numeric_fields:
            values = [float(item[field]) for item in items]
            mean = float(statistics.mean(values))
            sd = float(statistics.stdev(values)) if len(values) > 1 else 0.0
            half = (
                2.7764451051977987 * sd / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            record[field] = mean
            record[f"{field}_sd"] = sd
            record[f"{field}_ci_low"] = mean - half
            record[f"{field}_ci_high"] = mean + half
        output.append(record)
    return output


def assert_original_metric_rows(
    artifact: OriginalWLCRArtifact,
    scenario: Scenario,
    corruption_seed: int,
    values: Mapping[str, float],
    metrics: Mapping[str, object],
) -> None:
    key = (scenario.mechanism, canonical_rate(scenario.requested_rate), int(corruption_seed))
    raw = artifact.raw_rows[key]
    for name, value in values.items():
        reported = float(raw[name])
        if not math.isclose(reported, float(value), rel_tol=0.0, abs_tol=ORIGINAL_METRIC_ABS_TOLERANCE):
            raise ValueError(
                f"Original WLCR raw {name} mismatch at {key}: {reported} != {value}"
            )
    per_indicator = metrics["per_indicator"]
    for item in per_indicator:
        indicator = str(item["indicator"])
        row = artifact.indicator_rows[(*key, indicator)]
        if row["method"] != "original_wlcr":
            raise ValueError(f"Original WLCR per-indicator method mismatch: {key}/{indicator}")
        for name in ("wape", "mase", "smape"):
            reported = float(row[name])
            if not math.isclose(
                reported,
                float(item[name]),
                rel_tol=0.0,
                abs_tol=ORIGINAL_METRIC_ABS_TOLERANCE,
            ):
                raise ValueError(
                    f"Original WLCR per-indicator {name} mismatch at {key}/{indicator}"
                )


def curve_row(
    *,
    method: str,
    scenario: Scenario,
    corruption_seed: int,
    mask_sha256: str,
    flat_stats: Mapping[str, object],
    values: Mapping[str, float],
) -> dict[str, object]:
    meta = METHOD_META[method]
    return {
        "method": method,
        "label": meta["label"],
        "training_view": meta["training_view"],
        "comparison_class": meta["comparison_class"],
        "comparison_note": meta["comparison_note"],
        "scenario": scenario.scenario,
        "mechanism": scenario.mechanism,
        "mechanism_display": scenario.mechanism_display,
        "requested_rate": scenario.requested_rate,
        "corruption_seed": corruption_seed,
        "mask_sha256": mask_sha256,
        **flat_stats,
        **values,
    }


def indicator_rows(
    *,
    method: str,
    scenario: Scenario,
    corruption_seed: int,
    mask_sha256: str,
    metrics: Mapping[str, object],
) -> list[dict[str, object]]:
    meta = METHOD_META[method]
    result: list[dict[str, object]] = []
    for item in metrics["per_indicator"]:
        result.append(
            {
                "method": method,
                "label": meta["label"],
                "training_view": meta["training_view"],
                "comparison_class": meta["comparison_class"],
                "comparison_note": meta["comparison_note"],
                "scenario": scenario.scenario,
                "mechanism": scenario.mechanism,
                "mechanism_display": scenario.mechanism_display,
                "requested_rate": scenario.requested_rate,
                "corruption_seed": corruption_seed,
                "mask_sha256": mask_sha256,
                "indicator": item["indicator"],
                "wape": item["wape"],
                "mase": item["mase"],
                "smape": item["smape"],
            }
        )
    return result


def bootstrap_row(
    *,
    scenario: Scenario,
    condition: str,
    baseline_key: str,
    baseline_label: str,
    comparison_class: str,
    result: Mapping[str, object],
) -> dict[str, object]:
    return {
        "condition": condition,
        "scenario": scenario.scenario,
        "mechanism": scenario.mechanism,
        "mechanism_display": scenario.mechanism_display,
        "requested_rate": scenario.requested_rate,
        "proposed": "A6_mixed_aug",
        "proposed_label": METHOD_META["A6_mixed_aug"]["label"],
        "baseline": baseline_key,
        "baseline_label": baseline_label,
        "comparison_class": comparison_class,
        "estimand": result["estimand"],
        "corruption_seed_aggregation": result["corruption_seed_aggregation"],
        "corruption_seed_count": result["corruption_seed_count"],
        "corruption_seeds": ",".join(str(value) for value in result["corruption_seeds"]),
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


def bootstrap_seed_rows(
    *,
    scenario: Scenario,
    condition: str,
    baseline_key: str,
    comparison_class: str,
    result: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, corruption_seed in enumerate(result["corruption_seeds"]):
        rows.append(
            {
                "condition": condition,
                "scenario": scenario.scenario,
                "mechanism": scenario.mechanism,
                "mechanism_display": scenario.mechanism_display,
                "requested_rate": scenario.requested_rate,
                "corruption_seed": corruption_seed,
                "proposed": "A6_mixed_aug",
                "baseline": baseline_key,
                "comparison_class": comparison_class,
                "a6_macro_wape": result["point_proposed_macro_wape_by_corruption_seed"][index],
                "baseline_macro_wape": result[
                    "point_baseline_macro_wape_by_corruption_seed"
                ][index],
                "delta_a6_minus_baseline": result[
                    "delta_proposed_minus_baseline_by_corruption_seed"
                ][index],
            }
        )
    return rows


def preflight_original_stress_asset(
    artifact: OriginalWLCRArtifact,
    *,
    dataset: neural.CachedDataset,
    holdout: np.ndarray,
    actual: np.ndarray,
    scales: np.ndarray,
    cells: np.ndarray,
    thresholds: np.ndarray,
    scenarios: Sequence[Scenario],
    corruption_seeds: Sequence[int],
    clean_reference: np.ndarray,
) -> dict[str, float]:
    """Validate every requested Original-WLCR mask/prediction without GPU writes."""
    clean_maximums: dict[str, float] = {}
    for scenario in scenarios:
        for corruption_seed in corruption_seeds:
            extra = missingness.global_corruption_mask(
                np.asarray(dataset.cells[holdout]),
                np.asarray(dataset.history_end_hours[holdout]),
                mechanism=scenario.mechanism,
                requested_rate=scenario.requested_rate,
                seed=corruption_seed,
            )
            prediction, raw = load_artifact_mask_and_prediction(
                artifact,
                scenario,
                corruption_seed,
                expected_mask=extra,
                prediction_shape=actual.shape,
            )
            values, metrics = metric_payload(prediction, actual, scales, cells, thresholds)
            if raw["mask_sha256"] != hash_mask(extra):
                raise ValueError("Original WLCR raw mask hash differs from the global mask")
            assert_original_metric_rows(
                artifact, scenario, corruption_seed, values, metrics
            )
            if scenario.requested_rate == 0.0:
                clean_maximums[str(corruption_seed)] = validate_original_clean_replay(
                    artifact,
                    corruption_seed=corruption_seed,
                    prediction=prediction,
                    clean_reference=clean_reference,
                )
    return clean_maximums


def condition_map(scenarios: Sequence[Scenario]) -> dict[tuple[str, float], str]:
    requested = {scenario.key for scenario in scenarios}
    values: dict[tuple[str, float], str] = {}
    for mechanism, rate, name in robustness.CONDITIONS:
        key = (mechanism, canonical_rate(rate))
        if key in requested:
            values[key] = name
    return values


def assert_cuda_devices(devices: Sequence[int]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Revision-9 missingness analysis requires CUDA")
    if max(devices) >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested GPU ids {list(devices)} exceed visible devices "
            f"{torch.cuda.device_count()}"
        )


def run(args: argparse.Namespace) -> int:
    root = runner.project_root()
    sea_root = resolve_inside(root, args.sea_root, strict=True)
    augmented_neural_root = resolve_inside(root, args.augmented_neural_root, strict=True)
    grud_root = resolve_inside(root, args.grud_root, strict=True)
    original_root = resolve_inside(root, args.original_wlcr_root, strict=True)
    original_clean = resolve_inside(root, args.original_wlcr_clean, strict=True)
    output = resolve_output(root, args.output)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    scenarios = smoke_scenarios() if args.smoke else full_scenarios()
    corruption_seeds = (CORRUPTION_SEEDS[0],) if args.smoke else CORRUPTION_SEEDS
    if args.validate_original_only:
        devices: tuple[int, ...] = ()
    else:
        devices = parse_gpu_devices(args.gpu_devices)
        assert_cuda_devices(devices)

    dataset, dataset_report, train_path = revision6.dataset_from_registered_train()
    registered_train = (root / "data/train_data.csv").resolve(strict=True)
    if train_path.resolve(strict=True) != registered_train:
        raise RuntimeError("Revision-9 loader did not resolve the registered training trace")
    train_hash_before = neural.sha256_file(train_path)
    fit = dataset.indices_for_dates(neural.FIT_DATES)
    inner = dataset.indices_for_dates(neural.INNER_DATES)
    final_train = np.concatenate((fit, inner))
    holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
    # Smoke keeps the full holdout identity intact; it reduces only conditions
    # and corruption seeds, never requests or histories.
    actual = np.asarray(dataset.targets[holdout], dtype=np.float32)
    scales = np.asarray(dataset.mase_scales[holdout], dtype=np.float32)
    cells = np.asarray(dataset.cells[holdout])
    shape = actual.shape
    thresholds = sea.frozen_low_activity_thresholds(
        dataset.targets, dataset.target_masks, final_train
    )

    original_artifact, original_manifest = load_original_artifact(
        original_root,
        dataset=dataset,
        holdout=holdout,
        actual=actual,
        scales=scales,
        thresholds=thresholds,
        train_hash=train_hash_before,
        scenarios=scenarios,
        corruption_seeds=corruption_seeds,
    )
    if runner.sha256_file(original_clean) != str(
        original_artifact.clean_replay.get("reference_sha256", "")
    ):
        raise ValueError("Original WLCR clean replay reference hash does not match disk")
    original_clean_prediction = align_original_clean_reference(
        original_clean.parent,
        revision6.load_prediction(original_clean, shape),
        expected_identity_rows(dataset, holdout),
    )
    if args.validate_original_only:
        clean_maximums = preflight_original_stress_asset(
            original_artifact,
            dataset=dataset,
            holdout=holdout,
            actual=actual,
            scales=scales,
            cells=cells,
            thresholds=thresholds,
            scenarios=scenarios,
            corruption_seeds=corruption_seeds,
            clean_reference=original_clean_prediction,
        )
        train_hash_after = neural.sha256_file(train_path)
        if train_hash_before != train_hash_after:
            raise RuntimeError(
                "registered training trace changed during Original-WLCR preflight"
            )
        print(
            json.dumps(
                {
                    "status": "original_wlcr_preflight_complete",
                    "artifact_root": str(original_root.relative_to(root)),
                    "scenario_count": len(scenarios),
                    "corruption_seeds": list(corruption_seeds),
                    "clean_replay_maximums": clean_maximums,
                    "finals_test_opened": False,
                }
            )
        )
        return 0

    atomic_output_directory(output)
    a6_device = torch.device(f"cuda:{devices[0]}")
    dlinear_device = torch.device(f"cuda:{devices[1]}")
    patchtst_device = torch.device(f"cuda:{devices[2]}")
    grud_device = torch.device(f"cuda:{devices[3] if len(devices) > 3 else devices[2]}")
    a6_models, prior, a6_records = load_a6_ensemble(root, sea_root, a6_device)
    dlinear_models, dlinear_norm, dlinear_records = load_augmented_neural_ensemble(
        root, augmented_neural_root, "dlinear", dlinear_device
    )
    patchtst_models, patchtst_norm, patchtst_records = load_augmented_neural_ensemble(
        root, augmented_neural_root, "patchtst", patchtst_device
    )
    grud_models, grud_norm, grud_records = load_augmented_neural_ensemble(
        root, grud_root, "grud_direct", grud_device
    )
    clean_references = {
        "A6_mixed_aug": load_clean_ensemble(sea_root, "A6_mixed_aug", shape),
        "dlinear_aug": load_clean_ensemble(augmented_neural_root, "dlinear", shape),
        "patchtst_aug": load_clean_ensemble(augmented_neural_root, "patchtst", shape),
        "grud_direct_aug": load_clean_ensemble(grud_root, "grud_direct", shape),
        "original_wlcr": original_clean_prediction,
    }

    raw_curve_rows: list[dict[str, object]] = []
    raw_indicator_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    paired_seed_rows: list[dict[str, object]] = []
    descriptive_rows: list[dict[str, object]] = []
    descriptive_seed_rows: list[dict[str, object]] = []
    detailed_bootstrap: list[dict[str, object]] = []
    clean_replay_maximums: dict[str, float] = {}
    conditions = condition_map(scenarios)

    for scenario in scenarios:
        scenario_predictions: dict[str, list[np.ndarray]] = {
            method: [] for method in METHOD_META
        }
        for corruption_seed in corruption_seeds:
            extra = missingness.global_corruption_mask(
                np.asarray(dataset.cells[holdout]),
                np.asarray(dataset.history_end_hours[holdout]),
                mechanism=scenario.mechanism,
                requested_rate=scenario.requested_rate,
                seed=corruption_seed,
            )
            statistics_payload = missingness.corruption_statistics(
                np.asarray(dataset.x_masks[holdout]),
                extra,
                cells=np.asarray(dataset.cells[holdout]),
                history_end_hours=np.asarray(dataset.history_end_hours[holdout]),
                mechanism=scenario.mechanism,
                requested_rate=scenario.requested_rate,
                seed=corruption_seed,
            )
            flat_stats = missingness.flatten_statistics(statistics_payload)
            mask_sha256 = hash_mask(extra)
            _, sea_tensors = runner.make_eval_tensors(
                dataset, holdout, prior, additional_missing=extra
            )
            neural_values, neural_masks = shared_neural_request_view(dataset, holdout, extra)
            predictions: dict[str, np.ndarray] = {
                "A6_mixed_aug": predict_sea_ensemble(
                    a6_models, sea_tensors, a6_device, args.batch_size
                ),
                "dlinear_aug": predict_neural_ensemble(
                    dlinear_models,
                    normalized_neural_inputs(neural_values, neural_masks, dlinear_norm),
                    dlinear_norm,
                    dlinear_device,
                    args.batch_size,
                ),
                "patchtst_aug": predict_neural_ensemble(
                    patchtst_models,
                    normalized_neural_inputs(neural_values, neural_masks, patchtst_norm),
                    patchtst_norm,
                    patchtst_device,
                    args.batch_size,
                ),
                "grud_direct_aug": predict_neural_ensemble(
                    grud_models,
                    normalized_neural_inputs(neural_values, neural_masks, grud_norm),
                    grud_norm,
                    grud_device,
                    args.batch_size,
                ),
            }
            original_prediction, original_raw = load_artifact_mask_and_prediction(
                original_artifact,
                scenario,
                corruption_seed,
                expected_mask=extra,
                prediction_shape=shape,
            )
            predictions["original_wlcr"] = original_prediction
            for method, prediction in predictions.items():
                robustness.validate_prediction(
                    f"{scenario.scenario}/{corruption_seed}/{method}", prediction, shape
                )
                if scenario.requested_rate == 0.0:
                    maximum = float(
                        np.max(
                            np.abs(
                                prediction.astype(np.float64)
                                - clean_references[method].astype(np.float64)
                            )
                        )
                    )
                    prior_maximum = clean_replay_maximums.get(method, 0.0)
                    clean_replay_maximums[method] = max(prior_maximum, maximum)
                    if maximum > CLEAN_REPLAY_ABS_TOLERANCE:
                        raise ValueError(
                            f"{method} clean replay mismatch: {maximum} exceeds "
                            f"{CLEAN_REPLAY_ABS_TOLERANCE}"
                        )
                scenario_predictions[method].append(prediction)
                values, metrics = metric_payload(prediction, actual, scales, cells, thresholds)
                if method == "original_wlcr":
                    if original_raw["mask_sha256"] != mask_sha256:
                        raise ValueError("Original WLCR raw mask hash differs from recomputed mask")
                    assert_original_metric_rows(
                        original_artifact,
                        scenario,
                        corruption_seed,
                        values,
                        metrics,
                    )
                    if scenario.requested_rate == 0.0:
                        validate_original_clean_replay(
                            original_artifact,
                            corruption_seed=corruption_seed,
                            prediction=prediction,
                            clean_reference=original_clean_prediction,
                        )
                raw_curve_rows.append(
                    curve_row(
                        method=method,
                        scenario=scenario,
                        corruption_seed=corruption_seed,
                        mask_sha256=mask_sha256,
                        flat_stats=flat_stats,
                        values=values,
                    )
                )
                raw_indicator_rows.extend(
                    indicator_rows(
                        method=method,
                        scenario=scenario,
                        corruption_seed=corruption_seed,
                        mask_sha256=mask_sha256,
                        metrics=metrics,
                    )
                )

        condition = conditions.get(scenario.key)
        if condition is None:
            continue
        for baseline_key, baseline_label in MATCHED_BASELINES:
            result = robustness.paired_multi_seed_cell_cluster_bootstrap(
                actual,
                scenario_predictions["A6_mixed_aug"],
                scenario_predictions[baseline_key],
                cells,
                corruption_seeds=corruption_seeds,
                replicates=args.bootstrap_replicates,
                seed=robustness.stable_bootstrap_seed(
                    args.bootstrap_seed, condition, baseline_key
                ),
            )
            paired_rows.append(
                bootstrap_row(
                    scenario=scenario,
                    condition=condition,
                    baseline_key=baseline_key,
                    baseline_label=baseline_label,
                    comparison_class="augmentation_matched",
                    result=result,
                )
            )
            paired_seed_rows.extend(
                bootstrap_seed_rows(
                    scenario=scenario,
                    condition=condition,
                    baseline_key=baseline_key,
                    comparison_class="augmentation_matched",
                    result=result,
                )
            )
            detailed_bootstrap.append(
                {
                    "comparison_class": "augmentation_matched",
                    "condition": condition,
                    "mechanism": scenario.mechanism,
                    "requested_rate": scenario.requested_rate,
                    "baseline": baseline_key,
                    "baseline_label": baseline_label,
                    **result,
                }
            )
        original_key, original_label = DESCRIPTIVE_BASELINE
        original_result = robustness.paired_multi_seed_cell_cluster_bootstrap(
            actual,
            scenario_predictions["A6_mixed_aug"],
            scenario_predictions[original_key],
            cells,
            corruption_seeds=corruption_seeds,
            replicates=args.bootstrap_replicates,
            seed=robustness.stable_bootstrap_seed(
                args.bootstrap_seed, condition, original_key
            ),
        )
        descriptive_rows.append(
            bootstrap_row(
                scenario=scenario,
                condition=condition,
                baseline_key=original_key,
                baseline_label=original_label,
                comparison_class="descriptive_clean_trained_not_augmentation_matched",
                result=original_result,
            )
        )
        descriptive_seed_rows.extend(
            bootstrap_seed_rows(
                scenario=scenario,
                condition=condition,
                baseline_key=original_key,
                comparison_class="descriptive_clean_trained_not_augmentation_matched",
                result=original_result,
            )
        )
        detailed_bootstrap.append(
            {
                "comparison_class": "descriptive_clean_trained_not_augmentation_matched",
                "condition": condition,
                "mechanism": scenario.mechanism,
                "requested_rate": scenario.requested_rate,
                "baseline": original_key,
                "baseline_label": original_label,
                "interpretation_restriction": (
                    "not augmentation matched; descriptive stress comparison only"
                ),
                **original_result,
            }
        )

    matched_numeric = (
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
    curve_aggregate = aggregate_seed_rows(
        raw_curve_rows,
        group_fields=(
            "method",
            "label",
            "training_view",
            "comparison_class",
            "comparison_note",
            "scenario",
            "mechanism",
            "mechanism_display",
            "requested_rate",
        ),
        numeric_fields=matched_numeric,
    )
    indicator_aggregate = aggregate_seed_rows(
        raw_indicator_rows,
        group_fields=(
            "method",
            "label",
            "training_view",
            "comparison_class",
            "comparison_note",
            "scenario",
            "mechanism",
            "mechanism_display",
            "requested_rate",
            "indicator",
        ),
        numeric_fields=("wape", "mase", "smape"),
    )
    expected_raw_rows = len(scenarios) * len(corruption_seeds) * len(METHOD_META)
    expected_indicator_raw_rows = expected_raw_rows * len(sea.METRIC_NAMES)
    if len(raw_curve_rows) != expected_raw_rows:
        raise RuntimeError("Revision-9 curve row count is incomplete")
    if len(raw_indicator_rows) != expected_indicator_raw_rows:
        raise RuntimeError("Revision-9 per-indicator row count is incomplete")
    if not args.smoke and len(paired_rows) != len(robustness.CONDITIONS) * len(MATCHED_BASELINES):
        raise RuntimeError("Revision-9 matched bootstrap table is incomplete")
    if not args.smoke and len(descriptive_rows) != len(robustness.CONDITIONS):
        raise RuntimeError("Revision-9 descriptive Original WLCR table is incomplete")

    train_hash_after = neural.sha256_file(train_path)
    if train_hash_before != train_hash_after:
        raise RuntimeError("registered training trace changed during Revision-9 analysis")
    runner.atomic_csv(output / "comparative_missingness_by_seed.csv", raw_curve_rows)
    runner.atomic_csv(output / "comparative_missingness.csv", curve_aggregate)
    runner.atomic_csv(
        output / "comparative_missingness_per_indicator_by_seed.csv", raw_indicator_rows
    )
    runner.atomic_csv(output / "comparative_missingness_per_indicator.csv", indicator_aggregate)
    if paired_rows:
        runner.atomic_csv(output / "paired_cell_bootstrap.csv", paired_rows)
        runner.atomic_csv(output / "corruption_seed_wape.csv", paired_seed_rows)
    if descriptive_rows:
        runner.atomic_csv(
            output / "paired_cell_bootstrap_original_wlcr_descriptive.csv", descriptive_rows
        )
        runner.atomic_csv(
            output / "corruption_seed_wape_original_wlcr_descriptive.csv",
            descriptive_seed_rows,
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "purpose": (
            "Revision-9 missingness curves with a GRU-D-Direct-Aug control and "
            "a separately ingested Original WLCR clean-trained stress curve"
        ),
        "registered_train_file": "data/train_data.csv",
        "registered_train_sha256_before": train_hash_before,
        "registered_train_sha256_after": train_hash_after,
        "finals_test_opened": False,
        "protocol": {
            "scope": "absolute cell-time timeline before overlapping windows",
            "scenarios": [
                {
                    "scenario": scenario.scenario,
                    "mechanism": scenario.mechanism,
                    "requested_rate": scenario.requested_rate,
                }
                for scenario in scenarios
            ],
            "scenario_count": len(scenarios),
            "corruption_seeds": list(corruption_seeds),
            "full_revision8_protocol": not args.smoke,
            "rate_zero_deduplicated_as": "mcar/0.00/clean",
        },
        "models": {
            "A6_mixed_aug": a6_records,
            "dlinear_aug": dlinear_records,
            "patchtst_aug": patchtst_records,
            "grud_direct_aug": grud_records,
            "original_wlcr": {
                "artifact_root": str(original_root.relative_to(root)),
                "request_identity_sha256": original_artifact.request_identity_sha256,
                "manifest_records_verified": len(original_manifest),
                "training_status": "clean_trained_frozen",
                "comparison_class": "descriptive_clean_trained_not_augmentation_matched",
            },
        },
        "inference": {
            "gpu_devices": list(devices),
            "device_assignment": {
                "A6_mixed_aug": str(a6_device),
                "dlinear_aug": str(dlinear_device),
                "patchtst_aug": str(patchtst_device),
                "grud_direct_aug": str(grud_device),
            },
            "batch_size": args.batch_size,
            "five_model_seed_predictions_averaged_before_scoring": True,
            "shared_corruption_dependent_fill_view_for_neural_baselines": True,
        },
        "clean_replay": {
            "absolute_tolerance": CLEAN_REPLAY_ABS_TOLERANCE,
            "maximum_absolute_difference": clean_replay_maximums,
            "original_worker_clean_replay_verified": True,
        },
        "bootstrap_protocol": {
            "metric": "macro-over-indicator WAPE",
            "replicates": args.bootstrap_replicates,
            "base_seed": args.bootstrap_seed,
            "cluster": "cell",
            "within_cluster_preserved": "all requests, horizons, and indicators",
            "corruption_seed_handling": (
                "paired delta per fixed corruption seed, averaged inside every "
                "cell-cluster bootstrap replicate"
            ),
            "matched_table": "paired_cell_bootstrap.csv",
            "original_wlcr_table": "paired_cell_bootstrap_original_wlcr_descriptive.csv",
            "original_wlcr_excluded_from_matched_claims": True,
        },
        "dataset_report": dataset_report,
        "frozen_training_thresholds": thresholds.tolist(),
        "rows": {
            "curve_by_seed": len(raw_curve_rows),
            "curve_aggregate": len(curve_aggregate),
            "per_indicator_by_seed": len(raw_indicator_rows),
            "per_indicator_aggregate": len(indicator_aggregate),
            "paired_matched": len(paired_rows),
            "paired_original_descriptive": len(descriptive_rows),
        },
        "bootstrap_results": detailed_bootstrap,
    }
    runner.atomic_json(output / "summary.json", summary)
    runner.atomic_json(output / "manifest.json", runner.output_manifest(output))
    print(json.dumps({"status": "complete", "output": str(output.relative_to(root))}))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--sea-root", default=str(DEFAULT_SEA_ROOT))
    value.add_argument(
        "--augmented-neural-root", default=str(DEFAULT_AUGMENTED_NEURAL_ROOT)
    )
    value.add_argument("--grud-root", default=str(DEFAULT_GRUD_ROOT))
    value.add_argument("--original-wlcr-root", default=str(DEFAULT_ORIGINAL_WLCR_ROOT))
    value.add_argument("--original-wlcr-clean", default=str(DEFAULT_ORIGINAL_WLCR_CLEAN))
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    value.add_argument("--gpu-devices", default="0,1,2,3")
    value.add_argument("--batch-size", type=int, default=256)
    value.add_argument("--bootstrap-replicates", type=int, default=5000)
    value.add_argument("--bootstrap-seed", type=int, default=20260729)
    value.add_argument(
        "--smoke",
        action="store_true",
        help="evaluate only clean and 20 percent block with seed 142; never use for paper results",
    )
    value.add_argument(
        "--validate-original-only",
        action="store_true",
        help=(
            "validate the Original-WLCR artifact identity, masks, predictions, and "
            "metrics without GPU inference or output writes"
        ),
    )
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
