from __future__ import annotations

"""Revision-8 request-local invariance audit for the WLCR-SEA trace.

This is an implementation-property audit, not a new performance experiment.
It reconstructs the registered *training* trace, selects 256 deterministic
holdout requests across target-date and input-missingness strata, then changes
every other request (including other requests for the same cell).  The target
request's expert tensors and the frozen seed-42 model outputs must remain
bitwise identical.  The finals traffic file is deliberately never opened.
"""

import argparse
import hashlib
import inspect
import json
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from experiments import train_neural_baselines as neural
from experiments import train_wlcr_sea as runner


SCHEMA_VERSION = 1
DEFAULT_SOURCE = Path("artifacts/revision7/wlcr_sea")
DEFAULT_OUTPUT = Path("artifacts/revision8/audit")
DEFAULT_CHECKPOINT_VARIANT = "A6_mixed_aug"
DEFAULT_MODEL_SEED = 42
DEFAULT_SAMPLE_SIZE = 256
DEFAULT_AUDIT_SEED = 7042
EXPERT_FIELDS = ("values", "availability", "reliability", "context")
EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class AuditRequest:
    """A deterministic training-trace request selected for locality testing."""

    index: int
    cell: str
    target_date: str
    missing_count: int
    input_elements: int
    missingness_bin: str

    @property
    def missing_rate(self) -> float:
        return self.missing_count / self.input_elements

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "cell": self.cell,
            "target_date": self.target_date,
            "missing_count": self.missing_count,
            "input_elements": self.input_elements,
            "input_missing_rate": self.missing_rate,
            "missingness_bin": self.missingness_bin,
        }


def canonical_sha256(payload: object) -> str:
    """Hash JSON with a stable byte representation suitable for provenance."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def missingness_bin(missing_count: int, input_elements: int) -> str:
    """Return a fixed, predeclared input-missingness stratum label."""
    if input_elements <= 0:
        raise ValueError("request must contain at least one input element")
    if not 0 <= missing_count <= input_elements:
        raise ValueError("missing-count value is outside the request bounds")
    rate = missing_count / input_elements
    if missing_count == 0:
        return "none_0pct"
    if rate <= 0.10:
        return "low_0_to_10pct"
    if rate <= 0.25:
        return "moderate_10_to_25pct"
    return "high_above_25pct"


def build_audit_requests(
    dataset: neural.CachedDataset, holdout: np.ndarray
) -> list[AuditRequest]:
    """Describe holdout requests using only pre-inference identifiers and masks."""
    indices = np.asarray(holdout, dtype=np.int64).reshape(-1)
    if not len(indices):
        raise ValueError("holdout request list is empty")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("holdout request list contains duplicate indices")
    masks = np.asarray(dataset.x_masks[indices], dtype=np.uint8)
    if masks.ndim < 2:
        raise ValueError("input-mask tensor does not have a request axis")
    flattened = masks.reshape(len(indices), -1)
    input_elements = int(flattened.shape[1])
    counts = np.sum(flattened == 0, axis=1, dtype=np.int64)
    cells = np.asarray(dataset.cells).astype(str)
    target_hours = np.asarray(dataset.target_start_hours, dtype=np.int64)
    records: list[AuditRequest] = []
    for index, count in zip(indices.tolist(), counts.tolist(), strict=True):
        target_date = (EPOCH + timedelta(hours=int(target_hours[index]))).date().isoformat()
        records.append(
            AuditRequest(
                index=int(index),
                cell=str(cells[index]),
                target_date=target_date,
                missing_count=int(count),
                input_elements=input_elements,
                missingness_bin=missingness_bin(int(count), input_elements),
            )
        )
    return records


def _stable_rank(record: AuditRequest, audit_seed: int) -> tuple[str, int]:
    token = (
        f"revision8-request-local|{audit_seed}|{record.target_date}|"
        f"{record.missingness_bin}|{record.cell}|{record.index}"
    )
    return hashlib.sha256(token.encode("utf-8")).hexdigest(), record.index


def _strata(records: Sequence[AuditRequest]) -> dict[tuple[str, str], list[AuditRequest]]:
    grouped: dict[tuple[str, str], list[AuditRequest]] = {}
    for record in records:
        grouped.setdefault((record.target_date, record.missingness_bin), []).append(record)
    return grouped


def allocate_stratum_quotas(
    records: Sequence[AuditRequest], sample_size: int, audit_seed: int
) -> dict[tuple[str, str], int]:
    """Allocate exact quotas while retaining every nonempty stratum when possible."""
    if sample_size <= 0:
        raise ValueError("sample size must be positive")
    if sample_size > len(records):
        raise ValueError("sample size exceeds the available holdout requests")
    groups = _strata(records)
    keys = sorted(groups)
    if not keys:
        raise ValueError("no nonempty sampling strata")
    quotas = {key: 0 for key in keys}
    if sample_size >= len(keys):
        for key in keys:
            quotas[key] = 1
    else:
        ordered = sorted(
            keys,
            key=lambda key: hashlib.sha256(
                f"revision8-stratum|{audit_seed}|{key[0]}|{key[1]}".encode("utf-8")
            ).hexdigest(),
        )
        for key in ordered[:sample_size]:
            quotas[key] = 1

    remaining = sample_size - sum(quotas.values())
    while remaining:
        capacity = {key: len(groups[key]) - quotas[key] for key in keys}
        eligible = [key for key in keys if capacity[key] > 0]
        if not eligible:
            raise ValueError("stratum allocation exhausted before reaching sample size")
        total_capacity = sum(capacity[key] for key in eligible)
        additions = {
            key: min(
                capacity[key], (remaining * capacity[key]) // total_capacity
            )
            for key in eligible
        }
        assigned = sum(additions.values())
        if assigned:
            for key, value in additions.items():
                quotas[key] += value
            remaining -= assigned
            continue
        ordered = sorted(
            eligible,
            key=lambda key: (
                -((remaining * capacity[key]) % total_capacity),
                hashlib.sha256(
                    f"revision8-remainder|{audit_seed}|{key[0]}|{key[1]}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            ),
        )
        for key in ordered:
            if remaining == 0:
                break
            quotas[key] += 1
            remaining -= 1
    if sum(quotas.values()) != sample_size:
        raise AssertionError("stratum allocation did not conserve the sample size")
    if any(quotas[key] > len(groups[key]) for key in keys):
        raise AssertionError("stratum allocation exceeds an available population")
    return quotas


def select_stratified_requests(
    records: Sequence[AuditRequest], sample_size: int, audit_seed: int
) -> tuple[list[AuditRequest], dict[tuple[str, str], int]]:
    """Choose a deterministic date × missingness sample with cell diversity.

    Quotas are fixed before request ranking.  Within each stratum, the first
    pass prefers cells not selected elsewhere; this broadens the cell coverage
    without using labels or model outputs.  A repeated cell is used only when a
    stratum has no remaining unseen cell.
    """
    quotas = allocate_stratum_quotas(records, sample_size, audit_seed)
    grouped = _strata(records)
    ordered: dict[tuple[str, str], list[AuditRequest]] = {
        key: sorted(grouped[key], key=lambda record: _stable_rank(record, audit_seed))
        for key in sorted(grouped)
    }
    cursor = {key: 0 for key in ordered}
    selected_counts = {key: 0 for key in ordered}
    selected: list[AuditRequest] = []
    selected_cells: set[str] = set()
    while len(selected) < sample_size:
        progressed = False
        for key in sorted(ordered):
            if selected_counts[key] >= quotas[key]:
                continue
            candidates = ordered[key]
            start = cursor[key]
            unseen_position = next(
                (
                    position
                    for position in range(start, len(candidates))
                    if candidates[position].cell not in selected_cells
                ),
                None,
            )
            position = unseen_position if unseen_position is not None else start
            if position >= len(candidates):
                raise AssertionError("selection cursor exhausted before quota")
            record = candidates[position]
            if position != start:
                candidates[start], candidates[position] = candidates[position], candidates[start]
                record = candidates[start]
            cursor[key] = start + 1
            selected.append(record)
            selected_counts[key] += 1
            selected_cells.add(record.cell)
            progressed = True
            if len(selected) == sample_size:
                break
        if not progressed:
            raise AssertionError("sampling made no progress")
    if len({record.index for record in selected}) != len(selected):
        raise AssertionError("sampling selected a request more than once")
    return sorted(selected, key=lambda record: record.index), quotas


def target_list_payload(
    selected: Sequence[AuditRequest],
    quotas: Mapping[tuple[str, str], int],
    population: Sequence[AuditRequest],
    *,
    audit_seed: int,
) -> dict[str, object]:
    """Build compact, independently hashable selection provenance."""
    grouped = _strata(population)
    selected_grouped = _strata(selected)
    strata: dict[str, dict[str, int]] = {}
    for key in sorted(grouped):
        label = f"{key[0]}|{key[1]}"
        strata[label] = {
            "population": len(grouped[key]),
            "quota": int(quotas[key]),
            "selected": len(selected_grouped.get(key, ())),
        }
    targets = [record.as_dict() for record in selected]
    return {
        "schema_version": SCHEMA_VERSION,
        "sampling": {
            "algorithm": (
                "fixed date-by-input-missingness quotas with stable SHA256 ranking "
                "and first-pass cross-stratum cell diversity"
            ),
            "audit_seed": audit_seed,
            "requested_sample_size": len(selected),
            "strata": strata,
        },
        "targets": targets,
        "target_list_sha256": canonical_sha256(targets),
    }


def build_global_perturbation(
    values: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Generate a guaranteed bitwise change for every input value and mask."""
    original_values = np.asarray(values, dtype=np.float32)
    original_masks = np.asarray(masks, dtype=np.uint8)
    if original_values.shape != original_masks.shape:
        raise ValueError("input values and masks must have identical shapes")
    if not np.all(np.isfinite(original_values)):
        raise ValueError("request-local audit requires finite filled input values")
    if not np.all((original_masks == 0) | (original_masks == 1)):
        raise ValueError("request-local audit requires binary input masks")
    perturbed_values = np.nextafter(original_values, np.float32(np.inf)).astype(
        np.float32, copy=False
    )
    perturbed_masks = np.bitwise_xor(original_masks, np.uint8(1))
    if not np.all(perturbed_values != original_values):
        raise AssertionError("value perturbation did not change every input element")
    if not np.all(perturbed_masks != original_masks):
        raise AssertionError("mask perturbation did not change every input element")
    return perturbed_values, perturbed_masks, {
        "value_perturbation": "np.nextafter(value, +infinity) in float32",
        "mask_perturbation": "binary complement (1 XOR mask)",
        "globally_changed_value_elements": int(original_values.size),
        "globally_changed_mask_elements": int(original_masks.size),
    }


def _expert_equal(
    first: object, second: object
) -> tuple[dict[str, bool], dict[str, bool]]:
    array_equal = {
        field: np.array_equal(getattr(first, field), getattr(second, field))
        for field in EXPERT_FIELDS
    }
    tensor_equal = {
        field: torch.equal(
            torch.as_tensor(getattr(first, field)), torch.as_tensor(getattr(second, field))
        )
        for field in EXPERT_FIELDS
    }
    return array_equal, tensor_equal


def compare_target_request(
    model: torch.nn.Module,
    original: neural.CachedDataset,
    altered: neural.CachedDataset,
    index: int,
    prior: np.ndarray,
    *,
    tensor_builder: Callable[..., tuple[object, tuple[torch.Tensor, ...]]] = runner.make_eval_tensors,
) -> dict[str, object]:
    """Compare one target request while every other request is perturbed."""
    indices = np.asarray([index], dtype=np.int64)
    clean_batch, clean_tensors = tensor_builder(original, indices, prior)
    altered_batch, altered_tensors = tensor_builder(altered, indices, prior)
    expert_arrays, expert_tensors = _expert_equal(clean_batch, altered_batch)
    with torch.inference_mode():
        clean_output = model(
            clean_tensors[0], clean_tensors[1].bool(), clean_tensors[2], clean_tensors[3]
        )
        altered_output = model(
            altered_tensors[0],
            altered_tensors[1].bool(),
            altered_tensors[2],
            altered_tensors[3],
        )
    output_keys = sorted(set(clean_output) | set(altered_output))
    output_tensors = {
        field: (
            field in clean_output
            and field in altered_output
            and isinstance(clean_output[field], torch.Tensor)
            and isinstance(altered_output[field], torch.Tensor)
            and torch.equal(clean_output[field], altered_output[field])
        )
        for field in output_keys
    }
    passed = all(expert_arrays.values()) and all(expert_tensors.values()) and all(
        output_tensors.values()
    )
    return {
        "expert_arrays_bitwise_identical": expert_arrays,
        "expert_tensors_bitwise_identical": expert_tensors,
        "model_output_tensors_bitwise_identical": output_tensors,
        "bitwise_request_local_invariance_pass": passed,
    }


def audit_selected_requests(
    dataset: neural.CachedDataset,
    selected: Sequence[AuditRequest],
    model: torch.nn.Module,
    prior: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Mutate each target's complete complement and record the bitwise checks."""
    original_values = np.asarray(dataset.x_values, dtype=np.float32)
    original_masks = np.asarray(dataset.x_masks, dtype=np.uint8)
    perturbed_values, perturbed_masks, perturbation = build_global_perturbation(
        original_values, original_masks
    )
    altered = replace(dataset, x_values=perturbed_values, x_masks=perturbed_masks)
    cells = np.asarray(dataset.cells).astype(str)
    request_count = len(cells)
    perturbed_request_state = np.ones(request_count, dtype=bool)
    details: list[dict[str, object]] = []
    for request in selected:
        index = request.index
        target_cell = request.cell
        same_cell = cells == target_cell
        non_target = np.ones(request_count, dtype=bool)
        non_target[index] = False
        perturbed_values[index] = original_values[index]
        perturbed_masks[index] = original_masks[index]
        perturbed_request_state[index] = False
        try:
            exact_target_history = np.array_equal(
                perturbed_values[index], original_values[index]
            ) and np.array_equal(perturbed_masks[index], original_masks[index])
            complement_state = bool(
                np.array_equal(perturbed_request_state, non_target)
            )
            same_cell_noncurrent = same_cell & non_target
            check = compare_target_request(model, dataset, altered, index, prior)
            details.append(
                {
                    **request.as_dict(),
                    "same_request_history_and_mask_preserved": exact_target_history,
                    "all_non_target_requests_perturbed": complement_state,
                    "same_cell_noncurrent_requests_perturbed": bool(
                        np.all(perturbed_request_state[same_cell_noncurrent])
                    ),
                    "other_cell_requests_perturbed": bool(
                        np.all(perturbed_request_state[(~same_cell) & non_target])
                    ),
                    "same_cell_noncurrent_request_count": int(
                        np.count_nonzero(same_cell_noncurrent)
                    ),
                    "other_cell_request_count": int(
                        np.count_nonzero((~same_cell) & non_target)
                    ),
                    "non_target_request_count": int(np.count_nonzero(non_target)),
                    **check,
                }
            )
        except Exception as exc:  # preserve a complete audit trail if one request fails
            details.append(
                {
                    **request.as_dict(),
                    "same_request_history_and_mask_preserved": False,
                    "all_non_target_requests_perturbed": False,
                    "bitwise_request_local_invariance_pass": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        finally:
            perturbed_values[index] = np.nextafter(
                original_values[index], np.float32(np.inf)
            )
            perturbed_masks[index] = np.bitwise_xor(original_masks[index], np.uint8(1))
            perturbed_request_state[index] = True
    return details, perturbation


def _resolve_under_artifacts(text: str, *, root: Path, strict: bool) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=strict)
    allowed = (root / "artifacts").resolve(strict=False)
    if path != allowed and not path.is_relative_to(allowed):
        raise ValueError("Revision-8 audit paths must remain under artifacts/")
    return path


def source_dataset_fields() -> list[str]:
    text = inspect.getsource(runner.make_eval_tensors)
    return sorted(set(re.findall(r"dataset\.([a-z_]+)", text)))


def run_audit(args: argparse.Namespace) -> int:
    root = runner.project_root()
    source = _resolve_under_artifacts(args.source, root=root, strict=True)
    output = _resolve_under_artifacts(args.output, root=root, strict=False)
    revision8_root = (root / "artifacts/revision8").resolve(strict=False)
    if output != revision8_root and not output.is_relative_to(revision8_root):
        raise ValueError("Revision-8 audit output must remain under artifacts/revision8")
    checkpoint = source / "models" / f"{args.variant}_seed{args.model_seed}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing frozen audit checkpoint: {checkpoint}")
    source_manifest = source / "manifest.json"
    train_path = neural.resolve_train_path()
    train_before = neural.sha256_file(train_path)
    output.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.cpu_threads)
        runner.set_seed(args.model_seed)
        with tempfile.TemporaryDirectory(prefix="revision8-request-local-") as temporary:
            cache = Path(temporary)
            arrays, _ = neural.build_window_arrays(neural.read_training_series(train_path))
            neural.write_dataset_cache(cache, arrays)
            dataset = neural.load_dataset_cache(cache)
            holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
            population = build_audit_requests(dataset, holdout)
            selected, quotas = select_stratified_requests(
                population, args.sample_size, args.audit_seed
            )
            model, payload = runner.load_checkpoint(checkpoint, torch.device("cpu"))
            prior = np.asarray(payload["prior_log"], dtype=np.float32)
            details, perturbation = audit_selected_requests(dataset, selected, model, prior)
    finally:
        torch.set_num_threads(previous_threads)
    train_after = neural.sha256_file(train_path)
    if train_before != train_after:
        raise RuntimeError("registered training data changed during the Revision-8 audit")

    targets = target_list_payload(
        selected, quotas, population, audit_seed=args.audit_seed
    )
    dates = sorted({request.target_date for request in selected})
    bins = sorted({request.missingness_bin for request in selected})
    cells = sorted({request.cell for request in selected})
    violations = [
        item for item in details if not bool(item["bitwise_request_local_invariance_pass"])
    ]
    fields = source_dataset_fields()
    expected_fields = ["x_masks", "x_values"]
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "exploratory_redesign_on_existing_trace",
        "registered_train_path": str(neural.REGISTERED_TRAIN),
        "registered_train_sha256_before": train_before,
        "registered_train_sha256_after": train_after,
        "source": str(source.relative_to(root)),
        "source_manifest_sha256": (
            runner.sha256_file(source_manifest) if source_manifest.is_file() else None
        ),
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": runner.sha256_file(checkpoint),
        "variant": args.variant,
        "model_seed": args.model_seed,
        "device": "cpu",
        "cpu_threads": args.cpu_threads,
        "script_sha256": runner.sha256_file(Path(__file__).resolve()),
        "make_eval_tensors_source_sha256": runner.sha256_file(
            root / "experiments/train_wlcr_sea.py"
        ),
        "expert_builder_source_sha256": runner.sha256_file(root / "experiments/wlcr_sea_model.py"),
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
        "tested_cells": cells,
        "n_tested_cells": len(cells),
        "tested_dates": dates,
        "tested_missingness_bins": bins,
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
    runner.atomic_json(output / "targets.json", targets)
    runner.atomic_json(output / "details.json", {"requests": details})
    runner.atomic_json(output / "protocol.json", protocol)
    runner.atomic_json(output / "request_local_invariance.json", summary)
    runner.atomic_json(output / "manifest.json", runner.output_manifest(output))
    return 0 if not violations else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", default=str(DEFAULT_SOURCE))
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    value.add_argument("--variant", default=DEFAULT_CHECKPOINT_VARIANT)
    value.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    value.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    value.add_argument("--audit-seed", type=int, default=DEFAULT_AUDIT_SEED)
    value.add_argument("--cpu-threads", type=int, default=1)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    return run_audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
