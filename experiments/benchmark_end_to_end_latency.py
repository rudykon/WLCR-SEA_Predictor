from __future__ import annotations

"""Revision-8 deployment-latency audit for the manuscript models.

This is deliberately separate from the Revision-7 evidence.  It measures one
336-hour request at a time on one CPU thread, after model files have been
loaded, and includes request preprocessing, model inference, and the final
inverse transform.  The script reads only registered training traffic and
already registered paper-model assets.  In particular, it never opens
``data/test_data.csv``.

The PyTorch models use the repository's neural runtime.  The historical
Revision-3 LightGBM model has a separate NumPy/LightGBM runtime, so this script
launches itself in ``--lgbm-worker`` mode with the local ``.runtime/lightgbm`` path.
Keeping the worker isolated prevents accidental mixing of the two incompatible
runtime stacks while still producing one audit report under ``artifacts/revision8``.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


# The latency run is evidence generation, not a package build.  Avoid adding
# incidental ``__pycache__`` files to the dirty user workspace.
sys.dont_write_bytecode = True


SCHEMA_VERSION = 1
EVIDENCE_STATUS = "exploratory_redesign_on_existing_trace"
TRAIN_SHA256 = "d274407a3db51ba4871851ab447bcc75202bb567337464d85ea280662f3bf1da"
SEA_SEEDS = (42, 43, 44, 45, 46)
NEURAL_MODELS = ("dlinear", "patchtst")
FORECAST_HOURS = 24
TARGET_COUNT = 4
OUTPUT_ROOT = Path("artifacts/reproduction/latency")
SEA_SOURCE = Path("artifacts/reproduction/wlcr_sea")
NEURAL_SOURCE = Path("artifacts/reproduction/neural_baselines/mixed")
ORIGINAL_WLCR_SOURCE = Path("artifacts/reproduction/lightgbm/traffic_only_73d")
PRIMARY_SEA_VARIANT = "A6_mixed_aug"
ORIGINAL_WLCR_ROUNDS = (341, 332, 742, 678)
ORIGINAL_WLCR_SELECTED_COLUMNS = (0, *range(16, 88))
METRIC_LABELS = ("ul_active_users", "dl_active_users", "dl_prb", "ul_prb")


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
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty latency summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def resolve_inside(root: Path, text: str | Path, *, strict: bool) -> Path:
    """Resolve an existing project input without allowing a path escape."""
    requested = Path(text)
    if not requested.is_absolute():
        requested = root / requested
    resolved = requested.resolve(strict=strict)
    if not resolved.is_relative_to(root):
        raise ValueError(f"path must remain inside the repository: {text}")
    return resolved


def resolve_revision8_output(root: Path, text: str | Path) -> Path:
    output = resolve_inside(root, text, strict=False)
    allowed = (root / OUTPUT_ROOT).resolve(strict=False)
    if output != allowed and not output.is_relative_to(allowed):
        raise ValueError(f"latency outputs must remain under {OUTPUT_ROOT}")
    return output


def file_provenance(root: Path, path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"invalid evidence asset: {path}")
    return {
        "path": str(resolved.relative_to(root)),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def array_sha256(values: object) -> str:
    """Return a stable byte hash for a small frozen numeric array."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def latency_distribution(values: Sequence[float]) -> dict[str, float]:
    import numpy as np

    if not values:
        raise ValueError("latency distribution requires at least one measurement")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("latency measurements must be finite and non-negative")
    return {
        "mean_ms": float(np.mean(array)),
        "sample_sd_ms": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "p50_ms": float(np.quantile(array, 0.50)),
        "p90_ms": float(np.quantile(array, 0.90)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
    }


def benchmark(
    call: Callable[[int], object],
    *,
    identities: int,
    warmups: int,
    measured: int,
) -> dict[str, float]:
    if identities <= 0 or warmups < 0 or measured <= 0:
        raise ValueError("identities/measured must be positive and warmups non-negative")
    for index in range(warmups):
        call(index % identities)
    timings: list[float] = []
    for index in range(measured):
        started = time.perf_counter_ns()
        result = call(index % identities)
        # The call implementations return a materialized numeric tensor/array.
        # Keeping a local reference makes accidental lazy-return refactors obvious.
        if result is None:
            raise RuntimeError("latency call returned None instead of a prediction")
        elapsed = time.perf_counter_ns() - started
        timings.append(elapsed / 1_000_000.0)
    return latency_distribution(timings)


def configure_single_thread_torch(torch: Any) -> dict[str, object]:
    """Pin the local process to a single CPU thread and record the result."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A fresh CLI process accepts the setting.  The guard keeps imported use
        # in a test runner from failing while still exposing the actual values.
        pass
    return {
        "cpu_threads_requested": 1,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    }


def validate_model_source(root: Path, source: Path, expected_relative: Path) -> Path:
    """Reject a convenient-looking but wrong frozen baseline source."""
    expected = (root / expected_relative).resolve(strict=True)
    if source != expected:
        raise ValueError(
            f"this audit is pinned to {expected_relative}; received {source.relative_to(root)}"
        )
    return expected


def validate_sea_source(root: Path, source: Path) -> Path:
    """Accept only the fresh multi-seed main-method checkpoints."""
    allowed = (root / SEA_SOURCE).resolve(strict=True)
    if source != allowed and not source.is_relative_to(allowed):
        raise ValueError(
            "SEA source must be the registered artifacts/reproduction checkpoint root"
        )
    return source


def checkpoint_prior_metadata(priors: Sequence[object]) -> dict[str, object]:
    import numpy as np

    arrays = [np.asarray(item, dtype=np.float32) for item in priors]
    if not arrays or any(array.shape != (FORECAST_HOURS, TARGET_COUNT) for array in arrays):
        raise ValueError("SEA checkpoints must contain 24x4 frozen priors")
    hashes = [array_sha256(array) for array in arrays]
    identical = all(np.array_equal(arrays[0], item) for item in arrays[1:])
    return {
        "frozen_scalar_assets_per_checkpoint": int(arrays[0].size),
        "frozen_scalar_assets_total": int(sum(array.size for array in arrays)),
        "frozen_scalar_assets_unique_if_deduplicated": int(arrays[0].size)
        if identical
        else int(sum(array.size for array in arrays)),
        "all_frozen_priors_byte_identical": bool(identical),
        "prior_sha256_per_checkpoint": hashes,
    }


def normalization_metadata(normalizations: Sequence[object]) -> dict[str, object]:
    import numpy as np

    vectors = []
    for normalization in normalizations:
        vectors.append(
            np.asarray(
                (
                    *normalization.input_mean,
                    *normalization.input_std,
                    *normalization.target_mean,
                    *normalization.target_std,
                ),
                dtype=np.float32,
            )
        )
    if not vectors or any(vector.shape != (16,) for vector in vectors):
        raise ValueError("neural checkpoints must contain four-channel normalizations")
    hashes = [array_sha256(vector) for vector in vectors]
    identical = all(np.array_equal(vectors[0], item) for item in vectors[1:])
    return {
        "frozen_scalar_assets_per_checkpoint": int(vectors[0].size),
        "frozen_scalar_assets_total": int(sum(vector.size for vector in vectors)),
        "frozen_scalar_assets_unique_if_deduplicated": int(vectors[0].size)
        if identical
        else int(sum(vector.size for vector in vectors)),
        "all_normalizations_byte_identical": bool(identical),
        "normalization_sha256_per_checkpoint": hashes,
    }


def selected_request_records(dataset: object, indices: object, neural: Any) -> list[dict[str, object]]:
    import numpy as np

    records: list[dict[str, object]] = []
    for index in np.asarray(indices, dtype=np.int64).tolist():
        records.append(
            {
                "dataset_index": int(index),
                "cell": str(dataset.cells[index]),
                "target_start": neural.timestamp_from_hour(
                    int(dataset.target_start_hours[index])
                ).isoformat(sep=" "),
            }
        )
    if len({(item["cell"], item["target_start"]) for item in records}) != len(records):
        raise ValueError("selected latency identities are not unique")
    return records


def _load_sea_models(root: Path, source: Path, *, seeds: Sequence[int]) -> tuple[list[object], list[object], list[dict[str, object]]]:
    import numpy as np
    import torch

    from experiments import train_wlcr_sea as runner
    from experiments import wlcr_sea_model as sea

    models: list[object] = []
    payloads: list[object] = []
    provenance: list[dict[str, object]] = []
    for seed in seeds:
        path = source / "models" / f"{PRIMARY_SEA_VARIANT}_seed{seed}.pt"
        asset = file_provenance(root, path)
        payload = torch.load(path, map_location="cpu")
        if int(payload.get("seed", -1)) != int(seed):
            raise ValueError(f"SEA checkpoint seed mismatch: {path}")
        variant = sea.VariantConfig(**payload["variant"])
        if variant.name != PRIMARY_SEA_VARIANT:
            raise ValueError(f"SEA checkpoint is not {PRIMARY_SEA_VARIANT}: {path}")
        if variant.augmentation != "mixed" or not variant.hard_mask:
            raise ValueError(f"SEA checkpoint has unexpected Revision-7 protocol: {path}")
        prior = np.asarray(payload["prior_log"], dtype=np.float32)
        if prior.shape != (FORECAST_HOURS, TARGET_COUNT) or not np.all(np.isfinite(prior)):
            raise ValueError(f"invalid SEA prior in {path}")
        model = runner.model_from_config(variant, payload["selected_config"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
        payloads.append(payload)
        provenance.append(
            {
                **asset,
                "seed": int(seed),
                "selected_config": dict(payload["selected_config"]),
                "selected_epoch": int(payload["selected_epoch"]),
                "trainable_parameters": int(runner.count_parameters(model)),
                "prior_sha256": array_sha256(prior),
            }
        )
    return models, payloads, provenance


def _load_neural_models(
    root: Path,
    source: Path,
    model_name: str,
    *,
    seeds: Sequence[int],
) -> tuple[list[object], list[object], list[dict[str, object]]]:
    import numpy as np
    import torch

    from experiments import train_neural_baselines as neural

    models: list[object] = []
    payloads: list[object] = []
    provenance: list[dict[str, object]] = []
    for seed in seeds:
        path = source / "models" / f"{model_name}_seed{seed}.pt"
        asset = file_provenance(root, path)
        payload = torch.load(path, map_location="cpu")
        if payload.get("model") != model_name or int(payload.get("seed", -1)) != int(seed):
            raise ValueError(f"neural checkpoint/model seed mismatch: {path}")
        if payload.get("augmentation") != "mixed" or not np.isclose(
            float(payload.get("augmentation_rate", -1.0)), 0.15
        ):
            raise ValueError(f"neural checkpoint is not the fair mixed-augmentation run: {path}")
        normalization = neural.Normalization(**payload["normalization"])
        vector = np.asarray(
            (
                *normalization.input_mean,
                *normalization.input_std,
                *normalization.target_mean,
                *normalization.target_std,
            ),
            dtype=np.float32,
        )
        if vector.shape != (16,) or not np.all(np.isfinite(vector)) or np.any(vector[4:8] <= 0.0) or np.any(vector[12:16] <= 0.0):
            raise ValueError(f"invalid neural normalization in {path}")
        model = neural.build_model(model_name, payload["config"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
        payloads.append(payload)
        provenance.append(
            {
                **asset,
                "seed": int(seed),
                "selected_config": dict(payload["config"]),
                "selected_epoch": int(payload["selected_epoch"]),
                "trainable_parameters": int(neural.count_parameters(model)),
                "normalization_sha256": array_sha256(vector),
            }
        )
    return models, payloads, provenance


def _sea_call_factory(
    *,
    dataset: object,
    selected: object,
    models: Sequence[object],
    payloads: Sequence[object],
) -> Callable[[int], object]:
    import numpy as np
    import torch

    from experiments import train_wlcr_sea as runner
    from experiments import wlcr_sea_model as sea

    selected_indices = np.asarray(selected, dtype=np.int64)
    priors = [np.asarray(payload["prior_log"], dtype=np.float32) for payload in payloads]

    def call(position: int) -> object:
        index = int(selected_indices[position])
        raw_predictions = []
        # Every member rebuilds its own request tensor using the frozen prior
        # saved in that checkpoint.  This is conservative but faithfully times
        # the deployed five-checkpoint ensemble without assuming shareability.
        for model, prior in zip(models, priors):
            batch = sea.build_expert_batch(
                np.asarray(dataset.x_values[index : index + 1], dtype=np.float32),
                np.asarray(dataset.x_masks[index : index + 1], dtype=np.uint8),
                prior,
            )
            values, availability, reliability, context = runner.batch_to_tensors(batch)
            with torch.inference_mode():
                prediction = sea.prediction_from_log(
                    model(values, availability.bool(), reliability, context)["prediction_log"]
                )
            raw_predictions.append(prediction)
        result = raw_predictions[0] if len(raw_predictions) == 1 else torch.stack(raw_predictions).mean(dim=0)
        if not bool(torch.isfinite(result).all()) or bool(torch.any(result <= 0.0)):
            raise FloatingPointError("SEA latency path produced an invalid raw prediction")
        return result

    return call


def _neural_call_factory(
    *,
    dataset: object,
    selected: object,
    models: Sequence[object],
    payloads: Sequence[object],
) -> Callable[[int], object]:
    import numpy as np
    import torch

    from experiments import train_neural_baselines as neural

    selected_indices = np.asarray(selected, dtype=np.int64)
    normalizations = [neural.Normalization(**payload["normalization"]) for payload in payloads]

    def call(position: int) -> object:
        index = int(selected_indices[position])
        raw_predictions = []
        for model, normalization in zip(models, normalizations):
            # ``prepared_inputs`` is the inference preprocessing used by the
            # Revision-7 neural models: fill/mask representation then channel
            # normalization.  Input corruption is absent in this clean serving
            # benchmark, as it is in their ordinary holdout inference path.
            inputs = neural.prepared_inputs(
                dataset,
                np.asarray((index,), dtype=np.int64),
                normalization,
            )
            with torch.inference_mode():
                normalized = model(inputs).detach().cpu().numpy()
            raw_predictions.append(neural.inverse_target(normalized, normalization))
        result = raw_predictions[0] if len(raw_predictions) == 1 else np.mean(
            np.stack(raw_predictions, axis=0), axis=0, dtype=np.float32
        )
        if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
            raise FloatingPointError("neural latency path produced an invalid raw prediction")
        return result

    return call


def deployment_record(
    *,
    label: str,
    family: str,
    deployment: str,
    seeds: Sequence[int],
    latency: Mapping[str, float],
    checkpoints: Sequence[Mapping[str, object]],
    frozen_assets: Mapping[str, object],
    model_representation: str = "PyTorch neural network",
) -> dict[str, object]:
    checkpoint_bytes = int(sum(int(item["size_bytes"]) for item in checkpoints))
    trainable = int(sum(int(item["trainable_parameters"]) for item in checkpoints))
    return {
        "label": label,
        "family": family,
        "deployment": deployment,
        "model_seeds": [int(seed) for seed in seeds],
        "ensemble_rule": "single raw prediction" if len(seeds) == 1 else "arithmetic mean of five member raw predictions",
        "preprocessing_recomputed_per_member": True,
        "latency": dict(latency),
        "model_representation": model_representation,
        "trainable_parameters": trainable,
        "trainable_parameters_per_member": [
            int(item["trainable_parameters"]) for item in checkpoints
        ],
        "checkpoint_total_bytes": checkpoint_bytes,
        "serialized_asset_total_bytes": checkpoint_bytes,
        "checkpoints": [dict(item) for item in checkpoints],
        "frozen_assets": dict(frozen_assets),
    }


def run_main(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from experiments import train_neural_baselines as neural

    root = project_root()
    output = resolve_revision8_output(root, args.output)
    sea_source = validate_sea_source(
        root,
        resolve_inside(root, args.sea_source, strict=True),
    )
    neural_source = validate_model_source(
        root,
        resolve_inside(root, args.neural_source, strict=True),
        NEURAL_SOURCE,
    )
    original_source = validate_model_source(
        root,
        resolve_inside(root, args.original_wlcr_source, strict=True),
        ORIGINAL_WLCR_SOURCE,
    )
    if args.identities <= 0 or args.warmups < 0 or args.measured <= 0:
        raise ValueError("identities/measured must be positive and warmups non-negative")
    threading = configure_single_thread_torch(torch)
    train_path = neural.resolve_train_path()
    input_before = sha256_file(train_path)
    if input_before != TRAIN_SHA256:
        raise ValueError("registered training input hash changed before latency run")

    # No finals traffic is involved: this materializes only legal supervised
    # windows from data/train_data.csv to create representative request identities.
    arrays, dataset_report = neural.build_window_arrays(neural.read_training_series(train_path))
    dataset = neural.CachedDataset(root=Path("<in-memory-latency>"), **arrays)
    holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
    if len(holdout) != 5_110:
        raise ValueError(f"unexpected registered holdout window count: {len(holdout)}")
    identity_count = min(int(args.identities), int(len(holdout)))
    selected = holdout[np.linspace(0, len(holdout) - 1, identity_count, dtype=np.int64)]
    records = selected_request_records(dataset, selected, neural)

    request_file = output.with_name("request_identities.json")
    atomic_json(
        request_file,
        {
            "schema_version": SCHEMA_VERSION,
            "selection": "evenly spaced deterministic registered-training holdout requests",
            "request_identities": records,
            "finals_test_opened": False,
        },
    )

    sea_models, sea_payloads, sea_provenance = _load_sea_models(
        root, sea_source, seeds=SEA_SEEDS
    )
    sea_priors = [payload["prior_log"] for payload in sea_payloads]
    sea_assets = checkpoint_prior_metadata(sea_priors)
    sea_seed42 = deployment_record(
        label="WLCR-SEA A6 mixed augmentation",
        family="WLCR-SEA",
        deployment="seed-42",
        seeds=(42,),
        latency=benchmark(
            _sea_call_factory(
                dataset=dataset,
                selected=selected,
                models=sea_models[:1],
                payloads=sea_payloads[:1],
            ),
            identities=identity_count,
            warmups=args.warmups,
            measured=args.measured,
        ),
        checkpoints=sea_provenance[:1],
        frozen_assets=checkpoint_prior_metadata(sea_priors[:1]),
    )
    sea_ensemble = deployment_record(
        label="WLCR-SEA A6 mixed augmentation",
        family="WLCR-SEA",
        deployment="five-seed raw-prediction ensemble",
        seeds=SEA_SEEDS,
        latency=benchmark(
            _sea_call_factory(
                dataset=dataset,
                selected=selected,
                models=sea_models,
                payloads=sea_payloads,
            ),
            identities=identity_count,
            warmups=args.warmups,
            measured=args.measured,
        ),
        checkpoints=sea_provenance,
        frozen_assets=sea_assets,
    )

    deployments: list[dict[str, object]] = [sea_seed42, sea_ensemble]
    neural_source_protocol = file_provenance(root, neural_source / "protocol.json")
    for model_name, label in (("dlinear", "DLinear-Aug"), ("patchtst", "PatchTST-Aug")):
        models, payloads, provenance = _load_neural_models(
            root, neural_source, model_name, seeds=SEA_SEEDS
        )
        normalizations = [neural.Normalization(**payload["normalization"]) for payload in payloads]
        deployments.append(
            deployment_record(
                label=label,
                family=label,
                deployment="seed-42",
                seeds=(42,),
                latency=benchmark(
                    _neural_call_factory(
                        dataset=dataset,
                        selected=selected,
                        models=models[:1],
                        payloads=payloads[:1],
                    ),
                    identities=identity_count,
                    warmups=args.warmups,
                    measured=args.measured,
                ),
                checkpoints=provenance[:1],
                frozen_assets=normalization_metadata(normalizations[:1]),
            )
        )
        deployments.append(
            deployment_record(
                label=label,
                family=label,
                deployment="five-seed raw-prediction ensemble",
                seeds=SEA_SEEDS,
                latency=benchmark(
                    _neural_call_factory(
                        dataset=dataset,
                        selected=selected,
                        models=models,
                        payloads=payloads,
                    ),
                    identities=identity_count,
                    warmups=args.warmups,
                    measured=args.measured,
                ),
                checkpoints=provenance,
                frozen_assets=normalization_metadata(normalizations),
            )
        )

    # The LightGBM artifact was generated in the repository's NumPy-2 runtime.
    # Invoke a self-worker in that same local environment rather than importing
    # LightGBM into the PyTorch runtime process.
    original_output = output.with_name("original_wlcr_latency.json")
    original_log = output.with_name("original_wlcr_latency_worker.log")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--lgbm-worker",
        "--output",
        str(original_output),
        "--request-list",
        str(request_file),
        "--original-wlcr-source",
        str(original_source),
        "--warmups",
        str(args.warmups),
        "--measured",
        str(args.measured),
        "--verify-samples",
        str(args.verify_samples),
        "--verification-rtol",
        str(args.verification_rtol),
        "--verification-atol",
        str(args.verification_atol),
    ]
    if args.skip_original_verification:
        command.append("--skip-original-verification")
    worker_environment = os.environ.copy()
    worker_environment["PYTHONPATH"] = os.pathsep.join(
        (str(root / ".runtime/lightgbm"), str(root))
    )
    worker_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        worker_environment[key] = "1"
    worker_environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        command,
        cwd=root,
        env=worker_environment,
        capture_output=True,
        text=True,
    )
    original_log.parent.mkdir(parents=True, exist_ok=True)
    original_log.write_text(
        "COMMAND\n"
        + " ".join(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Original WLCR LightGBM latency worker failed; see "
            f"{original_log.relative_to(root)}"
        )
    original_payload = json.loads(original_output.read_text(encoding="utf-8"))
    if not original_payload.get("verification", {}).get("passed", False) and not args.skip_original_verification:
        raise RuntimeError("Original WLCR reconstruction verification did not pass")
    deployments.append(dict(original_payload["deployment"]))

    input_after = sha256_file(train_path)
    if input_after != input_before or input_after != TRAIN_SHA256:
        raise RuntimeError("registered training input changed during latency run")

    summary_rows: list[dict[str, object]] = []
    for item in deployments:
        latency = item["latency"]
        summary_rows.append(
            {
                "family": item["family"],
                "deployment": item["deployment"],
                "model_seeds": ",".join(str(seed) for seed in item["model_seeds"]),
                "p50_ms": latency["p50_ms"],
                "p95_ms": latency["p95_ms"],
                "p99_ms": latency["p99_ms"],
                "mean_ms": latency["mean_ms"],
                "serialized_asset_total_bytes": item["serialized_asset_total_bytes"],
                "trainable_parameters": item["trainable_parameters"],
                "model_representation": item["model_representation"],
            }
        )
    summary_path = output.with_name("latency_summary.csv")
    atomic_csv(summary_path, summary_rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": EVIDENCE_STATUS,
        "protocol": {
            "measurement_unit": "one 336-hour request, producing 24x4 raw forecasts",
            "batch_size": 1,
            "cpu_threads": 1,
            "measured_path": "preprocessing + inference + inverse transform / output floor",
            "startup_excluded": "checkpoint loading and one-time dataset parsing",
            "ensemble_rule": "all five member raw predictions are arithmetic-mean averaged after each member inverse transform",
            "ensemble_preprocessing": "recomputed separately for each member; conservative with respect to possible shared frozen transforms",
            "request_source": "deterministic evenly spaced holdout requests constructed only from registered train_data.csv",
            "finals_test_opened": False,
        },
        "threading": threading,
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "os_cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "request_identities": identity_count,
        "warmups": int(args.warmups),
        "measured_requests": int(args.measured),
        "dataset_construction": dataset_report,
        "source_provenance": {
            "script": file_provenance(root, Path(__file__).resolve()),
            "sea_protocol": file_provenance(root, sea_source / "protocol.json"),
            "neural_protocol": neural_source_protocol,
            "original_wlcr_worker_report": file_provenance(root, original_output),
            "original_wlcr_worker_log": file_provenance(root, original_log),
            "request_identities": file_provenance(root, request_file),
        },
        "deployments": deployments,
        "original_wlcr_verification": original_payload["verification"],
        "summary_csv": str(summary_path.relative_to(root)),
        "registered_train_sha256_before": input_before,
        "registered_train_sha256_after": input_after,
        "finals_test_opened": False,
    }
    atomic_json(output, payload)
    return 0


def _original_feature_matrix(
    example: object,
    baseline: object,
    columns: object,
) -> object:
    """Rebuild exactly the original 73 traffic-only WLCR input rows."""
    import numpy as np

    from Model.lightgbm_feature_baseline import build_test_matrix

    # Empty parameter/weather maps are intentional.  The selected Revision-3
    # columns are ``horizon`` plus the four traffic blocks [16:88], and this
    # call never opens either auxiliary CSV.  It preserves the original feature
    # construction code while making the traffic-only serving contract explicit.
    features, _ = build_test_matrix((example.window,), baseline, {}, {})
    selected = np.asarray(features, dtype=np.float32)[:, np.asarray(columns, dtype=np.int64)]
    if selected.shape != (FORECAST_HOURS, len(ORIGINAL_WLCR_SELECTED_COLUMNS)):
        raise ValueError(f"unexpected Original WLCR serving matrix shape: {selected.shape}")
    return selected


def _count_leaf_values(tree: Mapping[str, object]) -> int:
    if "leaf_value" in tree:
        return 1
    return _count_leaf_values(tree["left_child"]) + _count_leaf_values(tree["right_child"])


def _original_model_assets(root: Path, source: Path) -> tuple[list[object], dict[str, object], list[dict[str, object]]]:
    import lightgbm as lgb

    manifest_path = source / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("cache_config")
    if not isinstance(config, dict):
        raise ValueError("Original WLCR cache manifest lacks cache_config")
    if config.get("experiment_version") != "traffic_only_73d_reproduction_v1":
        raise ValueError("traffic-only source is not the registered 73D reproduction cache")
    rounds = tuple(int(value) for value in config.get("rounds", ()))
    if rounds != ORIGINAL_WLCR_ROUNDS:
        raise ValueError(f"Original WLCR rounds differ from frozen Revision-3 rounds: {rounds}")
    selected = config.get("selected_feature_names")
    if not isinstance(selected, list) or len(selected) != TARGET_COUNT:
        raise ValueError("Original WLCR feature manifest has no per-target schema")
    if any(not isinstance(item, list) or len(item) != 73 for item in selected):
        raise ValueError("Original WLCR must be the 73-dimensional traffic-only model")
    if any(item != selected[0] for item in selected[1:]):
        raise ValueError("Original WLCR target feature schemas differ unexpectedly")
    model_records = manifest.get("models")
    if not isinstance(model_records, list) or len(model_records) != TARGET_COUNT:
        raise ValueError("Original WLCR manifest must list four boosters")
    boosters: list[object] = []
    provenance = [file_provenance(root, manifest_path)]
    tree_counts: list[int] = []
    leaf_counts: list[int] = []
    for metric, record in enumerate(model_records):
        expected_name = f"metric_{metric}.txt"
        if record.get("file") != expected_name:
            raise ValueError(f"Original WLCR booster order mismatch at metric {metric}")
        path = source / expected_name
        asset = file_provenance(root, path)
        if asset["size_bytes"] != int(record.get("size_bytes", -1)) or asset["sha256"] != record.get("sha256"):
            raise ValueError(f"Original WLCR booster hash mismatch: {path}")
        # LightGBM 4.6.0 segfaults when reset_parameter() changes a model that
        # was trained with device_type=gpu. Constructing an in-memory CPU copy
        # changes only that parameter line; all serialized tree sections remain
        # byte-identical to the hash-verified Revision-3 source asset.
        model_text = path.read_text(encoding="utf-8")
        if model_text.count("[device_type: gpu]") != 1:
            raise ValueError("Original WLCR model lacks exactly one GPU device marker")
        cpu_model_text = model_text.replace("[device_type: gpu]", "[device_type: cpu]")
        booster = lgb.Booster(model_str=cpu_model_text)
        dumped = booster.dump_model()
        trees = dumped.get("tree_info", [])
        tree_counts.append(len(trees))
        leaf_counts.append(sum(_count_leaf_values(item["tree_structure"]) for item in trees))
        boosters.append(booster)
        provenance.append({**asset, "metric": metric, "trees": tree_counts[-1], "learned_leaf_values": leaf_counts[-1]})
    metadata = {
        "feature_count": 73,
        "selected_feature_names": selected[0],
        "rounds": list(rounds),
        "cpu_inference_model_loading": (
            "in-memory LightGBM model string with only [device_type: gpu] "
            "rewritten to [device_type: cpu]; hash-verified source tree text is unchanged"
        ),
        "tree_count_per_target": tree_counts,
        "tree_count_total": int(sum(tree_counts)),
        "learned_leaf_values_per_target": leaf_counts,
        "learned_leaf_values_total": int(sum(leaf_counts)),
        "frozen_metadata_scalars": {
            "seasonal_baseline_weights": 6,
            "seasonal_baseline_scales": 4,
            "selected_feature_indices": 73,
            "boosting_rounds": 4,
        },
    }
    return boosters, metadata, provenance


def _load_original_baseline(source: Path) -> object:
    from Model.traffic_window_forecasting import BaselineConfig

    report_path = source / "summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report.get("seasonal_baseline", {})
    if selected.get("name") != "weekly_median_s097":
        raise ValueError("traffic-only baseline selection changed")
    weights = tuple(float(value) for value in selected.get("weights", ()))
    scales = tuple(float(value) for value in selected.get("scales", ()))
    if len(weights) != 6 or len(scales) != TARGET_COUNT:
        raise ValueError("traffic-only baseline configuration is malformed")
    return BaselineConfig(str(selected["name"]), weights, scales)


def _load_worker_requests(path: Path) -> list[tuple[str, datetime]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("request_identities")
    if not isinstance(records, list) or not records:
        raise ValueError("request identity record is empty")
    result = []
    for record in records:
        cell = str(record["cell"])
        target_start = datetime.fromisoformat(str(record["target_start"]))
        result.append((cell, target_start))
    if len(set(result)) != len(result):
        raise ValueError("request identity record contains duplicate requests")
    return result


def _verify_original_predictions(
    *,
    root: Path,
    source: Path,
    examples: Sequence[object],
    predict: Callable[[int], object],
    samples: int,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    import numpy as np

    if samples <= 0:
        return {
            "performed": False,
            "passed": True,
            "reason": "verification disabled by --verify-samples 0",
        }
    chosen = list(range(min(samples, len(examples))))
    prediction_path = source / "holdout_predictions.npy"
    order_path = source / "holdout_order.csv"
    stored = np.load(prediction_path, allow_pickle=False)
    if stored.ndim != 3 or stored.shape[1:] != (FORECAST_HOURS, TARGET_COUNT):
        raise ValueError("traffic-only reference predictions have an invalid shape")
    order: dict[tuple[str, str], int] = {}
    with order_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"window_index", "cell", "target_start"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("traffic-only prediction order has an invalid schema")
        for row in reader:
            key = (str(row["cell"]), str(row["target_start"]))
            if key in order:
                raise ValueError(f"duplicate traffic-only prediction order key: {key}")
            order[key] = int(row["window_index"])
    if len(order) != len(stored):
        raise ValueError("traffic-only prediction order and array length differ")
    differences: list[np.ndarray] = []
    expected_values: list[np.ndarray] = []
    computed_values: list[np.ndarray] = []
    for position in chosen:
        example = examples[position]
        raw = np.asarray(predict(position), dtype=np.float64)
        if raw.shape != (FORECAST_HOURS, TARGET_COUNT):
            raise ValueError("traffic-only verification prediction has wrong shape")
        key = (str(example.window.cell), example.window.target_start.isoformat(sep=" "))
        if key not in order:
            raise ValueError(f"traffic-only reference lacks request: {key}")
        expected = np.asarray(stored[order[key]], dtype=np.float64)
        differences.append(raw - expected)
        expected_values.append(expected)
        computed_values.append(raw)
    differences_array = np.stack(differences, axis=0)
    expected_array = np.stack(expected_values, axis=0)
    computed_array = np.stack(computed_values, axis=0)
    maximum_absolute = float(np.max(np.abs(differences_array)))
    maximum_relative = float(
        np.max(np.abs(differences_array) / np.maximum(np.abs(expected_array), 1e-12))
    )
    passed = bool(np.allclose(computed_array, expected_array, rtol=rtol, atol=atol))
    return {
        "performed": True,
        "passed": passed,
        "sampled_requests": int(len(chosen)),
        "sampled_forecast_rows": int(len(chosen) * FORECAST_HOURS),
        "rtol": float(rtol),
        "atol": float(atol),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "prediction_source": {
            "predictions": file_provenance(root, prediction_path),
            "order": file_provenance(root, order_path),
        },
        "comparison": "reconstructed traffic-only prediction versus the fresh aligned holdout prediction",
    }


def run_lgbm_worker(args: argparse.Namespace) -> int:
    import numpy as np

    from Model.traffic_window_forecasting import build_training_backtests, read_traffic
    from experiments.train_lightgbm_baseline import feature_names

    root = project_root()
    output = resolve_revision8_output(root, args.output)
    source = validate_model_source(
        root,
        resolve_inside(root, args.original_wlcr_source, strict=True),
        ORIGINAL_WLCR_SOURCE,
    )
    request_path = resolve_inside(root, args.request_list, strict=True)
    allowed = (root / OUTPUT_ROOT).resolve(strict=False)
    if not request_path.is_relative_to(allowed):
        raise ValueError("LightGBM request-list must be a Revision-8 artifact")
    if args.warmups < 0 or args.measured <= 0:
        raise ValueError("measured must be positive and warmups non-negative")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"

    train_path = root / "data/train_data.csv"
    input_before = sha256_file(train_path)
    if input_before != TRAIN_SHA256:
        raise ValueError("registered training input hash changed before LightGBM latency run")
    requested = _load_worker_requests(request_path)
    all_examples = build_training_backtests(read_traffic(train_path))
    examples_by_key = {
        (str(example.window.cell), example.window.target_start): example
        for example in all_examples
        if example.window.target_start.date().isoformat() >= "2024-08-12"
        and example.window.target_start.date().isoformat() <= "2024-08-18"
    }
    examples: list[object] = []
    for key in requested:
        if key not in examples_by_key:
            raise ValueError(f"Original WLCR worker cannot locate request identity: {key}")
        examples.append(examples_by_key[key])
    if len(examples) != len(requested):
        raise ValueError("Original WLCR request alignment failed")
    baseline = _load_original_baseline(source)
    names = feature_names(examples[0], baseline, {}, {})
    if len(names) != 88:
        raise ValueError(f"Original WLCR feature schema has {len(names)} columns, expected 88")
    columns = np.asarray(ORIGINAL_WLCR_SELECTED_COLUMNS, dtype=np.int64)
    if [names[int(index)] for index in columns] != json.loads(
        (source / "cache_manifest.json").read_text(encoding="utf-8")
    )["cache_config"]["selected_feature_names"][0]:
        raise ValueError("Original WLCR traffic-only serving columns no longer match the model manifest")
    boosters, model_metadata, provenance = _original_model_assets(root, source)

    def call(position: int) -> object:
        matrix = _original_feature_matrix(examples[position], baseline, columns)
        prediction = np.empty((FORECAST_HOURS, TARGET_COUNT), dtype=np.float64)
        for metric, booster in enumerate(boosters):
            prediction[:, metric] = np.maximum(
                np.expm1(booster.predict(matrix, num_threads=1)), 1e-4
            )
        if not np.all(np.isfinite(prediction)) or np.any(prediction <= 0.0):
            raise FloatingPointError("Original WLCR latency path produced an invalid prediction")
        return prediction

    verification = (
        {
            "performed": False,
            "passed": True,
            "reason": "disabled by --skip-original-verification",
        }
        if args.skip_original_verification
        else _verify_original_predictions(
            root=root,
            source=source,
            examples=examples,
            predict=call,
            samples=args.verify_samples,
            rtol=args.verification_rtol,
            atol=args.verification_atol,
        )
    )
    if not verification["passed"]:
        raise RuntimeError(
            "Original WLCR reconstructed predictions differ from Revision-3 holdout evidence"
        )
    latency = benchmark(
        call,
        identities=len(examples),
        warmups=args.warmups,
        measured=args.measured,
    )
    input_after = sha256_file(train_path)
    if input_after != input_before or input_after != TRAIN_SHA256:
        raise RuntimeError("registered training input changed during LightGBM latency run")
    model_files = provenance[1:]
    model_bytes = int(sum(int(item["size_bytes"]) for item in model_files))
    manifest_bytes = int(provenance[0]["size_bytes"])
    deployment = {
        "label": "Original WLCR-LightGBM traffic-only",
        "family": "Original WLCR-LightGBM",
        "deployment": "traffic-only 73D seed-42",
        "model_seeds": [42],
        "ensemble_rule": "single Revision-3 traffic-only model",
        "preprocessing_recomputed_per_member": True,
        "latency": latency,
        "model_representation": "four target-specific LightGBM tree ensembles",
        "trainable_parameters": int(model_metadata["learned_leaf_values_total"]),
        "trainable_parameters_definition": "learned leaf values; neural parameter totals are not directly comparable",
        "checkpoint_total_bytes": model_bytes,
        "serialized_asset_total_bytes": model_bytes + manifest_bytes,
        "serialized_asset_breakdown": {
            "four_booster_files_bytes": model_bytes,
            "cache_manifest_bytes": manifest_bytes,
        },
        "checkpoints": model_files,
        "cache_manifest": provenance[0],
        "frozen_assets": model_metadata,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": EVIDENCE_STATUS,
        "protocol": {
            "measurement_unit": "one 336-hour request, producing 24x4 raw forecasts",
            "batch_size": 1,
            "cpu_threads": 1,
            "measured_path": "traffic-only seasonal/feature preprocessing + four booster predictions + expm1/output floor",
            "auxiliary_files_opened": [],
            "finals_test_opened": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
        "request_identities": len(examples),
        "warmups": int(args.warmups),
        "measured_requests": int(args.measured),
        "deployment": deployment,
        "verification": verification,
        "registered_train_sha256_before": input_before,
        "registered_train_sha256_after": input_after,
        "request_list": file_provenance(root, request_path),
        "traffic_only_summary": file_provenance(root, source / "summary.json"),
        "finals_test_opened": False,
    }
    atomic_json(output, payload)
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--output",
        default="artifacts/reproduction/latency/latency.json",
        help="JSON output under artifacts/reproduction/latency/",
    )
    value.add_argument("--sea-source", default=str(SEA_SOURCE))
    value.add_argument("--neural-source", default=str(NEURAL_SOURCE))
    value.add_argument("--original-wlcr-source", default=str(ORIGINAL_WLCR_SOURCE))
    value.add_argument("--identities", type=int, default=256)
    value.add_argument("--warmups", type=int, default=128)
    value.add_argument("--measured", type=int, default=2048)
    value.add_argument("--verify-samples", type=int, default=0)
    value.add_argument("--verification-rtol", type=float, default=2e-5)
    value.add_argument("--verification-atol", type=float, default=2e-5)
    value.add_argument("--skip-original-verification", action="store_true")
    value.add_argument("--lgbm-worker", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--request-list", default="", help=argparse.SUPPRESS)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.lgbm_worker:
        if not args.request_list:
            raise ValueError("--lgbm-worker requires --request-list")
        return run_lgbm_worker(args)
    return run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
