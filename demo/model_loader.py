"""Download, verify, and load the public WLCR-SEA A6 ensemble once."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np


MODEL_REPO_ID = "config-h/WLCR-SEA-Predictor"
MODEL_REVISION = "eb4447f4ebab8f9caa003d92c838ed8e750963bd"
MODEL_VARIANT = "A6_mixed_aug"
ENSEMBLE_AGGREGATION = "arithmetic mean in linear traffic space"


@dataclass(frozen=True)
class CheckpointSpec:
    seed: int
    filename: str
    sha256: str


CHECKPOINT_SPECS = (
    CheckpointSpec(
        42,
        "A6_mixed_aug_seed42.pt",
        "9d9ab530566f74612a235dc6ed60c02caf09d4f0b862c68af52fdd61863b8956",
    ),
    CheckpointSpec(
        43,
        "A6_mixed_aug_seed43.pt",
        "84de30c603c0df9729a56a27490aef70eec3db795494676bf5a637cf04cccd26",
    ),
    CheckpointSpec(
        44,
        "A6_mixed_aug_seed44.pt",
        "93725821eb9dd226479e4fe4964aa53bc8e0fc71caa1eb9ec80d51f9f793489e",
    ),
    CheckpointSpec(
        45,
        "A6_mixed_aug_seed45.pt",
        "e05c14b8478aeed7f84e1e0733c22aa25d3c30fd60e685fe0cf75037e0b38a9a",
    ),
    CheckpointSpec(
        46,
        "A6_mixed_aug_seed46.pt",
        "f37247358733003d55651af35bd4587f5ba9284485749be40548ddbf9fdc5713",
    ),
)


@dataclass(frozen=True)
class EnsembleMember:
    seed: int
    filename: str
    sha256: str
    schema_version: int
    experiment_version: str
    selected_config: Mapping[str, object]
    selected_epoch: int
    prior_log: np.ndarray
    parameter_count: int
    model: object


@dataclass(frozen=True)
class A6Ensemble:
    repo_id: str
    revision: str
    variant: str
    aggregation: str
    device: str
    members: tuple[EnsembleMember, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_checkpoint(spec: CheckpointSpec) -> Path | None:
    configured = os.getenv("WLCR_SEA_CHECKPOINT_DIR")
    if not configured:
        return None
    root = Path(configured).expanduser()
    candidates = (root / spec.filename, root / "checkpoints" / spec.filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{spec.filename} was not found below WLCR_SEA_CHECKPOINT_DIR={root}"
    )


def _resolve_checkpoint(spec: CheckpointSpec) -> Path:
    local = _local_checkpoint(spec)
    if local is not None:
        path = local
    else:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=f"checkpoints/{spec.filename}",
                revision=MODEL_REVISION,
            )
        )
    actual = _sha256(path)
    if actual != spec.sha256:
        raise RuntimeError(
            f"Checksum mismatch for {spec.filename}: expected {spec.sha256}, got {actual}"
        )
    return path


def _load_member(spec: CheckpointSpec, path: Path, device) -> EnsembleMember:
    import torch

    from experiments import wlcr_sea_model as sea

    # These are pinned, checksum-verified project checkpoints. Explicitly
    # opting into the full payload keeps loading compatible with PyTorch
    # versions whose default is weights-only deserialization.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema_version",
        "experiment_version",
        "variant",
        "seed",
        "selected_config",
        "selected_epoch",
        "prior_log",
        "state_dict",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"{spec.filename} is missing checkpoint fields: {missing}")
    if int(payload["seed"]) != spec.seed:
        raise RuntimeError(f"{spec.filename} declares the wrong seed")

    variant = sea.VariantConfig(**payload["variant"])
    if variant.name != MODEL_VARIANT:
        raise RuntimeError(f"{spec.filename} is {variant.name}, expected {MODEL_VARIANT}")
    config = dict(payload["selected_config"])
    prior = np.asarray(payload["prior_log"], dtype=np.float32)
    if prior.shape != (sea.FORECAST_HOURS, sea.TARGET_COUNT) or not np.isfinite(prior).all():
        raise RuntimeError(f"{spec.filename} has an invalid frozen training prior")

    model = sea.WLCRSEA(
        variant,
        token_dim=int(config["token_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        residual_bound=float(config["residual_bound"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return EnsembleMember(
        seed=spec.seed,
        filename=spec.filename,
        sha256=spec.sha256,
        schema_version=int(payload["schema_version"]),
        experiment_version=str(payload["experiment_version"]),
        selected_config=config,
        selected_epoch=int(payload["selected_epoch"]),
        prior_log=prior,
        parameter_count=int(parameter_count),
        model=model,
    )


@lru_cache(maxsize=1)
def load_a6_ensemble() -> A6Ensemble:
    """Return the verified CPU ensemble; subsequent calls reuse the same models."""

    import torch

    thread_count = max(1, min(int(os.getenv("TORCH_NUM_THREADS", "2")), 4))
    torch.set_num_threads(thread_count)
    device = torch.device("cpu")
    members = tuple(
        _load_member(spec, _resolve_checkpoint(spec), device) for spec in CHECKPOINT_SPECS
    )
    if tuple(member.seed for member in members) != (42, 43, 44, 45, 46):
        raise RuntimeError("The A6 ensemble member registry is incomplete or out of order")
    return A6Ensemble(
        repo_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        variant=MODEL_VARIANT,
        aggregation=ENSEMBLE_AGGREGATION,
        device=str(device),
        members=members,
    )
