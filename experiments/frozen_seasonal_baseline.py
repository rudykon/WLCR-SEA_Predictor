from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from Model.traffic_window_forecasting import (
    BaselineConfig,
    ContractError,
    baseline_candidates,
    load_baseline_config,
)


FROZEN_SEASONAL_CONFIG_PATH = "Model/seasonal_baseline_config.json"
FROZEN_SEASONAL_CONFIG_SHA256 = (
    "5c90e712bd2b5d3504c71948a1cc974c39d283a9221d7e812963adf29bc3834f"
)
SEASONAL_CANDIDATE_ORDER = (
    "lag1",
    "lag7",
    "lag14",
    "median7",
    "median14",
    "bounded_weekly_trend",
)
SEASONAL_CONFIG_ENTRYPOINTS = (
    (
        "experiments/train_lightgbm_baseline.py",
        "98c96dc394d15eea96e3f485b93ea33c0d5f2ba8a4e32e082e3768e893203730",
        "baseline_config",
    ),
    (
        "experiments/select_lightgbm_model.py",
        "9be41caed781fac886711f382192dcb7d62271fe6743f592a26b3ab5eb0b1494",
        "baseline",
    ),
)


@dataclass(frozen=True)
class FrozenSeasonalConfig:
    config: BaselineConfig
    path: str
    size_bytes: int
    sha256: str

    def report(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "expected_sha256": FROZEN_SEASONAL_CONFIG_SHA256,
            "name": self.config.name,
            "weights": list(self.config.weights),
            "scales": list(self.config.scales),
            "candidate_order": list(SEASONAL_CANDIDATE_ORDER),
        }


@dataclass(frozen=True)
class FrozenSeasonalVerification:
    frozen: FrozenSeasonalConfig
    entrypoints: tuple[dict[str, object], ...]

    def report(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": "verified",
            "frozen_seasonal_baseline_config": self.frozen.report(),
            "seasonal_config_entrypoints": list(self.entrypoints),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_file(root: Path, relative: str, description: str) -> Path:
    unresolved = root / relative
    cursor = unresolved
    while cursor != root:
        if cursor.is_symlink():
            raise ContractError(f"{description} contains a symbolic link: {relative}")
        cursor = cursor.parent
    try:
        path = unresolved.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ContractError(f"missing {description}: {relative}") from exc
    if not path.is_relative_to(root) or not path.is_file():
        raise ContractError(f"invalid {description} path: {relative}")
    return path


def load_frozen_seasonal_baseline_config(project_root: str | Path) -> FrozenSeasonalConfig:
    root = Path(project_root).resolve(strict=True)
    path = _resolve_project_file(
        root,
        FROZEN_SEASONAL_CONFIG_PATH,
        "frozen seasonal config",
    )

    actual_sha256 = _sha256_file(path)
    if actual_sha256 != FROZEN_SEASONAL_CONFIG_SHA256:
        raise ContractError(
            "frozen seasonal config SHA256 mismatch: "
            f"{actual_sha256} != {FROZEN_SEASONAL_CONFIG_SHA256}"
        )

    config = load_baseline_config(path)
    matching_candidates = [
        candidate
        for candidate in baseline_candidates()
        if candidate.name == config.name
    ]
    if matching_candidates != [config]:
        raise ContractError(
            "frozen seasonal config no longer matches its implemented candidate"
        )
    return FrozenSeasonalConfig(
        config=config,
        path=FROZEN_SEASONAL_CONFIG_PATH,
        size_bytes=path.stat().st_size,
        sha256=actual_sha256,
    )


def _literal_sequence(node: ast.AST) -> tuple[float, ...]:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left = ast.literal_eval(node.left)
            count = ast.literal_eval(node.right)
            if isinstance(left, (tuple, list)) and isinstance(count, int):
                value = left * count
            else:
                raise ContractError("unsupported seasonal sequence expression")
        else:
            raise ContractError("unsupported seasonal sequence expression")
    if not isinstance(value, (tuple, list)):
        raise ContractError("seasonal weights or scales are not a literal sequence")
    return tuple(float(item) for item in value)


def _read_declared_baseline_config(path: Path, variable: str) -> tuple[BaselineConfig, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches: list[tuple[BaselineConfig, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable
            for target in node.targets
        ):
            continue
        call = node.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "BaselineConfig"
            and len(call.args) == 3
        ):
            continue
        name = ast.literal_eval(call.args[0])
        if not isinstance(name, str):
            raise ContractError(f"{path} has a non-literal seasonal config name")
        matches.append(
            (
                BaselineConfig(
                    name,
                    _literal_sequence(call.args[1]),
                    _literal_sequence(call.args[2]),
                ),
                node.lineno,
            )
        )
    if len(matches) != 1:
        raise ContractError(
            f"expected exactly one {variable} BaselineConfig assignment in {path}"
        )
    return matches[0]


def verify_seasonal_config_consistency(
    project_root: str | Path,
) -> FrozenSeasonalVerification:
    root = Path(project_root).resolve(strict=True)
    frozen = load_frozen_seasonal_baseline_config(root)
    entrypoint_reports: list[dict[str, object]] = []
    for relative, expected_sha256, variable in SEASONAL_CONFIG_ENTRYPOINTS:
        path = _resolve_project_file(root, relative, "seasonal-config entrypoint")
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ContractError(
                f"seasonal-config entrypoint SHA256 mismatch for {relative}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        config, line = _read_declared_baseline_config(path, variable)
        if config != frozen.config:
            raise ContractError(
                f"declared seasonal constants disagree with frozen config in {relative}"
            )
        entrypoint_reports.append(
            {
                "path": relative,
                "sha256": actual_sha256,
                "expected_sha256": expected_sha256,
                "assignment": variable,
                "line": line,
                "name": config.name,
                "weights": list(config.weights),
                "scales": list(config.scales),
            }
        )
    return FrozenSeasonalVerification(frozen, tuple(entrypoint_reports))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the frozen seasonal config and seasonal configuration consistency"
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="repository root",
    )
    args = parser.parse_args(argv)
    try:
        report = verify_seasonal_config_consistency(args.root).report()
    except ContractError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
