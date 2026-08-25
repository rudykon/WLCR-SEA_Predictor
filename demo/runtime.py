"""Inference and audit adapter for the public WLCR-SEA A6 Demo."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wlcr-sea-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from demo.model_loader import A6Ensemble, load_a6_ensemble


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEMO_SEED = 2026
SCENARIOS = {
    "none": "none",
    "mcar": "mcar",
    "block": "block",
    "recent_tail": "recent_tail",
    "asynchronous": "asynchronous",
}
SCENARIO_LABELS = {
    "en": {
        "none": "Complete history",
        "mcar": "Random whole-hour loss",
        "block": "Contiguous block loss",
        "recent_tail": "Recent-tail outage",
        "asynchronous": "Asynchronous indicator loss",
    },
    "zh": {
        "none": "完整历史",
        "mcar": "随机整小时缺失",
        "block": "连续区块缺失",
        "recent_tail": "最近时段中断",
        "asynchronous": "指标异步缺失",
    },
}
METRIC_KEYS = ("ul_users", "dl_users", "dl_prb", "ul_prb")
METRIC_INDEX = {key: index for index, key in enumerate(METRIC_KEYS)}
METRIC_LABELS = {
    "en": ("UL active users", "DL active users", "DL PRB", "UL PRB"),
    "zh": ("上行激活用户", "下行激活用户", "下行 PRB", "上行 PRB"),
}
EXPERT_LABELS = {
    "en": (
        "Previous day",
        "Previous week",
        "Two weeks earlier",
        "7-day same-hour median",
        "14-day same-hour median",
        "Limited weekly trend",
        "Request median",
        "Frozen training prior",
    ),
    "zh": (
        "前一天",
        "前一周",
        "前两周",
        "7 日同小时中位数",
        "14 日同小时中位数",
        "限幅周趋势",
        "请求中位数",
        "冻结训练先验",
    ),
}
SHORT_EXPERT_LABELS = (
    "1 d",
    "7 d",
    "14 d",
    "Med 7",
    "Med 14",
    "Trend",
    "Local",
    "Prior",
)


class DemoInputError(ValueError):
    """Raised when a public Demo request violates the CSV contract."""


@dataclass(frozen=True)
class MemberAudit:
    seed: int
    filename: str
    sha256: str
    selected_config: dict[str, object]
    selected_epoch: int
    parameter_count: int
    prediction: np.ndarray
    expert_values: np.ndarray
    attention: np.ndarray
    baseline_log: np.ndarray
    residual_log: np.ndarray
    lower_envelope: np.ndarray
    upper_envelope: np.ndarray


@dataclass(frozen=True)
class AuditResult:
    """Inspectable outputs from one five-member A6 request."""

    source_path: Path
    input_sha256: str
    cell: str
    history_times: tuple[datetime, ...]
    history_values: np.ndarray
    original_mask: np.ndarray
    effective_mask: np.ndarray
    forecast_times: tuple[datetime, ...]
    prediction: np.ndarray
    lower_envelope: np.ndarray
    upper_envelope: np.ndarray
    expert_values: np.ndarray
    availability: np.ndarray
    reliability: np.ndarray
    attention: np.ndarray
    scenario: str
    requested_rate: float
    model_repo_id: str
    model_revision: str
    variant: str
    aggregation: str
    device: str
    members: tuple[MemberAudit, ...]


def _upload_path(upload: str | Path | Any) -> Path:
    if upload is None:
        raise DemoInputError("Choose the built-in sample or upload a CSV request.")
    candidate = (
        upload
        if isinstance(upload, (str, os.PathLike))
        else getattr(upload, "name", upload)
    )
    path = Path(str(candidate)).expanduser()
    if not path.is_file():
        raise DemoInputError("The selected CSV is no longer available. Choose it again.")
    if path.suffix.lower() != ".csv":
        raise DemoInputError("Use a UTF-8 CSV file with the documented six-column header.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise DemoInputError("The upload exceeds the 5 MB public-Demo limit.")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_request(upload: str | Path | Any):
    """Validate and load exactly one physical 336-hour request window."""

    from Model.traffic_window_forecasting import (
        ContractError,
        read_traffic,
        split_physical_windows,
    )

    path = _upload_path(upload)
    try:
        rows = read_traffic(path)
        windows = split_physical_windows(rows)
    except (ContractError, UnicodeDecodeError) as exc:
        raise DemoInputError(str(exc)) from exc
    if len(windows) != 1:
        raise DemoInputError("The public Demo accepts exactly one 336-row request window.")
    window = windows[0]
    if window.gaps:
        raise DemoInputError("Timestamps must be strictly hourly and contiguous for all 336 rows.")
    return path, window


def _request_arrays(window) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((1, 336, 4), dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    for hour, row in enumerate(window.rows):
        for metric, item in enumerate(row.metrics):
            if item is not None:
                values[0, hour, metric] = float(item)
                mask[0, hour, metric] = True
    if np.any(np.sum(mask, axis=1) == 0):
        raise DemoInputError("Every indicator needs at least one observed value in the request.")
    return values, mask


def _absolute_hour(value: datetime) -> int:
    return int((value - datetime(1970, 1, 1)).total_seconds() // 3600)


def run_a6_forecast(
    upload: str | Path | Any,
    *,
    scenario: str = "none",
    missing_rate: float = 0.2,
    ensemble: A6Ensemble | None = None,
) -> AuditResult:
    """Run the public five-member A6 ensemble with its frozen per-seed priors."""

    if scenario not in SCENARIOS:
        raise DemoInputError(f"Unknown missingness scenario: {scenario}")
    if not 0.0 <= float(missing_rate) <= 0.8:
        raise DemoInputError("Missingness rate must be between 0 and 80%.")

    path, window = load_request(upload)
    raw_values, original_mask = _request_arrays(window)
    mechanism = SCENARIOS[scenario]
    effective_rate = 0.0 if mechanism == "none" else float(missing_rate)

    from experiments.missingness_protocol import global_corruption_mask

    additional_missing = global_corruption_mask(
        np.asarray([window.cell]),
        np.asarray([_absolute_hour(window.rows[-1].timestamp)], dtype=np.int64),
        mechanism=mechanism,
        requested_rate=effective_rate,
        seed=DEMO_SEED,
    )
    effective_mask = original_mask & ~additional_missing
    if np.any(np.sum(effective_mask, axis=1) == 0):
        raise DemoInputError(
            "This corruption removes every observed value for an indicator; choose a lower rate."
        )

    from experiments import wlcr_sea_model as sea
    import torch

    bundle = ensemble or load_a6_ensemble()
    history_log = np.where(original_mask, np.log1p(raw_values), 0.0).astype(np.float32)
    member_audits: list[MemberAudit] = []
    common_availability: np.ndarray | None = None
    common_reliability: np.ndarray | None = None

    with torch.inference_mode():
        for member in bundle.members:
            experts = sea.build_expert_batch(
                history_log,
                original_mask,
                member.prior_log,
                additional_missing=additional_missing,
            )
            output = member.model(
                torch.from_numpy(experts.values),
                torch.from_numpy(experts.availability),
                torch.from_numpy(experts.reliability),
                torch.from_numpy(experts.context),
            )
            prediction_log = output["prediction_log"].cpu().numpy()[0]
            prediction = np.asarray(
                sea.prediction_from_log(prediction_log), dtype=np.float32
            )
            attention = output["attention"].cpu().numpy()[0]
            expert_values = np.asarray(
                sea.prediction_from_log(experts.values[0]), dtype=np.float32
            )
            residual_bound = float(member.selected_config["residual_bound"])
            lower_log, upper_log = sea.bounded_audit_envelope(
                experts.values[0], experts.availability[0], residual_bound
            )
            lower = np.asarray(sea.prediction_from_log(lower_log), dtype=np.float32)
            upper = np.asarray(sea.prediction_from_log(upper_log), dtype=np.float32)

            if np.any(attention[~experts.availability[0]] != 0.0):
                raise RuntimeError(
                    f"Seed {member.seed} assigned weight to an unavailable expert"
                )
            if not np.allclose(attention.sum(axis=-1), 1.0, atol=1e-7):
                raise RuntimeError(
                    f"Seed {member.seed} produced unnormalized routing weights"
                )
            if np.any(prediction < lower - 1e-5) or np.any(prediction > upper + 1e-5):
                raise RuntimeError(
                    f"Seed {member.seed} violated its bounded residual envelope"
                )

            if common_availability is None:
                common_availability = experts.availability[0].copy()
                common_reliability = experts.reliability[0].copy()
            elif not np.array_equal(common_availability, experts.availability[0]):
                raise RuntimeError("A6 members disagree on expert availability")

            member_audits.append(
                MemberAudit(
                    seed=member.seed,
                    filename=member.filename,
                    sha256=member.sha256,
                    selected_config=dict(member.selected_config),
                    selected_epoch=member.selected_epoch,
                    parameter_count=member.parameter_count,
                    prediction=prediction,
                    expert_values=expert_values,
                    attention=attention,
                    baseline_log=output["baseline_log"].cpu().numpy()[0],
                    residual_log=output["residual"].cpu().numpy()[0],
                    lower_envelope=lower,
                    upper_envelope=upper,
                )
            )

    if common_availability is None or common_reliability is None:
        raise RuntimeError("The A6 ensemble contains no members")
    prediction = np.mean([member.prediction for member in member_audits], axis=0)
    lower = np.mean([member.lower_envelope for member in member_audits], axis=0)
    upper = np.mean([member.upper_envelope for member in member_audits], axis=0)
    if np.any(prediction < lower - 1e-5) or np.any(prediction > upper + 1e-5):
        raise RuntimeError("The five-member ensemble violated its averaged envelope")

    start = window.target_start
    return AuditResult(
        source_path=path,
        input_sha256=_sha256(path),
        cell=window.cell,
        history_times=tuple(row.timestamp for row in window.rows),
        history_values=raw_values[0],
        original_mask=original_mask[0],
        effective_mask=effective_mask[0],
        forecast_times=tuple(start + timedelta(hours=index) for index in range(24)),
        prediction=np.asarray(prediction, dtype=np.float32),
        lower_envelope=np.asarray(lower, dtype=np.float32),
        upper_envelope=np.asarray(upper, dtype=np.float32),
        expert_values=np.mean(
            [member.expert_values for member in member_audits], axis=0
        ),
        availability=common_availability,
        reliability=common_reliability,
        attention=np.mean([member.attention for member in member_audits], axis=0),
        scenario=scenario,
        requested_rate=effective_rate,
        model_repo_id=bundle.repo_id,
        model_revision=bundle.revision,
        variant=bundle.variant,
        aggregation=bundle.aggregation,
        device=bundle.device,
        members=tuple(member_audits),
    )


def forecast_dataframe(result: AuditResult, lang: str = "en") -> pd.DataFrame:
    labels = METRIC_LABELS[lang]
    timestamp_key = "Timestamp" if lang == "en" else "时间"
    cell_key = "Cell" if lang == "en" else "小区"
    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(result.forecast_times):
        rows.append(
            {
                timestamp_key: timestamp.strftime("%Y/%m/%d %H:%M"),
                cell_key: result.cell,
                **{
                    label: round(float(result.prediction[index, metric]), 4)
                    for metric, label in enumerate(labels)
                },
            }
        )
    return pd.DataFrame(rows)


def expert_dataframe(
    result: AuditResult,
    metric_key: str = "dl_prb",
    horizon: int = 1,
    lang: str = "en",
) -> pd.DataFrame:
    q = METRIC_INDEX[metric_key]
    h = int(horizon) - 1
    rows: list[dict[str, object]] = []
    for expert, label in enumerate(EXPERT_LABELS[lang]):
        available = bool(result.availability[h, q, expert])
        if lang == "en":
            row = {
                "Seasonal expert": label,
                "Available": "Yes" if available else "No",
                "Support": round(float(result.reliability[h, q, expert]), 4),
                "Ensemble-mean value": (
                    round(float(result.expert_values[h, q, expert]), 4)
                    if available
                    else None
                ),
                "Mean routing weight": round(float(result.attention[h, q, expert]), 6),
            }
        else:
            row = {
                "季节专家": label,
                "可用": "是" if available else "否",
                "支持度": round(float(result.reliability[h, q, expert]), 4),
                "集成平均值": (
                    round(float(result.expert_values[h, q, expert]), 4)
                    if available
                    else None
                ),
                "平均路由权重": round(float(result.attention[h, q, expert]), 6),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def member_dataframe(result: AuditResult, lang: str = "en") -> pd.DataFrame:
    rows = []
    for member in result.members:
        config = member.selected_config
        if lang == "en":
            rows.append(
                {
                    "Seed": member.seed,
                    "Configuration": config.get("name"),
                    "Epoch": member.selected_epoch,
                    "Residual bound": config.get("residual_bound"),
                    "Parameters": member.parameter_count,
                    "Checkpoint SHA-256": member.sha256,
                }
            )
        else:
            rows.append(
                {
                    "种子": member.seed,
                    "配置": config.get("name"),
                    "轮次": member.selected_epoch,
                    "残差上限": config.get("residual_bound"),
                    "参数量": member.parameter_count,
                    "检查点 SHA-256": member.sha256,
                }
            )
    return pd.DataFrame(rows)


def make_forecast_figure(
    result: AuditResult, metric_key: str = "dl_prb", lang: str = "en"
):
    q = METRIC_INDEX[metric_key]
    metric_label = METRIC_LABELS[lang][q]
    shown = np.where(
        result.effective_mask[:, q], result.history_values[:, q], np.nan
    )
    figure, axis = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    axis.plot(
        result.history_times,
        shown,
        color="#3d6fb6",
        linewidth=1.25,
        label="Observed history" if lang == "en" else "已观测历史",
    )
    available_values = shown[np.isfinite(shown)]
    marker_y = float(np.min(available_values)) if available_values.size else 0.0
    missing = ~result.effective_mask[:, q]
    if np.any(missing):
        axis.scatter(
            np.asarray(result.history_times, dtype=object)[missing],
            np.full(int(np.sum(missing)), marker_y),
            marker="|",
            s=30,
            color="#b55a52",
            alpha=0.75,
            label="Missing position" if lang == "en" else "缺失位置",
        )
    axis.plot(
        result.forecast_times,
        result.prediction[:, q],
        color="#172b4d",
        linewidth=2.4,
        marker="o",
        markersize=3,
        label="A6 ensemble" if lang == "en" else "A6 五模型集成",
    )
    axis.fill_between(
        result.forecast_times,
        result.lower_envelope[:, q],
        result.upper_envelope[:, q],
        color="#168c7e",
        alpha=0.14,
        label="Audited envelope" if lang == "en" else "审计边界",
    )
    axis.axvline(
        result.forecast_times[0], color="#5b6573", linestyle="--", linewidth=1
    )
    axis.set_ylabel(metric_label)
    axis.set_title(
        f"{metric_label}: 336-hour history → 24-hour forecast"
        if lang == "en"
        else f"{metric_label}：336 小时历史 → 24 小时预测",
        loc="left",
        fontweight="bold",
    )
    axis.grid(alpha=0.18)
    axis.legend(loc="upper left", ncol=4, fontsize=8)
    return figure


def make_expert_figure(
    result: AuditResult,
    metric_key: str = "dl_prb",
    horizon: int = 1,
    lang: str = "en",
):
    h = int(horizon) - 1
    q = METRIC_INDEX[metric_key]
    available = result.availability[h, q]
    values = np.where(available, result.expert_values[h, q], np.nan)
    weights = result.attention[h, q]
    colors = np.where(available, "#3d6fb6", "#d9e1e8")
    figure, axis = plt.subplots(figsize=(10.5, 4.1), constrained_layout=True)
    positions = np.arange(len(SHORT_EXPERT_LABELS))
    bars = axis.bar(positions, np.nan_to_num(values), color=colors, alpha=0.9)
    for index, bar in enumerate(bars):
        if not available[index]:
            bar.set_hatch("///")
    axis.set_xticks(positions, SHORT_EXPERT_LABELS)
    axis.set_ylabel("Expert value" if lang == "en" else "专家值")
    axis.grid(axis="y", alpha=0.18)
    weight_axis = axis.twinx()
    weight_axis.plot(
        positions, weights * 100.0, color="#d9822b", marker="o", linewidth=2.1
    )
    weight_axis.set_ylabel(
        "Mean weight (%)" if lang == "en" else "平均权重（%）", color="#a85f1e"
    )
    weight_axis.set_ylim(0.0, max(105.0, float(np.max(weights) * 115.0)))
    label = METRIC_LABELS[lang][q]
    axis.set_title(
        f"Why this forecast? {label}, future hour {horizon}"
        if lang == "en"
        else f"为什么得到这个预测？{label}，未来第 {horizon} 小时",
        loc="left",
        fontweight="bold",
    )
    return figure


def _audit_checks(result: AuditResult) -> tuple[bool, bool]:
    mask_ok = all(
        np.all(member.attention[~result.availability] == 0.0)
        for member in result.members
    )
    bound_ok = bool(
        np.all(result.prediction >= result.lower_envelope - 1e-5)
        and np.all(result.prediction <= result.upper_envelope + 1e-5)
    )
    return mask_ok, bound_ok


def status_markdown(result: AuditResult, lang: str = "en") -> str:
    observed = float(np.mean(result.effective_mask))
    mask_ok, bound_ok = _audit_checks(result)
    audit = (
        f"Mask {'PASS' if mask_ok else 'FAIL'} · "
        f"Bound {'PASS' if bound_ok else 'FAIL'}"
    )
    if lang == "en":
        return (
            "| Model | Input | Observed | Audit |\n"
            "| --- | --- | ---: | --- |\n"
            f"| A6 ensemble · {len(result.members)} seeds | 336 h → 24 h | "
            f"{observed:.1%} | {audit} |"
        )
    audit_zh = (
        f"掩码{'通过' if mask_ok else '失败'} · "
        f"边界{'通过' if bound_ok else '失败'}"
    )
    return (
        "| 模型 | 输入输出 | 有效观测 | 审计 |\n"
        "| --- | --- | ---: | --- |\n"
        f"| A6 集成 · {len(result.members)} 个种子 | 336 小时 → 24 小时 | "
        f"{observed:.1%} | {audit_zh} |"
    )


def _array(values: np.ndarray) -> list:
    return np.round(np.asarray(values, dtype=np.float64), 7).tolist()


def export_outputs(result: AuditResult) -> tuple[str, str]:
    output_dir = Path(tempfile.mkdtemp(prefix="wlcr-sea-a6-"))
    forecast_path = output_dir / "wlcr_sea_a6_ensemble_forecast.csv"
    audit_path = output_dir / "wlcr_sea_a6_audit_record.json"
    forecast_dataframe(result, "en").to_csv(
        forecast_path, index=False, encoding="utf-8-sig"
    )
    mask_ok, bound_ok = _audit_checks(result)
    payload = {
        "schema": "wlcr-sea-audit/v2",
        "paper_model": True,
        "variant": result.variant,
        "ensemble": {
            "model_repo": result.model_repo_id,
            "revision": result.model_revision,
            "aggregation": result.aggregation,
            "members": [
                {
                    "seed": member.seed,
                    "checkpoint": member.filename,
                    "checkpoint_sha256": member.sha256,
                    "selected_config": member.selected_config,
                    "selected_epoch": member.selected_epoch,
                    "parameter_count": member.parameter_count,
                    "prediction": _array(member.prediction),
                    "expert_values": _array(member.expert_values),
                    "routing_weights": _array(member.attention),
                    "baseline_log": _array(member.baseline_log),
                    "residual_log": _array(member.residual_log),
                }
                for member in result.members
            ],
        },
        "input": {
            "sha256": result.input_sha256,
            "cell": result.cell,
            "history_hours": 336,
            "forecast_hours": 24,
            "scenario": result.scenario,
            "requested_missing_rate": result.requested_rate,
            "observed_fraction": float(np.mean(result.effective_mask)),
        },
        "ensemble_output": {
            "prediction": _array(result.prediction),
            "expert_values_mean": _array(result.expert_values),
            "availability": result.availability.astype(int).tolist(),
            "reliability": _array(result.reliability),
            "routing_weights_mean": _array(result.attention),
            "lower_envelope": _array(result.lower_envelope),
            "upper_envelope": _array(result.upper_envelope),
        },
        "checks": {
            "unavailable_expert_weight_is_zero": mask_ok,
            "weights_normalized": all(
                np.allclose(member.attention.sum(axis=-1), 1.0, atol=1e-7)
                for member in result.members
            ),
            "prediction_within_ensemble_envelope": bound_ok,
        },
    }
    audit_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return str(forecast_path), str(audit_path)
