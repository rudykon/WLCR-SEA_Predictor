"""Runtime adapter for the public WLCR-SEA audit lab.

The repository does not distribute the trained A6 checkpoint used for the
paper.  The demo therefore runs the registered, parameter-free ``A0_fixed``
baseline through the real expert-construction and masking code.  Its outputs
are useful for inspecting request-local evidence and failure behavior, but are
deliberately not presented as the paper model's reported forecasts.
"""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEMO_SEED = 2026
SCENARIOS = {
    "Clean request / 完整请求": "none",
    "Random whole-hour loss / 随机整小时缺失": "mcar",
    "Contiguous block loss / 连续区块缺失": "block",
    "Recent-tail outage / 最近时段中断": "recent_tail",
    "Asynchronous indicator loss / 指标异步缺失": "asynchronous",
}
METRIC_CHOICES = {
    "UL active users / 上行激活用户": 0,
    "DL active users / 下行激活用户": 1,
    "DL PRB / 下行 PRB": 2,
    "UL PRB / 上行 PRB": 3,
}
METRIC_LABELS = (
    "UL active users",
    "DL active users",
    "DL PRB",
    "UL PRB",
)
EXPERT_LABELS = (
    "Last day / 前一天",
    "Last week / 前一周",
    "Two-week lag / 两周滞后",
    "Same-hour median 7 d / 7 日同小时中位数",
    "Same-hour median 14 d / 14 日同小时中位数",
    "Bounded weekly trend / 有界周趋势",
    "Window-local median / 窗口局部中位数",
    "Demo fallback prior* / 演示回退先验*",
)
SHORT_EXPERT_LABELS = (
    "1 d",
    "7 d",
    "14 d",
    "Med 7",
    "Med 14",
    "Trend",
    "Local",
    "Fallback",
)


class DemoInputError(ValueError):
    """Raised when a public demo request violates the input contract."""


@dataclass(frozen=True)
class AuditResult:
    """All inspectable outputs from one deterministic method-demo request."""

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
    scenario_label: str
    mechanism: str
    requested_rate: float
    selected_metric: int
    selected_horizon: int
    device: str


def _upload_path(upload: str | Path | Any) -> Path:
    if upload is None:
        raise DemoInputError("Upload a CSV request or select the bundled example.")
    candidate = upload if isinstance(upload, (str, os.PathLike)) else getattr(upload, "name", upload)
    path = Path(str(candidate)).expanduser()
    if not path.is_file():
        raise DemoInputError("The uploaded file is no longer available. Upload it again.")
    if path.suffix.lower() != ".csv":
        raise DemoInputError("Use a UTF-8 CSV file with the documented six-column header.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise DemoInputError("The upload exceeds the 5 MB public-demo limit.")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_request(upload: str | Path | Any):
    """Validate and load exactly one physical 336-hour request window."""

    from Model.traffic_window_forecasting import ContractError, read_traffic, split_physical_windows

    path = _upload_path(upload)
    try:
        rows = read_traffic(path)
        windows = split_physical_windows(rows)
    except (ContractError, UnicodeDecodeError) as exc:
        raise DemoInputError(str(exc)) from exc
    if len(windows) != 1:
        raise DemoInputError("The public demo accepts exactly one 336-row request window.")
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


def _request_fallback_prior(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build a finite demo-only fallback without reading another request."""

    from experiments import wlcr_sea_model as sea

    prior = np.empty((sea.FORECAST_HOURS, sea.TARGET_COUNT), dtype=np.float32)
    for horizon in range(1, sea.FORECAST_HOURS + 1):
        indices = [sea.seasonal_history_index(horizon, day) for day in range(1, 15)]
        for metric in range(sea.TARGET_COUNT):
            seasonal = values[0, indices, metric][mask[0, indices, metric]]
            if seasonal.size:
                fallback = float(np.median(seasonal))
            else:
                observed = values[0, :, metric][mask[0, :, metric]]
                fallback = float(np.median(observed))
            prior[horizon - 1, metric] = np.log1p(max(fallback, 0.0))
    return prior


def _absolute_hour(value: datetime) -> int:
    return int((value - datetime(1970, 1, 1)).total_seconds() // 3600)


def run_audit_demo(
    upload: str | Path | Any,
    *,
    scenario_label: str = "Clean request / 完整请求",
    missing_rate: float = 0.2,
    metric_label: str = "DL PRB / 下行 PRB",
    horizon: int = 1,
) -> AuditResult:
    """Run the real A0 fixed expert mixture and return its audit record."""

    if scenario_label not in SCENARIOS:
        raise DemoInputError(f"Unknown missingness scenario: {scenario_label}")
    if metric_label not in METRIC_CHOICES:
        raise DemoInputError(f"Unknown indicator: {metric_label}")
    if not 0.0 <= float(missing_rate) <= 0.8:
        raise DemoInputError("Missingness rate must be between 0 and 80%.")
    if not 1 <= int(horizon) <= 24:
        raise DemoInputError("Forecast horizon must be between 1 and 24 hours.")

    path, window = load_request(upload)
    raw_values, original_mask = _request_arrays(window)
    mechanism = SCENARIOS[scenario_label]
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

    history_log = np.where(original_mask, np.log1p(raw_values), 0.0).astype(np.float32)
    prior_log = _request_fallback_prior(raw_values, original_mask)
    experts = sea.build_expert_batch(
        history_log,
        original_mask,
        prior_log,
        additional_missing=additional_missing,
    )

    import torch

    thread_count = max(1, min(int(os.getenv("TORCH_NUM_THREADS", "2")), 4))
    torch.set_num_threads(thread_count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sea.WLCRSEA(sea.VARIANTS["A0_fixed"], token_dim=16, hidden_dim=32).to(device)
    with torch.inference_mode():
        output = model(
            torch.from_numpy(experts.values).to(device),
            torch.from_numpy(experts.availability).to(device),
            torch.from_numpy(experts.reliability).to(device),
            torch.from_numpy(experts.context).to(device),
        )
    prediction_log = output["prediction_log"].cpu().numpy()[0]
    attention = output["attention"].cpu().numpy()[0]
    prediction = np.asarray(sea.prediction_from_log(prediction_log), dtype=np.float32)
    lower_log, upper_log = sea.bounded_audit_envelope(
        experts.values[0], experts.availability[0], residual_bound=0.0
    )
    lower = np.asarray(sea.prediction_from_log(lower_log), dtype=np.float32)
    upper = np.asarray(sea.prediction_from_log(upper_log), dtype=np.float32)
    if np.any(attention[~experts.availability[0]] != 0.0):
        raise RuntimeError("Unavailable expert received non-zero routing mass.")
    if np.any(prediction < lower - 1e-5) or np.any(prediction > upper + 1e-5):
        raise RuntimeError("Fixed-mixture forecast escaped the expert envelope.")

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
        prediction=prediction,
        lower_envelope=lower,
        upper_envelope=upper,
        expert_values=np.asarray(sea.prediction_from_log(experts.values[0]), dtype=np.float32),
        availability=experts.availability[0],
        reliability=experts.reliability[0],
        attention=attention,
        scenario_label=scenario_label,
        mechanism=mechanism,
        requested_rate=effective_rate,
        selected_metric=METRIC_CHOICES[metric_label],
        selected_horizon=int(horizon),
        device=str(device),
    )


def forecast_dataframe(result: AuditResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(result.forecast_times):
        rows.append(
            {
                "Timestamp / 时间": timestamp.strftime("%Y/%m/%d %H:%M"),
                "Cell / 小区": result.cell,
                **{
                    label: round(float(result.prediction[index, metric]), 4)
                    for metric, label in enumerate(METRIC_LABELS)
                },
            }
        )
    return pd.DataFrame(rows)


def expert_dataframe(result: AuditResult) -> pd.DataFrame:
    h = result.selected_horizon - 1
    q = result.selected_metric
    rows: list[dict[str, object]] = []
    for expert, label in enumerate(EXPERT_LABELS):
        available = bool(result.availability[h, q, expert])
        rows.append(
            {
                "Expert / 专家": label,
                "Available / 可用": "Yes / 是" if available else "No / 否",
                "Reliability / 可靠度": round(float(result.reliability[h, q, expert]), 4),
                "Candidate value / 候选值": (
                    round(float(result.expert_values[h, q, expert]), 4) if available else None
                ),
                "Routing weight / 路由权重": round(float(result.attention[h, q, expert]), 6),
            }
        )
    return pd.DataFrame(rows)


def make_forecast_figure(result: AuditResult):
    colors = ("#2563eb", "#7c3aed", "#0f766e", "#d97706")
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False, constrained_layout=True)
    for metric, axis in enumerate(axes):
        shown = np.where(result.effective_mask[:, metric], result.history_values[:, metric], np.nan)
        axis.plot(result.history_times, shown, color=colors[metric], linewidth=1.35, label="Observed history")
        missing = ~result.effective_mask[:, metric]
        if np.any(missing):
            observed = result.history_values[:, metric][result.original_mask[:, metric]]
            baseline = float(np.nanmedian(observed)) if observed.size else 0.0
            axis.scatter(
                np.asarray(result.history_times, dtype=object)[missing],
                np.full(int(np.sum(missing)), baseline),
                marker="x",
                s=13,
                color="#ef4444",
                alpha=0.55,
                label="Missing position",
            )
        axis.plot(
            result.forecast_times,
            result.prediction[:, metric],
            color="#111827",
            linewidth=2.4,
            marker="o",
            markersize=3,
            label="A0 fixed forecast",
        )
        axis.fill_between(
            result.forecast_times,
            result.lower_envelope[:, metric],
            result.upper_envelope[:, metric],
            color=colors[metric],
            alpha=0.10,
            label="Available-expert envelope",
        )
        axis.axvline(result.forecast_times[0], color="#64748b", linestyle="--", linewidth=1)
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.grid(alpha=0.18)
        if metric == 0:
            axis.legend(loc="upper left", ncol=4, fontsize=8)
    figure.suptitle(
        "336-hour request and 24-hour deterministic forecast · "
        f"{result.scenario_label.split(' / ', 1)[0]}",
        fontsize=14,
        fontweight="bold",
    )
    return figure


def make_expert_figure(result: AuditResult):
    h = result.selected_horizon - 1
    q = result.selected_metric
    available = result.availability[h, q]
    values = np.where(available, result.expert_values[h, q], np.nan)
    weights = result.attention[h, q]
    colors = np.where(available, "#4f46e5", "#cbd5e1")
    figure, axis = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    positions = np.arange(len(SHORT_EXPERT_LABELS))
    bars = axis.bar(positions, np.nan_to_num(values), color=colors, alpha=0.86)
    for index, bar in enumerate(bars):
        if not available[index]:
            bar.set_hatch("///")
    axis.set_xticks(positions, SHORT_EXPERT_LABELS)
    axis.set_ylabel("Candidate value")
    axis.grid(axis="y", alpha=0.18)
    weight_axis = axis.twinx()
    weight_axis.plot(positions, weights * 100.0, color="#f97316", marker="o", linewidth=2.3)
    weight_axis.set_ylabel("Routing weight (%)", color="#c2410c")
    weight_axis.set_ylim(0.0, max(105.0, float(np.max(weights) * 115.0)))
    axis.set_title(
        f"Expert audit · {METRIC_LABELS[q]} · horizon +{result.selected_horizon} h",
        fontweight="bold",
    )
    return figure


def status_markdown(result: AuditResult) -> str:
    removed = 1.0 - float(np.mean(result.effective_mask))
    unavailable_mass = float(np.sum(result.attention[~result.availability]))
    envelope_ok = bool(
        np.all(result.prediction >= result.lower_envelope - 1e-5)
        and np.all(result.prediction <= result.upper_envelope + 1e-5)
    )
    return (
        "### Audit complete / 审计完成\n"
        f"- **Engine:** registered `A0_fixed` parameter-free baseline on `{result.device}`; "
        "this is **not** the unpublished trained A6 checkpoint.\n"
        f"- **Request:** `{result.cell}`, 336 history hours → 24 forecast hours; "
        f"final missing rate **{removed:.1%}**.\n"
        f"- **Hard mask:** unavailable-expert mass `{unavailable_mass:.1f}`; "
        f"expert envelope `{'PASS' if envelope_ok else 'FAIL'}`.\n"
        f"- **Input SHA-256:** `{result.input_sha256[:16]}…`"
    )


def export_outputs(result: AuditResult) -> tuple[str, str]:
    output_dir = Path(tempfile.mkdtemp(prefix="wlcr-sea-demo-"))
    forecast_path = output_dir / "wlcr_sea_a0_fixed_forecast.csv"
    audit_path = output_dir / "wlcr_sea_audit_record.json"
    forecast_dataframe(result).to_csv(forecast_path, index=False, encoding="utf-8-sig")
    payload = {
        "schema": "wlcr-sea-demo-audit/v1",
        "method": "A0_fixed",
        "paper_model": False,
        "notice": "The trained A6 checkpoint is not distributed; this record is a deterministic method demonstration.",
        "input_sha256": result.input_sha256,
        "cell": result.cell,
        "history_hours": 336,
        "forecast_hours": 24,
        "missingness": {
            "mechanism": result.mechanism,
            "requested_rate": result.requested_rate,
            "final_rate": 1.0 - float(np.mean(result.effective_mask)),
        },
        "hard_mask": {
            "unavailable_expert_mass": float(np.sum(result.attention[~result.availability])),
            "all_weights_normalized": bool(
                np.allclose(np.sum(result.attention, axis=-1), 1.0, atol=1e-7)
            ),
        },
        "envelope": {
            "residual_bound_log": 0.0,
            "violations": int(
                np.sum(result.prediction < result.lower_envelope - 1e-5)
                + np.sum(result.prediction > result.upper_envelope + 1e-5)
            ),
        },
        "selected_expert_view": {
            "horizon": result.selected_horizon,
            "metric": METRIC_LABELS[result.selected_metric],
            "rows": expert_dataframe(result).to_dict(orient="records"),
        },
    }
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(forecast_path), str(audit_path)
