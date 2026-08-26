"""Gradio interface for the public five-model WLCR-SEA ensemble."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# The existing hosted repository is assigned ZeroGPU hardware and Hugging Face
# currently rejects an in-place downgrade for its free owner account. Startup
# therefore requires one detectable decorator even though inference is CPU-only.
# This marker is never connected to Gradio and is never called.
try:
    import spaces
except ImportError:

    class _LocalSpaces:
        @staticmethod
        def GPU(*, duration: int):
            del duration
            return lambda function: function

    spaces = _LocalSpaces()


@spaces.GPU(duration=1)
def _host_hardware_compatibility_marker():
    return None


import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.model_loader import load_ensemble
from demo.runtime import (
    DemoInputError,
    METRIC_INDEX,
    METRIC_KEYS,
    METRIC_LABELS,
    SCENARIO_LABELS,
    AuditResult,
    expert_dataframe,
    export_outputs,
    forecast_dataframe,
    make_expert_figure,
    make_forecast_figure,
    member_dataframe,
    run_forecast,
    source_commit,
    status_markdown,
)


LOGGER = logging.getLogger(__name__)
EXAMPLE_PATH = ROOT / "demo" / "examples" / "synthetic_traffic.csv"
# Space startup resolves, verifies, and loads all five public checkpoints once.
ENSEMBLE = load_ensemble()

CSS = """
:root { --wlcr-navy: #172b4d; --wlcr-blue: #3d6fb6; --wlcr-teal: #0f766e; }
.gradio-container { max-width: 1180px !important; }
.wlcr-header { padding: .35rem 0 1rem; border-bottom: 1px solid #d9e1e8; margin-bottom: 1rem; }
.wlcr-header h1 { color: var(--wlcr-navy); font-size: 2rem; margin: 0 0 .25rem; }
.wlcr-header p { color: #5b6573; margin: .2rem 0; }
.wlcr-meta { color: var(--wlcr-teal) !important; font-size: .9rem; font-weight: 650; }
.wlcr-links { display: flex; flex-wrap: wrap; gap: .85rem; margin-top: .55rem; }
.wlcr-links a { color: var(--wlcr-blue); font-size: .9rem; font-weight: 650; text-decoration: none; }
.wlcr-links a:hover { text-decoration: underline; }
.primary-action { background: var(--wlcr-blue) !important; border-color: var(--wlcr-blue) !important; }
.privacy-note { color: #5b6573; font-size: .86rem; }
.status-table table { margin: .35rem 0 0 !important; }
.compact-panel { border: 1px solid #d9e1e8; border-radius: 10px; padding: .8rem; }
"""
THEME = gr.themes.Base(primary_hue="blue", secondary_hue="teal")


@dataclass
class Panel:
    state: gr.State
    upload: gr.File
    scenario: gr.Dropdown
    missing_rate: gr.Slider
    metric: gr.Dropdown
    horizon: gr.Slider
    status: gr.Markdown
    forecast_plot: gr.Plot
    forecast_table: gr.Dataframe
    expert_plot: gr.Plot
    expert_table: gr.Dataframe
    member_table: gr.Dataframe
    forecast_download: gr.File
    audit_download: gr.File

    @property
    def result_outputs(self):
        return [
            self.state,
            self.status,
            self.forecast_plot,
            self.forecast_table,
            self.expert_plot,
            self.expert_table,
            self.member_table,
            self.forecast_download,
            self.audit_download,
        ]


TEXT = {
    "en": {
        "header": """
        <section class="wlcr-header">
          <h1>WLCR-SEA Forecast Demo</h1>
          <p class="wlcr-meta">Five-model ensemble · Verified public checkpoints · CPU inference</p>
          <p>One cell · 336 hours of history → 24 hours of forecast</p>
          <nav class="wlcr-links" aria-label="Project links">
            <a href="https://github.com/rudykon/WLCR-SEA_Predictor" target="_blank" rel="noopener noreferrer">GitHub repository</a>
            <a href="https://rudykon.github.io/WLCR-SEA_Predictor/" target="_blank" rel="noopener noreferrer">Project website</a>
          </nav>
        </section>
        """,
        "sample": "Reset to sample",
        "upload": "Upload a 336-hour CSV",
        "privacy": "Public Demo. Do not upload confidential operator traffic.",
        "scenario": "Missingness pattern",
        "rate": "Additional missingness",
        "run": "Run forecast",
        "metric": "Indicator",
        "horizon": "Future hour",
        "advanced": "Advanced details",
        "forecast_table": "Full 24 × 4 forecast",
        "expert_table": "Eight experts at the selected hour",
        "member_table": "Ensemble members",
        "forecast_download": "Download forecast CSV",
        "audit_download": "Download audit JSON",
        "download_hint": (
            "Run the forecast to enable CSV and audit JSON downloads. "
            "The automatic preview does not create temporary download files."
        ),
        "ready": "The built-in sample will run automatically.",
        "routing_title": "## Ensemble routing summary",
        "routing_note": (
            "Mean expert values and routing weights across five members. "
            "This view summarizes internal routing and does not exactly "
            "decompose the ensemble prediction."
        ),
    },
    "zh": {
        "header": """
        <section class="wlcr-header">
          <h1>WLCR-SEA 流量预测 Demo</h1>
          <p class="wlcr-meta">五模型集成 · 公开检查点已校验 · CPU 推理</p>
          <p>单个小区 · 336 小时历史 → 未来 24 小时预测</p>
          <nav class="wlcr-links" aria-label="项目链接">
            <a href="https://github.com/rudykon/WLCR-SEA_Predictor" target="_blank" rel="noopener noreferrer">GitHub 仓库</a>
            <a href="https://rudykon.github.io/WLCR-SEA_Predictor/zh/" target="_blank" rel="noopener noreferrer">项目网站</a>
          </nav>
        </section>
        """,
        "sample": "恢复内置样例",
        "upload": "上传 336 小时 CSV",
        "privacy": "这是公开 Demo，请勿上传运营商机密流量数据。",
        "scenario": "缺失模式",
        "rate": "追加缺失率",
        "run": "运行预测",
        "metric": "流量指标",
        "horizon": "未来第几小时",
        "advanced": "高级详情",
        "forecast_table": "完整 24 × 4 预测",
        "expert_table": "所选时刻的八个季节专家",
        "member_table": "集成成员",
        "forecast_download": "下载预测 CSV",
        "audit_download": "下载审计 JSON",
        "download_hint": (
            "点击“运行预测”后可下载 CSV 与审计 JSON；自动预览不会创建临时下载文件。"
        ),
        "ready": "页面将自动运行内置样例。",
        "routing_title": "## 集成路由摘要",
        "routing_note": (
            "展示五个成员的平均专家值与平均路由权重，用于观察内部路由倾向，"
            "不能直接重构最终集成预测。"
        ),
    },
}


def _render_result(
    result: AuditResult,
    metric_key: str,
    horizon: int,
    lang: str,
    exported: tuple[str | None, str | None] | None = None,
    *,
    create_exports: bool = True,
):
    forecast_path, audit_path = (
        exported
        if exported is not None
        else export_outputs(result)
        if create_exports
        else (None, None)
    )
    return (
        result,
        status_markdown(result, lang),
        make_forecast_figure(result, metric_key, lang),
        forecast_dataframe(result, lang),
        make_expert_figure(result, metric_key, horizon, lang),
        expert_dataframe(result, metric_key, horizon, lang),
        member_dataframe(result, lang),
        forecast_path,
        audit_path,
    )


def _run_request(upload, scenario, missing_rate, metric, horizon, *, lang: str):
    try:
        result = run_forecast(
            upload,
            scenario=str(scenario),
            missing_rate=float(missing_rate),
            ensemble=ENSEMBLE,
        )
        return _render_result(result, str(metric), int(horizon), lang)
    except DemoInputError as exc:
        raise gr.Error(exc.localized(lang)) from exc
    except Exception as exc:
        LOGGER.exception("WLCR-SEA Forecast Demo failed")
        message = (
            "The forecast could not finish. Check the 336-row input and try again."
            if lang == "en"
            else "预测未能完成，请检查 336 行输入后重试。"
        )
        raise gr.Error(message) from exc


def _use_sample(scenario, missing_rate, metric, horizon, *, lang: str):
    rendered = _run_request(
        str(EXAMPLE_PATH), scenario, missing_rate, metric, horizon, lang=lang
    )
    return (str(EXAMPLE_PATH), *rendered)


def _update_view(result, metric, horizon, *, lang: str):
    if result is None:
        return None, None, None
    return (
        make_forecast_figure(result, str(metric), lang),
        make_expert_figure(result, str(metric), int(horizon), lang),
        expert_dataframe(result, str(metric), int(horizon), lang),
    )


def _sync_missing_rate(scenario, current_rate, *, label: str):
    if str(scenario) == "none":
        return gr.Slider(
            0, 0.8, value=0.0, step=0.05, label=label, interactive=False
        )
    rate = float(current_rate)
    return gr.Slider(
        0,
        0.8,
        value=0.2 if rate <= 0.0 else rate,
        step=0.05,
        label=label,
        interactive=True,
    )


def _build_panel(lang: str) -> Panel:
    text = TEXT[lang]
    gr.HTML(text["header"])
    state = gr.State()
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, elem_classes=["compact-panel"]):
            sample_button = gr.Button(text["sample"])
            upload = gr.File(
                value=str(EXAMPLE_PATH),
                label=text["upload"],
                file_types=[".csv"],
                type="filepath",
            )
            gr.Markdown(text["privacy"], elem_classes=["privacy-note"])
            scenario = gr.Dropdown(
                choices=[
                    (label, key) for key, label in SCENARIO_LABELS[lang].items()
                ],
                value="none",
                label=text["scenario"],
            )
            missing_rate = gr.Slider(
                0,
                0.8,
                value=0.0,
                step=0.05,
                label=text["rate"],
                interactive=False,
            )
            run_button = gr.Button(
                text["run"], variant="primary", elem_classes=["primary-action"]
            )
        with gr.Column(scale=8):
            forecast_plot = gr.Plot(show_label=False)
            status = gr.Markdown(text["ready"], elem_classes=["status-table"])

    gr.Markdown(text["routing_title"])
    gr.Markdown(text["routing_note"], elem_classes=["privacy-note"])
    with gr.Row():
        metric = gr.Dropdown(
            choices=[
                (METRIC_LABELS[lang][METRIC_INDEX[key]], key) for key in METRIC_KEYS
            ],
            value="dl_prb",
            label=text["metric"],
            scale=2,
        )
        horizon = gr.Slider(
            1, 24, value=1, step=1, label=text["horizon"], scale=4
        )
    expert_plot = gr.Plot(show_label=False)

    with gr.Accordion(text["advanced"], open=False):
        forecast_table = gr.Dataframe(
            interactive=False, label=text["forecast_table"]
        )
        expert_table = gr.Dataframe(interactive=False, label=text["expert_table"])
        member_table = gr.Dataframe(interactive=False, label=text["member_table"])
        gr.Markdown(text["download_hint"], elem_classes=["privacy-note"])
        with gr.Row():
            forecast_download = gr.File(
                label=text["forecast_download"], interactive=False
            )
            audit_download = gr.File(
                label=text["audit_download"], interactive=False
            )
        gr.Markdown(
            "[Input schema](https://rudykon.github.io/WLCR-SEA_Predictor/reference/reproduction/#input-format) · "
            "[Pinned model weights](https://huggingface.co/config-h/WLCR-SEA-Predictor/tree/eb4447f4ebab8f9caa003d92c838ed8e750963bd)"
            if lang == "en"
            else "[输入格式](https://rudykon.github.io/WLCR-SEA_Predictor/zh/reference/reproduction/#input-format) · "
            "[固定版本模型权重](https://huggingface.co/config-h/WLCR-SEA-Predictor/tree/eb4447f4ebab8f9caa003d92c838ed8e750963bd)"
        )

    panel = Panel(
        state=state,
        upload=upload,
        scenario=scenario,
        missing_rate=missing_rate,
        metric=metric,
        horizon=horizon,
        status=status,
        forecast_plot=forecast_plot,
        forecast_table=forecast_table,
        expert_plot=expert_plot,
        expert_table=expert_table,
        member_table=member_table,
        forecast_download=forecast_download,
        audit_download=audit_download,
    )
    run_button.click(
        fn=partial(_run_request, lang=lang),
        inputs=[upload, scenario, missing_rate, metric, horizon],
        outputs=panel.result_outputs,
        api_name=f"forecast_{lang}",
        concurrency_limit=1,
    )
    sample_button.click(
        fn=partial(_use_sample, lang=lang),
        inputs=[scenario, missing_rate, metric, horizon],
        outputs=[upload, *panel.result_outputs],
        api_name=f"sample_{lang}",
        api_visibility="private",
        concurrency_limit=1,
    )
    scenario.change(
        fn=partial(_sync_missing_rate, label=text["rate"]),
        inputs=[scenario, missing_rate],
        outputs=missing_rate,
        show_progress="hidden",
        api_name=f"missingness_control_{lang}",
        api_visibility="private",
    )
    for view_index, component in enumerate((metric, horizon), start=1):
        component.change(
            fn=partial(_update_view, lang=lang),
            inputs=[state, metric, horizon],
            outputs=[forecast_plot, expert_plot, expert_table],
            show_progress="hidden",
            api_name=f"view_{lang}_{view_index}",
            api_visibility="private",
        )
    return panel


def build_app() -> gr.Blocks:
    with gr.Blocks(title="WLCR-SEA Cellular Traffic Forecast Demo") as app:
        with gr.Tabs():
            with gr.Tab("English"):
                english = _build_panel("en")
            with gr.Tab("中文"):
                chinese = _build_panel("zh")

        def load_default():
            result = run_forecast(EXAMPLE_PATH, ensemble=ENSEMBLE)
            return (
                *_render_result(
                    result, "dl_prb", 1, "en", create_exports=False
                ),
                *_render_result(
                    result, "dl_prb", 1, "zh", create_exports=False
                ),
            )

        app.load(
            fn=load_default,
            outputs=[*english.result_outputs, *chinese.result_outputs],
            api_name="load_default",
            api_visibility="private",
            concurrency_limit=1,
        )

        deployment_smoke = gr.Button(visible=False)
        deployment_smoke_output = gr.JSON(visible=False)

        def run_deployment_smoke():
            result = run_forecast(EXAMPLE_PATH, ensemble=ENSEMBLE)
            return {
                "source_commit": source_commit(),
                "model_revision": result.model_revision,
                "member_count": len(result.members),
                "prediction_shape": list(result.prediction.shape),
                "first_prediction": [
                    round(float(value), 7) for value in result.prediction[0]
                ],
            }

        deployment_smoke.click(
            fn=run_deployment_smoke,
            outputs=deployment_smoke_output,
            api_name="deployment_smoke",
            api_description=(
                "Run the bundled request through the deployed five-model predictor "
                "and report its exact source and model revisions."
            ),
            api_visibility="public",
            concurrency_limit=1,
        )
    return app


demo = build_app()


if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        theme=THEME,
        css=CSS,
        max_file_size="5mb",
    )
