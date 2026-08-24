"""Bilingual Gradio interface for the WLCR-SEA traffic forecast Demo."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import spaces  # ZeroGPU must be imported before the lazy PyTorch runtime path.
import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.runtime import (
    DemoInputError,
    METRIC_CHOICES,
    SCENARIOS,
    expert_dataframe,
    export_outputs,
    forecast_dataframe,
    make_expert_figure,
    make_forecast_figure,
    run_audit_demo,
    status_markdown,
)


LOGGER = logging.getLogger(__name__)
EXAMPLE_PATH = ROOT / "demo" / "examples" / "synthetic_traffic.csv"

CSS = """
:root { --wlcr-indigo: #4f46e5; --wlcr-violet: #7c3aed; --wlcr-ink: #0f172a; }
.gradio-container { max-width: 1260px !important; }
.wlcr-hero {
  margin-bottom: .85rem; padding: 1.4rem 1.55rem; border-radius: 22px;
  border: 1px solid rgba(79, 70, 229, .18);
  background: linear-gradient(135deg, rgba(238,242,255,.98), rgba(245,243,255,.98));
  box-shadow: 0 18px 52px rgba(79,70,229,.11);
}
.wlcr-hero h1 { margin: 0 0 .45rem; color: var(--wlcr-ink); font-size: clamp(1.7rem,4vw,2.6rem); }
.wlcr-hero p { margin: .22rem 0; color: #475569; max-width: 960px; }
.wlcr-pill { display: inline-block; margin-bottom: .65rem; padding: .28rem .7rem;
  border-radius: 999px; color: #4338ca; background: rgba(99,102,241,.11);
  font-size: .76rem; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; }
.primary-action { background: linear-gradient(135deg, var(--wlcr-indigo), var(--wlcr-violet)) !important; }
.scope-note { color: #64748b; font-size: .91rem; }
"""
THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet")


@spaces.GPU(duration=30)
def _run_zero_gpu_audit(upload, scenario, missing_rate, metric, horizon):
    return run_audit_demo(
        upload,
        scenario_label=str(scenario),
        missing_rate=float(missing_rate),
        metric_label=str(metric),
        horizon=int(horizon),
    )


def run_request(upload, scenario, missing_rate, metric, horizon):
    try:
        result = _run_zero_gpu_audit(upload, scenario, missing_rate, metric, horizon)
        forecast_csv, audit_json = export_outputs(result)
        return (
            status_markdown(result),
            make_forecast_figure(result),
            make_expert_figure(result),
            forecast_dataframe(result),
            expert_dataframe(result),
            forecast_csv,
            audit_json,
        )
    except DemoInputError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("WLCR-SEA public demo failed")
        raise gr.Error(
            "The Demo could not finish. Check that the input has 336 valid hourly rows and try again."
        ) from exc


def build_app() -> gr.Blocks:
    with gr.Blocks(title="WLCR-SEA Traffic Forecast Demo") as app:
        gr.HTML(
            """
            <section class="wlcr-hero">
              <span class="wlcr-pill">ZeroGPU · Real project code · 真实项目代码</span>
              <h1>WLCR-SEA Traffic Forecast Demo</h1>
              <p>Use one cell's previous 14 days to forecast the next 24 hours. Remove part of
              the history to see which reference forecasts remain available and how their
              weights change.</p>
              <p>使用一个小区过去 14 天的数据预测未来 24 小时。你可以移除部分历史数据，
              查看哪些参考预测仍然可用，以及它们的权重怎样变化。</p>
            </section>
            """
        )
        gr.Markdown(
            "**Important / 重要：** the repository does not include the trained A6 paper checkpoint. "
            "This Demo uses the simpler fixed `A0_fixed` baseline to show the workflow; its forecast is not a paper result. "
            "仓库未包含论文训练后的 A6 检查点；本 Demo 使用更简单的固定 `A0_fixed` 基线展示流程，输出不是论文结果。",
            elem_classes=["scope-note"],
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                upload = gr.File(
                    label="336-hour request CSV / 336 小时请求 CSV",
                    file_types=[".csv"],
                    type="filepath",
                )
                gr.Markdown(
                    "Required header: `时间, 小区名称` plus the four documented traffic indicators. "
                    "Use `NIL` or a blank field for an unavailable measurement."
                )
                run_button = gr.Button(
                    "Generate forecast / 生成预测", variant="primary", elem_classes=["primary-action"]
                )
            with gr.Column(scale=4):
                scenario = gr.Dropdown(
                    choices=list(SCENARIOS),
                    value="Clean request / 完整请求",
                    label="Telemetry scenario / 遥测场景",
                )
                missing_rate = gr.Slider(
                    0,
                    0.8,
                    value=0.2,
                    step=0.05,
                    label="Additional missingness / 追加缺失率",
                )
                metric = gr.Dropdown(
                    choices=list(METRIC_CHOICES),
                    value="DL PRB / 下行 PRB",
                    label="Indicator to inspect / 查看哪个指标",
                )
                horizon = gr.Slider(
                    1,
                    24,
                    value=1,
                    step=1,
                    label="Future hour to inspect / 查看未来第几小时",
                )

        gr.Examples(
            examples=[
                [str(EXAMPLE_PATH), "Clean request / 完整请求", 0.2, "DL PRB / 下行 PRB", 1],
                [str(EXAMPLE_PATH), "Recent-tail outage / 最近时段中断", 0.5, "DL active users / 下行激活用户", 12],
            ],
            inputs=[upload, scenario, missing_rate, metric, horizon],
            label="Bundled synthetic request / 内置合成请求",
            cache_examples=False,
        )

        status = gr.Markdown("### Ready / 就绪\nUpload a CSV or choose a built-in sample, then generate the forecast. / 上传 CSV 或选择内置样例，然后生成预测。")
        with gr.Tabs():
            with gr.Tab("Forecast / 预测"):
                forecast_plot = gr.Plot(label="History, forecast, and available-reference range")
                forecast_table = gr.Dataframe(interactive=False, label="24-hour forecast / 24 小时预测")
                forecast_download = gr.File(label="Download forecast CSV / 下载预测 CSV", interactive=False)
            with gr.Tab("Reference details / 参考项明细"):
                expert_plot = gr.Plot(label="Reference values and fixed weights")
                expert_table = gr.Dataframe(interactive=False, label="Eight reference forecasts / 八种参考预测")
                audit_download = gr.File(label="Download calculation JSON / 下载计算记录 JSON", interactive=False)

        with gr.Accordion("Input rules and limitations / 输入规则与局限", open=False):
            gr.Markdown(
                """
                - Exactly 336 contiguous hourly rows for one cell are accepted; uploads are limited to 5 MB.
                - Do not upload confidential operational traffic to a public Space. The app does not
                  intentionally persist files, but the host is shared public infrastructure.
                - The bundled example is deterministic synthetic traffic and contains no operator data.
                - Candidate weights shown here belong to the fixed baseline. They are not learned
                  A6 weights, uncertainty estimates, or the paper's reported predictions.

                - 仅接收同一小区连续 336 个小时的数据，上传文件上限为 5 MB。
                - 请勿向公开 Space 上传机密运营数据；应用不会主动持久化文件，但托管平台属于共享基础设施。
                - 内置样例是确定性合成流量，不包含运营商数据。
                - 页面中的候选权重属于固定基线，不是 A6 学习权重、置信度或论文正式预测。
                """
            )

        run_button.click(
            fn=run_request,
            inputs=[upload, scenario, missing_rate, metric, horizon],
            outputs=[
                status,
                forecast_plot,
                expert_plot,
                forecast_table,
                expert_table,
                forecast_download,
                audit_download,
            ],
            api_name="audit",
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
