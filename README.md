<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/brand/logo.svg">
    <img src="docs/assets/brand/logo.svg" width="360" alt="WLCR-SEA Predictor logo">
  </picture>
</p>

<h1 align="center">WLCR-SEA Predictor</h1>

<p align="center">
  <strong>Forecast the next 24 hours of one cell's traffic using only that cell's recent data</strong><br>
  Open-source code, paper results, and an interactive Demo for WLCR-SEA.
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README_CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://rudykon.github.io/WLCR-SEA_Predictor/"><img src="https://img.shields.io/badge/Project-Website-172B4D?style=flat-square" alt="Project website"></a>
  <a href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor"><img src="https://img.shields.io/badge/🤗-Live%20Demo-FEBD08?style=flat-square" alt="Hugging Face Live Demo"></a>
  <a href="#validation"><img src="https://img.shields.io/badge/Validation-unittest-2CA02C?style=flat-square" alt="Unit tests"></a>
  <a href="#live-demo"><img src="https://img.shields.io/badge/Demo-A0__fixed-6A5ACD?style=flat-square" alt="Demo uses A0 fixed baseline"></a>
</p>

<p align="center">
  <a href="https://rudykon.github.io/WLCR-SEA_Predictor/">Website</a> ·
  <a href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor">Live Demo</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#method">Method</a> ·
  <a href="#figures">Figures</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#validation">Validation</a> ·
  <a href="#resources">Resources</a>
</p>

<a id="overview"></a>
## Overview

Suppose an operator needs tomorrow's traffic forecast for one cell. For access
or system-isolation reasons, the model may be allowed to read only the data
included with that request—not live traffic from neighboring cells.

WLCR-SEA (Window-Local Context Representation with Seasonal Expert Attention)
is designed for this setting. It takes the previous 14 days (336 hours) of four
traffic indicators and predicts the next 24 hours. If some historical values
are missing, the method removes the affected references instead of treating a
placeholder as real data.

The model also reports which historical patterns influenced each prediction.
This makes it easier to inspect a result, compare it with available data, and
replay the same request later.

**At a glance:**

- **Input:** 336 hourly rows for one cell and four traffic indicators.
- **Output:** 24 hourly predictions for the same four indicators.
- **Missing data:** unavailable historical references receive zero weight.
- **Inspection:** candidate values, weights, and range checks can be exported.

<a id="method"></a>
## Method

WLCR-SEA first creates eight **candidate forecasts** for every future hour and
traffic indicator. The paper calls them *experts*, but each expert is simply a
clear rule based on a familiar historical pattern.

| Candidate | What it uses |
| --- | --- |
| Previous day | The same hour one day earlier |
| Previous week | The same hour one week earlier |
| Two weeks earlier | The same hour two weeks earlier |
| 7-day same-hour median | A robust summary of the previous seven matching hours |
| 14-day same-hour median | A robust summary of the previous fourteen matching hours |
| Weekly trend | Recent weekly change, limited to avoid extreme extrapolation |
| Request median | A fallback summary computed from the current input |
| Training prior | A fallback learned from the training set |

The model then gives weights to the candidates that can actually be computed.
A candidate that depends on missing data receives exactly zero weight. The
weighted average forms the main forecast, and a learned final adjustment is
kept within a fixed limit so that it cannot move arbitrarily far away.

See [How the method works](https://rudykon.github.io/WLCR-SEA_Predictor/guide/method/)
for the Entmax router, equations, and exact availability rules.

<a id="figures"></a>
## Figures from the Manuscript

These are the five figures used in the manuscript, rendered directly at
300 dpi. Figure 1 explains the usage setting; Figure 2 shows the method;
Figures 3–5 report accuracy, missing-data tests, and inspection results.

<p align="center">
  <a href="docs/images/Scene_Diagram.pdf">
    <img src="docs/images/paper_figure_scenario.png" alt="One cell's data moving from input preparation to forecast and calculation record" width="96%">
  </a>
</p>
<p align="center"><em>Figure 1 | How one cell's data is prepared, forecast, and recorded.</em></p>

<p align="center">
  <a href="docs/images/paper_figure_architecture.png">
    <img src="docs/images/paper_figure_architecture.png" alt="How WLCR-SEA builds and combines candidate forecasts" width="96%">
  </a>
</p>
<p align="center"><em>Figure 2 | How WLCR-SEA builds, filters, and combines candidate forecasts.</em></p>

<details>
<summary><strong>Open Figures 3–5: complete-data accuracy, missing-data tests, and calculation checks</strong></summary>

<br>

<p align="center">
  <a href="docs/images/paper_figure_clean_accuracy.png">
    <img src="docs/images/paper_figure_clean_accuracy.png" alt="Model comparison when historical data are complete" width="76%">
  </a>
</p>
<p align="center"><em>Figure 3 | Model comparison when historical data are complete.</em></p>

<p align="center">
  <a href="docs/images/paper_figure_missingness.png">
    <img src="docs/images/paper_figure_missingness.png" alt="Forecast errors after removing historical data in several patterns" width="96%">
  </a>
</p>
<p align="center"><em>Figure 4 | Forecast errors under several missing-data patterns.</em></p>

<p align="center">
  <a href="docs/images/paper_figure_auditability.png">
    <img src="docs/images/paper_figure_auditability.png" alt="Checks of candidate weights, deletion effects, and prediction limits" width="96%">
  </a>
</p>
<p align="center"><em>Figure 5 | Checks of candidate weights and their measured effects.</em></p>

</details>

<a id="quick-start"></a>
## Quick Start

~~~bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
~~~

For the full training, analysis, ablation, comparison, and result-checking workflow, see [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md).

## Live Demo

The [Hugging Face Space](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor)
lets you test how the method reacts when historical data is missing. Use the
built-in sample or upload a 336-row CSV, remove part of the history, and compare
the resulting 24-hour forecast with the historical patterns still available.
You can download both the forecast and a JSON record showing the candidate
values and weights.

**Important:** the repository does not include the trained A6 checkpoint used
for the paper's main results. The Space runs the real but simpler `A0_fixed`
baseline. Use it to understand the method, not to reproduce the numbers in the
paper. See [what the Demo runs](https://rudykon.github.io/WLCR-SEA_Predictor/deployment/hugging-face/).

<a id="validation"></a>
## Validation

Run the WLCR-SEA unit tests:

~~~bash
PYTHONPATH=. python3 -m unittest tests.test_wlcr_sea_model -v
~~~

<a id="resources"></a>
## Resources

| Resource | Link |
| --- | --- |
| Project website | [rudykon.github.io/WLCR-SEA_Predictor](https://rudykon.github.io/WLCR-SEA_Predictor/) |
| Live Demo | [Hugging Face Space](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor) |
| Data download | [Download ZIP](https://res-static.hc-cdn.cn/cloudbu-site/china/zh-cn/wuxian-gaoxiao2026/1780886490950118786.zip) |
| Source code | [github.com/rudykon/WLCR-SEA_Predictor](https://github.com/rudykon/WLCR-SEA_Predictor) |

<a id="license"></a>
## License

This repository is licensed under the Apache License 2.0; see <code>LICENSE</code>.

## Repository Layout

| Path | Purpose |
| --- | --- |
| <code>experiments/wlcr_sea_model.py</code> | WLCR-SEA experts, routing, bounded residual, losses, and metrics |
| <code>experiments/missingness_protocol.py</code> | Repeatable rules for removing historical data in experiments |
| <code>tests/test_wlcr_sea_model.py</code> | Focused unit tests for the public method |
| <code>demo/</code> | Gradio Demo and generated sample input |
| <code>docs/images/</code> | Five manuscript figures rendered for this README |
| <code>requirements.txt</code> | Research and Gradio runtime dependencies |
