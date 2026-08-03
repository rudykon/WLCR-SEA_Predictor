<h1 align="center">WLCR-SEA Predictor</h1>

<p align="center">
  <strong>Request-local cellular traffic forecasting with structured seasonal expert routing</strong><br>
  Open-source implementation of the WLCR-SEA method, its analysis/ablation baselines, and validation tests.
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README_CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#validation"><img src="https://img.shields.io/badge/Validation-unittest-2CA02C?style=flat-square" alt="Unit tests"></a>
  <a href="#scope"><img src="https://img.shields.io/badge/Release-WLCR--SEA%20only-6A5ACD?style=flat-square" alt="WLCR-SEA-only release"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#method">Method</a> ·
  <a href="#figures">Figures</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#validation">Validation</a> ·
  <a href="#resources">Resources</a>
</p>

<a id="overview"></a>
## Overview

WLCR-SEA (Window-Local Context Representation with Seasonal Expert Attention)
is a request-local traffic forecasting method. For each request, it uses an
ordered 336-hour history and an observation mask to produce a 24-hour forecast
for four traffic indicators. The implementation exposes seasonal evidence,
expert availability, routing weights, and a bounded correction.

<a id="method"></a>
## Method

For every horizon and indicator, WLCR-SEA constructs eight named experts from
the supplied request. Unavailable evidence is excluded before routing.

| Expert | Evidence role |
| --- | --- |
| Last day | Same hour one day earlier |
| Last week | Same hour seven days earlier |
| Two-week lag | Same hour fourteen days earlier |
| Same-hour median, 7 d | Robust seven-day seasonal summary |
| Same-hour median, 14 d | Robust fourteen-day seasonal summary |
| Bounded weekly trend | Conservative correction from weekly change |
| Window-local median | Request-level fallback summary |
| Frozen training prior | Always-available population fallback |

A horizon-conditioned Entmax router allocates mass across available experts.
The bounded residual adjusts the routed baseline while retaining a finite,
inspectable prediction envelope.

<a id="figures"></a>
## Figures from the Manuscript

The following five PNG files are direct 300 dpi renders of the five figures
referenced by the manuscript. No unrelated illustrations are included.

<p align="center">
  <a href="docs/images/paper_figure_scenario.png">
    <img src="docs/images/paper_figure_scenario.png" alt="Conceptual request-local serving scenario" width="96%">
  </a>
</p>
<p align="center"><em>Figure 1 | Conceptual request-local serving scenario.</em></p>

<p align="center">
  <a href="docs/images/paper_figure_architecture.png">
    <img src="docs/images/paper_figure_architecture.png" alt="WLCR-SEA structured seasonal expert-routing architecture" width="96%">
  </a>
</p>
<p align="center"><em>Figure 2 | WLCR-SEA as an instance of structured seasonal expert routing.</em></p>

<details>
<summary><strong>Open Figures 3–5: clean accuracy, missingness, and auditability</strong></summary>

<br>

<p align="center">
  <a href="docs/images/paper_figure_clean_accuracy.png">
    <img src="docs/images/paper_figure_clean_accuracy.png" alt="Routing hierarchy on the clean holdout" width="76%">
  </a>
</p>
<p align="center"><em>Figure 3 | Routing hierarchy on the clean holdout.</em></p>

<p align="center">
  <a href="docs/images/paper_figure_missingness.png">
    <img src="docs/images/paper_figure_missingness.png" alt="Missingness robustness" width="96%">
  </a>
</p>
<p align="center"><em>Figure 4 | Missingness robustness.</em></p>

<p align="center">
  <a href="docs/images/paper_figure_auditability.png">
    <img src="docs/images/paper_figure_auditability.png" alt="Auditability evidence" width="96%">
  </a>
</p>
<p align="center"><em>Figure 5 | Auditability evidence.</em></p>

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

For the full training, analysis, ablation, comparison, and audit workflow, see [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md).

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
| Data download | [Download ZIP](https://res-static.hc-cdn.cn/cloudbu-site/china/zh-cn/wuxian-gaoxiao2026/1780886490950118786.zip) |
| Source code | [github.com/rudykon/WLCR-SEA_Predictor](https://github.com/rudykon/WLCR-SEA_Predictor) |

<a id="license"></a>
## License

This repository is licensed under the Apache License 2.0; see <code>LICENSE</code>.

## Repository Layout

| Path | Purpose |
| --- | --- |
| <code>experiments/wlcr_sea_model.py</code> | WLCR-SEA experts, routing, bounded residual, losses, and metrics |
| <code>experiments/missingness_protocol.py</code> | Deterministic missing-telemetry protocol used by WLCR-SEA |
| <code>tests/test_wlcr_sea_model.py</code> | Focused unit tests for the public method |
| <code>docs/images/</code> | Five manuscript figures rendered for this README |
| <code>requirements.txt</code> | Minimal Python dependencies |
