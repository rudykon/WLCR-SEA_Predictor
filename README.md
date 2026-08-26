<p align="center">
  <img src="docs/assets/brand/logo-horizontal.svg" width="540" alt="WLCR-SEA Predictor logo">
</p>

<p align="center">
  <strong>Forecast one cell's next 24 hours from its own recent history, even when some telemetry is missing.</strong>
</p>

<p align="center">
  <a href="https://rudykon.github.io/WLCR-SEA_Predictor/">Website</a> ·
  <a href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor">Live Demo</a> ·
  <a href="https://huggingface.co/config-h/WLCR-SEA-Predictor">Model Weights</a> ·
  <a href="README_CN.md">中文</a>
</p>

WLCR-SEA reads one cell's previous **336 hours × 4 traffic indicators** and
forecasts its next **24 hours × 4 indicators**. Inference uses only the current
cell's data and the published model weights; it does not fetch live traffic
from other cells. The public Demo runs a five-model ensemble built from the
same released checkpoints used for the reported results.

## Why WLCR-SEA

- **Uses one cell's data.** Each prediction is a sealed one-cell computation. Cell
  identity is retained for validation and replay, but it is not used to query
  another live data source.
- **Handles missing readings.** Eight seasonal experts represent daily, weekly,
  robust-median, trend, request-level, and frozen-prior patterns. If an expert
  depends on unavailable history, hard masking gives it exactly zero routing
  weight.
- **Shows how the forecast was built.** A request can export expert values,
  availability, routing weights, bounded residuals, checkpoint identities, and
  range checks. These records explain the model's internal allocation; they
  are not causal explanations or uncertainty estimates.

This is a research implementation for constrained data access and missing
telemetry. It is not presented as a universally best forecaster or as a privacy
system.

## Method

<p align="center">
  <a href="docs/images/paper_figure_architecture.png">
    <img src="docs/images/paper_figure_architecture.png" width="94%" alt="WLCR-SEA architecture: eight seasonal experts, hard availability masking, sparse routing, bounded residual, forecast, and audit record">
  </a>
</p>

First, WLCR-SEA constructs eight seasonal experts from the 336-hour request and
each checkpoint's frozen training prior. Second, reliability-aware Entmax
routing operates only on the available expert set, so unusable experts remain
at zero weight. Finally, a bounded residual produces the 24-hour forecast while
preserving a checkable envelope. The five trained models are combined by averaging
their predictions in linear traffic space.

The full expert definitions, availability rules, and equations are on the
[Method page](https://rudykon.github.io/WLCR-SEA_Predictor/guide/method/).

## Evidence

| Question | Registered finding |
| --- | --- |
| Complete history | DLinear has lower macro-indicator WAPE in the reported comparison: **0.1854** versus **0.1955** for the five-member WLCR-SEA ensemble. |
| Severe missingness | WLCR-SEA has lower macro-indicator WAPE in all **9 prespecified comparisons** against DLinear-Aug, PatchTST-Aug, and GRU-D under the study's severe missingness settings. |
| Audit checks | Unavailable-expert weight is **0** and reported bounded-envelope violations are **0**. |
| CPU inference | The five-member ensemble reports **34.705 ms median / 38.684 ms P99** at batch 1 with one CPU thread; model assets total **148.8 KiB**. |

These results come from one anonymous region and roughly one month of data.
They should not be treated as deployment guarantees. See [Results and
limitations](https://rudykon.github.io/WLCR-SEA_Predictor/research/evidence/)
for the protocol, confidence intervals, missingness patterns, and scope.

Metric labels are intentionally explicit. **Macro-indicator WAPE** computes
WAPE separately for the four indicators and averages those values;
**macro-cell WAPE** averages cell-level values, while **pooled WAPE** combines
the numerator and denominator over all evaluated points. Results from these
different aggregation schemes are not silently compared.

## Run

```bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor
python -m pip install -r requirements-demo.txt
python demo/app.py
```

The app verifies and caches the five public model checkpoints, automatically runs
a synthetic 336-hour sample, and exports both a forecast CSV and a versioned
audit JSON. Public uploads are limited to 5 MB; use a local deployment for
sensitive operator data.

## Reproduce, cite, and license

- **Reproduce:** follow [`REPRODUCTION.md`](REPRODUCTION.md) for inputs,
  installation, public-checkpoint verification, tests, training, and evidence
  generation.
- **Cite:** until formal publication metadata is available, cite the repository
  commit and the pinned [Hugging Face model
  revision](https://huggingface.co/config-h/WLCR-SEA-Predictor).
- **License:** Apache License 2.0; see [`LICENSE`](LICENSE).
