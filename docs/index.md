---
hide:
  - toc
---

<section class="project-hero">
  <div class="project-wordmark">
    <img class="wordmark-light" src="assets/brand/logo.svg" alt="WLCR-SEA Predictor">
    <img class="wordmark-dark" src="assets/brand/logo-dark.svg" alt="WLCR-SEA Predictor">
  </div>
  <h1>WLCR-SEA</h1>
  <p class="project-subtitle">Forecast one cell's next 24 hours from its own recent history</p>
  <p class="project-lead">Use one cell's previous 336 hours and four traffic indicators to forecast its next 24 hours. Unavailable historical references are hard-masked, and every request can export expert values, routing weights, residuals, and bound checks.</p>
  <div class="project-actions">
    <a class="md-button md-button--primary" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">Try the Demo</a>
    <a class="md-button" href="guide/method/">Method</a>
    <a class="md-button" href="research/evidence/">Results</a>
  </div>
</section>

## Research setting and method

<div class="setting-layout">
  <figure class="paper-figure">
    <a href="images/paper_figure_architecture.png" target="_blank" rel="noopener">
      <img src="images/paper_figure_architecture.png" alt="WLCR-SEA architecture from one-cell request to eight seasonal experts, hard-masked routing, bounded forecast, and audit record">
    </a>
    <figcaption>One cell's history and the published model checkpoints produce a forecast and audit record.</figcaption>
  </figure>
  <div>
    <p>WLCR-SEA is designed for forecasting when a request may contain missing telemetry and inference must not query neighboring cells. The model uses the request plus frozen assets only.</p>
    <ul class="fact-list">
      <li><strong>One cell</strong><span>No live cross-cell lookup</span></li>
      <li><strong>336 h × 4</strong><span>History and observation mask</span></li>
      <li><strong>Eight experts</strong><span>Daily, weekly, median, trend, local, and prior</span></li>
      <li><strong>24 h × 4</strong><span>Five-model ensemble forecast</span></li>
      <li><strong>Audit record</strong><span>Values, availability, weights, residuals, and checks</span></li>
    </ul>
  </div>
</div>

## Main findings

| Question | Registered result |
| --- | --- |
| Complete history | DLinear has lower **macro-indicator WAPE**: 0.1854 versus 0.1955 for the WLCR-SEA five-model ensemble. |
| Severe missingness | WLCR-SEA has lower macro-indicator WAPE in all 9 prespecified comparisons against DLinear-Aug, PatchTST-Aug, and GRU-D in the study's severe settings. |
| Auditability | Unavailable-expert weight: 0. Reported bounded-envelope violations: 0. |
| CPU cost | Five-member ensemble: 34.705 ms median / 38.684 ms P99, batch 1, one CPU thread; 148.8 KiB of model assets. |

[Read the full protocol, intervals, and results →](research/evidence.md)

## Scope

<div class="scope-box">
  <ul>
    <li>Evaluated on one anonymous region and roughly one month of data.</li>
    <li>Routing entropy is not a calibrated uncertainty estimate.</li>
    <li>Request-local inference is not, by itself, a privacy or security guarantee.</li>
    <li>Network-control actions and business outcomes were not evaluated.</li>
  </ul>
</div>
