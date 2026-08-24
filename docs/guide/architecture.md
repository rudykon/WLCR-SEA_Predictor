# System architecture

WLCR-SEA turns the request-local evidence rule into a five-part serving path.
The architecture defines **where data may enter, how it moves, and what must be
recorded**. The forecasting method then operates inside that boundary.

<figure class="paper-figure">
  <a href="../../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_architecture.png" alt="WLCR-SEA system architecture from sealed request through expert construction and masked routing to forecast and audit record" loading="lazy">
  </a>
  <figcaption>Manuscript Figure 2. The online path keeps request assembly, expert construction, routing, correction and audit output explicit.</figcaption>
</figure>

## Five components, five responsibilities

<div class="story-steps">
  <article class="story-step">
    <span class="story-number">01</span>
    <div>
      <h3>Identity-aware ingress</h3>
      <p>The serving layer authenticates the source and resolves operational identity. It assembles one 336-hour history and its observation mask, then keeps the cell ID outside the forecasting feature path.</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">02</span>
    <div>
      <h3>Sealed request boundary</h3>
      <p>The scorer receives only the ordered four-indicator tensor, the authoritative mask and frozen global assets. It cannot reach into another request, another cell or a live topology service.</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">03</span>
    <div>
      <h3>Seasonal expert builder</h3>
      <p>For every horizon and indicator, the builder materializes eight named candidates plus their availability and reliability. Removed evidence is recomputed rather than hidden behind a numerical fill.</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">04</span>
    <div>
      <h3>Available-set router and bounded predictor</h3>
      <p>The router assigns mass only across surviving experts. Their convex combination forms the baseline; a bounded residual can refine it without leaving the finite expert envelope by more than the configured margin.</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">05</span>
    <div>
      <h3>Forecast and audit output</h3>
      <p>The response returns four 24-hour forecasts. A semantic audit record can retain the request hash, model version, expert state, routing weights, baseline, residual and envelope checks for replay.</p>
    </div>
  </article>
</div>

## End-to-end data flow

| Boundary | Receives | Produces | Enforced property |
| --- | --- | --- | --- |
| Ingress → request | Authorized raw telemetry | Ordered history + mask | Identity is not a model feature |
| Request → experts | One sealed request | Eight candidates per horizon and indicator | Every candidate is rebuildable locally |
| Experts → router | Values, availability, reliability | Sparse routing weights | Unavailable experts receive exactly zero mass |
| Router → predictor | Routed baseline | Baseline + bounded residual | Prediction remains near available evidence |
| Predictor → consumers | Forecast and trace fields | Planning input + replay record | Forecasting stays separate from downstream control |

## Trust and deployment boundary

The architecture is deliberately narrower than a complete telecom platform.
Authentication, authorization, encryption, retention and downstream scheduling
belong to the surrounding system. WLCR-SEA specifies the forecasting evidence
boundary and the proposed audit semantics; it does not turn request-local
processing into a privacy guarantee or a closed-loop controller.

<div class="notice-card">
  <strong>Architecture is not the algorithm.</strong> This page explains components, boundaries and data flow. The Method section explains how the eight experts, available-set routing and bounded residual calculate a forecast; Research evaluates the resulting behavior and limitations.
</div>

[Start with the business scenarios](problem.md){ .md-button }
[Inspect the forecasting method](method.md){ .md-button .md-button--primary }
[Review the research evidence](../research/evidence.md){ .md-button }
