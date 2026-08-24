# Problem and use cases

WLCR-SEA predicts one cell's next 24 hours from that cell's history alone. It is built for limited access, missing data, and reviewable results.

<figure class="paper-figure">
  <a href="../../images/paper_figure_scenario.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_scenario.png" alt="One cell's data moving from input preparation to forecast and calculation record" loading="lazy">
  </a>
  <figcaption>One prepared cell history enters the model. Other cells stay out.</figcaption>
</figure>

## One request

<div class="process-steps process-steps--compact">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>Request</h3>
      <p>Predict the next day, 00:00–23:00.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>Prepare</h3>
      <p>Collect 336 hours, four indicators, and a missing-value mask.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">03</span>
    <div>
      <h3>Predict</h3>
      <p>Use the prepared input and one fixed model version.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">04</span>
    <div>
      <h3>Return</h3>
      <p>Output four forecasts plus candidates, weights, and checks.</p>
    </div>
  </article>
</div>

<div class="notice-card">
  <strong>Scope:</strong> forecasting only. Business impact and network control were not tested.
</div>

## Use cases

<div class="scenario-grid scenario-grid--compact">
  <article class="scenario-card">
    <span class="scenario-tag">PLAN</span>
    <h3>Planning</h3>
    <p>Estimate demand before scheduling.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">MISSING</span>
    <h3>Missing data</h3>
    <p>Keep placeholders out of the calculation.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">REVIEW</span>
    <h3>Review</h3>
    <p>Replay a forecast and inspect its weights.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">COMPARE</span>
    <h3>Research</h3>
    <p>Compare models under the same outages.</p>
  </article>
</div>

## When to use another method

- **Neighbor data is reliable:** consider a graph or multi-cell model.
- **External events matter:** add weather, events, mobility, or holidays.
- **Uncertainty is required:** routing entropy is not reliable here.
- **Forecasts control the network:** run safety and field tests first.

## Model input

Input: 336 hours × four indicators, plus a present/missing flag for every value. The model applies `log1p`. Cell identity is not a feature. Output: the next 24 hours for the same four indicators.

## What the model may and may not use

| The model may use | The model may not use |
| --- | --- |
| Values and missing-value markers in the current input | Live traffic from another cell |
| Fixed statistics learned during training | Metadata looked up by cell ID |
| One versioned global checkpoint | Topology or neighbor tables |
| Fixed method configuration | External feature stores |
| Cell ID outside the model for access control and records | Traffic cached from another request |

This is **request-local forecasting**: after input preparation, the model cannot fetch more traffic data.

## Why WLCR-SEA fits

WLCR-SEA builds eight historical candidates, drops unavailable ones, combines the rest, and limits the final adjustment. The same input and model version reproduce the same calculation.

[Architecture](architecture.md){ .md-button .md-button--primary }
[Method](method.md){ .md-button }
[Demo](../deployment/hugging-face.md){ .md-button }

!!! note "Limiting model input is not the same as protecting data"
    The surrounding system still needs access control, encryption, safe logs, and retention rules.
