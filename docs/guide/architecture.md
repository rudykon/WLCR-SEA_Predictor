# System architecture

Five steps turn one cell's history into a forecast and calculation record.

<figure class="paper-figure">
  <a href="../../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_architecture.png" alt="WLCR-SEA architecture from prepared input through candidate weighting to forecast and calculation record" loading="lazy">
  </a>
  <figcaption>From prepared input to candidates, forecast, and record.</figcaption>
</figure>

## Components

<div class="process-steps">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>Data gateway</h3>
      <p>Prepare 336 hours and a missing-value mask. Keep cell ID outside the model.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>Model input</h3>
      <p>Pass four traffic series, masks, and fixed training data.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">03</span>
    <div>
      <h3>Candidates</h3>
      <p>Build eight candidates for each hour and indicator.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">04</span>
    <div>
      <h3>Routing</h3>
      <p>Weight usable candidates, then apply a bounded adjustment.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">05</span>
    <div>
      <h3>Output</h3>
      <p>Return four 24-hour forecasts and their calculation fields.</p>
    </div>
  </article>
</div>

## Data flow

| Step | Receives | Produces | Important rule |
| --- | --- | --- | --- |
| Gateway → model input | One cell's raw data | Ordered history + missing-value mask | Cell ID is not a prediction feature |
| Model input → candidates | One prepared input | Eight candidates per hour and indicator | Every candidate can be rebuilt from the input |
| Candidates → weighting | Candidate values and availability | Candidate weights | Missing references receive zero weight |
| Weighting → prediction | Weighted average | Average + limited adjustment | Adjustment cannot grow without limit |
| Prediction → users | Forecast and calculation fields | Planning input + saved record | Forecasting remains separate from control |

## Boundary

This is a forecasting module, not a network controller. Permissions, encryption, retention, logging, and scheduling belong to the surrounding system.

<div class="notice-card">
  <strong>Next:</strong> see Method for the calculation and Results for the evidence.
</div>

[Use cases](problem.md){ .md-button }
[Method](method.md){ .md-button .md-button--primary }
[Results](../research/evidence.md){ .md-button }
