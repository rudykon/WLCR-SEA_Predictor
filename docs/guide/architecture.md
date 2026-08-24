# System architecture

The system has five clear steps: prepare one cell's history, pass it to the
model, build candidate forecasts, combine the usable candidates, and return a
forecast with a calculation record.

<figure class="paper-figure">
  <a href="../../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_architecture.png" alt="WLCR-SEA architecture from prepared input through candidate weighting to forecast and calculation record" loading="lazy">
  </a>
  <figcaption>Manuscript Figure 2. The diagram shows how one input becomes candidate forecasts, a final prediction, and an inspection record.</figcaption>
</figure>

## Five components, five responsibilities

<div class="process-steps">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>Data gateway</h3>
      <p>The service verifies the cell and prepares 336 hours of data. A Boolean mask marks missing values. The cell ID is kept outside the model.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>Fixed model input</h3>
      <p>The model receives the four traffic series, their missing-value markers, and fixed information learned during training. It cannot fetch extra live traffic.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">03</span>
    <div>
      <h3>Candidate builder</h3>
      <p>For each future hour and indicator, the model creates eight candidate forecasts from known historical patterns and records which ones can be computed.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">04</span>
    <div>
      <h3>Weighting and limited adjustment</h3>
      <p>The model gives weights only to usable candidates. Their weighted average is the main forecast, followed by a final adjustment whose size is limited.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">05</span>
    <div>
      <h3>Forecast and calculation record</h3>
      <p>The response contains four 24-hour forecasts. It can also save the input hash, model version, candidate values, weights, adjustment, and range checks.</p>
    </div>
  </article>
</div>

## End-to-end data flow

| Step | Receives | Produces | Important rule |
| --- | --- | --- | --- |
| Gateway → model input | One cell's raw data | Ordered history + missing-value mask | Cell ID is not a prediction feature |
| Model input → candidates | One prepared input | Eight candidates per hour and indicator | Every candidate can be rebuilt from the input |
| Candidates → weighting | Candidate values and availability | Candidate weights | Missing references receive zero weight |
| Weighting → prediction | Weighted average | Average + limited adjustment | Adjustment cannot grow without limit |
| Prediction → users | Forecast and calculation fields | Planning input + saved record | Forecasting remains separate from control |

## What the forecasting service does—and does not do

This is only the forecasting part of a telecom system. The surrounding system
must still handle user permissions, encryption, data retention, logs, and
resource scheduling. Restricting what the model can read does not automatically
provide privacy, and the model does not directly control the network.

<div class="notice-card">
  <strong>This page explains the system flow.</strong> The Method page explains the exact calculation, and the Research pages report the results and limitations.
</div>

[Start with the business scenarios](problem.md){ .md-button }
[Inspect the forecasting method](method.md){ .md-button .md-button--primary }
[Review the research evidence](../research/evidence.md){ .md-button }
