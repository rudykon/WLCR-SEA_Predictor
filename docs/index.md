---
hide:
  - toc
---

<section class="home-hero">
  <div class="hero-copy">
    <span class="hero-kicker">Cellular traffic forecasting · Next 24 hours · Inspectable results</span>
    <h1>Forecast tomorrow's traffic for one cell—<span class="gradient-text">without reading other cells.</span></h1>
    <p class="hero-lead">
      WLCR-SEA reads the previous 14 days of four traffic indicators and predicts
      the next 24 hours. It is designed for settings where the model can use only
      the current cell's submitted data. The result includes both the forecast
      and a record of the historical patterns used to produce it.
    </p>
    <div class="hero-actions">
      <a class="hero-button primary" href="guide/problem/">See when to use it</a>
      <a class="hero-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">Try the Live Demo</a>
      <a class="hero-button" href="research/evidence/">See the results</a>
    </div>
    <div class="hero-proof">
      <span>One cell at a time</span>
      <span>14 days in, 24 hours out</span>
      <span>Forecast plus inspection record</span>
    </div>
  </div>
  <figure class="hero-visual">
    <a href="images/paper_figure_scenario.png" target="_blank" rel="noopener">
      <img src="images/paper_figure_scenario.png" alt="One cell's data moving from input preparation to forecast and calculation record" loading="eager" decoding="async">
    </a>
    <figcaption class="hero-caption">Manuscript Figure 1 · a gateway prepares one cell's data; the model cannot fetch traffic from other cells</figcaption>
  </figure>
</section>

<div class="metric-strip">
  <div class="metric"><strong>336 h</strong><span>input history</span></div>
  <div class="metric"><strong>24 h</strong><span>forecast period</span></div>
  <div class="metric"><strong>8</strong><span>historical candidates</span></div>
  <div class="metric"><strong>4</strong><span>traffic measures</span></div>
</div>

<span class="section-eyebrow">The problem</span>

## Useful forecasts with a strict data limit {: .section-title }

<p class="section-lead">A network planner may need tomorrow's traffic for one cell, but the prediction service may not be allowed to query neighboring cells or a live feature store. The model must therefore make the best use of the history included with that one request.</p>

<div class="process-steps">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>A next-day forecast is needed</h3>
      <p>At midnight, a planner asks for the next 24 hours of traffic. The forecast supports planning; it does not directly control the network.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>The gateway prepares one input</h3>
      <p>The service verifies the cell and packages 336 hourly rows. A Boolean mask records which values are truly present. The cell ID is not used as a prediction feature.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">03</span>
    <div>
      <h3>The model compares historical patterns</h3>
      <p>It builds eight candidate forecasts from yesterday, last week, two weeks ago, seasonal summaries and fallback values. Candidates that depend on missing data are removed.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">04</span>
    <div>
      <h3>The result includes its calculation record</h3>
      <p>The service returns four 24-hour forecasts and records the candidate values, their weights, the limited final adjustment and range checks.</p>
    </div>
  </article>
</div>

<p class="section-link"><a href="guide/problem/">See the use cases and boundaries →</a></p>

<span class="section-eyebrow">Where it fits</span>

## Situations in which this pattern is useful {: .section-title }

<p class="section-lead">WLCR-SEA is most useful when the model has limited access to data, historical records may be incomplete, and someone may later need to check why a forecast changed.</p>

<div class="scenario-grid">
  <article class="scenario-card">
    <span class="scenario-tag">PLAN</span>
    <h3>Daily radio-resource planning</h3>
    <p>Estimate next-day demand before allocating resources, while leaving scheduling and control to a separate downstream system.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">CAPACITY</span>
    <h3>Service-capacity outlook</h3>
    <p>Give teams a simple view of how four cell-level traffic measures may change over the next day.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">OUTAGE</span>
    <h3>Incomplete measurements</h3>
    <p>Ignore historical references that are no longer available and measure how different types of data loss affect the forecast.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">REVIEW</span>
    <h3>Review and incident analysis</h3>
    <p>Re-run a prediction from the same input and model version, then check which historical references were used or excluded.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">RESEARCH</span>
    <h3>Controlled forecasting studies</h3>
    <p>Compare complete-data accuracy, performance after data removal, input restrictions, calculation checks, and speed under the same test rules.</p>
  </article>
  <article class="scenario-card scenario-card--boundary">
    <span class="scenario-tag">NOT A CLAIM</span>
    <h3>Not autonomous network control</h3>
    <p>The paper evaluates forecasting behavior, not business impact or closed-loop control. New regions, seasons, policies and sudden events require prospective validation.</p>
  </article>
</div>

<span class="section-eyebrow">Why WLCR-SEA</span>

## How WLCR-SEA works within the data limit {: .section-title }

<p class="section-lead">If reliable neighbor-cell data are available, a graph model may be a better choice. WLCR-SEA addresses the narrower case where each prediction must use only the current input and fixed information learned during training.</p>

<figure class="paper-figure">
  <a href="images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="images/paper_figure_architecture.png" alt="How WLCR-SEA builds and combines candidate forecasts" loading="lazy" decoding="async">
  </a>
  <figcaption>Manuscript Figure 2 · eight historical candidates are filtered, weighted, and followed by a limited final adjustment.</figcaption>
</figure>

<div class="feature-grid">
  <article class="feature-card">
    <span class="feature-number">01 · REQUEST</span>
    <h3>One self-contained input</h3>
    <p>The model receives one cell's ordered history and missing-value mask. It cannot fetch data from another cell or an external live data service.</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">02 · ROUTE</span>
    <h3>Eight clear candidate forecasts</h3>
    <p>Each candidate is based on a known historical pattern. The model records its value, whether it can be used, and the weight it receives.</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">03 · RECORD</span>
    <h3>A limited final adjustment</h3>
    <p>The model may adjust the weighted average, but only within a fixed limit. The saved input and model version support later re-runs.</p>
  </article>
</div>

<p class="section-link"><a href="guide/architecture/">Open the complete system architecture →</a></p>

<span class="section-eyebrow">What the study found</span>

## What the experiments show—and what they do not {: .section-title }

<p class="section-lead">The paper uses 527,760 hourly records from 736 cells. DLinear is more accurate when the history is complete. WLCR-SEA's main strength appears when blocks of historical data are missing and when the calculation needs to be inspected.</p>

<figure class="paper-figure">
  <a href="images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="images/paper_figure_missingness.png" alt="Forecast errors after removing historical data in several patterns" loading="lazy" decoding="async">
  </a>
  <figcaption>Manuscript Figure 4. The reported gains apply to the fixed data-removal patterns and training runs used in the paper; they are not a guarantee for other regions.</figcaption>
</figure>

<div class="metric-strip">
  <div class="metric"><strong>0.1955</strong><span>WLCR-SEA WAPE, complete data</span></div>
  <div class="metric"><strong>0.1854</strong><span>DLinear WAPE, complete data</span></div>
  <div class="metric"><strong>0</strong><span>weight given to missing references</span></div>
  <div class="metric"><strong>6.8 ms</strong><span>single-seed median CPU latency</span></div>
</div>

<div class="demo-cta">
  <div>
    <h2>Try the method with one sample</h2>
    <p>Load the built-in 336-hour example, remove part of the history, and see which candidate forecasts remain. You can download the 24-hour forecast and its calculation record. The Demo uses the simpler A0 fixed baseline, not the unavailable trained A6 checkpoint.</p>
  </div>
  <a class="md-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">Launch the Live Demo</a>
</div>

<div class="notice-card">
  <strong>Read the limits before deployment.</strong> The data cover one anonymous region for about one month. The reported routing entropy is not a reliable uncertainty score, and limiting the model's input does not by itself guarantee privacy. See <a href="research/limitations/">the full limitations</a>.
</div>
