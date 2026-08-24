---
hide:
  - toc
---

<section class="home-hero">
  <div class="hero-copy">
    <span class="hero-kicker">Cell traffic · 24-hour forecast · Traceable output</span>
    <h1>Forecast one cell—<span class="gradient-text">without reading another.</span></h1>
    <p class="hero-lead">WLCR-SEA turns 14 days of one cell's traffic into a 24-hour forecast. It ignores missing references and records the values and weights behind each result.</p>
    <div class="hero-actions">
      <a class="hero-button primary" href="guide/problem/">Use cases</a>
      <a class="hero-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">Demo</a>
      <a class="hero-button" href="research/evidence/">Results</a>
    </div>
    <div class="hero-proof">
      <span>One cell</span>
      <span>14 days → 24 hours</span>
      <span>Traceable</span>
    </div>
  </div>
  <figure class="hero-visual hero-logo-visual">
    <img class="hero-brand-logo hero-brand-logo--light" src="assets/brand/logo.svg" alt="WLCR-SEA Predictor project logo" loading="eager" decoding="async">
    <img class="hero-brand-logo hero-brand-logo--dark" src="assets/brand/logo-dark.svg" alt="WLCR-SEA Predictor project logo for dark mode" loading="eager" decoding="async">
  </figure>
</section>

<div class="metric-strip">
  <div class="metric"><strong>336 h</strong><span>input history</span></div>
  <div class="metric"><strong>24 h</strong><span>forecast period</span></div>
  <div class="metric"><strong>8</strong><span>historical candidates</span></div>
  <div class="metric"><strong>4</strong><span>traffic measures</span></div>
</div>

<span class="section-eyebrow">Problem</span>

## Tomorrow's traffic, limited data {: .section-title }

<p class="section-lead">A planner needs tomorrow's demand. The model receives one cell's history—nothing else.</p>

<div class="process-steps">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>Need</h3>
      <p>Estimate the next 24 hours for four traffic indicators.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>Input</h3>
      <p>Pack 336 hourly rows and a missing-value mask.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">03</span>
    <div>
      <h3>Forecast</h3>
      <p>Build eight candidates. Drop any that need missing data.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">04</span>
    <div>
      <h3>Output</h3>
      <p>Return the forecast, weights, adjustment, and checks.</p>
    </div>
  </article>
</div>

<p class="section-link"><a href="guide/problem/">Use cases →</a></p>

<span class="section-eyebrow">Use cases</span>

## Where it fits {: .section-title }

<p class="section-lead">Use WLCR-SEA when data access is narrow, measurements may be missing, or forecasts need review.</p>

<div class="scenario-grid">
  <article class="scenario-card">
    <span class="scenario-tag">PLAN</span>
    <h3>Planning</h3>
    <p>Estimate demand before the next day's schedule is set.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">MISSING</span>
    <h3>Missing data</h3>
    <p>Ignore candidates that depend on unavailable values.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">REVIEW</span>
    <h3>Review</h3>
    <p>Replay a result and inspect its candidates and weights.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">BOUNDARY</span>
    <h3>Forecasting only</h3>
    <p>Scheduling and network control remain separate.</p>
  </article>
</div>

<span class="section-eyebrow">Method</span>

## Eight candidates. One forecast. {: .section-title }

<p class="section-lead">The model builds candidates from daily and weekly patterns, removes unavailable ones, then combines the rest.</p>

<figure class="paper-figure">
  <a href="images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="images/paper_figure_architecture.png" alt="How WLCR-SEA builds and combines candidate forecasts" loading="lazy" decoding="async">
  </a>
  <figcaption>Eight candidates are filtered, weighted, and adjusted within a fixed limit.</figcaption>
</figure>

<div class="feature-grid">
  <article class="feature-card">
    <span class="feature-number">01 · INPUT</span>
    <h3>One cell</h3>
    <p>Only the current history and mask enter the model.</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">02 · ROUTE</span>
    <h3>Eight candidates</h3>
    <p>Each candidate has a value, availability flag, and weight.</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">03 · OUTPUT</span>
    <h3>Bounded result</h3>
    <p>The final adjustment is limited and the calculation can be saved.</p>
  </article>
</div>

<p class="section-link"><a href="guide/architecture/">Architecture →</a></p>

<span class="section-eyebrow">Results</span>

## Stronger under tested outages {: .section-title }

<p class="section-lead">On 527,760 records from 736 cells, DLinear leads with complete history. WLCR-SEA leads in the study's severe missing-data tests.</p>

<figure class="paper-figure">
  <a href="images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="images/paper_figure_missingness.png" alt="Forecast errors after removing historical data in several patterns" loading="lazy" decoding="async">
  </a>
  <figcaption>Errors under the study's fixed missing-data tests.</figcaption>
</figure>

<div class="metric-strip">
  <div class="metric"><strong>0.1955</strong><span>WLCR-SEA WAPE, complete data</span></div>
  <div class="metric"><strong>0.1854</strong><span>DLinear WAPE, complete data</span></div>
  <div class="metric"><strong>0</strong><span>weight given to missing references</span></div>
  <div class="metric"><strong>6.8 ms</strong><span>single-model median CPU latency</span></div>
</div>

<div class="demo-cta">
  <div>
    <h2>Try it</h2>
    <p>Load the sample, remove history, and inspect the forecast. The Demo uses the A0 fixed baseline; the trained A6 checkpoint is not public.</p>
  </div>
  <a class="md-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">Open Demo</a>
</div>

<div class="notice-card">
  <strong>Limits:</strong> one region, about one month; no reliable uncertainty score; no privacy guarantee. <a href="research/limitations/">Read more</a>.
</div>
