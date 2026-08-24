---
hide:
  - toc
---

<section class="home-hero">
  <div class="hero-copy">
    <span class="hero-kicker">Cellular operations · Midnight horizon · Auditable evidence</span>
    <h1>One cell needs tomorrow's traffic plan. The scorer can read only <span class="gradient-text">this authorized request.</span></h1>
    <p class="hero-lead">
      After the final observed 23:00 hour, an operator may need a 24-hour demand
      signal for proactive radio-resource and service-capacity planning. In a
      compartmentalized edge domain, however, the scorer cannot quietly fetch
      live traffic from neighboring cells. WLCR-SEA turns one authorized
      336-hour history into a forecast and a replayable evidence trail.
    </p>
    <div class="hero-actions">
      <a class="hero-button primary" href="guide/problem/">Follow the operational story</a>
      <a class="hero-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">Try one request</a>
      <a class="hero-button" href="research/evidence/">Inspect the paper evidence</a>
    </div>
    <div class="hero-proof">
      <span>One target cell</span>
      <span>Authorized evidence only</span>
      <span>Forecast plus audit record</span>
    </div>
  </div>
  <figure class="hero-visual">
    <a href="images/paper_figure_scenario.png" target="_blank" rel="noopener">
      <img src="images/paper_figure_scenario.png" alt="Request-local cellular forecasting scenario from authorized ingress to auditable forecast" loading="eager" decoding="async">
    </a>
    <figcaption class="hero-caption">Manuscript Figure 1 · the ingress authorizes one request; the scorer stays inside its evidence boundary</figcaption>
  </figure>
</section>

<div class="metric-strip">
  <div class="metric"><strong>336 h</strong><span>authorized history</span></div>
  <div class="metric"><strong>24 h</strong><span>next-day horizon</span></div>
  <div class="metric"><strong>8</strong><span>named evidence experts</span></div>
  <div class="metric"><strong>4</strong><span>traffic indicators</span></div>
</div>

<span class="section-eyebrow">The operational story</span>

## From one authorized request to a forecast that can be replayed {: .section-title }

<p class="section-lead">The project begins with an operational constraint, not a model diagram. A downstream planner needs a short-horizon traffic signal, while the online prediction service must remain inside the evidence explicitly packaged for one cell.</p>

<div class="story-steps">
  <article class="story-step">
    <span class="story-number">01</span>
    <div>
      <h3>A forecast is requested</h3>
      <p>At a midnight forecast origin, the next 24 hours matter for proactive radio-resource and service-capacity planning. The prediction is decision support; scheduling and control remain downstream.</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">02</span>
    <div>
      <h3>The ingress seals the evidence</h3>
      <p>An identity-aware front end authorizes the target cell and materializes its ordered 336-hour history plus the authoritative observation mask. Identity is retained outside the model for routing and audit.</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">03</span>
    <div>
      <h3>The scorer explains its references</h3>
      <p>Yesterday, last week, two weeks ago, robust seasonal medians, a bounded trend and summary fallbacks become named candidates. Missing evidence is removed before routing, not hidden behind a numerical fill.</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">04</span>
    <div>
      <h3>The forecast leaves with a trace</h3>
      <p>The service returns four 24-hour series together with expert values, availability, routing mass, baseline, bounded correction and envelope checks that can be replayed from the same request.</p>
    </div>
  </article>
</div>

<p class="story-bridge"><a href="guide/problem/">Read the complete serving scenario →</a></p>

<span class="section-eyebrow">Where it fits</span>

## Situations in which this pattern is useful {: .section-title }

<p class="section-lead">WLCR-SEA is most relevant when the forecasting contract matters as much as the point estimate: the scorer has bounded evidence, telemetry may be incomplete, and someone may later need to explain what influenced the result.</p>

<div class="scenario-grid">
  <article class="scenario-card">
    <span class="scenario-tag">PLAN</span>
    <h3>Daily radio-resource planning</h3>
    <p>Provide a next-day demand signal for proactive resource planning while keeping the forecasting service separate from the downstream scheduler or controller.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">CAPACITY</span>
    <h3>Service-capacity outlook</h3>
    <p>Summarize the expected shape of four traffic indicators when teams need a compact cell-level view for the next operational horizon.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">OUTAGE</span>
    <h3>Incomplete telemetry</h3>
    <p>Remove unavailable seasonal references exactly and inspect how the forecast changes under random, block, recent-tail or asynchronous indicator loss.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">GOVERN</span>
    <h3>Audit and incident review</h3>
    <p>Replay a prediction from its request and model version, then examine which evidence was available, selected or structurally excluded.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">RESEARCH</span>
    <h3>Controlled forecasting studies</h3>
    <p>Compare clean accuracy, missingness robustness, request locality, audit properties and latency under one explicit serving contract.</p>
  </article>
  <article class="scenario-card scenario-card--boundary">
    <span class="scenario-tag">NOT A CLAIM</span>
    <h3>Not autonomous network control</h3>
    <p>The paper evaluates forecasting behavior, not business impact or closed-loop control. New regions, seasons, policies and sudden events require prospective validation.</p>
  </article>
</div>

<span class="section-eyebrow">Why WLCR-SEA</span>

## The evidence constraint changes the model shape {: .section-title }

<p class="section-lead">A graph predictor may be appropriate when authorized neighbor states are available. This project studies the different case in which every online value must come from the current request or a frozen global asset—and every surviving reference should remain visible.</p>

<figure class="paper-figure">
  <a href="images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="images/paper_figure_architecture.png" alt="WLCR-SEA structured seasonal expert-routing architecture" loading="lazy" decoding="async">
  </a>
  <figcaption>Manuscript Figure 2 · a finite expert interface, available-set routing, bounded correction and proposed semantic audit record.</figcaption>
</figure>

<div class="feature-grid">
  <article class="feature-card">
    <span class="feature-number">01 · REQUEST</span>
    <h3>Self-contained input</h3>
    <p>The scorer receives one ordered history and mask. It cannot query another cell, a topology table, a live feature store or a cross-request traffic cache.</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">02 · ROUTE</span>
    <h3>Named seasonal evidence</h3>
    <p>Eight candidates expose their value, availability, reliability and routing mass for every horizon and indicator; unavailable experts receive exactly zero mass.</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">03 · AUDIT</span>
    <h3>Bounded, replayable output</h3>
    <p>The learned correction stays within a finite log-space bound around the routed baseline, while the request and model version anchor later replay.</p>
  </article>
</div>

<p class="story-bridge"><a href="guide/architecture/">Walk through the complete system architecture →</a></p>

<span class="section-eyebrow">What the study found</span>

## Robustness and inspectability—with the trade-offs visible {: .section-title }

<p class="section-lead">The paper studies 527,760 cell-hour records from 736 cells. It does not claim universal accuracy leadership: DLinear has the lowest clean WAPE, while WLCR-SEA's strongest evidence is its structured-missingness and audit profile under the stated protocol.</p>

<figure class="paper-figure">
  <a href="images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="images/paper_figure_missingness.png" alt="Paper evidence comparing forecasting robustness under several missingness mechanisms" loading="lazy" decoding="async">
  </a>
  <figcaption>Manuscript Figure 4. Missingness results are conditional on the fixed masks and refits defined by the paper; they are not a cross-region guarantee.</figcaption>
</figure>

<div class="metric-strip">
  <div class="metric"><strong>0.1955</strong><span>WLCR-SEA clean WAPE</span></div>
  <div class="metric"><strong>0.1854</strong><span>lowest clean WAPE, DLinear</span></div>
  <div class="metric"><strong>0</strong><span>unavailable-expert mass in audits</span></div>
  <div class="metric"><strong>6.8 ms</strong><span>single-seed median CPU latency</span></div>
</div>

<div class="demo-cta">
  <div>
    <h2>Take one request through the same story</h2>
    <p>Load the bundled 336-hour example, remove telemetry, inspect the surviving experts and download the 24-hour forecast plus its audit record. The public lab runs the real A0 fixed method path—not the undistributed trained A6 checkpoint.</p>
  </div>
  <a class="md-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">Launch the guided audit lab</a>
</div>

<div class="notice-card">
  <strong>Interpretation matters.</strong> Entropy is not calibrated uncertainty in the reported study; the dataset covers one anonymous region for roughly one month; and request-local processing is an evidence boundary, not by itself a privacy guarantee. See <a href="research/limitations/">scope and limitations</a> before reusing the results.
</div>
