---
hide:
  - toc
---

<section class="home-hero">
  <div class="hero-copy">
    <span class="hero-kicker">Open source · Request-local · Auditable</span>
    <h1>Forecast cellular traffic from <span class="gradient-text">authorized evidence alone.</span></h1>
    <p class="hero-lead">
      WLCR-SEA turns one sealed 336-hour cell history into a 24-hour,
      four-indicator forecast through named seasonal experts, exact
      availability masking, sparse routing, and a bounded correction.
    </p>
    <div class="hero-actions">
      <a class="hero-button primary" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">Open the audit lab</a>
      <a class="hero-button" href="getting-started/quickstart/">Run the code</a>
      <a class="hero-button" href="research/evidence/">Read the evidence</a>
    </div>
    <div class="hero-proof">
      <span>Finite expert interface</span>
      <span>Hard missing-evidence mask</span>
      <span>Replayable request record</span>
    </div>
  </div>
  <figure class="hero-visual">
    <a href="images/paper_figure_architecture.png" target="_blank" rel="noopener">
      <img src="images/paper_figure_architecture.png" alt="WLCR-SEA structured seasonal expert-routing architecture" loading="eager" decoding="async">
    </a>
    <figcaption class="hero-caption">Manuscript Figure 2 · select to inspect the original 4,000 × 2,250 render</figcaption>
  </figure>
</section>

<div class="metric-strip">
  <div class="metric"><strong>336 h</strong><span>sealed request history</span></div>
  <div class="metric"><strong>24 h</strong><span>forecast horizon</span></div>
  <div class="metric"><strong>8</strong><span>named evidence experts</span></div>
  <div class="metric"><strong>4</strong><span>traffic indicators</span></div>
</div>

<span class="section-eyebrow">Method boundary</span>

## A prediction path built to be inspected {: .section-title}

<p class="section-lead">The scorer cannot query another cell, a topology table, or a live feature store. Every online value must come from the current request or from a frozen, globally shared asset.</p>

<div class="feature-grid">
  <article class="feature-card">
    <span class="feature-number">01 · REQUEST</span>
    <h3>Self-contained input</h3>
    <p>An identity-aware ingress authorizes and materializes one ordered history plus its observation mask. Cell identity is not a forecasting feature.</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">02 · ROUTE</span>
    <h3>Structured evidence</h3>
    <p>Eight seasonal and summary experts expose their value, availability, reliability, and routing mass for every horizon and indicator.</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">03 · AUDIT</span>
    <h3>Bounded output</h3>
    <p>Unavailable experts receive exactly zero mass, while the learned residual remains inside a finite log-space bound around the routed baseline.</p>
  </article>
</div>

<span class="section-eyebrow">Paper evidence</span>

## Robustness and inspectability—with the trade-offs visible {: .section-title}

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
    <h2>Remove evidence and inspect what changes</h2>
    <p>The public lab runs the repository's real expert builder, missingness protocol, hard mask, and parameter-free A0 fixed mixture. The trained paper checkpoint is not distributed, so the demo never presents itself as A6 inference.</p>
  </div>
  <a class="md-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">Launch on Hugging Face</a>
</div>

<div class="notice-card">
  <strong>Interpretation matters.</strong> Entropy is not calibrated uncertainty in the reported study; the dataset covers one anonymous region for roughly one month; and request-local processing is an evidence boundary, not by itself a privacy guarantee. See <a href="research/limitations/">scope and limitations</a> before reusing the results.
</div>
