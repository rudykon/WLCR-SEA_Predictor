# From a midnight request to an auditable forecast

WLCR-SEA starts with a practical question: **how can a forecasting service help
plan the next day when it is allowed to read live traffic only for the cell in
the current request?** The answer must be useful under incomplete telemetry and
replayable without silently consulting another cell.

<figure class="paper-figure">
  <a href="../../images/paper_figure_scenario.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_scenario.png" alt="Conceptual request-local serving scenario with an ingress, sealed request, shared checkpoint, scorer, and audit record" loading="lazy">
  </a>
  <figcaption>Manuscript Figure 1. The ingress and scorer have distinct responsibilities; the dashed boundary is an evidence boundary, not a claim of complete privacy.</figcaption>
</figure>

## Background: why this request exists

Short-horizon cellular-traffic forecasts can support proactive radio-resource
and service-capacity planning. A planning system may want the expected shape of
uplink users, downlink users, downlink used PRBs and uplink used PRBs over the
next 24 hours.

The forecasting environment studied here is deliberately constrained. An
identity-aware ingress can authorize a target cell and assemble its data, but
the scoring process is not allowed to fetch live traffic from neighboring
cells, a topology service or a cross-request cache. That separation may arise
in compartmentalized edge domains where access, fault containment and later
replay require a small, explicit online evidence surface.

## A concrete operating story

<div class="story-steps story-steps--compact">
  <article class="story-step">
    <span class="story-number">01</span>
    <div>
      <h3>The observed day closes</h3>
      <p>The final observed hour is 23:00. The next target is 00:00, and the requested horizon continues for 24 hours. The downstream goal is a demand signal—not an autonomous scheduling command.</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">02</span>
    <div>
      <h3>The ingress authorizes one cell</h3>
      <p>It materializes 336 consecutive history hours, four traffic indicators and one authoritative Boolean observation mask. An opaque origin ID may remain outside the model for authorization, routing and audit.</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">03</span>
    <div>
      <h3>The scorer stays inside the request</h3>
      <p>It combines the sealed window with a versioned global checkpoint and frozen global statistics. It does not receive the cell identity as a feature and cannot issue identity-conditioned lookups.</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">04</span>
    <div>
      <h3>A forecast and trace move downstream</h3>
      <p>The result contains four 24-hour series. A proposed audit record can preserve the request hash, model version, expert values, availability, weights, baseline, bounded residual and envelope checks for later review.</p>
    </div>
  </article>
</div>

<div class="notice-card">
  <strong>Illustrative workflow, measured forecasting task.</strong> The paper evaluates forecast accuracy, robustness, audit properties and latency. It does not evaluate closed-loop radio control, business outcomes or a production ingress deployment.
</div>

## Where this contract is useful

<div class="scenario-grid scenario-grid--compact">
  <article class="scenario-card">
    <span class="scenario-tag">OPERATE</span>
    <h3>Next-day resource outlook</h3>
    <p>A planner needs a per-cell demand signal, while the scheduler remains a separate downstream component.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">DEGRADE</span>
    <h3>Telemetry is partially missing</h3>
    <p>The observation mask—not the numerical fill—is authoritative, so invalid seasonal references can be removed exactly.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">REVIEW</span>
    <h3>A prediction must be replayed</h3>
    <p>Model governance or incident review needs to reconstruct what evidence was available and how routing changed.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">COMPARE</span>
    <h3>Methods need a fair evidence boundary</h3>
    <p>Researchers can compare forecasting families under the same request-local information and matched missingness protocol.</p>
  </article>
</div>

## When another formulation is better

WLCR-SEA should not be treated as the default answer to every cellular
forecasting problem.

- If authorized neighbor traffic and topology are reliable and important at
  scoring time, a graph or multi-cell model addresses a richer information set.
- If sudden events, weather, mobility, provisioning changes or calendar effects
  dominate, the manually specified seasonal expert bank may miss them unless
  those signals are explicitly introduced and validated.
- If an application requires calibrated predictive uncertainty, routing entropy
  is not a substitute; the reported study found no useful error correlation.
- If forecasts will drive autonomous control, the forecasting evidence here is
  insufficient without prospective system-level safety and impact validation.

## The sealed request

One request contains an ordered tensor and an authoritative observation mask:

- 336 consecutive history hours;
- four traffic indicators;
- observed values transformed with `log1p`;
- a Boolean state for every hour and indicator;
- no cell identity supplied to the forecasting function as a feature.

The target is the next 24 hours of the same four series.

## What the online path may use

| Allowed | Not available to the scorer |
| --- | --- |
| Current request values and masks | Live traffic from another cell |
| Frozen training statistics | Cell-conditioned metadata files |
| One versioned global checkpoint | Topology or neighbor tables |
| Fixed method configuration | External feature stores |
| Opaque ID outside the model for audit | Cross-request traffic caches |

## Why the constraint changes the model

A black-box predictor could produce a point estimate without making its
references explicit. WLCR-SEA instead concentrates the current request into a
finite set of named seasonal candidates that can be rebuilt from the same
input. Availability is structural, routing is limited to surviving evidence,
and the correction is bounded around the routed baseline.

[Follow the system architecture](architecture.md){ .md-button .md-button--primary }
[See how the eight experts are routed](method.md){ .md-button }
[Try the request audit lab](../deployment/hugging-face.md){ .md-button }

!!! note "What request-local does—and does not—mean"
    Request-local describes the online evidence path. It does not automatically
    provide anonymization, encryption, access control, differential privacy, or
    protection against leakage elsewhere in the serving system.
