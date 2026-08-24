# Problem and use cases

WLCR-SEA addresses one specific problem: **how can we predict the next 24 hours
for one cell when the model receives only that cell's recent history?** This
setting matters when data access is restricted, some measurements may be
missing, and the calculation may need to be checked later.

<figure class="paper-figure">
  <a href="../../images/paper_figure_scenario.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_scenario.png" alt="One cell's data moving from input preparation to forecast and calculation record" loading="lazy">
  </a>
  <figcaption>Manuscript Figure 1. A gateway prepares one cell's input; the forecasting model cannot fetch live data from other cells.</figcaption>
</figure>

## Why this forecast is useful

Short-term traffic forecasts can help teams plan radio resources and service
capacity before demand arrives. Here the goal is to predict four cell-level
measures for the next 24 hours: uplink users, downlink users, downlink used
PRBs, and uplink used PRBs.

The paper studies a deliberately limited setup. A gateway can verify the cell
and prepare its data, but the forecasting model cannot fetch neighbor traffic,
topology data, or values saved from another request. This can be useful when
systems are separated for access control, fault isolation, or easier review.

## How one request is processed

<div class="process-steps process-steps--compact">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>The available history ends at 23:00</h3>
      <p>The model predicts from 00:00 through the following 23:00. The output is a planning signal, not an automatic scheduling command.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>The gateway prepares one cell's data</h3>
      <p>It collects 336 consecutive hours and four indicators. A Boolean mask records which values are present. The cell ID can remain outside the model for access control and record keeping.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">03</span>
    <div>
      <h3>The model uses only the prepared input</h3>
      <p>It combines that history with one fixed model version and statistics learned during training. It cannot use the cell ID to look up extra traffic data.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">04</span>
    <div>
      <h3>The service returns predictions and a calculation record</h3>
      <p>The result contains four 24-hour series. The record can save the input hash, model version, candidate values, availability, weights, final adjustment, and range checks.</p>
    </div>
  </article>
</div>

<div class="notice-card">
  <strong>The paper evaluates the forecast, not a complete network system.</strong> It measures accuracy, performance with missing data, inspection checks, and latency. It does not measure business impact or automatic radio control.
</div>

## Where this setup is useful

<div class="scenario-grid scenario-grid--compact">
  <article class="scenario-card">
    <span class="scenario-tag">PLAN</span>
    <h3>Next-day resource outlook</h3>
    <p>A planner needs a next-day demand estimate for each cell, while a separate system handles scheduling.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">MISSING</span>
    <h3>Some measurements are missing</h3>
    <p>The model uses a clear missing-value marker, so a placeholder number cannot be mistaken for a real observation.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">REVIEW</span>
    <h3>A prediction must be replayed</h3>
    <p>A reviewer needs to see which historical references were available and how their weights changed.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">COMPARE</span>
    <h3>Models need a fair comparison</h3>
    <p>Researchers can compare models using the same input data and the same missing-data patterns.</p>
  </article>
</div>

## When another formulation is better

WLCR-SEA should not be treated as the default answer to every cellular
forecasting problem.

- If reliable neighbor traffic and topology are available, a graph or
  multi-cell model can use more information and may be a better choice.
- If weather, events, mobility, network upgrades, or holidays drive the traffic,
  these signals need to be added and tested; the current historical patterns
  do not include them.
- If an application needs a reliable uncertainty estimate, routing entropy is
  not enough; the study found no useful relationship between entropy and error.
- If predictions will directly control the network, additional safety and
  real-world impact testing is required.

## Model input

Each prediction uses:

- 336 consecutive history hours;
- four traffic indicators;
- values transformed with `log1p` inside the model;
- one Boolean marker for every value: present or missing;
- no cell identity used as a prediction feature.

The target is the next 24 hours of the same four series.

## What the model may and may not use

| The model may use | The model may not use |
| --- | --- |
| Values and missing-value markers in the current input | Live traffic from another cell |
| Fixed statistics learned during training | Metadata looked up by cell ID |
| One versioned global checkpoint | Topology or neighbor tables |
| Fixed method configuration | External feature stores |
| Cell ID outside the model for access control and records | Traffic cached from another request |

The paper calls this **request-local forecasting**: once the input has been
prepared, the model cannot fetch more traffic data for that prediction.

## Why the constraint changes the model

A conventional model may return only a number. WLCR-SEA first turns the input
into eight candidate forecasts based on known historical patterns. It removes
candidates that cannot be computed, combines the remaining ones, and limits
the size of the final adjustment. The same steps can be repeated from the same
input.

[Follow the system architecture](architecture.md){ .md-button .md-button--primary }
[See how the eight experts are routed](method.md){ .md-button }
[Try the Live Demo](../deployment/hugging-face.md){ .md-button }

!!! note "Limiting model input is not the same as protecting data"
    The surrounding system still needs access control, encryption, safe logs,
    and appropriate data-retention rules.
