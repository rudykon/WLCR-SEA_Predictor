# Problem and use cases

WLCR-SEA addresses one specific problem: **how can we predict one cell's traffic over the next 24 hours when the model receives only that cell's recent history?** This setting matters when data access is restricted, some measurements may be missing, or the calculation may need to be reviewed later.

<figure class="paper-figure">
  <a href="../../images/paper_figure_scenario.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_scenario.png" alt="One cell's data moving from input preparation to forecast and calculation record" loading="lazy">
  </a>
  <figcaption>Manuscript Figure 1. A gateway prepares one cell's input; the forecasting model cannot fetch live data from other cells.</figcaption>
</figure>

## Why this forecast is useful

Short-term traffic forecasts can help teams plan radio resources and service capacity before demand arrives. This project predicts four cell-level measures for the next 24 hours: active uplink users, active downlink users, used downlink PRBs, and used uplink PRBs.

The study deliberately limits what the forecasting model can read. A gateway may verify the cell and prepare its data, but the model cannot fetch neighboring-cell traffic, topology data, or values saved from another request. This separation is useful when a system needs tighter access control, fault isolation, or a calculation that can be reviewed later.

## How one request is processed

<div class="process-steps process-steps--compact">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>The latest day closes at 23:00</h3>
      <p>Using the history available through 23:00, the model predicts the next day from 00:00 to 23:00. The output supports planning; it is not an automatic scheduling command.</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>The gateway prepares one cell's data</h3>
      <p>It collects four indicators over 336 consecutive hours. A Boolean mask records which values are present. The cell ID remains outside the model and may be used for access control and record keeping.</p>
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
      <p>The result contains a 24-hour series for each of the four indicators. The calculation record can store the input hash, model version, candidate values, availability, weights, final adjustment, and range checks.</p>
    </div>
  </article>
</div>

<div class="notice-card">
  <strong>The study evaluates forecasting, not a complete network system.</strong> It measures accuracy, robustness to missing data, calculation checks, and latency. It does not measure business impact or automatic radio control.
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

## When another method may be a better fit

WLCR-SEA is not the default choice for every cellular forecasting problem.

- If reliable neighboring-cell traffic and topology are available, a graph or multi-cell model can use that additional information and may be more suitable.
- If weather, events, mobility, network upgrades, or holidays strongly affect traffic, those signals need to be added and evaluated; the current method does not include them.
- If the application requires a reliable uncertainty estimate, routing entropy is not sufficient; the study found no useful relationship between entropy and error.
- If predictions will directly control the network, additional safety testing and real-world impact evaluation are required.

## Model input

Each prediction uses:

- 336 consecutive hours of history;
- four traffic indicators;
- values transformed with `log1p` inside the model;
- one Boolean marker for every value: present or missing;
- no cell identity as a prediction feature.

The target is the next 24 hours of those same four indicators.

## What the model may and may not use

| The model may use | The model may not use |
| --- | --- |
| Values and missing-value markers in the current input | Live traffic from another cell |
| Fixed statistics learned during training | Metadata looked up by cell ID |
| One versioned global checkpoint | Topology or neighbor tables |
| Fixed method configuration | External feature stores |
| Cell ID outside the model for access control and records | Traffic cached from another request |

The project calls this **request-local forecasting**: once an input has been prepared, the model cannot fetch additional traffic data for that prediction.

## Why the constraint changes the model

A conventional forecasting model may return only its final prediction. WLCR-SEA instead builds eight candidates from known historical patterns, removes candidates that cannot be computed, combines the remaining candidates, and limits the final adjustment. Given the same input and model version, the calculation can be repeated.

[Follow the system architecture](architecture.md){ .md-button .md-button--primary }
[See how the eight experts are routed](method.md){ .md-button }
[Try the Live Demo](../deployment/hugging-face.md){ .md-button }

!!! note "Limiting model input is not the same as protecting data"
    The surrounding system still needs access control, encryption, safe logging, and appropriate data-retention rules.
