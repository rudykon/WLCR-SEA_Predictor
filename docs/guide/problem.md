# Request-local serving problem

WLCR-SEA starts from an operational constraint: the online scorer may read live
traffic only for the cell named by the current request. A front end can use an
identity to authorize and assemble the request, but the predictor does not use
that identity as a feature or a key for another lookup.

<figure class="paper-figure">
  <a href="../../images/paper_figure_scenario.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_scenario.png" alt="Conceptual request-local serving scenario with an ingress, sealed request, shared checkpoint, scorer, and audit record" loading="lazy">
  </a>
  <figcaption>Manuscript Figure 1. The ingress and scorer have distinct responsibilities; the dashed boundary is an evidence boundary, not a claim of complete privacy.</figcaption>
</figure>

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

A graph or neighbor-aware predictor can be valuable when authorized neighbor
states are available. It is not interchangeable with this problem. WLCR-SEA
instead concentrates the current request into finite seasonal candidates that
can be replayed later from the same input.

!!! note "What request-local does—and does not—mean"
    Request-local describes the online evidence path. It does not automatically
    provide anonymization, encryption, access control, differential privacy, or
    protection against leakage elsewhere in the serving system.
