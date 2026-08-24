# Method: structured seasonal expert routing

For every forecast horizon and indicator, WLCR-SEA builds a finite bank of
eight candidates. The router sees their values and availability rather than a
hidden, unconstrained feature store.

<figure class="paper-figure">
  <a href="../../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_architecture.png" alt="WLCR-SEA expert construction, masked routing, baseline, bounded residual, and audit record" loading="lazy">
  </a>
  <figcaption>Manuscript Figure 2. The online path is a finite expert interface followed by available-set routing and a bounded correction.</figcaption>
</figure>

## The eight experts

| Expert | Request-local evidence | Availability rule |
| --- | --- | --- |
| Last day | Same target hour one day earlier | That observation exists |
| Last week | Same hour seven days earlier | That observation exists |
| Two-week lag | Same hour fourteen days earlier | That observation exists |
| 7-day same-hour median | Robust median across seven seasonal positions | At least one position exists |
| 14-day same-hour median | Robust median across fourteen positions | At least one position exists |
| Bounded weekly trend | Last-week value plus clipped weekly change | Week and biweek values exist |
| Window-local median | Median of the current 336-hour request | At least one indicator value exists |
| Frozen training prior | Horizon–indicator training median | Always available after fitting |

Missing values can carry arbitrary numerical fills internally; the mask is
authoritative. Every aggregate is recomputed after artificial removal, which
prevents a deleted value from leaking through its fill.

## Available-set routing

The selected model uses horizon-conditioned Entmax over the **compacted set of
available experts**. Probabilities are then scattered back to eight slots.
This makes an unavailable expert's mass structurally equal to zero instead of
approximately zero after a large negative sentinel.

Reliability features tell the router how much support underlies a median or
summary expert. They do not override availability.

## Bounded residual

The routed convex combination yields a log-space baseline. A learned residual
is transformed through `tanh` and multiplied by a fixed bound. Therefore every
prediction remains inside:

\[
\left[\min_{j \in A} e_j - b,\; \max_{j \in A} e_j + b\right]
\]

where \(A\) is the available expert set and \(b\) is the residual bound.

## Audit fields

A replayable record can retain the request hash, model version, expert values,
availability, reliability, routing weights, residual, baseline, prediction,
and envelope checks. The repository's audit programs also verify a field
allowlist so identity-conditioned inputs do not silently enter the scorer.

!!! info "What the public Demo runs"
    The Space runs the registered `A0_fixed` parameter-free ablation: weekly
    lag 0.7, biweekly lag 0.2, and 7-day same-hour median 0.1, renormalized over
    available evidence. The trained A6 checkpoint is not distributed, so the
    demo does not claim to reproduce the paper model's predictions.

[See the system architecture](architecture.md){ .md-button }
[Start with the business scenarios](problem.md){ .md-button }
