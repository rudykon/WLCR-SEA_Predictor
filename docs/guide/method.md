# How WLCR-SEA makes a forecast

WLCR-SEA works in three steps:

1. build eight candidate forecasts from clear historical patterns;
2. remove candidates that depend on missing data and combine the rest;
3. allow a small, limited final adjustment.

This design makes it possible to see which historical references were used for
every future hour and traffic indicator.

!!! tip "Four terms used on this page"
    **Expert** means one candidate forecast. **Mask** marks whether a historical
    value exists. **Router** assigns weights to usable candidates. **Residual**
    is the final adjustment added after the weighted average.

<figure class="paper-figure">
  <a href="../../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_architecture.png" alt="WLCR-SEA candidate generation, missing-data filtering, weighting, limited adjustment, and calculation record" loading="lazy">
  </a>
  <figcaption>Manuscript Figure 2. Eight historical candidates are filtered, weighted, and followed by a limited final adjustment.</figcaption>
</figure>

## Step 1: build eight candidate forecasts

| Candidate | Plain meaning | Available when |
| --- | --- | --- |
| Previous day | Value at the same hour one day earlier | That value exists |
| Previous week | Value at the same hour one week earlier | That value exists |
| Two weeks earlier | Value at the same hour two weeks earlier | That value exists |
| 7-day same-hour median | Median of up to seven matching hours | At least one matching value exists |
| 14-day same-hour median | Median of up to fourteen matching hours | At least one matching value exists |
| Weekly trend | Previous-week value plus a limited weekly change | One- and two-week values exist |
| Input median | Median of the current 336-hour input | At least one value exists for the indicator |
| Training prior | Typical value learned from the training set | The model has been fitted |

Internally, software may place a number in a missing position to keep arrays
rectangular. The mask—not that placeholder number—decides whether the value can
be used. Summary candidates are recomputed after data are removed, so deleted
values cannot affect the forecast through a placeholder.

## Step 2: remove unavailable candidates and combine the rest

For each future hour and indicator, the model first keeps only the candidates
that can be computed. It then gives those candidates weights that add up to
one. A candidate based on missing history receives exactly zero weight.

Technically, the selected model uses horizon-conditioned **Entmax** on the
compacted list of available candidates, then places the resulting weights back
into the original eight positions. Entmax is a weighting function that can set
some weights exactly to zero. This implementation does not rely on an
approximate “very negative number” to hide unavailable candidates.

The model also receives a reliability value, such as how many observations
support a median. Reliability can influence the weight, but it cannot make a
missing candidate available.

## Step 3: limit the final adjustment

The weighted average of the candidates forms the main prediction in log space.
The model can then add a learned adjustment. That adjustment passes through
`tanh` and is multiplied by a fixed limit, so it cannot grow without bound.
The final prediction stays inside:

\[
\left[\min_{j \in A} e_j - b,\; \max_{j \in A} e_j + b\right]
\]

where \(A\) is the set of usable candidates and \(b\) is the maximum adjustment.
In plain language, the output must remain close to the range supported by the
available historical candidates.

## What can be saved for later inspection

A calculation record can save the input hash, model version, candidate values,
availability, reliability, weights, weighted average, final adjustment,
prediction, and range checks. Repository tests also verify that the model input
contains only approved fields and does not quietly add cell-specific data.

!!! info "What the public Demo runs"
    The Space uses the simpler `A0_fixed` baseline: previous week 0.7, two weeks
    earlier 0.2, and 7-day same-hour median 0.1. If one is unavailable, the
    remaining weights are scaled to add up to one. The trained A6 checkpoint is
    not distributed, so Demo outputs are not the paper model's predictions.

[See the system architecture](architecture.md){ .md-button }
[Start with the business scenarios](problem.md){ .md-button }
