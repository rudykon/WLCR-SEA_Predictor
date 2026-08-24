# How WLCR-SEA makes a forecast

WLCR-SEA works in three steps:

1. build eight candidate forecasts from clear historical patterns;
2. remove candidates that depend on missing data and combine the rest;
3. allow a small, limited final adjustment.

This design shows which historical references contributed to each future hour and traffic indicator.

!!! tip "Four terms used on this page"
    An **expert** is one candidate forecast. A **mask** marks whether a historical value is present. The **router** assigns weights to candidates that can be used. The **residual** is the final adjustment added to the weighted average.

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

To keep arrays rectangular, the software may place a placeholder number in a missing position. The mask—not the placeholder—determines whether that position can be used. Summary candidates are recalculated after data are removed, so deleted values cannot continue to influence the forecast.

## Step 2: remove unavailable candidates and combine the rest

For each future hour and indicator, the model keeps only the candidates that can be computed, then assigns them weights that add up to one. A candidate that depends on missing history receives exactly zero weight.

Technically, the selected model applies horizon-conditioned **Entmax** to the compacted list of available candidates, then maps the resulting weights back to the original eight positions. Entmax is a weighting function that can set some weights exactly to zero. This implementation therefore does not need an approximate “very negative number” to hide unavailable candidates.

The model also receives a reliability value, such as the number of observations behind a median. Reliability may affect a candidate's weight, but it cannot make an unavailable candidate usable.

## Step 3: limit the final adjustment

The weighted average of the candidates forms the main prediction in log space. The model can then add a learned adjustment. This adjustment passes through `tanh` and is multiplied by a fixed limit, so it cannot grow without bound. The final prediction stays inside:

\[
\left[\min_{j \in A} e_j - b,\; \max_{j \in A} e_j + b\right]
\]

where \(A\) is the set of usable candidates and \(b\) is the maximum adjustment. In plain language, the output cannot move far beyond the range supported by the available historical candidates.

## What can be saved for later inspection

A calculation record can store the input hash, model version, candidate values, availability, reliability, weights, weighted average, final adjustment, prediction, and range checks. Repository tests also verify that the model receives only approved fields and does not silently add cell-specific data.

!!! info "Which model the public Demo uses"
    The Space runs the simpler `A0_fixed` baseline: previous week 0.7, two weeks earlier 0.2, and 7-day same-hour median 0.1. If one candidate is unavailable, the remaining weights are rescaled to add up to one. Because the trained A6 checkpoint is not public, Demo outputs are not predictions from the paper's trained model.

[See the system architecture](architecture.md){ .md-button }
[Start with the business scenarios](problem.md){ .md-button }
