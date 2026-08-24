# Method

WLCR-SEA has three steps:

1. build eight historical candidates;
2. drop unavailable candidates and weight the rest;
3. apply a bounded adjustment.

The saved candidates and weights show what shaped each forecast.

!!! tip "Terms"
    **Expert:** candidate. **Mask:** present/missing flag. **Router:** weighting step. **Residual:** final adjustment.

<figure class="paper-figure">
  <a href="../../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_architecture.png" alt="WLCR-SEA candidate generation, missing-data filtering, weighting, limited adjustment, and calculation record" loading="lazy">
  </a>
  <figcaption>Eight candidates are filtered, weighted, and adjusted.</figcaption>
</figure>

## 1. Candidates

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

Missing positions may contain placeholders, but the mask decides availability. Summary candidates are rebuilt after deletion, so removed values cannot leak into the forecast.

## 2. Routing

For each hour and indicator, usable weights sum to one. Unavailable candidates receive zero.

The selected model applies horizon-conditioned **Entmax** only to the compact list of usable candidates, then restores the eight-position layout. No approximate masking value is needed.

Reliability, such as a median's sample count, may change weight. It cannot restore an unavailable candidate.

## 3. Bounded adjustment

The weighted average is the main log-space prediction. A learned adjustment passes through `tanh` and a fixed limit. The result stays inside:

\[
\left[\min_{j \in A} e_j - b,\; \max_{j \in A} e_j + b\right]
\]

where \(A\) is the usable candidate set and \(b\) is the limit. The output cannot move far beyond the available candidates.

## Record

The record can store the input hash, model version, candidates, availability, reliability, weights, average, adjustment, prediction, and range checks. Tests also reject unapproved input fields.

!!! info "Demo model"
    The Space runs `A0_fixed`: previous week 0.7, two weeks earlier 0.2, and 7-day median 0.1. Available weights are rescaled when needed. It is not the trained A6 model.

[Architecture](architecture.md){ .md-button }
[Use cases](problem.md){ .md-button }
