# Paper evidence

The manuscript evaluates an exploratory **robustness–inspectability–cost
profile**, not a universal leaderboard winner. Results below preserve the
paper's estimands and qualifications.

[English paper PDF](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/paper/main.pdf){ .md-button target="_blank" rel="noopener" }
[Chinese paper PDF](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/paper/main_zh.pdf){ .md-button target="_blank" rel="noopener" }

## Study frame

| Item | Reported setting |
| --- | --- |
| Records | 527,760 cell-hours |
| Cells | 736 |
| Request | 336-hour history plus observation mask |
| Target | Next 24 hours, four indicators |
| Clean split | Fixed chronological holdout defined in the paper |
| Missingness | Matched deterministic masks, with refits where specified |

<figure class="paper-figure">
  <a href="../../images/paper_figure_scenario.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_scenario.png" alt="Request-local evidence boundary used by the study" loading="lazy">
  </a>
  <figcaption>Figure 1 · The evaluated serving path is self-contained after request materialization.</figcaption>
</figure>

## Clean forecasting

<figure class="paper-figure">
  <a href="../../images/paper_figure_clean_accuracy.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_clean_accuracy.png" alt="Clean holdout routing hierarchy" loading="lazy">
  </a>
  <figcaption>Figure 3 · Routing hierarchy on the clean holdout.</figcaption>
</figure>

| Result | WAPE | Interpretation |
| --- | ---: | --- |
| DLinear | **0.1854** | Lowest clean WAPE in the comparison |
| Prior traffic-only method | 0.1951 | Paired reference for the selected method |
| WLCR-SEA, five-seed ensemble | 0.1955 | Difference +0.00045; 95% CI [-0.00312, 0.00366] |

The paired interval versus the prior method contains zero. The study therefore
does not detect a clean-history difference between those two methods. It also
does not obscure that DLinear is more accurate on the clean comparison.

## Structured missingness

<figure class="paper-figure">
  <a href="../../images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_missingness.png" alt="Matched missingness robustness results" loading="lazy">
  </a>
  <figcaption>Figure 4 · Robustness under matched missing-telemetry mechanisms.</figcaption>
</figure>

At moderate selected missingness, intervals versus DLinear-Aug and PatchTST-Aug
include zero. Under the listed 50% structured-corruption settings, WLCR-SEA
reports:

| 50% mechanism | WLCR-SEA WAPE |
| --- | ---: |
| Contiguous block | 0.2196 |
| Timeline tail | 0.2460 |
| Asynchronous indicators | 0.2172 |

All nine paired differences against the matched DLinear-Aug, PatchTST-Aug, and
GRU-D baselines are below zero under those fixed masks. This evidence is
conditional on five fixed corruption masks; it is not a claim about every
possible outage or retraining run.

## Auditability

<figure class="paper-figure">
  <a href="../../images/paper_figure_auditability.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_auditability.png" alt="Auditability evidence including expert deletion and routing influence" loading="lazy">
  </a>
  <figcaption>Figure 5 · Structural checks, expert deletion, and routing–influence alignment.</figcaption>
</figure>

| Audit | Reported result |
| --- | ---: |
| Unavailable-expert routing mass | 0 |
| Bounded-envelope violations | 0 |
| Bitwise serving-path differences across 256 request objects | 0 |
| Effective expert support | 5.223 [5.101, 5.345] |
| Top-weight expert deletion, WAPE increase | +0.00595 [0.00441, 0.00757] |
| Matched-random deletion, WAPE increase | +0.00104 [0.00079, 0.00131] |
| Weight–influence Spearman | 0.693 [0.678, 0.708] |

Routing entropy has mean correlation -0.0196 with absolute percentage error,
95% CI [-0.0407, 0.0016]. The interval includes zero, so entropy should **not**
be used as a calibrated uncertainty score based on this study.

## Cost and cell-disjoint audit

With one CPU thread and batch size one, the reported single-seed median latency
is 6.802 ms (P99 7.574 ms, 16.2 KiB). The five-seed ensemble median is 34.705 ms
(P99 38.684 ms, 148.8 KiB).

In the protocol-matched cell-disjoint refits, WLCR-SEA records WAPE 0.1967.
Intervals versus DLinear-Aug and the prior method include zero; the interval
versus PatchTST-Aug lies below zero. These fixed refits are a within-trace audit,
not evidence of retraining or cross-region generalization.
