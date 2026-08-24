# Experimental results

The main result is straightforward: **WLCR-SEA is not the most accurate model
when all historical data are present. Its value is clearer when blocks of
history are missing and when users need to inspect how a prediction was
formed.**

!!! info "How to read the numbers"
    **WAPE** is a forecasting error: lower is better. A **95% confidence
    interval** shows the range supported by the experiment. When an interval
    for a difference includes zero, the study cannot establish a clear
    difference between the two models.

[English paper PDF](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/paper/main.pdf){ .md-button target="_blank" rel="noopener" }
[Chinese paper PDF](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/paper/main_zh.pdf){ .md-button target="_blank" rel="noopener" }

## Data and evaluation setup

| Item | Reported setting |
| --- | --- |
| Records | 527,760 cell-hours |
| Cells | 736 |
| Request | 336-hour history plus observation mask |
| Target | Next 24 hours, four indicators |
| Complete-data test | Fixed later time period defined in the paper |
| Missing-data test | The same data points are removed for all compared models |

<figure class="paper-figure">
  <a href="../../images/paper_figure_scenario.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_scenario.png" alt="One cell's data moving from input preparation to forecast and calculation record" loading="lazy">
  </a>
  <figcaption>Figure 1 · Each prediction uses one prepared cell history and cannot fetch live traffic from other cells.</figcaption>
</figure>

## When all historical data are present

<figure class="paper-figure">
  <a href="../../images/paper_figure_clean_accuracy.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_clean_accuracy.png" alt="Model comparison when historical data are complete" loading="lazy">
  </a>
  <figcaption>Figure 3 · Model comparison when historical data are complete.</figcaption>
</figure>

| Model | WAPE | What it means |
| --- | ---: | --- |
| DLinear | **0.1854** | Lowest complete-data WAPE in the comparison |
| Prior traffic-only method | 0.1951 | Similar error to WLCR-SEA |
| WLCR-SEA, five-model ensemble | 0.1955 | Difference from prior method: +0.00045; 95% CI [-0.00312, 0.00366] |

**Conclusion:** DLinear has the lowest error in this complete-data comparison.
The interval for WLCR-SEA versus the prior traffic-only method includes zero,
so the experiment does not show a clear difference between those two methods.

## When parts of the history are missing

<figure class="paper-figure">
  <a href="../../images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_missingness.png" alt="Model comparison after removing the same historical values" loading="lazy">
  </a>
  <figcaption>Figure 4 · Forecast errors after removing the same values for all compared models.</figcaption>
</figure>

The paper removes data in several realistic patterns: one continuous block,
the most recent part of the timeline, or different times for different
indicators. With 50% of values removed in these patterns, WLCR-SEA reports:

| 50% mechanism | WLCR-SEA WAPE |
| --- | ---: |
| Contiguous block | 0.2196 |
| Timeline tail | 0.2460 |
| Asynchronous indicators | 0.2172 |

For these fixed tests, WLCR-SEA has lower error in all nine comparisons against
DLinear-Aug, PatchTST-Aug, and GRU-D. At some moderate missing rates, however,
the confidence intervals versus DLinear-Aug and PatchTST-Aug include zero.

**Conclusion:** the results support better performance under the tested severe
missing-data patterns. They do not guarantee better results for every outage,
dataset, or retraining run.

## Can the calculation be checked?

<figure class="paper-figure">
  <a href="../../images/paper_figure_auditability.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_auditability.png" alt="Checks of candidate weights, deletion effects, and prediction limits" loading="lazy">
  </a>
  <figcaption>Figure 5 · Checks of candidate weights, deletion effects, and prediction limits.</figcaption>
</figure>

| Check | Reported result |
| --- | ---: |
| Weight given to unavailable candidates | 0 |
| Predictions outside the allowed adjustment range | 0 |
| Output changes caused only by changing the request object wrapper (256 tests) | 0 |
| Average number of candidates receiving meaningful support | 5.223 [5.101, 5.345] |
| Error increase after removing the highest-weight candidate | +0.00595 [0.00441, 0.00757] |
| Error increase after removing a random matched candidate | +0.00104 [0.00079, 0.00131] |
| Relationship between assigned weight and measured influence (Spearman) | 0.693 [0.678, 0.708] |

Candidates with higher weights usually have a larger measured effect when
removed. This supports the usefulness of the saved weights for inspection.

One negative result is equally important: routing entropy has mean correlation
-0.0196 with absolute percentage error, 95% CI [-0.0407, 0.0016]. Because the
interval includes zero, entropy should **not** be treated as a reliable
uncertainty score.

## Speed and a stricter cell split

Using one CPU thread and predicting one request at a time, one WLCR-SEA model
takes a median of 6.802 ms (P99 7.574 ms, 16.2 KiB). The five-model ensemble
takes a median of 34.705 ms (P99 38.684 ms, 148.8 KiB).

In a stricter split where training and test sets use different cells,
WLCR-SEA records WAPE 0.1967. Differences from DLinear-Aug and the prior method
are not clear because their intervals include zero; the result is better than
PatchTST-Aug in this test. All cells still come from the same regional trace,
so this does not prove performance in another region or season.
