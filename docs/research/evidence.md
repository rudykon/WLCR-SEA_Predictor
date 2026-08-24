# Experimental results

The main result is straightforward: **WLCR-SEA is not the most accurate model when all historical data are available. Its strengths become clearer when blocks of history are missing and when users need to inspect how a prediction was produced.**

!!! info "How to read the numbers"
    **WAPE** measures forecasting error, so lower values are better. A **95% confidence interval** describes the range of differences supported by the experiment. If the interval for a difference includes zero, the experiment does not establish a clear difference between the two models.

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

**Conclusion:** DLinear has the lowest error in this complete-data comparison. The confidence interval for the difference between WLCR-SEA and the prior traffic-only method includes zero, so the experiment does not establish a clear difference between those two methods.

## When parts of the history are missing

<figure class="paper-figure">
  <a href="../../images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_missingness.png" alt="Model comparison after removing the same historical values" loading="lazy">
  </a>
  <figcaption>Figure 4 · Forecast errors after removing the same values for all compared models.</figcaption>
</figure>

The study removes data in several patterns that resemble practical failures: one continuous block, the most recent part of the timeline, or different periods for different indicators. When 50% of the values are removed, WLCR-SEA reports:

| 50% removal pattern | WLCR-SEA WAPE |
| --- | ---: |
| Contiguous block | 0.2196 |
| Timeline tail | 0.2460 |
| Asynchronous indicators | 0.2172 |

Across these fixed tests, WLCR-SEA has lower error in all nine comparisons with DLinear-Aug, PatchTST-Aug, and GRU-D. At some moderate missing rates, however, the confidence intervals for the differences from DLinear-Aug and PatchTST-Aug include zero.

**Conclusion:** the results support better performance under the severe missing-data patterns tested in this study. They do not guarantee better results for every outage, dataset, or retraining run.

## Can the basis of a prediction be checked?

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
| Output changes after only the request wrapper is changed (256 tests) | 0 |
| Average number of candidates receiving meaningful support | 5.223 [5.101, 5.345] |
| Error increase after removing the highest-weight candidate | +0.00595 [0.00441, 0.00757] |
| Error increase after removing a random matched candidate | +0.00104 [0.00079, 0.00131] |
| Relationship between assigned weight and measured influence (Spearman) | 0.693 [0.678, 0.708] |

Candidates with higher weights usually have a greater measured effect when removed. This result supports using the saved weights to inspect a prediction.

One negative result is equally important: routing entropy has a mean correlation of -0.0196 with absolute percentage error, with a 95% confidence interval of [-0.0407, 0.0016]. Because this interval includes zero, entropy should **not** be treated as a reliable uncertainty score.

## Speed and a stricter cell split

Using one CPU thread and processing one request at a time, a single WLCR-SEA model has a median latency of 6.802 ms and a P99 latency of 7.574 ms; its checkpoint and frozen assets total 16.2 KiB. The five-model ensemble has a median latency of 34.705 ms, a P99 latency of 38.684 ms, and 148.8 KiB of checkpoint and frozen assets.

In a stricter split with different cells in the training and test sets, WLCR-SEA records a WAPE of 0.1967. The differences from DLinear-Aug and the prior method remain unclear because their confidence intervals include zero; WLCR-SEA performs better than PatchTST-Aug in this test. All cells still come from the same regional trace, so this result does not establish performance in another region or season.
