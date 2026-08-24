# Experimental results

**DLinear leads on complete data. WLCR-SEA is stronger in the tested severe outages and keeps a traceable calculation.**

!!! info "Reading the tables"
    Lower **WAPE** is better. A 95% confidence interval that includes zero does not show a clear difference.

## Setup

| Item | Reported setting |
| --- | --- |
| Records | 527,760 cell-hours |
| Cells | 736 |
| Request | 336-hour history plus observation mask |
| Target | Next 24 hours, four indicators |
| Complete-data test | Fixed later time period defined in the paper |
| Missing-data test | The same data points are removed for all compared models |

## Complete data

<figure class="paper-figure">
  <a href="../../images/paper_figure_clean_accuracy.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_clean_accuracy.png" alt="Model comparison when historical data are complete" loading="lazy">
  </a>
  <figcaption>Complete-data comparison.</figcaption>
</figure>

| Model | WAPE | What it means |
| --- | ---: | --- |
| DLinear | **0.1854** | Lowest complete-data WAPE in the comparison |
| Prior traffic-only method | 0.1951 | Similar error to WLCR-SEA |
| WLCR-SEA, five-model ensemble | 0.1955 | Difference from prior method: +0.00045; 95% CI [-0.00312, 0.00366] |

**Result:** DLinear has the lowest error. WLCR-SEA and the prior method are not clearly different because their interval includes zero.

## Missing data

<figure class="paper-figure">
  <a href="../../images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_missingness.png" alt="Model comparison after removing the same historical values" loading="lazy">
  </a>
  <figcaption>Matched missing-data comparison.</figcaption>
</figure>

The study removes one block, the recent tail, or different periods by indicator. At 50% removal:

| 50% removal pattern | WLCR-SEA WAPE |
| --- | ---: |
| Contiguous block | 0.2196 |
| Timeline tail | 0.2460 |
| Asynchronous indicators | 0.2172 |

WLCR-SEA has lower error in all nine fixed comparisons with DLinear-Aug, PatchTST-Aug, and GRU-D. Some moderate-rate intervals against DLinear-Aug and PatchTST-Aug still include zero.

**Result:** better performance in the tested severe outages, not a guarantee for other data or runs.

## Traceability

<figure class="paper-figure">
  <a href="../../images/paper_figure_auditability.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_auditability.png" alt="Checks of candidate weights, deletion effects, and prediction limits" loading="lazy">
  </a>
  <figcaption>Weight, deletion, and range checks.</figcaption>
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

Higher-weight candidates usually have more effect when removed. Saved weights therefore help inspect a result.

Routing entropy correlates -0.0196 with error, 95% CI [-0.0407, 0.0016]. It is **not** a reliable uncertainty score.

## Speed and cell split

With one CPU thread, batch size 1, and sequential requests, a single model takes 6.802 ms median / 7.574 ms P99 and uses 16.2 KiB of model assets. The five-model ensemble takes 34.705 ms / 38.684 ms and 148.8 KiB.

With disjoint train/test cells, WLCR-SEA reaches 0.1967 WAPE. Differences from DLinear-Aug and the prior method remain unclear; WLCR-SEA beats PatchTST-Aug. All cells still come from one region.
