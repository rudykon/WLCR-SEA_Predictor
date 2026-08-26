# Results

WLCR-SEA is not uniformly the most accurate model. DLinear leads when history
is complete; WLCR-SEA is stronger in the study's severe missing-telemetry
comparisons and preserves a replayable routing record.

!!! info "Metric labels"
    Lower WAPE is better. Every WAPE below is explicitly labeled by aggregation.
    A confidence interval containing zero does not show a clear paired
    difference.

## Evaluation setting

| Item | Registered setting |
| --- | --- |
| Data | 527,760 cell-hours from 736 cells in one anonymous region |
| Request | One cell, 336-hour history, four indicators, observation mask |
| Target | The same cell's next 24 hours and four indicators |
| Primary model | WLCR-SEA five-model ensemble, seeds 42–46, mean in linear traffic space |
| Complete-history test | Fixed later time period defined by the study |
| Missingness test | Identical historical positions removed for every compared model |

The [public model release](https://huggingface.co/config-h/WLCR-SEA-Predictor)
reports the ensemble's complete-history **macro-cell WAPE 0.177612**, **pooled
WAPE 0.184915**, and **macro-indicator WAPE 0.195511**. These values describe
different aggregations of the same registered workflow and are not
interchangeable.

## Complete history

<figure class="paper-figure">
  <a href="../../images/paper_figure_clean_accuracy.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_clean_accuracy.png" alt="Complete-history model comparison with explicitly labeled macro-indicator WAPE">
  </a>
  <figcaption>Complete-history comparison on the registered holdout workflow.</figcaption>
</figure>

| Model | Macro-indicator WAPE | Interpretation |
| --- | ---: | --- |
| DLinear | **0.1854** | Lowest complete-history error in this comparison |
| Prior traffic-only method | 0.1951 | Similar error to WLCR-SEA |
| WLCR-SEA five-model ensemble | 0.1955 | Difference from prior method: +0.00045; 95% CI [-0.00312, 0.00366] |

**Finding:** DLinear is more accurate with complete history. WLCR-SEA and the
prior method are not clearly different because the paired interval includes
zero.

## Missing telemetry

<figure class="paper-figure">
  <a href="../../images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_missingness.png" alt="Matched missing-telemetry comparison under block, recent-tail, and asynchronous patterns">
  </a>
  <figcaption>Every model receives the same removed historical positions.</figcaption>
</figure>

The registered stress test removes a contiguous block, the recent tail, or
different periods by indicator. At 50% requested removal:

| Pattern | WLCR-SEA macro-indicator WAPE |
| --- | ---: |
| Contiguous block | 0.2196 |
| Recent tail | 0.2460 |
| Asynchronous indicators | 0.2172 |

WLCR-SEA has lower macro-indicator WAPE in all **9 prespecified severe-setting
comparisons** against DLinear-Aug, PatchTST-Aug, and GRU-D. Some moderate-rate
paired intervals against DLinear-Aug and PatchTST-Aug still include zero.

**Finding:** the advantage is specific to the tested missingness patterns,
rates, data, and registered runs. It is not a guarantee for arbitrary outages.

## Auditability

<figure class="paper-figure">
  <a href="../../images/paper_figure_auditability.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_auditability.png" alt="Hard-mask, bounded-envelope, request-locality, and expert-deletion audits">
  </a>
  <figcaption>Structural checks and measured influence after expert deletion.</figcaption>
</figure>

| Check | Registered result |
| --- | ---: |
| Weight assigned to unavailable experts | 0 |
| Predictions outside the registered bounded envelope | 0 |
| Output changes after changing only the request wrapper (256 tests) | 0 |
| Mean experts with meaningful support | 5.223 [5.101, 5.345] |
| Macro-indicator WAPE increase after removing the highest-weight expert | +0.00595 [0.00441, 0.00757] |
| Macro-indicator WAPE increase after removing a random matched expert | +0.00104 [0.00079, 0.00131] |
| Routing weight versus measured deletion influence, Spearman | 0.693 [0.678, 0.708] |

Higher-weight experts usually have more effect when removed, so weights help
review the model's internal allocation. Routing entropy correlates -0.0196 with
error, 95% CI [-0.0407, 0.0016]; it is **not** a reliable uncertainty score.

## CPU cost and cell-disjoint audit

With one CPU thread, batch size 1, and sequential requests:

| Predictor | Median | P99 | Model assets |
| --- | ---: | ---: | ---: |
| One WLCR-SEA model | 6.802 ms | 7.574 ms | 16.2 KiB |
| Five-model ensemble | 34.705 ms | 38.684 ms | 148.8 KiB |

With disjoint training and test cells, WLCR-SEA reaches **macro-indicator WAPE 0.1967**.
Differences from DLinear-Aug and the prior method remain unclear;
WLCR-SEA beats PatchTST-Aug. All cells still come from the same region.

## Limitations

- **Data and generalization:** one anonymous region, roughly one month; all
  forecasts begin at midnight; later-period data informed some design choices;
  the cell-disjoint audit remains within the same regional trace.
- **Accuracy:** DLinear leads with complete history; moderate missingness gains
  are not always clear; tested outage patterns do not cover all telemetry
  failures.
- **Uncertainty:** routing entropy and expert weights are not calibrated
  uncertainty estimates.
- **Privacy and serving:** request-local computation does not provide access
  control, encryption, anonymization, differential privacy, safe logging, or
  retention policy. Production systems need these controls separately.
- **Operational scope:** the study evaluates forecasting, not network-control
  actions, scheduling decisions, or business outcomes.

Use the project for research and controlled evaluation. Revalidate the model,
data contract, security controls, and failure modes before operational use.
