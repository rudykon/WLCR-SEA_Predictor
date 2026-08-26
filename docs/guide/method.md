# Method

WLCR-SEA maps a one-cell request $x \in \mathbb{R}^{336 \times 4}$ and its
observation mask to a forecast $\hat{y} \in \mathbb{R}^{24 \times 4}$. Inference may read the request
and frozen model assets, but it does not fetch live traffic from another cell.

## Terms

| Term | Meaning in this project |
| --- | --- |
| **Seasonal expert** | A candidate forecast derived from one historical pattern |
| **Availability mask** | Whether the evidence needed by an expert is present |
| **Routing weight** | That expert's contribution to the routed baseline |
| **Bounded residual** | A learned final correction with a fixed log-space limit |
| **Audit record** | Model identity, expert values, masks, weights, residuals, prediction, and checks |

These terms are used consistently across the website, Demo, exports, and code.

<figure class="paper-figure">
  <a href="../../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../../images/paper_figure_architecture.png" alt="WLCR-SEA candidate construction, availability masking, reliability-aware Entmax routing, bounded residual, forecast, and audit record">
  </a>
  <figcaption>Eight seasonal experts are filtered, routed, and corrected within a fixed bound.</figcaption>
</figure>

## 1. Eight seasonal experts

For every future hour and indicator, the model constructs the following
candidate values in `log1p` space.

| Seasonal expert | Evidence | Available when |
| --- | --- | --- |
| Previous day | Same hour one day earlier | That value is observed |
| Previous week | Same hour one week earlier | That value is observed |
| Two weeks earlier | Same hour two weeks earlier | That value is observed |
| 7-day same-hour median | Up to seven matching hours | At least one matching value is observed |
| 14-day same-hour median | Up to fourteen matching hours | At least one matching value is observed |
| Limited weekly trend | Previous-week value plus clipped week-to-week change | One- and two-week values are observed |
| Request median | Median of the current 336-hour request | The indicator has at least one observation |
| Frozen training prior | Typical target value estimated during training | Always available in a fitted checkpoint |

The observation mask is authoritative. When the missingness protocol removes a
value, every summary expert is rebuilt after removal; placeholder values cannot
leak into medians, trends, or context features.

Each of the five public model checkpoints contains its own frozen `(24, 4)`
training prior. The Demo never derives this prior from the uploaded request.

## 2. Hard-masked sparse routing

The router embeds the forecast horizon, indicator, expert type, expert value,
distance, reliability, and request-local context. It applies
reliability-aware **Entmax 1.5** only to the compact set of available experts,
then restores the eight-position layout.

For each horizon and indicator:

- unavailable experts receive exactly zero weight;
- available weights sum to one;
- reliability may change a weight but cannot restore an unavailable expert;
- sparse support makes the internal allocation inspectable.

Routing weight is an internal allocation, not a causal effect. The deletion
audit measures whether high-weight experts tend to have more influence when
removed; it does not turn attention into a causal explanation.

## 3. Bounded residual

The routed expert average is the baseline in log space. A small network produces
a correction passed through `tanh` and the selected checkpoint's bound $b$:

\[
\hat{y}_{h,q}^{\log} = \sum_{j \in A_{h,q}} w_{h,q,j} e_{h,q,j}
+ b\,\tanh(r_{h,q}).
\]

Therefore the member prediction remains inside:

\[
\left[\min_{j \in A_{h,q}} e_{h,q,j} - b,\;
      \max_{j \in A_{h,q}} e_{h,q,j} + b\right].
\]

The public checkpoints use the residual bound recorded in each selected
configuration. The exported audit performs the check per member and on the
linear-space ensemble.

## 4. Five-model ensemble

The primary predictor contains seeds 42–46. Each member builds experts with its
own frozen prior and produces a complete `24 × 4` forecast. The final output is
the arithmetic mean of the five arrays **after inverse transformation to linear
traffic space**.

[Open the live Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }
[Inspect the pinned checkpoints](https://huggingface.co/config-h/WLCR-SEA-Predictor/tree/eb4447f4ebab8f9caa003d92c838ed8e750963bd){ .md-button target="_blank" rel="noopener" }

## 5. Audit record

The versioned JSON export records the source commit and environment, input
hash, missingness seed and effective mask, model revision, checkpoint hashes,
and per-member predictions, routing components, envelopes, and violation
counts. Audit schema v4 distinguishes the configured removal rate, positions
selected for corruption, observations actually removed, and the final observed
fraction. It supports review and, when paired with the original request and
pinned source revision, replay. It does not claim calibrated uncertainty or
privacy.
