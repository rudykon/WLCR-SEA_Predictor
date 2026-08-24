# Code map

Key paths at a glance.

| Path | Responsibility |
| --- | --- |
| `experiments/wlcr_sea_model.py` | Candidates, Entmax, variants, losses, metrics, records |
| `experiments/missingness_protocol.py` | Repeatable data removal |
| `experiments/train_wlcr_sea.py` | Training |
| `experiments/analyze_matched_missingness.py` | Matched outage comparison |
| `experiments/audit_expert_routing.py` | Weight and deletion checks |
| `experiments/audit_request_locality.py` | Input-field checks |
| `experiments/benchmark_wlcr_sea_latency.py` | Latency and memory |
| `experiments/validate_evidence_integrity.py` | Result consistency |
| `Model/traffic_window_forecasting.py` | CSV input and seasonal baseline |
| `tests/test_wlcr_sea_model.py` | Core tests |
| `demo/` | Gradio app and sample |
| `docs/` | Guides and website |
| `paper/figures/` | Figure sources |

## Core tensor shapes

| Object | Shape | Meaning |
| --- | --- | --- |
| History values | `N × 336 × 4` | Traffic values after `log1p`; missing positions may contain placeholders |
| History mask | `N × 336 × 4` | Whether each historical value is present |
| Candidate values | `N × 24 × 4 × 8` | Eight candidate forecasts for each future hour and indicator |
| Availability | `N × 24 × 4 × 8` | Whether each candidate can be used |
| Reliability | `N × 24 × 4 × 8` | How much historical data supports each candidate |
| Prediction | `N × 24 × 4` | Log forecast before inverse transform |

## Variants

`VARIANTS` defines all models. The selected research model is `A6_mixed_aug`; the public Demo uses `A0_fixed`.
