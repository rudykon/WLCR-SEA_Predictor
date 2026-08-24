# Code map

This page helps developers find each implementation quickly. The repository separates model code, experiment scripts, CSV handling, result checks, manuscript sources, and the website/Demo.

| Path | Responsibility |
| --- | --- |
| `experiments/wlcr_sea_model.py` | Candidate construction, Entmax, WLCR-SEA variants, losses, metrics, and saved calculation fields |
| `experiments/missingness_protocol.py` | Removes data in fixed, repeatable patterns and records the actual rate |
| `experiments/train_wlcr_sea.py` | Runs WLCR-SEA training |
| `experiments/analyze_matched_missingness.py` | Compares models after removing the same data points |
| `experiments/audit_expert_routing.py` | Checks candidate weights and measures the effect of deleting candidates |
| `experiments/audit_request_locality.py` | Checks that the model reads only approved input fields |
| `experiments/benchmark_wlcr_sea_latency.py` | Single-thread latency and memory benchmark |
| `experiments/validate_evidence_integrity.py` | Checks that reported results agree across generated files |
| `Model/traffic_window_forecasting.py` | Six-column CSV input and fixed seasonal baseline |
| `tests/test_wlcr_sea_model.py` | Core behavior tests for the public method |
| `demo/` | Bilingual Gradio demo, synthetic sample, and Space configuration |
| `docs/` | Existing technical guides plus this bilingual MkDocs site |
| `paper/` | English and Chinese manuscript sources plus figure sources |

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

`VARIANTS` contains the fixed baseline and learned model variants evaluated in the study. The selected paper method is `A6_mixed_aug`. The public Demo uses `A0_fixed` because it does not require unpublished trained parameters.
