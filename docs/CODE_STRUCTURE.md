# Code structure and naming

The public repository contains the WLCR-SEA method, its paper-related
analysis/ablation/comparison programs, tests, and reproducibility documentation.
Manuscript sources, writing/figure-generation code, data, checkpoints, and
generated outputs are kept outside the upload boundary.

## Layers

| Directory | Responsibility |
| --- | --- |
| `Model/` | Request-window parsing, metrics, seasonal baselines, and LightGBM feature baseline |
| `experiments/` | WLCR-SEA training, baseline training, ablations, robustness analysis, audits, generalization, and latency |
| `tests/` | Unit tests for the method, baselines, metrics, and evidence contracts |
| `docs/` | Public reproduction instructions and code maps |

## Main modules

| File | Role |
| --- | --- |
| `Model/traffic_window_forecasting.py` | Traffic-window parsing, seasonal baselines, metrics, and input contracts |
| `Model/lightgbm_feature_baseline.py` | Feature construction and LightGBM baseline support |
| `experiments/wlcr_sea_model.py` | Expert construction, availability masking, sparse routing, bounded residual, losses, and metrics |
| `experiments/train_wlcr_sea.py` | Multi-seed WLCR-SEA training and A0–A6 ablation variants |
| `experiments/train_neural_baselines.py` | DLinear, PatchTST, and GRU-D controls |
| `experiments/train_lightgbm_baseline.py` | Statistical LightGBM baseline training and nested selection |
| `experiments/analyze_paper_clean_results.py` | Clean comparison table, including A1–A6 |
| `experiments/analyze_missingness_robustness.py` | Missingness stress and paired robustness analysis |
| `experiments/audit_method_evidence.py` | Routing, locality, deletion, and structural evidence audit |

## Naming rules

- `train_`: model fitting or baseline training.
- `analyze_`: clean, robustness, ablation, or comparison analysis.
- `audit_`: integrity, locality, routing, or metric audits.
- `evaluate_`: generalization or controlled evaluation.
- `benchmark_`: feature or inference latency.
- `verify_` / `validate_`: consistency and evidence checks.

New generated models, reports, predictions, and outputs must remain under a
local `artifacts/` directory and must not be committed.
