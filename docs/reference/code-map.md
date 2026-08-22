# Code map

The repository separates the public method, experiment orchestration, physical
CSV utilities, evidence audits, paper source, and website/Demo layers.

| Path | Responsibility |
| --- | --- |
| `experiments/wlcr_sea_model.py` | Expert construction, Entmax, WLCR-SEA variants, losses, metrics, audit envelope |
| `experiments/missingness_protocol.py` | Deterministic absolute cell–time corruption and rate accounting |
| `experiments/train_wlcr_sea.py` | WLCR-SEA training orchestration |
| `experiments/analyze_matched_missingness.py` | Matched robustness analysis |
| `experiments/audit_expert_routing.py` | Routing mass, deletion, and influence audits |
| `experiments/audit_request_locality.py` | Serving field allowlist and request-object invariance |
| `experiments/benchmark_wlcr_sea_latency.py` | Single-thread latency and memory benchmark |
| `experiments/validate_evidence_integrity.py` | Cross-artifact evidence consistency |
| `Model/traffic_window_forecasting.py` | Six-column CSV contract and deterministic seasonal baseline |
| `tests/test_wlcr_sea_model.py` | Focused public-method invariants |
| `demo/` | Bilingual Gradio audit lab, synthetic request, and Space metadata |
| `docs/` | Existing technical guides plus this bilingual MkDocs site |
| `paper/` | English/Chinese manuscript source, PDFs, and figure sources |

## Core tensor shapes

| Object | Shape | Meaning |
| --- | --- | --- |
| History values | `N × 336 × 4` | `log1p` traffic with arbitrary finite fills |
| History mask | `N × 336 × 4` | Authoritative observation state |
| Expert values | `N × 24 × 4 × 8` | Horizon–indicator candidate evidence |
| Availability | `N × 24 × 4 × 8` | Exact routing subset |
| Reliability | `N × 24 × 4 × 8` | Support fraction or binary support |
| Prediction | `N × 24 × 4` | Log forecast before inverse transform |

## Variants

`VARIANTS` exposes the progression from fixed and static routers to softmax,
Entmax, hard masking, reliability, bounded residuals, missingness augmentation,
consistency, and cross-indicator context. The selected paper method is
`A6_mixed_aug`; the public interactive Demo uses `A0_fixed` because it requires
no unpublished fitted parameters.
