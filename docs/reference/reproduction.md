# Reproduction map

Code, tests, scripts, and figures are included. Data, checkpoints, and generated results are not.

## Start here

1. Install the [environment](../getting-started/installation.md).
2. Follow [`REPRODUCTION_GUIDE.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/REPRODUCTION_GUIDE.md).
3. Use [`RESEARCH_REPRODUCTION.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/RESEARCH_REPRODUCTION.md) for result checks.
4. Read [`CODE_STRUCTURE.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/CODE_STRUCTURE.md) before changing paths.
5. Download and verify the source data listed in the root README.

## Quick checks

```bash
PYTHONPATH=. python -m unittest tests.test_wlcr_sea_model -v
PYTHONPATH=. python -m unittest tests.test_request_locality_audit -v
PYTHONPATH=. python -m unittest tests.test_evidence_integrity -v
PYTHONPATH=. python -m unittest tests.test_rq4_evidence_sync -v
```

With all local artifacts:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

## Script map

| Question | Representative scripts |
| --- | --- |
| Clean forecasting | `train_wlcr_sea.py`, `analyze_paper_clean_results.py` |
| Structured missingness | `missingness_protocol.py`, `analyze_matched_missingness.py` |
| Routing semantics | `audit_expert_routing.py`, `audit_method_evidence.py` |
| Request locality | `audit_request_locality.py` |
| Latency and memory | `benchmark_wlcr_sea_latency.py` |
| Cell-disjoint audit | `evaluate_cell_disjoint_generalization.py` |
| Manuscript consistency | `validate_evidence_integrity.py`, `tools/sync_rq4_evidence.py` |

## Local artifacts

Git excludes checkpoints, saved models, NumPy bundles, logs, and large result folders. A6 results require the documented training and evaluation; they do not come from the Demo.

## Website and Demo checks

```bash
mkdocs build --strict
python -m unittest tests.test_hf_space_demo -v
```

GitHub Actions runs both checks. Pages publishes the site; another workflow syncs the Space.
