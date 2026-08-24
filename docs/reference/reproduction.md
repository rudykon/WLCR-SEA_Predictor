# Reproduction map

The repository includes source code, tests, research scripts, and figures. Large datasets, trained checkpoints, and generated result folders are not stored in Git. Follow the steps below to recreate the required research artifacts locally.

## Start here

1. Install the [research environment](../getting-started/installation.md).
2. Read [`REPRODUCTION_GUIDE.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/REPRODUCTION_GUIDE.md) for the ordered workflow.
3. Use [`RESEARCH_REPRODUCTION.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/RESEARCH_REPRODUCTION.md) for paper-oriented checks.
4. Inspect [`CODE_STRUCTURE.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/CODE_STRUCTURE.md) before changing experiment paths.
5. Download the source data from the link recorded in the root README and verify any stated hashes.

## Quick checks that work after a fresh clone

```bash
PYTHONPATH=. python -m unittest tests.test_wlcr_sea_model -v
PYTHONPATH=. python -m unittest tests.test_request_locality_audit -v
PYTHONPATH=. python -m unittest tests.test_evidence_integrity -v
PYTHONPATH=. python -m unittest tests.test_rq4_evidence_sync -v
```

Run the complete suite when the required local artifacts are available:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

## Which script answers which question

| Question | Representative scripts |
| --- | --- |
| Clean forecasting | `train_wlcr_sea.py`, `analyze_paper_clean_results.py` |
| Structured missingness | `missingness_protocol.py`, `analyze_matched_missingness.py` |
| Routing semantics | `audit_expert_routing.py`, `audit_method_evidence.py` |
| Request locality | `audit_request_locality.py` |
| Latency and memory | `benchmark_wlcr_sea_latency.py` |
| Cell-disjoint audit | `evaluate_cell_disjoint_generalization.py` |
| Manuscript consistency | `validate_evidence_integrity.py`, `tools/sync_rq4_evidence.py` |

## Files that must be generated locally

The `.gitignore` excludes checkpoints (`*.pt`, `*.pth`), saved models, NumPy bundles, logs, and large experiment folders. Their absence after cloning is expected. The paper's A6 results require the documented A6 training and evaluation process; they do not come from the public Demo.

## Website and Demo checks

```bash
mkdocs build --strict
python -m unittest tests.test_hf_space_demo -v
```

GitHub Actions repeats both checks before changes reach the main branch. The Pages workflow publishes the bilingual site, while a separate workflow adds the Space frontmatter and mirrors the repository to Hugging Face.
