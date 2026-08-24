# Reproduction map

The repository keeps source, tests, bilingual paper, and rendered figures in
Git. Large datasets, fitted checkpoints, and generated result directories are
excluded and must be reconstructed from the documented workflow.

## Start here

1. Install the [research environment](../getting-started/installation.md).
2. Read [`REPRODUCTION_GUIDE.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/REPRODUCTION_GUIDE.md) for the ordered workflow.
3. Use [`RESEARCH_REPRODUCTION.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/RESEARCH_REPRODUCTION.md) for paper-oriented checks.
4. Inspect [`CODE_STRUCTURE.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/CODE_STRUCTURE.md) before changing experiment paths.
5. Download the source data from the link recorded in the root README and verify any stated hashes.

## Focused public checks

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

## Evidence families

| Question | Representative scripts |
| --- | --- |
| Clean forecasting | `train_wlcr_sea.py`, `analyze_paper_clean_results.py` |
| Structured missingness | `missingness_protocol.py`, `analyze_matched_missingness.py` |
| Routing semantics | `audit_expert_routing.py`, `audit_method_evidence.py` |
| Request locality | `audit_request_locality.py` |
| Latency and memory | `benchmark_wlcr_sea_latency.py` |
| Cell-disjoint audit | `evaluate_cell_disjoint_generalization.py` |
| Manuscript consistency | `validate_evidence_integrity.py`, `tools/sync_rq4_evidence.py` |

## Generated artifacts

The `.gitignore` excludes checkpoints (`*.pt`, `*.pth`), serialized models,
NumPy bundles, logs, and large experiment directories. Do not treat the absence
of those files from a fresh clone as evidence that the paper used the public
Demo baseline. The trained A6 evaluation must be reconstructed from its own
training and audit workflow.

## Website and Demo checks

```bash
mkdocs build --strict
python -m unittest tests.test_hf_space_demo -v
```

GitHub Actions repeats both checks before changes reach the main branch. The
Pages workflow publishes the bilingual site; a separate workflow stages Space
frontmatter and mirrors the repository to Hugging Face.
