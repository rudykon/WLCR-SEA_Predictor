# Reproduce

This page is the practical entry point for installation, input validation,
public A6 inference, tests, and the research code map. The complete authoritative
workflow lives in [`REPRODUCTION.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/REPRODUCTION.md).

## Run the A6 Demo locally

```bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
python demo/app.py
```

The app downloads a pinned Hugging Face model revision, verifies all five
checkpoint SHA-256 values, loads the models and frozen training priors once on
CPU, and automatically runs the synthetic sample. Set
`WLCR_SEA_CHECKPOINT_DIR` to a local model-repository checkout to run without a
second download.

## Input format { #input-format }

The Demo accepts exactly 336 contiguous hourly rows for one cell. Required
header:

```text
时间,小区名称,小区上行平均激活用户数,小区下行平均激活用户数,下行平均使用的PRB个数,上行平均使用的PRB个数
```

| Field | Rule |
| --- | --- |
| `时间` | `YYYY/MM/DD HH:MM`, strictly hourly and increasing |
| `小区名称` | One non-empty value shared by all rows |
| Four indicators | Finite non-negative number, `NIL`, or blank |
| Encoding | UTF-8 or UTF-8 with BOM |
| Public upload | At most 5 MB; never upload confidential operator traffic |

The forecast begins one hour after the final row and covers the next 24 hours.
`NIL` and blank fields are missing observations; the mask prevents their
placeholders from entering expert summaries.

[View the synthetic request](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/demo/examples/synthetic_traffic.csv){ .md-button }

## Model assets

The public model repository contains seeds 42–46 of `A6_mixed_aug`. Each file
stores the selected configuration and epoch, a frozen `(24, 4)` training prior,
and the CPU `state_dict`. The registered ensemble rule is an arithmetic mean of
the five predictions in linear traffic space.

[Inspect model weights](https://huggingface.co/config-h/WLCR-SEA-Predictor){ .md-button }
[Open the live Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary }

## Verify

```bash
PYTHONPATH=. python -m unittest tests.test_wlcr_sea_model -v
PYTHONPATH=. python -m unittest tests.test_hf_space_demo -v
PYTHONPATH=. python -m unittest tests.test_request_locality_audit -v
```

The A6 test independently evaluates the fixed sample through the core model
path and compares the five-member linear-space mean with the Demo runtime.

Build both website languages:

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

## Full research workflow

The research trace is not stored in Git. Place the registered file at
`data/train_data.csv`, verify its hash, install `requirements.txt`, and write
new outputs below `artifacts/reproduction/`. Full five-seed training requires
GPUs. The authoritative guide lists the exact primary training, baseline,
missingness, request-locality, cell-disjoint, auditability, and latency stages.

[Open the complete reproduction guide](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/REPRODUCTION.md){ .md-button .md-button--primary }

## Code map { #code-map }

| Path | Responsibility |
| --- | --- |
| `experiments/wlcr_sea_model.py` | Seasonal experts, masks, Entmax routing, bounded residual, losses, metrics |
| `experiments/missingness_protocol.py` | Repeatable missing-telemetry patterns |
| `experiments/train_wlcr_sea.py` | Multi-seed training, model selection, evaluation, checkpoint schema |
| `experiments/train_neural_baselines.py` | DLinear, PatchTST, and GRU-D controls |
| `experiments/audit_method_evidence.py` | Request-locality, masking, deletion, and bound audits |
| `experiments/benchmark_end_to_end_latency.py` | Matched CPU latency and model-asset size |
| `Model/traffic_window_forecasting.py` | Six-column CSV parsing and request-window validation |
| `demo/model_loader.py` | Pinned model download, integrity checks, one-time CPU load |
| `demo/runtime.py` | Five-member inference, figures, tables, CSV and JSON exports |
| `tests/test_hf_space_demo.py` | Public A6 and local/Space consistency checks |

Generated models, NumPy bundles, logs, and result folders remain outside the
versioned source. The public workflow does not contain or build unpublished
manuscript text.
