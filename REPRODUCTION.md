# WLCR-SEA reproduction guide

This is the single authoritative guide for running the public five-model predictor,
checking the implementation, and regenerating the research evidence. Commands
assume a clean clone at the repository root. The repository does not include
training data or generated experiment artifacts.

## 1. Environments

Use Python 3.11 or newer; CI and the public Demo use Python 3.12.

For public ensemble inference and the Gradio Demo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
python demo/app.py
```

For training, baselines, and analysis, install the research environment instead:

```bash
python -m pip install -r requirements.txt
```

Select a PyTorch build that matches the training host. Full five-seed training
uses GPUs; unit tests and public inference run on CPU.

## 2. Public model assets

The Demo downloads five checkpoints from
[`config-h/WLCR-SEA-Predictor`](https://huggingface.co/config-h/WLCR-SEA-Predictor)
at pinned revision `eb4447f4ebab8f9caa003d92c838ed8e750963bd`. Startup verifies
the registered SHA-256 for every file before loading it. Each checkpoint
contains its seed, selected architecture, selected epoch, frozen `(24, 4)`
training prior, and `state_dict`.

The checkpoint filenames retain the paper's internal experiment identifier
`A6_mixed_aug` so that hashes and result manifests remain reproducible. The
reader-facing name is **WLCR-SEA five-model ensemble**.

To use previously downloaded files, place them under a local `checkpoints/`
directory and set:

```bash
export WLCR_SEA_CHECKPOINT_DIR=/path/to/model-repository
python demo/app.py
```

The primary prediction is the arithmetic mean of all five forecast arrays in
linear traffic space. The audit JSON records the model revision, hashes,
configuration, seed, per-member output, expert values, weights, residuals, and
bound checks.

## 3. Input data

### Inference request

The Demo accepts one UTF-8 CSV containing exactly 336 contiguous hourly rows
for one cell. The required header is:

```text
时间,小区名称,小区上行平均激活用户数,小区下行平均激活用户数,下行平均使用的PRB个数,上行平均使用的PRB个数
```

- Time format: `YYYY/MM/DD HH:MM`, strictly increasing by one hour.
- Cell name: one non-empty value shared by every row.
- Indicators: finite non-negative numbers; `NIL` or blank means missing.
- Public upload limit: 5 MB.

`demo/examples/synthetic_traffic.csv` is deterministic synthetic data and
contains no operator traffic.

### Research trace

Download the registered trace from the link in the project website and place
it at `data/train_data.csv`. The current code registers SHA-256
`d274407a3db51ba4871851ab447bcc75202bb567337464d85ea280662f3bf1da`.
Do not commit the trace, credentials, held-out traffic, checkpoints, or generated
artifacts.

## 4. Verification

Run the focused implementation and public-ensemble checks:

```bash
PYTHONPATH=. python -m unittest tests.test_wlcr_sea_model -v
PYTHONPATH=. python -m unittest tests.test_hf_space_demo -v
PYTHONPATH=. python -m unittest tests.test_request_locality_audit -v
PYTHONPATH=. python -m unittest tests.test_evidence_integrity -v
```

Build both website languages in strict mode:

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

The Space consistency test independently reconstructs the fixed sample through the core
evaluation path and compares its five-member linear-space mean with the Space
runtime output.

## 5. Refit the research models

Write every independent run to a fresh child of `artifacts/reproduction/`.
For the paper's internal A0–A6 experiment sweep:

```bash
python experiments/train_wlcr_sea.py \
  --output artifacts/reproduction/wlcr_sea \
  --gpu-devices 0,1,2,3 \
  --seeds 42,43,44,45,46 \
  --max-epochs 100 \
  --patience 10 \
  --batch-size 256
```

Append `--smoke` for a short implementation check. Smoke results are diagnostic
and must not replace the registered evidence.

Train the controlled neural baselines with their declared batch size:

```bash
python experiments/train_neural_baselines.py \
  --output artifacts/reproduction/neural_baselines/clean \
  --models dlinear,patchtst \
  --seeds 42,43,44,45,46 \
  --gpu-devices 0,1,2,3 \
  --max-epochs 100 \
  --patience 10 \
  --batch-size 128
```

The traffic-only LightGBM and GRU-D controls have separate `train_*.py`
entrypoints. Run `python <script> --help` before launching them and preserve the
registered seeds, augmentation mode, and batch sizes recorded by each output
manifest.

## 6. Regenerate evidence

After the required model artifacts exist, run the registered analysis and audit
programs from the repository root:

```bash
python experiments/analyze_paper_clean_results.py
python experiments/analyze_missingness_robustness.py --gpu-devices 0,1,2,3
python experiments/audit_method_evidence.py --gpu-devices 0,1,2,3
python experiments/evaluate_cell_disjoint_generalization.py \
  --output artifacts/reproduction/cell_disjoint_protocol_matched \
  --gpu-devices 0,1,2,3 \
  --wlcr-batch-size 256 \
  --neural-batch-size 128
python experiments/benchmark_end_to_end_latency.py
```

These programs validate expected files, shapes, identities, and hashes before
writing clean-data, missingness, request-locality, cell-disjoint, auditability,
and latency evidence. This public repository intentionally does not contain or
build the unpublished manuscript source.

## 7. Code map

| Path | Responsibility |
| --- | --- |
| `experiments/wlcr_sea_model.py` | Eight experts, hard-masked Entmax routing, bounded residual, losses, and metrics |
| `experiments/missingness_protocol.py` | Deterministic missing-telemetry perturbations |
| `experiments/train_wlcr_sea.py` | Multi-seed fitting, selection, evaluation, and checkpoint format |
| `experiments/train_neural_baselines.py` | DLinear, PatchTST, and GRU-D controls |
| `experiments/audit_method_evidence.py` | Request-locality, masking, deletion, and envelope audits |
| `experiments/benchmark_end_to_end_latency.py` | Matched CPU latency and asset-size measurement |
| `demo/model_loader.py` | Pinned download, SHA-256 verification, and one-time ensemble loading |
| `demo/runtime.py` | CSV-to-ensemble inference, plots, tables, and audit exports |
| `tests/test_hf_space_demo.py` | Public checkpoint and local/Space consistency checks |

## 8. Reporting checklist

1. Record the Git commit, environment, data hash, and model revision.
2. Preserve the five seeds and linear-space ensemble rule for primary results.
3. Use fresh output directories and inspect every generated manifest.
4. Label WAPE as macro-cell, pooled, or macro-indicator; never report an
   unqualified value.
5. Keep the complementary cell-disjoint audit separate from the primary
   temporal comparison.
6. Treat routing weights as internal allocation records, not causal effects or
   calibrated uncertainty.
