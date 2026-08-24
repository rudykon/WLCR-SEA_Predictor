# WLCR-SEA paper reproduction guide

## Scope and canonical sources

WLCR-SEA is the only active research method in this workspace. It is a
request-local cellular traffic forecaster: a 336-hour, four-indicator history
and its observation mask yield a 24-hour forecast. The method makes its
seasonal evidence, availability masks, routing weights, baseline, and bounded
correction inspectable for each request.

This public guide documents the code, inputs, experiments, and checks needed
to reproduce the reported results.

The source code is open at
<https://github.com/rudykon/WLCR-SEA_Predictor>. The repository is licensed
under Apache-2.0; see [LICENSE](../LICENSE).

## Inputs and evidence boundary

The current paper workflows use one immutable, repository-local input:

| File | Role |
| --- | --- |
| [data/train_data.csv](../data/train_data.csv) | Cell-hour traffic trace used for training and evaluation |

`data/parameter.csv` and `data/weather.csv` belong to a historical full-context
LightGBM compatibility path. The current WLCR-SEA, neural, missingness,
cell-disjoint, and traffic-only LightGBM paper workflows do not open them.

The paper experiments start from the registered training trace. Generated
models and analysis outputs under artifacts/ are run evidence rather than
immutable source inputs; a clean checkout may omit them. Use a fresh output
directory for an independent run and keep generated files outside data/.

## Code map

| Path | Responsibility |
| --- | --- |
| [experiments/wlcr_sea_model.py](../experiments/wlcr_sea_model.py) | Core seasonal experts, masked Entmax routing, bounded residual, objectives, and metrics |
| [experiments/missingness_protocol.py](../experiments/missingness_protocol.py) | Deterministic missing-telemetry perturbation protocol |
| [experiments/train_wlcr_sea.py](../experiments/train_wlcr_sea.py) | Multi-seed WLCR-SEA fitting and ablations |
| [experiments/train_neural_baselines.py](../experiments/train_neural_baselines.py) | Controlled neural comparison runs |
| [experiments/audit_method_evidence.py](../experiments/audit_method_evidence.py) | Request-locality, availability-mask, envelope, and deletion audits |
| [experiments/benchmark_end_to_end_latency.py](../experiments/benchmark_end_to_end_latency.py) | Matched CPU latency measurement |
| [experiments/analyze_missingness_robustness.py](../experiments/analyze_missingness_robustness.py) | Final missingness comparison and paired bootstrap analysis |
| [tools/figures/render_paper_figures.R](../tools/figures/render_paper_figures.R) | Renders the data-driven manuscript figures from newly generated analysis outputs |
| [tests/test_wlcr_sea_model.py](../tests/test_wlcr_sea_model.py) | Focused method tests |

Auxiliary LightGBM code and its frozen models in Model/ are retained only to
replay the comparator shown in the manuscript. They are not an additional
active method.

## Reproduction levels

### 1. Verify the implementation

Create the Python environment described in
[requirements.txt](../requirements.txt), then run the focused unit suite:

~~~bash
PYTHONPATH=.runtime/neural:. python3 -m unittest tests.test_wlcr_sea_model -v
~~~

If the project-provided neural runtime is unavailable, install the compatible
dependencies in an isolated environment first. Do not combine the neural and
LightGBM runtimes; their numerical stacks are intentionally separate.

### 2. Refit the main method and controls

The full multi-seed run requires four compatible GPUs. Write new outputs to a
fresh directory under artifacts/. That directory is intentionally absent from
the cleaned workspace and will be created by the workflows; do not treat any
generated output as a source file.

~~~bash
PYTHONPATH=.runtime/neural:. python3 experiments/train_wlcr_sea.py \
  --output artifacts/reproduction/wlcr_sea \
  --gpu-devices 0,1,2,3 \
  --seeds 42,43,44,45,46 \
  --max-epochs 100 --patience 10 \
  --batch-size 256

PYTHONPATH=.runtime/neural:. python3 experiments/train_neural_baselines.py \
  --output artifacts/reproduction/neural_baselines/clean \
  --models dlinear,patchtst \
  --seeds 42,43,44,45,46 \
  --gpu-devices 0,1,2,3 \
  --max-epochs 100 --patience 10 \
  --batch-size 128

PYTHONPATH=.runtime/neural:. python3 experiments/train_neural_baselines.py \
  --output artifacts/reproduction/neural_baselines/mixed \
  --models dlinear,patchtst \
  --seeds 42,43,44,45,46 \
  --gpu-devices 0,1,2,3 \
  --max-epochs 100 --patience 10 \
  --batch-size 128 \
  --augmentation mixed --augmentation-rate 0.15

PYTHONPATH=.runtime/neural:. python3 experiments/train_neural_baselines.py \
  --output artifacts/reproduction/neural_baselines/grud_mixed \
  --models grud_direct \
  --seeds 42,43,44,45,46 \
  --gpu-devices 0,1,2,3 \
  --max-epochs 100 --patience 10 \
  --batch-size 128 \
  --augmentation mixed --augmentation-rate 0.15

PAPER_GPU_DEVICES=0,1,2,3 PYTHONPATH=.runtime/lightgbm:. \
python3 experiments/train_traffic_only_lightgbm.py \
  --output artifacts/reproduction/lightgbm/standard_stat \
  --round-cap 10000

PYTHONPATH=.runtime/lightgbm:. python3 experiments/train_traffic_only_73d_lightgbm.py \
  --output artifacts/reproduction/lightgbm/traffic_only_73d \
  --gpu-devices 0,1,2,3
~~~

For a short implementation check, append --smoke to the WLCR-SEA command.
Smoke outputs are diagnostic only and must not be substituted for manuscript
evidence.

### 3. Regenerate the evidence, figures, and audit records

Generated checkpoints, predictions, manifests, and experiment results may be
absent from a clean checkout. After fitting the main method and controls, run
the following commands in order. Each output must be a fresh child of
artifacts/reproduction/; do not re-use an existing output directory.

~~~bash
PYTHONPATH=.runtime/lightgbm:. python3 experiments/evaluate_lightgbm_missingness.py

PYTHONPATH=.runtime/neural:. python3 experiments/analyze_paper_clean_results.py

PYTHONPATH=.runtime/neural:. python3 experiments/audit_method_evidence.py \
  --gpu-devices 0,1,2,3

PYTHONPATH=.runtime/neural:. python3 experiments/evaluate_cell_disjoint_generalization.py \
  --output artifacts/reproduction/cell_disjoint_protocol_matched \
  --gpu-devices 0,1,2,3 \
  --wlcr-batch-size 256 \
  --neural-batch-size 128

python3 tools/sync_rq4_evidence.py \
  --evidence-root artifacts/reproduction/cell_disjoint_protocol_matched \
  --output paper/rq4_evidence.tex \
  --write

PYTHONPATH=.runtime/neural:. python3 experiments/benchmark_end_to_end_latency.py

PYTHONPATH=.runtime/neural:. python3 experiments/analyze_missingness_robustness.py \
  --gpu-devices 0,1,2,3
~~~

The analyses create the clean-data comparison, request-locality audit,
cell-disjoint refit, latency record, and missingness evidence from the new
model outputs. The cell-disjoint audit is explicitly a single-model-seed (42)
analysis: it freezes each seed-42 temporal configuration and epoch, uses WLCR
batch 256 and neural batch 128, and applies the same final-refit augmentation
seed 100042 to all three trainable methods. Each command fails if an input
hash, request identity, prediction shape, or expected artifact is inconsistent.

Then render the data-driven figures from those newly generated outputs:

~~~bash
PROJECT_ROOT="$(pwd)"
.runtime/r/run_r.sh \
  --file=tools/figures/render_paper_figures.R \
  --args \
  --clean-analysis "$PROJECT_ROOT/artifacts/reproduction/analysis/clean" \
  --audit "$PROJECT_ROOT/artifacts/reproduction/analysis/audit" \
  --revision9 "$PROJECT_ROOT/artifacts/reproduction/analysis/missingness" \
  --output "$PROJECT_ROOT/artifacts/reproduction/figures"
~~~

The figure script checks its declared inputs and hashes before export. It
intentionally writes only under artifacts/reproduction/; this prevents a
reproduction run from silently overwriting the versioned manuscript figures.
Review its revision9_figures_r_qa.json and the generated PDF/SVG/PNG/TIFF
files before any explicit, reviewed update of paper/figures. Build the English
manuscript with pdfLaTeX and the Chinese manuscript with XeLaTeX from paper/;
both use the versioned figure sources.

## Manuscript figures

paper/main.tex uses the following five figure files. The matching 300-dpi
README renders are retained in [docs/images/](images/).

| Manuscript file | README render |
| --- | --- |
| paper/figures/Scene_Diagram.pdf | images/paper_figure_scenario.png |
| paper/figures/fig_architecture.pdf | images/paper_figure_architecture.png |
| paper/figures/wlcr_sea_clean_accuracy.pdf | images/paper_figure_clean_accuracy.png |
| paper/figures/wlcr_sea_missingness.pdf | images/paper_figure_missingness.png |
| paper/figures/wlcr_sea_auditability.pdf | images/paper_figure_auditability.png |

## Integrity checks

Before reporting a reproduced result:

1. Confirm `data/train_data.csv` has not changed.
2. Inspect the output manifest.json and its recorded hashes.
3. Keep generated outputs outside data/. Preserve the five-seed protocol for
   primary temporal comparisons and the separately declared single-seed
   protocol for the complementary cell-disjoint audit.
4. Run the focused unit suite, then all six audit and analysis scripts above.
5. Run `tools/sync_rq4_evidence.py` with `--check` before compiling either
   manuscript; this validates the manifest and detects stale RQ4 values.
6. Run the R figure builder with --check-only, then the full export, and
   inspect its QA JSON plus every file listed in it.
7. Rebuild both manuscripts only after the figure-input checks pass: pdfLaTeX
   for main.tex and XeLaTeX for main_zh.tex.

This concise guide supersedes the former scattered iteration plans, review
notes, feature-schema snapshots, and submission drafts. The retained source,
documentation, inputs, and scripts — rather than stale prose snapshots or
generated outputs — are the reproducibility record.
