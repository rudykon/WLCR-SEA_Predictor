# WLCR-SEA experiment workflows

This directory contains the executable source for WLCR-SEA training,
paper-related analysis/ablation/comparison, robustness, audits, generalization,
and latency checks. It does not contain manuscript-writing or figure-generation
programs. Generated checkpoints and reports belong in a local `artifacts/`
directory and are not uploaded.

## Runtime separation

The neural workflows use PyTorch; LightGBM workflows use LightGBM. Install the
dependencies from the root requirements file, then use the runtime appropriate
for the selected workflow. Do not mix incompatible NumPy/SciPy stacks.

## Main method and controls

```bash
PYTHONPATH=. python3 experiments/train_wlcr_sea.py \
  --output artifacts/reproduction/wlcr_sea \
  --gpu-devices 0,1,2,3 \
  --seeds 42,43,44,45,46 \
  --max-epochs 100 --patience 10 \
  --batch-size 256

PYTHONPATH=. python3 experiments/train_neural_baselines.py \
  --output artifacts/reproduction/neural_baselines/mixed \
  --models dlinear,patchtst \
  --seeds 42,43,44,45,46 \
  --gpu-devices 0,1,2,3 \
  --max-epochs 100 --patience 10 \
  --batch-size 128 \
  --augmentation mixed --augmentation-rate 0.15

PYTHONPATH=. python3 experiments/train_neural_baselines.py \
  --output artifacts/reproduction/neural_baselines/grud_mixed \
  --models grud_direct \
  --seeds 42,43,44,45,46 \
  --gpu-devices 0,1,2,3 \
  --max-epochs 100 --patience 10 \
  --batch-size 128 \
  --augmentation mixed --augmentation-rate 0.15

PYTHONPATH=. python3 experiments/train_traffic_only_73d_lightgbm.py \
  --output artifacts/reproduction/lightgbm/traffic_only_73d \
  --gpu-devices 0,1,2,3
```

## Clean comparison and robustness

```bash
PYTHONPATH=. python3 experiments/analyze_paper_clean_results.py
PYTHONPATH=. python3 experiments/analyze_missingness_robustness.py \
  --gpu-devices 0,1,2,3
PYTHONPATH=. python3 experiments/evaluate_cell_disjoint_generalization.py \
  --output artifacts/reproduction/cell_disjoint_protocol_matched \
  --gpu-devices 0,1,2,3 \
  --wlcr-batch-size 256 --neural-batch-size 128
PYTHONPATH=. python3 experiments/audit_method_evidence.py \
  --gpu-devices 0,1,2,3
PYTHONPATH=. python3 experiments/benchmark_end_to_end_latency.py
```

The current analysis programs read only the registered training trace
`data/train_data.csv` and write new results under `artifacts/reproduction/`.
They do not require or open parameter/weather context or the held-out test
traffic file.

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py' -v
```

See [docs/REPRODUCTION_GUIDE.md](../docs/REPRODUCTION_GUIDE.md) for the public
repository scope and the full workflow map.
