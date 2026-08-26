---
license: apache-2.0
library_name: pytorch
pipeline_tag: time-series-forecasting
tags:
  - time-series
  - traffic-forecasting
  - cellular-networks
  - pytorch
---

# WLCR-SEA Predictor

WLCR-SEA (Window-Local Context Representation with Seasonal Expert Attention)
forecasts cellular traffic using only the target cell's request history and
frozen model assets. Each request contains an ordered 336-hour history and
observation mask. The model forecasts the next 24 hours for four indicators:

1. uplink active users;
2. downlink active users;
3. average used downlink PRBs;
4. average used uplink PRBs.

The two PRB fields are counts of average used physical resource blocks, not
utilization percentages. WLCR-SEA routes between eight seasonal experts with
hard availability masking, reliability-aware Entmax routing, and a bounded
residual.

## Public five-model predictor

The `checkpoints/` directory contains the five members used by the public
predictor. Each checkpoint stores `schema_version`, `experiment_version`, its
seed, selected configuration and epoch, a frozen `(24, 4)` training prior, and
the CPU `state_dict`. `SHA256SUMS` records their integrity hashes.

The artifact filenames retain the paper's internal experiment identifier so
that the release can be mapped exactly to research manifests:

| File | Seed | Selected configuration | Epoch |
| --- | ---: | --- | ---: |
| `A6_mixed_aug_seed42.pt` | 42 | `d16_h32_lr1e3_delta025` | 67 |
| `A6_mixed_aug_seed43.pt` | 43 | `d32_h64_lr5e4_delta050` | 40 |
| `A6_mixed_aug_seed44.pt` | 44 | `d16_h32_lr1e3_delta025` | 65 |
| `A6_mixed_aug_seed45.pt` | 45 | `d32_h64_lr5e4_delta050` | 56 |
| `A6_mixed_aug_seed46.pt` | 46 | `d32_h64_lr5e4_delta050` | 65 |

The reader-facing name is **WLCR-SEA five-model ensemble**. Its output is the
arithmetic mean of the five member forecasts after inverse transformation to
linear traffic space.

## Load the same predictor as the Demo

Clone the
[implementation repository](https://github.com/rudykon/WLCR-SEA_Predictor),
install `requirements-demo.txt`, and use the verified ensemble loader:

```python
from demo.model_loader import load_ensemble
from demo.runtime import run_forecast

ensemble = load_ensemble()
result = run_forecast("request.csv", ensemble=ensemble)
forecast = result.prediction  # shape: (24, 4), linear traffic space
```

The public runtime pins model revision
[`eb4447f4ebab`](https://huggingface.co/config-h/WLCR-SEA-Predictor/tree/eb4447f4ebab8f9caa003d92c838ed8e750963bd),
verifies every checkpoint SHA-256, and checks model metadata before loading.
Request parsing, observation-mask handling, five-member aggregation, and audit
export all use the same code path as the public Demo.

## Evaluation snapshot

On the project's reported holdout workflow, the five-model predictor reports
macro-cell WAPE `0.177612`, pooled WAPE `0.184915`, and macro-indicator WAPE
`0.195511`. These are different aggregations of one dataset and protocol, and
they are not general deployment guarantees.

## Intended use and limitations

This release supports research and reproducibility for request-local cellular
traffic forecasting. It is not a general-purpose pretrained time-series model
and should be validated on the target network before operational use. Training
data is not included, and the source dataset's terms are separate from this
model release. PyTorch `.pt` files use pickle-based serialization; load only
trusted files and verify their checksums.

Routing weights describe internal allocation. They are not causal effects,
feature attributions, or calibrated uncertainty estimates.

## License

Apache License 2.0. See `LICENSE`.
