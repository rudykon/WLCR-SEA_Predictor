# Quick start

Choose the path that matches what you want to inspect.

## 1. Explore one request interactively

[Open the Hugging Face Audit Lab](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }

Select the bundled synthetic CSV, choose a telemetry-loss mechanism, and run
the audit. You will receive:

- all four 24-hour forecasts from the registered `A0_fixed` baseline;
- a history/forecast plot and available-expert envelope;
- values, availability, reliability, and fixed routing weight for eight experts;
- a forecast CSV and a JSON audit record.

The trained A6 checkpoint is not in this repository. The interface states this
at the top and in every exported record.

## 2. Inspect the real expert builder in Python

```python
import numpy as np
from experiments import wlcr_sea_model as sea

# One request: log1p values, authoritative masks, and a frozen prior.
x = np.zeros((1, 336, 4), dtype=np.float32)
m = np.ones_like(x, dtype=bool)
prior = np.zeros((24, 4), dtype=np.float32)

experts = sea.build_expert_batch(x, m, prior)
print(experts.values.shape)        # (1, 24, 4, 8)
print(experts.availability.shape)  # (1, 24, 4, 8)
```

## 3. Run a deterministic seasonal forecast

```python
from Model.traffic_window_forecasting import (
    BaselineConfig,
    read_traffic,
    seasonal_forecast,
    split_physical_windows,
)

window = split_physical_windows(read_traffic("request.csv"))[0]
forecast = seasonal_forecast(window, BaselineConfig.default())
assert len(forecast) == 24
```

This utility and the Demo's fixed expert mixture are deterministic baselines.
They are useful for contracts and method inspection, not substitutes for the
trained WLCR-SEA results reported in the paper.

## 4. Reproduce experiments

Start with the [reproduction map](../reference/reproduction.md), then use the
tracked scripts for training, matched missingness, latency, auditability, and
evidence-integrity checks. Large data, generated checkpoints, and result
artifacts are intentionally not stored in Git.
