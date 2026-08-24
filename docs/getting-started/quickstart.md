# Quick start

Start with the Demo, or jump to Python.

## 1. Demo

[Open the Hugging Face Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }

Load the sample, remove some history, and run:

- four 24-hour forecasts;
- history and forecast plots;
- candidate ranges, values, and weights;
- forecast CSV and calculation JSON.

The Demo uses `A0_fixed`, not the trained A6 model.

## 2. Candidates

```python
import numpy as np
from experiments import wlcr_sea_model as sea

# One input: log1p values, present/missing markers, and a training prior.
x = np.zeros((1, 336, 4), dtype=np.float32)
m = np.ones_like(x, dtype=bool)
prior = np.zeros((24, 4), dtype=np.float32)

experts = sea.build_expert_batch(x, m, prior)
print(experts.values.shape)        # (1, 24, 4, 8)
print(experts.availability.shape)  # (1, 24, 4, 8)
```

## 3. Baseline

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

The same input returns the same result. This checks code and format; it does not reproduce A6.

## 4. Experiments

See the [reproduction map](../reference/reproduction.md) for training, outages, speed, and checks. Large data and checkpoints are not in Git.
