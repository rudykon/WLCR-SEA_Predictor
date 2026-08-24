# Quick start

Choose what you want to do. The Live Demo is the fastest way to understand the
project; the Python examples are for developers and researchers.

## 1. Try the Live Demo

[Open the Hugging Face Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }

Select the built-in sample, choose how much history to remove, and run the
forecast. You will receive:

- four 24-hour forecasts from the `A0_fixed` baseline;
- plots of the history and forecast;
- the range covered by currently usable candidates;
- values and weights for all eight candidates;
- a forecast CSV and a JSON calculation record.

The Demo uses a simple fixed baseline because the trained A6 checkpoint is not
included in the repository. It is for understanding the method, not reproducing
the paper's main result.

## 2. Build the eight candidates in Python

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

This utility and the Demo always return the same result for the same input.
They are useful for checking code and input format, but they do not replace the
trained WLCR-SEA results in the paper.

## 4. Reproduce experiments

Start with the [reproduction guide](../reference/reproduction.md). It points to
the scripts for training, missing-data tests, speed tests, and calculation
checks. Large datasets, trained checkpoints, and generated results are not
stored in Git.
