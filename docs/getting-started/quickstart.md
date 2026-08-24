# Quick start

Choose the entry point that matches your goal. The Live Demo is the fastest way to understand the project, while the Python examples are intended for developers and researchers.

## 1. Try the Live Demo

[Open the Hugging Face Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }

Select the built-in sample, choose how much history to remove, and run the forecast. The Demo returns:

- four 24-hour forecasts from the `A0_fixed` baseline;
- plots of the history and forecast;
- the range covered by currently usable candidates;
- values and weights for all eight candidates;
- a forecast CSV and a JSON calculation record.

The Demo uses a simple fixed baseline because the trained A6 checkpoint is not included in the repository. It demonstrates how the method works; it does not reproduce the paper's main results.

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

## 3. Run a reproducible seasonal baseline

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

This utility and the Demo return the same result whenever the input is unchanged. They are useful for checking the code and input format, but they do not replace the trained WLCR-SEA results reported in the paper.

## 4. Reproduce experiments

Start with the [reproduction guide](../reference/reproduction.md). It links to the scripts for training, missing-data evaluation, speed tests, and calculation checks. Large datasets, trained checkpoints, and generated results are not stored in Git.
