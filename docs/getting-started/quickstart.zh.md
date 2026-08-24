# 快速开始

先体验 Demo，或直接运行 Python。

## 1. Demo

[打开 Hugging Face Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }

载入样例，移除部分历史，再运行预测：

- 四项 24 小时预测；
- 历史与预测图；
- 候选范围、数值和权重；
- 预测 CSV 和计算记录 JSON。

Demo 使用 `A0_fixed`，不是训练后的 A6 模型。

## 2. 候选

```python
import numpy as np
from experiments import wlcr_sea_model as sea

# 一个输入：log1p 数值、存在/缺失标记和训练先验。
x = np.zeros((1, 336, 4), dtype=np.float32)
m = np.ones_like(x, dtype=bool)
prior = np.zeros((24, 4), dtype=np.float32)

experts = sea.build_expert_batch(x, m, prior)
print(experts.values.shape)        # (1, 24, 4, 8)
print(experts.availability.shape)  # (1, 24, 4, 8)
```

## 3. 基线

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

输入不变，结果不变。它可检查代码和格式，但不能复现 A6。

## 4. 实验

查看[复现地图](../reference/reproduction.md)，了解训练、缺失测试、速度和检查脚本。大型数据和检查点不在 Git 中。
