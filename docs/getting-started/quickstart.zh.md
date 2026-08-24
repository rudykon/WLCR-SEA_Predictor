# 快速开始

根据你的目标选择入口。在线 Demo 最适合快速理解项目；Python 示例面向开发者和研究人员。

## 1. 体验在线 Demo

[打开 Hugging Face Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }

选择内置样例、设定要移除多少历史数据，然后运行预测。页面将输出：

- `A0_fixed` 基线对四个指标的 24 小时预测；
- 历史和预测图；
- 当前可用候选所覆盖的范围；
- 八个候选的数值和权重；
- 预测 CSV 和 JSON 计算记录。

由于仓库没有提供训练后的 A6 检查点，Demo 使用简单的固定基线。它用于理解方法，
不能复现论文主结果。

## 2. 在 Python 中生成八个候选

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

## 3. 运行确定性季节预测

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

该工具和 Demo 对相同输入总会返回相同结果，适合检查代码和输入格式，
但不能替代论文中的训练后 WLCR-SEA 结果。

## 4. 复现实验

先查看[复现指南](../reference/reproduction.md)。其中列出了训练、缺失数据测试、速度测试和计算检查脚本。
大型数据集、训练检查点和生成结果不会存入 Git。
