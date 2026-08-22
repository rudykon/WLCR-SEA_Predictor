# 快速开始

根据想检查的内容选择路径。

## 1. 交互检查一个请求

[打开 Hugging Face 审计实验室](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }

选择内置合成 CSV、设定遥测缺失机制并开始审计。页面将输出：

- 仓库登记的 `A0_fixed` 基线对四个指标的 24 小时预测；
- 历史/预测图和可用专家范围；
- 八个专家的数值、可用性、可靠度与固定路由权重；
- 预测 CSV 和 JSON 审计记录。

仓库没有训练后的 A6 检查点；界面顶部和每个导出记录都会明确说明。

## 2. 在 Python 中检查真实专家构造

```python
import numpy as np
from experiments import wlcr_sea_model as sea

# 一个请求：log1p 数值、权威掩码和冻结先验。
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

该工具和 Demo 中的固定专家混合都是确定性基线，适合验证契约与检查方法，
但不能替代论文报告的训练后 WLCR-SEA 结果。

## 4. 复现实验

先查看[复现地图](../reference/reproduction.md)，再使用仓库脚本完成训练、配对缺失、延迟、
可审计性和证据完整性检查。大型数据、生成的检查点和结果产物不会存入 Git。
