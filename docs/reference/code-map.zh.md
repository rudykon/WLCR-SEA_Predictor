# 代码地图

关键路径一览。

| 路径 | 职责 |
| --- | --- |
| `experiments/wlcr_sea_model.py` | 候选、Entmax、变体、损失、指标、记录 |
| `experiments/missingness_protocol.py` | 可重复数据移除 |
| `experiments/train_wlcr_sea.py` | 训练 |
| `experiments/analyze_matched_missingness.py` | 相同缺失下的对比 |
| `experiments/audit_expert_routing.py` | 权重与删除检查 |
| `experiments/audit_request_locality.py` | 输入字段检查 |
| `experiments/benchmark_wlcr_sea_latency.py` | 延迟与内存 |
| `experiments/validate_evidence_integrity.py` | 结果一致性 |
| `Model/traffic_window_forecasting.py` | CSV 输入与季节性基线 |
| `tests/test_wlcr_sea_model.py` | 核心测试 |
| `demo/` | Gradio 应用与样例 |
| `docs/` | 指南与网站 |
| `paper/figures/` | 图片源文件 |

## 核心张量形状

| 对象 | 形状 | 含义 |
| --- | --- | --- |
| 历史数值 | `N × 336 × 4` | `log1p` 变换后的流量；缺失位置可能含占位数 |
| 历史掩码 | `N × 336 × 4` | 每个历史数值是否存在 |
| 候选值 | `N × 24 × 4 × 8` | 每个未来小时和指标的八个候选预测 |
| 可用性 | `N × 24 × 4 × 8` | 每个候选是否能够使用 |
| 可靠度 | `N × 24 × 4 × 8` | 每个候选获得多少历史数据支持 |
| 预测 | `N × 24 × 4` | 逆变换前的对数预测 |

## 变体

`VARIANTS` 定义全部模型。研究选择 `A6_mixed_aug`，公开 Demo 使用 `A0_fixed`。
