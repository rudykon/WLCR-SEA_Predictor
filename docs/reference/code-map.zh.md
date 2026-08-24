# 代码地图

本页帮助开发者快速找到各项功能的实现位置。仓库将模型代码、实验脚本、CSV 处理、结果检查、手稿源文件以及网站/Demo 分开组织。

| 路径 | 职责 |
| --- | --- |
| `experiments/wlcr_sea_model.py` | 候选构造、Entmax、WLCR-SEA 变体、损失函数、评估指标与计算记录字段 |
| `experiments/missingness_protocol.py` | 按固定、可重复的方式移除数据并统计实际比例 |
| `experiments/train_wlcr_sea.py` | 运行 WLCR-SEA 训练 |
| `experiments/analyze_matched_missingness.py` | 在移除相同数据点后比较不同模型 |
| `experiments/audit_expert_routing.py` | 检查候选权重并测量删除候选后的影响 |
| `experiments/audit_request_locality.py` | 检查模型是否只读取允许的输入字段 |
| `experiments/benchmark_wlcr_sea_latency.py` | 单线程延迟与内存基准 |
| `experiments/validate_evidence_integrity.py` | 检查不同生成文件中的报告结果是否一致 |
| `Model/traffic_window_forecasting.py` | 六列 CSV 输入与固定季节基线 |
| `tests/test_wlcr_sea_model.py` | 公开方法的核心行为测试 |
| `demo/` | 双语 Gradio Demo、合成示例和 Space 配置 |
| `docs/` | 既有技术指南与本双语 MkDocs 网站 |
| `paper/` | 中英文手稿与图件源文件 |

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

`VARIANTS` 包含研究所测试的固定基线和多种学习模型。论文最终选择 `A6_mixed_aug`，公开 Demo 则使用不依赖未发布训练参数的 `A0_fixed`。
