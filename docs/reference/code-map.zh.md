# 代码地图

仓库将公开方法、实验编排、物理 CSV 工具、证据审计、论文源码和网站/Demo 层分开组织。

| 路径 | 职责 |
| --- | --- |
| `experiments/wlcr_sea_model.py` | 专家构造、Entmax、WLCR-SEA 变体、损失、指标与审计范围 |
| `experiments/missingness_protocol.py` | 确定性绝对小区—时间损坏及比率统计 |
| `experiments/train_wlcr_sea.py` | WLCR-SEA 训练编排 |
| `experiments/analyze_matched_missingness.py` | 配对鲁棒性分析 |
| `experiments/audit_expert_routing.py` | 路由质量、删除与影响审计 |
| `experiments/audit_request_locality.py` | 服务字段白名单和请求对象不变性 |
| `experiments/benchmark_wlcr_sea_latency.py` | 单线程延迟与内存基准 |
| `experiments/validate_evidence_integrity.py` | 跨产物证据一致性 |
| `Model/traffic_window_forecasting.py` | 六列 CSV 契约与确定性季节基线 |
| `tests/test_wlcr_sea_model.py` | 公开方法聚焦不变量 |
| `demo/` | 双语 Gradio 审计实验室、合成请求和 Space 元数据 |
| `docs/` | 既有技术指南与本双语 MkDocs 网站 |
| `paper/` | 中英文论文源码、PDF 和图件源文件 |

## 核心张量形状

| 对象 | 形状 | 含义 |
| --- | --- | --- |
| 历史数值 | `N × 336 × 4` | `log1p` 流量，缺失位置可有任意有限填充 |
| 历史掩码 | `N × 336 × 4` | 权威观测状态 |
| 专家值 | `N × 24 × 4 × 8` | 预测步—指标候选证据 |
| 可用性 | `N × 24 × 4 × 8` | 精确路由子集 |
| 可靠度 | `N × 24 × 4 × 8` | 支持比例或二元支持 |
| 预测 | `N × 24 × 4` | 逆变换前的对数预测 |

## 变体

`VARIANTS` 展示了从固定/静态路由到 Softmax、Entmax、硬掩码、可靠度、有界残差、
缺失增广、一致性和跨指标上下文的递进。论文所选方法为 `A6_mixed_aug`；
公开交互 Demo 使用不需要未发布拟合参数的 `A0_fixed`。
