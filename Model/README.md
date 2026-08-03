# WLCR-SEA implementation support

WLCR-SEA 的核心网络、训练和论文相关分析位于 ../experiments/。本目录提供
共享的话务窗口、季节基线和 LightGBM 对照实现；每个文件名直接描述职责。

| 文件 | 作用 |
| --- | --- |
| traffic_window_forecasting.py | 话务 CSV 读取、336 小时请求窗口、季节基线、指标和输入完整性检查 |
| lightgbm_feature_baseline.py | LightGBM 特征矩阵、上下文表读取、基线训练和预测 |
| seasonal_baseline_config.json | 季节基线的冻结配置 |

新增或移动本目录模块时，必须同步更新其导入、相关测试和
[代码结构与命名规范](../docs/CODE_STRUCTURE.md)。生成的模型、预测和报告
应写入本地 artifacts/，不应作为源码资产保留。
