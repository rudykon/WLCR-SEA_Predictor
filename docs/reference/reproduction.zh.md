# 复现

本页统一提供环境安装、输入校验、公开 A6 推理、测试和代码地图。完整且唯一的权威流程见
[`REPRODUCTION.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/REPRODUCTION.md)。

## 在本地运行 A6 Demo

```bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
python demo/app.py
```

应用会下载固定的 Hugging Face 模型版本，校验五个检查点的 SHA-256，在 CPU 上一次性
加载模型及冻结训练先验，并自动运行合成样例。若已下载模型仓库，可设置
`WLCR_SEA_CHECKPOINT_DIR`，避免重复下载。

## 输入格式 { #input-format }

Demo 只接收同一个小区连续 336 小时的数据。必须使用以下表头：

```text
时间,小区名称,小区上行平均激活用户数,小区下行平均激活用户数,下行平均使用的PRB个数,上行平均使用的PRB个数
```

| 字段 | 规则 |
| --- | --- |
| `时间` | `YYYY/MM/DD HH:MM`，严格按小时递增 |
| `小区名称` | 336 行使用同一个非空值 |
| 四项指标 | 有限非负数、`NIL` 或空白 |
| 编码 | UTF-8 或带 BOM 的 UTF-8 |
| 公开上传 | 不超过 5 MB；不得上传运营商机密流量 |

预测从最后一行的下一小时开始，覆盖未来 24 小时。`NIL` 和空白表示缺失观测；掩码会
阻止占位值进入专家汇总。

[查看合成请求](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/demo/examples/synthetic_traffic.csv){ .md-button }

## 模型资源

公开模型仓库包含 `A6_mixed_aug` 的种子 42–46。每个文件保存所选配置和轮次、冻结的
`(24, 4)` 训练先验以及 CPU `state_dict`。登记的集成规则是在原始流量空间对五个
预测执行算术平均。

[查看模型权重](https://huggingface.co/config-h/WLCR-SEA-Predictor){ .md-button }
[打开在线 Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary }

## 验证

```bash
PYTHONPATH=. python -m unittest tests.test_wlcr_sea_model -v
PYTHONPATH=. python -m unittest tests.test_hf_space_demo -v
PYTHONPATH=. python -m unittest tests.test_request_locality_audit -v
```

A6 测试会通过核心模型路径独立计算固定样例，再将五成员的原始流量空间平均值与 Demo
运行时输出进行比较。

严格构建中英文网站：

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

## 完整研究流程

Git 不保存研究数据。请将登记数据放到 `data/train_data.csv`，校验哈希，安装
`requirements.txt`，并把新结果写入 `artifacts/reproduction/`。完整五种子训练需要
GPU。权威指南列出了主模型训练、基线、缺失、请求内计算、小区不重叠、可审计性和延迟
各阶段的准确命令。

[打开完整复现指南](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/REPRODUCTION.md){ .md-button .md-button--primary }

## 代码地图 { #code-map }

| 路径 | 职责 |
| --- | --- |
| `experiments/wlcr_sea_model.py` | 季节专家、掩码、Entmax 路由、有界残差、损失和指标 |
| `experiments/missingness_protocol.py` | 可重复的遥测缺失模式 |
| `experiments/train_wlcr_sea.py` | 多种子训练、模型选择、评估和检查点结构 |
| `experiments/train_neural_baselines.py` | DLinear、PatchTST 和 GRU-D 对照 |
| `experiments/audit_method_evidence.py` | 请求内计算、掩码、删除和边界审计 |
| `experiments/benchmark_end_to_end_latency.py` | 匹配 CPU 延迟和模型资源大小 |
| `Model/traffic_window_forecasting.py` | 六列 CSV 解析和请求窗口校验 |
| `demo/model_loader.py` | 固定版本下载、完整性检查和一次性 CPU 加载 |
| `demo/runtime.py` | 五成员推理、图片、表格、CSV 与 JSON 导出 |
| `tests/test_hf_space_demo.py` | 公开 A6 和本地/Space 一致性检查 |

生成模型、NumPy 文件、日志和结果目录均不进入版本化源码。公开流程不包含也不构建未发表
论文正文。
