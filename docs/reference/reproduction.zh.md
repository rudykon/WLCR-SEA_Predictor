# 复现地图

仓库把源码、测试、中英文论文和渲染图保存在 Git 中。大型数据集、拟合检查点与生成结果目录被排除，
需要按照文档工作流重新构建。

## 从这里开始

1. 安装[研究环境](../getting-started/installation.md)。
2. 按 [`REPRODUCTION_GUIDE.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/REPRODUCTION_GUIDE.md) 执行有序工作流。
3. 使用 [`RESEARCH_REPRODUCTION.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/RESEARCH_REPRODUCTION.md) 完成面向论文的检查。
4. 修改实验路径前阅读 [`CODE_STRUCTURE.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/CODE_STRUCTURE.md)。
5. 从根 README 记录的链接下载源数据，并验证文档列出的哈希值。

## 聚焦公开检查

```bash
PYTHONPATH=. python -m unittest tests.test_wlcr_sea_model -v
PYTHONPATH=. python -m unittest tests.test_request_locality_audit -v
PYTHONPATH=. python -m unittest tests.test_evidence_integrity -v
PYTHONPATH=. python -m unittest tests.test_rq4_evidence_sync -v
```

具备所需本地产物时运行完整测试：

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

## 证据类别

| 问题 | 代表脚本 |
| --- | --- |
| 洁净预测 | `train_wlcr_sea.py`、`analyze_paper_clean_results.py` |
| 结构化缺失 | `missingness_protocol.py`、`analyze_matched_missingness.py` |
| 路由语义 | `audit_expert_routing.py`、`audit_method_evidence.py` |
| 请求局部性 | `audit_request_locality.py` |
| 延迟与内存 | `benchmark_wlcr_sea_latency.py` |
| 小区不相交审计 | `evaluate_cell_disjoint_generalization.py` |
| 论文一致性 | `validate_evidence_integrity.py`、`tools/sync_rq4_evidence.py` |

## 生成产物

`.gitignore` 会排除检查点（`*.pt`、`*.pth`）、序列化模型、NumPy 包、日志和大型实验目录。
全新克隆中缺少这些文件，不代表论文使用了公开 Demo 基线。训练后的 A6 评估必须通过其训练与审计工作流重建。

## 网站与 Demo 检查

```bash
mkdocs build --strict
python -m unittest tests.test_hf_space_demo -v
```

GitHub Actions 会在改动进入主分支前重复两项检查。Pages 工作流发布双语网站；
另一个工作流单独生成 Space 元数据并把仓库镜像到 Hugging Face。
