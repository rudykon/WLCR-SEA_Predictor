# 复现地图

仓库包含代码、测试、脚本和图片，不包含数据、检查点和生成结果。

## 从这里开始

1. 安装[环境](../getting-started/installation.md)。
2. 按 [`REPRODUCTION_GUIDE.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/REPRODUCTION_GUIDE.md) 操作。
3. 用 [`RESEARCH_REPRODUCTION.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/RESEARCH_REPRODUCTION.md) 检查结果。
4. 修改路径前阅读 [`CODE_STRUCTURE.md`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/docs/CODE_STRUCTURE.md)。
5. 下载并验证根 README 中的数据。

## 快速检查

```bash
PYTHONPATH=. python -m unittest tests.test_wlcr_sea_model -v
PYTHONPATH=. python -m unittest tests.test_request_locality_audit -v
PYTHONPATH=. python -m unittest tests.test_evidence_integrity -v
PYTHONPATH=. python -m unittest tests.test_rq4_evidence_sync -v
```

具备全部本地产物后：

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

## 脚本地图

| 问题 | 代表脚本 |
| --- | --- |
| 完整数据预测 | `train_wlcr_sea.py`、`analyze_paper_clean_results.py` |
| 结构化缺失 | `missingness_protocol.py`、`analyze_matched_missingness.py` |
| 路由语义 | `audit_expert_routing.py`、`audit_method_evidence.py` |
| 请求局部性 | `audit_request_locality.py` |
| 延迟与内存 | `benchmark_wlcr_sea_latency.py` |
| 训练与测试小区不重叠评估 | `evaluate_cell_disjoint_generalization.py` |
| 结果与手稿一致性 | `validate_evidence_integrity.py`、`tools/sync_rq4_evidence.py` |

## 本地产物

Git 会排除检查点、保存的模型、NumPy 文件、日志和大型结果目录。A6 结果需要按文档训练和评估，不来自 Demo。

## 网站与 Demo 检查

```bash
mkdocs build --strict
python -m unittest tests.test_hf_space_demo -v
```

GitHub Actions 运行这两项检查。Pages 发布网站，另一工作流同步 Space。
