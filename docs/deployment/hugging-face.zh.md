# 在线 Demo

[打开 Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }
[源码](https://github.com/rudykon/WLCR-SEA_Predictor/tree/main/demo){ .md-button target="_blank" rel="noopener" }

Gradio 应用展示历史缺失如何改变候选、权重和预测。

## 快速体验

1. 载入 336 小时样例，运行**完整数据**。
2. 选择**最近时段缺失**，提高缺失率。
3. 选择指标和未来小时。
4. 查看候选可用性与权重。
5. 下载 CSV 和 JSON。

这里测试计算过程，不评估故障影响，也不运行训练后的 A6。

## 代码路径

- `read_traffic` + `split_physical_windows`：读取 CSV。
- `global_corruption_mask`：移除数据。
- `build_expert_batch`：生成八个候选。
- `WLCRSEA(VARIANTS["A0_fixed"])`：组合候选。
- `bounded_audit_envelope`：检查范围。

初始权重为前一周 0.7、前两周 0.2、7 日中位数 0.1。可用权重会重新归一；全部缺失时使用回退值。

<div class="notice-card">
  <strong>Demo ≠ A6。</strong>回退值来自当前输入，JSON 中 <code>paper_model: false</code>。
</div>

## 控件

| 控件 | 用途 |
| --- | --- |
| CSV | 一个小区，336 条小时记录 |
| 缺失方式 | 完整、随机、区块、最近时段或指标异步 |
| 缺失率 | 0–80%，可重复 |
| 指标 | 一项流量指标 |
| 未来小时 | 24 个预测之一 |

## 输出

- 四个历史/预测面板；
- 候选范围、数值、可用性、支持度和权重；
- 不可用候选权重与范围检查；
- 预测 CSV 和版本化 JSON。

## 本地运行

```bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor
python -m pip install -r requirements.txt
python demo/app.py
```

敏感数据请使用本地或私有部署。

## 部署

Space 使用免费 `zero-a10g`。计算由 `@spaces.GPU` 包装，GitHub Actions 会同步每次 `main` 更新。
