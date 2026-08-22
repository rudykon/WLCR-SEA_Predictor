# Hugging Face 审计实验室

[启动 WLCR-SEA 请求审计实验室](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }
[查看 Demo 源码](https://github.com/rudykon/WLCR-SEA_Predictor/tree/main/demo){ .md-button target="_blank" rel="noopener" }

公开 Gradio Space 让仓库中的请求局部机制可以交互运行，同时不会虚构训练检查点。

## 实际运行内容

执行路径直接导入并使用：

- `read_traffic` 和 `split_physical_windows`：物理 CSV 契约；
- `global_corruption_mask`：确定性遥测缺失场景；
- `build_expert_batch`：八个真实季节专家；
- `WLCRSEA(VARIANTS["A0_fixed"])`：仓库登记的无参数混合；
- `bounded_audit_envelope`：结构包含检查。

固定混合的初始权重为前周 0.7、双周 0.2、七日同小时中位数 0.1。
不可用部分会被移除，剩余质量重新归一化；若三者全部不可用，则使用回退槽。

<div class="notice-card">
  <strong>不是训练后 A6 推理。</strong>仓库没有发布拟合后的 A6 检查点或冻结训练先验。Demo 最后的回退槽使用请求派生值，并用星号标记；导出的 JSON 记录明确设置 <code>paper_model: false</code>。
</div>

## 控件

| 控件 | 用途 |
| --- | --- |
| 请求 CSV | 一个小区、336 小时的物理窗口 |
| 遥测场景 | 完整、随机整小时、连续区块、最近尾部或指标异步缺失 |
| 缺失率 | 0% 至 80% 的确定性追加移除 |
| 指标 | 选择详细专家记录的指标 |
| 预测步 | 从未来 24 小时中选择一个小时检查专家 |

## 输出

- 四个历史与 24 小时预测面板；
- 可用专家最小/最大范围；
- 候选值、可用性、可靠度和路由权重；
- 精确的不可用专家质量与范围状态；
- 可下载的 24 小时预测 CSV；
- 带输入 SHA-256 的版本化 JSON 审计记录。

## 本地运行

```bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor
python -m pip install -r requirements.txt
python demo/app.py
```

敏感流量应选择本地或私有部署，不要上传到公开 Space。

## 免费 ZeroGPU 部署

Space 声明为 Gradio `zero-a10g`。一个很小的固定模型计算由 `@spaces.GPU` 包装；
模型导入保留在运行路径内部，从而确保先导入 `spaces`。GitHub Actions 会单独生成 Space README 元数据，
并在主分支更新后镜像整个仓库。

GitHub 根 README 仍是普通 Markdown，开头不会出现 YAML 元数据块或表格。
