# Hugging Face 在线 Demo

[启动 WLCR-SEA Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }
[查看 Demo 源码](https://github.com/rudykon/WLCR-SEA_Predictor/tree/main/demo){ .md-button target="_blank" rel="noopener" }

公开的 Gradio 应用展示 WLCR-SEA 如何利用历史规律，以及部分历史数据缺失时预测会发生怎样的变化。

## 五分钟上手

假设某个小区的最新测量数据突然停止上报，但规划人员仍需要估计明天的需求。你可以通过 Demo 直接观察方法如何处理这种情况：

1. 载入内置的 336 小时合成数据，先运行**完整数据**场景；
2. 切换到**最近一段数据缺失**场景，逐步提高缺失比例；
3. 选择一项指标和一个未来小时，观察哪些历史候选随之消失；
4. 确认不可用候选获得的权重严格为零；
5. 将预测与可用候选范围比较，再下载预测 CSV 和 JSON 计算记录。

这个测试回答一个实际问题：*部分历史数据缺失时，方法会如何处理？* 它不会评估真实故障带来的业务影响，也不会运行检查点尚未公开的 A6 训练模型。

## Demo 使用的代码

界面直接调用仓库中的真实代码完成计算，执行路径包括：

- `read_traffic` 和 `split_physical_windows`：读取并验证 CSV；
- `global_corruption_mask`：按可重复的方式移除数据；
- `build_expert_batch`：生成八个历史候选；
- `WLCRSEA(VARIANTS["A0_fixed"])`：使用固定权重组合候选；
- `bounded_audit_envelope`：检查预测是否位于允许范围内。

初始权重分别为：前一周 0.7、前两周 0.2、7 日同小时中位数 0.1。某个候选不可用时，其余权重会按比例调整，使总和保持为 1；三者全部不可用时，方法会使用回退值。

<div class="notice-card">
  <strong>这不是论文中的 A6 训练模型。</strong>仓库没有提供 A6 检查点或训练集先验。Demo 最后的回退值从当前输入计算，并用星号标记；导出的 JSON 明确设置 <code>paper_model: false</code>。
</div>

## 控件

| 控件 | 用途 |
| --- | --- |
| 请求 CSV | 单个小区连续 336 小时的数据，每小时一行 |
| 缺失方式 | 完整数据、随机小时缺失、连续区块缺失、最近时段缺失，或不同指标在不同时间缺失 |
| 缺失率 | 以可重复方式额外移除 0% 至 80% 的数值 |
| 指标 | 选择要检查的流量指标 |
| 未来小时 | 从未来 24 个预测中选择一个进行检查 |

## 输出

- 四个面板，分别展示各项指标的历史数据和未来 24 小时预测；
- 当前可用候选的最小值和最大值；
- 候选值、可用性、支持程度和权重；
- 分给不可用候选的总权重和范围检查结果；
- 可下载的 24 小时预测 CSV；
- 带输入 SHA-256 的版本化 JSON 计算记录。

## 本地运行

```bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor
python -m pip install -r requirements.txt
python demo/app.py
```

如需处理敏感流量数据，请使用本地或私有部署，不要将其上传到公开 Space。

## 面向维护者的部署信息

Space 使用免费的 Gradio `zero-a10g` 硬件。模型计算由 `@spaces.GPU` 包装；每次 `main` 更新后，GitHub Actions 都会把仓库同步到 Space。

GitHub README 保持为标准 Markdown；Space 专用元数据仅在部署时添加。
