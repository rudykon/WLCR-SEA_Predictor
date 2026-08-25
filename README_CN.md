<p align="center">
  <img src="docs/assets/brand/logo-horizontal.svg" width="540" alt="WLCR-SEA Predictor 项目 Logo">
</p>

<p align="center">
  <strong>面向缺失遥测的单小区请求内流量预测，采用严格缺失掩码和可审计专家路由。</strong>
</p>

<p align="center">
  <a href="https://rudykon.github.io/WLCR-SEA_Predictor/zh/">项目网站</a> ·
  <a href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor">A6 Demo</a> ·
  <a href="https://huggingface.co/config-h/WLCR-SEA-Predictor">模型权重</a> ·
  <a href="README.md">English</a>
</p>

WLCR-SEA 读取单个小区过去 **336 小时 × 4 项流量指标**，预测未来
**24 小时 × 4 项指标**。推理阶段只使用当前请求和冻结模型资源，不读取其他小区的
实时流量。公开 Demo 运行项目报告的五检查点 `A6_mixed_aug` 集成。

## 为什么使用 WLCR-SEA

- **请求内计算：**每次预测都是封闭的单小区计算。小区标识只用于校验和回放，不用于
  查询其他实时数据源。
- **感知缺失：**八个季节专家覆盖日周期、周周期、稳健中位数、趋势、请求汇总和冻结
  训练先验。依赖缺失历史的专家经过严格掩码后，路由权重必定为零。
- **可审计：**可导出专家值、可用性、可靠度、路由权重、有界残差、检查点身份和边界
  检查。这里的权重解释模型内部如何分配信息，不代表因果解释或不确定性。

这是面向受限数据访问与缺失遥测的研究实现，不是“任何数据上都最准确”的通用预测器，
也不等同于隐私保护系统。

## 方法

<p align="center">
  <a href="docs/images/paper_figure_architecture.png">
    <img src="docs/images/paper_figure_architecture.png" width="94%" alt="WLCR-SEA 架构：八个季节专家、严格可用性掩码、稀疏路由、有界残差、预测与审计记录">
  </a>
</p>

模型先从 336 小时请求和每个检查点的冻结训练先验构造八个季节专家；再只对可用专家
执行带可靠度的 Entmax 路由，使不可用专家保持零权重；最后使用有界残差得到未来
24 小时预测，并保留可检查的预测边界。五个 A6 成员在原始流量空间逐元素平均。

完整专家定义、可用条件和公式见[方法页面](https://rudykon.github.io/WLCR-SEA_Predictor/zh/guide/method/)。

## 证据

| 问题 | 已登记结论 |
| --- | --- |
| 完整历史 | 当前对比中 DLinear 的宏指标 WAPE 更低：**0.1854**；WLCR-SEA 五模型集成为 **0.1955**。 |
| 严重缺失 | 在研究设定的严重缺失场景下，WLCR-SEA 在与 DLinear-Aug、PatchTST-Aug、GRU-D 的 **9 项预设对比**中宏指标 WAPE 均更低。 |
| 审计检查 | 不可用专家权重为 **0**，报告的预测边界违规为 **0**。 |
| CPU 推理 | 单线程、batch=1 时，五模型集成中位/P99 延迟为 **34.705/38.684 ms**，模型资源共 **148.8 KiB**。 |

这些结果来自一个匿名地区、约一个月的数据，不能直接视为部署保证。完整协议、置信区间、
缺失模式和适用边界见[实验结果](https://rudykon.github.io/WLCR-SEA_Predictor/zh/research/evidence/)。

## 运行

```bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor
python -m pip install -r requirements-demo.txt
python demo/app.py
```

应用会校验并缓存五个公开 A6 检查点，自动运行一个 336 小时合成样例，并导出预测
CSV 与带版本的审计 JSON。公开 Space 上传上限为 5 MB；敏感运营数据请在本地运行。

## 复现、引用与许可

- **复现：**按照 [`REPRODUCTION.md`](REPRODUCTION.md) 完成数据准备、环境安装、公开
  检查点校验、测试、训练和证据生成。
- **引用：**正式发表信息公开前，请同时注明所用 GitHub commit 与固定的
  [Hugging Face 模型版本](https://huggingface.co/config-h/WLCR-SEA-Predictor)。
- **许可：**Apache License 2.0，详见 [`LICENSE`](LICENSE)。
