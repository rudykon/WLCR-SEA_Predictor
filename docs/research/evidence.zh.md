# 实验结果

**完整数据下 DLinear 更准；在研究设定的严重缺失测试中，WLCR-SEA 更好，并保留可追溯的计算记录。**

!!! info "如何读表"
    **WAPE** 越低越好。95% 置信区间包含 0，表示差异不明确。

## 设置

| 项目 | 报告设置 |
| --- | --- |
| 记录 | 527,760 个小区小时 |
| 小区 | 736 |
| 请求 | 336 小时历史及观测掩码 |
| 目标 | 未来 24 小时、四个指标 |
| 完整数据测试 | 论文定义的固定后续时间段 |
| 缺失数据测试 | 所有对比模型移除相同的数据点 |

## 完整数据

<figure class="paper-figure">
  <a href="../../../images/paper_figure_clean_accuracy.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_clean_accuracy.png" alt="历史数据完整时的模型对比" loading="lazy">
  </a>
  <figcaption>完整数据下的模型对比。</figcaption>
</figure>

| 模型 | WAPE | 含义 |
| --- | ---: | --- |
| DLinear | **0.1854** | 对比中最低的完整数据 WAPE |
| 先前的仅流量方法 | 0.1951 | 与 WLCR-SEA 误差接近 |
| WLCR-SEA 五模型集成 | 0.1955 | 相对先前方法差值 +0.00045；95% CI [-0.00312, 0.00366] |

**结果：**DLinear 误差最低。WLCR-SEA 与先前方法的区间包含 0，差异不明确。

## 缺失数据

<figure class="paper-figure">
  <a href="../../../images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_missingness.png" alt="为所有对比模型移除相同历史数值后的结果" loading="lazy">
  </a>
  <figcaption>相同缺失数据下的模型对比。</figcaption>
</figure>

研究按连续区块、最近时段或指标异步方式移除数据。缺失 50% 时：

| 数据移除 50% 的方式 | WLCR-SEA WAPE |
| --- | ---: |
| 连续区块 | 0.2196 |
| 时间线尾部 | 0.2460 |
| 指标异步 | 0.2172 |

九项固定对比中，WLCR-SEA 均低于 DLinear-Aug、PatchTST-Aug 和 GRU-D。部分中等缺失率下，与前两者的差异仍不明确。

**结果：**在测试的严重缺失场景中更好，不代表其他数据或训练也会如此。

## 可追溯性

<figure class="paper-figure">
  <a href="../../../images/paper_figure_auditability.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_auditability.png" alt="候选权重、删除影响和预测限制检查" loading="lazy">
  </a>
  <figcaption>权重、删除和范围检查。</figcaption>
</figure>

| 检查 | 报告结果 |
| --- | ---: |
| 分给不可用候选的权重 | 0 |
| 超出允许修正范围的预测 | 0 |
| 仅改变请求对象封装后，输出发生变化的次数（256 次测试） | 0 |
| 平均获得有效权重的候选数量 | 5.223 [5.101, 5.345] |
| 删除最高权重候选后的误差增量 | +0.00595 [0.00441, 0.00757] |
| 删除随机匹配候选后的误差增量 | +0.00104 [0.00079, 0.00131] |
| 分配权重与实际影响的关系（Spearman） | 0.693 [0.678, 0.708] |

高权重候选被删除后通常影响更大，因此权重有助于复核结果。

路由熵与误差的相关系数为 -0.0196，95% CI [-0.0407, 0.0016]。它**不能**作为可靠的不确定性分数。

## 速度与小区划分

单 CPU 线程、batch=1、逐请求运行时，单模型中位/P99 延迟为 6.802/7.574 ms，模型资源 16.2 KiB；五模型集成为 34.705/38.684 ms，148.8 KiB。

训练集与测试集使用不同小区时，WLCR-SEA 的 WAPE 为 0.1967。它与 DLinear-Aug、先前方法的差异不明确，优于 PatchTST-Aug。所有小区仍来自同一地区。
