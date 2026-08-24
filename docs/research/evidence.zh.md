# 实验结果

主要结论很直接：**历史数据完整时，WLCR-SEA 不是最准确的模型；当连续一段历史缺失，
或用户需要检查预测怎样形成时，它的价值更加明显。**

!!! info "怎样阅读这些数值"
    **WAPE** 是预测误差，越低越好。**95% 置信区间**表示实验支持的差异范围。
    如果模型差异的区间包含 0，就说明该实验无法确认两个模型存在明确差异。

[英文论文 PDF](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/paper/main.pdf){ .md-button target="_blank" rel="noopener" }
[中文论文 PDF](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/paper/main_zh.pdf){ .md-button target="_blank" rel="noopener" }

## 数据与评估设置

| 项目 | 报告设置 |
| --- | --- |
| 记录 | 527,760 个小区小时 |
| 小区 | 736 |
| 请求 | 336 小时历史及观测掩码 |
| 目标 | 未来 24 小时、四个指标 |
| 完整数据测试 | 论文定义的固定后续时间段 |
| 缺失数据测试 | 所有对比模型移除相同的数据点 |

<figure class="paper-figure">
  <a href="../../../images/paper_figure_scenario.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_scenario.png" alt="一个小区的数据从输入准备到预测与计算记录的流程" loading="lazy">
  </a>
  <figcaption>图 1｜每次预测只使用一个准备好的小区历史，不能临时查询其他小区的实时流量。</figcaption>
</figure>

## 历史数据完整时

<figure class="paper-figure">
  <a href="../../../images/paper_figure_clean_accuracy.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_clean_accuracy.png" alt="历史数据完整时的模型对比" loading="lazy">
  </a>
  <figcaption>图 3｜历史数据完整时的模型对比。</figcaption>
</figure>

| 模型 | WAPE | 含义 |
| --- | ---: | --- |
| DLinear | **0.1854** | 对比中最低的完整数据 WAPE |
| 先前的仅流量方法 | 0.1951 | 与 WLCR-SEA 误差接近 |
| WLCR-SEA 五模型集成 | 0.1955 | 相对先前方法差值 +0.00045；95% CI [-0.00312, 0.00366] |

**结论：**DLinear 在完整数据对比中的误差最低。WLCR-SEA 与先前仅流量方法的差异区间包含 0，
因此实验无法确认二者存在明确差异。

## 部分历史数据缺失时

<figure class="paper-figure">
  <a href="../../../images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_missingness.png" alt="为所有对比模型移除相同历史数值后的结果" loading="lazy">
  </a>
  <figcaption>图 4｜为所有对比模型移除相同数值后的预测误差。</figcaption>
</figure>

论文使用几种接近实际故障的缺失方式：连续缺失一段、最近一段数据缺失，或不同指标在不同时间缺失。
在这些模式下移除 50% 数值时，WLCR-SEA 的结果为：

| 50% 机制 | WLCR-SEA WAPE |
| --- | ---: |
| 连续区块 | 0.2196 |
| 时间线尾部 | 0.2460 |
| 指标异步 | 0.2172 |

在这些固定测试中，WLCR-SEA 相对 DLinear-Aug、PatchTST-Aug 和 GRU-D 的九次对比误差都更低。
但在部分中等缺失率下，相对 DLinear-Aug 和 PatchTST-Aug 的置信区间仍包含 0。

**结论：**结果支持 WLCR-SEA 在本研究测试的严重缺失模式下表现更好，但不能保证适用于所有故障、
所有数据集或每一次重新训练。

## 计算过程能否检查

<figure class="paper-figure">
  <a href="../../../images/paper_figure_auditability.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_auditability.png" alt="候选权重、删除影响和预测限制检查" loading="lazy">
  </a>
  <figcaption>图 5｜候选权重、删除影响和预测限制检查。</figcaption>
</figure>

| 检查 | 报告结果 |
| --- | ---: |
| 分给不可用候选的权重 | 0 |
| 超出允许修正范围的预测 | 0 |
| 只改变请求对象包装造成的输出变化（256 次测试） | 0 |
| 平均获得有效支持的候选数量 | 5.223 [5.101, 5.345] |
| 删除最高权重候选后的误差增量 | +0.00595 [0.00441, 0.00757] |
| 删除随机匹配候选后的误差增量 | +0.00104 [0.00079, 0.00131] |
| 分配权重与实际影响的关系（Spearman） | 0.693 [0.678, 0.708] |

获得更高权重的候选在被删除时通常会产生更大影响，说明保存的权重有助于检查预测。

同时也有一个重要的负面结果：路由熵与绝对百分比误差的平均相关为 -0.0196，
95% CI [-0.0407, 0.0016]。由于区间包含 0，不能把熵当作可靠的不确定性分数。

## 运行速度与更严格的小区划分

使用单 CPU 线程、每次预测一个请求时，一个 WLCR-SEA 模型的中位延迟为 6.802 ms
（P99 7.574 ms，16.2 KiB）；五模型集成为 34.705 ms（P99 38.684 ms，148.8 KiB）。

在训练集和测试集使用不同小区的更严格划分中，WLCR-SEA 的 WAPE 为 0.1967。
相对 DLinear-Aug 和先前方法的差异区间包含 0，因此差异不明确；在该测试中优于 PatchTST-Aug。
但所有小区仍来自同一个区域数据，因此不能证明方法能推广到其他地区或季节。
