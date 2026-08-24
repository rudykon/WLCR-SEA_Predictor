# 实验结果

主要结论很直接：**历史数据完整时，WLCR-SEA 不是误差最低的模型；当成段的历史数据缺失，或用户需要检查预测是如何得到的时，它的优势更加明显。**

!!! info "怎样阅读这些数值"
    **WAPE** 用于衡量预测误差，数值越低越好。**95% 置信区间**表示实验所支持的差异范围。如果模型差异的置信区间包含 0，就说明该实验无法确认两个模型之间存在明确差异。

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

**结论：**DLinear 在完整数据对比中的误差最低。WLCR-SEA 与先前仅使用流量数据的方法之间，差异的置信区间包含 0，因此实验无法确认二者存在明确差异。

## 部分历史数据缺失时

<figure class="paper-figure">
  <a href="../../../images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_missingness.png" alt="为所有对比模型移除相同历史数值后的结果" loading="lazy">
  </a>
  <figcaption>图 4｜为所有对比模型移除相同数值后的预测误差。</figcaption>
</figure>

研究采用了几种接近实际故障的数据缺失方式：连续缺失一段、最近一段数据缺失，或不同指标在不同时间段缺失。当这些方式移除 50% 的数值时，WLCR-SEA 的结果如下：

| 数据移除 50% 的方式 | WLCR-SEA WAPE |
| --- | ---: |
| 连续区块 | 0.2196 |
| 时间线尾部 | 0.2460 |
| 指标异步 | 0.2172 |

在这些固定测试中，WLCR-SEA 与 DLinear-Aug、PatchTST-Aug 和 GRU-D 的九次对比均取得了更低误差。但在部分中等缺失率下，与 DLinear-Aug 和 PatchTST-Aug 的差异置信区间仍包含 0。

**结论：**结果支持 WLCR-SEA 在本研究测试的严重数据缺失场景中表现更好，但不能保证这一结论适用于所有故障、所有数据集或每一次重新训练。

## 能否核对预测依据

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
| 仅改变请求对象封装后，输出发生变化的次数（256 次测试） | 0 |
| 平均获得有效权重的候选数量 | 5.223 [5.101, 5.345] |
| 删除最高权重候选后的误差增量 | +0.00595 [0.00441, 0.00757] |
| 删除随机匹配候选后的误差增量 | +0.00104 [0.00079, 0.00131] |
| 分配权重与实际影响的关系（Spearman） | 0.693 [0.678, 0.708] |

权重更高的候选被删除后，通常会对结果产生更大的影响。这说明保存候选权重有助于复核预测依据。

同时也有一个重要的负面结果：路由熵与绝对百分比误差的平均相关系数为 -0.0196，95% 置信区间为 [-0.0407, 0.0016]。由于该区间包含 0，不能把路由熵当作可靠的不确定性分数。

## 运行速度与更严格的小区划分

使用单个 CPU 线程、每次处理一个请求时，单个 WLCR-SEA 模型的中位延迟为 6.802 ms，P99 延迟为 7.574 ms，检查点与冻结模型资源共 16.2 KiB；五模型集成的中位延迟为 34.705 ms，P99 延迟为 38.684 ms，检查点与冻结模型资源共 148.8 KiB。

在训练集与测试集使用不同小区的更严格划分中，WLCR-SEA 的 WAPE 为 0.1967。它与 DLinear-Aug 及先前方法之间的差异置信区间包含 0，因此尚不能确认存在明确差异；在该测试中，WLCR-SEA 优于 PatchTST-Aug。不过，所有小区仍来自同一个区域的数据，因此该结果不能证明方法能够推广到其他地区或季节。
