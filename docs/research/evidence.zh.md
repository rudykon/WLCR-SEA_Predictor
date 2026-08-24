# 论文证据

论文评估的是探索性的**鲁棒性—可检查性—代价特征**，而不是宣称普遍排行榜第一。
以下内容保留论文中的估计目标与限定条件。

[英文论文 PDF](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/paper/main.pdf){ .md-button target="_blank" rel="noopener" }
[中文论文 PDF](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/paper/main_zh.pdf){ .md-button target="_blank" rel="noopener" }

## 研究框架

| 项目 | 报告设置 |
| --- | --- |
| 记录 | 527,760 个小区小时 |
| 小区 | 736 |
| 请求 | 336 小时历史及观测掩码 |
| 目标 | 未来 24 小时、四个指标 |
| 洁净划分 | 论文定义的固定时间留出集 |
| 缺失 | 配对确定性掩码，并在指定场景重训练 |

<figure class="paper-figure">
  <a href="../../../images/paper_figure_scenario.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_scenario.png" alt="研究使用的请求局部证据边界" loading="lazy">
  </a>
  <figcaption>图 1｜请求生成后，评估的评分路径是自包含的。</figcaption>
</figure>

## 洁净预测

<figure class="paper-figure">
  <a href="../../../images/paper_figure_clean_accuracy.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_clean_accuracy.png" alt="洁净留出集路由层级" loading="lazy">
  </a>
  <figcaption>图 3｜洁净留出集上的路由层级。</figcaption>
</figure>

| 结果 | WAPE | 解释 |
| --- | ---: | --- |
| DLinear | **0.1854** | 对比中最低的洁净 WAPE |
| 先前的仅流量方法 | 0.1951 | 所选方法的配对参照 |
| WLCR-SEA 五种子集成 | 0.1955 | 差值 +0.00045；95% CI [-0.00312, 0.00366] |

相对于先前方法的配对区间包含零，因此研究没有检测到两者在洁净历史上的差异。
同时，论文也没有掩盖 DLinear 在洁净比较中更加准确这一事实。

## 结构化缺失

<figure class="paper-figure">
  <a href="../../../images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_missingness.png" alt="配对缺失鲁棒性结果" loading="lazy">
  </a>
  <figcaption>图 4｜配对缺失遥测机制下的鲁棒性。</figcaption>
</figure>

在所选中等缺失程度下，相对于 DLinear-Aug 与 PatchTST-Aug 的区间包含零。
在论文列出的 50% 结构化损坏下，WLCR-SEA 报告：

| 50% 机制 | WLCR-SEA WAPE |
| --- | ---: |
| 连续区块 | 0.2196 |
| 时间线尾部 | 0.2460 |
| 指标异步 | 0.2172 |

在这些固定掩码下，相对于匹配的 DLinear-Aug、PatchTST-Aug 和 GRU-D，九个配对差值均低于零。
该证据以五个固定损坏掩码为条件，不能推广到任意中断或任意重训练。

## 可审计性

<figure class="paper-figure">
  <a href="../../../images/paper_figure_auditability.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_auditability.png" alt="包括专家删除与路由影响的可审计性证据" loading="lazy">
  </a>
  <figcaption>图 5｜结构检查、专家删除与路由—影响一致性。</figcaption>
</figure>

| 审计 | 报告结果 |
| --- | ---: |
| 不可用专家路由质量 | 0 |
| 有界范围违规 | 0 |
| 256 个请求对象的位级服务路径差异 | 0 |
| 有效专家支持数 | 5.223 [5.101, 5.345] |
| 删除最高权重专家后的 WAPE 增量 | +0.00595 [0.00441, 0.00757] |
| 匹配随机删除后的 WAPE 增量 | +0.00104 [0.00079, 0.00131] |
| 权重—影响 Spearman | 0.693 [0.678, 0.708] |

路由熵与绝对百分比误差的平均相关为 -0.0196，95% CI [-0.0407, 0.0016]。
区间包含零，因此不能根据本研究把熵当作经过校准的不确定性分数。

## 代价与小区不相交审计

在单 CPU 线程、批量为 1 时，单种子中位延迟为 6.802 ms（P99 7.574 ms，16.2 KiB）；
五种子集成中位延迟为 34.705 ms（P99 38.684 ms，148.8 KiB）。

在协议匹配的小区不相交重训练中，WLCR-SEA 的 WAPE 为 0.1967。
相对于 DLinear-Aug 和先前方法的区间包含零；相对于 PatchTST-Aug 的区间低于零。
这些固定重训练属于同一轨迹内审计，不是重训练或跨区域泛化证据。
