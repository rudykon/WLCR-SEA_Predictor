# 系统架构

WLCR-SEA 把请求局部的证据规则落实为一条由五部分组成的服务路径。架构定义的是**数据可以从哪里进入、
如何流转，以及必须记录什么**；预测方法则在这条边界之内完成计算。

<figure class="paper-figure">
  <a href="../../../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_architecture.png" alt="WLCR-SEA 从密封请求、专家构造和掩码路由到预测与审计记录的系统架构" loading="lazy">
  </a>
  <figcaption>论文图 2。在线路径把请求装配、专家构造、路由、修正和审计输出明确分开。</figcaption>
</figure>

## 五个组件，五项职责

<div class="story-steps">
  <article class="story-step">
    <span class="story-number">01</span>
    <div>
      <h3>感知身份的入口层</h3>
      <p>服务层验证来源并解析运营身份，装配一份 336 小时历史及其观测掩码；小区 ID 随后保留在预测特征路径之外。</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">02</span>
    <div>
      <h3>密封请求边界</h3>
      <p>评分器只接收按顺序排列的四指标张量、具有最终权威性的掩码和冻结全局资产；它不能访问其他请求、其他小区或在线拓扑服务。</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">03</span>
    <div>
      <h3>季节专家构造器</h3>
      <p>构造器针对每个预测步和指标生成八个具名候选，并给出可用性与可靠度。被移除的证据会重新计算，而不是藏在数值填充之后。</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">04</span>
    <div>
      <h3>可用集路由与有界预测器</h3>
      <p>路由器只在仍然可用的专家之间分配质量。它们的凸组合形成基线；有界残差可以继续修正，但不会超过有限专家包络与配置边界之和。</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">05</span>
    <div>
      <h3>预测与审计输出</h3>
      <p>响应返回四条 24 小时预测。语义审计记录可以保留请求哈希、模型版本、专家状态、路由权重、基线、残差和包络检查，供日后重放。</p>
    </div>
  </article>
</div>

## 端到端数据流

| 边界 | 接收 | 产生 | 强制属性 |
| --- | --- | --- | --- |
| 入口 → 请求 | 经过授权的原始遥测 | 有序历史 + 掩码 | 身份不作为模型特征 |
| 请求 → 专家 | 一份密封请求 | 每个预测步和指标的八个候选 | 每个候选都能从本地请求重建 |
| 专家 → 路由器 | 数值、可用性、可靠度 | 稀疏路由权重 | 不可用专家的质量严格为零 |
| 路由器 → 预测器 | 路由基线 | 基线 + 有界残差 | 预测停留在可用证据附近 |
| 预测器 → 下游 | 预测与轨迹字段 | 规划输入 + 重放记录 | 预测与下游控制保持分离 |

## 信任与部署边界

这套架构有意比完整电信平台更窄。身份认证、权限控制、加密、留存和下游调度都属于外围系统。
WLCR-SEA 规定的是预测证据边界和拟议的审计语义；它不会把请求局部处理自动变成隐私保证，
也不是闭环控制器。

<div class="notice-card">
  <strong>架构不等于算法。</strong>本页解释组件、边界与数据流；“方法”解释八个专家、可用集路由和有界残差如何计算预测；“研究”评估这些设计表现出的行为与局限。
</div>

[从业务场景开始](problem.md){ .md-button }
[查看预测方法](method.md){ .md-button .md-button--primary }
[阅读研究证据](../research/evidence.md){ .md-button }
