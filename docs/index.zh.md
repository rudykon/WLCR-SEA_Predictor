---
hide:
  - toc
---

<section class="project-hero">
  <div class="project-wordmark">
    <img class="wordmark-light" src="../assets/brand/logo.svg" alt="WLCR-SEA Predictor">
    <img class="wordmark-dark" src="../assets/brand/logo-dark.svg" alt="WLCR-SEA Predictor">
  </div>
  <h1>WLCR-SEA</h1>
  <p class="project-subtitle">面向缺失遥测的单小区请求内流量预测</p>
  <p class="project-lead">使用单个小区过去 336 小时的四项流量指标，预测未来 24 小时。不可用历史参考被严格屏蔽，每次预测均可导出专家值、路由权重、残差与边界检查。</p>
  <div class="project-actions">
    <a class="md-button md-button--primary" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">体验 A6</a>
    <a class="md-button" href="guide/method/">方法</a>
    <a class="md-button" href="research/evidence/">实验结果</a>
  </div>
</section>

## 研究设定与方法

<div class="setting-layout">
  <figure class="paper-figure">
    <a href="../images/paper_figure_architecture.png" target="_blank" rel="noopener">
      <img src="../images/paper_figure_architecture.png" alt="WLCR-SEA 从单小区请求到八个季节专家、严格掩码路由、有界预测与审计记录的架构">
    </a>
    <figcaption>一个封闭的单小区请求与冻结 A6 资源共同生成预测和审计记录。</figcaption>
  </figure>
  <div>
    <p>WLCR-SEA 面向遥测可能缺失、且推理阶段不能查询相邻小区的预测任务。模型只使用当前请求和冻结资源。</p>
    <ul class="fact-list">
      <li><strong>单个小区</strong><span>不查询其他小区实时数据</span></li>
      <li><strong>336 小时 × 4</strong><span>历史数据与观测掩码</span></li>
      <li><strong>八个季节专家</strong><span>日、周、中位数、趋势、请求汇总与训练先验</span></li>
      <li><strong>24 小时 × 4</strong><span>A6 五模型集成预测</span></li>
      <li><strong>审计记录</strong><span>数值、可用性、权重、残差和检查</span></li>
    </ul>
  </div>
</div>

## 主要发现

| 问题 | 已登记结果 |
| --- | --- |
| 完整历史 | DLinear 的**宏指标 WAPE**更低：0.1854；WLCR-SEA A6 集成为 0.1955。 |
| 严重缺失 | 在研究设定的严重缺失场景中，WLCR-SEA 在与 DLinear-Aug、PatchTST-Aug、GRU-D 的 9 项预设对比中宏指标 WAPE 均更低。 |
| 可审计性 | 不可用专家权重为 0；报告的预测边界违规为 0。 |
| CPU 成本 | 单线程、batch=1 时，五模型集成中位/P99 延迟为 34.705/38.684 ms，模型资源为 148.8 KiB。 |

[查看完整协议、区间与结果 →](research/evidence.md)

## 适用边界

<div class="scope-box">
  <ul>
    <li>只在一个匿名地区、约一个月的数据上验证。</li>
    <li>路由熵不是经过校准的不确定性估计。</li>
    <li>请求内推理本身不构成隐私或安全保证。</li>
    <li>尚未验证网络控制动作与业务收益。</li>
  </ul>
</div>
