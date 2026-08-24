# 系统架构

五个步骤把一个小区的历史数据变成预测和计算记录。

<figure class="paper-figure">
  <a href="../../../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_architecture.png" alt="WLCR-SEA 从准备输入、候选加权到预测与计算记录的系统架构" loading="lazy">
  </a>
  <figcaption>从准备输入到候选、预测和计算记录。</figcaption>
</figure>

## 组件

<div class="process-steps">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>数据入口</h3>
      <p>准备 336 小时数据和缺失标记，小区 ID 留在模型外。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>模型输入</h3>
      <p>传入四项流量、缺失标记和固定训练信息。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">03</span>
    <div>
      <h3>候选</h3>
      <p>为每个小时和指标生成八个候选。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">04</span>
    <div>
      <h3>路由</h3>
      <p>加权可用候选，再进行受限修正。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">05</span>
    <div>
      <h3>输出</h3>
      <p>返回四项 24 小时预测和计算字段。</p>
    </div>
  </article>
</div>

## 数据流

| 步骤 | 接收 | 产生 | 重要规则 |
| --- | --- | --- | --- |
| 入口 → 模型输入 | 一个小区的原始数据 | 有序历史 + 缺失标记 | 小区 ID 不作为预测特征 |
| 模型输入 → 候选 | 一份准备好的输入 | 每小时、每指标八个候选 | 每个候选都能从输入重新计算 |
| 候选 → 权重 | 候选值及其可用性 | 候选权重 | 缺失参考的权重为零 |
| 权重 → 预测 | 加权平均值 | 平均值 + 受限修正 | 修正幅度不能无限增大 |
| 预测 → 用户 | 预测与计算字段 | 规划输入 + 保存记录 | 预测与控制保持分离 |

## 边界

这是预测模块，不是网络控制器。权限、加密、留存、日志和调度由外部系统负责。

<div class="notice-card">
  <strong>下一步：</strong>“方法”介绍计算过程，“实验结果”给出证据。
</div>

[适用场景](problem.md){ .md-button }
[方法](method.md){ .md-button .md-button--primary }
[实验结果](../research/evidence.md){ .md-button }
