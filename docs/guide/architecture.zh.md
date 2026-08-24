# 系统架构

系统包含五个明确步骤：准备单个小区的历史数据、传给模型、生成候选预测、组合可用候选，
最后返回预测和计算记录。

<figure class="paper-figure">
  <a href="../../../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_architecture.png" alt="WLCR-SEA 从准备输入、候选加权到预测与计算记录的系统架构" loading="lazy">
  </a>
  <figcaption>论文图 2。一个输入如何变成多个候选预测、最终结果和检查记录。</figcaption>
</figure>

## 五个组件，五项职责

<div class="process-steps">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>数据入口</h3>
      <p>服务验证小区并准备 336 小时数据。布尔标记说明哪些值缺失，小区 ID 保留在模型之外。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>固定的模型输入</h3>
      <p>模型接收四项流量序列、缺失标记和训练阶段得到的固定信息，不能再获取额外实时流量。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">03</span>
    <div>
      <h3>候选预测生成</h3>
      <p>针对每个未来小时和指标，模型根据已知历史规律生成八个候选预测，并记录哪些候选能够计算。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">04</span>
    <div>
      <h3>分配权重并限制修正</h3>
      <p>模型只给可用候选分配权重。加权平均得到主要预测，最后一次修正的幅度受到限制。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">05</span>
    <div>
      <h3>预测与计算记录</h3>
      <p>响应返回四条 24 小时预测，还可以保存输入哈希、模型版本、候选值、权重、修正和范围检查。</p>
    </div>
  </article>
</div>

## 端到端数据流

| 步骤 | 接收 | 产生 | 重要规则 |
| --- | --- | --- | --- |
| 入口 → 模型输入 | 一个小区的原始数据 | 有序历史 + 缺失标记 | 小区 ID 不作为预测特征 |
| 模型输入 → 候选 | 一份准备好的输入 | 每小时、每指标八个候选 | 每个候选都能从输入重新计算 |
| 候选 → 权重 | 候选值及其可用性 | 候选权重 | 缺失参考的权重为零 |
| 权重 → 预测 | 加权平均值 | 平均值 + 受限修正 | 修正幅度不能无限增大 |
| 预测 → 用户 | 预测与计算字段 | 规划输入 + 保存记录 | 预测与控制保持分离 |

## 预测服务负责什么、不负责什么

这里展示的只是电信系统中的预测部分。外围系统仍需负责权限、加密、数据留存、日志和资源调度。
限制模型能读取的数据不会自动带来隐私保护，模型也不会直接控制网络。

<div class="notice-card">
  <strong>本页解释系统流程。</strong>“方法”页说明精确计算方式，“研究”页报告实验结果和适用边界。
</div>

[从业务场景开始](problem.md){ .md-button }
[查看预测方法](method.md){ .md-button .md-button--primary }
[阅读研究证据](../research/evidence.md){ .md-button }
