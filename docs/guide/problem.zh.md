# 问题与适用场景

WLCR-SEA 只用一个小区的历史数据预测未来 24 小时。它适合数据访问受限、记录缺失或结果需要复核的场景。

<figure class="paper-figure">
  <a href="../../../images/paper_figure_scenario.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_scenario.png" alt="一个小区的数据从输入准备到预测与计算记录的流程" loading="lazy">
  </a>
  <figcaption>一个小区的历史数据进入模型，其他小区的数据不参与计算。</figcaption>
</figure>

## 一次请求

<div class="process-steps process-steps--compact">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>请求</h3>
      <p>预测次日 00:00 至 23:00。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>准备</h3>
      <p>整理 336 小时、四项指标和缺失标记。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">03</span>
    <div>
      <h3>预测</h3>
      <p>使用当前输入和固定版本的模型。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">04</span>
    <div>
      <h3>返回</h3>
      <p>输出四项预测，以及候选、权重和检查结果。</p>
    </div>
  </article>
</div>

<div class="notice-card">
  <strong>范围：</strong>只评估预测，未测试业务收益和网络控制。
</div>

## 适用场景

<div class="scenario-grid scenario-grid--compact">
  <article class="scenario-card">
    <span class="scenario-tag">规划</span>
    <h3>规划</h3>
    <p>调度前估计次日需求。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">缺失</span>
    <h3>缺失数据</h3>
    <p>避免占位数参与计算。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">复盘</span>
    <h3>复核</h3>
    <p>重算预测并检查权重。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">比较</span>
    <h3>研究</h3>
    <p>在相同缺失条件下比较模型。</p>
  </article>
</div>

## 什么时候换用其他方法

- **邻区数据可靠：**考虑图模型或多小区模型。
- **外部事件重要：**加入天气、事件、移动或节假日。
- **需要不确定性：**本研究中的路由熵不可靠。
- **直接控制网络：**先做安全和现场测试。

## 模型输入

输入为 336 小时 × 四项指标，并为每个值标记“存在/缺失”。模型内部使用 `log1p`，小区身份不作为特征。输出为同四项指标未来 24 小时的预测。

## 模型允许使用哪些信息

| 模型可以使用 | 模型不可以使用 |
| --- | --- |
| 当前输入的数值与缺失标记 | 其他小区的实时流量 |
| 训练阶段得到的固定统计量 | 按小区 ID 查询的元数据文件 |
| 单个版本化全局检查点 | 拓扑或邻区表 |
| 固定方法配置 | 外部特征库 |
| 模型外用于权限和记录的小区 ID | 其他请求留下的流量缓存 |

这称为**请求内局部预测（request-local forecasting）**：输入准备完成后，模型不能再获取额外流量数据。

## 为什么使用 WLCR-SEA

WLCR-SEA 生成八个历史候选，排除不可用项，组合其余候选，并限制最终修正。相同输入和模型版本可复现同一次计算。

[系统架构](architecture.md){ .md-button .md-button--primary }
[方法](method.md){ .md-button }
[Demo](../deployment/hugging-face.md){ .md-button }

!!! note "限制模型输入不等于保护数据"
    外部系统仍需提供访问控制、加密、安全日志和数据留存规则。
