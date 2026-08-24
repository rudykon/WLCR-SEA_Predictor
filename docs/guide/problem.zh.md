# 问题与适用场景

WLCR-SEA 解决一个明确的问题：**当模型只能获得单个小区的近期历史数据时，如何预测该小区未来 24 小时的流量？** 当数据访问受到限制、部分测量可能缺失，或事后需要复核计算过程时，这种设置具有实际意义。

<figure class="paper-figure">
  <a href="../../../images/paper_figure_scenario.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_scenario.png" alt="一个小区的数据从输入准备到预测与计算记录的流程" loading="lazy">
  </a>
  <figcaption>论文图 1。入口准备一个小区的输入，预测模型不能临时查询其他小区的实时数据。</figcaption>
</figure>

## 为什么需要这类预测

短期流量预测可以帮助团队提前规划无线资源和服务容量。本项目预测四项小区指标在未来 24 小时的变化：上行激活用户数、下行激活用户数、下行已用 PRB 数和上行已用 PRB 数。

本研究有意限制预测模型可读取的信息：数据入口可以验证小区并整理输入，但模型不能查询邻区实时流量、拓扑信息或其他请求留下的缓存。这种分离方式适合需要严格控制数据访问、隔离故障或支持事后复核的系统。

## 一次请求如何完成预测

<div class="process-steps process-steps--compact">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>当天数据更新至 23:00</h3>
      <p>模型使用截至 23:00 的历史数据，预测次日 00:00 至 23:00 的流量。结果用于辅助规划，不会直接生成调度命令。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>入口准备一个小区的数据</h3>
      <p>入口整理四项指标连续 336 小时的数据。布尔标记说明每个数值是否存在。小区 ID 保留在模型之外，可用于权限控制和记录。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">03</span>
    <div>
      <h3>模型只使用已经准备好的输入</h3>
      <p>模型把这些历史数据与固定版本的模型、训练阶段得到的统计量结合，不会根据小区 ID 继续查询其他流量数据。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">04</span>
    <div>
      <h3>同时返回预测和计算记录</h3>
      <p>四项指标各得到一条未来 24 小时的预测序列。计算记录可以保存输入哈希、模型版本、候选值、可用性、权重、最终修正和范围检查结果。</p>
    </div>
  </article>
</div>

<div class="notice-card">
  <strong>本研究评估的是预测方法，而不是完整的网络系统。</strong>实验测量了预测精度、数据缺失时的表现、计算检查结果和运行延迟，但没有评估业务收益或自动无线控制。
</div>

## 哪些场景适合这种设置

<div class="scenario-grid scenario-grid--compact">
  <article class="scenario-card">
    <span class="scenario-tag">规划</span>
    <h3>次日资源展望</h3>
    <p>规划人员需要每个小区的次日需求估计，具体调度由另一个系统完成。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">缺失</span>
    <h3>部分测量数据缺失</h3>
    <p>模型使用明确的缺失标记，因此不会把占位数值误认为真实观测。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">复盘</span>
    <h3>预测需要重新计算</h3>
    <p>复核人员需要查看当时哪些历史参考可用，以及它们的权重怎样变化。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">比较</span>
    <h3>模型需要公平比较</h3>
    <p>研究人员可以让不同模型使用相同输入和相同缺失方式进行比较。</p>
  </article>
</div>

## 什么时候更适合采用其他方法

WLCR-SEA 并不是所有蜂窝流量预测任务的默认选择。

- 如果能够可靠获得邻区流量和拓扑信息，图模型或多小区模型可以利用这些额外信息，可能更加合适。
- 如果天气、突发事件、用户移动、网络升级或节假日会显著影响流量，就需要加入这些信号并重新评估；当前方法尚未使用它们。
- 如果应用需要可靠的不确定性估计，仅使用路由熵并不够；本研究没有发现路由熵与预测误差之间存在有用关系。
- 如果预测结果将直接用于网络控制，还需要额外开展安全测试和真实业务影响评估。

## 模型输入

每次预测使用：

- 连续 336 小时的历史数据；
- 四个流量指标；
- 模型内部使用 `log1p` 变换数值；
- 每个数值都有“存在/缺失”布尔标记；
- 小区身份不作为预测特征。

预测目标是这四项指标在未来 24 小时的取值。

## 模型允许使用哪些信息

| 模型可以使用 | 模型不可以使用 |
| --- | --- |
| 当前输入的数值与缺失标记 | 其他小区的实时流量 |
| 训练阶段得到的固定统计量 | 按小区 ID 查询的元数据文件 |
| 单个版本化全局检查点 | 拓扑或邻区表 |
| 固定方法配置 | 外部特征库 |
| 模型外用于权限和记录的小区 ID | 其他请求留下的流量缓存 |

本项目把这种设置称为**请求内局部预测（request-local forecasting）**：输入准备完成后，模型不能再为本次预测获取额外的流量数据。

## 为什么这项约束会改变模型

普通预测模型通常只给出最终结果。WLCR-SEA 会先根据已知历史规律生成八个候选预测，排除无法计算的候选，再组合其余候选，并限制最终修正的幅度。使用相同的输入和模型版本，可以复现同一次计算。

[继续了解系统架构](architecture.md){ .md-button .md-button--primary }
[查看八个专家如何路由](method.md){ .md-button }
[体验在线 Demo](../deployment/hugging-face.md){ .md-button }

!!! note "限制模型输入不等于保护数据"
    模型之外的系统仍需提供访问控制、加密、安全日志和合适的数据留存规则。
