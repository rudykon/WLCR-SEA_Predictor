# 问题与适用场景

WLCR-SEA 解决一个明确的问题：**当模型只能获得一个小区自己的近期历史时，如何预测它未来
24 小时的流量？** 当数据访问受限、部分测量可能缺失，或以后需要检查计算过程时，这种设置具有实际意义。

<figure class="paper-figure">
  <a href="../../../images/paper_figure_scenario.png" target="_blank" rel="noopener">
    <img src="../../../images/paper_figure_scenario.png" alt="一个小区的数据从输入准备到预测与计算记录的流程" loading="lazy">
  </a>
  <figcaption>论文图 1。入口准备一个小区的输入，预测模型不能临时查询其他小区的实时数据。</figcaption>
</figure>

## 为什么需要这类预测

短期流量预测可以帮助团队提前规划无线资源和服务容量。本项目预测未来 24 小时的四项小区指标：
上行激活用户、下行激活用户、下行已用 PRB 和上行已用 PRB。

论文研究的是一个刻意受限的环境：入口可以验证小区并整理数据，但预测模型不能查询邻区实时流量、
拓扑信息或其他请求留下的缓存。这种分离方式适合需要控制数据访问、隔离故障或便于事后检查的系统。

## 一次请求如何完成预测

<div class="process-steps process-steps--compact">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>可用历史在 23:00 结束</h3>
      <p>模型预测次日 00:00 至 23:00。输出用于辅助规划，不是自动下发的调度命令。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>入口准备一个小区的数据</h3>
      <p>入口整理连续 336 小时和四项指标。布尔标记说明每个数值是否存在。小区 ID 可以留在模型外，用于权限控制和记录。</p>
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
      <p>结果包含四条 24 小时序列。记录可以保存输入哈希、模型版本、候选值、可用性、权重、最终修正和范围检查。</p>
    </div>
  </article>
</div>

<div class="notice-card">
  <strong>论文评估的是预测，不是完整网络系统。</strong>研究测量了精度、数据缺失时的表现、结果检查和延迟，没有测量业务收益或自动无线控制。
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

## 什么情况下应选择其他问题形式

WLCR-SEA 不应被视为所有蜂窝预测任务的默认答案。

- 如果能够可靠获得邻区流量和拓扑信息，图模型或多小区模型可以使用更多信息，可能更合适。
- 如果天气、突发事件、移动性、网络升级或节假日是主要影响因素，就需要加入并重新测试这些信号；当前方法并未包含它们。
- 如果应用需要可靠的不确定性估计，路由熵并不足够；本研究没有发现它与误差存在有用关系。
- 如果预测将直接控制网络，还需要额外的安全测试和真实业务影响评估。

## 模型输入

每次预测使用：

- 连续 336 个历史小时；
- 四个流量指标；
- 模型内部使用 `log1p` 变换数值；
- 每个数值都有“存在/缺失”布尔标记；
- 小区身份不作为预测特征。

预测目标是同四个序列的未来 24 小时。

## 模型可以和不可以使用什么

| 模型可以使用 | 模型不可以使用 |
| --- | --- |
| 当前输入的数值与缺失标记 | 其他小区的实时流量 |
| 训练阶段得到的固定统计量 | 按小区 ID 查询的元数据文件 |
| 单个版本化全局检查点 | 拓扑或邻区表 |
| 固定方法配置 | 外部特征库 |
| 模型外用于权限和记录的小区 ID | 其他请求留下的流量缓存 |

论文把这种设置称为**请求局部预测**：输入准备完成后，模型不能再为本次预测获取额外流量数据。

## 为什么这项约束会改变模型

普通模型可能只返回一个数值。WLCR-SEA 先根据已知历史规律生成八个候选预测，排除无法计算的候选，
再组合其余候选，并限制最后一次修正的幅度。使用相同输入可以重复这些计算步骤。

[继续了解系统架构](architecture.md){ .md-button .md-button--primary }
[查看八个专家如何路由](method.md){ .md-button }
[体验在线 Demo](../deployment/hugging-face.md){ .md-button }

!!! note "限制模型输入不等于保护数据"
    外围系统仍然需要访问控制、加密、安全日志和合适的数据留存规则。
