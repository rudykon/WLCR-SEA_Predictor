---
hide:
  - toc
---

<section class="home-hero">
  <div class="hero-copy">
    <span class="hero-kicker">蜂窝网络运营 · 午夜预测 · 证据可审计</span>
    <h1>一个小区需要明天的流量计划，但评分器只能读取<span class="gradient-text">本次已授权请求。</span></h1>
    <p class="hero-lead">
      在最后一个已观测的 23:00 小时结束后，运营方可能需要未来 24 小时的需求信号，
      用于主动无线资源与服务容量规划。但在隔离的边缘域中，评分器不能悄悄查询邻近小区的实时流量。
      WLCR-SEA 将一份已授权的 336 小时历史转化为预测，并同时留下可重放的证据链。
    </p>
    <div class="hero-actions">
      <a class="hero-button primary" href="guide/problem/">进入业务故事</a>
      <a class="hero-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">体验一份请求</a>
      <a class="hero-button" href="research/evidence/">查看论文证据</a>
    </div>
    <div class="hero-proof">
      <span>单个目标小区</span>
      <span>仅使用已授权证据</span>
      <span>预测与审计记录同时输出</span>
    </div>
  </div>
  <figure class="hero-visual">
    <a href="../images/paper_figure_scenario.png" target="_blank" rel="noopener">
      <img src="../images/paper_figure_scenario.png" alt="从授权入口到可审计预测的请求局部蜂窝流量预测场景" loading="eager" decoding="async">
    </a>
    <figcaption class="hero-caption">论文图 1 · 入口授权并封装单次请求，评分器始终位于明确的证据边界之内</figcaption>
  </figure>
</section>

<div class="metric-strip">
  <div class="metric"><strong>336 小时</strong><span>已授权历史</span></div>
  <div class="metric"><strong>24 小时</strong><span>次日预测范围</span></div>
  <div class="metric"><strong>8</strong><span>具名证据专家</span></div>
  <div class="metric"><strong>4</strong><span>流量指标</span></div>
</div>

<span class="section-eyebrow">业务故事</span>

## 从一份已授权请求，到一条可重放的预测路径 {: .section-title }

<p class="section-lead">这个项目的起点不是模型结构图，而是一项运营约束：下游规划者需要短期流量信号，在线预测服务却必须严格限制在为单个小区明确封装的证据范围内。</p>

<div class="story-steps">
  <article class="story-step">
    <span class="story-number">01</span>
    <div>
      <h3>预测请求到达</h3>
      <p>在午夜预测起点，未来 24 小时可为主动无线资源与服务容量规划提供需求信号。预测只承担决策支持，具体调度与控制仍属于下游系统。</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">02</span>
    <div>
      <h3>入口封装证据</h3>
      <p>身份感知入口授权目标小区，并生成其有序的 336 小时历史与权威观测掩码。身份只在模型外用于路由和审计，不会成为预测特征。</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">03</span>
    <div>
      <h3>评分器解释参考依据</h3>
      <p>昨天、上周、两周前、稳健季节中位数、有界趋势与汇总回退值被构造成具名候选。缺失证据会在路由前被移除，而不是藏在某个数值填充值后面。</p>
    </div>
  </article>
  <article class="story-step">
    <span class="story-number">04</span>
    <div>
      <h3>预测带着证据离开</h3>
      <p>服务返回四条 24 小时预测序列，同时保留专家值、可用性、路由质量、基线、有界修正与包络检查，使结果可以从同一请求重新播放。</p>
    </div>
  </article>
</div>

<p class="story-bridge"><a href="guide/problem/">阅读完整服务场景 →</a></p>

<span class="section-eyebrow">适用场景</span>

## 什么时候会需要这样的预测方式 {: .section-title }

<p class="section-lead">当预测契约和点估计同样重要时，WLCR-SEA 更有意义：评分器只能使用有限证据，遥测可能不完整，并且事后需要有人解释哪些信息真正影响了结果。</p>

<div class="scenario-grid">
  <article class="scenario-card">
    <span class="scenario-tag">规划</span>
    <h3>每日无线资源规划</h3>
    <p>为次日主动资源规划提供小区级需求信号，同时让预测服务与下游调度器或控制器保持职责分离。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">容量</span>
    <h3>服务容量展望</h3>
    <p>当团队需要面向下一运营周期的紧凑小区视图时，给出四个流量指标的预期变化形态。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">中断</span>
    <h3>遥测数据不完整</h3>
    <p>严格移除不可用季节参考，并观察随机缺失、连续区块、最近尾部或异步指标缺失如何改变预测路径。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">治理</span>
    <h3>审计与故障复盘</h3>
    <p>根据请求和模型版本重放预测，检查当时哪些证据可用、被选中，或因不可用而被结构性排除。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">研究</span>
    <h3>受控预测研究</h3>
    <p>在同一个明确服务契约下比较洁净精度、缺失鲁棒性、请求局部性、审计属性与推理延迟。</p>
  </article>
  <article class="scenario-card scenario-card--boundary">
    <span class="scenario-tag">不是结论</span>
    <h3>不是自动网络控制系统</h3>
    <p>论文评估的是预测行为，而非业务收益或闭环控制。面对新区域、新季节、新策略和突发事件时仍需前瞻性验证。</p>
  </article>
</div>

<span class="section-eyebrow">为什么是 WLCR-SEA</span>

## 证据约束改变了模型形态 {: .section-title }

<p class="section-lead">当经过授权的邻区状态可用时，图预测模型可能更合适。本项目研究的是另一种情形：每一个在线值都必须来自当前请求或冻结的全局资产，并且每一个仍在使用的参考依据都应该可见。</p>

<figure class="paper-figure">
  <a href="../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../images/paper_figure_architecture.png" alt="WLCR-SEA 结构化季节专家路由架构" loading="lazy" decoding="async">
  </a>
  <figcaption>论文图 2 · 有限专家接口、可用集合路由、有界修正与拟议的语义审计记录。</figcaption>
</figure>

<div class="feature-grid">
  <article class="feature-card">
    <span class="feature-number">01 · 请求</span>
    <h3>自包含输入</h3>
    <p>评分器只接收一份有序历史及掩码，不能查询其他小区、拓扑表、在线特征库或跨请求流量缓存。</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">02 · 路由</span>
    <h3>具名季节证据</h3>
    <p>八个候选针对每个预测步与指标公开数值、可用性、可靠度和路由质量；不可用专家获得严格为零的权重。</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">03 · 审计</span>
    <h3>有界且可重放的输出</h3>
    <p>学习修正被限制在路由基线周围的有限对数空间范围内，请求和模型版本则为后续重放提供锚点。</p>
  </article>
</div>

<span class="section-eyebrow">研究发现</span>

## 让鲁棒性、可审计性与代价同时可见 {: .section-title }

<p class="section-lead">论文研究了来自 736 个小区的 527,760 条小区小时记录，但没有宣称普遍精度领先：DLinear 的洁净 WAPE 最低；WLCR-SEA 的主要证据在于既定协议下的结构化缺失鲁棒性和审计特性。</p>

<figure class="paper-figure">
  <a href="../images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="../images/paper_figure_missingness.png" alt="论文中不同缺失机制下的预测鲁棒性比较" loading="lazy" decoding="async">
  </a>
  <figcaption>论文图 4。缺失结果以论文规定的固定掩码和重训练为条件，不能解释为跨区域保证。</figcaption>
</figure>

<div class="metric-strip">
  <div class="metric"><strong>0.1955</strong><span>WLCR-SEA 洁净 WAPE</span></div>
  <div class="metric"><strong>0.1854</strong><span>DLinear 最低洁净 WAPE</span></div>
  <div class="metric"><strong>0</strong><span>审计中的不可用专家质量</span></div>
  <div class="metric"><strong>6.8 ms</strong><span>单种子 CPU 中位延迟</span></div>
</div>

<div class="demo-cta">
  <div>
    <h2>让一份请求亲自走完这条故事线</h2>
    <p>载入内置的 336 小时样例，移除部分遥测，检查仍然可用的专家，并下载 24 小时预测与审计记录。公开实验室运行真实的 A0 固定方法路径，而不是未发布的 A6 训练检查点。</p>
  </div>
  <a class="md-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">启动引导式审计实验室</a>
</div>

<div class="notice-card">
  <strong>解释边界很重要。</strong>报告中的熵不是经过校准的不确定性；数据只覆盖一个匿名区域约一个月；请求局部处理定义的是证据边界，而不是天然的隐私保证。复用结果前请阅读<a href="research/limitations/">范围与局限</a>。
</div>
