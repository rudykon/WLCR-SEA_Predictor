---
hide:
  - toc
---

<section class="home-hero">
  <div class="hero-copy">
    <span class="hero-kicker">蜂窝流量预测 · 未来 24 小时 · 结果可检查</span>
    <h1>只使用本小区数据，<span class="gradient-text">预测明天的流量。</span></h1>
    <p class="hero-lead">WLCR-SEA 根据单个小区过去 14 天的四项流量指标，预测接下来 24 小时的流量。它适用于模型只能使用当前请求所附数据的场景。除了预测结果，系统还会返回本次计算采用了哪些历史规律。</p>
    <div class="hero-actions">
      <a class="hero-button primary" href="guide/problem/">查看适用场景</a>
      <a class="hero-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">体验在线 Demo</a>
      <a class="hero-button" href="research/evidence/">查看实验结果</a>
    </div>
    <div class="hero-proof">
      <span>每次预测一个小区</span>
      <span>输入 14 天，预测 24 小时</span>
      <span>预测结果附带计算记录</span>
    </div>
  </div>
  <figure class="hero-visual">
    <a href="../images/paper_figure_scenario.png" target="_blank" rel="noopener">
      <img src="../images/paper_figure_scenario.png" alt="一个小区的数据从输入准备到预测与计算记录的流程" loading="eager" decoding="async">
    </a>
    <figcaption class="hero-caption">论文图 1 · 数据入口准备单个小区的数据，模型不能查询其他小区的实时流量</figcaption>
  </figure>
</section>

<div class="metric-strip">
  <div class="metric"><strong>336 小时</strong><span>输入历史</span></div>
  <div class="metric"><strong>24 小时</strong><span>预测范围</span></div>
  <div class="metric"><strong>8</strong><span>历史候选预测</span></div>
  <div class="metric"><strong>4</strong><span>流量指标</span></div>
</div>

<span class="section-eyebrow">要解决的问题</span>

## 在严格的数据限制下完成有效预测 {: .section-title }

<p class="section-lead">网络规划人员需要某个小区明天的流量，但预测服务可能无权查询邻近小区或在线特征库。因此，模型必须充分利用本次提交的单小区历史数据。</p>

<div class="process-steps">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>需要次日预测</h3>
      <p>午夜时，规划人员需要未来 24 小时的流量。预测用于辅助规划，不会直接控制网络。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>入口准备单个小区的数据</h3>
      <p>服务验证小区身份，并整理连续 336 小时的数据。布尔标记用于区分真实观测与缺失值，小区 ID 不作为预测特征。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">03</span>
    <div>
      <h3>模型比较不同历史规律</h3>
      <p>模型根据近期的日周期和周周期、历史中位数及回退值生成八个候选预测。依赖缺失数据的候选会被自动排除。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">04</span>
    <div>
      <h3>同时输出预测和计算记录</h3>
      <p>服务分别为四项指标返回未来 24 小时的预测，同时记录候选值、候选权重、幅度受限的最终修正和范围检查结果。</p>
    </div>
  </article>
</div>

<p class="section-link"><a href="guide/problem/">查看适用场景与边界 →</a></p>

<span class="section-eyebrow">适用场景</span>

## 哪些场景适合这种预测方式 {: .section-title }

<p class="section-lead">当模型只能访问有限数据、历史记录可能不完整，或事后需要检查预测为何变化时，WLCR-SEA 更有意义。</p>

<div class="scenario-grid">
  <article class="scenario-card">
    <span class="scenario-tag">规划</span>
    <h3>每日无线资源规划</h3>
    <p>在分配资源前估计次日需求，同时把预测与后续调度、控制分开。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">容量</span>
    <h3>次日容量评估</h3>
    <p>用简洁的小区视图展示四项流量指标在次日可能发生的变化，为容量准备提供参考。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">中断</span>
    <h3>测量数据不完整</h3>
    <p>自动停用依赖缺失观测的候选，并评估不同数据缺失方式对预测的影响。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">复盘</span>
    <h3>结果检查与故障复盘</h3>
    <p>使用相同输入和模型版本重新计算，并检查哪些历史参考被使用或排除。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">研究</span>
    <h3>可复现的模型评测</h3>
    <p>在统一规则下比较完整数据精度、缺失数据下的表现、输入约束、计算检查和运行速度。</p>
  </article>
  <article class="scenario-card scenario-card--boundary">
    <span class="scenario-tag">不是结论</span>
    <h3>不是自动网络控制系统</h3>
    <p>本研究评估的是预测表现，而非业务收益或闭环控制。应用到新地区、新季节、新策略或突发事件前，仍需开展前瞻性验证。</p>
  </article>
</div>

<span class="section-eyebrow">为什么是 WLCR-SEA</span>

## WLCR-SEA 如何在数据限制下工作 {: .section-title }

<p class="section-lead">如果能够可靠获得邻区数据，图模型可能更合适。WLCR-SEA 解决的是更受限的情况：每次预测只能使用当前输入和训练阶段得到的固定信息。</p>

<figure class="paper-figure">
  <a href="../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../images/paper_figure_architecture.png" alt="WLCR-SEA 如何生成并组合候选预测" loading="lazy" decoding="async">
  </a>
  <figcaption>论文图 2 · 八个历史候选经过筛选和加权，再进行幅度受限的最终修正。</figcaption>
</figure>

<div class="feature-grid">
  <article class="feature-card">
    <span class="feature-number">01 · 请求</span>
    <h3>单次请求自带完整输入</h3>
    <p>模型只接收一个小区的历史数据和缺失标记，不能临时读取其他小区或外部在线服务。</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">02 · 路由</span>
    <h3>八个可解释的候选预测</h3>
    <p>每个候选都对应一种已知历史规律。模型会记录候选值、能否使用以及获得的权重。</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">03 · 记录</span>
    <h3>幅度受限、结果可复现</h3>
    <p>模型可以修正加权平均值，但修正幅度有固定上限。保存输入与模型版本后，即可复现同一次计算。</p>
  </article>
</div>

<p class="section-link"><a href="guide/architecture/">查看完整系统架构 →</a></p>

<span class="section-eyebrow">研究发现</span>

## 实验说明了什么，又没有说明什么 {: .section-title }

<p class="section-lead">研究使用了 736 个小区的 527,760 条小时记录。历史数据完整时，DLinear 的误差更低；在论文设定的严重数据缺失测试中，WLCR-SEA 表现更好，并能保留可供检查的计算记录。</p>

<figure class="paper-figure">
  <a href="../images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="../images/paper_figure_missingness.png" alt="按不同方式移除历史数据后的预测误差" loading="lazy" decoding="async">
  </a>
  <figcaption>论文图 4。报告的优势只适用于论文采用的固定数据移除方式和训练过程，不能视为跨区域保证。</figcaption>
</figure>

<div class="metric-strip">
  <div class="metric"><strong>0.1955</strong><span>WLCR-SEA 完整数据 WAPE</span></div>
  <div class="metric"><strong>0.1854</strong><span>DLinear 完整数据 WAPE</span></div>
  <div class="metric"><strong>0</strong><span>分给缺失参考的权重</span></div>
  <div class="metric"><strong>6.8 ms</strong><span>单模型 CPU 中位延迟</span></div>
</div>

<div class="demo-cta">
  <div>
    <h2>用一个样例体验方法</h2>
    <p>载入内置的 336 小时样例，移除部分历史数据，查看哪些候选仍可使用，再下载未来 24 小时的预测和计算记录。由于训练后的 A6 检查点尚未公开，Demo 运行的是较简单的 A0 固定基线。</p>
  </div>
  <a class="md-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">启动在线 Demo</a>
</div>

<div class="notice-card">
  <strong>部署前请阅读适用边界。</strong>数据只覆盖一个匿名区域约一个月；论文中的路由熵不能作为可靠的不确定性分数；限制模型输入也不等于自动获得隐私保护。详见<a href="research/limitations/">完整局限</a>。
</div>
