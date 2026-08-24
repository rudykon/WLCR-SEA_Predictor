---
hide:
  - toc
---

<section class="home-hero">
  <div class="hero-copy">
    <span class="hero-kicker">小区流量 · 24 小时预测 · 结果可追溯</span>
    <h1>只用一个小区的数据，<span class="gradient-text">预测明天流量。</span></h1>
    <p class="hero-lead">WLCR-SEA 用一个小区过去 14 天的数据预测未来 24 小时。缺失参考会被排除，每项结果都会记录候选值和权重。</p>
    <div class="hero-actions">
      <a class="hero-button primary" href="guide/problem/">适用场景</a>
      <a class="hero-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">Demo</a>
      <a class="hero-button" href="research/evidence/">实验结果</a>
    </div>
    <div class="hero-proof">
      <span>单个小区</span>
      <span>14 天 → 24 小时</span>
      <span>可追溯</span>
    </div>
  </div>
  <figure class="hero-visual hero-logo-visual">
    <img class="hero-brand-logo hero-brand-logo--light" src="../assets/brand/logo.svg" alt="WLCR-SEA Predictor 项目 Logo" loading="eager" decoding="async">
    <img class="hero-brand-logo hero-brand-logo--dark" src="../assets/brand/logo-dark.svg" alt="WLCR-SEA Predictor 深色模式项目 Logo" loading="eager" decoding="async">
  </figure>
</section>

<div class="metric-strip">
  <div class="metric"><strong>336 小时</strong><span>输入历史</span></div>
  <div class="metric"><strong>24 小时</strong><span>预测范围</span></div>
  <div class="metric"><strong>8</strong><span>历史候选预测</span></div>
  <div class="metric"><strong>4</strong><span>流量指标</span></div>
</div>

<span class="section-eyebrow">问题</span>

## 数据有限，也要预测明天 {: .section-title }

<p class="section-lead">规划人员需要明天的流量，模型却只能读取当前小区的历史数据。</p>

<div class="process-steps">
  <article class="process-step">
    <span class="step-number">01</span>
    <div>
      <h3>需求</h3>
      <p>预测四项指标未来 24 小时的变化。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">02</span>
    <div>
      <h3>输入</h3>
      <p>整理 336 条小时记录和缺失标记。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">03</span>
    <div>
      <h3>预测</h3>
      <p>生成八个候选，排除依赖缺失数据的候选。</p>
    </div>
  </article>
  <article class="process-step">
    <span class="step-number">04</span>
    <div>
      <h3>输出</h3>
      <p>返回预测、权重、修正和检查结果。</p>
    </div>
  </article>
</div>

<p class="section-link"><a href="guide/problem/">适用场景 →</a></p>

<span class="section-eyebrow">场景</span>

## 什么时候使用 {: .section-title }

<p class="section-lead">适合数据访问受限、记录不完整或结果需要复核的场景。</p>

<div class="scenario-grid">
  <article class="scenario-card">
    <span class="scenario-tag">规划</span>
    <h3>规划</h3>
    <p>在制定次日计划前估计需求。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">缺失</span>
    <h3>缺失数据</h3>
    <p>自动排除依赖缺失值的候选。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">复核</span>
    <h3>结果复核</h3>
    <p>重算结果并查看候选与权重。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag">边界</span>
    <h3>只做预测</h3>
    <p>调度和网络控制由其他系统负责。</p>
  </article>
</div>

<span class="section-eyebrow">方法</span>

## 八个候选，一个结果 {: .section-title }

<p class="section-lead">模型从日、周规律生成候选，排除不可用项，再组合其余候选。</p>

<figure class="paper-figure">
  <a href="../images/paper_figure_architecture.png" target="_blank" rel="noopener">
    <img src="../images/paper_figure_architecture.png" alt="WLCR-SEA 如何生成并组合候选预测" loading="lazy" decoding="async">
  </a>
  <figcaption>八个候选经过筛选和加权，最终修正受到固定限制。</figcaption>
</figure>

<div class="feature-grid">
  <article class="feature-card">
    <span class="feature-number">01 · 输入</span>
    <h3>一个小区</h3>
    <p>模型读取当前历史、缺失标记和固定训练资源，不读取其他实时数据。</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">02 · 路由</span>
    <h3>八个候选</h3>
    <p>每个候选都有数值、可用状态和权重。</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">03 · 输出</span>
    <h3>受限结果</h3>
    <p>最终修正有上限，计算过程可以保存。</p>
  </article>
</div>

<p class="section-link"><a href="guide/architecture/">系统架构 →</a></p>

<span class="section-eyebrow">结果</span>

## 完整数据下 DLinear 更准；严重缺失时 WLCR-SEA 更稳 {: .section-title }

<p class="section-lead">实验包含 736 个小区、527,760 条记录。完整数据下 DLinear 更准；在研究设定的严重缺失测试中，WLCR-SEA 更好。</p>

<figure class="paper-figure">
  <a href="../images/paper_figure_missingness.png" target="_blank" rel="noopener">
    <img src="../images/paper_figure_missingness.png" alt="按不同方式移除历史数据后的预测误差" loading="lazy" decoding="async">
  </a>
  <figcaption>研究设定的固定数据缺失测试。</figcaption>
</figure>

<div class="metric-strip">
  <div class="metric"><strong>0.1955</strong><span>WLCR-SEA 完整数据 WAPE</span></div>
  <div class="metric"><strong>0.1854</strong><span>DLinear 完整数据 WAPE</span></div>
  <div class="metric"><strong>0</strong><span>分给缺失参考的权重</span></div>
  <div class="metric"><strong>6.8 ms</strong><span>单模型 CPU 中位延迟</span></div>
</div>

<div class="demo-cta">
  <div>
    <h2>立即体验</h2>
    <p>载入样例、移除历史数据，再查看预测。Demo 使用 A0 固定基线；训练后的 A6 检查点尚未公开。</p>
  </div>
  <a class="md-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">打开 Demo</a>
</div>

<div class="notice-card">
  <strong>局限：</strong>一个地区、约一个月；没有可靠的不确定性分数；不提供隐私保证。<a href="research/limitations/">查看详情</a>。
</div>
