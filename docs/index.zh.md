---
hide:
  - toc
---

<section class="home-hero">
  <div class="hero-copy">
    <span class="hero-kicker">开源 · 请求局部 · 可审计</span>
    <h1>仅使用<span class="gradient-text">已授权证据</span>预测蜂窝流量。</h1>
    <p class="hero-lead">
      WLCR-SEA 将单个密封的 336 小时小区历史转化为未来 24 小时、四个指标的预测，
      整个过程由具名季节专家、精确可用性掩码、稀疏路由和有界修正组成。
    </p>
    <div class="hero-actions">
      <a class="hero-button primary" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">打开审计实验室</a>
      <a class="hero-button" href="getting-started/quickstart/">运行代码</a>
      <a class="hero-button" href="research/evidence/">查看论文证据</a>
    </div>
    <div class="hero-proof">
      <span>有限专家接口</span>
      <span>缺失证据硬掩码</span>
      <span>请求记录可重放</span>
    </div>
  </div>
  <figure class="hero-visual">
    <a href="../images/paper_figure_architecture.png" target="_blank" rel="noopener">
      <img src="../images/paper_figure_architecture.png" alt="WLCR-SEA 结构化季节专家路由架构" loading="eager" decoding="async">
    </a>
    <figcaption class="hero-caption">论文图 2 · 点击查看 4,000 × 2,250 原始图片</figcaption>
  </figure>
</section>

<div class="metric-strip">
  <div class="metric"><strong>336 小时</strong><span>密封请求历史</span></div>
  <div class="metric"><strong>24 小时</strong><span>预测范围</span></div>
  <div class="metric"><strong>8</strong><span>具名证据专家</span></div>
  <div class="metric"><strong>4</strong><span>流量指标</span></div>
</div>

<span class="section-eyebrow">方法边界</span>

## 为检查而设计的预测路径 {: .section-title}

<p class="section-lead">评分器不能查询其他小区、拓扑表或在线特征库。在线路径中的每一个值都必须来自当前请求，或来自冻结且全局共享的资产。</p>

<div class="feature-grid">
  <article class="feature-card">
    <span class="feature-number">01 · 请求</span>
    <h3>自包含输入</h3>
    <p>身份感知入口负责授权并生成一份有序历史及观测掩码。小区身份不会作为预测特征进入模型。</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">02 · 路由</span>
    <h3>结构化证据</h3>
    <p>八个季节与汇总专家针对每个预测步和指标显式输出候选值、可用性、可靠度和路由质量。</p>
  </article>
  <article class="feature-card">
    <span class="feature-number">03 · 审计</span>
    <h3>有界输出</h3>
    <p>不可用专家获得严格为零的权重；学习残差则被限制在路由基线周围的有限对数空间范围内。</p>
  </article>
</div>

<span class="section-eyebrow">论文证据</span>

## 让鲁棒性、可审计性与代价同时可见 {: .section-title}

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
    <h2>移除证据，观察路径如何变化</h2>
    <p>公开实验室运行仓库真实的专家构造、缺失协议、硬掩码与无参数 A0 固定混合。仓库未发布论文训练权重，因此演示绝不会把结果冒充为 A6 推理。</p>
  </div>
  <a class="md-button" href="https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor" target="_blank" rel="noopener">在 Hugging Face 启动</a>
</div>

<div class="notice-card">
  <strong>解释边界很重要。</strong>报告中的熵不是经过校准的不确定性；数据只覆盖一个匿名区域约一个月；请求局部处理定义的是证据边界，而不是天然的隐私保证。复用结果前请阅读<a href="research/limitations/">范围与局限</a>。
</div>
