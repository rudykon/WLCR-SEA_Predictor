<h1 align="center">WLCR-SEA Predictor</h1>

<p align="center">
  <strong>具有结构化季节专家路由的请求局部蜂窝流量预测</strong><br>
  WLCR-SEA 主方法、论文相关分析/消融/对比程序与验证测试的开源实现。
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README_CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#validation"><img src="https://img.shields.io/badge/Validation-unittest-2CA02C?style=flat-square" alt="单元测试"></a>
  <a href="#scope"><img src="https://img.shields.io/badge/Release-WLCR--SEA%20only-6A5ACD?style=flat-square" alt="仅发布 WLCR-SEA"></a>
</p>

<p align="center">
  <a href="#overview">概览</a> ·
  <a href="#method">方法</a> ·
  <a href="#figures">图件</a> ·
  <a href="#quick-start">快速开始</a> ·
  <a href="#validation">验证</a> ·
  <a href="#resources">资源</a>
</p>

<a id="overview"></a>
## 概览

WLCR-SEA（Window-Local Context Representation with Seasonal Expert Attention）
是一种请求局部的流量预测方法。每次请求仅使用按时间排序的 336 小时历史与观测掩码，
预测四个流量指标未来 24 小时的数值。实现会显式输出季节性证据、专家可用性、路由
权重和有界修正，便于检查。

<a id="method"></a>
## 方法

WLCR-SEA 会针对每个预测步和每个指标，从输入请求构建八个具名专家。不可用证据在
路由前被排除。

| 专家 | 证据角色 |
| --- | --- |
| 前一天 | 前一天同一小时 |
| 前一周 | 前一周同一小时 |
| 两周滞后 | 前两周同一小时 |
| 同小时中位数，7 天 | 稳健的七日季节性汇总 |
| 同小时中位数，14 天 | 稳健的十四日季节性汇总 |
| 有界周趋势 | 由周变化给出的保守修正 |
| 窗口局部中位数 | 请求级回退摘要 |
| 冻结训练先验 | 始终可用的总体回退先验 |

按预测步设置的 Entmax 路由器仅在可用专家之间分配权重。有界残差在保留有限且可检查
的预测范围的同时，对路由基线进行修正。

<a id="figures"></a>
## 论文图件

以下五张 PNG 均由论文正文引用的五个图直接以 300 dpi 导出，未包含任何无关插图。

<p align="center">
  <a href="paper/figures/Scene_Diagram.pdf">
    <img src="docs/images/paper_figure_scenario.png" alt="请求局部服务场景" width="96%">
  </a>
</p>
<p align="center"><em>图 1｜请求局部服务的概念场景。</em></p>

<p align="center">
  <a href="docs/images/paper_figure_architecture.png">
    <img src="docs/images/paper_figure_architecture.png" alt="WLCR-SEA 结构化季节专家路由架构" width="96%">
  </a>
</p>
<p align="center"><em>图 2｜作为结构化季节专家路由实例的 WLCR-SEA。</em></p>

<details>
<summary><strong>展开查看图 3–5：洁净精度、缺失鲁棒性和可审计性</strong></summary>

<br>

<p align="center">
  <a href="docs/images/paper_figure_clean_accuracy.png">
    <img src="docs/images/paper_figure_clean_accuracy.png" alt="洁净留出集上的路由层级" width="76%">
  </a>
</p>
<p align="center"><em>图 3｜洁净留出集上的路由层级。</em></p>

<p align="center">
  <a href="docs/images/paper_figure_missingness.png">
    <img src="docs/images/paper_figure_missingness.png" alt="缺失鲁棒性" width="96%">
  </a>
</p>
<p align="center"><em>图 4｜缺失鲁棒性。</em></p>

<p align="center">
  <a href="docs/images/paper_figure_auditability.png">
    <img src="docs/images/paper_figure_auditability.png" alt="可审计性证据" width="96%">
  </a>
</p>
<p align="center"><em>图 5｜可审计性证据。</em></p>

</details>

<a id="quick-start"></a>
## 快速开始

~~~bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
~~~

完整的训练、分析、消融、对比和审计流程见 [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md)。

<a id="validation"></a>
## 验证

运行 WLCR-SEA 单元测试：

~~~bash
PYTHONPATH=. python3 -m unittest tests.test_wlcr_sea_model -v
~~~

<a id="resources"></a>
## 资源

| 资源 | 链接 |
| --- | --- |
| 数据下载 | [下载 ZIP](https://res-static.hc-cdn.cn/cloudbu-site/china/zh-cn/wuxian-gaoxiao2026/1780886490950118786.zip) |
| 源码 | [github.com/rudykon/WLCR-SEA_Predictor](https://github.com/rudykon/WLCR-SEA_Predictor) |

<a id="license"></a>
## 许可证

本仓库采用 Apache License 2.0，详见 <code>LICENSE</code>。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| <code>experiments/wlcr_sea_model.py</code> | WLCR-SEA 专家、路由、有界残差、损失与指标 |
| <code>experiments/missingness_protocol.py</code> | WLCR-SEA 使用的确定性缺失遥测协议 |
| <code>tests/test_wlcr_sea_model.py</code> | 公开方法的聚焦单元测试 |
| <code>docs/images/</code> | 为 README 导出的五张论文图 |
| <code>requirements.txt</code> | 最小 Python 依赖 |
