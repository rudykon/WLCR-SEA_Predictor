# 安装

WLCR-SEA 是 Python 研究仓库。建议使用 Python 3.11 或更高版本；网站与托管 Demo 使用 Python 3.12 验证。

## 克隆并创建环境

```bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell 使用：

```powershell
.venv\Scripts\Activate.ps1
```

## 验证核心方法

```bash
PYTHONPATH=. python -m unittest tests.test_wlcr_sea_model -v
```

聚焦测试覆盖季节索引、专家定义、删除值不泄漏、精确硬掩码、Entmax 归一化、
有界残差和审计范围包含关系。

## 在本地运行公开方法 Demo

```bash
python demo/app.py
```

打开 Gradio 输出的本地地址。Examples 区已经提供内置合成请求。

## 构建本网站

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
mkdocs serve
```

`mkdocs build --strict` 会同时构建中英文页面，并把配置或导航警告视为错误。
