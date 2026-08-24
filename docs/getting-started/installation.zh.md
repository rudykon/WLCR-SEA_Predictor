# 安装

需要 Python 3.11+。CI 和 Demo 使用 Python 3.12。

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

测试覆盖候选选择、缺失值遮蔽、权重和修正范围。

## 在本地运行在线 Demo

```bash
python demo/app.py
```

打开 Gradio 地址并选择内置样例。

## 构建本网站

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
mkdocs serve
```

严格模式会构建中英文页面，并在出现警告时失败。
