# 安装

WLCR-SEA 是一个 Python 研究代码仓库。建议使用 Python 3.11 或更高版本；网站和托管 Demo 已在 Python 3.12 环境中通过验证。

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

这些测试会检查模型是否选择了正确的历史时刻、缺失位置的占位数是否被排除、候选权重是否合法，以及最终修正是否保持在设定范围内。

## 在本地运行在线 Demo

```bash
python demo/app.py
```

打开 Gradio 输出的本地地址。页面的“示例”区域已经提供了一份内置合成数据。

## 构建本网站

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
mkdocs serve
```

`mkdocs build --strict` 会同时构建中英文页面，并把配置或导航警告视为错误。
