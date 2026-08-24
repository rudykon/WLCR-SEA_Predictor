# CSV 输入格式

命令行工具和公开 Demo 使用相同的六列 CSV 格式。每次上传一个小区连续 336 小时的数据，每小时一行。

## 必需表头

```text
时间,小区名称,小区上行平均激活用户数,小区下行平均激活用户数,下行平均使用的PRB个数,上行平均使用的PRB个数
```

| 字段 | 规则 |
| --- | --- |
| `时间` | `YYYY/MM/DD HH:MM`，严格按小时递增 |
| `小区名称` | 336 行使用同一个非空值 |
| 四个指标 | 有限非负数、`NIL` 或空字段 |
| 行数 | 公开 Demo 必须恰好 336 个数据行 |
| 编码 | UTF-8 或带 BOM 的 UTF-8 |
| 上传大小 | 公开 Space 最多 5 MB |

预测从最后一条记录的下一小时开始，覆盖未来 24 小时。`NIL` 或空字段表示该值缺失。模型内部使用布尔标记保留这一信息，因此不会把占位数误认为真实观测。

## 最小预览

```csv
时间,小区名称,小区上行平均激活用户数,小区下行平均激活用户数,下行平均使用的PRB个数,上行平均使用的PRB个数
2026/07/01 00:00,synthetic-cell,18.0,26.0,32.0,14.0
2026/07/01 01:00,synthetic-cell,18.1,26.2,NIL,14.1
```

完整的内置样例见 [`demo/examples/synthetic_traffic.csv`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/demo/examples/synthetic_traffic.csv)。该文件由程序生成，每次内容相同，不包含任何真实运营商或参与者数据。

## 程序化验证

```python
from Model.traffic_window_forecasting import read_traffic, split_physical_windows

rows = read_traffic("request.csv")
windows = split_physical_windows(rows)
assert len(windows) == 1
assert not windows[0].gaps
```

!!! warning "请勿上传机密数据"
    请勿将运营商的机密流量数据上传到公开 Space。该应用并非为了保存请求，但 Hugging Face 属于共享的公开基础设施。如需处理敏感数据，请在自有受控环境中运行 `demo/app.py`。
