# CSV 输入格式

命令行和 Demo 使用同一种六列 CSV：一个小区、连续 336 小时、每小时一行。

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

预测从最后一行的下一小时开始，共 24 小时。`NIL` 或空字段表示缺失；掩码会阻止占位数参与计算。

## 最小预览

```csv
时间,小区名称,小区上行平均激活用户数,小区下行平均激活用户数,下行平均使用的PRB个数,上行平均使用的PRB个数
2026/07/01 00:00,synthetic-cell,18.0,26.0,32.0,14.0
2026/07/01 01:00,synthetic-cell,18.1,26.2,NIL,14.1
```

内置样例见 [`demo/examples/synthetic_traffic.csv`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/demo/examples/synthetic_traffic.csv)。它由程序生成，内容固定，不含真实运营商数据。

## 程序化验证

```python
from Model.traffic_window_forecasting import read_traffic, split_physical_windows

rows = read_traffic("request.csv")
windows = split_physical_windows(rows)
assert len(windows) == 1
assert not windows[0].gaps
```

!!! warning "公开 Space"
    请勿上传机密流量。敏感数据请在本地运行 `demo/app.py`。
