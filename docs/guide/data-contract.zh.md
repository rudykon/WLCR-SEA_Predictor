# 数据契约

仓库中的物理窗口工具与公开 Demo 使用同一套六列 CSV 契约。公开 Demo 的一次上传只包含一个请求。

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

预测的 24 个时间戳从最后输入行后一小时开始。缺失由观测掩码表示，数值填充不是证据。

## 最小预览

```csv
时间,小区名称,小区上行平均激活用户数,小区下行平均激活用户数,下行平均使用的PRB个数,上行平均使用的PRB个数
2026/07/01 00:00,synthetic-cell,18.0,26.0,32.0,14.0
2026/07/01 01:00,synthetic-cell,18.1,26.2,NIL,14.1
```

完整内置请求见
[`demo/examples/synthetic_traffic.csv`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/demo/examples/synthetic_traffic.csv)。
该文件由程序确定性生成，不包含运营商或参与者数据。

## 程序化验证

```python
from Model.traffic_window_forecasting import read_traffic, split_physical_windows

rows = read_traffic("request.csv")
windows = split_physical_windows(rows)
assert len(windows) == 1
assert not windows[0].gaps
```

!!! warning "公开基础设施"
    请勿向公开 Space 上传机密运营流量。应用不会主动持久化请求，但 Hugging Face 属于共享托管环境。
    对敏感数据，请在自有受控环境中运行 `demo/app.py`。
