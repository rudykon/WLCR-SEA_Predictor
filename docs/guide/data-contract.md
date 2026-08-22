# Data contract

The repository's physical-window utility and public Demo accept the same
six-column CSV contract. One public-demo upload contains exactly one request.

## Required header

```text
时间,小区名称,小区上行平均激活用户数,小区下行平均激活用户数,下行平均使用的PRB个数,上行平均使用的PRB个数
```

| Field | Rule |
| --- | --- |
| `时间` | `YYYY/MM/DD HH:MM`, strictly hourly and increasing |
| `小区名称` | One non-empty value shared by all 336 rows |
| Four indicators | Finite, non-negative number, `NIL`, or blank |
| Row count | Exactly 336 data rows for the public Demo |
| Encoding | UTF-8 or UTF-8 with BOM |
| Upload size | At most 5 MB on the public Space |

The next 24 timestamps begin one hour after the final input row. Missingness is
represented by the observation mask; numerical fill values are not evidence.

## Minimal preview

```csv
时间,小区名称,小区上行平均激活用户数,小区下行平均激活用户数,下行平均使用的PRB个数,上行平均使用的PRB个数
2026/07/01 00:00,synthetic-cell,18.0,26.0,32.0,14.0
2026/07/01 01:00,synthetic-cell,18.1,26.2,NIL,14.1
```

Use the complete bundled request at
[`demo/examples/synthetic_traffic.csv`](https://github.com/rudykon/WLCR-SEA_Predictor/blob/main/demo/examples/synthetic_traffic.csv).
It is generated deterministically and contains no operator or participant data.

## Programmatic validation

```python
from Model.traffic_window_forecasting import read_traffic, split_physical_windows

rows = read_traffic("request.csv")
windows = split_physical_windows(rows)
assert len(windows) == 1
assert not windows[0].gaps
```

!!! warning "Public infrastructure"
    Do not upload confidential operational traffic to the public Space. The
    application does not intentionally persist requests, but Hugging Face is a
    shared hosting environment. Run `demo/app.py` inside your own controlled
    environment for sensitive data.
