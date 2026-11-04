"""Generate the deterministic, non-user traffic request bundled with the demo."""

from __future__ import annotations

import csv
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model.traffic_window_forecasting import CSV_HEADER, WINDOW_ROWS


OUTPUT = Path(__file__).resolve().parent / "examples" / "synthetic_traffic.csv"


def _signals(hour_index: int, timestamp: datetime) -> tuple[float, float, float, float]:
    hour = timestamp.hour
    weekday = timestamp.weekday()
    daytime = max(0.0, math.sin(math.pi * (hour - 6) / 16.0))
    commute = math.exp(-((hour - 9.0) / 2.5) ** 2) + 0.8 * math.exp(
        -((hour - 19.0) / 3.0) ** 2
    )
    weekend = 0.88 if weekday >= 5 else 1.0
    slow = 1.0 + 0.025 * math.sin(2.0 * math.pi * hour_index / (24.0 * 7.0))
    ul_users = (18.0 + 13.0 * daytime + 4.0 * commute) * weekend * slow
    dl_users = (26.0 + 22.0 * daytime + 7.0 * commute) * weekend * slow
    dl_prb = (32.0 + 33.0 * daytime + 10.0 * commute) * weekend * slow
    ul_prb = (14.0 + 18.0 * daytime + 7.0 * commute) * weekend * slow
    return ul_users, dl_users, dl_prb, ul_prb


def generate(path: Path = OUTPUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 7, 1, 0, 0)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for index in range(WINDOW_ROWS):
            timestamp = start + timedelta(hours=index)
            writer.writerow(
                (
                    timestamp.strftime("%Y/%m/%d %H:%M"),
                    "synthetic-cell",
                    *(f"{value:.4f}" for value in _signals(index, timestamp)),
                )
            )
    return path


if __name__ == "__main__":
    print(generate())
