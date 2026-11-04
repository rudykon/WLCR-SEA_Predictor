from __future__ import annotations

"""Deterministic missing-telemetry protocols shared by paper experiments.

Corruption is generated on each cell's unique absolute time axis and then
projected into overlapping 336-hour requests.  Consequently, a real
cell--timestamp--indicator has one corruption state in every request that
contains it.  Rates on the unique time axis and rates on repeated window
exposures are deliberately reported as different estimands.
"""

import hashlib
from typing import Mapping

import numpy as np


INPUT_HOURS = 336
TARGET_COUNT = 4
SUPPORTED_MECHANISMS = (
    "none",
    "mcar",
    "block",
    "recent_tail",
    "asynchronous",
    "mixed",
)


def stable_uniform(*tokens: object) -> float:
    """Return a deterministic pseudo-uniform value in ``[0, 1)``."""
    payload = "|".join(str(token) for token in tokens).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64)


def global_corruption_mask(
    cells: np.ndarray,
    history_end_hours: np.ndarray,
    *,
    mechanism: str,
    requested_rate: float,
    seed: int,
) -> np.ndarray:
    """Generate an absolute-cell-timeline mask before window extraction.

    ``recent_tail`` means the tail of the complete unique timeline available
    for one cell in the evaluated split.  It is *not* a request-relative tail;
    that choice preserves one state for a timestamp shared by overlapping
    requests.
    """
    cells = np.asarray(cells).astype(str)
    end_hours = np.asarray(history_end_hours, dtype=np.int64)
    if cells.ndim != 1 or end_hours.shape != cells.shape:
        raise ValueError("cells and history_end_hours must be aligned vectors")
    if mechanism not in SUPPORTED_MECHANISMS:
        raise ValueError(f"unknown missingness mechanism: {mechanism}")
    if not 0.0 <= requested_rate <= 1.0:
        raise ValueError("requested_rate must be in [0,1]")
    output = np.zeros((len(cells), INPUT_HOURS, TARGET_COUNT), dtype=bool)
    if mechanism == "none" or requested_rate == 0.0:
        return output

    offsets = np.arange(INPUT_HOURS, dtype=np.int64) - (INPUT_HOURS - 1)
    for cell in sorted(set(cells.tolist())):
        window_indices = np.flatnonzero(cells == cell)
        hour_matrix = end_hours[window_indices, None] + offsets[None, :]
        unique_hours = np.unique(hour_matrix)
        local_mechanism = mechanism
        if mechanism == "mixed":
            choices = ("mcar", "block", "recent_tail", "asynchronous")
            choice = min(
                int(stable_uniform(seed, cell, "mixed") * len(choices)),
                len(choices) - 1,
            )
            local_mechanism = choices[choice]

        removed = np.zeros((len(unique_hours), TARGET_COUNT), dtype=bool)
        if local_mechanism == "mcar":
            for position, hour in enumerate(unique_hours):
                if stable_uniform(seed, cell, int(hour), "mcar") < requested_rate:
                    removed[position, :] = True
        elif local_mechanism == "asynchronous":
            for position, hour in enumerate(unique_hours):
                for metric in range(TARGET_COUNT):
                    removed[position, metric] = (
                        stable_uniform(seed, cell, int(hour), metric, "async")
                        < requested_rate
                    )
        elif local_mechanism in {"block", "recent_tail"}:
            length = int(round(requested_rate * len(unique_hours)))
            if requested_rate > 0.0:
                length = max(1, length)
            length = min(length, len(unique_hours))
            if local_mechanism == "recent_tail":
                start = len(unique_hours) - length
            else:
                maximum = len(unique_hours) - length
                start = int(stable_uniform(seed, cell, "block") * (maximum + 1))
                start = min(start, maximum)
            removed[start : start + length, :] = True

        positions = np.searchsorted(unique_hours, hour_matrix)
        output[window_indices] = removed[positions]
    return output


def _rate_statistics(original: np.ndarray, extra: np.ndarray) -> dict[str, float | int]:
    original = np.asarray(original, dtype=bool)
    extra = np.asarray(extra, dtype=bool)
    if original.shape != extra.shape:
        raise ValueError("corruption statistics require aligned masks")
    newly_removed = original & extra
    final_missing = ~original | extra
    observed = int(np.sum(original))
    return {
        "positions": int(original.size),
        "observed_positions": observed,
        "original_missing_rate": float(np.mean(~original)),
        "selected_for_corruption_rate": float(np.mean(extra)),
        "newly_removed_rate": float(np.mean(newly_removed)),
        "newly_removed_fraction_of_observed": (
            float(np.sum(newly_removed) / observed) if observed else 0.0
        ),
        "final_total_missing_rate": float(np.mean(final_missing)),
    }


def window_exposure_statistics(
    original_masks: np.ndarray, additional_missing: np.ndarray
) -> dict[str, float | int]:
    """Return rates over repeated request-window exposures."""
    return _rate_statistics(original_masks, additional_missing)


def unique_cell_time_statistics(
    original_masks: np.ndarray,
    additional_missing: np.ndarray,
    cells: np.ndarray,
    history_end_hours: np.ndarray,
) -> dict[str, float | int]:
    """Return rates over unique cell--hour--indicator positions.

    Repeated exposures must agree on both the original observation state and
    the generated corruption state.  A disagreement is a protocol violation.
    """
    original = np.asarray(original_masks, dtype=bool)
    extra = np.asarray(additional_missing, dtype=bool)
    cells = np.asarray(cells).astype(str)
    end_hours = np.asarray(history_end_hours, dtype=np.int64)
    if original.shape != extra.shape:
        raise ValueError("corruption statistics require aligned masks")
    if original.shape != (len(cells), INPUT_HOURS, TARGET_COUNT):
        raise ValueError("history masks do not match the supplied window identities")
    if end_hours.shape != cells.shape:
        raise ValueError("history_end_hours must align with cells")

    offsets = np.arange(INPUT_HOURS, dtype=np.int64) - (INPUT_HOURS - 1)
    unique_original: list[np.ndarray] = []
    unique_extra: list[np.ndarray] = []
    for cell in sorted(set(cells.tolist())):
        selected = np.flatnonzero(cells == cell)
        hours = end_hours[selected, None] + offsets[None, :]
        unique_hours = np.unique(hours)
        cell_original = np.zeros((len(unique_hours), TARGET_COUNT), dtype=bool)
        cell_extra = np.zeros_like(cell_original)
        initialized = np.zeros(len(unique_hours), dtype=bool)
        positions = np.searchsorted(unique_hours, hours)
        for local_window, global_window in enumerate(selected):
            for offset_index, position in enumerate(positions[local_window]):
                observed_value = original[global_window, offset_index]
                extra_value = extra[global_window, offset_index]
                if initialized[position]:
                    if not np.array_equal(cell_original[position], observed_value):
                        raise ValueError(
                            "original observation mask is inconsistent across overlapping requests"
                        )
                    if not np.array_equal(cell_extra[position], extra_value):
                        raise ValueError(
                            "corruption mask is inconsistent across overlapping requests"
                        )
                else:
                    cell_original[position] = observed_value
                    cell_extra[position] = extra_value
                    initialized[position] = True
        unique_original.append(cell_original)
        unique_extra.append(cell_extra)
    return _rate_statistics(
        np.concatenate(unique_original, axis=0),
        np.concatenate(unique_extra, axis=0),
    )


def corruption_statistics(
    original_masks: np.ndarray,
    additional_missing: np.ndarray,
    *,
    cells: np.ndarray | None = None,
    history_end_hours: np.ndarray | None = None,
    mechanism: str | None = None,
    requested_rate: float | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Report both unique-axis and repeated-exposure corruption rates.

    For backward compatibility, the three historical exposure keys are also
    repeated at the top level.
    """
    exposure = window_exposure_statistics(original_masks, additional_missing)
    payload: dict[str, object] = {
        "scope": "absolute_cell_timeline",
        "recent_tail_definition": "tail of each cell's unique evaluated timeline",
        "mechanism": mechanism,
        "requested_rate": requested_rate,
        "seed": seed,
        "window_exposure": exposure,
        "original_missing_rate": exposure["original_missing_rate"],
        "newly_removed_rate": exposure["newly_removed_rate"],
        "final_total_missing_rate": exposure["final_total_missing_rate"],
    }
    if (cells is None) != (history_end_hours is None):
        raise ValueError("cells and history_end_hours must be supplied together")
    if cells is not None and history_end_hours is not None:
        payload["unique_cell_time"] = unique_cell_time_statistics(
            original_masks,
            additional_missing,
            cells,
            history_end_hours,
        )
    return payload


def flatten_statistics(report: Mapping[str, object]) -> dict[str, object]:
    """Flatten a protocol report into stable CSV-friendly columns."""
    row: dict[str, object] = {
        "corruption_scope": report.get("scope", "absolute_cell_timeline"),
        "recent_tail_definition": report.get("recent_tail_definition", ""),
    }
    for prefix, key in (
        ("unique", "unique_cell_time"),
        ("exposure", "window_exposure"),
    ):
        values = report.get(key)
        if not isinstance(values, Mapping):
            continue
        for name, value in values.items():
            row[f"{prefix}_{name}"] = value
    return row
