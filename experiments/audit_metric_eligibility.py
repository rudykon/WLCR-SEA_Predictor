from __future__ import annotations

"""Audit the MASE eligibility denominator used by Revision-8 reporting.

The audit reads the registered training trace only.  It reconstructs the
predeclared temporal holdout and records which request--indicator groups have
a finite, strictly positive 168-hour seasonal scale.  This makes the MASE
exclusion rule explicit without opening the finals inference file.
"""

import argparse
from pathlib import Path

import numpy as np

from experiments import train_neural_baselines as neural
from experiments import train_wlcr_sea as runner
from experiments import wlcr_sea_model as sea


SCHEMA_VERSION = 1
SCALE_EPSILON = 1e-12
DEFAULT_OUTPUT = Path("artifacts/revision8/metric_coverage")


def mase_coverage(
    targets: np.ndarray,
    target_masks: np.ndarray,
    mase_scales: np.ndarray,
) -> dict[str, object]:
    """Summarize request-local seasonal-scale coverage by indicator.

    A group is eligible precisely when it has at least one observed target
    value and the precomputed request--indicator scale is finite and exceeds
    ``SCALE_EPSILON``.  Every observed horizon in an eligible group enters
    MASE; unavailable target values remain excluded from all metrics.
    """
    y = np.asarray(targets, dtype=np.float64)
    masks = np.asarray(target_masks, dtype=bool)
    scales = np.asarray(mase_scales, dtype=np.float64)
    expected_tail = (sea.FORECAST_HOURS, sea.TARGET_COUNT)
    if y.shape != masks.shape or y.ndim != 3 or y.shape[1:] != expected_tail:
        raise ValueError("targets and target masks must be aligned [request,24,4] tensors")
    if scales.shape != (y.shape[0], sea.TARGET_COUNT):
        raise ValueError("MASE scales must be an aligned [request,4] tensor")

    observed = masks & np.isfinite(y)
    observed_group = np.any(observed, axis=1)
    scale_eligible = np.isfinite(scales) & (scales > SCALE_EPSILON)
    eligible_group = observed_group & scale_eligible
    eligible_horizon = observed & eligible_group[:, None, :]

    per_indicator: list[dict[str, object]] = []
    for metric, name in enumerate(sea.METRIC_NAMES):
        groups_with_targets = int(np.sum(observed_group[:, metric]))
        groups_eligible = int(np.sum(eligible_group[:, metric]))
        observed_horizons = int(np.sum(observed[:, :, metric]))
        eligible_horizons = int(np.sum(eligible_horizon[:, :, metric]))
        per_indicator.append(
            {
                "indicator": name,
                "request_indicator_groups_with_observed_targets": groups_with_targets,
                "mase_eligible_request_indicator_groups": groups_eligible,
                "mase_excluded_request_indicator_groups": groups_with_targets - groups_eligible,
                "group_coverage": (
                    float(groups_eligible / groups_with_targets)
                    if groups_with_targets
                    else 0.0
                ),
                "observed_forecast_values": observed_horizons,
                "mase_eligible_forecast_values": eligible_horizons,
                "forecast_value_coverage": (
                    float(eligible_horizons / observed_horizons)
                    if observed_horizons
                    else 0.0
                ),
            }
        )

    group_total = int(np.sum(observed_group))
    group_eligible_total = int(np.sum(eligible_group))
    value_total = int(np.sum(observed))
    value_eligible_total = int(np.sum(eligible_horizon))
    return {
        "scale_definition": (
            "mean absolute finite endpoint difference |x[u,q]-x[u-168,q]| "
            "over the 336-hour request history"
        ),
        "seasonal_lag_hours": 168,
        "scale_epsilon": SCALE_EPSILON,
        "exclusion_rule": (
            "exclude only request-indicator groups with no finite 168-hour "
            "endpoint pair or with scale <= 1e-12"
        ),
        "request_indicator_groups_with_observed_targets": group_total,
        "mase_eligible_request_indicator_groups": group_eligible_total,
        "mase_excluded_request_indicator_groups": group_total - group_eligible_total,
        "group_coverage": float(group_eligible_total / group_total) if group_total else 0.0,
        "observed_forecast_values": value_total,
        "mase_eligible_forecast_values": value_eligible_total,
        "forecast_value_coverage": float(value_eligible_total / value_total) if value_total else 0.0,
        "per_indicator": per_indicator,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return value


def main() -> int:
    args = parser().parse_args()
    root = runner.project_root()
    output = runner.resolve_output(args.output)
    allowed = (root / "artifacts/revision8").resolve(strict=False)
    if not output.is_relative_to(allowed):
        raise ValueError("Revision-8 metric audit output must remain under artifacts/revision8")
    output.mkdir(parents=True, exist_ok=True)

    train_path = neural.resolve_train_path()
    before = neural.sha256_file(train_path)
    series = neural.read_training_series(train_path)
    arrays, dataset_report = neural.build_window_arrays(series)
    dataset = neural.CachedDataset(root=output, **arrays)
    holdout = dataset.indices_for_dates(neural.HOLDOUT_DATES)
    report = mase_coverage(
        np.asarray(dataset.targets[holdout]),
        np.asarray(dataset.target_masks[holdout]),
        np.asarray(dataset.mase_scales[holdout]),
    )
    after = neural.sha256_file(train_path)
    if before != after:
        raise RuntimeError("registered training data changed during metric coverage audit")
    report.update(
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_status": "exploratory_redesign_on_existing_trace",
            "registered_train_file": str(neural.REGISTERED_TRAIN),
            "registered_train_sha256_before": before,
            "registered_train_sha256_after": after,
            "dataset_report": dataset_report,
            "holdout_requests": int(len(holdout)),
            "finals_test_opened": False,
        }
    )
    runner.atomic_json(output / "summary.json", report)
    runner.atomic_json(output / "manifest.json", runner.output_manifest(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
