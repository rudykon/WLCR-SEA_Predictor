from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Model.traffic_window_forecasting import BaselineConfig, ContractError
from experiments.frozen_seasonal_baseline import (
    SEASONAL_CONFIG_ENTRYPOINTS,
    FROZEN_SEASONAL_CONFIG_PATH,
    FROZEN_SEASONAL_CONFIG_SHA256,
    SEASONAL_CANDIDATE_ORDER,
    load_frozen_seasonal_baseline_config,
    verify_seasonal_config_consistency,
)


class FrozenSeasonalConfigTest(unittest.TestCase):
    def test_registered_config_is_hash_bound_to_the_implemented_candidate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        frozen = load_frozen_seasonal_baseline_config(root)
        report = frozen.report()

        self.assertEqual(frozen.sha256, FROZEN_SEASONAL_CONFIG_SHA256)
        self.assertEqual(frozen.config.name, "weekly_median_s097")
        self.assertEqual(
            frozen.config.weights,
            (0.0, 0.7, 0.2, 0.1, 0.0, 0.0),
        )
        self.assertEqual(frozen.config.scales, (0.97, 0.97, 0.97, 0.97))
        self.assertEqual(report["path"], FROZEN_SEASONAL_CONFIG_PATH)
        self.assertEqual(report["sha256"], FROZEN_SEASONAL_CONFIG_SHA256)
        self.assertEqual(report["candidate_order"], list(SEASONAL_CANDIDATE_ORDER))

    def test_any_config_byte_drift_is_rejected_before_use(self) -> None:
        source_root = Path(__file__).resolve().parents[1]
        source = source_root / FROZEN_SEASONAL_CONFIG_PATH
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / FROZEN_SEASONAL_CONFIG_PATH
            destination.parent.mkdir(parents=True)
            destination.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(ContractError, "SHA256 mismatch"):
                load_frozen_seasonal_baseline_config(root)

    def test_implemented_candidate_drift_is_rejected(self) -> None:
        root = Path(__file__).resolve().parents[1]
        drifted = BaselineConfig(
            "weekly_median_s097",
            (0.0, 0.6, 0.3, 0.1, 0.0, 0.0),
            (0.97, 0.97, 0.97, 0.97),
        )
        with patch(
            "experiments.frozen_seasonal_baseline.baseline_candidates",
            return_value=[drifted],
        ):
            with self.assertRaisesRegex(ContractError, "implemented candidate"):
                load_frozen_seasonal_baseline_config(root)

    def test_seasonal_config_entrypoints_match_source_hashes_and_frozen_values(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = verify_seasonal_config_consistency(root).report()
        entrypoints = report["seasonal_config_entrypoints"]

        self.assertEqual(report["status"], "verified")
        self.assertEqual(len(entrypoints), 2)
        self.assertEqual(
            [(item["path"], item["sha256"]) for item in entrypoints],
            [(path, sha256) for path, sha256, _ in SEASONAL_CONFIG_ENTRYPOINTS],
        )
        self.assertTrue(
            all(
                item["weights"] == [0.0, 0.7, 0.2, 0.1, 0.0, 0.0]
                and item["scales"] == [0.97, 0.97, 0.97, 0.97]
                for item in entrypoints
            )
        )

    def test_seasonal_config_entrypoint_byte_drift_is_rejected(self) -> None:
        source_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                FROZEN_SEASONAL_CONFIG_PATH,
                *(path for path, _, _ in SEASONAL_CONFIG_ENTRYPOINTS),
            ):
                source = source_root / relative
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
            run_path = root / SEASONAL_CONFIG_ENTRYPOINTS[0][0]
            run_path.write_bytes(run_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ContractError, "entrypoint SHA256 mismatch"):
                verify_seasonal_config_consistency(root)


if __name__ == "__main__":
    unittest.main()
