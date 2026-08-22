from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from demo import runtime


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "demo" / "examples" / "synthetic_traffic.csv"


class HuggingFaceSpaceDemoTest(unittest.TestCase):
    def test_bundled_request_matches_the_public_contract(self) -> None:
        path, window = runtime.load_request(SAMPLE)
        self.assertEqual(path, SAMPLE)
        self.assertEqual(len(window.rows), 336)
        self.assertEqual(window.cell, "synthetic-cell")
        self.assertEqual(window.gaps, ())

    def test_clean_fixed_path_is_finite_normalized_and_auditable(self) -> None:
        result = runtime.run_audit_demo(SAMPLE)
        self.assertEqual(result.prediction.shape, (24, 4))
        self.assertEqual(result.attention.shape, (24, 4, 8))
        self.assertTrue(np.isfinite(result.prediction).all())
        self.assertTrue(np.allclose(result.attention.sum(axis=-1), 1.0, atol=1e-7))
        self.assertEqual(float(np.sum(result.attention[~result.availability])), 0.0)
        self.assertTrue(np.all(result.prediction >= result.lower_envelope - 1e-5))
        self.assertTrue(np.all(result.prediction <= result.upper_envelope + 1e-5))

    def test_structured_outage_removes_experts_without_leaking_mass(self) -> None:
        result = runtime.run_audit_demo(
            SAMPLE,
            scenario_label="Recent-tail outage / 最近时段中断",
            missing_rate=0.5,
            metric_label="DL active users / 下行激活用户",
            horizon=12,
        )
        self.assertGreater(float(np.mean(~result.effective_mask)), 0.45)
        self.assertTrue(np.any(~result.availability))
        self.assertEqual(float(np.sum(result.attention[~result.availability])), 0.0)
        self.assertTrue(np.allclose(result.attention.sum(axis=-1), 1.0, atol=1e-7))

    def test_demo_fallback_cannot_read_values_removed_by_the_effective_mask(self) -> None:
        _, window = runtime.load_request(SAMPLE)
        values, original_mask = runtime._request_arrays(window)
        effective_mask = original_mask.copy()
        effective_mask[:, -168:, :] = False
        original_prior = runtime._request_fallback_prior(values, effective_mask)

        poisoned = values.copy()
        poisoned[~effective_mask] = 1_000_000_000.0
        poisoned_prior = runtime._request_fallback_prior(poisoned, effective_mask)
        self.assertTrue(np.array_equal(original_prior, poisoned_prior))

    def test_exports_and_visuals_are_materialized(self) -> None:
        result = runtime.run_audit_demo(SAMPLE)
        forecast_path, audit_path = runtime.export_outputs(result)
        self.assertTrue(Path(forecast_path).is_file())
        self.assertTrue(Path(audit_path).is_file())
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        self.assertEqual(audit["method"], "A0_fixed")
        self.assertFalse(audit["paper_model"])
        self.assertEqual(audit["envelope"]["violations"], 0)
        forecast_figure = runtime.make_forecast_figure(result)
        expert_figure = runtime.make_expert_figure(result)
        self.assertEqual(len(forecast_figure.axes), 4)
        self.assertEqual(len(expert_figure.axes), 2)


if __name__ == "__main__":
    unittest.main()
