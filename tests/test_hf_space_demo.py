from __future__ import annotations

import json
import unittest
from pathlib import Path

from matplotlib._pylab_helpers import Gcf
import numpy as np
import torch

from demo import runtime
from demo.model_loader import (
    CHECKPOINT_SPECS,
    MODEL_REVISION,
    MODEL_VARIANT,
    load_ensemble,
    load_a6_ensemble,
)
from experiments import wlcr_sea_model as sea


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "demo" / "examples" / "synthetic_traffic.csv"


class HuggingFaceSpaceDemoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ensemble = load_a6_ensemble()
        cls.clean = runtime.run_a6_forecast(SAMPLE, ensemble=cls.ensemble)

    def test_bundled_request_matches_the_public_contract(self) -> None:
        path, window = runtime.load_request(SAMPLE)
        self.assertEqual(path, SAMPLE)
        self.assertEqual(len(window.rows), 336)
        self.assertEqual(window.cell, "synthetic-cell")
        self.assertEqual(window.gaps, ())

    def test_public_registry_is_complete_pinned_and_cached(self) -> None:
        self.assertIs(self.ensemble, load_a6_ensemble())
        self.assertIs(self.ensemble, load_ensemble())
        self.assertEqual(self.ensemble.variant, MODEL_VARIANT)
        self.assertEqual(self.ensemble.revision, MODEL_REVISION)
        self.assertEqual(tuple(member.seed for member in self.ensemble.members), (42, 43, 44, 45, 46))
        self.assertEqual(
            tuple(member.sha256 for member in self.ensemble.members),
            tuple(spec.sha256 for spec in CHECKPOINT_SPECS),
        )
        self.assertTrue(
            all(member.prior_log.shape == (24, 4) for member in self.ensemble.members)
        )

    def test_clean_a6_path_is_finite_normalized_and_auditable(self) -> None:
        result = self.clean
        self.assertTrue(result.variant == "A6_mixed_aug")
        self.assertEqual(result.prediction.shape, (24, 4))
        self.assertEqual(result.attention.shape, (24, 4, 8))
        self.assertEqual(len(result.members), 5)
        self.assertTrue(np.isfinite(result.prediction).all())
        for member in result.members:
            self.assertTrue(
                np.allclose(member.attention.sum(axis=-1), 1.0, atol=1e-7)
            )
            self.assertEqual(
                float(np.sum(member.attention[~result.availability])), 0.0
            )
        self.assertTrue(np.all(result.prediction >= result.lower_envelope - 1e-5))
        self.assertTrue(np.all(result.prediction <= result.upper_envelope + 1e-5))
        self.assertEqual(result.applied_rate, 0.0)
        expected_first_hour = np.asarray(
            [17.2086716, 24.8728180, 32.3873444, 13.6604710], dtype=np.float32
        )
        self.assertTrue(
            np.allclose(result.prediction[0], expected_first_hour, rtol=1e-6, atol=1e-5)
        )

    def test_space_output_matches_the_core_five_member_evaluation(self) -> None:
        _, window = runtime.load_request(SAMPLE)
        values, mask = runtime._request_arrays(window)
        history_log = np.where(mask, np.log1p(values), 0.0).astype(np.float32)
        predictions = []
        with torch.inference_mode():
            for member in self.ensemble.members:
                batch = sea.build_expert_batch(history_log, mask, member.prior_log)
                output = member.model(
                    torch.from_numpy(batch.values),
                    torch.from_numpy(batch.availability),
                    torch.from_numpy(batch.reliability),
                    torch.from_numpy(batch.context),
                )
                predictions.append(
                    np.asarray(
                        sea.prediction_from_log(output["prediction_log"].cpu().numpy()[0]),
                        dtype=np.float32,
                    )
                )
        reference = np.mean(predictions, axis=0)
        self.assertTrue(np.allclose(self.clean.prediction, reference, rtol=1e-6, atol=1e-6))

    def test_structured_outage_removes_experts_without_leaking_mass(self) -> None:
        result = runtime.run_a6_forecast(
            SAMPLE,
            scenario="recent_tail",
            missing_rate=0.2,
            ensemble=self.ensemble,
        )
        self.assertGreater(float(np.mean(~result.effective_mask)), 0.15)
        expected_removed = float(
            np.count_nonzero(result.original_mask & ~result.effective_mask)
            / np.count_nonzero(result.original_mask)
        )
        self.assertAlmostEqual(result.applied_rate, expected_removed)
        self.assertNotAlmostEqual(result.applied_rate, 0.2)
        self.assertTrue(np.any(~result.availability))
        for member in result.members:
            self.assertEqual(
                float(np.sum(member.attention[~result.availability])), 0.0
            )
            self.assertTrue(
                np.allclose(member.attention.sum(axis=-1), 1.0, atol=1e-7)
            )
        _, audit_path = runtime.export_outputs(result)
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        self.assertEqual(audit["missingness"]["seed"], runtime.DEMO_SEED)
        self.assertGreater(len(audit["missingness"]["removed_positions"]), 0)
        self.assertGreater(
            audit["missingness"]["removed_fraction_of_original_observations"], 0
        )
        self.assertAlmostEqual(
            audit["missingness"]["actually_removed_fraction_of_original_observations"],
            expected_removed,
        )
        self.assertAlmostEqual(audit["missingness"]["applied_rate"], expected_removed)
        self.assertAlmostEqual(
            audit["missingness"]["final_observed_fraction"],
            float(np.mean(result.effective_mask)),
        )
        self.assertEqual(len(audit["missingness"]["effective_mask"]), 336)

    def test_exports_record_real_a6_identity_and_per_seed_outputs(self) -> None:
        forecast_path, audit_path = runtime.export_outputs(self.clean)
        self.assertTrue(Path(forecast_path).is_file())
        self.assertTrue(Path(audit_path).is_file())
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        self.assertEqual(audit["schema"], "wlcr-sea-audit/v4")
        self.assertTrue(audit["paper_model"])
        self.assertEqual(audit["variant"], "A6_mixed_aug")
        self.assertEqual(audit["ensemble"]["revision"], MODEL_REVISION)
        self.assertEqual(len(audit["ensemble"]["members"]), 5)
        self.assertEqual(
            [member["seed"] for member in audit["ensemble"]["members"]],
            [42, 43, 44, 45, 46],
        )
        self.assertTrue(audit["checks"]["unavailable_expert_weight_is_zero"])
        self.assertTrue(audit["checks"]["prediction_within_ensemble_envelope"])
        self.assertEqual(
            audit["source"]["repository"],
            "https://github.com/rudykon/WLCR-SEA_Predictor",
        )
        self.assertTrue(audit["source"]["commit"])
        for key in (
            "runtime_version",
            "python_version",
            "torch_version",
            "numpy_version",
            "pandas_version",
            "gradio_version",
        ):
            self.assertIn(key, audit["source"])
        self.assertEqual(audit["missingness"]["scenario"], "none")
        self.assertEqual(audit["missingness"]["applied_rate"], 0.0)
        self.assertEqual(audit["missingness"]["removed_positions"], [])
        self.assertTrue(
            all(member["checks"]["passed"] for member in audit["ensemble"]["members"])
        )
        self.assertTrue(
            all(
                member["checks"]["unavailable_weight_violation_count"] == 0
                and member["checks"]["weight_normalization_violation_count"] == 0
                and member["checks"]["lower_bound_violation_count"] == 0
                and member["checks"]["upper_bound_violation_count"] == 0
                for member in audit["ensemble"]["members"]
            )
        )
        self.assertIn(
            "do not exactly decompose",
            audit["ensemble_output"]["routing_summary_note"],
        )
        self.assertEqual(
            (forecast_path, audit_path), runtime.export_outputs(self.clean)
        )

    def test_metric_and_horizon_are_view_only(self) -> None:
        prediction = self.clean.prediction.copy()
        managers_before = Gcf.get_num_fig_managers()
        forecast_figure = runtime.make_forecast_figure(self.clean, "ul_users", "en")
        expert_figure = runtime.make_expert_figure(self.clean, "ul_users", 24, "en")
        rows = runtime.expert_dataframe(self.clean, "ul_users", 24, "en")
        self.assertEqual(len(forecast_figure.axes), 1)
        self.assertEqual(len(expert_figure.axes), 2)
        self.assertEqual(len(rows), 8)
        self.assertTrue(np.array_equal(prediction, self.clean.prediction))
        self.assertEqual(Gcf.get_num_fig_managers(), managers_before)
        forecast_figure.clear()
        expert_figure.clear()

    def test_reader_friendly_runtime_alias_matches_the_legacy_api(self) -> None:
        result = runtime.run_forecast(SAMPLE, ensemble=self.ensemble)
        self.assertTrue(
            np.array_equal(result.prediction, self.clean.prediction)
        )
        self.assertEqual(result.requested_rate, 0.0)
        self.assertEqual(runtime.run_a6_forecast.__kwdefaults__["missing_rate"], 0.0)

    def test_status_uses_distinct_configured_and_actual_rate_labels(self) -> None:
        status_en = runtime.status_markdown(self.clean, "en")
        status_zh = runtime.status_markdown(self.clean, "zh")
        self.assertIn("Configured removal", status_en)
        self.assertIn("Actually removed", status_en)
        self.assertIn("Final observed", status_en)
        self.assertIn("配置移除率", status_zh)
        self.assertIn("实际移除率", status_zh)
        self.assertIn("最终观测比例", status_zh)

    def test_demo_input_errors_have_stable_localized_messages(self) -> None:
        error = runtime.DemoInputError("wrong_row_count")
        self.assertEqual(error.code, "wrong_row_count")
        self.assertIn("336-row", error.localized("en"))
        self.assertIn("336 行", error.localized("zh"))
        self.assertNotEqual(error.localized("en"), error.localized("zh"))


if __name__ == "__main__":
    unittest.main()
