from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from experiments import wlcr_sea as sea


class SeasonalExpertConstructionTest(unittest.TestCase):
    def test_midnight_origin_and_seasonal_indices(self) -> None:
        self.assertEqual(sea.seasonal_history_index(1, 1), 312)
        self.assertEqual(sea.seasonal_history_index(24, 1), 335)
        self.assertEqual(sea.seasonal_history_index(1, 7), 168)
        self.assertEqual(sea.seasonal_history_index(24, 7), 191)
        self.assertEqual(sea.seasonal_history_index(1, 14), 0)
        self.assertEqual(sea.seasonal_history_index(24, 14), 23)

    def test_eight_experts_match_their_definitions(self) -> None:
        history = np.zeros((1, 336, 4), dtype=np.float32)
        for metric in range(4):
            history[0, :, metric] = np.arange(336, dtype=np.float32) + metric * 1000
        masks = np.ones_like(history, dtype=np.uint8)
        prior = np.full((24, 4), 99.0, dtype=np.float32)
        batch = sea.build_expert_batch(history, masks, prior, trend_clip=10.0)
        h = 0
        q = 0
        self.assertEqual(batch.values.shape, (1, 24, 4, 8))
        self.assertAlmostEqual(batch.values[0, h, q, 0], 312.0)
        self.assertAlmostEqual(batch.values[0, h, q, 1], 168.0)
        self.assertAlmostEqual(batch.values[0, h, q, 2], 0.0)
        expected_7 = np.median([312, 288, 264, 240, 216, 192, 168])
        expected_14 = np.median([sea.seasonal_history_index(1, d) for d in range(1, 15)])
        self.assertAlmostEqual(batch.values[0, h, q, 3], expected_7)
        self.assertAlmostEqual(batch.values[0, h, q, 4], expected_14)
        self.assertAlmostEqual(batch.values[0, h, q, 5], 178.0)
        self.assertAlmostEqual(batch.values[0, h, q, 6], 167.5)
        self.assertAlmostEqual(batch.values[0, h, q, 7], 99.0)
        self.assertTrue(np.all(batch.availability))
        self.assertAlmostEqual(batch.reliability[0, h, q, 3], 1.0)

    def test_removed_observation_cannot_leak_through_filled_value(self) -> None:
        history = np.ones((1, 336, 4), dtype=np.float32)
        history[0, 312, 0] = 1_000_000.0
        masks = np.ones_like(history, dtype=np.uint8)
        removed = np.zeros_like(history, dtype=bool)
        removed[0, 312, 0] = True
        prior = np.full((24, 4), 3.0, dtype=np.float32)
        batch = sea.build_expert_batch(history, masks, prior, additional_missing=removed)
        self.assertFalse(batch.availability[0, 0, 0, 0])
        self.assertEqual(batch.values[0, 0, 0, 0], 3.0)
        self.assertLess(batch.values[0, 0, 0, 3], 10.0)


    def test_training_prior_is_horizon_indicator_median(self) -> None:
        targets = np.ones((3, 24, 4), dtype=np.float32)
        targets[1] *= 9.0
        targets[2] *= 999.0
        masks = np.ones_like(targets, dtype=np.uint8)
        prior = sea.training_prior_log(targets, masks, [0, 1, 2])
        self.assertTrue(np.allclose(prior, np.log1p(9.0)))


class SparseAttentionTest(unittest.TestCase):
    def test_entmax_is_sparse_normalized_and_differentiable(self) -> None:
        logits = torch.tensor([[5.0, 1.0, 0.0, -2.0]], requires_grad=True)
        weights = sea.entmax15(logits)
        self.assertTrue(torch.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertGreaterEqual(int(torch.sum(weights == 0.0)), 1)
        objective = torch.sum(weights * torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        objective.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_available_set_entmax_matches_compacted_expert_sets(self) -> None:
        logits = torch.tensor(
            [[-1_000_000_000.0, -9.0, 2.0, -4.0], [3.0, -7.0, -2.0, 1.0]],
            requires_grad=True,
        )
        availability = torch.tensor(
            [[False, True, True, False], [True, False, False, True]]
        )
        weights = sea.available_set_entmax15(logits, availability)
        expected = torch.zeros_like(logits)
        expected[0, [1, 2]] = sea.entmax15(logits[0, [1, 2]])
        expected[1, [0, 3]] = sea.entmax15(logits[1, [0, 3]])
        self.assertTrue(torch.equal(weights[~availability], torch.zeros_like(weights[~availability])))
        self.assertTrue(torch.allclose(weights, expected, atol=1e-7))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-7))
        (weights * torch.tensor([[1.0, 2.0, 3.0, 4.0]])).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_reliability_loss_is_disabled_for_no_reliability_variant(self) -> None:
        prediction = torch.zeros((1, 24, 4), requires_grad=True)
        output = {
            "prediction_log": prediction,
            "residual": torch.zeros_like(prediction),
            "attention": torch.full((1, 24, 4, 8), 1.0 / 8.0),
        }
        target = torch.ones_like(prediction)
        target_mask = torch.ones_like(prediction)
        low_reliability = torch.zeros((1, 24, 4, 8))
        high_reliability = torch.ones((1, 24, 4, 8))
        low_loss, low_pieces = sea.sea_loss(
            output, target, target_mask, low_reliability, reliability_weight=0.0
        )
        high_loss, high_pieces = sea.sea_loss(
            output, target, target_mask, high_reliability, reliability_weight=0.0
        )
        self.assertEqual(low_pieces["reliability"], 0.0)
        self.assertEqual(high_pieces["reliability"], 0.0)
        self.assertTrue(torch.equal(low_loss, high_loss))

    def test_hard_mask_assigns_exact_zero_weight(self) -> None:
        torch.manual_seed(3)
        model = sea.WLCRSEA(sea.VARIANTS["A4_reliability"], token_dim=16, hidden_dim=32)
        values = torch.randn(2, 24, 4, 8)
        availability = torch.ones_like(values, dtype=torch.bool)
        availability[..., 0] = False
        reliability = availability.float()
        reliability[..., 7] = 1.0
        context = torch.randn(2, 4, 5)
        output = model(values, availability, reliability, context)
        self.assertTrue(torch.equal(output["attention"][..., 0], torch.zeros_like(output["attention"][..., 0])))
        self.assertTrue(torch.allclose(output["attention"].sum(-1), torch.ones(2, 24, 4), atol=1e-6))

    def test_bounded_residual_cannot_bypass_experts_unboundedly(self) -> None:
        torch.manual_seed(4)
        model = sea.WLCRSEA(
            sea.VARIANTS["A5_residual"], token_dim=16, hidden_dim=32, residual_bound=0.25
        )
        values = torch.randn(2, 24, 4, 8)
        availability = torch.ones_like(values, dtype=torch.bool)
        reliability = torch.ones_like(values)
        context = torch.randn(2, 4, 5)
        output = model(values, availability, reliability, context)
        self.assertLessEqual(float(torch.max(torch.abs(output["residual"]))), 0.250001)
        self.assertTrue(torch.allclose(output["prediction_log"], output["baseline_log"] + output["residual"]))


    def test_fixed_and_learned_static_router_parameterization(self) -> None:
        fixed = sea.WLCRSEA(sea.VARIANTS["A0_fixed"], token_dim=16, hidden_dim=32)
        global_router = sea.WLCRSEA(
            sea.VARIANTS["A0_global_static"], token_dim=16, hidden_dim=32
        )
        horizon_router = sea.WLCRSEA(
            sea.VARIANTS["A0_horizon_indicator"], token_dim=16, hidden_dim=32
        )
        self.assertEqual(sum(p.numel() for p in fixed.parameters() if p.requires_grad), 0)
        self.assertEqual(
            sum(p.numel() for p in global_router.parameters() if p.requires_grad),
            sea.TARGET_COUNT * sea.EXPERT_COUNT,
        )
        self.assertEqual(
            sum(p.numel() for p in horizon_router.parameters() if p.requires_grad),
            sea.FORECAST_HOURS * sea.TARGET_COUNT * sea.EXPERT_COUNT,
        )
        values = torch.randn(2, 24, 4, 8)
        availability = torch.ones_like(values, dtype=torch.bool)
        availability[..., 0] = False
        reliability = availability.float()
        context = torch.randn(2, 4, 5)
        for model in (global_router, horizon_router):
            output = model(values, availability, reliability, context)
            self.assertTrue(
                torch.equal(
                    output["attention"][..., 0],
                    torch.zeros_like(output["attention"][..., 0]),
                )
            )
            self.assertTrue(
                torch.allclose(
                    output["attention"].sum(-1), torch.ones(2, 24, 4), atol=1e-6
                )
            )

    def test_bounded_audit_envelope_contains_every_valid_prediction_log(self) -> None:
        torch.manual_seed(8)
        model = sea.WLCRSEA(
            sea.VARIANTS["A5_residual"],
            token_dim=16,
            hidden_dim=32,
            residual_bound=0.25,
        )
        values = torch.randn(3, 24, 4, 8)
        availability = torch.ones_like(values, dtype=torch.bool)
        availability[..., 0] = False
        reliability = availability.float()
        reliability[..., 7] = 1.0
        context = torch.randn(3, 4, 5)
        output = model(values, availability, reliability, context)
        lower, upper = sea.bounded_audit_envelope(
            values.numpy(), availability.numpy(), 0.25
        )
        prediction = output["prediction_log"].detach().numpy()
        self.assertTrue(np.all(prediction >= lower - 1e-6))
        self.assertTrue(np.all(prediction <= upper + 1e-6))


class GlobalMissingnessTest(unittest.TestCase):
    def test_overlapping_requests_share_cell_time_state(self) -> None:
        cells = np.asarray(["cell-a", "cell-a"])
        ends = np.asarray([1000, 1024], dtype=np.int64)
        mask = sea.global_corruption_mask(
            cells, ends, mechanism="asynchronous", requested_rate=0.3, seed=19
        )
        first_hours = ends[0] - 335 + np.arange(336)
        second_hours = ends[1] - 335 + np.arange(336)
        common = sorted(set(first_hours.tolist()).intersection(second_hours.tolist()))
        for hour in common:
            left = int(np.flatnonzero(first_hours == hour)[0])
            right = int(np.flatnonzero(second_hours == hour)[0])
            self.assertTrue(np.array_equal(mask[0, left], mask[1, right]))

    def test_statistics_separate_requested_removal_from_total_missingness(self) -> None:
        original = np.ones((1, 336, 4), dtype=np.uint8)
        original[:, :24] = 0
        extra = np.zeros_like(original, dtype=bool)
        extra[:, 24:48] = True
        report = sea.corruption_statistics(original, extra)
        self.assertAlmostEqual(report["original_missing_rate"], 24 / 336)
        self.assertAlmostEqual(report["newly_removed_rate"], 24 / 336)
        self.assertAlmostEqual(report["final_total_missing_rate"], 48 / 336)


    def test_unique_axis_and_window_exposure_rates_are_reported_separately(self) -> None:
        cells = np.asarray(["cell-a", "cell-a"])
        ends = np.asarray([1000, 1024], dtype=np.int64)
        original = np.ones((2, 336, 4), dtype=np.uint8)
        extra = sea.global_corruption_mask(
            cells, ends, mechanism="block", requested_rate=0.5, seed=11
        )
        report = sea.corruption_statistics(
            original,
            extra,
            cells=cells,
            history_end_hours=ends,
            mechanism="block",
            requested_rate=0.5,
            seed=11,
        )
        unique = report["unique_cell_time"]
        exposure = report["window_exposure"]
        self.assertAlmostEqual(unique["selected_for_corruption_rate"], 0.5, places=3)
        self.assertNotAlmostEqual(
            unique["selected_for_corruption_rate"],
            exposure["selected_for_corruption_rate"],
            places=4,
        )
        self.assertEqual(
            report["recent_tail_definition"],
            "tail of each cell's unique evaluated timeline",
        )


class MetricContractTest(unittest.TestCase):
    def test_low_activity_thresholds_are_frozen_from_training_only(self) -> None:
        targets = np.ones((2, 24, 4), dtype=np.float32)
        targets[0] *= 10.0
        targets[1] *= 10_000.0
        masks = np.ones_like(targets, dtype=np.uint8)
        thresholds = sea.frozen_low_activity_thresholds(targets, masks, [0])
        self.assertTrue(np.allclose(thresholds, 10.0))
        score = sea.threshold_hit_score(targets[1:], targets[1:].copy(), thresholds)
        self.assertEqual(score["score"], 1.0)
        self.assertTrue(score["not_auc"])

    def test_bootstrap_recomputes_macro_indicator_ratio_of_sums(self) -> None:
        actual = np.ones((4, 24, 4), dtype=np.float32)
        actual[..., 0] *= 1.0
        actual[..., 1] *= 10.0
        actual[..., 2] *= 100.0
        actual[..., 3] *= 1000.0
        proposed = actual + np.asarray([0.1, 2.0, 30.0, 400.0], dtype=np.float32)
        baseline = actual + np.asarray([0.2, 1.0, 20.0, 100.0], dtype=np.float32)
        cells = np.asarray(["a", "a", "b", "b"])
        scales = np.ones((4, 4), dtype=np.float32)
        expected = (
            sea.forecast_metrics(actual, proposed, scales, cells)["macro_indicator"]["wape"]
            - sea.forecast_metrics(actual, baseline, scales, cells)["macro_indicator"]["wape"]
        )
        result = sea.cell_cluster_bootstrap_wape_delta(
            actual, proposed, baseline, cells, replicates=100, seed=7
        )
        self.assertEqual(result["estimand"], "macro_over_indicator_wape")
        self.assertAlmostEqual(result["delta_proposed_minus_baseline"], expected, places=12)

    def test_core_has_no_finals_or_network_data_access(self) -> None:
        source = Path(sea.__file__).read_text(encoding="utf-8")
        self.assertNotIn("test_data.csv", source)
        self.assertNotIn("requests.", source)
        self.assertNotIn("socket.", source)


if __name__ == "__main__":
    unittest.main()
