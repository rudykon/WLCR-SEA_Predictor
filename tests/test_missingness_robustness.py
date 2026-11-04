from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments import analyze_missingness_robustness as revision9


class Revision9MissingnessAnalysisTest(unittest.TestCase):
    def test_full_protocol_is_exactly_revision8_17_scenarios(self) -> None:
        scenarios = revision9.full_scenarios()
        self.assertEqual(len(scenarios), 17)
        self.assertEqual(
            [(item.mechanism, item.requested_rate, item.scenario) for item in scenarios[:2]],
            [("mcar", 0.0, "clean"), ("mcar", 0.1, "mcar_0.10")],
        )
        self.assertEqual(
            sum(item.requested_rate == 0.0 for item in scenarios),
            1,
        )
        self.assertEqual(
            revision9.scenario_name("recent_tail", 0.5), "timeline_tail_0.50"
        )
        self.assertEqual(
            len({item.key for item in scenarios}),
            17,
        )

    def test_mask_hash_uses_canonical_c_order_uint8_bytes(self) -> None:
        mask = np.asarray([[[True, False], [False, True]]], dtype=bool)
        expected = hashlib.sha256(
            np.ascontiguousarray(mask, dtype=np.uint8).tobytes(order="C")
        ).hexdigest()
        self.assertEqual(revision9.hash_mask(mask), expected)
        self.assertEqual(revision9.hash_mask(mask.astype(np.uint8)), expected)

    def test_identity_alignment_rejects_reordered_request(self) -> None:
        expected = (
            {
                "holdout_position": "0",
                "dataset_index": "10",
                "cell": "1001",
                "target_start": "2024-08-12 00:00:00",
                "history_end": "2024-08-11 23:00:00",
                "target_start_hour": "477000",
                "history_end_hour": "476999",
            },
        )
        revision9.validate_identity_rows(expected, expected)
        wrong = ({**expected[0], "cell": "9999"},)
        with self.assertRaisesRegex(ValueError, "request identity mismatch"):
            revision9.validate_identity_rows(wrong, expected)

    def test_clean_reference_is_aligned_by_persisted_request_identity(self) -> None:
        expected = (
            {"cell": "cell_b", "target_start": "2024-08-13 00:00:00"},
            {"cell": "cell_a", "target_start": "2024-08-12 00:00:00"},
        )
        reference = np.asarray([[10.0], [20.0]], dtype=np.float32)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "holdout_order.csv").write_text(
                "window_index,cell,target_start\n"
                "0,cell_a,2024-08-12 00:00:00\n"
                "1,cell_b,2024-08-13 00:00:00\n",
                encoding="utf-8",
            )
            aligned = revision9.align_original_clean_reference(root, reference, expected)

        np.testing.assert_array_equal(
            aligned, np.asarray([[20.0], [10.0]], dtype=np.float32)
        )

    def test_gpu_parser_and_output_rate_validation(self) -> None:
        self.assertEqual(revision9.parse_gpu_devices("0, 1,2,3"), (0, 1, 2, 3))
        with self.assertRaisesRegex(ValueError, "at least three"):
            revision9.parse_gpu_devices("0,1")
        with self.assertRaisesRegex(ValueError, "distinct"):
            revision9.parse_gpu_devices("0,1,1")
        with self.assertRaisesRegex(ValueError, "clean scenario"):
            revision9.scenario_name("block", 0.0)

    def test_smoke_summary_contract_uses_its_explicit_subset(self) -> None:
        scenarios = revision9.smoke_scenarios()
        summary = {
            "registered_train_file": "data/train_data.csv",
            "registered_train_sha256_before": "abc",
            "registered_train_sha256_after": "abc",
            "finals_test_opened": False,
            "corruption_seeds": [142],
            "scenario_count": 2,
            "seed_scenario_count": 2,
            "scenarios": [
                {
                    "mechanism": item.mechanism,
                    "requested_rate": item.requested_rate,
                    "scenario": item.scenario,
                }
                for item in scenarios
            ],
        }
        revision9.validate_original_summary(
            Path("."),
            summary,
            train_hash="abc",
            expected_scenarios=scenarios,
            expected_corruption_seeds=(142,),
        )
        summary["corruption_seeds"] = [143]
        with self.assertRaisesRegex(ValueError, "corruption seeds"):
            revision9.validate_original_summary(
                Path("."),
                summary,
                train_hash="abc",
                expected_scenarios=scenarios,
                expected_corruption_seeds=(142,),
            )


if __name__ == "__main__":
    unittest.main()
