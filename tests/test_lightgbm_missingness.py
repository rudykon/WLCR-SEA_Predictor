from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from experiments import evaluate_lightgbm_missingness as worker


class Revision9OriginalWlcrWorkerTest(unittest.TestCase):
    def test_revision8_scenario_inventory_is_17_conditions(self) -> None:
        rows = worker.scenario_specs(smoke=False)
        self.assertEqual(len(rows), 17)
        self.assertEqual(rows[0], ("mcar", 0.0))
        self.assertEqual(sum(rate == 0.0 for _, rate in rows), 1)
        self.assertEqual(
            set(rows[1:]),
            {
                (mechanism, rate)
                for mechanism in worker.ROBUSTNESS_MECHANISMS
                for rate in worker.ROBUSTNESS_RATES[1:]
            },
        )

    def test_smoke_inventory_retains_clean_and_one_structured_case(self) -> None:
        self.assertEqual(worker.scenario_specs(smoke=True), (("mcar", 0.0), ("block", 0.20)))

    def test_scenario_names_follow_revision8_display_semantics(self) -> None:
        self.assertEqual(worker.scenario_name("mcar", 0.0), "clean")
        self.assertEqual(worker.scenario_name("block", 0.2), "block_0.20")
        self.assertEqual(worker.scenario_name("recent_tail", 0.5), "timeline_tail_0.50")
        self.assertEqual(worker._mechanism_display("recent_tail"), "timeline_tail")

    def test_identity_digest_is_stable_and_order_sensitive(self) -> None:
        first = {
            "holdout_position": 0,
            "dataset_index": 9,
            "cell": "1018",
            "target_start": "2024-08-12 00:00:00",
            "history_end": "2024-08-11 23:00:00",
            "target_start_hour": 478728,
            "history_end_hour": 478727,
        }
        second = {**first, "holdout_position": 1, "dataset_index": 10}
        self.assertEqual(worker._identity_digest((first, second)), worker._identity_digest((first, second)))
        self.assertNotEqual(worker._identity_digest((first, second)), worker._identity_digest((second, first)))

    def test_persisted_predictions_are_aligned_by_request_identity(self) -> None:
        class IndexablePrediction:
            def __init__(self, values: tuple[str, ...]) -> None:
                self.values = values

            def __len__(self) -> int:
                return len(self.values)

            def __getitem__(self, positions: tuple[int, ...]) -> tuple[str, ...]:
                return tuple(self.values[int(position)] for position in positions)

        # Keep this ordinary unit test free of the isolated LightGBM runtime.
        # The helper's only needed NumPy operations here are index coercion and
        # contiguous-array wrapping, which this narrow stand-in models.
        numpy_standin = SimpleNamespace(
            int64=object(),
            float32=object(),
            asarray=lambda values, dtype: tuple(values),
            ascontiguousarray=lambda values, dtype: values,
        )
        persisted = IndexablePrediction(("first", "second"))
        identity_rows = (
            {"cell": "cell_b", "target_start": "2024-08-13 00:00:00"},
            {"cell": "cell_a", "target_start": "2024-08-12 00:00:00"},
        )
        bundle = SimpleNamespace(identity_rows=identity_rows)
        with TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "holdout_order.csv").write_text(
                "window_index,cell,target_start\n"
                "0,cell_a,2024-08-12 00:00:00\n"
                "1,cell_b,2024-08-13 00:00:00\n",
                encoding="utf-8",
            )
            with patch.dict(sys.modules, {"numpy": numpy_standin}):
                aligned = worker.align_reference_prediction(persisted, source, bundle)

        self.assertEqual(aligned, ("second", "first"))

    def test_source_contract_contains_no_final_or_preliminary_traffic_path(self) -> None:
        source = worker.__file__
        self.assertIsNotNone(source)
        text = Path(source).read_text(encoding="utf-8")
        self.assertNotIn("data/test_data.csv", text)
        self.assertNotIn("data/reference/preliminary/test_data.csv", text)
        self.assertIn("data/train_data.csv", text)


if __name__ == "__main__":
    unittest.main()
