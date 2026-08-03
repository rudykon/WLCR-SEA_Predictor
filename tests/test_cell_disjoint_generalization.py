from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments import train_neural_baselines as neural
from experiments import evaluate_cell_disjoint_generalization as unseen


class Revision7UnseenHelpersTest(unittest.TestCase):
    def test_fold_mapping_is_order_independent_and_balanced(self) -> None:
        cells = [f"cell-{index:02d}" for index in range(13)]
        first = unseen.fold_mapping(cells)
        second = unseen.fold_mapping(list(reversed(cells)))
        self.assertEqual(first, second)
        counts = [list(first.values()).count(fold) for fold in range(unseen.FOLDS)]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_align_examples_requires_exact_cell_time_keys(self) -> None:
        start = datetime(2024, 8, 3)
        hour = int((start - datetime(1970, 1, 1)).total_seconds() // 3600)
        dataset = neural.CachedDataset(
            root=Path("<memory>"),
            x_values=np.zeros((2, 336, 4), dtype=np.float32),
            x_masks=np.ones((2, 336, 4), dtype=np.uint8),
            targets=np.ones((2, 24, 4), dtype=np.float32),
            target_masks=np.ones((2, 24, 4), dtype=np.uint8),
            mase_scales=np.ones((2, 4), dtype=np.float32),
            cells=np.asarray(["cell-b", "cell-a"]),
            target_start_hours=np.asarray([hour + 24, hour], dtype=np.int64),
            history_end_hours=np.asarray([hour + 23, hour - 1], dtype=np.int64),
        )
        examples = [
            SimpleNamespace(window=SimpleNamespace(cell="cell-a", target_start=start)),
            SimpleNamespace(
                window=SimpleNamespace(
                    cell="cell-b",
                    target_start=neural.timestamp_from_hour(hour + 24),
                )
            ),
        ]
        aligned = unseen.align_examples(
            dataset, np.asarray([0, 1], dtype=np.int64), examples, "toy"
        )
        self.assertTrue(np.array_equal(aligned, np.asarray([1, 0])))
        with self.assertRaises(ValueError):
            unseen.align_examples(
                dataset,
                np.asarray([0, 1], dtype=np.int64),
                examples[:1],
                "toy",
            )


    def test_lgbm_smoke_caps_match_neural_worker(self) -> None:
        source = Path(
            "experiments/evaluate_cell_disjoint_lightgbm_worker.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "np.flatnonzero(train_window_mask)[:256]", source
        )
        self.assertIn(
            "np.flatnonzero(evaluation_window_mask)[:128]", source
        )


if __name__ == "__main__":
    unittest.main()
