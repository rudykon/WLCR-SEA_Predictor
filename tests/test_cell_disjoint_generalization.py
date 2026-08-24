from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments import train_neural_baselines as neural
from experiments import evaluate_cell_disjoint_generalization as unseen
from experiments import train_wlcr_sea as runner


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

    def test_refit_defaults_match_temporal_protocols(self) -> None:
        args = unseen.parser().parse_args([])
        self.assertEqual(args.wlcr_batch_size, runner.DEFAULT_BATCH_SIZE)
        self.assertEqual(args.neural_batch_size, neural.DEFAULT_BATCH_SIZE)
        protocol = unseen.refit_training_protocol(
            args.wlcr_batch_size, args.neural_batch_size
        )
        self.assertTrue(protocol["matches_temporal_batch_size_defaults"])
        self.assertTrue(protocol["matched_final_refit_augmentation_view"])
        self.assertEqual(
            protocol["batch_size_by_model"],
            {"wlcr_sea": 256, "dlinear_aug": 128, "patchtst_aug": 128},
        )
        self.assertEqual(
            set(protocol["augmentation_seed_by_model"].values()),
            {42 + neural.FINAL_AUGMENTATION_SEED_OFFSET},
        )

    def test_refit_protocol_rejects_nonpositive_batches(self) -> None:
        with self.assertRaises(ValueError):
            unseen.refit_training_protocol(0, neural.DEFAULT_BATCH_SIZE)
        with self.assertRaises(ValueError):
            unseen.refit_training_protocol(runner.DEFAULT_BATCH_SIZE, -1)

    def test_worker_report_validation_checks_actual_training_settings(self) -> None:
        protocol = unseen.refit_training_protocol(256, 128)
        shared_augmentation = {
            "mechanism": "mixed",
            "requested_rate": 0.15,
            "seed": 100042,
            "scope": "absolute_cell_timeline",
            "newly_removed_rate": 0.1,
            "final_total_missing_rate": 0.2,
            "unique_cell_time": {"positions": 10},
            "window_exposure": {"positions": 20},
        }
        report = {
            "model_seed": 42,
            "refit_training_protocol": protocol,
            "training_reports": {
                method: {
                    "batch_size": protocol["batch_size_by_model"][method],
                    "augmentation_seed": 100042,
                    "augmentation": dict(shared_augmentation),
                }
                for method in ("wlcr_sea", "dlinear_aug", "patchtst_aug")
            },
        }
        unseen.validate_worker_refit_report(report, protocol)
        report["training_reports"]["patchtst_aug"]["augmentation"]["seed"] = 7
        with self.assertRaises(ValueError):
            unseen.validate_worker_refit_report(report, protocol)

    def test_context_files_are_not_opened_by_cell_disjoint_cache(self) -> None:
        source = Path(
            "experiments/evaluate_cell_disjoint_generalization.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("load_parameters", source)
        self.assertNotIn("load_weather", source)
        self.assertIn("build_matrix(final_examples, baseline, {}, {})", source)
        self.assertIn("build_matrix(holdout_examples, baseline, {}, {})", source)

    def test_master_output_must_be_new_or_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-evidence"
            unseen.prepare_fresh_output(output)
            self.assertTrue(output.is_dir())
            (output / "stale.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                unseen.prepare_fresh_output(output)


if __name__ == "__main__":
    unittest.main()
