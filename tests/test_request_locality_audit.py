from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments import audit_request_locality as audit


@dataclass(frozen=True)
class TinyDataset:
    x_values: np.ndarray
    x_masks: np.ndarray
    cells: np.ndarray
    target_start_hours: np.ndarray


class Revision8RequestLocalAuditTest(unittest.TestCase):
    def test_stratified_targets_are_deterministic_and_cover_date_missingness_cells(self) -> None:
        records_per_stratum = 2
        dates = ("2024-08-12", "2024-08-13")
        masks: list[np.ndarray] = []
        cells: list[str] = []
        target_hours: list[int] = []
        start = int((np.datetime64("2024-08-12") - np.datetime64("1970-01-01")) / np.timedelta64(1, "h"))
        missing_counts = (0, 1, 2, 4)  # none, low, moderate, high over ten inputs
        for day_offset, _ in enumerate(dates):
            for bin_index, missing in enumerate(missing_counts):
                for repeat in range(records_per_stratum):
                    row = np.ones((10, 1), dtype=np.uint8)
                    row[:missing] = 0
                    masks.append(row)
                    cells.append(f"cell-{day_offset}-{bin_index}-{repeat}")
                    target_hours.append(start + day_offset * 24 + repeat)
        count = len(masks)
        dataset = TinyDataset(
            x_values=np.arange(count * 10, dtype=np.float32).reshape(count, 10, 1),
            x_masks=np.stack(masks),
            cells=np.asarray(cells),
            target_start_hours=np.asarray(target_hours, dtype=np.int64),
        )
        population = audit.build_audit_requests(dataset, np.arange(count, dtype=np.int64))
        first, first_quotas = audit.select_stratified_requests(population, 8, 7042)
        second, second_quotas = audit.select_stratified_requests(population, 8, 7042)
        self.assertEqual(first, second)
        self.assertEqual(first_quotas, second_quotas)
        self.assertEqual(len(first), 8)
        self.assertEqual({record.target_date for record in first}, set(dates))
        self.assertEqual(
            {record.missingness_bin for record in first},
            {
                "none_0pct",
                "low_0_to_10pct",
                "moderate_10_to_25pct",
                "high_above_25pct",
            },
        )
        self.assertEqual(len({record.cell for record in first}), 8)
        payload = audit.target_list_payload(first, first_quotas, population, audit_seed=7042)
        self.assertEqual(payload["target_list_sha256"], audit.canonical_sha256(payload["targets"]))

    def test_global_complement_mutation_preserves_target_expert_and_model_tensors(self) -> None:
        values = np.arange(3 * 4, dtype=np.float32).reshape(3, 2, 2)
        masks = np.ones_like(values, dtype=np.uint8)
        masks[1, 0, 0] = 0
        changed_values, changed_masks, report = audit.build_global_perturbation(values, masks)
        self.assertTrue(np.all(changed_values != values))
        self.assertTrue(np.all(changed_masks != masks))
        self.assertEqual(report["globally_changed_value_elements"], values.size)
        altered_values = changed_values.copy()
        altered_masks = changed_masks.copy()
        target = 0
        altered_values[target] = values[target]
        altered_masks[target] = masks[target]
        original = TinyDataset(
            x_values=values,
            x_masks=masks,
            cells=np.asarray(["same", "same", "other"]),
            target_start_hours=np.asarray([0, 24, 48], dtype=np.int64),
        )
        altered = TinyDataset(
            x_values=altered_values,
            x_masks=altered_masks,
            cells=original.cells,
            target_start_hours=original.target_start_hours,
        )

        def fake_builder(dataset: TinyDataset, indices: np.ndarray, prior: np.ndarray):
            selected_values = np.asarray(dataset.x_values[indices], dtype=np.float32)
            selected_masks = np.asarray(dataset.x_masks[indices], dtype=np.uint8)
            batch = SimpleNamespace(
                values=selected_values,
                availability=selected_masks,
                reliability=selected_masks.astype(np.float32),
                context=np.full((len(indices), 1, 1), prior.item(), dtype=np.float32),
            )
            tensors = (
                torch.from_numpy(batch.values),
                torch.from_numpy(batch.availability),
                torch.from_numpy(batch.reliability),
                torch.from_numpy(batch.context),
            )
            return batch, tensors

        class FakeModel(torch.nn.Module):
            def forward(self, values, availability, reliability, context):
                result = values + availability.to(values.dtype) + reliability + context[:, :1, :1]
                return {"prediction_log": result, "entropy": result.square()}

        self.assertFalse(np.array_equal(altered.x_values[1], original.x_values[1]))
        self.assertFalse(np.array_equal(altered.x_masks[1], original.x_masks[1]))
        check = audit.compare_target_request(
            FakeModel(), original, altered, target, np.asarray(1.0, dtype=np.float32), tensor_builder=fake_builder
        )
        self.assertTrue(check["bitwise_request_local_invariance_pass"])
        self.assertTrue(all(check["expert_tensors_bitwise_identical"].values()))
        self.assertTrue(all(check["model_output_tensors_bitwise_identical"].values()))


if __name__ == "__main__":
    unittest.main()
