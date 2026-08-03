from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_end_to_end_latency as latency


class Revision8LatencyHelpersTest(unittest.TestCase):
    def test_revision8_output_path_is_confined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = latency.resolve_revision8_output(
                root, "artifacts/reproduction/latency/latency.json"
            )
            self.assertEqual(
                destination, root / "artifacts/reproduction/latency/latency.json"
            )
            with self.assertRaises(ValueError):
                latency.resolve_revision8_output(root, "outside/latency.json")

    def test_latency_distribution_reports_required_percentiles(self) -> None:
        result = latency.latency_distribution((1.0, 2.0, 3.0, 4.0))
        self.assertAlmostEqual(result["p50_ms"], 2.5)
        self.assertAlmostEqual(result["p95_ms"], 3.85)
        self.assertAlmostEqual(result["p99_ms"], 3.97)
        self.assertGreater(result["sample_sd_ms"], 0.0)
