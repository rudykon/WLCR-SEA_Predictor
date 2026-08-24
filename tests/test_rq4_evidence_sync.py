from __future__ import annotations

import unittest
from decimal import Decimal
from pathlib import Path

from tools import sync_rq4_evidence as sync


class RQ4EvidenceSyncTest(unittest.TestCase):
    def test_fixed_formatting_uses_explicit_rounding_and_sign(self) -> None:
        self.assertEqual(sync.format_fixed(Decimal("0.19625"), 4), "0.1963")
        self.assertEqual(
            sync.format_fixed(Decimal("0.002047273"), 5, signed=True),
            "+0.00205",
        )
        self.assertEqual(
            sync.format_fixed(Decimal("-0.005497273"), 5, signed=True),
            "-0.00550",
        )

    def test_rendered_macros_are_language_independent(self) -> None:
        def metric(value: str) -> dict[str, object]:
            return {"macro_indicator": {"wape": Decimal(value)}}

        def comparison(delta: str, low: str, high: str) -> dict[str, Decimal]:
            return {
                "delta_proposed_minus_baseline": Decimal(delta),
                "ci_low": Decimal(low),
                "ci_high": Decimal(high),
            }

        evidence = {
            "summary": {
                "metrics": {
                    "wlcr_sea": metric("0.196181602"),
                    "dlinear_aug": metric("0.194134329"),
                    "original_wlcr_lightgbm": metric("0.200192759"),
                    "patchtst_aug": metric("0.217487125"),
                    "same_hour_median_7d": metric("0.209600415"),
                    "fixed_seasonal_mixture": metric("0.236759962"),
                    "standard_stat_lightgbm": metric("0.2270358"),
                },
                "paired_cell_cluster_bootstrap": {
                    "wlcr_sea_minus_dlinear_aug": comparison(
                        "0.002047273", "-0.001252113", "0.006138165"
                    ),
                    "wlcr_sea_minus_original_wlcr_lightgbm": comparison(
                        "-0.004011157", "-0.009594427", "0.000603454"
                    ),
                    "wlcr_sea_minus_patchtst_aug": comparison(
                        "-0.021305523", "-0.027818574", "-0.015270063"
                    ),
                },
                "unseen_mean_prior_mass": Decimal("0.015020035"),
                "statistical_unit": {"evaluable_cell_clusters": 727},
            },
            "protocol": {
                "refit_training_protocol": {
                    "model_seed": 42,
                    "batch_size_by_model": {
                        "wlcr_sea": 256,
                        "dlinear_aug": 128,
                    },
                    "augmentation_seed_by_model": {"wlcr_sea": 100042},
                }
            },
            "hashes": {
                "manifest.json": "a" * 64,
                "summary.json": "b" * 64,
                "protocol.json": "c" * 64,
            },
        }
        rendered = sync.render_tex(evidence)
        self.assertIn(r"\newcommand{\RQFourWLCRWAPE}{0.1962}", rendered)
        self.assertIn(r"\newcommand{\RQFourDLinearDelta}{+0.00205}", rendered)
        self.assertIn(r"\newcommand{\RQFourPriorMassPercent}{1.50}", rendered)
        self.assertIn(r"\newcommand{\RQFourClusterCount}{727}", rendered)

    def test_both_manuscripts_share_the_generated_file(self) -> None:
        for manuscript in (Path("paper/main.tex"), Path("paper/main_zh.tex")):
            source = manuscript.read_text(encoding="utf-8")
            self.assertIn(r"\input{rq4_evidence}", source)
            for macro in (
                "RQFourWLCRWAPE",
                "RQFourDLinearWAPE",
                "RQFourDLinearDelta",
                "RQFourOriginalDelta",
                "RQFourPatchDelta",
                "RQFourPriorMassPercent",
            ):
                self.assertIn(f"\\{macro}", source)
            for stale in ("0.1962", "0.1941", "+0.00205", "1.50\\%"):
                self.assertNotIn(stale, source)


if __name__ == "__main__":
    unittest.main()
