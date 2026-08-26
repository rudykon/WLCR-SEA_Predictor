from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = (
    "https://res-static.hc-cdn.cn/cloudbu-site/china/zh-cn/"
    "wuxian-gaoxiao2026/1780886490950118786.zip"
)
MODEL_REVISION = "eb4447f4ebab8f9caa003d92c838ed8e750963bd"


class PublicMetadataTest(unittest.TestCase):
    def _public_text(self) -> str:
        paths = [
            ROOT / "README.md",
            ROOT / "README_CN.md",
            ROOT / "REPRODUCTION.md",
            ROOT / "MODEL_CARD.md",
        ]
        paths.extend((ROOT / "docs").rglob("*.md"))
        paths.extend(
            [
                ROOT / "demo" / "README.md",
                ROOT / "demo" / "space-readme-frontmatter.md",
                ROOT / "demo" / "app.py",
                ROOT / "demo" / "runtime.py",
            ]
        )
        return "\n".join(path.read_text(encoding="utf-8") for path in paths)

    def test_obsolete_a6_unavailability_claims_are_absent(self) -> None:
        text = self._public_text().lower()
        banned = (
            "trained a6 checkpoint is not public",
            "does not include the trained a6",
            "does not contain the a6 checkpoint",
            "trained a6 checkpoint is not distributed",
            "demo ≠ a6",
            "space runs `a0_fixed`",
            "demo uses `a0_fixed`",
            "a6 检查点尚未公开",
            "仓库未包含论文训练后的 a6",
        )
        for phrase in banned:
            self.assertNotIn(phrase, text)

    def test_reader_facing_surfaces_do_not_use_the_paper_experiment_id(self) -> None:
        paths = (
            "README.md",
            "README_CN.md",
            "docs/index.md",
            "docs/index.zh.md",
            "docs/guide/method.md",
            "docs/guide/method.zh.md",
            "docs/research/evidence.md",
            "docs/research/evidence.zh.md",
            "docs/deployment/hugging-face.md",
            "docs/deployment/hugging-face.zh.md",
            "demo/README.md",
            "demo/space-readme-frontmatter.md",
            "mkdocs.yml",
        )
        for relative in paths:
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotRegex(
                text,
                r"(?i)\ba6(?:\b|_)",
                msg=f"Paper experiment ID leaked into reader-facing copy: {relative}",
            )

        space_card = (ROOT / "demo" / "space-readme-frontmatter.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("title: WLCR-SEA Cellular Traffic Forecast Demo", space_card)
        self.assertIn("Live Demo", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn("体验 Demo", (ROOT / "docs/index.zh.md").read_text(encoding="utf-8"))

    def test_public_metrics_name_their_aggregation(self) -> None:
        for relative in (
            "README.md",
            "README_CN.md",
            "docs/index.md",
            "docs/index.zh.md",
            "docs/research/evidence.md",
            "docs/research/evidence.zh.md",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            for line in text.splitlines():
                if any(value in line for value in ("0.1955", "0.1854", "0.2196", "0.2460", "0.2172", "0.1967")):
                    self.assertTrue(
                        "macro-indicator" in line.lower()
                        or "宏指标" in line
                        or "四指标宏平均" in line
                        or line.lstrip().startswith("|")
                        and "WAPE" not in line,
                        msg=f"Unqualified WAPE line in {relative}: {line}",
                    )

    def test_information_architecture_is_compact(self) -> None:
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        nav = config.split("\nnav:\n", 1)[1]
        self.assertEqual(sum(1 for line in nav.splitlines() if line.startswith("  - ")), 5)
        self.assertNotIn("fonts.googleapis.com", config)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("## Repository Layout", readme)
        self.assertNotIn("## Figures", readme)
        self.assertEqual(readme.count("paper_figure_"), 2)

    def test_reproduction_guide_has_no_removed_runtime_or_manuscript_paths(self) -> None:
        text = (ROOT / "REPRODUCTION.md").read_text(encoding="utf-8")
        self.assertNotIn(".runtime/", text)
        self.assertNotIn("paper/main.tex", text)
        self.assertNotIn("paper/main_zh.tex", text)

    def test_space_frontmatter_uses_supported_cpu_metadata(self) -> None:
        text = (ROOT / "demo" / "space-readme-frontmatter.md").read_text(
            encoding="utf-8"
        )
        frontmatter = text.split("---", 2)[1]
        metadata = {
            key.strip(): value.strip().strip('"')
            for line in frontmatter.splitlines()
            if ":" in line
            for key, value in (line.split(":", 1),)
        }
        valid_colors = {
            "red",
            "yellow",
            "green",
            "blue",
            "indigo",
            "purple",
            "pink",
            "gray",
        }
        self.assertIn(metadata["colorFrom"], valid_colors)
        self.assertIn(metadata["colorTo"], valid_colors)
        self.assertEqual(metadata["sdk"], "gradio")
        self.assertEqual(metadata["app_file"], "demo/app.py")
        self.assertEqual(metadata["suggested_hardware"], "cpu-basic")

        requirements = (ROOT / "requirements-demo.txt").read_text(encoding="utf-8")
        self.assertIn("torch==2.8.0", requirements.splitlines())
        self.assertNotIn("torch==2.8.0+cpu", requirements)

        app = (ROOT / "demo" / "app.py").read_text(encoding="utf-8")
        self.assertEqual(app.count("@spaces.GPU"), 1)
        self.assertIn("def _host_hardware_compatibility_marker", app)
        self.assertNotIn("@spaces.GPU\ndef _run_request", app)
        normalized = " ".join(text.split())
        self.assertIn("legacy ZeroGPU host configuration", normalized)
        self.assertIn("unused startup", normalized)

    def test_research_dataset_entry_is_public_complete_and_regression_guarded(self) -> None:
        paths = (
            "README.md",
            "README_CN.md",
            "REPRODUCTION.md",
            "docs/reference/reproduction.md",
            "docs/reference/reproduction.zh.md",
        )
        train_hash = (
            "d274407a3db51ba4871851ab447bcc75202bb567337464d85ea280662f3bf1da"
        )
        for relative in paths:
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(DATASET_URL, text, msg=f"Dataset link missing: {relative}")
            self.assertIn("data/train_data.csv", text)
            self.assertIn(train_hash, text)

        reproduction = (ROOT / "REPRODUCTION.md").read_text(encoding="utf-8")
        self.assertIn(
            "17d87ae40a9ddfd263ea60cba7f2a4ff05037b92cebdd37f9bb89a6c9e3094bf",
            reproduction,
        )
        self.assertIn("线上阶段数据集/AI数据集/train_data.csv", reproduction)
        self.assertIn("no separate data-license file", reproduction)
        self.assertIn("does not redistribute", reproduction)

    def test_routing_view_states_its_ensemble_summary_boundary(self) -> None:
        app = (ROOT / "demo" / "app.py").read_text(encoding="utf-8")
        runtime = (ROOT / "demo" / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("Why this forecast?", app)
        self.assertNotIn("Why this forecast?", runtime)
        self.assertIn("Ensemble routing summary", app)
        self.assertIn("does not exactly", app)
        self.assertIn("集成路由摘要", app)
        self.assertIn("不能直接重构", app)

    def test_audit_export_carries_replay_inputs_and_member_checks(self) -> None:
        runtime = (ROOT / "demo" / "runtime.py").read_text(encoding="utf-8")
        for value in (
            'AUDIT_SCHEMA = "wlcr-sea-audit/v3"',
            '"repository": SOURCE_REPOSITORY',
            '"commit": _source_commit()',
            '"runtime_version": RUNTIME_VERSION',
            '"python_version": platform.python_version()',
            '"torch_version": _package_version("torch")',
            '"seed": DEMO_SEED',
            'removed_positions = _removed_positions(result)',
            '"removed_fraction_of_original_observations"',
            '"effective_mask": result.effective_mask.astype(int).tolist()',
            '"checks": _member_checks(member, result.availability)',
        ):
            self.assertIn(value, runtime)

    def test_prb_terms_describe_counts_not_utilization(self) -> None:
        card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
        self.assertIn("average used downlink PRBs", card)
        self.assertIn("average used uplink PRBs", card)
        self.assertNotIn("downlink PRB utilization", card)
        self.assertNotIn("uplink PRB utilization", card)
        self.assertIn("not\nutilization percentages", card)

    def test_citation_and_links_pin_the_public_model_revision(self) -> None:
        for relative in (
            "README.md",
            "README_CN.md",
            "REPRODUCTION.md",
            "MODEL_CARD.md",
            "CITATION.cff",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(MODEL_REVISION, text, msg=f"Revision not pinned: {relative}")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("family-names: Kong", citation)
        self.assertIn("repository-code:", citation)

    def test_demo_resources_are_bounded_and_figures_avoid_pyplot_state(self) -> None:
        runtime = (ROOT / "demo" / "runtime.py").read_text(encoding="utf-8")
        app = (ROOT / "demo" / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("matplotlib.pyplot", runtime)
        self.assertIn("from matplotlib.figure import Figure", runtime)
        self.assertIn("EXPORT_TTL_SECONDS", runtime)
        self.assertIn("EXPORT_DIRECTORY_LIMIT", runtime)
        self.assertIn("create_exports=False", app)
        self.assertIn('value=0.0,', app)
        self.assertIn("interactive=False", app)
        self.assertIn("Reset to sample", app)
        self.assertIn("恢复内置样例", app)

    def test_space_sync_records_source_and_runs_post_deploy_sample(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "hugging-face-space.yml"
        ).read_text(encoding="utf-8")
        self.assertIn('> "${HF_SPACE_STAGE}/SOURCE_REVISION"', workflow)
        self.assertIn("tools/check_space_deployment.py", workflow)
        self.assertIn("--expected-commit", workflow)
        self.assertIn("Synchronize the public model card", workflow)


if __name__ == "__main__":
    unittest.main()
