from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicMetadataTest(unittest.TestCase):
    def _public_text(self) -> str:
        paths = [ROOT / "README.md", ROOT / "README_CN.md", ROOT / "REPRODUCTION.md"]
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


if __name__ == "__main__":
    unittest.main()
