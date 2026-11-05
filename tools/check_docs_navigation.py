"""Verify that built language links stay on the corresponding project page."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urljoin, urlsplit


class LanguageLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a" and "md-select__link" in (values.get("class") or "").split():
            self.links[values.get("hreflang") or ""] = values.get("href") or ""


def check_navigation(site_dir: Path, site_url: str) -> dict[str, int]:
    base = site_url.rstrip("/") + "/"
    if urlsplit(base).scheme not in {"http", "https"} or not urlsplit(base).netloc:
        raise ValueError("Pass the full deployment URL, including its project directory")
    pages = sorted(site_dir.rglob("index.html"))
    if not pages:
        raise ValueError(f"No built pages found in {site_dir}")
    checked = 0
    for page in pages:
        relative = page.relative_to(site_dir).as_posix().removesuffix("index.html")
        logical = relative.removeprefix("zh/")
        parser = LanguageLinks()
        parser.feed(page.read_text(encoding="utf-8"))
        if set(parser.links) != {"en", "zh"}:
            raise ValueError(f"Missing English/Chinese language links in {relative or 'index.html'}")
        for locale, href in parser.links.items():
            target = ("zh/" if locale == "zh" else "") + logical
            expected = urljoin(base, target)
            actual = urljoin(urljoin(base, relative), href)
            if actual != expected:
                raise ValueError(f"{relative or 'index.html'} [{locale}]: {actual} != {expected}")
            if not (site_dir / target / "index.html").is_file():
                raise ValueError(f"Missing translated destination: {target}")
            checked += 1
    return {"pages": len(pages), "language_links": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--site-url", required=True)
    args = parser.parse_args()
    print(json.dumps(check_navigation(args.site_dir, args.site_url)))


if __name__ == "__main__":
    main()
