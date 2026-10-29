"""Create a reproducible review archive from versioned source and a built site."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Mapping
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "wlcr-sea-anonymous-review/"
EXCLUDED = {".git", ".github", ".venv", ".runtime", "__pycache__", "artifacts", "data", "config", "dist"}
IDENTIFIABLE_URLS = (
    rb"https?://github[.]com/[^/\s\"<>]+/WLCR-SEA_Predictor",
    rb"https?://[^/\s\"<>]+[.]github[.]io/WLCR-SEA_Predictor",
    rb"https?://huggingface[.]co/[^/\s\"<>]+/WLCR-SEA[-_]Predictor",
)


def safe_path(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and not any(
        part in EXCLUDED for part in path.parts
    ) and path.name not in {"SOURCE_REVISION", ".env", "github_token.json"} and not path.name.startswith(".env.")


def collect_files(root: Path, include_site: bool = True) -> dict[str, bytes]:
    if not (root / ".git").exists():
        raise RuntimeError("Rebuild from a maintained checkout; the review archive has no Git metadata")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    result = {}
    for name in names:
        path = root / name
        if safe_path(name) and path.is_file() and not path.is_symlink():
            result[name] = path.read_bytes()
    if include_site:
        if not (root / "site/local-demo/index.html").is_file():
            raise RuntimeError("Build the bilingual website first: mkdocs build --strict")
        for path in (root / "site").rglob("*"):
            name = path.relative_to(root).as_posix()
            if path.is_file() and not path.is_symlink() and safe_path(name):
                result[name] = path.read_bytes()
    required = ["CITATION.cff", "LICENSE", "REVIEW.md", "docs/assets/local-demo/weights.bin"]
    required += [f"demo/checkpoints/A6_mixed_aug_seed{seed}.pt" for seed in range(42, 47)]
    if any(name not in result for name in required):
        raise RuntimeError("The versioned review files or bundled checkpoints are incomplete")
    return result


def validate_files(files: Mapping[str, bytes], denied: tuple[str, ...] = ()) -> None:
    for name, payload in files.items():
        if not safe_path(name):
            raise ValueError(f"Unsafe or private archive path: {name}")
        if any(re.search(pattern, payload, re.I) for pattern in IDENTIFIABLE_URLS):
            raise ValueError(f"Account-linked project URL remains in {name}")
        if any(value.encode("utf-8").lower() in payload.lower() for value in denied if value):
            raise ValueError(f"A supplied identity marker remains in {name}")
    citation = files.get("CITATION.cff", b"")
    if b"name: Anonymous Contributors" not in citation or b"family-names:" in citation:
        raise ValueError("Citation metadata is not anonymous")


def write_archive(files: Mapping[str, bytes], output: Path, denied: tuple[str, ...] = ()) -> dict:
    validate_files(files, denied)
    entries = dict(files)
    manifest = {
        "schema": "anonymous-review/v1",
        "provenance": "File SHA-256 checksums; no Git metadata",
        "files": [{"path": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
                  for name, payload in sorted(entries.items())],
    }
    entries["REVIEW_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    entries["REVIEW_SHA256SUMS"] = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n" for name, payload in sorted(entries.items())
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(entries.items()):
            info = ZipInfo(PREFIX + name, date_time=(2026, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, payload)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    return {"archive": output.name, "files": len(entries), "bytes": output.stat().st_size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/wlcr-sea-anonymous-review.zip")
    parser.add_argument("--deny-file", type=Path, help="Optional local identity markers, one per line; never packaged")
    args = parser.parse_args()
    denied = tuple(args.deny_file.read_text().splitlines()) if args.deny_file else ()
    print(json.dumps(write_archive(collect_files(ROOT), args.output, denied), indent=2))


if __name__ == "__main__":
    main()
