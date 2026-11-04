# Anonymous review package

Author names, personal contact details, account links, and Git provenance are
withheld in this review release. The five released model files are included in
`demo/checkpoints/`, with their original SHA-256 checksums in `SHA256SUMS`.
No operator dataset is included.

## Run the browser Demo

The distribution archive includes the built bilingual website in `site/`.
From the extracted package root, run:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory site
```

Open `http://localhost:8000/local-demo/` or
`http://localhost:8000/zh/local-demo/`. Inference runs on the visitor's CPU.
Model files are served from the package; request CSV contents stay in the browser.
Use HTTP on localhost rather than opening an HTML file directly.

When working from source rather than the distribution archive, build `site/`
with `python -m pip install -r requirements-docs.txt` and `mkdocs build --strict`.

## Run the optional Python interface

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python demo/app.py
```

The Python interface verifies the bundled checkpoints and runs them locally.
No model-hosting account or model download is required. The source identifier in
its audit JSON is a SHA-256 of the implementation files, rather than a Git commit.

## Verify the distribution

`REVIEW_MANIFEST.json` lists the paths and SHA-256 values in the archive.
`REVIEW_SHA256SUMS` can be checked from the extracted package root with:

```bash
sha256sum -c REVIEW_SHA256SUMS
node --test tests/test_browser_local_demo.mjs
python -m unittest tests.test_public_metadata tests.test_hf_space_demo -q
```

For the complete input contract, research environment, and reproduction commands,
see `REPRODUCTION.md`. The original Apache-2.0 license is preserved in `LICENSE`.

To rebuild a distribution from a maintained source checkout:

```bash
mkdocs build --strict
python tools/build_review_package.py --output dist/wlcr-sea-anonymous-review.zip
```

The package builder uses versioned files and the built website. It excludes
Git metadata, hosting workflows, datasets, experiment outputs, local settings,
and credentials. A public repository's owner and previously published history
remain outside the package; distribute the ZIP as an anonymous supplementary
artifact instead of linking to an identifiable account.
