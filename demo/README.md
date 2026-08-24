# WLCR-SEA Request Audit Lab

This Gradio app exercises the repository's real request-local expert builder,
missingness protocol, hard availability mask, and registered `A0_fixed`
parameter-free baseline.

It is intentionally a **method demo**, not a reproduction of the paper's
trained A6 predictions: the public repository does not contain the A6
checkpoint or frozen training prior. The app replaces that last-resort prior
with a request-derived fallback and labels it in every audit record.

Run locally:

```bash
python -m pip install -r requirements.txt
python demo/app.py
```

Generate the bundled deterministic request again:

```bash
python demo/generate_sample.py
```

The Space deployment frontmatter lives in
`demo/space-readme-frontmatter.md`. The GitHub workflow prepends it only in the
deployment staging directory, so the root GitHub README retains its normal
project presentation.
