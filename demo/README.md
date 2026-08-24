# WLCR-SEA Traffic Forecast Demo

This Gradio app uses the repository's real CSV parser, missing-data logic,
eight candidate forecasts, and fixed `A0_fixed` baseline. It shows how one
14-day input becomes a 24-hour forecast and how the result changes when data is
removed.

It is a **workflow Demo**, not a reproduction of the paper's trained A6
predictions: the public repository does not contain the A6 checkpoint or fixed
training prior. The app replaces the last fallback with a value calculated from
the current input and labels this clearly in every exported record.

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
