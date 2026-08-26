# WLCR-SEA Forecast Demo

The Demo runs a five-model ensemble from the public WLCR-SEA checkpoints on
CPU. At startup it downloads the pinned Hugging Face revision, verifies every
SHA-256, loads each frozen training prior and model once, and averages member
forecasts in linear traffic space.

```bash
python -m pip install -r requirements-demo.txt
python demo/app.py
```

The built-in synthetic request runs automatically. Uploads must contain one
cell, 336 contiguous hourly rows, and the four documented traffic indicators.
The automatic preview does not create temporary downloads; click **Run
forecast** to enable the forecast CSV and audit JSON. Do not upload
confidential operator traffic to the public Space.

The **Ensemble routing summary** shows expert values and routing weights
averaged separately across five members. It is useful for reviewing routing
tendencies but does not exactly decompose the final ensemble prediction.

The forecast CSV contains the full `24 × 4` output. The audit JSON records the
source commit and environment, input hash, missingness seed and effective mask,
model revision, checkpoint hashes, and per-member forecasts, routing
components, envelopes, and violation counts. Audit schema v4 separates the
configured removal rate from the fraction of original observations actually
removed and the final observed fraction. Replay requires the original request
matching the recorded input hash and the pinned source revision.
