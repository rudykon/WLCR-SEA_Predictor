# Live Demo

[Open Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }
[Source](https://github.com/rudykon/WLCR-SEA_Predictor/tree/main/demo){ .md-button target="_blank" rel="noopener" }

The Gradio app shows how missing history changes candidates, weights, and forecasts.

## Try it

1. Load the 336-hour sample and run **Clean**.
2. Select **Recent-tail outage** and raise the missing rate.
3. Pick an indicator and future hour.
4. Inspect candidate availability and weight.
5. Download CSV and JSON.

This tests the calculation, not outage impact. It does not run trained A6.

## Code path

- `read_traffic` + `split_physical_windows`: parse CSV.
- `global_corruption_mask`: remove data.
- `build_expert_batch`: build eight candidates.
- `WLCRSEA(VARIANTS["A0_fixed"])`: combine them.
- `bounded_audit_envelope`: check the range.

Initial weights are 0.7 (previous week), 0.2 (two weeks), and 0.1 (7-day median). Available weights are rescaled; all-missing cases use a fallback.

<div class="notice-card">
  <strong>Demo ≠ A6.</strong> The fallback comes from the current input. JSON sets <code>paper_model: false</code>.
</div>

## Controls

| Control | Purpose |
| --- | --- |
| CSV | One cell, 336 hourly rows |
| Pattern | Complete, random, block, recent tail, or asynchronous |
| Missing rate | 0–80%, repeatable |
| Indicator | Traffic measure |
| Future hour | One of 24 predictions |

## Outputs

- four history/forecast panels;
- candidate range, values, availability, support, and weights;
- unavailable weight and range check;
- forecast CSV and versioned JSON.

## Run locally

```bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor
python -m pip install -r requirements.txt
python demo/app.py
```

For sensitive data, deploy locally or privately.

## Deployment

The Space uses free `zero-a10g`. `@spaces.GPU` wraps the calculation, and GitHub Actions syncs every `main` update.
