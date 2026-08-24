# Hugging Face Live Demo

[Launch the WLCR-SEA Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }
[View Demo source](https://github.com/rudykon/WLCR-SEA_Predictor/tree/main/demo){ .md-button target="_blank" rel="noopener" }

The public Gradio app shows how WLCR-SEA uses historical patterns and what
happens when part of the history is missing.

## Try it in five minutes

Imagine that the latest measurements for one cell stop arriving, but a planner
still needs tomorrow's demand estimate. The Demo lets you test this directly:

1. load the bundled 336-hour synthetic request and run the **Clean** case;
2. switch to **Recent-tail outage** and move the missingness rate upward;
3. choose one indicator and future hour to see which historical candidates disappear;
4. confirm that unavailable candidates receive exactly zero weight;
5. compare the forecast with the range of available candidates, then download
   the forecast CSV and JSON calculation record.

This test answers a practical question: *what does the method do when part of
the history is missing?* It does not measure the business impact of a real
outage, and it does not run the unavailable trained A6 model.

## Code used by the Demo

The interface uses real repository code rather than a mocked calculation:

The execution path imports and uses:

- `read_traffic` and `split_physical_windows` read and validate the CSV;
- `global_corruption_mask` removes data in repeatable patterns;
- `build_expert_batch` builds the eight historical candidates;
- `WLCRSEA(VARIANTS["A0_fixed"])` combines them with fixed weights;
- `bounded_audit_envelope` checks that the forecast stays inside the allowed range.

The initial weights are previous week 0.7, two weeks earlier 0.2, and 7-day
same-hour median 0.1. If one candidate is unavailable, the remaining weights
are scaled to add up to one. If all three are unavailable, a fallback is used.

<div class="notice-card">
  <strong>This is not the trained A6 paper model.</strong> The repository does not include the A6 checkpoint or training-set prior. The Demo's final fallback is calculated from the current input and marked with an asterisk. Exported JSON records set <code>paper_model: false</code>.
</div>

## Controls

| Control | Purpose |
| --- | --- |
| Request CSV | One cell with 336 hourly rows |
| Missing-data pattern | Complete, random hours, one continuous block, recent hours, or different times by indicator |
| Missing rate | Removes an additional 0% to 80% of values in a repeatable way |
| Indicator | Chooses which traffic measure to inspect |
| Future hour | Chooses one of the 24 predictions to inspect |

## Outputs

- four history and 24-hour forecast panels;
- the minimum and maximum of currently usable candidates;
- candidate values, availability, support, and weights;
- total weight given to unavailable candidates and the range-check result;
- downloadable 24-hour forecast CSV;
- a downloadable versioned JSON calculation record with the input SHA-256.

## Run locally

```bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor
python -m pip install -r requirements.txt
python demo/app.py
```

For sensitive traffic, local or private deployment is the appropriate path.
Do not upload confidential operator data to the public Space.

## Deployment details for maintainers

The Space uses free Gradio `zero-a10g` hardware. The small model calculation is
wrapped in `@spaces.GPU`, and GitHub Actions mirrors the repository after each
update to `main`.

The GitHub README remains ordinary Markdown; Space-specific metadata is added
only during deployment.
