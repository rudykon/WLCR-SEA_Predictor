# Hugging Face Live Demo

[Launch the WLCR-SEA Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }
[View Demo source](https://github.com/rudykon/WLCR-SEA_Predictor/tree/main/demo){ .md-button target="_blank" rel="noopener" }

The public Gradio app demonstrates how WLCR-SEA uses historical patterns and how its behavior changes when part of the history is missing.

## Try it in five minutes

Suppose the latest measurements from one cell stop arriving, but a planner still needs an estimate of tomorrow's demand. The Demo lets you explore this situation directly:

1. load the bundled 336-hour synthetic request and run the **Clean** case;
2. switch to **Recent-tail outage** and gradually increase the missing-data rate;
3. choose one indicator and future hour to see which historical candidates disappear;
4. confirm that unavailable candidates receive exactly zero weight;
5. compare the forecast with the range of available candidates, then download the forecast CSV and JSON calculation record.

This exercise answers a practical question: *what does the method do when part of the history is missing?* It does not measure the business impact of a real outage, and it does not run the trained A6 model, whose checkpoint is not public.

## Code used by the Demo

The interface performs a real calculation with repository code. Its execution path uses:

- `read_traffic` and `split_physical_windows` read and validate the CSV;
- `global_corruption_mask` removes data in repeatable patterns;
- `build_expert_batch` builds the eight historical candidates;
- `WLCRSEA(VARIANTS["A0_fixed"])` combines them with fixed weights;
- `bounded_audit_envelope` checks that the forecast stays inside the allowed range.

The initial weights are 0.7 for the previous week, 0.2 for two weeks earlier, and 0.1 for the 7-day same-hour median. If one candidate is unavailable, the remaining weights are rescaled to add up to one. If all three are unavailable, the method uses a fallback value.

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

For sensitive traffic data, use a local or private deployment. Do not upload confidential operator data to the public Space.

## Deployment details for maintainers

The Space uses free Gradio `zero-a10g` hardware. The model calculation is wrapped in `@spaces.GPU`, and GitHub Actions mirrors the repository after every update to `main`.

The GitHub README remains standard Markdown; Space-specific metadata is added only during deployment.
