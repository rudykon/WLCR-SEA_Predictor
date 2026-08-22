# Hugging Face Audit Lab

[Launch WLCR-SEA Request Audit Lab](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor){ .md-button .md-button--primary target="_blank" rel="noopener" }
[View Demo source](https://github.com/rudykon/WLCR-SEA_Predictor/tree/main/demo){ .md-button target="_blank" rel="noopener" }

The public Gradio Space makes the repository's request-local mechanics
interactive without inventing a trained checkpoint.

## What it runs

The execution path imports and uses:

- `read_traffic` and `split_physical_windows` for the physical CSV contract;
- `global_corruption_mask` for deterministic telemetry-loss scenarios;
- `build_expert_batch` for the eight real seasonal experts;
- `WLCRSEA(VARIANTS["A0_fixed"])` for the registered parameter-free mixture;
- `bounded_audit_envelope` for a structural containment check.

The fixed mixture starts from weekly lag 0.7, biweekly lag 0.2, and seven-day
same-hour median 0.1. Unavailable components are removed and the remaining mass
is renormalized. If all three are unavailable, the method uses the fallback
slot.

<div class="notice-card">
  <strong>Not trained A6 inference.</strong> The repository does not distribute the fitted A6 checkpoint or frozen training prior. The Demo's last-resort slot is a request-derived fallback and is marked with an asterisk. Exported JSON records set <code>paper_model: false</code>.
</div>

## Controls

| Control | Purpose |
| --- | --- |
| Request CSV | One 336-hour, one-cell physical window |
| Telemetry scenario | Clean, random hour, contiguous block, recent tail, or asynchronous indicator loss |
| Missingness rate | Deterministic additional removal from 0% to 80% |
| Indicator | Selects the detailed expert record |
| Horizon | Selects one of the 24 future hours for expert inspection |

## Outputs

- four history and 24-hour forecast panels;
- the available-expert min/max envelope;
- candidate values, availability, reliability, and routing weight;
- exact unavailable-expert mass and envelope status;
- downloadable 24-hour forecast CSV;
- downloadable versioned JSON audit record with the input SHA-256.

## Run locally

```bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor
python -m pip install -r requirements.txt
python demo/app.py
```

For sensitive traffic, local or private deployment is the appropriate path.
Do not upload confidential operator data to the public Space.

## Free ZeroGPU deployment

The Space is declared as a Gradio `zero-a10g` deployment. One tiny fixed-model
pass is wrapped in `@spaces.GPU`; model imports remain inside the runtime path so
`spaces` is imported first. GitHub Actions stages the Space README frontmatter
separately and mirrors the repository after each main-branch update.

The root GitHub README remains ordinary Markdown—there is no YAML metadata block
or table at its top.
