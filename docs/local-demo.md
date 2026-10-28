---
hide:
  - toc
---

# Local Demo

Forecast directly in your browser using your computer's CPU. No Python installation,
account, or inference server is needed. The small model assets load from this site;
your CSV stays on your device.

<div class="local-demo" data-local-demo data-lang="en">
  <noscript>Enable JavaScript to run the local forecast.</noscript>
</div>

For the six-column CSV schema, see the [data contract](guide/data-contract.md).
This page uses the same [five public checkpoints](https://huggingface.co/config-h/WLCR-SEA-Predictor/tree/eb4447f4ebab8f9caa003d92c838ed8e750963bd)
as the project Demo. Missingness simulation follows the single-request protocol
with seed 2026; it does not reproduce a multi-request benchmark.

Prefer a hosted interface? Open the [Hugging Face Demo](https://huggingface.co/spaces/config-h/WLCR-SEA_Predictor).
