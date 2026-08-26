---
title: WLCR-SEA Cellular Traffic Forecast Demo
emoji: 📡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.25.0
python_version: "3.12"
app_file: demo/app.py
pinned: false
license: apache-2.0
suggested_hardware: cpu-basic
---

# WLCR-SEA Cellular Traffic Forecast Demo

One cell · 336 hours of history → 24 hours of forecast.

This CPU Demo runs a five-model ensemble from the public WLCR-SEA checkpoints,
verifies their SHA-256 hashes, uses their frozen training priors, and averages
forecasts in linear traffic space. Use the built-in synthetic sample or upload
a compatible CSV. Do not upload confidential operator traffic.

The automatic preview does not create download files. Click **Run forecast**
to enable the forecast CSV and audit JSON downloads.

The routing chart is an ensemble summary: it averages expert values and
routing weights separately and does not exactly decompose the final forecast.

Inference is CPU-only. The repository remains assigned to a legacy ZeroGPU
host configuration; the GPU decorator in the source is an unused startup
compatibility marker.

[Project website](https://rudykon.github.io/WLCR-SEA_Predictor/) ·
[Pinned model weights](https://huggingface.co/config-h/WLCR-SEA-Predictor/tree/eb4447f4ebab8f9caa003d92c838ed8e750963bd) ·
[Source](https://github.com/rudykon/WLCR-SEA_Predictor)
