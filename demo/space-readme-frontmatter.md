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

[Project website](https://rudykon.github.io/WLCR-SEA_Predictor/) ·
[Model weights](https://huggingface.co/config-h/WLCR-SEA-Predictor) ·
[Source](https://github.com/rudykon/WLCR-SEA_Predictor)
