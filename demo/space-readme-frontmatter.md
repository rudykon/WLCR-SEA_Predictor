---
title: WLCR-SEA A6 Forecast Demo
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

# WLCR-SEA A6 Forecast Demo

One cell · 336 hours of history → 24 hours of forecast.

This CPU Demo runs the five public `A6_mixed_aug` checkpoints, verifies their
SHA-256 hashes, uses their frozen training priors, and averages forecasts in
linear traffic space. Use the built-in synthetic sample or upload a compatible
CSV. Do not upload confidential operator traffic.

[Project website](https://rudykon.github.io/WLCR-SEA_Predictor/) ·
[Model weights](https://huggingface.co/config-h/WLCR-SEA-Predictor) ·
[Source](https://github.com/rudykon/WLCR-SEA_Predictor)
