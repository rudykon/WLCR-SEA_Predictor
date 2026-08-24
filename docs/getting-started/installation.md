# Installation

Requires Python 3.11+. CI and Demo use Python 3.12.

## Clone and create an environment

```bash
git clone https://github.com/rudykon/WLCR-SEA_Predictor.git
cd WLCR-SEA_Predictor

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Verify the core method

```bash
PYTHONPATH=. python -m unittest tests.test_wlcr_sea_model -v
```

Tests cover candidate selection, missing-value masking, weights, and adjustment limits.

## Run the Live Demo locally

```bash
python demo/app.py
```

Open the Gradio URL and select the bundled example.

## Build this website

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
mkdocs serve
```

Strict mode builds both languages and fails on warnings.
