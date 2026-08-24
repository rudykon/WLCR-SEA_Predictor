# Installation

WLCR-SEA is a Python research repository. Use Python 3.11 or newer; the
website and hosted Demo are validated with Python 3.12.

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

The focused tests check that historical hours are selected correctly, missing
values cannot affect the result, candidate weights are valid, and the final
adjustment stays within its configured limit.

## Run the public method Demo locally

```bash
python demo/app.py
```

Open the local URL printed by Gradio. The bundled synthetic request is already
available in the Examples section.

## Build this website

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
mkdocs serve
```

`mkdocs build --strict` builds both English and Chinese pages and treats
configuration or navigation warnings as errors.
