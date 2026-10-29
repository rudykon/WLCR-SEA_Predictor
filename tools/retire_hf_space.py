"""Retire the former hosted Demo without deleting its source or model repository.

Runs only with the existing repository-scoped HF_TOKEN in the manual retirement
workflow. All mutation calls target the exact Space named by the user.
"""
from __future__ import annotations

import json
import os
import time

SPACE_ID = "config-h/WLCR-SEA_Predictor"
MODEL_ID = "config-h/WLCR-SEA-Predictor"
MODEL_REVISION = "eb4447f4ebab8f9caa003d92c838ed8e750963bd"


def retire() -> dict:
    from huggingface_hub import HfApi

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("The existing HF_TOKEN Actions secret is required")
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.whoami()["name"] != "config-h":
        raise RuntimeError("The retirement credential must belong to the Space owner")
    info = api.space_info(repo_id=SPACE_ID)
    if info.id != SPACE_ID or SPACE_ID == MODEL_ID:
        raise RuntimeError("Unexpected retirement target")

    runtime = api.get_space_runtime(repo_id=SPACE_ID)
    if getattr(runtime.stage, "value", runtime.stage) != "PAUSED":
        api.pause_space(repo_id=SPACE_ID)
    if not info.private:
        api.update_repo_settings(repo_id=SPACE_ID, repo_type="space", private=True)

    # Visibility changes can update runtime state asynchronously.
    for _ in range(24):
        info = api.space_info(repo_id=SPACE_ID)
        runtime = api.get_space_runtime(repo_id=SPACE_ID)
        stage = str(getattr(runtime.stage, "value", runtime.stage))
        if info.private and stage == "PAUSED":
            break
        if stage not in {"PAUSED", "PAUSING"}:
            api.pause_space(repo_id=SPACE_ID)
        time.sleep(5)
    else:
        raise RuntimeError("Space retirement did not reach private + PAUSED")

    # Read-only check: the separate public model stays available at its pin.
    model = HfApi(token=False).model_info(repo_id=MODEL_ID, revision=MODEL_REVISION)
    if model.private or model.sha != MODEL_REVISION:
        raise RuntimeError("The existing public model release is not available")
    result = {
        "space": SPACE_ID,
        "private": bool(info.private),
        "stage": stage,
        "source_deleted": False,
        "public_model": MODEL_ID,
        "model_revision": model.sha,
        "model_still_public": not model.private,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    retire()
