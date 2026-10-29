"""Delete only the former hosted Demo; keep the public model repository intact.

Runs only with the existing repository-scoped HF_TOKEN in the manual retirement
workflow. The Space implementation remains versioned in this GitHub repository.
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
    from huggingface_hub.errors import RepositoryNotFoundError

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("The existing HF_TOKEN Actions secret is required")
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.whoami()["name"] != "config-h":
        raise RuntimeError("The retirement credential must belong to the Space owner")
    if SPACE_ID != "config-h/WLCR-SEA_Predictor" or SPACE_ID == MODEL_ID:
        raise RuntimeError("Unexpected retirement target")

    # Read-only preflight: the separately published model must remain available.
    public = HfApi(token=False)
    model = public.model_info(repo_id=MODEL_ID, revision=MODEL_REVISION)
    if model.private or model.sha != MODEL_REVISION:
        raise RuntimeError("The existing public model release is not available")
    try:
        info = api.space_info(repo_id=SPACE_ID)
    except RepositoryNotFoundError:
        info = None  # Allow retry after successful deletion.
    if info is not None and info.id != SPACE_ID:
        raise RuntimeError("Unexpected retirement target")

    api.delete_repo(repo_id=SPACE_ID, repo_type="space", missing_ok=True)

    # Check as the owner: a private or paused Space still appears in this list.
    for _ in range(24):
        try:
            api.space_info(repo_id=SPACE_ID)
        except RepositoryNotFoundError:
            if not any(space.id == SPACE_ID for space in api.list_spaces(author="config-h")):
                break
        time.sleep(5)
    else:
        raise RuntimeError("The former hosted Space still exists after deletion")

    model = public.model_info(repo_id=MODEL_ID, revision=MODEL_REVISION)
    if model.private or model.sha != MODEL_REVISION:
        raise RuntimeError("The existing public model release is not available")
    result = {
        "space": SPACE_ID,
        "space_exists": False,
        "space_deleted": True,
        "github_source_retained": True,
        "public_model": MODEL_ID,
        "model_revision": model.sha,
        "model_still_public": not model.private,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    retire()
