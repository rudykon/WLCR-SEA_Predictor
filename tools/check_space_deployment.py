"""Poll the public Space until its bundled sample runs on the expected source."""

from __future__ import annotations

import argparse
import json
import math
import time


def _payload(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-url", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            from gradio_client import Client

            client = Client(args.space_url, verbose=False)
            result = _payload(client.predict(api_name="/deployment_smoke"))
            if result.get("source_commit") != args.expected_commit:
                raise RuntimeError(
                    "Space still serves source "
                    f"{result.get('source_commit')!r}; expected {args.expected_commit!r}"
                )
            if result.get("member_count") != 5:
                raise RuntimeError(f"Unexpected member count: {result!r}")
            if result.get("prediction_shape") != [24, 4]:
                raise RuntimeError(f"Unexpected prediction shape: {result!r}")
            first = result.get("first_prediction", [])
            if len(first) != 4 or not all(math.isfinite(float(value)) for value in first):
                raise RuntimeError(f"Invalid deployed prediction: {result!r}")
            print(json.dumps(result, sort_keys=True))
            return
        except Exception as exc:
            last_error = exc
            time.sleep(15)

    raise SystemExit(
        f"Space did not serve the expected sample before timeout: {last_error}"
    )


if __name__ == "__main__":
    main()
