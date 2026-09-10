"""The driver for the MinT + Terminal-Bench + Reef iteration loop.

One pass:

    episode - run the real Terminus episode and task verifier
    report  - report only the verifier's real failure score against a receipt;
              infrastructure errors are never converted into model failures
    poll    - wait for the step's verdict: the proposer (the same MinT model)
              reads the failure and proposes a rules/skill mutation; the gate
              runs current and candidate trees on the tasks; a win publishes

Usage:  python -m recipes.mint_terminus.run_loop
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx

from recipes.mint_terminus import baseline

SERVICE_URL = os.environ.get("REEF_MINT_SERVICE_URL", "http://127.0.0.1:8912")
SCENARIO = os.environ.get("REEF_MINT_SCENARIO", "mint-terminus-demo")
TOKEN = os.environ.get("REEF_MINT_TOKEN", "reef-local")
MODEL = os.environ.get("REEF_MINT_MODEL", "openai/macaron-v1-tall")
STEP_TIMEOUT_S = float(os.environ.get("REEF_MINT_STEP_TIMEOUT_S", "3600"))
INFERENCE_TIMEOUT_S = float(os.environ.get("REEF_MINT_INFERENCE_TIMEOUT_S", "660"))
INFERENCE_ATTEMPTS = int(os.environ.get("REEF_MINT_INFERENCE_ATTEMPTS", "3"))
INFERENCE_MAX_TOKENS = int(os.environ.get("REEF_MINT_INFERENCE_MAX_TOKENS", "256"))


def _inference(client: httpx.Client, instruction: str) -> httpx.Response:
    """Send the bootstrap inference, retrying transient Reef/upstream failures."""
    last_error: Exception | None = None
    for attempt in range(1, INFERENCE_ATTEMPTS + 1):
        print(f"  inference attempt {attempt}/{INFERENCE_ATTEMPTS}...", flush=True)
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": instruction}],
                    "max_tokens": INFERENCE_MAX_TOKENS,
                },
                timeout=INFERENCE_TIMEOUT_S,
            )
            if response.status_code < 500:
                response.raise_for_status()
                return response
            last_error = httpx.HTTPStatusError(
                f"Reef returned HTTP {response.status_code}: {response.text[:500]}",
                request=response.request,
                response=response,
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
            last_error = exc
        print(f"  transient inference failure: {last_error}", flush=True)
        if attempt < INFERENCE_ATTEMPTS:
            delay = 5 * attempt
            print(f"  retrying in {delay}s", flush=True)
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def main() -> int:
    tasks = [os.environ.get("REEF_MINT_TASK", "/root/.reef-eval-tasks/continual-learning/terminal-bench/fix-git")]
    client = httpx.Client(
        base_url=SERVICE_URL,
        headers={"Authorization": f"Bearer {TOKEN}", "x-reef-scenario": SCENARIO},
        timeout=INFERENCE_TIMEOUT_S,
    )

    # Run the same real verifier used by the evolve gate. A proxy turn is only
    # metadata; it must never be used to invent a failure score.
    records = []
    for index, task in enumerate(tasks, start=1):
        print(f"episode: running task {index} with the seed harness", flush=True)
        record = baseline.run_one(task)
        records.append(record)
        print(f"task {index}: reward={record.get('reward')} ({baseline.classify_record(record)})", flush=True)
    reported = baseline.report_results(records, client, report_tag="mint-terminus")
    if not reported:
        print("all tasks passed or were unscored; no evolve step triggered", flush=True)
        return 0
    print(f"reported {reported} real task failure(s); the evolve step is running", flush=True)

    # poll the releases catalog for the step's verdict
    deadline = time.monotonic() + STEP_TIMEOUT_S
    seen = 0
    while time.monotonic() < deadline:
        try:
            rows = client.get("/reef/harness/releases").json()["releases"]
        except (httpx.HTTPError, KeyError):
            time.sleep(5.0)
            continue
        training = [row for row in rows if row.get("operation") == "training"]
        for row in training[seen:]:
            metrics = row.get("metrics") or {}
            verdict = "published" if metrics.get("published") else metrics.get("skipped") or "rejected"
            print(f"step: {verdict} (release {row['release_id'][:12]})")
            print("  feedback/evaluation:")
            print(json.dumps({k: v for k, v in metrics.items() if k != "steps"}, indent=2, default=str)[:6000])
        seen = max(seen, len(training))
        if seen:
            published = [row for row in training if (row.get("metrics") or {}).get("published")]
            if published:
                head = client.get("/reef/harness").json()
                print(f"\npublished head: {head['release_id']}")
                for path in sorted(head["files"]):
                    if any(part in path for part in ("AGENTS.md", "SKILL.md", "config.json")):
                        print(f"--- {path} ---")
                        print(head["files"][path][:1500])
                return 0
            return 1
        time.sleep(10.0)
    print("no step verdict within the timeout; check the service log")
    return 1


if __name__ == "__main__":
    sys.exit(main())
