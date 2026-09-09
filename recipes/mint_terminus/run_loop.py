"""The driver for the MinT + Terminal-Bench + Reef iteration loop.

One pass:

    record  - each task goes once through reef inference (a proxy turn that
              reef records against a receipt; the score comes from the task's
              own verdict, reported below)
    report  - report a failing score (0.0) against the first receipt, which
              batches (max_score: 0.0) and triggers one gated evolve step
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

    # record: one proxy inference per task, keeping the receipt
    receipts = []
    for index, task in enumerate(tasks, start=1):
        print(f"record: sending task {index} through Reef", flush=True)
        with open(f"{task}/instruction.md", encoding="utf-8") as handle:
            instruction = handle.read()
        response = _inference(client, instruction)
        receipt = response.headers["x-reef-agent-record-id"]
        receipts.append(receipt)
        print(f"task {index} recorded (receipt {receipt[:12]}...)", flush=True)
        try:
            body = response.json()
            choices = body.get("choices") or []
            content = ((choices[0].get("message") or {}).get("content") if choices else None)
            if content:
                print("  model response:", flush=True)
                print(str(content)[:2000], flush=True)
            elif choices:
                reasoning = (choices[0].get("message") or {}).get("reasoning")
                if reasoning:
                    print("  model reasoning (response content was empty):", flush=True)
                    print(str(reasoning)[:2000], flush=True)
        except (ValueError, IndexError, AttributeError, TypeError):
            pass

    # report: a failing score batches and opens one evolve step
    report = client.post(
        "/reef/report",
        json={
            "agent_record_id": "mint-terminus-1",
            "score": 0.0,
            "feedback": "the agent did not finish the task within its turn budget",
            "references": receipts,
        },
    )
    report.raise_for_status()
    print("failure reported; the evolve step is running (episodes take many minutes)", flush=True)
    try:
        print("  report:", json.dumps(report.json(), default=str))
    except (ValueError, TypeError):
        print("  report response:", report.text[:1000])

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
