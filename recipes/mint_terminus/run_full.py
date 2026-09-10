"""The full-benchmark self-improvement driver.

One round:

    episode run every task with the current served harness and its verifier
    report  report only real verifier failures as one grouped sample
    wait    the batch opens one evolve step: the proposer reads all the
            round's failures and proposes one general change; the gate runs
            current and candidate trees over the 12-task stratified set
    repeat  until a round publishes nothing or the round budget is spent

Usage:
    PYTHONPATH=<repo> python -m recipes.mint_terminus.run_full \
        [--rounds N] [--eval-set PATH.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

from recipes.mint_terminus import baseline

#: Unique per driver launch: report ids must not collide across runs.
RUN_TAG = uuid.uuid4().hex[:8]

SERVICE_URL = os.environ.get("REEF_MINT_SERVICE_URL", "http://127.0.0.1:8912")
SCENARIO = os.environ.get("REEF_MINT_SCENARIO", "mint-terminus-full")
TOKEN = os.environ.get("REEF_MINT_TOKEN", "reef-local")
MODEL = os.environ.get("REEF_MINT_MODEL", "openai/macaron-v1-tall")
TASKS_ROOT = "/root/.reef-eval-tasks/continual-learning/terminal-bench"
DEFAULT_EVAL_SET = Path("/tmp/usable_tasks.json")
STEP_TIMEOUT_S = float(os.environ.get("REEF_MINT_STEP_TIMEOUT_S", "21600"))  # 6h per step
BASELINE_SUMMARY = Path(os.environ.get("REEF_MINT_BASELINE_SUMMARY", str(baseline.WORK / "baseline" / "_summary.json")))
ROUND_JOBS = int(os.environ.get("REEF_MINT_ROUND_JOBS", "4"))


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=SERVICE_URL,
        headers={"Authorization": f"Bearer {TOKEN}", "x-reef-scenario": SCENARIO},
        timeout=600.0,
    )


def _training_rows(client: httpx.Client) -> list[dict]:
    rows = client.get("/reef/harness/releases").json()["releases"]
    return [row for row in rows if row.get("operation") == "training"]


def one_round(client: httpx.Client, tasks: list[str], round_no: int) -> str:
    """Record, report failures, and wait for the step verdict.

    Returns 'published', 'rejected', 'skipped', or 'timeout'.
    """
    try:
        response = client.get("/reef/harness")
        response.raise_for_status()
        manifest = response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise
        # The first recorded request creates the scenario and exposes its seed.
        instruction = (Path(tasks[0]) / "instruction.md").read_text(encoding="utf-8")
        bootstrap = client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": instruction}], "max_tokens": 256},
        )
        bootstrap.raise_for_status()
        response = client.get("/reef/harness")
        response.raise_for_status()
        manifest = response.json()
    files = manifest.get("files") or {}
    print(f"  round {round_no}: running {len(tasks)} real verifier episodes", flush=True)
    records = []
    out_dir = baseline.WORK / "rounds" / str(round_no)
    out_dir.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=ROUND_JOBS) as pool:
        futures = {pool.submit(baseline.run_one, task, files): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"task": task, "reward": None, "failed": True, "error": f"{type(exc).__name__}: {exc}"}
            records.append(record)
            (out_dir / f"{Path(task).name}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
            print(f"  [{index}/{len(tasks)}] {Path(task).name}: {record.get('reward')} ({baseline.classify_record(record)})", flush=True)
    summary = baseline.summarize_records(records)
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  round {round_no}: {summary['passed']} passed, {summary['failed']} task failures, {summary['infra_error']} infra errors", flush=True)
    # Snapshot before reporting: the service may finish the step before the
    # report request returns, and we still need to observe that new row.
    seen = _training_rows(client)
    reported = baseline.report_results(records, client, report_tag=f"round-{round_no}-{RUN_TAG}")
    if not reported:
        print("  no real task failures: no evolve step triggered", flush=True)
        return "skipped"

    deadline = time.monotonic() + STEP_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            rows = _training_rows(client)
        except (httpx.HTTPError, KeyError):
            time.sleep(15.0)
            continue
        for row in rows[len(seen):]:
            metrics = row.get("metrics") or {}
            verdict = "published" if metrics.get("published") else metrics.get("skipped") or "rejected"
            print(f"  round {round_no} step verdict: {verdict}", flush=True)
            print("   ", json.dumps({
                "proposer_seconds": metrics.get("proposer_seconds"),
                "candidate_scores": (metrics.get("selection") or {}).get("evaluation", {}).get("metrics", {}).get("candidate_scores"),
                "current_scores": (metrics.get("selection") or {}).get("evaluation", {}).get("metrics", {}).get("current_scores"),
                "wins": metrics.get("wins"), "losses": metrics.get("losses"), "ties": metrics.get("ties"),
            }, default=str), flush=True)
            return verdict if verdict in ("published", "skipped") else "rejected"
        time.sleep(30.0)
    return "timeout"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    args = parser.parse_args()

    if args.eval_set.exists():
        tasks = [str(Path(TASKS_ROOT) / entry["name"]) for entry in json.loads(args.eval_set.read_text())]
    else:
        tasks = [str(p) for p in sorted(Path(TASKS_ROOT).iterdir()) if (p / "task.toml").exists()]
    tasks = [t for t in tasks if (Path(t) / "instruction.md").exists()]
    print(f"evaluation set: {len(tasks)} tasks; {args.rounds} rounds", flush=True)

    if not BASELINE_SUMMARY.exists():
        raise SystemExit(
            f"baseline summary is required before optimization: {BASELINE_SUMMARY}. "
            "Run recipes.mint_terminus.baseline first."
        )
    baseline_summary = json.loads(BASELINE_SUMMARY.read_text())
    if baseline_summary.get("total") != len(tasks):
        raise SystemExit(
            f"baseline task count ({baseline_summary.get('total')}) does not match evaluation set ({len(tasks)})"
        )
    if baseline_summary.get("infra_error", 0):
        raise SystemExit(
            f"baseline contains {baseline_summary['infra_error']} infrastructure error(s); "
            "repair or explicitly exclude them before optimization"
        )

    client = _client()
    for round_no in range(1, args.rounds + 1):
        print(f"== round {round_no} ==", flush=True)
        verdict = one_round(client, tasks, round_no)
        if verdict == "timeout":
            print("step did not settle in time; stopping", flush=True)
            return 1
        if verdict == "skipped":
            print("no real failures in this round; stopping", flush=True)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
