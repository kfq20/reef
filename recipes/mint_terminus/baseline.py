"""Run and persist real Terminus baseline episodes.

This is the honest measurement the loop needs: for each task in the
evaluation set it runs one complete Terminus-2 episode (agent in an E2B
sandbox, multi-turn terminal work, the task's own verifier), then records:

  - the verifier reward and outcome class (pass, task failure, or infrastructure
    error)
  - the full trial record (steps, reward, error) under work/baseline/<task>.json

Reports are optional. When requested, only verifier-scored task failures are
reported. Timeouts, runner errors, and missing rewards never become synthetic
zero scores.

Usage:
    PYTHONPATH=<repo> python -m recipes.mint_terminus.baseline [--jobs 4] [--report]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECIPE_ROOT = REPO / "recipes" / "mint_terminus"
WORK = RECIPE_ROOT / "work-full"
TASKS_ROOT = Path("/root/.reef-eval-tasks/continual-learning/terminal-bench")
DEFAULT_EVAL_SET = Path("/tmp/usable_tasks.json")
EPISODE_TIMEOUT_S = 4800.0
WRAPPER = str(RECIPE_ROOT / "bin" / "reef-terminus-mint")

import httpx

SERVICE_URL = os.environ.get("REEF_MINT_SERVICE_URL", "http://127.0.0.1:8912")
SCENARIO = os.environ.get("REEF_MINT_SCENARIO", "mint-terminus-full")
TOKEN = os.environ.get("REEF_MINT_TOKEN", "reef-local")
MODEL = os.environ.get("REEF_MINT_MODEL", "openai/macaron-v1-tall")

OUTCOME_PASSED = "passed"
OUTCOME_FAILED = "failed"
OUTCOME_INFRA_ERROR = "infra_error"


def render_tree(target: Path, files: Mapping[str, str] | None = None) -> None:
    """Render the seed, or materialize a published tree, with this run's binding."""
    if files is None:
        from reef.harness.adapters import get_adapter
        from reef.harness.tree.render import render_composition

        nodes = [
            ("rules", {"text": "Work carefully: run commands to verify every claim before you\nfinish, and re-read the task instruction before you answer."}),
            ("skill", {"name": "verify-results", "text": "# verify-results\n---\nname: verify-results\ndescription: Check work by running commands before declaring done.\n---\n\nBefore declaring a task complete, run a command that proves it:\nre-run the failing case, print the file's content, or query the\ninstalled state. If the check fails, keep working."}),
            ("config", {"data": {"llm_call_kwargs": {"max_tokens": 16384}}}),
        ]
        adapter = get_adapter("terminus")
        rendered = dict(render_composition(nodes, adapter))
    else:
        rendered = dict(files)
    config = json.loads(rendered.get("terminus/config.json", "{}"))
    config.update({
        "model_name": "openai/macaron-v1-tall",
        "api_base": "http://127.0.0.1:8971/v1",
        "llm_kwargs": {"api_key": os.environ.get("MINT_API_KEY", "")},
    })
    rendered["terminus/config.json"] = json.dumps(config, indent=2)
    for relative, text in rendered.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def run_one(task: str, files: Mapping[str, str] | None = None) -> dict:
    """One full episode for one task; returns the trial record."""
    task_name = Path(task).name
    with tempfile.TemporaryDirectory(prefix=f"baseline-{task_name}-") as tmp:
        root = Path(tmp) / "root"
        root.mkdir()
        render_tree(root, files)
        sessions = root / "terminus" / "sessions"
        trials = root / "terminus" / "trials"
        env = {
            **{k: v for k, v in os.environ.items() if k in ("PATH",)},
            "REEF_TERMINUS_DIR": str(root),
            "REEF_TERMINUS_SESSION_DIR": str(sessions),
            "REEF_TERMINUS_TRIALS_DIR": str(trials),
        }
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [WRAPPER, "--task", task],
                env=env, cwd=str(root), capture_output=True, text=True, timeout=EPISODE_TIMEOUT_S,
            )
            stderr_tail = proc.stderr[-400:]
        except subprocess.TimeoutExpired:
            return {"task": task, "reward": None, "failed": True, "error": f"episode exceeded {EPISODE_TIMEOUT_S}s", "seconds": EPISODE_TIMEOUT_S}
        seconds = round(time.monotonic() - started, 1)
        session_file = sessions / f"{Path(task).name}.json"
        if not session_file.exists():
            return {"task": task, "reward": None, "failed": True, "error": f"no session file; exit {proc.returncode}; {stderr_tail}", "seconds": seconds}
        record = json.loads(session_file.read_text())
        record["seconds"] = seconds
        return record


def classify_record(record: Mapping) -> str:
    """Separate model/task failures from failures of the measurement itself."""
    reward = record.get("reward")
    if record.get("failed") or record.get("error") or not isinstance(reward, (int, float)):
        return OUTCOME_INFRA_ERROR
    return OUTCOME_PASSED if float(reward) >= 1.0 else OUTCOME_FAILED


def summarize_records(records: list[dict]) -> dict:
    outcomes = {name: 0 for name in (OUTCOME_PASSED, OUTCOME_FAILED, OUTCOME_INFRA_ERROR)}
    for record in records:
        outcomes[classify_record(record)] += 1
    scored = outcomes[OUTCOME_PASSED] + outcomes[OUTCOME_FAILED]
    return {
        **outcomes,
        "total": len(records),
        "scored": scored,
        "pass_rate": outcomes[OUTCOME_PASSED] / scored if scored else None,
        "results": [
            {
                "task": Path(record["task"]).name,
                "reward": record.get("reward"),
                "outcome": classify_record(record),
                "error": record.get("error") or None,
            }
            for record in records
        ],
    }


def load_records(out_dir: Path, tasks: list[str]) -> tuple[list[dict], list[str]]:
    """Load completed task records in task order and return missing tasks."""
    records = []
    missing = []
    for task in tasks:
        path = out_dir / f"{Path(task).name}.json"
        try:
            record = json.loads(path.read_text())
            if record.get("task") != task:
                raise ValueError("record belongs to a different task")
        except (OSError, ValueError, TypeError, AttributeError):
            missing.append(task)
            continue
        records.append(record)
    return records, missing


def report_results(records: list[dict], client: httpx.Client | None = None, *, report_tag: str = "baseline") -> int:
    """Report one real, grouped failure sample; return its number of task receipts."""
    failures = [record for record in records if classify_record(record) == OUTCOME_FAILED]
    if not failures:
        return 0
    owned_client = client is None
    if client is None:
        client = httpx.Client(
            base_url=SERVICE_URL,
            headers={"Authorization": f"Bearer {TOKEN}", "x-reef-scenario": SCENARIO},
            timeout=900.0,
        )
    receipts = []
    reported = []
    for record in failures:
        task = record["task"]
        score = float(record["reward"])
        instruction = (Path(task) / "instruction.md").read_text(encoding="utf-8")
        for attempt in range(3):
            try:
                response = client.post(
                    "/v1/chat/completions",
                    json={"model": MODEL, "messages": [{"role": "user", "content": instruction}], "max_tokens": 2048},
                )
                response.raise_for_status()
                receipts.append(response.headers["x-reef-agent-record-id"])
                reported.append({"task": Path(task).name, "score": score})
                break
            except httpx.HTTPError as exc:
                print(f"  report retry {attempt + 1} for {Path(task).name}: {type(exc).__name__}", flush=True)
                time.sleep(5.0)
    if receipts:
        # A report over several receipts is one trajectory sample. Its score is
        # the minimum verifier reward: the trajectory passes only if every task
        # passes. Per-task scores remain explicit in feedback.
        response = client.post(
            "/reef/report",
            json={
                "agent_record_id": f"{report_tag}-{uuid.uuid4().hex[:12]}",
                "score": min(item["score"] for item in reported),
                "feedback": {"kind": "terminal_bench_failures", "results": reported},
                "references": receipts,
            },
        )
        response.raise_for_status()
        print(f"  reported one real failure batch with {len(receipts)} task(s)", flush=True)
    if owned_client:
        client.close()
    return len(receipts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--report", action="store_true", help="report scores to reef after the run")
    args = parser.parse_args()

    if args.eval_set.exists():
        tasks = [str(TASKS_ROOT / entry["name"]) for entry in json.loads(args.eval_set.read_text())]
    else:
        tasks = [str(p) for p in sorted(TASKS_ROOT.iterdir()) if (p / "task.toml").exists()]
    tasks = [t for t in tasks if (Path(t) / "instruction.md").exists()]
    print(f"baseline: {len(tasks)} tasks, {args.jobs} parallel episodes", flush=True)

    out_dir = WORK / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    records, _ = load_records(out_dir, tasks)
    done = {record["task"] for record in records}
    remaining = [t for t in tasks if t not in done]
    print(f"resuming: {len(done)} already measured, {len(remaining)} to run", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, task): task for task in remaining}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 - one task's crash must not kill the run
                record = {"task": task, "reward": None, "failed": True, "error": f"{type(exc).__name__}: {exc}", "seconds": 0}
            records.append(record)
            (out_dir / f"{Path(task).name}.json").write_text(json.dumps(record, indent=2, default=str))
            reward = record.get("reward")
            print(f"[{index}/{len(remaining)}] {Path(task).name}: reward={reward} ({record.get('seconds')}s)" + (f" err={str(record.get('error'))[:100]}" if record.get("error") else ""), flush=True)

    summary = summarize_records(records)
    rate = summary["pass_rate"]
    rate_text = f"{rate:.1%}" if rate is not None else "n/a"
    print(
        f"\nbaseline: {summary['passed']}/{summary['scored']} scored episodes passed ({rate_text}); "
        f"{summary['infra_error']} infrastructure error(s)",
        flush=True,
    )
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.report:
        print("\nreporting real scores to reef...", flush=True)
        report_results(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
