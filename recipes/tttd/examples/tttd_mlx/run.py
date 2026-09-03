"""Drive one TTT-Discover grid per step against a Mac-local Reef service.

TTT-Discover makes repeated attempts at a hard problem at test time: each
step samples ``rollouts_per_group`` sibling attempts from each of
``groups_per_step`` parents, trains on the whole grid, and runs the next grid
on the weights that step produced. This driver is the smallest honest version
of that loop — the search is a fixed problem set rather than a PUCT archive,
and the scorer checks arithmetic instead of certifying a bound.

Every rollout in a step must be served by the same release, so the driver
waits for the step's publication before starting the next grid; a step whose
reports span releases is discarded by the processor, by design.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "third_party" / "reef-client"))

from reef_client import ReefClient, ReefClientError

SCENARIO = "tttd-mlx"

#: Problems hard enough for a 0.5B model to get some attempts wrong, so a
#: group's rewards vary. A group with no reward variance carries no relative
#: signal, and TTT-Discover drops it.
PROBLEMS: tuple[tuple[str, str], ...] = (
    ("What is 17 times 23? Reason briefly, then end with 'Answer: <number>'.", "391"),
    ("What is 144 divided by 9, then times 7? Reason briefly, then end with 'Answer: <number>'.", "112"),
    ("A shop sells pens at 13 each. What do 27 pens cost? Reason briefly, then end with 'Answer: <number>'.", "351"),
    ("What is 19 squared minus 41? Reason briefly, then end with 'Answer: <number>'.", "320"),
)


@dataclass(frozen=True)
class Rollout:
    group: int
    rollout: int
    reward: float
    correct: bool
    text: str


def reward_for(text: str, answer: str) -> tuple[float, bool]:
    """Correctness plus brevity.

    Correctness alone is often constant across a group — every sibling right
    or every sibling wrong — and a constant group trains nothing. The brevity
    term keeps the relative signal alive without changing which answer wins.
    """
    numbers = re.findall(r"-?\d+", text.replace(",", ""))
    correct = bool(numbers) and numbers[-1] == answer
    brevity = max(0.0, 1.0 - len(text) / 400.0)
    return (1.0 if correct else 0.0) + 0.25 * brevity, correct


def current_release(client: ReefClient) -> str:
    body = client.post("/reef/scenarios", SCENARIO, {"name": SCENARIO})[0]
    release = body.get("release_id")
    if not isinstance(release, str) or not release:
        raise ReefClientError(502, f"scenario response carries no release_id: {body}")
    return release


def wait_for_publication(client: ReefClient, previous: str, *, timeout_s: float) -> str:
    """Block until the training step this grid fed publishes its release."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        release = current_release(client)
        if release != previous:
            return release
        time.sleep(2.0)
    raise TimeoutError(f"no new release within {timeout_s:g}s (still at {previous})")


def run_grid(
    client: ReefClient,
    *,
    step: int,
    groups_per_step: int,
    rollouts_per_group: int,
    max_tokens: int,
) -> list[Rollout]:
    """Serve and report one complete TTTD grid.

    No release header is sent: ``x-reef-release-id`` binds a scenario to its
    *base* release, not its head. What keeps a grid on one release is the
    barrier below — the driver does not start the next grid until this step's
    publication lands, and a step only trains once its grid is complete.
    """
    results: list[Rollout] = []
    for group in range(groups_per_step):
        prompt, answer = PROBLEMS[group % len(PROBLEMS)]
        for rollout in range(rollouts_per_group):
            response, record_id = client.inference_with_record(
                SCENARIO,
                "/v1/chat/completions",
                {
                    "model": "reef",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 1.0,
                },
            )
            text = response["choices"][0]["message"]["content"]
            reward, correct = reward_for(text, answer)
            client.report(
                SCENARIO,
                {
                    "score": reward,
                    "references": [record_id],
                    "metadata": {
                        "comparison_set": f"tttd-step-{step}-group-{group}",
                        "algorithm": "ttt-discover",
                        "step": step,
                        "group": group,
                        "rollout": rollout,
                        "groups_per_step": groups_per_step,
                        "rollouts_per_group": rollouts_per_group,
                    },
                },
            )
            results.append(Rollout(group, rollout, reward, correct, text))
            print(
                f"  step {step} group {group} rollout {rollout}: "
                f"reward={reward:.3f} correct={correct} chars={len(text)}",
                flush=True,
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8901")
    parser.add_argument("--token", default="reef-local")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--groups-per-step", type=int, default=2)
    parser.add_argument("--rollouts-per-group", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--publish-timeout-s", type=float, default=1800.0)
    parser.add_argument("--summary", type=Path, default=None, help="write a JSON run summary here")
    args = parser.parse_args()

    client = ReefClient(args.url, token=args.token, timeout_s=1800.0)
    release = current_release(client)
    print(f"scenario {SCENARIO} starts at release {release}", flush=True)

    history: list[dict[str, object]] = []
    for step in range(args.steps):
        started = time.monotonic()
        rollouts = run_grid(
            client,
            step=step,
            groups_per_step=args.groups_per_step,
            rollouts_per_group=args.rollouts_per_group,
            max_tokens=args.max_tokens,
        )
        rewards = [item.reward for item in rollouts]
        mean = sum(rewards) / len(rewards)
        accuracy = sum(item.correct for item in rollouts) / len(rollouts)
        print(f"step {step}: mean reward {mean:.4f}, accuracy {accuracy:.2f}; waiting for publication", flush=True)
        release = wait_for_publication(client, release, timeout_s=args.publish_timeout_s)
        elapsed = time.monotonic() - started
        print(f"step {step}: published {release} in {elapsed:.1f}s", flush=True)
        history.append(
            {
                "step": step,
                "mean_reward": mean,
                "accuracy": accuracy,
                "release_id": release,
                "seconds": elapsed,
            }
        )

    print("\nlearning curve (mean reward per step):")
    for entry in history:
        print(f"  step {entry['step']}: {entry['mean_reward']:.4f}  ->  {entry['release_id']}")
    releases = client.get(f"/reef/scenarios/{SCENARIO}/releases")
    print(f"\npublished releases: {json.dumps(releases, indent=2)[:2000]}")
    if args.summary is not None:
        args.summary.write_text(json.dumps({"steps": history, "releases": releases}, indent=2), encoding="utf-8")
        print(f"summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
