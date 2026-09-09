# MinT macaron-v1-tall + Terminal-Bench 2 + Reef: a training-free self-improvement loop

This recipe runs Reef's harness evolution (`CordisRecipe`) with the **terminus**
adapter: the agent under evolution is Terminus 2 driving Terminal-Bench tasks in
remote E2B sandboxes, the model under test is MinT's `macaron-v1-tall`
(a frozen Qwen3.6-35B-A3B base with 4×1B LoRA experts), and no local GPU or
weight training is involved — the loop improves the *harness* (rules, skills,
config knobs), not the weights.

One pass of the loop:

```text
record   the driver sends the task instruction through reef's inference proxy
         and keeps the receipt
report   a failing score (0.0) against the receipt batches (max_score: 0.0)
propose  the served model itself reads the failure + current tree and answers
         with a strict-JSON mutation: a rules entry, a skill, or Terminus 2
         config knobs (self-proposer, no human in the loop)
gate     current tree vs candidate tree, one real Terminal-Bench episode per
         task per side, in E2B, scored by the task's own verifier
publish  score_comparison: the candidate publishes only on more wins than
         losses; the rejected case keeps the current release serving
```

## Measured result (2026-09-09)

Baseline probe of hard Terminal-Bench tasks on the seed tree found real
failures. The loop then ran once against them:

| Task (difficulty: hard) | seed tree | evolved tree |
|---|---|---|
| `cancel-async-tasks` | 0.0 | **1.0** |
| `make-mips-interpreter` | timeout | timeout |

Verdict: 1 win / 0 losses / 1 tie → **published**. The proposer (the served
model, 17 s) rewrote the seed rules into: read the task from scratch and plan
before writing code, write complete robust code in one step, do not stop early,
verify against the requirements, re-read the instruction before declaring done.
The winning change is exactly that rules entry — no human wrote or selected it.

One gate run is one sample; episodes are stochastic (the upstream tutorial
records the same caveat). A rejected first attempt (candidate 1.0 vs current
2.0 on two easy tasks) showed the gate refusing a change that made a passing
task fail — the mechanism working in both directions.

## Directory layout

```text
deployment.yaml       the deployment: terminus adapter, tasks, seed composition,
                      MinT upstream through the shim, all credentials via env
harness/
  evolution.py        the method package: propose (self-proposer over failures,
                      rules/skill/config kinds, strict JSON parsing, knob
                      whitelist) and evaluate (verifier reward from the ATIF
                      trajectory; a run with no reward scores 0)
  openai_shim.py      ~100-line local reverse proxy: litellm demands a provider
                      prefix (openai/<model>) that the MinT endpoint refuses;
                      the shim strips it, so one URL serves both spellings
bin/
  reef-terminus-mint  episode wrapper: reef's LocalExecutor passes only PATH +
                      the descriptor env, so the E2B credentials and the e2b
                      environment mode are injected here
run_loop.py           the driver: record → report → poll verdicts (env-tunable)
run_all.sh            one-command smoke: starts shim + reef serve (reusing
                      either if already listening), runs the loop, prints the
                      final status
```

## Setup

Requirements: Python 3.12+ (reef-eval[harbor] pins it), Docker not required
locally — Terminal-Bench tasks run in remote E2B sandboxes.

```bash
# from the repo root
uv venv --python 3.12 .venv-tb && source .venv-tb/bin/activate
uv pip install -e ".[terminus]"        # reef + reef-eval[harbor] + harbor
git lfs install                         # once; reef's artifact history needs it
```

Fetch the task catalog (a first `resolve` downloads the pinned benchmark; if
the clone is slow, it can be copied from another machine):

```bash
export REEF_EVAL_TASKS_DIR=~/.reef-eval-tasks
python -c "from reef_eval.targets import tasks; print(len(tasks('terminal-bench')))"
```

Credentials (all via environment, nothing in the files):

```bash
export MINT_API_KEY=...                 # MinT provider key
export E2B_API_KEY=...                  # E2B sandbox cluster key
export E2B_API_URL=http://<host>:<port> # the cluster's API endpoint
```

## Run

```bash
# the default smoke (fix-git): one full record → report → evolve → gate pass
recipes/mint_terminus/run_all.sh

# against a real-failure task (the measured run):
REEF_MINT_TASK=~/.reef-eval-tasks/continual-learning/terminal-bench/cancel-async-tasks \
    recipes/mint_terminus/run_all.sh
```

Every gate episode runs current and candidate trees over the deployment's
`evolution.tasks` — with two tasks that is four full Terminal-Bench episodes,
each an E2B template build (first time only) plus a multi-turn agent run plus
the verifier. Expect roughly an hour per step; state and logs land under
`work/`.

## Findings and pitfalls (worth reading before extending)

- **MinT endpoint**: the documented tinker-prod OpenAI-compatible URL did not
  answer; the hosted model API at `https://mintcn.macaron.xin/v1` works and is
  OpenAI-compatible, including `tool_calls`. The Macaron models return
  reasoning in a separate `reasoning_content` and keep `content` clean.
- **litellm provider prefix**: Terminus 2 reaches the model through litellm,
  which refuses an unprefixed model name; the MinT endpoint refuses the
  prefixed one. The shim bridges the two spellings on one URL.
- **Local Docker may not work for Terminus 2**: on a restricted docker proxy
  (exec/run stdout stripped, host binds refused) the agent sees a dead terminal
  and every episode times out. Remote E2B sandboxes (with commands executed as
  `root`) work; note some E2B clusters only allow the root user.
- **Task image choice matters**: Terminus 2 needs tmux inside the task
  container and tries to apt-install it. On `ubuntu:24.04`-based images the
  ubuntu archives may be unreachable from the sandbox and the episode dies at
  "tmux: command not found"; `python:3.13-slim-bookworm` (debian) images
  installed fine. Pick debian-based tasks.
- **Reasoning models need a generous reply cap**: with the default max_tokens
  the model spent the whole budget reasoning, the JSON reply was truncated,
  Terminus 2's parser saw no JSON, and episodes died after ~5 steps. The seed
  config carries `llm_call_kwargs: {max_tokens: 16384}` and episodes survived
  to 30–40 steps. This knob is also an evolution target the proposer can tune.
- **Two adapter-level admission refusals seen from the proposer**, both fixed
  in `harness/evolution.py`: an id colliding with an existing entry (`create`
  where the tree already carries the id), and config keys outside Terminus 2's
  constructor whitelist. The parser now dedups ids and filters knobs.

## Known limitations

- The record phase is a single proxied chat turn plus a synthetic failing
  report — enough to drive the loop, but not a real episode failure trace. A
  truer closed loop points the episode's model endpoint at reef itself so
  agent traffic is recorded, and reports the verifier's score against those
  receipts.
- `episode_timeout_s: 1800` is tight for construction tasks like
  `make-mips-interpreter` (the model works correctly but slowly; both sides
  timed out into a tie).
- Ubuntu-based task images are effectively unusable here (tmux install); the
  measured runs used debian-based tasks only.
