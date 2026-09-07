# OpenClaw-RL on Apple Silicon — Qwen3.8-27B, MLX

The GSM8K homework stream, trained on a single Mac. The policy both serves and
trains in one process on the [MLX backend](../../../../../../reef/train/mlx_backend);
the two roles that only ever need text — the PRM's accept/reject votes and the
student's reactions — go to a hosted GLM-5.3 over an OpenAI-compatible endpoint.
The reference stack for this run needs seven GPUs (Megatron on four, a rollout
engine, a served PRM, and a 32B student); this fits it on one 48 GB machine.

**It learns.** The stream reaches the benchmark's own adaptation metric, and a
held-out control separates the trained policy from the base model at
`p = 1.7e-8`.

## How it was run

The reference harness ([`harness/agent.py`](../../harness/agent.py)) runs each
session as a Harbor task — a container with the real Hermes agent, a judge
sidecar, and reef-eval sequencing them. That stack does not run on Apple
Silicon, so this result was produced by a **Mac-native driver standing in for
Hermes**, aligned to the harness's session semantics: the same `problem.json`
files, the unmodified `StudentSession` judge and acceptance criterion,
`MAX_TURNS = 8`, an uncapped tool loop, and a dead turn recorded as a `failure`
rather than scored. The one behavioural difference is tool-call plumbing —
Hermes reads structured `tool_calls` from the API, while this driver parsed
them from the reply text. The training tokens and log-probs are identical
either way (they are captured from the raw stream, before any presentation
split), so this bears on the agent loop and the judged prose, not the gradient.

## Result

| metric | this run (MLX, 27B) | reference (4B, 7×GPU) |
|---|---|---|
| `sessions_to_adaptation` | **21** | 14 |
| trainable parameters | 45M (LoRA, 2 layers) | full-parameter 4B |
| hardware | 1× Apple Silicon, 48 GB | 7× GPU |

`sessions_to_adaptation` is the first of three consecutive accepted sessions,
computed by the recipe's own [`learning_curve.py`](../learning_curve.py) from
the verdicts on disk. Accepts landed at s21–24, then intermittently to s70.

![learning curve](learning_curve.png)

Accumulated accepts (top) start at s21; the style-violation rates the criterion
penalises (bottom) fall as the policy adapts.

## The control

The training verdicts are one turn each and could ride on luck or an easy
problem. So after the run, the opening turn (the homework file already read) was
resampled 10× from the **base model** and from two trained checkpoints on each
of the run's accepted problems, counting replies with no style violation.

| checkpoint | clean replies | vs base | one-sided Fisher |
|---|---|---|---|
| base | **0 / 120** | — | — |
| adapted (step 12) | 24 / 120 | +24 | **p = 1.7e-8** |
| final (step 71) | 37 / 120 | +37 | **p = 2.7e-13** |

![base vs trained control](control.png)

The base never produced a clean reply on any of the twelve problems (240
samples, zero). Note the final checkpoint answers *more* cleanly than the
adapted one, not less — see "What collapses" below.

## Why it is hard on this model

48 of Qwen3.8-27B's 64 layers are `GatedDeltaNet`, whose kernel has no vjp, so a
gradient cannot cross one. The reachable surface is **layer 63** (the last
full-attention block) whole, plus **layer 62's MLP** — the gradient reaches it
from 63 without passing through 62's attention. `lora_layers: 3` fails with
`[Primitive::vjp] Not implemented`. At rank 256 that is ~45M trainable
parameters. Learning two layers of a 27B is the whole difficulty; the KL-to-base
term below is what keeps those two layers from wandering.

## Configuration

See [`serve.yaml`](serve.yaml). The load-bearing choices:

- `lora_layers: 2`, `lora_rank: 256` — the reachable surface, measured not guessed.
- `kl_coef 0.05` (frozen-base k3 KL) + `weight_decay 0.1` — without the KL term,
  earlier runs collapsed into an unconditional tool loop; it prices the drift of
  the mass neither objective term targets.
- `prm_api: openai`, `prm_model: z-ai/glm-5.3` — the judge is a hosted model, so
  the deployment serves only the policy.
- `batch_size: 8` — half the reference's 16, to get more optimizer steps out of a
  72-session stream on one machine.

The API key is read from `$OPENROUTER_API_KEY`; the config records only the
variable name.

## What collapses (and what does not)

Four to five optimizer steps after adapting, the run enters a tool loop — the
model reads/writes its file repeatedly instead of answering — and sessions start
timing out. But the control above shows the *final* checkpoint answers better
than the adapted one: what degrades is confined to the **save turn** of the
agent loop, not the answer. The verifier grades only the first reply, which
stays healthy; the loop that eats later turns is the objective pushing the two
trainable layers past where they still generalise. Stabilising the tail (a lower
learning rate, or a stronger KL, past the adaptation point) is the open item.

## Files

- `learning_curve.png`, `curve.csv` — the run, from `learning_curve.py`.
- `control.png` — base vs trained clean-reply rates on the accepted problems.
- `serve.yaml` — the deployment (policy on MLX, judge + student on GLM-5.3).
- [`mlx-runtime-notes.md`](mlx-runtime-notes.md) — runtime-general operational
  notes: topology, how to read a step's training metrics, the adapter artifact
  format, and the measured capacity envelope (how large a model fits on MLX).
