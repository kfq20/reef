# tttd_mlx

TTT-Discover on one Apple Silicon machine, with MLX serving **and** training in a
single Reef process. No CUDA, no Ray, no Slime, no second GPU box.

This is a **functional smoke**, not the paper reproduction. The reproduction
lives in [examples/tttd](../tttd/README.md) and runs Qwen3-8B on four B200s.
What this example demonstrates is that the whole model-evolution lifecycle —
serve, record, batch, train, evaluate, select, publish, activate — closes on a
laptop, on the same recipe code the GPU run uses.

## What it exercises

`recipes/tttd` is unmodified. The recipe, its processor, its report contract and
its `tttd` step preparer are exactly what the Slime deployment runs; only the
runtime underneath changes, selected by one config key:

```yaml
reef:
  runtime_type: mlx
  runtime_config:
    checkpoint_dir: work/checkpoints
```

Rollouts still arrive from outside the runtime. `run.py` is an ordinary Reef
client: it asks for chat completions, scores them, and reports each one into
its `(step, group, rollout)` slot in the TTTD grid. The runtime never generates
its own training data — the same invariant the Slime path holds.

## Topology

**Colocated, single host, single process, synchronous.** One `MLXEngine` owns
the base model, the LoRA adapter and the optimizer; serving reads those
parameters and training writes them, and the two are kept apart by the
`InferenceAdmissionController` Reef already uses for colocated Slime. Every MLX
operation runs on one dedicated engine thread, because MLX streams are
per-thread.

The optimizer lives with the engine across steps, so Reef's step *N+1*
continues the Adam moments and step counter that step *N* left behind. The
`optimizer_step` metric below is that continuity, made observable.

The frozen-base log-probabilities TTT-Discover's KL term needs are computed by
zeroing `lora_b` in place and restoring it — LoRA computes `x @ A @ B * scale`,
so a zeroed `B` *is* the base model, and no second copy of the weights sits in
unified memory.

## Run it

Requires an Apple Silicon Mac, `git-lfs`, and the optional MLX extra:

```bash
pip install -e '.[mlx]'          # or: pip install 'reef-infra[mlx]'
git lfs install

reef serve -c recipes/tttd/examples/tttd_mlx/serve.yaml &
python recipes/tttd/examples/tttd_mlx/run.py --steps 3 --summary summary.json
```

The default grid is 2 groups x 4 rollouts on `Qwen2.5-0.5B-Instruct-4bit`.
Both `serve.yaml` and `run.py` default to the same grid; a mismatch is refused
by the processor rather than silently trained on a partial step.

## What a run looks like

Measured on an M4 Pro / 48 GB, 8 rollouts per step at 96 max new tokens:

```
step 0: mean reward 0.7914, accuracy 0.62  ->  published in 3.2s
step 1: mean reward 1.0625, accuracy 0.88  ->  published in 3.0s
step 2: mean reward 1.1977, accuracy 1.00  ->  published in 3.0s
```

and the training metrics each step carries into its commit record:

```
opt_step=1  delta_l2=0.080952  changed=32/32  ratio=0.999434  kl=+6.27e-04  tokens=426
opt_step=2  delta_l2=0.133624  changed=32/32  ratio=1.000663  kl=-6.11e-04  tokens=339
opt_step=3  delta_l2=0.129960  changed=32/32  ratio=0.999387  kl=+6.29e-04  tokens=309
```

Read them as follows:

- `opt_step` increments across Reef steps — the optimizer is not rebuilt per
  step.
- `delta_l2` and `changed` are the proof the update is real. The runtime
  **refuses to publish** a candidate whose adapter tensors did not move, so a
  step that trains nothing fails loudly instead of publishing an adapter
  identical to the base with a success record attached.
- `ratio` is `exp(logπθ − logπrollout)` before the update. It sits at 1.000 ±
  0.001 because the batch is on-policy, which also confirms that the log-probs
  recorded at generation line up token-for-token with the ones recomputed at
  training time. A drift away from 1 would mean the batch aged behind the
  serving weights.

Each step publishes a durable adapter artifact through Reef's normal Git-LFS
stack, carrying its compatibility provenance:

```json
{
  "schema": "reef.mlx.adapter/1",
  "base_model": ".../Qwen2.5-0.5B-Instruct-4bit",
  "lora_parameters": {"rank": 8, "scale": 2.0, "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
  "libraries": {"mlx": "0.32.2", "mlx-lm": "0.31.3"},
  "rollout": "in-process mlx-lm generate_step",
  "objective": "tttd-importance-sampling",
  "scenario_step": 0,
  "source_runtime_load_id": "mlx-37773-1"
}
```

The adapter directory is `mlx-lm`'s own format (`adapters.safetensors` plus
`adapter_config.json`), so `mlx_lm.tuner.utils.load_adapters` loads a published
artifact directly. It is written to a staging directory and renamed into place,
so an interrupted publication leaves either the previous complete adapter or
the new one, never a half-written one.

## What this does not show

- **Not a learning result.** Eight rollouts per step on a 0.5B model is far too
  little signal to claim the objective works; the reward movement above is
  within noise. Treat it as evidence the *lifecycle* closes, and see
  [examples/tttd](../tttd/README.md) for the result-level reproduction.
- **Not paper-scale.** The paper's grid is 8x64 on Qwen3-8B; this is 2x4 on a
  0.5B base with rank-8 LoRA over 8 blocks.
- **No multi-node, no asynchronous rollout, no bounded staleness.** The runtime
  refuses `max_staleness > 0` rather than pretending to correct for it.
- **Single scenario.** Per-scenario adapter slots are not implemented here.

## Tuning for your machine

Unified memory scales with the number of sequences held in one backward pass
times their length. In order of effect: lower `micro_batch_size`, then
`max_tokens`, then `rollouts_per_group`, then `lora_layers`. Generation
dominates step time, so `max_tokens` is also the main throughput knob.

`temperature` should stay at 1.0 for training rollouts. The recorded behaviour
proxy is the model's own log-softmax at the sampled token, so a tempered
sampling distribution makes the importance ratio approximate rather than exact.

**Do not set `rollouts_per_group: 2`.** This is a property of TTT-Discover's
adaptive-entropic advantages, not of the MLX path: the leave-one-out
normalizer has a single "other" sample at *G*=2, so the solved beta runs away
and the advantages blow up. Measured on rewards `[1, 0, …]`:

| group size | solved beta | max abs advantage |
| --- | --- | --- |
| 2 | 41.13 | 1e+12 |
| 3 | 2.848 | 16.26 |
| 4 | 2.553 | 11.85 |
| 8 | 2.465 | 10.77 |

Four is the smallest group worth training on; the paper uses 64.
