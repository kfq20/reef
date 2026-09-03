"""The single-process MLX model: generation, optimization, and adapter I/O.

One engine owns one base model, one LoRA adapter and one optimizer for the
life of the service. Serving generates through the adapter Reef has published;
training mutates the same parameters, so the two never run at once — the
runtime closes inference admission around every optimizer step.

Keeping the optimizer here rather than rebuilding it per step is what makes a
Reef training step continue the previous one: the Adam moments and the step
counter survive across the whole run, exactly as they do inside a long-lived
Megatron actor.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner import linear_to_lora_layers

from reef.core.errors import ReefError

ADAPTER_WEIGHTS = "adapters.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
PROVENANCE = "reef_provenance.json"
PROVENANCE_SCHEMA = "reef.mlx.adapter/1"

DEFAULT_LORA_KEYS = ("self_attn.q_proj", "self_attn.v_proj")


class MLXEngineError(ReefError):
    """The MLX engine was asked for something it cannot do correctly."""


@dataclass(frozen=True)
class MLXEngineConfig:
    """Everything about the model that a deployment chooses.

    Generation length, group size and adapted layer count are the knobs that
    bound unified-memory use, so they are all explicit rather than implied.
    """

    model_path: str
    lora_layers: int = 8
    lora_rank: int = 8
    lora_scale: float = 2.0
    lora_dropout: float = 0.0
    lora_keys: tuple[str, ...] = DEFAULT_LORA_KEYS
    learning_rate: float = 1e-5
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0
    #: Maximum sequences held in one backward pass. Lower it when a step
    #: exhausts unified memory; the step then accumulates over several
    #: micro-batches instead of failing.
    micro_batch_size: int = 8

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path must be non-empty")
        for name in ("lora_layers", "lora_rank", "max_tokens", "micro_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not self.lora_keys:
            raise ValueError("lora_keys must name at least one projection")


@dataclass(frozen=True)
class Rollout:
    """One generated completion, kept token-native from sampling to training."""

    prompt_tokens: tuple[int, ...]
    output_tokens: tuple[int, ...]
    rollout_log_probs: tuple[float, ...]
    text: str
    finish_reason: str


@dataclass(frozen=True)
class TrainingRow:
    """One reserved sample, already carrying its algorithm-supplied advantage."""

    tokens: tuple[int, ...]
    loss_mask: tuple[int, ...]
    rollout_log_probs: tuple[float, ...]
    #: One advantage per response token. TTT-Discover starts from a single
    #: trajectory advantage and then adds a per-token frozen-base KL term, so
    #: the row carries the expanded vector rather than a scalar.
    advantages: tuple[float, ...]

    def __post_init__(self) -> None:
        response_length = len(self.loss_mask)
        if response_length == 0 or len(self.tokens) <= response_length:
            raise ValueError("a training row needs at least one prompt token and one response token")
        if len(self.rollout_log_probs) != response_length:
            raise ValueError("rollout_log_probs must cover exactly the response tokens")
        if len(self.advantages) != response_length:
            raise ValueError("advantages must cover exactly the response tokens")


@dataclass(frozen=True)
class _StepTensors:
    """One micro-batch, padded to a common width.

    Every array except ``sequences`` is laid out on target positions
    (``width - 1`` columns), so no member needs shifting at use.
    """

    sequences: mx.array
    rollout_log_probs: mx.array
    advantages: mx.array
    mask: mx.array


def _as_float(value: mx.array) -> float:
    """One scalar MLX array as a float.

    ``mx.array.item()`` is typed ``int | float | complex`` because MLX
    supports complex dtypes. Every array narrowed here is a real loss,
    log-probability or norm, so the conversion is total — but it has to be
    stated rather than assumed.
    """
    scalar = value.item()
    if isinstance(scalar, complex):
        raise MLXEngineError("expected a real scalar, got a complex one")
    return float(scalar)


def _flat(tree: Any) -> list[tuple[str, Any]]:
    """``tree_flatten`` as the list of pairs it returns without a destination.

    Its declared return type is ``list | dict`` because passing ``destination``
    changes the shape; nothing here passes one.
    """
    flattened = tree_flatten(tree)
    if not isinstance(flattened, list):
        raise MLXEngineError("tree_flatten returned a mapping where a list of pairs was expected")
    return flattened


def _as_int(value: Any) -> int:
    """One token id, whether MLX hands back a 0-d array or a plain int."""
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, complex):
            raise MLXEngineError("expected an integral token id, got a complex one")
        return int(scalar)
    return int(value)


def _token_log_probs(model: nn.Module, sequences: mx.array, mask: mx.array) -> mx.array:
    """Per-response-token log-probs, shaped like ``mask``.

    ``mask`` is 1 exactly on the target positions that correspond to response
    tokens, so the prompt contributes no gradient: the policy is only ever
    credited for tokens it chose.
    """
    logits = model(sequences[:, :-1])
    targets = sequences[:, 1:]
    # -cross_entropy is log p(target | context) without materializing a
    # [batch, length, vocab] log-softmax, which matters at 150k vocabularies.
    return -nn.losses.cross_entropy(logits, targets, reduction="none") * mask


def _response_mask(prompt_lengths: mx.array, sequence_lengths: mx.array, width: int) -> mx.array:
    """1 on target positions holding a response token, 0 elsewhere.

    Target position ``i`` predicts ``sequences[i + 1]``, so the response spans
    ``prompt_length - 1 <= i < sequence_length - 1``.
    """
    positions = mx.arange(width - 1)[None, :]
    inside = (positions >= (prompt_lengths[:, None] - 1)) & (positions < (sequence_lengths[:, None] - 1))
    return inside.astype(mx.float32)


class MLXEngine:
    """Own the model, the adapter and the optimizer for one Reef deployment.

    Every MLX operation runs on one dedicated thread. MLX streams are
    per-thread — ``mlx_lm`` generation uses a thread-local stream, and a
    stream created on one thread does not exist on another — so a model
    loaded on the service's main thread cannot be generated from an arbitrary
    asyncio worker. Owning a single engine thread fixes that and, as a
    welcome consequence, serializes serving against training by construction:
    the colocated model is never read and written at once.
    """

    def __init__(self, config: MLXEngineConfig) -> None:
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reef-mlx")
        self._publication = 0
        self._run(self._bootstrap)

    def _bootstrap(self) -> None:
        """Load the model on the engine thread that will also generate on it."""
        mx.random.seed(self._config.seed)
        # `load` returns a 3-tuple only with return_config=True.
        self._model, self._tokenizer = load(self._config.model_path)[:2]
        self._model.freeze()
        linear_to_lora_layers(self._model, self._config.lora_layers, self._lora_parameters())
        self._optimizer = optim.AdamW(learning_rate=self._config.learning_rate)

    def _run(self, work: Callable[[], Any]) -> Any:
        """Execute ``work`` on the engine thread and re-raise what it raises."""
        return self._executor.submit(work).result()

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    @property
    def config(self) -> MLXEngineConfig:
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    def _lora_parameters(self) -> dict[str, Any]:
        return {
            "rank": self._config.lora_rank,
            "scale": self._config.lora_scale,
            "dropout": self._config.lora_dropout,
            "keys": list(self._config.lora_keys),
        }

    # ---------------------------------------------------------------- serving

    def render_prompt(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        return self._run(lambda: self._render_prompt(messages))

    def _render_prompt(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        """Tokenize a chat request exactly as generation will see it.

        The chat template is applied here and nowhere else, so the tokens that
        train are the tokens that were served. Skipping it makes a base model
        run to the token ceiling instead of emitting its end-of-turn marker.
        """
        prompt = self._tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        return list(self._tokenizer.encode(prompt))

    def generate(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Rollout:
        return self._run(lambda: self._generate(prompt_tokens, max_tokens, temperature))

    def _generate(
        self,
        prompt_tokens: Sequence[int],
        max_tokens: int | None,
        temperature: float | None,
    ) -> Rollout:
        """Sample one completion, recording the log-prob of every chosen token.

        The recorded value is the model's own log-softmax at the sampled
        token. It is the behaviour proxy the importance ratio divides by, so
        it must come from the engine that generated — Reef never reconstructs
        it by re-scoring decoded text.
        """
        temperature = self._config.temperature if temperature is None else temperature
        limit = self._config.max_tokens if max_tokens is None else max_tokens
        sampler = make_sampler(temp=temperature, top_p=self._config.top_p)
        prompt = mx.array(list(prompt_tokens))
        stop_ids = set(self._tokenizer.eos_token_ids)

        output: list[int] = []
        log_probs: list[float] = []
        finish_reason = "length"
        for token, step_log_probs in generate_step(prompt, self._model, max_tokens=limit, sampler=sampler):
            # mlx-lm yields a plain int here; older versions yield a 0-d array.
            token_id = _as_int(token)
            if token_id in stop_ids:
                finish_reason = "stop"
                break
            chosen = step_log_probs[token_id]
            mx.eval(chosen)
            output.append(token_id)
            log_probs.append(_as_float(chosen))
        return Rollout(
            prompt_tokens=tuple(int(value) for value in prompt_tokens),
            output_tokens=tuple(output),
            rollout_log_probs=tuple(log_probs),
            text=self._tokenizer.decode(output),
            finish_reason=finish_reason,
        )

    # --------------------------------------------------------------- training

    def base_log_probs(self, rows: Sequence[TrainingRow]) -> list[list[float]]:
        return self._run(lambda: self._base_log_probs(rows))

    def _base_log_probs(self, rows: Sequence[TrainingRow]) -> list[list[float]]:
        """Response log-probs under the frozen base, with the adapter disabled.

        LoRA computes ``x @ A @ B * scale``, so zeroing every ``lora_b`` turns
        the adapted model back into its base without a second copy of the
        weights in unified memory.
        """
        snapshot = self._adapter_snapshot()
        zeroed = {
            name: (mx.zeros_like(value) if name.endswith("lora_b") else value) for name, value in snapshot.items()
        }
        self._apply_adapter(zeroed)
        try:
            return self._response_log_probs(rows)
        finally:
            self._apply_adapter(snapshot)

    def _response_log_probs(self, rows: Sequence[TrainingRow]) -> list[list[float]]:
        collected: list[list[float]] = []
        for start in range(0, len(rows), self._config.micro_batch_size):
            chunk = rows[start : start + self._config.micro_batch_size]
            tensors = self._pack(chunk)
            values = _token_log_probs(self._model, tensors.sequences, tensors.mask)
            mx.eval(values)
            for index, row in enumerate(chunk):
                response_length = len(row.loss_mask)
                prompt_length = len(row.tokens) - response_length
                window = values[index, prompt_length - 1 : prompt_length - 1 + response_length]
                collected.append([_as_float(value) for value in window])
        return collected

    @staticmethod
    def _on_targets(row: TrainingRow, values: Sequence[float], width: int) -> list[float]:
        """Lay a per-response-token vector onto target positions.

        Response token ``j`` is predicted by target position
        ``prompt_length - 1 + j``, and there are ``width - 1`` target
        positions, so every packed vector lines up with the model's output
        without a shift at use.
        """
        prompt_length = len(row.tokens) - len(row.loss_mask)
        return [0.0] * (prompt_length - 1) + list(values) + [0.0] * (width - len(row.tokens))

    def _pack(self, rows: Sequence[TrainingRow]) -> _StepTensors:
        pad_id = self._tokenizer.pad_token_id
        pad_id = 0 if pad_id is None else int(pad_id)
        width = max(len(row.tokens) for row in rows)
        sequences = mx.array([list(row.tokens) + [pad_id] * (width - len(row.tokens)) for row in rows])
        sequence_lengths = mx.array([len(row.tokens) for row in rows])
        prompt_lengths = mx.array([len(row.tokens) - len(row.loss_mask) for row in rows])
        rollout = mx.array([self._on_targets(row, row.rollout_log_probs, width) for row in rows])
        advantages = mx.array([self._on_targets(row, row.advantages, width) for row in rows])
        mask = _response_mask(prompt_lengths, sequence_lengths, width)
        return _StepTensors(sequences, rollout, advantages, mask)

    def train_step(self, rows: Sequence[TrainingRow]) -> dict[str, Any]:
        return self._run(lambda: self._train_step(rows))

    def _train_step(self, rows: Sequence[TrainingRow]) -> dict[str, Any]:
        """Apply one TTT-Discover importance-sampling update over ``rows``.

        The objective is the reference's un-clipped surrogate,
        ``-exp(logπθ - logπrollout) · advantage`` summed over response tokens.
        Gradients accumulate across micro-batches so the update is the same
        full-batch sum however the batch is split for memory.
        """
        if not rows:
            raise MLXEngineError("a training step requires at least one row")
        # Policy log-probs before the update, computed once. They are the
        # step's staleness diagnostic: on a fresh rollout the ratio against
        # the recorded behaviour proxy is 1, and drift away from 1 is exactly
        # how far the batch has aged behind the serving weights.
        before = self._response_log_probs(rows)
        total_ratio = 0.0
        total_kl = 0.0
        counted = 0
        for row, current in zip(before, rows, strict=True):
            for policy, rollout in zip(row, current.rollout_log_probs, strict=True):
                total_ratio += math.exp(policy - rollout)
                total_kl += rollout - policy
                counted += 1

        total_loss = 0.0
        accumulated: dict[str, mx.array] | None = None
        for start in range(0, len(rows), self._config.micro_batch_size):
            chunk = rows[start : start + self._config.micro_batch_size]
            loss, flat = self._micro_batch_gradients(self._pack(chunk))
            accumulated = flat if accumulated is None else {n: accumulated[n] + g for n, g in flat.items()}
            total_loss += loss

        if accumulated is None:
            raise MLXEngineError("training step produced no gradients")
        self._optimizer.update(self._model, tree_unflatten(list(accumulated.items())))
        mx.eval(self._model.parameters(), self._optimizer.state)
        return {
            "loss": total_loss,
            "importance_ratio": total_ratio / counted if counted else 0.0,
            "ppo_kl": total_kl / counted if counted else 0.0,
            "response_tokens": counted,
            "rows": len(rows),
            "optimizer_step": _as_int(self._optimizer.step),
        }

    # ---------------------------------------------------------------- adapter

    def adapter_snapshot(self) -> dict[str, mx.array]:
        return self._run(self._adapter_snapshot)

    def _micro_batch_gradients(self, tensors: _StepTensors) -> tuple[float, dict[str, mx.array]]:
        """Loss and gradients for one micro-batch of the TTT-Discover objective."""

        def loss_fn(model: nn.Module) -> mx.array:
            log_probs = _token_log_probs(model, tensors.sequences, tensors.mask)
            # rollout_log_probs is already laid out on target positions, so
            # no shift is needed here.
            ratio = mx.exp(log_probs - tensors.rollout_log_probs) * tensors.mask
            return (-ratio * tensors.advantages * tensors.mask).sum()

        loss, grads = nn.value_and_grad(self._model, loss_fn)(self._model)
        mx.eval(loss, grads)
        return _as_float(loss), dict(_flat(grads))

    def _adapter_snapshot(self) -> dict[str, mx.array]:
        return {name: mx.array(value) for name, value in _flat(self._model.trainable_parameters())}

    def apply_adapter(self, snapshot: Mapping[str, mx.array]) -> None:
        self._run(lambda: self._apply_adapter(snapshot))

    def _apply_adapter(self, snapshot: Mapping[str, mx.array]) -> None:
        self._model.update(tree_unflatten(list(snapshot.items())))
        mx.eval(self._model.parameters())

    def adapter_delta(self, before: Mapping[str, mx.array], after: Mapping[str, mx.array]) -> tuple[float, int]:
        return self._run(lambda: self._adapter_delta(before, after))

    def _adapter_delta(self, before: Mapping[str, mx.array], after: Mapping[str, mx.array]) -> tuple[float, int]:
        """L2 distance and the count of tensors that actually moved.

        Reef publishes a candidate only when this is non-zero: a trainer that
        reports success while every weight is unchanged must not reach the
        artifact stack as a trained update.
        """
        total = mx.zeros((), dtype=mx.float32)
        changed = 0
        for name, old in before.items():
            new = after.get(name)
            if new is None or new.shape != old.shape:
                continue
            squared = mx.sum((new.astype(mx.float32) - old.astype(mx.float32)) ** 2)
            mx.eval(squared)
            total = total + squared
            if _as_float(squared) > 0.0:
                changed += 1
        mx.eval(total)
        return _as_float(total) ** 0.5, changed

    def next_runtime_load_id(self) -> str:
        """Mint the token that identifies these serving weights.

        Namespaced by process incarnation so a restart can never hand out a
        token an earlier incarnation already used for different weights.
        """
        self._publication += 1
        return f"mlx-{os.getpid()}-{self._publication}"

    def provenance(self, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        import importlib.metadata

        def installed(package: str) -> str:
            try:
                return importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                return "unknown"

        versions = {package: installed(package) for package in ("mlx", "mlx-lm")}
        record: dict[str, Any] = {
            "schema": PROVENANCE_SCHEMA,
            "base_model": self._config.model_path,
            "tokenizer": self._config.model_path,
            "lora_parameters": self._lora_parameters(),
            "libraries": versions,
            "rollout": "in-process mlx-lm generate_step",
            "objective": "tttd-importance-sampling",
        }
        if extra:
            record.update(extra)
        return record

    def save_adapter(self, destination: Path, *, provenance_extra: Mapping[str, Any] | None = None) -> Path:
        return self._run(lambda: self._save_adapter(destination, provenance_extra))

    def _save_adapter(self, destination: Path, provenance_extra: Mapping[str, Any] | None = None) -> Path:
        """Write the adapter atomically: readers see the old one or the new one.

        Everything is written into a sibling directory, fsynced, and renamed
        into place, so an interrupted publication can never leave a partially
        written adapter where an activated one is expected.
        """
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            weights = dict(_flat(self._model.trainable_parameters()))
            if not weights:
                raise MLXEngineError("the model exposes no trainable parameters to publish")
            mx.save_safetensors(str(staging / ADAPTER_WEIGHTS), weights)
            # mlx-lm's own adapter shape: tuner.utils.load_adapters reads
            # exactly these keys, so the stock loader can serve the artifact.
            (staging / ADAPTER_CONFIG).write_text(
                json.dumps(
                    {
                        "fine_tune_type": "lora",
                        "num_layers": self._config.lora_layers,
                        "lora_parameters": self._lora_parameters(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (staging / PROVENANCE).write_text(
                json.dumps(self.provenance(extra=provenance_extra), indent=2), encoding="utf-8"
            )
            for name in (ADAPTER_WEIGHTS, ADAPTER_CONFIG, PROVENANCE):
                handle = os.open(staging / name, os.O_RDONLY)
                try:
                    os.fsync(handle)
                finally:
                    os.close(handle)
            directory = os.open(staging, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination

    def load_adapter(self, source: Path) -> None:
        self._run(lambda: self._load_adapter(source))

    def _load_adapter(self, source: Path) -> None:
        """Load a published adapter's weights into the live model."""
        weights_path = Path(source) / ADAPTER_WEIGHTS
        if not weights_path.is_file():
            raise MLXEngineError(f"adapter at {source} has no {ADAPTER_WEIGHTS}")
        loaded = mx.load(str(weights_path))
        # mx.load returns an array for a .npy file and a mapping for
        # safetensors; only the mapping shape is a servable adapter.
        if not isinstance(loaded, dict) or not loaded:
            raise MLXEngineError(f"adapter at {source} carries no weight mapping")
        self._apply_adapter(loaded)


__all__ = [
    "ADAPTER_CONFIG",
    "ADAPTER_WEIGHTS",
    "PROVENANCE",
    "MLXEngine",
    "MLXEngineConfig",
    "MLXEngineError",
    "Rollout",
    "TrainingRow",
]
