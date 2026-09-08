"""A gated-delta recurrence whose backward fits in unified memory.

mlx-lm's GatedDeltaNet runs its recurrence as a Metal kernel in eval mode and,
because that kernel has no vjp, as a loop of plain ops in training mode. The
loop is differentiable, but MLX keeps every intermediate of every step for the
backward — three to four state-sized tensors per token per crossed layer,
about 10.5 MB a token on Qwen3.8-27B, linear in length. A 700-token row
crossing six such layers (``lora_layers: 8``) would need about 60 GB.

This is the same loop, cut into chunks that are ``mx.checkpoint``-ed: the
backward keeps only the recurrent state at each chunk boundary and recomputes
the steps inside a chunk when it needs them. The per-step arithmetic is
mlx-lm's own compiled step, so the numbers are identical to the letter; only
what is remembered changes. It buys less than the arithmetic promises — MLX
schedules the recomputation of many chunks before it frees any of them, so
the peak still grows with length, at roughly half the rate — but that half is
what makes eight layers fit: 25 GB measured where 60 GB was projected. A
hand-written backward that keeps chunk-boundary states only would need about
a hundredth of either.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import mlx.core as mx

_Recurrence = Callable[..., tuple[mx.array, mx.array]]


def _chunk_runner(step: Callable[..., tuple[mx.array, mx.array]], *, masked: bool) -> _Recurrence:
    """One checkpointed span of the recurrence, ``chunk`` tokens long."""

    def run(q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array, *rest: mx.array):
        mask = rest[0] if masked else None
        ys = []
        for t in range(q.shape[1]):
            y, state = step(
                q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state, None if mask is None else mask[:, t]
            )
            ys.append(y)
        return mx.stack(ys, axis=1), state

    # mx.checkpoint takes arrays only, so the mask travels positionally and
    # its absence is a separate function rather than a None argument.
    return mx.checkpoint(run)


def checkpointed_recurrence(step: Callable[..., tuple[mx.array, mx.array]], chunk: int) -> _Recurrence:
    """mlx-lm's ``gated_delta_ops``, remembering only every ``chunk``-th state.

    ``step`` is mlx-lm's single-token step (``_gated_delta_step_ops``); the
    signature and semantics of the returned function are those of
    ``mlx_lm.models.gated_delta.gated_delta_ops``: ``(q, k, v, g, beta,
    state=None, mask=None) -> (y, state)``.
    """
    if chunk <= 0:
        raise ValueError("the recurrence chunk must be a positive number of tokens")
    unmasked = _chunk_runner(step, masked=False)
    masked = _chunk_runner(step, masked=True)

    def gated_delta_ops(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        batch, length, key_heads, key_dim = q.shape
        value_heads, value_dim = v.shape[-2:]
        if state is None:
            state = mx.zeros((batch, value_heads, value_dim, key_dim), dtype=mx.float32)
        if (repeat := value_heads // key_heads) > 1:
            q = mx.repeat(q, repeat, -2)
            k = mx.repeat(k, repeat, -2)
        ys = []
        for start in range(0, length, chunk):
            stop = min(start + chunk, length)
            span = (q[:, start:stop], k[:, start:stop], v[:, start:stop], g[:, start:stop], beta[:, start:stop], state)
            if mask is None:
                y, state = unmasked(*span)
            else:
                y, state = masked(*span, mask[:, start:stop])
            ys.append(y)
        return mx.concatenate(ys, axis=1), state

    return gated_delta_ops


@contextmanager
def checkpointed_gated_delta(chunk: int) -> Iterator[None]:
    """Route mlx-lm's training-mode recurrence through the checkpointed loop.

    ``gated_delta_update`` looks ``gated_delta_ops`` up in its module at call
    time, so rebinding that name is enough, and undoing it on exit leaves
    mlx-lm exactly as found. A chunk of 0 changes nothing. An mlx-lm without
    this module has no such layer to route, so there is nothing to do.
    """
    if chunk <= 0:
        yield
        return
    try:
        from mlx_lm.models import gated_delta as module
    except ImportError:
        yield
        return
    original = module.gated_delta_ops
    module.gated_delta_ops = checkpointed_recurrence(module._gated_delta_step_ops, chunk)
    try:
        yield
    finally:
        module.gated_delta_ops = original
