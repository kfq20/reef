"""A gated-delta recurrence with a backward that fits in unified memory.

mlx-lm's GatedDeltaNet runs its recurrence as a Metal kernel in eval mode and,
because that kernel has no vjp, as a loop of plain ops in training mode. The
loop is differentiable, but MLX keeps every intermediate of every step for the
backward — three to four state-sized tensors per token per crossed layer,
about 10.5 MB a token on Qwen3.8-27B, linear in length. A 700-token row
crossing six such layers (``lora_layers: 8``) needs about 60 GB that way.

This module gives the recurrence the backward it should have had. The forward
runs mlx-lm's own kernel, one span of ``chunk`` tokens at a time, and each
span is an ``mx.custom_function`` whose vjp is written by hand: it recomputes
the span's states from the state at its boundary and walks the steps in
reverse, so what the backward keeps is one state per span plus one span's
worth of states while it works. Measured on the recurrence alone at
Qwen3.8-27B's head shape, 700 tokens: 14.6 MB per token and 5.9 s for
mlx-lm's loop, 1.0 MB per token and 1.0 s here, with gradients that agree to
about 1e-6 in float32.

Two things had to be said to MLX for that to hold. The span's vjp is one
compiled graph, so its arithmetic is fused; and it carries explicit
scheduling dependencies (``mx.depends``), because MLX evaluates a graph by
walking its outputs depth first, which on its own recomputes every span's
states before running any span's reverse scan — and then the backward holds
a state per token after all, whatever the span size.

The maths is the ops loop's, step for step::

    S' = S ⊙ decay              decay = g, per head or per key column
    kv = S' k                   [Dv]
    δ  = (v − kv) · β           [Dv]
    S''= S' + δ ⊗ k
    y  = S'' q                  [Dv]

and where a position is masked out the state passes through unchanged and
the output is zero, which is what the kernel does with a mask.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import mlx.core as mx

_Recurrence = Callable[..., tuple[mx.array, mx.array]]


def _kernel_usable(key_dim: int) -> bool:
    return mx.metal.is_available() and mx.default_device() == mx.gpu and key_dim >= 32 and key_dim % 32 == 0


def _decay(g: mx.array) -> mx.array:
    """``g`` broadcast against a ``[B, H, Dv, Dk]`` state."""
    if g.ndim == 2:  # [B, H]: one decay per head
        return g[..., None, None]
    if g.ndim == 3:  # [B, H, Dk]: one decay per key column
        return g[..., None, :]
    raise ValueError(f"unsupported gating shape {g.shape}")


@mx.compile
def _forward_step(q, k, v, g, beta, state):
    """One step of the recurrence, all in the state's precision."""
    decayed = state * _decay(g)
    kv = (decayed * k[..., None, :]).sum(axis=-1)
    delta = (v - kv) * beta[..., None]
    updated = decayed + delta[..., None] * k[..., None, :]
    y = (updated * q[..., None, :]).sum(axis=-1)
    return y, updated


@mx.compile
def _reverse_step(q, k, v, g, beta, previous, dy, dstate):
    """The vjp of one step, from the state it started from.

    ``dstate`` is the cotangent of the state the step produced, ``dy`` that
    of its output. Returns the cotangents of the step's inputs and of the
    state it started from.
    """
    decay = _decay(g)
    decayed = previous * decay
    kv = (decayed * k[..., None, :]).sum(axis=-1)
    delta = (v - kv) * beta[..., None]
    updated = decayed + delta[..., None] * k[..., None, :]

    # y = S'' q
    d_updated = dstate + dy[..., None] * q[..., None, :]
    dq = (dy[..., None] * updated).sum(axis=-2)
    # S'' = S' + δ ⊗ k
    d_delta = (d_updated * k[..., None, :]).sum(axis=-1)
    dk = (d_updated * delta[..., None]).sum(axis=-2)
    # δ = (v − kv) β
    dv = d_delta * beta[..., None]
    d_kv = -dv
    dbeta = (d_delta * (v - kv)).sum(axis=-1)
    # kv = S' k
    d_decayed = d_updated + d_kv[..., None] * k[..., None, :]
    dk = dk + (d_kv[..., None] * decayed).sum(axis=-2)
    # S' = S ⊙ decay
    d_previous = d_decayed * decay
    d_decay = d_decayed * previous
    dg = d_decay.sum(axis=(-2, -1)) if g.ndim == 2 else d_decay.sum(axis=-2)
    return dq, dk, dv, dg, dbeta, d_previous


def _span_backward(masked: bool):
    """The vjp of one span as a single compiled graph.

    Compiled so that MLX runs the span's recomputation and reverse scan as
    one unit whose intermediates are released as it goes. Left as ordinary
    lazy ops, the scheduler builds every span's recomputation before it
    frees any of them, and the backward keeps per-token states after all —
    measured, that lands within a third of mlx-lm's loop, and forcing an
    evaluation per span makes it worse, not better.
    """

    def backward(q, k, v, g, beta, state, dy, dstate, *rest):
        mask = rest[0] if masked else None
        length = q.shape[1]

        # Not a data dependency but a scheduling one: this span's
        # recomputation may not begin before the cotangent from the span
        # after it exists. Without it MLX's evaluation, which walks outputs
        # depth first, recomputes every span's states before it runs any
        # span's reverse scan, and the backward holds a state per token
        # after all.
        state = mx.depends([state], [dstate])[0]

        # The states each step started from, recomputed from the span's
        # boundary: this is the only per-token memory the backward holds,
        # and only for one span at a time.
        starts = []
        for t in range(length):
            starts.append(state)
            _, updated = _forward_step(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state)
            state = updated if mask is None else mx.where(mask[:, t][:, None, None, None], updated, state)

        dq, dk, dv, dg, dbeta = [], [], [], [], []
        for t in reversed(range(length)):
            step = _reverse_step(
                q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], starts[t], dy[:, t].astype(mx.float32), dstate
            )
            dq_t, dk_t, dv_t, dg_t, dbeta_t, d_previous = step
            if mask is None:
                dstate = d_previous
            else:
                keep = mask[:, t]

                def zero(value, keep=keep):
                    return mx.where(keep.reshape((-1,) + (1,) * (value.ndim - 1)), value, mx.zeros_like(value))

                dq_t, dk_t, dv_t, dg_t, dbeta_t = map(zero, (dq_t, dk_t, dv_t, dg_t, dbeta_t))
                # A masked step passes the state through, and its cotangent with it.
                dstate = mx.where(keep[:, None, None, None], d_previous, dstate)
            # The same scheduling dependency within the step: the small
            # cotangents are produced before the chain moves on, so the
            # state-sized intermediates they read are released with the step
            # instead of living until the stacked outputs are assembled.
            dstate = mx.depends([dstate], [dq_t, dk_t, dv_t, dg_t, dbeta_t])[0]
            dq.append(dq_t)
            dk.append(dk_t)
            dv.append(dv_t)
            dg.append(dg_t)
            dbeta.append(dbeta_t)

        return (
            mx.stack(dq[::-1], axis=1).astype(q.dtype),
            mx.stack(dk[::-1], axis=1).astype(k.dtype),
            mx.stack(dv[::-1], axis=1).astype(v.dtype),
            mx.stack(dg[::-1], axis=1).astype(g.dtype),
            mx.stack(dbeta[::-1], axis=1).astype(beta.dtype),
            dstate,
        )

    return mx.compile(backward)


_backward_unmasked = _span_backward(masked=False)
_backward_masked = _span_backward(masked=True)


def _span(mask: mx.array | None):
    """The custom function for one span; ``mask`` is ``[B, C]`` or None."""

    @mx.custom_function
    def forward(q, k, v, g, beta, state):
        if _kernel_usable(q.shape[-1]):
            from mlx_lm.models.gated_delta import gated_delta_kernel

            return gated_delta_kernel(q, k, v, g, beta, state, mask)
        ys = []
        for t in range(q.shape[1]):
            y, updated = _forward_step(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state)
            if mask is not None:
                keep = mask[:, t]
                state = mx.where(keep[:, None, None, None], updated, state)
                y = mx.where(keep[:, None, None], y, mx.zeros_like(y))
            else:
                state = updated
            ys.append(y.astype(q.dtype))
        return mx.stack(ys, axis=1), state

    @forward.vjp
    def backward(primals, cotangents, outputs):
        dy, dstate = cotangents
        if mask is None:
            return _backward_unmasked(*primals, dy, dstate)
        return _backward_masked(*primals, dy, dstate, mask)

    return forward


def chunked_recurrence(chunk: int) -> _Recurrence:
    """``mlx_lm.models.gated_delta.gated_delta_ops``, with a hand-written backward.

    Same signature and semantics: ``(q, k, v, g, beta, state=None, mask=None)
    -> (y, state)``, with ``q, k`` of ``[B, T, Hk, Dk]``, ``v`` of ``[B, T,
    Hv, Dv]``, ``g`` of ``[B, T, Hv]`` or ``[B, T, Hv, Dk]``, ``beta`` of
    ``[B, T, Hv]`` and the state ``[B, Hv, Dv, Dk]``.
    """
    if chunk <= 0:
        raise ValueError("the recurrence chunk must be a positive number of tokens")

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
            # Grouped queries and keys are shared across value heads; repeating
            # them here keeps the span's maths one head to one head, and
            # mx.repeat's own vjp sums the gradient back over the group.
            q = mx.repeat(q, repeat, -2)
            k = mx.repeat(k, repeat, -2)
        ys = []
        for start in range(0, length, chunk):
            stop = min(start + chunk, length)
            span = _span(None if mask is None else mask[:, start:stop])
            y, state = span(
                q[:, start:stop], k[:, start:stop], v[:, start:stop], g[:, start:stop], beta[:, start:stop], state
            )
            ys.append(y)
        return mx.concatenate(ys, axis=1), state

    return gated_delta_ops


@contextmanager
def chunked_gated_delta(chunk: int) -> Iterator[None]:
    """Route mlx-lm's training-mode recurrence through :func:`chunked_recurrence`.

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
    module.gated_delta_ops = chunked_recurrence(chunk)
    try:
        yield
    finally:
        module.gated_delta_ops = original
