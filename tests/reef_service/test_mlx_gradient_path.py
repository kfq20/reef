"""A gradient must be able to cross a GatedDeltaNet layer.

mlx-lm's GatedDeltaNet picks a Metal kernel with no vjp in eval mode and a
differentiable loop of plain ops in training mode. ``load`` leaves the model in
eval mode, so unless the engine switches the gradient's path into training
mode for the backward pass, LoRA on a hybrid model such as Qwen3.8 is confined
to the layers above the last GatedDeltaNet. This pins that the switch happens,
that it happens only on the path (the frozen prefix keeps the kernel), and
that it is undone afterwards so serving stays on the kernel.

Needs real MLX, but no model download: the hybrid is a synthetic ``qwen3_5``
with a handful of tiny layers, so this runs on any Apple Silicon machine with
the optional extra installed.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core", reason="the MLX engine needs the optional mlx extra")
pytest.importorskip("mlx_lm", reason="the MLX engine needs the optional mlx extra")

import mlx.nn as nn

from reef.train.mlx_backend import engine as engine_module
from reef.train.mlx_backend.engine import MLXEngine, MLXEngineConfig, TrainingRow, _gradient_path_layers

LAYERS = 8
#: Every fourth layer is full attention, as in the real model; the rest are
#: GatedDeltaNet. Layers 3 and 7 are attention here.
INTERVAL = 4


def tiny_hybrid() -> nn.Module:
    from mlx_lm.models import qwen3_5

    text = {
        "hidden_size": 64,
        "num_hidden_layers": LAYERS,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "full_attention_interval": INTERVAL,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "attn_output_gate": True,
        "partial_rotary_factor": 0.25,
        "max_position_embeddings": 4096,
    }
    return qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=text))


class _Tokenizer:
    pad_token_id = 0


def hybrid_engine(monkeypatch, *, lora_layers: int) -> MLXEngine:
    monkeypatch.setattr(engine_module, "load", lambda path: (tiny_hybrid(), _Tokenizer()))
    return MLXEngine(
        MLXEngineConfig(
            model_path="synthetic-qwen3_5",
            lora_layers=lora_layers,
            lora_rank=4,
            micro_batch_size=2,
            # Both kinds of layer are adapted, so the gradient has to reach
            # into a GatedDeltaNet's own projections, not just past it.
            lora_keys=("self_attn.q_proj", "linear_attn.in_proj_qkv", "mlp.down_proj"),
            seed=0,
        )
    )


def rows() -> list[TrainingRow]:
    return [
        TrainingRow(
            tokens=tuple((index * 7 + position) % 200 + 1 for position in range(24)),
            loss_mask=(1,) * 16,
            rollout_log_probs=tuple(-0.5 - 0.01 * position for position in range(16)),
            advantages=tuple(1.0 if index % 2 else -0.7 for _ in range(16)),
        )
        for index in range(2)
    ]


def layer_index(name: str) -> int:
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1])


def test_the_gradient_path_starts_at_the_lowest_adapted_layer_and_nothing_below(monkeypatch) -> None:
    engine = hybrid_engine(monkeypatch, lora_layers=6)
    try:
        layers = engine._run(lambda: list(engine._holder.model.layers))
        assert engine._gradient_path == layers[LAYERS - 6 :]
    finally:
        engine.close()


def test_a_model_with_no_adapter_has_no_gradient_path() -> None:
    model = tiny_hybrid()
    model.freeze()
    assert _gradient_path_layers(model) == []


def test_an_unrecognised_architecture_is_its_own_gradient_path() -> None:
    class Opaque(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

    model = Opaque()
    assert _gradient_path_layers(model) == [model]


def test_the_backward_crosses_gated_delta_layers_and_serving_stays_on_the_kernel(monkeypatch) -> None:
    engine = hybrid_engine(monkeypatch, lora_layers=6)
    try:
        # The path spans layers 2..7: GatedDeltaNet at 2, 4, 5, 6 and full
        # attention at 3 and 7. In eval mode this is exactly the shape that
        # fails with "[Primitive::vjp] Not implemented for CustomKernel".
        _, grads = engine._run(lambda: engine._micro_batch_gradients(engine._pack(rows())))
        moved = {layer_index(name) for name, grad in grads.items() if float(mx.abs(grad).sum()) > 0}
        assert moved == set(range(LAYERS - 6, LAYERS))

        # Every layer is back on the kernel, ready to serve: the prefix never
        # left it, and the path was restored when the backward finished.
        modes = engine._run(lambda: [layer.training for layer in engine._holder.model.layers])
        assert modes == [False] * LAYERS
    finally:
        engine.close()


def test_the_prefix_never_leaves_eval_mode_during_the_backward(monkeypatch) -> None:
    engine = hybrid_engine(monkeypatch, lora_layers=6)
    try:
        seen: list[list[bool]] = []
        original = engine_module._token_log_probs

        def observing(model, sequences, mask):
            seen.append([layer.training for layer in engine._holder.model.layers])
            return original(model, sequences, mask)

        monkeypatch.setattr(engine_module, "_token_log_probs", observing)
        engine._run(lambda: engine._micro_batch_gradients(engine._pack(rows())))
        assert seen and all(modes == [False] * (LAYERS - 6) + [True] * 6 for modes in seen)
    finally:
        engine.close()


def test_without_the_switch_the_kernel_refuses_the_backward(monkeypatch) -> None:
    # The reason the switch exists, kept next to it so that if mlx-lm ever
    # gives the kernel a vjp, this fails and the mechanism can be retired.
    engine = hybrid_engine(monkeypatch, lora_layers=6)
    try:
        monkeypatch.setattr(engine, "_gradient_path", [])
        with pytest.raises(ValueError, match="vjp"):
            engine._run(lambda: engine._micro_batch_gradients(engine._pack(rows())))
    finally:
        engine.close()


# ------------------------------------------------- hand-written recurrence backward


def recurrence_inputs(length: int, *, vector_gating: bool):
    from mlx_lm.models import gated_delta

    mx.random.seed(0)
    batch, key_heads, value_heads, key_dim, value_dim = 2, 2, 4, 32, 32
    q = mx.random.normal((batch, length, key_heads, key_dim))
    k = mx.random.normal((batch, length, key_heads, key_dim))
    v = mx.random.normal((batch, length, value_heads, value_dim))
    gating_shape = (batch, length, value_heads, key_dim) if vector_gating else (batch, length, value_heads)
    g = mx.random.uniform(shape=gating_shape)
    beta = mx.random.uniform(shape=(batch, length, value_heads))
    state = mx.random.normal((batch, value_heads, value_dim, key_dim)) * 0.1
    mask = mx.array([[1] * length, [1] * (length - 17) + [0] * 17]).astype(mx.bool_)
    return gated_delta, (q, k, v, g, beta, state), mask


def relative_deviation(expected: mx.array, actual: mx.array) -> float:
    return float(mx.max(mx.abs(expected - actual)) / mx.maximum(mx.max(mx.abs(expected)), mx.array(1e-12)))


@pytest.mark.parametrize("vector_gating", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_the_hand_written_backward_agrees_with_autodiff_through_mlx_lm_s_loop(
    masked: bool, vector_gating: bool
) -> None:
    from reef.train.mlx_backend.gated_delta import chunked_recurrence

    # 77 tokens: two full spans of 32 and a ragged tail; grouped queries
    # (two key heads to four value heads); a starting state that is not
    # zero, so its gradient is exercised; and a mask that freezes one row's
    # state partway through the last span. The loss reads both outputs.
    gated_delta, inputs, mask = recurrence_inputs(77, vector_gating=vector_gating)
    mask = mask if masked else None

    def loss_of(recurrence):
        def loss(q, k, v, g, beta, state):
            y, final = recurrence(q, k, v, g, beta, state, mask)
            # The kernel writes zeros at masked positions and the ops loop
            # does not; the outputs there are padding, so weight them out.
            weight = 1.0 if mask is None else mask[..., None, None].astype(y.dtype)
            return ((y * weight) ** 2).sum() + (final**2).sum()

        return mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5))(*inputs)

    reference_loss, reference_grads = loss_of(gated_delta.gated_delta_ops)
    loss, grads = loss_of(chunked_recurrence(32))
    mx.eval(reference_loss, reference_grads, loss, grads)

    # float32 throughout; the residue is the kernel's summation order.
    assert abs(float(loss) - float(reference_loss)) / abs(float(reference_loss)) < 1e-5
    for expected, actual in zip(reference_grads, grads, strict=True):
        assert relative_deviation(expected, actual) < 1e-4


def test_the_backward_keeps_one_span_of_states_rather_than_every_token() -> None:
    from mlx_lm.models import gated_delta

    from reef.train.mlx_backend.gated_delta import chunked_recurrence

    # The whole point. Measured on the recurrence alone, so the number is the
    # recurrence's: mlx-lm's loop keeps several states per token, the
    # hand-written backward keeps a span's worth in total. A generous bound,
    # so it fails only if the design regresses, not on allocator noise.
    key_heads, value_heads, key_dim, value_dim, length = 4, 8, 64, 64, 256
    state_bytes = value_heads * value_dim * key_dim * 4

    def peak_of(recurrence) -> int:
        mx.random.seed(0)
        q = mx.random.normal((1, length, key_heads, key_dim))
        k = mx.random.normal((1, length, key_heads, key_dim))
        v = mx.random.normal((1, length, value_heads, value_dim))
        g = mx.random.uniform(shape=(1, length, value_heads))
        beta = mx.random.uniform(shape=(1, length, value_heads))
        mx.eval(q, k, v, g, beta)

        def loss(q, k, v, g, beta):
            y, _ = recurrence(q, k, v, g, beta, None, None)
            return (y**2).sum()

        mx.reset_peak_memory()
        base = mx.get_active_memory()
        value, grads = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
        mx.eval(value, grads)
        return mx.get_peak_memory() - base

    loop = peak_of(gated_delta.gated_delta_ops)
    manual = peak_of(chunked_recurrence(32))
    assert loop > length * state_bytes  # the loop really does keep per-token states
    assert manual < loop / 4


def test_the_recurrence_is_rerouted_only_for_the_span_of_a_backward(monkeypatch) -> None:
    from mlx_lm.models import gated_delta

    from reef.train.mlx_backend import gated_delta as reef_gated_delta

    original = gated_delta.gated_delta_ops
    engine = hybrid_engine(monkeypatch, lora_layers=6)
    try:
        seen: list[bool] = []
        scoring = engine_module._token_log_probs

        def observing(model, sequences, mask):
            seen.append(gated_delta.gated_delta_ops is not original)
            return scoring(model, sequences, mask)

        monkeypatch.setattr(engine_module, "_token_log_probs", observing)
        engine._run(lambda: engine._micro_batch_gradients(engine._pack(rows())))
        assert seen == [True]
        assert gated_delta.gated_delta_ops is original

        with reef_gated_delta.chunked_gated_delta(0):
            assert gated_delta.gated_delta_ops is original
    finally:
        engine.close()


def test_a_negative_recurrence_chunk_is_refused() -> None:
    with pytest.raises(ValueError, match="recurrence_chunk_size"):
        MLXEngineConfig(model_path="x", recurrence_chunk_size=-1)
