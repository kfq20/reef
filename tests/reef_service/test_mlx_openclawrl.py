"""The MLX OPD kernel against the torch original, on identical inputs.

``reef/train/mlx_backend/objective.py`` is a port, so the only test that
means anything is a differential one: feed both implementations the same
tensors and compare. Expectations written by hand would just re-encode
whatever the port happens to do.

Skips unless both frameworks are importable — torch for the reference,
MLX for the port.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="the torch reference is the thing under comparison")
mx = pytest.importorskip("mlx.core", reason="the MLX port needs the optional mlx extra")

from reef.train.mlx_backend.objective import (
    candidate_log_probs,
    masked_softmax,
    opd_one_sample,
    policy_loss,
)

pytestmark = pytest.mark.integration

ROWS = 6
STUDENT_K = 5
TEACHER_K = 5
VOCAB = 64
EPS_LO = 0.2
EPS_HI = 0.28


@pytest.fixture
def reference(monkeypatch):
    """The torch objective with only its device glue stubbed.

    Mirrors ``tests/test_openclawrl.py``'s fixture: Megatron is unavailable
    off a GPU host, so the tensor-parallel group is stubbed while the tensor
    kernel and the real PPO helpers stay intact.
    """
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_tensor_model_parallel_group=lambda: None)
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)

    loss_module = ModuleType("slime.backends.megatron_utils.loss")
    loss_module.get_log_probs_and_entropy = lambda logits, **kwargs: (torch.empty(0), {"log_probs": [logits]})
    loss_module.get_responses = lambda logits, **_: iter([(logits, torch.empty(0, dtype=torch.long))])
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.loss", loss_module)

    name = "recipes.openclawrl.slime.objective"
    previous = sys.modules.pop(name, None)
    try:
        yield importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)
        if previous is not None:
            sys.modules[name] = previous


def inputs(seed: int, *, disjoint: bool):
    """One sample's tensors, as numpy, so both frameworks see the same bits."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(ROWS, VOCAB, generator=generator, dtype=torch.float32)
    if disjoint:
        # Student and teacher propose overlapping but different candidates,
        # which is the case the intersection mask exists for.
        student = torch.stack([torch.randperm(VOCAB, generator=generator)[:STUDENT_K] for _ in range(ROWS)])
        teacher = torch.stack([torch.randperm(VOCAB, generator=generator)[:TEACHER_K] for _ in range(ROWS)])
        # Guarantee at least one shared candidate per row, and one row with
        # none at all, so row_valid gets exercised in both directions.
        teacher[0, 0] = student[0, 0]
        teacher[1, 2] = student[1, 4]
    else:
        student = torch.stack([torch.randperm(VOCAB, generator=generator)[:STUDENT_K] for _ in range(ROWS)])
        teacher = student.clone()
    captured = torch.randn(ROWS, STUDENT_K, generator=generator, dtype=torch.float32) - 1.0
    teacher_lp = torch.randn(ROWS, TEACHER_K, generator=generator, dtype=torch.float32) - 1.0
    return logits, student, teacher, captured, teacher_lp


def run_reference(module, logits, student, teacher, captured, teacher_lp, diff_clip):
    outputs = module._opd_one_sample(
        logits,
        student_indices=student,
        student_captured_lp=captured,
        teacher_indices=teacher,
        teacher_lp=teacher_lp,
        eps_lo=EPS_LO,
        eps_hi=EPS_HI,
        diff_clip=diff_clip,
        tp_group=None,
    )
    return [value.detach().numpy() for value in outputs[:4]], outputs[4].numpy()


def run_port(logits, student, teacher, captured, teacher_lp, diff_clip):
    current = candidate_log_probs(mx.array(logits.numpy()), mx.array(student.numpy()))
    result = opd_one_sample(
        current,
        student_indices=mx.array(student.numpy()),
        student_captured_log_probs=mx.array(captured.numpy()),
        teacher_indices=mx.array(teacher.numpy()),
        teacher_log_probs=mx.array(teacher_lp.numpy()),
        eps_lo=EPS_LO,
        eps_hi=EPS_HI,
        diff_clip=diff_clip,
    )
    values = [result.per_token_pg, result.per_token_clip, result.per_token_diff, result.per_token_capture_drift]
    mx.eval(*values, result.row_valid)
    import numpy

    return [numpy.array(value) for value in values], numpy.array(result.row_valid)


NAMES = ("per_token_pg", "per_token_clip", "per_token_diff", "per_token_capture_drift")


@pytest.mark.parametrize("disjoint", [False, True], ids=["identical-candidates", "partial-overlap"])
@pytest.mark.parametrize("diff_clip", [None, 0.5], ids=["unclipped", "clipped"])
def test_the_port_reproduces_the_torch_kernel(reference, disjoint, diff_clip) -> None:
    import numpy

    tensors = inputs(seed=7 if disjoint else 3, disjoint=disjoint)
    expected, expected_valid = run_reference(reference, *tensors, diff_clip)
    actual, actual_valid = run_port(*tensors, diff_clip)

    assert numpy.array_equal(expected_valid, actual_valid)
    for name, want, got in zip(NAMES, expected, actual, strict=True):
        numpy.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, err_msg=name)


def test_a_row_with_no_shared_candidate_contributes_nothing(reference) -> None:
    import numpy

    logits, student, teacher, captured, teacher_lp = inputs(seed=11, disjoint=True)
    # Force row 3 to share nothing: its teacher candidates leave the student's set.
    teacher[3] = torch.tensor([(int(student[3].max().item()) + 1 + offset) % VOCAB for offset in range(TEACHER_K)])
    for candidate in student[3].tolist():
        assert candidate not in teacher[3].tolist()

    expected, expected_valid = run_reference(reference, logits, student, teacher, captured, teacher_lp, None)
    actual, actual_valid = run_port(logits, student, teacher, captured, teacher_lp, None)

    assert not bool(actual_valid[3])
    assert expected_valid[3] == actual_valid[3]
    for name, want, got in zip(NAMES, expected, actual, strict=True):
        assert got[3] == 0.0, name
        numpy.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, err_msg=name)


def test_the_gradient_pushes_the_policy_toward_the_teacher(reference) -> None:
    # The point of the term: raise the log-prob of candidates the teacher
    # likes more than the current policy does, lower the ones it likes less.
    import mlx.nn as nn  # noqa: F401  (imported for parity with engine usage)

    logits, student, teacher, captured, teacher_lp = inputs(seed=5, disjoint=False)

    def loss(raw: mx.array) -> mx.array:
        current = candidate_log_probs(raw, mx.array(student.numpy()))
        result = opd_one_sample(
            current,
            student_indices=mx.array(student.numpy()),
            student_captured_log_probs=mx.array(captured.numpy()),
            teacher_indices=mx.array(teacher.numpy()),
            teacher_log_probs=mx.array(teacher_lp.numpy()),
            eps_lo=EPS_LO,
            eps_hi=EPS_HI,
            diff_clip=None,
        )
        return mx.sum(result.per_token_pg)

    raw = mx.array(logits.numpy())
    gradient = mx.grad(loss)(raw)
    mx.eval(gradient)

    # Descending the loss raises the log-prob of a candidate the teacher
    # prefers, so its logit gradient must be negative.
    teacher_gap = teacher_lp.numpy() - candidate_log_probs(raw, mx.array(student.numpy()))
    mx.eval(teacher_gap)
    import numpy

    gaps = numpy.array(teacher_gap)
    grads = numpy.array(gradient)
    preferred = numpy.unravel_index(numpy.argmax(gaps), gaps.shape)
    row = int(preferred[0])
    token = int(student[row, int(preferred[1])].item())
    assert grads[row, token] < 0.0


def test_the_masked_softmax_ignores_entries_outside_the_subset(reference) -> None:
    import numpy

    values = mx.array([[1.0, 2.0, 3.0, 4.0]])
    mask = mx.array([[True, False, True, False]])
    weights = masked_softmax(values, mask)
    mx.eval(weights)
    got = numpy.array(weights)[0]

    assert got[1] == 0.0 and got[3] == 0.0
    numpy.testing.assert_allclose(got.sum(), 1.0, rtol=1e-6)
    # Same as a softmax over just the unmasked pair.
    expected = torch.softmax(torch.tensor([1.0, 3.0]), dim=-1).numpy()
    numpy.testing.assert_allclose([got[0], got[2]], expected, rtol=1e-6)


def test_policy_loss_matches_slime(reference) -> None:
    import numpy
    from slime.utils.ppo_utils import compute_policy_loss

    ppo_kl = torch.linspace(-1.5, 1.5, 24).reshape(4, 6)
    advantages = torch.linspace(-2.0, 2.0, 24).reshape(4, 6)
    want_loss, want_clip = compute_policy_loss(ppo_kl, advantages, EPS_LO, EPS_HI)
    got_loss, got_clip = policy_loss(mx.array(ppo_kl.numpy()), mx.array(advantages.numpy()), EPS_LO, EPS_HI)
    mx.eval(got_loss, got_clip)

    numpy.testing.assert_allclose(numpy.array(got_loss), want_loss.numpy(), rtol=1e-5, atol=1e-6)
    numpy.testing.assert_allclose(numpy.array(got_clip), want_clip.numpy(), rtol=1e-5, atol=1e-6)
