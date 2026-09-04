"""Contract tests for the MLX runtime that need no Apple hardware.

Everything MLX-specific lives behind the engine, so the runtime's contract
with Reef — preparation, candidate export, activation, rejection, rollback,
and the refusal paths — is testable against a fake engine on any machine.
The parts that genuinely need Metal (real generation, a real optimizer step,
memory ceilings) belong to the gated Apple Silicon qualification instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reef.runtime.base import RuntimeContractError
from reef.runtime.candidates import ModelCandidate
from reef.runtime.registry import RuntimeRegistry
from reef.train.algos.base import StepPreparer
from reef.train.algos.base import register_step_preparer
from reef.train.algos.signals import StepScheduling, StepSignal
from reef.train.evaluation.contracts import EvaluationResult, SelectionDecision
from reef.train.mlx_backend.runtime import MLXRuntime
from reef.train.types import PolicyBatch, PolicySample


class FakeEngineConfig:
    model_path = "fake/model"
    lora_layers = 2
    lora_rank = 4
    learning_rate = 1e-5


class FakeEngine:
    """Stand in for MLXEngine with plain Python state.

    ``moved`` decides whether a training step actually changes weights, which
    is what the refusal test needs to drive.
    """

    def __init__(self, *, moved: bool = True) -> None:
        self.config = FakeEngineConfig()
        self.weights = {"layer.lora_a": 0.0, "layer.lora_b": 0.0}
        self.moved = moved
        self.publications = 0
        self.saved: list[Path] = []
        self.loaded: list[Path] = []
        self.trained_rows: list[Any] = []
        self.distillation_rows: list[Any] | None = None

    def adapter_snapshot(self) -> dict[str, float]:
        return dict(self.weights)

    def apply_adapter(self, snapshot) -> None:
        self.weights = dict(snapshot)

    def adapter_delta(self, before, after) -> tuple[float, int]:
        changed = sum(1 for name, value in after.items() if before.get(name) != value)
        return (float(changed), changed)

    def train_step(self, rows) -> dict[str, Any]:
        self.trained_rows = list(rows)
        if self.moved:
            self.weights = {name: value + 1.0 for name, value in self.weights.items()}
        return {"loss": -1.0, "rows": len(rows)}

    def openclawrl_step(self, rows, **settings) -> dict[str, Any]:
        self.distillation_rows = list(rows)
        if self.moved:
            self.weights = {name: value + 1.0 for name, value in self.weights.items()}
        return {"loss": -1.0, "rows": len(rows), **settings}

    def base_log_probs(self, rows) -> list[list[float]]:
        return [[-1.0] * len(row.loss_mask) for row in rows]

    def next_runtime_load_id(self) -> str:
        self.publications += 1
        return f"fake-{self.publications}"

    def save_adapter(self, destination: Path, *, provenance_extra=None) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "adapters.safetensors").write_bytes(b"weights")
        (destination / "reef_provenance.json").write_text(json.dumps(dict(provenance_extra or {})))
        self.saved.append(destination)
        return destination

    def load_adapter(self, source: Path) -> None:
        self.loaded.append(Path(source))


def sample(reward: float) -> PolicySample:
    return PolicySample(
        source_agent_record_id=f"rec-{reward}",
        tokens=(1, 2, 3, 4),
        loss_mask=(1, 1),
        rollout_log_probs=(-0.5, -0.25),
        reward=reward,
    )


@register_step_preparer
class _TwoAdvantagePreparer(StepPreparer):
    name = "mlx-test-preparer"

    def __call__(self, batch, state):
        return StepSignal(
            "train",
            "fake-family",
            {"steps": int(state.get("steps", 0)) + 1},
            {"prepared": True},
            (1.0, -1.0),
            StepScheduling(unit="sample", batch_size="actual"),
        )


@register_step_preparer
class _TttdFamilyPreparer(StepPreparer):
    name = "mlx-test-tttd"

    def __call__(self, batch, state):
        return StepSignal(
            "train",
            "tttd",
            {"steps": int(state.get("steps", 0)) + 1},
            {"prepared": True},
            (1.0, -1.0),
            StepScheduling(unit="sample", batch_size="actual"),
        )


@register_step_preparer
class _MultiEpochPreparer(StepPreparer):
    name = "mlx-test-multi-epoch"

    def __call__(self, batch, state):
        return StepSignal(
            "train",
            "tttd",
            {},
            {},
            (1.0, -1.0),
            StepScheduling(unit="sample", batch_size="actual", epochs=3),
        )


def build_runtime(tmp_path: Path, *, moved: bool = True, kl_coef: float = 0.0) -> MLXRuntime:
    return MLXRuntime(FakeEngine(moved=moved), checkpoint_dir=str(tmp_path / "ckpt"), kl_coef=kl_coef)


def batch() -> PolicyBatch:
    return PolicyBatch("batch-1", (sample(1.0), sample(0.0)))


@pytest.mark.unit
def test_boot_names_the_weights_that_answer_the_first_request(tmp_path: Path) -> None:
    # A durable training record must name the weights that produced it, and
    # the first rollout is served before anything has been published.
    runtime = build_runtime(tmp_path)
    assert runtime.serving_runtime_load_id() == "fake-1"


@pytest.mark.unit
def test_preparation_runs_the_recipe_preparer_in_process(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {"steps": 4}, 7)

    assert prepared.action == "train"
    assert prepared.next_algorithm_state == {"steps": 5}
    assert prepared.payload["advantages"] == (1.0, -1.0)
    assert prepared.payload["rollout_id"] == 7
    assert len(prepared.payload["samples"]) == 2


@pytest.mark.unit
def test_unsupported_scheduling_fails_before_training(tmp_path: Path) -> None:
    # A schedule this runtime cannot honour must be refused, not ignored:
    # silently training one epoch when three were asked for changes the
    # objective without saying so.
    runtime = build_runtime(tmp_path)
    with pytest.raises(RuntimeContractError, match="epochs=3"):
        runtime.prepare_training_step(batch(), "mlx-test-multi-epoch", {}, 0)


@pytest.mark.unit
def test_a_candidate_exports_without_changing_serving(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    engine = runtime.engine
    before = engine.adapter_snapshot()
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 3)

    candidate = runtime.train_candidate(prepared.payload)

    assert isinstance(candidate, ModelCandidate)
    assert Path(candidate.checkpoint_path).is_dir()
    assert candidate.training_metrics["adapter_tensors_changed"] == 2
    # Serving still holds the pre-step weights: the trained parameters exist
    # only in the export until Reef selects them.
    assert engine.adapter_snapshot() == before
    assert runtime.serving_runtime_load_id() == "fake-1"


@pytest.mark.unit
def test_activation_moves_serving_to_the_selected_candidate(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    engine = runtime.engine
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)
    candidate = runtime.train_candidate(prepared.payload)

    activated = runtime.activate_candidate(candidate)

    assert activated.candidate_id == candidate.candidate_id
    assert activated.runtime_load_id == "fake-2"
    assert runtime.serving_runtime_load_id() == "fake-2"
    assert engine.adapter_snapshot() == {"layer.lora_a": 1.0, "layer.lora_b": 1.0}


@pytest.mark.unit
def test_a_rejected_candidate_never_reaches_serving(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    engine = runtime.engine
    before = engine.adapter_snapshot()
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)
    candidate = runtime.train_candidate(prepared.payload)

    runtime.reject_candidate(
        candidate,
        SelectionDecision("reject", "test", "1", "not better", EvaluationResult("test", "1", {})),
    )

    assert engine.adapter_snapshot() == before
    assert runtime.serving_runtime_load_id() == "fake-1"
    # A rejected candidate is gone: activating it afterwards must fail loudly
    # rather than resurrect weights Reef declined.
    with pytest.raises(RuntimeContractError, match="no pending mlx candidate"):
        runtime.activate_candidate(candidate)


@pytest.mark.unit
def test_a_step_that_moves_nothing_is_refused(tmp_path: Path) -> None:
    # The failure mode this guards against: a trainer reports success, writes
    # an adapter identical to the base, and Reef publishes it as a trained
    # update. The step must fail instead.
    runtime = build_runtime(tmp_path, moved=False)
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)

    with pytest.raises(RuntimeContractError, match="left every adapter tensor unchanged"):
        runtime.train_candidate(prepared.payload)

    assert runtime.engine.saved == []


@pytest.mark.unit
def test_the_frozen_base_kl_term_shifts_advantages_per_token(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path, kl_coef=0.5)
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)

    runtime.train_candidate(prepared.payload)

    rows = runtime.engine.trained_rows
    assert len(rows) == 2
    # rollout log-probs are (-0.5, -0.25) and the fake base returns -1.0, so
    # the per-token differences are (0.5, 0.75) with mean 0.625; every
    # advantage moves by kl_coef * (mean - difference).
    assert rows[0].advantages == pytest.approx((1.0 + 0.5 * 0.125, 1.0 + 0.5 * -0.125))
    assert rows[1].advantages == pytest.approx((-1.0 + 0.5 * 0.125, -1.0 + 0.5 * -0.125))


@pytest.mark.unit
def test_rollback_loads_a_published_adapter(tmp_path: Path) -> None:
    from reef.artifact.artifact import Artifact

    adapter = tmp_path / "published"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"weights")
    runtime = build_runtime(tmp_path)

    restored = runtime.restore_checkpoint(Artifact.local(adapter))

    assert restored == "fake-2"
    assert runtime.engine.loaded == [adapter]


@pytest.mark.unit
def test_the_factory_requires_a_checkpoint_directory() -> None:
    with pytest.raises(RuntimeContractError, match="checkpoint_dir"):
        RuntimeRegistry().build({"type": "mlx"}, model_path="fake/model")


@pytest.mark.unit
def test_the_factory_refuses_stale_sample_training() -> None:
    # Nothing in this runtime corrects for a batch produced by older weights,
    # so admitting one would train on a ratio it cannot compute.
    with pytest.raises(RuntimeContractError, match="exact-version"):
        RuntimeRegistry().build(
            {"type": "mlx", "checkpoint_dir": "/tmp/reef-mlx-test", "max_staleness": 4},
            model_path="fake/model",
        )


@pytest.mark.unit
def test_the_factory_refuses_template_kwargs_of_the_wrong_shape() -> None:
    with pytest.raises(RuntimeContractError, match="chat_template_kwargs"):
        RuntimeRegistry().build(
            {
                "type": "mlx",
                "checkpoint_dir": "/tmp/reef-mlx-test",
                "chat_template_kwargs": "enable_thinking=false",
            },
            model_path="fake/model",
        )


@pytest.mark.unit
def test_an_unsupported_loss_family_is_refused(tmp_path: Path) -> None:
    # The runtime implements one objective; training a recipe's data under a
    # different loss than it asked for would be a silent substitution.
    runtime = build_runtime(tmp_path)
    with pytest.raises(RuntimeContractError, match="loss family 'fake-family'"):
        runtime.prepare_training_step(batch(), "mlx-test-preparer", {}, 0)


@pytest.mark.unit
def test_inference_stays_closed_between_activation_and_the_durable_commit(tmp_path: Path) -> None:
    # Reopening at activation would let a request freeze the old artifact head
    # and then be answered by the new weights — a runtime-load mismatch.
    runtime = build_runtime(tmp_path)
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)
    candidate = runtime.train_candidate(prepared.payload)

    runtime.activate_candidate(candidate)

    assert runtime.inference_admission_status["open"] is False
    # The engine holds the new weights, but Reef has not published them yet.
    assert runtime.serving_runtime_load_id() == "fake-2"
    assert runtime.current_runtime_load_id() == "fake-1"

    runtime.reconcile_training_job(0, committed_training_job_id=candidate.training_job_id)

    assert runtime.inference_admission_status["open"] is True
    assert runtime.current_runtime_load_id() == "fake-2"


@pytest.mark.unit
def test_a_failed_step_restores_the_weights_that_were_serving(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    engine = runtime.engine
    before = engine.adapter_snapshot()
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)

    def explode(rows):
        # Mutate first, then fail: the shape of a step that dies after the
        # optimizer has already touched the parameters.
        engine.weights = {name: value + 99.0 for name, value in engine.weights.items()}
        raise RuntimeError("metal exploded")

    engine.train_step = explode
    with pytest.raises(RuntimeError, match="metal exploded"):
        runtime.train_candidate(prepared.payload)

    assert engine.adapter_snapshot() == before
    assert runtime.inference_admission_status["open"] is True


@pytest.mark.unit
def test_the_frozen_base_pass_runs_with_inference_closed(tmp_path: Path) -> None:
    # Zeroing lora_b turns the live model into the bare base. A request
    # admitted during that window would be answered by the wrong weights.
    runtime = build_runtime(tmp_path, kl_coef=0.5)
    engine = runtime.engine
    observed: list[bool] = []

    def watching_base_log_probs(rows):
        observed.append(runtime.inference_admission_status["open"])
        return [[-1.0] * len(row.loss_mask) for row in rows]

    engine.base_log_probs = watching_base_log_probs
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)
    runtime.train_candidate(prepared.payload)

    assert observed == [False]


class _FakeRollout:
    """The engine's Rollout shape, without loading a model."""

    def __init__(self, *, topk=True):
        self.prompt_tokens = (11, 12, 13)
        self.output_tokens = (21, 22)
        self.rollout_log_probs = (-0.5, -0.25)
        self.text = "an answer"
        self.finish_reason = "stop"
        self.topk_indices = ((21, 99), (22, 98)) if topk else ()
        self.topk_log_probs = ((-0.5, -3.0), (-0.25, -4.0)) if topk else ()


class _FakeEngineForServing:
    def __init__(self, rollout):
        self._rollout = rollout
        self.config = FakeEngineConfig()
        self.publications = 0
        self.template_kwargs = "unset"

    def next_runtime_load_id(self) -> str:
        self.publications += 1
        return f"fake-{self.publications}"

    def render_prompt(self, messages, *, template_kwargs=None):
        self.template_kwargs = template_kwargs
        return [11, 12, 13]

    def generate(self, prompt_tokens, *, max_tokens=None, temperature=None):
        return self._rollout


def _serve(payload, *, topk=True):
    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    runtime = MLXRuntime(
        _FakeEngineForServing(_FakeRollout(topk=topk)),
        checkpoint_dir="/tmp/reef-mlx-serving-test",
    )
    backend = MLXInferenceBackend(runtime)
    return asyncio.run(backend.inference(Artifact.local(Path("/tmp")), "/v1/chat/completions", payload))


@pytest.mark.unit
def test_a_served_response_carries_the_tensors_that_make_it_trainable() -> None:
    response = _serve({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    training = response["training"]

    # Full sequence, mask over the response only: what policy_row_violation
    # checks before a record can become a sample.
    assert training["tokens"] == [11, 12, 13, 21, 22]
    assert training["loss_mask"] == [1, 1]
    assert training["rollout_log_probs"] == [-0.5, -0.25]
    assert training["runtime_load_id"] == response["choices"][0]["meta_info"]["runtime_load_id"]


@pytest.mark.unit
def test_captured_candidates_reach_the_wire_when_the_engine_records_them() -> None:
    # A distillation objective trains on the candidate set the policy
    # considered; nothing downstream can rebuild it after generation.
    training = _serve({"messages": [{"role": "user", "content": "hi"}]})["training"]

    assert training["topk_indices"] == [[21, 99], [22, 98]]
    assert training["topk_log_probs"] == [[-0.5, -3.0], [-0.25, -4.0]]


@pytest.mark.unit
def test_no_candidate_channel_when_capture_is_off() -> None:
    # An on-policy objective needs none, and an absent key is what
    # make_policy_sample reads as "not captured".
    training = _serve({"messages": [{"role": "user", "content": "hi"}]}, topk=False)["training"]

    assert "topk_indices" not in training
    assert "topk_log_probs" not in training


@pytest.mark.unit
def test_a_request_steers_the_chat_template() -> None:
    """A reasoning model's ``<think>`` block is response tokens, so whether the
    template opens one has to be a per-request decision."""
    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    engine = _FakeEngineForServing(_FakeRollout())
    backend = MLXInferenceBackend(MLXRuntime(engine, checkpoint_dir="/tmp/reef-mlx-template-test"))
    asyncio.run(
        backend.inference(
            Artifact.local(Path("/tmp")),
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    )
    assert engine.template_kwargs == {"enable_thinking": False}


@pytest.mark.unit
def test_a_request_without_template_kwargs_leaves_the_deployment_default() -> None:
    engine = _FakeEngineForServing(_FakeRollout())
    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    backend = MLXInferenceBackend(MLXRuntime(engine, checkpoint_dir="/tmp/reef-mlx-template-test"))
    asyncio.run(
        backend.inference(
            Artifact.local(Path("/tmp")),
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
        )
    )
    assert engine.template_kwargs is None


@pytest.mark.unit
def test_a_malformed_template_kwargs_field_is_refused() -> None:
    from reef.runtime.inference import UpstreamStatusError

    with pytest.raises(UpstreamStatusError, match="chat_template_kwargs"):
        _serve({"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": "no-think"})


@pytest.mark.unit
def test_streaming_is_refused_rather_than_faked() -> None:
    from reef.runtime.inference import UpstreamStatusError

    with pytest.raises(UpstreamStatusError, match="streaming"):
        _serve({"messages": [{"role": "user", "content": "hi"}], "stream": True})


@pytest.mark.unit
def test_an_empty_completion_is_refused_so_a_grid_cannot_stall() -> None:
    from reef.runtime.inference import UpstreamStatusError

    rollout = _FakeRollout()
    rollout.output_tokens = ()
    rollout.rollout_log_probs = ()
    rollout.topk_indices = ()
    rollout.topk_log_probs = ()

    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    runtime = MLXRuntime(_FakeEngineForServing(rollout), checkpoint_dir="/tmp/reef-mlx-serving-test")
    backend = MLXInferenceBackend(runtime)
    with pytest.raises(UpstreamStatusError, match="no response tokens"):
        asyncio.run(backend.inference(Artifact.local(Path("/tmp")), "/v1/chat/completions", {"messages": [{}]}))


@register_step_preparer
class _OpenClawRLPreparer(StepPreparer):
    name = "mlx-test-openclawrl"

    def __call__(self, batch, state):
        return StepSignal(
            "train",
            "openclawrl",
            {},
            {},
            tuple(sample.reward for sample in batch.samples),
            # The real preparer leaves scheduling at its default, which names
            # the backend's own batch size.
            StepScheduling(),
        )


@register_step_preparer
class _SubBatchedPreparer(StepPreparer):
    name = "mlx-test-subbatched"

    def __call__(self, batch, state):
        return StepSignal("train", "tttd", {}, {}, (1.0, -1.0), StepScheduling(unit="sample", batch_size=4))


def _distillation_sample(*, topk=True, teacher=True) -> PolicySample:
    extras = {}
    if teacher:
        extras["teacher_cands"] = ({"hint": "Be terse.", "teacher_tokens": [7, 8, 1, 2]},)
    return PolicySample(
        source_agent_record_id="turn-1",
        tokens=(5, 6, 1, 2),
        loss_mask=(1, 1),
        rollout_log_probs=(-0.5, -0.25),
        reward=1.0,
        topk_indices=((1, 3), (2, 4)) if topk else (),
        topk_log_probs=((-0.5, -2.0), (-0.25, -3.0)) if topk else (),
        extras=extras,
    )


@pytest.mark.unit
def test_a_configured_batch_size_is_accepted_but_sub_batching_is_not(tmp_path: Path) -> None:
    # "configured" names a backend batch size that a single process does not
    # have, so the reserved batch is the step either way. An explicit integer
    # really does mean several steps, which this runtime cannot honour.
    runtime = build_runtime(tmp_path)
    batch = PolicyBatch("b", (_distillation_sample(),))

    prepared = runtime.prepare_training_step(batch, "mlx-test-openclawrl", {}, 0)
    assert prepared.action == "train"

    with pytest.raises(RuntimeContractError, match="batch_size=4"):
        runtime.prepare_training_step(batch, "mlx-test-subbatched", {}, 0)


@pytest.mark.unit
def test_the_distillation_objective_refuses_a_batch_with_no_captured_candidates(tmp_path: Path) -> None:
    # Training a distillation objective on rollouts that recorded no candidate
    # set would silently optimise nothing; say which setting is missing.
    runtime = build_runtime(tmp_path)
    batch = PolicyBatch("b", (_distillation_sample(topk=False),))
    prepared = runtime.prepare_training_step(batch, "mlx-test-openclawrl", {}, 0)

    with pytest.raises(RuntimeContractError, match="capture_topk"):
        runtime.train_candidate(prepared.payload)


@pytest.mark.unit
def test_the_distillation_objective_refuses_a_batch_with_no_teacher(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    batch = PolicyBatch("b", (_distillation_sample(teacher=False),))
    prepared = runtime.prepare_training_step(batch, "mlx-test-openclawrl", {}, 0)

    with pytest.raises(RuntimeContractError, match="teacher_cands"):
        runtime.train_candidate(prepared.payload)


@pytest.mark.unit
def test_the_openclawrl_family_reaches_the_distillation_step(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    engine = runtime.engine
    batch = PolicyBatch("b", (_distillation_sample(),))
    prepared = runtime.prepare_training_step(batch, "mlx-test-openclawrl", {}, 0)

    candidate = runtime.train_candidate(prepared.payload)

    # The distillation step ran, not the policy-gradient one.
    assert engine.distillation_rows is not None
    assert engine.trained_rows == []
    row = engine.distillation_rows[0]
    assert row.reward == 1.0
    assert row.candidates[0].hint == "Be terse."
    assert candidate.training_metrics["w_opd"] == 1.0
