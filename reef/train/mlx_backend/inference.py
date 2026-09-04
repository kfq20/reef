"""OpenAI-shaped chat inference executed in this process by MLX.

The response carries a private ``training`` block holding the exact token ids
the engine sampled and their log-probabilities. That block is what makes a
served exchange trainable: Reef never re-tokenizes decoded text to reconstruct
policy tensors, so a rollout that was not captured token-natively at
generation time can never become a policy sample.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from typing import Any

from reef.artifact.artifact import Artifact
from reef.runtime.inference import InferenceBackend, UpstreamStatusError


class MLXInferenceBackend(InferenceBackend):
    """Answer chat completions from the runtime's resident MLX model."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        # MLX evaluates on one process-wide stream, and training mutates the
        # same parameters generation reads. One lock keeps concurrent HTTP
        # requests from interleaving inside the engine.
        self._lock = asyncio.Lock()

    async def inference(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not path.rstrip("/").endswith("chat/completions"):
            raise UpstreamStatusError(f"the mlx backend serves chat completions, not {path!r}", status=404)
        if payload.get("stream") is True:
            # The base class would hand back the whole JSON body as one
            # "stream" chunk, which is not SSE and would leave the caller
            # parsing something that is not a provider stream.
            raise UpstreamStatusError("the mlx backend does not implement streaming completions", status=400)
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise UpstreamStatusError("chat completions require a non-empty messages list", status=400)

        max_tokens = payload.get("max_completion_tokens", payload.get("max_tokens"))
        temperature = payload.get("temperature")
        # The field OpenAI-compatible servers use to steer a chat template —
        # `enable_thinking` above all, which decides whether a reasoning
        # model's `<think>` block becomes response tokens.
        template_kwargs = payload.get("chat_template_kwargs")
        if template_kwargs is not None and not isinstance(template_kwargs, Mapping):
            raise UpstreamStatusError("chat_template_kwargs must be an object", status=400)
        async with self._lock:
            rollout = await asyncio.to_thread(
                self._generate,
                messages,
                None if max_tokens is None else int(max_tokens),
                None if temperature is None else float(temperature),
                template_kwargs,
            )
        if not rollout.output_tokens:
            # A completion with no response tokens can never be a policy
            # sample: `policy_row_violation` rejects an empty loss mask, the
            # processor marks the report terminally unusable, and a grid-based
            # recipe would then wait forever for a slot that can never fill.
            # Fail the request instead, so the caller retries a rollout.
            raise UpstreamStatusError("the model produced no response tokens", status=502)
        return self._response(payload, rollout)

    def _generate(
        self,
        messages: list[Any],
        max_tokens: int | None,
        temperature: float | None,
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        engine = self._runtime.engine
        prompt_tokens = engine.render_prompt(messages, template_kwargs=template_kwargs)
        return engine.generate(prompt_tokens, max_tokens=max_tokens, temperature=temperature)

    def _response(self, request: Mapping[str, Any], rollout: Any) -> dict[str, Any]:
        runtime_load_id = self._runtime.serving_runtime_load_id()
        prompt_tokens = list(rollout.prompt_tokens)
        output_tokens = list(rollout.output_tokens)
        response: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(request.get("model") or self._runtime.engine.config.model_path),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": rollout.text},
                    "finish_reason": rollout.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(output_tokens),
                "total_tokens": len(prompt_tokens) + len(output_tokens),
            },
            "training": {
                "tokens": [*prompt_tokens, *output_tokens],
                # Every generated token is a policy action here: this backend
                # runs single-turn completions with no environment tokens.
                "loss_mask": [1] * len(output_tokens),
                "rollout_log_probs": list(rollout.rollout_log_probs),
                "prompt_length": len(prompt_tokens),
                "response_length": len(output_tokens),
                "runtime_load_id": runtime_load_id,
                # Present only when the engine was asked to capture them. A
                # distillation objective trains on the candidate set the
                # policy actually considered, and nothing downstream can
                # reconstruct it after generation.
                **(
                    {
                        "topk_indices": [list(row) for row in rollout.topk_indices],
                        "topk_log_probs": [list(row) for row in rollout.topk_log_probs],
                    }
                    if rollout.topk_indices
                    else {}
                ),
            },
        }
        if runtime_load_id is not None:
            # Surface verification reads the engine-reported version from
            # meta_info; without it a live-weight artifact cannot prove which
            # weights answered.
            response["choices"][0]["meta_info"] = {"runtime_load_id": runtime_load_id}
        return response


__all__ = ["MLXInferenceBackend"]
