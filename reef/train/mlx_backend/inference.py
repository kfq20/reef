"""OpenAI-shaped chat inference executed in this process by MLX.

The response carries a private ``training`` block holding the exact token ids
the engine sampled and their log-probabilities. That block is what makes a
served exchange trainable: Reef never re-tokenizes decoded text to reconstruct
policy tensors, so a rollout that was not captured token-natively at
generation time can never become a policy sample.

The public ``message`` is a presentation of the same sample: reasoning split
off into ``reasoning_content`` and tool calls parsed into ``tool_calls`` in
the syntax the model's own chat template asked for. The split reads the
decoded text and never touches the tensors.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.artifact.artifact import Artifact
from reef.runtime.assistant_message import THINK_OPEN, split_assistant_message
from reef.runtime.inference import InferenceBackend, UpstreamStatusError


@dataclass(frozen=True)
class ParsedToolCall:
    """One call the parser recovered: what ``split_assistant_message`` reads."""

    name: str
    parameters: Any


class MLXToolCallParser:
    """mlx-lm's per-model tool parser behind the interface the message split reads.

    mlx-lm chooses the parser from the chat template when it loads the
    tokenizer (Qwen3.8's ``<tool_call>``-wrapped XML, Hermes-style JSON,
    Mistral's ``[TOOL_CALLS]``, ...) and exposes the markers that bound a call
    plus a ``parse_tool_call(text, tools)`` that coerces each argument by the
    type its declared schema gives it. This adapter finds the bounded spans,
    hands each to that function, and keeps the text outside them as the reply.
    """

    def __init__(self, tokenizer: Any, tools: Sequence[Mapping[str, Any]]) -> None:
        self.tool_call_start: str = tokenizer.tool_call_start
        self._tool_call_end: str = tokenizer.tool_call_end
        self._parse = tokenizer.tool_parser
        self._tools = [dict(tool) for tool in tools]

    @classmethod
    def for_tokenizer(cls, tokenizer: Any, tools: Sequence[Mapping[str, Any]] | None) -> MLXToolCallParser | None:
        """The parser for this request, or ``None`` when nothing could be a call.

        No declared toolset means the prompt stated no call syntax, so any
        markup in the reply is text. A template mlx-lm found no parser for
        leaves the reply as text too, rather than guessing at a syntax.
        """
        if not tools or not getattr(tokenizer, "has_tool_calling", False):
            return None
        return cls(tokenizer, tools)

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start in text

    def parse_non_stream(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        """The text outside every call, and the calls in order.

        A call that opens and never closes is a sample that hit the token cap
        mid-call; it raises, and the request fails so the agent retries a
        rollout instead of reading half a call as a reply.
        """
        outside: list[str] = []
        calls: list[ParsedToolCall] = []
        cursor = 0
        while True:
            start = text.find(self.tool_call_start, cursor)
            if start == -1:
                outside.append(text[cursor:])
                break
            outside.append(text[cursor:start])
            body_start = start + len(self.tool_call_start)
            end = text.find(self._tool_call_end, body_start)
            if end == -1:
                raise ValueError(f"a tool call opened with {self.tool_call_start!r} never closed")
            parsed = self._parse(text[body_start:end].strip(), self._tools)
            name = parsed.get("name") if isinstance(parsed, Mapping) else None
            if not isinstance(name, str) or not name:
                raise ValueError("a tool call names no function")
            arguments = parsed.get("arguments", parsed.get("parameters"))
            calls.append(ParsedToolCall(name=name, parameters={} if arguments is None else arguments))
            cursor = end + len(self._tool_call_end)
        return "".join(outside).strip(), calls


def prompt_opens_reasoning(tokenizer: Any, prompt_tokens: Sequence[int]) -> bool:
    """Whether this rendered prompt ends by opening the model's thinking block.

    Decided per request rather than once per model: the same Qwen3 template
    opens ``<think>`` by default and opens-and-closes it at once when a
    request sets ``enable_thinking`` false, and the split has to know which
    one produced the sample it is reading. The last few prompt tokens are
    enough: the marker is the final thing the template writes.
    """
    tail = tokenizer.decode(list(prompt_tokens[-8:]))
    return str(tail).rstrip().endswith(THINK_OPEN)


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
        # The toolset the caller declared. The chat template renders it into
        # the prompt — the schemas and the call syntax the model is meant to
        # answer in — so dropping it leaves the model inventing both.
        tools = payload.get("tools")
        if tools is not None and not isinstance(tools, list):
            raise UpstreamStatusError("tools must be an array", status=400)
        # The same toolset decides how the reply is read back: the parser
        # mlx-lm matched to the template turns the model's call markup into
        # `tool_calls`. `tool_choice: none` declares the tools for context
        # only, as the OpenAI dialect defines it.
        parser = None if payload.get("tool_choice") == "none" else self._tool_call_parser(tools)
        async with self._lock:
            rollout, opens_reasoning = await asyncio.to_thread(
                self._generate,
                messages,
                None if max_tokens is None else int(max_tokens),
                None if temperature is None else float(temperature),
                template_kwargs,
                tools,
            )
        if not rollout.output_tokens:
            # A completion with no response tokens can never be a policy
            # sample: `policy_row_violation` rejects an empty loss mask, the
            # processor marks the report terminally unusable, and a grid-based
            # recipe would then wait forever for a slot that can never fill.
            # Fail the request instead, so the caller retries a rollout.
            raise UpstreamStatusError("the model produced no response tokens", status=502)
        return self._response(payload, rollout, parser, force_reasoning=opens_reasoning)

    def _tool_call_parser(self, tools: list[Any] | None) -> MLXToolCallParser | None:
        return MLXToolCallParser.for_tokenizer(self._runtime.engine.tokenizer, tools)

    def _generate(
        self,
        messages: list[Any],
        max_tokens: int | None,
        temperature: float | None,
        template_kwargs: Mapping[str, Any] | None = None,
        tools: list[Any] | None = None,
    ) -> tuple[Any, bool]:
        engine = self._runtime.engine
        prompt_tokens = engine.render_prompt(messages, tools=tools, template_kwargs=template_kwargs)
        opens_reasoning = prompt_opens_reasoning(engine.tokenizer, prompt_tokens)
        return engine.generate(prompt_tokens, max_tokens=max_tokens, temperature=temperature), opens_reasoning

    def _response(
        self,
        request: Mapping[str, Any],
        rollout: Any,
        parser: MLXToolCallParser | None,
        *,
        force_reasoning: bool,
    ) -> dict[str, Any]:
        runtime_load_id = self._runtime.serving_runtime_load_id()
        prompt_tokens = list(rollout.prompt_tokens)
        output_tokens = list(rollout.output_tokens)
        try:
            message, called = split_assistant_message(
                rollout.text,
                parser,
                force_reasoning=force_reasoning,
                tool_call_start=None if parser is None else parser.tool_call_start,
                parser_label="the mlx tool-call",
            )
        except ValueError as exc:
            # Broken call markup is not a reply. Failing the request is what
            # the SGLang path does too; the agent's request loop retries.
            raise UpstreamStatusError(str(exc), status=502) from exc
        response: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(request.get("model") or self._runtime.engine.config.model_path),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if called else rollout.finish_reason,
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


__all__ = ["MLXInferenceBackend", "MLXToolCallParser", "ParsedToolCall", "prompt_opens_reasoning"]
