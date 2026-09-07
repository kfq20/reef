"""Turn one sampled completion into the assistant message a chat client reads.

A chat template asks the model for reasoning and for tool calls in a syntax
it states inside the prompt, and the sampled text carries that syntax
verbatim. OpenAI-shaped clients render ``content`` as the reply, so reasoning
has to ride ``reasoning_content`` and calls have to ride ``tool_calls``.
Otherwise every consumer downstream (agents, judges, user simulators) reads
chain-of-thought and raw call markup as what the agent said, and an agent
that gets its own markup echoed back as text copies it as an example of how
to call tools for the rest of the session.

This is presentation only. The tokens and log-probs that train are captured
from the raw stream before the split and never depend on it. Every backend
that serves a trainable chat completion runs its sampled text through
:func:`split_assistant_message`, so the SGLang and MLX paths cannot drift.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any, Protocol

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class ToolCallParser(Protocol):
    """What the split needs from a parser: SGLang's ``FunctionCallParser`` and the MLX adapter both fit."""

    def has_tool_call(self, text: str) -> bool: ...

    def parse_non_stream(self, text: str) -> tuple[str, Sequence[Any]]:
        """Return the text outside the calls and the calls; each call exposes ``name`` and ``parameters``."""
        ...


def reasoning_is_pre_opened(tokenizer: Any) -> bool:
    """Whether the chat template opens ``<think>`` on the model's behalf.

    Qwen3-*-Thinking templates end the generation prompt with an open
    ``<think>``, so the sample carries only the closing tag, and a sample
    without one is reasoning that ran out of tokens rather than an answer.
    Rendered from an empty conversation: the tail of the generation prompt
    does not depend on the messages.
    """
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True
        )
    except Exception:  # a template we cannot render tells us nothing
        return False
    return str(rendered).rstrip().endswith(THINK_OPEN)


def split_assistant_message(
    text: str,
    tool_parser: ToolCallParser | None,
    *,
    force_reasoning: bool = False,
    split_reasoning: bool = True,
    tool_call_start: str | None = None,
    parser_label: str = "tool-call",
) -> tuple[dict[str, Any], bool]:
    """Split sampled text into an OpenAI assistant message; ``True`` when it carries tool calls.

    Reasoning comes first. Thinking models emit it before the last
    ``</think>``; the opening tag is usually absent because the template
    inserted it. ``force_reasoning`` says the template did so for this
    prompt, and then a sample with no closing tag is reasoning that hit the
    token cap: the whole sample is ``reasoning_content`` and ``content`` is
    empty, because handing it back as content is exactly the
    chain-of-thought-as-reply this split exists to prevent, and a judge
    scoring it would score the wrong text.

    One exception, which Ollama's Qwen3 parser makes too: a model that opens
    a tool call while still inside its thinking has stopped thinking. When
    ``tool_call_start`` is known and appears before any ``</think>``, the
    text up to it is the reasoning and the call is parsed as a call, rather
    than the whole turn being lost as truncated reasoning.

    Then the calls. A parser that cannot parse text it recognised as a call
    raises ``ValueError``: the caller fails the request so the agent retries
    a rollout, instead of the agent reading broken markup as a reply.
    """
    reasoning: str | None = None
    if split_reasoning:
        close = text.rfind(THINK_CLOSE)
        thinking_open = force_reasoning or text.lstrip().startswith(THINK_OPEN)
        call = text.find(tool_call_start) if tool_call_start and thinking_open and close == -1 else -1
        if call != -1:
            reasoning = text[:call].replace(THINK_OPEN, "").strip()
            text = text[call:]
        elif close != -1:
            reasoning = text[:close].replace(THINK_OPEN, "").strip()
            text = text[close + len(THINK_CLOSE) :].lstrip("\n")
        elif force_reasoning:
            reasoning = text.replace(THINK_OPEN, "").strip()
            text = ""
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_parser is None or not tool_parser.has_tool_call(text):
        return message, False
    try:
        remaining_text, parsed = tool_parser.parse_non_stream(text)
    except Exception as exc:
        raise ValueError(f"{parser_label} parser could not parse the sampled output: {exc}") from exc
    if not parsed:
        return message, False
    tool_calls = []
    for call_item in parsed:
        arguments = call_item.parameters
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": call_item.name, "arguments": arguments},
            }
        )
    message["content"] = remaining_text or None
    message["tool_calls"] = tool_calls
    return message, True


__all__ = [
    "THINK_CLOSE",
    "THINK_OPEN",
    "ToolCallParser",
    "reasoning_is_pre_opened",
    "split_assistant_message",
]
