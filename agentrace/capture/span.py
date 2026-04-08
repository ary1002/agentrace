"""Manual span context manager for fine-grained events within traced runs."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry.trace import Span as OtelSpan

from agentrace.capture.adapters._span_utils import (
    get_tracer,
    record_exception,
    set_span_attributes,
)


@contextmanager
def span(
    name: str,
    *,
    span_type: str = "agent_step",
    tool: str | None = None,
    input: dict[str, Any] | None = None,
    output: dict[str, Any] | None = None,
    framework: str = "agentrace",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
) -> Iterator[OtelSpan]:
    """Open a typed child span and attach AgentTrace attributes."""
    tracer = get_tracer()
    input_dict: dict[str, Any] = dict(input or {})
    if tool:
        input_dict.setdefault("tool_name", tool)
    output_dict: dict[str, Any] = dict(output or {})

    with tracer.start_as_current_span(name) as otel_span:
        set_span_attributes(
            otel_span,
            span_type,
            input_dict,
            output_dict,
            framework,
            prompt_tokens,
            completion_tokens,
            cost_usd,
        )
        try:
            yield otel_span
        except Exception as exc:
            record_exception(otel_span, exc)
            raise
