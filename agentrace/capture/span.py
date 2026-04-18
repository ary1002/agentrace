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

_KNOWN_SPAN_TYPES: frozenset[str] = frozenset(
    ("llm_call", "tool_call", "memory_read", "memory_write", "agent_step")
)


class TraceSpan:
    """OpenTelemetry span plus :meth:`set_output` for results known after the block starts."""

    __slots__ = (
        "_otel",
        "_span_type",
        "_input_dict",
        "_output_dict",
        "_framework",
        "_prompt_tokens",
        "_completion_tokens",
        "_cost_usd",
    )

    def __init__(
        self,
        otel_span: OtelSpan,
        *,
        span_type: str,
        input_dict: dict[str, Any],
        output_dict: dict[str, Any],
        framework: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        self._otel = otel_span
        self._span_type = span_type
        self._input_dict = input_dict
        self._output_dict = output_dict
        self._framework = framework
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._cost_usd = cost_usd

    def set_output(self, result: Any) -> None:
        """Update ``agentrace.output`` on the span (dict is copied; other values become ``{"output": ...}``)."""
        if isinstance(result, dict):
            self._output_dict = dict(result)
        else:
            self._output_dict = {"output": result}
        set_span_attributes(
            self._otel,
            self._span_type,
            self._input_dict,
            self._output_dict,
            self._framework,
            self._prompt_tokens,
            self._completion_tokens,
            self._cost_usd,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._otel, name)


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
) -> Iterator[TraceSpan]:
    """Open a typed child span and attach AgentTrace attributes.

    The yielded object supports :meth:`TraceSpan.set_output` (per spec §3.2) and forwards
    other attributes to the underlying OpenTelemetry span.

    When ``span_type`` is left as default ``\"agent_step\"`` and ``name`` is one of the
    known type literals (e.g. ``\"tool_call\"``), ``name`` is also used as
    ``agentrace.span_type`` so ``with span(\"tool_call\", tool=...)`` matches the spec.
    """
    tracer = get_tracer()
    effective_type = (
        name if span_type == "agent_step" and name in _KNOWN_SPAN_TYPES else span_type
    )
    input_dict: dict[str, Any] = dict(input or {})
    if tool:
        input_dict.setdefault("tool_name", tool)
    output_dict: dict[str, Any] = dict(output or {})

    with tracer.start_as_current_span(name) as otel_span:
        wrapped = TraceSpan(
            otel_span,
            span_type=effective_type,
            input_dict=input_dict,
            output_dict=output_dict,
            framework=framework,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
        )
        set_span_attributes(
            otel_span,
            effective_type,
            input_dict,
            output_dict,
            framework,
            prompt_tokens,
            completion_tokens,
            cost_usd,
        )
        try:
            yield wrapped
        except Exception as exc:
            record_exception(otel_span, exc)
            raise
