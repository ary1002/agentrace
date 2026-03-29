"""Shared OpenTelemetry helpers for agentrace adapters."""

from __future__ import annotations

import json

import opentelemetry.trace as otel_trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer


_MAX_ATTR_LEN = 32_000
_TRUNC_SUFFIX = "...[truncated]"


def get_tracer() -> Tracer:
    from agentrace.capture.context_manager import peek_active_tracer

    session_tracer = peek_active_tracer()
    if session_tracer is not None:
        return session_tracer
    return otel_trace.get_tracer("agentrace")


def _json_truncate(obj: dict) -> str:
    raw = json.dumps(obj, default=str)
    if len(raw) <= _MAX_ATTR_LEN:
        return raw
    cap = _MAX_ATTR_LEN - len(_TRUNC_SUFFIX)
    if cap < 0:
        return _TRUNC_SUFFIX
    return raw[:cap] + _TRUNC_SUFFIX


def set_span_attributes(
    span: Span,
    span_type: str,
    input_dict: dict,
    output_dict: dict,
    framework: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
    *,
    omit_input: bool = False,
    omit_output: bool = False,
) -> None:
    span.set_attribute("agentrace.span_type", span_type)
    if not omit_input:
        span.set_attribute("agentrace.input", _json_truncate(input_dict))
    if not omit_output:
        span.set_attribute("agentrace.output", _json_truncate(output_dict))
    span.set_attribute("agentrace.framework", framework)
    span.set_attribute("agentrace.token_count.prompt", int(prompt_tokens))
    span.set_attribute("agentrace.token_count.completion", int(completion_tokens))
    span.set_attribute("agentrace.cost_usd", float(cost_usd))


def record_exception(span: Span, exc: BaseException) -> None:
    span.set_attribute("agentrace.error", f"{type(exc).__name__}: {exc}")
    span.set_status(Status(StatusCode.ERROR, str(exc)))


MODEL_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.000005, 0.000015),
    "gpt-4o-mini": (0.00000015, 0.0000006),
    "gpt-4-turbo": (0.00001, 0.00003),
    "gpt-3.5-turbo": (0.0000005, 0.0000015),
    "claude-opus-4-20250514": (0.000015, 0.000075),
    "claude-sonnet-4-20250514": (0.000003, 0.000015),
    "claude-haiku-4-5-20251001": (0.0000008, 0.000004),
}


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    m = str(model)
    while m:
        rates = MODEL_COST_TABLE.get(m)
        if rates is not None:
            pp, cp = rates
            return pp * prompt_tokens + cp * completion_tokens
        last_dash = m.rfind("-")
        if last_dash <= 0:
            break
        m = m[:last_dash]
    return 0.0
