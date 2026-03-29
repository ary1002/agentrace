"""LangChain integration via ``BaseCallbackHandler`` subclass."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import opentelemetry.trace as otel_trace

try:
    from langchain_core.callbacks.base import BaseCallbackHandler

    _AVAILABLE = True
except ImportError:
    BaseCallbackHandler = object  # type: ignore[misc, assignment]
    _AVAILABLE = False

from agentrace.capture.adapters._span_utils import (
    _json_truncate,
    compute_cost,
    get_tracer,
    record_exception,
    set_span_attributes,
)

_LOG = logging.getLogger(__name__)


def _llm_model_name(serialized: dict[str, Any]) -> str:
    name = serialized.get("name")
    if name:
        return str(name)
    id_parts = serialized.get("id")
    if isinstance(id_parts, (list, tuple)) and id_parts:
        return str(id_parts[-1])
    return "unknown"


class AgentTraceCallbackHandler(BaseCallbackHandler):
    """Maps LangChain callback events to OpenTelemetry spans (``agentrace`` tracer)."""

    def __init__(self) -> None:
        self._spans: dict[str, Any] = {}
        self._tracer: Any = None
        self._llm_models: dict[str, str] = {}
        if not _AVAILABLE:
            _LOG.warning("langchain_core is not installed; AgentTraceCallbackHandler is a no-op")

    def _get_tracer(self):
        if self._tracer is None:
            self._tracer = get_tracer()
        return self._tracer

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if not _AVAILABLE:
            return
        model_name = _llm_model_name(serialized)
        tracer = self._get_tracer()
        span = tracer.start_span(f"langchain/{model_name}")
        ctx = otel_trace.use_span(span, end_on_exit=False)
        ctx.__enter__()
        key = str(run_id)
        self._spans[key] = (span, ctx)
        self._llm_models[key] = model_name
        inp = {"model": model_name, "prompts": prompts}
        span.set_attribute("agentrace.span_type", "llm_call")
        span.set_attribute("agentrace.framework", "langchain")
        span.set_attribute("agentrace.input", _json_truncate(inp))

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if not _AVAILABLE:
            return
        key = str(run_id)
        pair = self._spans.pop(key, (None, None))
        if pair == (None, None):
            return
        span, ctx = pair
        model_name = self._llm_models.pop(key, "unknown")
        try:
            text = ""
            if getattr(response, "generations", None):
                row = response.generations[0]
                if row and getattr(row[0], "text", None) is not None:
                    text = row[0].text
            llm_output = getattr(response, "llm_output", None) or {}
            token_usage = llm_output.get("token_usage") or {}
            prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
            cost = compute_cost(model_name, prompt_tokens, completion_tokens)
            output_dict = {"text": text}
            set_span_attributes(
                span,
                "llm_call",
                {},
                output_dict,
                "langchain",
                prompt_tokens,
                completion_tokens,
                cost,
                omit_input=True,
            )
        finally:
            ctx.__exit__(None, None, None)
            span.end()

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        if not _AVAILABLE:
            return
        key = str(run_id)
        pair = self._spans.pop(key, (None, None))
        if pair == (None, None):
            return
        span, ctx = pair
        self._llm_models.pop(key, None)
        try:
            record_exception(span, error)
        finally:
            ctx.__exit__(type(error), error, None)
            span.end()

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if not _AVAILABLE:
            return
        tool_name = str(serialized.get("name", "tool"))
        tracer = self._get_tracer()
        span = tracer.start_span(f"tool/{tool_name}")
        ctx = otel_trace.use_span(span, end_on_exit=False)
        ctx.__enter__()
        self._spans[str(run_id)] = (span, ctx)
        inp = {"tool_name": tool_name, "input": input_str}
        span.set_attribute("agentrace.span_type", "tool_call")
        span.set_attribute("agentrace.framework", "langchain")
        span.set_attribute("agentrace.input", _json_truncate(inp))

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if not _AVAILABLE:
            return
        key = str(run_id)
        pair = self._spans.pop(key, (None, None))
        if pair == (None, None):
            return
        span, ctx = pair
        try:
            output_dict = {"output": str(output)}
            set_span_attributes(
                span,
                "tool_call",
                {},
                output_dict,
                "langchain",
                omit_input=True,
            )
        finally:
            ctx.__exit__(None, None, None)
            span.end()

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        if not _AVAILABLE:
            return
        key = str(run_id)
        pair = self._spans.pop(key, (None, None))
        if pair == (None, None):
            return
        span, ctx = pair
        try:
            record_exception(span, error)
        finally:
            ctx.__exit__(type(error), error, None)
            span.end()

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if not _AVAILABLE:
            return
        chain_name = str(serialized.get("name", "chain"))
        tracer = self._get_tracer()
        span = tracer.start_span(f"langchain/{chain_name}")
        ctx = otel_trace.use_span(span, end_on_exit=False)
        ctx.__enter__()
        self._spans[str(run_id)] = (span, ctx)
        inp = {"chain": chain_name, "inputs": inputs}
        span.set_attribute("agentrace.span_type", "agent_step")
        span.set_attribute("agentrace.framework", "langchain")
        span.set_attribute("agentrace.input", _json_truncate(inp))

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        if not _AVAILABLE:
            return
        key = str(run_id)
        pair = self._spans.pop(key, (None, None))
        if pair == (None, None):
            return
        span, ctx = pair
        try:
            output_dict = {"outputs": outputs}
            set_span_attributes(
                span,
                "agent_step",
                {},
                output_dict,
                "langchain",
                omit_input=True,
            )
        finally:
            ctx.__exit__(None, None, None)
            span.end()

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        if not _AVAILABLE:
            return
        key = str(run_id)
        pair = self._spans.pop(key, (None, None))
        if pair == (None, None):
            return
        span, ctx = pair
        try:
            record_exception(span, error)
        finally:
            ctx.__exit__(type(error), error, None)
            span.end()
