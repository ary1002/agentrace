"""Async ``agentrace.trace()`` context manager: OTel root span + in-memory export → ``AgentTrace``."""

from __future__ import annotations

import json
from contextlib import AbstractAsyncContextManager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer as OtelTracer

from agentrace.capture.adapters.llamaindex import reset_llamaindex, setup_llamaindex
from agentrace.normalizer.normalizer import Normalizer

if TYPE_CHECKING:
    from agentrace.normalizer.models import AgentTrace

_active_tracer: ContextVar[OtelTracer | None] = ContextVar(
    "agentrace_active_otel_tracer",
    default=None,
)


def peek_active_tracer() -> OtelTracer | None:
    """Return the session tracer when inside :func:`trace`, else ``None``.

    Used by :func:`agentrace.capture.adapters._span_utils.get_tracer` so SDK patches
    attach to the in-memory exporter without replacing the process-global
    ``TracerProvider`` (which OpenTelemetry allows only once).
    """
    return _active_tracer.get()


def current_tracer() -> OtelTracer:
    """Return the ``Tracer`` for the innermost active :func:`trace` block.

    Use this (or the returned tracer's ``start_as_current_span``) for nested steps so
    spans are captured to the same in-memory exporter as the session root.
    """
    t = _active_tracer.get()
    if t is None:
        raise RuntimeError(
            "current_tracer() called outside an active agentrace.trace() context"
        )
    return t


@dataclass
class TraceContext:
    """Values for one traced block; ``agent_trace`` is filled after successful exit."""

    session_id: str
    task: str
    agent_trace: AgentTrace | None = None


class _TraceAsyncContextManager(AbstractAsyncContextManager[TraceContext]):
    """Uses a dedicated ``TracerProvider`` with in-memory export for one async block.

    A separate provider is used for each session. The session tracer is published via
    :func:`peek_active_tracer` so adapter code calling :func:`get_tracer` in
    ``_span_utils`` records spans on this provider.
    """

    def __init__(self, session_id: str, task: str) -> None:
        self._session_id = session_id
        self._task = task
        self._exporter: InMemorySpanExporter | None = None
        self._provider: TracerProvider | None = None
        self._span_cm: Any = None
        self._ctx: TraceContext | None = None
        self._tracer_token: Token[OtelTracer | None] | None = None

    async def __aenter__(self) -> TraceContext:
        self._exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(self._exporter)
        self._provider = TracerProvider()
        self._provider.add_span_processor(processor)
        setup_llamaindex(self._provider)
        tracer = self._provider.get_tracer("agentrace")
        self._tracer_token = _active_tracer.set(tracer)
        self._span_cm = tracer.start_as_current_span(
            self._session_id,
            attributes={
                "agentrace.session_id": self._session_id,
                "agentrace.task": self._task,
                "agentrace.framework": "agentrace",
                "agentrace.input": json.dumps(
                    {"node": "__session_root__", "node_name": "__session_root__"}
                ),
                "agentrace.output": "{}",
            },
        )
        assert self._span_cm is not None
        self._span_cm.__enter__()
        self._ctx = TraceContext(session_id=self._session_id, task=self._task)
        return self._ctx

    async def __aexit__(self, exc_type, exc, tb) -> None:
        assert self._ctx is not None
        try:
            if self._span_cm is not None:
                self._span_cm.__exit__(exc_type, exc, tb)
            if self._exporter is not None:
                finished = list(self._exporter.get_finished_spans())
                self._ctx.agent_trace = Normalizer.build(
                    self._session_id, self._task, finished
                )
        finally:
            if self._tracer_token is not None:
                _active_tracer.reset(self._tracer_token)
            reset_llamaindex()
            if self._provider is not None:
                self._provider.shutdown()
        return None


def trace(session_id: str, task: str) -> AbstractAsyncContextManager[TraceContext]:
    """Open an async tracing block; after exit, ``TraceContext.agent_trace`` holds the DAG."""
    return _TraceAsyncContextManager(session_id, task)
