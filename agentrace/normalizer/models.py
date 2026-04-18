"""Dataclasses for normalized traces: ``Span``, ``AgentTrace``, ``SpanNode``, ``TokenCount``."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


class MalformedTraceError(Exception):
    """Raised when raw OpenTelemetry spans cannot be converted into a valid ``AgentTrace`` DAG."""


@dataclass
class TokenCount:
    """Token usage attributed to a single span."""

    prompt: int
    completion: int

    @property
    def total(self) -> int:
        return self.prompt + self.completion


@dataclass
class Span:
    """Normalized span aligned with instrumentation attributes."""

    span_id: str
    parent_span_id: str | None
    span_type: Literal[
        "llm_call", "tool_call", "memory_read", "memory_write", "agent_step"
    ]
    input: dict
    output: dict
    latency_ms: float
    token_count: TokenCount
    cost_usd: float
    timestamp: datetime
    framework: str
    error: str | None = None


@dataclass
class SpanNode:
    """A span and its child nodes in the execution tree."""

    span: Span
    children: list[SpanNode] = field(default_factory=list)


@dataclass
class AgentTrace:
    """Full agent session: flat span list and reconstructed tree."""

    session_id: str
    task: str
    spans: list[Span]
    trace_tree: SpanNode
    total_latency_ms: float
    total_cost_usd: float
    total_tokens: int
    outcome: Literal["success", "failure", "partial"] = "success"
