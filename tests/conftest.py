from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agentrace.normalizer.models import AgentTrace, Span, SpanNode, TokenCount


def make_span(
    span_id: str,
    *,
    parent_span_id: str | None = None,
    span_type: str = "agent_step",
    tool_name: str | None = None,
    latency_ms: float = 10.0,
    cost_usd: float = 0.0,
    error: str | None = None,
    offset_ms: int = 0,
) -> Span:
    input_obj = {}
    if tool_name is not None:
        input_obj["tool_name"] = tool_name
    return Span(
        span_id=span_id,
        parent_span_id=parent_span_id,
        span_type=span_type,  # type: ignore[arg-type]
        input=input_obj,
        output={},
        latency_ms=latency_ms,
        token_count=TokenCount(prompt=10, completion=5),
        cost_usd=cost_usd,
        timestamp=datetime.now(timezone.utc) + timedelta(milliseconds=offset_ms),
        framework="agentrace",
        error=error,
    )


def make_trace(
    spans: list[Span],
    *,
    outcome: str = "success",
    total_tokens: int = 0,
    total_cost: float = 0.0,
) -> AgentTrace:
    if spans:
        by_id = {s.span_id: SpanNode(span=s) for s in spans}
        root = by_id[spans[0].span_id]
        for s in spans[1:]:
            parent = by_id.get(s.parent_span_id or spans[0].span_id, root)
            parent.children.append(by_id[s.span_id])
    else:
        root_span = make_span("root", span_type="agent_step")
        root = SpanNode(span=root_span)
    return AgentTrace(
        session_id="s1",
        task="task",
        spans=spans,
        trace_tree=root,
        total_latency_ms=sum(s.latency_ms for s in spans),
        total_cost_usd=total_cost if total_cost else sum(s.cost_usd for s in spans),
        total_tokens=total_tokens,
        outcome=outcome,  # type: ignore[arg-type]
    )
