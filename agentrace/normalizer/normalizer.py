"""Convert finished OpenTelemetry ``ReadableSpan`` instances into ``AgentTrace``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal, cast, Mapping, Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import format_span_id

from agentrace.normalizer.models import AgentTrace, MalformedTraceError, Span, SpanNode, TokenCount

_SPAN_TYPES: tuple[str, ...] = (
    "llm_call",
    "tool_call",
    "memory_read",
    "memory_write",
    "agent_step",
)


class Normalizer:
    """Builds ``AgentTrace`` from exported SDK spans and attribute conventions."""

    @classmethod
    def build(cls, session_id: str, task: str, raw_spans: list[ReadableSpan]) -> AgentTrace:
        spans: list[Span] = [cls._convert_readable(r) for r in raw_spans]
        roots = [s for s in spans if s.parent_span_id is None]
        if len(roots) != 1:
            raise MalformedTraceError(
                f"expected exactly one root span (parent_span_id is None), found {len(roots)}"
            )
        root_id = roots[0].span_id

        nodes: dict[str, SpanNode] = {s.span_id: SpanNode(span=s) for s in spans}
        root_node = nodes[root_id]

        for s in spans:
            if s.span_id == root_id:
                continue
            parent_id = s.parent_span_id
            if parent_id is not None and parent_id in nodes:
                nodes[parent_id].children.append(nodes[s.span_id])
            else:
                root_node.children.append(nodes[s.span_id])

        cls._assert_acyclic(root_node)

        total_latency_ms = sum(s.latency_ms for s in spans)
        total_cost_usd = sum(s.cost_usd for s in spans)
        total_tokens = sum(s.token_count.total for s in spans)

        return AgentTrace(
            session_id=session_id,
            task=task,
            spans=spans,
            trace_tree=root_node,
            total_latency_ms=total_latency_ms,
            total_cost_usd=total_cost_usd,
            total_tokens=total_tokens,
            outcome="success",
        )

    @staticmethod
    def _attr_get(attributes: Any, key: str, default: object | None = None) -> object | None:
        if attributes is None:
            return default
        if isinstance(attributes, Mapping):
            return attributes.get(key, default)
        getter = getattr(attributes, "get", None)
        if callable(getter):
            return getter(key, default)
        return default

    @classmethod
    def _convert_readable(cls, raw: ReadableSpan) -> Span:
        ctx = raw.context
        if ctx is None:
            raise MalformedTraceError("span has no context")
        span_id = format_span_id(ctx.span_id)

        parent = raw.parent
        parent_span_id: str | None
        if parent is not None and getattr(parent, "is_valid", False):
            parent_span_id = format_span_id(parent.span_id)
        else:
            parent_span_id = None

        attrs = raw.attributes
        span_type = cls._parse_span_type(attrs)
        input_obj = cls._load_json_attr(attrs, "agentrace.input")
        output_obj = cls._load_json_attr(attrs, "agentrace.output")

        start_time = raw.start_time
        if start_time is None:
            raise MalformedTraceError("span has no start_time")
        end_time = raw.end_time
        if end_time is None:
            latency_ms = 0.0
        else:
            latency_ms = (end_time - start_time) / 1e6

        prompt_tok = cls._int_attr(attrs, "agentrace.token_count.prompt")
        completion_tok = cls._int_attr(attrs, "agentrace.token_count.completion")
        token_count = TokenCount(prompt=prompt_tok, completion=completion_tok)

        cost_usd = cls._float_attr(attrs, "agentrace.cost_usd", 0.0)
        ts = datetime.fromtimestamp(start_time / 1e9, tz=timezone.utc)
        framework = str(cls._attr_get(attrs, "agentrace.framework", "unknown") or "unknown")
        err_raw = cls._attr_get(attrs, "agentrace.error")
        error: str | None = None if err_raw is None else str(err_raw)

        return Span(
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_type=span_type,
            input=input_obj,
            output=output_obj,
            latency_ms=latency_ms,
            token_count=token_count,
            cost_usd=cost_usd,
            timestamp=ts,
            framework=framework,
            error=error,
        )

    @classmethod
    def _parse_span_type(
        cls, attributes: object
    ) -> Literal["llm_call", "tool_call", "memory_read", "memory_write", "agent_step"]:
        raw = cls._attr_get(attributes, "agentrace.span_type")
        if raw is None:
            return "agent_step"
        s = str(raw)
        if s in _SPAN_TYPES:
            return cast(
                Literal["llm_call", "tool_call", "memory_read", "memory_write", "agent_step"],
                s,
            )
        return "agent_step"

    @classmethod
    def _load_json_attr(cls, attributes: object, key: str) -> dict:
        raw = cls._attr_get(attributes, key, "{}")
        if raw is None:
            raw = "{}"
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            raise MalformedTraceError(f"{key!r} must be a JSON object string or dict, got {type(raw)}")
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MalformedTraceError(f"invalid JSON for {key!r}") from exc
        if not isinstance(obj, dict):
            raise MalformedTraceError(f"JSON for {key!r} must decode to an object")
        return obj

    @classmethod
    def _int_attr(cls, attributes: object, key: str) -> int:
        raw = cls._attr_get(attributes, key, 0)
        if raw is None:
            return 0
        if isinstance(raw, bool):
            return int(raw)
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw)
        if isinstance(raw, str):
            try:
                return int(raw)
            except ValueError:
                return 0
        return 0

    @classmethod
    def _float_attr(cls, attributes: object, key: str, default: float) -> float:
        raw = cls._attr_get(attributes, key, default)
        if raw is None:
            return default
        if isinstance(raw, float):
            return raw
        if isinstance(raw, int):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except ValueError:
                return default
        return default

    @staticmethod
    def _assert_acyclic(root: SpanNode) -> None:
        visiting: set[str] = set()

        def dfs(node: SpanNode) -> None:
            sid = node.span.span_id
            if sid in visiting:
                raise MalformedTraceError(f"cycle detected involving span id {sid!r}")
            visiting.add(sid)
            for ch in node.children:
                dfs(ch)
            visiting.remove(sid)

        dfs(root)
