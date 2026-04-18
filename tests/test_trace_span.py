from __future__ import annotations

import asyncio

import pytest

import agentrace


@pytest.mark.asyncio
async def test_span_set_output_visible_in_agent_trace() -> None:
    async with agentrace.trace(session_id="sid", task="q1") as t:
        with agentrace.span("tool_call", tool="pubmed_search") as sp:
            sp.set_output({"hits": 3})
    assert t.agent_trace is not None
    tool_spans = [s for s in t.agent_trace.spans if s.span_type == "tool_call"]
    assert tool_spans
    assert any(s.output.get("hits") == 3 for s in tool_spans)


@pytest.mark.asyncio
async def test_span_set_output_scalar_wrapped() -> None:
    async with agentrace.trace(session_id="sid2", task="q2") as t:
        with agentrace.span("tool_call", tool="t") as sp:
            sp.set_output("plain-result")
    assert t.agent_trace is not None
    assert any(s.output.get("output") == "plain-result" for s in t.agent_trace.spans)


@pytest.mark.asyncio
async def test_span_forwards_otel_set_attribute() -> None:
    async with agentrace.trace(session_id="sid3", task="q3") as t:
        with agentrace.span("step", span_type="agent_step") as sp:
            sp.set_attribute("custom.key", "v")
    assert t.agent_trace is not None
