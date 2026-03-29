import asyncio

import agentrace
from agentrace import traced_node


@traced_node()
async def research_node(state: dict) -> dict:
    return {**state, "research": "some findings"}


async def test():
    async with agentrace.trace(session_id="lg_test", task="test") as ctx:
        await research_node({"query": "hello"})

    t = ctx.agent_trace
    node_spans = [
        s
        for s in t.spans
        if "research_node" in s.span_id or s.input.get("node") == "research_node"
    ]
    print(f"node spans: {len(node_spans)}")  # should be 1
    print(f"span_type: {node_spans[0].span_type}")  # should be 'agent_step'


asyncio.run(test())
