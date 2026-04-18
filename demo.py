from __future__ import annotations

import asyncio
import json

import agentrace
from agentrace.capture.context_manager import current_tracer
from agentrace.dataset.dataset import Dataset
from agentrace.report.cli_reporter import CLIReporter
from agentrace.runner.models import EvalTask


def _tool_span(name: str, tool_name: str, result: str) -> None:
    """Record one ``tool_call`` span on the active trace (``evaluate``'s context)."""
    tracer = current_tracer()
    with tracer.start_as_current_span(
        name,
        attributes={
            "agentrace.span_type": "tool_call",
            "agentrace.input": json.dumps({"tool_name": tool_name}),
            "agentrace.output": json.dumps({"result": result}),
        },
    ):
        pass


async def my_agent(query: str) -> str:
    """Fake agent that emits tool spans visible to the outer trace."""
    _tool_span("search_tool", "search", "some search result")
    _tool_span("lookup_tool", "lookup", "some lookup result")
    return f"answer for {query}"


dataset = Dataset(
    [
        EvalTask(
            id="t1", query="What is X?", expected_tools=["search"], optimal_steps=3
        ),
        EvalTask(
            id="t2", query="What is Y?", expected_tools=["lookup"], optimal_steps=2
        ),
        EvalTask(
            id="t3",
            query="What is Z?",
            expected_tools=["search", "lookup"],
            optimal_steps=4,
        ),
    ]
)


async def main() -> None:
    result = await agentrace.evaluate(
        agent=my_agent,
        dataset=dataset,
        metrics=["tool_selection_accuracy", "step_efficiency"],
    )
    CLIReporter().print_results(result)
    print("aggregate_scores:", result.aggregate_scores)


if __name__ == "__main__":
    asyncio.run(main())
