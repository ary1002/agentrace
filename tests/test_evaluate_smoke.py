from __future__ import annotations

import pytest

from agentrace.dataset.dataset import Dataset
from agentrace.runner.models import EvalTask
from agentrace.runner.runner import evaluate


@pytest.mark.asyncio
async def test_evaluate_smoke_with_mock_agent(tmp_path) -> None:
    async def mock_agent(query: str) -> str:
        return f"answer:{query}"

    dataset = Dataset(
        [
            EvalTask(id="task-1", query="hello", expected_tools=None),
            EvalTask(id="task-2", query="world", expected_tools=None),
        ]
    )
    out = await evaluate(
        agent=mock_agent,
        dataset=dataset,
        metrics=["latency_p50", "cost_per_task"],
        concurrency=2,
        output_dir=str(tmp_path / "reports"),
        storage_config={"backend": "sqlite", "path": str(tmp_path / "agentrace.db")},
        run_id="smoke_run",
    )
    assert len(out.task_results) == 2
    assert "latency_p50" in out.aggregate_scores
