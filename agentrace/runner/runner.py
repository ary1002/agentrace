"""Async ``evaluate()`` orchestration over a ``Dataset`` with tracing and metrics."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Awaitable, Callable, Any

import agentrace
from agentrace.dataset.dataset import Dataset
from agentrace.metrics.deterministic import METRICS_REGISTRY
from agentrace.runner.models import EvalResult, TaskResult


class _EvalJSONEncoder(json.JSONEncoder):
    """JSON encoder for evaluation exports: ``datetime`` as ISO, else ``str``."""

    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)


async def evaluate(
    agent: Callable[[str], Awaitable[str]],
    dataset: Dataset,
    metrics: list[str],
    judge_model: str = "claude-sonnet-4-20250514",
    concurrency: int = 1,
    checkpoint_dir: str | None = None,
    output_dir: str | None = "./eval_results/",
    thresholds: dict[str, float] | None = None,
) -> EvalResult:
    """Run each task with ``agentrace.trace``, score with named metrics, aggregate results."""

    _ = judge_model, concurrency, checkpoint_dir

    resolved_metrics: list[Any] = []
    for name in metrics:
        if name not in METRICS_REGISTRY:
            raise ValueError(
                f"Unknown metric: '{name}'. Available: {list(METRICS_REGISTRY.keys())}"
            )
        template = METRICS_REGISTRY[name]
        inst = type(template)()
        if thresholds is not None and inst.name in thresholds:
            setattr(inst, "_run_threshold", float(thresholds[inst.name]))
        else:
            setattr(inst, "_run_threshold", float(type(inst).default_threshold))
        resolved_metrics.append(inst)

    task_results: list[TaskResult] = []
    start_time = time.monotonic()

    for task in dataset:
        try:
            async with agentrace.trace(session_id=task.id, task=task.query) as t:
                await agent(task.query)
            agent_trace = t.agent_trace

            metric_scores: dict[str, float] = {}
            passed: dict[str, bool] = {}
            for metric in resolved_metrics:
                result = await metric.compute(agent_trace, task)
                metric_scores[metric.name] = result.score
                passed[metric.name] = result.passed

            task_results.append(
                TaskResult(
                    task_id=task.id,
                    metric_scores=metric_scores,
                    passed=passed,
                    failure_types=[],
                    trace=agent_trace,
                    error=None,
                )
            )
        except Exception as e:
            task_results.append(
                TaskResult(
                    task_id=task.id,
                    metric_scores={},
                    passed={},
                    failure_types=[],
                    trace=None,
                    error=str(e),
                )
            )

    aggregate_scores: dict[str, float] = {}
    for metric in resolved_metrics:
        name = metric.name
        values = [
            r.metric_scores[name]
            for r in task_results
            if name in r.metric_scores
        ]
        if values:
            aggregate_scores[name] = sum(values) / len(values)

    total_cost_usd = sum(
        r.trace.total_cost_usd for r in task_results if r.trace is not None
    )
    total_tokens = sum(
        r.trace.total_tokens for r in task_results if r.trace is not None
    )
    duration_s = time.monotonic() - start_time

    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    result = EvalResult(
        run_id=run_id,
        dataset_id=dataset.id,
        timestamp=datetime.utcnow(),
        task_results=task_results,
        aggregate_scores=aggregate_scores,
        failure_dist={},
        total_cost_usd=total_cost_usd,
        total_tokens=total_tokens,
        duration_s=duration_s,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{run_id}.json")
        payload = asdict(result)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, cls=_EvalJSONEncoder, indent=2)

    return result
