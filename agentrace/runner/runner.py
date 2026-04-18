"""Async ``evaluate()`` orchestration over a ``Dataset`` with tracing and metrics."""

from __future__ import annotations

import asyncio
import os
import time
import warnings
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import agentrace
from agentrace.classifier import FailureClassifier
from agentrace.dataset.dataset import Dataset
from agentrace.metrics.base import MetricResult
from agentrace.metrics.deterministic import LLM_METRIC_NAMES, METRICS_REGISTRY
from agentrace.metrics.llm_judge.judge_client import JudgeClient
from agentrace.report.html_reporter import HTMLReporter
from agentrace.report.json_reporter import JSONReporter
from agentrace.runner.checkpoint import CheckpointManager
from agentrace.runner.models import EvalResult, EvalTask, TaskResult
from agentrace.storage import get_storage


async def evaluate(
    agent: Callable[[str], Awaitable[str]],
    dataset: Dataset,
    metrics: list[str],
    judge_model: str = "claude-sonnet-4-20250514",
    concurrency: int = 10,
    checkpoint_dir: str | None = None,
    output_dir: str | None = "./eval_results/",
    thresholds: dict[str, float] | None = None,
    storage_config: dict | None = None,
    run_id: str | None = None,
    timeout_per_task: float | None = None,
) -> EvalResult:
    """Run each task with ``agentrace.trace``, score with named metrics, aggregate results.

    If ``timeout_per_task`` is set (seconds), each ``await agent(query)`` is bounded by
    :func:`asyncio.wait_for`; on expiry the task is recorded as failed with a clear error.
    """

    if run_id is not None:
        resolved_run_id = run_id
    else:
        resolved_run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    cfg = dict(storage_config) if storage_config else {"backend": "sqlite", "path": "./agentrace.db"}
    backend = str(cfg.get("backend", "sqlite"))
    sqlite_path = str(cfg.get("path", "./agentrace.db"))

    if checkpoint_dir:
        if backend != "sqlite":
            warnings.warn(
                f"checkpoint_dir is set but storage backend is {backend!r}; "
                "checkpoint data is stored via storage (e.g. storage.path / DSN), not checkpoint_dir.",
                UserWarning,
                stacklevel=2,
            )
        elif sqlite_path != "./agentrace.db":
            warnings.warn(
                "checkpoint_dir is set together with a non-default storage.path; "
                "resume uses storage.path only. Put the DB under checkpoint_dir or omit checkpoint_dir.",
                UserWarning,
                stacklevel=2,
            )
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
            cfg["path"] = str(Path(checkpoint_dir) / "agentrace.db")

    storage = get_storage(cfg)
    await storage.connect()
    try:
        checkpoint = CheckpointManager(resolved_run_id, storage)
        already_done = await checkpoint.load_existing()
        if already_done:
            print(
                f"Resuming run {resolved_run_id} — "
                f"{len(already_done)}/{len(dataset)} tasks already complete, skipping."
            )

        await storage.save_run_meta(
            EvalResult(
                run_id=resolved_run_id,
                dataset_id=dataset.id,
                timestamp=datetime.now(timezone.utc),
                task_results=[],
                aggregate_scores={},
                failure_dist={},
                total_cost_usd=0.0,
                total_tokens=0,
                duration_s=0.0,
            )
        )

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

        needs_judge = any(m.name in LLM_METRIC_NAMES for m in resolved_metrics)
        judge = JudgeClient(model=judge_model, temperature=0.0) if needs_judge else None
        classifier = FailureClassifier(judge=judge, run_stage2=(judge is not None))

        semaphore = asyncio.Semaphore(concurrency)
        start_time = time.monotonic()
        task_order = {t.id: i for i, t in enumerate(dataset)}

        async def run_one(task: EvalTask) -> TaskResult | None:
            if checkpoint.is_complete(task.id):
                return None

            async with semaphore:
                try:
                    async with agentrace.trace(session_id=task.id, task=task.query) as t:
                        try:
                            if timeout_per_task is not None:
                                await asyncio.wait_for(
                                    agent(task.query),
                                    timeout=float(timeout_per_task),
                                )
                            else:
                                await agent(task.query)
                        except asyncio.TimeoutError:
                            raise RuntimeError(
                                f"Agent exceeded timeout_per_task ({float(timeout_per_task)}s)"
                            ) from None

                    agent_trace = t.agent_trace
                    if agent_trace is None:
                        raise RuntimeError("No agent trace captured for task")
                    metric_results: dict[str, MetricResult] = {}
                    for metric in resolved_metrics:
                        r = await metric.compute(agent_trace, task, judge=judge)
                        metric_results[metric.name] = r
                    pass_flags = [r.passed for r in metric_results.values()]
                    if pass_flags and all(pass_flags):
                        agent_trace.outcome = "success"
                    elif pass_flags and any(pass_flags):
                        agent_trace.outcome = "partial"
                    elif pass_flags:
                        agent_trace.outcome = "failure"
                    else:
                        agent_trace.outcome = "success"

                    wasted: list[str] = []
                    for r in metric_results.values():
                        wasted.extend(getattr(r, "wasted_steps", []))

                    failure_records = await classifier.classify(
                        agent_trace,
                        task.id,
                        wasted_step_ids=wasted,
                    )

                    task_result = TaskResult(
                        task_id=task.id,
                        metric_scores={n: r.score for n, r in metric_results.items()},
                        passed={n: r.passed for n, r in metric_results.items()},
                        failure_types=[r.failure_type.value for r in failure_records],
                        trace=agent_trace,
                        error=None,
                    )
                except Exception as e:
                    task_result = TaskResult(
                        task_id=task.id,
                        metric_scores={},
                        passed={},
                        failure_types=[],
                        trace=None,
                        error=str(e),
                    )

                await checkpoint.save(task_result)
                return task_result

        new_results = await asyncio.gather(
            *[run_one(task) for task in dataset],
            return_exceptions=False,
        )

        all_results: list[TaskResult] = already_done + [
            r for r in new_results if r is not None
        ]
        all_results.sort(key=lambda r: task_order.get(r.task_id, 10**9))

        aggregate_scores: dict[str, float] = {}
        for metric in resolved_metrics:
            name = metric.name
            values = [
                r.metric_scores[name]
                for r in all_results
                if name in r.metric_scores
            ]
            if values:
                aggregate_scores[name] = sum(values) / len(values)

        total_cost_usd = sum(
            r.trace.total_cost_usd for r in all_results if r.trace is not None
        )
        total_tokens = sum(
            r.trace.total_tokens for r in all_results if r.trace is not None
        )
        duration_s = time.monotonic() - start_time

        all_failure_types = [ft for tr in all_results for ft in tr.failure_types]
        failure_dist = dict(Counter(all_failure_types))

        final_result = EvalResult(
            run_id=resolved_run_id,
            dataset_id=dataset.id,
            timestamp=datetime.now(timezone.utc),
            task_results=all_results,
            aggregate_scores=aggregate_scores,
            failure_dist=failure_dist,
            total_cost_usd=total_cost_usd,
            total_tokens=total_tokens,
            duration_s=duration_s,
        )

        await storage.save_run_meta(final_result)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            json_reporter = JSONReporter()
            json_reporter.write(final_result, output_dir)

            prev_result = None
            try:
                runs = await storage.list_runs(dataset_id=dataset.id)
                other_runs = [r for r in runs if r["run_id"] != resolved_run_id]
                if other_runs:
                    prev_json = os.path.join(
                        output_dir, f"{other_runs[0]['run_id']}.json"
                    )
                    if os.path.exists(prev_json):
                        prev_result = json_reporter.read(prev_json)
            except Exception:
                pass

            html_reporter = HTMLReporter()
            html_reporter.generate(final_result, prev_result, output_dir)

        return final_result
    finally:
        await storage.disconnect()
