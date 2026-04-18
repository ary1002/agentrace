"""Serialise ``TaskResult`` and ``EvalResult`` shells for storage backends."""

from __future__ import annotations

import dataclasses
import json
import warnings
from datetime import datetime
from typing import Any

from agentrace.runner.models import EvalResult, TaskResult


class StorageSerializer:
    @staticmethod
    def _json_default(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)

    @staticmethod
    def task_result_to_dict(result: TaskResult) -> dict[str, Any]:
        trace_json: str | None = None
        if result.trace is not None:
            try:
                trace_dict = dataclasses.asdict(result.trace)
                trace_json = json.dumps(
                    trace_dict, default=StorageSerializer._json_default
                )
            except Exception as e:  # noqa: BLE001 — must not block checkpoint
                warnings.warn(
                    f"Trace serialization failed for task {result.task_id}: {e}",
                    stacklevel=2,
                )
                trace_json = None

        return {
            "task_id": result.task_id,
            "metric_scores": json.dumps(result.metric_scores),
            "passed": json.dumps(result.passed),
            "failure_types": json.dumps(result.failure_types),
            "error": result.error,
            "trace_json": trace_json,
        }

    @staticmethod
    def task_result_from_dict(row: dict[str, Any]) -> TaskResult:
        def _loads_str(key: str, default: str) -> Any:
            raw = row.get(key, default)
            if isinstance(raw, (dict, list)):
                return raw
            return json.loads(raw if isinstance(raw, str) else default)

        metric_scores = _loads_str("metric_scores", "{}")
        passed = _loads_str("passed", "{}")
        failure_types = _loads_str("failure_types", "[]")
        if not isinstance(metric_scores, dict):
            metric_scores = {}
        if not isinstance(passed, dict):
            passed = {}
        if not isinstance(failure_types, list):
            failure_types = []

        err = row.get("error")
        error_str: str | None = err if err is None or isinstance(err, str) else str(err)

        return TaskResult(
            task_id=str(row["task_id"]),
            metric_scores=metric_scores,
            passed=passed,
            failure_types=failure_types,
            trace=None,
            error=error_str,
        )

    @staticmethod
    def eval_result_to_dict(result: EvalResult) -> dict[str, Any]:
        return {
            "run_id": result.run_id,
            "dataset_id": result.dataset_id,
            "timestamp": result.timestamp.isoformat()
            if isinstance(result.timestamp, datetime)
            else str(result.timestamp),
            "aggregate_scores": json.dumps(result.aggregate_scores),
            "failure_dist": json.dumps(result.failure_dist),
            "total_cost_usd": float(result.total_cost_usd),
            "total_tokens": int(result.total_tokens),
            "duration_s": float(result.duration_s),
        }

    @staticmethod
    def eval_result_from_dict(row: dict[str, Any]) -> EvalResult:
        ts = row["timestamp"]
        if isinstance(ts, datetime):
            timestamp = ts
        else:
            timestamp = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))

        def _loads_scores(key: str, default: str) -> Any:
            raw = row.get(key, default)
            if isinstance(raw, dict):
                return raw
            return json.loads(raw if isinstance(raw, str) else default)

        agg = _loads_scores("aggregate_scores", "{}")
        fd = _loads_scores("failure_dist", "{}")
        if not isinstance(agg, dict):
            agg = {}
        if not isinstance(fd, dict):
            fd = {}

        return EvalResult(
            run_id=str(row["run_id"]),
            dataset_id=str(row["dataset_id"]),
            timestamp=timestamp,
            task_results=[],
            aggregate_scores={str(k): float(v) for k, v in agg.items()},
            failure_dist={str(k): int(v) for k, v in fd.items()},
            total_cost_usd=float(row.get("total_cost_usd", 0.0)),
            total_tokens=int(row.get("total_tokens", 0)),
            duration_s=float(row.get("duration_s", 0.0)),
        )
