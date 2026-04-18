"""Versioned JSON export of ``EvalResult`` for diffing and HTML comparison."""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime
from typing import Any

from agentrace.runner.models import EvalResult, TaskResult

SCHEMA_VERSION = "1.0"


class _ExportJSONEncoder(json.JSONEncoder):
    """Handle datetime, dataclasses, sets, and fall back to str."""

    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, set):
            return sorted(o)
        return str(o)


class JSONReporter:
    """Write and read schema-versioned evaluation JSON (no traces)."""

    def write(
        self,
        result: EvalResult,
        output_dir: str = "./eval_results/",
    ) -> str:
        """Serialise ``EvalResult`` to JSON under ``{output_dir}/{run_id}.json``."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{result.run_id}.json")
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": result.run_id,
            "dataset_id": result.dataset_id,
            "timestamp": result.timestamp,
            "duration_s": result.duration_s,
            "total_cost_usd": result.total_cost_usd,
            "total_tokens": result.total_tokens,
            "aggregate_scores": dict(result.aggregate_scores),
            "failure_dist": dict(result.failure_dist),
            "task_results": [
                {
                    "task_id": tr.task_id,
                    "metric_scores": dict(tr.metric_scores),
                    "passed": dict(tr.passed),
                    "failure_types": list(tr.failure_types),
                    "error": tr.error,
                }
                for tr in result.task_results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, cls=_ExportJSONEncoder, indent=2)
        return path

    def read(self, path: str) -> EvalResult:
        """Load JSON written by ``write`` back into an ``EvalResult``."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported JSON schema version: expected '{SCHEMA_VERSION}', "
                f"found {version!r}. Re-export with the current AgentTrace version."
            )

        ts_raw = data["timestamp"]
        if isinstance(ts_raw, str):
            timestamp = datetime.fromisoformat(ts_raw)
        else:
            raise ValueError("timestamp must be an ISO format string")

        task_results: list[TaskResult] = []
        for row in data["task_results"]:
            task_results.append(
                TaskResult(
                    task_id=str(row["task_id"]),
                    metric_scores={
                        str(k): float(v) for k, v in row["metric_scores"].items()
                    },
                    passed={str(k): bool(v) for k, v in row["passed"].items()},
                    failure_types=[str(x) for x in row["failure_types"]],
                    trace=None,
                    error=(None if row.get("error") is None else str(row["error"])),
                )
            )

        return EvalResult(
            run_id=str(data["run_id"]),
            dataset_id=str(data["dataset_id"]),
            timestamp=timestamp,
            task_results=task_results,
            aggregate_scores={
                str(k): float(v) for k, v in data["aggregate_scores"].items()
            },
            failure_dist={str(k): int(v) for k, v in data["failure_dist"].items()},
            total_cost_usd=float(data["total_cost_usd"]),
            total_tokens=int(data["total_tokens"]),
            duration_s=float(data["duration_s"]),
        )
