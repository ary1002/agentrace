"""Evaluation task definitions and batched run results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class EvalTask:
    """A single evaluation scenario with optional gold references."""

    id: str
    query: str
    expected_tools: list[str] | None = None
    expected_answer: str | None = None
    optimal_steps: int | None = None
    tags: list[str] = field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"


@dataclass
class TaskResult:
    """Per-task metrics, pass/fail flags, and captured trace."""

    task_id: str
    metric_scores: dict[str, float]
    passed: dict[str, bool]
    failure_types: list[str]
    trace: Any | None = None
    error: str | None = None


@dataclass
class EvalResult:
    """Aggregate outcome for an evaluation run over a dataset."""

    run_id: str
    dataset_id: str
    timestamp: datetime
    task_results: list[TaskResult]
    aggregate_scores: dict[str, float]
    failure_dist: dict[str, int]
    total_cost_usd: float
    total_tokens: int
    duration_s: float
