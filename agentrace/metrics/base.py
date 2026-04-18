"""Abstract evaluation metric interface and structured ``MetricResult``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class MetricResult:
    """Outcome of applying one metric to a trace (and optional gold task)."""

    metric_name: str
    score: float
    passed: bool
    reason: str
    evidence: list[str]
    wasted_steps: list[str] = field(default_factory=list)
    optimal_path: list[str] = field(default_factory=list)
    deviation_reason: str = ""


class BaseMetric(ABC):
    """Async metric with a class-level name and default pass threshold."""

    name: ClassVar[str]
    default_threshold: ClassVar[float]
    _run_threshold: float

    def __init__(self) -> None:
        self._run_threshold = float(type(self).default_threshold)

    @abstractmethod
    async def compute(
        self,
        trace: Any,
        expected: Any | None = None,
        judge: Any | None = None,
    ) -> MetricResult:
        """Score ``trace``; ``expected`` is typically an ``EvalTask`` when available."""

    def passes(self, score: float) -> bool:
        return score >= type(self).default_threshold
