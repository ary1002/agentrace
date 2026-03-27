"""Abstract evaluation metric interface and structured ``MetricResult``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class MetricResult:
    """Outcome of applying one metric to a trace (and optional gold task)."""

    metric_name: str
    score: float
    passed: bool
    reason: str
    evidence: list[str]


class BaseMetric(ABC):
    """Async metric with a class-level name and default pass threshold."""

    name: ClassVar[str]
    default_threshold: ClassVar[float]

    @abstractmethod
    async def compute(self, trace: Any, expected: Any | None = None) -> MetricResult:
        """Score ``trace``; ``expected`` is typically an ``EvalTask`` when available."""

    def passes(self, score: float) -> bool:
        return score >= type(self).default_threshold
