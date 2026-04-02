"""Cost per task: map trace cost to [0,1] using a soft USD ceiling (lower cost = higher score)."""

from __future__ import annotations

from typing import Any

from agentrace.metrics.base import BaseMetric, MetricResult


class CostPerTask(BaseMetric):
    """Score = min(1, ceiling_usd / max(actual_cost, epsilon))."""

    name = "cost_per_task"
    default_threshold = 0.50
    _CEILING_USD = 0.05

    def _effective_threshold(self) -> float:
        return getattr(self, "_run_threshold", type(self).default_threshold)

    async def compute(
        self,
        trace: Any,
        expected: Any | None = None,
        judge: Any | None = None,
    ) -> MetricResult:
        thr = self._effective_threshold()
        cost = float(trace.total_cost_usd)
        eps = 1e-9
        score = min(1.0, self._CEILING_USD / max(cost, eps))
        reason = f"total_cost_usd={cost:.6f}, ceiling={self._CEILING_USD}, score={score:.3f}"
        evidence = [reason]
        passed = score >= thr
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            evidence=evidence,
        )
