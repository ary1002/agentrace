"""Token efficiency: reward lower total token usage relative to a soft reference."""

from __future__ import annotations

from typing import Any

from agentrace.metrics.base import BaseMetric, MetricResult


class TokenEfficiency(BaseMetric):
    """
    Maps total trace tokens to [0,1] using a reference budget derived from optimal_steps
    (default 500 tokens per optimal step) or a floor of 500 tokens.
    """

    name = "token_efficiency"
    default_threshold = 0.50
    _TOKENS_PER_OPTIMAL_STEP = 500

    def _effective_threshold(self) -> float:
        return getattr(self, "_run_threshold", type(self).default_threshold)

    async def compute(
        self,
        trace: Any,
        expected: Any | None = None,
        judge: Any | None = None,
    ) -> MetricResult:
        thr = self._effective_threshold()
        total = int(trace.total_tokens)
        if total <= 0:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reason="No token usage recorded on trace",
                evidence=["total_tokens=0"],
            )

        if expected is not None and expected.optimal_steps is not None:
            budget = max(
                200, int(expected.optimal_steps) * self._TOKENS_PER_OPTIMAL_STEP
            )
        else:
            budget = 2000

        score = min(1.0, budget / total)
        reason = f"total_tokens={total}, reference_budget={budget}, score={score:.3f}"
        evidence = [reason]
        passed = score >= thr
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            evidence=evidence,
        )
