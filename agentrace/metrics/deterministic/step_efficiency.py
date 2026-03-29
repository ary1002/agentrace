"""Step efficiency: compare span count to ``EvalTask.optimal_steps``."""

from __future__ import annotations

from collections import Counter
from typing import Any

from agentrace.metrics.base import BaseMetric, MetricResult


class StepEfficiency(BaseMetric):
    """Ratio of optimal steps to actual span count (capped at 1.0 when better than optimal)."""

    name = "step_efficiency"
    default_threshold = 0.70

    def _effective_threshold(self) -> float:
        return getattr(self, "_run_threshold", type(self).default_threshold)

    async def compute(
        self,
        trace: Any,
        expected: Any | None = None,
        judge: Any | None = None,
    ) -> MetricResult:
        thr = self._effective_threshold()

        actual_steps = len(trace.spans)

        if expected is None or expected.optimal_steps is None:
            return MetricResult(
                metric_name="step_efficiency",
                score=1.0,
                passed=True,
                reason="No optimal_steps provided — metric skipped",
                evidence=[],
            )

        optimal = int(expected.optimal_steps)
        if actual_steps == 0:
            score = 0.0
        else:
            score = min(1.0, optimal / actual_steps)

        wasted_steps = max(0, actual_steps - optimal)

        counts = Counter(s.span_type for s in trace.spans)
        label = {
            "llm_call": "llm_calls",
            "tool_call": "tool_calls",
            "memory_read": "memory_reads",
            "memory_write": "memory_writes",
            "agent_step": "agent_steps",
        }
        type_parts = [f"{label.get(stype, stype)}={counts[stype]}" for stype in sorted(counts.keys())]
        evidence = [
            f"actual_steps={actual_steps}, optimal_steps={optimal}, wasted={wasted_steps}",
            ", ".join(type_parts) if type_parts else "no spans by type",
        ]

        if score >= 1.0 and actual_steps <= optimal:
            reason = f"Agent completed task in {actual_steps} steps (optimal or better)"
        elif score >= thr:
            reason = f"Agent used {actual_steps} steps vs optimal {optimal} ({wasted_steps} wasted)"
        else:
            reason = (
                f"Agent was significantly inefficient: {actual_steps} steps vs optimal {optimal}"
            )

        passed = score >= thr

        return MetricResult(
            metric_name="step_efficiency",
            score=score,
            passed=passed,
            reason=reason,
            evidence=evidence,
        )
