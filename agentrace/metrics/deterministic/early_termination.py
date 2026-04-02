"""Early termination rate: penalize stopping before expected tool use or with very short traces."""

from __future__ import annotations

from typing import Any

from agentrace.metrics.base import BaseMetric, MetricResult


class EarlyTerminationRate(BaseMetric):
    """
    Score 1.0 when the agent appears to have run a full trajectory; lower when it stops
    early (no tool calls despite expected_tools, or very few spans vs optimal_steps).
    """

    name = "early_termination_rate"
    default_threshold = 0.60

    def _effective_threshold(self) -> float:
        return getattr(self, "_run_threshold", type(self).default_threshold)

    async def compute(
        self,
        trace: Any,
        expected: Any | None = None,
        judge: Any | None = None,
    ) -> MetricResult:
        thr = self._effective_threshold()
        n = len(trace.spans)
        tool_spans = [s for s in trace.spans if s.span_type == "tool_call"]

        penalties = 0.0
        evidence: list[str] = []

        if expected is not None and expected.expected_tools:
            exp_tools = [str(t) for t in expected.expected_tools]
            if not tool_spans:
                penalties += 0.7
                evidence.append(f"no_tool_calls_but_expected={exp_tools}")
            else:
                names = []
                for s in tool_spans:
                    raw = s.input.get("tool_name") or s.input.get("name")
                    names.append("" if raw is None else str(raw))
                matched = any(
                    any(et in n or n in et for et in exp_tools) for n in names if n
                )
                if not matched:
                    penalties += 0.4
                    evidence.append(f"tool_calls={names!r}_miss_expected={exp_tools!r}")

        if expected is not None and expected.optimal_steps is not None:
            opt = int(expected.optimal_steps)
            if n > 0 and n < max(2, opt // 2):
                penalties += 0.3
                evidence.append(f"span_count={n}_below_half_of_optimal={opt}")

        if n < 2:
            penalties += 0.2
            evidence.append("fewer_than_2_spans")

        score = max(0.0, min(1.0, 1.0 - penalties))
        if not evidence:
            evidence = ["trajectory_looks_complete"]

        reason = f"early_termination_penalty={penalties:.2f} -> score={score:.2f}"
        passed = score >= thr
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            evidence=evidence,
        )
