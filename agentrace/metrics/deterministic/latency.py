"""Latency percentiles over span durations; score vs a reference latency (lower pXX = higher score)."""

from __future__ import annotations

from typing import Any

from agentrace.metrics.base import BaseMetric, MetricResult


class _LatencyPercentileBase(BaseMetric):
    percentile: int = 95
    reference_ms: float = 30_000.0

    def _effective_threshold(self) -> float:
        return getattr(self, "_run_threshold", type(self).default_threshold)

    async def compute(
        self,
        trace: Any,
        expected: Any | None = None,
        judge: Any | None = None,
    ) -> MetricResult:
        thr = self._effective_threshold()
        latencies = sorted(s.latency_ms for s in trace.spans if s.latency_ms > 0)
        if not latencies:
            return MetricResult(
                metric_name=self.name,
                score=0.5,
                passed=True,
                reason="No positive latencies — neutral score",
                evidence=["no_span_latencies"],
            )

        k = max(0, min(len(latencies) - 1, int(round((self.percentile / 100.0) * (len(latencies) - 1)))))
        p_ms = float(latencies[k])
        score = min(1.0, self.reference_ms / max(p_ms, 1.0))
        reason = f"p{self.percentile}={p_ms:.1f}ms vs ref={self.reference_ms:.0f}ms -> score={score:.3f}"
        evidence = [reason, f"span_count={len(trace.spans)}"]
        passed = score >= thr
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            evidence=evidence,
        )


class LatencyP50(_LatencyPercentileBase):
    name = "latency_p50"
    default_threshold = 0.50
    percentile = 50


class LatencyP95(_LatencyPercentileBase):
    name = "latency_p95"
    default_threshold = 0.50
    percentile = 95
