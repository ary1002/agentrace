"""Redundancy rate: penalize near-duplicate consecutive tool calls (same tool, similar args)."""

from __future__ import annotations

import difflib
from typing import Any

from agentrace.metrics.base import BaseMetric, MetricResult


class RedundancyRate(BaseMetric):
    """1.0 minus normalized count of redundant tool pairs (higher is better — less redundancy)."""

    name = "redundancy_rate"
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
        tool_spans = [s for s in sorted(trace.spans, key=lambda s: s.timestamp) if s.span_type == "tool_call"]
        if len(tool_spans) < 2:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="Fewer than 2 tool calls — no redundancy to measure",
                evidence=[],
            )

        redundant_pairs: list[tuple[str, str]] = []
        for i in range(1, len(tool_spans)):
            prev, cur = tool_spans[i - 1], tool_spans[i]
            raw_p = prev.input.get("tool_name") or prev.input.get("name")
            raw_c = cur.input.get("tool_name") or cur.input.get("name")
            name_p = "" if raw_p is None else str(raw_p)
            name_c = "" if raw_c is None else str(raw_c)
            if name_p != name_c or not name_p:
                continue
            ratio = difflib.SequenceMatcher(None, str(prev.input), str(cur.input)).ratio()
            if ratio > 0.85:
                redundant_pairs.append((prev.span_id, cur.span_id))

        n_tools = len(tool_spans)
        max_pairs = max(1, n_tools - 1)
        redundancy_fraction = len(redundant_pairs) / max_pairs
        score = max(0.0, min(1.0, 1.0 - redundancy_fraction))

        evidence = [f"redundant_pair_{i}: {a} -> {b}" for i, (a, b) in enumerate(redundant_pairs)]
        if not evidence:
            evidence = ["no_near_duplicate_tool_calls"]

        reason = (
            f"{len(redundant_pairs)} redundant tool pair(s) of {max_pairs} possible; "
            f"score={score:.2f}"
        )
        passed = score >= thr
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            evidence=evidence[:20],
        )
