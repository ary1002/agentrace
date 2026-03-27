"""Tool selection accuracy: match actual tool spans to expected tool names (exact + fuzzy)."""

from __future__ import annotations

import difflib
from typing import Any

from agentrace.metrics.base import BaseMetric, MetricResult


class ToolSelectionAccuracy(BaseMetric):
    """Fraction of tool calls that align with ``EvalTask.expected_tools``."""

    name = "tool_selection_accuracy"
    default_threshold = 0.80

    def _effective_threshold(self) -> float:
        return getattr(self, "_run_threshold", type(self).default_threshold)

    def _names_match(self, actual: str, expected: str) -> bool:
        if actual == expected:
            return True
        if difflib.SequenceMatcher(None, actual, expected).ratio() >= 0.85:
            return True
        return False

    async def compute(self, trace: Any, expected: Any | None = None) -> MetricResult:
        thr = self._effective_threshold()

        tool_spans = [s for s in trace.spans if s.span_type == "tool_call"]
        actual_calls: list[tuple[Any, str]] = []
        for span in tool_spans:
            raw = span.input.get("tool_name")
            if raw is None:
                raw = span.input.get("name")
            name = "" if raw is None else str(raw)
            actual_calls.append((span, name))

        if expected is None or expected.expected_tools is None:
            return MetricResult(
                metric_name="tool_selection_accuracy",
                score=1.0,
                passed=True,
                reason="No expected tools provided — metric skipped",
                evidence=[],
            )

        expected_tools: list[str] = [str(t) for t in expected.expected_tools]

        if not actual_calls:
            return MetricResult(
                metric_name="tool_selection_accuracy",
                score=0.0,
                passed=False,
                reason="Agent made no tool calls",
                evidence=[],
            )

        matched_flags: list[bool] = []
        for _span, actual_name in actual_calls:
            matched = any(self._names_match(actual_name, exp) for exp in expected_tools)
            matched_flags.append(matched)

        matched_count = sum(1 for m in matched_flags if m)
        denom = max(len(actual_calls), len(expected_tools))
        score = max(0.0, min(1.0, matched_count / denom))

        evidence: list[str] = []
        for (span, tool_name), matched in zip(actual_calls, matched_flags, strict=True):
            status = "matched" if matched else "unexpected"
            evidence.append(f"span {span.span_id}: called '{tool_name}' — {status}")

        unexpected_names = [tool_name for (_, tool_name), m in zip(actual_calls, matched_flags, strict=True) if not m]
        if matched_count == len(actual_calls):
            unexpected_part = ""
        else:
            unexpected_part = f" Unexpected: [{', '.join(repr(n) for n in unexpected_names)}]"

        reason = (
            f"{matched_count}/{len(actual_calls)} tool calls matched expected tools "
            f"[{', '.join(expected_tools)}].{unexpected_part}"
        ).strip()

        passed = score >= thr

        return MetricResult(
            metric_name="tool_selection_accuracy",
            score=score,
            passed=passed,
            reason=reason,
            evidence=evidence,
        )
