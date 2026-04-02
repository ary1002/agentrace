"""Argument correctness: structural checks on tool_call span inputs vs expected shapes."""

from __future__ import annotations

from typing import Any

from agentrace.metrics.base import BaseMetric, MetricResult


def _tool_input_dict(inp: Any) -> dict[str, Any]:
    if isinstance(inp, dict):
        return inp
    return {}


class ArgumentCorrectness(BaseMetric):
    """
    For known tools (web_search: requires ``query``), count correct argument payloads.
    Score = correct / max(1, applicable_tool_calls).
    """

    name = "argument_correctness"
    default_threshold = 0.70

    def _effective_threshold(self) -> float:
        return getattr(self, "_run_threshold", type(self).default_threshold)

    def _check_args(self, tool_name: str, inp: dict[str, Any]) -> tuple[bool, str]:
        t = tool_name.lower()
        if "search" in t or t == "web_search":
            args = inp.get("args") if isinstance(inp.get("args"), dict) else inp
            if not isinstance(args, dict):
                args = inp
            if args.get("query") not in (None, ""):
                return True, "web_search has query"
            if args.get("input") and isinstance(args.get("input"), str):
                return True, "web_search args in input field"
            return False, "web_search missing query"
        if "calculat" in t or t == "calculate":
            args = inp.get("args") if isinstance(inp.get("args"), dict) else inp
            if not isinstance(args, dict):
                args = inp
            expr = args.get("expression") or args.get("input")
            if expr not in (None, ""):
                return True, "calculator has expression"
            return False, "calculator missing expression"
        return True, f"unknown_tool_{tool_name}_skipped"

    async def compute(
        self,
        trace: Any,
        expected: Any | None = None,
        judge: Any | None = None,
    ) -> MetricResult:
        thr = self._effective_threshold()
        tool_spans = [s for s in trace.spans if s.span_type == "tool_call"]
        if not tool_spans:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="No tool calls — nothing to validate",
                evidence=[],
            )

        ok = 0
        evidence: list[str] = []
        for s in tool_spans:
            inp = _tool_input_dict(s.input)
            raw = inp.get("tool_name") or inp.get("name") or ""
            name = str(raw) if raw else "tool"
            inner = inp.get("input")
            if isinstance(inner, str):
                try:
                    import json

                    parsed = json.loads(inner)
                    if isinstance(parsed, dict):
                        inp = {**inp, **parsed}
                except (json.JSONDecodeError, TypeError):
                    pass
            good, msg = self._check_args(name, inp)
            if good:
                ok += 1
            evidence.append(f"{s.span_id}: {name} -> {msg}")

        score = ok / len(tool_spans)
        reason = f"{ok}/{len(tool_spans)} tool calls have plausible arguments"
        passed = score >= thr
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            evidence=evidence[:25],
        )
