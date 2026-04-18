"""LLM-judge metric: trajectory optimality (TOS) — headline score."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentrace.metrics.base import BaseMetric, MetricResult

if TYPE_CHECKING:
    from agentrace.metrics.llm_judge.judge_client import JudgeClient
    from agentrace.normalizer.models import AgentTrace, Span


def _tool_name(span: Span) -> str:
    if span.span_type != "tool_call":
        return ""
    raw = span.input.get("tool_name")
    if raw is None:
        raw = span.input.get("name")
    return "?" if raw is None else str(raw)


def _summarize(obj: Any, max_len: int) -> str:
    s = str(obj) if obj is not None else ""
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


class TrajectoryOptimality(BaseMetric):
    name = "trajectory_optimality"
    default_threshold = 0.65

    def _build_prompt(self, trace: AgentTrace) -> str:
        ordered = sorted(trace.spans, key=lambda s: s.timestamp)
        lines: list[str] = []
        for i, span in enumerate(ordered, start=1):
            tn = _tool_name(span) if span.span_type == "tool_call" else "N/A"
            lines.append(
                f"{i}. span_id={span.span_id} type={span.span_type} tool={tn} "
                f"in={_summarize(span.input, 100)!r} out={_summarize(span.output, 100)!r} "
                f"latency_ms={span.latency_ms:.1f}"
            )
        formatted_trace = "\n".join(lines) if lines else "(empty)"
        return f"""You are an expert AI agent evaluator assessing trajectory optimality.

Task: {trace.task}

Agent execution trace ({len(trace.spans)} steps):
{formatted_trace}

Your job:
1. Determine the OPTIMAL sequence of steps to complete this task
2. Score how closely the agent followed the optimal path (0.0 = completely suboptimal, 1.0 = optimal or better)
3. Identify specific wasted or suboptimal steps by their span_id

Consider wasted: redundant tool calls, wrong tools, unnecessary LLM calls, steps that did not contribute to the final answer.

Respond ONLY with a JSON object:
{{
  "score": <float 0.0-1.0>,
  "optimal_path": [<list of step description strings — the ideal sequence>],
  "actual_steps": <int>,
  "optimal_steps": <int>,
  "wasted_steps": [<list of span_id strings that were wasteful>],
  "deviation_reason": "<one sentence — main reason the agent deviated from optimal>",
  "reasoning": "<two sentence overall explanation>"
}}"""

    def _parse_response(self, parsed: dict, trace: AgentTrace) -> MetricResult:
        score = float(parsed["score"])
        actual = parsed.get("actual_steps", len(trace.spans))
        optimal = parsed.get("optimal_steps", actual)
        wasted = parsed.get("wasted_steps", [])
        if not isinstance(wasted, list):
            wasted = []
        wasted_strs = [str(sid) for sid in wasted]
        optimal_path = parsed.get("optimal_path", [])
        if not isinstance(optimal_path, list):
            optimal_path = []
        optimal_path_strs = [str(x) for x in optimal_path]
        deviation = str(parsed.get("deviation_reason", ""))
        evidence = [f"wasted span: {sid}" for sid in wasted_strs]
        reason = (
            f"Actual: {actual} steps, Optimal: {optimal} steps. "
            f"{parsed.get('deviation_reason', '')} "
            f"{parsed.get('reasoning', '')}"
        )
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=self.passes(score),
            reason=reason,
            evidence=evidence,
            wasted_steps=wasted_strs,
            optimal_path=optimal_path_strs,
            deviation_reason=deviation,
        )

    async def compute(
        self,
        trace: Any,
        expected: Any | None = None,
        judge: JudgeClient | None = None,
    ) -> MetricResult:
        if judge is None:
            raise ValueError("TrajectoryOptimality requires a JudgeClient")

        prompt = self._build_prompt(trace)
        response = await judge.judge(
            prompt,
            response_schema={
                "score": "float",
                "optimal_path": "list",
                "actual_steps": "int",
                "optimal_steps": "int",
                "wasted_steps": "list",
                "deviation_reason": "str",
                "reasoning": "str",
            },
        )
        return self._parse_response(response.parsed, trace)
