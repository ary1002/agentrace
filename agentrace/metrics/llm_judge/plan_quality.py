"""LLM-judge metric: quality of the agent's initial plan (first llm_call output)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentrace.metrics.base import BaseMetric, MetricResult

if TYPE_CHECKING:
    from agentrace.metrics.llm_judge.judge_client import JudgeClient
    from agentrace.normalizer.models import AgentTrace


def _summarize(obj: Any, max_len: int) -> str:
    s = str(obj) if obj is not None else ""
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


class PlanQuality(BaseMetric):
    name = "plan_quality"
    default_threshold = 0.70

    def _build_prompt(self, trace: AgentTrace, first_llm_output: str) -> str:
        return f"""You are evaluating the quality of an AI agent's initial plan for a task.

Task: {trace.task}

Agent's initial response / plan:
{first_llm_output}

Assess the plan on:
- Appropriateness: does it address the right subtasks?
- Completeness: does it cover all necessary steps?
- Efficiency: does it avoid unnecessary steps?
- Tool awareness: does it plan to use appropriate tools?

Respond ONLY with a JSON object:
{{
  "score": <float 0.0-1.0>,
  "appropriateness": <float 0.0-1.0>,
  "completeness": <float 0.0-1.0>,
  "efficiency": <float 0.0-1.0>,
  "missing_elements": [<list of strings>],
  "reasoning": "<one sentence explanation>"
}}"""

    def _parse_response(self, parsed: dict) -> MetricResult:
        score = float(parsed["score"])
        missing = parsed.get("missing_elements", [])
        if not isinstance(missing, list):
            missing = []
        evidence = [str(x) for x in missing]
        app = float(parsed.get("appropriateness", 0.0))
        comp = float(parsed.get("completeness", 0.0))
        eff = float(parsed.get("efficiency", 0.0))
        reason = (
            f"appropriateness={app:.2f}, completeness={comp:.2f}, efficiency={eff:.2f}. "
            f"{parsed.get('reasoning', '')}"
        )
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=self.passes(score),
            reason=reason,
            evidence=evidence,
        )

    async def compute(
        self,
        trace: Any,
        expected: Any | None = None,
        judge: JudgeClient | None = None,
    ) -> MetricResult:
        if judge is None:
            raise ValueError("PlanQuality requires a JudgeClient")

        llm_spans = [
            s
            for s in sorted(trace.spans, key=lambda x: x.timestamp)
            if s.span_type == "llm_call"
        ]
        if not llm_spans:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="No llm_call spans — metric skipped",
                evidence=[],
            )

        first_llm_output = _summarize(llm_spans[0].output, 12000)
        prompt = self._build_prompt(trace, first_llm_output)
        response = await judge.judge(
            prompt,
            response_schema={
                "score": "float",
                "appropriateness": "float",
                "completeness": "float",
                "efficiency": "float",
                "missing_elements": "list",
                "reasoning": "str",
            },
        )
        return self._parse_response(response.parsed)
