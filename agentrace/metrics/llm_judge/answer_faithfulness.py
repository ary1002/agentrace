"""LLM-judge metric: final answer faithfulness vs tool-retrieved context."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from agentrace.metrics.base import BaseMetric, MetricResult

if TYPE_CHECKING:
    from agentrace.metrics.llm_judge.judge_client import JudgeClient
    from agentrace.normalizer.models import AgentTrace


def _summarize(obj: Any, max_len: int) -> str:
    s = str(obj) if obj is not None else ""
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


class AnswerFaithfulness(BaseMetric):
    name = "answer_faithfulness"
    default_threshold = 0.80

    def _build_prompt(self, trace: AgentTrace, formatted_tool_outputs: str, final_answer: str) -> str:
        return f"""You are evaluating whether an AI agent's final answer is faithful to the information it retrieved.

Task: {trace.task}

Retrieved context (tool outputs):
{formatted_tool_outputs}

Final answer:
{final_answer}

Assess whether every claim in the final answer is supported by the retrieved context. Penalise claims that contradict or go beyond the retrieved information.

Respond ONLY with a JSON object:
{{
  "score": <float 0.0-1.0>,
  "supported_claims": <int>,
  "unsupported_claims": <int>,
  "contradictions": [<list of brief strings describing contradictions>],
  "reasoning": "<one sentence explanation>"
}}"""

    def _parse_response(self, parsed: dict) -> MetricResult:
        score = float(parsed["score"])
        contradictions = parsed.get("contradictions", [])
        if not isinstance(contradictions, list):
            contradictions = []
        evidence = [str(c) for c in contradictions]
        reason = parsed.get("reasoning", "")
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
        expected: Optional[Any] = None,
        judge: Optional[JudgeClient] = None,
    ) -> MetricResult:
        if judge is None:
            raise ValueError("AnswerFaithfulness requires a JudgeClient")

        tool_spans = [s for s in sorted(trace.spans, key=lambda x: x.timestamp) if s.span_type == "tool_call"]
        llm_spans = [s for s in sorted(trace.spans, key=lambda x: x.timestamp) if s.span_type == "llm_call"]

        if not llm_spans:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="No llm_call spans — metric skipped",
                evidence=[],
            )

        final_answer = _summarize(llm_spans[-1].output, 8000)

        if not tool_spans:
            formatted_tool_outputs = (
                "(No tool_call spans — no retrieval occurred. Evaluate whether the final answer "
                "is appropriately cautious given the lack of retrieved evidence.)"
            )
        else:
            parts: list[str] = []
            for s in tool_spans:
                parts.append(f"span {s.span_id}: {_summarize(s.output, 2000)}")
            formatted_tool_outputs = "\n".join(parts)

        prompt = self._build_prompt(trace, formatted_tool_outputs, final_answer)
        response = await judge.judge(
            prompt,
            response_schema={
                "score": "float",
                "supported_claims": "int",
                "unsupported_claims": "int",
                "contradictions": "list",
                "reasoning": "str",
            },
        )
        return self._parse_response(response.parsed)
