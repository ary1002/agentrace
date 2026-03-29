"""LLM-judge metric: reasoning coherence across ordered spans."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from agentrace.metrics.base import BaseMetric, MetricResult

if TYPE_CHECKING:
    from agentrace.metrics.llm_judge.judge_client import JudgeClient
    from agentrace.normalizer.models import AgentTrace


def _summarize(obj: Any, max_len: int) -> str:
    s = str(obj) if obj is not None else ""
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


class ReasoningCoherence(BaseMetric):
    name = "reasoning_coherence"
    default_threshold = 0.75

    def _build_prompt(self, trace: AgentTrace) -> str:
        ordered = sorted(trace.spans, key=lambda s: s.timestamp)
        lines: list[str] = []
        for span in ordered:
            inp = _summarize(span.input, 500)
            out = _summarize(span.output, 500)
            lines.append(
                f"- [{span.span_type}] input_summary={inp!r} output_summary={out!r}"
            )
        formatted_steps = "\n".join(lines) if lines else "(no steps)"

        return f"""You are evaluating whether an AI agent's reasoning steps are coherent and logically connected.

Task: {trace.task}

Agent execution steps (in order):
{formatted_steps}

For each consecutive pair of steps, assess whether step N+1 logically follows from step N given the task context.

Respond ONLY with a JSON object:
{{
  "score": <float 0.0-1.0>,
  "coherent_transitions": <int — number of step transitions that are coherent>,
  "total_transitions": <int>,
  "incoherent_steps": [<list of 1-based step indices where reasoning breaks down>],
  "reasoning": "<one sentence explanation>"
}}"""

    def _parse_response(self, parsed: dict, trace: AgentTrace) -> MetricResult:
        score = float(parsed["score"])
        incoherent = parsed.get("incoherent_steps", [])
        if not isinstance(incoherent, list):
            incoherent = []
        n_spans = len(sorted(trace.spans, key=lambda s: s.timestamp))
        ordered = sorted(trace.spans, key=lambda s: s.timestamp)
        evidence = [
            f"step {i}: span {ordered[i - 1].span_id}"
            for i in incoherent
            if isinstance(i, int) and 0 < i <= n_spans
        ]
        ct = parsed.get("coherent_transitions", 0)
        tt = parsed.get("total_transitions", 0)
        reason = f"{ct}/{tt} transitions coherent. {parsed['reasoning']}"
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
            raise ValueError("ReasoningCoherence requires a JudgeClient")
        prompt = self._build_prompt(trace)
        response = await judge.judge(
            prompt,
            response_schema={
                "score": "float",
                "coherent_transitions": "int",
                "total_transitions": "int",
                "incoherent_steps": "list",
                "reasoning": "str",
            },
        )
        return self._parse_response(response.parsed, trace)
