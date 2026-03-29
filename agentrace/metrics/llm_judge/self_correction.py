"""LLM-judge metric: quality of retries after failed or adjusted tool calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from agentrace.metrics.base import BaseMetric, MetricResult

if TYPE_CHECKING:
    from agentrace.metrics.llm_judge.judge_client import JudgeClient
    from agentrace.normalizer.models import AgentTrace, Span


def _tool_name(span: Span) -> str:
    raw = span.input.get("tool_name")
    if raw is None:
        raw = span.input.get("name")
    return "" if raw is None else str(raw)


def _find_retry_pairs(trace: AgentTrace) -> list[tuple[Span, Span]]:
    tool_spans = [s for s in sorted(trace.spans, key=lambda x: x.timestamp) if s.span_type == "tool_call"]
    pairs: list[tuple[Span, Span]] = []
    for i in range(len(tool_spans) - 1):
        a, b = tool_spans[i], tool_spans[i + 1]
        if _tool_name(a) != _tool_name(b) or not _tool_name(a):
            continue
        if a.error is not None:
            pairs.append((a, b))
            continue
        if str(a.input) != str(b.input):
            pairs.append((a, b))
            continue
    return pairs


class SelfCorrectionQuality(BaseMetric):
    name = "self_correction_quality"
    default_threshold = 0.70

    def _format_retry_pairs(self, pairs: list[tuple[Span, Span]]) -> str:
        lines: list[str] = []
        for failed, retry in pairs:
            lines.append(
                f"Failed span {failed.span_id} (tool={_tool_name(failed)!r}, "
                f"error={failed.error!r}, input={str(failed.input)[:400]})\n"
                f"  Retry span {retry.span_id} (input={str(retry.input)[:400]}, "
                f"output={str(retry.output)[:400]})"
            )
        return "\n\n".join(lines)

    def _build_prompt(self, trace: AgentTrace, pairs: list[tuple[Span, Span]]) -> str:
        formatted = self._format_retry_pairs(pairs)
        return f"""You are evaluating whether an AI agent improved its approach when retrying a failed or suboptimal step.

Task: {trace.task}

Retry instances detected:
{formatted}

For each retry, assess whether the agent diagnosed the problem correctly and made a meaningful improvement.

Respond ONLY with a JSON object:
{{
  "score": <float 0.0-1.0>,
  "retry_count": <int>,
  "improved_retries": <int — retries where the approach meaningfully improved>,
  "retry_assessments": [
    {{"span_id": "<failed span id>", "improved": <bool>, "reason": "<str>"}}
  ],
  "reasoning": "<one sentence overall explanation>"
}}"""

    def _parse_response(self, parsed: dict) -> MetricResult:
        score = float(parsed["score"])
        assessments = parsed.get("retry_assessments", [])
        evidence: list[str] = []
        if isinstance(assessments, list):
            for item in assessments:
                if not isinstance(item, dict):
                    continue
                if not item.get("improved", True):
                    sid = item.get("span_id", "")
                    if sid:
                        evidence.append(str(sid))
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
            raise ValueError("SelfCorrectionQuality requires a JudgeClient")

        pairs = _find_retry_pairs(trace)
        if not pairs:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="No retries detected — metric skipped",
                evidence=[],
            )

        prompt = self._build_prompt(trace, pairs)
        response = await judge.judge(
            prompt,
            response_schema={
                "score": "float",
                "retry_count": "int",
                "improved_retries": "int",
                "retry_assessments": "list",
                "reasoning": "str",
            },
        )
        return self._parse_response(response.parsed)
