"""Stage 2 classification: batched LLM judge over remaining spans."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

from agentrace.classifier.models import FailureRecord, FailureType
from agentrace.metrics.llm_judge.judge_client import JudgeClient

if TYPE_CHECKING:
    from agentrace.normalizer.models import AgentTrace, Span

_LOG = logging.getLogger(__name__)

Severity = Literal["critical", "moderate", "minor"]


def _summarize(obj: object, max_len: int) -> str:
    s = str(obj) if obj is not None else ""
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


class LLMClassifier:
    def __init__(self, judge: JudgeClient) -> None:
        self.judge = judge

    def _brief_full_trace(self, trace: AgentTrace) -> str:
        ordered = sorted(trace.spans, key=lambda s: s.timestamp)
        lines: list[str] = []
        for i, s in enumerate(ordered, start=1):
            lines.append(
                f"{i}. [{s.span_type}] id={s.span_id} "
                f"in={_summarize(s.input, 80)} out={_summarize(s.output, 80)}"
            )
        return "\n".join(lines) if lines else "(no spans)"

    def _build_classification_prompt(
        self,
        trace: AgentTrace,
        candidate_spans: list[Span],
    ) -> str:
        total = len(trace.spans)
        brief_full_trace = self._brief_full_trace(trace)
        parts: list[str] = []
        for i, s in enumerate(candidate_spans, start=1):
            raw = s.input.get("tool_name")
            if raw is None:
                raw = s.input.get("name")
            tool_name = "N/A" if raw is None else str(raw)
            err = s.error if s.error is not None else "none"
            parts.append(
                f"Step {i}: [{s.span_type}] tool={tool_name}\n"
                f"  Input:  {_summarize(s.input, 200)}\n"
                f"  Output: {_summarize(s.output, 200)}\n"
                f"  Error:  {err}"
            )
        formatted_candidates = "\n\n".join(parts) if parts else "(none)"
        n = len(candidate_spans)
        return f"""You are classifying failures in an AI agent execution trace.

Task: {trace.task}

Full trace context ({total} total steps):
{brief_full_trace}

Candidate spans to classify ({n} spans):
{formatted_candidates}

For each candidate span, classify it into one of these failure types (or 'NO_FAILURE' if it is not a failure):
  - WRONG_TOOL_SELECTED: correct type of action but wrong specific tool
  - CORRECT_TOOL_WRONG_ARGS: right tool, malformed/missing/wrong arguments
  - REASONING_BREAK: this step does not logically follow the previous step
  - FAITHFULNESS_FAILURE: output makes claims unsupported by retrieved context

Respond ONLY with a JSON object:
{{
  "classifications": [
    {{
      "span_id": "<span_id>",
      "failure_type": "<FAILURE_TYPE or NO_FAILURE>",
      "severity": "<critical|moderate|minor>",
      "explanation": "<one sentence>",
      "confidence": <float 0.0-1.0>
    }}
  ]
}}"""

    def _fix_prompt(
        self, failure_type: str, span: Span, task: str, explanation: str
    ) -> str:
        input_summary = _summarize(span.input, 400)
        return f"""Given this agent failure in the task '{task}':
Failure type: {failure_type}
Offending step: [{span.span_type}] {input_summary}
Explanation: {explanation}

Provide a single, concrete, actionable fix for the agent developer.
Two sentences maximum. Respond ONLY with a JSON object:
{{'suggested_fix': '<string>'}}"""

    async def _generate_fix(
        self,
        failure_type: str,
        span: Span,
        task: str,
        explanation: str,
    ) -> str:
        prompt = self._fix_prompt(failure_type, span, task, explanation)
        resp = await self.judge.judge(prompt, response_schema={"suggested_fix": "str"})
        return str(resp.parsed.get("suggested_fix", ""))

    def _parse_classification_response(
        self,
        parsed: dict,
        trace: AgentTrace,
        task_id: str,
    ) -> list[FailureRecord]:
        by_id = {s.span_id: s for s in trace.spans}
        raw_list = parsed.get("classifications", [])
        if not isinstance(raw_list, list):
            return []
        out: list[FailureRecord] = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            ft_raw = str(item.get("failure_type", "")).strip()
            if ft_raw == "NO_FAILURE":
                continue
            try:
                conf = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if conf < 0.6:
                continue
            try:
                ftype = FailureType[ft_raw]
            except KeyError:
                _LOG.warning("unknown failure_type from judge: %s", ft_raw)
                continue
            sid = str(item.get("span_id", ""))
            if sid not in by_id:
                _LOG.warning("span_id not in trace: %s", sid)
                continue
            sev_raw = item.get("severity", "moderate")
            if sev_raw not in ("critical", "moderate", "minor"):
                sev_raw = "moderate"
            sev = cast(Severity, sev_raw)
            out.append(
                FailureRecord(
                    task_id=task_id,
                    failure_type=ftype,
                    span_id=sid,
                    severity=sev,
                    explanation=str(item.get("explanation", "")),
                    suggested_fix="",
                )
            )
        return out

    async def classify(
        self,
        trace: AgentTrace,
        task_id: str,
        already_classified_span_ids: set[str],
        wasted_step_ids: list[str] | None = None,
    ) -> list[FailureRecord]:
        ordered = sorted(trace.spans, key=lambda s: s.timestamp)
        candidates: list[Span] = []
        seen_ids: set[str] = set()

        if wasted_step_ids:
            for wid in wasted_step_ids:
                if wid in already_classified_span_ids or wid in seen_ids:
                    continue
                sp = next((s for s in ordered if s.span_id == wid), None)
                if sp is not None:
                    candidates.append(sp)
                    seen_ids.add(wid)

        for s in ordered:
            if s.span_id in already_classified_span_ids or s.span_id in seen_ids:
                continue
            candidates.append(s)
            seen_ids.add(s.span_id)

        if not candidates:
            return []

        prompt = self._build_classification_prompt(trace, candidates)
        response = await self.judge.judge(
            prompt,
            response_schema={"classifications": "list"},
        )
        records = self._parse_classification_response(response.parsed, trace, task_id)
        if not records:
            return records

        fix_prompts: list[str] = []
        for rec in records:
            span = next((s for s in trace.spans if s.span_id == rec.span_id), None)
            if span is None:
                fix_prompts.append("")
                continue
            fix_prompts.append(
                self._fix_prompt(
                    rec.failure_type.name, span, trace.task, rec.explanation
                )
            )

        indices = [i for i, p in enumerate(fix_prompts) if p]
        valid_prompts = [fix_prompts[i] for i in indices]
        if valid_prompts:
            fix_responses = await self.judge.judge_batch(
                valid_prompts,
                response_schema={"suggested_fix": "str"},
                concurrency=5,
            )
            for idx_pos, rec_idx in enumerate(indices):
                if idx_pos >= len(fix_responses):
                    break
                fix = fix_responses[idx_pos].parsed.get("suggested_fix", "")
                old = records[rec_idx]
                records[rec_idx] = FailureRecord(
                    task_id=old.task_id,
                    failure_type=old.failure_type,
                    span_id=old.span_id,
                    severity=old.severity,
                    explanation=old.explanation,
                    suggested_fix=str(fix) if fix is not None else "",
                )

        return records
