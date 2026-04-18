"""Stage 1 classification: deterministic checks on traces."""

from __future__ import annotations

import difflib
import logging
from typing import TYPE_CHECKING

from agentrace.classifier.models import FailureRecord, FailureType

if TYPE_CHECKING:
    from agentrace.normalizer.models import AgentTrace

_LOG = logging.getLogger(__name__)

_ERROR_HALLUCINATION_MARKERS = (
    "tool not found",
    "unknown tool",
    "function not found",
)

_CONTEXT_MARKERS = (
    "context length",
    "token limit",
    "maximum context",
)

_PREMATURE_PHRASES = (
    "i cannot",
    "i don't know",
    "unable to",
    "insufficient information",
    "no information available",
)


class RuleBasedClassifier:
    def classify(
        self,
        trace: AgentTrace,
        task_id: str,
        known_tools: list[str] | None = None,
    ) -> list[FailureRecord]:
        records: list[FailureRecord] = []
        seen: set[str] = set()

        def add(recs: list[FailureRecord]) -> None:
            for r in recs:
                if r.span_id in seen:
                    continue
                seen.add(r.span_id)
                records.append(r)

        add(self._check_hallucinated_tool(trace, task_id, known_tools))
        add(self._check_redundant_loop(trace, task_id))
        # Context overflow before premature termination: both can match the same span's error;
        # we keep the more specific CONTEXT_OVERFLOW classification.
        add(self._check_context_overflow(trace, task_id))
        add(self._check_premature_termination(trace, task_id))
        return records

    def _check_hallucinated_tool(
        self,
        trace: AgentTrace,
        task_id: str,
        known_tools: list[str] | None,
    ) -> list[FailureRecord]:
        try:
            out: list[FailureRecord] = []
            ordered = sorted(trace.spans, key=lambda s: s.timestamp)
            for span in ordered:
                if span.span_type != "tool_call":
                    continue
                raw = span.input.get("tool_name")
                if raw is None:
                    raw = span.input.get("name")
                tool_name = "" if raw is None else str(raw)
                err_l = (span.error or "").lower()
                hallucinated = False
                if (
                    known_tools is not None
                    and tool_name
                    and tool_name not in known_tools
                ):
                    hallucinated = True
                if not hallucinated:
                    for m in _ERROR_HALLUCINATION_MARKERS:
                        if m in err_l:
                            hallucinated = True
                            break
                if hallucinated:
                    out.append(
                        FailureRecord(
                            task_id=task_id,
                            failure_type=FailureType.HALLUCINATED_TOOL_CALL,
                            span_id=span.span_id,
                            severity="critical",
                            explanation=(
                                f"Agent called '{tool_name}' which is not in the tool registry"
                            ),
                            suggested_fix=(
                                "Ensure the agent's system prompt lists only available tools. "
                                "Check tool name spelling and casing."
                            ),
                        )
                    )
            return out
        except Exception as e:
            _LOG.warning("_check_hallucinated_tool failed: %s", e)
            return []

    def _check_redundant_loop(
        self, trace: AgentTrace, task_id: str
    ) -> list[FailureRecord]:
        try:
            out: list[FailureRecord] = []
            tool_spans = [
                s
                for s in sorted(trace.spans, key=lambda s: s.timestamp)
                if s.span_type == "tool_call"
            ]
            for i in range(1, len(tool_spans)):
                prev, cur = tool_spans[i - 1], tool_spans[i]
                raw_p = prev.input.get("tool_name") or prev.input.get("name")
                raw_c = cur.input.get("tool_name") or cur.input.get("name")
                name_p = "" if raw_p is None else str(raw_p)
                name_c = "" if raw_c is None else str(raw_c)
                if name_p != name_c or not name_p:
                    continue
                ratio = difflib.SequenceMatcher(
                    None, str(prev.input), str(cur.input)
                ).ratio()
                if ratio > 0.85:
                    out.append(
                        FailureRecord(
                            task_id=task_id,
                            failure_type=FailureType.REDUNDANT_LOOP,
                            span_id=cur.span_id,
                            severity="minor",
                            explanation=(
                                f"Tool '{name_c}' called again with near-identical arguments "
                                f"(similarity {ratio:.0%})"
                            ),
                            suggested_fix=(
                                "Add a memory check before tool calls to avoid repeating the same query. "
                                "Consider caching tool results."
                            ),
                        )
                    )
            return out
        except Exception as e:
            _LOG.warning("_check_redundant_loop failed: %s", e)
            return []

    def _check_premature_termination(
        self, trace: AgentTrace, task_id: str
    ) -> list[FailureRecord]:
        try:
            ordered = sorted(trace.spans, key=lambda s: s.timestamp)
            if not ordered:
                return []
            last = ordered[-1]
            detected_reason = ""

            if trace.outcome in ("failure", "partial"):
                detected_reason = f"trace outcome is '{trace.outcome}'"
            elif last.error is not None:
                detected_reason = f"last span error: {last.error}"
            else:
                llm_spans = [s for s in ordered if s.span_type == "llm_call"]
                if llm_spans:
                    content = str(llm_spans[-1].output).lower()
                    for phrase in _PREMATURE_PHRASES:
                        if phrase in content:
                            detected_reason = f"last llm output suggests inability: matched {phrase!r}"
                            break
            if not detected_reason and len(trace.spans) < 2:
                detected_reason = (
                    f"only {len(trace.spans)} span(s); agent did almost nothing"
                )

            if not detected_reason:
                return []

            return [
                FailureRecord(
                    task_id=task_id,
                    failure_type=FailureType.PREMATURE_TERMINATION,
                    span_id=last.span_id,
                    severity="critical",
                    explanation=(
                        f"Agent terminated before satisfying task criteria. Reason: {detected_reason}"
                    ),
                    suggested_fix=(
                        "Review the agent's stopping condition. Ensure it checks task completion "
                        "criteria before returning."
                    ),
                )
            ]
        except Exception as e:
            _LOG.warning("_check_premature_termination failed: %s", e)
            return []

    def _check_context_overflow(
        self, trace: AgentTrace, task_id: str
    ) -> list[FailureRecord]:
        try:
            out: list[FailureRecord] = []
            ordered = sorted(trace.spans, key=lambda s: s.timestamp)
            for span in ordered:
                err_l = (span.error or "").lower()
                for m in _CONTEXT_MARKERS:
                    if m in err_l:
                        out.append(
                            FailureRecord(
                                task_id=task_id,
                                failure_type=FailureType.CONTEXT_OVERFLOW,
                                span_id=span.span_id,
                                severity="moderate",
                                explanation=(
                                    f"Span error indicates context limits: matched {m!r} in error message"
                                ),
                                suggested_fix=(
                                    "Implement context summarisation or sliding window memory. "
                                    "Consider chunking the task into smaller subtasks."
                                ),
                            )
                        )
                        break

            llm_calls = [s for s in ordered if s.span_type == "llm_call"]
            n = len(llm_calls)
            if n >= 3:
                k = max(1, n // 3)
                lengths_early = [len(str(s.output)) for s in llm_calls[:k]]
                lengths_late = [len(str(s.output)) for s in llm_calls[-k:]]
                early_mean = sum(lengths_early) / len(lengths_early)
                late_mean = sum(lengths_late) / len(lengths_late)
                if early_mean > 0 and late_mean < 0.5 * early_mean:
                    degraded_span = llm_calls[-k].span_id
                    if not any(r.span_id == degraded_span for r in out):
                        out.append(
                            FailureRecord(
                                task_id=task_id,
                                failure_type=FailureType.CONTEXT_OVERFLOW,
                                span_id=degraded_span,
                                severity="moderate",
                                explanation=(
                                    "Output quality degraded in later steps, suggesting context window pressure. "
                                    f"Output length dropped from {early_mean:.0f} to {late_mean:.0f} chars on average."
                                ),
                                suggested_fix=(
                                    "Implement context summarisation or sliding window memory. "
                                    "Consider chunking the task into smaller subtasks."
                                ),
                            )
                        )
            return out
        except Exception as e:
            _LOG.warning("_check_context_overflow failed: %s", e)
            return []
