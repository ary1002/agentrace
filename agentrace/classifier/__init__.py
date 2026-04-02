"""Failure taxonomy: rule-based stage and LLM structured-output stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentrace.classifier.llm_classifier import LLMClassifier
from agentrace.classifier.models import FailureRecord, FailureType
from agentrace.classifier.rule_based import RuleBasedClassifier
from agentrace.metrics.llm_judge.judge_client import JudgeClient

if TYPE_CHECKING:
    from agentrace.normalizer.models import AgentTrace


class FailureClassifier:
    """Orchestrates Stage 1 (rule-based) and Stage 2 (LLM)."""

    def __init__(
        self,
        judge: JudgeClient | None = None,
        known_tools: list[str] | None = None,
        run_stage2: bool = True,
    ) -> None:
        self._rule_classifier = RuleBasedClassifier()
        self._llm_classifier = LLMClassifier(judge) if judge and run_stage2 else None
        self.known_tools = known_tools
        self.last_stage1_span_ids: set[str] = set()

    async def classify(
        self,
        trace: AgentTrace,
        task_id: str,
        wasted_step_ids: list[str] | None = None,
    ) -> list[FailureRecord]:
        self.last_stage1_span_ids = set()
        stage1 = self._rule_classifier.classify(trace, task_id, self.known_tools)
        self.last_stage1_span_ids = {r.span_id for r in stage1}
        if self._llm_classifier is None:
            return stage1

        already_classified = {r.span_id for r in stage1}
        stage2 = await self._llm_classifier.classify(
            trace,
            task_id,
            already_classified,
            wasted_step_ids,
        )
        return stage1 + stage2


__all__ = [
    "FailureClassifier",
    "FailureType",
    "FailureRecord",
    "RuleBasedClassifier",
    "LLMClassifier",
]
