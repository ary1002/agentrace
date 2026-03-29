"""LLM-as-judge metrics via JudgeClient (litellm only inside judge_client)."""

from __future__ import annotations

from agentrace.metrics.base import BaseMetric
from agentrace.metrics.llm_judge.answer_faithfulness import AnswerFaithfulness
from agentrace.metrics.llm_judge.judge_client import JudgeClient, JudgeError, JudgeParseError
from agentrace.metrics.llm_judge.plan_quality import PlanQuality
from agentrace.metrics.llm_judge.reasoning_coherence import ReasoningCoherence
from agentrace.metrics.llm_judge.self_correction import SelfCorrectionQuality
from agentrace.metrics.llm_judge.trajectory_optimality import TrajectoryOptimality

LLM_METRICS_REGISTRY: dict[str, BaseMetric] = {
    ReasoningCoherence.name: ReasoningCoherence(),
    AnswerFaithfulness.name: AnswerFaithfulness(),
    PlanQuality.name: PlanQuality(),
    SelfCorrectionQuality.name: SelfCorrectionQuality(),
    TrajectoryOptimality.name: TrajectoryOptimality(),
}

__all__ = [
    "LLM_METRICS_REGISTRY",
    "ReasoningCoherence",
    "AnswerFaithfulness",
    "PlanQuality",
    "SelfCorrectionQuality",
    "TrajectoryOptimality",
    "JudgeClient",
    "JudgeError",
    "JudgeParseError",
]
