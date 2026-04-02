"""Deterministic + LLM-judge metrics registry."""

from __future__ import annotations

from agentrace.metrics.base import BaseMetric
from agentrace.metrics.deterministic.argument_correctness import ArgumentCorrectness
from agentrace.metrics.deterministic.cost_per_task import CostPerTask
from agentrace.metrics.deterministic.early_termination import EarlyTerminationRate
from agentrace.metrics.deterministic.latency import LatencyP50, LatencyP95
from agentrace.metrics.deterministic.redundancy_rate import RedundancyRate
from agentrace.metrics.deterministic.step_efficiency import StepEfficiency
from agentrace.metrics.deterministic.token_efficiency import TokenEfficiency
from agentrace.metrics.deterministic.tool_selection import ToolSelectionAccuracy
from agentrace.metrics.llm_judge import LLM_METRICS_REGISTRY

METRICS_REGISTRY: dict[str, BaseMetric] = {
    ToolSelectionAccuracy.name: ToolSelectionAccuracy(),
    StepEfficiency.name: StepEfficiency(),
    RedundancyRate.name: RedundancyRate(),
    EarlyTerminationRate.name: EarlyTerminationRate(),
    TokenEfficiency.name: TokenEfficiency(),
    CostPerTask.name: CostPerTask(),
    LatencyP50.name: LatencyP50(),
    LatencyP95.name: LatencyP95(),
    ArgumentCorrectness.name: ArgumentCorrectness(),
    **LLM_METRICS_REGISTRY,
}

LLM_METRIC_NAMES: frozenset[str] = frozenset(LLM_METRICS_REGISTRY.keys())

__all__ = [
    "METRICS_REGISTRY",
    "LLM_METRIC_NAMES",
    "ToolSelectionAccuracy",
    "StepEfficiency",
    "RedundancyRate",
    "EarlyTerminationRate",
    "TokenEfficiency",
    "CostPerTask",
    "LatencyP50",
    "LatencyP95",
    "ArgumentCorrectness",
]
