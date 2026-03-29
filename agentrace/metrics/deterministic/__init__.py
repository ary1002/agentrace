"""Deterministic + LLM-judge metrics registry."""

from __future__ import annotations

from agentrace.metrics.base import BaseMetric
from agentrace.metrics.deterministic.step_efficiency import StepEfficiency
from agentrace.metrics.deterministic.tool_selection import ToolSelectionAccuracy
from agentrace.metrics.llm_judge import LLM_METRICS_REGISTRY

METRICS_REGISTRY: dict[str, BaseMetric] = {
    ToolSelectionAccuracy.name: ToolSelectionAccuracy(),
    StepEfficiency.name: StepEfficiency(),
    **LLM_METRICS_REGISTRY,
}

LLM_METRIC_NAMES: frozenset[str] = frozenset(LLM_METRICS_REGISTRY.keys())

__all__ = ["METRICS_REGISTRY", "LLM_METRIC_NAMES", "ToolSelectionAccuracy", "StepEfficiency"]
