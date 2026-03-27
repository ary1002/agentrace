"""Deterministic metrics derived from traces and task outputs without LLM calls."""

from __future__ import annotations

from agentrace.metrics.base import BaseMetric
from agentrace.metrics.deterministic.step_efficiency import StepEfficiency
from agentrace.metrics.deterministic.tool_selection import ToolSelectionAccuracy

METRICS_REGISTRY: dict[str, BaseMetric] = {
    ToolSelectionAccuracy.name: ToolSelectionAccuracy(),
    StepEfficiency.name: StepEfficiency(),
}

__all__ = ["METRICS_REGISTRY", "ToolSelectionAccuracy", "StepEfficiency"]
