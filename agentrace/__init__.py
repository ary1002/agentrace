"""AgentRace: framework-agnostic evaluation and tracing for LLM agent pipelines."""

from __future__ import annotations

import importlib
from typing import Any

from agentrace.capture.context_manager import trace
from agentrace.classifier.models import FailureRecord, FailureType
from agentrace.metrics.base import BaseMetric, MetricResult
from agentrace.normalizer.models import AgentTrace, Span, SpanNode, TokenCount
from agentrace.runner.models import EvalResult, EvalTask, TaskResult

evaluate: Any
try:
    _runner_mod = importlib.import_module("agentrace.runner.runner")
except ImportError:
    evaluate = None
else:
    evaluate = getattr(_runner_mod, "evaluate", None)

__version__ = "0.1.0"

__all__ = [
    "trace",
    "evaluate",
    "AgentTrace",
    "Span",
    "SpanNode",
    "TokenCount",
    "EvalTask",
    "EvalResult",
    "TaskResult",
    "BaseMetric",
    "MetricResult",
    "FailureType",
    "FailureRecord",
]
