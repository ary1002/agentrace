"""AgentRace: framework-agnostic evaluation and tracing for LLM agent pipelines."""

from __future__ import annotations

import importlib
from typing import Any

from agentrace.dataset.benchmarks import load_suite as load_benchmark_suite
from agentrace.capture.context_manager import trace
from agentrace.capture.decorator import agent
from agentrace.capture.span import span
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

# Auto-patch OpenAI and Anthropic SDKs if installed
# These are no-ops if the libraries are not installed
from agentrace.capture.adapters.anthropic_sdk import patch_anthropic
from agentrace.capture.adapters.openai_sdk import patch_openai

patch_openai()
patch_anthropic()

# Expose adapter surface at top level for ergonomic imports
from agentrace.capture.adapters import (
    AgentTraceCallbackHandler,
    instrument_crew,
    instrument_graph,
    traced_node,
)

__version__ = "1.0.0"

__all__ = [
    "trace",
    "agent",
    "span",
    "evaluate",
    "load_benchmark_suite",
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
    "AgentTraceCallbackHandler",
    "traced_node",
    "instrument_graph",
    "instrument_crew",
]
