"""Capture layer: decorators, context managers, spans, and framework adapters."""

from agentrace.capture.context_manager import current_tracer, trace
from agentrace.capture.decorator import agent
from agentrace.capture.span import span

__all__ = ["trace", "current_tracer", "agent", "span"]
