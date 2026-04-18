from __future__ import annotations

from conftest import make_span, make_trace

from agentrace.classifier.models import FailureType
from agentrace.classifier.rule_based import RuleBasedClassifier


def test_rule_based_hallucinated_tool_call() -> None:
    trace = make_trace([make_span("1", span_type="tool_call", tool_name="ghost_tool")])
    out = RuleBasedClassifier().classify(trace, "t1", known_tools=["web_search"])
    assert any(r.failure_type == FailureType.HALLUCINATED_TOOL_CALL for r in out)


def test_rule_based_redundant_loop() -> None:
    s1 = make_span("1", span_type="tool_call", tool_name="web_search", offset_ms=1)
    s1.input = {"tool_name": "web_search", "query": "q"}
    s2 = make_span("2", span_type="tool_call", tool_name="web_search", offset_ms=2)
    s2.input = {"tool_name": "web_search", "query": "q"}
    trace = make_trace([s1, s2])
    out = RuleBasedClassifier().classify(trace, "t1")
    assert any(r.failure_type == FailureType.REDUNDANT_LOOP for r in out)


def test_rule_based_premature_termination() -> None:
    trace = make_trace([make_span("1", span_type="llm_call")], outcome="partial")
    out = RuleBasedClassifier().classify(trace, "t1")
    assert any(r.failure_type == FailureType.PREMATURE_TERMINATION for r in out)


def test_rule_based_context_overflow() -> None:
    span = make_span("1", span_type="llm_call", error="Maximum context length exceeded")
    trace = make_trace([span])
    out = RuleBasedClassifier().classify(trace, "t1")
    assert any(r.failure_type == FailureType.CONTEXT_OVERFLOW for r in out)
