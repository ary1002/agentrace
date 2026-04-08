from __future__ import annotations

import pytest

from agentrace.metrics.deterministic.argument_correctness import ArgumentCorrectness
from agentrace.metrics.deterministic.cost_per_task import CostPerTask
from agentrace.metrics.deterministic.early_termination import EarlyTerminationRate
from agentrace.metrics.deterministic.latency import LatencyP50, LatencyP95
from agentrace.metrics.deterministic.redundancy_rate import RedundancyRate
from agentrace.metrics.deterministic.step_efficiency import StepEfficiency
from agentrace.metrics.deterministic.token_efficiency import TokenEfficiency
from agentrace.metrics.deterministic.tool_selection import ToolSelectionAccuracy
from agentrace.runner.models import EvalTask
from conftest import make_span, make_trace


@pytest.mark.asyncio
async def test_tool_selection_missing_expected_tools_defaults_to_pass() -> None:
    metric = ToolSelectionAccuracy()
    trace = make_trace([make_span("1", span_type="tool_call", tool_name="web_search")])
    out = await metric.compute(trace, expected=EvalTask(id="t1", query="q", expected_tools=None))
    assert out.score == 1.0 and out.passed


@pytest.mark.asyncio
async def test_step_efficiency_handles_optimal_steps_zero() -> None:
    metric = StepEfficiency()
    trace = make_trace([make_span("1"), make_span("2")])
    out = await metric.compute(trace, expected=EvalTask(id="t1", query="q", optimal_steps=0))
    assert out.score == 0.0


@pytest.mark.asyncio
async def test_redundancy_rate_with_zero_spans_scores_full() -> None:
    metric = RedundancyRate()
    out = await metric.compute(make_trace([]))
    assert out.score == 1.0 and out.passed


@pytest.mark.asyncio
async def test_early_termination_penalizes_tiny_trace() -> None:
    metric = EarlyTerminationRate()
    out = await metric.compute(make_trace([make_span("1")]), expected=EvalTask(id="t1", query="q", expected_tools=["web_search"]))
    assert out.score < 1.0


@pytest.mark.asyncio
async def test_token_efficiency_score_caps_at_one() -> None:
    metric = TokenEfficiency()
    out = await metric.compute(make_trace([make_span("1")], total_tokens=50))
    assert out.score == 1.0


@pytest.mark.asyncio
async def test_cost_per_task_score_caps_at_one() -> None:
    metric = CostPerTask()
    out = await metric.compute(make_trace([make_span("1", cost_usd=0.001)], total_cost=0.001))
    assert out.score == 1.0


@pytest.mark.asyncio
async def test_latency_metrics_zero_spans_neutral_score() -> None:
    t1 = await LatencyP50().compute(make_trace([]))
    t2 = await LatencyP95().compute(make_trace([]))
    assert t1.score == 0.5 and t2.score == 0.5


@pytest.mark.asyncio
async def test_argument_correctness_validates_missing_query() -> None:
    metric = ArgumentCorrectness()
    bad = make_span("1", span_type="tool_call", tool_name="web_search")
    bad.input = {"tool_name": "web_search", "args": {}}
    out = await metric.compute(make_trace([bad]))
    assert out.score == 0.0 and not out.passed
