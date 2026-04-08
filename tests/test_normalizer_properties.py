from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given, strategies as st

from agentrace.normalizer.models import MalformedTraceError
from agentrace.normalizer.normalizer import Normalizer


@dataclass
class FakeCtx:
    span_id: int
    is_valid: bool = True


@dataclass
class FakeReadable:
    context: FakeCtx
    parent: FakeCtx | None
    start_time: int
    end_time: int
    attributes: dict


def _mk(span_id: int, parent_id: int | None, ts: int) -> FakeReadable:
    return FakeReadable(
        context=FakeCtx(span_id=span_id),
        parent=FakeCtx(span_id=parent_id) if parent_id is not None else None,
        start_time=1_000_000_000 + ts,
        end_time=1_000_000_500 + ts,
        attributes={"agentrace.span_type": "agent_step", "agentrace.input": "{}", "agentrace.output": "{}"},
    )


@given(st.integers(min_value=2, max_value=8))
def test_build_multiple_roots_creates_virtual_root(n: int) -> None:
    raws = [_mk(i + 1, None, i) for i in range(n)]
    out = Normalizer.build("s", "task", raws)
    assert out.trace_tree.span.span_id == "virtual_root"
    assert len(out.trace_tree.children) == n


@given(st.integers(min_value=1, max_value=6))
def test_build_orphaned_spans_are_attached_to_root(n_orphans: int) -> None:
    raws = [_mk(1, None, 0)]
    raws.extend(_mk(10 + i, 9999 + i, i + 1) for i in range(n_orphans))
    out = Normalizer.build("s", "task", raws)
    child_ids = {c.span.span_id for c in out.trace_tree.children}
    assert len(child_ids) == n_orphans


@given(st.integers(min_value=2, max_value=8))
def test_build_rejects_parent_cycle_without_root(n: int) -> None:
    raws = []
    for i in range(n):
        cur = i + 1
        parent = 1 if i == n - 1 else cur + 1
        raws.append(_mk(cur, parent, i))
    with pytest.raises(MalformedTraceError):
        Normalizer.build("s", "task", raws)
