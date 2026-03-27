"""Failure taxonomy and structured failure records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class FailureType(str, Enum):
    """Eight-class failure taxonomy for agent traces."""

    WRONG_TOOL_SELECTED = "WRONG_TOOL_SELECTED"
    CORRECT_TOOL_WRONG_ARGS = "CORRECT_TOOL_WRONG_ARGS"
    REASONING_BREAK = "REASONING_BREAK"
    HALLUCINATED_TOOL_CALL = "HALLUCINATED_TOOL_CALL"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    PREMATURE_TERMINATION = "PREMATURE_TERMINATION"
    REDUNDANT_LOOP = "REDUNDANT_LOOP"
    FAITHFULNESS_FAILURE = "FAITHFULNESS_FAILURE"


@dataclass
class FailureRecord:
    """A classified failure anchored to a task and span."""

    task_id: str
    failure_type: FailureType
    span_id: str
    severity: Literal["critical", "moderate", "minor"]
    explanation: str
    suggested_fix: str
