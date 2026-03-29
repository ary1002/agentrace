"""CrewAI integration via task-level callback hooks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

try:
    import crewai  # noqa: F401

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

from agentrace.capture.adapters._span_utils import get_tracer, set_span_attributes

_LOG = logging.getLogger(__name__)


def make_task_callback(task_name: str) -> Callable[[Any], None]:
    """
    Returns a callback function suitable for task.callback = make_task_callback('research').
    The callback is called by CrewAI with the task output when the task completes.
    It emits an agent_step span retroactively (best effort — no start time available).
    """

    def callback(output: Any) -> None:
        tracer = get_tracer()
        output_str = str(output)
        with tracer.start_as_current_span(f"crewai/{task_name}") as span:
            set_span_attributes(
                span,
                "agent_step",
                input_dict={"task": task_name},
                output_dict={"output": output_str[:4000]},
                framework="crewai",
            )

    return callback


def instrument_crew(crew) -> None:
    """
    Auto-instrument a CrewAI Crew by attaching make_task_callback to each task.
    Iterates crew.tasks, sets task.callback = make_task_callback(task.description[:50])
    for any task that does not already have a callback set.
    Best-effort — never raises.

    Usage:
        crew = Crew(agents=[...], tasks=[...])
        instrument_crew(crew)
    """
    if not _AVAILABLE:
        _LOG.warning("crewai is not installed; instrument_crew is a no-op")
        return
    try:
        for task in getattr(crew, "tasks", []):
            if getattr(task, "callback", None) is None:
                desc = getattr(task, "description", "task") or "task"
                task_name = str(desc)[:50]
                task.callback = make_task_callback(task_name)
    except Exception:
        return
