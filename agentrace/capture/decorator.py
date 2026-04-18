"""`@agentrace.agent` decorator for async traced agent entrypoints."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar
from uuid import uuid4

from agentrace.capture.async_context import (
    attach_context,
    detach_context,
    get_current_context,
)
from agentrace.capture.context_manager import trace

P = ParamSpec("P")
R = TypeVar("R")


def _infer_task_name(
    func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str:
    if "task" in kwargs and kwargs["task"] is not None:
        return str(kwargs["task"])
    if "query" in kwargs and kwargs["query"] is not None:
        return str(kwargs["query"])
    if args:
        return str(args[0])
    return func_name


def agent(
    fn: Callable[P, Awaitable[R]] | None = None,
    *,
    task_name: str | None = None,
) -> (
    Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]
    | Callable[P, Awaitable[R]]
):
    """Wrap an async agent function in ``agentrace.trace()``.

    The decorator snapshots the current OpenTelemetry context before function execution and
    re-attaches it inside the traced task body to keep parent context across asyncio task
    boundaries.
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            parent_ctx = get_current_context()
            inferred_task = task_name or _infer_task_name(func.__name__, args, kwargs)
            session_id = f"{func.__name__}:{uuid4().hex}"
            async with trace(session_id=session_id, task=inferred_task):
                token = attach_context(parent_ctx)
                try:
                    return await func(*args, **kwargs)
                finally:
                    detach_context(token)

        return wrapped

    if fn is not None:
        return decorator(fn)
    return decorator
