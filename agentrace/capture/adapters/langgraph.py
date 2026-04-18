"""LangGraph integration via node-wrapping decorator."""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import TypeVar

from agentrace.capture.adapters._span_utils import (
    get_tracer,
    record_exception,
    set_span_attributes,
)

_LOG = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def traced_node(name: str | None = None):
    """
    Decorator for LangGraph node functions. Wraps the node in an agentrace
    agent_step span.

    Usage:
        @traced_node()
        async def my_node(state: dict) -> dict:
            ...

        # or with explicit name:
        @traced_node(name='research_step')
        async def my_node(state: dict) -> dict:
            ...
    """

    def decorator(fn: F) -> F:
        node_name = name or fn.__name__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                state = args[0] if args else {}
                input_dict = {
                    "node": node_name,
                    "node_name": node_name,
                    "state_keys": (
                        list(state.keys())
                        if isinstance(state, dict)
                        else str(type(state))
                    ),
                }

                with tracer.start_as_current_span(f"node/{node_name}") as span:
                    try:
                        result = await fn(*args, **kwargs)
                        output_dict = {
                            "state_keys": (
                                list(result.keys())
                                if isinstance(result, dict)
                                else str(type(result))
                            ),
                        }
                        set_span_attributes(
                            span,
                            "agent_step",
                            input_dict,
                            output_dict,
                            "langgraph",
                        )
                        return result
                    except Exception as e:
                        record_exception(span, e)
                        raise

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            state = args[0] if args else {}
            input_dict = {
                "node": node_name,
                "node_name": node_name,
                "state_keys": (
                    list(state.keys()) if isinstance(state, dict) else str(type(state))
                ),
            }

            with tracer.start_as_current_span(f"node/{node_name}") as span:
                try:
                    result = fn(*args, **kwargs)
                    output_dict = {
                        "state_keys": (
                            list(result.keys())
                            if isinstance(result, dict)
                            else str(type(result))
                        ),
                    }
                    set_span_attributes(
                        span,
                        "agent_step",
                        input_dict,
                        output_dict,
                        "langgraph",
                    )
                    return result
                except Exception as e:
                    record_exception(span, e)
                    raise

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def instrument_graph(graph) -> None:
    """
    Attempt to auto-instrument a compiled LangGraph by wrapping all nodes.
    Accepts a CompiledGraph or StateGraph. Iterates graph.nodes dict and
    wraps each node callable with traced_node(name=node_name).

    This is best-effort — if the graph structure is not accessible, log a
    warning and return without raising.

    Usage:
        graph = builder.compile()
        instrument_graph(graph)
    """
    try:
        nodes = getattr(graph, "nodes", None)
        if nodes is None:
            _LOG.warning("instrument_graph: graph has no accessible .nodes; skipping")
            return
        for node_name, node_fn in list(nodes.items()):
            if callable(node_fn) and not getattr(node_fn, "_agentrace_traced", False):
                wrapped = traced_node(name=node_name)(node_fn)
                wrapped._agentrace_traced = True
                nodes[node_name] = wrapped
    except Exception:
        return
