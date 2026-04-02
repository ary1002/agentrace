"""Minimal async agent for smoke tests and default ``eval.yaml`` examples."""

from __future__ import annotations


async def my_agent(query: str) -> str:
    """Return a fixed string (traced by ``evaluate()``)."""
    _ = query
    return "answer"
