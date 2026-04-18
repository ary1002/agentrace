"""Built-in benchmark suites (web research, code agent, RAG)."""

from __future__ import annotations

from agentrace.dataset.dataset import Dataset

from . import code_agent, rag_agent, web_research

AVAILABLE_SUITES: dict[str, str] = {
    "web_research": "agentrace.dataset.benchmarks.web_research  (50 tasks)",
    "code_agent": "agentrace.dataset.benchmarks.code_agent    (30 tasks)",
    "rag_agent": "agentrace.dataset.benchmarks.rag_agent     (40 tasks)",
}


def load_suite(name: str) -> Dataset:
    """Load a built-in benchmark suite by name."""
    if name == "web_research":
        return web_research.load()
    if name == "code_agent":
        return code_agent.load()
    if name == "rag_agent":
        return rag_agent.load()
    available = "\n  ".join(AVAILABLE_SUITES.values())
    raise ValueError(
        f"Unknown benchmark suite: '{name}'.\nAvailable suites:\n  {available}"
    )


__all__ = ["load_suite", "AVAILABLE_SUITES"]
