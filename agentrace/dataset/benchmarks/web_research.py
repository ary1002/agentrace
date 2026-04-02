"""Web research benchmark suite (50 tasks)."""

from __future__ import annotations

from agentrace.dataset.benchmarks._base import load_benchmark
from agentrace.dataset.dataset import Dataset


def load() -> Dataset:
    return load_benchmark("web_research.json")
