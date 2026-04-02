"""Code agent benchmark suite (30 tasks)."""

from __future__ import annotations

from agentrace.dataset.benchmarks._base import load_benchmark
from agentrace.dataset.dataset import Dataset


def load() -> Dataset:
    return load_benchmark("code_agent.json")
