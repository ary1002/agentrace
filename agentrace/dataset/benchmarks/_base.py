"""Load bundled benchmark JSON into a ``Dataset``."""

from __future__ import annotations

import json
import os

from agentrace.dataset.dataset import Dataset
from agentrace.runner.models import EvalTask


def load_benchmark(filename: str) -> Dataset:
    """Load ``agentrace/dataset/benchmarks/{filename}``."""
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    tasks = [
        EvalTask(
            id=t["id"],
            query=t["query"],
            expected_tools=t.get("expected_tools"),
            expected_answer=t.get("expected_answer"),
            optimal_steps=t.get("optimal_steps"),
            tags=t.get("tags", []),
            difficulty=t.get("difficulty", "medium"),
        )
        for t in raw
    ]
    return Dataset(tasks)
