"""In-memory ``Dataset`` of ``EvalTask`` rows with filtering and JSON load."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path

from agentrace.runner.models import EvalTask


class Dataset:
    """Ordered collection of evaluation tasks."""

    def __init__(self, tasks: list[EvalTask]) -> None:
        self._tasks = list(tasks)

    @property
    def tasks(self) -> list[EvalTask]:
        return list(self._tasks)

    @property
    def id(self) -> str:
        raw = "".join(t.id for t in self._tasks)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return digest[:8]

    def __iter__(self) -> Iterator[EvalTask]:
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def filter(
        self,
        tags: list[str] | None = None,
        difficulty: str | None = None,
    ) -> Dataset:
        """Return tasks matching all given ``tags`` (subset) and/or ``difficulty``."""

        def ok(task: EvalTask) -> bool:
            if tags is not None:
                for tag in tags:
                    if tag not in task.tags:
                        return False
            if difficulty is not None and task.difficulty != difficulty:
                return False
            return True

        return Dataset([t for t in self._tasks if ok(t)])

    @classmethod
    def from_json(cls, path: str) -> Dataset:
        """Load tasks from a JSON file: a list of objects with ``EvalTask`` field names."""

        raw_path = Path(path)
        text = raw_path.read_text(encoding="utf-8")
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError("Dataset JSON root must be a list")

        field_names = {f.name for f in fields(EvalTask)}
        tasks: list[EvalTask] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each dataset row must be a JSON object")
            kwargs = {k: v for k, v in row.items() if k in field_names}
            tasks.append(EvalTask(**kwargs))

        return cls(tasks)
