"""Coordinates per-task checkpoint writes between the runner and storage."""

from __future__ import annotations

import warnings

from agentrace.runner.models import TaskResult
from agentrace.storage.base import BaseStorage


class CheckpointManager:
    def __init__(self, run_id: str, storage: BaseStorage) -> None:
        self.run_id = run_id
        self.storage = storage
        self._completed: set[str] = set()

    async def load_existing(self) -> list[TaskResult]:
        """
        Called once at runner startup.
        Populates self._completed from storage.
        Returns the list of already-completed TaskResult objects so the
        runner can include them in the final EvalResult without re-running.
        Prints nothing — caller handles the "resuming" message.
        """
        completed_results = await self.storage.load_task_results(self.run_id)
        self._completed = {r.task_id for r in completed_results}
        return completed_results

    def is_complete(self, task_id: str) -> bool:
        return task_id in self._completed

    async def save(self, result: TaskResult) -> None:
        """
        Persist a task result and update the in-memory cache.
        Must not raise — wrap in try/except and log warning on failure.
        The runner must never crash because checkpointing failed.
        """
        try:
            await self.storage.save_task_result(self.run_id, result)
            self._completed.add(result.task_id)
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"Checkpoint save failed for task {result.task_id}: {e}",
                stacklevel=2,
            )

    @property
    def completed_count(self) -> int:
        return len(self._completed)
