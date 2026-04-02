"""Abstract storage interface for evaluation run metadata and task results."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentrace.runner.models import EvalResult, TaskResult


class BaseStorage(ABC):
    @abstractmethod
    async def connect(self) -> None:
        """Open connection / create tables if not exist. Called once on startup."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection gracefully."""

    @abstractmethod
    async def save_run_meta(self, result: EvalResult) -> None:
        """
        Upsert the run shell — everything in EvalResult EXCEPT task_results.
        Called once at the start of a run with empty aggregate_scores,
        then again at the end with final aggregates.
        Keyed on result.run_id.
        """

    @abstractmethod
    async def save_task_result(self, run_id: str, result: TaskResult) -> None:
        """
        Persist a single completed TaskResult immediately after it finishes.
        Keyed on (run_id, result.task_id).
        If the row already exists (resume scenario), upsert silently.
        """

    @abstractmethod
    async def load_completed_task_ids(self, run_id: str) -> set[str]:
        """
        Return the set of task_ids already persisted for this run_id.
        Used at runner startup to determine which tasks to skip.
        Returns empty set if run_id not found.
        """

    @abstractmethod
    async def load_task_results(self, run_id: str) -> list[TaskResult]:
        """
        Load all persisted TaskResult rows for a run_id.
        Used on resume to reconstruct already-completed results.
        Returns [] if none found.
        """

    @abstractmethod
    async def load_run_meta(self, run_id: str) -> EvalResult | None:
        """
        Load the run shell for a given run_id.
        Returns None if not found.
        """

    @abstractmethod
    async def list_runs(self, dataset_id: str | None = None) -> list[dict]:
        """
        Return a list of run metadata dicts for display in CLI.
        Each dict has at minimum: run_id, dataset_id, timestamp, duration_s,
        aggregate_scores, total_cost_usd, task_count (when available).
        Optionally filter by dataset_id.
        Ordered by timestamp descending.
        """

    async def __aenter__(self) -> BaseStorage:
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.disconnect()
