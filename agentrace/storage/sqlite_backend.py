"""SQLite persistence via aiosqlite (lazy import)."""

from __future__ import annotations

import json
from typing import Any

from agentrace.runner.models import EvalResult, TaskResult
from agentrace.storage.base import BaseStorage
from agentrace.storage.serialization import StorageSerializer


class SQLiteStorage(BaseStorage):
    def __init__(self, db_path: str = "./agentrace.db") -> None:
        self.db_path = db_path
        self._conn: Any = None

    async def connect(self) -> None:
        try:
            import aiosqlite
        except ImportError as e:
            raise ImportError(
                "aiosqlite is required. pip install agentrace[sqlite]"
            ) from e

        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._create_tables()
        await self._conn.commit()

    async def disconnect(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _create_tables(self) -> None:
        assert self._conn is not None
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id           TEXT PRIMARY KEY,
                dataset_id       TEXT NOT NULL,
                timestamp        TEXT NOT NULL,
                aggregate_scores TEXT NOT NULL,
                failure_dist     TEXT NOT NULL,
                total_cost_usd   REAL NOT NULL DEFAULT 0.0,
                total_tokens     INTEGER NOT NULL DEFAULT 0,
                duration_s       REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_results (
                run_id        TEXT NOT NULL,
                task_id       TEXT NOT NULL,
                metric_scores TEXT NOT NULL,
                passed        TEXT NOT NULL,
                failure_types TEXT NOT NULL,
                error         TEXT,
                trace_json    TEXT,
                PRIMARY KEY (run_id, task_id)
            )
            """
        )

    async def save_run_meta(self, result: EvalResult) -> None:
        assert self._conn is not None
        row = StorageSerializer.eval_result_to_dict(result)
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, dataset_id, timestamp, aggregate_scores, failure_dist,
                total_cost_usd, total_tokens, duration_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["run_id"],
                row["dataset_id"],
                row["timestamp"],
                row["aggregate_scores"],
                row["failure_dist"],
                row["total_cost_usd"],
                row["total_tokens"],
                row["duration_s"],
            ),
        )
        await self._conn.commit()

    async def save_task_result(self, run_id: str, result: TaskResult) -> None:
        assert self._conn is not None
        d = StorageSerializer.task_result_to_dict(result)
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO task_results (
                run_id, task_id, metric_scores, passed, failure_types, error, trace_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                d["task_id"],
                d["metric_scores"],
                d["passed"],
                d["failure_types"],
                d["error"],
                d["trace_json"],
            ),
        )
        await self._conn.commit()

    async def load_completed_task_ids(self, run_id: str) -> set[str]:
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT task_id FROM task_results WHERE run_id = ?",
            (run_id,),
        )
        rows = await cur.fetchall()
        return {str(r["task_id"]) for r in rows}

    async def load_task_results(self, run_id: str) -> list[TaskResult]:
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT * FROM task_results WHERE run_id = ? ORDER BY task_id",
            (run_id,),
        )
        rows = await cur.fetchall()
        return [StorageSerializer.task_result_from_dict(dict(r)) for r in rows]

    async def load_run_meta(self, run_id: str) -> EvalResult | None:
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return StorageSerializer.eval_result_from_dict(dict(row))

    async def list_runs(self, dataset_id: str | None = None) -> list[dict]:
        assert self._conn is not None
        base = """
            SELECT
                r.run_id,
                r.dataset_id,
                r.timestamp,
                r.aggregate_scores,
                r.failure_dist,
                r.total_cost_usd,
                r.total_tokens,
                r.duration_s,
                (SELECT COUNT(*) FROM task_results t WHERE t.run_id = r.run_id) AS task_count
            FROM runs r
        """
        if dataset_id is not None:
            cur = await self._conn.execute(
                base + " WHERE r.dataset_id = ? ORDER BY r.timestamp DESC",
                (dataset_id,),
            )
        else:
            cur = await self._conn.execute(base + " ORDER BY r.timestamp DESC")
        rows = await cur.fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            d["aggregate_scores"] = json.loads(d["aggregate_scores"])
            d["failure_dist"] = json.loads(d["failure_dist"])
            out.append(d)
        return out
