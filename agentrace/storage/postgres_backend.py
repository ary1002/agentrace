"""PostgreSQL persistence via asyncpg connection pool (lazy import)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from agentrace.runner.models import EvalResult, TaskResult
from agentrace.storage.base import BaseStorage
from agentrace.storage.serialization import StorageSerializer


def _record_to_task_row(rec: Any) -> dict[str, Any]:
    d = dict(rec)
    for key in ("metric_scores", "passed", "failure_types"):
        v = d.get(key)
        if not isinstance(v, str):
            d[key] = json.dumps(v)
    return d


def _record_to_run_row(rec: Any) -> dict[str, Any]:
    d = dict(rec)
    ts = d.get("timestamp")
    if isinstance(ts, datetime):
        d["timestamp"] = ts.isoformat()
    for key in ("aggregate_scores", "failure_dist"):
        v = d.get(key)
        if not isinstance(v, str):
            d[key] = json.dumps(v)
    return d


class PostgreSQLStorage(BaseStorage):
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: Any = None

    async def connect(self) -> None:
        try:
            import asyncpg  # type: ignore[import-untyped, import-not-found]
        except ImportError as e:
            raise ImportError(
                "asyncpg is required. pip install agentrace[postgres]"
            ) from e

        self._pool = await asyncpg.create_pool(
            self.dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        async with self._pool.acquire() as conn:
            await self._create_tables(conn)

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def _create_tables(self, conn: Any) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id           TEXT PRIMARY KEY,
                dataset_id       TEXT NOT NULL,
                timestamp        TIMESTAMPTZ NOT NULL,
                aggregate_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
                failure_dist     JSONB NOT NULL DEFAULT '{}'::jsonb,
                total_cost_usd   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                total_tokens     BIGINT NOT NULL DEFAULT 0,
                duration_s       DOUBLE PRECISION NOT NULL DEFAULT 0.0
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_results (
                run_id        TEXT NOT NULL,
                task_id       TEXT NOT NULL,
                metric_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
                passed        JSONB NOT NULL DEFAULT '{}'::jsonb,
                failure_types JSONB NOT NULL DEFAULT '[]'::jsonb,
                error         TEXT,
                trace_json    TEXT,
                PRIMARY KEY (run_id, task_id)
            )
            """
        )

    async def save_run_meta(self, result: EvalResult) -> None:
        assert self._pool is not None
        row = StorageSerializer.eval_result_to_dict(result)
        ts = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
        agg = json.loads(row["aggregate_scores"])
        fd = json.loads(row["failure_dist"])
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO runs (
                    run_id, dataset_id, timestamp, aggregate_scores, failure_dist,
                    total_cost_usd, total_tokens, duration_s
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (run_id) DO UPDATE SET
                    dataset_id = EXCLUDED.dataset_id,
                    timestamp = EXCLUDED.timestamp,
                    aggregate_scores = EXCLUDED.aggregate_scores,
                    failure_dist = EXCLUDED.failure_dist,
                    total_cost_usd = EXCLUDED.total_cost_usd,
                    total_tokens = EXCLUDED.total_tokens,
                    duration_s = EXCLUDED.duration_s
                """,
                row["run_id"],
                row["dataset_id"],
                ts,
                agg,
                fd,
                row["total_cost_usd"],
                row["total_tokens"],
                row["duration_s"],
            )

    async def save_task_result(self, run_id: str, result: TaskResult) -> None:
        assert self._pool is not None
        d = StorageSerializer.task_result_to_dict(result)
        ms = json.loads(d["metric_scores"])
        passed = json.loads(d["passed"])
        fts = json.loads(d["failure_types"])
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO task_results (
                    run_id, task_id, metric_scores, passed, failure_types, error, trace_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (run_id, task_id) DO UPDATE SET
                    metric_scores = EXCLUDED.metric_scores,
                    passed = EXCLUDED.passed,
                    failure_types = EXCLUDED.failure_types,
                    error = EXCLUDED.error,
                    trace_json = EXCLUDED.trace_json
                """,
                run_id,
                d["task_id"],
                ms,
                passed,
                fts,
                d["error"],
                d["trace_json"],
            )

    async def load_completed_task_ids(self, run_id: str) -> set[str]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT task_id FROM task_results WHERE run_id = $1",
                run_id,
            )
        return {str(r["task_id"]) for r in rows}

    async def load_task_results(self, run_id: str) -> list[TaskResult]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM task_results WHERE run_id = $1 ORDER BY task_id",
                run_id,
            )
        return [
            StorageSerializer.task_result_from_dict(_record_to_task_row(r))
            for r in rows
        ]

    async def load_run_meta(self, run_id: str) -> EvalResult | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM runs WHERE run_id = $1", run_id)
        if row is None:
            return None
        return StorageSerializer.eval_result_from_dict(_record_to_run_row(row))

    async def list_runs(self, dataset_id: str | None = None) -> list[dict]:
        assert self._pool is not None
        sql = """
            SELECT
                r.run_id,
                r.dataset_id,
                r.timestamp,
                r.aggregate_scores,
                r.failure_dist,
                r.total_cost_usd,
                r.total_tokens,
                r.duration_s,
                (SELECT COUNT(*)::bigint FROM task_results t WHERE t.run_id = r.run_id) AS task_count
            FROM runs r
        """
        async with self._pool.acquire() as conn:
            if dataset_id is not None:
                rows = await conn.fetch(
                    sql + " WHERE r.dataset_id = $1 ORDER BY r.timestamp DESC",
                    dataset_id,
                )
            else:
                rows = await conn.fetch(sql + " ORDER BY r.timestamp DESC")
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            ts = d["timestamp"]
            if isinstance(ts, datetime):
                d["timestamp"] = ts.isoformat()
            agg = d["aggregate_scores"]
            fd = d["failure_dist"]
            if not isinstance(agg, dict):
                agg = json.loads(agg) if isinstance(agg, str) else {}
            if not isinstance(fd, dict):
                fd = json.loads(fd) if isinstance(fd, str) else {}
            d["aggregate_scores"] = agg
            d["failure_dist"] = fd
            out.append(d)
        return out
