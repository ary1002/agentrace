# eval.yaml additions for Phase 10:
#
# storage:
#   backend: sqlite              # default — zero infrastructure
#   path: ./agentrace.db
#
# For team mode:
# storage:
#   backend: postgres
#   dsn: postgresql://user:pass@localhost:5432/agentrace

from __future__ import annotations

from agentrace.storage.base import BaseStorage
from agentrace.storage.postgres_backend import PostgreSQLStorage
from agentrace.storage.serialization import StorageSerializer
from agentrace.storage.sqlite_backend import SQLiteStorage


def get_storage(config: dict) -> BaseStorage:
    """
    Factory function. Reads config dict with shape:
      {'backend': 'sqlite', 'path': './agentrace.db'}
      {'backend': 'postgres', 'dsn': 'postgresql://...'}

    Returns the appropriate BaseStorage subclass instance.
    Raises ValueError for unknown backend.
    """
    backend = config.get("backend", "sqlite")
    if backend == "sqlite":
        path = config.get("path", "./agentrace.db")
        return SQLiteStorage(db_path=path)
    if backend == "postgres":
        dsn = config.get("dsn")
        if not dsn:
            raise ValueError("PostgreSQL backend requires 'dsn' in storage config")
        return PostgreSQLStorage(dsn=dsn)
    raise ValueError(
        f"Unknown storage backend: '{backend}'. Choose 'sqlite' or 'postgres'."
    )


__all__ = [
    "BaseStorage",
    "SQLiteStorage",
    "PostgreSQLStorage",
    "StorageSerializer",
    "get_storage",
]
