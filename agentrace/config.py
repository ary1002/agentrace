"""YAML-backed evaluation configuration (eval.yaml schema).

Core domain models elsewhere remain stdlib dataclasses; this module defines
config containers and loading helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass
class StorageConfig:
    backend: str = "sqlite"
    path: str = "./agentrace.db"
    dsn: str = ""


@dataclass
class EvalConfig:
    version: int | None = None
    agent: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    metrics: list[Any] = field(default_factory=list)
    runner: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    storage: StorageConfig = field(default_factory=StorageConfig)


def eval_config_from_mapping(raw: dict[str, Any]) -> EvalConfig:
    """Build ``EvalConfig`` from a YAML-loaded mapping (``eval.yaml`` root)."""
    storage_raw = raw.get("storage") or {}
    if not isinstance(storage_raw, dict):
        storage_raw = {}
    storage = StorageConfig(
        backend=str(storage_raw.get("backend", "sqlite")),
        path=str(storage_raw.get("path", "./agentrace.db")),
        dsn=str(storage_raw.get("dsn", "") or ""),
    )

    agent = raw.get("agent")
    dataset = raw.get("dataset")
    metrics = raw.get("metrics")
    runner = raw.get("runner")
    thresholds = raw.get("thresholds")
    judge = raw.get("judge")

    return EvalConfig(
        version=raw.get("version") if raw.get("version") is not None else None,
        agent=dict(agent) if isinstance(agent, dict) else {},
        dataset=dict(dataset) if isinstance(dataset, dict) else {},
        metrics=list(metrics) if isinstance(metrics, list) else [],
        runner=dict(runner) if isinstance(runner, dict) else {},
        thresholds=dict(thresholds) if isinstance(thresholds, dict) else {},
        judge=dict(judge) if isinstance(judge, dict) else {},
        storage=storage,
    )


def load_eval_config(path: str | Path) -> EvalConfig:
    """Load ``eval.yaml`` from disk and return a structured ``EvalConfig``."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    return eval_config_from_mapping(raw)
