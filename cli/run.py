"""``agentrace run`` — load config, run evaluation, print Rich report."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

import typer
from rich.console import Console
from rich.table import Table

from agentrace.config import load_eval_config
from agentrace.dataset.dataset import Dataset
from agentrace.report.cli_reporter import CLIReporter
from agentrace.runner.runner import evaluate
from agentrace.storage import SQLiteStorage


def run(
    config: str = typer.Option(
        "eval.yaml",
        "--config",
        "-c",
        help="Path to eval.yaml config file",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Resume a specific run by ID. Get IDs from `agentrace runs` command.",
    ),
) -> None:
    """Load YAML config, import agent, run ``evaluate``, print results, enforce thresholds."""

    try:
        path = Path(config)
        ec = load_eval_config(path)

        agent_cfg = ec.agent
        module_name = agent_cfg["module"]
        function_name = agent_cfg["function"]

        dataset_cfg = ec.dataset
        console = Console()
        if dataset_cfg.get("suite"):
            suite_name = str(dataset_cfg["suite"])
            suite_name = suite_name.replace("agentrace.benchmarks.", "")
            from agentrace.dataset.benchmarks import load_suite

            try:
                dataset = load_suite(suite_name)
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
                raise typer.Exit(1) from None
        elif dataset_cfg.get("path"):
            dataset = Dataset.from_json(str(dataset_cfg["path"]))
        else:
            console.print(
                "[red]dataset config must specify either suite or path[/red]"
            )
            raise typer.Exit(1)

        metrics = ec.metrics
        if not isinstance(metrics, list):
            raise ValueError("metrics must be a list")

        runner_cfg = ec.runner
        concurrency = int(runner_cfg.get("concurrency", 1))
        output_dir = runner_cfg.get("output_dir", "./eval_results/")
        raw_timeout = runner_cfg.get("timeout_per_task")
        timeout_per_task = float(raw_timeout) if raw_timeout is not None else None
        checkpoint_dir_raw = runner_cfg.get("checkpoint_dir")
        checkpoint_dir = str(checkpoint_dir_raw) if checkpoint_dir_raw else None

        thresholds = ec.thresholds
        if not isinstance(thresholds, dict):
            raise ValueError("thresholds must be a mapping")

        judge_cfg = ec.judge
        judge_model = judge_cfg.get("model", "claude-sonnet-4-20250514")

        storage_config = {
            "backend": ec.storage.backend,
            "path": ec.storage.path,
            "dsn": ec.storage.dsn,
        }

        module = importlib.import_module(module_name)
        agent_fn = getattr(module, function_name)

        if inspect.iscoroutinefunction(agent_fn):
            agent: Callable[[str], Awaitable[str]] = agent_fn
        else:

            async def _wrap(q: str) -> str:
                out = agent_fn(q)
                if inspect.isawaitable(out):
                    return await out  # type: ignore[no-any-return]
                return str(out)

            agent = _wrap

        result = asyncio.run(
            evaluate(
                agent=agent,
                dataset=dataset,
                metrics=[str(m) for m in metrics],
                judge_model=str(judge_model),
                concurrency=concurrency,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                thresholds={str(k): float(v) for k, v in thresholds.items()},
                storage_config=storage_config,
                run_id=run_id,
                timeout_per_task=timeout_per_task,
            )
        )

        json_path: str | None = None
        html_path: str | None = None
        if output_dir:
            od = str(output_dir)
            json_path = os.path.join(od, f"{result.run_id}.json")
            html_path = os.path.join(od, f"{result.run_id}.html")

        reporter = CLIReporter(thresholds={str(k): float(v) for k, v in thresholds.items()})
        reporter.print_results(result, output_path=json_path, html_path=html_path)

        for metric_name, score in result.aggregate_scores.items():
            threshold = float(thresholds.get(metric_name, 0.75))
            if score < threshold:
                raise typer.Exit(code=1)

    except FileNotFoundError:
        typer.echo(f"Config file not found: {config}", err=True)
        raise typer.Exit(1) from None
    except KeyError as e:
        typer.echo(f"Missing required config key: {e}", err=True)
        raise typer.Exit(1) from None
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1) from None


def runs(
    storage_path: str = typer.Option(
        "./agentrace.db",
        "--db",
        help="Path to the SQLite database used for evaluation runs.",
    ),
) -> None:
    """List all evaluation runs stored in the local database."""

    async def _list() -> list[dict]:
        async with SQLiteStorage(db_path=storage_path) as s:
            return await s.list_runs()

    rows = asyncio.run(_list())
    if not rows:
        print("No runs found.")
        return

    console = Console()
    table = Table(title="AgentTrace Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Dataset", style="dim")
    table.add_column("Timestamp", style="dim")
    table.add_column("Duration", style="dim")
    table.add_column("Cost", style="dim")
    table.add_column("Tasks", style="dim")
    table.add_column("Scores", style="dim")

    for run in rows:
        scores = run.get("aggregate_scores", {})
        score_str = (
            ", ".join(f"{k}: {float(v):.2f}" for k, v in scores.items())
            if scores
            else "—"
        )
        ts = run["timestamp"]
        ts_display = (
            ts[:19].replace("T", " ") if isinstance(ts, str) and len(ts) >= 19 else str(ts)
        )
        task_n = run.get("task_count", "—")
        table.add_row(
            run["run_id"],
            run["dataset_id"],
            ts_display,
            f"{float(run['duration_s']):.1f}s",
            f"${float(run['total_cost_usd']):.4f}",
            str(task_n),
            score_str,
        )
    console.print(table)
