"""``agentrace run`` — load config, run evaluation, print Rich report."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

import typer
import yaml  # type: ignore[import-untyped]

from agentrace.dataset.dataset import Dataset
from agentrace.report.cli_reporter import CLIReporter
from agentrace.runner.runner import evaluate


def run(
    config: str = typer.Option(
        "eval.yaml",
        "--config",
        "-c",
        help="Path to eval.yaml config file",
    ),
) -> None:
    """Load YAML config, import agent, run ``evaluate``, print results, enforce thresholds."""

    try:
        path = Path(config)
        text = path.read_text(encoding="utf-8")
        cfg: dict[str, Any] = yaml.safe_load(text)
        if not isinstance(cfg, dict):
            raise ValueError("Config root must be a mapping")

        agent_cfg = cfg["agent"]
        module_name = agent_cfg["module"]
        function_name = agent_cfg["function"]

        dataset_cfg = cfg["dataset"]
        if dataset_cfg.get("suite"):
            typer.echo("built-in suites coming in v0.3", err=True)
            raise typer.Exit(1)
        dataset_path = dataset_cfg.get("path")
        if not dataset_path:
            raise KeyError("dataset.path")

        metrics = cfg["metrics"]
        if not isinstance(metrics, list):
            raise ValueError("metrics must be a list")

        runner_cfg = cfg.get("runner") or {}
        concurrency = int(runner_cfg.get("concurrency", 1))
        output_dir = runner_cfg.get("output_dir", "./eval_results/")
        timeout_per_task = runner_cfg.get("timeout_per_task")
        _ = timeout_per_task

        thresholds = cfg.get("thresholds") or {}
        if not isinstance(thresholds, dict):
            raise ValueError("thresholds must be a mapping")

        judge_cfg = cfg.get("judge") or {}
        judge_model = judge_cfg.get("model", "claude-sonnet-4-20250514")

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

        dataset = Dataset.from_json(str(dataset_path))

        result = asyncio.run(
            evaluate(
                agent=agent,
                dataset=dataset,
                metrics=[str(m) for m in metrics],
                judge_model=str(judge_model),
                concurrency=concurrency,
                output_dir=output_dir,
                thresholds={str(k): float(v) for k, v in thresholds.items()},
            )
        )

        out_path: str | None = None
        if output_dir:
            out_path = os.path.join(str(output_dir), f"{result.run_id}.json")

        reporter = CLIReporter(thresholds={str(k): float(v) for k, v in thresholds.items()})
        reporter.print_results(result, output_path=out_path)

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
