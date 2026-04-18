"""``agentrace benchmark`` — run bundled suite against an agent."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import typer
from rich.console import Console


def benchmark(
    suite: str = typer.Option(
        ...,
        "--suite",
        "-s",
        help="Benchmark suite name: web_research, code_agent, rag_agent",
    ),
    agent: str = typer.Option(
        ...,
        "--agent",
        "-a",
        help="Path to agent module: my_project.agent or ./agent.py",
    ),
    function: str = typer.Option(
        "run_agent",
        "--function",
        "-f",
        help="Agent function name within the module",
    ),
    metrics: str = typer.Option(
        "tool_selection_accuracy,step_efficiency,reasoning_coherence",
        "--metrics",
        "-m",
        help="Comma-separated metric names",
    ),
    judge_model: str = typer.Option(
        "claude-sonnet-4-20250514",
        "--judge-model",
    ),
    concurrency: int = typer.Option(5, "--concurrency", "-c"),
    output_dir: str = typer.Option("./eval_results/", "--output-dir"),
    filter_tags: str = typer.Option(
        "",
        "--tags",
        help="Comma-separated tags to filter tasks e.g. multi-hop,hard",
    ),
    difficulty: str = typer.Option(
        "",
        "--difficulty",
        help="Filter by difficulty: easy, medium, hard",
    ),
    min_score_threshold: float = typer.Option(
        0.75,
        "--min-score-threshold",
        help="Exit 1 if any aggregate metric score is below this (use 0 to always exit 0)",
    ),
    max_tasks: int = typer.Option(
        0,
        "--max-tasks",
        help="Run at most this many tasks after filters (0 = no limit)",
    ),
) -> None:
    """Run a built-in AgentTrace benchmark suite against your agent."""
    import agentrace
    from agentrace.dataset.benchmarks import load_suite
    from agentrace.report.cli_reporter import CLIReporter

    console = Console()

    try:
        dataset = load_suite(suite)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    tag_list = [t.strip() for t in filter_tags.split(",") if t.strip()]
    diff_filter = difficulty.strip() or None
    if tag_list or diff_filter:
        dataset = dataset.filter(tags=tag_list or None, difficulty=diff_filter)
        console.print(
            f"  Filtered to {len(dataset)} tasks "
            f"(tags={tag_list or 'any'}, difficulty={diff_filter or 'any'})"
        )

    if max_tasks > 0:
        from agentrace.dataset.dataset import Dataset as _Dataset

        sliced = list(dataset)[:max_tasks]
        dataset = _Dataset(sliced)
        console.print(f"  Limited to first {len(dataset)} task(s) (--max-tasks)")

    agent_path = Path(agent)
    try:
        if agent.endswith(".py") or agent_path.suffix == ".py":
            path = agent_path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Agent file not found: {path}")
            spec = importlib.util.spec_from_file_location(
                "_agentrace_benchmark_agent", path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_agentrace_benchmark_agent"] = mod
            spec.loader.exec_module(mod)
        else:
            mod = importlib.import_module(agent)
        agent_fn = getattr(mod, function)
    except (ImportError, AttributeError, FileNotFoundError, OSError) as e:
        console.print(f"[red]Could not load agent: {e}[/red]")
        raise typer.Exit(1) from None

    if inspect.iscoroutinefunction(agent_fn):
        agent_callable: Callable[[str], Awaitable[str]] = agent_fn
    else:

        async def _wrap(q: str) -> str:
            out = agent_fn(q)
            if inspect.isawaitable(out):
                return await out  # type: ignore[no-any-return]
            return str(out)

        agent_callable = _wrap

    suite_info = {
        "web_research": ("🌐", "50 tasks", "Multi-hop web research"),
        "code_agent": ("💻", "30 tasks", "Code generation & debugging"),
        "rag_agent": ("📄", "40 tasks", "Retrieval-augmented generation"),
    }
    icon, size, desc = suite_info.get(suite, ("📊", "? tasks", suite))
    console.print(f"\n{icon}  [bold]AgentTrace Benchmark — {suite}[/bold]")
    console.print(f"   {desc}  ·  {size}  ·  agent: {agent}:{function}\n")

    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]

    result = asyncio.run(
        agentrace.evaluate(
            agent=agent_callable,
            dataset=dataset,
            metrics=metric_list,
            judge_model=judge_model,
            concurrency=concurrency,
            output_dir=output_dir,
        )
    )

    json_path: str | None = None
    html_path: str | None = None
    if output_dir:
        od = str(output_dir)
        json_path = str(Path(od) / f"{result.run_id}.json")
        html_path = str(Path(od) / f"{result.run_id}.html")

    CLIReporter().print_results(result, output_path=json_path, html_path=html_path)

    failed = [
        m for m, score in result.aggregate_scores.items() if score < min_score_threshold
    ]
    if failed:
        console.print(
            f"\n[red]✗ Below threshold ({min_score_threshold}): {', '.join(failed)}[/red]"
        )
        raise typer.Exit(1)
    console.print("\n[green]✓ All metrics passed threshold[/green]")
