"""``agentrace diff`` — compare two versioned JSON eval exports."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text


def diff(
    run_a: str = typer.Argument(..., help="Path to first run JSON file"),
    run_b: str = typer.Argument(..., help="Path to second run JSON file"),
    html: bool = typer.Option(False, "--html", help="Also generate HTML comparison report"),
    output_dir: str = typer.Option("./eval_results/", "--output-dir"),
) -> None:
    """Compare two AgentTrace runs and print a diff table to the terminal."""
    from agentrace.report.html_reporter import HTMLReporter
    from agentrace.report.json_reporter import JSONReporter

    console = Console()
    reporter = JSONReporter()

    try:
        result_a = reporter.read(run_a)
        result_b = reporter.read(run_b)
    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        raise typer.Exit(1) from None
    except ValueError as e:
        console.print(f"[red]Schema error: {e}[/red]")
        raise typer.Exit(1) from None

    console.print("\n[bold]Comparing runs:[/bold]")
    console.print(
        f"  A: {result_a.run_id}  ({result_a.timestamp.strftime('%Y-%m-%d %H:%M')})"
    )
    console.print(
        f"  B: {result_b.run_id}  ({result_b.timestamp.strftime('%Y-%m-%d %H:%M')})\n"
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("METRIC", style="dim", width=28)
    table.add_column("RUN A", justify="right", width=10)
    table.add_column("RUN B", justify="right", width=10)
    table.add_column("DELTA", justify="right", width=10)
    table.add_column("WINNER", justify="center", width=8)

    all_metrics = sorted(
        set(result_a.aggregate_scores) | set(result_b.aggregate_scores)
    )

    for metric in all_metrics:
        score_a = result_a.aggregate_scores.get(metric)
        score_b = result_b.aggregate_scores.get(metric)

        a_str = f"{score_a:.3f}" if score_a is not None else "—"
        b_str = f"{score_b:.3f}" if score_b is not None else "—"

        if score_a is not None and score_b is not None:
            delta = score_b - score_a
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            delta_colour = (
                "green" if delta > 0.005 else ("red" if delta < -0.005 else "dim")
            )
            delta_text = Text(delta_str, style=delta_colour)
            winner = (
                Text("B ▲", style="green")
                if delta > 0.005
                else (
                    Text("A ▲", style="blue")
                    if delta < -0.005
                    else Text("tie", style="dim")
                )
            )
        else:
            delta_text = Text("—", style="dim")
            winner = Text("—", style="dim")

        table.add_row(metric, a_str, b_str, delta_text, winner)

    console.print(table)

    all_failures = sorted(set(result_a.failure_dist) | set(result_b.failure_dist))
    if all_failures:
        console.print("\n[bold]Failure distribution:[/bold]")
        ftable = Table(show_header=True, header_style="bold")
        ftable.add_column("FAILURE TYPE", width=30)
        ftable.add_column("RUN A", justify="right", width=8)
        ftable.add_column("RUN B", justify="right", width=8)
        ftable.add_column("DELTA", justify="right", width=8)
        for ft in all_failures:
            cnt_a = result_a.failure_dist.get(ft, 0)
            cnt_b = result_b.failure_dist.get(ft, 0)
            d = cnt_b - cnt_a
            d_str = f"+{d}" if d > 0 else str(d)
            d_col = "red" if d > 0 else ("green" if d < 0 else "dim")
            ftable.add_row(ft, str(cnt_a), str(cnt_b), Text(d_str, style=d_col))
        console.print(ftable)

    console.print(
        f"\n  cost:   A=${result_a.total_cost_usd:.4f}  "
        f"B=${result_b.total_cost_usd:.4f}\n"
        f"  tokens: A={result_a.total_tokens:,}  B={result_b.total_tokens:,}"
    )

    if html:
        html_reporter = HTMLReporter()
        path = html_reporter.generate(
            result_b, prev_result=result_a, output_dir=output_dir
        )
        console.print(f"\n  [dim]HTML comparison report: {path}[/dim]")
