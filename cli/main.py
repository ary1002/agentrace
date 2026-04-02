"""Typer application entrypoint for the ``agentrace`` CLI."""

from __future__ import annotations

import typer

from cli.benchmark import benchmark
from cli.diff import diff
from cli.run import run, runs

app = typer.Typer(
    name="agentrace",
    help="Framework-agnostic evaluation and tracing for LLM agents.",
    add_completion=False,
)

app.command("run")(run)
app.command("runs")(runs)
app.command("benchmark")(benchmark)
app.command("diff")(diff)


def main() -> None:
    """Invoke the Typer application."""
    app()


if __name__ == "__main__":
    main()
