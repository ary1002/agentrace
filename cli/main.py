"""Typer application entrypoint for the ``agentrace`` CLI."""

from __future__ import annotations

import typer

from cli.run import run

app = typer.Typer(
    name="agentrace",
    help="Framework-agnostic evaluation and tracing for LLM agents.",
    add_completion=False,
)

app.command("run")(run)


@app.command("benchmark")
def benchmark() -> None:
    """Placeholder for bundled benchmark suites."""
    typer.echo("coming soon")


@app.command("diff")
def diff() -> None:
    """Placeholder for comparing eval JSON exports."""
    typer.echo("coming soon")


def main() -> None:
    """Invoke the Typer application."""
    app()


if __name__ == "__main__":
    main()
