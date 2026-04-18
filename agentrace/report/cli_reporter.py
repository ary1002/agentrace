"""Rich-based terminal summary for ``EvalResult``."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from agentrace.runner.models import EvalResult


class CLIReporter:
    """Print evaluation summaries with tables, bars, and pass rates."""

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self.thresholds = thresholds or {}
        self.console = Console()

    def _score_style(self, score: float, thr: float) -> str:
        if score >= thr:
            return "green"
        if score >= thr - 0.10:
            return "yellow"
        return "red"

    def _score_bar(self, score: float) -> str:
        s = max(0.0, min(1.0, score))
        filled = min(6, max(0, int(round(s * 6))))
        return "█" * filled + "░" * (6 - filled)

    def print_results(
        self,
        result: EvalResult,
        prev_result: EvalResult | None = None,
        *,
        output_path: str | None = None,
        html_path: str | None = None,
    ) -> None:
        """Render header, metric table, cost line, errors, and optional footer path."""

        n_tasks = len(result.task_results)
        subtitle = (
            f"dataset: {result.dataset_id}  ·  {n_tasks} tasks  ·  "
            f"{result.duration_s:.1f}s"
        )
        header = Text()
        header.append("AgentTrace  ·  ", style=Style(dim=True))
        header.append(result.run_id, style=Style(bold=True))
        header.append("\n")
        header.append(subtitle, style=Style(dim=True))

        panel = Panel(
            header,
            border_style=Style(dim=True),
            title_align="left",
        )
        self.console.print(panel)

        table = Table(show_header=True, header_style="bold")
        table.add_column("METRIC")
        table.add_column("SCORE")
        table.add_column("PASS RATE")
        table.add_column("vs LAST RUN")

        for metric_name in sorted(result.aggregate_scores.keys()):
            score = result.aggregate_scores[metric_name]
            thr = float(self.thresholds.get(metric_name, 0.75))
            style_name = self._score_style(score, thr)
            bar = self._score_bar(score)
            score_cell = Text()
            score_cell.append(f"{bar}  ", style=style_name)
            score_cell.append(f"{score:.2f}", style=style_name)

            passed_n = sum(
                1
                for r in result.task_results
                if metric_name in r.passed and r.passed[metric_name]
            )
            pass_cell = f"{passed_n}/{n_tasks}"

            vs_cell = Text("—", style=Style(dim=True))
            if prev_result is not None and metric_name in prev_result.aggregate_scores:
                prev_s = prev_result.aggregate_scores[metric_name]
                delta = score - prev_s
                if abs(delta) < 1e-9:
                    vs_cell = Text("─", style=Style(dim=True))
                elif delta > 0:
                    vs_cell = Text(f"▲ +{delta:.2f}", style="green")
                else:
                    vs_cell = Text(f"▼ {delta:.2f}", style="red")

            table.add_row(metric_name, score_cell, pass_cell, vs_cell)

        self.console.print(table)

        if result.failure_dist:
            total_cases = sum(result.failure_dist.values())
            sorted_types = sorted(
                result.failure_dist.items(),
                key=lambda x: (-x[1], x[0]),
            )
            top_type, top_n = sorted_types[0]
            top_pct = 100.0 * top_n / total_cases if total_cases else 0.0

            div = Text(
                "─────────────────────────────────────────", style=Style(dim=True)
            )
            self.console.print(div)
            title = Text("FAILURE BREAKDOWN\n", style=Style(bold=True))
            self.console.print(title)
            top_line = Text()
            top_line.append("Top failure type: ", style=Style(dim=True))
            top_line.append(top_type, style=Style(bold=True))
            top_line.append(
                f" ({top_n} cases, {top_pct:.0f}%)",
                style=Style(dim=True),
            )
            self.console.print(top_line)
            self.console.print()

            critical_types = frozenset(
                {
                    "HALLUCINATED_TOOL_CALL",
                    "PREMATURE_TERMINATION",
                    "FAITHFULNESS_FAILURE",
                }
            )
            moderate_types = frozenset(
                {
                    "WRONG_TOOL_SELECTED",
                    "CORRECT_TOOL_WRONG_ARGS",
                    "REASONING_BREAK",
                    "CONTEXT_OVERFLOW",
                }
            )

            for ft, count in sorted_types:
                pct = 100.0 * count / total_cases if total_cases else 0.0
                filled = min(10, max(0, round(pct / 10)))
                bar = "█" * filled + "░" * (10 - filled)
                if ft in critical_types:
                    style = "red"
                elif ft in moderate_types:
                    style = "yellow"
                elif ft == "REDUNDANT_LOOP":
                    style = "dim"
                else:
                    style = "dim"
                line = Text()
                line.append(f"{ft:24} ", style=style)
                line.append(f"{bar}  ", style=style)
                line.append(f"{count:3} cases  {pct:3.0f}%", style=style)
                self.console.print(line)
            self.console.print(div)

        per_task_cost = result.total_cost_usd / max(n_tasks, 1)
        cost_line = Text()
        cost_line.append("  ", style=Style(dim=True))
        cost_line.append(f"cost_per_task ${per_task_cost:.3f}", style=Style(dim=True))
        cost_line.append("   ·   ", style=Style(dim=True))
        cost_line.append(
            f"total_tokens {result.total_tokens:,}",
            style=Style(dim=True),
        )
        cost_line.append("   ·   ", style=Style(dim=True))
        cost_line.append(
            f"duration {result.duration_s:.1f}s",
            style=Style(dim=True),
        )
        self.console.print(cost_line)

        errors = [(r.task_id, r.error) for r in result.task_results if r.error]
        if errors:
            warn = Text()
            warn.append(
                f"⚠  {len(errors)} tasks failed to execute:\n",
                style="yellow",
            )
            self.console.print(warn)
            for task_id, err in errors:
                line = Text(f"    {task_id}: {err}\n", style="yellow")
                self.console.print(line)

        if output_path:
            self.console.print(
                Text(f"  JSON: {output_path}", style=Style(dim=True, italic=True))
            )
        if html_path:
            self.console.print(
                Text(f"  HTML: {html_path}", style=Style(dim=True, italic=True))
            )
