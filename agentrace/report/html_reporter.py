"""Self-contained HTML report rendered via Jinja2 + Chart.js."""

from __future__ import annotations

import html
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agentrace.runner.models import EvalResult, TaskResult

_CHART_JS_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"


def _metric_names_union(result: EvalResult) -> list[str]:
    keys: set[str] = set(result.aggregate_scores.keys())
    for tr in result.task_results:
        keys |= set(tr.metric_scores.keys())
    return sorted(keys)


def _task_status_pass(tr: TaskResult) -> bool:
    return tr.error is None and bool(tr.passed) and all(tr.passed.values())


def _bar_color(score: float, threshold: float = 0.75) -> str:
    if score >= threshold:
        return "var(--green)"
    if score >= threshold - 0.10:
        return "var(--yellow)"
    return "var(--red)"


class HTMLReporter:
    """Generate a single file-friendly HTML evaluation report."""

    def generate(
        self,
        result: EvalResult,
        prev_result: EvalResult | None = None,
        output_dir: str = "./eval_results/",
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{result.run_id}.html")
        content = self._render(result, prev_result)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _render(self, result: EvalResult, prev_result: EvalResult | None) -> str:
        env = Environment(
            loader=FileSystemLoader(
                str(Path(__file__).resolve().parents[2] / "templates")
            ),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template("report.html.j2")
        return template.render(
            page_title=result.run_id,
            css=self._css(),
            chart_tag=self._chart_tag(),
            sections=[
                self._build_header(result),
                self._build_metric_table(result, prev_result),
                self._build_metric_radar(result),
                self._build_failure_chart(result),
                self._build_task_table(result),
                self._build_run_comparison(result, prev_result),
                self._build_cost_latency(result),
            ],
        )

    def _chart_tag(self) -> str:
        try:
            with urllib.request.urlopen(_CHART_JS_URL, timeout=5) as resp:
                body = resp.read().decode("utf-8")
            return f"<script>\n{body}\n</script>"
        except (urllib.error.URLError, OSError, TimeoutError):
            return f'<script src="{html.escape(_CHART_JS_URL, quote=True)}"></script>'

    def _css(self) -> str:
        return """
:root { --bg:#0f1117; --surface:#1a1d27; --border:#2a2d3a; --text:#e2e8f0; --text-dim:#64748b; --green:#22c55e; --yellow:#eab308; --red:#ef4444; --accent:#6366f1; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; line-height: 1.5; }
.wrap { max-width: 1280px; margin: 0 auto; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 16px; }
h1 { margin: 0 0 8px 0; font-size: 1.75rem; }
h2 { margin: 0 0 12px 0; font-size: 1.1rem; color: var(--text-dim); }
table.data { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
table.data th, table.data td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }
table.data th { color: var(--text-dim); font-weight: 600; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: .85rem; font-weight: 600; }
.badge-pass { background: rgba(34,197,94,.2); color: var(--green); }
.badge-fail { background: rgba(239,68,68,.2); color: var(--red); }
.score-bar-wrap { max-width: 200px; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
.score-bar-fill { height: 8px; border-radius: 4px; }
.meta { color: var(--text-dim); font-size: .9rem; }
.meta strong { color: var(--text); }
button.toggle-fail { background: var(--accent); color: #fff; border: none; padding: 8px 14px; border-radius: 6px; cursor: pointer; margin-bottom: 12px; }
.grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.delta-up { color: var(--green); } .delta-down { color: var(--red); } .delta-none { color: var(--text-dim); }
details.trace { margin: 4px 0; padding-left: 8px; border-left: 1px solid var(--border); }
.trace-line { color: var(--text-dim); font-size: .85rem; }
canvas { max-height: 360px; }
"""

    def _build_header(self, result: EvalResult) -> str:
        all_pass = bool(result.aggregate_scores) and all(
            v >= 0.75 for v in result.aggregate_scores.values()
        )
        badge_class = "badge-pass" if all_pass else "badge-fail"
        badge_text = "PASS" if all_pass else "FAIL"
        ts_str = result.timestamp.strftime("%B %d, %Y %H:%M:%S")
        return (
            f'<div class="card"><h1>AgentTrace Evaluation Report</h1>'
            f'<p class="meta"><strong>Run ID</strong> {html.escape(result.run_id)} · '
            f"<strong>Dataset ID</strong> {html.escape(result.dataset_id)} · "
            f"<strong>Timestamp</strong> {html.escape(ts_str)}</p>"
            f'<p class="meta"><strong>Duration</strong> {result.duration_s:.1f}s · '
            f"<strong>Total cost</strong> ${result.total_cost_usd:.4f} · "
            f"<strong>Total tokens</strong> {result.total_tokens:,}</p>"
            f'<p><span class="badge {badge_class}">{badge_text}</span></p></div>'
        )

    def _build_metric_table(
        self, result: EvalResult, prev_result: EvalResult | None
    ) -> str:
        n_tasks = len(result.task_results)
        rows: list[str] = []
        for metric in sorted(result.aggregate_scores.keys()):
            score = result.aggregate_scores[metric]
            bar_w = f"{max(0.0, min(1.0, score)) * 100:.0f}%"
            passed_n = sum(
                1 for r in result.task_results if r.passed.get(metric, False)
            )
            if prev_result is None or metric not in prev_result.aggregate_scores:
                vs = "—"
            else:
                delta = score - prev_result.aggregate_scores[metric]
                vs = (
                    '<span class="delta-none">─</span>'
                    if abs(delta) < 0.005
                    else (
                        f'<span class="delta-up">▲ +{delta:.2f}</span>'
                        if delta > 0
                        else f'<span class="delta-down">▼ {delta:.2f}</span>'
                    )
                )
            rows.append(
                f"<tr><td>{html.escape(metric)}</td><td>{score:.3f}</td>"
                f'<td><div class="score-bar-wrap"><div class="score-bar-fill" style="width:{bar_w};background-color:{_bar_color(score)}"></div></div></td>'
                f"<td>{passed_n}/{n_tasks}</td><td>{vs}</td></tr>"
            )
        return (
            '<div class="card"><h2>Metrics</h2><table class="data"><thead>'
            "<tr><th>Metric</th><th>Score</th><th>Bar</th><th>Pass rate</th><th>vs Last Run</th></tr>"
            f"</thead><tbody>{''.join(rows)}</tbody></table></div>"
        )

    def _build_metric_radar(self, result: EvalResult) -> str:
        if not result.aggregate_scores:
            return ""
        labels = sorted(result.aggregate_scores.keys())
        data = [result.aggregate_scores[m] for m in labels]
        return f"""
<div class="card">
  <h2>Metric Radar</h2>
  <canvas id="metricRadar"></canvas>
  <script>(function(){{
    const el=document.getElementById('metricRadar');
    if(el && typeof Chart!=='undefined'){{
      new Chart(el,{{type:'radar',data:{{labels:{json.dumps(labels)},datasets:[{{label:'Aggregate score',data:{json.dumps(data)},fill:true}}]}},options:{{scales:{{r:{{min:0,max:1}}}}}}}});
    }}
  }})();</script>
</div>"""

    def _build_failure_chart(self, result: EvalResult) -> str:
        if not result.failure_dist:
            return ""
        labels = list(result.failure_dist.keys())
        vals = [result.failure_dist[k] for k in labels]
        rows = "".join(
            f"<tr><td>{html.escape(ft)}</td><td>{cnt}</td><td>{(100.0 * cnt / max(1, len(result.task_results))):.1f}%</td></tr>"
            for ft, cnt in sorted(
                result.failure_dist.items(), key=lambda x: (-x[1], x[0])
            )
        )
        return f"""
<div class="card"><h2>Failure Taxonomy</h2><canvas id="failureChart"></canvas>
<script>(function(){{
const ctx=document.getElementById('failureChart');
if(ctx && typeof Chart!=='undefined'){{new Chart(ctx,{{type:'doughnut',data:{{labels:{json.dumps(labels)},datasets:[{{data:{json.dumps(vals)}}}]}}}});}}
}})();</script>
<table class="data"><thead><tr><th>Failure Type</th><th>Count</th><th>% of tasks</th></tr></thead><tbody>{rows}</tbody></table></div>"""

    def _render_trace_node(self, node, depth: int = 0) -> str:
        span = node.span
        label = f"{span.span_type} · tool={span.input.get('tool_name', '—')} · {span.latency_ms:.1f}ms · ${span.cost_usd:.4f}"
        children = "".join(
            self._render_trace_node(ch, depth + 1) for ch in node.children
        )
        return (
            f'<details class="trace" {"open" if depth == 0 else ""}>'
            f"<summary>{html.escape(label)}</summary>{children}</details>"
        )

    def _build_task_table(self, result: EvalResult) -> str:
        metric_cols = _metric_names_union(result)
        th_metrics = "".join(f"<th>{html.escape(m)}</th>" for m in metric_cols)
        rows: list[str] = []
        for tr in result.task_results:
            status = "pass" if _task_status_pass(tr) else "fail"
            status_html = (
                '<span style="color:var(--green)">✓ pass</span>'
                if status == "pass"
                else '<span style="color:var(--red)">✗ fail</span>'
            )
            metric_cells = "".join(
                f'<td style="color:{_bar_color(tr.metric_scores[m])}">{tr.metric_scores[m]:.2f}</td>'
                if m in tr.metric_scores
                else '<td style="color:var(--text-dim)">—</td>'
                for m in metric_cols
            )
            fails = ", ".join(html.escape(ft) for ft in tr.failure_types) or "—"
            err = (
                html.escape(tr.error[:80] + ("…" if len(tr.error) > 80 else ""))
                if tr.error
                else "—"
            )
            trace_html = (
                self._render_trace_node(tr.trace.trace_tree)
                if tr.trace is not None
                else "—"
            )
            rows.append(
                f'<tr class="task-row" data-status="{status}"><td>{html.escape(tr.task_id)}</td><td>{status_html}</td>{metric_cells}<td>{fails}</td><td>{err}</td><td>{trace_html}</td></tr>'
            )
        return f"""
<div class="card"><h2>Per-Task Results</h2>
<button type="button" class="toggle-fail" onclick="toggleErrors()">Show only failed tasks</button>
<script>function toggleErrors(){{document.querySelectorAll('.task-row').forEach(r=>{{if(r.dataset.status==='pass'){{r.style.display=r.style.display==='none'?'':'none';}}}});}}</script>
<table class="data"><thead><tr><th>Task ID</th><th>Status</th>{th_metrics}<th>Failures</th><th>Error</th><th>Trace</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>"""

    def _build_run_comparison(
        self, result: EvalResult, prev_result: EvalResult | None
    ) -> str:
        if prev_result is None:
            return ""
        metrics = sorted(
            set(result.aggregate_scores) | set(prev_result.aggregate_scores)
        )
        if not metrics:
            return ""
        cur = [result.aggregate_scores.get(m, 0.0) for m in metrics]
        prev = [prev_result.aggregate_scores.get(m, 0.0) for m in metrics]
        return f"""
<div class="card"><h2>Run Comparison</h2><canvas id="comparisonChart"></canvas>
<script>(function(){{
const el=document.getElementById('comparisonChart');
if(el && typeof Chart!=='undefined'){{new Chart(el,{{type:'bar',data:{{labels:{json.dumps(metrics)},datasets:[{{label:{json.dumps(result.run_id)},data:{json.dumps(cur)}}},{{label:{json.dumps(prev_result.run_id)},data:{json.dumps(prev)}}}]}}}});}}
}})();</script></div>"""

    def _build_cost_latency(self, result: EvalResult) -> str:
        task_ids = [tr.task_id for tr in result.task_results]
        latencies = [
            tr.trace.total_latency_ms if tr.trace is not None else 0.0
            for tr in result.task_results
        ]
        costs = [
            tr.trace.total_cost_usd if tr.trace is not None else 0.0
            for tr in result.task_results
        ]
        n = len(result.task_results)
        denom = max(n, 1)
        pass_n = sum(1 for t in result.task_results if _task_status_pass(t))
        return f"""
<div class="card"><h2>Cost, Latency, and Token Summary</h2>
<div class="grid-2">
  <div class="meta"><p><strong>Total cost</strong> ${result.total_cost_usd:.4f}</p><p><strong>Total tokens</strong> {result.total_tokens:,}</p><p><strong>Duration</strong> {result.duration_s:.1f}s</p><p><strong>Pass rate</strong> {(100.0 * pass_n / denom):.0f}%</p></div>
  <div><canvas id="latencyHistogram"></canvas><canvas id="costHistogram"></canvas></div>
</div>
<script>(function(){{
if(typeof Chart==='undefined') return;
new Chart(document.getElementById('latencyHistogram'),{{type:'bar',data:{{labels:{json.dumps(task_ids)},datasets:[{{label:'Latency (ms)',data:{json.dumps(latencies)}}}]}}}});
new Chart(document.getElementById('costHistogram'),{{type:'bar',data:{{labels:{json.dumps(task_ids)},datasets:[{{label:'Cost (USD)',data:{json.dumps(costs)}}}]}}}});
}})();</script>
</div>"""
