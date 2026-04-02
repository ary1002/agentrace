"""Self-contained HTML report with inline CSS and Chart.js (fetch or CDN fallback)."""

from __future__ import annotations

import html
import json
import os
import urllib.error
import urllib.request

from agentrace.runner.models import EvalResult, TaskResult

_CHART_JS_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"


def _metric_names_union(result: EvalResult) -> list[str]:
    keys: set[str] = set(result.aggregate_scores.keys())
    for tr in result.task_results:
        keys |= set(tr.metric_scores.keys())
    return sorted(keys)


def _task_status_pass(tr: TaskResult) -> bool:
    if tr.error is not None:
        return False
    if not tr.passed:
        return False
    return all(tr.passed.values())


def _bar_color(score: float, threshold: float = 0.75) -> str:
    if score >= threshold:
        return "var(--green)"
    if score >= threshold - 0.10:
        return "var(--yellow)"
    return "var(--red)"


def _text_color(score: float, threshold: float = 0.75) -> str:
    return _bar_color(score, threshold)


class HTMLReporter:
    """Generate a single file://-friendly HTML evaluation report."""

    def generate(
        self,
        result: EvalResult,
        prev_result: EvalResult | None = None,
        output_dir: str = "./eval_results/",
    ) -> str:
        """Write ``{output_dir}/{run_id}.html`` and return the path."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{result.run_id}.html")
        content = self._render(result, prev_result)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _render(self, result: EvalResult, prev_result: EvalResult | None) -> str:
        parts = [
            self._build_header(result),
            self._build_metric_table(result, prev_result),
            self._build_failure_chart(result),
            self._build_task_table(result),
            self._build_run_comparison(result, prev_result),
            self._build_cost_latency(result),
        ]
        body = "\n".join(parts)
        return self._build_html_shell(body, result.run_id)

    def _build_html_shell(self, body: str, page_title: str) -> str:
        chart_script_inner = ""
        try:
            with urllib.request.urlopen(_CHART_JS_URL, timeout=5) as resp:
                chart_script_inner = resp.read().decode("utf-8")
        except (urllib.error.URLError, OSError, TimeoutError):
            chart_script_inner = ""

        if chart_script_inner:
            chart_tag = f"<script>\n{chart_script_inner}\n</script>"
        else:
            chart_tag = (
                f'<script src="{html.escape(_CHART_JS_URL, quote=True)}"></script>'
            )

        css = """
:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --border: #2a2d3a;
  --text: #e2e8f0;
  --text-dim: #64748b;
  --green: #22c55e;
  --yellow: #eab308;
  --red: #ef4444;
  --blue: #3b82f6;
  --accent: #6366f1;
}
* { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  margin: 0;
  padding: 24px;
  line-height: 1.5;
}
.wrap { max-width: 1200px; margin: 0 auto; }
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 16px;
}
h1 { margin: 0 0 8px 0; font-size: 1.75rem; }
h2 { margin: 0 0 12px 0; font-size: 1.1rem; color: var(--text-dim); }
table.data { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
table.data th, table.data td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
table.data th { color: var(--text-dim); font-weight: 600; }
table.data tr:nth-child(even) { background: var(--bg); }
table.data tr:nth-child(odd) { background: var(--surface); }
.badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
}
.badge-pass { background: rgba(34, 197, 94, 0.2); color: var(--green); }
.badge-fail { background: rgba(239, 68, 68, 0.2); color: var(--red); }
.score-bar-wrap { max-width: 200px; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
.score-bar-fill { height: 8px; border-radius: 4px; }
.meta { color: var(--text-dim); font-size: 0.9rem; }
.meta strong { color: var(--text); }
button.toggle-fail {
  background: var(--accent);
  color: white;
  border: none;
  padding: 8px 14px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
  margin-bottom: 12px;
}
button.toggle-fail:hover { opacity: 0.9; }
.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
@media (max-width: 700px) { .grid-2 { grid-template-columns: 1fr; } }
.delta-up { color: var(--green); }
.delta-down { color: var(--red); }
.delta-none { color: var(--text-dim); }
canvas { max-height: 360px; }
"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(page_title)} — AgentTrace</title>
  <style>{css}</style>
  {chart_tag}
</head>
<body>
<div class="wrap">
{body}
</div>
</body>
</html>"""

    def _build_header(self, result: EvalResult) -> str:
        thr = 0.75
        all_pass = bool(result.aggregate_scores) and all(
            result.aggregate_scores[m] >= thr for m in result.aggregate_scores
        )
        badge_class = "badge-pass" if all_pass else "badge-fail"
        badge_text = "PASS" if all_pass else "FAIL"
        ts = result.timestamp
        ts_str = ts.strftime("%B %d, %Y %H:%M:%S")
        return f"""
<div class="card">
  <h1>AgentTrace Evaluation Report</h1>
  <p class="meta">
    <strong>Run ID</strong> {html.escape(result.run_id)} &nbsp;·&nbsp;
    <strong>Dataset ID</strong> {html.escape(result.dataset_id)} &nbsp;·&nbsp;
    <strong>Timestamp</strong> {html.escape(ts_str)}
  </p>
  <p class="meta">
    <strong>Duration</strong> {result.duration_s:.1f}s &nbsp;·&nbsp;
    <strong>Total cost</strong> ${result.total_cost_usd:.4f} &nbsp;·&nbsp;
    <strong>Total tokens</strong> {result.total_tokens:,}
  </p>
  <p><span class="badge {badge_class}">{badge_text}</span></p>
</div>"""

    def _build_metric_table(
        self, result: EvalResult, prev_result: EvalResult | None
    ) -> str:
        n_tasks = len(result.task_results)
        metrics = sorted(result.aggregate_scores.keys())
        rows_html = []
        thr = 0.75
        for metric in metrics:
            score = result.aggregate_scores[metric]
            bar_w = f"{max(0.0, min(1.0, score)) * 100:.0f}%"
            bar_bg = _bar_color(score, thr)
            passed_n = sum(
                1
                for r in result.task_results
                if r.passed.get(metric, False)
            )
            pass_rate = f"{passed_n}/{n_tasks}"

            if prev_result is None:
                vs_cell = "—"
            elif metric not in prev_result.aggregate_scores:
                vs_cell = "—"
            else:
                prev_s = prev_result.aggregate_scores[metric]
                delta = score - prev_s
                if abs(delta) < 0.005:
                    vs_cell = '<span class="delta-none">─</span>'
                elif delta > 0:
                    vs_cell = f'<span class="delta-up">▲ +{delta:.2f}</span>'
                else:
                    vs_cell = f'<span class="delta-down">▼ {delta:.2f}</span>'

            rows_html.append(
                f"<tr><td>{html.escape(metric)}</td>"
                f"<td>{score:.3f}</td>"
                f'<td><div class="score-bar-wrap">'
                f'<div class="score-bar-fill" style="width:{bar_w};background-color:{bar_bg}"></div></div></td>'
                f"<td>{pass_rate}</td><td>{vs_cell}</td></tr>"
            )

        thead = (
            "<tr><th>Metric</th><th>Score</th><th>Bar</th>"
            "<th>Pass rate</th><th>vs Last Run</th></tr>"
        )
        return f"""
<div class="card">
  <h2>Metrics</h2>
  <table class="data">
    <thead>{thead}</thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>"""

    def _build_failure_chart(self, result: EvalResult) -> str:
        if not result.failure_dist:
            return ""
        n_tasks = max(len(result.task_results), 1)
        labels = list(result.failure_dist.keys())
        values = [result.failure_dist[k] for k in labels]
        labels_json = json.dumps(labels)
        data_json = json.dumps(values)

        sorted_items = sorted(
            result.failure_dist.items(), key=lambda x: (-x[1], x[0])
        )
        table_rows = []
        for ft, cnt in sorted_items:
            pct = 100.0 * cnt / n_tasks
            table_rows.append(
                f"<tr><td>{html.escape(ft)}</td><td>{cnt}</td>"
                f"<td>{pct:.1f}%</td></tr>"
            )

        script = f"""
<script>
(function() {{
  const ctx = document.getElementById('failureChart');
  if (ctx && typeof Chart !== 'undefined') {{
    new Chart(ctx, {{
      type: 'doughnut',
      data: {{
        labels: {labels_json},
        datasets: [{{
          data: {data_json},
          backgroundColor: [
            '#ef4444','#f97316','#eab308',
            '#22c55e','#3b82f6','#6366f1','#8b5cf6','#ec4899'
          ],
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'right' }} }}
      }}
    }});
  }}
}})();
</script>"""
        return f"""
<div class="card">
  <h2>Failure Taxonomy</h2>
  <canvas id="failureChart"></canvas>
  {script}
  <table class="data" style="margin-top:16px">
    <thead><tr><th>Failure Type</th><th>Count</th><th>% of tasks</th></tr></thead>
    <tbody>{"".join(table_rows)}</tbody>
  </table>
</div>"""

    def _build_task_table(self, result: EvalResult) -> str:
        metric_cols = _metric_names_union(result)
        th_metrics = "".join(f"<th>{html.escape(m)}</th>" for m in metric_cols)

        rows: list[str] = []
        for tr in result.task_results:
            ok = _task_status_pass(tr)
            status = "pass" if ok else "fail"
            status_html = (
                '<span style="color:var(--green)">✓ pass</span>'
                if ok
                else '<span style="color:var(--red)">✗ fail</span>'
            )
            metric_cells = []
            for m in metric_cols:
                if m in tr.metric_scores:
                    sc = tr.metric_scores[m]
                    col = _text_color(sc)
                    metric_cells.append(
                        f'<td style="color:{col}">{sc:.2f}</td>'
                    )
                else:
                    metric_cells.append('<td style="color:var(--text-dim)">—</td>')
            fails = ", ".join(html.escape(ft) for ft in tr.failure_types)
            if not fails:
                fails = "—"
            err_full = tr.error or ""
            if tr.error:
                disp = html.escape(tr.error[:80] + ("…" if len(tr.error) > 80 else ""))
                err_cell = (
                    f'<span title="{html.escape(tr.error, quote=True)}">{disp}</span>'
                )
            else:
                err_cell = "—"
            rows.append(
                f'<tr class="task-row" data-status="{status}">'
                f"<td>{html.escape(tr.task_id)}</td>"
                f"<td>{status_html}</td>"
                f'{"".join(metric_cells)}'
                f"<td>{fails}</td><td>{err_cell}</td></tr>"
            )

        toggle_script = """
<script>
function toggleErrors() {
  const rows = document.querySelectorAll('.task-row');
  rows.forEach(r => {
    if (r.dataset.status === 'pass') {
      r.style.display = r.style.display === 'none' ? '' : 'none';
    }
  });
}
</script>"""
        return f"""
<div class="card">
  <h2>Per-Task Results</h2>
  <button type="button" class="toggle-fail" onclick="toggleErrors()">Show only failed tasks</button>
  {toggle_script}
  <table class="data">
    <thead><tr><th>Task ID</th><th>Status</th>{th_metrics}<th>Failures</th><th>Error</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>"""

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
        cur_scores = [result.aggregate_scores.get(m, 0.0) for m in metrics]
        prev_scores = [prev_result.aggregate_scores.get(m, 0.0) for m in metrics]
        labels_json = json.dumps(metrics)
        cur_json = json.dumps(cur_scores)
        prev_json = json.dumps(prev_scores)
        rid_js = json.dumps(result.run_id)
        pid_js = json.dumps(prev_result.run_id)

        script = f"""
<script>
(function() {{
  const el = document.getElementById('comparisonChart');
  if (el && typeof Chart !== 'undefined') {{
    new Chart(el, {{
      type: 'bar',
      data: {{
        labels: {labels_json},
        datasets: [
          {{
            label: {rid_js},
            data: {cur_json},
            backgroundColor: '#3b82f6',
          }},
          {{
            label: {pid_js},
            data: {prev_json},
            backgroundColor: '#374151',
          }}
        ]
      }},
      options: {{
        scales: {{ y: {{ min: 0, max: 1 }} }},
        plugins: {{ legend: {{ position: 'top' }} }}
      }}
    }});
  }}
}})();
</script>"""
        return f"""
<div class="card">
  <h2>Run Comparison</h2>
  <canvas id="comparisonChart"></canvas>
  {script}
</div>"""

    def _build_cost_latency(self, result: EvalResult) -> str:
        n = len(result.task_results)
        denom = max(n, 1)
        cost_pt = result.total_cost_usd / denom
        tok_pt = result.total_tokens // denom
        err_n = sum(1 for t in result.task_results if t.error)
        pass_n = sum(1 for t in result.task_results if _task_status_pass(t))
        pass_pct = 100.0 * pass_n / denom
        return f"""
<div class="card">
  <h2>Cost &amp; Token Summary</h2>
  <div class="grid-2">
    <div class="meta">
      <p><strong>Total cost</strong> ${result.total_cost_usd:.4f}</p>
      <p><strong>Cost per task</strong> ${cost_pt:.4f}</p>
      <p><strong>Total tokens</strong> {result.total_tokens:,}</p>
      <p><strong>Tokens per task</strong> {tok_pt:,}</p>
    </div>
    <div class="meta">
      <p><strong>Duration</strong> {result.duration_s:.1f}s</p>
      <p><strong>Tasks</strong> {n}</p>
      <p><strong>Errors</strong> {err_n}</p>
      <p><strong>Pass rate</strong> {pass_pct:.0f}%</p>
    </div>
  </div>
</div>"""
