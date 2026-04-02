# AgentRace

**AgentRace** is a framework-agnostic library for **tracing** and **evaluating** LLM agent pipelines. It captures runs via OpenTelemetry-style spans, normalizes traces, runs deterministic and LLM-as-judge metrics, and persists results (SQLite by default, PostgreSQL optional).

**Public API** (stable for 1.x): `agentrace.trace`, `agentrace.evaluate` (async), `load_benchmark_suite`, core models (`AgentTrace`, `Span`, `EvalTask`, `EvalResult`, …), metric base types, and optional adapters (`AgentTraceCallbackHandler`, LangGraph/Crew helpers). Everything else under `agentrace.*` should be treated as internal unless documented here.

## Requirements

- Python **3.10+**
- **Async-first**: use `await agentrace.evaluate(...)` inside `async` code. Calling `evaluate` without `await` returns a coroutine object and will not run the evaluation.

## Installation

```bash
pip install agentrace
```

**Optional extras**

| Extra | Purpose |
|-------|---------|
| `agentrace[judge]` | LLM-as-judge metrics (installs **litellm**). Required if your metric set uses judge-backed scores. |
| `agentrace[postgres]` | PostgreSQL storage backend (installs **asyncpg**). |
| `agentrace[all]` | Both of the above. |

Development install from a clone (library + pytest/mypy only):

```bash
pip install -e ".[dev]"
```

For local work that touches judge metrics or PostgreSQL storage, also install optional stacks (same as CI):

```bash
pip install -e ".[dev,all]"
```

The published package does not depend on integration-test stacks; those live only under `v1_testing/requirements.txt` in the repo.

## Environment variables

Judge metrics and traced LLM calls go through **LiteLLM**, which follows each provider’s usual env vars:

| Variable | When it matters |
|----------|-----------------|
| `ANTHROPIC_API_KEY` | Anthropic models (agents or judge). |
| `OPENAI_API_KEY` | OpenAI models (agents or judge). |
| Other provider keys | As required by LiteLLM for your chosen `judge_model` / agent stack. |

If a key is missing, API calls fail with the provider’s error (not a silent success). Tracing and deterministic metrics do not require API keys.

## Quickstart (under five minutes)

`trace` is an **async** context manager (session id and task label):

```python
import asyncio
from agentrace import trace

async def main():
    async with trace("demo-session", "my-task") as ctx:
        # Your agent / LLM calls here (SDK patches or adapters attach to this session).
        pass
    # After the block, ctx.agent_trace holds the normalized trace (if exit was clean).

asyncio.run(main())
```

For full evaluation you typically define an async agent callable, a `Dataset`, and metric names, then `await evaluate(agent, dataset, metrics, ...)`. See `eval.yaml` and the CLI (`agentrace --help`) for project-style runs.

## CLI

After install, the `agentrace` command is available (Typer + Rich). Subcommands include `run`, `runs`, `benchmark`, and `diff`.

## What is not in v1.0

Non-exhaustive list; see issues and future milestones for detail:

- First-class adapters for every agent framework (community coverage will grow).
- Synchronous wrappers for `evaluate` (library remains async-first).
- Guaranteed compatibility with every pinned third-party stack; use extras and version ranges, and report conflicts.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Development

```bash
pip install -e ".[dev,all]"
pytest
mypy agentrace cli
```

Integration tests with real APIs live under `v1_testing/` (repo-only; not on PyPI); see [v1_testing/README.md](v1_testing/README.md).
