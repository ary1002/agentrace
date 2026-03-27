# AgentRace

**AgentRace** (working title aligned with **AgentTrace**) is a framework-agnostic, production-oriented library for **tracing** and **evaluating** LLM-based agent pipelines. This repository currently contains the **package scaffold** only: modules are stubs with docstrings and `TODO`s; there is no runtime logic yet.

## Layout

- **`agentrace/`** — core package: capture (instrumentation), normalizer (OTel → DAG), metrics (deterministic + LLM judge via **litellm**), classifier, dataset/benchmarks, async **runner**, reporters, and async storage (**aiosqlite** / **asyncpg**).
- **`cli/`** — **Typer** + **Rich** CLI (`agentrace` console script).
- **`tests/`** — pytest placeholders.
- **`templates/`** — Jinja2 HTML report template.

Configuration will be driven by **`eval.yaml`** using **pydantic-settings** in `agentrace.config` (core domain models remain **dataclasses**).

## Requirements

- Python **3.10+**
- Async-first: **asyncio** + **anyio**

## Development

```bash
pip install -e ".[dev]"
pytest
mypy agentrace cli
```

(Specific type-check scope may be adjusted once implementations land.)
