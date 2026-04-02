# Contributing

Thanks for your interest in AgentRace.

## Development setup

```bash
git clone <repository-url>
cd agentrace
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,all]"
```

The `dev` extra is only pytest, pytest-asyncio, and mypy. CI and full local checks use `.[dev,all]` so optional judge (LiteLLM) and Postgres (asyncpg) dependencies match what users install via `agentrace[judge]` / `agentrace[postgres]`. Integration tests under `v1_testing/` add their own deps via `v1_testing/requirements.txt` — nothing from that tree is declared on the PyPI package.

## Running checks

```bash
pytest
mypy agentrace cli
```

Real-LLM integration tests are under `v1_testing/`; see [v1_testing/README.md](v1_testing/README.md) for API keys and how to run them.

## Pull requests

- Keep changes focused on one concern when possible.
- Ensure `pytest` and `mypy agentrace cli` pass locally.
- Update [CHANGELOG.md](CHANGELOG.md) for user-visible changes (use the Unreleased section or the next version heading as appropriate).

## Code style

Match existing patterns in the tree (imports, typing, async style). The library is async-first; prefer `async`/`await` for I/O.
