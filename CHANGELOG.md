# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-03

### Added

- Initial stable public release: tracing (`agentrace.trace`), async evaluation (`evaluate`), datasets, deterministic and LLM-judge metrics, SQLite storage by default, optional PostgreSQL, CLI, and HTML/JSON reporters.
- Packaging extras: `agentrace[judge]`, `agentrace[postgres]`, `agentrace[all]`.
- PEP 561 support via `py.typed`.
- Project documentation: README, CONTRIBUTING, SECURITY, and this changelog.
