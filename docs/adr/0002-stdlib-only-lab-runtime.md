# 0002. Standard-library-only lab runtime

Status: Accepted (2026-11-01)

## Context

The plan suggested pydantic, PyYAML, httpx, FastAPI and motmetrics. The project rule is to
keep dependencies minimal, and the lab must run on one local machine.

## Decision

Use the standard library for the lab runtime: `tomllib` and dataclasses for configuration,
`sqlite3` (FTS5 with a LIKE fallback) for storage, `urllib` for model endpoints and
`http.server` for the dashboard and webhook. No new runtime dependencies beyond numpy,
onnxruntime and OpenCV.

## Consequences

- Fewer packages to lock and audit; configuration is TOML, not YAML.
- Validation is hand-written in `core/config.py`.
- The dashboard is simple and has no authentication; it is bound to loopback (ADR 0004).
- A new runtime dependency needs a stated reason and `uv add`.
