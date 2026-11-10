---
name: verify
description: Run the project checks (pytest, ruff check, ruff format --check) and report the real results. Use before saying a change is done, after edits to src/ or tests/, or when asked to verify, test or lint.
allowed-tools: Bash(uv run pytest*), Bash(uv run ruff *), Bash(uv sync*)
---

# Verify

Run, in this order, from the repository root:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Report each command with its actual outcome (counts, failing tests, lint messages). Do not
summarise a failure as a pass and do not skip a command.

If a check fails:
- fix the cause when it is in code you changed; never relax a test or a lint rule to pass;
- if the failure is unrelated to your change, say so and show the evidence (for example by
  running the same test on a clean checkout with `git stash` only if the user agrees).

If `uv` itself fails (for example Windows error 32, a locked cache directory) rather than the
checks, retry once with a different cache (the project uses Python 3.11; see `.python-version`): `UV_CACHE_DIR=<scratchpad>/uv-cache uv run pytest`.
If it still fails, report the uv error as-is and what was therefore not verified.

For a CLI flag change also confirm the README table and the CLI test were updated
(`/cli-change`).
