# depth-based-volumetry

Python project with two packages built on aligned colour + depth images (pinhole model):

- `depth_volumetry`: library and CLI that estimate a box's length, width and height.
- `perception_lab`: local home perception lab (`lab` CLI); first app is pet monitoring.

Map and dependency rules: `ARCHITECTURE.md`. Current status and next steps: `ROADMAP.md`.
Decisions: `docs/adr/`. Geometry and model contract: `docs/method.md`. Lab design and
evaluation: `docs/pet-monitoring.md`. `docs/plans/` and `docs/reports/` are dated history,
not the current state. Package-specific rules load from `.claude/rules/` when you touch
matching files.

## Commands (uv only; do not use bare pip)

```
uv sync                       # add --extra realsense when needed
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run lab run --frames 300   # synthetic end-to-end run, no hardware
uv run lab eval               # synthetic evaluation report
```

Add dependencies with `uv add` / `uv add --dev` so `uv.lock` stays in sync. Never edit
`uv.lock` by hand.

## Conventions

- Units go in the name: `_cm`, `_mm`, `_px`, `_m`, `_s`.
- Points are `(x, y)`; arrays are indexed `[row, col]`; shapes are `(height, width)`.
- Do not clamp or silently substitute values: raise (`MeasurementError`, `CalibrationError`)
  or return `None` with a quality flag.
- Keep dependencies minimal. RealSense stays an optional extra and a lazy import.
- Python 3.11+ (the pinned `pyrealsense2` has no wheel for 3.12+). Code, comments and docs in
  English. Commits follow Conventional Commits.

## Workflow

1. Explore first: read the module, its test and the matching rule file before editing.
2. For anything beyond a small fix, state the plan before changing files.
3. Keep changes small and in the existing style; no unrelated refactors.
4. Before saying a change is done, run `/verify` (pytest, ruff check, ruff format --check)
   and report the actual results, including failures.
5. When behaviour, a CLI flag or status changes, update README, the relevant doc and
   `ROADMAP.md` in the same change (`/cli-change` for flags, `/excalidraw` to rebuild the
   generated diagrams in `docs/diagrams/`).

## Hard constraints

- Git commits, pushes, branch changes and remote changes need explicit user authorization.
- Never add credentials, company or client names, or non-loopback endpoints to the
  repository. Loopback defaults (`127.0.0.1`, `localhost`) in examples are allowed.
- Never commit model weights (`*.onnx`, `*.pt`, `*.engine`, `*.gguf`, anything under
  `models/`), captures, recordings, databases or real calibration files. The only exception is
  `docs/gallery/`: one image per pet picked by the owner, with no people in it.
- Tests use synthetic data only and need no network, camera or model.
- Do not claim real-model inference, RealSense capture or real-world accuracy unless it was
  actually run on that hardware or model.
