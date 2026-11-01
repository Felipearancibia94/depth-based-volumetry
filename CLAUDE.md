# depth-based-volumetry

Python library and CLI that estimate a box's length, width and height from an aligned
colour + depth image pair (pinhole camera model). User docs: `README.md`. Geometry, model
contract and calibration procedure: `docs/method.md`.

## Layout

- `src/depth_volumetry/calibration.py`: `Calibration` dataclass, TOML loading and validation.
- `src/depth_volumetry/measurement.py`: footprint, surface distance, `measure_box`, `MeasurementError`.
- `src/depth_volumetry/segmentation.py`: pure decode/NMS/mask functions plus a thin ONNX Runtime wrapper.
- `src/depth_volumetry/io.py`: reading and writing images, depth maps, masks.
- `src/depth_volumetry/overlay.py`: annotated output image.
- `src/depth_volumetry/realsense.py`: optional capture; `pyrealsense2` is imported lazily.
- `src/depth_volumetry/cli.py`: `measure` and `capture` commands.
- `src/perception_lab/`: home perception lab (`lab` CLI), first app pet monitoring; design in
  `docs/pet-monitoring.md`. `core/` types and TOML config, `sources/`, `vision/`,
  `spatial/`, `events/`, `vlm/`, `store/`, `notify/`, `web/`, `eval/`, `apps/pets/` (wiring).
  It reuses `depth_volumetry` (segmentation decoding, RealSense loader, image I/O).
- `tests/`: one test module per source module, synthetic data only; lab tests in `tests/lab/`
  (synthetic scene, `tests/lab/fake_rs.py` for the RealSense SDK, fake HTTP for the VLM).

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
- `measurement.py` and the decode functions in `segmentation.py` are pure: no file or
  model access. Keep I/O in `io.py`, `cli.py` and the model wrapper.
- Failures to measure raise `MeasurementError`; invalid configuration raises
  `CalibrationError`. Do not clamp or silently substitute values.
- Keep dependencies minimal. RealSense stays an optional extra and a lazy import.
- Python 3.11+ (`tomllib` from the standard library). Code, comments and docs in English.
- Commit messages follow Conventional Commits.
- Lab: alerts come from deterministic rules; the VLM only describes. Keep observed facts,
  VLM interpretation and uncertainty in separate fields; `unknown` is a valid answer.
- Lab: local only. VLM, webhook and dashboard endpoints stay on loopback unless the config
  explicitly allows otherwise; no new runtime dependencies without need (stdlib HTTP, SQLite).

## Testing

- Tests use synthetic arrays and tensors only. They must need no network, camera or model.
- Never commit model binaries (`*.onnx`), captures or real calibration files.
- When a CLI flag changes, update the README and the matching test in `tests/test_cli.py`
  (or `tests/lab/test_cli.py` for `lab`).

## Constraints

- Before saying a change is done, run `uv run pytest` and `uv run ruff check .` and report
  the actual results.
- Git commits, pushes, branch changes and remote changes need explicit user authorization.
- Never add credentials, company or client names, or endpoints to the repository.
- Do not claim real-model inference, RealSense capture or real-world accuracy work unless it
  was actually run on that hardware or model.
