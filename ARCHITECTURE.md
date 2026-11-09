# Architecture

Two packages in one distribution (`pyproject.toml` lists both under
`[tool.uv.build-backend] module-name`). See [ADR 0001](docs/adr/0001-perception-lab-as-sibling-package.md).

## Dependency direction

```
perception_lab  ──imports──▶  depth_volumetry
```

`depth_volumetry` never imports `perception_lab`. The lab reuses exactly three things:

| From `depth_volumetry` | Used by |
|---|---|
| `segmentation` (`decode_detections`, `nms`, `preprocess`, `xywh_to_xyxy`) | `perception_lab/vision/detector.py` |
| `realsense.load_pyrealsense2` (lazy import) | `perception_lab/sources/realsense.py` |
| `io` (`read_color`, `write_image`) | `perception_lab/eval/bench.py`, `perception_lab/store/media.py` |

## `depth_volumetry`: box measurement

Aligned depth + mask + calibration in, length/width/height out. Geometry, model contract and
calibration procedure: [`docs/method.md`](docs/method.md).

| Module | Role | I/O allowed |
|---|---|---|
| `calibration.py` | `Calibration` dataclass, TOML loading, validation (`CalibrationError`) | reads TOML |
| `measurement.py` | footprint, surface distance, `measure_box` (`MeasurementError`) | no (pure) |
| `segmentation.py` | decode, NMS, mask functions (pure) plus a thin ONNX Runtime wrapper | wrapper only |
| `io.py` | read/write images, depth maps, masks | yes |
| `overlay.py` | annotated output image | no |
| `realsense.py` | optional capture; `pyrealsense2` imported lazily | yes |
| `cli.py` | `measure` and `capture` commands | yes |

## `perception_lab`: pet monitoring

Design, rules, evaluation protocol and privacy defaults: [`docs/pet-monitoring.md`](docs/pet-monitoring.md).
Data flow diagram: [`docs/diagrams/pet-monitoring-architecture.svg`](docs/diagrams/pet-monitoring-architecture.svg).

Per frame (synchronous): `sources` → `vision` (detector, tracker) → `spatial` → track history
→ `events` rules. Per event: `apps/pets/pipeline.py` builds facts and evidence, writes to
`store`, notifies through `notify`, and queues the optional `vlm` worker. `web` serves the
stored events on loopback.

| Package | Role |
|---|---|
| `core/` | shared dataclasses (`FrameBundle`, `Detection`, `SpatialObs`, `TrackObservation`) in `types.py`, TOML config, run record |
| `sources/` | frame sources: RealSense live, `.bag`, video, synthetic scene |
| `vision/` | YOLOv8-layout ONNX detector, ByteTrack-style tracker, drawing |
| `spatial/` | floor plane, zones, robust per-track depth facts |
| `events/` | features, deterministic rules, event model, evidence ring buffer |
| `vlm/` | strict JSON schema, versioned prompt, providers, async worker |
| `store/` | SQLite history, evidence JPEGs, retention |
| `notify/`, `web/` | notifications and the standard-library dashboard |
| `eval/` | labels, metrics, repeatable runs, on-machine benchmarks |
| `apps/pets/` | wiring of the above into `PetPipeline` |

## Invariants

- Alerts come from deterministic rules and never wait for the VLM ([ADR 0003](docs/adr/0003-rules-decide-vlm-describes.md)).
- Measurement and decode functions are pure; I/O lives in `io.py`, `cli.py`, the model wrapper
  and the lab's `sources/`, `store/`, `web/`, `notify/`.
- The lab is local only; the runtime uses the standard library ([ADR 0002](docs/adr/0002-stdlib-only-lab-runtime.md),
  [ADR 0004](docs/adr/0004-local-only-loopback-endpoints.md)).
- Tests mirror modules: one test module per source module, lab tests in `tests/lab/`.
