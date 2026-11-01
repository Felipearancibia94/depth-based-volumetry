# Pet activity monitoring (`perception_lab`)

Local-first monitoring of cats and dogs with an Intel RealSense camera: detection, tracking,
depth-based spatial facts, deterministic event rules, optional descriptions from a small
local vision-language model (VLM), local notifications and a local dashboard. Everything
runs on one machine; nothing is sent to a cloud service.

**Status.** The pipeline is implemented and tested against a synthetic scene, a fake
RealSense SDK and a fake VLM runtime. It has **not** been run with a real camera, a real
detection model or a real VLM in this repository. Every accuracy, latency or memory figure
must come from your own runs (`lab eval`, `lab bench`).

## Architecture

```
FrameSource ──▶ Detector ──▶ Tracker ──▶ SpatialEstimator ──▶ TrackHistory
 (RealSense,    (YOLOv8      (ByteTrack   (depth, 3D point,     (per track)
  .bag, video,   ONNX)        style)       elevation, zones)          │
  synthetic)                                                          ▼
                       Dashboard ◀── EventStore ◀── PetPipeline ◀── EventEngine (rules)
                       Notifier  ◀───────────────────┤
                                                     └──▶ VLMWorker (event-gated, async)
```

| Module | Responsibility |
|---|---|
| `sources/` | `FrameBundle` (colour, aligned depth, intrinsics, timestamp) from RealSense live, `.bag` replay, video or the synthetic scene; `throttle` drops frames instead of queuing |
| `vision/` | YOLOv8-layout ONNX detector reusing `depth_volumetry.segmentation`; ByteTrack-style tracker; live-view drawing |
| `spatial/` | floor plane (RANSAC), zones (image polygons, optional elevation range), robust depth per track |
| `events/` | track histories, temporal features, rules, start/end hysteresis and cooldowns, evidence ring buffer, event model |
| `vlm/` | strict JSON schema, versioned prompt, Ollama / OpenAI-compatible providers, background worker |
| `store/` | SQLite event history with full-text search, evidence JPEGs, retention |
| `notify/`, `web/` | `notify-send`, loopback webhook, log lines; standard-library dashboard |
| `eval/` | labels, metrics, repeatable evaluation runs, on-machine benchmarks |
| `apps/pets/` | wiring of the above into `PetPipeline` |

Design rules:

- **Rules decide, the VLM describes.** Events and notifications come from deterministic
  rules over measured facts. The VLM is asked only after an event starts, receives a few
  evidence crops plus the measured facts, and its answer is merged later.
- **Facts, interpretation and uncertainty are separate fields.** `observed_facts` holds
  measurements, `hypothesis` the rule's prior (e.g. a dwell at the food bowl suggests
  `eating`), `interpretation` the VLM answer, and `uncertainty` known reasons to doubt
  either. `unknown` is a valid VLM answer. `confidence` stays empty until it can be
  calibrated against reviewed events.
- **No silent substitution.** Missing depth gives `None` plus a quality flag; speeds and
  distances say whether they are in metres or body lengths.

## Rules and expected difficulty

| Behaviour | Rule (`rule` → `hypothesis`) | Main signal | Difficulty |
|---|---|---|---|
| Entering a restricted area | `zone_entry` → `entered_restricted_area` | foot point in a `restricted` zone | easy |
| On furniture or counter | `elevated_surface` → `on_elevated_surface` | elevation above the floor, or an `elevated` zone | easy–medium (needs floor calibration) |
| Running | `speed_burst` → `running` | mean speed over 1 s | easy–medium |
| Resting / sleeping | `stationary` → `resting_or_sleeping` | stillness outside dwell zones | medium; sleep vs rest is not visually decidable |
| Eating / drinking | `zone_dwell` → `eating` / `drinking` | dwell in a `food` / `water` zone; VLM to separate eating from sniffing | medium–hard |
| Animals interacting | `proximity` → `interacting` | two animals within 0.5 m | medium; play vs conflict needs the VLM and is unreliable |
| Unusual inactivity | `long_inactivity` → `unusual_inactivity` | stillness for an hour | needs per-animal baselines; not validated |
| Playing, grooming | VLM only (`playing`, `grooming`) | — | hard; expect `unknown` |

Repeated movement patterns and per-animal identity are not implemented.

## What depth adds, and how far to trust it

| Measurement | Reliability | Conditions |
|---|---|---|
| Distance to the animal | usable at room distances, with `depth_valid_ratio` | stereo depth error grows with distance; dark or fluffy fur, sunlight and motion leave holes |
| 3D position | usable | median over the shrunk box or mask; biased towards background on loose boxes |
| Zones | reliable after calibration | per camera; redo `lab calibrate` if the camera moves |
| Elevation / jumping | controlled conditions only | needs the floor plane; treat as floor / furniture / counter, not centimetres |
| Proximity between animals | reliable at coarse thresholds | 2D box overlap is not proximity; depth separates them |
| Object moved or missing | reliable for rigid static objects | not part of this app yet (see `depth_volumetry` for box measurement) |
| Room mapping | single-view floor plane only | multi-view mapping is out of scope |
| Animal size or volume | not reliable | not used |

Depth is never used as evidence of identity or behaviour on its own.

## First steps on the target machine

Run these on the Ubuntu machine with the camera and GPU, and keep the outputs; they replace
every assumption in this document.

```bash
nvidia-smi
```

```bash
uv sync --extra realsense
```

```bash
uv run lab discover
```

`lab discover` lists the camera model, firmware, USB type and supported profiles. If
`pyrealsense2` has no wheel for your Python version, install librealsense from Intel's
packages or use Python 3.11.

Record a few minutes and check the streams:

```bash
uv run lab record --seconds 120 --out data/recordings/first.bag
```

```bash
uv run lab replay --source bag --path data/recordings/first.bag --preview-dir outputs/preview
```

### VLM runtime

Install [Ollama](https://ollama.com) (or llama.cpp `llama-server`) locally and pull the
smallest candidate first. Check the exact model tag in the Ollama library; `qwen3-vl:2b` is
the configured default but was not verified from this repository.

```bash
ollama pull qwen3-vl:2b
```

```bash
uv run lab bench vlm --images 1,3,6 --repeats 3 --out outputs/bench/vlm-2b.json
```

```bash
uv run lab bench vlm --images 1,3 --montage --out outputs/bench/vlm-2b-montage.json
```

The benchmark reports, per image count: latency percentiles, how many answers matched the
schema, the answers, errors, peak GPU memory (whole GPU, via `nvidia-smi`) and Ollama's own
`size_vram`. Repeat with the 4B quantised model (`--model <tag>`). Decide from these
numbers, not from model file sizes:

- if multi-image requests fail or are ignored, set `vlm.montage = true`;
- keep the model whose peak memory leaves room for the detector (6 GB card) and whose
  latency is acceptable; alerts never wait for the VLM;
- check the model licence on its official model card before relying on it.

### Detector

Export a COCO-trained detector with the YOLOv8 output layout to ONNX (detection or
segmentation; segmentation masks improve depth sampling) and measure it:

```bash
uv run lab bench detector --source bag --path data/recordings/first.bag --frames 200
```

Ultralytics weights are AGPL-3.0; check the licence of whatever weights you use. Set
`detector.kind = "yolov8_onnx"` and `detector.model_path` in `lab.toml`.

The base dependency `onnxruntime` is the CPU build, and `providers_in_use` in the benchmark
output shows what actually ran. GPU inference needs `onnxruntime-gpu` in its place, matched
to the installed CUDA and cuDNN versions; that is a dependency change to make and lock on
the target machine. On a 6 GB card, compare the detector on CPU against GPU while the VLM is
loaded before choosing.

### Calibration and zones

1. With the floor visible and mostly clear, fit the floor plane:

   ```bash
   uv run lab calibrate --source realsense --out scene.toml
   ```

   It prints the camera height above the floor and its tilt; check them with a tape.
2. Start monitoring with the dashboard, click points on the live view and paste the
   generated `[[zones]]` snippets into `scene.toml`. Tags: `food`, `water`, `bed`
   (dwell rules), `restricted` (entry rule), `elevated` (raised surface without depth).
3. Set `spatial.scene_path = "scene.toml"` in `lab.toml`.

### Monitoring

```bash
uv run lab init-config --out lab.toml
```

```bash
uv run lab run --config lab.toml --source realsense --vlm --serve
```

The dashboard is at http://127.0.0.1:8000/. Use **Correct** / **Wrong** on events: reviews
are stored and become labels for calibration later.

## Evaluation protocol

### Dataset

- 15–25 recording sessions (`.bag`) of 5–30 minutes: each animal alone, animals together,
  feeding times, rest periods, daylight, artificial light, night/IR.
- Several hours of ordinary activity without labelled events, to measure false events per
  day.
- Labels as JSON (see `examples/labels.example.json`): activity, start and end in seconds
  from the first frame, animal class. Commit labels if useful, never the recordings.

### Splits without leakage

Frames of one session are strongly correlated, so split **by session**, never by frame or
clip. Keep at least one whole day and one night session only in the test split. Roughly
60/20/20 sessions for train/validation/test; tune thresholds and prompts on validation
only; evaluate on test only for milestone reports, with the configuration frozen.

### Procedure

```bash
uv run lab eval --config lab.toml --source bag --path data/recordings/val-01.bag --labels data/labels/val-01.json
```

Each run writes `outputs/runs/<run-id>/` with `run.json` (code revision, packages, config,
model and labels hashes), `metrics.json`, `events.jsonl` and `report.md`:

| Metric | Definition |
|---|---|
| Event precision / recall / F1 | per activity; a prediction matches a segment with the same activity and compatible animal class whose interval, widened by the tolerance, overlaps it (one-to-one) |
| False events per day | unmatched rule events per day of recording |
| VLM accuracy when answered | correct answers among non-`unknown` answers, against the overlapping labelled segment |
| Unknown rate | `unknown` answers among all answers |
| Detection precision / recall, identity switches | when frames carry ground-truth boxes (synthetic source) |
| Latency, throughput, memory | `stage_ms` in the run, `lab bench` |

Compare runs that differ in one thing only: detector, input size, VLM model, prompt
version, number of frames, crops vs montage.

The synthetic source has built-in labels (`lab eval` without `--labels`). It checks the
plumbing and robustness to controlled detector noise (`detector.gt_miss_rate`,
`detector.gt_jitter_px`); it says nothing about real-world accuracy. Its cat rests for about
a minute, so with the default `rules.stationary_s = 300` that rest is (correctly) reported
as missed; set `stationary_s = 30` to exercise the rule.

## Privacy and retention

- No continuous recording: frames live in a short in-memory ring buffer; only evidence for
  events is written to disk.
- Defaults: evidence images 14 days, event rows 365 days (`lab purge`, also `--dry-run`).
- The dashboard binds to 127.0.0.1 and refuses other addresses unless
  `web.allow_remote = true`; it has no authentication.
- VLM and webhook endpoints must be loopback unless explicitly allowed.
- Recordings, databases, media, models and calibration files are git-ignored.

## Known limitations

- Not validated with real hardware, a real detector or a real VLM.
- One camera; multi-camera fusion is not implemented.
- Track ids are not animal identities; identity can switch when animals overlap.
- Under heavy detector noise (synthetic: 40 % missed detections, 8 px jitter) the
  `interacting` rule is missed.
- `unusual_inactivity` uses a fixed threshold, not learned per-animal baselines.
- The zone editor only produces TOML snippets; zones are edited in `scene.toml`.
