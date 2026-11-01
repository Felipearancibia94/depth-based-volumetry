# Home AI Perception Lab — Audit, Architecture and Implementation Plan

## Context
Goal: evolve `depth-based-volumetry` into a local-first **Home AI Perception Lab** (Intel RealSense + computer vision + small local LLM/VLM) whose first application is **Home Pet Intelligence** (multi-animal cat/dog detection, tracking, depth-based spatial analysis, event-gated VLM interpretation, local notifications), with components reusable by later home applications.

This is the planning deliverable. **No repository file is modified, nothing is installed, committed or pushed until you approve.**

Evidence rules used in this plan:
- Repository facts: verified by reading every tracked file.
- Model/runtime facts: from third-party web summaries only. `huggingface.co` and `ollama.com` are not reachable from this cloud sandbox, and the sandbox has no GPU or camera. Every such fact is marked **[unverified]**. Every performance number is **[to measure]**: none is a benchmark.
- Camera model and connected hardware are **not assumed**.

The plan follows the 13 requested deliverables in order.

---

## D1. Repository capability inventory (verified in source)

Single commit `6907823 Scaffold project with uv, packaging and tooling`. Tracked files: `.gitignore`, `.python-version` (3.12), `LICENSE` (MIT), `README.md`, `pyproject.toml`, `uv.lock`, `src/depth_volumetry/{__init__.py, __main__.py, cli.py}`.

| Capability | Status in source | Evidence |
|---|---|---|
| Volumetric measurement (box L×W×H) | **Not implemented**: only the project description | `pyproject.toml` `description`, `src/depth_volumetry/__init__.py` docstring |
| RealSense integration | **Not implemented**: `pyrealsense2>=2.50` declared as optional extra `realsense`, never imported | `pyproject.toml` `[project.optional-dependencies]` |
| Camera acquisition, RGB/depth alignment, calibration, deprojection, point clouds | **None** | no code |
| Models / inference | **None**: `onnxruntime>=1.17` (CPU build) and `opencv-python-headless>=4.8` declared, unused; no weights (`*.onnx` gitignored) | `pyproject.toml`, `.gitignore` |
| CLI | Entry point `depth-volumetry` → `main()` raises "not implemented yet" | `src/depth_volumetry/cli.py` |
| Tests / CI | None (pytest configured with `testpaths=["tests"]`, dir absent; no CI) | `pyproject.toml` |
| Tooling | uv + `uv_build` backend, Python >=3.11 (pinned 3.12), ruff (E,F,I,UP,B,SIM; line 100), pytest >=8 | `pyproject.toml`, `.python-version` |
| Data hygiene | `.gitignore` excludes `/data/`, `/outputs/`, `*.onnx`, `calibration.toml`, `.env*`; mentions a future `examples/` calibration example | `.gitignore` |
| Claude Code config | None (no `CLAUDE.md`, no `.claude/`) | — |

Conclusion: the repository is a clean, well-configured **scaffold with no perception functionality**. The architecture is greenfield; the reuse question concerns conventions, not algorithms.

## D2. Existing functionality worth preserving
- `src/` layout + uv + `uv_build` + `uv.lock` workflow.
- ruff and pytest configuration (unchanged rules).
- Optional-extra pattern (`realsense`), extended to `gpu`, `web`, `eval`.
- `.gitignore` privacy defaults (data, outputs, models, calibration, secrets): keep and extend (`*.bag`, `media/`, `*.db`).
- MIT license.
- The original goal (box volumetry from aligned RGB-D) is kept as **planned application A6** (D4), built on the same spatial module, not deleted from scope.
- Package naming: propose renaming the import package `depth_volumetry` → `perception_lab` (CLI `lab`); keep the repository/distribution name unless you decide otherwise (open question Q1).

## D3. Technical gaps and proposed improvements

Gaps for real-time local perception (everything below is missing today):

| Gap | Proposed improvement |
|---|---|
| No frame source abstraction | `FrameSource` protocol with RealSense-live, `.bag` replay, video-file/webcam and synthetic implementations |
| No device discovery / compatibility check | `lab discover`: model, serial, firmware, USB type, supported profiles, depth scale, intrinsics/extrinsics; warns on USB 2 and unsupported profiles |
| No alignment / sync / timestamps | Use RealSense framesets (colour+depth from one device are delivered together), `rs.align(color)`, keep device timestamp + timestamp domain + host monotonic time in every `FrameBundle` |
| No rate control | Separate camera fps from processing fps; latest-frame-wins bounded queue (drop, never lag); per-stage rate config |
| No recording/replay | `lab record` (bag + metadata JSON), `lab replay` deterministic at recorded or max speed |
| No calibration | `calibration.toml` per camera: intrinsics snapshot, floor plane (RANSAC), camera height/tilt, zone polygons |
| No detection/tracking | ONNX detector + ByteTrack-style tracker (D6) |
| No event/memory system | Rule engine + SQLite (WAL) store + FTS5 text search + evidence media folder + retention |
| No VLM/LLM layer | `VLMProvider` over an OpenAI-compatible HTTP API (Ollama / llama.cpp `llama-server`), schema-validated JSON |
| No status/monitoring | Health object per module (fps, queue depth, last frame age, GPU memory via NVML, model loaded/unloaded) exposed to dashboard |
| No configuration / experiment management | Single YAML config validated with pydantic; every run writes a run record (git SHA, config, model hashes, runtime versions) |
| No tests / CI | Hardware-free unit tests + synthetic-source integration test; optional GitHub Actions running ruff + pytest (no GPU) |
| CPU-only onnxruntime | Move to `gpu` extra with `onnxruntime-gpu`; CUDA/cuDNN compatibility checked on host |
| pyrealsense2 + Python 3.12 | **[unverified]** wheel availability on the host; fallback: build librealsense or pin Python 3.11 |

## D4. Ranked home application opportunities

Scores 1–5 (5 = best: most useful, most feasible, least effort). Rank = usefulness × feasibility ÷ effort, judged qualitatively.

| Rank | Application | Problem solved | RealSense contribution | CV models | LLM/VLM value | Complexity | Main limits | Validation experiment | U / F / E |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **A1 Restricted-area & furniture/counter detection** | Know when a pet enters the kitchen counter, a closed room, etc. | 3D zones independent of perspective; elevation above floor distinguishes "on counter" from "in front of counter" | detector + tracker | Low: optional confirmation image only | Low | calibration per camera, occlusion, depth holes at edges | Scripted session: 30+ known zone entries (by you with a toy/pet), measure entry precision/recall and latency | 5/5/4 |
| 2 | **A2 Pet activity monitoring (MVP)** | Daily log of what each pet did, alerts on notable behaviour | distance, speed in m/s, elevation, proximity | detector + tracker (+ optional pose later) | **Medium–High**: activity description for ambiguous events | Medium | activity ambiguity, 6 GB VRAM, ID switches | Full suite D10 | 5/3/3 |
| 3 | **A3 Food & water station activity** | Did the cat eat/drink today, how often, how long | proximity to bowl in 3D, dwell; bowl-moved detection | detector + tracker | Medium: eating vs. sniffing vs. passing | Low–Medium | bowl occlusion by the animal, "visit" ≠ "eating" | 3 days of bowl-visit labels; compare rules-only vs. rules+VLM | 4/4/4 |
| 4 | **A4 Animal interaction monitoring** | Play vs. conflict between pets, resource guarding | inter-animal 3D distance (2D overlap is misleading) | multi-object tracking | Medium: interaction type | Medium–High | ID swaps during contact, rare events | Annotated contact segments; ID-switch count during contact | 3/3/2 |
| 5 | **A5 Room occupancy & spatial heatmaps** | Where pets/people spend time; feeds automation | metric floor-plane heatmap | person/animal detector | Low (deterministic) | Low–Medium | privacy with people, multi-room needs multiple cameras | Compare heatmap with manually logged positions | 3/4/4 |
| 6 | **A6 Object change & volumetry (original repo goal)** | Box dimensions; "object moved/missing" (bowl, litter box) | **Core**: plane fitting + depth differencing | none / plane fitting / optional segmentation | None | Medium | depth noise, reflective/dark surfaces, calibration | Measure known boxes at known distances; report error distribution | 3/4/3 |
| 7 | **A7 Local smart-home automation hooks** | Trigger lights/notifications from events | via events only | reuses events | None | Low | safety of physical actions | Mock MQTT/webhook consumer with event replay | 2/5/5 |

Build order: shared core → A1+A2 (same pipeline, A1 is A2's deterministic subset) → A3 → A6 → A5/A4 → A7. Deterministic CV is the right tool for A1, A5, A6, A7; VLM is justified only for A2, A3, A4 and only on ambiguous, event-gated cases.

## D5. Home Pet Intelligence architecture

### Dataflow
```
FrameSource ──FrameBundle(ts, color, depth|None, intrinsics, depth_scale, camera_id)──▶
  Detector (cat/dog, every Nth frame) ─▶ Tracker (every frame or every detection)
  ─▶ SpatialEstimator (robust depth, 3D point, elevation, zone ids, quality flags)
  ─▶ TrackStateStore (ring buffer per track: kinematics, zone history, thumbnails)
  ─▶ EventEngine (deterministic rules, hysteresis, cooldowns) ─▶ CandidateEvent
  ─▶ EvidenceBuilder (keyframes + crops from ring buffer) ─▶ VLM queue (concurrency 1)
  ─▶ VLMInterpreter (structured JSON, may return unknown) ─▶ EventStore (SQLite + media)
  ─▶ Notifier (desktop / local webhook) and Dashboard (FastAPI, localhost)
```
Single process; the VLM call runs in a worker thread so perception never blocks. If the VLM is busy or unavailable, events are stored rule-only and can be interpreted later ("deferred interpretation").

### Stage details
1. **Detection**: COCO cat/dog classes from a small pretrained detector (D6), run at a configurable rate (start 5 fps, measured). Confidence threshold and class filter in config.
2. **Tracking**: ByteTrack-style (Kalman + IoU, two-stage association with low-score detections) implemented in-repo (~200 lines, testable) or the MIT reference implementation. Track lifecycle: tentative → confirmed → lost (with grace period). Appearance re-ID only if evaluation shows ID switches are a problem. Track IDs are not pet identities (per-pet identity is out of MVP scope).
3. **Spatial**: per track, median depth of valid pixels in the central part of the bbox (shrunk to reduce background), `valid_ratio`, deprojected 3D point, elevation above calibrated floor plane, zone membership, distance to other tracks and to static objects (bowl, bed, sofa). Missing/invalid depth → fields `null` + quality flag, never guessed.
4. **Temporal features** (windows ~2 s / 10 s / 60 s, configurable): speed, motion energy (bbox and pixel-difference inside bbox), bbox aspect ratio trend, elevation change, zone dwell, pairwise proximity duration, periodicity score (repeated back-and-forth paths).
5. **Rules → candidate events** (YAML-configurable thresholds, hysteresis, per-track cooldown): zone_entry, zone_dwell, elevated_surface (on furniture/counter), stationary_long, speed_burst, elevation_jump, animals_close, repetitive_path, no_activity_timeout.
6. **Evidence**: 3–6 frames (start / middle / end / sharpest), each a crop around the animal with margin plus one full context frame; timestamps burned into metadata (not into pixels). If the runtime lacks multi-image input, fallback is a single **grid montage** image (verified first, D6).
7. **VLM interpretation**: prompt contains the measured facts as text ("in zone food_bowl for 42 s, low speed") plus images; output constrained to the schema below; temperature 0; one retry on invalid JSON, otherwise `activity="unknown"` with raw text stored for review.
8. **Notifications**: templated text from rule + measured facts; the VLM description is attached when available. Alerts never depend on the VLM.

### Short-term context and memory
- Short-term: in-memory ring buffer per camera (~10–15 s of frames, configurable) and per-track state; the last K events per track are passed as text context to the VLM prompt (optional, evaluated).
- Persistent: SQLite tables `events`, `tracks`, `evidence`, `runs`, `reviews`; FTS5 index over description/observations for local search; media under `data/media/<date>/<event_id>/`.
- Daily summary (phase 2): small text LLM summarises the day's structured events (input = rows, not images). Optional and evaluated against a hand-written summary.

### Activity difficulty and method

| Activity | Difficulty | Method | Validation data needed |
|---|---|---|---|
| Presence / entering restricted area | Easy | detection + tracking + zone | zone-entry timestamps |
| Jumping onto furniture/counters | Easy–Medium | elevation change + elevated zone; VLM not needed | labelled jump/on-surface segments |
| Walking vs. running vs. stationary | Easy–Medium | speed thresholds (m/s with depth, body-lengths/s without) | segments with locomotion labels |
| Resting / sleeping | Medium | long stationary + low motion energy; VLM to describe posture; label `resting_or_sleeping` (sleep vs. rest often not visually decidable) | long rest segments, night/IR footage |
| Interacting with other animals | Medium | 3D proximity + relative motion; VLM for type, no emotion claims | contact segments |
| Eating / drinking | Medium–Hard | bowl-zone dwell + head-down cue + VLM | bowl-visit segments with eat/drink/sniff labels |
| Unusual inactivity / repetitive movement | Medium (statistical) | per-animal baselines over days, deviation scores | weeks of data; **not validated in MVP** |
| Playing | Hard | motion-energy bursts + VLM → `possible_play` | play segments; expect high FP |
| Grooming | Hard | small localised motion while stationary + VLM; likely needs fine-tuned classifier | grooming segments; accept `unknown` |

Activities ship enabled only if they pass the thresholds agreed after the first evaluation (D10); others remain `possible_*` or `unknown`.

### Event representation
Facts, interpretation and uncertainty are separate fields; numeric confidence is not taken from the VLM (LLM-stated numbers are poorly calibrated).
```json
{
  "event_id": "uuid", "schema_version": 1, "event_type": "pet_activity",
  "camera_id": "rs-living", "track_id": 3, "animal_class": "cat",
  "detector_confidence_mean": 0.81,
  "timestamp_start": "ISO-8601", "timestamp_end": "ISO-8601",
  "trigger": {"rule": "zone_dwell", "zone": "food_bowl", "params": {"min_dwell_s": 20}},
  "observed_facts": {
    "dwell_s": 42.0, "mean_speed_mps": 0.02, "zones": ["food_bowl"],
    "median_depth_m": 2.31, "depth_valid_ratio": 0.82, "elevation_m": null
  },
  "interpretation": {
    "source": "vlm", "model": "qwen3-vl-2b-instruct@Q4_K_M", "prompt_version": "pets-v1",
    "activity": "eating",
    "description": "The cat stands beside the bowl and repeatedly lowers its head toward it.",
    "visual_observations": ["Head lowered toward bowl in 4 of 5 frames", "Body position stable"],
    "certainty": "medium"
  },
  "uncertainty": ["Bowl contents not visible", "Eating vs. sniffing not distinguishable from frames"],
  "confidence": null,
  "evidence": [{"kind": "frame", "path": "media/2026-11-01/<id>/f1.jpg", "ts": "ISO-8601", "crop": true}],
  "review": {"status": "unreviewed", "human_label": null}
}
```
`activity` enum: eating, drinking, resting_or_sleeping, walking, running, playing, grooming, jumping, on_elevated_surface, interacting, other, **unknown**. `interpretation` may be absent (rule-only event). `confidence` is filled only later from calibration against reviewed events.

## D6. Recommended local models and inference runtimes

### VLM: Qwen3-VL 2B vs. 4B quantized

| Aspect | Qwen3-VL-2B-Instruct | Qwen3-VL-4B-Instruct (Q4_K_M) | Status |
|---|---|---|---|
| llama.cpp support | GGUF + separate `mmproj` vision projector; support added ~30 Oct 2025 | same | [unverified]; check the llama.cpp release on the host |
| Ollama | `qwen3-vl:2b` tag reported (~1.9 GB download) | `qwen3-vl:4b` reported with image input | [unverified] |
| Weights size | ~2 GB (reported) | ~3.1–3.3 GB (reported) | [unverified] |
| Actual VRAM peak | weights + mmproj + KV cache (context × image tokens) + CUDA overhead | same, larger | **[to measure]** with NVML at 1/3/6 images and 2 image resolutions |
| Single image | yes (reported) | yes (reported) | verify in spike |
| Multi-image per request | model supports it; **runtime support [unverified]** | same | verify; fallback grid montage |
| Video input | model family advertises video; **llama.cpp/Ollama video path [unverified], treated as unsupported** | same | do not design for it |
| Structured output | Ollama `format` JSON schema / llama.cpp grammar | same | verify with image inputs |
| Latency | **[to measure]** time-to-first-token and total, per image count | **[to measure]** | — |
| License | Apache-2.0 per secondary sources | same | **[unverified]**: confirm on official model card |

Recommendation: start with **2B**, measure; adopt **4B Q4** only if evaluation shows a meaningful accuracy gain that fits the VRAM/latency budget with the detector resident. Comparison baselines if Qwen3-VL fails a requirement: a newer Qwen VL release if one exists at implementation time, SmolVLM2, Moondream 2, InternVL 2B-class (all to be checked for license and runtime support; not pre-selected).

### Runtimes
- **Ollama** (default for setup simplicity): model management, REST, JSON schema `format`, `keep_alive` to unload after idle; `ollama ps` shows loaded size.
- **llama.cpp `llama-server`** (control): explicit context size, GPU layers, mmproj, OpenAI-compatible endpoint.
- **transformers** in-process: reference-quality baseline only, offline evaluation; likely too heavy for 6 GB during live operation [to measure].
`VLMProvider` uses an OpenAI-compatible HTTP client so Ollama and llama-server are interchangeable via config.

Resource limits (config): `max_requests_per_minute`, `max_concurrency=1`, `max_images`, `image_max_side`, `context_tokens`, `keep_alive_s`, `timeout_s`, queue policy (merge duplicates per track, drop oldest when full).

### Detector and tracker
- Candidates (all exported to ONNX, run with onnxruntime-gpu, CPU fallback): **YOLOX-S/Tiny (Apache-2.0)** as default, **RT-DETR-R18 / D-FINE-S (Apache-2.0)** as accuracy alternatives, Ultralytics YOLO11n as a measured reference only (**AGPL-3.0** weights/code: acceptable for a private lab, a problem if the repo is ever distributed; Q3).
- Input 640 (or 416 if latency-bound) [to measure].
- Tracker: ByteTrack-style; OC-SORT as alternative if occlusion handling is poor.
- Pose (animal keypoints) excluded from MVP; reconsider only if eating/grooming accuracy is insufficient.

### Text LLM (phase 2, optional)
Small instruct model (e.g. Qwen3 1.7B / 4B class, [license and Ollama availability unverified]) for daily summaries and natural-language queries over **structured event rows**, never in the detection loop. Not loaded concurrently with the VLM on 6 GB.

## D7. RealSense integration strategy

### Integration
- `lab discover` lists devices and profiles and writes `outputs/devices/<serial>.json`.
- Live: `rs.pipeline` + `rs.config` (configurable colour/depth resolution and fps, start with a profile the device reports as supported), `rs.align(rs.stream.color)`, depth scale from the device, intrinsics from the aligned profile.
- Post-processing filters (decimation, spatial, temporal, hole filling): configurable, **off for measurements by default**, evaluated separately because they change depth values.
- Recording: `rs.recorder` / `config.enable_record_to_file` for `.bag` + sidecar JSON (device, firmware, profiles, calibration version, notes). Replay with `config.enable_device_from_file` and `set_real_time(False)` for deterministic evaluation.
- Multi-camera: out of MVP; the design keeps `camera_id` everywhere so it can be added.

### No camera connected → development fallback
1. `.bag` recordings made once on the host with the camera (or public RealSense sample bags, license permitting).
2. Plain RGB video (phone or webcam) with `depth=None`: all depth features degrade to 2D equivalents with an explicit `depth_unavailable` flag.
3. `SyntheticSource`: generated scene with moving boxes, known depth and known zones, for unit/integration tests and CI.

### Reliability of measurements (to be validated on the actual device)

| Measurement | Expected reliability | Requirements / caveats |
|---|---|---|
| Distance to animal | Usable at room distances with `valid_ratio` check | Stereo depth error grows with distance (roughly quadratically for stereo devices); dark/fluffy fur, low texture, sunlight, motion blur create holes; model-dependent (stereo D4xx vs. other families) |
| Approximate 3D position | Usable (intrinsics + depth) | bbox-median bias toward background; use shrunk bbox or mask |
| Movement through zones | Reliable after calibration | per-camera floor plane + zone polygons; recalibrate if the camera moves (detect via plane-fit drift) |
| Height / jumping | **Controlled conditions only** | needs floor plane; report coarse classes (floor / furniture / counter) rather than cm until validated against tape measurements |
| Occupancy / proximity | Reliable at coarse thresholds | thresholds validated on recordings; 2D overlap is not proximity |
| Object position changes | Reliable for rigid static objects | static background reference + depth/RGB difference; lighting changes affect RGB |
| Simple room mapping | Single-view point cloud / floor plan feasible | multi-view mapping needs SLAM: out of scope |
| Animal size/volume | **Not reliable** for moving, deformable animals | not used |

Depth is never used as evidence of identity or behaviour without validation; it supplies localisation facts with quality flags.

## D8. Proposed directory structure
```
CLAUDE.md
.claude/settings.json                  # minimal, see D-CC below
configs/  lab.example.yaml  zones.example.yaml  rules.pets.yaml
src/perception_lab/
  core/     types.py  config.py  pipeline.py  health.py  runrecord.py
  sources/  base.py  realsense.py  bag.py  video.py  synthetic.py  discover.py
  vision/   detector.py  tracker.py
  spatial/  calibration.py  geometry.py  zones.py  depth_stats.py
  events/   features.py  rules.py  engine.py  evidence.py
  vlm/      provider.py  openai_compat.py  schema.py  prompts/pets_v1.txt
  store/    db.py  retention.py  media.py
  notify/   desktop.py  webhook.py
  web/      app.py  static/
  apps/pets/   config.py  rules.py  labels.py
  eval/     datasets.py  metrics.py  runner.py  report.py  bench.py
  cli.py    # lab discover | record | replay | run | bench | eval | serve | purge
tests/      unit + synthetic integration (no hardware, no GPU, no network)
docs/       architecture.md (short), calibration.md, adr/
data/ outputs/   # git-ignored: recordings, media, db, models, reports
```
Dependencies (minimal, by extra): base `numpy`, `opencv-python-headless`, `pydantic`, `pyyaml`, `httpx`; `realsense`: `pyrealsense2`; `gpu`: `onnxruntime-gpu` (else `onnxruntime`); `web`: `fastapi`, `uvicorn`; `eval`: `motmetrics` (or TrackEval scripts); `nvidia-ml-py` for VRAM monitoring. No Open3D, no agent framework, no message broker.

### Privacy and retention defaults
- No continuous recording; only event clips/frames from the ring buffer are saved.
- Retention (configurable): media 14 days, event rows 365 days, recordings for datasets kept only in explicitly named dataset folders; `lab purge [--dry-run]`.
- Dashboard binds to `127.0.0.1`; no code path makes outbound network calls except to the configured local model endpoint (localhost by default).
- Option to blur people in saved evidence; people detections are not stored as events in the pet app.

## D9. Hardware and resource feasibility
Known: Ubuntu, GTX 1660 SUPER (6 GB, Turing TU116, **no tensor cores**), CUDA available. Unknown: CPU, RAM, driver/CUDA versions, USB controller, camera model.

VRAM budget (all **[to measure]**; this table is the measurement plan, not the result):

| Component | How measured |
|---|---|
| CUDA context + onnxruntime-gpu detector at 416/640 | NVML peak during `lab bench detector` |
| VLM 2B resident, idle | `ollama ps` + NVML |
| VLM 2B during request with 1/3/6 images at 448/672 px max side | NVML peak sampling at 10 Hz |
| VLM 4B Q4 same matrix | same |
| Detector + VLM concurrently | combined run; check for OOM / CPU offload |

Feasibility decision rule: default model = largest configuration whose concurrent peak stays below ~5.5 GB and whose p95 VLM latency is acceptable for alerts (target to be set by you; proposal ≤ 30 s, since alerts do not wait for the VLM). If neither fits with the detector on GPU, run the detector on CPU (measure fps) or unload the VLM between events.

## D10. Experimental methodology and evaluation metrics

**Baselines and configurations**
- Detection: YOLOX-S vs. one Apache-2.0 alternative vs. YOLO11n reference, at 2 input sizes.
- Tracking: ByteTrack vs. OC-SORT.
- Events: rules-only (no VLM) baseline vs. rules + VLM 2B vs. rules + VLM 4B Q4; prompt versions; 1 vs. 3 vs. 6 frames; crops vs. crops+context; multi-image vs. montage.
- Every run writes `outputs/runs/<run_id>/run.json`: git SHA, config, model file hashes, runtime versions (Ollama/llama.cpp/onnxruntime/driver), dataset split hash, seed.

**Metrics**

| Metric | Definition |
|---|---|
| Latency / throughput | per-stage p50/p95 ms, end-to-end frame-to-event ms, processed fps |
| VLM latency | time-to-first-token, total, per image count |
| VRAM / RAM | NVML peak and mean, process RSS peak |
| Detection | precision, recall, AP@0.5 for cat/dog on annotated frames |
| Tracking | IDF1, HOTA (if TrackEval), ID switches per minute |
| Event precision/recall | predicted event matches ground truth if same type, same animal, temporal IoU ≥ 0.3 (or start within ±tolerance); per event type |
| False alerts per day | unmatched notifications ÷ recorded hours × 24, measured on long "normal day" recordings |
| VLM activity accuracy | accuracy and confusion matrix vs. human label on evaluated events; `unknown` counted separately |
| Unknown rate | fraction of interpreted events returning `unknown` (and how many of them were truly ambiguous per annotator) |
| Unsupported-claim rate | fraction of observations not verifiable in the evidence frames, human-audited sample |
| Schema validity | fraction of VLM responses valid on first attempt |

**Dataset**
- 15–25 recording sessions of 5–30 min (bag format) covering: each pet alone, pets together, feeding times, rest periods, day light, artificial light, night/IR, plus 4–8 h of uneventful footage for false-alert measurement.
- Annotation: track boxes on sampled frames (e.g. 1 fps) with CVAT (local, self-hosted) and activity/zone segments as start–end intervals in a simple CSV/JSON (`session, animal, activity, t_start, t_end`). Labels are committed; media are not.
- Inter-annotator check on a subset if a second person is available.

**Splits without leakage**
- Split by **recording session**, never by frame or clip; frames from one session are strongly correlated.
- Group sessions by day and lighting; keep at least one day (and one night session) entirely in test.
- ~60/20/20 (train/val/test) by session; train is used only if fine-tuning is later justified; thresholds and prompts are tuned on val only; test is frozen and evaluated only for milestone reports.
- Record split membership in `data/datasets/<name>/splits.json` with a hash used in run records.

**Repeatable procedure**
1. `uv run lab eval --suite pets --split val --config configs/lab.yaml` replays bags with `set_real_time(False)`, VLM temperature 0, fixed seed.
2. Writes `outputs/runs/<id>/metrics.json` and `report.md`.
3. `uv run lab eval compare <run_a> <run_b>` prints a diff table.

## D11. MVP scope

**In scope**
- Sources: RealSense live, bag replay, video file, synthetic; `lab discover`, `lab record`.
- Calibration: floor plane + zone polygons per camera (simple CLI/dashboard tool).
- Detection + tracking of multiple cats/dogs.
- Spatial facts with quality flags; zone/elevation/proximity/speed/stationary rules.
- Event-gated VLM (Qwen3-VL 2B first) with structured output and `unknown`.
- SQLite event store + FTS5 search + evidence media + retention + purge.
- Desktop notification (libnotify / `notify-send`) and optional localhost webhook.
- Dashboard: camera/model status, live annotated view, zone editor (polygon on image), event timeline with evidence and review buttons (confirm/reject/label).
- Evaluation harness and benchmark commands; first annotated dataset.

**Explicit exclusions**
Cloud APIs or paid services; per-pet identity recognition; model training/fine-tuning; pose/keypoint models; direct video input to the VLM; multi-camera fusion; SLAM/room mapping; animal volume; daily LLM summaries (phase 2); smart-home actuation; authentication/multi-user/remote access; mobile app; Docker deployment; applications A3–A7 beyond what falls out of the shared pipeline.

## D12. Ordered implementation plan with validation commands
Each step ends with ruff + pytest green; no step needs hardware unless marked [HW].

**Step 0: Host spike [HW], scripts outside the package, no repo changes**
Records facts that replace the [unverified] items.
```bash
nvidia-smi
```
```bash
rs-enumerate-devices
```
```bash
uv run --with pyrealsense2 python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) for d in rs.context().devices])"
```
```bash
ollama pull qwen3-vl:2b
```
```bash
ollama ps
```
Test: 1 image, 3 images, montage, video attempt, JSON-schema `format`; log VRAM with `nvidia-smi --query-gpu=memory.used --format=csv -lms 100`. Output: `docs/spike-results.md` (after approval).

**Step 1: Scaffolding**
Rename package to `perception_lab`, CLI `lab`, core types, pydantic config, run record, CLAUDE.md, `.claude/settings.json`, extend `.gitignore`.
```bash
uv sync
```
```bash
uv run ruff check . && uv run ruff format --check .
```
```bash
uv run pytest
```
```bash
uv run lab --help
```

**Step 2: Sources + recording/replay** (synthetic, video, bag, RealSense, discover)
```bash
uv run pytest tests/sources
```
```bash
uv run lab discover
```
```bash
uv run lab record --seconds 60 --out data/recordings/
```
```bash
uv run lab replay data/recordings/<file>.bag --show
```

**Step 3: Spatial + calibration** (deprojection, floor plane, zones, depth stats)
```bash
uv run pytest tests/spatial
```
```bash
uv run lab calibrate --source data/recordings/<file>.bag
```
Manual check: tape-measured distances to 3 markers at ~1, 2, 3 m; record error.

**Step 4: Detection + tracking + benchmark**
```bash
uv run lab bench detector --model data/models/yolox_s.onnx --source data/recordings/<file>.bag
```
```bash
uv run lab eval --suite detect-track --split val
```

**Step 5: Event engine + store + retention**
```bash
uv run pytest tests/events tests/store
```
```bash
uv run lab run --source data/recordings/<file>.bag --no-vlm
```
```bash
uv run lab eval --suite pets --split val --no-vlm
```
```bash
uv run lab purge --dry-run
```

**Step 6: VLM provider + schema + prompts; 2B vs 4B Q4**
```bash
uv run pytest tests/vlm
```
(provider tests use a fake HTTP server; no model needed)
```bash
uv run lab bench vlm --provider ollama --model qwen3-vl:2b --images 1,3,6
```
```bash
uv run lab eval --suite pets --split val
```

**Step 7: Notifications + dashboard**
```bash
uv run lab serve
```
Opens http://127.0.0.1:8000. Check: binds to localhost only, status updates, timeline, review actions persist.

**Step 8: Dataset growth, soak, decision**
Multi-day live run; false alerts per day; frozen test-set report; decide which activities ship enabled and the default VLM.

Commits: one per step on branch `claude/home-ai-perception-lab-c9w6uv`, **only after you approve** committing (you asked for no commits/pushes during planning).

## D-CC. Claude Code project configuration (draft, created in Step 1)
Root `CLAUDE.md` draft (concise; current official format to be re-checked at implementation, docs not reachable from this sandbox):
```markdown
# Home AI Perception Lab
Local-first RealSense + CV + local VLM lab. First app: pet monitoring (src/perception_lab/apps/pets).

## Architecture
sources → vision (detector, tracker) → spatial → events → vlm (event-gated) → store → notify/web.
Modules communicate through dataclasses in core/types.py. Config: configs/*.yaml (pydantic, core/config.py).

## Commands
- Install: `uv sync` (extras: `--extra realsense --extra gpu --extra web --extra eval`)
- Lint/format: `uv run ruff check . && uv run ruff format --check .`
- Tests (no hardware/GPU/network): `uv run pytest`
- Replay: `uv run lab replay <bag>`; Run: `uv run lab run --source <bag|realsense>`
- Evaluate: `uv run lab eval --suite pets --split val`

## Constraints
- Local only: no cloud APIs, no outbound calls except the configured local model endpoint.
- Never commit media, recordings, models, databases or calibration.toml (see .gitignore).
- Tests must not require a camera, GPU or network; use SyntheticSource and fake providers.
- VLM output must be schema-validated; `unknown` is valid; facts, interpretation, uncertainty stay separate.
- Never state performance numbers without a run record in outputs/runs/.
- Depth values carry quality flags; never infer identity/behaviour from depth alone.
```
`.claude/settings.json`: permission allow-list for `uv run pytest*`, `uv run ruff*`, `uv sync`; nothing else. No custom skills, rules or subagents initially; add an evaluation-report skill only if the eval/compare loop becomes repetitive in practice.

## D13. Risks, assumptions and unresolved hardware questions

**Risks**
- Qwen3-VL runtime support for multi-image, structured output with images and video is [unverified]; mitigation: Step 0 spike, montage fallback, provider abstraction.
- 6 GB VRAM contention (detector + VLM); mitigation: measured budget, CPU detector fallback, VLM unload.
- Hard activities (playing, grooming, eating vs. sniffing) may not reach useful accuracy with pretrained models; mitigation: `possible_*`/`unknown`, ship only validated labels.
- Depth holes on dark/fluffy fur and at furniture edges; mitigation: `valid_ratio`, 2D fallback, coarse elevation classes.
- ID switches when pets overlap; mitigation: measure, then consider re-ID.
- Licenses: detector weights (AGPL for Ultralytics), VLM license [unverified].
- Privacy: household members appear in frames; mitigation: no continuous recording, retention, localhost only, optional blur.
- Reported model facts may be outdated by implementation time (Nov 2026); re-check newer small VLMs at the spike.

**Assumptions**
- Single host running everything; one camera for MVP; Ubuntu with working NVIDIA driver; you can provide recordings of your pets.

**Unresolved questions**
- Q1: rename the import package to `perception_lab` (keep repo name)?
- Q2: which RealSense model(s), how many, and where mounted? USB 3 available?
- Q3: will the repository ever be shared/distributed (decides whether AGPL detectors are acceptable even as references)?
- Q4: number and species of pets; is per-pet identity needed later?
- Q5: host CPU/RAM, driver and CUDA versions; is Ollama already installed?
- Q6: acceptable alert latency and maximum false alerts per day.
- Q7: allow committing step-by-step to `claude/home-ai-perception-lab-c9w6uv` after approval?

## Verification of this plan
Planning only. After approval, Step 0 results replace every [unverified]/[to measure] item before model and runtime defaults are frozen; each subsequent step is validated with the commands listed in D12.

---

## Desviaciones durante la implementación

Este plan se aprobó antes de tocar código. Estas son las diferencias con lo que se implementó (ver `docs/reports/2026-11-01-pet-monitoring-mvp-report.md`):

- **Sin renombrar el paquete.** `main` ya contenía `depth_volumetry` con código funcional. Se agregó `perception_lab` como paquete hermano que reutiliza sus módulos.
- **Sin dependencias nuevas.** Config en TOML (`tomllib`), SQLite y HTTP con la librería estándar. No se usó pydantic, pyyaml, httpx, FastAPI ni motmetrics.
- **Detector YOLOv8 en lugar de YOLOX.** Reutiliza el formato de salida que ya decodificaba `depth_volumetry.segmentation`. Ultralytics es AGPL-3.0.
- **Tracker propio estilo ByteTrack**, con una etapa adicional de emparejamiento por distancia de centros para movimientos rápidos.
- **Ramas y commits.** Se trabajó en `feat/pet-activity-monitoring` desde `main`, con commits en Conventional Commits y autoría de Felipe Arancibia.
- **Nombre del comando.** `lab` en lugar de `depth-volumetry` para el nuevo paquete; el comando anterior se mantiene.
- **Hardware y modelos.** Nada de esto se validó con cámara, GPU ni VLM reales, en línea con las reglas de `CLAUDE.md`.
