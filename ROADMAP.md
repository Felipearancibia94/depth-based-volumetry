# Roadmap

Single place for current status and next steps. Statuses come from the code, the tests and
[`docs/reports/2026-11-01-pet-monitoring-mvp-report.md`](docs/reports/2026-11-01-pet-monitoring-mvp-report.md).
Update this file when a phase item changes state.

A visual summary with the hardware and models is in
[`docs/diagrams/home-perception-lab-overview.png`](docs/diagrams/home-perception-lab-overview.png).

Legend: done = implemented and covered by synthetic tests. Nothing below is validated on a
real camera, detector or VLM unless it says so.

## Phase 0: box volumetry (`depth_volumetry`): done

- Calibration, measurement, segmentation decoding, overlay, `measure` / `capture` CLI.
- Not validated with a real model or RealSense capture. Known limits: one box per
  measurement, camera looking straight down at a flat base ([README](README.md#limitations)).

## Phase 1: pet monitoring MVP (`perception_lab`): done on synthetic data

- Sources, detector, tracker, spatial facts, 7 rules, VLM interpreter, SQLite store,
  notifications, dashboard, evaluation and benchmarks, `lab` CLI.
- Branch `feat/pet-activity-monitoring`; not pushed.

## Phase 2: hardware validation: next

Run on the target machine and keep the outputs ([`docs/pet-monitoring.md`](docs/pet-monitoring.md#first-steps-on-the-target-machine)):

- [ ] `nvidia-smi`, `lab discover`, `lab record`; confirm the `pyrealsense2` wheel for the Python version.
- [ ] Confirm the VLM tag (`qwen3-vl:2b`) and its licence on the official model card.
- [ ] `lab bench detector` and `lab bench vlm`; decide CPU vs GPU (`onnxruntime-gpu`) on a 6 GB card.
- [ ] Calibrate the floor plane and zones with a tape measure.

## Phase 3: dataset and evaluation

- [ ] Record 15–25 sessions and label them (`examples/labels.example.json`).
- [ ] Split by session; tune on validation only; freeze the test split.
- [ ] Measure false events per day, VLM accuracy with `unknown` separate, id switches.
- [ ] Decide which activities ship enabled.

## Later (not scheduled)

- CI: `.github/workflows/ci.yml` runs ruff and pytest on Ubuntu and Windows (no GPU, camera
  or network). Written but not yet run on GitHub; the branch is not pushed.
- Per-animal baselines for `unusual_inactivity`; per-pet identity.
- Further applications from the plan: food/water station, interaction monitoring,
  occupancy heatmaps, smart-home hooks.
- Multi-camera support; daily summaries with a text LLM.
- HTTP API and Docker image for `depth_volumetry` (README "Future work").

## Open decisions

- Whether to push `feat/pet-activity-monitoring` and open a PR.
- Detector weights licence (Ultralytics weights are AGPL-3.0).
- Translate the Spanish sections of the dated plan and the MVP report to English.
