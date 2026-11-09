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

- [x] `lab discover`, live capture and `lab record` on the Windows development PC
      (2026-11-09): RealSense D435, firmware 5.13.0.50, serial recorded in the sidecar, GTX 1660
      SUPER (6 GB), `pyrealsense2` 2.59.1 on Python 3.12. Live depth aligned to colour works at
      640x480, 15 fps (about 70 % valid pixels at about 1.1 m median in one indoor scene).
      The camera enumerates as USB 2.1, which limits profiles (for example 1280x720 depth is
      6 fps); a USB 3 port and cable are still to be confirmed.
- [ ] **Recording format.** With `pyrealsense2` 2.59.1, recordings must be `.db3` (the lab's
      default `.bag` name is rejected), and replaying a `.db3` returns corrupt depth-to-colour
      extrinsics, so the aligned depth is all zeros although the file holds valid depth. With
      `pyrealsense2` 2.55.1 on Python 3.11 a `.bag` records and replays correctly
      (about 71 % valid aligned depth). Decide: pin `pyrealsense2` below 2.56 and Python 3.11
      for the `realsense` extra, or make the replay not depend on the file's extrinsics.
- [ ] **VLM (`qwen3-vl:2b` through Ollama, 2026-11-09, same PC).** The tag exists and Ollama
      lists Apache licence (confirm on the official model card). This tag is the *thinking*
      variant: it spends tokens on reasoning before the JSON answer, and `think: false` did
      not turn that off. Ollama's default context is 4096 tokens, which three images already
      fill (3348 prompt tokens), so the answer is cut (`done_reason=length`, empty content);
      six images need 6485 tokens. With `num_ctx = 8192`, one image answered in 3-15 s with
      valid JSON, three images 35 s (valid once, 4842 tokens and invalid on the repeat) and six
      images were cut at 58 s. Whole-GPU memory with the model loaded at 8192 context: about
      4.4 GB, of which 0.7 GB belongs to other programs.
- [ ] **VLM `qwen3-vl:2b-instruct` (1.9 GB, no thinking; tag listed on ollama.com 2026-11-09).**
      `num_ctx = 8192`, model already loaded, same PC. Answers were valid against the schema
      in every request (15 of 15 on frames without a cat; 3 of 3 on a live cat), with 195-252
      output tokens and no reasoning tokens. **Latency depends on Ollama's prompt cache.** An
      identical request repeated three times took 42.5 s, then 2.9 s and 2.9 s (prompt
      evaluation 39.4 s, then 0.0 s); an earlier run that repeated identical requests showed
      1.8-3.6 s medians and was inflated by that cache. Treat a new event as about 40-55 s for
      4-5 images (about 1100 prompt tokens per image, about 110 prompt tokens/s here, whatever
      the crop size). Whole-GPU memory with the model loaded: about 4.3 GB (0.7 GB of it
      other programs).
- [ ] **Live cat check (detector `l` + VLM instruct, 2026-11-09, live D435).** A black cat on a
      cat tree: detected in 48 of 48 analysed frames in one run (score 0.54-0.90, valid depth
      in all, median 1.03 m), 31 of 48 in a second run and 3 of 48 in a third (the animal may
      have moved; not investigated). The VLM answered `resting_or_sleeping` in three runs with
      certainty high, high and medium, while listing its own uncertainties; do not use the
      model's certainty as confidence. One image read by eye matched the description (a cat
      sitting on the lower platform). This is not an accuracy measurement.
- [x] `vlm.num_ctx` (default 8192) is now in `VLMConfig`, sent to Ollama and covered by tests.
      Through `lab bench vlm` with the instruct tag and real frames, first (uncached) request:
      1 image 16.0 s, 2 images 21.6 s, 4 images 42.1 s; repeats 2.0-3.2 s from the cache; 6 of
      6 answers valid; whole-GPU peak 4.25 GB. Still open: change the default model tag from
      `qwen3-vl:2b` to `qwen3-vl:2b-instruct`, compare 2 against 4 images and the 4B model on
      labelled cat events, and decide how many images per event given about 10 s per image.
- [ ] **Detector (multi-class YOLO-layout exports in five sizes, 640 input, synthetic frames, same PC).** CPU
      (`onnxruntime` 1.31): n 37 ms, s 82 ms, m 235 ms, l 281 ms, x 577 ms (p50). GPU through
      DirectML with the VLM already loaded: n 9 ms, s 14 ms, m 27 ms, l 32-34 ms, x 59-63 ms
      (p50), adding 100, 168, 289, 324 and 664 MiB. CUDA through `onnxruntime-gpu` did not
      initialise here (1.31 needs CUDA 13; 1.22 with CUDA 12 libraries failed in cuDNN on the
      first convolution and fell back to CPU), so the CUDA path on Ubuntu is untested.
      One live check detected a black cat at score 0.39 on a cat tree; this is not an accuracy
      measurement. Class ids are 55 (cat) and 86 (dog), not the COCO defaults.
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
- Gallery update: a command (opt-in, with the commit left to the owner) that copies the best
  recent crop of each named pet into [`docs/gallery/`](docs/gallery/README.md) for a personal
  site. It needs per-pet identity, which does not exist yet; today `negrita.jpg` and
  `negrita.json` were made by hand from the last live capture.
- Multi-camera support; daily summaries with a text LLM.
- HTTP API and Docker image for `depth_volumetry` (README "Future work").

## Open decisions

- Whether to push `feat/pet-activity-monitoring` and open a PR.
- Detector weights licence (Ultralytics weights are AGPL-3.0).
- Translate the Spanish sections of the dated plan and the MVP report to English.
