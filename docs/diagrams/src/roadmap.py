"""Roadmap diagram. Keep in step with ROADMAP.md (the source of truth)."""

from excalidraw_lib import BLUE, GREEN, GREY, MUTED, ORANGE, PURPLE

OUTPUT = "docs/diagrams/roadmap"
SEED = 7000
COL_W, GAP = 340, 50


def build(d, repo):
    d.header(
        "Roadmap: depth-based-volumetry and the home perception lab",
        "Green = done, synthetic tests only.  Blue = measured on the development PC "
        "(2026-11-09).  Orange = open.  Grey = later.  Purple = open decision.",
    )
    phases = [
        ("p0", "Phase 0\nBox volumetry\ndone", GREEN),
        ("p1", "Phase 1\nPet monitoring MVP\ndone on synthetic data", GREEN),
        ("p2", "Phase 2\nHardware validation\nIN PROGRESS", ORANGE),
        ("p3", "Phase 3\nDataset and evaluation\nnext", ORANGE),
        ("pl", "Later\nnot scheduled", GREY),
    ]
    xs = [i * (COL_W + GAP) for i in range(len(phases))]
    for (eid, text, colors), x in zip(phases, xs, strict=True):
        d.box(eid, x, 95, COL_W, 90, text, colors, 18, dashed=eid == "pl",
              stroke_width=4 if eid == "p2" else 2)  # fmt: skip
    for a, b in zip(phases, phases[1:], strict=False):
        d.arrow(f"a_{a[0]}_{b[0]}", a[0], b[0], stroke_width=3)
    d.label("here", xs[2] + 95, 190, "you are here", 16, ORANGE[1])

    y0 = 225
    y_p0 = d.column("i0_", xs[0], y0, COL_W, [
        ("Calibration, measurement, segmentation decoding, overlay", GREEN),
        ("measure / capture CLI", GREEN),
        ("Not run with a real model or RealSense capture; one box, camera straight down",
         ORANGE),
    ])  # fmt: skip
    y_p1 = d.column("i1_", xs[1], y0, COL_W, [
        ("Sources, detector, tracker, spatial facts, 7 deterministic rules", GREEN),
        ("VLM interpreter, SQLite store, notifications, dashboard", GREEN),
        ("Evaluation, benchmarks, lab CLI", GREEN),
        ("Pet names: gate on detector score, owner correction, pet filter, lab summary", GREEN),
    ])  # fmt: skip
    d.column("i2_", xs[2], y0, COL_W, [
        ("lab discover + live capture: D435, 640x480 at 15 fps, enumerates as USB 2.1", BLUE),
        ("Detector latency (model l): 281 ms CPU, 32 ms GPU via DirectML; CUDA failed here",
         BLUE),
        ("VLM qwen3-vl:2b-instruct, num_ctx 8192: valid JSON; first request 16 s (1 image) "
         "to 42 s (4 images)", BLUE),
        ("Live black cat detected and described (not an accuracy measurement)", BLUE),
        ("lab record / replay with pyrealsense2 2.55 on Python 3.11", ORANGE),
        ("Live run with light and a pet: names, tune gate 0.4, re-run the no-animal case",
         ORANGE),
        ("Test Leia (dog); confirm USB 3 port and cable", ORANGE),
        ("2 vs 4 images per event; try the 4B model on labelled events", ORANGE),
        ("Night: infrared stream path", ORANGE),
        ("Calibrate floor plane and zones with a tape measure", ORANGE),
    ])  # fmt: skip
    d.column("i3_", xs[3], y0, COL_W, [
        ("Record 15-25 sessions and label them", ORANGE),
        ("Split by session, tune on validation, freeze the test split", ORANGE),
        ("False events per day, VLM accuracy with unknown separate, id switches", ORANGE),
        ("Decide which activities ship enabled", ORANGE),
    ])  # fmt: skip
    d.column("il_", xs[4], y0, COL_W, [
        ("Run CI on GitHub (workflow written, Python 3.11)", GREY),
        ("Per-animal baselines; per-pet detector if VLM naming is too weak", GREY),
        ("Food/water station, interaction, occupancy heatmaps, smart-home hooks", GREY),
        ("Gallery update command (best crop per pet)", GREY),
        ("Multi-camera; daily summaries with a text LLM", GREY),
        ("HTTP API and Docker for depth_volumetry", GREY),
    ])  # fmt: skip

    yd = max(y_p0, y_p1) + 30
    d.zone("z_dec", 0, yd, 2 * COL_W + GAP, 225, ("#e5dbff", "#9775fa"))
    d.label("dec_t", 14, yd + 10, "Open decisions", 20, PURPLE[1])
    d.column("d_", 20, yd + 48, 2 * COL_W + GAP - 40, [
        ("Open a PR for feat/pet-activity-monitoring", PURPLE),
        ("Licence of the detector weights", PURPLE),
        ("Translate the Spanish sections of the dated plan and MVP report", PURPLE),
    ])  # fmt: skip
    d.label("foot", 0, yd + 250,
            "Source of truth: ROADMAP.md.  Nothing here is a real-world accuracy figure.",
            15, MUTED)  # fmt: skip
