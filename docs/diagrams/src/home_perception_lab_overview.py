"""Level 1 overview: hardware and weights, the two packages, local services and data.

Structure comes from ARCHITECTURE.md; status and measurements from ROADMAP.md. Detail per
level is in docs/architecture/ and docs/diagrams/mermaid/.
"""

from excalidraw_lib import BLUE, GREEN, GREY, MUTED, ORANGE

OUTPUT = "docs/diagrams/home-perception-lab-overview"
SEED = 2000

COL1_X, COL1_W = 0, 470
COL2_X, COL2_W = 560, 520
COL3_X, COL3_W = 1170, 560
TOP = 135


def build(d, repo):
    d.header(
        "Home AI Perception Lab: system overview (level 1)",
        "Blue = measured on the development PC (2026-11-09).  Green = implemented, synthetic "
        "tests only.  Orange = not validated on real hardware or a real model.  Grey = later.",
    )

    d.zone("z1", COL1_X - 15, 85, COL1_W + 30, 640, ("#ffe8cc", "#e8590c"))
    d.label("z1l", COL1_X, 92, "Hardware and weights (this PC)", 18, ORANGE[1])
    d.zone("z2", COL2_X - 15, 85, COL2_W + 30, 640, ("#dbe4ff", "#4a9eed"))
    d.label("z2l", COL2_X, 92, "This repository: two packages", 18, BLUE[1])
    d.zone("z3", COL3_X - 15, 85, COL3_W + 30, 640, ("#e5dbff", "#9775fa"))
    d.label("z3l", COL3_X, 92, "Local services and data (loopback only)", 18, "#6741d9")

    # Column 1: hardware and weights
    d.box("cam", COL1_X, TOP, COL1_W, 100,
          "Intel RealSense D435 (fw 5.13.0.50)\nlive at 640x480, 15 fps\n"
          "USB 2.1 today: a USB 3 port is to confirm", BLUE, 16)  # fmt: skip
    d.box("gpu", COL1_X, 260, COL1_W, 100,
          "NVIDIA GTX 1660 SUPER, 6 GB\nno tensor cores; detector and VLM share it\n"
          "CUDA path failed here, DirectML works", BLUE, 16)  # fmt: skip
    d.box("onnx", COL1_X, 385, COL1_W, 125,
          "Detector weights (not in git)\nmulti-class YOLO-layout ONNX, cat = 55, dog = 86\n"
          "l: 281 ms CPU, 32 ms GPU (DirectML)\nlicence of the weights to confirm",
          BLUE, 16)  # fmt: skip

    # Column 2: repository
    d.box("pipe", COL2_X, TOP, COL2_W, 100,
          "perception_lab: pet monitoring\nframe loop: detect, track, spatial facts, 7 rules\n"
          "alerts never wait for the VLM", GREEN, 16)  # fmt: skip
    d.box("vol", COL2_X, 260, COL2_W, 100,
          "depth_volumetry: box length, width, height\nmeasure and capture CLI\n"
          "not run with a real model or capture yet", ORANGE, 16)  # fmt: skip
    d.box("vlmbox", COL2_X, 385, COL2_W, 100,
          "VLM interpretation + pet-name gate\ndescribes the event after the alert\n"
          "name kept only if detector score >= 0.4", ORANGE, 16)  # fmt: skip
    d.box("dash", COL2_X, 510, COL2_W, 90,
          "Dashboard on 127.0.0.1\nevents, pet filter, owner correction, daily summary",
          GREEN, 16)  # fmt: skip
    d.box("evalb", COL2_X, 625, COL2_W, 80,
          "Evaluation and benchmarks\nlab eval, lab bench (no real labelled data yet)",
          ORANGE, 16)  # fmt: skip

    # Column 3: services and data
    d.box(
        "notif",
        COL3_X,
        TOP,
        COL3_W,
        80,
        "Notifications\nnotify-send or loopback webhook",
        GREEN,
        16,
    )
    d.box(
        "db",
        COL3_X,
        240,
        COL3_W,
        90,
        "SQLite events + evidence JPEGs\ndata/lab.db, data/media (git-ignored)",
        GREEN,
        16,
    )
    d.box("ollama", COL3_X, 385, COL3_W, 100,
          "Ollama (loopback only)\nqwen3-vl:2b-instruct, num_ctx 8192\n"
          "first request 16 s (1 image) to 42 s (4 images)", BLUE, 16)  # fmt: skip
    d.box("rec", COL3_X, 510, COL3_W, 90,
          "Recordings (.bag + JSON sidecar)\nrecord / replay with the pinned SDK: to verify",
          ORANGE, 16)  # fmt: skip
    d.box("later", COL3_X, 625, COL3_W, 80,
          "Not scheduled: night / infrared path, per-animal\n"
          "identity, multi-camera, text-LLM summaries",
          GREY, 16, dashed=True)  # fmt: skip

    # Data flow (solid = synchronous or direct, dashed = asynchronous or optional)
    d.arrow("a_cam_pipe", "cam", "pipe")
    d.arrow("a_cam_vol", "cam", "vol")
    d.arrow("a_onnx_pipe", "onnx", "pipe")
    d.arrow("a_pipe_vol", "pipe", "vol")
    d.arrow("a_pipe_notif", "pipe", "notif")
    d.arrow("a_pipe_db", "pipe", "db")
    d.arrow("a_vlm_ollama", "vlmbox", "ollama", dashed=True)
    d.arrow("a_dash_db", "dash", "db", dashed=True)

    d.label("foot", 0, 745,
            "Arrows: solid = direct, dashed = asynchronous or reading.  pipe -> vol: the lab "
            "reuses decode, io and the lazy SDK import.  Detail: ARCHITECTURE.md (levels 1-4).  "
            "Status: ROADMAP.md.  Not measured: CUDA, accuracy.", 15, MUTED)  # fmt: skip
