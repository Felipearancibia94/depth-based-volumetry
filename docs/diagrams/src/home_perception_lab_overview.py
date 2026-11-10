"""Overview: pipeline, hardware, models and ToDo. Status comes from ROADMAP.md."""

from excalidraw_lib import BLUE, GREEN, GREY, MUTED, ORANGE, PURPLE

OUTPUT = "docs/diagrams/home-perception-lab-overview"
SEED = 2000


def build(d, repo):
    d.header(
        "Home AI Perception Lab: architecture, hardware, models and ToDo",
        "Green = implemented, tested on synthetic data only.  Orange = needs real hardware or "
        "a real model to validate.  Grey = later.",
    )

    # Pipeline
    d.zone("z_frame", 0, 85, 1840, 150, ("#dbe4ff", "#4a9eed"))
    d.label("z_frame_l", 12, 92, "Per frame (synchronous)", 18, BLUE[1])
    frame = [
        ("src", "FrameSource\nRealSense, .bag,\nvideo, synthetic", ORANGE),
        ("det", "Detector\nYOLOv8-layout ONNX\ncat / dog", ORANGE),
        ("trk", "Tracker\nByteTrack style", GREEN),
        ("spa", "SpatialEstimator\nfloor plane, zones,\nrobust depth", ORANGE),
        ("his", "TrackHistory\nper track", GREEN),
        ("rul", "EventEngine\n7 deterministic\nrules", GREEN),
    ]
    for i, (eid, text, colors) in enumerate(frame):
        d.box(eid, 20 + 300 * i, 130, 245, 90, text, colors, 17)
    for (a, _, _), (b, _, _) in zip(frame, frame[1:], strict=False):
        d.arrow(f"a_{a}_{b}", a, b)

    d.zone("z_event", 0, 255, 1840, 150, ("#e5dbff", "#9775fa"))
    d.label("z_event_l", 170, 262, "Per event (alerts never wait for the VLM)", 18, PURPLE[1])
    event = [
        ("pip", "PetPipeline\nfacts + evidence", GREEN),
        ("sto", "EventStore\nSQLite + images", GREEN),
        ("not", "Notifier\nnotify-send / webhook", GREEN),
        ("vlm", "VLMWorker\nasync, local, optional", ORANGE),
        ("das", "Dashboard\n127.0.0.1", GREEN),
        ("eva", "Eval + bench\nlab eval / lab bench", ORANGE),
    ]
    for i, (eid, text, colors) in enumerate(event):
        d.box(eid, 20 + 317 * i, 300, 215, 90, text, colors, 17)
    d.arrow("a_pip_sto", "pip", "sto")
    d.arrow("a_sto_not", "sto", "not")
    d.arrow("a_vlm_das", "vlm", "das", dashed=True)
    d.arrow("a_das_eva", "das", "eva", dashed=True)
    d.arrow("a_rul_pip", "rul", "pip", elbow_y=245)

    # Hardware
    d.label("h_t", 0, 440, "Hardware needed", 24, ORANGE[1])
    hardware = [
        ("Intel RealSense D435 (fw 5.13.0.50)\nworks live at 640x480, 15 fps\n"
         "enumerates as USB 2.1: needs USB 3", ORANGE),
        ("NVIDIA GTX 1660 SUPER\n6 GB, no tensor cores\n(detector + VLM must share it)", GREEN),
        ("Ubuntu + NVIDIA driver / CUDA\nversions to check:\nnvidia-smi", ORANGE),
        ("CPU and RAM\nnot yet recorded", GREY),
        ("Tape measure\nfloor and zone calibration", ORANGE),
    ]  # fmt: skip
    for i, (text, colors) in enumerate(hardware):
        d.box(f"hw{i + 1}", 0, 485 + 105 * i, 540, 90, text, colors, 17)

    # Models
    d.label("m_t", 600, 440, "Models and runtimes", 24, PURPLE[1])
    models = [
        ("Detector: multi-class YOLO-layout ONNX (cat = 55, dog = 86)\n"
         "l: 281 ms CPU, 32 ms GPU (DirectML); CUDA untested\nlicence of the weights to confirm",
         ORANGE),
        ("Tracker: ByteTrack style, in-repo\nno weights", GREEN),
        ("VLM: qwen3-vl:2b-instruct through Ollama (num_ctx 8192)\n"
         "names the pet; gated on detector score; 4B untested", ORANGE),
        ("Text LLM for daily summaries\nlater, never next to the VLM on 6 GB", GREY),
        ("Runtime: Ollama or llama.cpp llama-server\nloopback only", ORANGE),
    ]  # fmt: skip
    for i, (text, colors) in enumerate(models):
        d.box(f"md{i + 1}", 600, 485 + 105 * i, 640, 90, text, colors, 16)

    # ToDo
    d.label("t_t", 1280, 440, "ToDo", 24, GREEN[1])
    todo2 = (
        "Phase 2: hardware validation\n"
        "- done: lab discover and live capture (D435)\n"
        "- record/replay with pyrealsense2 2.55 (pinned)\n"
        "- connect on USB 3, run lab discover again\n"
        "- check the VLM licence on the model card\n"
        "- live run with light: pet names + summary\n"
        "- lab calibrate: floor plane + zones"
    )
    todo3 = (
        "Phase 3: dataset and evaluation\n"
        "- record 15-25 sessions, label them\n"
        "- split by session, freeze the test split\n"
        "- false events per day, VLM accuracy,\n"
        "  unknown rate, id switches\n"
        "- decide which activities ship enabled"
    )
    later = (
        "Later\n"
        "- per-animal baselines, pet identity\n"
        "- multi-camera, daily summaries\n"
        "- HTTP API / Docker for depth_volumetry"
    )
    d.box("td2", 1280, 485, 560, 215, todo2, ORANGE, 15)
    d.box("td3", 1280, 715, 560, 150, todo3, ORANGE, 15)
    d.box("td4", 1280, 880, 560, 100, later, GREY, 15, dashed=True)

    d.label("foot", 0, 1030,
            "Source of truth for status: ROADMAP.md.  Design: docs/pet-monitoring.md.  "
            "Measured so far: live D435 capture, detector latency (CPU, DirectML) and VLM "
            "latency. Not measured: CUDA, accuracy.", 15, MUTED)  # fmt: skip
