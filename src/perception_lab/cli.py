"""``lab`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from perception_lab import __version__
from perception_lab.core.config import ConfigError, LabConfig, dump_config, load_config
from perception_lab.sources import SourceError, limit, open_source
from perception_lab.spatial.geometry import SpatialError

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE_OR_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab", description="Home perception lab: local-first RGB-D pet monitoring."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p: argparse.ArgumentParser, source: bool = True) -> argparse.ArgumentParser:
        p.add_argument("--config", type=Path, help="lab TOML file (default: built-in defaults)")
        if source:
            p.add_argument(
                "--source",
                choices=("synthetic", "video", "bag", "realsense"),
                help="override source.kind",
            )
            p.add_argument("--path", help="override source.path (video or .bag file)")
            p.add_argument("--frames", type=int, help="stop after this many frames")
        return p

    p = sub.add_parser("init-config", help="write the default configuration as TOML")
    p.add_argument("--out", type=Path, default=Path("lab.toml"))
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(handler=_run_init_config)

    p = with_config(sub.add_parser("run", help="run pet monitoring on the configured source"))
    vlm = p.add_mutually_exclusive_group()
    vlm.add_argument("--vlm", action="store_true", help="enable VLM interpretation")
    vlm.add_argument("--no-vlm", action="store_true", help="disable VLM interpretation")
    p.add_argument("--quiet", action="store_true", help="do not print notifications")
    p.add_argument("--serve", action="store_true", help="also start the local dashboard")
    p.set_defaults(handler=_run_run)

    p = with_config(sub.add_parser("serve", help="dashboard over the stored events"), source=False)
    p.set_defaults(handler=_run_serve)

    p = sub.add_parser("discover", help="list connected RealSense devices and stream profiles")
    p.set_defaults(handler=_run_discover)

    p = with_config(sub.add_parser("record", help="record a RealSense session to a .bag file"))
    p.add_argument("--out", type=Path, help="output .bag (default data/recordings/<time>.bag)")
    p.add_argument("--seconds", type=float, default=60.0)
    p.set_defaults(handler=_run_record)

    p = with_config(sub.add_parser("replay", help="read a recording and report stream statistics"))
    p.add_argument("--preview-dir", type=Path, help="write colour + depth previews here")
    p.add_argument("--preview-every", type=int, default=30)
    p.set_defaults(handler=_run_replay)

    p = with_config(
        sub.add_parser("calibrate", help="fit the floor plane and write the scene file")
    )
    p.add_argument(
        "--out", type=Path, help="scene TOML (default: spatial.scene_path or scene.toml)"
    )
    p.add_argument("--calib-frames", type=int, default=30, help="frames to average")
    p.add_argument("--roi-top", type=float, default=0.5, help="use rows below this fraction")
    p.set_defaults(handler=_run_calibrate)

    p = with_config(sub.add_parser("eval", help="replay a source and score events against labels"))
    p.add_argument(
        "--labels", type=Path, help="labels JSON (synthetic sources have built-in labels)"
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/runs"))
    p.add_argument("--tolerance", type=float, default=2.0, help="time tolerance in seconds")
    vlm = p.add_mutually_exclusive_group()
    vlm.add_argument("--vlm", action="store_true", help="enable VLM interpretation")
    vlm.add_argument("--no-vlm", action="store_true", help="disable VLM interpretation")
    p.set_defaults(handler=_run_eval)

    p = sub.add_parser("bench", help="measure latency and GPU memory on this machine")
    bench = p.add_subparsers(dest="target", required=True)
    b = with_config(bench.add_parser("detector", help="detector latency per frame"))
    b.add_argument("--warmup", type=int, default=5)
    b.add_argument("--out", type=Path, help="also write the JSON result here")
    b.set_defaults(handler=_run_bench_detector)
    b = with_config(bench.add_parser("vlm", help="VLM latency, schema validity and VRAM"), False)
    b.add_argument("--images", default="1,3", help="comma-separated image counts per request")
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--image", help="test image (default: a synthetic frame)")
    b.add_argument("--model", help="override vlm.model")
    b.add_argument("--montage", action="store_true", help="send frames as one grid image")
    b.add_argument("--out", type=Path, help="also write the JSON result here")
    b.set_defaults(handler=_run_bench_vlm)

    p = with_config(
        sub.add_parser("summary", help="events and time per pet and activity for one day"),
        source=False,
    )
    p.add_argument("--date", help="day as YYYY-MM-DD in local time (default: today)")
    p.set_defaults(handler=_run_summary)

    p = with_config(sub.add_parser("purge", help="apply media and event retention"), source=False)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(handler=_run_purge)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ConfigError, SourceError, SpatialError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_OR_INPUT


def load_with_overrides(args: argparse.Namespace) -> LabConfig:
    config = load_config(getattr(args, "config", None))
    source = config.source
    if getattr(args, "source", None):
        source = replace(source, kind=args.source)
    if getattr(args, "path", None):
        source = replace(source, path=args.path)
    if getattr(args, "frames", None) is not None:
        source = replace(source, max_frames=args.frames)
    config = replace(config, source=source)
    if getattr(args, "vlm", False):
        config = replace(config, vlm=replace(config.vlm, enabled=True))
    if getattr(args, "no_vlm", False):
        config = replace(config, vlm=replace(config.vlm, enabled=False))
    return config


def _run_init_config(args: argparse.Namespace) -> int:
    if args.out.exists() and not args.force:
        raise ValueError(f"{args.out} exists; use --force to overwrite it")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump_config(LabConfig()))
    print(f"wrote {args.out}")
    return EXIT_OK


def _run_run(args: argparse.Namespace) -> int:
    from perception_lab.apps.pets.app import build_app

    config = load_with_overrides(args)
    app = build_app(
        config, log_stream=None if args.quiet else sys.stdout, keep_live_view=args.serve
    )
    server = None
    if args.serve:
        from perception_lab.web.server import DashboardState, start_server

        state = DashboardState(
            app.store,
            config.storage.media_dir,
            app.pipeline.scene,
            app.pipeline,
            pet_names=tuple(p.name for p in config.vlm.pets),
        )
        server, _ = start_server(
            state, config.web.host, config.web.port, allow_remote=config.web.allow_remote
        )
        print(f"dashboard: http://{config.web.host}:{server.server_address[1]}/", flush=True)
    stop = threading.Event()
    try:
        status = app.pipeline.run(app.frames(), stop)
    except KeyboardInterrupt:
        stop.set()
        app.pipeline.finish()
        status = app.pipeline.status
    finally:
        if server is not None:
            server.shutdown()
        app.close()
    summary = {
        k: status.to_dict()[k] for k in ("frames", "detections", "events", "notifications", "vlm")
    }
    summary["stage_ms"] = status.stage_ms
    summary["db_path"] = config.storage.db_path
    print(json.dumps(summary, indent=2))
    return EXIT_OK


def _run_serve(args: argparse.Namespace) -> int:
    from perception_lab.spatial.scene import load_scene
    from perception_lab.store.db import EventStore
    from perception_lab.web.server import DashboardState, start_server

    config = load_with_overrides(args)
    store = EventStore(config.storage.db_path)
    scene = load_scene(config.spatial.scene_path) if config.spatial.scene_path else None
    state = DashboardState(
        store, config.storage.media_dir, scene, pet_names=tuple(p.name for p in config.vlm.pets)
    )
    server, thread = start_server(
        state, config.web.host, config.web.port, allow_remote=config.web.allow_remote
    )
    print(f"dashboard: http://{config.web.host}:{server.server_address[1]}/ (Ctrl-C to stop)")
    try:
        thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        store.close()
    return EXIT_OK


def _run_discover(args: argparse.Namespace) -> int:
    from perception_lab.sources.realsense import discover

    devices = discover()
    print(json.dumps(devices, indent=2))
    if not devices:
        print("no RealSense device found", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


def _run_record(args: argparse.Namespace) -> int:
    from perception_lab.sources.realsense import record

    config = load_with_overrides(args)
    out = args.out or Path("data/recordings") / time.strftime("%Y%m%d-%H%M%S.bag")
    sidecar = record(replace(config.source, kind="realsense"), out, args.seconds)
    print(
        f"recorded {sidecar['frames_seen']} framesets to {out} (sidecar {out.with_suffix('.json')})"
    )
    return EXIT_OK


def _run_replay(args: argparse.Namespace) -> int:
    import cv2

    from depth_volumetry.io import write_image

    config = load_with_overrides(args)
    source = open_source(config.source)
    count, first_s, last_s, valid, shape = 0, None, None, [], None
    for bundle in limit(source.frames(), config.source.max_frames):
        count += 1
        first_s = bundle.timestamp_s if first_s is None else first_s
        last_s = bundle.timestamp_s
        shape = bundle.color_bgr.shape
        if bundle.depth is not None:
            valid.append(float((bundle.depth > 0).mean()))
        if args.preview_dir and count % args.preview_every == 1:
            preview = bundle.color_bgr
            if bundle.depth is not None:
                depth_m = bundle.depth_m()
                scaled = np.clip(depth_m / 5.0 * 255, 0, 255).astype(np.uint8)
                colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
                colored[bundle.depth == 0] = 0
                preview = np.hstack([preview, colored])
            write_image(args.preview_dir / f"frame_{bundle.index:06d}.jpg", preview)
    source.close()
    duration = (last_s - first_s) if count > 1 else 0.0
    report = {
        "frames": count,
        "duration_s": round(duration, 2),
        "mean_fps": round((count - 1) / duration, 2) if duration > 0 else None,
        "color_shape": list(shape) if shape else None,
        "depth_valid_fraction_median": round(float(np.median(valid)), 3) if valid else None,
    }
    print(json.dumps(report, indent=2))
    return EXIT_OK if count else EXIT_FAILED


def _run_calibrate(args: argparse.Namespace) -> int:
    from perception_lab.spatial.calibrate import fit_floor_plane
    from perception_lab.spatial.scene import SceneCalibration, dump_scene, load_scene

    config = load_with_overrides(args)
    out = args.out or Path(config.spatial.scene_path or "scene.toml")
    source = open_source(config.source)
    frames, intrinsics = [], None
    for bundle in source.frames():
        if bundle.has_depth:
            frames.append(bundle.depth_m())
            intrinsics = bundle.intrinsics
        if len(frames) >= args.calib_frames:
            break
    source.close()
    if not frames or intrinsics is None:
        raise SpatialError("the source produced no depth frames to calibrate from")
    plane, inliers = fit_floor_plane(frames, intrinsics, roi_top_fraction=args.roi_top)
    scene = load_scene(out) if out.exists() else SceneCalibration(config.source.camera_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_scene(replace(scene, floor=plane)))
    print(
        json.dumps(
            {
                "scene": str(out),
                "camera_height_above_floor_m": round(plane.offset_m, 3),
                "tilt_from_vertical_deg": round(plane.tilt_from_up_deg(), 1),
                "inlier_ratio": round(inliers, 3),
                "frames": len(frames),
                "zones_kept": len(scene.zones),
            },
            indent=2,
        )
    )
    if inliers < 0.5:
        print(
            "warning: under half the points fit the plane; check the floor is visible and clear",
            file=sys.stderr,
        )
    return EXIT_OK


def _run_eval(args: argparse.Namespace) -> int:
    from perception_lab.eval.runner import run_evaluation

    config = load_with_overrides(args)
    metrics = run_evaluation(
        config, labels_path=args.labels, out_dir=args.out_dir, tolerance_s=args.tolerance
    )
    run_dir = args.out_dir / metrics["run_id"]
    print((run_dir / "report.md").read_text())
    print(f"results: {run_dir}")
    return EXIT_OK


def _emit(result: dict, out: Path | None) -> None:
    text = json.dumps(result, indent=2)
    print(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)


def _run_bench_detector(args: argparse.Namespace) -> int:
    from perception_lab.core.runrecord import build_run_record
    from perception_lab.eval.bench import bench_detector

    config = load_with_overrides(args)
    result = bench_detector(config, frames=config.source.max_frames or 100, warmup=args.warmup)
    result["run_record"] = build_run_record(config)
    _emit(result, args.out)
    return EXIT_OK


def _run_bench_vlm(args: argparse.Namespace) -> int:
    from perception_lab.core.runrecord import build_run_record
    from perception_lab.eval.bench import bench_vlm

    config = load_with_overrides(args)
    vlm = config.vlm
    if args.model:
        vlm = replace(vlm, model=args.model)
    if args.montage:
        vlm = replace(vlm, montage=True)
    config = replace(config, vlm=replace(vlm, enabled=True))
    counts = [int(v) for v in args.images.split(",") if v.strip()]
    result = bench_vlm(config, image_counts=counts, repeats=args.repeats, image_path=args.image)
    result["run_record"] = build_run_record(config)
    _emit(result, args.out)
    return EXIT_OK


def _run_summary(args: argparse.Namespace) -> int:
    import datetime as dt

    from perception_lab.store.db import EventStore
    from perception_lab.store.summary import daily_summary

    day = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    config = load_with_overrides(args)
    store = EventStore(config.storage.db_path)
    try:
        print(json.dumps(daily_summary(store, day), indent=2))
    finally:
        store.close()
    return EXIT_OK


def _run_purge(args: argparse.Namespace) -> int:
    from perception_lab.store.db import EventStore
    from perception_lab.store.retention import purge

    config = load_with_overrides(args)
    store = EventStore(config.storage.db_path)
    try:
        report = purge(
            store,
            config.storage.media_dir,
            media_retention_days=config.storage.media_retention_days,
            event_retention_days=config.storage.event_retention_days,
            dry_run=args.dry_run,
        )
    finally:
        store.close()
    print(json.dumps(report.__dict__, indent=2))
    return EXIT_OK
