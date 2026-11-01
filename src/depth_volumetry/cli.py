"""Command-line interface: ``depth-volumetry measure`` and ``depth-volumetry capture``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from depth_volumetry import __version__
from depth_volumetry.calibration import CalibrationError, load_calibration
from depth_volumetry.io import read_color, read_depth, read_mask, write_image
from depth_volumetry.measurement import MeasurementError, footprint_px, measure_box
from depth_volumetry.overlay import draw_measurement
from depth_volumetry.realsense import RealSenseUnavailableError

EXIT_OK = 0
EXIT_MEASUREMENT_FAILED = 1
EXIT_USAGE_OR_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depth-volumetry",
        description="Estimate box dimensions from an aligned colour + depth image pair.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    measure = subcommands.add_parser(
        "measure",
        help="measure the box in a colour + depth capture",
        description="Measure one box. The object is found either by an ONNX segmentation "
        "model (--model) or from a ready-made binary mask image (--mask).",
    )
    measure.add_argument(
        "--depth", required=True, type=Path, help="depth map (.npy or 16-bit .png)"
    )
    measure.add_argument(
        "--color",
        type=Path,
        help="colour image; required with --model and --overlay",
    )
    source = measure.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="ONNX instance-segmentation model")
    source.add_argument("--mask", type=Path, help="binary mask image of the box")
    measure.add_argument("--calibration", required=True, type=Path, help="calibration TOML file")
    measure.add_argument("--overlay", type=Path, help="write an annotated image to this path")
    measure.add_argument("--json", action="store_true", help="print the result as JSON")
    measure.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="minimum detection score when using --model (default: %(default)s)",
    )
    measure.set_defaults(handler=_run_measure)

    capture = subcommands.add_parser(
        "capture",
        help="capture an aligned colour + depth pair from a RealSense camera",
        description="Requires the optional extra: uv sync --extra realsense",
    )
    capture.add_argument("--out", required=True, type=Path, help="output directory")
    capture.set_defaults(handler=_run_capture)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except MeasurementError as exc:
        print(f"measurement failed: {exc}", file=sys.stderr)
        return EXIT_MEASUREMENT_FAILED
    except (CalibrationError, RealSenseUnavailableError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_OR_INPUT


def _run_measure(args: argparse.Namespace) -> int:
    if (args.model or args.overlay) and args.color is None:
        raise ValueError("--color is required with --model and --overlay")

    calibration = load_calibration(args.calibration)
    depth = read_depth(args.depth)
    color = read_color(args.color) if args.color else None
    if color is not None and color.shape[:2] != depth.shape:
        raise ValueError(
            f"colour image {color.shape[:2]} and depth map {depth.shape} must have the same "
            "height and width (the depth map must be aligned to the colour image)"
        )

    if args.mask:
        mask = read_mask(args.mask)
    else:
        from depth_volumetry.segmentation import SegmentationModel, select_target

        model = SegmentationModel.from_file(args.model, conf_threshold=args.confidence)
        detections = model.predict(color)
        mask = select_target(detections, depth.shape, calibration.reference_point).mask

    result = measure_box(depth, mask, calibration)

    if args.overlay:
        write_image(args.overlay, draw_measurement(color, footprint_px(mask), result))

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        for key, value in result.to_dict().items():
            print(f"{key}: {value:.1f}")
    return EXIT_OK


def _run_capture(args: argparse.Namespace) -> int:
    from depth_volumetry.realsense import capture_frames

    frames = capture_frames()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_image(out / "color.png", frames.color_bgr)
    np.save(out / "depth.npy", frames.depth)
    (out / "device.toml").write_text(
        f"focal_length_px = {frames.focal_length_px:.3f}\n"
        f"depth_unit_mm = {frames.depth_unit_mm:.6f}\n"
    )
    print(f"saved color.png, depth.npy and device.toml to {out}")
    print(f"focal_length_px = {frames.focal_length_px:.3f}")
    print(f"depth_unit_mm = {frames.depth_unit_mm:.6f}")
    return EXIT_OK
