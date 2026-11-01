"""Capture aligned colour and depth frames from an Intel RealSense camera.

This module needs the optional ``realsense`` extra (``pyrealsense2``). The dependency is
imported lazily, so the rest of the package works without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


class RealSenseUnavailableError(RuntimeError):
    """Raised when ``pyrealsense2`` is not installed."""


@dataclass(frozen=True)
class CapturedFrames:
    """One aligned colour + depth capture and the device parameters needed to measure it."""

    color_bgr: NDArray[np.uint8]
    depth: NDArray[np.uint16]
    focal_length_px: float
    depth_unit_mm: float


def load_pyrealsense2() -> Any:
    """Import ``pyrealsense2`` or raise :class:`RealSenseUnavailableError` with install help."""
    try:
        import pyrealsense2
    except ImportError as exc:
        raise RealSenseUnavailableError(
            "pyrealsense2 is not installed. Install the optional extra with "
            "`uv sync --extra realsense` (or `pip install 'depth-based-volumetry[realsense]'`)."
        ) from exc
    return pyrealsense2


def capture_frames(
    *, width: int = 640, height: int = 480, fps: int = 30, warmup_frames: int = 15
) -> CapturedFrames:
    """Grab one depth frame aligned to the colour frame.

    The first ``warmup_frames`` frames are discarded so auto-exposure can settle.
    """
    rs = load_pyrealsense2()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    try:
        align = rs.align(rs.stream.color)
        frames = None
        for _ in range(max(1, warmup_frames)):
            frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("the camera did not return both a colour and a depth frame")

        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        depth_scale_m = profile.get_device().first_depth_sensor().get_depth_scale()
        return CapturedFrames(
            color_bgr=np.asanyarray(color_frame.get_data()).copy(),
            depth=np.asanyarray(depth_frame.get_data()).copy(),
            focal_length_px=float(intrinsics.fx),
            depth_unit_mm=float(depth_scale_m) * 1000.0,
        )
    finally:
        pipeline.stop()
