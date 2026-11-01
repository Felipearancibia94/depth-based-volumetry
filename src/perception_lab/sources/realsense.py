"""Intel RealSense streaming, ``.bag`` replay, recording and device discovery.

Needs the optional ``realsense`` extra; ``pyrealsense2`` is imported lazily through
:func:`depth_volumetry.realsense.load_pyrealsense2`. Not validated on hardware in this
repository: run ``lab discover`` and ``lab record`` on the target machine first.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from depth_volumetry.realsense import load_pyrealsense2
from perception_lab.core.config import SourceConfig
from perception_lab.core.types import FrameBundle, Intrinsics
from perception_lab.sources.base import SourceError

_INFO_FIELDS = (
    "name",
    "serial_number",
    "firmware_version",
    "product_line",
    "usb_type_descriptor",
    "physical_port",
)


class RealSenseSource:
    """Aligned colour + depth frames from a live camera (``kind="realsense"``) or a
    recorded ``.bag`` file (``kind="bag"``)."""

    def __init__(self, config: SourceConfig, *, rs_module: Any = None) -> None:
        if config.kind not in ("realsense", "bag"):
            raise ValueError(f"RealSenseSource cannot open source kind {config.kind!r}")
        if config.kind == "bag" and not config.path:
            raise ValueError("source.path must point to a .bag file")
        self.config = config
        self._rs = rs_module
        self._pipeline: Any = None

    def frames(self) -> Iterator[FrameBundle]:
        rs = self._rs or load_pyrealsense2()
        cfg = self.config
        pipeline, rs_config = rs.pipeline(), rs.config()
        playback = cfg.kind == "bag"
        if playback:
            if not Path(cfg.path).is_file():
                raise SourceError(f"recording not found: {cfg.path}")
            rs_config.enable_device_from_file(cfg.path, repeat_playback=False)
        else:
            if cfg.serial:
                rs_config.enable_device(cfg.serial)
            _enable_streams(rs, rs_config, cfg)
        try:
            profile = pipeline.start(rs_config)
        except RuntimeError as exc:
            raise SourceError(f"could not start the RealSense pipeline: {exc}") from exc
        self._pipeline = pipeline
        device = profile.get_device()
        if playback:
            device.as_playback().set_real_time(cfg.real_time)
        depth_unit_mm = _depth_unit_mm(device) if cfg.depth else 1.0
        align = rs.align(rs.stream.color) if cfg.depth else None
        index = 0
        try:
            while True:
                if playback:
                    ok, frames = pipeline.try_wait_for_frames(5000)
                    if not ok:
                        return  # end of recording
                else:
                    frames = pipeline.wait_for_frames(5000)
                if align is not None:
                    frames = align.process(frames)
                color = frames.get_color_frame()
                if not color:
                    continue
                depth = frames.get_depth_frame() if cfg.depth else None
                yield _to_bundle(rs, index, color, depth, depth_unit_mm, cfg.camera_id)
                index += 1
        finally:
            self.close()

    def close(self) -> None:
        if self._pipeline is not None:
            with contextlib.suppress(RuntimeError):
                self._pipeline.stop()
            self._pipeline = None


def record(
    config: SourceConfig, out_path: str | Path, seconds: float, *, rs_module: Any = None
) -> dict[str, Any]:
    """Record a ``.bag`` file and write a JSON sidecar (device, streams, intrinsics)."""
    rs = rs_module or load_pyrealsense2()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pipeline, rs_config = rs.pipeline(), rs.config()
    if config.serial:
        rs_config.enable_device(config.serial)
    _enable_streams(rs, rs_config, config)
    rs_config.enable_record_to_file(str(out))
    profile = pipeline.start(rs_config)
    started = time.time()
    frames_seen, intrinsics = 0, None
    try:
        while time.time() - started < seconds:
            frames = pipeline.wait_for_frames(5000)
            frames_seen += 1
            if intrinsics is None and frames.get_color_frame():
                intrinsics = _intrinsics(frames.get_color_frame())
    finally:
        pipeline.stop()
    device = profile.get_device()
    sidecar = {
        "recording": out.name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "seconds": seconds,
        "frames_seen": frames_seen,
        "device": _device_info(rs, device),
        "streams": {
            "width_px": config.width_px,
            "height_px": config.height_px,
            "fps": config.fps,
            "depth": config.depth,
        },
        "depth_unit_mm": _depth_unit_mm(device) if config.depth else None,
        "color_intrinsics": intrinsics.__dict__ if intrinsics else None,
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return sidecar


def discover(*, rs_module: Any = None) -> list[dict[str, Any]]:
    """Describe every connected device: identity, USB type and supported video profiles."""
    rs = rs_module or load_pyrealsense2()
    devices = []
    for device in rs.context().query_devices():
        info = _device_info(rs, device)
        sensors = []
        for sensor in device.query_sensors():
            profiles = set()
            for p in sensor.get_stream_profiles():
                if p.is_video_stream_profile():
                    v = p.as_video_stream_profile()
                    profiles.add(
                        (str(p.stream_type()), str(p.format()), v.width(), v.height(), p.fps())
                    )
            sensors.append(
                {
                    "name": sensor.get_info(rs.camera_info.name),
                    "profiles": [
                        {"stream": s, "format": f, "width_px": w, "height_px": h, "fps": fps}
                        for s, f, w, h, fps in sorted(profiles)
                    ],
                }
            )
        usb = info.get("usb_type_descriptor") or ""
        warnings = (
            ["connected over USB 2: high resolutions/rates unavailable"]
            if usb.startswith("2")
            else []
        )
        devices.append({**info, "sensors": sensors, "warnings": warnings})
    return devices


def _enable_streams(rs: Any, rs_config: Any, cfg: SourceConfig) -> None:
    rs_config.enable_stream(rs.stream.color, cfg.width_px, cfg.height_px, rs.format.bgr8, cfg.fps)
    if cfg.depth:
        rs_config.enable_stream(
            rs.stream.depth, cfg.width_px, cfg.height_px, rs.format.z16, cfg.fps
        )


def _depth_unit_mm(device: Any) -> float:
    return float(device.first_depth_sensor().get_depth_scale()) * 1000.0


def _device_info(rs: Any, device: Any) -> dict[str, str]:
    info = {}
    for name in _INFO_FIELDS:
        key = getattr(rs.camera_info, name, None)
        if key is not None and device.supports(key):
            info[name] = device.get_info(key)
    return info


def _intrinsics(frame: Any) -> Intrinsics:
    i = frame.profile.as_video_stream_profile().intrinsics
    return Intrinsics(
        int(i.width), int(i.height), float(i.fx), float(i.fy), float(i.ppx), float(i.ppy)
    )


def _to_bundle(
    rs: Any, index: int, color: Any, depth: Any, depth_unit_mm: float, camera_id: str
) -> FrameBundle:
    image = np.asanyarray(color.get_data()).copy()
    if color.profile.format() == rs.format.rgb8:  # recordings may store RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    depth_array = np.asanyarray(depth.get_data()).copy() if depth else None
    return FrameBundle(
        index=index,
        timestamp_s=float(color.get_timestamp()) / 1000.0,
        color_bgr=image,
        depth=depth_array,
        depth_unit_mm=depth_unit_mm,
        intrinsics=_intrinsics(color),
        camera_id=camera_id,
        meta={
            "frame_number": int(color.get_frame_number()),
            "timestamp_domain": str(color.get_frame_timestamp_domain()),
        },
    )
