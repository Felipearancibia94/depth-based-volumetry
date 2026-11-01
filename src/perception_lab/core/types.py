"""Plain data types shared by all lab modules.

Conventions (same as ``depth_volumetry``): units go in the name, points are ``(x, y)``,
arrays are indexed ``[row, col]`` and shapes are ``(height, width)``.
The camera frame follows the RealSense convention: +X right, +Y down, +Z forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

BoxXYXY = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics of the stream the depth map is aligned to (distortion ignored)."""

    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def deproject_m(self, x_px: float, y_px: float, depth_m: float) -> Point3:
        """Pixel ``(x, y)`` at ``depth_m`` -> 3D point in the camera frame, in metres."""
        return (
            (x_px - self.cx_px) * depth_m / self.fx_px,
            (y_px - self.cy_px) * depth_m / self.fy_px,
            depth_m,
        )

    def project_px(self, point_m: Point3) -> tuple[float, float]:
        x, y, z = point_m
        return (x * self.fx_px / z + self.cx_px, y * self.fy_px / z + self.cy_px)


@dataclass
class FrameBundle:
    """One colour frame and, when available, the depth map aligned to it."""

    index: int
    timestamp_s: float  # source clock (device or file time)
    color_bgr: NDArray[np.uint8]  # (H, W, 3)
    depth: NDArray[np.uint16] | None = None  # raw units, aligned to colour, (H, W)
    depth_unit_mm: float = 1.0  # millimetres per raw depth unit
    intrinsics: Intrinsics | None = None
    camera_id: str = "cam0"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_depth(self) -> bool:
        return self.depth is not None and self.intrinsics is not None

    def depth_m(self) -> NDArray[np.float32] | None:
        if self.depth is None:
            return None
        return self.depth.astype(np.float32) * np.float32(self.depth_unit_mm / 1000.0)


@dataclass(frozen=True)
class Detection:
    """One detected animal in image coordinates."""

    box_xyxy: BoxXYXY
    score: float
    label: str
    class_id: int = -1
    mask: NDArray[np.bool_] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class SpatialObs:
    """Depth-derived facts for one track in one frame. ``None`` means not measured."""

    median_depth_m: float | None = None
    depth_valid_ratio: float = 0.0
    point_m: Point3 | None = None  # body centre in the camera frame
    elevation_m: float | None = None  # lowest body part above the calibrated floor
    zones: tuple[str, ...] = ()
    quality: tuple[str, ...] = ()  # e.g. "no_depth", "low_valid_ratio", "no_floor"


@dataclass(frozen=True)
class TrackObservation:
    """A confirmed track at one timestamp, with its spatial facts."""

    timestamp_s: float
    track_id: int
    label: str
    box_xyxy: BoxXYXY
    score: float
    spatial: SpatialObs = SpatialObs()
    frame_index: int = -1

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_xy(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the animal touches its support surface."""
        x1, _, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def height_px(self) -> float:
        return self.box_xyxy[3] - self.box_xyxy[1]


def box_iou(a: BoxXYXY, b: BoxXYXY) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_to_slices(box_xyxy: BoxXYXY, shape: tuple[int, int]) -> tuple[slice, slice]:
    """Row/column slices of ``box_xyxy`` inside an array of ``shape`` (height, width)."""
    height, width = shape
    x1 = int(np.clip(np.floor(box_xyxy[0]), 0, width))
    y1 = int(np.clip(np.floor(box_xyxy[1]), 0, height))
    x2 = int(np.clip(np.ceil(box_xyxy[2]), 0, width))
    y2 = int(np.clip(np.ceil(box_xyxy[3]), 0, height))
    return slice(y1, y2), slice(x1, x2)
