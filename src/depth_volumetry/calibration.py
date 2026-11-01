"""Camera and scene calibration, loaded from a TOML file."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CalibrationError(ValueError):
    """Raised when a calibration file or mapping is missing keys or has invalid values."""


_REQUIRED = ("focal_length_px", "camera_height_cm")
_OPTIONAL = (
    "base_height_cm",
    "depth_unit_mm",
    "distance_factor",
    "depth_percentiles",
    "reference_point",
)


@dataclass(frozen=True)
class Calibration:
    """Parameters that turn pixels and depth values into centimetres.

    Attributes:
        focal_length_px: Focal length of the colour camera, in pixels.
        camera_height_cm: Distance from the camera to the surface the box rests on.
        base_height_cm: Height of anything between that surface and the box (e.g. a pallet).
        depth_unit_mm: Millimetres represented by one unit of the depth map.
        distance_factor: Multiplier applied to the measured surface distance.
        depth_percentiles: ``(low, high)`` percentile band kept when averaging depth.
        reference_point: ``(x, y)`` pixel used to choose among several detections.
            ``None`` means the image centre.
    """

    focal_length_px: float
    camera_height_cm: float
    base_height_cm: float = 0.0
    depth_unit_mm: float = 1.0
    distance_factor: float = 1.0
    depth_percentiles: tuple[float, float] = (20.0, 80.0)
    reference_point: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _require_positive("focal_length_px", self.focal_length_px)
        _require_positive("camera_height_cm", self.camera_height_cm)
        _require_positive("depth_unit_mm", self.depth_unit_mm)
        _require_positive("distance_factor", self.distance_factor)
        if not math.isfinite(self.base_height_cm) or self.base_height_cm < 0:
            raise CalibrationError("base_height_cm must be a finite number >= 0")
        if self.base_height_cm >= self.camera_height_cm:
            raise CalibrationError("base_height_cm must be smaller than camera_height_cm")
        low, high = self.depth_percentiles
        if not (0 <= low < high <= 100):
            raise CalibrationError("depth_percentiles must satisfy 0 <= low < high <= 100")
        if self.reference_point is not None and not all(
            math.isfinite(v) for v in self.reference_point
        ):
            raise CalibrationError("reference_point must contain finite numbers")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Calibration:
        """Build a calibration from a plain mapping (for example parsed TOML)."""
        unknown = sorted(set(data) - set(_REQUIRED) - set(_OPTIONAL))
        if unknown:
            raise CalibrationError(f"unknown calibration key(s): {', '.join(unknown)}")
        missing = [key for key in _REQUIRED if key not in data]
        if missing:
            raise CalibrationError(f"missing required calibration key(s): {', '.join(missing)}")

        values: dict[str, Any] = {}
        for key in ("focal_length_px", "camera_height_cm", "base_height_cm"):
            if key in data:
                values[key] = _as_float(key, data[key])
        for key in ("depth_unit_mm", "distance_factor"):
            if key in data:
                values[key] = _as_float(key, data[key])
        if "depth_percentiles" in data:
            values["depth_percentiles"] = _as_pair("depth_percentiles", data["depth_percentiles"])
        if "reference_point" in data:
            values["reference_point"] = _as_pair("reference_point", data["reference_point"])
        return cls(**values)


def load_calibration(path: str | Path) -> Calibration:
    """Read a TOML calibration file."""
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise CalibrationError(f"calibration file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CalibrationError(f"invalid TOML in {path}: {exc}") from exc
    return Calibration.from_mapping(data)


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise CalibrationError(f"{name} must be a finite number > 0")


def _as_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CalibrationError(f"{name} must be a number")
    return float(value)


def _as_pair(name: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise CalibrationError(f"{name} must be a list of two numbers")
    return (_as_float(name, value[0]), _as_float(name, value[1]))
