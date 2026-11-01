"""Box dimensions from a segmentation mask and an aligned depth map.

Everything in this module is a pure function of its inputs: no files, no models.

Conventions: points are ``(x, y)``; arrays are indexed ``[row, col]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from depth_volumetry.calibration import Calibration


class MeasurementError(RuntimeError):
    """Raised when a box cannot be measured (no object, no valid depth, impossible height)."""


@dataclass(frozen=True)
class Footprint:
    """Oriented bounding rectangle of a mask, in pixels."""

    center_xy: tuple[float, float]
    length_px: float
    width_px: float
    corners_xy: NDArray[np.float32]  # shape (4, 2)


@dataclass(frozen=True)
class BoxMeasurement:
    """Measured box dimensions, in centimetres."""

    length_cm: float
    width_cm: float
    height_cm: float
    surface_distance_cm: float

    @property
    def volume_cm3(self) -> float:
        return self.length_cm * self.width_cm * self.height_cm

    def to_dict(self, decimals: int = 2) -> dict[str, float]:
        """Plain dictionary of the results, rounded to ``decimals`` places."""
        values = {**asdict(self), "volume_cm3": self.volume_cm3}
        return {key: round(float(value), decimals) for key, value in values.items()}


def footprint_px(mask: NDArray) -> Footprint:
    """Oriented rectangle enclosing the non-zero pixels of ``mask``.

    ``length_px`` is the longer side of the rectangle.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    points = cv2.findNonZero(np.ascontiguousarray(mask != 0, dtype=np.uint8))
    if points is None:
        raise MeasurementError("no object: the mask is empty")
    rect = cv2.minAreaRect(points)
    (cx, cy), (side_a, side_b), _angle = rect
    corners = cv2.boxPoints(rect)
    return Footprint(
        center_xy=(float(cx), float(cy)),
        length_px=float(max(side_a, side_b)),
        width_px=float(min(side_a, side_b)),
        corners_xy=corners.astype(np.float32),
    )


def surface_distance_cm(
    depth: NDArray,
    mask: NDArray,
    *,
    depth_unit_mm: float = 1.0,
    percentiles: tuple[float, float] = (20.0, 80.0),
    distance_factor: float = 1.0,
) -> float:
    """Distance from the camera to the top of the box, in centimetres.

    Takes the depth samples under ``mask``, drops non-finite and non-positive values,
    keeps those inside the percentile band, averages them and applies ``distance_factor``.
    """
    if depth.shape != mask.shape:
        raise ValueError(f"depth {depth.shape} and mask {mask.shape} must have the same shape")
    samples = depth[mask != 0].astype(np.float64) * (depth_unit_mm / 10.0)
    samples = samples[np.isfinite(samples) & (samples > 0)]
    if samples.size == 0:
        raise MeasurementError("no valid depth: every sample under the mask is zero or invalid")
    low, high = np.percentile(samples, percentiles)
    band = samples[(samples >= low) & (samples <= high)]
    if band.size == 0:
        # Very few samples can leave the interpolated band empty; fall back to the median.
        return float(np.median(samples)) * distance_factor
    return float(band.mean()) * distance_factor


def measure_box(depth: NDArray, mask: NDArray, calibration: Calibration) -> BoxMeasurement:
    """Measure the box covered by ``mask`` using the pinhole model.

    ``length = length_px * d / f``, ``width = width_px * d / f`` and
    ``height = camera_height - base_height - d``, where ``d`` is the surface distance.
    """
    footprint = footprint_px(mask)
    distance = surface_distance_cm(
        depth,
        mask,
        depth_unit_mm=calibration.depth_unit_mm,
        percentiles=calibration.depth_percentiles,
        distance_factor=calibration.distance_factor,
    )
    scale = distance / calibration.focal_length_px
    height = calibration.camera_height_cm - calibration.base_height_cm - distance
    if height <= 0:
        raise MeasurementError(
            f"non-positive height ({height:.1f} cm): the surface distance ({distance:.1f} cm) "
            "is not smaller than camera_height_cm - base_height_cm; check the calibration"
        )
    return BoxMeasurement(
        length_cm=footprint.length_px * scale,
        width_cm=footprint.width_px * scale,
        height_cm=height,
        surface_distance_cm=distance,
    )
