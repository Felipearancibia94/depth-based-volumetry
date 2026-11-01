"""Annotated output image."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from depth_volumetry.measurement import BoxMeasurement, Footprint

_COLOR = (0, 200, 0)  # BGR


def draw_measurement(
    color_bgr: NDArray, footprint: Footprint, measurement: BoxMeasurement
) -> NDArray[np.uint8]:
    """Return a copy of ``color_bgr`` with the oriented footprint and the dimensions drawn."""
    canvas = np.ascontiguousarray(color_bgr, dtype=np.uint8).copy()
    corners = np.round(footprint.corners_xy).astype(np.int32)
    cv2.polylines(canvas, [corners], isClosed=True, color=_COLOR, thickness=2)

    lines = (
        f"L {measurement.length_cm:.1f} cm",
        f"W {measurement.width_cm:.1f} cm",
        f"H {measurement.height_cm:.1f} cm",
    )
    x, y = 10, 25
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLOR, 2, cv2.LINE_AA)
        y += 28
    return canvas
