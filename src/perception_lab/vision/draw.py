"""Annotated live view: zones, tracks and their spatial facts."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from perception_lab.core.types import TrackObservation
from perception_lab.spatial.scene import Zone

_TRACK_COLORS = ((0, 200, 0), (0, 160, 255), (255, 120, 0), (200, 0, 200), (0, 220, 220))
_ZONE_COLOR = (255, 255, 0)
_RESTRICTED_COLOR = (0, 0, 255)


def draw_observations(
    color_bgr: NDArray, observations: Sequence[TrackObservation], zones: Sequence[Zone] = ()
) -> NDArray[np.uint8]:
    canvas = np.ascontiguousarray(color_bgr, dtype=np.uint8).copy()
    for zone in zones:
        pts = np.round(np.asarray(zone.polygon_xy_px)).astype(np.int32)
        color = _RESTRICTED_COLOR if "restricted" in zone.tags else _ZONE_COLOR
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=1)
        cv2.putText(canvas, zone.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    for obs in observations:
        color = _TRACK_COLORS[obs.track_id % len(_TRACK_COLORS)]
        x1, y1, x2, y2 = (int(round(v)) for v in obs.box_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{obs.label} #{obs.track_id}"
        if obs.spatial.median_depth_m is not None:
            text += f" {obs.spatial.median_depth_m:.2f}m"
        if obs.spatial.elevation_m is not None:
            text += f" h{obs.spatial.elevation_m:+.2f}"
        cv2.putText(
            canvas, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3
        )
        cv2.putText(canvas, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return canvas
