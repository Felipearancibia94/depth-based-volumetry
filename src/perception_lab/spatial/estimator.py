"""Depth-derived facts for a detected animal: distance, 3D position, elevation, zones.

Depth on animals is noisy (dark or fluffy fur, motion, edges), so every value comes with
its valid-pixel ratio and quality flags. When depth is missing or unreliable the field is
``None``; nothing is extrapolated.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from perception_lab.core.config import SpatialConfig
from perception_lab.core.types import BoxXYXY, FrameBundle, SpatialObs, box_to_slices
from perception_lab.spatial.scene import SceneCalibration


def sample_depth_m(
    depth_m: NDArray,
    box_xyxy: BoxXYXY,
    *,
    mask: NDArray | None = None,
    shrink: float = 0.2,
    depth_min_m: float = 0.2,
    depth_max_m: float = 8.0,
) -> tuple[float | None, float]:
    """Median of valid depth inside the box (shrunk on each side, or under ``mask``).

    Returns ``(median_m, valid_ratio)``; the median is ``None`` when no pixel is valid.
    """
    x1, y1, x2, y2 = box_xyxy
    dx, dy = (x2 - x1) * shrink, (y2 - y1) * shrink
    rows, cols = box_to_slices((x1 + dx, y1 + dy, x2 - dx, y2 - dy), depth_m.shape)
    patch = depth_m[rows, cols]
    if mask is not None:
        patch = patch[mask[rows, cols]]
    patch = np.asarray(patch, dtype=np.float32).ravel()
    if patch.size == 0:
        return None, 0.0
    valid = patch[np.isfinite(patch) & (patch >= depth_min_m) & (patch <= depth_max_m)]
    ratio = valid.size / patch.size
    if valid.size == 0:
        return None, 0.0
    return float(np.median(valid)), float(ratio)


class SpatialEstimator:
    def __init__(self, scene: SceneCalibration, config: SpatialConfig | None = None) -> None:
        self.scene = scene
        self.config = config or SpatialConfig()

    def measure(
        self, bundle: FrameBundle, box_xyxy: BoxXYXY, mask: NDArray | None = None
    ) -> SpatialObs:
        cfg = self.config
        x1, y1, x2, y2 = box_xyxy
        foot_xy = ((x1 + x2) / 2.0, y2)
        quality: list[str] = []
        median_m, ratio, point_m, elevation_m = None, 0.0, None, None

        depth_m = bundle.depth_m() if bundle.has_depth else None
        if depth_m is None:
            quality.append("no_depth")
        else:
            intr = bundle.intrinsics
            assert intr is not None
            median_m, ratio = sample_depth_m(
                depth_m,
                box_xyxy,
                mask=mask,
                shrink=cfg.box_shrink,
                depth_min_m=cfg.depth_min_m,
                depth_max_m=cfg.depth_max_m,
            )
            if median_m is None or ratio < cfg.min_valid_ratio:
                quality.append("low_valid_ratio")
                median_m = None
            else:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                point_m = intr.deproject_m(cx, cy, median_m)
                elevation_m, from_body = self._elevation_m(
                    depth_m, box_xyxy, mask, bundle, median_m
                )
                if elevation_m is None:
                    quality.append("no_floor")
                elif from_body:
                    quality.append("elevation_from_body_depth")

        zones = tuple(z.name for z in self.scene.zones if z.contains(foot_xy, elevation_m))
        return SpatialObs(
            median_depth_m=median_m,
            depth_valid_ratio=ratio,
            point_m=point_m,
            elevation_m=elevation_m,
            zones=zones,
            quality=tuple(quality),
        )

    def _elevation_m(
        self,
        depth_m: NDArray,
        box_xyxy: BoxXYXY,
        mask: NDArray | None,
        bundle: FrameBundle,
        body_m: float,
    ) -> tuple[float | None, bool]:
        """Height of the bottom of the box above the calibrated floor.

        Uses the depth of the lowest band of the box. When that band is empty or clearly
        belongs to something else (a loose box showing the floor or wall behind the
        animal), it falls back to the body depth and says so.
        Returns ``(elevation_m, used_body_depth)``.
        """
        floor = self.scene.floor
        if floor is None or bundle.intrinsics is None:
            return None, False
        cfg = self.config
        x1, y1, x2, y2 = box_xyxy
        band_y1 = y2 - (y2 - y1) * cfg.foot_band
        foot_m, _ = sample_depth_m(
            depth_m,
            (x1, band_y1, x2, y2),
            mask=mask,
            shrink=0.0,
            depth_min_m=cfg.depth_min_m,
            depth_max_m=cfg.depth_max_m,
        )
        from_body = foot_m is None or abs(foot_m - body_m) > cfg.foot_body_max_diff_m
        depth_used = body_m if from_body else foot_m
        foot_point = bundle.intrinsics.deproject_m((x1 + x2) / 2.0, y2, depth_used)
        return floor.signed_distance_m(foot_point), from_body
