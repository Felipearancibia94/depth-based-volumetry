"""Pure 3D geometry on aligned depth maps: back-projection and plane fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from perception_lab.core.types import Intrinsics, Point3


class SpatialError(ValueError):
    """Raised when a spatial quantity cannot be computed from the available data."""


@dataclass(frozen=True)
class Plane:
    """Plane ``n . p + offset_m = 0`` with unit normal ``n`` in the camera frame.

    The normal is oriented so that the camera (origin) lies on the positive side; for a
    floor this means the normal points up and :meth:`signed_distance_m` is the height
    above the floor.
    """

    normal: Point3
    offset_m: float

    def signed_distance_m(self, point_m: Point3) -> float:
        n = self.normal
        return n[0] * point_m[0] + n[1] * point_m[1] + n[2] * point_m[2] + self.offset_m

    def signed_distances_m(self, points_m: NDArray) -> NDArray:
        return points_m @ np.asarray(self.normal) + self.offset_m

    def tilt_from_up_deg(self, up: Point3 = (0.0, -1.0, 0.0)) -> float:
        """Angle between the normal and ``up`` (camera -Y by default)."""
        cos = float(np.clip(np.dot(self.normal, up), -1.0, 1.0))
        return math.degrees(math.acos(cos))

    @classmethod
    def from_normal_point(cls, normal: NDArray, point: NDArray) -> Plane:
        n = np.asarray(normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        offset = -float(n @ np.asarray(point, dtype=np.float64))
        if offset < 0:  # orient so the camera origin is on the positive side
            n, offset = -n, -offset
        return cls((float(n[0]), float(n[1]), float(n[2])), offset)


def depth_to_points_m(
    depth_m: NDArray,
    intrinsics: Intrinsics,
    *,
    stride_px: int = 4,
    mask: NDArray | None = None,
    depth_min_m: float = 0.1,
    depth_max_m: float = 10.0,
) -> NDArray[np.float64]:
    """Back-project valid depth pixels (every ``stride_px``) to an ``(N, 3)`` point array."""
    if depth_m.ndim != 2:
        raise ValueError(f"depth must be 2-D, got shape {depth_m.shape}")
    rows, cols = np.mgrid[0 : depth_m.shape[0] : stride_px, 0 : depth_m.shape[1] : stride_px]
    z = depth_m[rows, cols].astype(np.float64)
    valid = np.isfinite(z) & (z >= depth_min_m) & (z <= depth_max_m)
    if mask is not None:
        valid &= mask[rows, cols].astype(bool)
    z, xs, ys = z[valid], cols[valid].astype(np.float64), rows[valid].astype(np.float64)
    x = (xs - intrinsics.cx_px) * z / intrinsics.fx_px
    y = (ys - intrinsics.cy_px) * z / intrinsics.fy_px
    return np.stack([x, y, z], axis=1)


def fit_plane_ransac(
    points_m: NDArray,
    *,
    threshold_m: float = 0.02,
    iterations: int = 300,
    seed: int = 0,
    min_points: int = 100,
    max_tilt_deg: float | None = None,
) -> tuple[Plane, float]:
    """Fit the dominant plane with RANSAC, then refine it on the inliers by least squares.

    Args:
        points_m: ``(N, 3)`` points in metres.
        threshold_m: Inlier distance.
        max_tilt_deg: If set, reject candidate planes whose normal is further than this
            from camera "up" (-Y), so a wall is not mistaken for the floor.

    Returns:
        The plane and the inlier ratio.
    """
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if len(points) < max(3, min_points):
        raise SpatialError(f"not enough valid depth points to fit a plane ({len(points)})")

    rng = np.random.default_rng(seed)
    best_count, best_plane = 0, None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        plane = Plane.from_normal_point(normal / norm, sample[0])
        if max_tilt_deg is not None and plane.tilt_from_up_deg() > max_tilt_deg:
            continue
        count = int((np.abs(plane.signed_distances_m(points)) < threshold_m).sum())
        if count > best_count:
            best_count, best_plane = count, plane
    if best_plane is None:
        raise SpatialError("no plane satisfies the constraints; check the region and tilt limit")

    inliers = points[np.abs(best_plane.signed_distances_m(points)) < threshold_m]
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    refined = Plane.from_normal_point(vt[-1], centroid)
    if max_tilt_deg is not None and refined.tilt_from_up_deg() > max_tilt_deg:
        refined = best_plane
    return refined, len(inliers) / len(points)
