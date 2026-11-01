"""Floor-plane calibration from depth frames of the empty scene."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from perception_lab.core.types import Intrinsics
from perception_lab.spatial.geometry import (
    Plane,
    SpatialError,
    depth_to_points_m,
    fit_plane_ransac,
)


def median_depth_m(frames_m: Sequence[NDArray]) -> NDArray[np.float32]:
    """Per-pixel median over frames, ignoring zero/invalid samples (holes stay 0)."""
    if not frames_m:
        raise SpatialError("no depth frames given")
    stack = np.stack([np.where(f > 0, f, np.nan) for f in frames_m]).astype(np.float32)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
            median = np.nanmedian(stack, axis=0)
    return np.nan_to_num(median, nan=0.0).astype(np.float32)


def fit_floor_plane(
    frames_m: Sequence[NDArray],
    intrinsics: Intrinsics,
    *,
    roi_top_fraction: float = 0.5,
    threshold_m: float = 0.02,
    max_tilt_deg: float = 60.0,
    seed: int = 0,
) -> tuple[Plane, float]:
    """Fit the floor in the lower part of the image (rows below ``roi_top_fraction``).

    Use frames where the floor is visible and mostly free. The tilt limit rejects walls.
    Returns the plane (normal pointing up) and its inlier ratio.
    """
    if not 0.0 <= roi_top_fraction < 1.0:
        raise ValueError("roi_top_fraction must be in [0, 1)")
    depth = median_depth_m(frames_m)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[int(depth.shape[0] * roi_top_fraction) :, :] = True
    points = depth_to_points_m(depth, intrinsics, mask=mask)
    return fit_plane_ransac(points, threshold_m=threshold_m, max_tilt_deg=max_tilt_deg, seed=seed)
