import cv2
import numpy as np
import pytest

from depth_volumetry.calibration import Calibration
from depth_volumetry.measurement import (
    MeasurementError,
    footprint_px,
    measure_box,
    surface_distance_cm,
)

SHAPE = (480, 640)


def rotated_rect_mask(center=(320, 240), size=(200, 100), angle=30.0, shape=SHAPE):
    corners = cv2.boxPoints((center, size, angle)).astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [corners], 1)
    return mask


def scene(box_top_mm=1500.0, floor_mm=2000.0, **mask_kwargs):
    mask = rotated_rect_mask(**mask_kwargs)
    depth = np.full(SHAPE, floor_mm, dtype=np.uint16)
    depth[mask > 0] = box_top_mm
    return depth, mask


CALIBRATION = Calibration(focal_length_px=500.0, camera_height_cm=200.0)


def test_footprint_recovers_rotated_rectangle():
    footprint = footprint_px(rotated_rect_mask())
    assert footprint.length_px == pytest.approx(200, rel=0.01)
    assert footprint.width_px == pytest.approx(100, rel=0.02)
    assert footprint.center_xy == pytest.approx((320, 240), abs=1.5)
    assert footprint.corners_xy.shape == (4, 2)


def test_footprint_length_is_the_longer_side_whatever_the_angle():
    for angle in (0.0, 30.0, 60.0, 90.0, 120.0):
        footprint = footprint_px(rotated_rect_mask(angle=angle))
        assert footprint.length_px > footprint.width_px


def test_footprint_of_empty_mask():
    with pytest.raises(MeasurementError, match="no object"):
        footprint_px(np.zeros(SHAPE, dtype=np.uint8))


def test_footprint_rejects_non_2d_mask():
    with pytest.raises(ValueError, match="2-D"):
        footprint_px(np.zeros((4, 4, 3), dtype=np.uint8))


def test_measure_box_on_synthetic_scene():
    depth, mask = scene()
    result = measure_box(depth, mask, CALIBRATION)
    assert result.length_cm == pytest.approx(60.0, rel=0.01)
    assert result.width_cm == pytest.approx(30.0, rel=0.02)
    assert result.height_cm == pytest.approx(50.0, rel=0.01)
    assert result.surface_distance_cm == pytest.approx(150.0)
    assert result.volume_cm3 == pytest.approx(result.length_cm * result.width_cm * 50.0)
    assert result.to_dict()["volume_cm3"] == pytest.approx(result.volume_cm3)


def test_base_height_reduces_box_height():
    depth, mask = scene(box_top_mm=1500.0)
    calibration = Calibration(focal_length_px=500.0, camera_height_cm=200.0, base_height_cm=15.0)
    assert measure_box(depth, mask, calibration).height_cm == pytest.approx(35.0)


def test_depth_unit_converts_float_metres():
    depth, mask = scene()
    metres = depth.astype(np.float32) / 1000.0
    calibration = Calibration(focal_length_px=500.0, camera_height_cm=200.0, depth_unit_mm=1000.0)
    assert measure_box(metres, mask, calibration).height_cm == pytest.approx(50.0)


def test_distance_factor_scales_distance():
    depth, mask = scene()
    assert surface_distance_cm(depth, mask, distance_factor=1.02) == pytest.approx(153.0)


def test_percentile_band_rejects_outliers():
    mask = np.ones((10, 10), dtype=np.uint8)
    depth = np.full((10, 10), 1000, dtype=np.uint16)
    depth.flat[:5] = 5000  # far outliers
    depth.flat[5:10] = 100  # near outliers
    assert surface_distance_cm(depth, mask) == pytest.approx(100.0)
    assert surface_distance_cm(depth, mask, percentiles=(0, 100)) != pytest.approx(100.0)


def test_zero_and_nan_depth_are_ignored():
    mask = np.ones((4, 4), dtype=np.uint8)
    depth = np.full((4, 4), 1000.0, dtype=np.float32)
    depth[0, :] = 0.0
    depth[1, :] = np.nan
    assert surface_distance_cm(depth, mask) == pytest.approx(100.0)


def test_no_valid_depth():
    mask = np.ones((4, 4), dtype=np.uint8)
    with pytest.raises(MeasurementError, match="no valid depth"):
        surface_distance_cm(np.zeros((4, 4), dtype=np.uint16), mask)


def test_two_samples_fall_back_to_median():
    mask = np.ones((1, 2), dtype=np.uint8)
    depth = np.array([[100, 1000]], dtype=np.uint16)
    assert surface_distance_cm(depth, mask) == pytest.approx(55.0)


def test_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        surface_distance_cm(np.zeros((4, 4)), np.ones((5, 5), dtype=np.uint8))


def test_non_positive_height_is_an_error():
    depth, mask = scene(box_top_mm=2100.0)  # surface farther than the camera height
    with pytest.raises(MeasurementError, match="non-positive height"):
        measure_box(depth, mask, CALIBRATION)


def test_empty_mask_in_measure_box():
    depth, _ = scene()
    with pytest.raises(MeasurementError, match="no object"):
        measure_box(depth, np.zeros(SHAPE, dtype=np.uint8), CALIBRATION)
