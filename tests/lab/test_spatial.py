import math

import numpy as np
import pytest

from perception_lab.core.config import ConfigError, SpatialConfig
from perception_lab.core.types import FrameBundle, Intrinsics
from perception_lab.spatial.calibrate import fit_floor_plane, median_depth_m
from perception_lab.spatial.estimator import SpatialEstimator, sample_depth_m
from perception_lab.spatial.geometry import Plane, SpatialError, depth_to_points_m, fit_plane_ransac
from perception_lab.spatial.scene import (
    SceneCalibration,
    Zone,
    dump_scene,
    load_scene,
    point_in_polygon,
    scene_from_mapping,
)

K = Intrinsics(640, 480, 525.0, 525.0, 320.0, 240.0)
HEIGHT_M, PITCH_DEG = 1.2, 35.0


def floor_depth_m(height_m=HEIGHT_M, pitch_deg=PITCH_DEG):
    """Depth of a flat floor seen by a camera ``height_m`` above it, pitched down."""
    t = math.radians(pitch_deg)
    down = np.array([0.0, math.cos(t), math.sin(t)])
    ys, xs = np.mgrid[0:480, 0:640].astype(np.float64)
    rays = np.stack([(xs - K.cx_px) / K.fx_px, (ys - K.cy_px) / K.fy_px, np.ones_like(xs)], -1)
    along = rays @ down
    with np.errstate(divide="ignore"):
        z = np.where(along > 1e-3, height_m / along, 0.0)
    return np.where(z < 8.0, z, 0.0).astype(np.float32)


def expected_floor():
    t = math.radians(PITCH_DEG)
    return Plane((0.0, -math.cos(t), -math.sin(t)), HEIGHT_M)


def test_plane_orientation_and_distance():
    plane = Plane.from_normal_point(np.array([0, 1, 0]), np.array([0, 1.5, 0]))
    assert plane.normal == pytest.approx((0, -1, 0))  # flipped: camera on positive side
    assert plane.offset_m == pytest.approx(1.5)
    assert plane.signed_distance_m((0, 1.0, 2.0)) == pytest.approx(0.5)
    assert plane.tilt_from_up_deg() == pytest.approx(0.0)


def test_ransac_recovers_floor_with_outliers():
    depth = floor_depth_m()
    points = depth_to_points_m(depth, K)
    rng = np.random.default_rng(1)
    outliers = rng.uniform([-2, -1, 0.5], [2, 1, 5], size=(len(points) // 5, 3))
    plane, ratio = fit_plane_ransac(np.vstack([points, outliers]), threshold_m=0.01)
    ref = expected_floor()
    assert plane.normal == pytest.approx(ref.normal, abs=1e-3)
    assert plane.offset_m == pytest.approx(ref.offset_m, abs=1e-3)
    assert ratio > 0.75


def test_ransac_rejects_wall_with_tilt_limit():
    ys, xs = np.mgrid[0:100, 0:100].astype(np.float64)
    wall = np.stack([xs / 50 - 1, ys / 50 - 1, np.full_like(xs, 3.0)], -1).reshape(-1, 3)
    with pytest.raises(SpatialError):
        fit_plane_ransac(wall, max_tilt_deg=45)
    with pytest.raises(SpatialError, match="not enough"):
        fit_plane_ransac(wall[:10])


def test_fit_floor_plane_from_noisy_frames():
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(5):
        f = floor_depth_m() + rng.normal(0, 0.005, (480, 640)).astype(np.float32)
        f[rng.random((480, 640)) < 0.1] = 0.0
        frames.append(f)
    plane, ratio = fit_floor_plane(frames, K)
    assert plane.offset_m == pytest.approx(HEIGHT_M, abs=0.02)
    assert plane.tilt_from_up_deg() == pytest.approx(PITCH_DEG, abs=1.0)
    assert median_depth_m(frames).min() >= 0.0


def test_sample_depth_ignores_invalid_and_reports_ratio():
    depth = np.full((100, 100), 2.0, np.float32)
    depth[40:60, 40:50] = 0.0  # holes inside the box
    median, ratio = sample_depth_m(depth, (30, 30, 70, 70), shrink=0.0)
    assert median == pytest.approx(2.0)
    assert ratio == pytest.approx(1 - 200 / 1600)
    assert sample_depth_m(np.zeros((10, 10), np.float32), (0, 0, 10, 10)) == (None, 0.0)


def test_point_in_polygon_and_zone_elevation():
    square = ((0, 0), (10, 0), (10, 10), (0, 10))
    assert point_in_polygon((5, 5), square)
    assert not point_in_polygon((15, 5), square)
    counter = Zone("counter", square, ("restricted",), min_elevation_m=0.6)
    assert counter.contains((5, 5), 0.9)
    assert not counter.contains((5, 5), 0.1)
    assert not counter.contains((5, 5), None)  # elevation unknown: not claimed


def test_scene_toml_roundtrip(tmp_path):
    scene = SceneCalibration(
        "kitchen",
        expected_floor(),
        (
            Zone("bowl", ((1, 2), (3, 2), (3, 4)), ("food",)),
            Zone("counter", ((0, 0), (5, 0), (5, 5)), ("restricted",), 0.6, 1.5),
        ),
    )
    path = tmp_path / "scene.toml"
    path.write_text(dump_scene(scene))
    loaded = load_scene(path)
    assert loaded.camera_id == "kitchen"
    assert loaded.zones == scene.zones
    assert loaded.floor.offset_m == pytest.approx(HEIGHT_M)
    assert loaded.zones_with_tag("food")[0].name == "bowl"


@pytest.mark.parametrize(
    "data",
    [
        {"bogus": 1},
        {"zones": [{"name": "a", "polygon_xy_px": [[0, 0], [1, 1]]}]},
        {"zones": [{"name": "a", "polygon_xy_px": [[0, 0], [1, 0], [1, 1]], "colour": 1}]},
        {"floor": {"normal": [0, 1]}},
        {
            "zones": [
                {"name": "a", "polygon_xy_px": [[0, 0], [1, 0], [1, 1]]},
                {"name": "a", "polygon_xy_px": [[0, 0], [1, 0], [1, 1]]},
            ]
        },
    ],
)
def test_invalid_scene_is_rejected(data):
    with pytest.raises(ConfigError):
        scene_from_mapping(data)


def _bundle_with_animal(elevation_m: float):
    """Floor plus a flat 'animal' patch whose bottom rests ``elevation_m`` above the floor."""
    depth_m = floor_depth_m()
    floor = expected_floor()
    t = math.radians(PITCH_DEG)
    down = np.array([0.0, math.cos(t), math.sin(t)])
    forward = np.array([0.0, -math.sin(t), math.cos(t)])
    # Elevated points are close to camera height, so they are only in view when near.
    forward_m, body_m = (2.0, 0.25) if elevation_m == 0 else (0.7, 0.15)
    foot = (HEIGHT_M - elevation_m) * down + forward_m * forward
    top = foot - body_m * down
    (u, v_foot), (_, v_top) = K.project_px(tuple(foot)), K.project_px(tuple(top))
    box = (u - 40, v_top, u + 40, v_foot)
    assert 0 <= v_top < v_foot <= 480, "test animal must be in view"
    rows = np.arange(int(v_top), int(np.ceil(v_foot)))
    frac = (rows - v_top) / (v_foot - v_top)
    z = top[2] + frac * (foot[2] - top[2])
    depth_m[int(v_top) : int(np.ceil(v_foot)), int(u - 40) : int(u + 40)] = z[:, None]
    depth = np.round(depth_m * 1000).astype(np.uint16)
    bundle = FrameBundle(0, 0.0, np.zeros((480, 640, 3), np.uint8), depth, 1.0, K)
    return bundle, box, floor


@pytest.mark.parametrize("elevation_m", [0.0, 0.9])
def test_estimator_measures_distance_and_elevation(elevation_m):
    bundle, box, floor = _bundle_with_animal(elevation_m)
    zone = Zone("all", ((0, 0), (640, 0), (640, 480), (0, 480)), ("restricted",), 0.5)
    obs = SpatialEstimator(SceneCalibration(floor=floor, zones=(zone,))).measure(bundle, box)
    assert obs.quality == ()
    assert obs.depth_valid_ratio == pytest.approx(1.0)
    assert obs.elevation_m == pytest.approx(elevation_m, abs=0.03)
    assert ("all" in obs.zones) == (elevation_m > 0.5)
    assert obs.point_m[2] == pytest.approx(obs.median_depth_m)


def test_estimator_without_depth_or_floor():
    bundle, box, _ = _bundle_with_animal(0.0)
    zone = Zone("all", ((0, 0), (640, 0), (640, 480), (0, 480)))
    no_floor = SpatialEstimator(SceneCalibration(zones=(zone,))).measure(bundle, box)
    assert no_floor.elevation_m is None and "no_floor" in no_floor.quality
    assert no_floor.zones == ("all",)  # image zone still works

    rgb_only = FrameBundle(0, 0.0, bundle.color_bgr)
    obs = SpatialEstimator(SceneCalibration(zones=(zone,))).measure(rgb_only, box)
    assert obs.quality == ("no_depth",) and obs.median_depth_m is None
    assert obs.zones == ("all",)


def test_estimator_flags_low_valid_ratio():
    bundle, box, floor = _bundle_with_animal(0.0)
    bundle.depth[:] = 0
    obs = SpatialEstimator(SceneCalibration(floor=floor), SpatialConfig()).measure(bundle, box)
    assert obs.median_depth_m is None and obs.point_m is None
    assert "low_valid_ratio" in obs.quality


def test_elevation_falls_back_to_body_depth_when_feet_show_background():
    bundle, box, floor = _bundle_with_animal(0.9)
    x1, y1, x2, y2 = box
    loose = (x1, y1, x2, y2 + 20)  # box extends well below the animal onto far background
    obs = SpatialEstimator(SceneCalibration(floor=floor)).measure(bundle, loose)
    assert "elevation_from_body_depth" in obs.quality
    assert obs.elevation_m == pytest.approx(0.9, abs=0.1)
