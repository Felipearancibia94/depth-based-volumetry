import json

import cv2
import numpy as np
import pytest

from perception_lab.core.config import SourceConfig
from perception_lab.core.types import FrameBundle
from perception_lab.sources import SourceError, limit, open_source, throttle
from perception_lab.sources.realsense import RealSenseSource, discover, record
from perception_lab.sources.synthetic import SyntheticSource, default_scenario
from perception_lab.sources.video import VideoSource
from perception_lab.spatial.estimator import SpatialEstimator

from . import fake_rs


def _bundles(timestamps):
    return [FrameBundle(i, t, np.zeros((2, 2, 3), np.uint8)) for i, t in enumerate(timestamps)]


def test_throttle_keeps_newest_rate():
    frames = _bundles([i / 30 for i in range(30)])
    kept = list(throttle(frames, 10))
    assert len(kept) == 10
    assert list(throttle(frames, None)) == frames
    assert len(list(limit(frames, 5))) == 5


def test_synthetic_is_deterministic_and_has_ground_truth():
    a = SyntheticSource(fps=5, duration_s=2, seed=3)
    b = SyntheticSource(fps=5, duration_s=2, seed=3)
    fa, fb = list(a.frames()), list(b.frames())
    assert len(fa) == 10
    assert np.array_equal(fa[4].depth, fb[4].depth)
    assert fa[0].has_depth and fa[0].depth.dtype == np.uint16
    labels = {g["label"] for g in fa[0].meta["ground_truth"]}
    assert labels == {"cat", "dog"}


def test_synthetic_scene_matches_rendered_geometry():
    source = SyntheticSource(fps=1, depth_noise=False)
    scene = source.scene()
    estimator = SpatialEstimator(scene)
    eating = source.render(20)  # cat at the bowl
    cat_box = next(g for g in eating.meta["ground_truth"] if g["label"] == "cat")["box_xyxy"]
    obs = estimator.measure(eating, cat_box)
    assert "food_bowl" in obs.zones
    assert obs.elevation_m == pytest.approx(0.0, abs=0.05)

    on_counter = source.render(45)
    cat_box = next(g for g in on_counter.meta["ground_truth"] if g["label"] == "cat")["box_xyxy"]
    obs = estimator.measure(on_counter, cat_box)
    assert obs.elevation_m == pytest.approx(0.9, abs=0.05)
    assert "counter" in obs.zones


def test_default_scenario_segments_are_ordered():
    for segment in default_scenario().segments:
        assert segment.t_start_s < segment.t_end_s


def test_open_source_factory():
    assert isinstance(open_source(SourceConfig()), SyntheticSource)
    assert isinstance(open_source(SourceConfig(kind="video", path="x.mp4")), VideoSource)
    assert isinstance(open_source(SourceConfig(kind="bag", path="x.bag")), RealSenseSource)
    with pytest.raises(ValueError, match="path"):
        open_source(SourceConfig(kind="video"))


def test_video_source_reads_frames(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 40, np.uint8))
    writer.release()
    frames = list(VideoSource(str(path)).frames())
    assert len(frames) == 5
    assert frames[1].timestamp_s == pytest.approx(0.1)
    assert frames[0].depth is None


def test_video_source_missing_file(tmp_path):
    with pytest.raises(SourceError):
        list(VideoSource(str(tmp_path / "missing.mp4")).frames())


def test_realsense_live_with_fake_sdk():
    fake_rs.calls.clear()
    cfg = SourceConfig(kind="realsense", serial="000111", width_px=64, height_px=48)
    source = RealSenseSource(cfg, rs_module=fake_rs)
    frames = list(limit(source.frames(), 2))
    assert len(frames) == 2
    f = frames[0]
    assert f.depth_unit_mm == pytest.approx(1.0)
    assert f.intrinsics.fx_px == 50.0
    assert f.depth_m()[0, 0] == pytest.approx(2.0)
    assert ("enable_device", "000111") in fake_rs.calls
    assert ("align",) in fake_rs.calls
    assert fake_rs.calls[-1] == ("stop",)


def test_bag_replay_stops_at_end_and_converts_rgb(tmp_path):
    bag = tmp_path / "rec.bag"
    bag.write_bytes(b"")
    fake_rs.calls.clear()
    fake_rs.pipeline.color_format = fake_rs.format.rgb8
    try:
        frames = list(
            RealSenseSource(SourceConfig(kind="bag", path=str(bag)), rs_module=fake_rs).frames()
        )
    finally:
        fake_rs.pipeline.color_format = None
    assert len(frames) == fake_rs.pipeline.frames_available
    assert tuple(frames[0].color_bgr[0, 0]) == (30, 20, 10)  # RGB -> BGR
    assert ("set_real_time", False) in fake_rs.calls
    with pytest.raises(SourceError, match="not found"):
        list(
            RealSenseSource(
                SourceConfig(kind="bag", path=str(tmp_path / "x.bag")), rs_module=fake_rs
            ).frames()
        )


def test_record_writes_sidecar(tmp_path):
    out = tmp_path / "rec" / "session.bag"
    sidecar = record(SourceConfig(kind="realsense"), out, 0.05, rs_module=fake_rs)
    assert out.is_file()
    saved = json.loads(out.with_suffix(".json").read_text())
    assert saved["device"]["serial_number"] == "000111" == sidecar["device"]["serial_number"]
    assert saved["color_intrinsics"]["fx_px"] == 50.0


def test_discover_lists_devices_and_warns_on_usb2():
    devices = discover(rs_module=fake_rs)
    assert [d["serial_number"] for d in devices] == ["000111", "222"]
    assert devices[0]["warnings"] == []
    assert "USB 2" in devices[1]["warnings"][0]
    assert devices[0]["sensors"][0]["profiles"][0]["width_px"] == 64
