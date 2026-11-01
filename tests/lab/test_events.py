import numpy as np
import pytest

from perception_lab.core.config import RulesConfig
from perception_lab.core.types import FrameBundle, SpatialObs, TrackObservation
from perception_lab.events.engine import EventEngine
from perception_lab.events.evidence import FrameRingBuffer, crop, montage, select_frames
from perception_lab.events.features import (
    TrackHistory,
    condition_duration_s,
    speed,
    stationary_duration_s,
)
from perception_lab.events.model import PetEvent
from perception_lab.events.rules import (
    ElevatedRule,
    ProximityRule,
    SpeedBurstRule,
    StationaryRule,
    ZoneRule,
    default_rules,
)
from perception_lab.spatial.scene import SceneCalibration, Zone

SCENE = SceneCalibration(
    zones=(
        Zone("bowl", ((0, 0), (100, 0), (100, 100), (0, 100)), ("food",)),
        Zone("door", ((200, 0), (300, 0), (300, 100), (200, 100)), ("restricted",)),
        Zone("sofa", ((400, 0), (500, 0), (500, 100), (400, 100)), ("elevated",)),
    )
)


def obs(t, tid=1, x_m=0.0, zones=(), elevation=None, label="cat", depth=True, x_px=None):
    point = (x_m, 0.0, 2.0) if depth else None
    x = x_px if x_px is not None else 50 + x_m * 100
    return TrackObservation(
        t,
        tid,
        label,
        (x - 20, 40, x + 20, 80),
        0.9,
        SpatialObs(2.0 if depth else None, 0.9, point, elevation, tuple(zones)),
    )


def history(observations, tid=1):
    h = TrackHistory(tid)
    for o in observations:
        h.add(o)
    return h


def test_speed_uses_metres_or_body_lengths():
    h = history([obs(t / 10, x_m=t / 10) for t in range(11)])  # 1 m/s
    v = speed(h, 1.0)
    assert v.unit == "m/s" and v.value == pytest.approx(1.0)
    h2 = history(
        [obs(t / 10, depth=False, x_px=50 + 8 * t) for t in range(11)]
    )  # 80 px/s, 40 px body
    v2 = speed(h2, 1.0)
    assert v2.unit == "body_lengths/s" and v2.value == pytest.approx(2.0)
    assert speed(history([obs(0.0)]), 1.0) is None


def test_condition_duration_breaks_on_gap_and_false():
    h = history([obs(t, zones=("bowl",)) for t in range(10)])
    assert condition_duration_s(h, lambda o: "bowl" in o.spatial.zones) == 9
    h.add(obs(15, zones=("bowl",)))  # 6 s gap
    assert condition_duration_s(h, lambda o: "bowl" in o.spatial.zones) == 0
    h.add(obs(16))
    assert condition_duration_s(h, lambda o: "bowl" in o.spatial.zones) == 0


def test_stationary_duration():
    walk = [obs(t, x_m=t * 0.5) for t in range(5)]
    still = [obs(5 + t, x_m=2.5 + 0.01 * (t % 2)) for t in range(10)]
    assert stationary_duration_s(history(walk + still), 0.15, 0.3) == 9


def _run(rule, timeline, tracks=1):
    histories = {}
    hits = []
    for frame in timeline:
        for o in frame:
            histories.setdefault(o.track_id, TrackHistory(o.track_id)).add(o)
        hits = rule.evaluate(frame[0].timestamp_s, frame, histories)
    return hits


def test_zone_dwell_and_entry_rules():
    dwell = ZoneRule("zone_dwell", SCENE, {"food": 15.0}, {"food": "eating"})
    (hit,) = _run(dwell, [[obs(t, zones=("bowl",))] for t in range(20)])
    assert (hit.hypothesis, hit.zone, hit.started_s) == ("eating", "bowl", 0)
    assert hit.facts["duration_s"] == 19
    assert _run(dwell, [[obs(t, zones=("bowl",))] for t in range(10)]) == []
    entry = ZoneRule("zone_entry", SCENE, {"restricted": 0.5}, {"restricted": "entered"})
    assert _run(entry, [[obs(t / 10, zones=("door",))] for t in range(6)])[0].zone == "door"


def test_elevated_rule_from_depth_or_zone():
    rule = ElevatedRule(SCENE, 0.5, 2.0)
    assert (
        _run(rule, [[obs(t, elevation=0.9)] for t in range(4)])[0].hypothesis
        == "on_elevated_surface"
    )
    assert _run(rule, [[obs(t, zones=("sofa",), depth=False)] for t in range(4)])
    assert _run(rule, [[obs(t, elevation=0.1)] for t in range(4)]) == []


def test_speed_burst_and_stationary_rules():
    run = SpeedBurstRule(1.0, 1.2, 2.5)
    assert (
        _run(run, [[obs(t / 10, x_m=t * 0.15)] for t in range(15)])[0].facts["speed_unit"] == "m/s"
    )
    assert _run(run, [[obs(t / 10, x_m=t * 0.05)] for t in range(15)]) == []
    still = StationaryRule("stationary", "resting_or_sleeping", 30, 0.15, 0.3)
    assert _run(still, [[obs(t)] for t in range(31)])[0].facts["duration_s"] == 30
    in_bowl = StationaryRule("stationary", "resting", 30, 0.15, 0.3, SCENE, ("food",))
    assert _run(in_bowl, [[obs(t, zones=("bowl",))] for t in range(31)]) == []


def test_proximity_rule_pairs_and_resets():
    rule = ProximityRule(0.5, 5.0)
    near = [[obs(t, 1, 0.0), obs(t, 2, 0.3, label="dog")] for t in range(7)]
    (hit,) = _run(rule, near)
    assert hit.track_ids == (1, 2) and hit.facts["classes"] == ["cat", "dog"]
    far = [[obs(t, 1, 0.0), obs(t, 2, 2.0, label="dog")] for t in range(7)]
    assert _run(rule, far) == []


def test_engine_starts_once_ends_after_release_and_respects_cooldown():
    cfg = RulesConfig(cooldown_s=100.0, release_s=3.0)
    engine = EventEngine(
        [ZoneRule("zone_entry", SCENE, {"restricted": 0.0}, {"restricted": "x"})], cfg
    )
    histories = {1: TrackHistory(1)}

    def step(t, zones):
        o = obs(t, zones=zones)
        histories[1].add(o)
        return engine.update(t, [o], histories)

    started = [len(step(t, ("door",)).started) for t in range(5)]
    assert started == [1, 0, 0, 0, 0]
    outs = [step(t, ()) for t in range(5, 10)]
    ended = [e for o in outs for e in o.ended]
    assert ended == [(("zone_entry", (1,), "door"), 4)]
    assert step(10, ("door",)).started == []  # within cooldown
    assert engine.flush() == []  # cooldown suppressed it, nothing open


def test_default_rules_cover_mvp():
    names = {r.name for r in default_rules(RulesConfig(), SCENE)}
    assert names == {
        "zone_entry",
        "zone_dwell",
        "elevated_surface",
        "stationary",
        "long_inactivity",
        "speed_burst",
        "proximity",
    }


def test_event_roundtrip_and_summary():
    event = PetEvent(
        "e1", "cam", "zone_dwell", "eating", (3,), ("cat",), 5.0, 35.0, "2026-01-01T00:00:00",
        "bowl", {"duration_s": 30.0},
    )  # fmt: skip
    assert PetEvent.from_dict(event.to_dict()) == event
    assert event.activity is None and event.confidence is None
    assert event.summary() == "cat #3 at bowl (possible eating) for 30 s"
    event.interpretation, event.interpretation_status = {"activity": "unknown"}, "done"
    assert event.activity == "unknown"


def _frame(t, index):
    image = np.random.default_rng(index).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    return FrameBundle(index, t, image)


def test_ring_buffer_rate_retention_and_selection():
    buf = FrameRingBuffer(seconds=5.0, fps=2.0, max_side_px=320)
    added = [buf.maybe_add(_frame(t / 10, t), [obs(t / 10)]) for t in range(100)]
    assert sum(added) == 20
    # Last frame kept is at 9.5 s (2 fps), so everything older than 4.5 s is dropped.
    assert [f.timestamp_s for f in buf.frames][0] == pytest.approx(4.5)
    assert len(buf.frames) == 11
    assert buf.frames[0].image.shape == (240, 320, 3)
    chosen = select_frames(buf, [1], 6.0, 9.9, 3)
    assert len(chosen) == 3
    assert all(6.0 <= f.timestamp_s <= 9.9 for f in chosen)
    assert select_frames(buf, [99], 0, 10, 3) == []
    c = crop(chosen[0], [1])
    assert c.shape[0] < 240 and c.shape[1] < 320


def test_montage_layout():
    tiles = [np.full((100, 200, 3), i * 50, np.uint8) for i in range(3)]
    grid = montage(tiles, tile_px=160, columns=2)
    assert grid.shape == (320, 320, 3)
    with pytest.raises(ValueError):
        montage([])
