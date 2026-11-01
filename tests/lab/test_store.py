import time

import numpy as np
import pytest

from perception_lab.events.model import PetEvent
from perception_lab.store.db import EventStore
from perception_lab.store.media import resolve_media_path, save_images
from perception_lab.store.retention import purge


def _event(event_id="e1", rule="zone_dwell", hypothesis="eating", zone="bowl"):
    return PetEvent(
        event_id, "cam", rule, hypothesis, (3,), ("cat",), 5.0, 35.0, "2026-01-01T00:00:00", zone,
        {"duration_s": 30.0},
    )  # fmt: skip


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "lab.db")
    yield s
    s.close()


def test_add_get_update_and_list(store):
    store.add(_event("e1"), created_unix=100.0)
    store.add(_event("e2", "speed_burst", "running", None), created_unix=200.0)
    assert store.count() == 2
    assert [e.event_id for e in store.list()] == ["e2", "e1"]
    assert [e.event_id for e in store.list(rule="zone_dwell")] == ["e1"]
    event = store.get("e1")
    event.interpretation = {"activity": "eating", "description": "Head lowered into the bowl."}
    event.interpretation_status = "done"
    store.update(event)
    assert store.get("e1").activity == "eating"
    assert store.get("missing") is None
    with pytest.raises(KeyError):
        store.update(_event("missing"))


def test_search_finds_interpretations_and_survives_odd_queries(store):
    store.add(_event("e1"))
    event = store.get("e1")
    event.interpretation = {"activity": "eating", "description": "The cat lowers its head"}
    event.interpretation_status = "done"
    store.update(event)
    store.add(_event("e2", "speed_burst", "running", None))
    assert [e.event_id for e in store.search("lowers head")] == ["e1"]
    assert [e.event_id for e in store.search("running")] == ["e2"]
    assert store.search('"unbalanced AND (') == []
    assert store.search("   ") == []


def test_review(store):
    store.add(_event())
    assert store.set_review("e1", "rejected", "sniffing").review_label == "sniffing"
    assert store.get("e1").review_status == "rejected"
    with pytest.raises(ValueError):
        store.set_review("e1", "maybe")


def test_media_paths_are_confined(tmp_path):
    paths = save_images(tmp_path, "e1", [("f0", np.zeros((8, 8, 3), np.uint8))], created_unix=0)
    assert paths[0].endswith("e1/f0.jpg")
    assert resolve_media_path(tmp_path, paths[0]) is not None
    assert resolve_media_path(tmp_path, "../etc/passwd") is None
    assert resolve_media_path(tmp_path, "missing.jpg") is None


def test_purge_media_and_events(store, tmp_path):
    media = tmp_path / "media"
    now = time.mktime((2026, 3, 30, 12, 0, 0, 0, 0, -1))
    old_day = time.mktime((2026, 3, 1, 12, 0, 0, 0, 0, -1))
    save_images(media, "old", [("f", np.zeros((4, 4, 3), np.uint8))], created_unix=old_day)
    save_images(media, "new", [("f", np.zeros((4, 4, 3), np.uint8))], created_unix=now)
    (media / "not-a-day").mkdir()
    store.add(_event("old"), created_unix=now - 400 * 86400)
    store.add(_event("new"), created_unix=now)

    dry = purge(
        store, media, media_retention_days=14, event_retention_days=365, now_unix=now, dry_run=True
    )
    assert dry.media_days_removed == ["2026-03-01"] and dry.events_removed == 1
    assert (media / "2026-03-01").exists() and store.count() == 2

    report = purge(store, media, media_retention_days=14, event_retention_days=365, now_unix=now)
    assert not (media / "2026-03-01").exists() and (media / "2026-03-30").exists()
    assert (media / "not-a-day").exists()
    assert report.events_removed == 1 and store.get("old") is None
    assert store.search("eating")[0].event_id == "new"
