import datetime as dt

from perception_lab.events.model import PetEvent
from perception_lab.store.db import EventStore
from perception_lab.store.summary import daily_summary, day_bounds_unix, summarize


def _event(event_id, hypothesis, start, end, pet=None, activity=None):
    e = PetEvent(event_id, "cam", "rule", hypothesis, (1,), ("cat",), start, end, "t")
    if activity or pet:
        e.interpretation = {"activity": activity or hypothesis, "description": "x"}
        if pet:
            e.interpretation["pet"] = pet
        e.interpretation_status = "done"
    return e


def test_summarize_groups_by_pet_and_activity():
    report = summarize(
        [
            _event("a", "eating", 0, 30, pet="Nube"),
            _event("b", "eating", 100, 120, pet="Nube"),
            _event("c", "resting_or_sleeping", 0, 600, pet="Negrita", activity="unknown"),
            _event("d", "running", 0, 5),
        ]
    )
    assert report["events"] == 4
    nube = report["pets"]["Nube"]
    assert nube["events"] == 2 and nube["seconds"] == 50.0
    assert nube["activities"]["eating"] == {
        "events": 2, "seconds": 50.0, "from_vlm": 2, "from_rule": 0,
    }  # fmt: skip
    assert report["pets"]["Negrita"]["activities"]["unknown"]["from_vlm"] == 1
    assert report["pets"]["unnamed"]["activities"]["running"]["from_rule"] == 1


def test_summarize_empty():
    assert summarize([]) == {"events": 0, "pets": {}}


def test_daily_summary_only_counts_that_day(tmp_path):
    store = EventStore(tmp_path / "lab.db")
    day = dt.date(2026, 11, 9)
    start, end = day_bounds_unix(day)
    store.add(_event("in", "eating", 0, 10, pet="Leia"), created_unix=start + 60)
    store.add(_event("before", "eating", 0, 10, pet="Leia"), created_unix=start - 60)
    store.add(_event("after", "eating", 0, 10, pet="Leia"), created_unix=end + 60)
    report = daily_summary(store, day)
    store.close()
    assert report["date"] == "2026-11-09" and report["events"] == 1
    assert end - start in (23 * 3600, 24 * 3600, 25 * 3600)
