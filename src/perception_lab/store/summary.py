"""Daily summary per pet, computed from stored events (no model involved)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import Any

from perception_lab.events.model import PetEvent
from perception_lab.store.db import EventStore

NO_NAME = "unnamed"  # events where no pet name was asked for or answered


def day_bounds_unix(day: dt.date) -> tuple[float, float]:
    """Start and end of ``day`` in local time, as Unix seconds."""
    start = dt.datetime.combine(day, dt.time.min).astimezone()
    return start.timestamp(), (start + dt.timedelta(days=1)).timestamp()


def summarize(events: Iterable[PetEvent]) -> dict[str, Any]:
    """Events and seconds per pet and activity.

    The activity is the VLM's when it answered, otherwise the rule's hypothesis; each activity
    counts how many came from the VLM and how many from the rule alone.
    """
    pets: dict[str, dict[str, Any]] = {}
    for e in events:
        name = e.pet or NO_NAME
        activity = e.activity or e.hypothesis
        entry = pets.setdefault(name, {"events": 0, "seconds": 0.0, "activities": {}})
        entry["events"] += 1
        entry["seconds"] += max(0.0, e.duration_s)
        act = entry["activities"].setdefault(
            activity, {"events": 0, "seconds": 0.0, "from_vlm": 0, "from_rule": 0}
        )
        act["events"] += 1
        act["seconds"] += max(0.0, e.duration_s)
        act["from_vlm" if e.activity else "from_rule"] += 1
    for entry in pets.values():
        entry["seconds"] = round(entry["seconds"], 1)
        for act in entry["activities"].values():
            act["seconds"] = round(act["seconds"], 1)
    return {"events": sum(p["events"] for p in pets.values()), "pets": pets}


def daily_summary(store: EventStore, day: dt.date) -> dict[str, Any]:
    start, end = day_bounds_unix(day)
    events = store.list(limit=100_000, created_after_unix=start, created_before_unix=end)
    return {"date": day.isoformat(), **summarize(events)}
