"""Ground-truth activity segments for a recording.

Labels file (JSON), times in seconds from the first frame of the recording::

    {
      "recording": "living-room-2026-11-01.bag",
      "duration_s": 1800,
      "segments": [
        {"activity": "eating", "t_start_s": 12.0, "t_end_s": 40.0, "animal_class": "cat"},
        {"activity": "interacting", "t_start_s": 300.0, "t_end_s": 312.0, "animal_class": null}
      ]
    }

Use the rule hypothesis names (``entered_restricted_area``, ``on_elevated_surface``, ...) or
the VLM activity names (``eating``, ``grooming``, ...). ``animal_class`` ``null`` matches any
animal (pairwise activities).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perception_lab.events.model import ACTIVITIES, HYPOTHESES

KNOWN_ACTIVITIES = frozenset(ACTIVITIES) | frozenset(HYPOTHESES)


@dataclass(frozen=True)
class Segment:
    activity: str
    t_start_s: float
    t_end_s: float
    animal_class: str | None = None


@dataclass(frozen=True)
class Labels:
    segments: tuple[Segment, ...]
    duration_s: float | None = None
    recording: str | None = None


def labels_from_mapping(data: dict[str, Any]) -> Labels:
    segments = []
    for i, s in enumerate(data.get("segments", [])):
        try:
            seg = Segment(
                str(s["activity"]),
                float(s["t_start_s"]),
                float(s["t_end_s"]),
                s.get("animal_class"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"segments[{i}] needs activity, t_start_s and t_end_s") from exc
        if seg.activity not in KNOWN_ACTIVITIES:
            raise ValueError(f"segments[{i}]: unknown activity {seg.activity!r}")
        if seg.t_end_s < seg.t_start_s:
            raise ValueError(f"segments[{i}]: t_end_s is before t_start_s")
        segments.append(seg)
    duration = data.get("duration_s")
    return Labels(
        tuple(segments), float(duration) if duration is not None else None, data.get("recording")
    )


def load_labels(path: str | Path) -> Labels:
    return labels_from_mapping(json.loads(Path(path).read_text()))
