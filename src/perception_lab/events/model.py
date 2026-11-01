"""The stored event record.

An event keeps three kinds of information apart:

* ``observed_facts``: measured by deterministic code (durations, zones, speeds, depth);
* ``interpretation``: what a vision-language model said about the evidence, if asked;
* ``uncertainty``: known reasons the facts or the interpretation may be wrong.

``hypothesis`` is the activity the triggering rule suggests (a food-zone dwell suggests
"eating"); it is a prior, not a verified activity. ``confidence`` stays ``None`` until it
can be calibrated against reviewed events: numbers stated by a language model are not
calibrated probabilities.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1

# Activities a rule can suggest.
HYPOTHESES = (
    "entered_restricted_area",
    "eating",
    "drinking",
    "resting_or_sleeping",
    "on_elevated_surface",
    "running",
    "interacting",
    "unusual_inactivity",
    "dwelling",
)

# Activities the VLM may answer with. "unknown" is always allowed.
ACTIVITIES = (
    "eating",
    "drinking",
    "resting_or_sleeping",
    "walking",
    "running",
    "playing",
    "grooming",
    "jumping",
    "on_elevated_surface",
    "interacting",
    "other",
    "unknown",
)

INTERPRETATION_STATES = ("not_requested", "pending", "done", "failed", "skipped")
REVIEW_STATES = ("unreviewed", "confirmed", "rejected")


def new_event_id() -> str:
    return uuid.uuid4().hex


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass
class PetEvent:
    event_id: str
    camera_id: str
    rule: str
    hypothesis: str
    track_ids: tuple[int, ...]
    animal_classes: tuple[str, ...]
    timestamp_start_s: float  # source clock
    timestamp_end_s: float
    detected_at: str  # host wall clock, ISO-8601
    zone: str | None = None
    observed_facts: dict[str, Any] = field(default_factory=dict)
    interpretation: dict[str, Any] | None = None
    interpretation_status: str = "not_requested"
    uncertainty: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    review_status: str = "unreviewed"
    review_label: str | None = None
    confidence: float | None = None
    event_type: str = "pet_activity"
    schema_version: int = SCHEMA_VERSION

    @property
    def activity(self) -> str | None:
        """The VLM's activity when interpreted, otherwise ``None`` (see ``hypothesis``)."""
        if self.interpretation_status == "done" and self.interpretation:
            return self.interpretation.get("activity")
        return None

    @property
    def duration_s(self) -> float:
        return self.timestamp_end_s - self.timestamp_start_s

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["track_ids"] = list(self.track_ids)
        data["animal_classes"] = list(self.animal_classes)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PetEvent:
        values = dict(data)
        values["track_ids"] = tuple(values.get("track_ids", ()))
        values["animal_classes"] = tuple(values.get("animal_classes", ()))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in values.items() if k in known})

    def summary(self) -> str:
        """Short deterministic sentence built from facts only (used for notifications)."""
        who = " and ".join(
            f"{label} #{tid}"
            for label, tid in zip(self.animal_classes, self.track_ids, strict=False)
        )
        text = {
            "entered_restricted_area": f"{who} entered {self.zone}",
            "eating": f"{who} at {self.zone} (possible eating)",
            "drinking": f"{who} at {self.zone} (possible drinking)",
            "resting_or_sleeping": f"{who} has been still (possible resting)",
            "on_elevated_surface": f"{who} is on a raised surface",
            "running": f"{who} moved fast (possible running)",
            "interacting": f"{who} are close together",
            "unusual_inactivity": f"{who} has not moved for a long time",
        }.get(self.hypothesis, f"{who}: {self.rule}")
        duration = self.observed_facts.get("duration_s")
        if duration is not None:
            text += f" for {duration:.0f} s"
        return text
