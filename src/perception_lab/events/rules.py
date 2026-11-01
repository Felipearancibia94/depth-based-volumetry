"""Deterministic rules that turn track features into candidate events.

A rule returns a :class:`RuleHit` for every condition that currently holds; the
:class:`~perception_lab.events.engine.EventEngine` decides when a hit starts a new event.
Hypotheses are priors from geometry (e.g. a dwell at the food bowl suggests eating), not
verified activities.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from perception_lab.core.config import RulesConfig
from perception_lab.core.types import TrackObservation
from perception_lab.events.features import (
    TrackHistory,
    condition_duration_s,
    distance,
    speed,
    stationary_duration_s,
)
from perception_lab.spatial.scene import SceneCalibration

DWELL_HYPOTHESES = {"food": "eating", "water": "drinking", "bed": "resting_or_sleeping"}


@dataclass(frozen=True)
class RuleHit:
    rule: str
    hypothesis: str
    track_ids: tuple[int, ...]
    started_s: float
    zone: str | None = None
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, tuple[int, ...], str | None]:
        return (self.rule, self.track_ids, self.zone)


class Rule(Protocol):
    name: str

    def evaluate(
        self,
        now_s: float,
        observations: Sequence[TrackObservation],
        histories: Mapping[int, TrackHistory],
    ) -> list[RuleHit]: ...


def _base_facts(obs: TrackObservation) -> dict[str, Any]:
    s = obs.spatial
    facts: dict[str, Any] = {"depth_valid_ratio": round(s.depth_valid_ratio, 2)}
    if s.median_depth_m is not None:
        facts["distance_to_camera_m"] = round(s.median_depth_m, 2)
    if s.elevation_m is not None:
        facts["elevation_m"] = round(s.elevation_m, 2)
    if s.zones:
        facts["zones"] = list(s.zones)
    if s.quality:
        facts["depth_quality"] = list(s.quality)
    return facts


@dataclass
class ZoneRule:
    """Fires when a track stays in a zone carrying one of ``tags`` for ``min_s``."""

    name: str
    scene: SceneCalibration
    tag_min_s: dict[str, float]
    tag_hypothesis: dict[str, str]

    def evaluate(self, now_s, observations, histories):
        hits = []
        for obs in observations:
            for zone_name in obs.spatial.zones:
                zone = self.scene.zone(zone_name)
                tags = [t for t in (zone.tags if zone else ()) if t in self.tag_min_s]
                if not tags:
                    continue
                tag = tags[0]
                dur = condition_duration_s(
                    histories[obs.track_id], lambda o, z=zone_name: z in o.spatial.zones
                )
                if dur >= self.tag_min_s[tag]:
                    facts = {**_base_facts(obs), "duration_s": round(dur, 1), "zone_tag": tag}
                    hits.append(
                        RuleHit(
                            self.name,
                            self.tag_hypothesis[tag],
                            (obs.track_id,),
                            now_s - dur,
                            zone_name,
                            facts,
                        )
                    )
        return hits


@dataclass
class ElevatedRule:
    """Animal on a raised surface: measured elevation, or a zone tagged ``elevated``."""

    scene: SceneCalibration
    min_elevation_m: float
    min_s: float
    name: str = "elevated_surface"

    def _elevated(self, obs: TrackObservation) -> bool:
        e = obs.spatial.elevation_m
        if e is not None and e >= self.min_elevation_m:
            return True
        return any(
            (z := self.scene.zone(name)) is not None and "elevated" in z.tags
            for name in obs.spatial.zones
        )

    def evaluate(self, now_s, observations, histories):
        hits = []
        for obs in observations:
            dur = condition_duration_s(histories[obs.track_id], self._elevated)
            if dur >= self.min_s:
                facts = {**_base_facts(obs), "duration_s": round(dur, 1)}
                hits.append(
                    RuleHit(
                        self.name, "on_elevated_surface", (obs.track_id,), now_s - dur, None, facts
                    )
                )
        return hits


@dataclass
class StationaryRule:
    name: str
    hypothesis: str
    min_s: float
    radius_m: float
    radius_bl: float

    def evaluate(self, now_s, observations, histories):
        hits = []
        for obs in observations:
            dur = stationary_duration_s(histories[obs.track_id], self.radius_m, self.radius_bl)
            if dur >= self.min_s:
                facts = {**_base_facts(obs), "duration_s": round(dur, 1)}
                hits.append(
                    RuleHit(self.name, self.hypothesis, (obs.track_id,), now_s - dur, None, facts)
                )
        return hits


@dataclass
class SpeedBurstRule:
    """Mean speed over ``window_s`` above a threshold (m/s with depth, body lengths/s without)."""

    window_s: float
    min_mps: float
    min_bl: float
    name: str = "speed_burst"

    def evaluate(self, now_s, observations, histories):
        hits = []
        for obs in observations:
            v = speed(histories[obs.track_id], self.window_s)
            if v is None:
                continue
            limit = self.min_mps if v.unit == "m/s" else self.min_bl
            if v.value >= limit:
                facts = {
                    **_base_facts(obs),
                    "speed": round(v.value, 2),
                    "speed_unit": v.unit,
                    "duration_s": self.window_s,
                }
                hits.append(
                    RuleHit(
                        self.name, "running", (obs.track_id,), now_s - self.window_s, None, facts
                    )
                )
        return hits


@dataclass
class ProximityRule:
    """Two animals closer than ``max_m`` (or one body length without depth) for ``min_s``."""

    max_m: float
    min_s: float
    name: str = "proximity"
    _since: dict[tuple[int, int], float] = field(default_factory=dict)

    def evaluate(self, now_s, observations, histories):
        hits, close_now = [], set()
        for a, b in itertools.combinations(sorted(observations, key=lambda o: o.track_id), 2):
            dist, unit = distance(a, b)
            if dist > (self.max_m if unit == "m" else 1.0):
                continue
            pair = (a.track_id, b.track_id)
            close_now.add(pair)
            since = self._since.setdefault(pair, now_s)
            if now_s - since >= self.min_s:
                facts = {
                    "distance": round(dist, 2),
                    "distance_unit": unit,
                    "duration_s": round(now_s - since, 1),
                    "classes": [a.label, b.label],
                }
                hits.append(RuleHit(self.name, "interacting", pair, since, None, facts))
        for pair in list(self._since):
            if pair not in close_now:
                del self._since[pair]
        return hits


def default_rules(config: RulesConfig, scene: SceneCalibration) -> list[Rule]:
    return [
        ZoneRule(
            "zone_entry",
            scene,
            {tag: config.zone_entry_min_s for tag in config.zone_entry_tags},
            {tag: "entered_restricted_area" for tag in config.zone_entry_tags},
        ),
        ZoneRule(
            "zone_dwell",
            scene,
            dict(config.dwell_tags),
            {tag: DWELL_HYPOTHESES.get(tag, "dwelling") for tag in config.dwell_tags},
        ),
        ElevatedRule(scene, config.elevated_min_m, config.elevated_min_s),
        StationaryRule(
            "stationary",
            "resting_or_sleeping",
            config.stationary_s,
            config.stationary_radius_m,
            config.stationary_radius_bl,
        ),
        StationaryRule(
            "long_inactivity",
            "unusual_inactivity",
            config.long_inactivity_s,
            config.stationary_radius_m,
            config.stationary_radius_bl,
        ),
        SpeedBurstRule(config.speed_burst_min_s, config.speed_burst_mps, config.speed_burst_bl),
        ProximityRule(config.proximity_m, config.proximity_min_s),
    ]
