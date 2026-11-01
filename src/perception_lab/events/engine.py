"""Turns continuous rule hits into discrete event starts and ends.

A condition (rule, tracks, zone) *starts* an event the first time it holds, unless the
same condition fired less than ``cooldown_s`` ago. It *ends* once it has been false for
``release_s``; the end time is the last time it held. This hysteresis prevents a flickering
detection from producing a stream of duplicate events.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from perception_lab.core.config import RulesConfig
from perception_lab.core.types import TrackObservation
from perception_lab.events.features import TrackHistory
from perception_lab.events.rules import Rule, RuleHit

Key = tuple[str, tuple[int, ...], str | None]


@dataclass
class EngineOutput:
    started: list[RuleHit] = field(default_factory=list)
    ended: list[tuple[Key, float]] = field(default_factory=list)  # (key, last time it held)


class EventEngine:
    def __init__(self, rules: Sequence[Rule], config: RulesConfig | None = None) -> None:
        self.rules = list(rules)
        self.config = config or RulesConfig()
        self._last_true: dict[Key, float] = {}
        self._open: set[Key] = set()  # conditions whose event has started and not ended
        self._last_fired: dict[Key, float] = {}

    def update(
        self,
        now_s: float,
        observations: Sequence[TrackObservation],
        histories: Mapping[int, TrackHistory],
    ) -> EngineOutput:
        out = EngineOutput()
        for rule in self.rules:
            for hit in rule.evaluate(now_s, observations, histories):
                key = hit.key
                if key not in self._last_true:  # condition (re)appeared
                    last = self._last_fired.get(key)
                    if last is None or now_s - last >= self.config.cooldown_s:
                        self._last_fired[key] = now_s
                        self._open.add(key)
                        out.started.append(hit)
                self._last_true[key] = now_s
        for key, last in list(self._last_true.items()):
            if now_s - last > self.config.release_s:
                del self._last_true[key]
                if key in self._open:
                    self._open.discard(key)
                    out.ended.append((key, last))
        return out

    def flush(self) -> list[tuple[Key, float]]:
        """End every open event (e.g. when the source stops)."""
        ended = [(key, self._last_true.get(key, 0.0)) for key in self._open]
        self._open.clear()
        self._last_true.clear()
        return ended
