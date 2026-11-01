"""Temporal features of one track: speed, stillness and condition durations.

All functions are pure. Speeds use metres when both endpoints have a 3D position and
fall back to body lengths (box height) per second otherwise; the unit is returned so a
caller never mixes them.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from perception_lab.core.types import TrackObservation

MAX_GAP_S = 2.0  # a hole longer than this breaks a continuous condition


class TrackHistory:
    def __init__(self, track_id: int, max_age_s: float = 600.0) -> None:
        self.track_id = track_id
        self.max_age_s = max_age_s
        self.observations: deque[TrackObservation] = deque()

    def add(self, obs: TrackObservation) -> None:
        self.observations.append(obs)
        cutoff = obs.timestamp_s - self.max_age_s
        while self.observations and self.observations[0].timestamp_s < cutoff:
            self.observations.popleft()

    @property
    def latest(self) -> TrackObservation | None:
        return self.observations[-1] if self.observations else None

    def __len__(self) -> int:
        return len(self.observations)


@dataclass(frozen=True)
class Speed:
    value: float
    unit: str  # "m/s" or "body_lengths/s"


def distance(a: TrackObservation, b: TrackObservation) -> tuple[float, str]:
    """Distance between two observations: metres if both have 3D points, else body lengths."""
    pa, pb = a.spatial.point_m, b.spatial.point_m
    if pa is not None and pb is not None:
        return math.dist(pa, pb), "m"
    (ax, ay), (bx, by) = a.center_xy, b.center_xy
    body_px = max(1.0, (a.height_px + b.height_px) / 2)
    return math.hypot(ax - bx, ay - by) / body_px, "body_lengths"


def speed(history: TrackHistory, window_s: float) -> Speed | None:
    """Mean speed over the last ``window_s`` (``None`` if less than half a window is covered)."""
    latest = history.latest
    if latest is None:
        return None
    start = None
    for obs in reversed(history.observations):
        if latest.timestamp_s - obs.timestamp_s > window_s:
            break
        start = obs
    if start is None or latest.timestamp_s - start.timestamp_s < window_s / 2:
        return None
    dist, unit = distance(start, latest)
    dt = latest.timestamp_s - start.timestamp_s
    return Speed(dist / dt, "m/s" if unit == "m" else "body_lengths/s")


def condition_duration_s(
    history: TrackHistory,
    predicate: Callable[[TrackObservation], bool],
    max_false_s: float = 0.0,
) -> float:
    """How long ``predicate`` has held up to the latest observation.

    Runs of false observations no longer than ``max_false_s`` (detector flicker, a box
    edge crossing a zone border) do not break the condition; gaps in the history longer
    than ``MAX_GAP_S`` always do.
    """
    obs = list(history.observations)
    if not obs or not predicate(obs[-1]):
        return 0.0
    first = obs[-1]
    false_run_end: float | None = None  # time of the true observation after a false run
    for prev, cur in zip(reversed(obs[:-1]), reversed(obs[1:]), strict=True):
        if cur.timestamp_s - prev.timestamp_s > MAX_GAP_S:
            break
        if predicate(prev):
            first, false_run_end = prev, None
            continue
        if false_run_end is None:
            false_run_end = cur.timestamp_s
        if false_run_end - prev.timestamp_s > max_false_s:
            break
    return obs[-1].timestamp_s - first.timestamp_s


def stationary_duration_s(history: TrackHistory, radius_m: float, radius_bl: float) -> float:
    """How long the track has stayed within a radius of its current position."""
    obs = list(history.observations)
    if not obs:
        return 0.0
    latest = obs[-1]
    first = latest
    for prev, cur in zip(reversed(obs[:-1]), reversed(obs[1:]), strict=True):
        if cur.timestamp_s - prev.timestamp_s > MAX_GAP_S:
            break
        dist, unit = distance(prev, latest)
        if dist > (radius_m if unit == "m" else radius_bl):
            break
        first = prev
    return latest.timestamp_s - first.timestamp_s
