"""Frame source interface and helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol

from perception_lab.core.types import FrameBundle


class SourceError(RuntimeError):
    """Raised when a source cannot be opened or stops unexpectedly."""


class FrameSource(Protocol):
    """Anything that yields :class:`FrameBundle` objects in timestamp order."""

    def frames(self) -> Iterator[FrameBundle]: ...

    def close(self) -> None: ...


def throttle(frames: Iterable[FrameBundle], max_fps: float | None) -> Iterator[FrameBundle]:
    """Drop frames so at most ``max_fps`` are processed per second of source time.

    Frames are dropped, never queued: the newest frame always wins, so a slow consumer
    does not fall behind a live camera.
    """
    if not max_fps:
        yield from frames
        return
    period_s = 1.0 / max_fps
    next_s: float | None = None
    for bundle in frames:
        if next_s is None or bundle.timestamp_s >= next_s - 1e-6:
            next_s = bundle.timestamp_s + period_s
            yield bundle


def limit(frames: Iterable[FrameBundle], max_frames: int | None) -> Iterator[FrameBundle]:
    for i, bundle in enumerate(frames):
        if max_frames is not None and i >= max_frames:
            return
        yield bundle
