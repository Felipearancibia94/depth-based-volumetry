"""Multi-object tracking in the style of ByteTrack: Kalman prediction + two-stage IoU matching.

High-score detections are matched first; remaining tracks then try the low-score ones,
which keeps tracks alive through partial occlusion without creating tracks from weak
detections. Track ids are per-session numbers, not animal identities.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from perception_lab.core.config import TrackerConfig
from perception_lab.core.types import BoxXYXY, Detection, box_iou


class KalmanBox:
    """Constant-velocity Kalman filter on ``(cx, cy, w, h)``."""

    def __init__(self, box: BoxXYXY) -> None:
        x1, y1, x2, y2 = box
        self.x = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0, 0, 0, 0], float)
        size = max(self.x[2], self.x[3], 1.0)
        self.P = np.diag([size, size, size, size, 10 * size, 10 * size, size, size]) ** 2 * 0.01
        self.H = np.eye(4, 8)

    def predict(self, dt_s: float) -> None:
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt_s
        size = max(self.x[2], self.x[3], 1.0)
        q = (0.05 * size) ** 2 * max(dt_s, 1e-3)
        Q = np.diag([q, q, q, q, 25 * q, 25 * q, q, q])  # velocity adapts within a few frames
        self.x = F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.P = F @ self.P @ F.T + Q

    def update(self, box: BoxXYXY) -> None:
        x1, y1, x2, y2 = box
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        size = max(z[2], z[3], 1.0)
        R = np.eye(4) * (0.05 * size) ** 2
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def box(self) -> BoxXYXY:
        cx, cy, w, h = self.x[:4]
        return (float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2))


@dataclass
class Track:
    track_id: int
    kalman: KalmanBox
    score: float
    last_seen_s: float
    hits: int = 1
    confirmed: bool = False
    updated: bool = True  # matched in the latest update
    mask: NDArray | None = None
    detection_box: BoxXYXY | None = None
    _votes: Counter = field(default_factory=Counter)

    @property
    def box(self) -> BoxXYXY:
        return self.detection_box if self.updated and self.detection_box else self.kalman.box

    @property
    def label(self) -> str:
        return self._votes.most_common(1)[0][0] if self._votes else "unknown"

    def vote(self, label: str, score: float) -> None:
        self._votes[label] += score


class Tracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 1
        self._last_s: float | None = None

    def update(self, detections: Sequence[Detection], timestamp_s: float) -> list[Track]:
        """Advance to ``timestamp_s`` with this frame's detections.

        Returns the confirmed tracks matched in this frame.
        """
        cfg = self.config
        dt = 0.0 if self._last_s is None else max(0.0, timestamp_s - self._last_s)
        self._last_s = timestamp_s
        for track in self.tracks:
            track.kalman.predict(dt)
            track.updated = False

        high = [d for d in detections if d.score >= cfg.high_score]
        low = [d for d in detections if cfg.low_score <= d.score < cfg.high_score]

        unmatched_tracks = list(range(len(self.tracks)))
        matches, unmatched_high = _greedy_match(
            [self.tracks[i].kalman.box for i in unmatched_tracks], high, cfg.match_iou
        )
        for ti, di in matches:
            self._apply(self.tracks[unmatched_tracks[ti]], high[di], timestamp_s)
        remaining = [
            unmatched_tracks[i]
            for i in range(len(unmatched_tracks))
            if i not in {t for t, _ in matches}
        ]

        low_matches, _ = _greedy_match(
            [self.tracks[i].kalman.box for i in remaining], low, cfg.low_match_iou
        )
        for ti, di in low_matches:
            self._apply(self.tracks[remaining[ti]], low[di], timestamp_s)

        # Fast motion at low frame rates can leave no overlap with the prediction: match the
        # leftovers of the same class by centre distance relative to box size.
        leftover = [i for i in remaining if not self.tracks[i].updated and self.tracks[i].confirmed]
        free_high = [high[di] for di in unmatched_high]
        shift_matches, still_free = _center_match(
            [self.tracks[i] for i in leftover], free_high, cfg.max_center_shift
        )
        for ti, di in shift_matches:
            self._apply(self.tracks[leftover[ti]], free_high[di], timestamp_s)
        unmatched_high = [unmatched_high[i] for i in still_free]

        for di in unmatched_high:
            d = high[di]
            track = Track(self._next_id, KalmanBox(d.box_xyxy), d.score, timestamp_s)
            track.detection_box, track.mask = d.box_xyxy, d.mask
            track.vote(d.label, d.score)
            track.confirmed = cfg.min_hits <= 1
            self._next_id += 1
            self.tracks.append(track)

        self.tracks = [
            t
            for t in self.tracks
            if (t.updated or t.confirmed) and timestamp_s - t.last_seen_s <= cfg.max_lost_s
        ]
        return [t for t in self.tracks if t.confirmed and t.updated]

    def _apply(self, track: Track, detection: Detection, timestamp_s: float) -> None:
        track.kalman.update(detection.box_xyxy)
        track.detection_box, track.mask = detection.box_xyxy, detection.mask
        track.score = detection.score
        track.last_seen_s = timestamp_s
        track.hits += 1
        track.updated = True
        track.vote(detection.label, detection.score)
        if track.hits >= self.config.min_hits:
            track.confirmed = True


def _center_match(
    tracks: Sequence[Track], detections: Sequence[Detection], max_shift: float
) -> tuple[list[tuple[int, int]], list[int]]:
    """Greedy nearest-centre assignment between same-label pairs, gated by
    ``distance / max(box diagonal) <= max_shift``."""
    pairs = []
    for ti, track in enumerate(tracks):
        tb = track.kalman.box
        for di, d in enumerate(detections):
            if d.label != track.label:
                continue
            db = d.box_xyxy
            dist = np.hypot(
                (tb[0] + tb[2] - db[0] - db[2]) / 2, (tb[1] + tb[3] - db[1] - db[3]) / 2
            )
            diag = max(
                np.hypot(tb[2] - tb[0], tb[3] - tb[1]), np.hypot(db[2] - db[0], db[3] - db[1])
            )
            if diag > 0 and dist / diag <= max_shift:
                pairs.append((dist / diag, ti, di))
    pairs.sort()
    used_t, used_d, matches = set(), set(), []
    for _, ti, di in pairs:
        if ti not in used_t and di not in used_d:
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))
    return matches, [i for i in range(len(detections)) if i not in used_d]


def _greedy_match(
    track_boxes: Sequence[BoxXYXY], detections: Sequence[Detection], min_iou: float
) -> tuple[list[tuple[int, int]], list[int]]:
    """Greedy highest-IoU-first assignment. Returns matches and unmatched detection indices."""
    pairs = []
    for ti, tb in enumerate(track_boxes):
        for di, d in enumerate(detections):
            iou = box_iou(tb, d.box_xyxy)
            if iou >= min_iou:
                pairs.append((iou, ti, di))
    pairs.sort(reverse=True)
    used_t, used_d, matches = set(), set(), []
    for _, ti, di in pairs:
        if ti not in used_t and di not in used_d:
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))
    return matches, [i for i in range(len(detections)) if i not in used_d]
