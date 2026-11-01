"""Short-term frame memory and evidence selection for events.

Only a few downscaled frames per second are kept in memory for ``ring_buffer_s``; nothing
is written to disk unless an event selects it. This bounds both memory and how much
household video is ever stored.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from perception_lab.core.types import BoxXYXY, FrameBundle, TrackObservation


@dataclass
class BufferedFrame:
    timestamp_s: float
    frame_index: int
    image: NDArray[np.uint8]  # downscaled BGR
    scale: float  # image pixels per original pixel
    boxes: dict[int, BoxXYXY] = field(default_factory=dict)  # original pixels


class FrameRingBuffer:
    def __init__(self, seconds: float = 20.0, fps: float = 4.0, max_side_px: int = 640) -> None:
        self.seconds = seconds
        self.period_s = 1.0 / fps if fps > 0 else 0.0
        self.max_side_px = max_side_px
        self.frames: deque[BufferedFrame] = deque()

    def maybe_add(self, bundle: FrameBundle, observations: Iterable[TrackObservation]) -> bool:
        if self.frames and bundle.timestamp_s - self.frames[-1].timestamp_s < self.period_s:
            return False
        image, scale = _downscale(bundle.color_bgr, self.max_side_px)
        boxes = {o.track_id: o.box_xyxy for o in observations}
        self.frames.append(BufferedFrame(bundle.timestamp_s, bundle.index, image, scale, boxes))
        cutoff = bundle.timestamp_s - self.seconds
        while self.frames and self.frames[0].timestamp_s < cutoff:
            self.frames.popleft()
        return True

    def frames_with(
        self, track_ids: Sequence[int], start_s: float, end_s: float
    ) -> list[BufferedFrame]:
        return [
            f
            for f in self.frames
            if start_s <= f.timestamp_s <= end_s and any(t in f.boxes for t in track_ids)
        ]


def sharpness(image: NDArray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_frames(
    buffer: FrameRingBuffer, track_ids: Sequence[int], start_s: float, end_s: float, n: int
) -> list[BufferedFrame]:
    """Up to ``n`` frames spread over the interval, the sharpest crop in each time slot.

    Falls back to the whole buffer when the interval itself has no frame with the tracks.
    """
    candidates = buffer.frames_with(track_ids, start_s, end_s) or buffer.frames_with(
        track_ids, float("-inf"), float("inf")
    )
    if len(candidates) <= n:
        return candidates
    slots = np.array_split(np.arange(len(candidates)), n)
    chosen = []
    for slot in slots:
        best = max(slot, key=lambda i: sharpness(crop(candidates[i], track_ids)))
        chosen.append(candidates[int(best)])
    return chosen


def crop(frame: BufferedFrame, track_ids: Sequence[int], margin: float = 0.3) -> NDArray[np.uint8]:
    """Crop around the union of the tracks' boxes, with ``margin`` of the box size."""
    boxes = [frame.boxes[t] for t in track_ids if t in frame.boxes]
    if not boxes:
        return frame.image
    x1 = min(b[0] for b in boxes) * frame.scale
    y1 = min(b[1] for b in boxes) * frame.scale
    x2 = max(b[2] for b in boxes) * frame.scale
    y2 = max(b[3] for b in boxes) * frame.scale
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    h, w = frame.image.shape[:2]
    xa, ya = int(max(0, x1 - mx)), int(max(0, y1 - my))
    xb, yb = int(min(w, x2 + mx)), int(min(h, y2 + my))
    if xb - xa < 2 or yb - ya < 2:
        return frame.image
    return frame.image[ya:yb, xa:xb].copy()


def fit_max_side(image: NDArray, max_side_px: int) -> NDArray:
    return _downscale(image, max_side_px)[0]


def montage(images: Sequence[NDArray], tile_px: int = 320, columns: int = 2) -> NDArray[np.uint8]:
    """Grid of letterboxed tiles, for runtimes that accept a single image only."""
    if not images:
        raise ValueError("no images for the montage")
    rows = -(-len(images) // columns)
    canvas = np.zeros((rows * tile_px, columns * tile_px, 3), np.uint8)
    for i, image in enumerate(images):
        tile = fit_max_side(image, tile_px)
        r, c = divmod(i, columns)
        y = r * tile_px + (tile_px - tile.shape[0]) // 2
        x = c * tile_px + (tile_px - tile.shape[1]) // 2
        canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return canvas


def _downscale(image: NDArray, max_side_px: int) -> tuple[NDArray[np.uint8], float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_side_px / max(h, w))
    if scale >= 1.0:
        return image.copy(), 1.0
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale
