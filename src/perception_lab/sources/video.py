"""Colour-only source from a video file (or a V4L2 device index). No depth."""

from __future__ import annotations

from collections.abc import Iterator

import cv2

from perception_lab.core.types import FrameBundle
from perception_lab.sources.base import SourceError


class VideoSource:
    def __init__(self, path: str, *, camera_id: str = "video", fps: float | None = None) -> None:
        self.path = path
        self.camera_id = camera_id
        self._fps = fps
        self._capture: cv2.VideoCapture | None = None

    def frames(self) -> Iterator[FrameBundle]:
        target = int(self.path) if self.path.isdigit() else self.path
        capture = cv2.VideoCapture(target)
        if not capture.isOpened():
            raise SourceError(f"could not open video source: {self.path}")
        self._capture = capture
        fps = self._fps or capture.get(cv2.CAP_PROP_FPS) or 15.0
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                yield FrameBundle(index, index / fps, frame, camera_id=self.camera_id)
                index += 1
        finally:
            self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
