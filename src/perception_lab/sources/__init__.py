"""Frame sources: synthetic scene, video file, RealSense camera and ``.bag`` replay."""

from __future__ import annotations

from perception_lab.core.config import SourceConfig
from perception_lab.sources.base import FrameSource, SourceError, limit, throttle


def open_source(config: SourceConfig) -> FrameSource:
    """Create the source described by ``config`` (opened lazily on ``frames()``)."""
    if config.kind == "synthetic":
        from perception_lab.sources.synthetic import SyntheticSource

        return SyntheticSource(
            fps=config.fps,
            duration_s=config.synthetic_seconds,
            seed=config.seed,
            camera_id=config.camera_id,
        )
    if config.kind == "video":
        from perception_lab.sources.video import VideoSource

        if not config.path:
            raise ValueError("source.path is required for kind = 'video'")
        return VideoSource(config.path, camera_id=config.camera_id)
    from perception_lab.sources.realsense import RealSenseSource

    return RealSenseSource(config)


__all__ = ["FrameSource", "SourceError", "limit", "open_source", "throttle"]
