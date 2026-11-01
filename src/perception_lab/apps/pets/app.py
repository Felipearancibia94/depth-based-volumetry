"""Build a ready-to-run pet pipeline from a :class:`LabConfig`."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TextIO

from perception_lab.apps.pets.pipeline import PetPipeline
from perception_lab.core.config import LabConfig
from perception_lab.core.types import FrameBundle
from perception_lab.notify.notifier import build_notifier
from perception_lab.sources import FrameSource, limit, open_source, throttle
from perception_lab.spatial.scene import SceneCalibration, load_scene
from perception_lab.store.db import EventStore
from perception_lab.vision.detector import create_detector
from perception_lab.vlm.interpreter import Interpreter
from perception_lab.vlm.providers import PostJSON, create_provider, post_json


@dataclass
class PetApp:
    config: LabConfig
    source: FrameSource
    pipeline: PetPipeline
    store: EventStore

    def frames(self) -> Iterator[FrameBundle]:
        cfg = self.config.source
        return limit(throttle(self.source.frames(), cfg.process_fps), cfg.max_frames)

    def close(self) -> None:
        self.source.close()
        self.store.close()


def resolve_scene(
    config: LabConfig, source: FrameSource, warn: TextIO = sys.stderr
) -> SceneCalibration:
    if config.spatial.scene_path:
        return load_scene(config.spatial.scene_path)
    scene_fn = getattr(source, "scene", None)
    if callable(scene_fn):  # synthetic source knows its exact scene
        return scene_fn()
    print(
        "warning: no spatial.scene_path: no zones and no floor plane, so zone and elevation "
        "rules are inactive (run `lab calibrate`)",
        file=warn,
    )
    return SceneCalibration(config.source.camera_id)


def build_app(
    config: LabConfig,
    *,
    log_stream: TextIO | None = None,
    keep_live_view: bool = False,
    vlm_async: bool = True,
    vlm_post: PostJSON = post_json,
) -> PetApp:
    source = open_source(config.source)
    scene = resolve_scene(config, source)
    store = EventStore(config.storage.db_path)
    interpreter = None
    if config.vlm.enabled:
        interpreter = Interpreter(create_provider(config.vlm, vlm_post), config.vlm)
    pipeline = PetPipeline(
        config,
        detector=create_detector(config.detector),
        scene=scene,
        store=store,
        notifier=build_notifier(config.notify, log_stream=log_stream),
        interpreter=interpreter,
        vlm_async=vlm_async,
        keep_live_view=keep_live_view,
    )
    return PetApp(config, source, pipeline, store)
