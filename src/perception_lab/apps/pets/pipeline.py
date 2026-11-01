"""The pet monitoring loop.

Per frame: detect animals, track them, measure depth facts, update histories and evaluate
rules. When a rule starts an event: store it with its facts, save evidence crops from the
in-memory ring buffer, notify, and (optionally) queue the evidence for the VLM. The VLM
answer is merged into the stored event later; alerts never wait for it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import cv2

from perception_lab.core.config import LabConfig
from perception_lab.core.types import FrameBundle, TrackObservation
from perception_lab.events.engine import EventEngine, Key
from perception_lab.events.evidence import FrameRingBuffer, crop, fit_max_side, select_frames
from perception_lab.events.features import TrackHistory
from perception_lab.events.model import PetEvent, iso_now, new_event_id
from perception_lab.events.rules import RuleHit, default_rules
from perception_lab.notify.notifier import NotificationPolicy
from perception_lab.spatial.estimator import SpatialEstimator
from perception_lab.spatial.scene import SceneCalibration
from perception_lab.store.db import EventStore
from perception_lab.store.media import save_images
from perception_lab.vision.detector import Detector
from perception_lab.vision.draw import draw_observations
from perception_lab.vision.tracker import Tracker
from perception_lab.vlm.interpreter import (
    InterpretationResult,
    Interpreter,
    VLMWorker,
    evidence_images,
)

_QUALITY_NOTES = {
    "no_depth": "no depth: distances and speeds are in body lengths, elevation unknown",
    "low_valid_ratio": "too few valid depth pixels on the animal: 3D position not measured",
    "no_floor": "no floor calibration: elevation not measured",
    "no_foot_depth": "no valid depth at the animal's feet: elevation not measured",
}


@dataclass
class PipelineStatus:
    started_at: str = field(default_factory=iso_now)
    frames: int = 0
    detections: int = 0
    events: int = 0
    notifications: int = 0
    processing_fps: float = 0.0
    last_frame_timestamp_s: float | None = None
    last_frame_host_monotonic: float | None = None
    active_tracks: list[dict[str, Any]] = field(default_factory=list)
    stage_ms: dict[str, float] = field(default_factory=dict)  # moving averages
    vlm: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        if self.last_frame_host_monotonic is not None:
            data["last_frame_age_s"] = round(time.monotonic() - self.last_frame_host_monotonic, 2)
        return data


class PetPipeline:
    def __init__(
        self,
        config: LabConfig,
        *,
        detector: Detector,
        scene: SceneCalibration,
        store: EventStore,
        notifier: NotificationPolicy | None = None,
        interpreter: Interpreter | None = None,
        vlm_async: bool = True,
        keep_live_view: bool = False,
    ) -> None:
        self.config = config
        self.detector = detector
        self.scene = scene
        self.store = store
        self.notifier = notifier
        self.tracker = Tracker(config.tracker)
        self.estimator = SpatialEstimator(scene, config.spatial)
        self.engine = EventEngine(default_rules(config.rules, scene), config.rules)
        self.ring = FrameRingBuffer(
            config.storage.ring_buffer_s,
            config.storage.ring_buffer_fps,
            config.storage.evidence_max_side_px,
        )
        self.histories: dict[int, TrackHistory] = {}
        self.status = PipelineStatus()
        self.keep_live_view = keep_live_view
        self.live_jpeg: bytes | None = None
        self._open: dict[Key, str] = {}
        self._event_lock = threading.Lock()
        self.interpreter = interpreter
        self.worker: VLMWorker | None = None
        if interpreter is not None and vlm_async:
            self.worker = VLMWorker(interpreter, self._on_interpretation, config.vlm)
            self.worker.start()

    # -- per frame ------------------------------------------------------------------

    def process(self, bundle: FrameBundle) -> list[PetEvent]:
        t_start = time.perf_counter()
        status = self.status
        status.frames += 1
        if (bundle.index % max(1, self.config.detector.every_n_frames)) != 0:
            return []
        detections = self.detector.detect(bundle)
        t_det = time.perf_counter()
        tracks = self.tracker.update(detections, bundle.timestamp_s)
        observations = [
            TrackObservation(
                bundle.timestamp_s,
                tr.track_id,
                tr.label,
                tr.box,
                tr.score,
                self.estimator.measure(bundle, tr.box, tr.mask),
                bundle.index,
            )
            for tr in tracks
        ]
        t_spatial = time.perf_counter()
        for obs in observations:
            history = self.histories.get(obs.track_id)
            if history is None:
                history = self.histories[obs.track_id] = TrackHistory(
                    obs.track_id, self.config.rules.history_s
                )
            history.add(obs)
        alive = {t.track_id for t in self.tracker.tracks}
        for tid in [t for t in self.histories if t not in alive]:
            del self.histories[tid]

        self.ring.maybe_add(bundle, observations)
        output = self.engine.update(bundle.timestamp_s, observations, self.histories)
        started = [self._start_event(hit, bundle, observations) for hit in output.started]
        for key, end_s in output.ended:
            self._end_event(key, end_s)
        t_end = time.perf_counter()

        if self.keep_live_view:
            view = draw_observations(bundle.color_bgr, observations, self.scene.zones)
            ok, jpeg = cv2.imencode(".jpg", fit_max_side(view, 960))
            if ok:
                self.live_jpeg = jpeg.tobytes()
        self._update_status(
            bundle, detections, observations, started, t_start, t_det, t_spatial, t_end
        )
        return started

    def run(
        self,
        frames: Iterable[FrameBundle],
        stop: threading.Event | None = None,
        on_frame: Callable[[FrameBundle], None] | None = None,
    ) -> PipelineStatus:
        try:
            for bundle in frames:
                if stop is not None and stop.is_set():
                    break
                self.process(bundle)
                if on_frame is not None:
                    on_frame(bundle)
        finally:
            self.finish()
        return self.status

    def finish(self, vlm_timeout_s: float = 600.0) -> None:
        for key, end_s in self.engine.flush():
            self._end_event(key, end_s)
        if self.worker is not None:
            self.worker.wait_idle(vlm_timeout_s)
            self.worker.stop()
            self.status.vlm = dict(self.worker.stats)

    # -- events ---------------------------------------------------------------------

    def _start_event(
        self, hit: RuleHit, bundle: FrameBundle, observations: list[TrackObservation]
    ) -> PetEvent:
        by_id = {o.track_id: o for o in observations}
        labels = tuple(by_id[t].label if t in by_id else "unknown" for t in hit.track_ids)
        quality = {q for t in hit.track_ids if t in by_id for q in by_id[t].spatial.quality}
        uncertainty = [_QUALITY_NOTES[q] for q in sorted(quality) if q in _QUALITY_NOTES]
        event = PetEvent(
            event_id=new_event_id(),
            camera_id=bundle.camera_id,
            rule=hit.rule,
            hypothesis=hit.hypothesis,
            track_ids=hit.track_ids,
            animal_classes=labels,
            timestamp_start_s=hit.started_s,
            timestamp_end_s=bundle.timestamp_s,
            detected_at=iso_now(),
            zone=hit.zone,
            observed_facts=dict(hit.facts),
            uncertainty=uncertainty,
        )
        frames = select_frames(
            self.ring, hit.track_ids, hit.started_s, bundle.timestamp_s,
            self.config.storage.evidence_frames,
        )  # fmt: skip
        if frames:
            images = [(f"crop_{i}", crop(f, hit.track_ids)) for i, f in enumerate(frames)]
            images.append(("context", frames[-1].image))
            paths = save_images(self.config.storage.media_dir, event.event_id, images)
            times = [f.timestamp_s for f in frames] + [frames[-1].timestamp_s]
            event.evidence = [
                {
                    "kind": "context" if name == "context" else "crop",
                    "path": path,
                    "timestamp_s": ts,
                    "track_ids": list(hit.track_ids),
                }
                for (name, _), path, ts in zip(images, paths, times, strict=True)
            ]
        else:
            event.uncertainty.append("no evidence frames in the buffer")

        vlm_images, vlm_times = (
            evidence_images(frames, hit.track_ids, self.config.vlm) if frames else ([], [])
        )
        if self.interpreter is not None and vlm_images:
            event.interpretation_status = "pending"
        with self._event_lock:
            self.store.add(event)
        self._open[hit.key] = event.event_id
        if self.notifier is not None and self.notifier.notify(event):
            self.status.notifications += 1
        if self.interpreter is not None and vlm_images:
            if self.worker is not None:
                self.worker.submit(event, vlm_images, vlm_times)
            else:
                self._on_interpretation(
                    event, self.interpreter.interpret(event, vlm_images, vlm_times)
                )
        return event

    def _end_event(self, key: Key, end_s: float) -> None:
        event_id = self._open.pop(key, None)
        if event_id is None:
            return
        with self._event_lock:
            event = self.store.get(event_id)
            if event is None:
                return
            event.timestamp_end_s = max(event.timestamp_end_s, end_s)
            event.observed_facts["duration_s"] = round(event.duration_s, 1)
            self.store.update(event)

    def _on_interpretation(self, event: PetEvent, result: InterpretationResult) -> None:
        with self._event_lock:
            stored = self.store.get(event.event_id)
            if stored is None:
                return
            stored.interpretation_status = result.status
            if result.status == "done":
                stored.interpretation = result.interpretation
                activity = (result.interpretation or {}).get("activity")
                if activity == "unknown":
                    stored.uncertainty.append("the VLM could not identify the activity")
                elif activity and activity != stored.hypothesis:
                    stored.uncertainty.append(
                        f"VLM activity '{activity}' differs from rule hypothesis "
                        f"'{stored.hypothesis}'"
                    )
            else:
                stored.interpretation = {"error": result.error, "raw_text": result.raw_text}
                stored.uncertainty.append(f"interpretation {result.status}: {result.error}")
            self.store.update(stored)

    # -- status ---------------------------------------------------------------------

    def _update_status(self, bundle, detections, observations, started, t0, t1, t2, t3) -> None:
        s = self.status
        s.detections += len(detections)
        s.events += len(started)
        now = time.monotonic()
        if s.last_frame_host_monotonic is not None:
            dt = now - s.last_frame_host_monotonic
            if dt > 0:
                s.processing_fps = round(0.9 * s.processing_fps + 0.1 * (1.0 / dt), 2)
        s.last_frame_host_monotonic = now
        s.last_frame_timestamp_s = bundle.timestamp_s
        for name, ms in (("detect", t1 - t0), ("track_spatial", t2 - t1), ("events", t3 - t2)):
            prev = s.stage_ms.get(name)
            value = ms * 1000.0
            s.stage_ms[name] = round(value if prev is None else 0.9 * prev + 0.1 * value, 2)
        s.active_tracks = [
            {
                "track_id": o.track_id,
                "label": o.label,
                "zones": list(o.spatial.zones),
                "distance_m": o.spatial.median_depth_m,
                "elevation_m": o.spatial.elevation_m,
            }
            for o in observations
        ]
        if self.worker is not None:
            s.vlm = {**self.worker.stats, "pending": self.worker.pending}
