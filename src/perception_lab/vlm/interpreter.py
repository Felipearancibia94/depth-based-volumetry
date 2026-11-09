"""Event-gated interpretation: evidence frames + measured facts -> validated JSON answer."""

from __future__ import annotations

import collections
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from numpy.typing import NDArray

from perception_lab.core.config import VLMConfig
from perception_lab.events.evidence import BufferedFrame, crop, fit_max_side, montage
from perception_lab.events.model import PetEvent
from perception_lab.vlm.prompts import RETRY_SUFFIX, build_prompt, prompt_version
from perception_lab.vlm.providers import ProviderError, VLMProvider, encode_jpeg
from perception_lab.vlm.schema import InterpretationError, build_json_schema, parse_interpretation


@dataclass
class InterpretationResult:
    status: str  # "done" | "failed" | "skipped"
    interpretation: dict[str, Any] | None = None
    error: str | None = None
    raw_text: str | None = None
    latency_s: float = 0.0
    attempts: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def evidence_images(
    frames: Sequence[BufferedFrame], track_ids: Sequence[int], config: VLMConfig
) -> tuple[list[NDArray], list[float]]:
    """Crops around the tracks (plus one uncropped context frame), sized for the model."""
    n_crops = max(1, config.max_images - (1 if config.context_frame else 0))
    chosen = list(frames)[-n_crops:] if len(frames) > n_crops else list(frames)
    images = [fit_max_side(crop(f, track_ids), config.image_max_side_px) for f in chosen]
    times = [f.timestamp_s for f in chosen]
    if config.context_frame and frames:
        images.append(fit_max_side(frames[-1].image, config.image_max_side_px))
        times.append(frames[-1].timestamp_s)
    if config.montage and images:
        return [montage(images, tile_px=config.image_max_side_px // 2)], times
    return images, times


class Interpreter:
    def __init__(self, provider: VLMProvider, config: VLMConfig) -> None:
        self.provider = provider
        self.config = config

    def interpret(
        self, event: PetEvent, images: Sequence[NDArray], times_s: Sequence[float]
    ) -> InterpretationResult:
        if not images:
            return InterpretationResult("skipped", error="no evidence frames")
        jpegs = [encode_jpeg(i) for i in images]
        pets = self.config.pets
        schema = build_json_schema(pets)
        base_prompt = build_prompt(event, times_s, montage=self.config.montage, pets=pets)
        prompt, total_latency, raw, error = base_prompt, 0.0, None, None
        for attempt in (1, 2):
            try:
                reply = self.provider.complete(prompt, jpegs, schema)
            except ProviderError as exc:
                return InterpretationResult("failed", error=str(exc), attempts=attempt)
            total_latency += reply.latency_s
            raw = reply.text
            try:
                parsed = parse_interpretation(reply.text, pets)
            except InterpretationError as exc:
                error = str(exc)
                prompt = base_prompt + RETRY_SUFFIX.format(error=error)
                continue
            interpretation = {
                **parsed.to_dict(),
                "model": self.provider.model,
                "provider": self.provider.name,
                "prompt_version": prompt_version(pets),
                "image_count": len(jpegs),
                "montage": self.config.montage,
                "latency_s": round(total_latency, 3),
                "attempts": attempt,
            }
            return InterpretationResult(
                "done", interpretation, raw_text=raw, latency_s=total_latency,
                attempts=attempt, meta=reply.meta,
            )  # fmt: skip
        return InterpretationResult(
            "failed", error=error, raw_text=(raw or "")[:2000], latency_s=total_latency, attempts=2
        )


Job = tuple[PetEvent, list[NDArray], list[float]]
ResultCallback = Callable[[PetEvent, InterpretationResult], None]


class VLMWorker:
    """Runs interpretations on one background thread, never blocking the perception loop.

    The queue is bounded; when it is full the oldest job is dropped and reported as
    ``skipped``. Requests are spaced to respect ``max_requests_per_minute``.
    """

    def __init__(
        self,
        interpreter: Interpreter,
        on_result: ResultCallback,
        config: VLMConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.interpreter = interpreter
        self.on_result = on_result
        self.config = config
        self._clock = clock
        self._jobs: collections.deque[Job] = collections.deque()
        self._cond = threading.Condition()
        self._busy = False
        self._stop = False
        self._last_request: float | None = None
        self._thread: threading.Thread | None = None
        self.stats = {"submitted": 0, "dropped": 0, "done": 0, "failed": 0}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="vlm-worker", daemon=True)
        self._thread.start()

    def submit(self, event: PetEvent, images: list[NDArray], times_s: list[float]) -> None:
        dropped = None
        with self._cond:
            if len(self._jobs) >= self.config.queue_size:
                dropped = self._jobs.popleft()
                self.stats["dropped"] += 1
            self._jobs.append((event, images, times_s))
            self.stats["submitted"] += 1
            self._cond.notify_all()
        if dropped is not None:
            self.on_result(dropped[0], InterpretationResult("skipped", error="VLM queue full"))

    def wait_idle(self, timeout_s: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._jobs or self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def stop(self, timeout_s: float = 5.0) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout_s)

    @property
    def pending(self) -> int:
        with self._cond:
            return len(self._jobs) + (1 if self._busy else 0)

    def _run(self) -> None:
        min_interval = 60.0 / self.config.max_requests_per_minute
        while True:
            with self._cond:
                while not self._jobs and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                event, images, times = self._jobs.popleft()
                self._busy = True
            if self._last_request is not None:
                wait = min_interval - (self._clock() - self._last_request)
                if wait > 0:
                    time.sleep(wait)
            self._last_request = self._clock()
            try:
                result = self.interpreter.interpret(event, images, times)
            except Exception as exc:  # keep the worker alive whatever the provider does
                result = InterpretationResult("failed", error=f"{type(exc).__name__}: {exc}")
            self.stats["done" if result.status == "done" else "failed"] += 1
            try:
                self.on_result(event, result)
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()
