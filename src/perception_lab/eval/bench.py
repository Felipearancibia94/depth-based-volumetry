"""Measure latency and GPU memory on this machine. Nothing here is a published benchmark.

GPU memory comes from ``nvidia-smi`` (whole-GPU usage, sampled ~10 times per second), so
close other GPU programs for meaningful numbers. Ollama's ``/api/ps`` adds the runtime's own
view of the model's VRAM when available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import urllib.request
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from depth_volumetry.io import read_color
from perception_lab.core.config import LabConfig
from perception_lab.events.model import PetEvent
from perception_lab.sources import limit, open_source
from perception_lab.sources.synthetic import SyntheticSource
from perception_lab.vision.detector import create_detector
from perception_lab.vlm.interpreter import Interpreter
from perception_lab.vlm.providers import PostJSON, create_provider, post_json, require_loopback


def gpu_memory_used_mb() -> list[int] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return [int(v) for v in out.split() if v.strip().isdigit()]


class GpuSampler:
    """Background sampler of total GPU memory; ``peak_mb`` is ``None`` without nvidia-smi."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.baseline_mb = _total(gpu_memory_used_mb())
        self.peak_mb = self.baseline_mb
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> GpuSampler:
        if self.baseline_mb is not None:
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(2)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            value = _total(gpu_memory_used_mb())
            if value is not None and (self.peak_mb is None or value > self.peak_mb):
                self.peak_mb = value

    def to_dict(self) -> dict[str, Any]:
        return {"gpu_baseline_mb": self.baseline_mb, "gpu_peak_mb": self.peak_mb}


def _total(values: list[int] | None) -> int | None:
    return sum(values) if values else None


def _summary(samples_s: Sequence[float]) -> dict[str, Any]:
    if not samples_s:
        return {"n": 0}
    ms = np.asarray(samples_s) * 1000.0
    return {
        "n": len(ms),
        "mean_ms": round(float(ms.mean()), 2),
        "p50_ms": round(float(np.percentile(ms, 50)), 2),
        "p95_ms": round(float(np.percentile(ms, 95)), 2),
        "max_ms": round(float(ms.max()), 2),
    }


def bench_detector(config: LabConfig, *, frames: int = 100, warmup: int = 5) -> dict[str, Any]:
    detector = create_detector(config.detector)
    source = open_source(config.source)
    bundles = list(limit(source.frames(), frames + warmup))
    source.close()
    if not bundles:
        raise ValueError("the source produced no frames")
    for b in bundles[:warmup]:
        detector.detect(b)
    latencies, detections = [], 0
    with GpuSampler() as gpu:
        for b in bundles[warmup:]:
            start = time.perf_counter()
            detections += len(detector.detect(b))
            latencies.append(time.perf_counter() - start)
    session = getattr(detector, "_session", None)
    return {
        "detector": config.detector.kind,
        "model_path": config.detector.model_path,
        "input_size": list(getattr(detector, "input_size", ())) or None,
        "providers_in_use": session.get_providers() if session is not None else None,
        "frame_shape": list(bundles[0].color_bgr.shape),
        "latency": _summary(latencies),
        "detections": detections,
        **gpu.to_dict(),
    }


def ollama_loaded_models(base_url: str) -> list[dict[str, Any]] | None:
    try:
        require_loopback(base_url)
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/ps", timeout=5) as r:  # noqa: S310
            data = json.loads(r.read().decode())
    except Exception:
        return None
    return [
        {
            "name": m.get("name"),
            "size_mb": _mb(m.get("size")),
            "size_vram_mb": _mb(m.get("size_vram")),
        }
        for m in data.get("models", [])
    ]


def _mb(value: Any) -> float | None:
    return round(value / 2**20, 1) if isinstance(value, int | float) else None


def bench_vlm(
    config: LabConfig,
    *,
    image_counts: Sequence[int] = (1, 3),
    repeats: int = 3,
    image_path: str | None = None,
    post: PostJSON = post_json,
) -> dict[str, Any]:
    """Send the same request ``repeats`` times per image count and record what happens.

    The first request may include model loading; it is reported separately as ``cold``.
    """
    if image_path:
        base = read_color(image_path)
    else:
        bundle = SyntheticSource(fps=1, depth_noise=False).render(20)
        base = bundle.color_bgr
    side = config.vlm.image_max_side_px
    h, w = base.shape[:2]
    scale = min(1.0, side / max(h, w))
    image = cv2.resize(base, (round(w * scale), round(h * scale))) if scale < 1 else base
    interpreter = Interpreter(create_provider(config.vlm, post), config.vlm)
    event = PetEvent(
        "bench", "bench", "bench", "unknown", (1,), ("cat",), 0.0, 1.0, "bench",
        observed_facts={"note": "benchmark request"},
    )  # fmt: skip
    results: dict[str, Any] = {
        "model": config.vlm.model,
        "provider": config.vlm.provider,
        "runs": {},
    }
    cold_done = False
    with GpuSampler() as gpu:
        for n in image_counts:
            latencies, statuses, activities, errors = [], [], [], []
            for _ in range(repeats):
                r = interpreter.interpret(event, [image] * n, [float(i) for i in range(n)])
                if not cold_done:
                    results["cold_latency_s"] = round(r.latency_s, 3)
                    cold_done = True
                else:
                    latencies.append(r.latency_s)
                statuses.append(r.status)
                if r.interpretation and "activity" in r.interpretation:
                    activities.append(r.interpretation["activity"])
                if r.error:
                    errors.append(r.error[:200])
            results["runs"][str(n)] = {
                "latency": _summary(latencies),
                "schema_valid": statuses.count("done"),
                "requests": repeats,
                "activities": dict(_count(activities)),
                "errors": errors[:3],
            }
    results.update(gpu.to_dict())
    if config.vlm.provider == "ollama":
        results["ollama_ps"] = ollama_loaded_models(config.vlm.base_url)
    return results


def _count(values: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
