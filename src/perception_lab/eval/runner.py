"""Repeatable evaluation: replay a source through the pipeline and score it.

Each run writes ``<out_dir>/<run_id>/``: ``run.json`` (code revision, packages, config,
model hash, labels hash), ``metrics.json``, ``events.jsonl``, ``report.md`` and the run's
own event database and evidence. The VLM runs synchronously so results do not depend on
queue timing; use ``temperature = 0`` for repeatability.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from perception_lab.apps.pets.app import build_app
from perception_lab.core.config import LabConfig
from perception_lab.core.runrecord import build_run_record, new_run_id, write_json
from perception_lab.eval.labels import Labels, Segment, load_labels
from perception_lab.eval.metrics import (
    DetectionTally,
    event_metrics,
    predictions_from_events,
    vlm_metrics,
)
from perception_lab.vlm.providers import PostJSON, post_json


def synthetic_labels(config: LabConfig) -> Labels:
    from perception_lab.sources.synthetic import default_scenario

    scenario = default_scenario()
    duration = min(config.source.synthetic_seconds, scenario.duration_s)
    segments = tuple(
        Segment(s.activity, s.t_start_s, min(s.t_end_s, duration), s.animal_class)
        for s in scenario.segments
        if s.t_start_s < duration
    )
    return Labels(segments, duration, "synthetic:default_scenario")


def run_evaluation(
    config: LabConfig,
    *,
    labels_path: str | Path | None = None,
    out_dir: str | Path = "outputs/runs",
    run_id: str | None = None,
    tolerance_s: float = 2.0,
    vlm_post: PostJSON = post_json,
) -> dict[str, Any]:
    run_id = run_id or new_run_id("eval")
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = replace(
        config,
        storage=replace(
            config.storage, db_path=str(run_dir / "events.db"), media_dir=str(run_dir / "media")
        ),
        notify=replace(config.notify, desktop=False, webhook_url=None),
    )
    if labels_path is not None:
        labels = load_labels(labels_path)
        labels_hash = hashlib.sha256(Path(labels_path).read_bytes()).hexdigest()
    elif config.source.kind == "synthetic":
        labels, labels_hash = synthetic_labels(config), "synthetic"
    else:
        raise ValueError("a labels file is required for recorded sources (--labels)")

    write_json(
        run_dir / "run.json",
        build_run_record(
            config,
            {
                "run_id": run_id,
                "labels": str(labels_path or labels.recording),
                "labels_sha256": labels_hash,
            },
        ),
    )

    app = build_app(config, vlm_async=False, vlm_post=vlm_post)
    tally = DetectionTally()
    every_n = max(1, config.detector.every_n_frames)
    t0_s, last_s, frames = None, None, 0
    started = time.perf_counter()
    try:
        for bundle in app.frames():
            t0_s = bundle.timestamp_s if t0_s is None else t0_s
            last_s = bundle.timestamp_s
            frames += 1
            app.pipeline.process(bundle)
            truth = bundle.meta.get("ground_truth")
            if truth is not None and bundle.index % every_n == 0:
                tally.add(
                    [(g["key"], g["box_xyxy"]) for g in truth],
                    [(o.track_id, o.box_xyxy) for o in app.pipeline.last_observations],
                )
        app.pipeline.finish()
        wall_s = time.perf_counter() - started
        events = app.store.all_events()
    finally:
        app.close()

    t0_s = t0_s or 0.0
    duration = labels.duration_s or ((last_s - t0_s) if last_s is not None else None)
    rule_preds, _ = predictions_from_events(events, source="rules", t0_s=t0_s)
    vlm_preds, abstained = predictions_from_events(events, source="vlm", t0_s=t0_s)
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "frames": frames,
        "source_duration_s": duration,
        "processing_wall_s": round(wall_s, 2),
        "processing_fps": round(frames / wall_s, 2) if wall_s > 0 else None,
        "stage_ms": app.pipeline.status.stage_ms,
        "events": len(events),
        "rules": event_metrics(
            rule_preds, labels.segments, duration_s=duration, tolerance_s=tolerance_s
        ),
    }
    if config.vlm.enabled:
        metrics["vlm_events"] = event_metrics(
            vlm_preds, labels.segments, duration_s=duration, tolerance_s=tolerance_s
        )
        metrics["vlm_events"]["abstained_unknown"] = abstained
        metrics["vlm"] = vlm_metrics(events, labels.segments, t0_s=t0_s, tolerance_s=tolerance_s)
    if tally.frames:
        metrics["detection_tracking"] = tally.result()

    write_json(run_dir / "metrics.json", metrics)
    with open(run_dir / "events.jsonl", "w") as handle:
        for e in events:
            handle.write(json.dumps(e.to_dict()) + "\n")
    (run_dir / "report.md").write_text(render_report(metrics, config))
    return metrics


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_report(metrics: dict[str, Any], config: LabConfig) -> str:
    vlm = f"`{config.vlm.model}` via {config.vlm.provider}" if config.vlm.enabled else "disabled"
    lines = [
        f"# Evaluation {metrics['run_id']}",
        "",
        f"- Source: `{config.source.kind}` {config.source.path or ''}",
        f"- Detector: `{config.detector.kind}` {config.detector.model_path or ''}",
        f"- VLM: {vlm}",
        f"- Frames: {metrics['frames']}, source duration {_fmt(metrics['source_duration_s'])} s, "
        f"processing {_fmt(metrics['processing_fps'])} frames/s on this machine",
        "",
    ]
    for key, title in (("rules", "Rule events"), ("vlm_events", "VLM activities")):
        if key not in metrics:
            continue
        m = metrics[key]
        o = m["overall"]
        lines += [
            f"## {title}",
            "",
            f"Overall: precision {_fmt(o['precision'])}, recall {_fmt(o['recall'])}, "
            f"F1 {_fmt(o['f1'])}; false events per day {_fmt(m['false_events_per_day'])} "
            f"(tolerance {m['tolerance_s']} s)",
            "",
            "| Activity | TP | FP | FN | Precision | Recall |",
            "|---|---|---|---|---|---|",
        ]
        for activity, s in m["per_activity"].items():
            lines.append(
                f"| {activity} | {s['tp']} | {s['fp']} | {s['fn']} | "
                f"{_fmt(s['precision'])} | {_fmt(s['recall'])} |"
            )
        lines.append("")
    if "vlm" in metrics:
        v = metrics["vlm"]
        lines += [
            "## VLM answers",
            "",
            f"- Status: {v['interpretation_status']}",
            f"- Accuracy when answered: {_fmt(v['accuracy_when_answered'])} "
            f"({v['answered_with_truth']} answers with a label)",
            f"- Unknown rate: {_fmt(v['unknown_rate'])}",
            f"- Answers without any overlapping label: {v['answered_without_label']}",
            "",
        ]
    if "detection_tracking" in metrics:
        d = metrics["detection_tracking"]
        lines += [
            "## Detection and tracking",
            "",
            f"Precision {_fmt(d['precision'])}, recall {_fmt(d['recall'])}, "
            f"identity switches {d['id_switches']} over {d['frames']} frames.",
            "",
        ]
    lines.append(
        "Numbers are valid only for this source, configuration and machine (see run.json)."
    )
    return "\n".join(lines) + "\n"
