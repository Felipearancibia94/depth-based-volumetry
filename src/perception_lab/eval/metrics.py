"""Evaluation metrics. All functions are pure.

Event matching: a predicted event matches a ground-truth segment when the activity is the
same, the animal class is compatible and the time intervals overlap once the segment is
widened by ``tolerance_s`` on each side. Matching is one-to-one, largest overlap first.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from perception_lab.core.types import BoxXYXY, box_iou
from perception_lab.eval.labels import Segment
from perception_lab.events.model import PetEvent


@dataclass(frozen=True)
class Prediction:
    activity: str
    t_start_s: float
    t_end_s: float
    animal_classes: tuple[str, ...]


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _overlap(p: Prediction, s: Segment, tolerance_s: float) -> float:
    start = max(p.t_start_s, s.t_start_s - tolerance_s)
    end = min(p.t_end_s, s.t_end_s + tolerance_s)
    return end - start


def _compatible(p: Prediction, s: Segment) -> bool:
    return s.animal_class is None or s.animal_class in p.animal_classes


def match(
    predictions: Sequence[Prediction], segments: Sequence[Segment], tolerance_s: float = 2.0
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return matched (prediction, segment) index pairs, unmatched predictions and segments."""
    pairs = []
    for pi, p in enumerate(predictions):
        for si, s in enumerate(segments):
            if p.activity == s.activity and _compatible(p, s):
                ov = _overlap(p, s, tolerance_s)
                if ov >= 0:
                    pairs.append((ov, pi, si))
    pairs.sort(reverse=True)
    used_p, used_s, matches = set(), set(), []
    for _, pi, si in pairs:
        if pi not in used_p and si not in used_s:
            used_p.add(pi)
            used_s.add(si)
            matches.append((pi, si))
    return (
        matches,
        [i for i in range(len(predictions)) if i not in used_p],
        [i for i in range(len(segments)) if i not in used_s],
    )


def predictions_from_events(
    events: Sequence[PetEvent], *, source: str = "rules", t0_s: float = 0.0
) -> tuple[list[Prediction], int]:
    """Predictions from rule hypotheses (``source="rules"``) or from VLM activities
    (``source="vlm"``). VLM "unknown" answers are abstentions, not predictions; their count
    is returned as the second value."""
    preds, abstained = [], 0
    for e in events:
        if source == "rules":
            activity = e.hypothesis
        else:
            activity = e.activity
            if activity is None:
                continue
            if activity == "unknown":
                abstained += 1
                continue
        preds.append(
            Prediction(
                activity, e.timestamp_start_s - t0_s, e.timestamp_end_s - t0_s, e.animal_classes
            )
        )
    return preds, abstained


def event_metrics(
    predictions: Sequence[Prediction],
    segments: Sequence[Segment],
    *,
    duration_s: float | None,
    tolerance_s: float = 2.0,
) -> dict[str, Any]:
    matches, unmatched_p, unmatched_s = match(predictions, segments, tolerance_s)
    per = defaultdict(lambda: Counter())
    for pi, _ in matches:
        per[predictions[pi].activity]["tp"] += 1
    for pi in unmatched_p:
        per[predictions[pi].activity]["fp"] += 1
    for si in unmatched_s:
        per[segments[si].activity]["fn"] += 1
    per_activity = {a: _prf(c["tp"], c["fp"], c["fn"]) for a, c in sorted(per.items())}
    overall = _prf(len(matches), len(unmatched_p), len(unmatched_s))
    false_per_day = (
        len(unmatched_p) / duration_s * 86400.0 if duration_s and duration_s > 0 else None
    )
    return {
        "per_activity": per_activity,
        "overall": overall,
        "false_events_per_day": false_per_day,
        "predictions": len(predictions),
        "segments": len(segments),
        "tolerance_s": tolerance_s,
    }


def vlm_metrics(
    events: Sequence[PetEvent],
    segments: Sequence[Segment],
    *,
    t0_s: float = 0.0,
    tolerance_s: float = 2.0,
) -> dict[str, Any]:
    """Compare VLM answers with the ground-truth segment overlapping each event in time.

    ``accuracy_when_answered`` excludes "unknown"; ``unknown_rate`` is reported separately so
    abstaining is never counted as correct or as wrong.
    """
    status = Counter(e.interpretation_status for e in events)
    confusion: dict[str, Counter] = defaultdict(Counter)
    answered = correct = unknown = without_truth = 0
    for e in events:
        if e.interpretation_status != "done" or not e.activity:
            continue
        p = Prediction(
            e.activity, e.timestamp_start_s - t0_s, e.timestamp_end_s - t0_s, e.animal_classes
        )
        overlapping = [
            (s, _overlap(p, s, tolerance_s))
            for s in segments
            if _compatible(p, s) and _overlap(p, s, tolerance_s) >= 0
        ]
        if not overlapping:
            without_truth += 1
            confusion["(no label)"][e.activity] += 1
            continue
        truth = {s.activity for s, _ in overlapping}
        best = max(overlapping, key=lambda x: x[1])[0].activity
        confusion[best][e.activity] += 1
        if e.activity == "unknown":
            unknown += 1
            continue
        answered += 1
        correct += e.activity in truth
    interpreted = status.get("done", 0)
    return {
        "interpretation_status": dict(status),
        "interpreted": interpreted,
        "unknown_rate": unknown / (unknown + answered) if unknown + answered else None,
        "accuracy_when_answered": correct / answered if answered else None,
        "answered_with_truth": answered,
        "answered_without_label": without_truth,
        "confusion": {gt: dict(c) for gt, c in sorted(confusion.items())},
    }


@dataclass
class DetectionTally:
    """Frame-level detection precision/recall at an IoU threshold, plus identity switches."""

    iou_threshold: float = 0.5
    tp: int = 0
    fp: int = 0
    fn: int = 0
    id_switches: int = 0
    frames: int = 0
    _last_id: dict[str, int] = field(default_factory=dict)

    def add(
        self, truth: Sequence[tuple[str, BoxXYXY]], tracks: Sequence[tuple[int, BoxXYXY]]
    ) -> None:
        self.frames += 1
        pairs = sorted(
            (
                (box_iou(tb, kb), ti, ki)
                for ki, (_, kb) in enumerate(truth)
                for ti, (_, tb) in enumerate(tracks)
            ),
            reverse=True,
        )
        used_t, used_k = set(), set()
        for iou, ti, ki in pairs:
            if iou < self.iou_threshold or ti in used_t or ki in used_k:
                continue
            used_t.add(ti)
            used_k.add(ki)
            key, track_id = truth[ki][0], tracks[ti][0]
            if key in self._last_id and self._last_id[key] != track_id:
                self.id_switches += 1
            self._last_id[key] = track_id
        self.tp += len(used_k)
        self.fn += len(truth) - len(used_k)
        self.fp += len(tracks) - len(used_t)

    def result(self) -> dict[str, Any]:
        return {
            **_prf(self.tp, self.fp, self.fn),
            "id_switches": self.id_switches,
            "frames": self.frames,
        }
