import json
from dataclasses import replace

import pytest

from perception_lab.core.config import (
    DetectorConfig,
    LabConfig,
    RulesConfig,
    SourceConfig,
    VLMConfig,
)
from perception_lab.eval.bench import bench_detector, bench_vlm
from perception_lab.eval.labels import Segment, labels_from_mapping, load_labels
from perception_lab.eval.metrics import (
    DetectionTally,
    Prediction,
    event_metrics,
    match,
    predictions_from_events,
    vlm_metrics,
)
from perception_lab.eval.runner import run_evaluation
from perception_lab.events.model import PetEvent


def _event(hypothesis, start, end, classes=("cat",), activity=None, status="not_requested"):
    e = PetEvent("e", "c", "r", hypothesis, (1,), classes, start, end, "t")
    if activity is not None:
        e.interpretation, e.interpretation_status = {"activity": activity}, "done"
    elif status != "not_requested":
        e.interpretation_status = status
    return e


def test_labels_validation(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"duration_s": 60, "segments": [
        {"activity": "eating", "t_start_s": 1, "t_end_s": 5, "animal_class": "cat"}]}))  # fmt: skip
    labels = load_labels(path)
    assert labels.segments[0] == Segment("eating", 1.0, 5.0, "cat") and labels.duration_s == 60
    with pytest.raises(ValueError, match="unknown activity"):
        labels_from_mapping({"segments": [{"activity": "dancing", "t_start_s": 0, "t_end_s": 1}]})
    with pytest.raises(ValueError, match="before"):
        labels_from_mapping({"segments": [{"activity": "eating", "t_start_s": 5, "t_end_s": 1}]})


def test_match_tolerance_class_and_one_to_one():
    segs = [Segment("eating", 10, 20, "cat"), Segment("interacting", 30, 40, None)]
    preds = [
        Prediction("eating", 21, 25, ("cat",)),  # within 2 s tolerance
        Prediction("eating", 12, 18, ("cat",)),  # better overlap: wins the match
        Prediction("eating", 12, 18, ("dog",)),  # wrong animal
        Prediction("interacting", 35, 36, ("cat", "dog")),
        Prediction("running", 30, 31, ("dog",)),
    ]
    matches, unmatched_p, unmatched_s = match(preds, segs, tolerance_s=2.0)
    assert sorted(matches) == [(1, 0), (3, 1)]
    assert unmatched_p == [0, 2, 4] and unmatched_s == []
    assert match(preds[:1], segs[:1], tolerance_s=0.5)[0] == []


def test_event_metrics_and_false_events_per_day():
    segs = [Segment("eating", 10, 20, "cat"), Segment("running", 50, 52, "dog")]
    preds = [Prediction("eating", 11, 19, ("cat",)), Prediction("eating", 100, 110, ("cat",))]
    m = event_metrics(preds, segs, duration_s=3600.0)
    eating = m["per_activity"]["eating"]
    assert (eating["tp"], eating["fp"], eating["fn"]) == (1, 1, 0)
    assert (eating["precision"], eating["recall"]) == (0.5, 1.0)
    assert eating["f1"] == pytest.approx(2 / 3)
    assert m["per_activity"]["running"]["recall"] == 0.0
    assert m["false_events_per_day"] == pytest.approx(24.0)
    assert m["overall"]["tp"] == 1


def test_predictions_from_events_rules_vs_vlm():
    events = [
        _event("eating", 1, 2, activity="eating"),
        _event("eating", 3, 4, activity="unknown"),
        _event("running", 5, 6),
    ]
    rules, _ = predictions_from_events(events, source="rules", t0_s=1.0)
    assert [p.activity for p in rules] == ["eating", "eating", "running"] and rules[
        0
    ].t_start_s == 0
    vlm, abstained = predictions_from_events(events, source="vlm")
    assert [p.activity for p in vlm] == ["eating"] and abstained == 1


def test_vlm_metrics_separates_unknown_from_errors():
    segs = [Segment("eating", 0, 10, "cat"), Segment("grooming", 20, 30, "cat")]
    events = [
        _event("eating", 1, 5, activity="eating"),
        _event("resting_or_sleeping", 21, 29, activity="resting_or_sleeping"),
        _event("eating", 6, 9, activity="unknown"),
        _event("running", 100, 101, activity="running"),
        _event("eating", 2, 3, status="failed"),
    ]
    m = vlm_metrics(events, segs)
    assert m["accuracy_when_answered"] == 0.5
    assert m["unknown_rate"] == pytest.approx(1 / 3)
    assert m["answered_without_label"] == 1
    assert m["confusion"]["grooming"] == {"resting_or_sleeping": 1}
    assert m["interpretation_status"] == {"done": 4, "failed": 1}


def test_detection_tally_counts_id_switches():
    tally = DetectionTally()
    tally.add([("cat", (0, 0, 10, 10))], [(1, (0, 0, 10, 10))])
    tally.add([("cat", (0, 0, 10, 10))], [(2, (1, 0, 11, 10)), (3, (50, 50, 60, 60))])
    tally.add([("cat", (0, 0, 10, 10))], [])
    r = tally.result()
    assert (r["tp"], r["fp"], r["fn"], r["id_switches"]) == (2, 1, 1, 1)


def _short_config():
    return LabConfig(
        source=SourceConfig(fps=5, synthetic_seconds=40.0),
        rules=RulesConfig(stationary_s=30.0),
    )


def test_run_evaluation_on_synthetic_writes_artifacts(tmp_path):
    metrics = run_evaluation(_short_config(), out_dir=tmp_path, run_id="r1")
    run_dir = tmp_path / "r1"
    for name in ("run.json", "metrics.json", "events.jsonl", "report.md"):
        assert (run_dir / name).is_file()
    assert metrics["rules"]["per_activity"]["eating"]["tp"] == 1
    assert metrics["detection_tracking"]["id_switches"] == 0
    assert "Rule events" in (run_dir / "report.md").read_text()
    assert json.loads((run_dir / "run.json").read_text())["labels_sha256"] == "synthetic"


def test_run_evaluation_with_vlm_and_labels_file(tmp_path):
    labels = tmp_path / "labels.json"
    segment = {"activity": "eating", "t_start_s": 5, "t_end_s": 35, "animal_class": "cat"}
    labels.write_text(json.dumps({"duration_s": 40, "segments": [segment]}))

    def fake_vlm(url, payload, timeout_s):
        answer = {"activity": "eating", "description": "d", "visual_observations": [],
                  "uncertainty": [], "certainty": "low"}  # fmt: skip
        return {"message": {"content": json.dumps(answer)}}

    cfg = replace(_short_config(), vlm=VLMConfig(enabled=True))
    metrics = run_evaluation(
        cfg, labels_path=labels, out_dir=tmp_path, run_id="r2", vlm_post=fake_vlm
    )
    assert metrics["vlm_events"]["per_activity"]["eating"]["tp"] == 1
    assert metrics["vlm"]["accuracy_when_answered"] is not None


def test_recorded_source_requires_labels(tmp_path):
    cfg = replace(_short_config(), source=SourceConfig(kind="bag", path="x.bag"))
    with pytest.raises(ValueError, match="labels"):
        run_evaluation(cfg, out_dir=tmp_path)


def test_bench_detector_and_vlm_with_fakes():
    cfg = replace(_short_config(), detector=DetectorConfig(kind="ground_truth"))
    result = bench_detector(cfg, frames=10, warmup=2)
    assert result["latency"]["n"] == 10 and result["detections"] > 0

    seen = []

    def fake_vlm(url, payload, timeout_s):
        seen.append(len(payload["messages"][0]["images"]))
        return {"message": {"content": "not json"}}

    result = bench_vlm(
        replace(cfg, vlm=VLMConfig(enabled=True)), image_counts=(1, 2), repeats=2, post=fake_vlm
    )
    assert result["runs"]["1"]["schema_valid"] == 0 and result["runs"]["2"]["requests"] == 2
    assert sorted(set(seen)) == [1, 2]  # images per request as asked (two attempts each)
    assert result["cold_latency_s"] is not None
