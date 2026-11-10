import io
import json
from dataclasses import replace

import pytest

from perception_lab.apps.pets.app import build_app
from perception_lab.core.config import LabConfig, NotifyConfig, RulesConfig, SourceConfig, VLMConfig
from perception_lab.notify.notifier import LogNotifier, NotificationPolicy, WebhookNotifier
from perception_lab.vlm.providers import ProviderError


def _config(tmp_path, **overrides):
    cfg = LabConfig(
        source=SourceConfig(fps=5, synthetic_seconds=120.0),
        rules=RulesConfig(stationary_s=30.0, cooldown_s=60.0),
        notify=NotifyConfig(desktop=False, min_interval_s=0.0),
    )
    cfg = replace(
        cfg,
        storage=replace(
            cfg.storage, db_path=str(tmp_path / "lab.db"), media_dir=str(tmp_path / "media")
        ),
    )
    return replace(cfg, **overrides)


def _vlm_reply(activity):
    answer = {
        "activity": activity,
        "description": f"The animal appears to be {activity}.",
        "visual_observations": ["Animal visible"],
        "uncertainty": [],
        "certainty": "low",
    }
    return {"message": {"content": json.dumps(answer)}}


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    calls = []

    def fake_vlm(url, payload, timeout_s):
        calls.append(payload)
        prompt = payload["messages"][0]["content"]
        return _vlm_reply("eating" if '"eating"' in prompt.split("suggests")[1][:20] else "unknown")

    log = io.StringIO()
    cfg = _config(tmp, vlm=VLMConfig(enabled=True, max_requests_per_minute=60000))
    app = build_app(cfg, log_stream=log, vlm_post=fake_vlm, keep_live_view=True)
    status = app.pipeline.run(app.frames())
    events = app.store.all_events()
    yield app, status, events, calls, log.getvalue(), tmp
    app.close()


def test_every_scenario_behaviour_produces_one_event(synthetic_run):
    _, status, events, *_ = synthetic_run
    by_hypothesis = {}
    for e in events:
        by_hypothesis.setdefault(e.hypothesis, []).append(e)
    assert set(by_hypothesis) >= {
        "eating",
        "entered_restricted_area",
        "on_elevated_surface",
        "running",
        "interacting",
        "resting_or_sleeping",
    }
    (eating,) = by_hypothesis["eating"]
    assert eating.animal_classes == ("cat",) and eating.zone == "food_bowl"
    # The zone extends 0.3 m around the bowl, so the dwell starts shortly before t = 5 s.
    assert 2 <= eating.timestamp_start_s <= 8 and 33 <= eating.timestamp_end_s <= 37
    (entry,) = by_hypothesis["entered_restricted_area"]
    assert entry.animal_classes == ("dog",) and entry.zone == "bedroom_door"
    (resting,) = by_hypothesis["resting_or_sleeping"]  # not while eating at the bowl
    assert resting.timestamp_start_s > 50
    (interacting,) = by_hypothesis["interacting"]
    assert sorted(interacting.animal_classes) == ["cat", "dog"]
    assert status.frames == 600


def test_events_have_evidence_and_merged_interpretations(synthetic_run):
    _, status, events, calls, _, tmp = synthetic_run
    assert calls, "the VLM should have been asked"
    assert all(len(c["messages"][0]["images"]) >= 1 for c in calls)
    for e in events:
        assert e.evidence, e.rule
        for item in e.evidence:
            assert (tmp / "media" / item["path"]).is_file()
        assert e.interpretation_status == "done"
    eating = next(e for e in events if e.hypothesis == "eating")
    assert eating.activity == "eating"
    other = next(e for e in events if e.hypothesis == "running")
    assert other.activity == "unknown"
    assert "the VLM could not identify the activity" in other.uncertainty
    assert status.vlm["done"] == len(events)


def test_facts_are_measured_and_notifications_are_logged(synthetic_run):
    app, status, events, _, log, _ = synthetic_run
    elevated = next(e for e in events if e.hypothesis == "on_elevated_surface")
    assert elevated.observed_facts["elevation_m"] == pytest.approx(0.9, abs=0.1)
    running = next(e for e in events if e.hypothesis == "running")
    assert running.observed_facts["speed_unit"] == "m/s"
    assert running.observed_facts["speed"] > 1.2
    assert "cat #" in log and "possible eating" in log
    assert status.notifications == len(events)
    assert app.pipeline.live_jpeg[:2] == b"\xff\xd8"
    assert app.store.search("bowl")


def test_vlm_failure_keeps_rule_event(tmp_path):
    def down(url, payload, timeout_s):
        raise ProviderError("connection refused")

    cfg = _config(
        tmp_path,
        source=SourceConfig(fps=5, synthetic_seconds=40.0),
        vlm=VLMConfig(enabled=True, max_requests_per_minute=60000),
    )
    app = build_app(cfg, vlm_post=down, vlm_async=False)
    app.pipeline.run(app.frames())
    eating = next(e for e in app.store.all_events() if e.hypothesis == "eating")
    assert eating.interpretation_status == "failed" and eating.activity is None
    assert any("connection refused" in u for u in eating.uncertainty)
    app.close()


def test_rgb_only_run_reports_missing_depth(tmp_path):
    cfg = _config(tmp_path, source=SourceConfig(fps=5, synthetic_seconds=40.0))
    app = build_app(cfg)
    frames = ({**b.__dict__} for b in app.frames())
    from perception_lab.core.types import FrameBundle

    for data in frames:
        data["depth"] = None
        app.pipeline.process(FrameBundle(**data))
    app.pipeline.finish()
    events = app.store.all_events()
    eating = next(e for e in events if e.hypothesis == "eating")
    assert any("no depth" in u for u in eating.uncertainty)
    assert "distance_to_camera_m" not in eating.observed_facts
    app.close()


def test_notification_policy_rate_limit_and_filter():
    from perception_lab.events.model import PetEvent

    out = io.StringIO()
    policy = NotificationPolicy(
        [LogNotifier(out)], NotifyConfig(min_interval_s=60, rules=("zone_entry",))
    )
    e = PetEvent(
        "e", "c", "zone_entry", "entered_restricted_area", (1,), ("dog",), 0, 10, "t", "door"
    )
    assert policy.notify(e)
    assert not policy.notify(replace(e, timestamp_end_s=30))
    assert policy.notify(replace(e, timestamp_end_s=80))
    assert not policy.notify(replace(e, rule="speed_burst"))
    assert out.getvalue().count("dog #1 entered door") == 2


def test_webhook_refuses_remote_and_errors_do_not_propagate():
    with pytest.raises(ProviderError):
        WebhookNotifier("http://10.0.0.5/hook")

    def broken(url, payload, timeout_s):
        raise ProviderError("down")

    from perception_lab.events.model import PetEvent

    policy = NotificationPolicy(
        [WebhookNotifier("http://127.0.0.1:9/hook", post=broken)], NotifyConfig()
    )
    assert policy.notify(PetEvent("e", "c", "r", "h", (1,), ("cat",), 0, 0, "t"))


@pytest.mark.parametrize("min_score, expected", [(0.5, "Negrita"), (0.95, "unknown")])
def test_pet_name_is_gated_on_the_detector_score(tmp_path, min_score, expected):
    from perception_lab.core.config import PetConfig

    def named(url, payload, timeout_s):
        answer = json.loads(_vlm_reply("eating")["message"]["content"])
        answer.update(certainty="high", pet="Negrita")
        return {"message": {"content": json.dumps(answer)}}

    cfg = _config(
        tmp_path,
        source=SourceConfig(fps=5, synthetic_seconds=40.0),
        vlm=VLMConfig(
            enabled=True,
            max_requests_per_minute=60000,
            pets=(PetConfig("Negrita", "cat", "a black cat"),),
            pet_min_detector_score=min_score,
        ),
    )
    app = build_app(cfg, vlm_post=named, vlm_async=False)
    app.pipeline.run(app.frames())
    event = next(e for e in app.store.all_events() if e.hypothesis == "eating")
    assert event.observed_facts["detector_score_mean"] == pytest.approx(0.9)  # synthetic detector
    assert event.pet == expected
    assert event.event_id in {e.event_id for e in app.store.list(pet=expected, limit=500)}
    if expected == "unknown":
        assert event.interpretation["pet_vlm"] == "Negrita"
        assert any("withheld" in u for u in event.uncertainty)
    app.close()
