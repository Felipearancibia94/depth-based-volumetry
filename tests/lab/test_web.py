import http.client
import json

import numpy as np
import pytest

from perception_lab.events.model import PetEvent
from perception_lab.spatial.scene import SceneCalibration, Zone
from perception_lab.store.db import EventStore
from perception_lab.store.media import save_images
from perception_lab.web.server import DashboardState, start_server


class _FakePipeline:
    live_jpeg = b"\xff\xd8fake"

    class status:  # noqa: N801 - mimics the attribute
        @staticmethod
        def to_dict():
            return {"frames": 10, "processing_fps": 5.0, "active_tracks": []}


@pytest.fixture
def dashboard(tmp_path):
    store = EventStore(tmp_path / "lab.db")
    media = tmp_path / "media"
    paths = save_images(media, "e1", [("crop_0", np.zeros((8, 8, 3), np.uint8))])
    event = PetEvent(
        "e1", "cam", "zone_dwell", "eating", (3,), ("cat",), 5.0, 35.0, "2026-01-01T00:00:00",
        "bowl", {"duration_s": 30.0}, evidence=[{"kind": "crop", "path": paths[0]}],
    )  # fmt: skip
    store.add(event)
    scene = SceneCalibration(zones=(Zone("bowl", ((0, 0), (1, 0), (1, 1)), ("food",)),))
    named = PetEvent(
        "e2", "cam", "stationary", "resting_or_sleeping", (4,), ("cat",), 40.0, 70.0,
        "2026-01-01T00:01:00", None, {"duration_s": 30.0},
        interpretation={"activity": "resting_or_sleeping", "description": "rests", "pet": "Nube"},
        interpretation_status="done",
    )  # fmt: skip
    store.add(named)
    state = DashboardState(
        store, str(media), scene, _FakePipeline(), pet_names=("Negrita", "Nube", "Leia")
    )
    server, _ = start_server(state, "127.0.0.1", 0)
    yield server.server_address[1], paths[0]
    server.shutdown()
    store.close()


def _request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, response.getheader("Content-Type"), data


def test_page_status_events_and_search(dashboard):
    port, _ = dashboard
    status, ctype, body = _request(port, "GET", "/")
    assert status == 200 and ctype.startswith("text/html") and b"Pet Monitor" in body
    status, _, body = _request(port, "GET", "/api/status")
    data = json.loads(body)
    assert data["events_total"] == 2 and data["pipeline"]["frames"] == 10
    events = json.loads(_request(port, "GET", "/api/events?limit=5")[2])
    assert sorted(e["event_id"] for e in events) == ["e1", "e2"]
    assert json.loads(_request(port, "GET", "/api/search?q=bowl")[2])[0]["event_id"] == "e1"
    assert json.loads(_request(port, "GET", "/api/events/e1")[2])["zone"] == "bowl"
    assert _request(port, "GET", "/api/events/nope")[0] == 404
    assert json.loads(_request(port, "GET", "/api/zones")[2])[0]["name"] == "bowl"


def test_media_and_live_view(dashboard):
    port, path = dashboard
    status, ctype, body = _request(port, "GET", "/media/" + path)
    assert status == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8"
    assert _request(port, "GET", "/media/../../etc/passwd")[0] == 404
    assert _request(port, "GET", "/api/live.jpg")[2] == b"\xff\xd8fake"


def test_review_requires_json_and_valid_status(dashboard):
    port, _ = dashboard
    body = json.dumps({"status": "rejected", "label": "sniffing"})
    form = {"Content-Type": "application/x-www-form-urlencoded"}
    assert _request(port, "POST", "/api/events/e1/review", body, form)[0] == 415
    js = {"Content-Type": "application/json"}
    status, _, data = _request(port, "POST", "/api/events/e1/review", body, js)
    assert status == 200 and json.loads(data)["review_label"] == "sniffing"
    assert _request(port, "POST", "/api/events/e1/review", '{"status": "maybe"}', js)[0] == 400
    assert _request(port, "POST", "/api/events/zz/review", body, js)[0] == 404


def test_foreign_host_header_is_rejected(dashboard):
    port, _ = dashboard
    assert _request(port, "GET", "/api/status", headers={"Host": "evil.example:80"})[0] == 403


def test_non_loopback_bind_refused_without_opt_in(tmp_path):
    store = EventStore(tmp_path / "x.db")
    with pytest.raises(ValueError, match="no authentication"):
        start_server(DashboardState(store, str(tmp_path)), "0.0.0.0", 0)
    store.close()


def test_pet_filter_and_pet_list(dashboard):
    port, _ = dashboard
    pets = json.loads(_request(port, "GET", "/api/pets")[2])
    assert pets == {"pets": ["Negrita", "Nube", "Leia"]}
    nube = json.loads(_request(port, "GET", "/api/events?pet=Nube")[2])
    assert [e["event_id"] for e in nube] == ["e2"]
    assert json.loads(_request(port, "GET", "/api/events?pet=Leia")[2]) == []
    status, _, body = _request(port, "GET", "/")
    assert b'id="pet"' in body


def test_review_can_correct_the_pet(dashboard):
    port, _ = dashboard
    headers = {"Content-Type": "application/json"}
    body = json.dumps({"status": "confirmed", "pet": "Leia"})
    status, _, data = _request(port, "POST", "/api/events/e2/review", body, headers)
    assert status == 200 and json.loads(data)["review_pet"] == "Leia"
    assert [
        e["event_id"] for e in json.loads(_request(port, "GET", "/api/events?pet=Leia")[2])
    ] == ["e2"]
    bad = json.dumps({"status": "confirmed", "pet": "Rex"})
    assert _request(port, "POST", "/api/events/e2/review", bad, headers)[0] == 400


def test_summary_endpoint(dashboard):
    port, _ = dashboard
    report = json.loads(_request(port, "GET", "/api/summary")[2])
    assert report["events"] == 2 and report["pets"]["Nube"]["events"] == 1
    assert _request(port, "GET", "/api/summary?date=nope")[0] == 400
    assert json.loads(_request(port, "GET", "/api/summary?date=2001-01-01")[2])["events"] == 0
