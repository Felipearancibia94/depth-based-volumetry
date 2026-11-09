import base64
import json
import threading

import numpy as np
import pytest

from perception_lab.core.config import PetConfig, VLMConfig
from perception_lab.events.evidence import BufferedFrame
from perception_lab.events.model import PetEvent
from perception_lab.vlm.interpreter import Interpreter, VLMWorker, evidence_images
from perception_lab.vlm.prompts import build_prompt, prompt_version
from perception_lab.vlm.providers import (
    OllamaProvider,
    OpenAICompatProvider,
    ProviderError,
    require_loopback,
)
from perception_lab.vlm.schema import (
    JSON_SCHEMA,
    InterpretationError,
    build_json_schema,
    parse_interpretation,
)

GOOD = {
    "activity": "eating",
    "description": "The cat stands at the bowl with its head lowered.",
    "visual_observations": ["Head lowered toward the bowl", "Body position stable"],
    "uncertainty": ["Bowl contents not visible"],
    "certainty": "medium",
}


def _event():
    return PetEvent(
        "e1", "cam", "zone_dwell", "eating", (3,), ("cat",), 5.0, 35.0, "2026-01-01T00:00:00",
        "bowl", {"duration_s": 30.0, "zones": ["bowl"]},
    )  # fmt: skip


def _images(n=2):
    return [np.full((64, 64, 3), 100 + i, np.uint8) for i in range(n)]


class FakeRuntime:
    """Records requests and replays canned replies (Ollama response shape)."""

    def __init__(self, *texts, fail=False):
        self.texts = list(texts)
        self.requests = []
        self.fail = fail

    def __call__(self, url, payload, timeout_s):
        self.requests.append((url, payload))
        if self.fail:
            raise ProviderError("connection refused")
        return {"message": {"content": self.texts.pop(0)}, "eval_count": 42}


def test_parse_accepts_valid_and_code_fenced_answers():
    assert parse_interpretation(json.dumps(GOOD)).activity == "eating"
    fenced = "```json\n" + json.dumps({**GOOD, "activity": "unknown"}) + "\n```"
    assert parse_interpretation(fenced).activity == "unknown"


@pytest.mark.parametrize(
    "answer, message",
    [
        ("not json", "not valid JSON"),
        ("[]", "JSON object"),
        (json.dumps({**GOOD, "activity": "dancing"}), "allowed list"),
        (json.dumps({**GOOD, "certainty": 0.9}), "low/medium/high"),
        (json.dumps({k: v for k, v in GOOD.items() if k != "uncertainty"}), "missing"),
        (json.dumps({**GOOD, "confidence": 0.99}), "unexpected"),
        (json.dumps({**GOOD, "visual_observations": "head down"}), "list of strings"),
        (json.dumps({**GOOD, "description": "x" * 700}), "longer than"),
    ],
)
def test_parse_rejects_invalid_answers(answer, message):
    with pytest.raises(InterpretationError, match=message):
        parse_interpretation(answer)


def test_schema_allows_unknown():
    assert "unknown" in JSON_SCHEMA["properties"]["activity"]["enum"]


def test_prompt_separates_facts_from_hypothesis():
    prompt = build_prompt(_event(), [5.0, 6.0, 7.5], montage=False)
    assert "3 images" in prompt and "t+2.5s" in prompt
    assert "- duration_s: 30.0" in prompt
    assert 'suggests "eating"' in prompt and "may be wrong" in prompt
    assert '"unknown"' in prompt
    assert "grid" in build_prompt(_event(), [5.0, 6.0], montage=True)


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:11434", "http://localhost:8080", "http://[::1]:1"]
)
def test_loopback_endpoints_allowed(url):
    require_loopback(url)


def test_remote_endpoint_refused_unless_allowed():
    with pytest.raises(ProviderError, match="non-loopback"):
        OllamaProvider(VLMConfig(base_url="http://192.168.1.20:11434"))
    require_loopback("http://192.168.1.20:11434", allow_remote=True)


def test_ollama_request_shape():
    runtime = FakeRuntime(json.dumps(GOOD))
    reply = OllamaProvider(VLMConfig(), runtime).complete("hi", [b"\xff\xd8x"], JSON_SCHEMA)
    url, payload = runtime.requests[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert payload["format"] == JSON_SCHEMA and payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["num_ctx"] == 8192
    assert base64.b64decode(payload["messages"][0]["images"][0]) == b"\xff\xd8x"
    assert reply.meta == {"eval_count": 42}


def test_ollama_sends_the_configured_context_window():
    runtime = FakeRuntime(json.dumps(GOOD))
    OllamaProvider(VLMConfig(num_ctx=16384), runtime).complete("hi", [b"\xff\xd8x"], JSON_SCHEMA)
    assert runtime.requests[0][1]["options"]["num_ctx"] == 16384


def test_openai_compat_request_shape():
    seen = {}

    def post(url, payload, timeout_s):
        seen.update(url=url, payload=payload)
        return {"choices": [{"message": {"content": json.dumps(GOOD)}}]}

    cfg = VLMConfig(provider="openai_compat", base_url="http://127.0.0.1:8080")
    OpenAICompatProvider(cfg, post).complete("hi", [b"abc", b"def"], JSON_SCHEMA)
    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    content = seen["payload"]["messages"][0]["content"]
    assert [c["type"] for c in content] == ["text", "image_url", "image_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert seen["payload"]["response_format"]["json_schema"]["schema"] == JSON_SCHEMA


def test_interpreter_success_records_provenance():
    runtime = FakeRuntime(json.dumps(GOOD))
    interp = Interpreter(OllamaProvider(VLMConfig(), runtime), VLMConfig())
    result = interp.interpret(_event(), _images(2), [5.0, 6.0])
    assert result.status == "done" and result.attempts == 1
    i = result.interpretation
    assert i["activity"] == "eating" and i["model"] == "qwen3-vl:2b-instruct"
    assert i["prompt_version"] == "pets_v1" and i["image_count"] == 2
    assert len(runtime.requests[0][1]["messages"][0]["images"]) == 2


def test_interpreter_retries_once_then_fails_without_inventing():
    runtime = FakeRuntime("garbage", json.dumps({**GOOD, "activity": "unknown"}))
    interp = Interpreter(OllamaProvider(VLMConfig(), runtime), VLMConfig())
    result = interp.interpret(_event(), _images(), [5.0, 6.0])
    assert result.status == "done" and result.attempts == 2
    assert result.interpretation["activity"] == "unknown"
    assert "not valid" in runtime.requests[1][1]["messages"][0]["content"]

    failing = Interpreter(OllamaProvider(VLMConfig(), FakeRuntime("x", "y")), VLMConfig())
    result = failing.interpret(_event(), _images(), [5.0, 6.0])
    assert result.status == "failed" and result.interpretation is None
    assert result.raw_text == "y"


def test_interpreter_handles_unreachable_runtime_and_no_images():
    interp = Interpreter(OllamaProvider(VLMConfig(), FakeRuntime(fail=True)), VLMConfig())
    assert interp.interpret(_event(), _images(), [5.0, 6.0]).status == "failed"
    assert interp.interpret(_event(), [], []).status == "skipped"


def _frames(n):
    return [
        BufferedFrame(
            float(i), i, np.full((240, 320, 3), i * 10, np.uint8), 0.5, {3: (100, 100, 300, 300)}
        )
        for i in range(n)
    ]


def test_evidence_images_crops_context_and_montage():
    cfg = VLMConfig(max_images=4, image_max_side_px=200)
    images, times = evidence_images(_frames(6), [3], cfg)
    assert len(images) == 4 and times == [3.0, 4.0, 5.0, 5.0]
    assert max(images[-1].shape[:2]) == 200  # context frame, resized
    assert images[0].shape[:2] != images[-1].shape[:2]  # crops differ from the full frame
    one, _ = evidence_images(_frames(6), [3], VLMConfig(montage=True, image_max_side_px=200))
    assert len(one) == 1


def test_worker_processes_drops_oldest_and_reports():
    cfg = VLMConfig(queue_size=1, max_requests_per_minute=6000)
    gate = threading.Event()

    class SlowInterpreter:
        def interpret(self, event, images, times):
            gate.wait(5)
            from perception_lab.vlm.interpreter import InterpretationResult

            return InterpretationResult("done", {"activity": "unknown"})

    results = []
    worker = VLMWorker(SlowInterpreter(), lambda e, r: results.append((e.event_id, r.status)), cfg)
    worker.start()
    events = [PetEvent(f"e{i}", "c", "r", "h", (1,), ("cat",), 0, 0, "t") for i in range(3)]
    worker.submit(events[0], [], [])
    while worker.pending and not worker._busy:  # wait until e0 is being processed
        pass
    worker.submit(events[1], [], [])
    worker.submit(events[2], [], [])  # queue full: e1 dropped
    gate.set()
    assert worker.wait_idle(5)
    worker.stop()
    assert ("e1", "skipped") in results
    assert {("e0", "done"), ("e2", "done")} <= set(results)
    assert worker.stats["dropped"] == 1


PETS = (
    PetConfig("Negrita", "cat", "a cat with a fully black coat"),
    PetConfig("Nube", "cat", "a cat with a tricolor coat"),
    PetConfig("Leia", "dog", "a dog"),
)


def test_without_pets_nothing_about_pets_is_asked():
    prompt = build_prompt(_event(), [5.0, 6.0], montage=False)
    assert "household" not in prompt and '"pet"' not in prompt
    assert build_json_schema() is JSON_SCHEMA and prompt_version() == "pets_v1"
    assert "pet" not in parse_interpretation(json.dumps(GOOD)).to_dict()
    with pytest.raises(InterpretationError, match="unexpected"):
        parse_interpretation(json.dumps({**GOOD, "pet": "Nube"}))


def test_prompt_lists_the_pets_and_asks_for_a_short_value():
    prompt = build_prompt(_event(), [5.0, 6.0], montage=False, pets=PETS)
    assert "- Negrita (cat): a cat with a fully black coat" in prompt
    assert "- Leia (dog): a dog" in prompt
    assert '"pet": the name of the listed pet' in prompt and '"unknown"' in prompt
    assert prompt_version(PETS) == "pets_v1+pet"


def test_schema_restricts_pet_to_the_configured_names_and_unknown():
    schema = build_json_schema(PETS)
    assert schema["properties"]["pet"]["enum"] == ["Negrita", "Nube", "Leia", "unknown"]
    assert "pet" in schema["required"]
    assert "pet" not in JSON_SCHEMA["properties"]  # the base schema is not modified


def test_parse_validates_pet():
    ok = parse_interpretation(json.dumps({**GOOD, "pet": "Nube"}), PETS)
    assert ok.pet == "Nube" and ok.to_dict()["pet"] == "Nube"
    assert parse_interpretation(json.dumps({**GOOD, "pet": "unknown"}), PETS).pet == "unknown"
    with pytest.raises(InterpretationError, match="not a configured pet"):
        parse_interpretation(json.dumps({**GOOD, "pet": "Rex"}), PETS)
    with pytest.raises(InterpretationError, match="missing"):
        parse_interpretation(json.dumps(GOOD), PETS)


def test_interpreter_asks_for_and_stores_the_pet():
    config = VLMConfig(pets=PETS)
    runtime = FakeRuntime(json.dumps({**GOOD, "pet": "Negrita"}))
    result = Interpreter(OllamaProvider(config, runtime), config).interpret(
        _event(), _images(), [5.0, 6.0]
    )
    payload = runtime.requests[0][1]
    assert payload["format"]["properties"]["pet"]["enum"][-1] == "unknown"
    assert "Nube (cat)" in payload["messages"][0]["content"]
    assert result.interpretation["pet"] == "Negrita"
    assert result.interpretation["prompt_version"] == "pets_v1+pet"
